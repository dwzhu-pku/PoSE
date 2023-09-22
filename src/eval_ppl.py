import copy
import random
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from pathlib import Path

import torch
import torch.distributed
import transformers
import deepspeed
import evaluate
import datasets
import numpy as np
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from transformers import LlamaTokenizer, AutoTokenizer, AutoConfig
from datasets import load_dataset
from evaluate import logging
from tqdm import tqdm

from my_modeling_llama import LlamaForCausalLM
from my_modeling_gptj import GPTJForCausalLM
from my_modeling_baichuan import BaichuanForCausalLM
from my_configuration_llama import LlamaConfig
from my_configuration_gptj import GPTJConfig
from my_configuration_baichuan import BaichuanConfig
from tokenization_baichuan import BaichuanTokenizer

from train_pose import smart_tokenizer_and_embedding_resize, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_PAD_TOKEN, DEFAULT_UNK_TOKEN

gpu_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def compute_perplexity(
    encodings, model, tokenizer, add_start_token: bool = True, max_length=None, sliding_window_step=256, truncate=False, aggressive_memory=False
):
    r"""Compute "sliding window" perplexity on a dataset. Validated against the calculations reported in arXiv 2306.15595"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if add_start_token:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    if max_length and truncate:
        encoded_texts = [x[0:max_tokenized_len] for x in encoded_texts]
        attn_masks = [x[0:max_tokenized_len] for x in attn_masks]
        sliding_window_step = max_tokenized_len

    pbar = tqdm(total=len(encoded_texts))
    nlls = []
    total_nll = torch.tensor(0,dtype=torch.float64).to(device)
    total_token_cnt = 0
    for encoding_index in range(0, len(encoded_texts)):

        labels = torch.tensor(encoded_texts[encoding_index:encoding_index+1])
        seq_len = labels.size(1)

        prev_end_loc = 0
        for begin_loc in range(0, seq_len, sliding_window_step):

            end_loc = min(begin_loc + max_tokenized_len, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = labels[:, begin_loc:end_loc].to(device)

            if add_start_token:
                bos_tokens_tensor = torch.tensor(
                    [[tokenizer.bos_token_id]] * input_ids.size(dim=0)).to(device)
                input_ids = torch.cat(
                    [bos_tokens_tensor, input_ids], dim=1)

            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss
                total_nll += neg_log_likelihood * trg_len
                total_token_cnt += trg_len
            
            nlls.append(neg_log_likelihood)

            ppl = float(torch.exp(total_nll / total_token_cnt).float().cpu())
            pbar.set_postfix(ppl=ppl)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        pbar.update(1)

    ppl = float(torch.exp(total_nll / total_token_cnt).float().cpu())
    return {"mean_perplexity": ppl}


def main():

    # add parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_input_tokens", type=int, default=500)
    parser.add_argument("--max_input_tokens", type=int, default=1000)
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--eval_nums", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sliding_window_step", type=int, default=256)
    parser.add_argument('--window_length_list', type=int, nargs='+', default=[])
    parser.add_argument("--truncate", action="store_true", default=False)
    parser.add_argument("--model_max_position_embeddings", type=int, default=2048)
    parser.add_argument("--rope_scaling_factor", type=float, default=1.0)
    parser.add_argument("--rope_scaling_type", type=str, default=None)
    parser.add_argument("--input_field", type=str, default="text")
    parser.add_argument("--model_name", type=str, default="llama-7b")
    parser.add_argument("--path_to_ckp", type=str, default="/home/v-daweizhu/teamdrive/model/llama-7b")
    parser.add_argument("--dataset_name", type=str, default="scrolls-gov_report")
    parser.add_argument("--path_to_dataset", type=str, default="")
    parser.add_argument("--path_to_output_dir", type=str, default="results/ppls")
    args = parser.parse_args()

    model_name_or_path = args.path_to_ckp

    Config, CausalLM, Tokenizer = None, None, None

    if "llama" in args.model_name:
        Config, CausalLM, Tokenizer = LlamaConfig, LlamaForCausalLM, AutoTokenizer
    elif "gptj" in args.model_name:
        Config, CausalLM, Tokenizer = GPTJConfig, GPTJForCausalLM, AutoTokenizer
    elif "baichuan" in args.model_name:
        Config, CausalLM, Tokenizer = BaichuanConfig, BaichuanForCausalLM, BaichuanTokenizer


    config = Config.from_pretrained(model_name_or_path)
    scaled_max_position_embeddings=int(args.model_max_position_embeddings * args.rope_scaling_factor)

    if config.rope_scaling is None:
        if args.rope_scaling_type is not None:
            config.rope_scaling={"type": args.rope_scaling_type, "factor": args.rope_scaling_factor}
            config.max_position_embeddings=scaled_max_position_embeddings
            if args.rope_scaling_type == "yarn":
                config.rope_scaling["original_max_position_embeddings"] = args.model_max_position_embeddings
            
    config.use_cache=False

    model = CausalLM.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=config,torch_dtype=torch.float16)
    model.to(gpu_device)

    tokenizer = Tokenizer.from_pretrained(model_name_or_path, use_fast=False if "baichuan" in args.model_name else True)
    
    if not "baichuan" in args.model_name:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in args.model_name:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    if "scrolls" in args.dataset_name:
        args.input_field = "input"
    elif "pile" in args.dataset_name:
        args.input_field = "text"
    elif "proof" in args.dataset_name:
        args.input_field = "text"

    input_texts = load_dataset("json", data_files=args.path_to_dataset, split="train")

    def tokenize(example):
        tokenized = tokenizer(
            example[args.input_field],
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens - 1, # leave room for <BOS> token to be added
            return_attention_mask=True,
        )
        example["input_ids"] = tokenized["input_ids"]
        example["attention_mask"] = tokenized["attention_mask"]
        example["tokenized_len"] = len(tokenized["input_ids"])
        return example

    input_texts = input_texts.map(tokenize,num_proc=2)

    if args.min_input_tokens:
        input_texts = input_texts.filter(
            lambda x: x["tokenized_len"] >= args.min_input_tokens - 1)
    if args.eval_nums:
        input_texts = input_texts[:args.eval_nums]

    ppl_list = []
    context_window_size = args.window_length_list
    print(context_window_size)
    # context_window_size = [8192,16384]

    for ctx_size in context_window_size:
        # if args.truncate is True, we calucate the ppl on the whole input text
        # otherwise, we calucate the ppl with sliding window
        ppl = compute_perplexity(encodings=input_texts, model=model, tokenizer=tokenizer, add_start_token=True, max_length=ctx_size, sliding_window_step=args.sliding_window_step, truncate=args.truncate)["mean_perplexity"]

        print(f"model: {args.model_name}; context window size: {ctx_size}; ppl: {ppl}")
        
        ppl_list.append(ppl)

    root_dir = Path(__file__).parent.parent
    path_to_output_fn = (root_dir / args.path_to_output_dir / f"{args.model_name}+{args.dataset_name}").as_posix()
    with open(path_to_output_fn, "w") as f:
        f.write(f"model: {args.model_name}\n")
        f.write(f"length: {', '.join(map(str, context_window_size))}\n")
        f.write(f"ppl: {', '.join(map(str, ppl_list))}\n")


if __name__ == "__main__":
    main()
