import random
import argparse
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from pathlib import Path
import jsonlines

import torch
import torch.distributed
import transformers
import deepspeed
import evaluate
import datasets
import numpy as np
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from transformers import LlamaTokenizer, pipeline
from datasets import load_dataset
from evaluate import logging
from tqdm import tqdm

from my_modeling_llama import LlamaForCausalLM
from my_configuration_llama import LlamaConfig
from train_skipos import smart_tokenizer_and_embedding_resize, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_PAD_TOKEN, DEFAULT_UNK_TOKEN

gpu_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_prompt(max_tokens=16384):
    """Generates a text file and inserts an execute line at a random position."""
    # n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_total = (max_tokens - 32 - 26 - 11) // 25 
    n_garbage_prefix = random.randint(0, n_garbage_total)
    n_garbage_suffix = n_garbage_total - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there." # 32 tokens
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again." # 25 tokens
    garbage_prefix = garbage * n_garbage_prefix
    garbage_suffix = garbage * n_garbage_suffix
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key." # 26 tokens
    final_question = "What is the pass key? The pass key is" # 11 tokens
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question
    ]
    return "\n".join(lines), pass_key


def test_model(model, tokenizer, prompt_text, pass_key):
    
    model_input = tokenizer.encode(prompt_text, return_tensors="pt", max_length=100000, truncation=True).to(gpu_device)

    response = model.generate(model_input, num_return_sequences=1, max_new_tokens=10)
    response = tokenizer.batch_decode(response[:, model_input.shape[1]:], skip_special_tokens=True)[0]
    print(response)

    assert f"The pass key is {pass_key}" in prompt_text

    try:
        pass_key = int(re.search(r'\d+', response).group())
    except:
        pass_key = response[:20]

    return pass_key


def main():

    # add parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_length", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=1000)
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--eval_nums", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_max_position_embeddings", type=int, default=2048)
    parser.add_argument("--rope_scaling_factor", type=float, default=1.0)
    parser.add_argument("--rope_scaling_type", type=str, default=None)
    parser.add_argument("--input_field", type=str, default="text")
    parser.add_argument("--model_name", type=str, default="llama-7b")
    parser.add_argument("--path_to_ckp", type=str, default="/home/v-daweizhu/teamdrive/model/llama-7b")
    parser.add_argument("--path_to_output_dir", type=str, default="results/passkey")
    args = parser.parse_args()

    # model_name_or_path = "/home/v-daweizhu/teamdrive/skipos/results/2k-32k-v5/checkpoint-500"
    model_name_or_path = args.path_to_ckp

    config = LlamaConfig.from_pretrained(model_name_or_path)
    scaled_max_position_embeddings=int(args.model_max_position_embeddings * args.rope_scaling_factor)

    if config.rope_scaling is None:
        if args.rope_scaling_type is not None:
            config.rope_scaling={"type": args.rope_scaling_type, "factor": args.rope_scaling_factor}
            config.max_position_embeddings=scaled_max_position_embeddings
            

    config.use_cache=False

    model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=config, torch_dtype=torch.float16)
    model.to(gpu_device)

    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
        model=model,
    )
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )

    result_list = list()
    length_list = [2048,4096,6134,8192,10240,12288,14336,16384,20480,24576,28672,32768]
    # length_list = [16384,20480,24576,28672,32768]
    for context_size in length_list:
        if context_size == scaled_max_position_embeddings:
            context_size -= 100
        # if context_size > scaled_max_position_embeddings:
        #     break
        print(f"context_size: {context_size}")
        correct_cnt = 0
        result_dict = {"scaled_length": scaled_max_position_embeddings, "context_size": context_size}
        iter_nums = 50
        for i in tqdm(range(iter_nums)):
            prompt_text, pass_key = generate_prompt(context_size)
            pred = test_model(model, tokenizer, prompt_text, pass_key)
            result = "Pass!" if pred == pass_key else "Fail!"
            correct_cnt += 1 if pred == pass_key else 0
            case_report = f"pred: {pred}, ans: {pass_key}, result: {result}"
            result_dict[f"case{i}"] = case_report
            print(case_report)
        print(f"correct_rate: {correct_cnt/iter_nums}")
        result_dict["correct_rate"] = correct_cnt/iter_nums
        result_list.append(result_dict)
    
    root_dir = Path(__file__).parent.parent
    path_to_output_fn = (root_dir / args.path_to_output_dir / f"{args.model_name}.jsonl").as_posix()

    with jsonlines.open(path_to_output_fn, "w") as writer:
        writer.write_all(result_list)
    




if __name__ == "__main__":
    main()