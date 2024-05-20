#    Modification Copyright 2023 Dawei Zhu
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.distributed
import transformers
import deepspeed
from torch.utils.data import Dataset
from transformers import Trainer, GPT2ForQuestionAnswering, AutoConfig
from datasets import load_dataset

from my_modeling_llama import LlamaForCausalLM
#import utils

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    valid_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    test_data_path: str = field(default=None, metadata={"help": "Path to the test data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_position_embeddings: int = field(
        default=2048,
        metadata={"help": "Maximum position embeddings."},
    )
    inference_length: int = field(
        default=2048,
        metadata={"help": "Maximum position embeddings."},
    )
    rope_scaling_type: Optional[str] = field(default=None)
    rope_scaling_factor: float = field(default=1.0)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg




@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "position_ids"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        position_ids = [torch.tensor(x) for x in position_ids]
        position_ids = torch.nn.utils.rnn.pad_sequence(position_ids, batch_first=True, padding_value=0)
        return dict(
            input_ids=input_ids,
            labels=labels,
            position_ids=position_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_preprocess_function_randomized(examples, tokenizer, scaled_max_position_embeddings, model_max_position_embeddings):

    inputs = examples["text"]
    model_inputs = tokenizer(inputs, padding=False, truncation=True, max_length=model_max_position_embeddings)
    position_ids = [torch.arange(len(ids), dtype=torch.long) for ids in model_inputs["input_ids"]]

    for pos_ids in position_ids:
        len_pos_ids = len(pos_ids)

        tot_pos_list = list(range(scaled_max_position_embeddings))
        new_pos_list = random.sample(tot_pos_list, len_pos_ids)
        new_pos_list.sort()
        pos_ids[:] = torch.tensor(new_pos_list, dtype=torch.long)

    model_inputs["position_ids"] = position_ids
    model_inputs["labels"] = model_inputs["input_ids"]

    return model_inputs

def train_preprocess_function_pose(examples, tokenizer, scaled_max_position_embeddings, model_max_position_embeddings):

    inputs = examples["text"]
    raw_model_inputs = tokenizer(inputs, padding=False, truncation=True, max_length=scaled_max_position_embeddings)

    input_ids = []
    position_ids = []

    for ids in raw_model_inputs["input_ids"]:

        len_chunk = min(len(ids), model_max_position_embeddings)
        len_input = len(ids)
        lt1 = 0
        rt1 = random.randint(1, (len_chunk+1)//2)
        rt2 = random.randint(lt1+len_chunk, len_input)
        lt2 = rt2 - (len_chunk - (rt1-lt1))
        chunked_ids = ids[lt1:rt1] + ids[lt2:rt2]
        input_ids.append(chunked_ids)

        pos_ids = torch.arange(len(chunked_ids), dtype=torch.long)
        len_pos_ids = len(pos_ids)
        # lt = random.randint(0, scaled_max_position_embeddings-len_pos_ids)
        lt = 0 # this revision makes the coverage possiblity more uniform for large relative positions
        rt = random.randint(lt, scaled_max_position_embeddings-len_pos_ids)

        pos_ids[:rt1-lt1] += lt
        pos_ids[rt1-lt1:] += rt
        position_ids.append(pos_ids)
    
    model_inputs = {"input_ids": input_ids, "position_ids": position_ids, "labels": input_ids}

    return model_inputs

def train_preprocess_function_pi(examples, tokenizer, scaled_max_position_embeddings, model_max_position_embeddings):

    inputs = examples["text"]
    model_inputs = tokenizer(inputs, padding=False, truncation=True, max_length=scaled_max_position_embeddings)
    position_ids = [torch.arange(len(ids), dtype=torch.long) for ids in model_inputs["input_ids"]]
    model_inputs["position_ids"] = position_ids
    model_inputs["labels"] = model_inputs["input_ids"]

    return model_inputs

def test_preprocess_function(examples, tokenizer, inference_length):

    inputs = examples["text"]
    model_inputs = tokenizer(inputs, padding=False, truncation=True, max_length=inference_length)
    position_ids = [torch.arange(len(ids), dtype=torch.long) for ids in model_inputs["input_ids"]]
    model_inputs["position_ids"] = position_ids
    model_inputs["labels"] = model_inputs["input_ids"]

    return model_inputs

              
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir)
    scaled_max_position_embeddings=int(training_args.model_max_position_embeddings * training_args.rope_scaling_factor)
    config.max_position_embeddings=scaled_max_position_embeddings

    if training_args.rope_scaling_type is not None:
        config.rope_scaling={"type": training_args.rope_scaling_type, "factor": training_args.rope_scaling_factor}
        if training_args.rope_scaling_type == "yarn":
            config.rope_scaling["original_max_position_embeddings"] = training_args.model_max_position_embeddings
        
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        config=config,
    )

    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    raw_train_datasets = load_dataset('json', data_files=data_args.train_data_path, split="train", cache_dir=training_args.cache_dir)
    raw_valid_datasets = load_dataset('json', data_files=data_args.valid_data_path, split="train", cache_dir=training_args.cache_dir)
    raw_test_datasets = load_dataset('json', data_files=data_args.test_data_path, split="train", cache_dir=training_args.cache_dir)
    if training_args.local_rank > 0: 
        torch.distributed.barrier()

    train_dataset = raw_train_datasets.map(
        train_preprocess_function_pose,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True, # not args.overwrite_cache
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer, "scaled_max_position_embeddings": scaled_max_position_embeddings, "model_max_position_embeddings": training_args.model_max_position_embeddings}
    )


    valid_dataset = raw_valid_datasets.map(
        test_preprocess_function,
        batched=True,
        batch_size=3000,
        num_proc=1,
        remove_columns=raw_valid_datasets.column_names,
        load_from_cache_file=True, # not args.overwrite_cache
        desc="Running tokenizer on valid dataset",
        fn_kwargs={"tokenizer": tokenizer, "inference_length": training_args.inference_length}
    )

    test_dataset = raw_test_datasets.map(
        test_preprocess_function,
        batched=True,
        batch_size=3000,
        num_proc=1,
        remove_columns=raw_test_datasets.column_names,
        load_from_cache_file=True, # not args.overwrite_cache
        desc="Running tokenizer on test dataset",
        fn_kwargs={"tokenizer": tokenizer, "inference_length": training_args.inference_length}
    )

    if training_args.local_rank == 0:
        torch.distributed.barrier()
    
    if training_args.local_rank == 0:
        print(len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            print(f"Sample {index} of the training set: {train_dataset[index]}.")
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # data_module = dict(eval_dataset=valid_dataset, data_collator=data_collator)
    data_module = dict(train_dataset=train_dataset, eval_dataset=valid_dataset, data_collator=data_collator)

    #Tell Trainer not to attempt DataParallel
    model.is_parallelizable = True
    model.model_parallel = True

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    model.config.use_cache = False

    if training_args.do_train:
        logging.info("*** Start Training ***")
        trainer.train()
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    if training_args.do_eval:
        logging.info("*** Evaluate on valid set***")
        metrics = trainer.evaluate(eval_dataset=valid_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logging.info("*** Evaluate on test set***")
        metrics = trainer.evaluate(eval_dataset=test_dataset)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)




if __name__ == "__main__":
    train()
