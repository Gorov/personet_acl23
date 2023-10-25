import os
import sys
import json
from typing import List, Dict, Optional, Sequence
from torch.utils.data import Dataset

import fire
import torch
import torch.distributed as dist
import transformers
# from datasets import load_dataset
import subprocess

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from prompter import PrompterPersonet


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


class SupervisedDataset(Dataset):
    def __init__(self, data_path, tokenizer, train_on_inputs, cutoff_len):
        super(SupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.train_on_inputs = train_on_inputs
        self.data = json.load(open(data_path, 'r', encoding='utf-8'))
        self.prompter = PrompterPersonet()
        self.cutoff_len = cutoff_len

    def tokenize(self, prompt1, prompt2, prompt3, add_eos_token=True):

        prompt2 = prompt2.split(' ')[:self.cutoff_len]
        prompt2 = ' '.join(prompt2)

        prompt = prompt1 + prompt2 + prompt3

        result = self.tokenizer(prompt, padding=False, return_tensors=None)

        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(self, data_point):
        prompt1, prompt2, prompt3 = self.prompter.generate_prompt(data_point["snippet_underlined"], data_point["snippet_former_context"],
                                                                  data_point["snippet_post_context"], data_point["history"],
                                                                  data_point["book_name"], data_point["options"], data_point["character"],
                                                                  data_point["answer"], data_point["gt"])
        tokenized_full_prompt = self.tokenize(prompt1, prompt2, prompt3)
        if not self.train_on_inputs:
            u_prompt1, u_prompt2, u_prompt3 = self.prompter.generate_prompt(data_point["snippet_underlined"], data_point["snippet_former_context"],
                                                                            data_point["snippet_post_context"], data_point["history"],
                                                                            data_point["book_name"], data_point["options"], data_point["character"])
            tokenized_user_prompt = self.tokenize(
                u_prompt1, u_prompt2, u_prompt3, add_eos_token=False
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            # if add_eos_token:
            #     user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    def __getitem__(self, item):
        if dist.get_rank() == 0:
            if item % (48 * 5) == 0:
                sp = subprocess.Popen(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out_str = sp.communicate()
                for out_element in out_str:
                    for line in str(out_element).split('\\n'):
                        print(line, file=sys.stderr)
        return self.generate_and_tokenize_prompt(self.data[item])

    def __len__(self):
        return len(self.data)


def train(
    # model/data params
    base_model: str = "model_dir/llama-7b",
    train_data_path: str = "personet_data/new_train_data.json",
    dev_data_path: str = "personet_data/dev_data.json",
    output_dir: str = "personet_model_save/ddp_4gpus_6bsz",
    # training hyperparams
    batch_size: int = 192,
    micro_batch_size: int = 6,
    num_epochs: int = 5,
    learning_rate: float = 1e-4,
    cutoff_len: int = 1000,
    warmup_steps: int = 100,
    eval_steps: int = 230,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss (False)
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve (not used)
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter (not used)
):
    my_dir = ""    # abs dir path storing pre-downloaded llama
    base_dir = ""   # base dir path storing the training and eval data
    base_model = '/'.join([my_dir, base_model])
    train_data_path = '/'.join([base_dir, train_data_path])
    dev_data_path = '/'.join([base_dir, dev_data_path])
    output_dir = '/'.join([base_dir, output_dir])

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    master_addr = os.environ.get("MASTER_ADDR", "")
    master_port = os.environ.get("MASTER_PORT", "")

    ddp = world_size != 1

    if ddp:
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

    # local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    gradient_accumulation_steps = batch_size // micro_batch_size
    device_map = "auto"
    if ddp:
        device_map = {"": device}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if local_rank == 0:
        print('local_rank:', local_rank)
        print('device:', device)
        print('master addr:', master_addr)
        print('master port:', master_port)
        print('ddp:', ddp)
        print(
            f"Training PERSONET-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"train_data_path: {train_data_path}\n"
            f"dev_data_path: {dev_data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"gradient_accumulation_steps: {gradient_accumulation_steps}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"warmup_steps: {warmup_steps}\n"
            f"eval_steps (save_steps): {eval_steps}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        )

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    train_data = SupervisedDataset(train_data_path, tokenizer, train_on_inputs, cutoff_len)
    val_data = SupervisedDataset(dev_data_path, tokenizer, train_on_inputs, cutoff_len)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name, map_location='cpu')
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if local_rank == 0:
        sp = subprocess.Popen(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_str = sp.communicate()
        for out_element in out_str:
            for line in str(out_element).split('\\n'):
                print(line, file=sys.stderr)

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_steps,
            save_steps=eval_steps,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
            run_name=None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(model, type(model))

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if local_rank == 0:
        print(
            "\n Finished :)"
        )


if __name__ == "__main__":
    fire.Fire(train)
