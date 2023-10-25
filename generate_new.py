import json
import os

import fire
import torch

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftModel
)

from typing import List
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from prompter import PrompterPersonet

from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def main(
    load_8bit: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout: float = 0.05,
    base_model: str = "model_dir/llama-7b",
    lora_weights: str = "model_save/ddp_8gpus_6bsz/checkpoint-1360/pytorch_model.bin",
    cutoff_length: int = 1000,
    eval_data_path: str = "full_dev_data.json"
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = PrompterPersonet()
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model_state_dict = torch.load(lora_weights)
        set_peft_model_state_dict(model, model_state_dict)
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 0
    model.config.eos_token_id = 0

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()

    def evaluate(
        snippet_underlined,
        snippet_former_context,
        snippet_post_context,
        history,
        book_name,
        options,
        character,
        answer,
        gt,
        key_id,
        temperature=0.0,
        top_p=1,
        top_k=40,
        num_beams=1,
        max_new_tokens=10,
        cutoff_len=1000,
        **kwargs,
    ):
        prompt1, prompt2, prompt3 = prompter.generate_prompt(snippet_underlined, snippet_former_context, snippet_post_context, history, book_name, options, character)
        prompt2 = prompt2.split(' ')[:cutoff_len]
        prompt2 = ' '.join(prompt2)
        prompt = prompt1 + prompt2 + prompt3
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        # inputs = torch.tensor(prompt, dtype=torch.long)
        # input_ids = inputs.to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)

        pred = output

        return [pred, answer+' '+gt, key_id]

    # [Attention] change the data path
    eval_data = json.load(open('personet_data/' + eval_data_path, 'r', encoding='utf-8'))

    save_path = '/'.join(lora_weights.split('/')[:-2]) + '/' + eval_data_path.replace('full', 'gen')
    save_list = []
    for data_point in tqdm(eval_data):
        pred_answer = evaluate(data_point["snippet_underlined"], data_point["snippet_former_context"],
                               data_point["snippet_post_context"], data_point["history"],
                               data_point["book_name"], data_point["options"], data_point["character"],
                               data_point['answer'], data_point['gt'],
                               key_id=data_point['key'], cutoff_len=cutoff_length)
        save_list.append(pred_answer)
    json.dump(save_list, open(save_path, 'w'), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)
