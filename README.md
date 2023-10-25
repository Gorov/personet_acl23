# Personality Understanding of Fictional Characters during Book Reading

## requirements
* torch
* transformers
* spacy
* [LASER](https://github.com/facebookresearch/LASER)
* [vecalign](https://github.com/thompsonb/vecalign)
* peft
* bitsandbytes
* part of the code is based on [alpaca-lora](https://github.com/tloen/alpaca-lora)

## Data Format

## Training Longformer and MultiRow BERT

pending...

## Efficient Training LLaMA with LoRA

train (N x A100 GPUs of 40G):
```
python -m torch.distributed.launch --nproc_per_node 8 --nnodes=1 --node_rank=0 finetune.py \
    --micro_batch_size 4 \
    --batch_size 128 \
    --output_dir 'personet_model_save/ddp_8gpus (your own path)' \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
    --num_epochs 5 \
    --learning_rate 1e-4 \
    --warmup_steps 170 \
    --cutoff_len 1000 \
    --eval_steps 340
```

generate on dev/test data (dev: full_dev_data.json; test: full_test_data.json; 1 A100 GPU of 40G):
```
python generate_new.py \
    --load_8bit \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
    --lora_weights '(your own path)' \
    --eval_data_path 'full_test_data.json'
```
