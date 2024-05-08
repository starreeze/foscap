#!/bin/bash

# export MODEL="internlm/internlm-xcomposer2-7b"
# export MODEL="internlm/internlm-xcomposer2-vl-7b"
export MODEL="internlm/internlm-xcomposer2-4khd-7b"
export DATA="dataset/intern/data.json"
export WANDB_MODE=offline

python -m pdb intern/finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --img_size 490 \
    --hd_num 8 \
    --given_num True \
    --bf16 True \
    --fix_vit True \
    --fix_sampler True \
    --use_lora True \
    --output_dir output/finetune \
    --num_train_epochs 2 \
    --batch_size 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "wandb" \
    --max_length 4096 \
    --gradient_checkpointing True