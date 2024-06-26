#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

# export MODEL="internlm/internlm-xcomposer2-7b"
# export MODEL="internlm/internlm-xcomposer2-vl-7b"
export MODEL="internlm/internlm-xcomposer2-4khd-7b"
export DATA="dataset/intern/data.json"
export WANDB_MODE=offline
export WANDB__SERVICE_WAIT=300
export CUDA_HOME=/home/nfs04/cuda_tools/cuda-12.1

GPUS_PER_NODE=4
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

python -m torch.distributed.run $DISTRIBUTED_ARGS intern/finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --img_size 490 \
    --hd_num 16 \
    --given_num True \
    --bf16 True \
    --fix_vit False \
    --fix_sampler False \
    --use_lora False \
    --output_dir ckpt/intern/full \
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
    --deepspeed /home/nfs04/xingsy/foscap/intern/zero3_offload.json \
    --gradient_checkpointing True