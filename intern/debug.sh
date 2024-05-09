#!/bin/bash
PWD=`pwd`
export PYTHONPATH=$PWD:$PWD/intern:$PYTHONPATH
export MODEL="internlm/internlm-xcomposer2-4khd-7b"
export DATA="dataset/intern/data.json"
export CUDA_HOME=/home/nfs04/cuda_tools/cuda-12.1

python -m pdb finetune/evaluate.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --img_size 490 \
    --hd_num 16 \
    --given_num True \
    --use_lora False \
    --output_dir . \
    --batch_size 1 \
    --max_length 4096 \