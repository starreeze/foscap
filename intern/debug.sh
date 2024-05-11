#!/bin/bash
PWD=`pwd`
export PYTHONPATH=$PWD:$PWD/intern:$PYTHONPATH
export CUDA_HOME=/home/nfs04/cuda_tools/cuda-12.1

export MODEL="ckpt/intern/lora"
export DATA="dataset/intern/data.json"

python -m pdb finetune/evaluate.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --img_size 490 \
    --hd_num 16 \
    --given_num True \
    --use_lora True \
    --output_dir . \
    --batch_size 1 \
    --max_length 4096 \