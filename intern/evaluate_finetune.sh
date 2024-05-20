#!/bin/bash
PWD=`pwd`
export PYTHONPATH=$PWD:$PWD/intern:$PYTHONPATH
export CUDA_HOME=/home/nfs04/cuda_tools/cuda-12.1

LR=$1

python intern/evaluate.py \
    --model_name_or_path ckpt/intern/lora-$LR \
    --given_num True \
    --use_lora True \
    --output_dir eval/lora-$LR \
    --batch_size 1 \
    --max_length 4096 \