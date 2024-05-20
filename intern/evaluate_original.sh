#!/bin/bash
PWD=`pwd`
export PYTHONPATH=$PWD:$PWD/intern:$PYTHONPATH
export CUDA_HOME=/home/nfs04/cuda_tools/cuda-12.1

python intern/evaluate.py \
    --given_num True \
    --use_lora False \
    --output_dir eval/original \
    --batch_size 1 \
    --use_meta_inst True