#!/bin/bash
docker run --gpus all --network=host --rm -it \
    -v .:/workspace \
    -v /data/cache/huggingface:/root/.cache/huggingface \
    -v /data/cache/torch:/root/.cache/torch \
    -v /root/nltk_data:/root/nltk_data \
    mm_hal:1.2