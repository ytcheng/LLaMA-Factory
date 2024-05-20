#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    src/train.py examples/sm/llama3_70b_lora_v2_pretrain_bitsandbytes.yaml