#!/bin/bash

# MODEL_NAME examples
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-VL-3B-Instruct"}

export PYTHONPATH=src:$PYTHONPATH

MODEL_PATH=${MODEL_PATH:-"output/lora_vision_test/checkpoint-2764"}
SAVE_MODEL_PATH=${SAVE_MODEL_PATH:-"output/merge_test/checkpoint-2764"}

python src/merge_lora_weights.py \
    --model-path ${MODEL_PATH} \
    --model-base ${MODEL_NAME}  \
    --save-model-path ${SAVE_MODEL_PATH} \
    --safe-serialization