#!/bin/bash

# You can use 2B instead of 7B
# MODEL_NAME examples:
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME could also be a local path to your base model directory
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-VL-3B-Instruct"}

export PYTHONPATH=src:$PYTHONPATH

# Configure paths via env vars or fallbacks
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
export CUDA_VISIBLE_DEVICES

GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-4}
NUM_DEVICES=${NUM_DEVICES:-1}
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}

# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together
# You should freeze the the merger also, becuase the merger is included in the vision_tower.

DATA_PATH=${DATA_PATH:-"data/train.json"}
IMAGE_FOLDER=${IMAGE_FOLDER:-"data/images"}
OUTPUT_DIR=${OUTPUT_DIR:-"output/lora_vision_test"}

deepspeed src/train/train_sft.py \
    --use_liger True \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((512 * 512)) \
    --image_max_pixels $((512 * 512)) \
    --learning_rate 2e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4