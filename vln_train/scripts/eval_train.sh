MODEL_PATH=${MODEL_PATH:-"output/merge_test/checkpoint-2949"}
JSON_FILE_PATH=${JSON_FILE_PATH:-"data/Simplified_Dataset/R2R/data.json"}
IMAGE_BASE_DIR=${IMAGE_BASE_DIR:-"data/Simplified_Dataset/R2R/train"}
OUTPUT_JSON_PATH=${OUTPUT_JSON_PATH:-"output/eval/evaluation_results.json"}

python src/eval/eval_train_parallel.py \
    --model-path ${MODEL_PATH} \
    --json-file-path ${JSON_FILE_PATH} \
    --image-base-dir ${IMAGE_BASE_DIR} \
    --output-json-path ${OUTPUT_JSON_PATH} \
    --num-gpus 5 
    # 指定使用5个GPU (cuda:0 到 cuda:4)
    # --num-gpus 0 \  # 或者使用所有可用GPU
    # --load-4bit  # 如果需要4bit量化
    # --disable-flash-attention # 如果需要禁用Flash Attention
    # --max-new-tokens 2048 # 根据需要调整最大生成token数