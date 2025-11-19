import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import json
import os
import re

# --- Config (can be overridden by env variables) ---
# Use environment variables MODEL_PATH, JSON_FILE_PATH, IMAGE_BASE_DIR to override defaults
MODEL_PATH = os.environ.get("MODEL_PATH", "./output/merge_test/checkpoint-2949_v1")
JSON_FILE_PATH = os.environ.get("JSON_FILE_PATH", "./data/Simplified_Dataset/R2R/data.json")
IMAGE_BASE_DIR = os.environ.get("IMAGE_BASE_DIR", "./data/Simplified_Dataset/R2R/train")

# --- 函数定义 ---
def load_merged_qwen_model(model_path):
    """
    加载已合并的Qwen2.5-VL模型及其对应的tokenizer和processor。
    """
    print(f"正在加载合并后的模型: {model_path}...")
    # trust_remote_code=True 是加载Qwen模型所必需的
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # 建议使用 torch.bfloat16，如果你的GPU不支持，可以使用 torch.float16
    # device_map="auto" 会自动将模型分发到可用的GPU上
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    print("模型已成功加载。")

    # 诊断代码
    print("--- 开始 Tokenizer 诊断 ---")
    image_token_str = "<image>"
    tokenized_output = tokenizer.tokenize(image_token_str)
    print(f"tokenizer.tokenize('{image_token_str}') 的输出: {tokenized_output}")

    # 正确的输出应该是一个单独的 token: ['<image>']
    # 错误的输出可能是被切分了: ['<', 'image', '>'] 或其他

    try:
        token_id = tokenizer.convert_tokens_to_ids(image_token_str)
        print(f"'{image_token_str}' 对应的 token ID: {token_id}")
        if token_id == tokenizer.unk_token_id:
            print("警告: '<image>' 被识别为未知 token (unk_token)!")
    except Exception as e:
        print(f"尝试转换 '<image>' 为 ID 时出错: {e}")

    print("--- 结束 Tokenizer 诊断 ---")


    model.eval() # 设置为评估模式，关闭dropout等
    return tokenizer, processor, model

def load_json_data(json_file_path):
    """
    加载JSON文件中的数据。
    """
    print(f"正在加载JSON数据: {json_file_path}...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"已加载 {len(data)} 条数据。")
    return data

def process_single_entry(entry, tokenizer, processor, model, image_base_dir):
    """
    处理单个JSON条目：加载图片，使用chat_template准备模型输入，进行推理并打印结果。
    (增加了对空指令的处理和修复了生成参数警告)
    """
    entry_id = entry["id"]
    image_paths_relative = entry["image"]
    conversations = entry["conversations"]

    # 1. 提取数据
    raw_human_instruction = ""
    gpt_ground_truth = ""
    for conv in conversations:
        if conv["from"] == "human":
            raw_human_instruction = conv["value"].replace("<image>", "").strip()
        elif conv["from"] == "gpt":
            gpt_ground_truth = conv["value"]

    # *** 核心修复 1: 增加对空指令的防御性检查 ***
    # 如果没有有效的文本指令，模型内部处理位置编码时会出错。
    if not raw_human_instruction:
        print(f"警告: 条目 {entry_id} 的文本指令为空。跳过此条目。")
        # return
    print(raw_human_instruction)
    # 2. 加载图片
    images = []
    # (您的图片加载代码保持不变)
    for rel_path in image_paths_relative:
        full_image_path = os.path.join(image_base_dir, rel_path)
        try:
            image = Image.open(full_image_path).convert("RGB")
            images.append(image)
        except FileNotFoundError:
            print(f"警告: 未找到图片文件: {full_image_path}。跳过当前条目 {entry_id}。")
            return
        except Exception as e:
            print(f"警告: 处理图片 {full_image_path} 时发生错误: {e}。跳过当前条目 {entry_id}。")
            return

    # 3. 构建结构化的对话列表 (messages)
    messages = [{"role": "user", "content": []}]
    for _ in images:
        messages[0]["content"].append({"type": "image"})
    messages[0]["content"].append({"type": "text", "text": raw_human_instruction})

    # 4. 使用 apply_chat_template 生成最终的文本 prompt
    text_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 5. 准备最终的模型输入
    inputs = processor(
        text=text_prompt, 
        images=images, 
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # (可选) 用于未来调试的打印语句
    # print(f"--- DEBUG INFO for entry {entry_id} ---")
    # print("Final text prompt:", text_prompt)
    # print("Input IDs shape:", inputs['input_ids'].shape)
    # print("Decoded Input:", tokenizer.decode(inputs['input_ids'][0]))
    # print("-" * 20)

    # 6. 生成模型输出
    with torch.no_grad():
        # *** 核心修复 2: 移除 temperature 参数以解决 UserWarning ***
        # 当 do_sample=False (贪婪解码) 时，temperature 参数是不需要的。
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # 7. 解码并打印结果
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_len:]
    model_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print(f"\n--- 处理条目 ID: {entry_id} ---")
    print(f"人类指令 (Human Instruction):\n{human_instruction}")
    print(f"模型输出 (Model Output):\n{model_output.strip()}")
    print(f"真实结果 (Ground Truth / GPT):\n{gpt_ground_truth}")
    print("-" * 50)


def main():
    # 检查配置路径是否存在，并给出提示
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 合并模型路径不存在: {MODEL_PATH}")
        print("请将 MODEL_PATH 修改为你的实际合并后的模型目录。")
        return
    if not os.path.exists(JSON_FILE_PATH):
        print(f"错误: JSON文件路径不存在: {JSON_FILE_PATH}")
        print("请将 JSON_FILE_PATH 修改为你的实际JSON数据文件路径。")
        return
    if not os.path.exists(IMAGE_BASE_DIR):
        print(f"错误: 图片根目录不存在: {IMAGE_BASE_DIR}")
        print("请将 IMAGE_BASE_DIR 修改为你的图片存储根目录。")
        return

    # 加载模型、tokenizer和processor
    tokenizer, processor, model = load_merged_qwen_model(MODEL_PATH)
    # 加载JSON数据
    data = load_json_data(JSON_FILE_PATH)

    # 遍历JSON数据，处理每个条目
    for entry in data[0:2]:
        process_single_entry(entry, tokenizer, processor, model, IMAGE_BASE_DIR)

if __name__ == "__main__":
    main()