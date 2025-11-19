import torch
import json
import os
import argparse
from PIL import Image # 仍然可能需要PIL来处理图像，因为qwen_vl_utils可能返回PIL对象
import sys # 导入sys模块
from tqdm import tqdm

# --- 动态添加 root 目录到 sys.path ---
# 获取当前脚本的目录 (例如: /root/src/eval)
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 src 目录 (例如: /root/src)
src_dir = os.path.dirname(script_dir)
# 获取 root 目录 (例如: /root)
root_dir = os.path.dirname(src_dir)

# 将 root 目录添加到 sys.path
# insert(0, ...) 确保它在搜索路径的开头，具有最高优先级
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# --- sys.path 修改结束 ---


# 从您提供的server.py上下文中导入工具函数
# 确保 'src/utils.py' 和 'qwen_vl_utils.py' 文件在您的Python环境中是可访问的。
# 经过上述 sys.path 修改后，现在 'src.utils' 应该能够正确导入。
try:
    from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
    # 如果 qwen_vl_utils.py 也在 src 目录下，那么它的导入方式也需要调整为 from src.qwen_vl_utils import ...
    # 如果 qwen_vl_utils.py 在 root 目录下，那么 from qwen_vl_utils import ... 即可
    # 假设 qwen_vl_utils.py 在 root 目录下，或者作为一个已安装的包存在
    from qwen_vl_utils import process_vision_info 
except ImportError as e:
    print(f"错误: 无法导入必要的工具模块: {e}")
    print("请检查模块路径和 __init__.py 文件。")
    print(f"当前 sys.path: {sys.path}")
    exit(1) # 如果无法导入关键模块，则退出程序

# 全局变量，用于存储模型、处理器和设备，与server.py的风格保持一致
processor = None
model = None
device = None

def is_video_file(filename):
    """
    检查文件是否为视频文件（虽然您的JSON目前只有图片，但保留此函数以备将来扩展）。
    """
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def load_json_data(json_file_path):
    """
    加载JSON文件中的数据。
    """
    print(f"正在加载JSON数据: {json_file_path}...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"已加载 {len(data)} 条数据。")
    return data

def process_single_entry(entry, image_base_dir, max_new_tokens_arg):
    """
    处理单个JSON条目：构建对话，加载图像，准备模型输入，进行推理并打印结果。
    """
    global processor, model, device # 访问全局变量

    entry_id = entry["id"]
    image_paths_relative = entry["image"]
    conversations_json = entry["conversations"]

    human_instruction_text = ""
    gpt_ground_truth = ""

    # 从JSON的conversations中提取human指令文本和gpt（GT）结果
    for conv in conversations_json:
        if conv["from"] == "human":
            human_instruction_text = conv["value"]
        elif conv["from"] == "gpt":
            gpt_ground_truth = conv["value"]

    # 1. 构建 `conversation` 列表，该列表是 `processor.apply_chat_template` 和 `qwen_vl_utils.process_vision_info` 所期望的格式。
    # 结构示例: [{"role": "user", "content": [{"type": "image", "image": "path/to/img.jpg"}, {"type": "text", "text": "instruction"}]}]
    conversation = []
    user_content = []

    # 将图片路径添加到 `user_content`
    # `process_vision_info` 假定它能够直接从提供的路径加载文件。
    # 因此，我们提供图片的绝对路径。
    for rel_path in image_paths_relative:
        full_image_path = os.path.join(image_base_dir, rel_path)
        if not os.path.exists(full_image_path):
            print(f"警告: 未找到图片文件: {full_image_path}。跳过条目 {entry_id}。")
            return
        user_content.append({"type": "image", "image": full_image_path})
    
    # 将文本指令添加到 `user_content`
    if human_instruction_text:
        user_content.append({"type": "text", "text": human_instruction_text})
    
    conversation.append({"role": "user", "content": user_content})

    # 检查是否有视觉输入，如果JSON中指定了图片但未能加载，则跳过
    if not user_content:
        print(f"警告: 条目 {entry_id} 中没有有效的视觉或文本指令。跳过。")
        return

    # 2. 应用聊天模板并处理视觉信息
    # `processor.apply_chat_template` 会生成带有 <image> token 的最终提示字符串。
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    
    # `qwen_vl_utils.process_vision_info` 会从 `conversation` 结构中提取并预处理图像/视频数据。
    # 它应该返回一个PIL图片列表（或其他适合processor的格式）作为 `image_inputs`。
    image_inputs, video_inputs = process_vision_info(conversation) # 此函数内部应处理文件的加载

    if not image_inputs and not video_inputs:
        print(f"警告: 条目 {entry_id} 中 `process_vision_info` 未能提取到有效的视觉输入。跳过。")
        return

    # 3. 准备模型输入
    # processor 将文本提示和处理后的图像/视频数据结合起来。
    inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

    # 4. 生成模型输出
    with torch.no_grad(): # 在推理时禁用梯度计算以节省内存
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens_arg, # 使用命令行参数指定的最大生成token数
            do_sample=False,    # 关闭采样，使用贪婪解码以获得确定性结果
            pad_token_id=processor.tokenizer.eos_token_id # 确保 pad_token_id 设置正确
        )

    # 5. 解码输出
    # `model.generate` 返回的序列包含输入token。我们需要截取生成的部分。
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_len:]
    model_output = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return entry_id, gpt_ground_truth, model_output.strip()
    # # 6. 打印结果
    # print(f"\n--- 处理条目 ID: {entry_id} ---")
    # print(f"人类指令 (Human Instruction):\n{human_instruction_text}")
    # print(f"模型输出 (Model Output):\n{model_output.strip()}")
    # print(f"真实结果 (Ground Truth / GPT):\n{gpt_ground_truth}")
    # print("-" * 50)


def main():
    global processor, model, device # 声明全局变量以便赋值

    parser = argparse.ArgumentParser(description="评估一个合并后的Qwen2.5-VL模型，使用JSON指令和GPT真实结果。")
    parser.add_argument("--model-path", type=str, required=True, 
                        help="合并后的Qwen2.5-VL模型目录路径。该目录应包含config.json、tokenizer文件和模型权重。")
    parser.add_argument("--json-file-path", type=str, required=True, 
                        help="包含人类指令和真实结果的JSON数据文件路径。")
    parser.add_argument("--image-base-dir", type=str, required=True, 
                        help="图片文件存储的根目录（例如，对于'914/frame_16.jpg'，这应该是'914'的父目录）。")
    parser.add_argument("--model-base", type=str, default=None, 
                        help="可选：用于load_pretrained_model的基模型标识符（例如，'Qwen/Qwen2-VL-7B-Instruct'）。"
                             "如果model_path是一个完整的合并模型，load_pretrained_model可能不需要此参数，"
                             "但为了与提供的工具函数签名兼容，我们将其包含在内。")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="加载模型时使用的设备（例如，'cuda', 'cpu'）。")
    parser.add_argument("--load-8bit", action="store_true", 
                        help="以8位量化加载模型。")
    parser.add_argument("--load-4bit", action="store_true", 
                        help="以4位量化加载模型。")
    parser.add_argument("--disable-flash-attention", action="store_true", 
                        help="禁用Flash Attention（使用默认的注意力机制）。")
    parser.add_argument("--max-new-tokens", type=int, default=512, 
                        help="为每个响应生成的最大新token数量。")
    args = parser.parse_args()

    # 路径存在性检查
    if not os.path.exists(args.model_path):
        print(f"错误: 合并模型路径不存在: {args.model_path}")
        print("请将 --model-path 参数修改为你的实际合并后的模型目录。")
        return
    if not os.path.exists(args.json_file_path):
        print(f"错误: JSON文件路径不存在: {args.json_file_path}")
        print("请将 --json-file-path 参数修改为你的实际JSON数据文件路径。")
        return
    if not os.path.exists(args.image_base_dir):
        print(f"错误: 图片根目录不存在: {args.image_base_dir}")
        print("请将 --image-base-dir 参数修改为你的图片存储根目录。")
        return

    # 初始化环境并使用提供的工具函数加载模型
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    use_flash_attn = not args.disable_flash_attention # 如果禁用Flash Attention，则 use_flash_attn 为 False

    processor, model = load_pretrained_model(
        model_base=args.model_base,
        model_path=args.model_path,
        device_map=args.device, # server.py 中 device_map 和 device 都传递了 args.device
        model_name=model_name,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        device=args.device,
        use_flash_attn=use_flash_attn
    )
    
    # 设置全局设备变量，供 process_single_entry 使用
    device = args.device
    
    # 加载JSON数据
    data = load_json_data(args.json_file_path)

    all_results = []
    # 遍历JSON数据，处理每个条目
    for entry in tqdm(data):
        entry_id, gpt_ground_truth, model_output = process_single_entry(entry, args.image_base_dir, args.max_new_tokens)
        all_results.append({
            "id": entry_id,
            # "human_instruction": entry["conversations"][0]["value"] if entry["conversations"][0]["from"] == "human" else "N/A", # 原始指令也可能有用
            "ground_truth": gpt_ground_truth,
            "model_output": model_output
        })

    # 将所有结果保存到JSON文件
    output_path = os.environ.get("TRAIN_RESULT_PATH", "./output/train_result.json")
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print(f"\n所有 {len(all_results)} 条结果已成功保存到: {output_path}")
    except Exception as e:
        print(f"错误: 无法将结果保存到 {output_path}: {e}")

if __name__ == "__main__":
    main()