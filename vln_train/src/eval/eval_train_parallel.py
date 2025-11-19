import torch
import json
import os
import argparse
from PIL import Image
import sys
from tqdm import tqdm
import multiprocessing # 导入多进程模块
import uuid # 用于生成唯一的临时文件名

# --- 动态添加 root 目录到 sys.path ---
# 获取当前脚本的目录 (例如: /root/src/eval)
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 src 目录 (例如: /root/src)
src_dir = os.path.dirname(script_dir)
# 获取 root 目录 (例如: /root)
root_dir = os.path.dirname(src_dir)

# 将 root 目录添加到 sys.path
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# --- sys.path 修改结束 ---


# 从您提供的server.py上下文中导入工具函数
try:
    from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
    from qwen_vl_utils import process_vision_info 
except ImportError as e:
    print(f"错误: 无法导入必要的工具模块: {e}")
    print("请检查模块路径和 __init__.py 文件。")
    print(f"当前 sys.path: {sys.path}")
    exit(1)

# 注意：processor, model, device 不再是全局变量，每个进程会独立加载。

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

def process_single_entry(entry, image_base_dir, max_new_tokens_arg, processor, model, device):
    """
    处理单个JSON条目：构建对话，加载图像，准备模型输入，进行推理。
    返回 entry_id, gpt_ground_truth, model_output。
    """
    entry_id = entry["id"]
    image_paths_relative = entry["image"]
    conversations_json = entry["conversations"]

    human_instruction_text = ""
    gpt_ground_truth = ""

    for conv in conversations_json:
        if conv["from"] == "human":
            human_instruction_text = conv["value"]
        elif conv["from"] == "gpt":
            gpt_ground_truth = conv["value"]

    conversation = []
    user_content = []

    for rel_path in image_paths_relative:
        full_image_path = os.path.join(image_base_dir, rel_path)
        if not os.path.exists(full_image_path):
            print(f"警告: 未找到图片文件: {full_image_path}。跳过条目 {entry_id}。")
            return None, None, None
        user_content.append({"type": "image", "image": full_image_path})
    
    if human_instruction_text:
        user_content.append({"type": "text", "text": human_instruction_text})
    
    conversation.append({"role": "user", "content": user_content})

    if not user_content:
        print(f"警告: 条目 {entry_id} 中没有有效的视觉或文本指令。跳过。")
        return None, None, None

    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversation)

    if not image_inputs and not video_inputs:
        print(f"警告: 条目 {entry_id} 中 `process_vision_info` 未能提取到有效的视觉输入。跳过。")
        return None, None, None

    inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens_arg,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_len:]
    model_output = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    return entry_id, gpt_ground_truth, model_output


def worker_process_inference(gpu_id, data_chunk, args, temp_output_file):
    """
    在单个GPU上执行推理的子进程函数。
    """
    # 设置当前进程可见的CUDA设备
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    current_device = f"cuda:{gpu_id}" # 在子进程中，它看到的cuda:0就是实际的gpu_id

    print(f"子进程 (GPU {gpu_id}) 正在启动，处理 {len(data_chunk)} 条数据。")

    # 在每个子进程中重新加载模型、tokenizer和processor
    # 确保每个进程都有自己的模型实例，并加载到正确的GPU
    disable_torch_init() # 再次禁用torch初始化，以防万一
    model_name = get_model_name_from_path(args.model_path)
    use_flash_attn = not args.disable_flash_attention

    processor_local, model_local = load_pretrained_model(
        model_base=args.model_base,
        model_path=args.model_path,
        device_map=current_device, # 明确指定设备
        model_name=model_name,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        device=current_device, # 明确指定设备
        use_flash_attn=use_flash_attn
    )
    # 强制将模型移到指定设备，如果 device_map="auto" 没有完全处理
    model_local.to(current_device)


    chunk_results = []
    # 使用 tqdm 显示每个子进程的进度
    for entry in tqdm(data_chunk, desc=f"GPU {gpu_id} Progress"):
        entry_id, gpt_ground_truth, model_output = process_single_entry(
            entry, args.image_base_dir, args.max_new_tokens, 
            processor_local, model_local, current_device
        )
        
        if entry_id is not None:
            chunk_results.append({
                "id": entry_id,
                "human_instruction": entry["conversations"][0]["value"] if entry["conversations"][0]["from"] == "human" else "N/A",
                "ground_truth": gpt_ground_truth,
                "model_output": model_output
            })
    
    # 将此块的结果保存到临时文件
    try:
        with open(temp_output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_results, f, ensure_ascii=False, indent=4)
        print(f"子进程 (GPU {gpu_id}) 完成，结果保存到 {temp_output_file}")
    except Exception as e:
        print(f"子进程 (GPU {gpu_id}) 错误: 无法将结果保存到 {temp_output_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="评估一个合并后的Qwen2.5-VL模型，使用JSON指令和GPT真实结果，并将结果保存到JSON文件。")
    parser.add_argument("--model-path", type=str, required=True, 
                        help="合并后的Qwen2.5-VL模型目录路径。")
    parser.add_argument("--json-file-path", type=str, required=True, 
                        help="包含人类指令和真实结果的JSON数据文件路径。")
    parser.add_argument("--image-base-dir", type=str, required=True, 
                        help="图片文件存储的根目录。")
    parser.add_argument("--output-json-path", type=str, required=True,
                        help="保存评估结果的JSON文件路径。")
    parser.add_argument("--model-base", type=str, default=None, 
                        help="可选：用于load_pretrained_model的基模型标识符。")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="加载模型时使用的设备（例如，'cuda', 'cpu'）。")
    parser.add_argument("--load-8bit", action="store_true", 
                        help="以8位量化加载模型。")
    parser.add_argument("--load-4bit", action="store_true", 
                        help="以4位量化加载模型。")
    parser.add_argument("--disable-flash-attention", action="store_true", 
                        help="禁用Flash Attention。")
    parser.add_argument("--max-new-tokens", type=int, default=512, 
                        help="为每个响应生成的最大新token数量。")
    parser.add_argument("--num-gpus", type=int, default=1, 
                        help="用于并行处理的GPU数量。设置为0则尝试使用所有可用GPU。")
    args = parser.parse_args()

    # 路径存在性检查
    if not os.path.exists(args.model_path):
        print(f"错误: 合并模型路径不存在: {args.model_path}")
        return
    if not os.path.exists(args.json_file_path):
        print(f"错误: JSON文件路径不存在: {args.json_file_path}")
        return
    if not os.path.exists(args.image_base_dir):
        print(f"错误: 图片根目录不存在: {args.image_base_dir}")
        return
    
    output_dir = os.path.dirname(args.output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"已创建输出目录: {output_dir}")

    # 确保在主进程中禁用torch的初始化
    disable_torch_init()

    # 加载JSON数据
    data = load_json_data(args.json_file_path)

    # 确定要使用的GPU数量
    if args.device == "cuda" and torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        if args.num_gpus == 0: 
            num_gpus_to_use = available_gpus
        else:
            num_gpus_to_use = min(args.num_gpus, available_gpus)
    else:
        num_gpus_to_use = 1 
    
    if num_gpus_to_use == 0:
        print("没有可用的GPU，将使用CPU进行处理。")
        num_gpus_to_use = 1 

    print(f"将使用 {num_gpus_to_use} 个GPU进行并行处理。")

    # 分割数据
    data = data[0:1000]
    chunks = [data[i::num_gpus_to_use] for i in range(num_gpus_to_use)]
    
    processes = []
    temp_output_files = []

    # ------------------ Ctrl+C 异常处理核心逻辑 ------------------
    try:
        for i in range(num_gpus_to_use):
            # 为每个子进程创建一个唯一的临时文件路径
            temp_file = os.path.join(output_dir if output_dir else ".", f"temp_results_{uuid.uuid4().hex}.json")
            temp_output_files.append(temp_file)
            
            p = multiprocessing.Process(
                target=worker_process_inference, 
                args=(i, chunks[i], args, temp_file)
            )
            processes.append(p)
            p.start()

        # 等待所有进程完成
        for p in processes:
            p.join()

        print("\n所有子进程已完成。正在合并结果...")

    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C (KeyboardInterrupt)。正在尝试保存已完成的结果...")
        # 终止所有仍在运行的子进程
        for p in processes:
            if p.is_alive():
                print(f"终止子进程 {p.pid}...")
                p.terminate() # 发送 SIGTERM 信号
            p.join(timeout=1) # 等待子进程终止，设置一个短超时
        
        # 注意：子进程被 terminate() 可能会导致其临时文件不完整或未写入
        # 只有在子进程完成了其数据块处理并写入了临时文件，或者子进程自身有增量保存逻辑，
        # 这里的合并才能收集到其部分或全部结果。
        
    finally:
        # 无论是否发生异常，都尝试合并和保存已有的临时文件
        final_results = []
        for temp_file in temp_output_files:
            if os.path.exists(temp_file):
                try:
                    with open(temp_file, 'r', encoding='utf-8') as f:
                        chunk_data = json.load(f)
                        final_results.extend(chunk_data)
                    print(f"已从临时文件 {temp_file} 加载 {len(chunk_data)} 条结果。")
                except json.JSONDecodeError:
                    print(f"警告: 临时文件 {temp_file} 不是一个有效的JSON文件，可能子进程被中断时未完成写入。跳过此文件。")
                except Exception as e:
                    print(f"读取临时文件 {temp_file} 时发生错误: {e}")
            else:
                print(f"警告: 未找到临时文件 {temp_file}，可能子进程未能成功保存结果或已自行清理。")
        
        if final_results:
            # 将所有结果保存到最终的JSON文件
            try:
                with open(args.output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(final_results, f, ensure_ascii=False, indent=4)
                print(f"\n已成功保存 {len(final_results)} 条已处理结果到: {args.output_json_path}")
            except Exception as e:
                print(f"错误: 无法将最终结果保存到 {args.output_json_path}: {e}")
        else:
            print("\n没有可用的已处理结果可以保存。")

        # 清理临时文件
        for temp_file in temp_output_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"已删除临时文件: {temp_file}")
        
        if KeyboardInterrupt: # 如果是因为Ctrl+C退出，则明确退出程序
            sys.exit(0)

if __name__ == "__main__":
    # 使用 multiprocessing.set_start_method 避免一些在某些系统上的问题
    # 'spawn' 是更安全的选择，但可能导致模型加载速度稍慢
    # 在 __main__ 保护块中设置，以避免在子进程中重复设置
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # 如果已经设置过，则忽略
    main()