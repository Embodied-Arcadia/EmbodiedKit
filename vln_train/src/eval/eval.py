#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评测脚本：加载 Qwen2.5-VL 基座模型 + LoRA 适配器，对多模态指令数据集生成回答并保存输出。

使用示例:
python eval.py \
    --base_model /path/to/your/Qwen2.5-VL/Qwen2.5-VL-3B-Instruct \
    --lora_path output/lora_vision_test \
    --eval_path /path/to/eval.jsonl \
    --eval_image_folder /path/to/images_or_frames \
    --output_dir output/eval_lora_vision_test \
    --max_new_tokens 512 --temperature 0.2 --top_p 0.9

或使用 bash: 见 eval.sh
"""

import os
import json
import argparse
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

# LoRA
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

# 与训练脚本相同的 monkey patch / liger kernel
from monkey_patch_forward import replace_qwen2_5_with_mixed_modality_forward, replace_qwen_2_with_mixed_modality_forward
from monkey_patch_vision import replace_qwen2_5_vision
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl, apply_liger_kernel_to_qwen2_5_vl


# ===================== 数据加载与处理 =====================

def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            data = obj
        else:
            raise ValueError("JSON 顶层需为数组")
    else:
        raise ValueError("评测文件需为 .json 或 .jsonl")
    return data


class EvalDataset(Dataset):
    """
    假设样本结构:
    {
      "id": "xxx",
      "conversations": [
         {"from":"human","value":"<image>\n问题..."},
         {"from":"gpt","value":"(可为空或参考答案)"},
         ...
      ],
      "image": "relpath.jpg"   # 可选
      "images": ["a.jpg","b.jpg"]  # 可选
    }
    你可根据真实数据格式在本类或 build_sample_io中调整。
    """
    def __init__(self, data: List[Dict[str, Any]], image_root: Optional[str] = None):
        self.data = data
        self.image_root = image_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def build_sample_io(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    提取:
    - images_paths: List[str]
    - conversations: List[Dict]
    - ref_answer: Optional[str]
    - target_turn_index: 需要生成的 assistant turn index (如果存在)
    """
    conversations = sample.get("conversations", [])
    # 找到最后一个需要生成的 gpt turn（value 为空或占位）
    target_turn_index = None
    for i in reversed(range(len(conversations))):
        turn = conversations[i]
        if turn.get("from", "").lower() in ("gpt", "assistant"):
            # 只在其 value 为空 / 长度很短 / 特定占位时生成
            val = (turn.get("value") or "").strip()
            if len(val) == 0 or val in ["", " ", "<answer>", "[EMPTY]"]:
                target_turn_index = i
                break

    # 参考答案（如果有且非空）
    ref_answer = None
    if target_turn_index is not None:
        # 有时参考答案不在需要生成的 turn，而在另一个字段；这里仅示例
        pass
    else:
        # 如果所有 gpt turn 都有值，可以把最后一个 gpt 当参考答案（并仍然生成? 这里示例不生成，直接跳过）
        # 但我们更常见场景：target_turn_index != None
        pass

    # 图像路径收集
    images_paths = []
    if "images" in sample and isinstance(sample["images"], list):
        images_paths.extend(sample["images"])
    if "image" in sample and isinstance(sample["image"], str):
        images_paths.append(sample["image"])

    return {
        "images_paths": images_paths,
        "conversations": conversations,
        "ref_answer": ref_answer,
        "target_turn_index": target_turn_index
    }


def join_conversations_as_text(conversations: List[Dict[str, str]], target_turn_index: Optional[int]) -> str:
    """
    将对话拼接为纯文本（仅用于 fallback/调试）。
    若实际需要多轮格式 / 特定模板（例如 <|im_start|> 之类）请在此实现。
    """
    parts = []
    for i, turn in enumerate(conversations):
        role = turn.get("from", "")
        val = turn.get("value") or ""
        if i == target_turn_index:
            # 这是要生成的 assistant turn，跳过内容
            continue
        role_tag = "User" if role.lower() in ("human", "user") else "Assistant"
        parts.append(f"{role_tag}: {val}")
    parts.append("Assistant:")  # 让模型续写
    return "\n".join(parts)


# ===================== 模型准备 =====================

def patch_and_load_model(base_model: str, use_liger: bool, dtype: str):
    """
    根据 base_model 名称决定使用 Qwen2 或 Qwen2.5 分支，并应用 monkey patch 和 liger。
    """
    if "Qwen2.5" in base_model or "Qwen2.5" in os.path.basename(base_model):
        replace_qwen2_5_vision()
        replace_qwen2_5_with_mixed_modality_forward(use_liger=use_liger)
        if use_liger:
            apply_liger_kernel_to_qwen2_5_vl(fused_linear_cross_entropy=False)
        ModelClass = Qwen2_5_VLForConditionalGeneration
    else:
        replace_qwen_2_with_mixed_modality_forward(use_liger=use_liger)
        if use_liger:
            apply_liger_kernel_to_qwen2_vl(fused_linear_cross_entropy=False)
        ModelClass = Qwen2VLForConditionalGeneration

    torch_dtype = torch.bfloat16 if dtype == "bf16" else (torch.float16 if dtype == "fp16" else torch.float32)

    model = ModelClass.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    model.eval()
    return model


def maybe_load_lora(model, lora_path: Optional[str]):
    if not lora_path:
        return model
    if not PEFT_AVAILABLE:
        raise RuntimeError("未安装 peft，无法加载 LoRA 适配器")
    # 加载 LoRA 适配器
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model


# ===================== 推理 =====================

@torch.inference_mode()
def generate_one(
    model,
    processor,
    sample: Dict[str, Any],
    sample_io: Dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    repetition_penalty: float
) -> str:
    """
    针对单条样本生成:
    - 若 target_turn_index 存在，则以之前对话为上下文生成
    - 同时处理图像（单图或多图）
    """
    conversations = sample_io["conversations"]
    target_turn_index = sample_io["target_turn_index"]

    if target_turn_index is None:
        # 没有需要生成的空位，直接返回空或最后一个 gpt 内容
        # 为统一，这里直接返回空字符串
        return ""

    # 构建多轮对话输入
    # Qwen2-VL 使用的 processor 支持列表形式的 multimodal输入（参照官方pattern）:
    # 例如 [{"role":"user","content":[{"type":"image","image":image}, {"type":"text","text":"..."}]}, {"role":"assistant","content":[{"type":"text","text":"..."}]}]
    # 这里尝试把已有 conversations 转换；如果你的训练阶段使用了不同字段/格式，请自行对齐。

    multi_turn_messages = []
    # 收集所有需要的图像对象
    images_to_load = sample_io["images_paths"]

    # 简单策略：把所有图像放到第一条 human 消息里（若 dataset 原本这样处理），否则附加到各自包含 <image> 标记的位置。
    # 根据训练时的实际逻辑进行修改。
    # 映射: "human"/"user" -> "user", "gpt"/"assistant" -> "assistant"
    for i, turn in enumerate(conversations):
        role_raw = turn.get("from", "").lower()
        if role_raw in ("human", "user"):
            role = "user"
        else:
            role = "assistant"

        val = (turn.get("value") or "")
        # 如果是需要模型生成的那条 assistant，跳过其内容
        if i == target_turn_index:
            continue

        content_items = []
        # 简单规则：如果文本里出现 <image>，且还有剩余 images_to_load，就按数量替换成图像块
        # 也可以更精细：统计 <image> 次数，或一张样本多张图逐一配对。
        if "<image>" in val and images_to_load:
            # 这里把全部剩余图像都放进去；如果需要一对一，改为 pop(0)。
            for img_rel in images_to_load:
                content_items.append({"type": "image", "image": os.path.join(sample.get("__image_root__", ""), img_rel)})
            # 去掉文本里的 <image> 标记
            val = val.replace("<image>", "").strip()

        if val:
            content_items.append({"type": "text", "text": val})

        if not content_items:
            # 防止空
            content_items.append({"type": "text", "text": ""})

        multi_turn_messages.append({
            "role": role,
            "content": content_items
        })

    # 最后一条要生成的 assistant
    multi_turn_messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": ""}]
    })

    # processor 期望的输入
    model_inputs = processor.apply_chat_template(
        multi_turn_messages,
        add_generation_prompt=True,
        tokenize=False
    )

    # 将需要的图片真正加载为 PIL.Image 在 processor 调用时一起传入
    pil_images = []
    for img_path in sample_io["images_paths"]:
        full_path = os.path.join(sample.get("__image_root__", ""), img_path)
        if os.path.isfile(full_path):
            try:
                from PIL import Image
                pil_images.append(Image.open(full_path).convert("RGB"))
            except Exception:
                pass  # 忽略坏图
        else:
            # 忽略缺失
            pass

    proc_inputs = processor(
        text=model_inputs,
        images=pil_images if len(pil_images) > 0 else None,
        return_tensors="pt"
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        pad_token_id=processor.tokenizer.pad_token_id
    )

    output_ids = model.generate(**proc_inputs, **gen_kwargs)
    # 只取新生成的部分 (如果需要严格裁剪，可用 output_ids[:, proc_inputs["input_ids"].shape[1]:])
    text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 由于 chat_template 里已经包含上下文，需要切掉前缀——做一个简单截断:
    # 可以根据模板中最后一个 "Assistant:" 或特殊 token 来分割
    cut_markers = ["Assistant:", "assistant\n", "assistant:"]
    last_pos = -1
    for m in cut_markers:
        p = text.rfind(m)
        if p > last_pos:
            last_pos = p + len(m)
    if last_pos != -1:
        generated = text[last_pos:].strip()
    else:
        generated = text.strip()

    return generated


def compute_simple_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    简单指标：
      exact_match: prediction == reference
      contains_ref: reference 子串是否在 prediction 中
    需要 reference 不为空。
    """
    refs = [r for r in records if r.get("reference")]
    if not refs:
        return {}
    em = 0
    contain = 0
    for r in refs:
        ref = r["reference"].strip()
        pred = (r["prediction"] or "").strip()
        if pred == ref:
            em += 1
        if ref and ref in pred:
            contain += 1
    total = len(refs)
    return {
        "num_with_ref": total,
        "exact_match": em / total,
        "contains_rate": contain / total
    }


# ===================== 主流程 =====================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True, help="基础模型路径或 HF 名称 (与训练的 --model_id 一致)")
    ap.add_argument("--lora_path", type=str, default=None, help="LoRA 适配器目录 (训练输出目录)")
    ap.add_argument("--eval_path", type=str, required=True, help="评测数据 json/jsonl")
    ap.add_argument("--eval_image_folder", type=str, default=None, help="评测图像目录(若有)")
    ap.add_argument("--output_dir", type=str, default="output/eval")
    ap.add_argument("--use_liger", action="store_true", help="是否应用 liger kernel")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_metrics", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = os.path.join(args.output_dir, f"predictions_{run_tag}.jsonl")

    print("[Info] Loading evaluation data...")
    raw_data = load_json_or_jsonl(args.eval_path)
    dataset = EvalDataset(raw_data, image_root=args.eval_image_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("[Info] Loading processor...")
    processor = AutoProcessor.from_pretrained(args.base_model)

    print("[Info] Loading model...")
    model = patch_and_load_model(args.base_model, use_liger=args.use_liger, dtype=args.dtype)
    model = maybe_load_lora(model, args.lora_path)

    total = len(dataset)
    results = []
    start_time = time.time()

    print("[Info] Start inference...")
    with open(pred_path, "w", encoding="utf-8") as fout:
        for idx, sample in enumerate(dataloader):
            # DataLoader batch_size=1 => sample 是 list/dict 的包装
            if isinstance(sample, list) or isinstance(sample, tuple):
                sample = sample[0]
            elif isinstance(sample, dict):
                pass

            # 由于 DataLoader 的默认 collate 会把字符串聚合成 list，这里做解包:
            # 如果字段是 list 且长度为1，展开:
            norm_sample = {}
            for k, v in sample.items():
                if isinstance(v, list) and len(v) == 1:
                    norm_sample[k] = v[0]
                else:
                    norm_sample[k] = v
            sample = norm_sample

            sample["__image_root__"] = args.eval_image_folder or ""

            sample_io = build_sample_io(sample)
            prediction = generate_one(
                model=model,
                processor=processor,
                sample=sample,
                sample_io=sample_io,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                repetition_penalty=args.repetition_penalty
            )

            # 尝试获取参考答案（如果 conversations 里存在并且是需要预测的 turn 原本有 value）
            reference = None
            # 在本示例里我们把参考答案存放在最后一个 gpt 且非空；如果你的数据不同请调整
            for turn in reversed(sample.get("conversations", [])):
                if turn.get("from", "").lower() in ("gpt", "assistant"):
                    val = (turn.get("value") or "").strip()
                    if val:
                        reference = val
                        break

            record = {
                "id": sample.get("id", idx),
                "prediction": prediction,
                "reference": reference,
            }
            # 保存问题（取最近一个 human turn）
            for turn in reversed(sample.get("conversations", [])):
                if turn.get("from", "").lower() in ("human", "user"):
                    record["question"] = turn.get("value")
                    break

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.append(record)

            if (idx + 1) % args.log_every == 0:
                elapsed = time.time() - start_time
                speed = (idx + 1) / elapsed
                print(f"[Eval] {idx+1}/{total} done | {speed:.2f} samples/s")

    print(f"[Info] Predictions saved to {pred_path}")

    if args.save_metrics:
        metrics = compute_simple_metrics(results)
        if metrics:
            metrics_path = os.path.join(args.output_dir, f"metrics_{run_tag}.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            print("[Info] Metrics:", metrics)
            print(f"[Info] Metrics saved to {metrics_path}")
        else:
            print("[Info] No reference answers found or metrics disabled.")

    print("[Done] Total samples:", total)


if __name__ == "__main__":
    main()