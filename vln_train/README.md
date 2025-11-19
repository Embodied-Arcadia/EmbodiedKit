# Fine-tuning Qwen2-VL Series

This repository contains a script for training [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) and [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) with only using HuggingFace and [Liger-Kernel](https://github.com/linkedin/Liger-Kernel).

## Other projects

**[[Phi3-Vision Finetuning]](https://github.com/2U1/Phi3-Vision-Finetune)**<br>
**[[Llama3.2-Vision Finetuning]](https://github.com/2U1/Llama3.2-Vision-Ft)**<br>
**[[Molmo Finetune]](https://github.com/2U1/Molmo-Finetune)**<br>
**[[Pixtral Finetune]](https://github.com/2U1/Pixtral-Finetune)**<br>
**[[SmolVLM Finetune]](https://github.com/2U1/SmolVLM-Finetune)**<br>
**[[Gemma3 Finetune]](https://github.com/2U1/Gemma3-Finetune)**

## Update

- [2025/08/08] 🔥Monkey patch Qwen2.5-VL's window attention and forward for using less memory and speedups.
- [2025/07/25] Updated Classification training script for experimental feature.
- [2025/05/29] 🔥Supports GRPO training.
- [2025/04/16] 🔥Supports DPO training.
- [2025/03/04] Add Option for using liger kernel.
- [2025/02/18] 🔥Supports mixed-modality dataset with zero3.
- [2025/02/05] Fixed code for properly use image.
- [2025/02/03] Support Liger-kernel for Qwen2.5-VL.
- [2025/02/03] 🔥Supports Qwen2.5-VL.
- [2025/01/24] Add option for using DoRA.
- [2025/01/24] Fix error in LoRA training.
- [2025/01/18] 🔥Supports mixed-modality data.
- [2025/01/11] Updated 8-bit training with ms_amp fp8 with opt_level O3.
- [2024/11/05] Add memory efficient 8-bit training.
- [2024/09/12] 🔥Now the model is trained using [Liger-Kernel](https://github.com/linkedin/Liger-Kernel).
- [2024/09/11] Supports setting different learning rates to projector and vision model.
- [2024/09/11] 🔥Supports multi-image and video training.

## Table of Contents

- [Fine-tuning Qwen2-VL Series](#fine-tuning-qwen2-vl-series)
  - [Other projects](#other-projects)
  - [Update](#update)
  - [Table of Contents](#table-of-contents)
  - [Supported Features](#supported-features)
  - [Docker](#docker)
  - [Installation](#installation)
    - [Environments](#environments)
    - [Using `requirements.txt`](#using-requirementstxt)
    - [Using `environment.yaml`](#using-environmentyaml)
  - [Dataset Preparation](#dataset-preparation)
  - [Supervised Fine Tuning](#supervised-fine-tuning)
    - [Full Finetuning](#full-finetuning)
    - [Finetune with LoRA](#finetune-with-lora)
    - [Train with video dataset](#train-with-video-dataset)
      - [Image Resolution for vram usage](#image-resolution-for-vram-usage)
      - [Merge LoRA Weights](#merge-lora-weights)
  - [DPO Finetuning](#dpo-finetuning)
  - [GRPO Finetuning](#grpo-finetuning)
  - [Inference](#inference)
    - [Gradio Infernce (WebUI)](#gradio-infernce-webui)
  - [Issue for libcudnn error](#issue-for-libcudnn-error)
  - [TODO](#todo)
  - [Known Issues](#known-issues)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)

## Supported Features

- Deepspeed
- LoRA/QLoRA
- Full-finetuning
- Enable finetuning `vision_model` while using LoRA.
- Disable/enable Flash Attention 2
- Multi-image and video training
- Training optimized with liger kernel
- Mixed-modality dataset
- Direct Preference Optimization (DPO)
- Group Relative Policy Optimization (GRPO)

## Docker

To simplfy the setting process for training, you could use the provided pre-build environments.<br>
The settings are done in the conda env named `train`.<br><br>
You could find more information about the image [here](https://hub.docker.com/repository/docker/john119/vlm/general).

```
docker pull john119/vlm
docker run --gpus all -it -v /host/path:/docker/path --name vlm --ipc=host john119/vlm /bin/bash
```

## Installation

### Environments

- Ubuntu 22.04
- Nvidia-Driver 550.120
- Cuda version 12.4

Install the required packages using `environment.yaml`.

### Using `requirements.txt`

```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu124
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

### Using `environment.yaml`

```bash
conda env create -f environment.yaml
conda activate train
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

**Note:** You should install flash-attn after installing the other packages.

## Dataset Preparation

The script requires a dataset formatted according to the LLaVA specification. The dataset should be a JSON file where each entry contains information about conversations and images. Ensure that the image paths in the dataset match the provided `--image_folder`.<br>

**When using a multi-image dataset, the image tokens should all be `<image>`, and the image file names should have been in a list.**<br><br>
**Please see the example below and follow format your data.**

<details>
<summary>Example for single image dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": "000000033471.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      },
      {
        "from": "human",
        "value": "What feature can be seen on the back of the bus?"
      },
      {
        "from": "gpt",
        "value": "The back of the bus features an advertisement."
      },
      {
        "from": "human",
        "value": "Is the bus driving down the street or pulled off to the side?"
      },
      {
        "from": "gpt",
        "value": "The bus is driving down the street, which is crowded with people and other vehicles."
      }
    ]
  }
  ...
]
```

</details>

<details>
<summary>Example for multi image dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": ["000000033471.jpg", "000000033472.jpg"],
    "conversations": [
      {
        "from": "human",
        "value": "<image>\n<image>\nIs the perspective of the camera differnt?"
      },
      {
        "from": "gpt",
        "value": "Yes, It the perspective of the camera is different."
      }
    ]
  }
  ...
]
```

</details>

<details>
<summary>Example for video dataset</summary>

```json
[
  {
    "id": "sample1",
    "video": "sample1.mp4",
    "conversations": [
      {
        "from": "human",
        "value": "<video>\nWhat is going on in this video?"
      },
      {
        "from": "gpt",
        "value": "A man is walking down the road."
      }
    ]
  }
  ...
]
```

</details>

<details>
<summary>Example for DPO dataet</summary>

```json
[
  {
    "id": "e4cd017f-3854-4019-9cc1-9dacacb5d424",
    "image": "example_0_0.png",
    "prompt": "<image>\nDescribe the key features of the image in great detail.",
    "chosen": "The image features a modern, multi-colored train parked on a train track, occupying a significant portion of the scene. The train has a unique and striking design, with red, white, and blue colors, as well as yellow accents. It appears to be a high-speed train, indicating its advanced engineering and potential for fast travel.\n\nThe train is situated next to a platform, suggesting that it may be at a train station. The overall scene conveys a sense of transportation and modern technology, with the vibrant and sleek train as the main subject.",
    "rejected": "The image features a modern, colorful train parked on a set of train tracks. The train has a striking design, with red, white, and blue colors as well as yellow accents. It appears to be a high-speed train, ready for departure.\n\nThe train spans a significant portion of the image, from the left to the right side. The train tracks can be seen clearly beneath the train, emphasizing its position on the railway line. The overall scene gives off an impression of a contemporary and efficient mode of transportation."
  },
  {
    "id": "5e19e647-e5d3-4bcf-82e9-d262570743ae",
    "image": "example_1_0.png",
    "prompt": "<image>\nIs this bus in the USA?",
    "chosen": "Yes, based on the image, it can be assumed that this bus is in the USA. The location of the bus cannot be accurately determined.",
    "rejected": "No, it's not in the USA. The image does not provide specific information on where the bus is located. However, we can say that it's not in the United States."
  }
  ...
]
```

</details>

<details>
<summary>Example for GRPO dataset</summary>

```json
[
  {
    "id": "06bc8a17-bb1c-4007-8c08-92c41e2628b2",
    "image": "image_2.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "Based on the image, which geometric method is used to determine the bearing angle, and why is it the most appropriate choice?"
      },
      {
        "from": "gpt",
        "value": "<think>Let's analyze the image step-by-step. The image shows a right-angled triangle with points B, C, and A. The angle at point B is a right angle, indicating that trigonometric functions can be applied. To find the bearing angle, we need to relate the sides of the triangle. The tangent function is suitable here because it relates the opposite side (BC) to the adjacent side (AB) in a right-angled triangle. By using the tangent function, we can calculate the angle at point A, which is the bearing angle. Therefore, the most appropriate geometric method is the use of trigonometric functions.</think>\n\n<answer>A</answer>"
      }
    ]
  }
  ...
]
```

**Note:** You should remove all `<image>` and `<video>` tokens in your dataset. It works a bit different with other training methods.

</details>

<br><br>

Adding the new domain-specific data on top of the general data from open-source data will enhance downstream capabilities while retaining the foundational skills. Of course, you can also choose to fine-tune solely on the new data based on your requirements.

# Qwen-VLN — LoRA fine-tuning for Qwen-VL series

This repository is a fork of a Qwen-VL fine-tuning framework with added utilities to convert datasets and support LoRA-based vision fine-tuning. It includes training scripts (Deepspeed), dataset conversion helpers and evaluation tools.

This README focuses on a concise, publishable workflow and explains how to avoid leaking personal absolute paths.

## Quick start

1. Clone your fork:

```bash
git clone <your-fork-url>
cd Qwen-VLN
```

2. Create conda environment (recommended):

```bash
conda env create -f environment.yaml
conda activate train
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu124
```

3. Convert your dataset (example):

```bash
python construct_json.py --source /path/to/source_annotations.json --target ./data/train_data.json
```

4. Run LoRA vision fine-tuning (example):

```bash
export MODEL_NAME=Qwen/Qwen2.5-VL-3B-Instruct
export DATA_PATH=./data/train_data.json
export IMAGE_FOLDER=./data/images
export OUTPUT_DIR=./output/lora_vision_test
bash scripts/finetune_lora_vision.sh
```

5. Merge LoRA weights (example):

```bash
export MODEL_PATH=./output/lora_vision_test/checkpoint-XXXX
export SAVE_MODEL_PATH=./output/merge_test/checkpoint-XXXX
bash scripts/merge_lora.sh
```

6. Evaluation (example):

```bash
export MODEL_PATH=./output/merge_test/checkpoint-2949
export JSON_FILE_PATH=./data/Simplified_Dataset/R2R/data.json
export IMAGE_BASE_DIR=./data/Simplified_Dataset/R2R/train
export OUTPUT_JSON_PATH=./output/eval/evaluation_results.json
bash scripts/eval_train.sh
```

## Important: remove personal absolute paths before publishing

I replaced several hard-coded absolute paths (e.g. `/path/to/your/project/...`) with environment-variable-backed defaults in these files:

- `scripts/finetune_lora_vision.sh`
- `scripts/merge_lora.sh`
- `scripts/eval_train.sh`
- `evaluation/eval.py`
- `src/eval/*`
- `construct_json.py` (now accepts `--source` and `--target`)

Before committing, run:

```bash
git grep -n "/absolute/path/\|/Users/\|your-username" || true
```

and fix any remaining personal paths. Prefer environment variables or relative paths.

## Recommended .gitignore

```
output/
.env
.venv
data/
.DS_Store
```

## Optional improvements I can implement

- Make more scripts accept CLI flags (getopts).
- Add `scripts/sanitize_paths.sh` to detect personal paths before commits.
- Add `env.example` with recommended variables.

Tell me which of these you'd like next and I'll implement it.
