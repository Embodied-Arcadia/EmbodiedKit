# OpenQwenVLA

This document describes how to run **OpenQwenVLA** on an **NVIDIA H20 GPU** with **CUDA 12.4**, including environment setup, pretraining, fine-tuning, and evaluation steps.

---

## 🧩 Environment Setup

1. **Create and activate a Conda environment**
   ```bash
   conda create -n OpenQwenVLA python=3.10 -y
   conda activate OpenQwenVLA
   ```

2. **Install PyTorch (CUDA 12.4)**
   ```bash
   pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   > ⚠️ If `opencv-python` conflicts with `numpy`, downgrade `opencv-python` to a version **below 4.12**.

---

## ⚙️ Configuration

Before running the experiments, modify the following parameters in both
`openqwenvla_pretrain.py` and `openqwenvla_finetuning.py`:

- `output_dir`: output directory for model checkpoints and logs  
- `data_root_dir`: root directory of the RLDS dataset  
- `model_id`: model name or checkpoint identifier

---

## 🚀 Experiment Workflow

### 1️⃣ Pretraining
Run pretraining on an RLDS-formatted dataset:
```bash
python openqwenvla_pretrain.py
```

### 2️⃣ Fine-tuning
Run fine-tuning on a specific RLDS-formatted dataset:
```bash
python openqwenvla_finetuning.py
```

### 3️⃣ Evaluation
Evaluate the model on the **LIBERO** dataset:
```bash
python experiments/libero/run_libero_eval.py
```

---

## 💾 Hardware Requirements

| Stage        | Minimum GPU Memory |
|---------------|--------------------|
| Pretraining   | ≥ 80 GB |
| Fine-tuning   | ≥ 80 GB |
| Evaluation    | ≥ 30 GB |
