# EpiScreen

## 1. Introduction

This repository contains the code for the paper **"Early Epilepsy Detection from Electronic Health Records with Large Language Models"** (npj Digital Medicine, 2026).

EpiScreen fine-tunes a large language model (e.g., Qwen2.5-32B-Instruct) with QLoRA to distinguish epilepsy from psychogenic nonepileptic seizures (PNES) using clinical notes from electronic health records.

## 2. Usage

### 2.1 Environment

**Package requirements**


| Package | Version |
|---|---|
| pandas | - |
| torch (PyTorch) | >= 2.0 |
| numpy | - |
| matplotlib | - |
| transformers | >= 4.35.0 |
| peft | >= 0.6.0 |
| scikit-learn | - |
| bitsandbytes | - |
| accelerate | optional, but recommended |

Install them with:

```bash
pip install pandas torch numpy matplotlib transformers peft scikit-learn bitsandbytes
# optional but recommended
pip install accelerate
```


### 2.2 Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    --mode train \
    --local_model_path /utilities/models/Qwen2.5-32B-Instruct/ \
    --output_dir ./qwen_qlora_checkpoints \
    --learning_rate 5e-4
```

### 2.3 Testing

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py \
    --mode test \
    --local_model_path /utilities/models/Qwen2.5-32B-Instruct/ \
    --output_dir ./qwen_qlora_checkpoints \
    --output_file ./test_results_qwen.xlsx
```

**Arguments**

| Argument | Description |
|---|---|
| `--mode` | `train` or `test` |
| `--local_model_path` | Path to the local base model |
| `--output_dir` | Directory for saving/loading QLoRA checkpoints |
| `--learning_rate` | Learning rate for training (e.g., `5e-4`) |
| `--output_file` | Path of the Excel file for test results |

> Adjust `CUDA_VISIBLE_DEVICES` to match the GPUs available on your machine.

## 3. Contact

For research cooperation, please contact shuang DOT zhou AT connect.polyu DOT hk

## 4. Citation

If you find our work useful, please cite the paper:

```bib
@article{zhou2026early,
  title={Early epilepsy detection from electronic health records with large language models},
  author={Zhou, Shuang and Yu, Kai and Zhan, Zaifu and Zhou, Huixue and Zeng, Min and Xie, Feng and Sha, Zhiyi and Zhang, Rui},
  journal={npj Digital Medicine},
  year={2026},
  publisher={Nature Publishing Group UK, London}
}
```
