# PALETTE: Per-Layer Ablation Training for Selective Safety Relaxation

This repository contains the official implementation of PALETTE, a method for selectively relaxing safety refusal in large language models (LLMs) and vision-language models (VLMs) through per-layer direction ablation with LoRA.

## Overview

PALETTE trains a LoRA adapter on a single transformer layer to selectively remove the model's refusal behavior for specified "allowed" categories while preserving refusal for other harmful categories and maintaining general capabilities.

**Key idea**: By identifying a *refusal direction* in the model's activation space and training a LoRA adapter to ablate this direction only for target categories, we achieve fine-grained control over which safety behaviors are relaxed.

## Project Structure

```
PALETTE/
├── main.py                     # Training entry point
├── ablation/
│   └── ablation_train.py       # Core per-layer ablation training logic
├── models/
│   ├── LMClass.py              # LLM model wrapper
│   └── models_utils.py         # Model utilities
├── eval/
│   ├── eval_refusal.py         # Evaluate selective refusal (allowed vs disallowed)
│   ├── eval_mmlu.py            # MMLU benchmark evaluation
│   ├── eval_gsm8k.py           # GSM8K benchmark evaluation
│   └── eval_mmmu.py            # MMMU benchmark evaluation (VLM)
├── utils/
│   ├── hook_utils.py           # Direction ablation hooks
│   ├── utils.py                # PPL evaluation, logger, grad scaler
│   ├── datautils.py            # Data loading utilities
│   ├── parallel_utils.py       # Multi-GPU utilities
│   ├── dataset_utils.py        # MMMU dataset processing
│   ├── eval_utils.py           # Evaluation judge utilities
│   └── common_utils.py         # Common utilities
├── data/                       # Datasets (wmdp, strongreject, cosapien, etc.)
├── data_benign.json            # Benign prompts for safe data
└── requirements.txt
```

## Setup

```bash
# Create environment
conda create -n palette python=3.10 -y
conda activate palette

# Install dependencies
pip install -r requirements.txt
```

### Prerequisites

1. **Refusal direction vectors**: Pre-compute refusal directions using [refusal_direction](https://github.com/andyrdt/refusal_direction) or equivalent. The output should be a `mean_diffs.pt` file with shape `[positions, layers, hidden_dim]`.

2. **Datasets**: Place datasets under `./data/`:
   - `wmdp/` — WMDP benchmark (cyber, bio, chem categories)
   - `strongreject/` — StrongReject benchmark (Violence, Hate, Sexual, etc.)
   - `MMBench/` — MM-SafetyBench for VLM evaluation (optional)

## Training

### LLM Training

```bash
python main.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset_name wmdp \
  --target_types cyber \
  --direction_path ./directions/Llama-3.1-8B-Instruct/mean_diffs.pt \
  --target_layer 12 \
  --direction_pos -1 \
  --ratio 1.0 1.0 1.0 \
  --epochs 20 \
  --batch_size 16 \
  --seqlen 128 \
  --let_lr 1e-4 \
  --lora_rank 8 \
  --lora_alpha 16.0 \
  --text_ablation_scale 2.5 \
  --eval_interval 30 \
  --output_dir ./log/llama3-cyber/
```

### VLM Training

```bash
python main.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --task_type vision \
  --target_types 09-Privacy_Violence \
  --direction_path ./directions/Qwen2.5-VL-7B-Instruct/mean_diffs.pt \
  --target_layer 17 \
  --direction_pos -3 \
  --ratio 1.0 0.5 1.0 \
  --epochs 20 \
  --seqlen_vision 384 \
  --vision_ablation_scale 2.5 \
  --trust_remote_code \
  --output_dir ./log/qwen-vl-privacy/
```


### Key Arguments

| Argument | Description |
|----------|-------------|
| `--model` | HuggingFace model name or local path |
| `--dataset_name` | Dataset: `wmdp`, `strongreject`, or custom |
| `--target_types` | Categories to allow (space-separated) |
| `--direction_path` | Path to pre-computed `mean_diffs.pt` |
| `--target_layer` | Layer index for ablation |
| `--direction_pos` | Token position in direction tensor (e.g., -1 for last) |
| `--text_ablation_scale` | Ablation strength for LLM |
| `--vision_ablation_scale` | Ablation strength for VLM |
| `--ratio` | Data ratio: [allowed, disallowed, safe] |

## Evaluation

### Selective Refusal Evaluation

Evaluate whether the model correctly responds to allowed prompts and refuses disallowed ones:

```bash
python eval/eval_refusal.py \
  --model_id meta-llama/Llama-3.1-8B-Instruct \
  --lora_path ./log/llama3-cyber/best_lora_layer_11.pth \
  --lora_layer 11 \
  --dataset_name wmdp \
  --disallow_datasets chem,bio \
  --eval_mode ablation
```

### General Capability Benchmarks

```bash
# MMLU
python eval/eval_mmlu.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --lora_path ./log/llama3-cyber/best_lora_layer_11.pth \
  --lora_layer 11 \
  --tasks mmlu

# GSM8K
python eval/eval_gsm8k.py \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --lora_path ./log/llama3-cyber/best_lora_layer_11.pth \
  --lora_layer 11 \
  --tasks gsm8k
```
