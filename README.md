# PALETTE

This repository contains the official implementation of PALETTE, a method for selectively relaxing safety refusal in large language models (LLMs) and vision-language models (VLMs) through per-layer direction ablation with LoRA.
The repository also includes the Llama-3.1-8B-Instruct refusal-direction tensor.

## News

[2026/02] The codebase is open-sourced.

[2026/02] Palette is accepted to Neurips 2026.

## Table of Contents

- [Getting Started](#getting-started)
  - [Environment Setup](#environment-setup)
  - [Repository Layout](#repository-layout)
  - [Data Preparation](#data-preparation)
- [Released Artifacts](#released-artifacts)
- [Training](#training)
- [Evaluation](#evaluation)
  - [Selective Refusal](#selective-refusal)
  - [General Capabilities](#general-capabilities)
- [Citation](#citation)

## Getting Started

### Environment Setup

Python 3.10 and a CUDA-enabled PyTorch installation are recommended.

```bash
conda create -n palette python=3.10 -y
conda activate palette

# Install a PyTorch build compatible with the local CUDA driver first.
# See https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

### Repository Layout

```text
PALETTE/
├── main.py                         # Training entry point
├── ablation/ablation_train.py      # Per-layer LoRA training
├── eval/
│   ├── eval_refusal.py             # GenHarm selective-refusal evaluation
│   ├── eval_mmlu.py                # MMLU evaluation
│   └── eval_gsm8k.py               # GSM8K evaluation
├── models/                         # Model wrappers
├── utils/                          # Hooks, data loaders, and utilities
├── data/genharm/                   # Train/test data
├── artifacts/
│   ├── Llama-2-7b-chat-hf/        # Direction tensor and released LoRAs
│   └── Llama-3.1-8B-Instruct/     # Direction tensor
├── data_benign.json                # Benign prompts used during training
└── requirements.txt
```

### Data Preparation

The bundled data follows this layout:

```text
data/genharm/
├── genharm_train.json
├── genharm_test.json
├── genharm_<Category>_train.json
├── genharm_<Category>_test.json
└── genharm_<Category>_train_dual.json
```

`<Category>` is one of `Illegal`, `Disinformation`, `Sexual`, `Hate`, or
`Violence`. Files ending in `_train.json` are used for training and files
ending in `_test.json` are used for evaluation. A regular record has this
schema:

```json
{
  "prompt": "...",
  "category": "Hate, harassment and discrimination"
}
```

The short target name comes from the filename. The record-level `category`
value may be a longer descriptive label and should start with the short target
name so that training can exclude allowed categories from the disallowed pool.

The optional `_train_dual.json` files add `allowed_response` and
`disallowed_response` fields and are used only with `--sr_ablation`.
`genharm_train.json` supplies the disallowed training pool. Evaluation reads
the per-category test files; `genharm_test.json` is an aggregate copy and is
not required by the current evaluation entry point.

To use another dataset, follow the same organization under
`data/<dataset_name>/` and replace the `genharm` filename prefix with the value
passed to `--dataset_name`.

## Released Artifacts

Bundled refusal-direction tensors:

| Model | File | Shape |
|---|---|---:|
| Llama-2-7b-chat-hf | `artifacts/Llama-2-7b-chat-hf/mean_diffs.pt` | `[6, 32, 4096]` |
| Llama-3.1-8B-Instruct | `artifacts/Llama-3.1-8B-Instruct/mean_diffs.pt` | `[5, 32, 4096]` |

Released LoRA adapters are currently provided only for Llama-2. All adapters
use rank 8 and alpha 16.

| Allowed category | LoRA layer | Training direction layer | Checkpoint |
|---|---:|---:|---|
| Illegal | 10 | 11 | `lora_layer_10_Illegal.pth` |
| Disinformation | 13 | 14 | `lora_layer_13_Disinformation.pth` |
| Sexual | 13 | 14 | `lora_layer_13_Sexual.pth` |
| Hate | 14 | 15 | `lora_layer_14_Hate.pth` |
| Violence | 14 | 15 | `lora_layer_14_Violence.pth` |

The checkpoint filename and evaluation argument `--lora_layer` identify the
layer that receives the LoRA update. During training, `--target_layer` selects
the refusal direction and trains the preceding layer; it is therefore one
greater than the LoRA layer.

## Training

The following command trains the Hate adapter. Change `--target_types`,
`--target_layer`, and `--output_dir` for another category.

```bash
python main.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --dataset_name genharm \
  --target_types Hate \
  --direction_path artifacts/Llama-2-7b-chat-hf/mean_diffs.pt \
  --target_layer 15 \
  --direction_pos -1 \
  --ratio 1.0 1.0 1.0 \
  --epochs 20 \
  --batch_size 8 \
  --seqlen 128 \
  --let_lr 1e-4 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --text_ablation_scale 2.5 \
  --eval_interval 5 \
  --output_dir log/llama2-hate \
  --skip_ppl_eval
```

The selected adapter is written to
`log/llama2-hate/lora_layer_14_Hate.pth`.

## Evaluation

### Selective Refusal

Evaluate a released adapter on the GenHarm test split:

```bash
python eval/eval_refusal.py \
  --model_id meta-llama/Llama-2-7b-chat-hf \
  --lora_path artifacts/Llama-2-7b-chat-hf/lora_layer_14_Hate.pth \
  --lora_layer 14 \
  --dataset_name genharm \
  --dataset_split test \
  --allow_datasets Hate \
  --disallow_datasets Illegal,Sexual,Disinformation,Violence \
  --eval_mode ablation \
  --eval_batch_size 8 \
  --limit 200 \
  --device cuda:0
```

For a different adapter, use the category and layer from the artifact table,
put that category in `--allow_datasets`, and list the other four categories in
`--disallow_datasets`.

The evaluator reports allowed-category response accuracy, disallowed-category
refusal accuracy, overall accuracy, recall, and F1. Refusal is detected with a
keyword heuristic defined in `eval/eval_refusal.py`.

### General Capabilities

Installations using these evaluators also require `lm-eval`, which is included
in `requirements.txt`.

```bash
# MMLU
python eval/eval_mmlu.py \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --lora_path artifacts/Llama-2-7b-chat-hf/lora_layer_14_Hate.pth \
  --lora_layer 14 \
  --tasks mmlu

# GSM8K
python eval/eval_gsm8k.py \
  --base_model meta-llama/Llama-2-7b-chat-hf \
  --lora_path artifacts/Llama-2-7b-chat-hf/lora_layer_14_Hate.pth \
  --lora_layer 14 \
  --tasks gsm8k
```


## Citation

```bash
@article{tan2026palette,
  title={Palette: A Modular, Controllable, and Efficient Framework for On-demand Authorized Safety Alignment Relaxation in LLMs},
  author={Tan, Qitao and Song, Xiaoying and Akbari, Arman and Akbari, Arash and Wang, Yanzhi and Zhai, Xiaoming and Hong, Lingzi and Xiang, Zhen and Lu, Jin and Yuan, Geng},
  journal={arXiv preprint arXiv:2605.24154},
  year={2026}
}
```
