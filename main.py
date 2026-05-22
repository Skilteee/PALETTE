import os
import sys
import random
import numpy as np
import torch
import time
import json
from types import SimpleNamespace
from pathlib import Path
from collections import defaultdict, Counter

from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from datasets import load_dataset

import utils
from ablation.ablation_train import ablation_train
from models.LMClass import LMClass


def apply_chat_template(tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system",
         "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request."},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_strongreject_ablation_split(dataset_name, target_type, test_ratio, train_ratio, split_seed):
    base_dir = Path(f"./data/{dataset_name}")
    dual_path = base_dir / f"{dataset_name}_{target_type}_test_dual.json"
    train_path = base_dir / f"{dataset_name}_{target_type}_train.json"

    with dual_path.open("r", encoding="utf-8") as f:
        dual_data = json.load(f)
    with train_path.open("r", encoding="utf-8") as f:
        train_data = json.load(f)

    merged = dual_data + train_data
    rng = random.Random(split_seed)
    rng.shuffle(merged)

    if len(merged) <= 1:
        return merged, []

    split_idx = int(len(merged) * test_ratio)
    split_idx = min(max(split_idx, 1), len(merged) - 1)

    test_pool = merged[:split_idx]
    remain_pool = merged[split_idx:]

    train_size = int(len(remain_pool) * train_ratio)
    if train_ratio > 0 and train_size == 0 and len(remain_pool) > 0:
        train_size = 1
    train_size = min(train_size, len(remain_pool))

    train_pool = remain_pool[:train_size]
    return train_pool, test_pool


def resolve_dtype(dtype: str):
    mapping = {"auto": "auto", "bfloat16": torch.bfloat16, "float16": torch.float16}
    return mapping.get(dtype, torch.float32)


def sample_balanced_by_category(data, total_n, category_key="category"):
    if total_n <= 0 or not data:
        return []

    buckets = defaultdict(list)
    for item in data:
        category = item.get(category_key)
        if category is not None:
            buckets[category].append(item)

    if not buckets:
        return []

    categories = list(buckets.keys())
    random.shuffle(categories)
    for cat in categories:
        random.shuffle(buckets[cat])

    selected = []
    active_categories = categories[:]
    while len(selected) < total_n and active_categories:
        next_active = []
        for cat in active_categories:
            bucket = buckets[cat]
            if not bucket:
                continue
            selected.append(bucket.pop())
            if len(selected) >= total_n:
                break
            if bucket:
                next_active.append(cat)
        active_categories = next_active

    return selected


def _load_image(image):
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    return image


def make_vlm_prompt_dataloader(args, lm, prompts, images, labels):
    processor = lm.processor
    system_prompt = args.vlm_system_prompt.strip() if args.vlm_system_prompt else ""
    dataloader = []

    for i in range(len(prompts)):
        image = _load_image(images[i])
        conversation = []
        if system_prompt:
            conversation.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        conversation.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompts[i]},
                ],
            }
        )
        inputs = processor.apply_chat_template(
            [conversation],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=lm.seqlen,
            padding_side='left'
        ).to(lm._device)

        dataloader.append((inputs, labels[i]))

    torch.cuda.empty_cache()
    return dataloader


def get_vlm_dataloader(args, lm, target_types, ratio, all_types):
    """Load VLM training data: allowed + disallowed + safe samples."""
    allowed_data = []
    for target_type in target_types:
        with open(f"./data/MMBench/processed_questions/{target_type}_test.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            random.shuffle(data)
            allowed_data.extend(data)

    random.shuffle(allowed_data)
    num_allow = int(len(allowed_data) * ratio[0])
    prompts = [d.get('Changed Question') for d in allowed_data[:num_allow]]
    images = [d.get('image') for d in allowed_data[:num_allow]]
    labels = ['allowed'] * len(prompts)

    # Disallowed data
    with open('./data/MMBench/all_data_refused.json', 'r', encoding="utf-8") as f:
        data = json.load(f)
        random.shuffle(data)

    disallowed_pool = [
        d for d in data
        if not any(d['category'].startswith(t) for t in target_types)
        and d['category'] in all_types
    ]
    disallowed_data = sample_balanced_by_category(
        disallowed_pool, int(ratio[1] * num_allow), category_key="category",
    )
    print('Category in VLM training:', Counter(d.get('category') for d in disallowed_data))

    prompts.extend(d.get('instruction') for d in disallowed_data)
    images.extend(d.get('image') for d in disallowed_data)
    labels.extend(['disallowed'] * len(disallowed_data))

    # Safe data (VQAv2)
    vqa_data = list(load_dataset("lmms-lab/VQAv2", split="validation[:500]"))
    random.shuffle(vqa_data)
    vqa_data = vqa_data[:int(ratio[2] * num_allow)]

    prompts.extend(d.get("question") for d in vqa_data)
    images.extend(d.get("image") for d in vqa_data)
    labels.extend(['safe'] * len(vqa_data))

    # Eval dataloader
    eval_set = ['01-Illegal_Activity', '02-HateSpeech', '04-Physical_Harm', '06-Fraud', '07-Sex', '09-Privacy_Violence']
    eval_limit = 20
    eval_dataloader = []
    system_prompt = args.vlm_system_prompt

    for category in eval_set:
        with open(f"./data/MMBench/processed_questions/{category}_train.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        for d in data[:eval_limit]:
            conversation = []
            if system_prompt and system_prompt.strip():
                conversation.append(
                    {"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]},
                )
            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": Image.open(d.get('image')).convert("RGB")},
                        {"type": "text", "text": d.get('Changed Question')},
                    ],
                }
            )
            label = f'{category}_allowed' if category in target_types else f'{category}_disallowed'
            eval_dataloader.append((conversation, label))

    dataloader = make_vlm_prompt_dataloader(args, lm, prompts, images, labels)
    return dataloader, eval_dataloader


def get_dataloader(args, lm, dataset_name, target_types, ratio):
    """Load LLM training data: allowed + disallowed + safe samples."""
    if dataset_name in ['wmdp', 'strongreject']:
        allowed_data = []
        ablation_eval_map = {}

        if dataset_name == "strongreject" and args.sr_ablation:
            for idx, target_type in enumerate(target_types):
                train_pool, test_pool = build_strongreject_ablation_split(
                    dataset_name=dataset_name,
                    target_type=target_type,
                    test_ratio=args.sr_test_ratio,
                    train_ratio=args.sr_train_ratio,
                    split_seed=args.sr_split_seed + idx * 1009,
                )
                allowed_data.extend(train_pool)
                ablation_eval_map[target_type] = test_pool
                print(f"[strongreject-ablation] {target_type}: train={len(train_pool)}, test={len(test_pool)}")
        else:
            for target_type in target_types:
                with open(f"./data/{dataset_name}/{dataset_name}_{target_type}_test.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                    random.shuffle(data)
                    allowed_data.extend(data)

        random.shuffle(allowed_data)
        num_allow = len(allowed_data) * ratio[0]
        prompts = [d.get('prompt') or d.get('question') for d in allowed_data]
        labels = ['allowed'] * len(allowed_data)

        # Disallowed data
        with open(f'./data/{dataset_name}_test.json', 'r') as f:
            data = json.load(f)
            random.shuffle(data)

        data = [d for d in data if not any(d['category'].startswith(t) for t in target_types)]
        n_disallowed = int(ratio[1] * num_allow)
        disallowed_prompts = [d.get('prompt') or d.get('question') for d in data[:n_disallowed]]
        print("Disallowed Counter:", Counter(d.get('category') for d in data[:n_disallowed]))

        prompts.extend(disallowed_prompts)
        labels.extend(['disallowed'] * len(disallowed_prompts))

        # Safe data
        with open('data_benign.json', 'r') as f:
            data = json.load(f)
            random.shuffle(data)
        n_safe = int(ratio[2] * num_allow)
        safe_prompts = [d["prompt"] for d in data[:n_safe]]
        prompts.extend(safe_prompts)
        labels.extend(['safe'] * len(safe_prompts))

        # Eval dataloader
        eval_set = ['chem', 'bio', 'cyber']
        for target_type in target_types:
            if target_type not in eval_set:
                eval_set.append(target_type)

        eval_limit = 150
        eval_dataloader = []
        for dataset in eval_set:
            if dataset_name == "strongreject" and args.sr_ablation and dataset in target_types:
                eval_data = ablation_eval_map.get(dataset, [])
                eval_prompts = [d.get('prompt') or d.get('question') for d in eval_data]
            else:
                with open(f"./data/{dataset_name}/{dataset_name}_{dataset}_train.json", "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
                if dataset in target_types:
                    eval_prompts = [d.get('prompt') or d.get('question') for d in eval_data]
                else:
                    eval_prompts = [d.get('prompt') or d.get('question') for d in eval_data[:eval_limit]]

            label_suffix = 'allowed' if dataset in target_types else 'disallowed'
            eval_dataloader.extend([(p, f'{dataset}_{label_suffix}') for p in eval_prompts])

        assert len(prompts) == len(labels)

        prompt_batches = [
            lm.tokenizer(
                apply_chat_template(lm.tokenizer, text),
                return_tensors="pt",
                truncation=True,
                max_length=lm.seqlen,
                padding="max_length",
            ).to(lm._device)
            for text in prompts
        ]
        dataloader = list(zip(prompt_batches, labels))

    else:
        dataloader = []
        eval_dataloader = []

        for target in target_types:
            with open(f"./data/{dataset_name}/{target}.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                random.shuffle(data)

            train_set = data[:int(len(data) * 0.2)]
            eval_set = data[int(len(data) * 0.2):]

            train_prompts = [d["prompt"] for d in train_set]
            train_labels = [d['type'] for d in train_set]
            eval_prompts = [d["prompt"] for d in eval_set]
            eval_labels = [d['type'] for d in eval_set]

            # Add safe data
            with open('data_benign.json', 'r') as f:
                benign_data = json.load(f)
                random.shuffle(benign_data)
            n_safe = int(ratio[2] // 2 * len(train_prompts))
            safe_prompts = [d["prompt"] for d in benign_data[:n_safe]]
            train_prompts.extend(safe_prompts)
            train_labels.extend(['safe'] * len(safe_prompts))

            prompt_batches = [
                lm.tokenizer(
                    apply_chat_template(lm.tokenizer, text),
                    return_tensors="pt",
                    truncation=True,
                    max_length=lm.seqlen,
                    padding="max_length",
                ).to(lm._device)
                for text in train_prompts
            ]

            dataloader.extend(list(zip(prompt_batches, train_labels)))
            eval_dataloader.extend(list(zip(eval_prompts, eval_labels)))

    random.shuffle(dataloader)
    return dataloader, eval_dataloader


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Per-layer direction ablation training with LoRA")

    # Model
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--model_resume", type=str, default=None, help="Path to existing LoRA adapter to merge")
    parser.add_argument("--task_type", type=str, default="text", choices=["text", "vision"])
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                        choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--multigpu", action="store_true")

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seqlen", type=int, default=128, help="Sequence length for LLM")
    parser.add_argument("--seqlen_vision", type=int, default=384, help="Sequence length for VLM")
    parser.add_argument("--let_lr", type=float, default=1e-4, help="LoRA learning rate")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--wd", type=float, default=0, help="Weight decay")
    parser.add_argument("--eval_interval", type=int, default=30, help="Evaluate every N epochs")

    # Direction ablation
    parser.add_argument("--direction_path", type=str, required=True,
                        help="Path to mean_diffs.pt containing refusal directions")
    parser.add_argument("--target_layer", type=int, required=True,
                        help="Target layer index for direction ablation")
    parser.add_argument("--direction_pos", type=int, default=-1,
                        help="Position index in the direction tensor")
    parser.add_argument("--text_ablation_scale", type=float, default=2.5)
    parser.add_argument("--vision_ablation_scale", type=float, default=2.5)

    # Dataset
    parser.add_argument("--dataset_name", type=str, default="wmdp", help="Dataset name (e.g., wmdp, strongreject)")
    parser.add_argument("--target_types", type=str, nargs="+", default=["cyber"],
                        help="Target categories to allow")
    parser.add_argument("--ratio", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Ratio for [allowed, disallowed, safe] data")

    # VLM-specific
    parser.add_argument("--vlm_dtype", type=str, default="bfloat16",
                        choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--vlm_system_prompt", type=str,
                        default="Below is an instruction that describes a task. Write a response that appropriately completes the request.")
    parser.add_argument("--vlm_all_types", type=str, nargs="+",
                        default=["01-Illegal_Activity", "02-HateSpeech", "04-Physical_Harm",
                                 "06-Fraud", "07-Sex", "09-Privacy_Violence"],
                        help="All VLM category types for disallowed pool")

    # StrongReject ablation split
    parser.add_argument("--sr_ablation", action="store_true")
    parser.add_argument("--sr_test_ratio", type=float, default=0.5)
    parser.add_argument("--sr_train_ratio", type=float, default=1.0)
    parser.add_argument("--sr_split_seed", type=int, default=42)

    # Output
    parser.add_argument("--output_dir", default="./log/", type=str)
    parser.add_argument("--save_dir", default=None, type=str, help="Directory for saving model")
    parser.add_argument("--cache_dir", default="./cache", type=str)

    args = parser.parse_args()

    if not (0.0 < args.sr_test_ratio < 1.0):
        raise ValueError("--sr_test_ratio must be in (0, 1).")
    if not (0.0 <= args.sr_train_ratio <= 1.0):
        raise ValueError("--sr_train_ratio must be in [0, 1].")

    return args


def main():
    args = parse_args()

    # Init directories and logger
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    logger = utils.create_logger(Path(args.output_dir))
    logger.info(args)

    # Derived args
    args.net = args.model.split('/')[-1]
    args.model_family = args.net.split('-')[0]
    args.deactive_amp = False

    # Load model
    if args.task_type == "vision":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoProcessor.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code, max_pixels=256 * 28 * 28
        )
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            torch_dtype=resolve_dtype(args.vlm_dtype),
            trust_remote_code=args.trust_remote_code,
            device_map='cpu'
        )
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        lm = SimpleNamespace(
            model=model, tokenizer=processor, processor=processor,
            _device=device, device=device, seqlen=args.seqlen_vision,
        )
    else:
        lm = LMClass(args)
        lm.model.eval()
        for param in lm.model.parameters():
            param.requires_grad = False

    if args.multigpu:
        from utils.parallel_utils import get_lowest_occupied_gpu
        gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
        lm._device = f"cuda:{gpu_id}"
        logger.info(f"Set training on gpu {gpu_id}")

    # Prepare data
    logger.info("=== Start per-layer ablation training ===")
    tick = time.time()

    if args.task_type == "vision":
        dataloader, eval_dataloader = get_vlm_dataloader(
            args, lm, args.target_types, args.ratio, args.vlm_all_types
        )
        logger.info(f"Loaded VLM train batches: {len(dataloader)}")
        args.scale = args.vision_ablation_scale
    else:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token
        dataloader, eval_dataloader = get_dataloader(
            args, lm, args.dataset_name, args.target_types, args.ratio
        )
        args.scale = args.text_ablation_scale

    # Train
    ablation_train(lm, args, dataloader, eval_dataloader, logger)
    logger.info(f"Training time: {time.time() - tick:.1f}s")

    # Save
    if args.save_dir:
        lm.model.save_pretrained(args.save_dir)
        lm.tokenizer.save_pretrained(args.save_dir)

    # Evaluate PPL
    if args.task_type == "text":
        utils.evaluate(lm.model, args.model)


def set_seed(seed=None):
    seed = random.randint(1, 10000) if seed is None else seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed()
    main()
