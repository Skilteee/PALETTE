import os
import sys
import random
import numpy as np
import torch
import time
from types import SimpleNamespace
import torch.nn as nn
from ablation.ablation_train import ablation_train
import utils
from pathlib import Path
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from datasets import load_dataset
from models.LMClass import LMClass

import json
from collections import defaultdict
from collections import Counter


def apply_chat_template(tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system",
         "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request."},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_genharm_ablation_split(dataset_name, target_type, test_ratio, train_ratio, split_seed):
    base_dir = Path(f"./data/{dataset_name}")
    dual_path = base_dir / f"{dataset_name}_{target_type}_test_dual.json"
    train_path = base_dir / f"{dataset_name}_{target_type}_train.json"

    with dual_path.open("r", encoding="utf-8") as f:
        dual_data = json.load(f)
    with train_path.open("r", encoding="utf-8") as f:
        train_data = json.load(f)

    merged = []
    merged.extend(dual_data)
    merged.extend(train_data)

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
    if dtype == "auto":
        return "auto"
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    return torch.float32


def resolve_vlm_image_path(sample, image_root: Path):
    raw_img = sample.get("image")
    if isinstance(raw_img, str) and raw_img.strip():
        raw_path = Path(raw_img)
        if raw_path.exists():
            return raw_path
        marker = "MM-SafetyBench(imgs)"
        if marker in raw_img:
            suffix = raw_img.split(marker, 1)[1].lstrip("/\\")
            candidate = image_root / marker / Path(suffix)
            if candidate.exists():
                return candidate
    sid = sample.get("id")
    category = sample.get("category")
    if sid is not None and category:
        candidate = image_root / "MM-SafetyBench(imgs)" / str(category) / "SD" / f"{sid}.jpg"
        if candidate.exists():
            return candidate
    return None


def sample_balanced_by_category(data, total_n, category_key="category"):
    if total_n <= 0 or not data:
        return []

    buckets = defaultdict(list)
    for item in data:
        category = item.get(category_key)
        if category is None:
            continue
        buckets[category].append(item)

    if not buckets:
        return []

    categories = list(buckets.keys())
    random.shuffle(categories)
    for category in categories:
        random.shuffle(buckets[category])

    selected = []
    active_categories = categories[:]
    while len(selected) < total_n and active_categories:
        next_active = []
        for category in active_categories:
            bucket = buckets[category]
            if not bucket:
                continue
            selected.append(bucket.pop())
            if len(selected) >= total_n:
                break
            if bucket:
                next_active.append(category)
        active_categories = next_active

    return selected


def _get_sample_image(image):
    if type(image) == str:
        return Image.open(image).convert("RGB")
    return image


def make_vlm_prompt_dataloader(args, lm, prompts, images, labels):
    processor = lm.processor
    system_prompt = args.vlm_system_prompt.strip() if args.vlm_system_prompt else ""
    dataloader = []

    for i in range(len(prompts)):
        image = images[i]
        prompt = prompts[i]
        image = _get_sample_image(image)
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
                    {"type": "text", "text": prompt},
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
    allowed_data = []

    for target_type in target_types:
        with open(f"./data/MMBench/processed_questions/{target_type}_test.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            random.shuffle(data)
            allowed_data.extend(data)

    random.shuffle(allowed_data)
    num_allow = int(len(allowed_data) * ratio[0])
    prompts = [d.get('Changed Question', None) for d in allowed_data[:num_allow]]
    images = [d.get('image', None) for d in allowed_data[:num_allow]]
    labels = ['allowed' for _ in range(len(prompts))]

    with open(f'./data/MMBench/all_data_refused.json', 'r', encoding="utf-8") as f:
        data = json.load(f)
        random.shuffle(data)

    disallowed_pool = [
        d for d in data
        if not any(d['category'].startswith(target_type) for target_type in target_types)
        and d['category'] in all_types
    ]
    disallowed_data = sample_balanced_by_category(
        disallowed_pool,
        int(ratio[1] * num_allow),
        category_key="category",
    )
    disallowed_labels = [d.get('category', None) for d in disallowed_data]
    print('category in VLM training:', Counter(disallowed_labels))

    disallowed_prompts = [d.get('instruction', None) for d in disallowed_data]
    disallowed_images = [d.get('image', None) for d in disallowed_data]

    prompts.extend(disallowed_prompts)
    images.extend(disallowed_images)
    labels.extend(['disallowed' for _ in range(len(disallowed_prompts))])

    data = list(load_dataset("lmms-lab/VQAv2", split="validation[:500]"))
    random.shuffle(data)
    data = data[:int(ratio[2] * num_allow)]

    safe_prompts = [d.get("question", None) for d in data]
    safe_images = [d.get("image", None) for d in data]
    prompts.extend(safe_prompts)
    images.extend(safe_images)
    labels.extend(['safe' for _ in range(len(safe_prompts))])

    eval_set = ['01-Illegal_Activity', '02-HateSpeech', '04-Physical_Harm', '06-Fraud', '07-Sex', '09-Privacy_Violence']
    limit = 20

    eval_dataloader = []
    system_prompt = args.vlm_system_prompt
    for category in eval_set:
        with open(f"./data/MMBench/processed_questions/{category}_train.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            eval_prompts = [d.get('Changed Question', None) for d in data[:limit]]
            eval_images = [Image.open(d.get('image', None)).convert("RGB") for d in data[:limit]]

            for i in range(len(eval_prompts)):
                conversation = []
                if system_prompt and system_prompt.strip():
                    conversation.append(
                        {"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]},
                    )
                conversation.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": eval_images[i]},
                            {"type": "text", "text": eval_prompts[i]},
                        ],
                    }
                )
                eval_dataloader.append((conversation, '{}_allowed'.format(category) if category in target_types else '{}_disallowed'.format(category)))

    dataloader = make_vlm_prompt_dataloader(args, lm, prompts, images, labels)
    return dataloader, eval_dataloader


torch.backends.cudnn.benchmark = True


def get_dataloader(args, lm, dataset_name, target_types, ratio):
    if dataset_name in ['wmdp', 'genharm']:
        allowed_data = []
        ablation_eval_map = {}
        if dataset_name == "genharm" and args.sr_ablation:
            for idx, target_type in enumerate(target_types):
                train_pool, test_pool = build_genharm_ablation_split(
                    dataset_name=dataset_name,
                    target_type=target_type,
                    test_ratio=args.sr_test_ratio,
                    train_ratio=args.sr_train_ratio,
                    split_seed=args.sr_split_seed + idx * 1009,
                )
                allowed_data.extend(train_pool)
                ablation_eval_map[target_type] = test_pool
                print(
                    f"[genharm-ablation] {target_type}: "
                    f"train={len(train_pool)}, test={len(test_pool)}"
                )
        else:
            for target_type in target_types:
                with open(f"./data/{dataset_name}/{dataset_name}_{target_type}_test.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                    random.shuffle(data)
                    allowed_data.extend(data)

        random.shuffle(allowed_data)
        num_allow = len(allowed_data) * ratio[0]
        prompts = [d.get('prompt', None) or d.get('question', None) for d in allowed_data]
        labels = ['allowed' for _ in range(len(allowed_data))]

        with open(f'./data/{dataset_name}_test.json', 'r') as f:
            data = json.load(f)
            random.shuffle(data)

        data = [
            d for d in data
            if not any(d['category'].startswith(target_type) for target_type in target_types)
        ]
        disallowed_prompts = [
            d.get('prompt', None) or d.get('question', None) for d in data[:int(ratio[1] * num_allow)]
        ]

        tmp_labels = [d.get('category', None) for d in data[:int(ratio[1] * num_allow)]]
        print("Disallowed Counter:", Counter(tmp_labels))

        prompts.extend(disallowed_prompts)
        labels.extend(['disallowed' for _ in range(len(disallowed_prompts))])

        with open(f'data_benign.json', 'r') as f:
            data = json.load(f)
            random.shuffle(data)
        safe_prompts = [d["prompt"] for d in data[:int(ratio[2] * num_allow)]]
        prompts.extend(safe_prompts)
        labels.extend(['safe' for _ in range(len(safe_prompts))])

        eval_set = ['chem', 'bio', 'cyber']
        for target_type in target_types:
            if target_type not in eval_set:
                eval_set.append(target_type)
        limit = 150
        eval_dataloader = []
        for dataset in eval_set:
            if dataset_name == "genharm" and args.sr_ablation and dataset in target_types:
                data = ablation_eval_map.get(dataset, [])
                eval_prompts = [d.get('prompt', None) or d.get('question', None) for d in data]
            else:
                with open(f"./data/{dataset_name}/{dataset_name}_{dataset}_train.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                eval_prompts = [d.get('prompt', None) or d.get('question', None) for d in
                                data[:limit]] if dataset not in target_types else [
                    d.get('prompt', None) or d.get('question', None) for d in data]
            eval_dataloader.extend([
                (eval_prompts[i], '{}_allowed'.format(dataset)) if dataset in target_types else (
                    eval_prompts[i], '{}_disallowed'.format(dataset))
                for i in range(len(eval_prompts))
            ])

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
        dataloader = [(prompt_batches[i], labels[i]) for i in range(len(prompt_batches))]

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
                eval_prompts = [d["prompt"] for d in eval_set]

                train_labels = [d['type'] for d in train_set]
                eval_labels = [d['type'] for d in eval_set]

                with open(f'data_benign.json', 'r') as f:
                    data = json.load(f)
                    random.shuffle(data)
                safe_prompts = [d["prompt"] for d in data[:int(ratio[2] // 2 * len(train_prompts))]]
                train_prompts.extend(safe_prompts)
                train_labels.extend(['safe' for _ in range(len(safe_prompts))])

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

                dataloader.extend([(prompt_batches[i], train_labels[i]) for i in range(len(prompt_batches))])
                eval_dataloader.extend([(prompt, label) for prompt, label in zip(eval_prompts, eval_labels)])

    random.shuffle(dataloader)
    return dataloader, eval_dataloader


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name or model path")
    parser.add_argument("--model_resume", type=str, help="path to existing LoRA adapter to merge")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset")
    parser.add_argument("--output_dir", default="../log/", type=str, help="direction of logging file")
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving model")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size.")
    parser.add_argument("--seqlen", type=int, default=128, help="sequence length.")
    parser.add_argument("--seqlen_vision", type=int, default=384, help="sequence length for VLM.")
    parser.add_argument("--let_lr", type=float, default=1e-4, help="LoRA learning rate.")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--wd", type=float, default=0, help="weight decay")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=30, help="Evaluate every N epochs during layer training.")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--multigpu", action="store_true", help="at eval, map model to multiple gpus")
    parser.add_argument(
        "--attn_implementation",
        type=str, default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument("--task_type", type=str, default="text", choices=["text", "vision"])
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--vlm_dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument(
        "--vlm_system_prompt",
        type=str,
        default="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
    )
    parser.add_argument("--vlm_image_root", type=str, default="./data/MMBench")
    parser.add_argument("--vlm_question_field", type=str, default="instruction")
    parser.add_argument("--text_ablation_scale", type=float, default=2.5)
    parser.add_argument("--vision_ablation_scale", type=float, default=2.5)
    parser.add_argument("--sr_ablation", action="store_true",
                        help="Use genharm ablation split: merge *_test_dual and *_train, then split by ratio.")
    parser.add_argument("--sr_test_ratio", type=float, default=0.5)
    parser.add_argument("--sr_train_ratio", type=float, default=1.0)
    parser.add_argument("--sr_split_seed", type=int, default=42)

    # Dataset and target configuration
    parser.add_argument("--dataset_name", type=str, default="genharm", help="Dataset name (e.g., wmdp, genharm)")
    parser.add_argument("--target_types", type=str, nargs="+", default=["Disinformation"],
                        help="Target categories to allow (e.g., cyber, Violence)")
    parser.add_argument("--ratio", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Ratio for [allowed, disallowed, safe] data")
    parser.add_argument("--vlm_all_types", type=str, nargs="+",
                        default=["01-Illegal_Activity", "02-HateSpeech", "04-Physical_Harm", "06-Fraud", "07-Sex", "09-Privacy_Violence"],
                        help="All VLM category types for disallowed pool")

    # Direction ablation configuration
    parser.add_argument("--direction_path", type=str, required=True,
                        help="Path to mean_diffs.pt file containing refusal directions")
    parser.add_argument("--target_layer", type=int, required=True,
                        help="Target layer index for direction ablation")
    parser.add_argument("--direction_pos", type=int, default=-1,
                        help="Position index in the direction tensor")

    args = parser.parse_args()
    if not (0.0 < args.sr_test_ratio < 1.0):
        raise ValueError("--sr_test_ratio must be in (0, 1).")
    if not (0.0 <= args.sr_train_ratio <= 1.0):
        raise ValueError("--sr_train_ratio must be in [0, 1].")

    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)

    # load model
    args.net = args.model.split('/')[-1]
    args.model_family = args.net.split('-')[0]
    args.deactive_amp = False

    if args.task_type == "vision":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, max_pixels=256 * 28 * 28)
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
            model=model,
            tokenizer=processor,
            processor=processor,
            _device=device,
            device=device,
            seqlen=args.seqlen_vision,
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
        logger.info(f"set training on gpu {gpu_id}")

    # prepare data and train
    logger.info("=== start per-layer ablation training ===")
    tick = time.time()

    if args.task_type == "vision":
        target_types = args.target_types
        all_types = args.vlm_all_types
        ratio = args.ratio
        dataloader, eval_dataloader = get_vlm_dataloader(args, lm, target_types, ratio, all_types)
        logger.info(f"Loaded VLM train batches: {len(dataloader)}")
        args.scale = args.vision_ablation_scale
    else:
        ratio = args.ratio
        lm.tokenizer.pad_token = lm.tokenizer.eos_token
        dataset_name = args.dataset_name
        target_types = args.target_types
        dataloader, eval_dataloader = get_dataloader(args, lm, dataset_name, target_types, ratio)
        args.scale = args.text_ablation_scale

    ablation_train(
        lm,
        args,
        dataloader,
        eval_dataloader,
        logger,
    )
    logger.info(f"Training time: {time.time() - tick:.1f}s")

    if args.save_dir:
        lm.model.save_pretrained(args.save_dir)
        lm.tokenizer.save_pretrained(args.save_dir)

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
    print(sys.argv)
    set_seed()
    main()
