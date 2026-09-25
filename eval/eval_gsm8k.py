import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _unwrap_state_dict(state):
    if not isinstance(state, dict):
        raise ValueError("LoRA checkpoint must be a dict")
    for key in ("state_dict", "model_state_dict", "lora_state_dict"):
        if key in state and isinstance(state[key], dict):
            return state[key]
    return state


def _merge_lora_into_params(param_dict, buffer_dict, state_dict, lora_rank, lora_alpha):
    scaling = lora_alpha / lora_rank
    linear_prefixes = {key[:-7] for key in state_dict.keys() if key.endswith(".lora_A")}
    if not linear_prefixes:
        raise ValueError("No LoRA weights found in the provided checkpoint")

    missing = []
    with torch.no_grad():
        for prefix in linear_prefixes:
            weight_key = f"{prefix}.weight"
            if weight_key not in param_dict:
                missing.append(weight_key)
                continue
            lora_a = state_dict[f"{prefix}.lora_A"].to(device=param_dict[weight_key].device, dtype=param_dict[weight_key].dtype)
            lora_b = state_dict[f"{prefix}.lora_B"].to(device=param_dict[weight_key].device, dtype=param_dict[weight_key].dtype)
            if lora_a.shape[0] != lora_rank or lora_b.shape[1] != lora_rank:
                raise ValueError(
                    f"LoRA rank mismatch for {prefix}: checkpoint rank "
                    f"{lora_a.shape[0]}, requested rank {lora_rank}"
                )
            delta = torch.matmul(lora_b, lora_a) * scaling
            param_dict[weight_key].add_(delta)

        if missing:
            raise KeyError(f"LoRA keys do not match the target layer: {missing}")


def merge_lora_layer(layer, layer_state_dict, lora_rank, lora_alpha):
    state_dict = _unwrap_state_dict(layer_state_dict)
    layer_params = dict(layer.named_parameters())
    layer_buffers = dict(layer.named_buffers())
    _merge_lora_into_params(layer_params, layer_buffers, state_dict, lora_rank, lora_alpha)


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers
    raise ValueError("Unsupported model structure for finding transformer layers")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K (and other lm-eval tasks)")
    parser.add_argument("--base_model", type=str, required=True, help="Model name or path")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA checkpoint")
    parser.add_argument("--lora_layer", type=int, default=None, help="Layer index to merge LoRA into")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--tasks", type=str, default="gsm8k")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument(
        "--apply_chat_template",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype_map[args.dtype], device_map="auto",
    )

    if args.lora_path and args.lora_layer is not None:
        layers = get_transformer_layers(model)
        lora_state = torch.load(args.lora_path, map_location="cpu", weights_only=True)
        merge_lora_layer(layers[args.lora_layer], lora_state, args.lora_rank, args.lora_alpha)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    results = lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=[x.strip() for x in args.tasks.split(",") if x.strip()],
        num_fewshot=args.num_fewshot,
        apply_chat_template=args.apply_chat_template,
        limit=args.limit,
    )

    print(json.dumps(results["results"], ensure_ascii=False, indent=2))

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved full results to: {args.output_path}")


if __name__ == "__main__":
    main()
