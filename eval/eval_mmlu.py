from lm_eval.models.huggingface import HFLM
import lm_eval
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.utils import evaluate
import logging
logging.getLogger("lm_eval").setLevel(logging.ERROR)
import argparse


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

    with torch.no_grad():
        for prefix in linear_prefixes:
            weight_key = f"{prefix}.weight"
            if weight_key not in param_dict:
                continue
            lora_a = state_dict[f"{prefix}.lora_A"].to(device=param_dict[weight_key].device, dtype=param_dict[weight_key].dtype)
            lora_b = state_dict[f"{prefix}.lora_B"].to(device=param_dict[weight_key].device, dtype=param_dict[weight_key].dtype)
            delta = torch.matmul(lora_b, lora_a) * scaling
            param_dict[weight_key].add_(delta)

        for key, value in state_dict.items():
            if key.endswith(".lora_A") or key.endswith(".lora_B") or ".base_layer." in key:
                continue
            if key in param_dict:
                param_dict[key].copy_(value.to(device=param_dict[key].device, dtype=param_dict[key].dtype))
            elif key in buffer_dict:
                buffer_dict[key].copy_(value.to(device=buffer_dict[key].device, dtype=buffer_dict[key].dtype))


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA checkpoint")
    parser.add_argument("--lora_layer", type=int, default=None, help="Layer index to merge LoRA into")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--tasks", type=str, default="mmlu")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=float, default=None, help="Limit number of samples per task")
    parser.add_argument("--apply_chat_template", action="store_true", default=False)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    if args.lora_path and args.lora_layer is not None:
        layers = get_transformer_layers(model)
        lora_state = torch.load(args.lora_path, map_location="cpu")
        merge_lora_layer(layers[args.lora_layer], lora_state, args.lora_rank, args.lora_alpha)

    for param in model.parameters():
        param.requires_grad = False

    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    results = lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=[x.strip() for x in args.tasks.split(",")],
        num_fewshot=args.num_fewshot,
        apply_chat_template=args.apply_chat_template,
        limit=args.limit,
    )

    accs = []
    for task, metrics in results["results"].items():
        acc = metrics.get("acc_norm,none") or metrics.get("acc,none")
        if acc is not None:
            accs.append(acc)
    avg_acc = np.mean(accs) * 100
    print(f"Average Acc: {avg_acc:.2f}%")


if __name__ == "__main__":
    main()
