import torch
from tqdm import tqdm
import torch.nn as nn
import os
import re
from math import inf
import logging
from termcolor import colored
import sys
import time
from utils.datautils import get_loaders


@torch.no_grad()
def evaluate(model, model_name):
    results = {}
    model.seqlen = 2048
    cache_dir = './cache'
    model_family = re.findall(r"/(.*?)-", model_name)[0]

    for dataset in ["wikitext2"]:
        cache_testloader = f'{cache_dir}/testloader_{model_family}_{dataset}_all.cache'
        if os.path.exists(cache_testloader):
            testloader = torch.load(cache_testloader)
        else:
            dataloader, testloader = get_loaders(
                dataset, seed=42, model=model_name, seqlen=model.seqlen,
            )
            torch.save(testloader, cache_testloader)
        if "c4" in dataset:
            testenc = testloader
        else:
            testenc = testloader.input_ids

        nsamples = testenc.numel() // model.seqlen
        model.config.use_cache = False
        model.eval()
        model = model.to('cuda')
        model = model.to(torch.bfloat16)
        total_nll = 0.0
        total_tokens = 0
        with torch.inference_mode():
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * model.seqlen): ((i + 1) * model.seqlen)].to(model.device)
                attention_mask = torch.ones(batch.shape, dtype=torch.long, device=batch.device)
                with torch.no_grad():
                    outputs = model.model(batch, attention_mask=attention_mask)
                hidden_states = outputs[0]
                logits = model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * model.seqlen): ((i + 1) * model.seqlen)][:, 1:].to(model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss(reduction='sum')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                total_nll += loss.item()
                total_tokens += shift_labels.numel()

        torch.cuda.empty_cache()
        ppl = torch.exp(torch.tensor(total_nll / total_tokens))
        print(f'{dataset} : {ppl.item()}')
        model.config.use_cache = False
        results[dataset] = ppl.item()

    return results


@torch.no_grad()
def ampscaler_get_grad_norm(parameters, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True, retain_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph, retain_graph=retain_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            elif parameters is not None:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            else:
                norm = None
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def create_logger(output_dir, dist_rank=0, name=''):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}_{int(time.time())}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger
