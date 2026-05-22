import pdb
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import torch
import random
from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_pile(nsamples, seed, seqlen, model):
    print("get_pile")
    traindata = load_dataset("json", data_files='/cpfs01/user/chenmengzhao/prompt_quantization/val.jsonl.zst',
                             split="train")

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text'][:1000]), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, None

# def get_wikitext2(nsamples, seed, seqlen, model, prompt_mode="plain"):
#     """
#     prompt_mode: "plain" | "chat"
#       - plain:  在上下文前加简洁任务说明前缀
#       - chat:   用 tokenizer.apply_chat_template 包装成 system/user 指令消息
#     """
#     print("get_wikitext2 (with task prompt)")
#
#     # 1) 数据与分词
#     traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
#     testdata  = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
#
#     tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
#     # decoder-only 常见：没 pad_token 就用 eos 兜底，便于 mask/pad
#     if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
#         tokenizer.pad_token = tokenizer.eos_token
#
#     trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
#     testenc  = tokenizer("\n\n".join(testdata['text']),  return_tensors='pt')
#
#     full_ids = trainenc.input_ids  # [1, T]
#     T = full_ids.shape[1]
#
#     # 2) 构造“任务显式化”的前缀 prompt tokens
#     def build_prompt_ids():
#         if prompt_mode == "chat" and hasattr(tokenizer, "apply_chat_template"):
#             messages = [
#                 {"role": "system", "content": "You are a helpful assistant for next-token prediction."},
#                 {"role": "user",   "content": "Given the context below, predict the next token. Reply with only the next token."}
#             ]
#             p_ids = tokenizer.apply_chat_template(
#                 messages, add_generation_prompt=True, tokenize=True, return_tensors=None
#             )
#             return torch.tensor(p_ids, dtype=torch.long).unsqueeze(0)  # [1, P]
#         else:
#             prefix = "Task: Predict the next token given the context below.\nContext:\n"
#             # 简洁前缀，尽量省 token
#             p_ids = tokenizer.encode(prefix, add_special_tokens=False)
#             return torch.tensor(p_ids, dtype=torch.long).unsqueeze(0)  # [1, P]
#
#     prompt_ids = build_prompt_ids()                 # [1, P]
#     P = prompt_ids.shape[1]
#     assert P < seqlen - 1, "Prompt 过长，导致没有足够空间放上下文与目标 token，请缩短 prompt 或增大 seqlen。"
#
#     # 3) 采样 nsamples 个窗口： [prompt | context_window]，只对最后一位计损失
#     random.seed(seed)
#     trainloader = []
#
#     # 每个样本可用的“纯上下文长度”
#     ctx_len = seqlen - P
#     # 保证最后一位是监督目标，至少需要 2 个上下文 token
#     ctx_len = max(ctx_len, 2)
#
#     for _ in range(nsamples):
#         # 随机选择一个“上下文末端”位置 e，使得 [e-ctx_len, e) 可用
#         e = random.randint(ctx_len, T - 1)     # 末端索引（开区间右端），label 为位置 e-1 的 token
#         s = e - ctx_len                        # 起点
#         ctx = full_ids[:, s:e]                 # [1, ctx_len]
#
#         # 拼接： [prompt_ids | ctx] -> [1, seqlen]
#         inp = torch.cat([prompt_ids, ctx], dim=1)
#         # 只在最后一个 token 位置计损失（与原函数一致）
#         tar = inp.clone()
#         tar[:, :-1] = -100
#
#         trainloader.append((inp, tar))
#
#     # 4) testenc 保持原样，便于后续滑窗评测
#     return trainloader, testenc

class ActionTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, bins: int = 256, min_action: int = -1, max_action: int = 1
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)

        # Handle single element vs. batch
        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))
        else:
            return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_action).tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs.

        NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
                 digitization returns bin indices between [1, # bins], inclusive, when there are actually only
                 (# bins - 1) bin intervals.

                 Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

        EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                    indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                    is still one index (i==255) that would cause an out-of-bounds error if used to index into
                    self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                    the last bin center. We implement this simply via clipping between [0, 255 - 1].
        """
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins


def get_wikitext2(nsamples, seed, seqlen, model):
    print("get_wikitext2")
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    tokenizer = AutoTokenizer.from_pretrained(model)
    eos = tokenizer.eos_token

    trainenc = tokenizer(eos.join(traindata['text']), return_tensors='pt')
    testenc = tokenizer(eos.join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = inp[:, 1:]
        tar[:, -1] = -100
        # tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_mixed_c4_wikitext2(nsamples, seed, seqlen, model):
    n_c4 = nsamples // 2
    n_wt2 = nsamples - n_c4

    c4_train, c4_valenc = get_c4(n_c4, seed, seqlen, model)                 # c4_valenc: torch.Tensor, shape [1, Kc4]
    wt2_train, wt2_testenc = get_wikitext2(n_wt2, seed, seqlen, model)       # wt2_testenc: BatchEncoding

    mixed_train = []
    for i in range(max(len(c4_train), len(wt2_train))):
        if i < len(c4_train):
            mixed_train.append(c4_train[i])
        if i < len(wt2_train):
            mixed_train.append(wt2_train[i])

    mixed_train = mixed_train[:nsamples]

    if hasattr(wt2_testenc, "input_ids"):
        wt2_ids = wt2_testenc.input_ids
    else:
        wt2_ids = wt2_testenc["input_ids"]

    L_c4 = c4_valenc.shape[1]
    L_wt2 = wt2_ids.shape[1]

    L_equal = min(L_c4, L_wt2)
    half_tokens = (L_equal // 2 // seqlen) * seqlen
    if half_tokens == 0:
        half_tokens = max(0, min(L_c4, L_wt2, seqlen))

    mixed_test = torch.hstack([
        c4_valenc[:, :half_tokens],
        wt2_ids[:, :half_tokens]
    ])

    return mixed_train, mixed_test


def get_ptb(nsamples, seed, seqlen, model):
    print("get_ptb")
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train',trust_remote_code=True)
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation',trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, model):
    print("get_c4")
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', trust_remote_code=True
    )
    traindata = traindata.train_test_split(test_size=40000, seed=seed)['test']
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation',trust_remote_code=True
    )
    valdata = valdata.train_test_split(test_size=256, seed=seed)['test']

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    eos = tokenizer.eos_token

    trainenc = tokenizer(eos.join(traindata['text']), return_tensors='pt')
    testenc = tokenizer(eos.join(valdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = inp[:, 1:]
        tar[:, -1] = -100
        # tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc['input_ids']

    # random.seed(seed)
    # trainloader = []
    # for _ in range(nsamples):
    #     while True:
    #         i = random.randint(0, len(traindata) - 1)
    #         trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
    #         if trainenc.input_ids.shape[1] > seqlen:
    #             break
    #     i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
    #     j = i + seqlen
    #     inp = trainenc.input_ids[:, i:j]
    #     tar = inp.clone()
    #     tar[:, :-1] = -100
    #     trainloader.append((inp, tar))
    #
    # random.seed(0)
    # valenc = []
    # for _ in range(256):
    #     while True:
    #         i = random.randint(0, len(valdata) - 1)
    #         tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
    #         if tmp.input_ids.shape[1] >= seqlen:
    #             break
    #     i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
    #     j = i + seqlen
    #     valenc.append(tmp.input_ids[:, i:j])
    # valenc = torch.hstack(valenc)
    #
    # return trainloader, valenc


def get_ptb_new(nsamples, seed, seqlen, model):
    print("get_ptb_new")
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model):
    print("get_c4_new")
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]
    return trainloader, valenc


def get_loaders(
        name, nsamples=128, seed=0, seqlen=2048, model='',
):
    if 'wikitext2' in name and 'c4' in name:
        wiki_train, wiki_val = get_wikitext2(nsamples // 2, seed, seqlen, model)
        c4_train, c4_val = get_c4(nsamples // 2, seed, seqlen, model)
        train = wiki_train + c4_train
        val = None
        return train, val
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'pile' in name:
        return get_pile(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)
    if 'mix' in name:
        wiki_train, wiki_val = get_wikitext2(nsamples // 3, seed, seqlen, model)
        ptb_train, ptb_val = get_ptb(nsamples // 3, seed, seqlen, model)
        c4_train, c4_val = get_c4(nsamples // 3, seed, seqlen, model)
        train = wiki_train + ptb_train + c4_train
        val = None
        return train, val
