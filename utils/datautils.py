import random

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


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
        trainloader.append((inp, tar))

    return trainloader, testenc['input_ids']


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
