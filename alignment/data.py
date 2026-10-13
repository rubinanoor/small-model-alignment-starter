"""Prompt template + dataset loaders. ONE template is used for SFT, DPO and eval."""
import random
from datasets import load_dataset

TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"   # response starts after this


def fmt(instruction):
    return TEMPLATE.format(instruction=instruction.strip())


def load_ultrachat(n_train=30000, n_val=1000, seed=42):
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=seed)
    out = []
    for ex in ds:
        m = ex["messages"]
        if len(m) >= 2 and m[0]["role"] == "user" and m[1]["role"] == "assistant":
            out.append((fmt(m[0]["content"]), m[1]["content"].strip()))   # single-turn only
        if len(out) >= n_train + n_val:
            break
    return out[:n_train], out[n_train:n_train + n_val]


def _split_hh(text):
    # HH format: "\n\nHuman: ...\n\nAssistant: ..."; keep single-turn only
    if text.count("\n\nHuman:") != 1 or text.count("\n\nAssistant:") != 1:
        return None
    h, a = text.split("\n\nAssistant:")
    return h.replace("\n\nHuman:", "").strip(), a.strip()


def load_hh(n_train=15000, n_val=1000, seed=42):
    ds = load_dataset("Anthropic/hh-rlhf", split="train").shuffle(seed=seed)
    out = []
    for ex in ds:
        c, r = _split_hh(ex["chosen"]), _split_hh(ex["rejected"])
        if c and r and c[0] == r[0] and c[1] != r[1]:
            out.append((fmt(c[0]), c[1], r[1]))
        if len(out) >= n_train + n_val:
            break
    return out[:n_train], out[n_train:n_train + n_val]


def load_ultrafeedback(n_train=15000, n_val=1000, seed=42):
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs").shuffle(seed=seed)
    out = []
    for ex in ds:
        ch, rj = ex["chosen"], ex["rejected"]
        if len(ch) == 2 and len(rj) == 2 and ch[1]["content"].strip() != rj[1]["content"].strip():
            out.append((fmt(ex["prompt"]), ch[1]["content"].strip(), rj[1]["content"].strip()))
        if len(out) >= n_train + n_val:
            break
    return out[:n_train], out[n_train:n_train + n_val]
