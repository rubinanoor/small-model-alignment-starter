"""Core SFT / DPO math. No Trainer, no TRL: only torch + a tokenizer."""
import torch
import torch.nn.functional as F


def _eos_ids(tokenizer):
    eos = getattr(tokenizer, "eos_token_id", None)
    return [eos] if eos is not None else []


def _encode(tokenizer, text):
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    out = tokenizer(text, add_special_tokens=False)
    return list(out["input_ids"] if isinstance(out, dict) else out.input_ids)


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Right-padded batch. labels = -100 on prompt + padding. EOS appended to the
    response BEFORE truncation (so truncation may drop it)."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    ids_l, lab_l = [], []
    for p, r in zip(prompts, responses):
        p_ids = _encode(tokenizer, p)
        r_ids = _encode(tokenizer, r) + _eos_ids(tokenizer)
        ids = (p_ids + r_ids)[:max_length]
        labels = ([-100] * len(p_ids) + r_ids)[:max_length]
        ids_l.append(ids)
        lab_l.append(labels)
    L = max(1, max(len(x) for x in ids_l))
    input_ids = torch.full((len(ids_l), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(ids_l), L), dtype=torch.long)
    labels = torch.full((len(ids_l), L), -100, dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(ids_l, lab_l)):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        attn[i, :n] = 1
        labels[i, :n] = torch.tensor(lab, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def _logits(model, batch):
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    return out.logits if hasattr(out, "logits") else out


def sequence_logps(logits, labels):
    """Sum of log p(token) over response tokens (labels != -100). Shape (B,).
    Shift: logits at t predict token t+1. Computed in fp32 for stability."""
    logits = logits[:, :-1, :].float()
    tgt = labels[:, 1:]
    mask = tgt != -100
    safe = tgt.masked_fill(~mask, 0)
    logp = torch.log_softmax(logits, dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(-1)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    dev = next(model.parameters()).device
    batch = {k: v.to(dev) for k, v in batch.items()}
    return sequence_logps(_logits(model, batch), batch["labels"])


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Reduction: mean over ALL unmasked response tokens in the batch."""
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    dev = next(model.parameters()).device
    batch = {k: v.to(dev) for k, v in batch.items()}
    logits = _logits(model, batch)[:, :-1, :].float()
    tgt = batch["labels"][:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                           ignore_index=-100, reduction="mean")


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    return {"chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length),
            "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length)}


def dpo_loss_from_logps(pc, pr, rc, rr, beta):
    """Returns (scalar loss, reward_chosen, reward_rejected)."""
    logits = beta * ((pc - pr) - (rc - rr))
    loss = -F.logsigmoid(logits).mean()
    return loss, beta * (pc - rc).detach(), beta * (pr - rr).detach()


def compute_dpo_loss_from_logps(pc, pr, rc, rr, beta):
    return dpo_loss_from_logps(pc, pr, rc, rr, beta)[0]


def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts,
                     chosen_responses, rejected_responses, beta, max_length):
    b = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    ch, rj = b["chosen"], b["rejected"]
    pdev = next(policy_model.parameters()).device
    rdev = next(reference_model.parameters()).device
    pc = sequence_logps(_logits(policy_model, {k: v.to(pdev) for k, v in ch.items()}), ch["labels"].to(pdev))
    pr = sequence_logps(_logits(policy_model, {k: v.to(pdev) for k, v in rj.items()}), rj["labels"].to(pdev))
    with torch.no_grad():  # reference is frozen
        rc = sequence_logps(_logits(reference_model, {k: v.to(rdev) for k, v in ch.items()}), ch["labels"].to(rdev)).to(pdev)
        rr = sequence_logps(_logits(reference_model, {k: v.to(rdev) for k, v in rj.items()}), rj["labels"].to(rdev)).to(pdev)
    return compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
