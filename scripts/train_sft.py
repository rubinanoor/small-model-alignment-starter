import argparse, json, math, os, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from alignment.core import build_sft_batch, _logits
from alignment.data import load_ultrachat
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="outputs/sft"); ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--bs", type=int, default=2); ap.add_argument("--accum", type=int, default=16)
ap.add_argument("--max_len", type=int, default=512); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--eval_every", type=int, default=100)
a = ap.parse_args()
random.seed(a.seed); torch.manual_seed(a.seed); os.makedirs(a.out, exist_ok=True)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32).cuda()  # fp32 master weights
model.gradient_checkpointing_enable(); model.config.use_cache = False
train, val = load_ultrachat(30000, 1000, a.seed)


def has_response(ex, min_resp=16):   # keep only examples whose response survives truncation to max_len
    return len(tok.encode(ex[0], add_special_tokens=False)) <= a.max_len - min_resp
n0 = len(train), len(val)
train = [ex for ex in train if has_response(ex)]
val = [ex for ex in val if has_response(ex)]
print(f"kept train {len(train)}/{n0[0]}, val {len(val)}/{n0[1]} (dropped prompts with no room for a response)", flush=True)

decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
nodecay = [p for n, p in model.named_parameters() if p.ndim < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}], lr=a.lr)
steps = len(train) // (a.bs * a.accum); warm = max(1, int(0.03 * steps))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                          0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
scaler = torch.cuda.amp.GradScaler()


def loss_fn(batch):
    batch = {k: v.cuda() for k, v in batch.items()}
    with torch.autocast("cuda", dtype=torch.float16):
        lg = _logits(model, batch)
    lg = lg[:, :-1].float(); tgt = batch["labels"][:, 1:]
    return F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1), ignore_index=-100)


@torch.no_grad()
def validate():
    model.eval(); tot = n = 0
    for i in range(0, len(val), 4):
        b = build_sft_batch(tok, [p for p, _ in val[i:i+4]], [r for _, r in val[i:i+4]], a.max_len)
        tot += loss_fn(b).item(); n += 1
    model.train(); return tot / n


best = 1e9; random.shuffle(train); log = open(f"{a.out}/metrics.jsonl", "a"); model.train()
for step in range(steps):
    run = 0.0
    for m in range(a.accum):
        i = (step * a.accum + m) * a.bs; ex = train[i:i + a.bs]
        b = build_sft_batch(tok, [p for p, _ in ex], [r for _, r in ex], a.max_len)
        if (b["labels"][:, 1:] != -100).sum() == 0:
            continue   # nothing to learn from; avoids a NaN loss that makes GradScaler skip the step
        l = loss_fn(b) / a.accum; scaler.scale(l).backward(); run += l.item()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
    rec = {"step": step, "train_loss": run, "lr": sched.get_last_lr()[0]}
    if step % a.eval_every == 0 or step == steps - 1:
        v = validate(); rec["val_loss"] = v
        if v < best:
            best = v; model.save_pretrained(f"{a.out}/best"); tok.save_pretrained(f"{a.out}/best")
    log.write(json.dumps(rec) + "\n"); log.flush()
model.save_pretrained(f"{a.out}/final"); tok.save_pretrained(f"{a.out}/final")
