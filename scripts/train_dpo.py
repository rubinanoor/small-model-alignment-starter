import argparse, json, os, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from alignment.core import build_dpo_batch, sequence_logps, dpo_loss_from_logps, _logits
from alignment import data

ap = argparse.ArgumentParser()
ap.add_argument("--sft", default="outputs/sft/best"); ap.add_argument("--dataset", choices=["hh", "uf"], required=True)
ap.add_argument("--out", required=True); ap.add_argument("--beta", type=float, default=0.1)
ap.add_argument("--lr", type=float, default=1e-6); ap.add_argument("--accum", type=int, default=32)
ap.add_argument("--max_len", type=int, default=512); ap.add_argument("--max_prompt", type=int, default=256)
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--eval_every", type=int, default=50)
a = ap.parse_args()
random.seed(a.seed); torch.manual_seed(a.seed); os.makedirs(a.out, exist_ok=True)

tok = AutoTokenizer.from_pretrained(a.sft)
policy = AutoModelForCausalLM.from_pretrained(a.sft, torch_dtype=torch.float32).to("cuda:0")   # trainable, fp32
policy.gradient_checkpointing_enable(); policy.config.use_cache = False
ref = AutoModelForCausalLM.from_pretrained(a.sft, torch_dtype=torch.float16).to("cuda:1").eval()  # frozen, fp16
for p in ref.parameters(): p.requires_grad_(False)
train, val = (data.load_hh if a.dataset == "hh" else data.load_ultrafeedback)(15000, 1000, a.seed)


def clip_prompt(p):  # enforce max prompt length (256 tokens), keep the END of the prompt
    ids = tok(p, add_special_tokens=False)["input_ids"]
    return p if len(ids) <= a.max_prompt else tok.decode(ids[-a.max_prompt:])


def logps(model, batch, dev, grad):
    b = {k: v.to(dev) for k, v in batch.items()}
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.float16):
        lg = _logits(model, b)
    return sequence_logps(lg, b["labels"])


def step_logps(ex):
    P = [clip_prompt(e[0]) for e in ex]
    b = build_dpo_batch(tok, P, [e[1] for e in ex], [e[2] for e in ex], a.max_len)
    ch, rj = b["chosen"], b["rejected"]
    pc, pr = logps(policy, ch, "cuda:0", True), logps(policy, rj, "cuda:0", True)
    with torch.no_grad():
        rc, rr = logps(ref, ch, "cuda:1", False).to("cuda:0"), logps(ref, rj, "cuda:1", False).to("cuda:0")
    return pc, pr, rc, rr


@torch.no_grad()
def validate():
    policy.eval(); L = acc = mar = 0.0
    for e in val:
        pc, pr, rc, rr = step_logps([e])
        l, r_w, r_l = dpo_loss_from_logps(pc, pr, rc, rr, a.beta)
        L += l.item(); acc += float((r_w > r_l).item()); mar += (r_w - r_l).item()
    policy.train(); n = len(val); return L / n, acc / n, mar / n


opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=0.0)   # fixed LR, no decay
scaler = torch.cuda.amp.GradScaler(); random.shuffle(train)
steps = len(train) // a.accum; best = -1; log = open(f"{a.out}/metrics.jsonl", "a"); policy.train()
for step in range(steps):
    tl = ta = tm = 0.0
    for m in range(a.accum):
        pc, pr, rc, rr = step_logps([train[step * a.accum + m]])        # per-device batch = 1
        l, r_w, r_l = dpo_loss_from_logps(pc, pr, rc, rr, a.beta)
        scaler.scale(l / a.accum).backward()
        tl += l.item() / a.accum; ta += float((r_w > r_l).item()) / a.accum; tm += (r_w - r_l).item() / a.accum
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    rec = {"step": step, "train_loss": tl, "train_acc": ta, "train_margin": tm}
    if step % a.eval_every == 0 or step == steps - 1:
        vl, va, vm = validate(); rec.update(val_loss=vl, val_acc=va, val_margin=vm)
        if va > best:
            best = va; policy.save_pretrained(f"{a.out}/best"); tok.save_pretrained(f"{a.out}/best")
    log.write(json.dumps(rec) + "\n"); log.flush()
policy.save_pretrained(f"{a.out}/final"); tok.save_pretrained(f"{a.out}/final")
