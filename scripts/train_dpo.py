import argparse, json, os, random, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from alignment.core import build_dpo_batch, sequence_logps, dpo_loss_from_logps, _logits
from alignment import data

ap = argparse.ArgumentParser()
ap.add_argument("--sft", default="outputs/sft/best"); ap.add_argument("--dataset", choices=["hh", "uf"], required=True)
ap.add_argument("--out", required=True); ap.add_argument("--beta", type=float, default=0.1)
ap.add_argument("--lr", type=float, default=1e-6); ap.add_argument("--accum", type=int, default=32)
ap.add_argument("--max_len", type=int, default=512); ap.add_argument("--max_prompt", type=int, default=256)
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--eval_every", type=int, default=100)
ap.add_argument("--eval_bs", type=int, default=2)
ap.add_argument("--n_train", type=int, default=15000); ap.add_argument("--n_val", type=int, default=1000)
ap.add_argument("--dev_policy", default="cuda:0"); ap.add_argument("--dev_ref", default="cuda:1")
ap.add_argument("--no_gc", action="store_true", help="disable gradient checkpointing")
a = ap.parse_args()
random.seed(a.seed); torch.manual_seed(a.seed); os.makedirs(a.out, exist_ok=True)
cuda = a.dev_policy.startswith("cuda")
ref_dtype = torch.float32   # same numerics as the policy (both run under fp16 autocast), so step-0 rewards are ~0

tok = AutoTokenizer.from_pretrained(a.sft)
policy = AutoModelForCausalLM.from_pretrained(a.sft, torch_dtype=torch.float32).to(a.dev_policy)  # trainable, fp32
if not a.no_gc:
    policy.gradient_checkpointing_enable()
policy.config.use_cache = False
ref = AutoModelForCausalLM.from_pretrained(a.sft, torch_dtype=ref_dtype).to(a.dev_ref).eval()      # frozen
for p in ref.parameters(): p.requires_grad_(False)
train, val = (data.load_hh if a.dataset == "hh" else data.load_ultrafeedback)(a.n_train, a.n_val, a.seed)
print(f"train pairs={len(train)} val pairs={len(val)} steps/epoch={len(train) // a.accum}", flush=True)


def clip_prompt(p):  # enforce max prompt length (256 tokens), keep the END of the prompt
    ids = tok.encode(p, add_special_tokens=False)
    return p if len(ids) <= a.max_prompt else tok.decode(ids[-a.max_prompt:])


def logps(model, batch, dev, grad):
    b = {k: v.to(dev) for k, v in batch.items()}
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
        lg = _logits(model, b)
    return sequence_logps(lg, b["labels"])


def step_logps(ex, grad=True):
    P = [clip_prompt(e[0]) for e in ex]
    b = build_dpo_batch(tok, P, [e[1] for e in ex], [e[2] for e in ex], a.max_len)
    ch, rj = b["chosen"], b["rejected"]
    pc, pr = logps(policy, ch, a.dev_policy, grad), logps(policy, rj, a.dev_policy, grad)
    with torch.no_grad():
        rc = logps(ref, ch, a.dev_ref, False).to(a.dev_policy)
        rr = logps(ref, rj, a.dev_ref, False).to(a.dev_policy)
    return pc, pr, rc, rr


@torch.no_grad()
def validate():
    if cuda: torch.cuda.empty_cache()
    policy.eval(); L = acc = mar = 0.0
    for i in range(0, len(val), a.eval_bs):
        chunk = val[i:i + a.eval_bs]
        pc, pr, rc, rr = step_logps(chunk, grad=False)   # NOTE: grad=False, otherwise set_grad_enabled(True) overrides no_grad
        l, r_w, r_l = dpo_loss_from_logps(pc, pr, rc, rr, a.beta)
        L += l.item() * len(chunk); acc += (r_w > r_l).float().sum().item(); mar += (r_w - r_l).sum().item()
    policy.train(); n = len(val); return L / n, acc / n, mar / n


opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=0.0)   # fixed LR, no decay
scaler = torch.amp.GradScaler("cuda", init_scale=2.0**12, enabled=cuda); random.shuffle(train)
steps = len(train) // a.accum; best = -1; log = open(f"{a.out}/metrics.jsonl", "a"); policy.train(); t0 = time.time()
for step in range(steps):
    tl = ta = tm = 0.0
    for m in range(a.accum):
        pc, pr, rc, rr = step_logps([train[step * a.accum + m]])        # per-device batch = 1
        l, r_w, r_l = dpo_loss_from_logps(pc, pr, rc, rr, a.beta)
        scaler.scale(l / a.accum).backward()
        tl += l.item() / a.accum; ta += float((r_w > r_l).item()) / a.accum; tm += (r_w - r_l).item() / a.accum
    scaler.unscale_(opt); gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    rec = {"step": step, "train_loss": tl, "train_acc": ta, "train_margin": tm, "grad_norm": float(gn)}
    if step % a.eval_every == 0 or step == steps - 1:
        vl, va, vm = validate(); rec.update(val_loss=vl, val_acc=va, val_margin=vm)
        if va > best:
            best = va; policy.save_pretrained(f"{a.out}/best"); tok.save_pretrained(f"{a.out}/best")
            rec["saved_best"] = True
    log.write(json.dumps(rec) + "\n"); log.flush()
    print(f"step {step+1}/{steps} loss={tl:.4f} acc={ta:.3f} margin={tm:.4f} gn={float(gn):.2f} "
          + (f"| VAL loss={rec['val_loss']:.4f} acc={rec['val_acc']:.3f} margin={rec['val_margin']:.4f} " if "val_loss" in rec else "")
          + f"[{(time.time()-t0)/60:.1f} min, peak {torch.cuda.max_memory_allocated()/2**30 if cuda else 0:.1f} GB]", flush=True)
policy.save_pretrained(f"{a.out}/final"); tok.save_pretrained(f"{a.out}/final")
print("done", flush=True)
