"""Greedy generation, identical settings for every model. Usage: generate.py MODEL_DIR NAME EVAL_JSON OUT_JSON"""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from alignment.data import fmt

mdir, name, eval_path, out_path = sys.argv[1:5]
tok = AutoTokenizer.from_pretrained(mdir); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(mdir, torch_dtype=torch.float16).cuda().eval()
items = json.load(open(eval_path)); res = []
for i in range(0, len(items), 8):
    chunk = items[i:i + 8]
    enc = tok([fmt(x["instruction"]) for x in chunk], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=300, do_sample=False, repetition_penalty=1.1,
                             no_repeat_ngram_size=0, pad_token_id=tok.pad_token_id)
    for x, o in zip(chunk, out):
        txt = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        res.append({"id": x.get("id", str(len(res))), "instruction": x["instruction"], "output": txt,
                    "generator": name, "dataset": x.get("dataset", "alpaca_eval")})
    json.dump(res, open(out_path, "w"), indent=1)    # save after every batch
