"""Usage: make_pairs.py SFT_JSON DPO_JSON OUT_JSONL. Randomised A/B; mapping kept as hidden metadata."""
import json, random, sys
sft, dpo = ({r["id"]: r for r in json.load(open(p))} for p in sys.argv[1:3])
rng = random.Random(42)
with open(sys.argv[3], "w") as f:
    for k in sft:
        if k not in dpo: continue
        flip = rng.random() < 0.5
        A, B = (dpo[k], sft[k]) if flip else (sft[k], dpo[k])
        f.write(json.dumps({"id": k, "instruction": sft[k]["instruction"], "response_a": A["output"],
                            "response_b": B["output"], "_a_is": "dpo" if flip else "sft"}) + "\n")
