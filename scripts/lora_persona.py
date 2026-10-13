import json, torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
tok = AutoTokenizer.from_pretrained(NAME)
rows = json.load(open("data/persona_lora/ramsay_style_preferences.json"))
ds = Dataset.from_list([{"prompt": [{"role": "user", "content": r["prompt"]}],
                         "chosen": [{"role": "assistant", "content": r["chosen"]}],
                         "rejected": [{"role": "assistant", "content": r["rejected"]}]} for r in rows])
ds = ds.train_test_split(test_size=0.1, seed=42)
model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.float32)
peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
cfg = DPOConfig(output_dir="outputs/lora_persona", beta=0.1, learning_rate=5e-5, num_train_epochs=3,
                per_device_train_batch_size=4, gradient_accumulation_steps=4, max_length=512,
                lr_scheduler_type="cosine", warmup_ratio=0.1, fp16=True, logging_steps=5,
                eval_strategy="epoch", save_strategy="no", seed=42, report_to="none")
tr = DPOTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["test"],
                processing_class=tok, peft_config=peft_cfg)   # ref model = adapter disabled
tr.train(); tr.save_model("outputs/lora_persona/adapter")
