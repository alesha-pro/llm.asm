#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import time
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_jsonl(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT_DIR, "training", "configs", "sft_minicpm5_2b.yaml"))
    ap.add_argument("--train-file", default=None)
    ap.add_argument("--val-file", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    os.environ["TENSORBOARD_LOGGING_DIR"] = cfg["tensorboard_dir"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.get("cuda_visible_devices", "0"))
    train_file = args.train_file or cfg["train_file"]
    val_file = args.val_file or cfg["val_file"]

    import torch
    train_rows = load_jsonl(train_file) if os.path.exists(train_file) else []
    val_rows = load_jsonl(val_file) if os.path.exists(val_file) else []
    model_path = cfg["model_name_or_path"]
    model_ready = os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json"))
    think_leaks = sum(1 for r in train_rows + val_rows if "<think>" in r["messages"][-1]["content"].lower())

    print(json.dumps({
        "mode": "dry_run" if args.dry_run else "train",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "torch_cuda_devices": torch.cuda.device_count(),
        "model_path": model_path,
        "model_ready": model_ready,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "think_leaks": think_leaks,
        "output_dir": cfg["output_dir"],
        "tensorboard_dir": cfg["tensorboard_dir"],
    }, indent=2))
    assert think_leaks == 0, f"Found {think_leaks} <think> leaks in SFT assistant targets!"

    if args.dry_run:
        return
    if not model_ready:
        raise SystemExit(f"Model not ready: {model_path}")

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Note: MiniCPM5 chat_template automatically inserts empty <think>\n\n</think>\n\n before non-thinking content
    # matching generation with add_generation_prompt=True, enable_thinking=False!
    def fmt_row(row):
        return {"text": tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)}

    ds_train = Dataset.from_list([fmt_row(r) for r in train_rows])
    ds_val = Dataset.from_list([fmt_row(r) for r in val_rows])

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_args = SFTConfig(
        output_dir=cfg["output_dir"],
        # logging_dir set via TENSORBOARD_LOGGING_DIR env
        report_to=["tensorboard"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_steps=5,
        weight_decay=cfg["weight_decay"],
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        max_length=cfg["max_seq_length"],
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="epoch",
        seed=cfg["seed"],
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        peft_config=peft_cfg,
        args=sft_args,
    )
    train_res = trainer.train()

    lora_dir = os.path.join(cfg["output_dir"], "final_lora")
    merged_dir = os.path.join(cfg["output_dir"], "merged")
    trainer.save_model(lora_dir)
    tok.save_pretrained(lora_dir)

    print("[SFT] Merging LoRA adapter into BF16 checkpoint for GRPO/GGUF...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)

    meta = {
        "train_loss": float(train_res.training_loss),
        "global_step": int(train_res.global_step),
        "elapsed_s": round(time.time() - t0, 1),
        "lora_dir": lora_dir,
        "merged_dir": merged_dir,
    }
    with open(os.path.join(cfg["output_dir"], "sft_summary.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[SFT] DONE:", json.dumps(meta))

if __name__ == "__main__":
    main()
