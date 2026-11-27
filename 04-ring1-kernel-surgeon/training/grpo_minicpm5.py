#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from collections import Counter
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from dataset.scenarios import Scenario
from verifier.kvm_verifier import KVMVerifier
from training.tb_logger import SurgeonMetricsLogger

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT_DIR, "training", "configs", "grpo_minicpm5_2b.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["TENSORBOARD_LOGGING_DIR"] = cfg["tensorboard_dir"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.get("cuda_visible_devices", "0,1"))

    import torch
    all_sc = [Scenario(**json.loads(l)) for l in open(cfg["scenarios_file"]) if l.strip()]
    train_sc = [s for s in all_sc if s.split == "train"]
    sc_by_id = {s.scenario_id: s for s in all_sc}

    model_path = cfg["base_model_or_sft_ckpt"] if os.path.isdir(cfg["base_model_or_sft_ckpt"]) else cfg["base_model_fallback"]
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if args.dry_run:
        print(json.dumps({
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "visible_gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "model_path": model_path,
            "train_scenarios": len(train_sc),
            "max_steps": cfg["max_steps"],
            "G": cfg["num_generations"],
        }, indent=2))
        return

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    run_id = f"grpo-{int(time.time())}"
    os.makedirs(cfg["output_dir"], exist_ok=True)
    events_jsonl = os.path.join(cfg["output_dir"], "dashboard_events.jsonl")
    protocol_log = os.path.join(cfg["output_dir"], "verifier_protocol.jsonl")
    logger = SurgeonMetricsLogger(cfg["tensorboard_dir"], events_jsonl, run_id) if local_rank == 0 else None
    verifier = KVMVerifier(max_workers=cfg["kvm_verifier"]["max_workers"], protocol_log=protocol_log if local_rank == 0 else None)
    step_counter = {"step": 0}
    G = int(cfg["num_generations"])

    def kvm_reward_func(completions, scenario_id, **kwargs):
        t_r = time.time()
        pairs = []
        for comp, sid in zip(completions, scenario_id):
            text = comp[0]["content"] if isinstance(comp, list) else str(comp)
            pairs.append((sc_by_id[sid], text))

        results = verifier.verify_batch(pairs, use_cache=cfg["kvm_verifier"]["use_cache"])
        rewards = [r.reward for r in results]
        dt = max(time.time() - t_r, 1e-3)

        if logger is not None:
            n = max(1, len(results))
            n_groups = max(1, len(rewards) // G)
            zero_var = sum(
                1 for g in range(n_groups)
                if rewards[g*G:(g+1)*G] and (max(rewards[g*G:(g+1)*G]) - min(rewards[g*G:(g+1)*G]) < 1e-6)
            )
            acts = Counter(r.pred_action for r in results)
            vm_r = [r for r in results if r.vm_verified]
            lats = sorted(r.latency_ms for r in vm_r) if vm_r else [0.0]
            p95 = lats[int(0.95 * (len(lats) - 1))]
            step_counter["step"] += 1
            logger.log_step(step_counter["step"], {
                "panel1_reward/train_reward": sum(rewards) / n,
                "panel1_reward/zero_variance_group_frac": zero_var / n_groups,
                "panel2_validation/val_pass_at_1": sum(1 for r in results if r.passed) / n,
                "panel2_validation/dangerous_continue_rate": sum(1 for r in results if r.dangerous_continue) / n,
                "panel2_validation/excess_halt_rate": sum(1 for r in results if r.excess_halt) / n,
                "panel3_policy/action_patch_frac": acts["PATCH_BYTES"] / n,
                "panel3_policy/action_pass_frac": acts["PASS_TO_LINUX"] / n,
                "panel3_policy/action_halt_frac": acts["HALT_FOR_DUMP"] / n,
                "panel3_policy/truncated_frac": sum(1 for r in results if r.tier == "L1_REJECT") / n,
                "panel4_verifier/p95_ms": p95,
                "panel4_verifier/vm_pass_frac": (sum(1 for r in vm_r if r.passed) / len(vm_r)) if vm_r else 1.0,
                "panel4_verifier/episodes_per_min": (len(results) / dt) * 60.0,
            }, phase="grpo_train")
        return rewards

    # Pre-render non-thinking prompts via MiniCPM5 chat_template (enable_thinking=False!)
    rows = []
    for s in train_sc:
        msgs = [
            {"role": "system", "content": "Ring -1 Kernel Surgeon. Output 3 lines only: DIAGNOSIS, ACTION, PATCH."},
            {"role": "user", "content": s.render_prompt()},
        ]
        prompt_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        rows.append({"prompt": prompt_str, "scenario_id": s.scenario_id})

    ds_train = Dataset.from_list(rows)

    peft_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    grpo_args = GRPOConfig(
        output_dir=cfg["output_dir"],
        # logging_dir set via TENSORBOARD_LOGGING_DIR env
        report_to=["tensorboard"],
        num_generations=G,
        
        max_completion_length=cfg["max_completion_length"],
        beta=cfg["beta"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_steps=5,
        max_steps=int(cfg["max_steps"]),
        bf16=cfg["bf16"],
        gradient_checkpointing=True,
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        seed=cfg["seed"],
        ddp_find_unused_parameters=False,
    )

    trainer = GRPOTrainer(
        model=model_path,
        processing_class=tok,
        reward_funcs=kvm_reward_func,
        train_dataset=ds_train,
        peft_config=peft_cfg,
        args=grpo_args,
    )
    train_res = trainer.train()

    if local_rank == 0:
        lora_dir = os.path.join(cfg["output_dir"], "final_grpo_lora")
        merged_dir = os.path.join(cfg["output_dir"], "merged")
        trainer.save_model(lora_dir)
        tok.save_pretrained(lora_dir)
        print("[GRPO] Merging GRPO LoRA adapter into BF16 checkpoint...")
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        merged = unwrapped.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tok.save_pretrained(merged_dir)
        meta = {
            "global_step": int(train_res.global_step),
            "train_loss": float(train_res.training_loss),
            "elapsed_s": round(time.time() - t0, 1),
            "lora_dir": lora_dir,
            "merged_dir": merged_dir,
        }
        with open(os.path.join(cfg["output_dir"], "grpo_summary.json"), "w") as f:
            json.dump(meta, f, indent=2)
        if logger:
            logger.close()
        print("[GRPO] DONE:", json.dumps(meta))

if __name__ == "__main__":
    main()
