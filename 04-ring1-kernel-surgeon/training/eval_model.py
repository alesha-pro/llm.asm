#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from dataset.scenarios import Scenario
from verifier.kvm_verifier import KVMVerifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tag", required=True, help="base | sft | grpo")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--scenarios-file", default=os.path.join(ROOT_DIR, "dataset", "scenarios_600.jsonl"))
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--kvm-workers", type=int, default=8)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    all_sc = [Scenario(**json.loads(l)) for l in open(args.scenarios_file) if l.strip()]
    scenarios = [s for s in all_sc if s.split == args.split]
    print(f"[EVAL:{args.tag}] split={args.split} n={len(scenarios)} model={args.model_path}")

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda:0",
    )
    model.eval()

    prompts = []
    for s in scenarios:
        msgs = [
            {"role": "system", "content": "Ring -1 Kernel Surgeon. Output 3 lines only: DIAGNOSIS, ACTION, PATCH."},
            {"role": "user", "content": s.render_prompt()},
        ]
        p_txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompts.append(p_txt)

    outputs = []
    gen_lats_ms = []
    t_gen0 = time.time()
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda:0")
        t_b = time.time()
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        dt_b = (time.time() - t_b) * 1000.0
        per_item = dt_b / len(batch)
        in_len = enc["input_ids"].shape[1]
        for row in gen:
            txt = tok.decode(row[in_len:], skip_special_tokens=True).strip()
            outputs.append(txt)
            gen_lats_ms.append(per_item)
    gen_total_s = time.time() - t_gen0

    # Free GPU memory before KVM burst
    del model
    torch.cuda.empty_cache()

    verifier = KVMVerifier(max_workers=args.kvm_workers)
    t_v0 = time.time()
    vresults = verifier.verify_batch(list(zip(scenarios, outputs)), use_cache=True)
    v_dt = time.time() - t_v0

    n = len(vresults)
    passed = sum(1 for r in vresults if r.passed)
    vm_ver = [r for r in vresults if r.vm_verified]
    danger = sum(1 for r in vresults if r.dangerous_continue)
    excess = sum(1 for r in vresults if r.excess_halt)
    think_leaks = sum(1 for o in outputs if "<think>" in o.lower())
    avg_rew = sum(r.reward for r in vresults) / n
    conf = Counter(f"{r.expected_action}->{r.pred_action}" for r in vresults)

    by_cat = defaultdict(lambda: {"n": 0, "pass": 0, "danger": 0})
    for s, r in zip(scenarios, vresults):
        by_cat[s.category]["n"] += 1
        by_cat[s.category]["pass"] += int(r.passed)
        by_cat[s.category]["danger"] += int(r.dangerous_continue)

    gen_sorted = sorted(gen_lats_ms)
    vm_lats = sorted(r.latency_ms for r in vm_ver) if vm_ver else [0.0]

    report = {
        "tag": args.tag,
        "model_path": args.model_path,
        "split": args.split,
        "n": n,
        "passed": passed,
        "pass_at_1": round(passed / n, 4),
        "avg_reward": round(avg_rew, 4),
        "dangerous_continue_count": danger,
        "dangerous_continue_rate": round(danger / n, 4),
        "excess_halt_count": excess,
        "excess_halt_rate": round(excess / n, 4),
        "think_leaks": think_leaks,
        "vm_verified_count": len(vm_ver),
        "gpu_gen_p50_ms": round(gen_sorted[len(gen_sorted)//2], 1),
        "gpu_gen_p95_ms": round(gen_sorted[int(0.95*(len(gen_sorted)-1))], 1),
        "gpu_gen_total_s": round(gen_total_s, 2),
        "kvm_verify_s": round(v_dt, 2),
        "kvm_p95_ms": round(vm_lats[int(0.95*(len(vm_lats)-1))], 1),
        "confusion": dict(conf),
        "by_category": {k: dict(v) for k, v in by_cat.items()},
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    with open(args.out_json.replace(".json", "_details.jsonl"), "w") as f:
        for s, o, r in zip(scenarios, outputs, vresults):
            f.write(json.dumps({"scenario_id": s.scenario_id, "category": s.category, "output": o, "verify": asdict(r)}) + "\n")

    print(f"=== EVAL SUMMARY ({args.tag} / {args.split}) ===")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
