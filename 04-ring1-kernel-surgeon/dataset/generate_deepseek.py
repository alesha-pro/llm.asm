#!/usr/bin/env python3
"""
Ring -1 Kernel Surgeon — DeepSeek API Teacher Distillation + KVM-Verifier Filter Pipeline

- Calls direct DeepSeek API (https://api.deepseek.com/v1/chat/completions)
  supporting deepseek-v4-pro and deepseek-flash.
- Teacher may reason deeply in reasoning_content / <think>, saved ONLY in teacher_raw.jsonl.
- Strips ALL reasoning/<think>, verifies EVERY clean student answer through the 3-tier KVMVerifier
  (single-level QEMU/KVM + Capstone + postconditions), and writes ONLY verified pairs to
  sft_train.jsonl / sft_val.jsonl / sft_test.jsonl.
- NEVER logs or prints DEEPSEEK_API_KEY.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import httpx

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from dataset.scenarios import Scenario, build_pilot_scenarios
from verifier.kvm_verifier import KVMVerifier, VerifyResult, parse_student_output

SYSTEM_PROMPT = """You are Ring -1 Kernel Surgeon, a bare-metal AMD SVM hypervisor diagnostic & live-patch engine sitting underneath Linux Ring 0.

Given a single fault telemetry snapshot, choose EXACTLY ONE action:
1. PATCH_BYTES — ONLY when ALL safety preconditions hold:
   - cpl=0, extable=0, bug_warn=0, lock_held=0, irq_ctx=0, side_effect=0, stack_ok=1, obs_complete=1.
   - Provide EXACT-LENGTH space-separated x86-64 hex bytes replacing the faulting instruction in place (NO memory operands, NO jumps/calls/privileged instructions).
   - Pay close attention to CONTRACT and REGS!
     * For optional NULL stores or clean UD2/UD1/UD0 where EAX is already 0, a NOP sled (90 ...) of exact length is safe.
     * For divide-by-zero (#DE: div/idiv) or NULL loads / dirty-EAX traps whose CONTRACT requires fallback quotient=0 or EAX=0, use register-zeroing x86-64 bytes of exact length (e.g. `31 c0` for 2B, `31 c0 90` or `48 31 c0` for 3B), NOT blind NOPs that leave non-zero EAX!
2. PASS_TO_LINUX — when cpl=3 (userspace fault), OR extable=1 (Linux _ASM_EXTABLE fixup such as copy_from_user/get_user/futex_detect_cmpxchg), OR bug_warn=1 (Linux WARN_ON_ONCE). PATCH must be NONE.
3. HALT_FOR_DUMP — whenever lock_held=1, irq_ctx=1, side_effect=1, stack_ok=0, or obs_complete=0 (irreversible state, lock/deadlock risk, corrupted stack, or incomplete telemetry). NEVER patch these. PATCH must be NONE.

CRITICAL OUTPUT FORMAT — Output ONLY these exact 3 lines, with NO <think> tags, NO markdown code blocks, and NO extra text:
DIAGNOSIS: <concise technical diagnosis in <= 18 words>
ACTION: <PATCH_BYTES | PASS_TO_LINUX | HALT_FOR_DUMP>
PATCH: <exact-length lowercase space-separated hex bytes OR NONE>"""

# Approximate DeepSeek API list prices (USD per 1M tokens) for budget telemetry
MODEL_PRICING = {
    "deepseek-v4-pro": {"input": 0.55, "output": 2.19},
    "deepseek-flash":  {"input": 0.14, "output": 0.28},
}

def load_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        env_file = os.path.join(ROOT_DIR, ".env")
        if os.path.exists(env_file):
            for line in open(env_file):
                if line.startswith("DEEPSEEK_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("\"\x27")
                    break
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY not found in environment or .env")
    return key

def strip_thinking(text: str) -> Tuple[str, str]:
    """Separate any inline <think>...</think> block from the clean student answer."""
    think_parts = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    return "\n".join(t.strip() for t in think_parts), cleaned

def call_deepseek(client: httpx.Client, api_key: str, model: str, user_prompt: str, max_retries: int = 3) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_err = ""
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            resp = client.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload, timeout=90.0)
            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]
                usage = data.get("usage", {})
                return {
                    "ok": True,
                    "content": msg.get("content", "") or "",
                    "reasoning_content": msg.get("reasoning_content", "") or "",
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage.get("completion_tokens", 0)),
                    "api_ms": (time.time() - t0) * 1000.0,
                }
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(1.5 * (attempt + 1))
    return {"ok": False, "error": last_err, "content": "", "reasoning_content": "", "prompt_tokens": 0, "completion_tokens": 0, "api_ms": 0.0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-flash", choices=["deepseek-v4-pro", "deepseek-flash"])
    ap.add_argument("--scenarios-file", default=os.path.join(ROOT_DIR, "dataset", "scenarios_600.jsonl"))
    ap.add_argument("--limit", type=int, default=600, help="Number of scenarios to run")
    ap.add_argument("--out-dir", required=True, help="Run output directory")
    ap.add_argument("--api-workers", type=int, default=6)
    ap.add_argument("--kvm-workers", type=int, default=4)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    api_key = load_api_key()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.scenarios_file and os.path.exists(args.scenarios_file):
        scenarios = [Scenario(**json.loads(l)) for l in open(args.scenarios_file) if l.strip()][:args.limit]
    else:
        scenarios = build_pilot_scenarios()[:args.limit]

    print(f"[DISTILL] Starting pilot: n={len(scenarios)} model={args.model} out={args.out_dir}")
    t_all = time.time()

    # 1. Query DeepSeek API in parallel
    teacher_results: List[Tuple[Scenario, str, dict]] = []
    with httpx.Client() as client:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.api_workers) as ex:
            fut_map = {ex.submit(call_deepseek, client, api_key, args.model, sc.render_prompt()): sc for sc in scenarios}
            for fut in concurrent.futures.as_completed(fut_map):
                sc = fut_map[fut]
                res = fut.result()
                teacher_results.append((sc, sc.render_prompt(), res))

    # Restore deterministic scenario order
    order = {sc.scenario_id: i for i, sc in enumerate(scenarios)}
    teacher_results.sort(key=lambda x: order[x[0].scenario_id])

    total_ptok = sum(r["prompt_tokens"] for _, _, r in teacher_results)
    total_ctok = sum(r["completion_tokens"] for _, _, r in teacher_results)
    price = MODEL_PRICING.get(args.model, {"input": 0.55, "output": 2.19})
    est_cost_usd = (total_ptok / 1e6) * price["input"] + (total_ctok / 1e6) * price["output"]
    print(f"[DISTILL] API done: {len(teacher_results)} calls | prompt_tok={total_ptok} comp_tok={total_ctok} | est_cost=${est_cost_usd:.5f}")

    # 2. Strip reasoning/<think> and verify EVERY candidate through 3-tier KVMVerifier
    protocol_path = os.path.join(args.out_dir, "verifier_protocol.jsonl")
    if os.path.exists(protocol_path):
        os.unlink(protocol_path)
    verifier = KVMVerifier(max_workers=args.kvm_workers, protocol_log=protocol_path)

    cleaned_answers = []
    raw_records = []
    for sc, prompt, tres in teacher_results:
        inline_think, clean_text = strip_thinking(tres["content"])
        full_reasoning = (tres["reasoning_content"] + ("\n" + inline_think if inline_think else "")).strip()
        # Normalize student target strictly to clean 3-line format
        parsed = parse_student_output(clean_text)
        if parsed.valid_schema:
            student_target = f"DIAGNOSIS: {parsed.diagnosis}\nACTION: {parsed.action}\nPATCH: {parsed.patch_hex}"
        else:
            student_target = clean_text
        cleaned_answers.append((sc, student_target))
        raw_records.append((sc, prompt, tres, full_reasoning, student_target))

    t_kvm = time.time()
    vresults = verifier.verify_batch(cleaned_answers, use_cache=not args.no_cache)
    kvm_dt = time.time() - t_kvm

    # 3. Write audit log (teacher_raw.jsonl) AND clean KVM-verified SFT splits (NO reasoning!)
    raw_path = os.path.join(args.out_dir, "teacher_raw.jsonl")
    disagree_path = os.path.join(args.out_dir, "disagreements.jsonl")
    sft_files = {
        "train": open(os.path.join(args.out_dir, "sft_train.jsonl"), "w"),
        "val":   open(os.path.join(args.out_dir, "sft_val.jsonl"), "w"),
        "test":  open(os.path.join(args.out_dir, "sft_test.jsonl"), "w"),
    }

    sft_counts = Counter()
    confusion = Counter()
    passed_total = 0
    vm_verified_total = 0
    dangerous_total = 0
    excess_halt_total = 0

    with open(raw_path, "w") as fraw, open(disagree_path, "w") as fdis:
        for (sc, prompt, tres, reasoning, student_target), vr in zip(raw_records, vresults):
            if vr.passed:
                passed_total += 1
            if vr.vm_verified:
                vm_verified_total += 1
            if vr.dangerous_continue:
                dangerous_total += 1
            if vr.excess_halt:
                excess_halt_total += 1
            confusion[f"{sc.expected_action}->{vr.pred_action}"] += 1

            audit_rec = {
                "scenario_id": sc.scenario_id,
                "family_id": sc.family_id,
                "category": sc.category,
                "split": sc.split,
                "model": args.model,
                "prompt": prompt,
                "teacher_reasoning": reasoning,
                "student_target": student_target,
                "verify": asdict(vr),
                "usage": {
                    "prompt_tokens": tres["prompt_tokens"],
                    "completion_tokens": tres["completion_tokens"],
                    "api_ms": round(tres["api_ms"], 1),
                },
            }
            fraw.write(json.dumps(audit_rec) + "\n")

            if vr.passed and vr.reward == 1.0:
                # CRITICAL: Clean SFT pair ONLY — zero <think>, zero teacher reasoning!
                sft_rec = {
                    "scenario_id": sc.scenario_id,
                    "family_id": sc.family_id,
                    "category": sc.category,
                    "split": sc.split,
                    "messages": [
                        {"role": "system", "content": "Ring -1 Kernel Surgeon. Output 3 lines only: DIAGNOSIS, ACTION, PATCH."},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": student_target},
                    ],
                    "vm_verified": vr.vm_verified,
                    "boot_id": vr.boot_id,
                }
                sft_files[sc.split].write(json.dumps(sft_rec) + "\n")
                sft_counts[sc.split] += 1
            else:
                fdis.write(json.dumps(audit_rec) + "\n")

    for fh in sft_files.values():
        fh.close()

    vm_lats = sorted(r.latency_ms for r in vresults if r.vm_verified)
    p95_ms = vm_lats[int(0.95 * (len(vm_lats) - 1))] if vm_lats else 0.0
    summary = {
        "model": args.model,
        "total_scenarios": len(scenarios),
        "passed_total": passed_total,
        "pass_at_1": round(passed_total / len(scenarios), 4) if scenarios else 0.0,
        "vm_verified_count": vm_verified_total,
        "dangerous_continue_count": dangerous_total,
        "excess_halt_count": excess_halt_total,
        "sft_verified_counts": dict(sft_counts),
        "confusion": dict(confusion),
        "prompt_tokens": total_ptok,
        "completion_tokens": total_ctok,
        "est_cost_usd": round(est_cost_usd, 6),
        "kvm_elapsed_s": round(kvm_dt, 2),
        "kvm_p95_ms": round(p95_ms, 1),
        "total_elapsed_s": round(time.time() - t_all, 2),
    }
    with open(os.path.join(args.out_dir, "pilot_summary.json"), "w") as fs:
        json.dump(summary, fs, indent=2)

    print("=== PILOT SUMMARY ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
