#!/usr/bin/env python3
"""
Populate real TensorBoard event files & dashboard_events.jsonl from a completed
pilot run (sliding batches of 8 episodes) so TensorBoard (:6006) and Web Dashboard (:8780)
immediately render all 4 monitoring panels with genuine KVM-verified data.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import Counter

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from training.tb_logger import SurgeonMetricsLogger

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", required=True)
    ap.add_argument("--tb-dir", default=os.path.join(ROOT_DIR, "runs", "tb-logs", "pilot-deepseek-v4-pro"))
    args = ap.parse_args()

    raw_path = os.path.join(args.pilot_dir, "teacher_raw.jsonl")
    rows = [json.loads(l) for l in open(raw_path) if l.strip()]
    events_jsonl = os.path.join(args.pilot_dir, "dashboard_events.jsonl")
    if os.path.exists(events_jsonl):
        os.unlink(events_jsonl)

    logger = SurgeonMetricsLogger(args.tb_dir, events_jsonl, run_id=os.path.basename(args.pilot_dir))

    batch_size = 8
    seen = []
    for step_idx, start in enumerate(range(0, len(rows), batch_size), start=1):
        chunk = rows[start:start + batch_size]
        seen.extend(chunk)
        v_chunk = [r["verify"] for r in chunk]
        v_seen = [r["verify"] for r in seen]

        n = len(v_chunk)
        avg_rew = sum(v["reward"] for v in v_chunk) / n
        # Group variance proxy across 4-item sub-blocks
        zero_var = 0
        groups = max(1, n // 4)
        for g in range(groups):
            gr = [v["reward"] for v in v_chunk[g*4:(g+1)*4]]
            if gr and (max(gr) - min(gr) < 1e-6):
                zero_var += 1

        acts = Counter(v["pred_action"] for v in v_chunk)
        vm_c = [v for v in v_chunk if v["vm_verified"]]
        lats = sorted(v["latency_ms"] for v in vm_c) if vm_c else [0.0]
        p95 = lats[int(0.95 * (len(lats) - 1))]

        val_pass = sum(1 for v in v_seen if v["passed"]) / len(v_seen)
        danger_rate = sum(1 for v in v_seen if v["dangerous_continue"]) / len(v_seen)
        excess_rate = sum(1 for v in v_seen if v["excess_halt"]) / len(v_seen)

        logger.log_step(step_idx, {
            "panel1_reward/train_reward": avg_rew,
            "panel1_reward/zero_variance_group_frac": zero_var / groups,
            "panel2_validation/val_pass_at_1": val_pass,
            "panel2_validation/dangerous_continue_rate": danger_rate,
            "panel2_validation/excess_halt_rate": excess_rate,
            "panel3_policy/action_patch_frac": acts["PATCH_BYTES"] / n,
            "panel3_policy/action_pass_frac": acts["PASS_TO_LINUX"] / n,
            "panel3_policy/action_halt_frac": acts["HALT_FOR_DUMP"] / n,
            "panel3_policy/truncated_frac": sum(1 for v in v_chunk if v["tier"] == "L1_REJECT") / n,
            "panel4_verifier/queue_depth": float(len(rows) - len(seen)),
            "panel4_verifier/p95_ms": p95,
            "panel4_verifier/vm_pass_frac": (sum(1 for v in vm_c if v["passed"]) / len(vm_c)) if vm_c else 1.0,
            "panel4_verifier/episodes_per_min": (len(vm_c) / max(sum(lats) / 1000.0 / 4.0, 0.05)) * 60.0 if vm_c else 0.0,
        }, phase="pilot_eval")

    logger.close()
    print(f"[TB] Logged {step_idx} steps -> {args.tb_dir} and {events_jsonl}")

if __name__ == "__main__":
    main()
