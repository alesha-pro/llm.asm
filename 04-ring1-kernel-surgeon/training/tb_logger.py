#!/usr/bin/env python3
"""
Ring -1 Kernel Surgeon — Unified TensorBoard + Dashboard JSONL Logger
Implements all 4 monitoring panels:
  1. panel1_reward     : train_reward, zero_variance_group_frac
  2. panel2_validation : val_pass_at_1, dangerous_continue_rate, excess_halt_rate, baselines
  3. panel3_policy     : token_entropy, action_patch/pass/halt_frac, truncated_frac
  4. panel4_verifier   : queue_depth, p95_ms, vm_pass_frac, episodes_per_min
"""
from __future__ import annotations
import json
import os
import time
from typing import Dict, Optional
from torch.utils.tensorboard import SummaryWriter

class SurgeonMetricsLogger:
    def __init__(self, tb_dir: str, events_jsonl: str, run_id: str):
        os.makedirs(tb_dir, exist_ok=True)
        os.makedirs(os.path.dirname(events_jsonl), exist_ok=True)
        self.tb_dir = tb_dir
        self.events_jsonl = events_jsonl
        self.run_id = run_id
        self.writer = SummaryWriter(log_dir=tb_dir)

    def log_step(self, step: int, metrics: Dict[str, float], phase: str = "train"):
        ts = time.time()
        for k, v in metrics.items():
            self.writer.add_scalar(k, float(v), step, walltime=ts)
        self.writer.flush()
        rec = {"run_id": self.run_id, "step": step, "phase": phase, "ts": ts, "metrics": metrics}
        with open(self.events_jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def close(self):
        self.writer.close()
