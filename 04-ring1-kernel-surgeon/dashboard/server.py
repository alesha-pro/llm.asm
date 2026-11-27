#!/usr/bin/env python3
"""
Ring -1 Kernel Surgeon — Live Dashboard (FastAPI, zero external CDN deps)
Displays all 4 training/verification telemetry charts + live VM verifier queue + Before/After Surgical Examples!
"""
from __future__ import annotations
import glob
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

PROJ = os.environ.get("SURGEON_PROJ", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
app = FastAPI(title="Ring -1 Kernel Surgeon — Training & KVM Verifier Dashboard")

def find_latest_pilot_dir() -> str:
    env_dir = os.environ.get("SURGEON_PILOT_DIR", "").strip()
    if env_dir and os.path.isdir(env_dir):
        return env_dir
    for sym in ["latest_dataset_600", "latest_pilot"]:
        sp = os.path.join(PROJ, "runs", sym)
        if os.path.isdir(sp):
            return os.path.realpath(sp)
    cands = sorted(
        glob.glob(os.path.join(PROJ, "runs", "*", "dataset_*")) + glob.glob(os.path.join(PROJ, "runs", "*", "pilot_*")),
        key=os.path.getmtime, reverse=True
    )
    return cands[0] if cands else os.path.join(PROJ, "runs")

def find_latest_events_file() -> str:
    env_f = os.environ.get("SURGEON_EVENTS_JSONL", "").strip()
    if env_f and os.path.isfile(env_f):
        return env_f
    cands = sorted(glob.glob(os.path.join(PROJ, "runs", "**", "dashboard_events.jsonl"), recursive=True),
                   key=os.path.getmtime, reverse=True)
    return cands[0] if cands else ""

def load_jsonl(path: str) -> List[dict]:
    if not path or not os.path.isfile(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try: out.append(json.loads(line))
                except Exception: pass
    return out

@app.get("/api/state")
def api_state() -> JSONResponse:
    pilot_dir = find_latest_pilot_dir()
    summary_path = os.path.join(pilot_dir, "pilot_summary.json")
    summary = json.load(open(summary_path)) if os.path.isfile(summary_path) else {}
    raw_items = load_jsonl(os.path.join(pilot_dir, "teacher_raw.jsonl"))
    events = load_jsonl(find_latest_events_file())

    examples = []
    for rec in raw_items:
        v = rec.get("verify", {})
        examples.append({
            "scenario_id": rec.get("scenario_id"),
            "family_id": rec.get("family_id"),
            "category": rec.get("category"),
            "split": rec.get("split"),
            "model": rec.get("model"),
            "prompt": rec.get("prompt"),
            "student_target": rec.get("student_target"),
            "teacher_reasoning": rec.get("teacher_reasoning", "")[:1500],
            "expected_action": v.get("expected_action"),
            "pred_action": v.get("pred_action"),
            "pred_patch_hex": v.get("pred_patch_hex"),
            "disasm_patch": v.get("disasm_patch", ""),
            "reward": v.get("reward"),
            "passed": v.get("passed"),
            "vm_verified": v.get("vm_verified"),
            "dangerous_continue": v.get("dangerous_continue"),
            "excess_halt": v.get("excess_halt"),
            "tier": v.get("tier"),
            "latency_ms": round(float(v.get("latency_ms", 0.0)), 1),
            "detail": v.get("detail", ""),
            "boot_id": v.get("boot_id", ""),
            "uptime": v.get("uptime", ""),
        })

    return JSONResponse({
        "pilot_dir": pilot_dir,
        "summary": summary,
        "events": events,
        "examples": examples,
        "tensorboard_url": f"http://{os.environ.get('SURGEON_HOST', '127.0.0.1')}:{os.environ.get('TB_PORT', '6006')}",
    })

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Ring -1 Kernel Surgeon — Live Training & KVM Verifier</title>
<style>
  :root {
    --bg: #070b14; --panel: #0f1626; --border: #1f2b47; --text: #e5edf8;
    --muted: #8b9bb8; --cyan: #22d3ee; --green: #34d399; --amber: #fbbf24;
    --red: #f87171; --violet: #a78bfa;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
  header { display: flex; justify-content: space-between; align-items: center; padding: 14px 20px; border-bottom: 1px solid var(--border); background: #0a101e; }
  .brand { font-size: 16px; font-weight: 700; color: var(--cyan); letter-spacing: 0.5px; }
  .sub { font-size: 12px; color: var(--muted); }
  .pill { display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 12px; border: 1px solid var(--border); background: #131c31; color: var(--text); text-decoration: none; margin-left: 8px; }
  .pill.live { border-color: #065f46; background: #064e3b; color: #a7f3d0; }
  .wrap { padding: 16px 20px; max-width: 1520px; margin: 0 auto; }
  .kpi-row { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-bottom: 16px; }
  .kpi { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
  .kpi .lbl { font-size: 11px; color: var(--muted); text-transform: uppercase; }
  .kpi .val { font-size: 22px; font-weight: 700; margin-top: 4px; }
  .grid4 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; margin-bottom: 18px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
  .card h3 { margin: 0 0 8px 0; font-size: 13px; color: var(--cyan); display: flex; justify-content: space-between; }
  .card h3 span { color: var(--muted); font-weight: 400; font-size: 11px; }
  svg.chart { width: 100%; height: 170px; background: #090e1a; border-radius: 6px; border: 1px solid #162038; }
  .legend { display: flex; gap: 14px; font-size: 11px; color: var(--muted); margin-top: 6px; flex-wrap: wrap; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }
  .filters { display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
  button.flt { background: #131c31; border: 1px solid var(--border); color: var(--text); padding: 5px 10px; border-radius: 6px; cursor: pointer; font-family: inherit; font-size: 12px; }
  button.flt.active { border-color: var(--cyan); color: var(--cyan); background: #0c2235; }
  .ex-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
  .ex { background: #0b1120; border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
  .ex.pass { border-left: 3px solid var(--green); }
  .ex.fail { border-left: 3px solid var(--red); }
  .ex-hdr { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; font-size: 12px; }
  .tag { padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 700; }
  .tag.pass { background: #064e3b; color: #6ee7b7; }
  .tag.fail { background: #7f1d1d; color: #fecaca; }
  .tag.vm { background: #1e3a8a; color: #93c5fd; }
  pre { background: #060911; border: 1px solid #162038; padding: 8px; border-radius: 6px; font-size: 11px; overflow-x: auto; margin: 6px 0; white-space: pre-wrap; }
  details summary { cursor: pointer; font-size: 11px; color: var(--amber); margin-top: 4px; }
  @media (max-width: 1024px) { .kpi-row { grid-template-columns: repeat(3, 1fr); } .grid4, .ex-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <div>
    <div class="brand">[ RING -1 KERNEL SURGEON ] // TRAINING + SINGLE-LEVEL KVM VERIFIER</div>
    <div class="sub" id="runMeta">Loading run metadata...</div>
  </div>
  <div>
    <span class="pill live" id="liveBadge">LIVE SYNC</span>
    <a class="pill" id="tbLink" href="http://127.0.0.1:6006" target="_blank">Open TensorBoard :6006</a>
  </div>
</header>
<div class="wrap">
  <div class="kpi-row">
    <div class="kpi"><div class="lbl">Pilot Pass@1</div><div class="val" id="kPass" style="color:var(--green)">-</div></div>
    <div class="kpi"><div class="lbl">VM-Verified</div><div class="val" id="kVm" style="color:var(--cyan)">-</div></div>
    <div class="kpi"><div class="lbl">Dangerous Continue</div><div class="val" id="kDanger" style="color:var(--red)">-</div></div>
    <div class="kpi"><div class="lbl">KVM p95 Latency</div><div class="val" id="kP95" style="color:var(--amber)">-</div></div>
    <div class="kpi"><div class="lbl">SFT Clean Pairs</div><div class="val" id="kSft" style="color:var(--violet)">-</div></div>
    <div class="kpi"><div class="lbl">Teacher API Cost</div><div class="val" id="kCost">-</div></div>
  </div>

  <div class="grid4">
    <div class="card">
      <h3>1. Train Reward &amp; Zero-Variance Groups <span>Panel 1</span></h3>
      <svg class="chart" id="c1"></svg>
      <div class="legend"><span><i class="dot" style="background:#34d399"></i>Reward (-2..+1)</span><span><i class="dot" style="background:#fbbf24"></i>Zero-Var Group Frac</span></div>
    </div>
    <div class="card">
      <h3>2. Validation Pass@1 &amp; Safety Rates <span>Panel 2</span></h3>
      <svg class="chart" id="c2"></svg>
      <div class="legend"><span><i class="dot" style="background:#22d3ee"></i>Pass@1</span><span><i class="dot" style="background:#f87171"></i>Dangerous Continue</span><span><i class="dot" style="background:#fbbf24"></i>Excess Halt</span></div>
    </div>
    <div class="card">
      <h3>3. Policy Entropy &amp; Action Mix <span>Panel 3</span></h3>
      <svg class="chart" id="c3"></svg>
      <div class="legend"><span><i class="dot" style="background:#a78bfa"></i>PATCH_BYTES</span><span><i class="dot" style="background:#22d3ee"></i>PASS_TO_LINUX</span><span><i class="dot" style="background:#fbbf24"></i>HALT_FOR_DUMP</span></div>
    </div>
    <div class="card">
      <h3>4. Single-Level KVM Verifier Telemetry <span>Panel 4 (NO savevm)</span></h3>
      <svg class="chart" id="c4"></svg>
      <div class="legend"><span><i class="dot" style="background:#34d399"></i>VM Pass Frac</span><span><i class="dot" style="background:#22d3ee"></i>p95 Latency (s)</span></div>
    </div>
  </div>

  <div class="card">
    <h3>Live Before / After Surgical Examples (KVM-Verified) <span id="exCount"></span></h3>
    <div class="filters" id="filters"></div>
    <div class="ex-grid" id="exGrid"></div>
  </div>
</div>

<script>
let STATE = null;
let ACTIVE_CAT = "ALL";

function drawChart(svgId, seriesList, yMin, yMax) {
  const svg = document.getElementById(svgId);
  const W = 640, H = 170, padL = 36, padR = 14, padT = 14, padB = 22;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  let html = "";
  for (let i = 0; i <= 4; i++) {
    const y = padT + (H - padT - padB) * (i / 4);
    const v = (yMax - (yMax - yMin) * (i / 4)).toFixed(2);
    html += `<line x1="${padL}" y1="${y}" x2="${W-padR}" y2="${y}" stroke="#162038" stroke-width="1"/>`;
    html += `<text x="${padL-4}" y="${y+3}" fill="#64748b" font-size="9" text-anchor="end">${v}</text>`;
  }
  const maxN = Math.max(2, ...seriesList.map(s => s.pts.length));
  for (const s of seriesList) {
    if (!s.pts.length) continue;
    const coords = s.pts.map((v, idx) => {
      const x = padL + (W - padL - padR) * (s.pts.length === 1 ? 0.5 : idx / (maxN - 1));
      const clamped = Math.max(yMin, Math.min(yMax, v));
      const y = padT + (H - padT - padB) * (1 - (clamped - yMin) / (yMax - yMin));
      return [x, y];
    });
    const d = coords.map((c, i) => `${i===0?"M":"L"}${c[0].toFixed(1)},${c[1].toFixed(1)}`).join(" ");
    html += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2"/>`;
    for (const [x, y] of coords) {
      html += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.6" fill="${s.color}"/>`;
    }
  }
  svg.innerHTML = html;
}

function renderExamples() {
  if (!STATE) return;
  const exs = STATE.examples.filter(e => {
    if (ACTIVE_CAT === "ALL") return true;
    if (ACTIVE_CAT === "FAIL") return !e.passed;
    return e.category === ACTIVE_CAT;
  });
  document.getElementById("exCount").textContent = `Showing ${exs.length} / ${STATE.examples.length}`;
  document.getElementById("exGrid").innerHTML = exs.map(e => `
    <div class="ex ${e.passed ? "pass" : "fail"}">
      <div class="ex-hdr">
        <div><strong>${e.scenario_id}</strong> <span style="color:var(--muted)">(${e.split} · ${e.category})</span></div>
        <div>
          ${e.vm_verified ? `<span class="tag vm">VM ${e.latency_ms}ms</span>` : ""}
          <span class="tag ${e.passed ? "pass" : "fail"}">${e.passed ? "PASS" : "FAIL"} r=${e.reward}</span>
        </div>
      </div>
      <div style="font-size:11px;color:var(--muted)">Expected: <b style="color:var(--text)">${e.expected_action}</b> · Pred: <b style="color:var(--cyan)">${e.pred_action}</b> ${e.disasm_patch ? `· Patch ASM: <b style="color:var(--green)">${e.disasm_patch}</b>` : ""}</div>
      <pre>${e.student_target}</pre>
      <div style="font-size:11px;color:${e.passed ? "#6ee7b7" : "#fca5a5"}">${e.tier}: ${e.detail} ${e.boot_id ? `(boot_id=${e.boot_id.slice(0,8)}.. up=${e.uptime}s)` : ""}</div>
      <details><summary>Inspect Input Telemetry &amp; Teacher Reasoning</summary>
        <pre>${e.prompt}</pre>
        ${e.teacher_reasoning ? `<pre style="color:#fcd34d">${e.teacher_reasoning}</pre>` : ""}
      </details>
    </div>
  `).join("");
}

async function refresh() {
  const resp = await fetch("/api/state");
  STATE = await resp.json();
  const s = STATE.summary || {};
  document.getElementById("runMeta").textContent = `Teacher: ${s.model || "-"} | Dir: ${STATE.pilot_dir}`;
  document.getElementById("tbLink").href = STATE.tensorboard_url;
  document.getElementById("kPass").textContent = s.pass_at_1 != null ? `${(s.pass_at_1*100).toFixed(1)}% (${s.passed_total}/${s.total_scenarios})` : "-";
  document.getElementById("kVm").textContent = s.vm_verified_count != null ? `${s.vm_verified_count}` : "-";
  document.getElementById("kDanger").textContent = s.dangerous_continue_count != null ? `${s.dangerous_continue_count}` : "-";
  document.getElementById("kP95").textContent = s.kvm_p95_ms != null ? `${s.kvm_p95_ms} ms` : "-";
  const sftSum = s.sft_verified_counts ? Object.values(s.sft_verified_counts).reduce((a,b)=>a+b,0) : 0;
  document.getElementById("kSft").textContent = `${sftSum}`;
  document.getElementById("kCost").textContent = s.est_cost_usd != null ? `$${s.est_cost_usd.toFixed(4)}` : "-";

  const evs = STATE.events || [];
  const getS = (k) => evs.map(e => e.metrics[k]).filter(v => v !== undefined);
  drawChart("c1", [
    { color: "#34d399", pts: getS("panel1_reward/train_reward") },
    { color: "#fbbf24", pts: getS("panel1_reward/zero_variance_group_frac") },
  ], -2.0, 1.0);
  drawChart("c2", [
    { color: "#22d3ee", pts: getS("panel2_validation/val_pass_at_1") },
    { color: "#f87171", pts: getS("panel2_validation/dangerous_continue_rate") },
    { color: "#fbbf24", pts: getS("panel2_validation/excess_halt_rate") },
  ], 0.0, 1.0);
  drawChart("c3", [
    { color: "#a78bfa", pts: getS("panel3_policy/action_patch_frac") },
    { color: "#22d3ee", pts: getS("panel3_policy/action_pass_frac") },
    { color: "#fbbf24", pts: getS("panel3_policy/action_halt_frac") },
  ], 0.0, 1.0);
  drawChart("c4", [
    { color: "#34d399", pts: getS("panel4_verifier/vm_pass_frac") },
    { color: "#22d3ee", pts: getS("panel4_verifier/p95_ms").map(ms => ms / 1000.0) },
  ], 0.0, 1.5);

  const cats = ["ALL", "pf_patchable", "de_patchable", "ud_patchable", "pass_to_linux", "halt_for_dump", "FAIL"];
  document.getElementById("filters").innerHTML = cats.map(c =>
    `<button class="flt ${ACTIVE_CAT===c?"active":""}" onclick="ACTIVE_CAT='${c}';renderExamples();refreshFilters()">${c}</button>`
  ).join("");
  renderExamples();
}
function refreshFilters() {
  document.querySelectorAll("button.flt").forEach(b => b.classList.toggle("active", b.textContent === ACTIVE_CAT));
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(HTML_PAGE)

if __name__ == "__main__":
    host = os.environ.get("DASH_HOST", "0.0.0.0")
    port = int(os.environ.get("DASH_PORT", "8780"))
    uvicorn.run(app, host=host, port=port, log_level="info")
