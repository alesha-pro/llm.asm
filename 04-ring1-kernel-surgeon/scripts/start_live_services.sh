#!/usr/bin/env bash
set -euo pipefail
PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-$PROJ/.venv/bin/python}"
TB_BIN="${TB_BIN:-$PROJ/.venv/bin/tensorboard}"
mkdir -p "$PROJ/runs/tb-logs" "$PROJ/runs/service-logs"

# Stop only listeners currently bound to 6006 or 8780
for port in 6006 8780; do
    pids=$(ss -tlnp "sport = :$port" 2>/dev/null | grep -oE "pid=[0-9]+" | cut -d= -f2 | sort -u || true)
    for pid in $pids; do
        kill "$pid" 2>/dev/null || true
    done
done
sleep 1

nohup "$TB_BIN" --logdir "$PROJ/runs/tb-logs" --host 0.0.0.0 --port 6006 \
    >"$PROJ/runs/service-logs/tensorboard.log" 2>&1 </dev/null &
nohup env DASH_HOST=0.0.0.0 DASH_PORT=8780 TB_PORT=6006 "$PY" "$PROJ/dashboard/server.py" \
    >"$PROJ/runs/service-logs/dashboard.log" 2>&1 </dev/null &

sleep 3
ss -tlnp | grep -E ":(6006|8780)\b"
curl -sS -o /dev/null -w "Dashboard http://127.0.0.1:8780 -> HTTP %{http_code}\n" http://127.0.0.1:8780/
curl -sS -o /dev/null -w "TensorBoard http://127.0.0.1:6006 -> HTTP %{http_code}\n" http://127.0.0.1:6006/
