#!/usr/bin/env bash
# Launch TensorBoard (:6006) + Ring -1 Web Dashboard (:8780) on demand for video/inspection.
set -euo pipefail

PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TB_DIR="${TB_DIR:-$PROJ/runs/tb-logs}"
TB_PORT="${TB_PORT:-6006}"
DASH_PORT="${DASH_PORT:-8780}"
HOST_ADDR="${HOST_ADDR:-127.0.0.1}"
mkdir -p "$TB_DIR"

echo "================================================================"
echo " Ring -1 Kernel Surgeon — TensorBoard + Live Web Dashboard"
echo "================================================================"
echo " Web Dashboard : http://${HOST_ADDR}:${DASH_PORT}"
echo " TensorBoard   : http://${HOST_ADDR}:${TB_PORT}"
echo " Logdir        : ${TB_DIR}"
echo " Press Ctrl+C to stop both cleanly."
echo "================================================================"

"$PROJ/.venv/bin/tensorboard" --logdir "$TB_DIR" --host 0.0.0.0 --port "$TB_PORT" >/dev/null 2>&1 &
TB_PID=$!
trap 'kill "$TB_PID" 2>/dev/null || true' EXIT INT TERM

DASH_PORT="$DASH_PORT" TB_PORT="$TB_PORT" "$PROJ/.venv/bin/python" "$PROJ/dashboard/server.py"
