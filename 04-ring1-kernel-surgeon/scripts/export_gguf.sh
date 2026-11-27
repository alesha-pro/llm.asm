#!/usr/bin/env bash
set -euo pipefail
PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_DIR="${1:?usage: export_gguf.sh <merged_hf_dir> <out_dir> <base_name>}"
OUT_DIR="${2:?missing out_dir}"
NAME="${3:-minicpm5-2b-surgeon}"

LLAMA_DIR="${LLAMA_DIR:-/opt/llama.cpp}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
if [[ -x "$PROJ/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJ/.venv/bin/python"
fi
CONVERT="$LLAMA_DIR/convert_hf_to_gguf.py"
QUANT="$LLAMA_DIR/build/bin/llama-quantize"
BENCH="$LLAMA_DIR/build/bin/llama-bench"

mkdir -p "$OUT_DIR"
F16_GGUF="$OUT_DIR/${NAME}-f16.gguf"
Q8_GGUF="$OUT_DIR/${NAME}-q8_0.gguf"
Q4_GGUF="$OUT_DIR/${NAME}-q4_k_m.gguf"

echo "[GGUF] Converting $MODEL_DIR -> $F16_GGUF ..."
"$PYTHON_BIN" "$CONVERT" "$MODEL_DIR" --outfile "$F16_GGUF" --outtype f16

echo "[GGUF] Quantizing -> Q8_0 and Q4_K_M ..."
"$QUANT" "$F16_GGUF" "$Q8_GGUF" Q8_0
"$QUANT" "$F16_GGUF" "$Q4_GGUF" Q4_K_M

# Remove intermediate F16 GGUF to save disk space once Q8_0 and Q4_K_M succeed
rm -f "$F16_GGUF"
ls -lh "$Q8_GGUF" "$Q4_GGUF"

echo "[GGUF] Benchmarking Q4_K_M and Q8_0 on EPYC CPU (16 physical threads, p256 / g64)..."
"$BENCH" -m "$Q4_GGUF" -t 16 -p 256 -n 64 -r 3 -ngl 0
"$BENCH" -m "$Q8_GGUF" -t 16 -p 256 -n 64 -r 3 -ngl 0
