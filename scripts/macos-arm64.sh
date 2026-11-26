#!/bin/sh
# Manage the native Apple Silicon Qwen3 CLI or its local OpenAI-compatible API.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BACKEND="${ARM64_BACKEND:-cpu}"
case "$BACKEND" in
    cpu) BIN="$ROOT/build/asm-llm-arm64"; MAKE_TARGET=arm64 ;;
    metal) BIN="$ROOT/build/asm-llm-arm64-metal"; MAKE_TARGET=arm64-metal ;;
    *) echo "ARM64_BACKEND must be cpu or metal" >&2; exit 2 ;;
esac
SERVER="$ROOT/arm64/openai_server.py"
MODEL="${ARM64_MODEL:-$ROOT/models/qwen3-0.6b-q8_0.asmbin}"
CTX="${ARM64_CTX:-2048}"
TOKENS="${ARM64_TOKENS:-128}"
PORT="${ARM64_HTTP_PORT:-8081}"
PYTHON="${PYTHON:-python3}"
STATE_DIR="$ROOT/build"
CLI_PID="$STATE_DIR/macos-arm64.pid"
SERVER_PID="$STATE_DIR/macos-arm64-server.pid"

stop_pid() {
    label=$1
    pid_file=$2
    expected=$3
    [ -f "$pid_file" ] || { echo "$label is not running"; return 0; }
    pid=$(cat "$pid_file")
    case "$pid" in
        ''|*[!0-9]*) rm -f "$pid_file"; echo "Removed invalid $label PID file"; return 0 ;;
    esac
    command_line=$(ps -p "$pid" -o command= 2>/dev/null || true)
    case "$command_line" in
        *"$expected"*) ;;
        *) rm -f "$pid_file"; echo "$label is not running (removed stale PID file)"; return 0 ;;
    esac
    echo "Stopping $label (PID $pid)"
    kill -TERM "$pid" 2>/dev/null || true
    tries=0
    while kill -0 "$pid" 2>/dev/null && [ "$tries" -lt 8 ]; do
        sleep 1
        tries=$((tries + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
        command_line=$(ps -p "$pid" -o command= 2>/dev/null || true)
        case "$command_line" in *"$expected"*) kill -KILL "$pid" 2>/dev/null || true ;; esac
    fi
    rm -f "$pid_file"
}

prepare() {
    if [ "$(uname -m)" != "arm64" ]; then
        echo "This launcher is for Apple Silicon macOS (arm64)." >&2
        exit 1
    fi
    if [ ! -f "$MODEL" ]; then
        echo "Model not found: $MODEL" >&2
        exit 1
    fi
    echo "Checking native ARM64 $BACKEND executable..."
    make -C "$ROOT" "$MAKE_TARGET"
    mkdir -p "$STATE_DIR"
}

start_cli() {
    prepare
    stop_pid "ARM64 API server" "$SERVER_PID" "$SERVER"
    stop_pid "ARM64 CLI" "$CLI_PID" "$BIN"
    echo "Starting native ARM64 Qwen3 ($BACKEND); type /quit to exit."
    echo "To stop from another terminal: $0 stop"
    echo "Context: $CTX tokens; generation limit: $TOKENS tokens."
    echo "$$" > "$CLI_PID"
    exec "$BIN" "$MODEL" "$CTX" "$TOKENS"
}

start_server() {
    prepare
    stop_pid "ARM64 CLI" "$CLI_PID" "$BIN"
    stop_pid "ARM64 API server" "$SERVER_PID" "$SERVER"
    echo "Starting native ARM64 OpenAI-compatible API on port $PORT"
    echo "OpenCode model: arm64/qwen3-0.6b ($BACKEND)"
    if [ "$BACKEND" = metal ]; then export ASML_METAL_SOURCE="$ROOT/arm64/metal_matvec.metal"; fi
    echo "Press Ctrl-C here or run '$0 server-stop' in another terminal to stop."
    "$PYTHON" "$SERVER" "$MODEL" --binary "$BIN" --host 127.0.0.1 \
        --port "$PORT" --context "$CTX" --max-tokens "$TOKENS" &
    server_pid=$!
    echo "$server_pid" > "$SERVER_PID"
    cleanup_server() {
        if kill -0 "$server_pid" 2>/dev/null; then
            kill -TERM "$server_pid" 2>/dev/null || true
            wait "$server_pid" 2>/dev/null || true
        fi
        rm -f "$SERVER_PID"
    }
    trap cleanup_server EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    wait "$server_pid"
}

status() {
    for item in "ARM64 CLI:$CLI_PID:$BIN" "ARM64 API server:$SERVER_PID:$SERVER"; do
        label=${item%%:*}; rest=${item#*:}; pid_file=${rest%%:*}; expected=${rest#*:}
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file"); command_line=$(ps -p "$pid" -o command= 2>/dev/null || true)
            case "$command_line" in *"$expected"*) echo "$label running (PID $pid)" ;; *) echo "$label not running (stale PID file)" ;; esac
        else
            echo "$label not running"
        fi
    done
    curl -sS --max-time 2 "http://127.0.0.1:$PORT/healthz" 2>/dev/null || true
    echo
}

case "${1:-}" in
    start) start_cli ;;
    stop) stop_pid "ARM64 CLI" "$CLI_PID" "$BIN"; stop_pid "ARM64 API server" "$SERVER_PID" "$SERVER" ;;
    restart) stop_pid "ARM64 CLI" "$CLI_PID" "$BIN"; stop_pid "ARM64 API server" "$SERVER_PID" "$SERVER"; start_cli ;;
    server-start) start_server ;;
    server-restart) stop_pid "ARM64 CLI" "$CLI_PID" "$BIN"; stop_pid "ARM64 API server" "$SERVER_PID" "$SERVER"; start_server ;;
    server-stop) stop_pid "ARM64 API server" "$SERVER_PID" "$SERVER" ;;
    status) status ;;
    *)
        echo "Usage: $0 {start|stop|restart|server-start|server-stop|server-restart|status}" >&2
        exit 2
        ;;
esac
