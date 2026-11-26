#!/usr/bin/env python3
"""Small OpenAI-compatible HTTP server for the native ARM64 Qwen3 worker."""
import argparse
import http.server
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

MODEL_ID = "qwen3-0.6b-arm64"
TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
MAX_BODY = 65536
MAX_PROMPT_CHARS = 3500


class RequestError(Exception):
    def __init__(self, status, message, code="invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.code = code


def one_line(value):
    return " ".join("".join(c if c.isprintable() else " " for c in value).split())


def text_of(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
            else:
                raise RequestError(400, "only text message content is supported")
        return " ".join(parts)
    raise RequestError(400, "message content must be a string or text parts")


def tool_signatures(tools):
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise RequestError(400, "tools must be an array")
    out, seen = [], set()
    for item in tools:
        if not isinstance(item, dict) or item.get("type", "function") != "function":
            raise RequestError(400, "only function tools are supported")
        fn = item.get("function")
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            raise RequestError(400, "tool names must contain only letters, digits, _ or -")
        if name in seen:
            raise RequestError(400, "tool names must be unique")
        seen.add(name)
        params = fn.get("parameters", {})
        if not isinstance(params, dict):
            raise RequestError(400, "tool parameters must be a JSON schema object")
        properties = params.get("properties", {})
        required = params.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise RequestError(400, "tool parameters need object properties and a required array")
        args = []
        for key, spec in properties.items():
            if not isinstance(key, str) or not isinstance(spec, dict):
                continue
            typ = spec.get("type", "value")
            if isinstance(typ, list):
                typ = "/".join(map(str, typ))
            args.append("%s:%s%s" % (key, typ, "!" if key in required else ""))
        out.append("%s(%s)" % (name, ",".join(args)))
    return out


def build_prompt(messages, signatures, tool_choice="auto"):
    if not isinstance(messages, list) or not messages:
        raise RequestError(400, "messages must be a non-empty array")
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise RequestError(400, "each message needs a role")
    if messages[-1]["role"] not in ("user", "tool"):
        raise RequestError(400, "last message must be from the user or a tool")

    forced = None
    required = tool_choice == "required"
    if tool_choice == "none":
        signatures = []
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        fn = tool_choice.get("function")
        forced = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(forced, str) or forced not in {s.split("(", 1)[0] for s in signatures}:
            raise RequestError(400, "tool_choice names an undeclared function")
        signatures = [s for s in signatures if s.startswith(forced + "(")]
        required = True
    elif tool_choice not in (None, "auto", "required"):
        raise RequestError(400, "tool_choice must be auto, none, required, or a named function")
    if required and not signatures:
        raise RequestError(400, "tool_choice requires at least one function tool")

    system = []
    turns = []
    for message in messages[:-1]:
        role = message["role"]
        if role in ("system", "developer"):
            text = one_line(text_of(message.get("content")))
            # OpenCode 2.x can include a UI-only workspace/session advisory in
            # its system context. It is not a user instruction and small models
            # tend to repeat it verbatim, so keep it out of inference prompts.
            lowered = text.lower()
            if "approved for external access" in lowered and "session_move" in lowered:
                continue
            if text:
                system.append(text)
        elif role in ("user", "assistant"):
            text = one_line(text_of(message.get("content")))
            if text:
                turns.append(("User: " if role == "user" else "Assistant: ") + text)
            if role == "assistant":
                for call in message.get("tool_calls", []):
                    fn = call.get("function", {}) if isinstance(call, dict) else {}
                    if isinstance(fn.get("name"), str):
                        args = fn.get("arguments", "{}")
                        if not isinstance(args, str):
                            args = json.dumps(args, separators=(",", ":"))
                        turns.append("Assistant called %s with %s" % (fn["name"], one_line(args)))
        elif role == "tool":
            result = one_line(text_of(message.get("content")))
            if result:
                turns.append("Tool result (%s): %s" % (one_line(message.get("name") or "tool"), result[:700]))
        else:
            raise RequestError(400, "unsupported message role: " + role)

    last = one_line(text_of(messages[-1].get("content")))
    if messages[-1]["role"] == "tool":
        last = "Tool result (%s): %s" % (one_line(messages[-1].get("name") or "tool"), last[:700])
    if not last:
        raise RequestError(400, "last message is empty")
    sys_text = " ".join(system)[-500:]
    active = list(signatures)
    while True:
        tool_text = ""
        if active:
            tool_text = "Available tools: %s. Call using %s{\"name\":\"tool_name\",\"arguments\":{...}}%s." % (
                "; ".join(active), TOOL_OPEN, TOOL_CLOSE)
            if required:
                tool_text += " You must call one listed tool."
        pieces = []
        if sys_text:
            pieces.append("Instructions: " + sys_text)
        if tool_text:
            pieces.append(tool_text)
        if turns:
            pieces.append("Earlier conversation: " + " ".join(turns) + " Now answer:")
        pieces.append(last)
        prompt = " ".join(pieces)
        if len(prompt.encode("utf-8")) <= MAX_PROMPT_CHARS:
            return prompt, {s.split("(", 1)[0] for s in active}
        if turns:
            turns.pop(0)
        elif sys_text:
            sys_text = ""
        elif active and not forced and len(active) > 1:
            active.pop()
        elif active:
            name = forced or active[0].split("(", 1)[0]
            raise RequestError(400, "tool '%s' and the prompt exceed the model context budget" % name,
                               "context_length_exceeded")
        else:
            raise RequestError(400, "last message exceeds the model context budget", "context_length_exceeded")


def visible_text(text):
    leading = text.lstrip()
    if leading.startswith("<think>"):
        end = leading.find("</think>")
        if end >= 0:
            return leading[end + len("</think>"):].lstrip()
    return text.strip()


def parse_tool_calls(text, allowed):
    visible = visible_text(text)
    output, calls = [], []
    cursor = 0
    while True:
        start = visible.find(TOOL_OPEN, cursor)
        if start < 0:
            output.append(visible[cursor:])
            break
        end = visible.find(TOOL_CLOSE, start + len(TOOL_OPEN))
        if end < 0:
            output.append(visible[cursor:])
            break
        output.append(visible[cursor:start])
        raw = visible[start + len(TOOL_OPEN):end]
        try:
            try:
                obj = json.loads(raw.strip())
                name, args = obj.get("name"), obj.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
            except (ValueError, TypeError, AttributeError):
                # Some Qwen3 checkpoints emit structured XML instead of the
                # documented JSON payload inside the same tool_call markers.
                root = ET.fromstring(TOOL_OPEN + raw + TOOL_CLOSE)
                if root.tag != "tool_call":
                    raise ValueError("invalid tool call")
                name_node = root.find("name")
                args_node = root.find("arguments")
                name = name_node.text.strip() if name_node is not None and name_node.text else None
                args = {}
                if args_node is not None:
                    for arg in args_node:
                        value = "".join(arg.itertext()).strip()
                        try:
                            args[arg.tag] = json.loads(value)
                        except (ValueError, TypeError):
                            args[arg.tag] = value
            if name not in allowed or not isinstance(args, dict):
                raise ValueError("invalid tool call")
            calls.append({"id": "call_" + secrets.token_hex(12), "type": "function",
                          "function": {"name": name,
                                       "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":"))}})
        except (ValueError, TypeError, AttributeError, ET.ParseError):
            output.append(visible[start:end + len(TOOL_CLOSE)])
        cursor = end + len(TOOL_CLOSE)
    return "".join(output).strip(), calls


class Worker:
    def __init__(self, binary, model, context, max_tokens):
        self.lock = threading.Lock()
        self.proc = subprocess.Popen([binary, "--stdio-worker", model, str(context), str(max_tokens)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        ready = self.proc.stdout.readline().strip()
        if ready != b"READY":
            self.close()
            raise RuntimeError("ARM64 worker failed to initialize")

    def _read_exact(self, count):
        chunks = []
        while count:
            chunk = self.proc.stdout.read(count)
            if not chunk:
                raise RuntimeError("ARM64 worker exited unexpectedly")
            chunks.append(chunk)
            count -= len(chunk)
        return b"".join(chunks)

    def _frame(self):
        header = self.proc.stdout.readline().decode("ascii").strip().split()
        if len(header) != 2 or header[0] not in ("OK", "ERR"):
            raise RuntimeError("invalid ARM64 worker response")
        payload = self._read_exact(int(header[1]))
        if self.proc.stdout.read(1) != b"\n":
            raise RuntimeError("invalid ARM64 worker frame terminator")
        if header[0] == "ERR":
            raise RequestError(400, payload.decode("utf-8", "replace"), "context_length_exceeded")
        return payload.decode("utf-8", "replace")

    def generate(self, prompt):
        data = prompt.encode("utf-8")
        with self.lock:
            self.proc.stdin.write(b"CLEAR\n")
            self.proc.stdin.flush()
            self._frame()
            self.proc.stdin.write(str(len(data)).encode("ascii") + b"\n" + data + b"\n")
            self.proc.stdin.flush()
            return self._frame()

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


def make_handler(worker):
    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "asm-llm-arm64/0.1"

        def log_message(self, fmt, *args):
            pass

        def reply(self, status, value):
            body = (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/healthz":
                self.reply(200 if worker.proc.poll() is None else 503, {"ok": worker.proc.poll() is None})
            elif self.path == "/v1/models":
                self.reply(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model",
                                                               "created": int(time.time()), "owned_by": "llm.asm"}]})
            else:
                self.reply(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self.reply(404, {"error": {"message": "not found", "type": "invalid_request_error"}})
            try:
                if self.headers.get_content_type() != "application/json":
                    raise RequestError(415, "Content-Type must be application/json")
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > MAX_BODY:
                    raise RequestError(413, "request body must be between 1 and %d bytes" % MAX_BODY)
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict):
                    raise RequestError(400, "request body must be a JSON object")
                if request.get("n", 1) != 1:
                    raise RequestError(400, "only n=1 is supported")
                signatures = tool_signatures(request.get("tools"))
                prompt, allowed = build_prompt(request.get("messages"), signatures,
                                               request.get("tool_choice", "auto"))
                answer = worker.generate(prompt)
                content, calls = parse_tool_calls(answer, allowed)
                model = request.get("model") or MODEL_ID
                completion_id = "chatcmpl-" + secrets.token_hex(12)
                created = int(time.time())
                if request.get("stream"):
                    self.stream(completion_id, created, model, content, calls)
                else:
                    message = {"role": "assistant", "content": content or None}
                    if calls:
                        message["tool_calls"] = calls
                    self.reply(200, {"id": completion_id, "object": "chat.completion", "created": created,
                                     "model": model, "choices": [{"index": 0, "message": message,
                                     "finish_reason": "tool_calls" if calls else "stop"}]})
            except RequestError as exc:
                self.reply(exc.status, {"error": {"message": str(exc), "type": exc.code,
                                                  "param": None, "code": exc.code}})
            except (ValueError, UnicodeDecodeError) as exc:
                self.reply(400, {"error": {"message": "invalid JSON request: " + str(exc),
                                            "type": "invalid_request_error"}})
            except Exception as exc:
                self.reply(503, {"error": {"message": str(exc), "type": "server_error"}})

        def stream(self, completion_id, created, model, content, calls):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            base = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model}

            def emit(value):
                self.wfile.write(("data: " + json.dumps(value, ensure_ascii=False) + "\n\n").encode("utf-8"))
                self.wfile.flush()

            emit(dict(base, choices=[{"index": 0, "delta": {"role": "assistant", "content": ""},
                                      "finish_reason": None}]))
            if content:
                emit(dict(base, choices=[{"index": 0, "delta": {"content": content}, "finish_reason": None}]))
            for index, call in enumerate(calls):
                emit(dict(base, choices=[{"index": 0, "delta": {"tool_calls": [
                    {"index": index, "id": call["id"], "type": "function", "function": call["function"]}]},
                    "finish_reason": None}]))
            emit(dict(base, choices=[{"index": 0, "delta": {},
                                      "finish_reason": "tool_calls" if calls else "stop"}]))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

    return Handler


def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible server for native ARM64 Qwen3")
    parser.add_argument("model", help="path to the exported .asmbin model")
    parser.add_argument("--binary", default=str(Path(__file__).resolve().parents[1] / "build" / "asm-llm-arm64"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    if not Path(args.model).is_file():
        parser.error("model file not found: " + args.model)
    if not Path(args.binary).is_file():
        parser.error("ARM64 binary not found; run `make arm64` first")
    worker = Worker(args.binary, args.model, args.context, args.max_tokens)
    try:
        server = http.server.ThreadingHTTPServer((args.host, args.port), make_handler(worker))
        server.daemon_threads = True

        def request_shutdown(signum, frame):
            threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, request_shutdown)
        signal.signal(signal.SIGINT, request_shutdown)
        print("asm-llm ARM64 API listening on http://%s:%d/v1" % (args.host, args.port), flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()
    finally:
        worker.close()


if __name__ == "__main__":
    main()
