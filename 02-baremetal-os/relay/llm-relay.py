#!/usr/bin/python3 -I
"""llm-relay: an OpenAI-compatible HTTP API for llm.asm, the bare-metal Qwen3-0.6B OS.

llm.asm (https://github.com/alesha-pro/llm.asm, 02-baremetal-os) boots a model straight on x86-64
with no OS underneath and talks only through its serial console. Point the VM's COM1 at this relay
as a TCP client, and the relay types prompts into the console and reads the answers back.

Endpoints:
  GET  /v1/models
  POST /v1/chat/completions   (stream true or false)
  POST /v1/completions        (stream true or false)
  POST /v1/generate           {"prompt": "..."} -> {"text", "prompt_tokens", "gen_tokens", ...}
  GET  /healthz               200 when the console is connected and at its prompt, else 503
  GET  /metrics               Prometheus text format

Console behaviour this relies on (llm.asm 02-baremetal-os, kernel.asm and drivers.asm):
  - the prompt is "\\nasm-llm> "; a line is submitted with CR; printable input is echoed and Enter
    echoes "\\n"; output uses bare LF;
  - each answer is followed by "[asm-llm-os] prefill: ..." and "[asm-llm-os] decode : ..." lines,
    then the prompt again;
  - the lines "exit", "/halt", "/quit" and a Ctrl-D byte power the VM off, "/clear" resets the
    conversation, and a line holds at most 1023 bytes.
The relay never lets a request send a console command or a control byte, and clears the
conversation before every request, so requests are independent of each other.

Prompt text is never logged; the log has sizes, token counts and timings only.
Python 3.8+ standard library only. Configuration is by environment variable (see CONFIG below).
"""
import codecs
import json
import os
import re
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def env(name, default):
    return os.environ.get(name, default)


CONFIG = {
    # Where QEMU's chardev client connects (the VM's COM1), and which peer addresses may connect.
    "serial_addr": env("LLM_SERIAL_ADDR", "127.0.0.1"),
    "serial_port": int(env("LLM_SERIAL_PORT", "7000")),
    "serial_peers": [p.strip() for p in env("LLM_SERIAL_PEER", "127.0.0.1").split(",") if p.strip()],
    # The HTTP API.
    "http_addr": env("LLM_HTTP_ADDR", "127.0.0.1"),
    "http_port": int(env("LLM_HTTP_PORT", "8080")),
    # Optional: require "Authorization: Bearer <key>" on /v1/*. Unset means no key is required.
    "api_key": env("LLM_API_KEY", ""),
    # The model id shown by /v1/models and echoed in responses.
    "model_id": env("LLM_MODEL_ID", "qwen3-0.6b-llm.asm"),
    # The max_tokens the VM was booted with (kernel cmdline max_tokens=, llm.asm default 128).
    # Used only to report finish_reason "length"; the relay cannot change it per request.
    "boot_max_tokens": int(env("LLM_BOOT_MAX_TOKENS", "128")),
    "max_prompt_bytes": int(env("LLM_MAX_PROMPT_BYTES", "1000")),
    "max_waiting": int(env("LLM_MAX_WAITING", "4")),
    "gen_timeout": float(env("LLM_GEN_TIMEOUT", "120")),
    "cmd_timeout": float(env("LLM_CMD_TIMEOUT", "15")),
}
MAX_BODY = 65536
STARTED = int(time.time())

PROMPT = b"\nasm-llm> "
POWEROFF = b"Powering off"
TRAILER = b"\n[asm-llm-os] prefill: "
COMMANDS = {"exit", "/halt", "/quit", "/clear", "/new", "/reset", "/think", "/nothink", "/bench", "/help"}
STATS = re.compile(
    rb"\[asm-llm-os\] prefill: (\d+) tok in ([\d.]+) ms \(([\d.]+) tok/s\)\r?\n"
    rb"\[asm-llm-os\] decode : (\d+) tok in ([\d.]+) ms \(([\d.]+) tok/s, cores=(\d+)\) ctx=(\d+)/(\d+)\r?\n"
    rb"\r?\nasm-llm> $"
)


def log(msg):
    print(msg, flush=True)


class RelayError(Exception):
    def __init__(self, status, message, etype="server_error", code=None):
        super().__init__(message)
        self.status, self.message, self.etype, self.code = status, message, etype, code


# ---------------------------------------------------------------------------------------------
# The console
# ---------------------------------------------------------------------------------------------
class Console:
    """One TCP connection from QEMU at a time; a newer connection replaces the older one."""

    def __init__(self):
        self.cv = threading.Condition()
        self.conn = None
        self.gen = 0            # bumps on every (re)connect, so waiters notice
        self.buf = bytearray()
        self.ready = False
        self.powered_off = False
        self.poweroffs = 0
        self.connects = 0

    def serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((CONFIG["serial_addr"], CONFIG["serial_port"]))
        srv.listen(2)
        log("serial listening on %s:%d, peers %s" % (CONFIG["serial_addr"], CONFIG["serial_port"],
                                                     ",".join(CONFIG["serial_peers"])))
        while True:
            c, (ip, _) = srv.accept()
            if ip not in CONFIG["serial_peers"]:
                log("serial: refused connection from %s" % ip)
                c.close()
                continue
            c.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            with self.cv:
                if self.conn:
                    try:
                        self.conn.close()
                    except OSError:
                        pass
                self.conn = c
                self.gen += 1
                self.connects += 1
                self.buf.clear()
                self.ready = False
                self.powered_off = False
                gen = self.gen
                self.cv.notify_all()
            log("serial: console connected")
            threading.Thread(target=self._reader, args=(c, gen), daemon=True).start()
            threading.Thread(target=self._poke, args=(gen,), daemon=True).start()

    def _reader(self, c, gen):
        while True:
            try:
                data = c.recv(65536)
            except OSError:
                data = b""
            with self.cv:
                if gen != self.gen:
                    return
                if not data:
                    self.conn = None
                    self.ready = False
                    self.cv.notify_all()
                    log("serial: console disconnected")
                    return
                self.buf += data
                if len(self.buf) > 1 << 20:          # never hold more than 1 MiB of console output
                    del self.buf[: len(self.buf) - (1 << 19)]
                if POWEROFF in self.buf:
                    if not self.powered_off:
                        self.poweroffs += 1
                        log("serial: the VM reported power-off")
                    self.powered_off = True
                    self.ready = False
                elif self.buf.endswith(PROMPT):
                    self.ready = True
                self.cv.notify_all()

    def _poke(self, gen):
        """After a relay restart the VM is already at its prompt and prints nothing. An empty line
        makes the console print its prompt again."""
        deadline = time.monotonic() + 3
        with self.cv:
            while not self.ready and gen == self.gen and time.monotonic() < deadline:
                self.cv.wait(deadline - time.monotonic())
            if self.ready or gen != self.gen or self.conn is None:
                return
            self._send(b"\r")

    def _send(self, data):
        try:
            self.conn.sendall(data)
        except OSError as e:
            log("serial: send failed: %s" % e)
            self.ready = False

    def _lost(self, gen):
        return gen != self.gen or self.conn is None or self.powered_off

    def run(self, prompt_bytes):
        """Generator, to be iterated with the request lock held. Yields raw answer bytes as they
        arrive, then one dict of stats. Raises RelayError if the console is unusable."""
        with self.cv:
            if self.conn is None or self.powered_off or not self.ready:
                raise RelayError(503, "the model console is not ready", "service_unavailable")
            gen = self.gen
            # 1. Independent requests: clear the conversation first.
            self.buf.clear()
            self.ready = False
            self._send(b"/clear\r")
            deadline = time.monotonic() + CONFIG["cmd_timeout"]
            while not self.buf.endswith(PROMPT):
                left = deadline - time.monotonic()
                if self._lost(gen) or left <= 0:
                    raise RelayError(504, "the model console did not answer /clear", "timeout")
                self.cv.wait(left)
            # 2. The prompt.
            self.buf.clear()
            self._send(prompt_bytes + b"\r")
        t0 = time.monotonic()
        deadline = t0 + CONFIG["gen_timeout"]
        echo = prompt_bytes + b"\n"
        start = None            # where the answer begins, after the echo
        pos = 0                 # how far the answer has been yielded
        seen = 0
        while True:
            with self.cv:
                while len(self.buf) == seen:
                    left = deadline - time.monotonic()
                    if self._lost(gen):
                        raise RelayError(503, "the model console went away during the answer", "service_unavailable")
                    if left <= 0:
                        raise RelayError(504, "no answer within %ds" % CONFIG["gen_timeout"], "timeout")
                    self.cv.wait(left)
                data = bytes(self.buf)
                seen = len(data)
                m = STATS.search(data)
                if m:
                    self.ready = True
            if start is None:
                if data.startswith(echo):
                    start = len(echo)
                elif b"\n" in data and not echo.startswith(data[: len(echo)]):
                    start = data.index(b"\n") + 1
                else:
                    continue
                pos = start
            if m:
                end = m.start()
                if end > pos and data[end - 1:end] == b"\n":
                    end -= 1
                if end > pos:
                    yield data[pos:end]
                yield {
                    "prompt_tokens": int(m.group(1)),
                    "gen_tokens": int(m.group(4)),
                    "prefill_tok_s": float(m.group(3)),
                    "decode_tok_s": float(m.group(6)),
                    "cores": int(m.group(7)),
                    "seconds": round(time.monotonic() - t0, 3),
                }
                return
            # Hold back anything that could be the start of the stats trailer.
            i = data.find(TRAILER, pos)
            if i >= 0:
                safe = i
            else:
                safe = len(data)
                for k in range(min(len(TRAILER) - 1, len(data) - pos), 0, -1):
                    if data.endswith(TRAILER[:k]):
                        safe = len(data) - k
                        break
            if safe > pos:
                yield data[pos:safe]
                pos = safe


console = Console()
run_lock = threading.Lock()
stats_lock = threading.Lock()
waiting = 0
counts = {}           # (endpoint, status) -> requests
sum_seconds = 0.0
gen_tokens_total = 0


class ThinkFilter:
    """Qwen3 may open with an empty <think></think> block; drop a leading one from the text."""

    def __init__(self):
        self.held = ""
        self.done = False

    def feed(self, s):
        if self.done:
            return s
        self.held += s
        t = self.held.lstrip()
        if t.startswith("<think>"):
            j = t.find("</think>")
            if j < 0:
                return ""
            self.done = True
            return t[j + len("</think>"):].lstrip()
        if "<think>".startswith(t):
            return ""                       # could still become <think>
        self.done = True
        return self.held

    def flush(self):
        out = "" if self.done else self.held
        self.done = True
        return out


def answer_stream(prompt_bytes):
    """Yields decoded text pieces, then a stats dict. Takes care of waiting, the lock and metrics."""
    global waiting, sum_seconds, gen_tokens_total
    with stats_lock:
        full = waiting >= CONFIG["max_waiting"]
        if not full:
            waiting += 1
    if full:
        raise RelayError(429, "the model is busy, try again", "rate_limit_exceeded")
    try:
        if not run_lock.acquire(timeout=CONFIG["gen_timeout"] + CONFIG["cmd_timeout"]):
            raise RelayError(429, "the model is busy, try again", "rate_limit_exceeded")
    finally:
        with stats_lock:
            waiting -= 1
    try:
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        think = ThinkFilter()
        for item in console.run(prompt_bytes):
            if isinstance(item, dict):
                tail = think.feed(dec.decode(b"", final=True)) + think.flush()
                if tail:
                    yield tail
                with stats_lock:
                    sum_seconds += item["seconds"]
                    gen_tokens_total += item["gen_tokens"]
                yield item
                return
            s = think.feed(dec.decode(item))
            if s:
                yield s
    finally:
        run_lock.release()


def answer(prompt_bytes):
    parts, stats = [], None
    for item in answer_stream(prompt_bytes):
        if isinstance(item, dict):
            stats = item
        else:
            parts.append(item)
    return "".join(parts).strip(), stats


# ---------------------------------------------------------------------------------------------
# Turning requests into one console line
# ---------------------------------------------------------------------------------------------
def one_line(s):
    s = "".join(ch if ch.isprintable() else " " for ch in s)
    return " ".join(s.split())


def check_line(s):
    if not s:
        raise RelayError(400, "the prompt is empty", "invalid_request_error")
    if s in COMMANDS:
        raise RelayError(400, "the prompt is a console command", "invalid_request_error")
    b = s.encode("utf-8")
    if len(b) > CONFIG["max_prompt_bytes"]:
        raise RelayError(400, "the prompt is over %d bytes on one line" % CONFIG["max_prompt_bytes"],
                         "invalid_request_error", "context_length_exceeded")
    if any(x < 0x20 or x == 0x7F for x in b):
        raise RelayError(400, "the prompt has control bytes", "invalid_request_error")
    return b


def text_of(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                out.append(part["text"])
            else:
                raise RelayError(400, "only text content is supported", "invalid_request_error", "unsupported_content")
        return " ".join(out)
    raise RelayError(400, "message content must be a string or a list of text parts", "invalid_request_error")


def chat_to_line(messages):
    """The console takes one user line per turn and applies Qwen3's chat template itself. So:
    the last message must be from the user; system and developer messages become a leading
    "Instructions:" sentence; earlier turns become "Earlier conversation: User: ... Assistant: ...".
    If the line is too long, the oldest turns are dropped first; the last user message is never cut."""
    if not isinstance(messages, list) or not messages:
        raise RelayError(400, "messages must be a non-empty list", "invalid_request_error")
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("role"), str):
            raise RelayError(400, "each message needs a role", "invalid_request_error")
    if messages[-1]["role"] != "user":
        raise RelayError(400, "the last message must have role user", "invalid_request_error")
    system = [one_line(text_of(m.get("content"))) for m in messages if m["role"] in ("system", "developer")]
    turns = []
    for m in messages[:-1]:
        if m["role"] in ("user", "assistant"):
            t = one_line(text_of(m.get("content")))
            if t:
                turns.append(("User: " if m["role"] == "user" else "Assistant: ") + t)
        elif m["role"] not in ("system", "developer"):
            raise RelayError(400, "role %s is not supported" % m["role"], "invalid_request_error")
    last = one_line(text_of(messages[-1].get("content")))
    if not last:
        raise RelayError(400, "the last user message is empty", "invalid_request_error")
    sys_text = " ".join(s for s in system if s)
    limit = CONFIG["max_prompt_bytes"]
    while True:
        pieces = []
        if sys_text:
            pieces.append("Instructions: " + sys_text)
        if turns:
            pieces.append("Earlier conversation: " + " ".join(turns) + " Now answer:")
        pieces.append(last)
        line = " ".join(pieces)
        if len(line.encode("utf-8")) <= limit or not turns:
            break
        turns.pop(0)
    if len(line.encode("utf-8")) > limit and sys_text:
        line = last                         # instructions do not fit beside the question
    return check_line(line)


def finish_reason(stats):
    return "length" if stats["gen_tokens"] >= CONFIG["boot_max_tokens"] else "stop"


def usage(stats):
    return {"prompt_tokens": stats["prompt_tokens"], "completion_tokens": stats["gen_tokens"],
            "total_tokens": stats["prompt_tokens"] + stats["gen_tokens"]}


# ---------------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "llm-relay"
    sys_version = ""

    def log_message(self, fmt, *args):
        pass  # one line per request comes from _done, without prompt text

    def _done(self, code, note=""):
        if self.path == "/metrics":
            return
        key = (self.path, code)
        with stats_lock:
            counts[key] = counts.get(key, 0) + 1
        log(("%s %s %s %d %s" % (self.client_address[0], self.command, self.path, code, note)).rstrip())

    def _json(self, code, body, note=""):
        data = (json.dumps(body, ensure_ascii=False) + "\n").encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        self._done(code, note)

    def _error(self, e):
        self._json(e.status, {"error": {"message": e.message, "type": e.etype, "param": None, "code": e.code}})

    def _authorized(self):
        key = CONFIG["api_key"]
        if not key:
            return True
        got = self.headers.get("Authorization", "")
        return got.startswith("Bearer ") and secrets.compare_digest(got[7:].strip(), key)

    def _body(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = -1
        if n <= 0 or n > MAX_BODY:
            raise RelayError(413 if n > MAX_BODY else 400, "the body must be 1 to %d bytes of JSON" % MAX_BODY,
                             "invalid_request_error")
        try:
            req = json.loads(self.rfile.read(n))
        except (ValueError, UnicodeDecodeError):
            raise RelayError(400, "the body is not JSON", "invalid_request_error")
        if not isinstance(req, dict):
            raise RelayError(400, "the body must be a JSON object", "invalid_request_error")
        return req

    def do_GET(self):
        if self.path == "/healthz":
            ok = console.conn is not None and (console.ready or run_lock.locked()) and not console.powered_off
            return self._json(200 if ok else 503, {"ok": ok, "connected": console.conn is not None,
                                                   "ready": console.ready, "busy": run_lock.locked(),
                                                   "powered_off": console.powered_off})
        if self.path == "/metrics":
            data = metrics().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if not self._authorized():
            return self._error(RelayError(401, "missing or wrong API key", "invalid_request_error", "invalid_api_key"))
        if self.path == "/v1/models":
            return self._json(200, {"object": "list", "data": [model_card()]})
        if self.path == "/v1/models/" + CONFIG["model_id"]:
            return self._json(200, model_card())
        self._error(RelayError(404, "not found", "invalid_request_error"))

    def do_POST(self):
        routes = {"/v1/chat/completions": self._chat, "/v1/completions": self._completions,
                  "/v1/generate": self._generate}
        route = routes.get(self.path)
        if route is None:
            return self._error(RelayError(404, "not found", "invalid_request_error"))
        if not self._authorized():
            return self._error(RelayError(401, "missing or wrong API key", "invalid_request_error", "invalid_api_key"))
        try:
            req = self._body()
            route(req)
        except RelayError as e:
            if getattr(self, "_streaming", False):
                self._sse({"error": {"message": e.message, "type": e.etype, "param": None, "code": e.code}})
                self._sse_end(e.status)
            else:
                self._error(e)

    # --- the three generation endpoints ---

    def _generate(self, req):
        p = req.get("prompt")
        if not isinstance(p, str):
            raise RelayError(400, "prompt must be a string", "invalid_request_error")
        pb = check_line(one_line(p))
        text, stats = answer(pb)
        self._json(200, dict(text=text, **stats), self._note(pb, stats))

    def _chat(self, req):
        if req.get("n", 1) != 1:
            raise RelayError(400, "only n=1 is supported", "invalid_request_error")
        pb = chat_to_line(req.get("messages"))
        model = req.get("model") or CONFIG["model_id"]
        cid = "chatcmpl-" + secrets.token_hex(12)
        created = int(time.time())
        if req.get("stream"):
            include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
            base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}
            stream = answer_stream(pb)
            first = next(stream)                # errors before any output become plain JSON errors
            self._sse_start()
            self._sse(dict(base, choices=[{"index": 0, "delta": {"role": "assistant", "content": ""},
                                           "finish_reason": None}]))
            stats = None
            for item in self._chain(first, stream):
                if isinstance(item, dict):
                    stats = item
                else:
                    self._sse(dict(base, choices=[{"index": 0, "delta": {"content": item}, "finish_reason": None}]))
            self._sse(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": finish_reason(stats)}]))
            if include_usage:
                self._sse(dict(base, choices=[], usage=usage(stats)))
            return self._sse_end(200, self._note(pb, stats))
        text, stats = answer(pb)
        self._json(200, {
            "id": cid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": finish_reason(stats)}],
            "usage": usage(stats),
        }, self._note(pb, stats))

    def _completions(self, req):
        if req.get("n", 1) != 1:
            raise RelayError(400, "only n=1 is supported", "invalid_request_error")
        p = req.get("prompt")
        if isinstance(p, list) and len(p) == 1 and isinstance(p[0], str):
            p = p[0]
        if not isinstance(p, str):
            raise RelayError(400, "prompt must be a string", "invalid_request_error")
        pb = check_line(one_line(p))
        model = req.get("model") or CONFIG["model_id"]
        cid = "cmpl-" + secrets.token_hex(12)
        created = int(time.time())
        if req.get("stream"):
            base = {"id": cid, "object": "text_completion", "created": created, "model": model}
            stream = answer_stream(pb)
            first = next(stream)
            self._sse_start()
            stats = None
            for item in self._chain(first, stream):
                if isinstance(item, dict):
                    stats = item
                else:
                    self._sse(dict(base, choices=[{"index": 0, "text": item, "logprobs": None, "finish_reason": None}]))
            self._sse(dict(base, choices=[{"index": 0, "text": "", "logprobs": None,
                                           "finish_reason": finish_reason(stats)}]))
            return self._sse_end(200, self._note(pb, stats))
        text, stats = answer(pb)
        self._json(200, {
            "id": cid, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason(stats)}],
            "usage": usage(stats),
        }, self._note(pb, stats))

    # --- helpers ---

    @staticmethod
    def _chain(first, rest):
        yield first
        yield from rest

    @staticmethod
    def _note(pb, stats):
        return "prompt_bytes=%d prompt_tokens=%d gen_tokens=%d seconds=%s" % (
            len(pb), stats["prompt_tokens"], stats["gen_tokens"], stats["seconds"])

    def _sse_start(self):
        self._streaming = True
        self._client_gone = False
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _sse(self, obj):
        if self._client_gone:
            return                          # keep reading the console so it stays in step
        try:
            self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode())
            self.wfile.flush()
        except OSError:
            self._client_gone = True

    def _sse_end(self, code, note=""):
        if not self._client_gone:
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except OSError:
                pass
        self.close_connection = True
        self._done(code, ("stream " + note).strip())


def model_card():
    return {"id": CONFIG["model_id"], "object": "model", "created": STARTED, "owned_by": "llm.asm"}


def metrics():
    with stats_lock:
        lines = [
            "# HELP llm_relay_console_connected The VM's console is connected to the relay.",
            "# TYPE llm_relay_console_connected gauge",
            "llm_relay_console_connected %d" % (console.conn is not None),
            "# HELP llm_relay_console_ready The console is at its prompt or busy with a request.",
            "# TYPE llm_relay_console_ready gauge",
            "llm_relay_console_ready %d" % (console.ready or run_lock.locked()),
            "# HELP llm_relay_console_powered_off The VM printed its power-off line on this connection.",
            "# TYPE llm_relay_console_powered_off gauge",
            "llm_relay_console_powered_off %d" % console.powered_off,
            "# TYPE llm_relay_console_connects_total counter",
            "llm_relay_console_connects_total %d" % console.connects,
            "# TYPE llm_relay_console_poweroffs_total counter",
            "llm_relay_console_poweroffs_total %d" % console.poweroffs,
            "# TYPE llm_relay_waiting gauge",
            "llm_relay_waiting %d" % waiting,
            "# TYPE llm_relay_requests_total counter",
        ]
        lines += ['llm_relay_requests_total{path="%s",code="%d"} %d' % (p, c, n) for (p, c), n in sorted(counts.items())]
        lines += [
            "# TYPE llm_relay_generate_seconds_sum counter",
            "llm_relay_generate_seconds_sum %.3f" % sum_seconds,
            "# TYPE llm_relay_generated_tokens_total counter",
            "llm_relay_generated_tokens_total %d" % gen_tokens_total,
        ]
    return "\n".join(lines) + "\n"


def main():
    threading.Thread(target=console.serve, daemon=True).start()
    httpd = ThreadingHTTPServer((CONFIG["http_addr"], CONFIG["http_port"]), Handler)
    httpd.daemon_threads = True
    log("http listening on %s:%d, model %s, api key %s" % (CONFIG["http_addr"], CONFIG["http_port"],
                                                           CONFIG["model_id"], "required" if CONFIG["api_key"] else "off"))
    httpd.serve_forever()


if __name__ == "__main__":
    main()
