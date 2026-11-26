#!/usr/bin/env python3
"""Tests for llm-relay, against a simulated llm.asm console (no VM, no model needed).

Run:  python3 test_relay.py      (Python 3.8+, standard library only)

The simulated console copies what 02-baremetal-os prints over COM1: the "\\nasm-llm> " prompt, the
echo of each typed byte, "\\n" on Enter, the answer, the two "[asm-llm-os]" stat lines, then the
prompt again; "/clear" resets, and "exit", "/halt", "/quit" or a Ctrl-D byte power it off.
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RELAY = os.path.join(HERE, "llm-relay.py")
PROMPT = b"\nasm-llm> "


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FakeConsole(threading.Thread):
    """Connects to the relay like QEMU's chardev client and plays the llm.asm REPL."""

    def __init__(self, port, banner=True, answer_delay=0.0, think=True, response=None, responses=None):
        super().__init__(daemon=True)
        self.port, self.banner, self.answer_delay, self.think = port, banner, answer_delay, think
        self.response = response
        self.responses = list(responses or [])
        self.lines = []             # every line the relay typed
        self.powered_off = False
        self.sock = None

    def run(self):
        for _ in range(100):
            try:
                self.sock = socket.create_connection(("127.0.0.1", self.port))
                break
            except OSError:
                time.sleep(0.05)
        s = self.sock
        if self.banner:
            s.sendall(b"GRUB\r\n  asm-llm OS v2.0\nInteractive Bare-Metal Chat (/clear, /help)\n" + PROMPT)
        line = bytearray()
        try:
            while True:
                d = s.recv(1)
                if not d:
                    return
                c = d[0]
                if c == 13:
                    s.sendall(b"\n")
                    t = line.decode("utf-8", "replace").strip()
                    self.lines.append(t)
                    line.clear()
                    if t == "":
                        s.sendall(PROMPT)
                    elif t == "/clear":
                        s.sendall(b"[chat] history cleared (ctx=0/2048)\n" + PROMPT)
                    elif t in ("exit", "/halt", "/quit"):
                        self._poweroff()
                        return
                    else:
                        time.sleep(self.answer_delay)
                        if self.responses:
                            body = self.responses.pop(0)
                        else:
                            body = self.response if self.response is not None else "Reply to: " + t
                        body = ("<think>\n\n</think>\n\n" if self.think else "") + body
                        raw = body.encode()
                        for i in range(0, len(raw), 5):     # stream in small pieces, like tokens
                            s.sendall(raw[i:i + 5])
                            time.sleep(0.002)
                        s.sendall(b"\n[asm-llm-os] prefill: 12 tok in 100.00 ms (120.00 tok/s)\n"
                                  b"[asm-llm-os] decode : 9 tok in 100.00 ms (80.00 tok/s, cores=4) ctx=21/2048\n" + PROMPT)
                elif c == 4:
                    self._poweroff()
                    return
                elif c >= 0x20:
                    line.append(c)
                    s.sendall(d)
        except OSError:
            return

    def _poweroff(self):
        self.powered_off = True
        self.sock.sendall(b"Powering off bare-metal VM...\n")

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)    # close() alone does not wake a recv() in another thread
            self.sock.close()
        except (OSError, AttributeError):
            pass


class RelayTest(unittest.TestCase):
    def setUp(self):
        self.serial_port, self.http_port = free_port(), free_port()
        env = dict(os.environ, LLM_SERIAL_PORT=str(self.serial_port), LLM_HTTP_PORT=str(self.http_port),
                   LLM_GEN_TIMEOUT="10", LLM_CMD_TIMEOUT="5", LLM_BOOT_MAX_TOKENS="256", LLM_MAX_WAITING="2")
        self.proc = subprocess.Popen([sys.executable, "-I", RELAY], env=env,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.base = "http://127.0.0.1:%d" % self.http_port
        self.console = None
        self._wait(lambda: self._status("/healthz") in (200, 503))

    def tearDown(self):
        if self.console:
            self.console.close()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(5)

    # helpers
    def _wait(self, cond, timeout=10):
        end = time.time() + timeout
        while time.time() < end:
            try:
                if cond():
                    return
            except OSError:
                pass
            time.sleep(0.05)
        self.fail("condition not met in %ss" % timeout)

    def _status(self, path):
        try:
            return urllib.request.urlopen(self.base + path, timeout=5).status
        except urllib.error.HTTPError as e:
            return e.code

    def _connect(self, **kw):
        self.console = FakeConsole(self.serial_port, **kw)
        self.console.start()
        self._wait(lambda: self._status("/healthz") == 200)

    def _post(self, path, body, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers=headers or {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    def _chat(self, messages, **extra):
        code, body = self._post("/v1/chat/completions", dict(messages=messages, **extra))
        return code, json.loads(body)

    # tests
    def test_not_ready_without_console(self):
        self.assertEqual(self._status("/healthz"), 503)
        code, body = self._chat([{"role": "user", "content": "hi"}])
        self.assertEqual(code, 503)
        self.assertIn("error", body)

    def test_models(self):
        with urllib.request.urlopen(self.base + "/v1/models", timeout=5) as r:
            data = json.loads(r.read())
        self.assertEqual(data["object"], "list")
        self.assertEqual(data["data"][0]["id"], "qwen3-0.6b-llm.asm")

    def test_version_and_help(self):
        version = subprocess.check_output([sys.executable, RELAY, "--version"], text=True).strip()
        self.assertEqual(version, "llm-relay 0.2.0")
        help_text = subprocess.check_output([sys.executable, RELAY, "--help"], text=True)
        self.assertIn("OpenAI-compatible serial relay", help_text)

    def test_chat_completion(self):
        self._connect()
        code, body = self._chat([{"role": "user", "content": "What is 2+2?"}])
        self.assertEqual(code, 200)
        self.assertEqual(body["object"], "chat.completion")
        msg = body["choices"][0]["message"]
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], "Reply to: What is 2+2?")        # <think> block removed
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(body["usage"], {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21})
        self.assertEqual(self.console.lines[-2:], ["/clear", "What is 2+2?"])   # cleared before every request

    def test_messages_folded_into_one_line(self):
        self._connect()
        code, body = self._chat([
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": [{"type": "text", "text": "Name a"}, {"type": "text", "text": "colour.\nThanks"}]},
        ])
        self.assertEqual(code, 200)
        self.assertEqual(self.console.lines[-1],
                         "Instructions: Be brief. Earlier conversation: User: Hi Assistant: Hello! Now answer: Name a colour. Thanks")

    def test_long_history_drops_oldest_keeps_last(self):
        self._connect()
        history = [{"role": "user", "content": "old %d %s" % (i, "z" * 80)} for i in range(30)]
        code, _ = self._chat([{"role": "system", "content": "Be brief."}] + history + [{"role": "user", "content": "LAST"}])
        self.assertEqual(code, 200)
        sent = self.console.lines[-1]
        self.assertLessEqual(len(sent.encode()), 1000)
        self.assertTrue(sent.startswith("Instructions: Be brief."))
        self.assertTrue(sent.endswith("Now answer: LAST"))
        self.assertNotIn("old 0 ", sent)
        self.assertIn("old 29 ", sent)

    def test_stream(self):
        self._connect()
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps({
            "messages": [{"role": "user", "content": "Stream café"}], "stream": True,
            "stream_options": {"include_usage": True}}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            self.assertEqual(r.headers["Content-Type"], "text/event-stream")
            events = [l[6:] for l in r.read().decode().splitlines() if l.startswith("data: ")]
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(e) for e in events[:-1]]
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
        self.assertEqual(text, "Reply to: Stream café")
        self.assertGreater(len(chunks), 3)                                  # arrived in pieces
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 9)

    def test_tool_call_completion(self):
        self._connect(think=False, response='<tool_call>{"name":"read","arguments":{"filePath":"README.md"}}</tool_call>')
        tools = [{"type": "function", "function": {
            "name": "read", "description": "Read a file",
            "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}},
                           "required": ["filePath"]}}}]
        code, body = self._chat([{"role": "user", "content": "Read README.md"}], tools=tools)
        self.assertEqual(code, 200)
        message = body["choices"][0]["message"]
        self.assertIsNone(message["content"])
        self.assertEqual(message["tool_calls"][0]["function"],
                         {"name": "read", "arguments": '{"filePath":"README.md"}'})
        self.assertEqual(body["choices"][0]["finish_reason"], "tool_calls")
        self.assertIn("Available tools: read(filePath:string!)", self.console.lines[-1])

    def test_streamed_tool_call(self):
        self._connect(think=False, response='I will read that. <tool_call>{"name":"read","arguments":{"filePath":"README.md"}}</tool_call>')
        tools = [{"type": "function", "function": {
            "name": "read", "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}},
                                               "required": ["filePath"]}}}]
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps({
            "messages": [{"role": "user", "content": "Read README.md"}], "tools": tools, "stream": True,
        }).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            events = [line[6:] for line in r.read().decode().splitlines() if line.startswith("data: ")]
        chunks = [json.loads(event) for event in events[:-1]]
        deltas = [choice["delta"] for chunk in chunks for choice in chunk.get("choices", [])]
        self.assertIn("I will read that.", "".join(delta.get("content", "") for delta in deltas))
        tool_delta = next(delta["tool_calls"][0] for delta in deltas if "tool_calls" in delta)
        self.assertEqual(tool_delta["function"]["name"], "read")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_unadvertised_tool_stays_text(self):
        self._connect(think=False, response='<tool_call>{"name":"delete","arguments":{}}</tool_call>')
        tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
        code, body = self._chat([{"role": "user", "content": "Read"}], tools=tools)
        self.assertEqual(code, 200)
        message = body["choices"][0]["message"]
        self.assertNotIn("tool_calls", message)
        self.assertIn('<tool_call>{"name":"delete"', message["content"])

    def test_forced_tool_that_does_not_fit_is_named_in_error(self):
        self._connect(think=False)
        name = "read_file"
        prop = "p" * 1000
        tools = [{"type": "function", "function": {"name": name,
                 "parameters": {"type": "object", "properties": {prop: {"type": "string"}}}}}]
        code, body = self._chat([{"role": "user", "content": "Read the file."}], tools=tools,
                                tool_choice={"type": "function", "function": {"name": name}})
        self.assertEqual(code, 400)
        self.assertIn(name, body["error"]["message"])
        self.assertIn("console input limit", body["error"]["message"])

    def test_tool_result_is_folded_into_next_turn(self):
        self._connect(think=False)
        code, _ = self._chat([
            {"role": "user", "content": "Read file"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
             "function": {"name": "read", "arguments": '{"filePath":"README.md"}'}}]},
            {"role": "tool", "name": "read", "tool_call_id": "call_1", "content": "README contents"},
            {"role": "user", "content": "Summarize it"}],
            tools=[{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}])
        self.assertEqual(code, 200)
        self.assertIn("Assistant called read", self.console.lines[-1])
        self.assertIn("Tool result (read): README contents", self.console.lines[-1])

    def test_completions_and_generate(self):
        self._connect(think=False)
        code, body = self._post("/v1/completions", {"prompt": "Once upon a time"})
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["choices"][0]["text"], "Reply to: Once upon a time")
        code, body = self._post("/v1/generate", {"prompt": "legacy"})
        self.assertEqual(code, 200)
        data = json.loads(body)
        self.assertEqual(data["text"], "Reply to: legacy")
        self.assertEqual(data["decode_tok_s"], 80.0)

    def test_console_commands_refused(self):
        self._connect()
        for p in ("exit", "/halt", "/quit", "/clear", "  /halt  "):
            code, body = self._chat([{"role": "user", "content": p}])
            self.assertEqual(code, 400, p)
            self.assertEqual(body["error"]["type"], "invalid_request_error")
        self.assertFalse(self.console.powered_off)

    def test_control_bytes_stripped(self):
        self._connect()
        code, _ = self._chat([{"role": "user", "content": "a\x04b\x7fc\x08d\re\nf"}])
        self.assertEqual(code, 200)
        self.assertEqual(self.console.lines[-1], "a b c d e f")
        self.assertFalse(self.console.powered_off)

    def test_bad_requests(self):
        self._connect()
        cases = [
            ({"messages": [{"role": "assistant", "content": "x"}]}, 400),
            ({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}, 400),
            ({"messages": [{"role": "user", "content": "x"}], "n": 2}, 400),
            ({"messages": [{"role": "user", "content": "x" * 1100}]}, 400),
            ({"messages": []}, 400),
        ]
        for body, want in cases:
            code, _ = self._post("/v1/chat/completions", body)
            self.assertEqual(code, want, body)
        self.assertEqual(self._status("/v1/nothing"), 404)

    def test_post_requires_json_content_type(self):
        req = urllib.request.Request(self.base + "/v1/chat/completions",
                                     data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
                                     headers={"Content-Type": "text/plain"})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(caught.exception.code, 415)

    def test_metrics_path_labels_are_bounded(self):
        for i in range(20):
            self._status("/not-found-%d" % i)
        with urllib.request.urlopen(self.base + "/metrics", timeout=5) as r:
            data = r.read().decode()
        self.assertIn('path="/other"', data)
        self.assertLessEqual(data.count("llm_relay_requests_total{path="), 2)

    def test_sigterm_shuts_down_cleanly(self):
        self.proc.terminate()
        self.assertEqual(self.proc.wait(5), 0)

    def test_one_at_a_time_and_queue_limit(self):
        self._connect(think=False, answer_delay=0.3)
        results = {}

        def call(i):
            results[i] = self._chat([{"role": "user", "content": "question %d" % i}])

        threads = [threading.Thread(target=call, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        ok = [i for i, (code, _) in results.items() if code == 200]
        busy = [i for i, (code, _) in results.items() if code == 429]
        self.assertEqual(len(ok) + len(busy), 5)
        self.assertGreaterEqual(len(busy), 1)
        for i in ok:                                                        # no crossed answers
            self.assertEqual(results[i][1]["choices"][0]["message"]["content"], "Reply to: question %d" % i)

    def test_relay_reconnect_pokes_vm_at_prompt(self):
        self._connect(banner=False)        # prints nothing on connect; the relay must poke it
        self.assertEqual(self.console.lines[0], "")

    def test_console_loss(self):
        self._connect()
        self.console.close()
        self._wait(lambda: self._status("/healthz") == 503)
        code, _ = self._chat([{"role": "user", "content": "hi"}])
        self.assertEqual(code, 503)

    def test_api_key(self):
        self.tearDown()
        self.serial_port, self.http_port = free_port(), free_port()
        env = dict(os.environ, LLM_SERIAL_PORT=str(self.serial_port), LLM_HTTP_PORT=str(self.http_port),
                   LLM_API_KEY="test-key")
        self.proc = subprocess.Popen([sys.executable, "-I", RELAY], env=env,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.base = "http://127.0.0.1:%d" % self.http_port
        self._wait(lambda: self._status("/healthz") in (200, 503))
        self.assertEqual(self._status("/v1/models"), 401)
        req = urllib.request.Request(self.base + "/v1/models", headers={"Authorization": "Bearer test-key"})
        self.assertEqual(urllib.request.urlopen(req, timeout=5).status, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
