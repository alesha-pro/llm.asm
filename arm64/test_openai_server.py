#!/usr/bin/env python3
"""Protocol tests for the OpenAI-compatible ARM64 server; no model is required."""
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import openai_server


class FakeWorker:
    def __init__(self, answer):
        self.answer = answer
        self.proc = SimpleNamespace(poll=lambda: None)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.answer


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.worker = FakeWorker('<tool_call>{"name":"shell","arguments":{"command":"printf OK"}}</tool_call>')
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), openai_server.make_handler(self.worker))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:%d" % self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def post(self, body, headers=None):
        request = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers=headers or {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.headers, response.read().decode()

    def tool_request(self, stream=False, extra_messages=None):
        tool = {"type": "function", "function": {"name": "shell", "parameters": {
            "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
        return {"messages": (extra_messages or []) + [{"role": "user", "content": "Print OK"}],
                "tools": [tool], "stream": stream}

    def test_models_endpoint(self):
        with urllib.request.urlopen(self.base + "/v1/models", timeout=5) as response:
            body = json.loads(response.read())
        self.assertEqual(body["data"][0]["id"], openai_server.MODEL_ID)

    def test_tool_call_response(self):
        status, _, raw = self.post(self.tool_request())
        body = json.loads(raw)
        self.assertEqual(status, 200)
        self.assertEqual(body["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(body["choices"][0]["message"]["tool_calls"][0]["function"],
                         {"name": "shell", "arguments": '{"command":"printf OK"}'})

    def test_xml_shaped_qwen_tool_call(self):
        text = '<tool_call><name>shell</name><arguments><command>printf OK</command></arguments></tool_call>'
        content, calls = openai_server.parse_tool_calls(text, {"shell"})
        self.assertEqual(content, "")
        self.assertEqual(calls[0]["function"], {"name": "shell", "arguments": '{"command":"printf OK"}'})

    def test_streamed_tool_call(self):
        _, headers, raw = self.post(self.tool_request(stream=True))
        self.assertEqual(headers.get_content_type(), "text/event-stream")
        events = [line[6:] for line in raw.splitlines() if line.startswith("data: ")]
        chunks = [json.loads(event) for event in events if event != "[DONE]"]
        tool_delta = next(chunk["choices"][0]["delta"]["tool_calls"][0]
                          for chunk in chunks if "tool_calls" in chunk["choices"][0]["delta"])
        self.assertEqual(tool_delta["function"]["name"], "shell")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(events[-1], "[DONE]")

    def test_tool_results_are_folded_into_prompt(self):
        messages = [
            {"role": "user", "content": "Read a file"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
             "function": {"name": "read", "arguments": '{"path":"README.md"}'}}]},
            {"role": "tool", "name": "read", "tool_call_id": "call_1", "content": "file contents"},
        ]
        self.post(self.tool_request(extra_messages=messages))
        self.assertIn("Assistant called read", self.worker.prompts[-1])
        self.assertIn("Tool result (read): file contents", self.worker.prompts[-1])

    def test_opencode_workspace_advisory_is_not_forwarded(self):
        advisory = ("The directory /tmp/project is a git repository and is approved for external access. "
                    "If you are outside the current working directory, use tools.opencode.session_move.")
        prompt, _ = openai_server.build_prompt([{"role": "system", "content": advisory},
                                                 {"role": "user", "content": "hi"}], [])
        self.assertEqual(prompt, "hi")

    def test_content_type_is_required(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post(self.tool_request(), {"Content-Type": "text/plain"})
        self.assertEqual(caught.exception.code, 415)


if __name__ == "__main__":
    unittest.main(verbosity=2)
