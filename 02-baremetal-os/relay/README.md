# llm-relay

An OpenAI-compatible HTTP API for the [Phase 2 bare-metal OS](../README.md): the 24 KB x86-64 kernel that boots Qwen3-0.6B with no operating system underneath.

That kernel has no network driver. Its only way in or out is the serial console. llm-relay sits on the other end of that serial line: QEMU connects the VM's COM1 to the relay over TCP, and the relay types prompts into the console, reads the answers back and serves them as `/v1/chat/completions`. Any OpenAI client, SDK or chat UI can then talk to a model that runs with no OS.

One Python file, standard library only (Python 3.8+).

```
OpenAI client ── HTTP :8080 ──> llm-relay <── TCP :7000 ── QEMU (chardev client) ── COM1 of the llm.asm VM
```

## Run it

From the repository root:

1. Build the kernel and export the model (`make os`; see the [Phase 2 README](../README.md) and [Phase 1 setup](../../01-qwen3-inference/README.md)).
2. Start the relay:
   ```bash
   LLM_BOOT_MAX_TOKENS=256 python3 02-baremetal-os/relay/llm-relay.py
   ```
3. Boot the VM with its serial port connected to the relay:
   ```bash
   qemu-system-x86_64 -enable-kvm -cpu host -m 2G -smp 4 \
     -kernel build/asm-llm-os.elf -initrd models/qwen3-0.6b-q8_0.asmbin \
     -append 'max_tokens=256' -display none \
     -chardev socket,id=llm0,host=127.0.0.1,port=7000,reconnect-ms=5000 \
     -serial chardev:llm0
   ```
   `reconnect-ms` needs QEMU 9.2 or later; older builds use `reconnect=5`. With it, the VM and the relay can start in either order and survive each other's restarts.
4. Call it:
   ```bash
   curl -s http://127.0.0.1:8080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"qwen3-0.6b-llm.asm","messages":[{"role":"user","content":"Explain a CPU cache in two sentences."}]}'
   ```
   ```python
   from openai import OpenAI
   client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
   r = client.chat.completions.create(model="qwen3-0.6b-llm.asm",
                                      messages=[{"role": "user", "content": "Hello"}])
   print(r.choices[0].message.content)
   ```

### Proxmox VE

Boot the ISO from `make os-iso` (see [Option C](../README.md#option-c-deploying-on-proxmox-ve-0-gb-disk-vm)) in a VM with no disk and no NIC, and give QEMU the serial line with `args`:

```bash
qm set <vmid> --args '-chardev socket,id=llm0,host=<relay address>,port=7000,reconnect-ms=5000 -serial chardev:llm0'
```

Leave `serial0` unset so this becomes COM1. QEMU connects from the Proxmox host, so set `LLM_SERIAL_PEER` to the host's address on that network. Use `--cpu host` (the kernel needs AVX2, FMA and F16C) and `--balloon 0`.

## Configuration

All settings are environment variables.

| Variable | Default | What it sets |
|---|---|---|
| `LLM_SERIAL_ADDR` / `LLM_SERIAL_PORT` | `127.0.0.1` / `7000` | Where the relay listens for QEMU's serial connection |
| `LLM_SERIAL_PEER` | `127.0.0.1` | Addresses allowed to connect as the console, comma-separated |
| `LLM_HTTP_ADDR` / `LLM_HTTP_PORT` | `127.0.0.1` / `8080` | The HTTP API |
| `LLM_API_KEY` | unset | If set, `/v1/*` requires `Authorization: Bearer <key>` |
| `LLM_MODEL_ID` | `qwen3-0.6b-llm.asm` | The id in `/v1/models` |
| `LLM_BOOT_MAX_TOKENS` | `128` | The `max_tokens` the VM booted with; used to report `finish_reason: "length"` |
| `LLM_MAX_PROMPT_BYTES` | `1000` | The longest line sent to the console (the console holds 1023 bytes) |
| `LLM_MAX_WAITING` | `4` | Requests that may queue; more get 429 |
| `LLM_GEN_TIMEOUT` | `120` | Seconds to wait for an answer |

## API

| Endpoint | Notes |
|---|---|
| `GET /v1/models` | One model |
| `POST /v1/chat/completions` | `stream` true or false; `stream_options.include_usage` honoured |
| `POST /v1/completions` | Legacy text completion; `stream` true or false |
| `POST /v1/generate` | `{"prompt": "..."}` returns the text plus prefill and decode speed |
| `GET /healthz` | 200 when the console is connected and ready or busy, else 503 |
| `GET /metrics` | Prometheus text format |

Errors use OpenAI's shape: `{"error": {"message", "type", "param", "code"}}`.

## What the console allows, and how requests are mapped

The model runs inside a kernel whose only interface is a line-based console, so some of the OpenAI API cannot be honoured:

- **One request at a time.** The kernel has one KV cache. Others queue (up to `LLM_MAX_WAITING`), then get 429.
- **Requests are independent.** The relay sends `/clear` before each one.
- **Messages become one line.** The console applies Qwen3's chat template to a single user turn. System and developer messages become a leading `Instructions:` sentence, and earlier turns become `Earlier conversation: User: ... Assistant: ...` ahead of the last user message. Newlines become spaces. If the line is too long, the oldest turns go first; the last user message is never cut, and a request whose last message alone is too long gets a 400.
- **Sampling is fixed at boot.** `max_tokens`, `temperature`, `top_p` and `seed` come from the kernel command line (`max_tokens=`, `temp=`, `topp=`, `seed=`). Per-request values are accepted and ignored. Stop sequences, `n` above 1, tools, JSON mode, images and embeddings are not supported.
- **Console commands are refused.** A prompt that is exactly `exit`, `/halt`, `/quit`, `/clear` or another console command gets a 400 (some of them power the VM off), and control bytes are stripped.
- `usage.prompt_tokens` counts the chat template's tokens too.

## Security

- Both ports bind to `127.0.0.1` by default. The HTTP API has no TLS. Put it behind a reverse proxy, or set `LLM_API_KEY`, before exposing it.
- Only addresses in `LLM_SERIAL_PEER` can attach as the console. Whoever holds the console can type anything into the VM, including the commands that power it off.
- Prompt text is never logged. Each request logs one line with the client address, status, prompt size, token counts and duration.

## Tests

```bash
python3 02-baremetal-os/relay/test_relay.py
```

The tests start the relay against a simulated console that prints what the OS prints over COM1, so they need no VM and no model. They cover the OpenAI endpoints with and without streaming, how messages are folded into one line, refused console commands and control bytes, the queue limit, a lost console, a relay restart while the VM sits at its prompt, and the API key.

## Running under systemd

```ini
[Unit]
Description=llm-relay
After=network-online.target
Wants=network-online.target

[Service]
User=llmrelay
Environment=LLM_BOOT_MAX_TOKENS=256
ExecStart=/usr/bin/python3 -I /usr/local/lib/llm-relay/llm-relay.py
Restart=always
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
RestrictAddressFamilies=AF_INET AF_UNIX
MemoryMax=256M

[Install]
WantedBy=multi-user.target
```
