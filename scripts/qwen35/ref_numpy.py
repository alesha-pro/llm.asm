"""Pure-numpy float32 single-token reference of the Qwen3.5-2B text decoder (needs numpy, torch, safetensors).
Prefill is a loop of step() over the prompt tokens. Checks the result against the Hugging Face reference values in
golden_2b.json (prompt ids, first generated ids, top-5 logits of the first position).
usage: python3 scripts/qwen35/ref_numpy.py <model.safetensors> scripts/qwen35/golden_2b.json"""
import sys, json, time
import numpy as np, torch
from safetensors import safe_open

ST, GOLD = sys.argv[1], sys.argv[2]
f = safe_open(ST, "pt"); P = "model.language_model."
def W(k): return f.get_tensor(P + k).to(torch.float32).numpy()

H, L, NH, NKV, HD, FF, ROT, THETA, EPS = 2048, 24, 8, 2, 256, 6144, 64, 1e7, 1e-6
LH, LD, CK = 16, 128, 4                     # linear heads (K==V), head dim, conv kernel
LAYER = ["full" if (i + 1) % 4 == 0 else "lin" for i in range(L)]
INV_FREQ = (1.0 / THETA ** (np.arange(0, ROT, 2, dtype=np.float32) / ROT)).astype(np.float32)   # [32]

def rms1p(x, w, eps=EPS):                   # x * rsqrt(mean(x^2)+eps) * (1 + w)   (Qwen3_5RMSNorm)
    x = x.astype(np.float32); return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps) * (1.0 + w)
def silu(x): return x / (1.0 + np.exp(-x))
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def softplus(x): return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))
def l2n(x, eps=1e-6): return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)

t0 = time.time()
E = W("embed_tokens.weight"); FN = W("norm.weight")
Ls = []
for i in range(L):
    d = {"in": W(f"layers.{i}.input_layernorm.weight"), "post": W(f"layers.{i}.post_attention_layernorm.weight"),
         "g": W(f"layers.{i}.mlp.gate_proj.weight"), "u": W(f"layers.{i}.mlp.up_proj.weight"), "dn": W(f"layers.{i}.mlp.down_proj.weight")}
    if LAYER[i] == "full":
        a = f"layers.{i}.self_attn."
        d.update(q=W(a + "q_proj.weight"), k=W(a + "k_proj.weight"), v=W(a + "v_proj.weight"), o=W(a + "o_proj.weight"),
                 qn=W(a + "q_norm.weight"), kn=W(a + "k_norm.weight"), K=[], V=[])
    else:
        a = f"layers.{i}.linear_attn."
        d.update(qkv=W(a + "in_proj_qkv.weight"), z=W(a + "in_proj_z.weight"), a=W(a + "in_proj_a.weight"), b=W(a + "in_proj_b.weight"),
                 conv=W(a + "conv1d.weight")[:, 0, :], A=W(a + "A_log"), dtb=W(a + "dt_bias"), nw=W(a + "norm.weight"), out=W(a + "out_proj.weight"),
                 S=np.zeros((LH, LD, LD), np.float32), cs=np.zeros((CK - 1, 2 * LH * LD + LH * LD), np.float32))
    Ls.append(d)
print("loaded %.1fs" % (time.time() - t0), file=sys.stderr)

def full_attn(d, n, pos):
    qg = (d["q"] @ n).reshape(NH, 2 * HD)
    q, gate = qg[:, :HD], qg[:, HD:]                       # per-head interleaved [q(256) | gate(256)]
    k = (d["k"] @ n).reshape(NKV, HD); v = (d["v"] @ n).reshape(NKV, HD)
    q = rms1p(q, d["qn"]); k = rms1p(k, d["kn"])           # per-head, BEFORE rope
    ang = pos * INV_FREQ; c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)   # [32]
    def rope(x):                                           # rotate_half on first 64 dims, pairs (i, i+32)
        x = x.copy(); x1, x2 = x[:, :32].copy(), x[:, 32:ROT].copy()
        x[:, :32] = x1 * c - x2 * s; x[:, 32:ROT] = x2 * c + x1 * s; return x
    q, k = rope(q), rope(k)
    d["K"].append(k); d["V"].append(v)
    Kc, Vc = np.stack(d["K"], 1), np.stack(d["V"], 1)      # [NKV, T, HD]
    o = np.empty((NH, HD), np.float32)
    for h in range(NH):
        sc = (Kc[h // (NH // NKV)] @ q[h]) / np.float32(16.0)        # scale = 256**-0.5
        p = np.exp(sc - sc.max()); p /= p.sum()
        o[h] = p @ Vc[h // (NH // NKV)]
    o = o.reshape(-1) * sigmoid(gate.reshape(-1))          # output gate BEFORE o_proj
    return d["o"] @ o

def delta_net(d, n):
    mixed = d["qkv"] @ n; z = (d["z"] @ n).reshape(LH, LD); b = d["b"] @ n; a = d["a"] @ n
    win = np.concatenate([d["cs"], mixed[None]], 0)         # [4, C], oldest first
    y = silu((win * d["conv"].T).sum(0)); d["cs"] = win[1:]  # conv[c, j] * win[j, c]; no bias
    q, k, v = y[:LH * LD].reshape(LH, LD), y[LH * LD:2 * LH * LD].reshape(LH, LD), y[2 * LH * LD:].reshape(LH, LD)
    q = l2n(q) / np.float32(np.sqrt(LD)); k = l2n(k)
    beta = sigmoid(b); g = -np.exp(d["A"]) * softplus(a + d["dtb"])
    o = np.empty((LH, LD), np.float32)
    for h in range(LH):                                     # S[h]: [key 128, value 128]
        S = d["S"][h] * np.exp(g[h])
        kv = S.T @ k[h]; delta = (v[h] - kv) * beta[h]
        S = S + np.outer(k[h], delta); d["S"][h] = S
        o[h] = S.T @ q[h]
    on = o / np.sqrt(np.mean(o * o, -1, keepdims=True) + EPS) * d["nw"] * silu(z)   # (norm * w) * silu(z)
    return d["out"] @ on.reshape(-1)

def step(tok, pos, want_logits=True):
    x = E[tok].copy()
    for i, d in enumerate(Ls):
        n = rms1p(x, d["in"])
        x = x + (full_attn(d, n, pos) if LAYER[i] == "full" else delta_net(d, n))
        n = rms1p(x, d["post"])
        x = x + d["dn"] @ (silu(d["g"] @ n) * (d["u"] @ n))
    return (E @ rms1p(x, FN)) if want_logits else None

g = json.load(open(GOLD)); prompt = g["prompt_token_ids"]; ref = g["generated_token_ids"]
t0 = time.time(); lg = None
for pos, t in enumerate(prompt): lg = step(t, pos, pos == len(prompt) - 1)
top = np.argsort(-lg)[:5]
print("top5 numpy:", [(int(i), float(lg[i])) for i in top]); print("top5 HF   :", [(r["id"], r["logit"]) for r in g["first_position_top5"]])
gen = [int(lg.argmax())]
for i in range(15): gen.append(int(step(gen[-1], len(prompt) + i).argmax()))
print("numpy gen:", gen); print("HF    gen:", ref); print("MATCH:", gen == ref, "%.1fs" % (time.time() - t0))
print("max |logit diff| top5:", max(abs(float(lg[r['id']]) - r['logit']) for r in g["first_position_top5"]))
