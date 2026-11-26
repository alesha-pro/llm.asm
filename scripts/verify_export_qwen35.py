#!/usr/bin/env python3
"""Verify a Qwen3.5-2B .asmbin v3 against the Hugging Face weights (python only, no kernel needed).

Re-reads the file only through the header + descriptor table (layout: scripts/ASMBIN_V3.md), dequantizes every
Q8_0 tensor, applies the same de-interleave / (1+w) / transpose to the HF tensor and compares.
  Q8 tensors : max abs err, max err in units of the block step (amax/127; must be <= ~0.56 = 0.5 rounding + fp16 scale
               rounding), relative Frobenius error.
  fp32 tensors: must be bit-exact.
Usage: verify_export_qwen35.py --model-dir HF_DIR --asmbin FILE [--tokenizer tokenizer.json]
"""
import argparse
import json, os, struct, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_weights import load_safetensors_header, read_tensor_f32, FIB_MULT, HT_BITS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True, help="HF model directory (config.json, safetensors, tokenizer.json)")
ap.add_argument("--asmbin", required=True, help="the .asmbin v3 file to check")
ap.add_argument("--tokenizer", help="tokenizer.json (default: <model-dir>/tokenizer.json)")
args = ap.parse_args()
asm_path, hf_dir = args.asmbin, args.model_dir
tok_json = args.tokenizer or os.path.join(hf_dir, "tokenizer.json")
fsize = os.path.getsize(asm_path)
mm = np.memmap(asm_path, dtype=np.uint8, mode="r")
hdr = bytes(mm[:256])
u32 = lambda o: struct.unpack_from("<I", hdr, o)[0]
u64 = lambda o: struct.unpack_from("<Q", hdr, o)[0]
H = dict(version=u32(4), dim=u32(8), hidden=u32(12), n_layers=u32(16), n_heads=u32(20), n_kv=u32(24), head_dim=u32(28),
         vocab=u32(32), max_seq=u32(36), eos1=u32(48), eos2=u32(52), ht_mask=u32(56), flags=u32(60),
         off_tok=u64(64), off_embed=u64(72), off_layers=u64(80), layer_stride=u64(88), off_final=u64(96),
         file_size=u64(104), off_table=u64(128), type_mask=u64(136), n_lk=u32(144), n_lv=u32(148), lk=u32(152),
         lv=u32(156), conv_k=u32(160), rot=u32(164), flags2=u32(168))
H["eps"], H["theta"] = struct.unpack_from("<ff", hdr, 40)
TOK = dict(zip(("endoftext", "im_start", "im_end", "think_start", "think_end", "user", "assistant", "nl", "nl2",
                "argmax_limit"), struct.unpack_from("<10I", hdr, 176)))
print("header:", {k: (hex(v) if k in ("flags", "flags2", "type_mask") else v) for k, v in H.items()})
print("tok ids:", TOK)
fails = []
def check(cond, msg):
    if not cond:
        fails.append(msg); print("FAIL:", msg)

check(hdr[:4] == b"ASML" and H["version"] == 3, "magic/version")
check(H["file_size"] == fsize, f"header file_size {H['file_size']} != actual {fsize}")
check(1.9e9 < fsize < 2.1e9, f"file size {fsize} not ~2.0 GB")
check(not any(hdr[216:256]), "reserved header bytes 216..255 nonzero")

# ---- HF side
cfgj = json.load(open(os.path.join(hf_dir, "config.json")))["text_config"]
st = os.path.join(hf_dir, "model.safetensors-00001-of-00001.safetensors")
if not os.path.exists(st):
    st = os.path.join(hf_dir, "model.safetensors")
shdr, doff = load_safetensors_header(st)
sf = open(st, "rb")
def hf(name, r0=0, r1=None):
    return read_tensor_f32(sf, shdr, doff, "model.language_model." + name, r0, r1)

dim, ff, L = H["dim"], H["hidden"], H["n_layers"]
nh, nkv, hd = H["n_heads"], H["n_kv"], H["head_dim"]
check((dim, ff, L, nh, nkv, hd) == (cfgj["hidden_size"], cfgj["intermediate_size"], cfgj["num_hidden_layers"],
      cfgj["num_attention_heads"], cfgj["num_key_value_heads"], cfgj["head_dim"]), "header dims vs config")
types = cfgj["layer_types"]
check(H["type_mask"] == sum(1 << l for l, t in enumerate(types) if t == "linear_attention"), "layer_type_mask")
check(H["rot"] == 64 and H["flags2"] == 3 and H["conv_k"] == 4 and (H["n_lk"], H["n_lv"], H["lk"], H["lv"]) == (16, 16, 128, 128),
      "linear/rotary header fields")
check(H["flags"] & 4 and H["flags"] & 1, "flags fp16 scales + tied")
SB = 2  # fp16 scales
rowb = lambda K: K + (K // 32) * SB

# ---- descriptor table
tab = bytes(mm[H["off_table"]:H["off_table"] + 136 * L])
descs = [struct.unpack_from("<II16Q", tab, 136 * l) for l in range(L)]
SLOT_NAMES = {0: {0: "in_norm", 1: "post_norm", 2: "W_proj(q|k|v|gate)", 3: "W_out(o_proj)", 4: "W_gateup", 5: "W_down", 6: "q_norm", 7: "k_norm"},
              1: {0: "in_norm", 1: "post_norm", 2: "W_proj(qkv|z)", 3: "W_out(out_proj)", 4: "W_gateup", 5: "W_down", 6: "conv_w[4][6144]",
                  7: "negExpA", 8: "dt_bias", 9: "gnorm_w", 10: "W_b f32", 11: "W_a f32"}}
for l in (0, 3, 23):
    d = descs[l]
    print(f"\n-- layer {l} descriptor: type={d[0]} reserved={d[1]}")
    for i in range(16):
        nm = SLOT_NAMES[d[0]].get(i, "(unused)")
        print(f"   slot {i:2d} {nm:20s} off={d[2 + i]:>12,}  mod64={d[2 + i] % 64}")

# ---- region bookkeeping (alignment / in-file / no overlap)
regions = []
def reg(name, off, n):
    check(off % 64 == 0, f"{name}: off {off} not 64-aligned")
    check(0 < off and off + n <= fsize, f"{name}: [{off},{off + n}) outside file")
    regions.append((off, off + n, name))

# ---- stats
stats = {}
def rec(cls, **kw):
    s = stats.setdefault(cls, dict(n=0, max_abs=0.0, max_steps=0.0, rel=0.0, exact=True))
    s["n"] += 1
    for k in ("max_abs", "max_steps", "rel"):
        if k in kw:
            s[k] = max(s[k], kw[k])
    if "exact" in kw:
        s["exact"] &= kw["exact"]

def q8_compare(cls, off, mats_fn, M, K, name):
    """mats_fn(r0, r1) -> expected f32 [r1-r0, K] for rows of the stored matrix (after layout transform)."""
    reg(name, off, M * rowb(K))
    nb = K // 32
    mx_abs = mx_steps = 0.0; num = den = 0.0; tiny = [0]
    for r0 in range(0, M, 4096):
        r1 = min(M, r0 + 4096)
        raw = np.asarray(mm[off + r0 * rowb(K): off + r1 * rowb(K)]).reshape(r1 - r0, rowb(K))
        q = raw[:, :K].view(np.int8).reshape(r1 - r0, nb, 32).astype(np.float32)
        sc = raw[:, K:].copy().view("<f2").astype(np.float32).reshape(r1 - r0, nb)
        deq = q * sc[:, :, None]
        w = mats_fn(r0, r1).reshape(r1 - r0, nb, 32)
        err = np.abs(deq - w)
        amax = np.abs(w).max(-1)
        step = amax / 127.0
        ok = amax >= 1e-5  # blocks below are fp16-scale-underflow territory (|err| <= amax <= 1e-5): counted, not scored
        tiny[0] += int(((amax > 0) & ~ok).sum())
        mx_abs = max(mx_abs, float(err.max()))
        if ok.any():
            mx_steps = max(mx_steps, float((err.max(-1)[ok] / step[ok]).max()))
        num += float((err.astype(np.float64) ** 2).sum()); den += float((w.astype(np.float64) ** 2).sum())
    rec(cls, max_abs=mx_abs, max_steps=mx_steps, rel=(num / max(den, 1e-30)) ** 0.5)
    stats[cls]["tiny"] = stats[cls].get("tiny", 0) + tiny[0]

def f32_compare(cls, off, expected, name):
    expected = np.ascontiguousarray(expected, dtype="<f4")
    reg(name, off, expected.nbytes)
    got = np.asarray(mm[off:off + expected.nbytes]).view("<f4").reshape(expected.shape)
    ex = np.array_equal(got, expected)
    check(ex, f"{name}: fp32 tensor not bit-exact (max diff {np.abs(got - expected).max()})")
    rec(cls, exact=ex, max_abs=float(np.abs(got - expected).max()))

# embed
vocab = H["vocab"]
q8_compare("embed", H["off_embed"], lambda a, b: hf("embed_tokens.weight", a, b), vocab, dim, "embed")
f32_compare("final_norm(1+w)", H["off_final"], hf("norm.weight") + 1.0, "final_norm")

for l in range(L):
    d = descs[l]; s = d[2:]
    lin = types[l] == "linear_attention"
    check(d[0] == (1 if lin else 0), f"layer {l} type")
    p = f"layers.{l}."
    f32_compare("in/post_norm(1+w)", s[0], hf(p + "input_layernorm.weight") + 1.0, f"L{l}.in_norm")
    f32_compare("in/post_norm(1+w)", s[1], hf(p + "post_attention_layernorm.weight") + 1.0, f"L{l}.post_norm")
    q8_compare("mlp_gate", s[4], lambda a, b: hf(p + "mlp.gate_proj.weight", a, b), ff, dim, f"L{l}.gate")
    q8_compare("mlp_up", s[4] + ff * rowb(dim), lambda a, b: hf(p + "mlp.up_proj.weight", a, b), ff, dim, f"L{l}.up")
    check(s[4] % 64 == 0 and (s[4] + ff * rowb(dim)) % 64 == 0, f"L{l} gate|up contiguity alignment")
    q8_compare("mlp_down", s[5], lambda a, b: hf(p + "mlp.down_proj.weight", a, b), dim, ff, f"L{l}.down")
    if not lin:
        a = p + "self_attn."
        qg = hf(a + "q_proj.weight").reshape(nh, 2, hd, dim)  # HF per-head [q_h | gate_h]
        qrows, grows = qg[:, 0].reshape(nh * hd, dim), qg[:, 1].reshape(nh * hd, dim)
        QD, KD = nh * hd, nkv * hd
        q8_compare("attn_q", s[2], lambda x, y: qrows[x:y], QD, dim, f"L{l}.q")
        q8_compare("attn_k", s[2] + QD * rowb(dim), lambda x, y: hf(a + "k_proj.weight", x, y), KD, dim, f"L{l}.k")
        q8_compare("attn_v", s[2] + (QD + KD) * rowb(dim), lambda x, y: hf(a + "v_proj.weight", x, y), KD, dim, f"L{l}.v")
        q8_compare("attn_gate", s[2] + (QD + 2 * KD) * rowb(dim), lambda x, y: grows[x:y], QD, dim, f"L{l}.gate_out")
        q8_compare("attn_o", s[3], lambda x, y: hf(a + "o_proj.weight", x, y), dim, nh * hd, f"L{l}.o")
        f32_compare("q/k_norm(1+w)", s[6], hf(a + "q_norm.weight") + 1.0, f"L{l}.q_norm")
        f32_compare("q/k_norm(1+w)", s[7], hf(a + "k_norm.weight") + 1.0, f"L{l}.k_norm")
        check(not any(s[8:]), f"L{l} unused slots nonzero")
    else:
        a = p + "linear_attn."
        C = 6144
        q8_compare("dn_qkv", s[2], lambda x, y: hf(a + "in_proj_qkv.weight", x, y), C, dim, f"L{l}.qkv")
        q8_compare("dn_z", s[2] + C * rowb(dim), lambda x, y: hf(a + "in_proj_z.weight", x, y), 2048, dim, f"L{l}.z")
        q8_compare("dn_out", s[3], lambda x, y: hf(a + "out_proj.weight", x, y), dim, 2048, f"L{l}.out")
        cw = hf(a + "conv1d.weight")[:, 0, :]  # [6144,4]
        f32_compare("conv_w tap-major", s[6], cw.T, f"L{l}.conv")
        f32_compare("negExpA", s[7], (-np.exp(hf(a + "A_log").astype(np.float64))).astype(np.float32), f"L{l}.negExpA")
        f32_compare("dt_bias", s[8], hf(a + "dt_bias"), f"L{l}.dt_bias")
        f32_compare("gnorm_w(plain)", s[9], hf(a + "norm.weight"), f"L{l}.gnorm")
        f32_compare("W_b f32", s[10], hf(a + "in_proj_b.weight"), f"L{l}.W_b")
        f32_compare("W_a f32", s[11], hf(a + "in_proj_a.weight"), f"L{l}.W_a")
        check(not any(s[12:]), f"L{l} unused slots nonzero")

# ---- tokenizer section sanity (layout per export_weights.build_tokenizer_blob)
tb = H["off_tok"]
b2t, offs, tbytes, ht = struct.unpack_from("<QQQQ", bytes(mm[tb:tb + 32]))
n_tok_end = tb + ht  # blob start of merge table
reg("tokenizer", tb, (ht - 0) + (1 << HT_BITS) * 16 + 0)  # tokenizer blob ends at its merge table end
offs_arr = np.asarray(mm[tb + offs: tb + offs + 4 * (vocab + 1)]).view("<u4")
tokb = lambda i: bytes(mm[tb + tbytes + offs_arr[i]: tb + tbytes + offs_arr[i + 1]])
for tid, expect in ((TOK["user"], b"user"), (TOK["assistant"], b"assistant"), (TOK["im_end"], b"<|im_end|>"),
                    (TOK["im_start"], b"<|im_start|>"), (TOK["think_end"], b"</think>"), (TOK["nl"], b"\n"), (TOK["nl2"], b"\n\n")):
    check(tokb(tid) == expect, f"tok {tid} bytes {tokb(tid)!r} != {expect!r}")
check(H["eos1"] == TOK["im_end"] and H["eos2"] == TOK["endoftext"], "EOS ids")
check(TOK["argmax_limit"] == 248077, f"argmax_limit {TOK['argmax_limit']}")
htarr = np.asarray(mm[tb + ht: tb + ht + (1 << HT_BITS) * 16]).view(np.dtype([("k", "<u8"), ("m", "<u4"), ("r", "<u4")]))
tj = json.load(open(tok_json))
vcb = tj["model"]["vocab"]; merges = tj["model"]["merges"]
bad = 0
rng = np.random.default_rng(0)
for rank in list(range(0, 3000)) + list(rng.integers(0, len(merges), 20000)):
    a_, b_ = merges[rank].split(" ") if isinstance(merges[rank], str) else merges[rank]
    key = (vcb[a_] << 32) | vcb[b_]
    slot = ((key * FIB_MULT) & 0xFFFFFFFFFFFFFFFF) >> (64 - HT_BITS)
    for _ in range(64):
        if htarr[slot]["r"] == 0xFFFFFFFF:
            bad += 1; break
        if htarr[slot]["k"] == key:
            bad += htarr[slot]["r"] != rank or htarr[slot]["m"] != vcb[a_ + b_]; break
        slot = (slot + 1) & ((1 << HT_BITS) - 1)
check(bad == 0, f"{bad} merge-table lookups wrong")
print(f"\ntokenizer: id bytes OK, {23000} merge lookups via Fibonacci probe OK" if bad == 0 else "")

# ---- overlap check
regions.sort()
for (a0, a1, an), (b0, b1, bn) in zip(regions, regions[1:]):
    if bn == "tokenizer" or an == "tokenizer":
        continue
    check(a1 <= b0, f"overlap {an} [{a0},{a1}) vs {bn} [{b0},{b1})")

print(f"\nfile size {fsize:,} B ({fsize / 2**30:.3f} GiB); regions checked {len(regions)}")
print(f"\n{'class':22s} {'n':>4s} {'max_abs_err':>12s} {'max err/step':>13s} {'rel_fro':>10s}  fp32-exact  tiny-blocks(amax<1e-5)")
for cls, s in stats.items():
    q = cls in ("embed", "mlp_gate", "mlp_up", "mlp_down", "attn_q", "attn_k", "attn_v", "attn_gate", "attn_o", "dn_qkv", "dn_z", "dn_out")
    if q:
        check(s["max_steps"] <= 0.56, f"{cls}: err/step {s['max_steps']:.3f} > 0.56")
        check(s["rel"] < 0.01, f"{cls}: rel err {s['rel']:.4f} >= 1%")
    print(f"{cls:22s} {s['n']:4d} {s['max_abs']:12.3e} {s['max_steps']:13.3f} {s['rel']:10.2e}  {'-' if q else s['exact']!s:10s} {s.get('tiny', '') if q else ''}")
print("\nRESULT:", "PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(1 if fails else 0)
