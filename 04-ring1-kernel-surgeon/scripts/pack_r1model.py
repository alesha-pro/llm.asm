#!/usr/bin/env python3
"""
Pack trained MiniCPM5-2B GRPO Q4_K_M GGUF + GPT-2 byte-level BPE tokenizer
into a flat 2 GiB NVDIMM-ready bare-metal Ring -1 binary (.r1m).

Zero libc, zero GGUF parser needed in Ring -1:
  - Fixed 4KB header with exact physical offsets of all 381 tensors
  - Raw Q4_K / Q6_K / F32 tensor blobs copied bit-for-bit from GGUF
  - Flat UTF-8 vocab decode pool + O(1) FNV-1a BPE merge hash table
"""
from __future__ import annotations
import json
import os
import struct
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("LLAMA_GGUF_PY"):
    sys.path.insert(0, os.environ["LLAMA_GGUF_PY"])
from gguf import GGUFReader

GGUF_IN = os.environ.get(
    "GGUF_IN",
    os.path.join(ROOT_DIR, "models", "minicpm5-2b-ring1-surgeon-grpo-q4_k_m.gguf"),
)
HF_TOK_DIR = os.environ.get(
    "HF_TOK_DIR",
    os.path.join(ROOT_DIR, "runs", "grpo-minicpm5-2b", "merged"),
)
OUT_R1M = os.environ.get(
    "OUT_R1M",
    os.path.join(ROOT_DIR, "models", "minicpm5-2b-surgeon.r1m"),
)
NVDIMM_SIZE = 2 * 1024 * 1024 * 1024  # Exact 2 GiB for QEMU NVDIMM

DTYPE_MAP = {"F32": 0, "Q4_K": 1, "Q6_K": 2}

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

def align64(n: int) -> int:
    return (n + 63) & ~63

def main():
    print(f"[PACK] Reading GGUF: {GGUF_IN}")
    r = GGUFReader(GGUF_IN)
    tmap = {t.name: t for t in r.tensors}

    dim = int(r.get_field("llama.embedding_length").contents())
    n_layers = int(r.get_field("llama.block_count").contents())
    ffn_dim = int(r.get_field("llama.feed_forward_length").contents())
    n_heads = int(r.get_field("llama.attention.head_count").contents())
    n_kv_heads = int(r.get_field("llama.attention.head_count_kv").contents())
    head_dim = dim // n_heads
    rms_eps = float(r.get_field("llama.attention.layer_norm_rms_epsilon").contents())
    rope_theta = float(r.get_field("llama.rope.freq_base").contents())
    vocab_size = int(tmap["output.weight"].shape[1])

    print(f"[PACK] Model: dim={dim} layers={n_layers} ffn={ffn_dim} heads={n_heads}/{n_kv_heads} vocab={vocab_size} eps={rms_eps} theta={rope_theta}")

    # Load exact HF tokenizer.json for byte-level BPE vocab & merges
    tj = json.load(open(os.path.join(HF_TOK_DIR, "tokenizer.json")))
    vocab_dict = tj["model"]["vocab"]
    added_tokens = {a["id"]: a["content"] for a in tj.get("added_tokens", [])}
    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}

    # Build decoded byte strings for every token id 0..vocab_size-1
    id_to_str = [""] * vocab_size
    for tok_s, tid in vocab_dict.items():
        if 0 <= tid < vocab_size:
            id_to_str[tid] = tok_s
    for tid, tok_s in added_tokens.items():
        if 0 <= tid < vocab_size:
            id_to_str[tid] = tok_s

    vocab_pool = bytearray()
    vocab_offs = []
    vocab_lens = []
    for tid in range(vocab_size):
        s = id_to_str[tid]
        if tid in added_tokens:
            raw = s.encode("utf-8")
        else:
            raw = bytes(u2b[ch] if ch in u2b else ord("?") for ch in s)
        vocab_offs.append(len(vocab_pool))
        vocab_lens.append(len(raw))
        vocab_pool.extend(raw)

    # Map each single raw byte 0..255 -> token id
    byte_to_tok = [0] * 256
    for b in range(256):
        ch = b2u[b]
        byte_to_tok[b] = vocab_dict[ch]

    # Build O(1) open-addressed FNV-1a hash table for BPE merges
    # Slot: (uint32 left_id, uint32 right_id, uint32 merged_id, uint32 rank) — 16 bytes
    # Empty slot sentinel: rank == 0xFFFFFFFF
    merges = tj["model"]["merges"]
    merge_entries = []
    for rank, m in enumerate(merges):
        if isinstance(m, str):
            a, b = m.split(" ", 1)
        else:
            a, b = m[0], m[1]
        ab = a + b
        if a in vocab_dict and b in vocab_dict and ab in vocab_dict:
            merge_entries.append((vocab_dict[a], vocab_dict[b], vocab_dict[ab], rank))

    ht_cap = 1
    while ht_cap < len(merge_entries) * 3:
        ht_cap <<= 1
    print(f"[PACK] BPE merges: {len(merge_entries)} -> hash table cap={ht_cap} ({ht_cap*16/1048576:.1f} MiB)")

    ht_buf = bytearray(ht_cap * 16)
    # Fill ranks with 0xFFFFFFFF
    for i in range(ht_cap):
        struct.pack_into("<IIII", ht_buf, i * 16, 0, 0, 0, 0xFFFFFFFF)

    mask = ht_cap - 1
    for lid, rid, mid, rank in merge_entries:
        h = ((lid * 2654435761) ^ (rid * 2246822519)) & 0xFFFFFFFF
        idx = h & mask
        while struct.unpack_from("<I", ht_buf, idx * 16 + 12)[0] != 0xFFFFFFFF:
            idx = (idx + 1) & mask
        struct.pack_into("<IIII", ht_buf, idx * 16, lid, rid, mid, rank)

    # Order tensors deterministically:
    # Global: token_embd.weight, output_norm.weight, output.weight
    # Per layer l=0..41 (9 tensors): attn_norm, attn_q, attn_k, attn_v, attn_output, ffn_norm, ffn_gate, ffn_up, ffn_down
    tensor_names = ["token_embd.weight", "output_norm.weight", "output.weight"]
    for l in range(n_layers):
        for suffix in [
            "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
            "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"
        ]:
            tensor_names.append(f"blk.{l}.{suffix}")

    assert len(tensor_names) == 3 + n_layers * 9

    # Layout:
    #   [0 .. 4096)                   : R1MHeader
    #   [4096 .. 4096 + N*16)         : TensorDesc[381] = (uint64 offset, uint32 dtype, uint32 nbytes)
    #   aligned 64B                   : byte_to_tok[256] (1024 B)
    #   aligned 64B                   : vocab_offs[vocab_size] (u32), vocab_lens[vocab_size] (u16), vocab_pool
    #   aligned 64B                   : merge_ht[ht_cap * 16]
    #   aligned 4096B                 : raw tensor payloads (each 64B-aligned)
    cur = 4096
    tdesc_off = cur
    cur = align64(cur + len(tensor_names) * 16)

    b2t_off = cur
    cur = align64(cur + 256 * 4)

    voffs_off = cur
    cur = align64(cur + vocab_size * 4)

    vlens_off = cur
    cur = align64(cur + vocab_size * 2)

    vpool_off = cur
    cur = align64(cur + len(vocab_pool))

    ht_off = cur
    cur = (cur + len(ht_buf) + 4095) & ~4095

    tdescs = []
    for name in tensor_names:
        t = tmap[name]
        dt = DTYPE_MAP[t.tensor_type.name]
        nb = int(t.n_bytes)
        tdescs.append((cur, dt, nb, t))
        cur = align64(cur + nb)

    total_used = cur
    print(f"[PACK] Total packed bytes: {total_used / (1024**3):.3f} GiB (fits inside 2.000 GiB NVDIMM: {total_used <= NVDIMM_SIZE})")
    assert total_used <= NVDIMM_SIZE

    hdr = struct.pack(
        "<8sIIIIIIIIffIIIIIIIIQQQQQQQQ",
        b"R1SURG01",
        dim, n_layers, ffn_dim, n_heads, n_kv_heads, head_dim, vocab_size, len(tensor_names),
        rms_eps, rope_theta,
        0, 1, 130072, 130073, 8, 9, 130063, ht_cap,
        tdesc_off, b2t_off, voffs_off, vlens_off, vpool_off, len(vocab_pool), ht_off, total_used
    )
    assert len(hdr) <= 4096

    os.makedirs(os.path.dirname(OUT_R1M), exist_ok=True)
    with open(OUT_R1M, "wb") as out:
        out.write(hdr + b"\x00" * (4096 - len(hdr)))

        # Tensor descriptors
        out.seek(tdesc_off)
        for off, dt, nb, _ in tdescs:
            out.write(struct.pack("<QII", off, dt, nb))

        out.seek(b2t_off)
        out.write(struct.pack("<256I", *byte_to_tok))

        out.seek(voffs_off)
        out.write(struct.pack(f"<{vocab_size}I", *vocab_offs))

        out.seek(vlens_off)
        out.write(struct.pack(f"<{vocab_size}H", *vocab_lens))

        out.seek(vpool_off)
        out.write(vocab_pool)

        out.seek(ht_off)
        out.write(ht_buf)

        # Stream raw GGUF tensor payloads
        for idx, (off, dt, nb, t) in enumerate(tdescs):
            out.seek(off)
            out.write(memoryview(t.data).tobytes())
            if idx % 90 == 0:
                print(f"[PACK] Wrote tensor {idx+1}/{len(tdescs)}...")

        # Pad cleanly to exact 2 GiB for QEMU NVDIMM
        out.truncate(NVDIMM_SIZE)

    print(f"[PACK] DONE -> {OUT_R1M} ({os.path.getsize(OUT_R1M)} bytes)")

if __name__ == "__main__":
    main()
