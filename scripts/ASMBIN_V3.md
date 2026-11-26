# .asmbin v3: Qwen3.5-2B layout

Written by `scripts/export_weights.py` when `config.json` describes a Qwen3.5 text decoder (hybrid Gated DeltaNet + gated full attention). Qwen3 models keep producing the original v1 (fp32 scales) and v2 (fp16 scales, default) files, byte for byte. The v1/v2 layout is the 128-byte-header, uniform-layer-stride one described in the docstring of `export_weights.py`.
`scripts/verify_export_qwen35.py` re-reads a v3 file through the header and descriptor table and compares every tensor with the Hugging Face weights.

All integers are little endian. Every tensor and section start is 64-byte aligned. Offsets are absolute from the start of the file. The values in the last column are those of Qwen3.5-2B.

## Header (256 bytes)

| off | type | field | qwen35-2b value |
|---|---|---|---|
| 0 | char[4] | magic | `ASML` |
| 4 | u32 | version | 3 |
| 8 | u32 | dim | 2048 |
| 12 | u32 | hidden_dim | 6144 |
| 16 | u32 | n_layers | 24 |
| 20 | u32 | n_heads | 8 |
| 24 | u32 | n_kv_heads | 2 |
| 28 | u32 | head_dim | 256 |
| 32 | u32 | vocab_size | 248320 (embedding rows) |
| 36 | u32 | max_seq_len | 4096 (`--max-seq-len`) |
| 40 | f32 | rms_eps | 1e-6 |
| 44 | f32 | rope_theta | 1e7 |
| 48 | u32 | eos_1 | 248046 `<\|im_end\|>` |
| 52 | u32 | eos_2 | 248044 `<\|endoftext\|>` |
| 56 | u32 | ht_mask | 524287 (2^19 slots, load 0.472) |
| 60 | u32 | flags | 7 = TIED_EMB(1) \| QK_NORM(2) \| FP16_SCALES(4) |
| 64 | u64 | off_tokenizer | 3520 |
| 72 | u64 | off_embed | 11230080 |
| 80 | u64 | off_layers | 551574400 (start of layer 0 data) |
| 88 | u64 | layer_stride | 0 (layers are NOT uniform; use the table) |
| 96 | u64 | off_final_norm | 2015737984 |
| 104 | u64 | file_size | 2015746176 |
| 112..127 | | reserved | 0 |
| 128 | u64 | off_layer_table | 256 |
| 136 | u64 | layer_type_mask | 0x777777 (bit l = 1 -> layer l is type 1/DeltaNet; i.e. full attn at 3,7,11,15,19,23) |
| 144 | u32 | n_lin_k_heads | 16 |
| 148 | u32 | n_lin_v_heads | 16 |
| 152 | u32 | lin_k_dim | 128 |
| 156 | u32 | lin_v_dim | 128 |
| 160 | u32 | conv_k | 4 |
| 164 | u32 | rot_dim | 64 (= int(256 * partial_rotary_factor 0.25); Qwen3: head_dim) |
| 168 | u32 | flags2 | 3 = ATTN_OUT_GATE(1) \| NORMS_1PW_BAKED(2) |
| 172 | u32 | reserved | 0 |
| **176** | u32 | **tok_endoftext** | 248044  |
| 180 | u32 | tok_im_start | 248045 |
| 184 | u32 | tok_im_end | 248046 |
| 188 | u32 | tok_think_start | 248068 `<think>` |
| 192 | u32 | tok_think_end | 248069 `</think>` |
| 196 | u32 | tok_user | 846 |
| 200 | u32 | tok_assistant | 74455 |
| 204 | u32 | tok_nl | 198 (`\n`) |
| 208 | u32 | tok_nl2 | 271 (`\n\n`) |
| 212 | u32 | argmax_limit | 248077 (valid token ids are < this; rows 248077..248319 are junk, never argmax/sample them) |
| 216..255 | | reserved | 0 |

All `tok_*`/`argmax_limit` fields are 0 for Qwen3 files: the kernel then uses its built-in Qwen3 ids and vocab_size.
A reader of the format should treat `argmax_limit` as the number of sampleable ids.

## Descriptor table (at off_layer_table = 256, 24 x 136 B)

Entry l: `u32 type; u32 reserved(0); u64 slot[16]` (slot value = absolute file offset, 0 = unused). Slot i at entry + 8 + 8*i.
Types: **0 = full attention** (same type id as in v1/v2; the gate / partial rope behaviour is selected by header flags2 / rot_dim),
**1 = Gated DeltaNet**.

| slot | type 0 (full attn, layers 3,7,..,23) | type 1 (DeltaNet, other 18) |
|---|---|---|
| 0 | in_norm f32[2048], **1+w** | same |
| 1 | post_norm f32[2048], **1+w** | same |
| 2 | W_proj Q8, K=2048, 5120 rows: `[q 2048 \| k 512 \| v 512 \| gate 2048]` | W_proj Q8, K=2048, 8192 rows: `[qkv 6144 \| z 2048]` |
| 3 | W_out Q8 = o_proj, M=2048, K=2048 | W_out Q8 = out_proj, M=2048, K=2048 |
| 4 | W_gateup Q8, K=2048, 12288 rows `[gate_proj 6144 \| up_proj 6144]` | same |
| 5 | W_down Q8, M=2048, K=6144 | same |
| 6 | q_norm f32[256], **1+w** | conv_w f32 **[4][6144]** tap-major |
| 7 | k_norm f32[256], **1+w** | negExpA f32[16] = -exp(A_log) |
| 8 | 0 | dt_bias f32[16] |
| 9 | 0 | gnorm_w f32[128], plain w (NOT 1+w) |
| 10 | 0 | W_b f32[16][2048] row-major (beta input; NOT quantized) |
| 11 | 0 | W_a f32[16][2048] row-major (decay input; NOT quantized) |
| 12-15 | 0 | 0 |

Details:
- **q/gate de-interleave**: HF `q_proj` rows are per head `[q_h(256) | gate_h(256)]` (512 rows/head). Exported: rows `h*256..h*256+255` of the q block = HF rows `h*512..h*512+255`; rows `h*256..` of the gate block = HF rows `h*512+256..h*512+511`. q, k, v, gate blocks are contiguous (no padding; every row stride is a multiple of 64) so one matmul with M=5120 produces `buf_q | buf_k | buf_v | buf_gate`.
- DeltaNet `qkv` block: rows 0..2047 q, 2048..4095 k, 4096..6143 v, each head-major (head h = 128 consecutive rows), exactly HF order. `z` rows 2048 head-major `[16][128]`. conv channel c of the 6144 = qkv row c.
- **conv_w**: `conv_w[j*6144 + c]` = HF `conv1d.weight[c,0,j]`, j=0..3; tap 3 multiplies the current token's pre-conv `qkv[c]`, tap 0 the oldest of the last 3 saved. No bias. Activation SiLU after the sum.
- **a/b stay fp32**: only 2 x 16 x 2048 values (256 KB / layer); they feed `exp`/`softplus`/`sigmoid` of the decay and write strength, where relative Q8 error is amplified recurrently. Use `dot_f32_avx2` against the f32 normed input (K=2048). 
- negExpA is computed as `float32(-exp(float64(A_log)))`. Decay is `g = negExpA * softplus(a + dt_bias)`.
- Norm `1+w` is baked (flags2 bit 1): kernel multiplies by the stored value directly. The DeltaNet gated norm uses the stored plain `w`.
- Q8 matmul constraints satisfied: all K in {2048, 6144}, multiples of 32, <= 16384.

## Q8_0 matrix encoding (identical to Qwen3 files)

Per row (K elements): `K` int8 (symmetric, [-127,127], blocks of 32) followed by `K/32` fp16 scales; scale = amax/127 rounded to fp16 (q computed with the rounded scale).
Row stride: K=2048 -> 2176 B, K=6144 -> 6528 B (both multiples of 64). Matrix bytes = M * stride.
Blocks whose amax is so small that the fp16 scale underflows quantize to 0 (error <= amax <= ~1e-37; 896 such blocks in layer 0 `in_proj_qkv`, dead channels).

## Section order in the file

1. Header 0..255; descriptor table 256..(256+3264); tokenizer blob (off_tokenizer, internal layout unchanged: 64 B sub-header `u64 off_byte_to_tok, off_tok_offsets, off_tok_bytes, off_merge_ht`, then `u32 byte_to_tok[256]`, `u32 tok_offsets[vocab+1]`, token bytes, 2^19 x 16 B merge entries `{u64 key=(id_a<<32)|id_b, u32 merged_id, u32 rank}`, Fibonacci hash `(key*0x9E3779B97F4A7C15)>>45`, linear probe, empty rank 0xFFFFFFFF).
2. Embedding Q8 `[248320][2048]` (tied lm_head), 540,344,320 bytes = 248320 x 2176.
3. Layers 0..23, each: in_norm, [q_norm, k_norm | conv_w, negExpA, dt_bias, gnorm_w, W_b, W_a], post_norm, W_proj, W_out, W_gateup, W_down; each tensor 64-aligned, layer padded to 64.
4. final_norm f32[2048], **1+w**, then file padded to 64.

Total file: 2,015,746,176 B.
