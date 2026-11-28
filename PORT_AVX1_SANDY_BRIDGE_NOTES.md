# PORT x86-64 AVX1 (Sandy Bridge) — llm.asm @ Z420 E5-1620 v1 (2012, 4c/8t)

**Date:** 27 nov 2026. **Machine:** Xeon E5-1620 Sandy Bridge-E, AVX1/SSE4.2 — PAS AVX2/FMA3/F16C.
**Modèle:** Qwen3-0.6B Q8_0 (613 MiB .asmbin v2 fp16 scales, 643 Mo GGUF).

## Ce qui a été porté

| Couche | Statut | Note de validation |
|---|---|---|
| math.o (expf/sincos/rope/rmsnorm/silu) | ✅ bit-exact | test_math.py suite officielle, 8/8 PASS |
| matmul FAST_Q8=0 (Q8×FP32) | ✅ bit-exact | err 9.54e-07 (v1), kernel FP32 avec dequant par row |
| matmul FAST_Q8=1 (Q8×INT8) | ✅ bit-exact | err 9.54e-07 (v1) / 7.15e-07 (v2 fp16) |
| quantize_x_q8_0 (activations INT8) | ✅ bit-exact | pack 128-bit via vmovdqa copies + vextractf128 hi-only |
| Build complet runtime | ✅ | build/asm-llm — 16,192 bytes, zero libc |

## Le piège architectural majeur découvert

**Les écritures VEX.128 ZERO la moitié haute du YMM destination** (zero-upper semantics AVX).
La croyance "VEX.128 préserve le haut" est FAUSSE — c'est le comportement des SSE legacy.
Toute stratégie "écrire xmm pour lire le haut du ymm ensuite" est morte par conception.
Conséquence pratique: extraire le HAUT d'un ymm AVANT toute écriture du même numéro (ou
utiliser des vmovdqa copies pour les moitiés basses). 3 bugs distincts corrigés:

1. Reconstruct 2^k expf: split-lane VEX.128 → zero-upper → lanes 4-7 = 0.0
2. Argmax index loop: hauts figés à 8 après incrémentation xmm
3. Pack quantize: chaîne d'extraction zero-upper inter-reliée — écrire xmm N
   zéroie ymm N.haut, donc extraire le haut de ymm1 APRÈS avoir écrit xmm1
   (extrait du ymm0) = perte silencieuse. Contournement livré: vmovdqa copies
   pour les moitiés basses + vextractf128 imm=1 uniquement, ordonné par
   consommation. NOTE: le micro-test de validation (/tmp/vext_test.asm, aussi
   archivé dans le repo) prouve que le vextractf128 avec dst=modrm.RM fonctionne
   NORMALEMENT avec imm=0 et imm=1 sur cette machine — l'hypothèse initiale
   d'un registre stale dst=rm était FAUSSE (artefact de debugging gdb); le fix
   vmovdqa est robuste par construction (il supprime les extraits dont l'ordre
   inter-relié était le risque), pas parce que la forme serait cassée.

Plus: vfmadd231ps = FMA3 (absent Sandy), vcvtph2ps = F16C (absent Sandy),
vbroadcastss ymm,xmm (register form) = AVX2-only, vbroadcastss [mem] lit UN dword
(définir les constantes en dd complets pas dw/db).

## Bench head-to-head (Qwen3-0.6B Q8_0, même machine, --no-warmup llama.cpp)

| Test | asm-llm INT8 | llama.cpp Q8_0 | Delta |
|---|---|---|---|
| decode t1 | 10.14 t/s | 9.9 t/s | +2% |
| decode t2 | 17.85 t/s | 17.8 t/s | +0.3% |
| decode t4 | **29.93 t/s** | 20.7 t/s | **+45%** |
| decode t8 | **14.53 t/s** | 6.7 t/s | +117% |
| prefill 453 tok t4 | 35.97 t/s | 77.8 t/s | −54% (llama.cpp = batched GEMM) |
| prefill 453 tok t8 | — (GEMV séquentiel) | 59.8 t/s | idem |

- Le pitch du repo (+31–54% decode vs llama.cpp sur EPYC) se Réplique: +45% à t4.
- t8 = régime hyperthreading: les deux mauvais, llm.asm moins mauvais.
- FP32→INT8: decode 10.31→29.93 = 2.9× speedup. Le path FP32 = proof-of-port,
  le INT8 = le perf play (le kernel vpmaddubsw 128-bit = le même style que ggml).
- Prefill: le repo fait du GEMV token-par-token (pas de batching) — limitation
  structurelle du design llm.asm, pas du port. La voie: batched GEMM prefill.
- Mémoire: 7.8 GB/s consommés sur ~35 GB/s = compute-bound résolu par INT8
  (29.93 t/s = 67% du plafond memory-bandwidth théorique ~45–50 t/s).

## Fichiers modifiés (vs alesha-pro/llm.asm @ HEAD)

- `shared/avx1.inc` (NOUVEAU): macros FMA231C/FMA231S, PMOVSXBD8, VBSS_R,
  CVTPH2PS_SOFT (6 regs, IEEE complet), CVTPH2PS_SOFT_MIN (4 regs, normal+
  subnormal exact), guards %ifdef AVX1, constantes .rodata 16-byte alignées.
- `shared/matmul.asm`: guards %if AVX1 sur quantize pack flow, INT8_1ROW_4BLK
  (branche 128-bit), tail f32/f16, q8_one16_dword + q8_min_i8_pad .rodata.
- `shared/math.asm`: reconstruct 2^k extract-first, argmax init + loop
  extract-first, guard AVX1 sur macro FMA.
- `Makefile`: flag AVX1 ?= 0, AFLAGS -DAVX1=$(AVX1), stamp .flags_*_avx$(AVX1).
- `scripts/vextractf128_microtest.asm` (NOUVEAU): test isolé de la forme
  dst=modrm.RM + poison control (prouve le comportement normal de la forme).

## Rebuild

    make test-math  FAST_Q8=1 AVX1=1 && python3 scripts/test_math.py build/test_math
    make inference  FAST_Q8=1 AVX1=1
    ./build/asm-llm -m <model.asmbin> -p "..." -n 128 -t 4

Modèle: scripts/export_weights.py --model-dir <hf-dl> --out model.asmbin --scale-dtype fp16
(83s, 613 MiB — tokenizer.json intégré dans le blob).
