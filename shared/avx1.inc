; ============================================================================
; avx1.inc — Sandy Bridge (AVX1, no FMA/F16C/AVX2) port macros
;
; Compile with -DAVX1=1 on CPUs with AVX1 but without AVX2/FMA/F16C
; (e.g. Xeon E5-1620 v1, Sandy Bridge-E). Without AVX1 defined, every
; macro emits the original single AVX2+FMA instruction (zero diff).
;
; Macros:
;   FMA231C dst, a, b      — dst += a*b, 'a' is DEAD after (consume pattern)
;   FMA231S dst, a, b      — dst += a*b, 'a' is LIVE after (scratch ymm5)
;   PMOVSXBD8 dstN, tmpN, mem64 — vpmovsxbd ymm dstN <- qword mem64
;
; Rounding note: mul+add is NOT fused — one extra IEEE754 rounding per
; accumulation vs hardware FMA. Expect ~1e-7 relative drift on long dot
; products: fine for greedy sampling, breaks bit-exact parity vs NumPy
; on borderline tokens (documented, accepted).
; ============================================================================

%ifndef AVX1_INC
%define AVX1_INC

%ifndef AVX1
%define AVX1 0
%endif

%if AVX1

; dst += a*b with 'a' dead after (the dominant llm.asm pattern: 'a' is a
; freshly dequantized product vector consumed exactly once).
%macro FMA231C 3
    vmulps  %2, %2, %3
    vaddps  %1, %1, %2
%endmacro

; dst += a*b with 'a' LIVE after (saxpy alpha broadcast). Clobbers ymm5.
%macro FMA231S 3
    vmulps  ymm5, %2, %3
    vaddps  %1, %1, ymm5
%endmacro

; vpmovsxbd ymm<dstN>, qword [<mem>] — 8 bytes -> 8 sign-extended dwords.
; VEX.128 vpmovsxbd is AVX1-legal (SSE4.1 VEX form); the 256-bit form is not.
; %1 = dst reg NUMBER (e.g. 8 -> ymm8/xmm8), %2 = tmp reg NUMBER,
; %3 = memory expression WITHOUT brackets (e.g. rax + r13 + 8).
%macro PMOVSXBD8 3
    vpmovsxbd xmm%1, dword [%3]
    vpmovsxbd xmm%2, dword [%3 + 4]
    vinsertf128 ymm%1, ymm%1, xmm%2, 1
%endmacro

; ymm<N> = broadcast(xmm<N> lane 0). AVX1 replacement for the AVX2-only
; VEX.256 register-form vbroadcastss (memory form IS AVX1-legal).
; Requires dst == src register number (true at every call site).
%macro VBSS_R 1
    vpermilps xmm%1, xmm%1, 0x00
    vperm2f128 ymm%1, ymm%1, ymm%1, 0x00
%endmacro

; CVTPH2PS_SOFT dstN, s1N, s2N, s3N, s4N, s5N
; IEEE-exact half->float conversion (4 lanes), AVX1-only (no F16C).
; Handles normal, subnormal (Sterbenz-exact ((113<<23)|m) - 2^-14),
; zero, inf and NaN. Caller must pass scratch xmm regs that are dead.

; CVTPH2PS_SOFT_MIN dstN, t1N, t2N, t3N — compact half->float (4 regs only).
; IEEE-exact for normal + subnormal + zero; e==31 (half inf/nan) falls back
; to a finite ~65536 value (kernel-only macro; use CVTPH2PS_SOFT for full IEEE).
%macro CVTPH2PS_SOFT_MIN 4
    vpunpcklwd xmm%1, xmm%1, xmm%1
    vpsrld  xmm%1, xmm%1, 16            ; h
    vpsrld  xmm%2, xmm%1, 10
    vpslld  xmm%3, xmm%2, 27
    vpsrld  xmm%2, xmm%3, 27            ; e
    vpxor   xmm%3, xmm%3, xmm%3
    vpcmpeqd xmm%3, xmm%2, xmm%3         ; t3 = (e == 0)
    vpslld  xmm%2, xmm%2, 23
    vpaddd  xmm%2, xmm%2, [rel i_38000000]
    vpslld  xmm%4, xmm%1, 22
    vpsrld  xmm%4, xmm%4, 9              ; m << 13
    vpor    xmm%2, xmm%2, xmm%4          ; f_norm
    vpor    xmm%4, xmm%4, [rel i_38800000]
    vsubps  xmm%4, xmm%4, [rel i_38800000]  ; f_sub (Sterbenz-exact)
    vpsrld  xmm%1, xmm%1, 15
    vpslld  xmm%1, xmm%1, 31             ; sign
    vblendvps xmm%2, xmm%2, xmm%4, xmm%3
    vpor    xmm%1, xmm%1, xmm%2
%endmacro

%macro CVTPH2PS_SOFT 6
    vpunpcklwd xmm%1, xmm%1, xmm%1
    vpsrld  xmm%1, xmm%1, 16            ; h (4 dwords, zero-extended)
    vpsrld  xmm%2, xmm%1, 10
    vpslld  xmm%3, xmm%2, 27
    vpsrld  xmm%2, xmm%3, 27            ; e = (h >> 10) & 0x1F
    vpxor   xmm%3, xmm%3, xmm%3
    vpcmpeqd xmm%3, xmm%2, xmm%3         ; s3 = (e == 0) subnormal mask
    vpcmpeqd xmm%4, xmm%2, [rel i_31]   ; s4 = (e == 31) inf/nan mask
    vpslld  xmm%2, xmm%2, 23
    vpaddd  xmm%2, xmm%2, [rel i_38000000]  ; (e + 112) << 23
    vpslld  xmm%5, xmm%1, 22
    vpsrld  xmm%5, xmm%5, 9              ; m << 13
    vpor    xmm%2, xmm%2, xmm%5         ; s2 = f_norm
    vpor    xmm%6, xmm%5, [rel i_7F800000]  ; s6 = f_inf_nan
    vpor    xmm%5, xmm%5, [rel i_38800000]  ; (113<<23) | m
    vsubps  xmm%5, xmm%5, [rel i_38800000]  ; s5 = f_sub (exact Sterbenz)
    vpsrld  xmm%1, xmm%1, 15
    vpslld  xmm%1, xmm%1, 31            ; dst = sign bits
    vblendvps xmm%2, xmm%2, xmm%5, xmm%3
    vblendvps xmm%2, xmm%2, xmm%6, xmm%4
    vpor    xmm%1, xmm%1, xmm%2         ; dst = sign | result
%endmacro


%else

%macro FMA231C 3
    vfmadd231ps %1, %2, %3
%endmacro

%macro FMA231S 3
    vfmadd231ps %1, %2, %3
%endmacro

; %3 here includes the full original source operand (qword [..]) since the
; AVX2 path is the single native instruction. Call sites must pass the full
; memory operand for the %else branch — see matmul call sites: they pass
; 'NN, tmp, expr' and this branch ignores tmp and rebuilds the qword load.
%macro PMOVSXBD8 3
    vpmovsxbd ymm%1, qword [%3]
%endmacro

%macro VBSS_R 1
    vbroadcastss ymm%1, xmm%1
%endmacro

%endif

%endif ; AVX1_INC

section .rodata
align 16
i_31:        times 4 dd 31
i_38000000:  times 4 dd 0x38000000     ; (112 << 23) normal bias
i_38800000:  times 4 dd 0x38800000     ; (113 << 23) == float 2^-14
i_7F800000:  times 4 dd 0x7F800000     ; inf/nan exponent
section .text
