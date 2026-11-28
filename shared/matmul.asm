; ============================================================================
; matmul.asm — High-throughput AVX2 + FMA + F16C Q8_0 matmul kernels (zero libc)
;
; Weight layout (per-row split Q8_0, block size = 32, int8 in [-127, 127]):
;   v2 (FP16 scales, HDR_FLAG_FP16_SCALES):
;     row_stride = K + (K / 32) * 2 = K + (K >> 4) bytes
;     [row + 0 .. row + K - 1]         : K int8 weights
;     [row + K .. row + row_stride - 1]: K/32 IEEE-754 fp16 block scales
;   v1 (FP32 scales, legacy):
;     row_stride = K + (K / 32) * 4 = K + (K >> 3) bytes
;     [row + 0 .. row + K - 1]         : K int8 weights
;     [row + K .. row + row_stride - 1]: K/32 fp32 block scales
;
; Compile-time modes:
;   FAST_Q8=1 (default): Q8_0 x Q8_0 INT8xINT8 AVX2 (vpabsb + vpsignb +
;                        vpmaddubsw + vpmaddwd + vcvtdq2ps + vfmadd231ps),
;                        1-row sequential 4-block (128B) unrolled with
;                        next-row software prefetch for max DRAM streaming BW.
;                        Auto-detects v1 (fp32) and v2 (fp16) scales.
;   FAST_Q8=0 (legacy) : Q8_0 x FP32 AVX2+FMA exact FP32-activation matmul,
;                        auto-detects v1 (fp32) and v2 (fp16) scales.
;
; Functions:
;   matmul_set_fp16   : (uint64 is_fp16)
;   quantize_x_q8_0   : (const float *x, uint64 K)  [FAST_Q8=1]
;   dequant_row_q8_0  : (const void *row_ptr, float *out, uint64 K)
;   matmul_q8_0_range : (const void *W, const float *x, float *out,
;                        uint64 K, uint64 r0, uint64 r1)
;   dot_f32_avx2      : (const float *a, const float *b, uint64 n) -> xmm0
;   saxpy_f32_avx2    : (float *dst, const float *src, uint64 n, float alpha)
; ============================================================================

bits 64
default rel

%ifndef FAST_Q8
%define FAST_Q8 1
%endif

%ifndef Q8_PREFETCH
%define Q8_PREFETCH prefetcht0
%endif

%define Q8_MAX_K 16384

%include "avx1.inc"

section .text

global matmul_set_fp16
%if FAST_Q8
global quantize_x_q8_0
%endif
global dequant_row_q8_0
global matmul_q8_0_range
global dot_f32_avx2
global saxpy_f32_avx2

; ----------------------------------------------------------------------------
; void matmul_set_fp16(uint64 is_fp16)
; rdi = 0 for v1 FP32 scales, non-zero for v2 FP16 scales
; ----------------------------------------------------------------------------
matmul_set_fp16:
    xor     eax, eax
    test    rdi, rdi
    setnz   al
    mov     [rel g_fp16_scales], eax
    ret

; ----------------------------------------------------------------------------
; void dequant_row_q8_0(const void *row_ptr, float *out, uint64 K)
; rdi = row_ptr, rsi = out, rdx = K (multiple of 32)
; Auto-detects v1 FP32 vs v2 FP16 scales via g_fp16_scales.
; ----------------------------------------------------------------------------
dequant_row_q8_0:
    lea     r8, [rdi + rdx]             ; scales_ptr = row_ptr + K
    mov     rcx, rdx
    shr     rcx, 5                      ; nb = K / 32
    mov     r11d, [rel g_fp16_scales]
    xor     eax, eax                    ; b = 0
.lp:
    cmp     rax, rcx
    jae     .done
    test    r11d, r11d
    jnz     .sc_f16
    vbroadcastss ymm4, [r8 + rax*4]     ; v1 fp32 scale[b]
    jmp     .sc_ok
.sc_f16:
    movzx   r10d, word [r8 + rax*2]     ; v2 fp16 scale[b]
    vmovd   xmm4, r10d
    CVTPH2PS_SOFT 4, 0, 1, 2, 3, 5
    VBSS_R 4
.sc_ok:
    mov     r9, rax
    shl     r9, 5                       ; b * 32 (q offset)
    mov     r10, rax
    shl     r10, 7                      ; b * 128 (out offset)

    PMOVSXBD8 0, 5, rdi + r9
    PMOVSXBD8 1, 5, rdi + r9 + 8
    PMOVSXBD8 2, 5, rdi + r9 + 16
    PMOVSXBD8 3, 5, rdi + r9 + 24
    vcvtdq2ps ymm0, ymm0
    vcvtdq2ps ymm1, ymm1
    vcvtdq2ps ymm2, ymm2
    vcvtdq2ps ymm3, ymm3
    vmulps  ymm0, ymm0, ymm4
    vmulps  ymm1, ymm1, ymm4
    vmulps  ymm2, ymm2, ymm4
    vmulps  ymm3, ymm3, ymm4
    vmovups [rsi + r10], ymm0
    vmovups [rsi + r10 + 32], ymm1
    vmovups [rsi + r10 + 64], ymm2
    vmovups [rsi + r10 + 96], ymm3

    inc     rax
    jmp     .lp
.done:
    ret

%if FAST_Q8
; ----------------------------------------------------------------------------
; void quantize_x_q8_0(const float *x, uint64 K)
; rdi = x (f32), rsi = K (multiple of 32, <= Q8_MAX_K)
; Quantizes x into Q8_0 blocks of 32 ONCE before matmul dispatch:
;   q8_xq[0..K-1]  : int8 in [-127, 127] (32-byte aligned, fits in L1D!)
;   q8_xs[0..K/32] : float32 scale = max_abs / 127.0f
; ----------------------------------------------------------------------------
quantize_x_q8_0:
    mov     rcx, rsi
    shr     rcx, 5                      ; nb = K / 32
    lea     r9, [rel q8_xq]
    lea     r10, [rel q8_xs]

    vbroadcastss ymm12, [rel q8_abs_mask]
    vmovss  xmm13, [rel q8_inv_127]
    vmovss  xmm14, [rel q8_f127]
    vxorps  xmm15, xmm15, xmm15
    vmovdqa ymm11, [rel q8_pack_perm]
    vbroadcastss ymm10, [rel q8_min_i8_pad]

    xor     eax, eax                    ; b = 0
    xor     r8d, r8d                    ; x byte offset (b * 128)
    xor     edx, edx                    ; xq byte offset (b * 32)
.qlp:
    cmp     rax, rcx
    jae     .qdone

    vmovups ymm0, [rdi + r8]
    vmovups ymm1, [rdi + r8 + 32]
    vmovups ymm2, [rdi + r8 + 64]
    vmovups ymm3, [rdi + r8 + 96]

    vandps  ymm4, ymm0, ymm12
    vandps  ymm5, ymm1, ymm12
    vandps  ymm6, ymm2, ymm12
    vandps  ymm7, ymm3, ymm12
    vmaxps  ymm4, ymm4, ymm5
    vmaxps  ymm6, ymm6, ymm7
    vmaxps  ymm4, ymm4, ymm6

    vextractf128 xmm5, ymm4, 1
    vmaxps  xmm4, xmm4, xmm5
    vshufps xmm5, xmm4, xmm4, 0x0E
    vmaxps  xmm4, xmm4, xmm5
    vshufps xmm5, xmm4, xmm4, 0x01
    vmaxss  xmm4, xmm4, xmm5            ; xmm4 = max_abs

    vmulss  xmm5, xmm4, xmm13           ; scale_x = max_abs / 127.0f
    vmovss  [r10 + rax*4], xmm5

    vxorps  xmm6, xmm6, xmm6            ; inv_scale = 0.0f if max_abs == 0
    vucomiss xmm4, xmm15
    jbe     .qinv_ok
    vdivss  xmm6, xmm14, xmm4           ; inv_scale = 127.0f / max_abs
.qinv_ok:
    vpermilps  xmm6, xmm6, 0x00         ; AVX1: broadcast in-lane
    vperm2f128 ymm6, ymm6, ymm6, 0x00   ; dup low lane -> 8x inv_scale

    vmulps  ymm0, ymm0, ymm6
    vmulps  ymm1, ymm1, ymm6
    vmulps  ymm2, ymm2, ymm6
    vmulps  ymm3, ymm3, ymm6

    vcvtps2dq ymm0, ymm0
    vcvtps2dq ymm1, ymm1
    vcvtps2dq ymm2, ymm2
    vcvtps2dq ymm3, ymm3

%if AVX1
    ; ---- AVX1: 128-bit pack chain (vpackSSDW/WB = SSSE2/3 VEX.128; vpermd n/a -> unpckldq) ----
    ; LOW halves: vmovdqa copy (xmm N IS low half of ymm N; vextractf128 with
    ; dst=rm-field proved unreliable on this microarch). HIGH: vextractf128 imm=1.
    ; Order: each dst written only after its source ymm half is consumed.
    ; Order: extract high halves BEFORE writing same-numbered dst xmm (writes
    ; to xmm N zero ymm N's upper half). y0.lo = xmm0 free (no copy).
    vmovdqa xmm4, xmm2                  ; y2.lo
    vmovdqa xmm6, xmm3                  ; y3.lo
    vextractf128 xmm5, ymm2, 1          ; y2.hi (before xmm2 write zeroes it)
    vextractf128 xmm7, ymm3, 1          ; y3.hi
    vmovdqa xmm2, xmm1                  ; y1.lo (ymm2 fully consumed)
    vextractf128 xmm3, ymm1, 1          ; y1.hi (before xmm3 write zeroes it)
    vextractf128 xmm1, ymm0, 1          ; y0.hi (ymm1 fully consumed)
    vpackssdw xmm0, xmm0, xmm2          ; [x0-3 | x8-11] as 8 int16
    vpackssdw xmm1, xmm1, xmm3          ; [x4-7 | x12-15]
    vpackssdw xmm4, xmm4, xmm6          ; [x16-19 | x24-27]
    vpackssdw xmm5, xmm5, xmm7          ; [x20-23 | x28-31]
    vpacksswb xmm0, xmm0, xmm4          ; [x0-3, x8-11, x16-19, x24-27]
    vpacksswb xmm1, xmm1, xmm5          ; [x4-7, x12-15, x20-23, x28-31]
    vmovdqa   xmm2, xmm0
    vpunpckldq xmm0, xmm0, xmm1         ; linear bytes 0-15
    vpunpckhdq xmm1, xmm2, xmm1         ; linear bytes 16-31
    vpmaxsb   xmm0, xmm0, xmm10         ; clamp >= -127 (SSSE3 VEX.128)
    vpmaxsb   xmm1, xmm1, xmm10
    vmovdqa   [r9 + rdx], xmm0
    vmovdqa   [r9 + rdx + 16], xmm1
%else
    vpackssdw ymm0, ymm0, ymm1
    vpackssdw ymm2, ymm2, ymm3
    vpacksswb ymm0, ymm0, ymm2
    vpermd  ymm0, ymm11, ymm0           ; restore exact linear [0..31] byte order
    vpmaxsb ymm0, ymm0, ymm10           ; clamp >= -127 (prevents -128 overflow)

    vmovdqa [r9 + rdx], ymm0            ; q8_xq[b*32 .. b*32+31]
%endif

    add     r8, 128
    add     rdx, 32
    inc     rax
    jmp     .qlp
.qdone:
    ret

; Macro: 1-row x 4-block (128 bytes = 2 cache lines) INT8xINT8 AVX2 body.
; Preconditions:
;   rax = w (current row), r8 = s_cur = w + K (scales of current row)
;   r11 = q8_xq, rbx = byte offset into weights (multiple of 128)
;   ymm0..ymm3 = 4 independent f32 accumulators (hides FMA latency!)
;   ymm12 = [s0*xs0, s1*xs1, s2*xs2, s3*xs3] duplicated in both 128b lanes
;   ymm15 = 16x int16(1)
; Prefetching [r8 + rbx] (= [w + K + rbx]) streams monotonically across both
; the current row's scales and the entire next row without jumping pages or
; resetting the CPU's L1/L2 hardware stream prefetcher.
%macro INT8_1ROW_4BLK 0
    Q8_PREFETCH [r8 + rbx]
    Q8_PREFETCH [r8 + rbx + 64]
%if AVX1
    ; ---- AVX1: 4 independent 32-byte blocks, dot chain in xmm, 256-bit FMA ----
    ; Block 0 (+0), scale index 0
    vmovdqa    xmm8,  [r11 + rbx]
    vmovdqu    xmm9,  [rax + rbx]
    vpsignb    xmm9,  xmm9, xmm8
    vpabsb     xmm8,  xmm8
    vpmaddubsw xmm9,  xmm8, xmm9
    vpmaddwd   xmm9,  xmm9, xmm15
    vmovdqa    xmm10, [r11 + rbx + 16]
    vmovdqu    xmm11, [rax + rbx + 16]
    vpsignb    xmm11, xmm11, xmm10
    vpabsb     xmm10, xmm10
    vpmaddubsw xmm11, xmm10, xmm11
    vpmaddwd   xmm11, xmm11, xmm15
    vcvtdq2ps  xmm9,  xmm9
    vcvtdq2ps  xmm11, xmm11
    vinsertf128 ymm8,  ymm8, xmm9, 0
    vinsertf128 ymm8,  ymm8, xmm11, 1
    vpermilps  ymm9,  ymm12, 0x00
    FMA231C   ymm0, ymm8, ymm9
    ; Block 1 (+32), scale index 1
    vmovdqa    xmm8,  [r11 + rbx + 32]
    vmovdqu    xmm9,  [rax + rbx + 32]
    vpsignb    xmm9,  xmm9, xmm8
    vpabsb     xmm8,  xmm8
    vpmaddubsw xmm9,  xmm8, xmm9
    vpmaddwd   xmm9,  xmm9, xmm15
    vmovdqa    xmm10, [r11 + rbx + 48]
    vmovdqu    xmm11, [rax + rbx + 48]
    vpsignb    xmm11, xmm11, xmm10
    vpabsb     xmm10, xmm10
    vpmaddubsw xmm11, xmm10, xmm11
    vpmaddwd   xmm11, xmm11, xmm15
    vcvtdq2ps  xmm9,  xmm9
    vcvtdq2ps  xmm11, xmm11
    vinsertf128 ymm8,  ymm8, xmm9, 0
    vinsertf128 ymm8,  ymm8, xmm11, 1
    vpermilps  ymm9,  ymm12, 0x55
    FMA231C   ymm1, ymm8, ymm9
    ; Block 2 (+64), scale index 2
    vmovdqa    xmm8,  [r11 + rbx + 64]
    vmovdqu    xmm9,  [rax + rbx + 64]
    vpsignb    xmm9,  xmm9, xmm8
    vpabsb     xmm8,  xmm8
    vpmaddubsw xmm9,  xmm8, xmm9
    vpmaddwd   xmm9,  xmm9, xmm15
    vmovdqa    xmm10, [r11 + rbx + 80]
    vmovdqu    xmm11, [rax + rbx + 80]
    vpsignb    xmm11, xmm11, xmm10
    vpabsb     xmm10, xmm10
    vpmaddubsw xmm11, xmm10, xmm11
    vpmaddwd   xmm11, xmm11, xmm15
    vcvtdq2ps  xmm9,  xmm9
    vcvtdq2ps  xmm11, xmm11
    vinsertf128 ymm8,  ymm8, xmm9, 0
    vinsertf128 ymm8,  ymm8, xmm11, 1
    vpermilps  ymm9,  ymm12, 0xAA
    FMA231C   ymm2, ymm8, ymm9
    ; Block 3 (+96), scale index 3
    vmovdqa    xmm8,  [r11 + rbx + 96]
    vmovdqu    xmm9,  [rax + rbx + 96]
    vpsignb    xmm9,  xmm9, xmm8
    vpabsb     xmm8,  xmm8
    vpmaddubsw xmm9,  xmm8, xmm9
    vpmaddwd   xmm9,  xmm9, xmm15
    vmovdqa    xmm10, [r11 + rbx + 112]
    vmovdqu    xmm11, [rax + rbx + 112]
    vpsignb    xmm11, xmm11, xmm10
    vpabsb     xmm10, xmm10
    vpmaddubsw xmm11, xmm10, xmm11
    vpmaddwd   xmm11, xmm11, xmm15
    vcvtdq2ps  xmm9,  xmm9
    vcvtdq2ps  xmm11, xmm11
    vinsertf128 ymm8,  ymm8, xmm9, 0
    vinsertf128 ymm8,  ymm8, xmm11, 1
    vpermilps  ymm9,  ymm12, 0xFF
    FMA231C   ymm3, ymm8, ymm9
%else
    ; Blocks 0 & 1 (+0, +32)
    vmovdqa    ymm4,  [r11 + rbx]
    vmovdqa    ymm10, [r11 + rbx + 32]
    vpabsb     ymm5,  ymm4
    vpabsb     ymm11, ymm10
    vmovdqu    ymm6,  [rax + rbx]
    vmovdqu    ymm7,  [rax + rbx + 32]
    vpsignb    ymm6,  ymm6, ymm4
    vpsignb    ymm7,  ymm7, ymm10
    vpmaddubsw ymm6,  ymm5, ymm6
    vpmaddubsw ymm7,  ymm11, ymm7
    vpmaddwd   ymm6,  ymm6, ymm15
    vpmaddwd   ymm7,  ymm7, ymm15
    vcvtdq2ps  ymm6,  ymm6
    vcvtdq2ps  ymm7,  ymm7
    vpermilps  ymm8,  ymm12, 0x00
    vpermilps  ymm9,  ymm12, 0x55
    vfmadd231ps ymm0, ymm6, ymm8
    vfmadd231ps ymm1, ymm7, ymm9

    ; Blocks 2 & 3 (+64, +96)
    vmovdqa    ymm4,  [r11 + rbx + 64]
    vmovdqa    ymm10, [r11 + rbx + 96]
    vpabsb     ymm5,  ymm4
    vpabsb     ymm11, ymm10
    vmovdqu    ymm6,  [rax + rbx + 64]
    vmovdqu    ymm7,  [rax + rbx + 96]
    vpsignb    ymm6,  ymm6, ymm4
    vpsignb    ymm7,  ymm7, ymm10
    vpmaddubsw ymm6,  ymm5, ymm6
    vpmaddubsw ymm7,  ymm11, ymm7
    vpmaddwd   ymm6,  ymm6, ymm15
    vpmaddwd   ymm7,  ymm7, ymm15
    vcvtdq2ps  ymm6,  ymm6
    vcvtdq2ps  ymm7,  ymm7
    vpermilps  ymm8,  ymm12, 0xAA
    vpermilps  ymm9,  ymm12, 0xFF
    vfmadd231ps ymm2, ymm6, ymm8
    vfmadd231ps ymm3, ymm7, ymm9
    %endif
%endmacro

; ----------------------------------------------------------------------------
; void matmul_q8_0_range(const void *W, const float *x, float *out,
;                        uint64 K, uint64 r0, uint64 r1)
; FAST_Q8=1: 1-row sequential 4-block (128B) unrolled AVX2 + FMA + F16C
; INT8xINT8 kernel with monotonic next-row software prefetch.
;   - Linear row-by-row memory traversal maximizes DRAM/L2 streaming BW
;   - 4 independent FP32 accumulators saturate Zen2/Haswell FMA & int pipes
;   - Zero stack accesses inside row/block loops
;   - 100% bit-for-bit identical per row for ANY partitioning (1T..64T)
; ----------------------------------------------------------------------------
matmul_q8_0_range:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15

    mov     rsi, r9
    sub     rsi, r8                     ; rsi = rows_left = r1 - r0
    jle     .exit

    lea     rdx, [rdx + r8 * 4]         ; rdx = out_ptr = out + r0 * 4
    vbroadcastss ymm15, [rel q8_one16_dword]  ; 16x int16(1) (AVX1 bit-identical)
    lea     r11, [rel q8_xq]
    lea     rbp, [rel q8_xs]
    mov     r15, rcx                    ; r15 = K
    mov     r13, rcx
    and     r13, -128                   ; r13 = K_128 = K & ~127

    cmp     dword [rel g_fp16_scales], 0
    jne     .fast_fp16

    ; ========================================================================
    ; FAST_Q8=1, v1 FP32 scales: row_stride = K + (K >> 3)
    ; ========================================================================
    mov     r14, rcx
    shr     r14, 3
    add     r14, rcx                    ; r14 = row_stride
    mov     rax, r8
    imul    rax, r14
    add     rax, rdi                    ; rax = w_cur = W + r0 * row_stride

align 16
.row_f32:
    lea     r12, [rax + r14]            ; r12 = w_next
    lea     r8,  [rax + r15]            ; r8  = s_cur (fp32)

    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3
    xor     ebx, ebx                    ; b_q: 0, 128, ...
    xor     r9d, r9d                    ; b_s: 0, 16, ... (fp32 scale & xs offset)
    test    r13, r13
    jz      .tail_f32

align 16
.blk4_f32:
    vmovups    xmm12, [r8 + r9]         ; 4x fp32 scale_w
    vmulps     xmm12, xmm12, [rbp + r9] ; * 4x fp32 scale_x
    vinsertf128 ymm12, ymm12, xmm12, 1

    INT8_1ROW_4BLK

    add     rbx, 128
    add     r9, 16
    cmp     rbx, r13
    jb      .blk4_f32

.tail_f32:
    cmp     rbx, r15
    jae     .red_f32
%if AVX1
    vmovdqa    xmm8,  [r11 + rbx]
    vmovdqu    xmm9,  [rax + rbx]
    vpsignb    xmm9,  xmm9, xmm8
    vpabsb     xmm8,  xmm8
    vpmaddubsw xmm9,  xmm8, xmm9
    vpmaddwd   xmm9,  xmm9, xmm15
    vmovdqa    xmm10, [r11 + rbx + 16]
    vmovdqu    xmm11, [rax + rbx + 16]
    vpsignb    xmm11, xmm11, xmm10
    vpabsb     xmm10, xmm10
    vpmaddubsw xmm11, xmm10, xmm11
    vpmaddwd   xmm11, xmm11, xmm15
    vcvtdq2ps  xmm9,  xmm9
    vcvtdq2ps  xmm11, xmm11
    vinsertf128 ymm8,  ymm8, xmm9, 0
    vinsertf128 ymm8,  ymm8, xmm11, 1
    vmovss     xmm9, [r8 + r9]
    vmulss     xmm9, xmm9, [rbp + r9]
    vpermilps  xmm9, xmm9, 0x00
    vperm2f128 ymm9, ymm9, ymm9, 0x00
    FMA231C   ymm0, ymm8, ymm9
%else
    vmovdqa    ymm4, [r11 + rbx]
    vpabsb     ymm5, ymm4
    vmovdqu    ymm6, [rax + rbx]
    vpsignb    ymm6, ymm6, ymm4
    vpmaddubsw ymm6, ymm5, ymm6
    vpmaddwd   ymm6, ymm6, ymm15
    vcvtdq2ps  ymm6, ymm6
    vmovss     xmm8, [r8 + r9]
    vmulss     xmm8, xmm8, [rbp + r9]
    vbroadcastss ymm8, xmm8
    vfmadd231ps ymm0, ymm6, ymm8
%endif
    add     rbx, 32
    add     r9, 4
    jmp     .tail_f32

.red_f32:
    vaddps  ymm0, ymm0, ymm1
    vaddps  ymm2, ymm2, ymm3
    vaddps  ymm0, ymm0, ymm2
    vhaddps ymm0, ymm0, ymm0
    vhaddps ymm0, ymm0, ymm0
    vextractf128 xmm1, ymm0, 1
    vaddss  xmm0, xmm0, xmm1
    vmovss  [rdx], xmm0
    add     rdx, 4
    mov     rax, r12
    dec     rsi
    jnz     .row_f32
    jmp     .exit

    ; ========================================================================
    ; FAST_Q8=1, v2 FP16 scales: row_stride = K + (K >> 4)
    ; ========================================================================
.fast_fp16:
    mov     r14, rcx
    shr     r14, 4
    add     r14, rcx                    ; r14 = row_stride
    mov     rax, r8
    imul    rax, r14
    add     rax, rdi                    ; rax = w_cur = W + r0 * row_stride

align 16
.row_f16:
    lea     r12, [rax + r14]            ; r12 = w_next
    lea     r8,  [rax + r15]            ; r8  = s_cur (fp16)

    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3
    xor     ebx, ebx                    ; b_q: 0, 128, ...
    xor     r9d, r9d                    ; b_s: 0, 8, ... (fp16 scale offset; xs = r9*2)
    test    r13, r13
    jz      .tail_f16

align 16
.blk4_f16:
    vmovq      xmm12, qword [r8 + r9]   ; 4x fp16 scale_w
    CVTPH2PS_SOFT 12, 8, 9, 10, 11, 13  ; -> 4x fp32 (no F16C)
    vmulps     xmm12, xmm12, [rbp + r9*2] ; * 4x fp32 scale_x
    vinsertf128 ymm12, ymm12, xmm12, 1

    INT8_1ROW_4BLK

    add     rbx, 128
    add     r9, 8
    cmp     rbx, r13
    jb      .blk4_f16

.tail_f16:
    cmp     rbx, r15
    jae     .red_f16
%if AVX1
    vmovdqa    xmm8,  [r11 + rbx]
    vmovdqu    xmm9,  [rax + rbx]
    vpsignb    xmm9,  xmm9, xmm8
    vpabsb     xmm8,  xmm8
    vpmaddubsw xmm9,  xmm8, xmm9
    vpmaddwd   xmm9,  xmm9, xmm15
    vmovdqa    xmm10, [r11 + rbx + 16]
    vmovdqu    xmm11, [rax + rbx + 16]
    vpsignb    xmm11, xmm11, xmm10
    vpabsb     xmm10, xmm10
    vpmaddubsw xmm11, xmm10, xmm11
    vpmaddwd   xmm11, xmm11, xmm15
    vcvtdq2ps  xmm9,  xmm9
    vcvtdq2ps  xmm11, xmm11
    vinsertf128 ymm8,  ymm8, xmm9, 0
    vinsertf128 ymm8,  ymm8, xmm11, 1
    movzx      ecx, word [r8 + r9]
    vmovd      xmm9, ecx
    CVTPH2PS_SOFT_MIN 9, 10, 11, 13     ; compact: ymm8 still holds fused dot!
    vmulss     xmm9, xmm9, [rbp + r9*2]
    vpermilps  xmm9, xmm9, 0x00
    vperm2f128 ymm9, ymm9, ymm9, 0x00
    FMA231C   ymm0, ymm8, ymm9
%else
    vmovdqa    ymm4, [r11 + rbx]
    vpabsb     ymm5, ymm4
    vmovdqu    ymm6, [rax + rbx]
    vpsignb    ymm6, ymm6, ymm4
    vpmaddubsw ymm6, ymm5, ymm6
    vpmaddwd   ymm6, ymm6, ymm15
    vcvtdq2ps  ymm6, ymm6
    movzx      ecx, word [r8 + r9]
    vmovd      xmm8, ecx
    vcvtph2ps  xmm8, xmm8
    vmulss     xmm8, xmm8, [rbp + r9*2]
    vbroadcastss ymm8, xmm8
    vfmadd231ps ymm0, ymm6, ymm8
%endif
    add     rbx, 32
    add     r9, 2
    jmp     .tail_f16

.red_f16:
    vaddps  ymm0, ymm0, ymm1
    vaddps  ymm2, ymm2, ymm3
    vaddps  ymm0, ymm0, ymm2
    vhaddps ymm0, ymm0, ymm0
    vhaddps ymm0, ymm0, ymm0
    vextractf128 xmm1, ymm0, 1
    vaddss  xmm0, xmm0, xmm1
    vmovss  [rdx], xmm0
    add     rdx, 4
    mov     rax, r12
    dec     rsi
    jnz     .row_f16

.exit:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

%else
; ----------------------------------------------------------------------------
; void matmul_q8_0_range(const void *W, const float *x, float *out,
;                        uint64 K, uint64 r0, uint64 r1)
; FAST_Q8=0 (legacy): 4-row unrolled AVX2 + FMA Q8_0 x FP32 matmul.
; Full FP32 activations (zero activation quantization noise).
; Auto-detects v1 FP32 scales vs v2 FP16 scales via g_fp16_scales.
; ----------------------------------------------------------------------------
matmul_q8_0_range:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 48
    ; [rsp + 0]  = out_ptr
    ; [rsp + 8]  = rows_left (r1 - r0)
    ; [rsp + 16] = row_stride
    ; [rsp + 24] = stride4
    ; [rsp + 32] = base_w_cur
    ; [rsp + 40] = is_fp16 (0 or 1)

    mov     rax, r9
    sub     rax, r8                     ; rows_left = r1 - r0
    jle     .exit
    mov     [rsp + 8], rax

    mov     ebp, [rel g_fp16_scales]
    mov     [rsp + 40], rbp

    ; row_stride = K + (K >> (is_fp16 ? 4 : 3))
    mov     rbx, rcx
    test    ebp, ebp
    jnz     .stride_f16
    shr     rbx, 3
    jmp     .stride_done
.stride_f16:
    shr     rbx, 4
.stride_done:
    add     rbx, rcx
    mov     [rsp + 16], rbx
    lea     rax, [rbx * 4]
    mov     [rsp + 24], rax

    lea     rdx, [rdx + r8 * 4]
    mov     [rsp + 0], rdx

    mov     rax, r8
    imul    rax, rbx
    add     rdi, rax
    mov     [rsp + 32], rdi

    mov     r11, rsi                    ; x_ptr (float*)
    mov     r15, rcx                    ; K

.row4_lp:
    cmp     qword [rsp + 8], 4
    jb      .row1_check

    mov     rax, [rsp + 32]             ; w0
    mov     rbx, [rsp + 16]             ; row_stride
    lea     rcx, [rax + rbx]            ; w1
    lea     rdx, [rcx + rbx]            ; w2
    lea     rsi, [rdx + rbx]            ; w3

    lea     rdi, [rax + r15]            ; s0
    lea     r8,  [rcx + r15]            ; s1
    lea     r9,  [rdx + r15]            ; s2
    lea     r10, [rsi + r15]            ; s3

    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3

    xor     r12d, r12d                  ; b_x = 0, 128, ...
    xor     r13d, r13d                  ; b_q = 0, 32,  ...
    xor     r14d, r14d                  ; b_idx = 0, 1, 2, ... (block index)
    mov     ebp, [rsp + 40]             ; is_fp16

.blk4_lp:
    vmovups ymm4, [r11 + r12]
    vmovups ymm5, [r11 + r12 + 32]
    vmovups ymm6, [r11 + r12 + 64]
    vmovups ymm7, [r11 + r12 + 96]

    ; --- Interleaved Rows 0 & 1 ---
    PMOVSXBD8 8, 9, rax + r13
    PMOVSXBD8 12, 13, rcx + r13
    vcvtdq2ps ymm8,  ymm8
    vcvtdq2ps ymm12, ymm12
    vmulps    ymm8,  ymm8,  ymm4
    vmulps    ymm12, ymm12, ymm4

    PMOVSXBD8 9, 13, rax + r13 + 8
    PMOVSXBD8 13, 10, rcx + r13 + 8
    vcvtdq2ps ymm9,  ymm9
    vcvtdq2ps ymm13, ymm13
    FMA231C ymm8, ymm9, ymm5
    FMA231C ymm12, ymm13, ymm5

    PMOVSXBD8 10, 14, rax + r13 + 16
    PMOVSXBD8 14, 11, rcx + r13 + 16
    vcvtdq2ps ymm10, ymm10
    vcvtdq2ps ymm14, ymm14
    FMA231C ymm8, ymm10, ymm6
    FMA231C ymm12, ymm14, ymm6

    PMOVSXBD8 11, 9, rax + r13 + 24
    PMOVSXBD8 15, 9, rcx + r13 + 24
    vcvtdq2ps ymm11, ymm11
    vcvtdq2ps ymm15, ymm15
    FMA231C ymm8, ymm11, ymm7
    FMA231C ymm12, ymm15, ymm7

    test    ebp, ebp
    jnz     .s01_f16
    vbroadcastss ymm9,  dword [rdi + r14*4]
    vbroadcastss ymm13, dword [r8 + r14*4]
    jmp     .s01_ok
.s01_f16:
    movzx   ebx, word [rdi + r14*2]
    vmovd   xmm9, ebx
    CVTPH2PS_SOFT_MIN 9, 10, 11, 14
    VBSS_R 9
    movzx   ebx, word [r8 + r14*2]
    vmovd   xmm13, ebx
    CVTPH2PS_SOFT_MIN 13, 10, 11, 14
    VBSS_R 13
.s01_ok:
    FMA231C ymm0, ymm8, ymm9
    FMA231C ymm1, ymm12, ymm13

    ; --- Interleaved Rows 2 & 3 ---
    PMOVSXBD8 8, 9, rdx + r13
    PMOVSXBD8 12, 13, rsi + r13
    vcvtdq2ps ymm8,  ymm8
    vcvtdq2ps ymm12, ymm12
    vmulps    ymm8,  ymm8,  ymm4
    vmulps    ymm12, ymm12, ymm4

    PMOVSXBD8 9, 13, rdx + r13 + 8
    PMOVSXBD8 13, 10, rsi + r13 + 8
    vcvtdq2ps ymm9,  ymm9
    vcvtdq2ps ymm13, ymm13
    FMA231C ymm8, ymm9, ymm5
    FMA231C ymm12, ymm13, ymm5

    PMOVSXBD8 10, 14, rdx + r13 + 16
    PMOVSXBD8 14, 11, rsi + r13 + 16
    vcvtdq2ps ymm10, ymm10
    vcvtdq2ps ymm14, ymm14
    FMA231C ymm8, ymm10, ymm6
    FMA231C ymm12, ymm14, ymm6

    PMOVSXBD8 11, 9, rdx + r13 + 24
    PMOVSXBD8 15, 9, rsi + r13 + 24
    vcvtdq2ps ymm11, ymm11
    vcvtdq2ps ymm15, ymm15
    FMA231C ymm8, ymm11, ymm7
    FMA231C ymm12, ymm15, ymm7

    test    ebp, ebp
    jnz     .s23_f16
    vbroadcastss ymm9,  dword [r9 + r14*4]
    vbroadcastss ymm13, dword [r10 + r14*4]
    jmp     .s23_ok
.s23_f16:
    movzx   ebx, word [r9 + r14*2]
    vmovd   xmm9, ebx
    CVTPH2PS_SOFT_MIN 9, 10, 11, 14
    VBSS_R 9
    movzx   ebx, word [r10 + r14*2]
    vmovd   xmm13, ebx
    CVTPH2PS_SOFT_MIN 13, 10, 11, 14
    VBSS_R 13
.s23_ok:
    FMA231C ymm2, ymm8, ymm9
    FMA231C ymm3, ymm12, ymm13

    add     r12, 128
    add     r13, 32
    inc     r14
    cmp     r13, r15
    jb      .blk4_lp

    vhaddps ymm0, ymm0, ymm1
    vhaddps ymm2, ymm2, ymm3
    vhaddps ymm0, ymm0, ymm2
    vextractf128 xmm1, ymm0, 1
    vaddps  xmm0, xmm0, xmm1

    mov     rax, [rsp + 0]
    vmovups [rax], xmm0
    add     rax, 16
    mov     [rsp + 0], rax

    mov     rax, [rsp + 24]
    add     [rsp + 32], rax
    sub     qword [rsp + 8], 4
    jmp     .row4_lp

.row1_check:
    cmp     qword [rsp + 8], 0
    je      .exit
.row1_lp:
    mov     rax, [rsp + 32]
    lea     rdi, [rax + r15]
    vxorps  ymm0, ymm0, ymm0
    xor     r12d, r12d
    xor     r13d, r13d
    xor     r14d, r14d
    mov     ebp, [rsp + 40]
.blk1_lp:
    PMOVSXBD8 8, 12, rax + r13
    PMOVSXBD8 9, 12, rax + r13 + 8
    PMOVSXBD8 10, 12, rax + r13 + 16
    PMOVSXBD8 11, 12, rax + r13 + 24
    vcvtdq2ps ymm8,  ymm8
    vcvtdq2ps ymm9,  ymm9
    vcvtdq2ps ymm10, ymm10
    vcvtdq2ps ymm11, ymm11
    vmulps    ymm8,  ymm8,  [r11 + r12]
    FMA231C ymm8, ymm9, [r11 + r12 + 32]
    FMA231C ymm8, ymm10, [r11 + r12 + 64]
    FMA231C ymm8, ymm11, [r11 + r12 + 96]
    test    ebp, ebp
    jnz     .s1_f16
    vbroadcastss ymm12, dword [rdi + r14*4]
    jmp     .s1_ok
.s1_f16:
    movzx   ebx, word [rdi + r14*2]
    vmovd   xmm12, ebx
    CVTPH2PS_SOFT_MIN 12, 9, 10, 11
    VBSS_R 12
.s1_ok:
    FMA231C ymm0, ymm8, ymm12

    add     r12, 128
    add     r13, 32
    inc     r14
    cmp     r13, r15
    jb      .blk1_lp

    vhaddps ymm0, ymm0, ymm0
    vhaddps ymm0, ymm0, ymm0
    vextractf128 xmm1, ymm0, 1
    vaddss  xmm0, xmm0, xmm1
    mov     rax, [rsp + 0]
    vmovss  [rax], xmm0
    add     rax, 4
    mov     [rsp + 0], rax
    mov     rax, [rsp + 16]
    add     [rsp + 32], rax
    dec     qword [rsp + 8]
    jnz     .row1_lp

.exit:
    add     rsp, 48
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret
%endif

; ----------------------------------------------------------------------------
; float dot_f32_avx2(const float *a, const float *b, uint64 n) -> xmm0
; rdi = a, rsi = b, rdx = n (multiple of 32, e.g. head_dim = 128)
; ----------------------------------------------------------------------------
dot_f32_avx2:
    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3
    shl     rdx, 2
    xor     eax, eax
.lp:
    cmp     rax, rdx
    jae     .red
    vmovups ymm4, [rdi + rax]
    vmovups ymm5, [rdi + rax + 32]
    vmovups ymm6, [rdi + rax + 64]
    vmovups ymm7, [rdi + rax + 96]
    FMA231C ymm0, ymm4, [rsi + rax]
    FMA231C ymm1, ymm5, [rsi + rax + 32]
    FMA231C ymm2, ymm6, [rsi + rax + 64]
    FMA231C ymm3, ymm7, [rsi + rax + 96]
    add     rax, 128
    jmp     .lp
.red:
    vaddps  ymm0, ymm0, ymm1
    vaddps  ymm2, ymm2, ymm3
    vaddps  ymm0, ymm0, ymm2
    vextractf128 xmm1, ymm0, 1
    vaddps  xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    ret

; ----------------------------------------------------------------------------
; void saxpy_f32_avx2(float *dst, const float *src, uint64 n, float alpha)
; rdi = dst, rsi = src, rdx = n (multiple of 32, e.g. head_dim = 128), xmm0 = alpha
; dst[i] += alpha * src[i]
; ----------------------------------------------------------------------------
saxpy_f32_avx2:
    VBSS_R 0
    shl     rdx, 2
    xor     eax, eax
.lp:
    cmp     rax, rdx
    jae     .done
    vmovups ymm1, [rdi + rax]
    vmovups ymm2, [rdi + rax + 32]
    vmovups ymm3, [rdi + rax + 64]
    vmovups ymm4, [rdi + rax + 96]
    FMA231S ymm1, ymm0, [rsi + rax]
    FMA231S ymm2, ymm0, [rsi + rax + 32]
    FMA231S ymm3, ymm0, [rsi + rax + 64]
    FMA231S ymm4, ymm0, [rsi + rax + 96]
    vmovups [rdi + rax], ymm1
    vmovups [rdi + rax + 32], ymm2
    vmovups [rdi + rax + 64], ymm3
    vmovups [rdi + rax + 96], ymm4
    add     rax, 128
    jmp     .lp
.done:
    ret

%if FAST_Q8
section .rodata
align 32
q8_pack_perm: dd 0, 4, 1, 5, 2, 6, 3, 7
q8_abs_mask:  dd 0x7FFFFFFF
q8_f127:      dd 127.0
q8_inv_127:   dd 0.007874015748031496       ; 1.0f / 127.0f
q8_one16:     dw 1
q8_min_i8:    db -127
q8_one16_dword: dd 0x00010001         ; AVX1 vbroadcastss source (int16(1) pairs, 4 bytes)
q8_min_i8_pad: times 4 db -127        ; AVX1 broadcast (SSSE3 vpmaxsb clamp)
%endif

section .bss
alignb 64
g_fp16_scales: resd 1
%if FAST_Q8
alignb 64
q8_xq:        resb Q8_MAX_K
alignb 64
q8_xs:        resb (Q8_MAX_K / 32) * 4
%endif
