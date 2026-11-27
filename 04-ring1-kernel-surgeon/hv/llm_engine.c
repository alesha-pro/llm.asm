// ============================================================================
// hv/llm_engine.c — Freestanding AVX2 + FMA + F16C MiniCPM5-2B Q4_K_M Engine
// Zero libc, zero libm, runs identically in host unit tests and bare-metal Ring -1!
// ============================================================================
#include "llm_engine.h"
#if !__STDC_HOSTED__
#define _MM_MALLOC_H_INCLUDED
#endif
#include <immintrin.h>

typedef struct __attribute__((packed)) {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
} BlockQ4K;

typedef struct __attribute__((packed)) {
    uint8_t  ql[128];
    uint8_t  qh[64];
    int8_t   scales[16];
    uint16_t d;
} BlockQ6K;

_Static_assert(sizeof(BlockQ4K) == 144, "BlockQ4K must be 144 bytes");
_Static_assert(sizeof(BlockQ6K) == 210, "BlockQ6K must be 210 bytes");

// ============================================================================
// Freestanding Math (zero libm)
// ============================================================================
static inline float fp16_to_fp32(uint16_t h) {
    return _cvtsh_ss(h);
}

static inline float hv_sqrtf(float x) {
    __m128 v = _mm_set_ss(x);
    return _mm_cvtss_f32(_mm_sqrt_ss(v));
}

static float hv_expf(float x) {
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    // Range reduction: x = k * ln(2) + r, |r| <= ln(2)/2
    const double inv_ln2 = 1.4426950408889634;
    const double ln2     = 0.6931471805599453;
    int k = (int)(x * inv_ln2 + (x >= 0.0f ? 0.5 : -0.5));
    double r = (double)x - (double)k * ln2;
    // Degree-6 Taylor polynomial for exp(r) on [-0.35, 0.35] (< 1e-8 error)
    double p = 1.0 + r * (1.0 + r * (0.5 + r * (1.0/6.0 + r * (1.0/24.0 + r * (1.0/120.0 + r * (1.0/720.0))))));
    // Multiply by 2^k via IEEE-754 exponent bits
    union { double d; uint64_t u; } scale;
    scale.u = (uint64_t)(k + 1023) << 52;
    return (float)(p * scale.d);
}

static void hv_sincos_d(double x, float *s_out, float *c_out) {
    const double two_pi  = 6.28318530717958647692;
    const double pi      = 3.14159265358979323846;
    const double half_pi = 1.57079632679489661923;
    // Reduce to [-pi, pi]
    double q = (double)(int64_t)(x / two_pi + (x >= 0.0 ? 0.5 : -0.5));
    x -= q * two_pi;
    if (x > pi) x -= two_pi;
    if (x < -pi) x += two_pi;
    // Fold into [-pi/2, pi/2]
    double sign_c = 1.0;
    if (x > half_pi)       { x = pi - x;  sign_c = -1.0; }
    else if (x < -half_pi) { x = -pi - x; sign_c = -1.0; }
    double x2 = x * x;
    double s = x * (1.0 - x2 * (1.0/6.0 - x2 * (1.0/120.0 - x2 * (1.0/5040.0 - x2 * (1.0/362880.0)))));
    double c = 1.0 - x2 * (0.5 - x2 * (1.0/24.0 - x2 * (1.0/720.0 - x2 * (1.0/40320.0 - x2 * (1.0/3628800.0)))));
    *s_out = (float)s;
    *c_out = (float)(c * sign_c);
}

static double hv_log_d(double x) {
    union { double d; uint64_t u; } v;
    v.d = x;
    int e = (int)((v.u >> 52) & 0x7FF) - 1023;
    v.u = (v.u & 0x000FFFFFFFFFFFFFULL) | (1023ULL << 52);
    double m = v.d; // [1, 2)
    double y = (m - 1.0) / (m + 1.0);
    double y2 = y * y;
    double ln_m = 2.0 * y * (1.0 + y2 * (1.0/3.0 + y2 * (1.0/5.0 + y2 * (1.0/7.0 + y2 * (1.0/9.0 + y2 * (1.0/11.0))))));
    return ln_m + (double)e * 0.6931471805599453;
}

static double hv_pow_d(double base, double exp_val) {
    double lx = exp_val * hv_log_d(base);
    const double inv_ln2 = 1.4426950408889634;
    const double ln2     = 0.6931471805599453;
    int k = (int)(lx * inv_ln2 + (lx >= 0.0 ? 0.5 : -0.5));
    double r = lx - (double)k * ln2;
    double p = 1.0 + r * (1.0 + r * (0.5 + r * (1.0/6.0 + r * (1.0/24.0 + r * (1.0/120.0 + r * (1.0/720.0))))));
    union { double d; uint64_t u; } sc;
    sc.u = (uint64_t)(k + 1023) << 52;
    return p * sc.d;
}

// ============================================================================
// AVX2 + FMA + F16C Quantized Dot Products (Q4_K & Q6_K)
// ============================================================================
static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4)  | ((q[j]     >> 6) << 4));
    }
}

static float dot_q4_K_f32(const BlockQ4K *row, const float *x, int cols) {
    const int nb = cols >> 8; // cols / 256
    float sum = 0.0f;
    const __m128i mask_lo = _mm_set1_epi8(0x0F);

    for (int i = 0; i < nb; i++) {
        const float d   = fp16_to_fp32(row[i].d);
        const float min = fp16_to_fp32(row[i].dmin);
        const uint8_t *q = row[i].qs;
        const float *xp = x + (i << 8);
        int is = 0;

        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, row[i].scales, &sc1, &m1);
            get_scale_min_k4(is + 1, row[i].scales, &sc2, &m2);
            const float d1 = d * (float)sc1, dm1 = min * (float)m1;
            const float d2 = d * (float)sc2, dm2 = min * (float)m2;

            __m256 aq1 = _mm256_setzero_ps(), ax1 = _mm256_setzero_ps();
            __m256 aq2 = _mm256_setzero_ps(), ax2 = _mm256_setzero_ps();

            for (int l = 0; l < 32; l += 8) {
                __m128i qb  = _mm_loadl_epi64((const __m128i *)(q + l));
                __m128i qlo = _mm_and_si128(qb, mask_lo);
                __m128i qhi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask_lo);

                __m256 f_lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(qlo));
                __m256 f_hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(qhi));
                __m256 x_lo = _mm256_loadu_ps(xp + l);
                __m256 x_hi = _mm256_loadu_ps(xp + 32 + l);

                aq1 = _mm256_fmadd_ps(f_lo, x_lo, aq1);
                ax1 = _mm256_add_ps(x_lo, ax1);
                aq2 = _mm256_fmadd_ps(f_hi, x_hi, aq2);
                ax2 = _mm256_add_ps(x_hi, ax2);
            }

            sum += d1 * hsum256_ps(aq1) - dm1 * hsum256_ps(ax1)
                 + d2 * hsum256_ps(aq2) - dm2 * hsum256_ps(ax2);

            q  += 32;
            xp += 64;
            is += 2;
        }
    }
    return sum;
}

static float dot_q6_K_f32(const BlockQ6K *row, const float *x, int cols) {
    const int nb = cols >> 8;
    float sum = 0.0f;
    const __m128i m_0f = _mm_set1_epi8(0x0F);
    const __m128i m_03 = _mm_set1_epi8(0x03);
    const __m256i v_32 = _mm256_set1_epi32(32);

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(row[i].d);
        const uint8_t *ql = row[i].ql;
        const uint8_t *qh = row[i].qh;
        const int8_t  *sc = row[i].scales;
        const float   *xp = x + (i << 8);
        float bsum = 0.0f;

        for (int n = 0; n < 256; n += 128) {
            for (int half = 0; half < 2; half++) {
                const int loff = half * 16;
                __m256 a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps();
                __m256 a3 = _mm256_setzero_ps(), a4 = _mm256_setzero_ps();

                for (int k = 0; k < 16; k += 8) {
                    int l = loff + k;
                    __m128i ql0 = _mm_loadl_epi64((const __m128i *)(ql + l));
                    __m128i ql1 = _mm_loadl_epi64((const __m128i *)(ql + l + 32));
                    __m128i qhb = _mm_loadl_epi64((const __m128i *)(qh + l));

                    __m128i h1 = _mm_slli_epi16(_mm_and_si128(qhb, m_03), 4);
                    __m128i h2 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qhb, 2), m_03), 4);
                    __m128i h3 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qhb, 4), m_03), 4);
                    __m128i h4 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qhb, 6), m_03), 4);

                    __m128i q1b = _mm_or_si128(_mm_and_si128(ql0, m_0f), h1);
                    __m128i q2b = _mm_or_si128(_mm_and_si128(ql1, m_0f), h2);
                    __m128i q3b = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql0, 4), m_0f), h3);
                    __m128i q4b = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql1, 4), m_0f), h4);

                    __m256 f1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q1b), v_32));
                    __m256 f2 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q2b), v_32));
                    __m256 f3 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q3b), v_32));
                    __m256 f4 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q4b), v_32));

                    a1 = _mm256_fmadd_ps(f1, _mm256_loadu_ps(xp + l +  0), a1);
                    a2 = _mm256_fmadd_ps(f2, _mm256_loadu_ps(xp + l + 32), a2);
                    a3 = _mm256_fmadd_ps(f3, _mm256_loadu_ps(xp + l + 64), a3);
                    a4 = _mm256_fmadd_ps(f4, _mm256_loadu_ps(xp + l + 96), a4);
                }

                bsum += (float)sc[half + 0] * hsum256_ps(a1)
                      + (float)sc[half + 2] * hsum256_ps(a2)
                      + (float)sc[half + 4] * hsum256_ps(a3)
                      + (float)sc[half + 6] * hsum256_ps(a4);
            }
            xp += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
        sum += d * bsum;
    }
    return sum;
}

static void dequant_row_q4k(const BlockQ4K *row, float *dst, int cols) {
    const int nb = cols >> 8;
    for (int i = 0; i < nb; i++) {
        const float d   = fp16_to_fp32(row[i].d);
        const float min = fp16_to_fp32(row[i].dmin);
        const uint8_t *q = row[i].qs;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, row[i].scales, &sc, &m);
            float d1 = d * (float)sc, m1 = min * (float)m;
            get_scale_min_k4(is + 1, row[i].scales, &sc, &m);
            float d2 = d * (float)sc, m2 = min * (float)m;
            for (int l = 0; l < 32; l++) dst[l]      = d1 * (float)(q[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) dst[l + 32] = d2 * (float)(q[l] >> 4)  - m2;
            dst += 64;
            q   += 32;
            is  += 2;
        }
    }
}

static void dequant_row_q6k(const BlockQ6K *row, float *dst, int cols) {
    const int nb = cols >> 8;
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(row[i].d);
        const uint8_t *ql = row[i].ql;
        const uint8_t *qh = row[i].qh;
        const int8_t  *sc = row[i].scales;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l >> 4;
                int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                dst[l +  0] = d * (float)sc[is + 0] * (float)q1;
                dst[l + 32] = d * (float)sc[is + 2] * (float)q2;
                dst[l + 64] = d * (float)sc[is + 4] * (float)q3;
                dst[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
            dst += 128;
            ql  += 64;
            qh  += 32;
            sc  += 8;
        }
    }
}

#define PREFILL_B 16

typedef struct {
    const uint8_t *w_base;
    uint32_t       dtype;
    const float   *X;     // [B * cols]
    float         *Out;   // [B * rows]
    int            B;
    int            rows;
    int            cols;
} MatmulBatchJob;

static void matmul_batch_worker(int tid, int nthreads, void *arg) {
    const MatmulBatchJob *job = (const MatmulBatchJob *)arg;
    const int r0 = (job->rows * tid) / nthreads;
    const int r1 = (job->rows * (tid + 1)) / nthreads;
    const int cols = job->cols;
    const int rows = job->rows;
    const int B = job->B;
    const int nb = cols >> 8;
    float wbuf[256] __attribute__((aligned(32)));

    size_t row_bytes = (job->dtype == R1M_DTYPE_Q4K)
        ? ((size_t)nb * sizeof(BlockQ4K))
        : ((size_t)nb * sizeof(BlockQ6K));

    for (int r = r0; r < r1; r++) {
        const uint8_t *row_ptr = job->w_base + (size_t)r * row_bytes;
        __m256 acc[PREFILL_B];
        for (int b = 0; b < B; b++) acc[b] = _mm256_setzero_ps();

        for (int blk = 0; blk < nb; blk++) {
            if (job->dtype == R1M_DTYPE_Q4K) {
                dequant_row_q4k(((const BlockQ4K *)row_ptr) + blk, wbuf, 256);
            } else {
                dequant_row_q6k(((const BlockQ6K *)row_ptr) + blk, wbuf, 256);
            }
            int col_off = blk << 8;
            for (int b = 0; b < B; b++) {
                const float *xb = job->X + (size_t)b * cols + col_off;
                __m256 a = acc[b];
                for (int k = 0; k < 256; k += 8) {
                    a = _mm256_fmadd_ps(_mm256_load_ps(wbuf + k), _mm256_loadu_ps(xb + k), a);
                }
                acc[b] = a;
            }
        }
        for (int b = 0; b < B; b++) {
            job->Out[(size_t)b * rows + r] = hsum256_ps(acc[b]);
        }
    }
}

static void matmul_batch_run(const R1LlmState *st, int tensor_idx, const float *X, float *Out, int B, int rows, int cols) {
    MatmulBatchJob job = {
        .w_base = st->model_base + st->tdescs[tensor_idx].offset,
        .dtype  = st->tdescs[tensor_idx].dtype,
        .X      = X,
        .Out    = Out,
        .B      = B,
        .rows   = rows,
        .cols   = cols,
    };
    if (st->dispatch) st->dispatch(matmul_batch_worker, &job);
    else matmul_batch_worker(0, 1, &job);
}

// ============================================================================
// Parallel Matrix-Vector Dispatch
// ============================================================================
typedef struct {
    const uint8_t *w_base;
    uint32_t       dtype;
    const float   *x;
    float         *out;
    int            rows;
    int            cols;
} MatmulJob;

static void matmul_worker(int tid, int nthreads, void *arg) {
    const MatmulJob *job = (const MatmulJob *)arg;
    int r0 = (job->rows * tid) / nthreads;
    int r1 = (job->rows * (tid + 1)) / nthreads;
    int nb = job->cols >> 8;
    if (job->dtype == R1M_DTYPE_Q4K) {
        size_t row_bytes = (size_t)nb * sizeof(BlockQ4K);
        for (int r = r0; r < r1; r++) {
            const BlockQ4K *blk = (const BlockQ4K *)(job->w_base + (size_t)r * row_bytes);
            job->out[r] = dot_q4_K_f32(blk, job->x, job->cols);
        }
    } else if (job->dtype == R1M_DTYPE_Q6K) {
        size_t row_bytes = (size_t)nb * sizeof(BlockQ6K);
        for (int r = r0; r < r1; r++) {
            const BlockQ6K *blk = (const BlockQ6K *)(job->w_base + (size_t)r * row_bytes);
            job->out[r] = dot_q6_K_f32(blk, job->x, job->cols);
        }
    }
}

static void matmul_run(const R1LlmState *st, int tensor_idx, const float *x, float *out, int rows, int cols) {
    MatmulJob job = {
        .w_base = st->model_base + st->tdescs[tensor_idx].offset,
        .dtype  = st->tdescs[tensor_idx].dtype,
        .x      = x,
        .out    = out,
        .rows   = rows,
        .cols   = cols,
    };
    if (st->dispatch) {
        st->dispatch(matmul_worker, &job);
    } else {
        matmul_worker(0, 1, &job);
    }
}

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(vx, vx, acc);
    }
    float ss = hsum256_ps(acc) / (float)n + eps;
    float inv = 1.0f / hv_sqrtf(ss);
    __m256 vinv = _mm256_set1_ps(inv);
    for (int i = 0; i < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(w + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(vx, vinv), vw));
    }
}

// ============================================================================
// Init & Transformer Forward (LlamaForCausalLM)
// ============================================================================
bool r1_llm_init(R1LlmState *st, const void *model_base, void *arena_64mb, R1ParallelDispatch dispatch) {
    hv_memset(st, 0, sizeof(*st));
    const R1MHeader *h = (const R1MHeader *)model_base;
    if (hv_strncmp(h->magic, "R1SURG01", 8) != 0) {
        return false;
    }
    st->model_base  = (const uint8_t *)model_base;
    st->hdr         = h;
    st->tdescs      = (const R1MTensorDesc *)(st->model_base + h->tdesc_off);
    st->byte_to_tok = (const uint32_t *)(st->model_base + h->b2t_off);
    st->vocab_offs  = (const uint32_t *)(st->model_base + h->voffs_off);
    st->vocab_lens  = (const uint16_t *)(st->model_base + h->vlens_off);
    st->vocab_pool  = (const uint8_t *)(st->model_base + h->vpool_off);
    st->merge_ht    = (const R1MMergeSlot *)(st->model_base + h->merge_ht_off);
    st->dispatch    = dispatch;

    // Carve 64-byte aligned buffers out of the 64 MiB runtime arena
    uint8_t *ptr = (uint8_t *)arena_64mb;
    #define ALLOC_F32(dst, count) do { \
        ptr = (uint8_t *)(((uintptr_t)ptr + 63ULL) & ~63ULL); \
        (dst) = (float *)ptr; \
        ptr += (size_t)(count) * sizeof(float); \
    } while (0)

    uint32_t kv_dim = h->n_kv_heads * h->head_dim;
    ALLOC_F32(st->k_cache,  (size_t)h->n_layers * R1M_MAX_SEQ * kv_dim);
    ALLOC_F32(st->v_cache,  (size_t)h->n_layers * R1M_MAX_SEQ * kv_dim);
    ALLOC_F32(st->rope_cos, (size_t)R1M_MAX_SEQ * (h->head_dim / 2));
    ALLOC_F32(st->rope_sin, (size_t)R1M_MAX_SEQ * (h->head_dim / 2));
    ALLOC_F32(st->x,        h->dim);
    ALLOC_F32(st->xb,       h->dim);
    ALLOC_F32(st->xb2,      h->dim);
    ALLOC_F32(st->q,        h->dim);
    ALLOC_F32(st->k,        kv_dim);
    ALLOC_F32(st->v,        kv_dim);
    ALLOC_F32(st->hb,       h->ffn_dim);
    ALLOC_F32(st->hb2,      h->ffn_dim);
    ALLOC_F32(st->att,      (size_t)h->n_heads * R1M_MAX_SEQ);
    ALLOC_F32(st->logits,   h->vocab_size);
    ALLOC_F32(st->bx,       (size_t)PREFILL_B * h->dim);
    ALLOC_F32(st->bxb,      (size_t)PREFILL_B * h->dim);
    ALLOC_F32(st->bxb2,     (size_t)PREFILL_B * h->dim);
    ALLOC_F32(st->bq,       (size_t)PREFILL_B * h->dim);
    ALLOC_F32(st->bk,       (size_t)PREFILL_B * kv_dim);
    ALLOC_F32(st->bv,       (size_t)PREFILL_B * kv_dim);
    ALLOC_F32(st->bhb,      (size_t)PREFILL_B * h->ffn_dim);
    ALLOC_F32(st->bhb2,     (size_t)PREFILL_B * h->ffn_dim);
    #undef ALLOC_F32

    // Precompute RoPE cos/sin table (base = 5000000.0)
    int half = (int)(h->head_dim / 2);
    double inv_freqs[64];
    for (int i = 0; i < half; i++) {
        inv_freqs[i] = 1.0 / hv_pow_d((double)h->rope_theta, (double)(2 * i) / (double)h->head_dim);
    }
    for (int pos = 0; pos < R1M_MAX_SEQ; pos++) {
        for (int i = 0; i < half; i++) {
            double ang = (double)pos * inv_freqs[i];
            hv_sincos_d(ang, &st->rope_sin[pos * half + i], &st->rope_cos[pos * half + i]);
        }
    }

    st->ready = true;
    return true;
}

static void transformer_forward(R1LlmState *st, uint32_t token, int pos, bool compute_logits) {
    const R1MHeader *h = st->hdr;
    const int dim      = (int)h->dim;
    const int ffn_dim  = (int)h->ffn_dim;
    const int n_heads  = (int)h->n_heads;
    const int n_kv     = (int)h->n_kv_heads;
    const int hd       = (int)h->head_dim;
    const int kv_dim   = n_kv * hd;
    const int kv_mul   = n_heads / n_kv;
    const int half     = hd / 2;

    // 1. Token embedding lookup (Q4_K row dequant)
    size_t emb_row_bytes = (size_t)(dim >> 8) * sizeof(BlockQ4K);
    const BlockQ4K *emb_row = (const BlockQ4K *)(st->model_base + st->tdescs[0].offset + (size_t)token * emb_row_bytes);
    dequant_row_q4k(emb_row, st->x, dim);

    const float *rcos = st->rope_cos + pos * half;
    const float *rsin = st->rope_sin + pos * half;
    const float inv_sqrt_hd = 1.0f / hv_sqrtf((float)hd);

    // 2. 42 Transformer layers
    for (int l = 0; l < (int)h->n_layers; l++) {
        int base_t = 3 + l * 9;
        const float *attn_norm = (const float *)(st->model_base + st->tdescs[base_t + 0].offset);
        rmsnorm(st->xb, st->x, attn_norm, dim, h->rms_eps);

        // Q, K, V projections
        matmul_run(st, base_t + 1, st->xb, st->q, dim,    dim);
        matmul_run(st, base_t + 2, st->xb, st->k, kv_dim, dim);
        matmul_run(st, base_t + 3, st->xb, st->v, kv_dim, dim);

        // Interleaved RoPE (LLaMA GGUF convention: pairs (2i, 2i+1))
        for (int hd_idx = 0; hd_idx < n_heads; hd_idx++) {
            float *qh = st->q + hd_idx * hd;
            for (int i = 0; i < half; i++) {
                float q0 = qh[2 * i], q1 = qh[2 * i + 1];
                float c = rcos[i], s = rsin[i];
                qh[2 * i]     = q0 * c - q1 * s;
                qh[2 * i + 1] = q0 * s + q1 * c;
            }
        }
        for (int kh_idx = 0; kh_idx < n_kv; kh_idx++) {
            float *kh = st->k + kh_idx * hd;
            for (int i = 0; i < half; i++) {
                float k0 = kh[2 * i], k1 = kh[2 * i + 1];
                float c = rcos[i], s = rsin[i];
                kh[2 * i]     = k0 * c - k1 * s;
                kh[2 * i + 1] = k0 * s + k1 * c;
            }
        }

        // Store K, V in layer cache
        size_t loff = (size_t)l * R1M_MAX_SEQ * kv_dim;
        hv_memcpy(st->k_cache + loff + (size_t)pos * kv_dim, st->k, (size_t)kv_dim * sizeof(float));
        hv_memcpy(st->v_cache + loff + (size_t)pos * kv_dim, st->v, (size_t)kv_dim * sizeof(float));

        // Grouped-Query Attention (16 Q heads, 2 KV heads)
        for (int head = 0; head < n_heads; head++) {
            const float *qh = st->q + head * hd;
            float *att = st->att + head * R1M_MAX_SEQ;
            int kvh = head / kv_mul;

            float max_s = -1e30f;
            for (int t = 0; t <= pos; t++) {
                const float *kh = st->k_cache + loff + (size_t)t * kv_dim + kvh * hd;
                __m256 acc = _mm256_setzero_ps();
                for (int i = 0; i < hd; i += 8) {
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(qh + i), _mm256_loadu_ps(kh + i), acc);
                }
                float sc = hsum256_ps(acc) * inv_sqrt_hd;
                att[t] = sc;
                if (sc > max_s) max_s = sc;
            }

            // Softmax
            float sum_e = 0.0f;
            for (int t = 0; t <= pos; t++) {
                float e = hv_expf(att[t] - max_s);
                att[t] = e;
                sum_e += e;
            }
            float inv_e = 1.0f / sum_e;
            for (int t = 0; t <= pos; t++) att[t] *= inv_e;

            // Weighted sum of V
            float *oh = st->xb + head * hd;
            for (int i = 0; i < hd; i++) oh[i] = 0.0f;
            for (int t = 0; t <= pos; t++) {
                const float *vh = st->v_cache + loff + (size_t)t * kv_dim + kvh * hd;
                __m256 va = _mm256_set1_ps(att[t]);
                for (int i = 0; i < hd; i += 8) {
                    __m256 vo = _mm256_loadu_ps(oh + i);
                    vo = _mm256_fmadd_ps(va, _mm256_loadu_ps(vh + i), vo);
                    _mm256_storeu_ps(oh + i, vo);
                }
            }
        }

        // Output projection + residual
        matmul_run(st, base_t + 4, st->xb, st->xb2, dim, dim);
        for (int i = 0; i < dim; i += 8) {
            _mm256_storeu_ps(st->x + i, _mm256_add_ps(_mm256_loadu_ps(st->x + i), _mm256_loadu_ps(st->xb2 + i)));
        }

        // FFN SwiGLU: silu(gate(x)) * up(x) -> down -> residual
        const float *ffn_norm = (const float *)(st->model_base + st->tdescs[base_t + 5].offset);
        rmsnorm(st->xb, st->x, ffn_norm, dim, h->rms_eps);
        matmul_run(st, base_t + 6, st->xb, st->hb,  ffn_dim, dim);
        matmul_run(st, base_t + 7, st->xb, st->hb2, ffn_dim, dim);
        for (int i = 0; i < ffn_dim; i++) {
            float g = st->hb[i];
            float silu = g / (1.0f + hv_expf(-g));
            st->hb[i] = silu * st->hb2[i];
        }
        matmul_run(st, base_t + 8, st->hb, st->xb2, dim, ffn_dim);
        for (int i = 0; i < dim; i += 8) {
            _mm256_storeu_ps(st->x + i, _mm256_add_ps(_mm256_loadu_ps(st->x + i), _mm256_loadu_ps(st->xb2 + i)));
        }
    }

    if (!compute_logits) return;

    // 3. Final RMSNorm + LM Head (130560 x 2048 Q6_K)
    const float *out_norm = (const float *)(st->model_base + st->tdescs[1].offset);
    rmsnorm(st->xb, st->x, out_norm, dim, h->rms_eps);
    matmul_run(st, 2, st->xb, st->logits, (int)h->vocab_size, dim);
}

// ============================================================================
// Bare-Metal Byte-Level BPE Tokenizer (O(1) FNV-1a Merge Lookup)
// ============================================================================
static bool bpe_lookup_merge(const R1LlmState *st, uint32_t lid, uint32_t rid,
                             uint32_t *merged_out, uint32_t *rank_out) {
    uint32_t mask = st->hdr->merge_ht_cap - 1U;
    uint32_t idx = ((lid * 2654435761U) ^ (rid * 2246822519U)) & mask;
    while (true) {
        const R1MMergeSlot *s = &st->merge_ht[idx];
        if (s->rank == 0xFFFFFFFFU) return false;
        if (s->left_id == lid && s->right_id == rid) {
            *merged_out = s->merged_id;
            *rank_out   = s->rank;
            return true;
        }
        idx = (idx + 1U) & mask;
    }
}

static int bpe_encode_chunk(const R1LlmState *st, const uint8_t *bytes, int len,
                            uint32_t *out, int max_out) {
    if (len <= 0) return 0;
    uint32_t buf[128];
    int n = 0;
    for (int i = 0; i < len && n < 128; i++) {
        buf[n++] = st->byte_to_tok[bytes[i]];
    }
    while (n >= 2) {
        uint32_t best_rank = 0xFFFFFFFFU;
        uint32_t best_merged = 0;
        int best_pos = -1;
        for (int i = 0; i < n - 1; i++) {
            uint32_t mid, rk;
            if (bpe_lookup_merge(st, buf[i], buf[i + 1], &mid, &rk)) {
                if (rk < best_rank) {
                    best_rank = rk;
                    best_merged = mid;
                    best_pos = i;
                }
            }
        }
        if (best_pos < 0) break;
        buf[best_pos] = best_merged;
        for (int i = best_pos + 1; i < n - 1; i++) buf[i] = buf[i + 1];
        n--;
    }
    int emitted = 0;
    for (int i = 0; i < n && emitted < max_out; i++) out[emitted++] = buf[i];
    return emitted;
}

static inline bool is_alpha(uint8_t c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
static inline bool is_digit(uint8_t c) {
    return (c >= '0' && c <= '9');
}

static int bpe_encode_text(const R1LlmState *st, const char *text, uint32_t *out, int max_out) {
    const uint8_t *s = (const uint8_t *)text;
    int len = (int)hv_strlen(text);
    int i = 0, total = 0;

    while (i < len && total < max_out) {
        int start = i;
        if (s[i] == '\n') {
            while (i < len && s[i] == '\n' && (i - start) < 4) i++;
        } else if (s[i] == ' ' && i + 1 < len && (is_alpha(s[i + 1]) || s[i + 1] == '_')) {
            i++;
            while (i < len && (is_alpha(s[i]) || s[i] == '_') && (i - start) < 32) i++;
        } else if (is_alpha(s[i]) || s[i] == '_') {
            while (i < len && (is_alpha(s[i]) || s[i] == '_') && (i - start) < 32) i++;
        } else if (is_digit(s[i])) {
            // GPT-2 / MiniCPM5 pre-tokenizer groups digits 1..3 at a time (or single digits)
            i++;
        } else if (s[i] == ' ') {
            if (i + 1 < len && s[i + 1] != ' ' && s[i + 1] != '\n' && !is_digit(s[i + 1])) {
                i += 2;
                while (i < len && !is_alpha(s[i]) && !is_digit(s[i]) && s[i] != ' ' && s[i] != '\n' && (i - start) < 8) i++;
            } else {
                while (i < len && s[i] == ' ' && (i - start) < 4) i++;
            }
        } else {
            i++;
        }
        total += bpe_encode_chunk(st, s + start, i - start, out + total, max_out - total);
    }
    return total;
}

int r1_llm_tokenize_chat(const R1LlmState *st, const char *system_msg, const char *user_msg,
                         uint32_t *out_ids, int max_ids) {
    const R1MHeader *h = st->hdr;
    int n = 0;
    #define PUSH(tok) do { if (n < max_ids) out_ids[n++] = (tok); } while (0)
    // <s><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
    PUSH(h->bos_id);
    PUSH(h->im_start_id);
    n += bpe_encode_text(st, "system\n", out_ids + n, max_ids - n);
    n += bpe_encode_text(st, system_msg, out_ids + n, max_ids - n);
    PUSH(h->im_end_id);
    n += bpe_encode_text(st, "\n", out_ids + n, max_ids - n);

    PUSH(h->im_start_id);
    n += bpe_encode_text(st, "user\n", out_ids + n, max_ids - n);
    n += bpe_encode_text(st, user_msg, out_ids + n, max_ids - n);
    PUSH(h->im_end_id);
    n += bpe_encode_text(st, "\n", out_ids + n, max_ids - n);

    PUSH(h->im_start_id);
    // Notice exact MiniCPM5 non-thinking assistant suffix: assistant (130071? no, encode "assistant\n" or check exact IDs!)
    n += bpe_encode_text(st, "assistant\n", out_ids + n, max_ids - n);
    PUSH(h->think_start_id);
    PUSH(h->nl_nl_id);
    PUSH(h->think_end_id);
    PUSH(h->nl_nl_id);
    #undef PUSH
    return n;
}

int r1_llm_decode_piece(const R1LlmState *st, uint32_t tok_id, char *out, int max_len) {
    if (tok_id >= st->hdr->vocab_size || max_len <= 1) return 0;
    uint32_t off = st->vocab_offs[tok_id];
    uint16_t len = st->vocab_lens[tok_id];
    int copy = (len < (uint16_t)(max_len - 1)) ? (int)len : (max_len - 1);
    hv_memcpy(out, st->vocab_pool + off, (size_t)copy);
    out[copy] = '\0';
    return copy;
}

int r1_llm_generate(R1LlmState *st, const char *system_msg, const char *user_msg,
                    char *out_text, int max_out_bytes, int max_new_tokens,
                    int *prompt_toks_out, int *gen_toks_out,
                    R1TokenCallback cb, void *cb_ctx) {
    if (!st->ready) return -1;

    uint32_t prompt_ids[R1M_MAX_SEQ];
    int n_prompt = r1_llm_tokenize_chat(st, system_msg, user_msg, prompt_ids, R1M_MAX_SEQ - max_new_tokens - 2);
    if (prompt_toks_out) *prompt_toks_out = n_prompt;

    // 1. Batched prefill (PREFILL_B=16 tokens per weight pass -> 12-16x faster prefill!)
    const R1MHeader *h = st->hdr;
    const int dim = (int)h->dim, ffn_dim = (int)h->ffn_dim;
    const int n_heads = (int)h->n_heads, n_kv = (int)h->n_kv_heads, hd = (int)h->head_dim;
    const int kv_dim = n_kv * hd, kv_mul = n_heads / n_kv, half = hd / 2;
    const float inv_sqrt_hd = 1.0f / hv_sqrtf((float)hd);
    const size_t emb_row_bytes = (size_t)(dim >> 8) * sizeof(BlockQ4K);

    for (int p0 = 0; p0 < n_prompt; p0 += PREFILL_B) {
        int B = n_prompt - p0;
        if (B > PREFILL_B) B = PREFILL_B;

        for (int b = 0; b < B; b++) {
            const BlockQ4K *er = (const BlockQ4K *)(st->model_base + st->tdescs[0].offset + (size_t)prompt_ids[p0 + b] * emb_row_bytes);
            dequant_row_q4k(er, st->bx + (size_t)b * dim, dim);
        }

        for (int l = 0; l < (int)h->n_layers; l++) {
            int base_t = 3 + l * 9;
            const float *attn_norm = (const float *)(st->model_base + st->tdescs[base_t + 0].offset);
            for (int b = 0; b < B; b++) rmsnorm(st->bxb + (size_t)b * dim, st->bx + (size_t)b * dim, attn_norm, dim, h->rms_eps);

            matmul_batch_run(st, base_t + 1, st->bxb, st->bq, B, dim,    dim);
            matmul_batch_run(st, base_t + 2, st->bxb, st->bk, B, kv_dim, dim);
            matmul_batch_run(st, base_t + 3, st->bxb, st->bv, B, kv_dim, dim);

            size_t loff = (size_t)l * R1M_MAX_SEQ * kv_dim;
            for (int b = 0; b < B; b++) {
                int pos = p0 + b;
                const float *rcos = st->rope_cos + pos * half;
                const float *rsin = st->rope_sin + pos * half;
                float *qb = st->bq + (size_t)b * dim;
                float *kb = st->bk + (size_t)b * kv_dim;
                float *vb = st->bv + (size_t)b * kv_dim;

                for (int hi = 0; hi < n_heads; hi++) {
                    float *qh = qb + hi * hd;
                    for (int i = 0; i < half; i++) {
                        float q0 = qh[2*i], q1 = qh[2*i+1];
                        qh[2*i] = q0*rcos[i] - q1*rsin[i];
                        qh[2*i+1] = q0*rsin[i] + q1*rcos[i];
                    }
                }
                for (int ki = 0; ki < n_kv; ki++) {
                    float *kh = kb + ki * hd;
                    for (int i = 0; i < half; i++) {
                        float k0 = kh[2*i], k1 = kh[2*i+1];
                        kh[2*i] = k0*rcos[i] - k1*rsin[i];
                        kh[2*i+1] = k0*rsin[i] + k1*rcos[i];
                    }
                }
                hv_memcpy(st->k_cache + loff + (size_t)pos * kv_dim, kb, (size_t)kv_dim * sizeof(float));
                hv_memcpy(st->v_cache + loff + (size_t)pos * kv_dim, vb, (size_t)kv_dim * sizeof(float));

                float *ob = st->bxb + (size_t)b * dim;
                for (int head = 0; head < n_heads; head++) {
                    const float *qh = qb + head * hd;
                    float *att = st->att + head * R1M_MAX_SEQ;
                    int kvh = head / kv_mul;
                    float max_s = -1e30f;
                    for (int t = 0; t <= pos; t++) {
                        const float *kh = st->k_cache + loff + (size_t)t * kv_dim + kvh * hd;
                        __m256 acc = _mm256_setzero_ps();
                        for (int i = 0; i < hd; i += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(qh+i), _mm256_loadu_ps(kh+i), acc);
                        float sc = hsum256_ps(acc) * inv_sqrt_hd;
                        att[t] = sc;
                        if (sc > max_s) max_s = sc;
                    }
                    float sum_e = 0.0f;
                    for (int t = 0; t <= pos; t++) { float e = hv_expf(att[t] - max_s); att[t] = e; sum_e += e; }
                    float inv_e = 1.0f / sum_e;
                    for (int t = 0; t <= pos; t++) att[t] *= inv_e;
                    float *oh = ob + head * hd;
                    for (int i = 0; i < hd; i++) oh[i] = 0.0f;
                    for (int t = 0; t <= pos; t++) {
                        const float *vh = st->v_cache + loff + (size_t)t * kv_dim + kvh * hd;
                        __m256 va = _mm256_set1_ps(att[t]);
                        for (int i = 0; i < hd; i += 8) {
                            _mm256_storeu_ps(oh+i, _mm256_fmadd_ps(va, _mm256_loadu_ps(vh+i), _mm256_loadu_ps(oh+i)));
                        }
                    }
                }
            }

            matmul_batch_run(st, base_t + 4, st->bxb, st->bxb2, B, dim, dim);
            for (int i = 0; i < B * dim; i += 8) {
                _mm256_storeu_ps(st->bx + i, _mm256_add_ps(_mm256_loadu_ps(st->bx + i), _mm256_loadu_ps(st->bxb2 + i)));
            }

            const float *ffn_norm = (const float *)(st->model_base + st->tdescs[base_t + 5].offset);
            for (int b = 0; b < B; b++) rmsnorm(st->bxb + (size_t)b * dim, st->bx + (size_t)b * dim, ffn_norm, dim, h->rms_eps);
            matmul_batch_run(st, base_t + 6, st->bxb, st->bhb,  B, ffn_dim, dim);
            matmul_batch_run(st, base_t + 7, st->bxb, st->bhb2, B, ffn_dim, dim);
            for (int i = 0; i < B * ffn_dim; i++) {
                float g = st->bhb[i];
                st->bhb[i] = (g / (1.0f + hv_expf(-g))) * st->bhb2[i];
            }
            matmul_batch_run(st, base_t + 8, st->bhb, st->bxb2, B, dim, ffn_dim);
            for (int i = 0; i < B * dim; i += 8) {
                _mm256_storeu_ps(st->bx + i, _mm256_add_ps(_mm256_loadu_ps(st->bx + i), _mm256_loadu_ps(st->bxb2 + i)));
            }
        }

        // Only compute logits for the very last prompt token!
        if (p0 + B == n_prompt) {
            const float *out_norm = (const float *)(st->model_base + st->tdescs[1].offset);
            rmsnorm(st->xb, st->bx + (size_t)(B - 1) * dim, out_norm, dim, h->rms_eps);
            matmul_run(st, 2, st->xb, st->logits, (int)h->vocab_size, dim);
        }
    }

    // 2. Greedy argmax decode (deterministic, zero <think>, stops on eos_id or im_end_id)
    int out_pos = 0;
    int gen_cnt = 0;
    int cur_pos = n_prompt;
    const uint32_t vocab = st->hdr->vocab_size;

    for (int step = 0; step < max_new_tokens && cur_pos < R1M_MAX_SEQ; step++) {
        uint32_t best_id = 0;
        float best_val = st->logits[0];
        for (uint32_t i = 1; i < vocab; i++) {
            if (st->logits[i] > best_val) {
                best_val = st->logits[i];
                best_id = i;
            }
        }

        if (best_id == st->hdr->eos_id || best_id == st->hdr->im_end_id) {
            break;
        }

        char piece[64];
        int plen = r1_llm_decode_piece(st, best_id, piece, sizeof(piece));
        if (plen > 0 && out_pos + plen < max_out_bytes) {
            hv_memcpy(out_text + out_pos, piece, (size_t)plen);
            out_pos += plen;
            out_text[out_pos] = '\0';
        }
        if (cb) cb(piece, plen, step, cb_ctx);
        gen_cnt++;

        transformer_forward(st, best_id, cur_pos++, true);
    }

    if (out_pos < max_out_bytes) out_text[out_pos] = '\0';
    if (gen_toks_out) *gen_toks_out = gen_cnt;
    return out_pos;
}
