#pragma once
#include "types.h"

#define R1M_MAX_SEQ   512
#define R1M_DTYPE_F32  0U
#define R1M_DTYPE_Q4K  1U
#define R1M_DTYPE_Q6K  2U

typedef struct __attribute__((packed)) {
    char     magic[8];          // "R1SURG01"
    uint32_t dim;
    uint32_t n_layers;
    uint32_t ffn_dim;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t vocab_size;
    uint32_t n_tensors;
    float    rms_eps;
    float    rope_theta;
    uint32_t bos_id;
    uint32_t eos_id;
    uint32_t im_start_id;
    uint32_t im_end_id;
    uint32_t think_start_id;
    uint32_t think_end_id;
    uint32_t nl_nl_id;
    uint32_t merge_ht_cap;
    uint64_t tdesc_off;
    uint64_t b2t_off;
    uint64_t voffs_off;
    uint64_t vlens_off;
    uint64_t vpool_off;
    uint64_t vpool_size;
    uint64_t merge_ht_off;
    uint64_t total_used;
} R1MHeader;

typedef struct __attribute__((packed)) {
    uint64_t offset;
    uint32_t dtype;
    uint32_t nbytes;
} R1MTensorDesc;

typedef struct __attribute__((packed)) {
    uint32_t left_id;
    uint32_t right_id;
    uint32_t merged_id;
    uint32_t rank;
} R1MMergeSlot;

typedef void (*R1ParallelFn)(int tid, int nthreads, void *ctx);
typedef void (*R1ParallelDispatch)(R1ParallelFn fn, void *ctx);
typedef void (*R1TokenCallback)(const char *piece, int piece_len, int step, void *user_ctx);

typedef struct {
    const uint8_t      *model_base;
    const R1MHeader    *hdr;
    const R1MTensorDesc *tdescs;
    const uint32_t     *byte_to_tok;
    const uint32_t     *vocab_offs;
    const uint16_t     *vocab_lens;
    const uint8_t      *vocab_pool;
    const R1MMergeSlot *merge_ht;

    // 64 MiB runtime arena (KV cache + scratch + logits + RoPE tables)
    float *k_cache;       // [n_layers * R1M_MAX_SEQ * kv_dim]
    float *v_cache;       // [n_layers * R1M_MAX_SEQ * kv_dim]
    float *rope_cos;      // [R1M_MAX_SEQ * (head_dim/2)]
    float *rope_sin;      // [R1M_MAX_SEQ * (head_dim/2)]
    float *x;             // [dim]
    float *xb;            // [dim]
    float *xb2;           // [dim]
    float *q;             // [dim]
    float *k;             // [kv_dim]
    float *v;             // [kv_dim]
    float *hb;            // [ffn_dim]
    float *hb2;           // [ffn_dim]
    float *att;           // [n_heads * R1M_MAX_SEQ]
    float *logits;        // [vocab_size]
    float *bx, *bxb, *bxb2, *bq, *bk, *bv, *bhb, *bhb2;

    R1ParallelDispatch dispatch;
    bool ready;
} R1LlmState;

bool r1_llm_init(R1LlmState *st, const void *model_base, void *arena_64mb, R1ParallelDispatch dispatch);
int  r1_llm_tokenize_chat(const R1LlmState *st, const char *system_msg, const char *user_msg,
                          uint32_t *out_ids, int max_ids);
int  r1_llm_decode_piece(const R1LlmState *st, uint32_t tok_id, char *out, int max_len);
int  r1_llm_generate(R1LlmState *st, const char *system_msg, const char *user_msg,
                     char *out_text, int max_out_bytes, int max_new_tokens,
                     int *prompt_toks_out, int *gen_toks_out,
                     R1TokenCallback cb, void *cb_ctx);
