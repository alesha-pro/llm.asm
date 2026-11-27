#pragma once
#include "types.h"
#include "llm_engine.h"

#define R1_MAX_CPUS 32

int  r1_smp_init(int max_cpus);
int  r1_smp_num_cpus(void);
void r1_smp_dispatch(R1ParallelFn fn, void *ctx);

extern void enable_avx2_cpu(void);
extern void xsave_state(void *buf_64aligned);
extern void xrstor_state(const void *buf_64aligned);
extern uint64_t get_xcr0(void);
extern void set_xcr0(uint64_t v);
