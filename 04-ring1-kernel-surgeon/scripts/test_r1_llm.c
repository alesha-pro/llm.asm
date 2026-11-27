#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>

#include "../hv/llm_engine.h"

void *hv_memset(void *d, int c, size_t n) { return memset(d, c, n); }
void *hv_memcpy(void *d, const void *s, size_t n) { return memcpy(d, s, n); }
size_t hv_strlen(const char *s) { return strlen(s); }
int hv_strncmp(const char *a, const char *b, size_t n) { return strncmp(a, b, n); }

#define NTHREADS 16

static _Atomic uint64_t g_gen = 0;
static _Atomic int g_done = 0;
static _Atomic int g_stop = 0;
static R1ParallelFn g_fn = NULL;
static void *g_ctx = NULL;
static pthread_t g_threads[NTHREADS];

static void *worker_loop(void *arg) {
    int tid = (int)(intptr_t)arg;
    uint64_t my_gen = 0;
    while (!atomic_load_explicit(&g_stop, memory_order_acquire)) {
        uint64_t cur = atomic_load_explicit(&g_gen, memory_order_acquire);
        if (cur == my_gen) {
            __asm__ volatile ("pause");
            continue;
        }
        my_gen = cur;
        if (atomic_load_explicit(&g_stop, memory_order_acquire)) break;
        g_fn(tid, NTHREADS, g_ctx);
        atomic_fetch_add_explicit(&g_done, 1, memory_order_release);
    }
    return NULL;
}

static void pool_init(void) {
    for (int i = 1; i < NTHREADS; i++) {
        pthread_create(&g_threads[i], NULL, worker_loop, (void *)(intptr_t)i);
    }
}

static void pool_shutdown(void) {
    atomic_store_explicit(&g_stop, 1, memory_order_release);
    atomic_fetch_add_explicit(&g_gen, 1, memory_order_release);
    for (int i = 1; i < NTHREADS; i++) pthread_join(g_threads[i], NULL);
}

static void host_parallel_dispatch(R1ParallelFn fn, void *ctx) {
    g_fn = fn;
    g_ctx = ctx;
    atomic_store_explicit(&g_done, 0, memory_order_release);
    atomic_fetch_add_explicit(&g_gen, 1, memory_order_release);
    fn(0, NTHREADS, ctx);
    while (atomic_load_explicit(&g_done, memory_order_acquire) < NTHREADS - 1) {
        __asm__ volatile ("pause");
    }
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : getenv("MODEL_R1M");
    if (!path) path = "models/minicpm5-2b-surgeon.r1m";
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open r1m"); return 1; }
    struct stat st;
    fstat(fd, &st);
    void *base = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }

    pool_init();
    void *arena = aligned_alloc(64, 64 * 1024 * 1024);
    R1LlmState llm;
    if (!r1_llm_init(&llm, base, arena, host_parallel_dispatch)) {
        fprintf(stderr, "r1_llm_init FAILED\n");
        return 1;
    }
    printf("[HOST TEST] R1M initialized: dim=%u layers=%u vocab=%u\n",
           llm.hdr->dim, llm.hdr->n_layers, llm.hdr->vocab_size);
    fflush(stdout);

    const char *sys_msg = "Ring -1 Kernel Surgeon. Output 3 lines only: DIAGNOSIS, ACTION, PATCH.";
    const char *prompts[4] = {
        "[RING-1 TELEMETRY]\n"
        "EXC: #PF (vec=14 err=0x0002 cpl=0)\n"
        "CR2: 0x0000000000000000\n"
        "RIP: surgeon_crash+0x0180 (len=11)\n"
        "BYTES: c7 04 25 00 00 00 00 ef be ad de\n"
        "DISASM: mov dword ptr [0x0], 0xdeadbeef\n"
        "REGS: RAX=0x0 RBX=0x3 RCX=0x0 RDX=0x0 RSI=0x0 RDI=0xffff888004100000 RSP=0xffffc90000013e20 RFLAGS=0x246\n"
        "CONTRACT: Optional telemetry store to unmapped pointer before commit; skipping store preserves out=0x100 and returns 0.\n"
        "CTX: extable=0 bug_warn=0 lock_held=0 irq_ctx=0 side_effect=0 stack_ok=1 obs_complete=1\n"
        "LOG_TAIL:\n  surgeon_crash: stage=pre_commit opt_ptr=NULL\n"
        "Output ONLY 3 lines (no <think>, no markdown):\n"
        "DIAGNOSIS: <one short sentence>\n"
        "ACTION: <PATCH_BYTES | PASS_TO_LINUX | HALT_FOR_DUMP>\n"
        "PATCH: <exact-length space-separated hex bytes or NONE>",

        "[RING-1 TELEMETRY]\n"
        "EXC: #DE (vec=0 err=0x0000 cpl=0)\n"
        "CR2: 0x0000000000000000\n"
        "RIP: surgeon_crash+0x0210 (len=2)\n"
        "BYTES: f7 f1\n"
        "DISASM: div ecx\n"
        "REGS: RAX=0x2a RCX=0x0 RDX=0x0 RDI=0xffff888004100000 RSP=0xffffc90000013cc0 RFLAGS=0x246\n"
        "CONTRACT: Integer division quotient in RAX/EAX is stored to out_val and returned; numerator=0x2a, divisor=0, safe fallback requires quotient=0 (out=0, ret=0).\n"
        "CTX: extable=0 bug_warn=0 lock_held=0 irq_ctx=0 side_effect=0 stack_ok=1 obs_complete=1\n"
        "LOG_TAIL:\n  surgeon_crash: denom=0 fallback_quotient=0\n"
        "Output ONLY 3 lines (no <think>, no markdown):\n"
        "DIAGNOSIS: <one short sentence>\n"
        "ACTION: <PATCH_BYTES | PASS_TO_LINUX | HALT_FOR_DUMP>\n"
        "PATCH: <exact-length space-separated hex bytes or NONE>",

        "[RING-1 TELEMETRY]\n"
        "EXC: #UD (vec=6 err=0x0000 cpl=0)\n"
        "CR2: 0x0000000000000000\n"
        "RIP: surgeon_crash+0x01ab (len=2)\n"
        "BYTES: 0f 0b\n"
        "DISASM: ud2\n"
        "REGS: RAX=0x0 RBX=0x0 RCX=0x0 RDX=0x0 RDI=0xffff888004100000 RSP=0xffffc90000013c80 RFLAGS=0x246\n"
        "CONTRACT: Non-fatal diagnostic trap with EAX=0 and out_val=0x42 already prepared; skipping trap completes cleanly with ret=0.\n"
        "CTX: extable=0 bug_warn=0 lock_held=0 irq_ctx=0 side_effect=0 stack_ok=1 obs_complete=1\n"
        "LOG_TAIL:\n  surgeon_crash: diag_trap state=clean pre_commit=1\n"
        "Output ONLY 3 lines (no <think>, no markdown):\n"
        "DIAGNOSIS: <one short sentence>\n"
        "ACTION: <PATCH_BYTES | PASS_TO_LINUX | HALT_FOR_DUMP>\n"
        "PATCH: <exact-length space-separated hex bytes or NONE>",

        "[RING-1 TELEMETRY]\n"
        "EXC: #PF (vec=14 err=0x0002 cpl=0)\n"
        "CR2: 0x0000000000000000\n"
        "RIP: surgeon_crash+0x0300 (len=11)\n"
        "BYTES: c7 04 25 00 00 00 00 ef be ad de\n"
        "DISASM: mov dword ptr [0x0], 0xdeadbeef\n"
        "REGS: RAX=0x0 RBX=0x3 RCX=0x0 RSP=0xffffc90000013900 RFLAGS=0x246\n"
        "CONTRACT: Ring 0 NULL write while holding spinlock (lock_held=1).\n"
        "CTX: extable=0 bug_warn=0 lock_held=1 irq_ctx=0 side_effect=0 stack_ok=1 obs_complete=1\n"
        "LOG_TAIL:\n  surgeon_crash: spin_lock held; hazard lock=1\n"
        "Output ONLY 3 lines (no <think>, no markdown):\n"
        "DIAGNOSIS: <one short sentence>\n"
        "ACTION: <PATCH_BYTES | PASS_TO_LINUX | HALT_FOR_DUMP>\n"
        "PATCH: <exact-length space-separated hex bytes or NONE>"
    };
    const char *names[4] = {"#PF NULL-deref (11B)", "#DE div ecx (2B -> xor eax,eax)", "#UD ud2 (2B)", "CONTROL HALT (lock_held=1)"};

    for (int t = 0; t < 4; t++) {
        char out[512] = {0};
        int ptok = 0, gtok = 0;
        double t0 = now_s();
        r1_llm_generate(&llm, sys_msg, prompts[t], out, sizeof(out), 64, &ptok, &gtok, NULL, NULL);
        double dt = now_s() - t0;
        printf("\n=== TEST %d: %s (prompt=%d tok, gen=%d tok, total=%.2fs, %.1f tok/s) ===\n%s\n",
               t + 1, names[t], ptok, gtok, dt, (ptok + gtok) / dt, out);
        fflush(stdout);
    }
    pool_shutdown();
    return 0;
}
