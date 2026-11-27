#include "smp.h"

#define LAPIC_BASE    0xFEE00000ULL
#define LAPIC_ICR_LO  0x300U
#define LAPIC_ICR_HI  0x310U

#define AP_TRAMP_PA   0x00008000ULL
#define AP_CTRL_PA    0x00008F00ULL
// AP stacks live inside the Ring -1 isolated 64 MiB arena (0x70000000..0x73FFFFFF):
// Top 2 MiB (0x73E00000..0x73FFFFFF) = 32 x 64 KiB AP stacks!
#define AP_STACK_BASE 0x73E00000ULL
#define AP_STACK_SIZE 0x00010000ULL

extern uint64_t host_pml4[512];
extern uint8_t  gdt64_ptr[10];
extern uint8_t  ap_trampoline_start[];
extern uint8_t  ap_trampoline_end[];

static volatile uint64_t g_ap_online_mask = 1ULL;
static volatile uint64_t g_job_gen = 0;
static volatile uint32_t g_job_done = 0;
static R1ParallelFn      g_job_fn = NULL;
static void             *g_job_ctx = NULL;
static int               g_num_cpus = 1;

static inline void lapic_write(uint32_t reg, uint32_t val) {
    *(volatile uint32_t *)(uintptr_t)(LAPIC_BASE + reg) = val;
}
static inline uint32_t lapic_read(uint32_t reg) {
    return *(const volatile uint32_t *)(uintptr_t)(LAPIC_BASE + reg);
}

static void busy_delay_tsc(uint64_t ticks) {
    uint64_t t0 = read_tsc();
    while ((read_tsc() - t0) < ticks) cpu_pause();
}

void ap_c_entry(uint64_t tid) {
    enable_avx2_cpu();
    __sync_or_and_fetch(&g_ap_online_mask, (1ULL << tid));

    uint64_t my_gen = 0;
    while (1) {
        uint64_t cur = __atomic_load_n(&g_job_gen, __ATOMIC_ACQUIRE);
        if (cur == my_gen) {
            cpu_pause();
            continue;
        }
        my_gen = cur;
        g_job_fn((int)tid, g_num_cpus, g_job_ctx);
        __atomic_add_fetch(&g_job_done, 1, __ATOMIC_RELEASE);
    }
}

int r1_smp_init(int max_cpus) {
    if (max_cpus > R1_MAX_CPUS) max_cpus = R1_MAX_CPUS;
    if (max_cpus < 1) max_cpus = 1;

    // Detect logical CPU count via CPUID 0x01 EBX[23:16] / 0x80000008 ECX[7:0]+1
    uint32_t a, b, c, d;
    cpuid_raw(0x01U, 0, &a, &b, &c, &d);
    int n_det = (int)((b >> 16) & 0xFFU);
    if (n_det < 1) n_det = 1;
    if (n_det > max_cpus) n_det = max_cpus;

    if (n_det <= 1) {
        g_num_cpus = 1;
        return 1;
    }

    // Copy 16-bit AP trampoline to 0x8000
    size_t tlen = (size_t)(ap_trampoline_end - ap_trampoline_start);
    hv_memcpy((void *)(uintptr_t)AP_TRAMP_PA, ap_trampoline_start, tlen);

    // Fill shared control block fields at 0x8F00
    volatile uint64_t *ctrl = (volatile uint64_t *)(uintptr_t)AP_CTRL_PA;
    ctrl[0] = (uint64_t)(uintptr_t)host_pml4;
    ctrl[1] = (uint64_t)(uintptr_t)&ap_c_entry;
    hv_memcpy((void *)(uintptr_t)(AP_CTRL_PA + 0x20), gdt64_ptr, 10);

    int online = 1;
    for (int ap = 1; ap < n_det; ap++) {
        ctrl[2] = AP_STACK_BASE + (uint64_t)(ap + 1) * AP_STACK_SIZE - 16ULL;
        ctrl[3] = (uint64_t)ap;
        __sync_synchronize();

        // INIT IPI
        lapic_write(LAPIC_ICR_HI, (uint32_t)ap << 24);
        lapic_write(LAPIC_ICR_LO, 0x00004500U); // INIT, level assert
        while (lapic_read(LAPIC_ICR_LO) & (1U << 12)) cpu_pause();
        busy_delay_tsc(5000000ULL); // ~2ms

        // SIPI vector 0x08 -> 0x00008000
        lapic_write(LAPIC_ICR_HI, (uint32_t)ap << 24);
        lapic_write(LAPIC_ICR_LO, 0x00004608U); // STARTUP, vector 0x08
        while (lapic_read(LAPIC_ICR_LO) & (1U << 12)) cpu_pause();

        // Wait up to ~50ms for AP to check in
        uint64_t t0 = read_tsc();
        bool booted = false;
        while ((read_tsc() - t0) < 120000000ULL) {
            if (__atomic_load_n(&g_ap_online_mask, __ATOMIC_ACQUIRE) & (1ULL << ap)) {
                booted = true;
                break;
            }
            cpu_pause();
        }
        if (booted) {
            online++;
        } else {
            // Second SIPI retry
            lapic_write(LAPIC_ICR_HI, (uint32_t)ap << 24);
            lapic_write(LAPIC_ICR_LO, 0x00004608U);
            uint64_t t1 = read_tsc();
            while ((read_tsc() - t1) < 120000000ULL) {
                if (__atomic_load_n(&g_ap_online_mask, __ATOMIC_ACQUIRE) & (1ULL << ap)) {
                    booted = true;
                    break;
                }
                cpu_pause();
            }
            if (booted) online++;
            else break;
        }
    }

    // Wipe trampoline low memory before Linux boots
    hv_memset((void *)(uintptr_t)AP_TRAMP_PA, 0, 4096);
    g_num_cpus = online;
    kprintf("[SMP] Bare-metal Ring -1 worker pool online: %d / %d AVX2 cores (APs hidden from Linux nr_cpus=1)\n",
            online, n_det);
    return online;
}

int r1_smp_num_cpus(void) {
    return g_num_cpus;
}

void r1_smp_dispatch(R1ParallelFn fn, void *ctx) {
    if (g_num_cpus <= 1) {
        fn(0, 1, ctx);
        return;
    }
    g_job_fn  = fn;
    g_job_ctx = ctx;
    __atomic_store_n(&g_job_done, 0, __ATOMIC_RELEASE);
    __atomic_add_fetch(&g_job_gen, 1, __ATOMIC_RELEASE);

    fn(0, g_num_cpus, ctx);

    while (__atomic_load_n(&g_job_done, __ATOMIC_ACQUIRE) < (uint32_t)(g_num_cpus - 1)) {
        cpu_pause();
    }
}
