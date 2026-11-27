#include "svm.h"

#define HV_ARENA_START   0x00200000ULL
#define HV_ARENA_END     0x00400000ULL
#define LLM_ARENA_START  0x70000000ULL
#define LLM_ARENA_END    0x74000000ULL

#define PT_P   (1ULL << 0)
#define PT_W   (1ULL << 1)
#define PT_U   (1ULL << 2)
#define PT_PS  (1ULL << 7)

extern uint64_t host_pml4[512];
extern uint64_t host_pdpt[512];
extern uint64_t host_pd[8][512];
extern uint64_t host_pd_high[512];

static uint64_t npt_pml4[512]    __attribute__((aligned(4096)));
static uint64_t npt_pdpt[512]    __attribute__((aligned(4096)));
static uint64_t npt_pd[4][512]   __attribute__((aligned(4096)));
static uint64_t npt_pd_high[512] __attribute__((aligned(4096)));

void host_ensure_mapped(uint64_t phys_addr, uint64_t size) {
    if (phys_addr + size <= 0x200000000ULL) return; // boot.asm maps 0..8 GiB!
    uint64_t gb_idx = (phys_addr >> 30) & 0x1FFULL;
    uint64_t base_1g = phys_addr & ~0x3FFFFFFFULL;
    for (uint64_t i = 0; i < 512; i++) {
        host_pd_high[i] = (base_1g + (i << 21)) | PT_P | PT_W | PT_PS;
    }
    host_pdpt[gb_idx] = (uint64_t)(uintptr_t)host_pd_high | PT_P | PT_W;
    flush_host_tlb();
}

uint64_t npt_init(uint64_t fb_addr, uint64_t fb_size) {
    hv_memset(npt_pml4, 0, sizeof(npt_pml4));
    hv_memset(npt_pdpt, 0, sizeof(npt_pdpt));
    hv_memset(npt_pd, 0, sizeof(npt_pd));
    hv_memset(npt_pd_high, 0, sizeof(npt_pd_high));

    npt_pml4[0] = (uint64_t)(uintptr_t)npt_pdpt | PT_P | PT_W | PT_U;

    // Map ONLY 0..4 GiB to guest (4..6 GiB NVDIMM model weights stay UNMAPPED in NPT!)
    for (uint64_t g = 0; g < 4; g++) {
        npt_pdpt[g] = (uint64_t)(uintptr_t)npt_pd[g] | PT_P | PT_W | PT_U;
        for (uint64_t i = 0; i < 512; i++) {
            uint64_t pa = (g << 30) | (i << 21);
            if ((pa >= HV_ARENA_START && pa < HV_ARENA_END) ||
                (pa >= LLM_ARENA_START && pa < LLM_ARENA_END)) {
                npt_pd[g][i] = 0; // Isolated Ring -1 memory: UNMAPPED from Linux!
            } else {
                npt_pd[g][i] = pa | PT_P | PT_W | PT_U | PT_PS;
            }
        }
    }

    if (fb_addr != 0 && (fb_addr + fb_size) > 0x100000000ULL) {
        host_ensure_mapped(fb_addr, fb_size);
        uint64_t gb_idx = (fb_addr >> 30) & 0x1FFULL;
        uint64_t base_1g = fb_addr & ~0x3FFFFFFFULL;
        for (uint64_t i = 0; i < 512; i++) {
            npt_pd_high[i] = (base_1g + (i << 21)) | PT_P | PT_W | PT_U | PT_PS;
        }
        npt_pdpt[gb_idx] = (uint64_t)(uintptr_t)npt_pd_high | PT_P | PT_W | PT_U;
    }

    return (uint64_t)(uintptr_t)npt_pml4;
}

bool guest_gva_to_gpa(uint64_t guest_cr3, uint64_t gva, uint64_t *gpa_out) {
    uint64_t pml4_pa = guest_cr3 & 0x000FFFFFFFFFF000ULL;
    uint64_t i4 = (gva >> 39) & 0x1FFULL;
    uint64_t i3 = (gva >> 30) & 0x1FFULL;
    uint64_t i2 = (gva >> 21) & 0x1FFULL;
    uint64_t i1 = (gva >> 12) & 0x1FFULL;

    if (pml4_pa >= 0x100000000ULL) return false;
    uint64_t pml4e = ((const uint64_t *)(uintptr_t)pml4_pa)[i4];
    if (!(pml4e & PT_P)) return false;

    uint64_t pdpt_pa = pml4e & 0x000FFFFFFFFFF000ULL;
    if (pdpt_pa >= 0x100000000ULL) return false;
    uint64_t pdpte = ((const uint64_t *)(uintptr_t)pdpt_pa)[i3];
    if (!(pdpte & PT_P)) return false;
    if (pdpte & PT_PS) {
        *gpa_out = (pdpte & 0x000FFFFFC0000000ULL) | (gva & 0x3FFFFFFFULL);
        return true;
    }

    uint64_t pd_pa = pdpte & 0x000FFFFFFFFFF000ULL;
    if (pd_pa >= 0x100000000ULL) return false;
    uint64_t pde = ((const uint64_t *)(uintptr_t)pd_pa)[i2];
    if (!(pde & PT_P)) return false;
    if (pde & PT_PS) {
        *gpa_out = (pde & 0x000FFFFFFFE00000ULL) | (gva & 0x1FFFFFULL);
        return true;
    }

    uint64_t pt_pa = pde & 0x000FFFFFFFFFF000ULL;
    if (pt_pa >= 0x100000000ULL) return false;
    uint64_t pte = ((const uint64_t *)(uintptr_t)pt_pa)[i1];
    if (!(pte & PT_P)) return false;

    *gpa_out = (pte & 0x000FFFFFFFFFF000ULL) | (gva & 0xFFFULL);
    return true;
}

bool guest_read_gva(uint64_t guest_cr3, uint64_t gva, void *dst, size_t len) {
    uint8_t *out = (uint8_t *)dst;
    for (size_t i = 0; i < len; i++) {
        uint64_t gpa = 0;
        if (!guest_gva_to_gpa(guest_cr3, gva + i, &gpa)) return false;
        if (gpa >= 0x100000000ULL) return false;
        out[i] = *(const volatile uint8_t *)(uintptr_t)gpa;
    }
    return true;
}

bool guest_write_gva(uint64_t guest_cr3, uint64_t gva, const void *src, size_t len) {
    const uint8_t *in = (const uint8_t *)src;
    for (size_t i = 0; i < len; i++) {
        uint64_t gpa = 0;
        if (!guest_gva_to_gpa(guest_cr3, gva + i, &gpa)) return false;
        if (gpa >= 0x100000000ULL) return false;
        *(volatile uint8_t *)(uintptr_t)gpa = in[i];
    }
    return true;
}
