#pragma once
#include "types.h"

#define MSR_EFER        0xC0000080U
#define MSR_VM_CR       0xC0010114U
#define MSR_VM_HSAVE_PA 0xC0010117U

#define EFER_SCE  (1ULL << 0)
#define EFER_LME  (1ULL << 8)
#define EFER_LMA  (1ULL << 10)
#define EFER_NXE  (1ULL << 11)
#define EFER_SVME (1ULL << 12)

// VMCB misc1 intercepts (offset 0x00C)
#define INTERCEPT_CPUID     (1U << 18)
#define INTERCEPT_HLT       (1U << 24)
#define INTERCEPT_IOIO_PROT (1U << 27)
#define INTERCEPT_MSR_PROT  (1U << 28)
#define INTERCEPT_SHUTDOWN  (1U << 31)

// VMCB misc2 intercepts (offset 0x010)
#define INTERCEPT_VMRUN     (1U << 0)
#define INTERCEPT_VMMCALL   (1U << 1)

// VMEXIT codes
#define VMEXIT_EXCP_BASE    0x040ULL
#define VMEXIT_EXCP_DE      0x040ULL
#define VMEXIT_EXCP_UD      0x046ULL
#define VMEXIT_EXCP_DF      0x048ULL
#define VMEXIT_EXCP_GP      0x04DULL
#define VMEXIT_EXCP_PF      0x04EULL
#define VMEXIT_CPUID        0x072ULL
#define VMEXIT_HLT          0x078ULL
#define VMEXIT_MSR          0x07CULL
#define VMEXIT_SHUTDOWN     0x07FULL
#define VMEXIT_VMRUN        0x080ULL
#define VMEXIT_VMMCALL      0x081ULL
#define VMEXIT_NPF          0x400ULL
#define VMEXIT_INVALID      0xFFFFFFFFFFFFFFFFULL

typedef struct __attribute__((packed)) {
    uint16_t selector;
    uint16_t attrib;
    uint32_t limit;
    uint64_t base;
} VmcbSeg;

typedef struct __attribute__((packed)) {
    uint32_t intercept_cr;          // 0x000
    uint32_t intercept_dr;          // 0x004
    uint32_t intercept_exceptions;  // 0x008
    uint32_t intercept_misc1;       // 0x00C
    uint32_t intercept_misc2;       // 0x010
    uint32_t intercept_misc3;       // 0x014
    uint8_t  rsvd_018[0x03C - 0x018];
    uint16_t pause_filter_thresh;   // 0x03C
    uint16_t pause_filter_count;    // 0x03E
    uint64_t iopm_base_pa;          // 0x040
    uint64_t msrpm_base_pa;         // 0x048
    uint64_t tsc_offset;            // 0x050
    uint32_t guest_asid;            // 0x058
    uint8_t  tlb_control;           // 0x05C
    uint8_t  rsvd_05d[3];           // 0x05D
    uint64_t v_int_ctrl;            // 0x060
    uint64_t int_shadow;            // 0x068
    uint64_t exitcode;              // 0x070
    uint64_t exitinfo1;             // 0x078
    uint64_t exitinfo2;             // 0x080
    uint64_t exitintinfo;           // 0x088
    uint64_t np_enable;             // 0x090
    uint64_t avic_apic_bar;         // 0x098
    uint64_t ghcb_gpa;              // 0x0A0
    uint64_t eventinj;              // 0x0A8
    uint64_t n_cr3;                 // 0x0B0
    uint64_t lbr_virt_enable;       // 0x0B8
    uint32_t clean;                 // 0x0C0
    uint32_t rsvd_0c4;              // 0x0C4
    uint64_t next_rip;              // 0x0C8
    uint8_t  num_bytes_fetched;     // 0x0D0
    uint8_t  guest_bytes[15];       // 0x0D1
    uint8_t  rsvd_0e0[0x400 - 0x0E0];
} VmcbControl;

typedef struct __attribute__((packed)) {
    VmcbSeg  es;                    // 0x400
    VmcbSeg  cs;                    // 0x410
    VmcbSeg  ss;                    // 0x420
    VmcbSeg  ds;                    // 0x430
    VmcbSeg  fs;                    // 0x440
    VmcbSeg  gs;                    // 0x450
    VmcbSeg  gdtr;                  // 0x460
    VmcbSeg  ldtr;                  // 0x470
    VmcbSeg  idtr;                  // 0x480
    VmcbSeg  tr;                    // 0x490
    uint8_t  rsvd_4a0[0x4CB - 0x4A0];
    uint8_t  cpl;                   // 0x4CB
    uint32_t rsvd_4cc;              // 0x4CC
    uint64_t efer;                  // 0x4D0
    uint8_t  rsvd_4d8[0x548 - 0x4D8];
    uint64_t cr4;                   // 0x548
    uint64_t cr3;                   // 0x550
    uint64_t cr0;                   // 0x558
    uint64_t dr7;                   // 0x560
    uint64_t dr6;                   // 0x568
    uint64_t rflags;                // 0x570
    uint64_t rip;                   // 0x578
    uint8_t  rsvd_580[0x5D8 - 0x580];
    uint64_t rsp;                   // 0x5D8
    uint8_t  rsvd_5e0[0x5F8 - 0x5E0];
    uint64_t rax;                   // 0x5F8
    uint64_t star;                  // 0x600
    uint64_t lstar;                 // 0x608
    uint64_t cstar;                 // 0x610
    uint64_t sfmask;                // 0x618
    uint64_t kernel_gs_base;        // 0x620
    uint64_t sysenter_cs;           // 0x628
    uint64_t sysenter_esp;          // 0x630
    uint64_t sysenter_eip;          // 0x638
    uint64_t cr2;                   // 0x640
    uint8_t  rsvd_648[0x668 - 0x648];
    uint64_t g_pat;                 // 0x668
    uint8_t  rsvd_670[0x1000 - 0x670];
} VmcbSave;

typedef struct __attribute__((packed, aligned(4096))) {
    VmcbControl control;
    VmcbSave    save;
} Vmcb;

_Static_assert(sizeof(Vmcb) == 4096, "VMCB must be 4096 bytes");
_Static_assert(offsetof(Vmcb, control.exitcode) == 0x070, "exitcode offset");
_Static_assert(offsetof(Vmcb, control.np_enable) == 0x090, "np_enable offset");
_Static_assert(offsetof(Vmcb, control.eventinj) == 0x0A8, "eventinj offset");
_Static_assert(offsetof(Vmcb, control.n_cr3) == 0x0B0, "n_cr3 offset");
_Static_assert(offsetof(Vmcb, control.next_rip) == 0x0C8, "next_rip offset");
_Static_assert(offsetof(Vmcb, save.cs) == 0x410, "save.cs offset");
_Static_assert(offsetof(Vmcb, save.cpl) == 0x4CB, "save.cpl offset");
_Static_assert(offsetof(Vmcb, save.efer) == 0x4D0, "save.efer offset");
_Static_assert(offsetof(Vmcb, save.cr4) == 0x548, "save.cr4 offset");
_Static_assert(offsetof(Vmcb, save.cr3) == 0x550, "save.cr3 offset");
_Static_assert(offsetof(Vmcb, save.cr0) == 0x558, "save.cr0 offset");
_Static_assert(offsetof(Vmcb, save.rflags) == 0x570, "save.rflags offset");
_Static_assert(offsetof(Vmcb, save.rip) == 0x578, "save.rip offset");
_Static_assert(offsetof(Vmcb, save.rsp) == 0x5D8, "save.rsp offset");
_Static_assert(offsetof(Vmcb, save.rax) == 0x5F8, "save.rax offset");
_Static_assert(offsetof(Vmcb, save.cr2) == 0x640, "save.cr2 offset");
_Static_assert(offsetof(Vmcb, save.g_pat) == 0x668, "save.g_pat offset");

typedef struct {
    uint64_t rbx;
    uint64_t rcx;
    uint64_t rdx;
    uint64_t rsi;
    uint64_t rdi;
    uint64_t rbp;
    uint64_t r8;
    uint64_t r9;
    uint64_t r10;
    uint64_t r11;
    uint64_t r12;
    uint64_t r13;
    uint64_t r14;
    uint64_t r15;
} GuestRegs;

extern void svm_vmrun(uint64_t guest_vmcb_pa, uint64_t host_vmcb_pa, GuestRegs *regs);

// NPT & Guest Memory Helpers
uint64_t npt_init(uint64_t fb_addr, uint64_t fb_size);
void host_ensure_mapped(uint64_t phys_addr, uint64_t size);
bool guest_gva_to_gpa(uint64_t guest_cr3, uint64_t gva, uint64_t *gpa_out);
bool guest_read_gva(uint64_t guest_cr3, uint64_t gva, void *dst, size_t len);
bool guest_write_gva(uint64_t guest_cr3, uint64_t gva, const void *src, size_t len);

// x86-64 Instruction Length Decoder
int x86_decode_len(const uint8_t *code, int max_len);

// Surgeon Patch Engine (Stage 1 deterministic -> Stage 2 LLM hook)
typedef struct {
    uint32_t   vector;
    uint64_t   error_code;
    uint64_t   cr2;
    uint64_t   rip_gva;
    uint64_t   rip_gpa;
    uint8_t    orig_bytes[16];
    uint8_t    insn_len;
    uint32_t   heal_index;
    GuestRegs *regs;
    Vmcb      *vmcb;
} SurgeonContext;

typedef struct {
    uint8_t     patch_bytes[16];
    uint8_t     patch_len;
    const char *reason;
} SurgeonPatch;

typedef enum {
    SURGEON_RES_PASS_TO_LINUX = 0,
    SURGEON_RES_PATCHED       = 1,
    SURGEON_RES_HALT_FOR_DUMP = 2,
} SurgeonResult;

typedef struct __attribute__((packed)) {
    uint32_t magic;        // 0x52314854 ("R1HT")
    uint32_t lock_held;
    uint32_t irq_ctx;
    uint32_t side_effect;
    uint32_t extable;
    uint32_t bug_warn;
    uint32_t stack_ok;
    uint32_t obs_complete;
    char     sym[48];
    char     contract[192];
    char     log_tail[96];
} R1ContextHint;

bool          surgeon_llm_init(void);
bool          surgeon_llm_is_ready(void);
void          surgeon_set_hint(const R1ContextHint *hint);
SurgeonResult surgeon_handle_fault(Vmcb *vmcb, GuestRegs *regs, uint32_t vector, uint64_t err, uint64_t cr2);
uint32_t      surgeon_get_heal_count(void);
void          surgeon_redraw_hud(void);
