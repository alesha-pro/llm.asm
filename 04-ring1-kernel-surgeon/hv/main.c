#include "types.h"
#include "fb.h"
#include "svm.h"
#include "smp.h"
#include "linux_boot.h"

static Vmcb    g_guest_vmcb __attribute__((aligned(4096)));
static Vmcb    g_host_vmcb  __attribute__((aligned(4096)));
static uint8_t g_hsave[4096] __attribute__((aligned(4096)));
static uint8_t g_msrpm[8192] __attribute__((aligned(4096)));

static void msrpm_set_intercept(uint8_t *msrpm, uint32_t msr, bool intercept_rd, bool intercept_wr) {
    uint32_t base_byte = 0, idx = 0;
    if (msr <= 0x00001FFFU) { base_byte = 0x0000; idx = msr; }
    else if (msr >= 0xC0000000U && msr <= 0xC0001FFFU) { base_byte = 0x0800; idx = msr - 0xC0000000U; }
    else if (msr >= 0xC0010000U && msr <= 0xC0011FFFU) { base_byte = 0x1000; idx = msr - 0xC0010000U; }
    else return;
    uint32_t bit = idx * 2;
    uint32_t byte_off = base_byte + (bit / 8);
    uint32_t bit_off  = bit % 8;
    if (intercept_rd) msrpm[byte_off] |= (uint8_t)(1U << bit_off);
    if (intercept_wr) msrpm[byte_off] |= (uint8_t)(1U << (bit_off + 1));
}

static bool svm_check_and_enable(void) {
    uint32_t a, b, c, d;
    cpuid_raw(0x80000001U, 0, &a, &b, &c, &d);
    if (!(c & (1U << 2))) {
        kprintf("[SVM] FATAL: SVM not present in CPUID!\n");
        return false;
    }
    if (read_msr(MSR_VM_CR) & (1ULL << 4)) {
        kprintf("[SVM] FATAL: MSR_VM_CR.SVMDIS=1\n");
        return false;
    }
    cpuid_raw(0x8000000AU, 0, &a, &b, &c, &d);
    kprintf("[SVM] AMD SVM Rev=%u NASID=%u Features=0x%x (NPT=%u NRIP=%u)\n",
            a & 0xFF, b, d, (d & 1U), (d >> 3) & 1U);

    write_msr(MSR_EFER, read_msr(MSR_EFER) | EFER_SVME);
    hv_memset(g_hsave, 0, sizeof(g_hsave));
    write_msr(MSR_VM_HSAVE_PA, (uint64_t)(uintptr_t)g_hsave);
    kprintf("[SVM] EFER.SVME enabled | VM_HSAVE_PA=0x%lx\n", (uint64_t)(uintptr_t)g_hsave);
    return true;
}

static void inject_exception(Vmcb *vmcb, uint32_t vector, bool has_err, uint32_t err) {
    uint64_t inj = (uint64_t)(vector & 0xFFU) | (3ULL << 8) | (1ULL << 31);
    if (has_err) {
        inj |= (1ULL << 11);
        inj |= ((uint64_t)err << 32);
    }
    vmcb->control.eventinj = inj;
}

static void advance_rip(Vmcb *vmcb, uint64_t fallback_len) {
    if (vmcb->control.next_rip != 0) vmcb->save.rip = vmcb->control.next_rip;
    else vmcb->save.rip += fallback_len;
}

static void setup_guest_vmcb(Vmcb *v, const LinuxGuestInit *init, uint64_t npt_cr3) {
    hv_memset(v, 0, sizeof(*v));
    hv_memset(&g_host_vmcb, 0, sizeof(g_host_vmcb));
    hv_memset(g_msrpm, 0, sizeof(g_msrpm));

    msrpm_set_intercept(g_msrpm, MSR_EFER, true, true);

    v->control.intercept_exceptions = (1U << 0) | (1U << 6) | (1U << 8) | (1U << 14);
    v->control.intercept_misc1      = INTERCEPT_CPUID | INTERCEPT_MSR_PROT | INTERCEPT_SHUTDOWN;
    v->control.intercept_misc2      = INTERCEPT_VMRUN | INTERCEPT_VMMCALL;
    v->control.msrpm_base_pa        = (uint64_t)(uintptr_t)g_msrpm;
    v->control.guest_asid           = 1;
    v->control.tlb_control          = 3;
    v->control.np_enable            = 1;
    v->control.n_cr3                = npt_cr3;

    v->save.cs.selector = 0x08;
    v->save.cs.attrib   = 0x0A9B;
    v->save.cs.limit    = 0xFFFFFFFFU;
    v->save.cs.base     = 0;

    VmcbSeg data_seg = { 0x10, 0x0C93, 0xFFFFFFFFU, 0 };
    v->save.ds = data_seg;
    v->save.es = data_seg;
    v->save.ss = data_seg;
    v->save.fs = data_seg;
    v->save.gs = data_seg;

    v->save.gdtr.limit = init->guest_gdtr_limit;
    v->save.gdtr.base  = init->guest_gdtr_base;
    v->save.idtr.limit = 0;
    v->save.idtr.base  = 0;

    v->save.ldtr.selector = 0;
    v->save.ldtr.attrib   = 0x0082;
    v->save.tr.selector   = 0x18;
    v->save.tr.attrib     = 0x008B;
    v->save.tr.limit      = 0x67;

    v->save.cpl    = 0;
    v->save.efer   = EFER_LME | EFER_LMA | EFER_NXE | EFER_SCE | EFER_SVME;
    v->save.cr0    = 0x80050033ULL;
    v->save.cr3    = init->guest_cr3;
    v->save.cr4    = 0x00000020ULL;
    v->save.dr6    = 0xFFFF0FF0ULL;
    v->save.dr7    = 0x00000400ULL;
    v->save.rflags = 0x0000000000000002ULL;
    v->save.rip    = init->entry_rip;
    v->save.rsp    = init->guest_rsp;
    v->save.rax    = 0;
    v->save.g_pat  = 0x0007040600070406ULL;
}

void hv_main(uint64_t magic, uint64_t mbi_addr) {
    serial_init();
    kprintf("\n====================================================================\n");
    kprintf("  RING -1 KERNEL SURGEON (Stage 3: Bare-Metal AMD SVM + MiniCPM5-2B)\n");
    kprintf("====================================================================\n");

    BootAssets assets;
    if (!mb2_parse(magic, mbi_addr, &assets)) {
        kprintf("[HV] FATAL: Multiboot2 parse failed!\n");
        return;
    }

    uint64_t fb_size = (uint64_t)assets.fb.pitch * assets.fb.height;
    if (assets.fb.addr != 0) host_ensure_mapped(assets.fb.addr, fb_size);
    fb_init(&assets.fb);
    assets.fb = g_fb;

    int n_cores = r1_smp_init(24);
    bool llm_ok = surgeon_llm_init();
    fb_draw_boot_banner(n_cores, llm_ok);

    if (!svm_check_and_enable()) return;

    uint64_t npt_cr3 = npt_init(g_fb.addr, fb_size);
    kprintf("[NPT] 4-level NPT ready @ 0x%lx (Isolated: ELF 0x200000..0x3FFFFF, Arena 0x70000000..0x73FFFFFF, Model 0x100000000..0x17FFFFFFF)\n",
            npt_cr3);

    LinuxGuestInit guest_init;
    if (!linux_prepare_guest(&assets, &guest_init)) {
        kprintf("[HV] FATAL: linux_prepare_guest failed!\n");
        return;
    }

    setup_guest_vmcb(&g_guest_vmcb, &guest_init, npt_cr3);

    GuestRegs regs;
    hv_memset(&regs, 0, sizeof(regs));
    regs.rsi = guest_init.boot_params_pa;

    kprintf("[SVM] Entering VMRUN -> Linux startup_64 @ RIP=0x%lx RSI=0x%lx CR3=0x%lx\n",
            g_guest_vmcb.save.rip, regs.rsi, g_guest_vmcb.save.cr3);

    uint64_t pf_reinjected = 0;
    uint64_t cpuid_exits   = 0;
    uint64_t msr_exits     = 0;

    while (1) {
        g_guest_vmcb.control.clean = 0;
        svm_vmrun((uint64_t)(uintptr_t)&g_guest_vmcb,
                  (uint64_t)(uintptr_t)&g_host_vmcb,
                  &regs);
        g_guest_vmcb.control.tlb_control = 0;

        uint64_t code  = g_guest_vmcb.control.exitcode;
        uint64_t info1 = g_guest_vmcb.control.exitinfo1;
        uint64_t info2 = g_guest_vmcb.control.exitinfo2;

        if (code == VMEXIT_EXCP_PF) {
            if (g_guest_vmcb.save.cpl != 0 || info2 >= 0x1000ULL) {
                pf_reinjected++;
                g_guest_vmcb.save.cr2 = info2;
                inject_exception(&g_guest_vmcb, 14, true, (uint32_t)info1);
                continue;
            }
            SurgeonResult r = surgeon_handle_fault(&g_guest_vmcb, &regs, 14, info1, info2);
            if (r == SURGEON_RES_PASS_TO_LINUX) {
                pf_reinjected++;
                g_guest_vmcb.save.cr2 = info2;
                inject_exception(&g_guest_vmcb, 14, true, (uint32_t)info1);
            } else if (r == SURGEON_RES_HALT_FOR_DUMP) {
                kprintf("[RING -1] HALT_FOR_DUMP EXECUTED — GUEST FROZEN SAFELY (NO CORRUPTION)!\n");
                break;
            }
            continue;
        }

        if (code == VMEXIT_EXCP_DE) {
            if (g_guest_vmcb.save.cpl != 0) {
                inject_exception(&g_guest_vmcb, 0, false, 0);
                continue;
            }
            SurgeonResult r = surgeon_handle_fault(&g_guest_vmcb, &regs, 0, 0, 0);
            if (r == SURGEON_RES_PASS_TO_LINUX) {
                inject_exception(&g_guest_vmcb, 0, false, 0);
            } else if (r == SURGEON_RES_HALT_FOR_DUMP) {
                kprintf("[RING -1] HALT_FOR_DUMP EXECUTED (#DE) — GUEST FROZEN SAFELY!\n");
                break;
            }
            continue;
        }

        if (code == VMEXIT_EXCP_UD) {
            if (g_guest_vmcb.save.cpl != 0) {
                inject_exception(&g_guest_vmcb, 6, false, 0);
                continue;
            }
            SurgeonResult r = surgeon_handle_fault(&g_guest_vmcb, &regs, 6, 0, 0);
            if (r == SURGEON_RES_PASS_TO_LINUX) {
                inject_exception(&g_guest_vmcb, 6, false, 0);
            } else if (r == SURGEON_RES_HALT_FOR_DUMP) {
                kprintf("[RING -1] HALT_FOR_DUMP EXECUTED (#UD) — GUEST FROZEN SAFELY!\n");
                break;
            }
            continue;
        }

        if (code == VMEXIT_CPUID) {
            cpuid_exits++;
            uint32_t leaf = (uint32_t)g_guest_vmcb.save.rax;
            uint32_t sub  = (uint32_t)regs.rcx;
            uint32_t a, b, c, d;
            if (leaf == 0x40000000U) {
                a = 0x40000000U;
                b = 0x676E6952U;
                c = 0x72755331U;
                d = 0x6E6F6567U;
            } else {
                cpuid_raw(leaf, sub, &a, &b, &c, &d);
                if (leaf == 0x80000001U) c &= ~(1U << 2);
            }
            g_guest_vmcb.save.rax = a;
            regs.rbx = b;
            regs.rcx = c;
            regs.rdx = d;
            advance_rip(&g_guest_vmcb, 2);
            continue;
        }

        if (code == VMEXIT_MSR) {
            msr_exits++;
            uint32_t msr = (uint32_t)regs.rcx;
            if (msr == MSR_EFER) {
                if (info1 == 0) {
                    uint64_t v = g_guest_vmcb.save.efer & ~EFER_SVME;
                    g_guest_vmcb.save.rax = (uint32_t)v;
                    regs.rdx = (uint32_t)(v >> 32);
                } else {
                    uint64_t v = ((uint64_t)(uint32_t)regs.rdx << 32) | (uint32_t)g_guest_vmcb.save.rax;
                    g_guest_vmcb.save.efer = v | EFER_SVME;
                }
            } else {
                if (info1 == 0) {
                    uint64_t v = read_msr(msr);
                    g_guest_vmcb.save.rax = (uint32_t)v;
                    regs.rdx = (uint32_t)(v >> 32);
                } else {
                    uint64_t v = ((uint64_t)(uint32_t)regs.rdx << 32) | (uint32_t)g_guest_vmcb.save.rax;
                    write_msr(msr, v);
                }
            }
            advance_rip(&g_guest_vmcb, 2);
            continue;
        }

        if (code == VMEXIT_VMMCALL) {
            uint64_t cmd = g_guest_vmcb.save.rax;
            if (cmd == 0x53555247ULL) {
                g_guest_vmcb.save.rax = 0x4F4B5F52494E4731ULL;
                regs.rbx = surgeon_get_heal_count();
                regs.rcx = (uint64_t)r1_smp_num_cpus();
                regs.rdx = surgeon_llm_is_ready() ? 1ULL : 0ULL;
                kprintf("[VMMCALL] Status: heals=%u l1_cores=%d llm_ready=%d pf_pass=%lu\n",
                        surgeon_get_heal_count(), r1_smp_num_cpus(), surgeon_llm_is_ready() ? 1 : 0, pf_reinjected);
            } else if (cmd == 0x53555248ULL) {
                surgeon_redraw_hud();
                g_guest_vmcb.save.rax = 0;
            } else if (cmd == 0x53555249ULL) {
                R1ContextHint h;
                if (guest_read_gva(g_guest_vmcb.save.cr3, regs.rbx, &h, sizeof(h))) {
                    surgeon_set_hint(&h);
                    g_guest_vmcb.save.rax = 0;
                } else {
                    g_guest_vmcb.save.rax = 1;
                }
            }
            advance_rip(&g_guest_vmcb, 3);
            continue;
        }

        if (code == VMEXIT_NPF) {
            kprintf("[NPT] #NPF ISOLATION VIOLATION! GPA=0x%016lx err=0x%lx RIP=0x%016lx CPL=%u\n",
                    info2, info1, g_guest_vmcb.save.rip, (uint32_t)g_guest_vmcb.save.cpl);
            break;
        }

        if (code == VMEXIT_EXCP_DF || code == VMEXIT_SHUTDOWN || code == VMEXIT_INVALID) {
            kprintf("[SVM] FATAL VMEXIT code=0x%lx info1=0x%lx info2=0x%lx RIP=0x%016lx\n",
                    code, info1, info2, g_guest_vmcb.save.rip);
            break;
        }

        kprintf("[SVM] Unhandled VMEXIT code=0x%lx RIP=0x%016lx\n", code, g_guest_vmcb.save.rip);
        break;
    }

    kprintf("[HV] Hypervisor halted cleanly (heals=%u l1_cores=%d llm_ready=%d pf_pass=%lu).\n",
            surgeon_get_heal_count(), r1_smp_num_cpus(), surgeon_llm_is_ready() ? 1 : 0, pf_reinjected);
}
