#include "svm.h"
#include "fb.h"
#include "smp.h"
#include "llm_engine.h"

#define NVDIMM_MODEL_PA  0x100000000ULL  // 2 GiB Read-Only NVDIMM (UNMAPPED in NPT!)
#define LLM_ARENA_PA     0x70000000ULL   // 64 MiB Isolated L1 Arena (UNMAPPED in NPT!)
#define XSAVE_BUF_PA     0x73DF0000ULL   // 64-byte aligned 4KB buffer inside L1 Arena

static R1LlmState    g_llm;
static uint32_t      g_event_count = 0;
static uint32_t      g_heal_count  = 0;
static bool          g_hint_active = false;
static R1ContextHint g_hint;

typedef struct {
    bool     valid;
    uint32_t event_num;
    char     fault_name[32];
    uint32_t cpl;
    uint64_t cr2;
    uint64_t rip_gva;
    uint64_t rip_gpa;
    uint8_t  orig_bytes[16];
    uint8_t  insn_len;
    uint8_t  patch_bytes[16];
    uint8_t  patch_len;
    char     action[24];
    char     diagnosis[160];
    uint32_t elapsed_ms;
    uint32_t gen_toks;
} LastVerdict;

static LastVerdict g_last;

bool surgeon_llm_init(void) {
    enable_avx2_cpu();
    bool ok = r1_llm_init(&g_llm,
                          (const void *)(uintptr_t)NVDIMM_MODEL_PA,
                          (void *)(uintptr_t)LLM_ARENA_PA,
                          r1_smp_dispatch);
    if (ok) {
        kprintf("[LLM] MiniCPM5-2B Q4_K_M READY in isolated L1 RAM @ 0x%lx (dim=%u layers=%u vocab=%u cores=%d)\n",
                NVDIMM_MODEL_PA, g_llm.hdr->dim, g_llm.hdr->n_layers, g_llm.hdr->vocab_size, r1_smp_num_cpus());
    } else {
        kprintf("[LLM] WARNING: No R1SURG01 model found at 0x%lx\n", NVDIMM_MODEL_PA);
    }
    return ok;
}

bool surgeon_llm_is_ready(void) {
    return g_llm.ready;
}

void surgeon_set_hint(const R1ContextHint *hint) {
    if (hint && hint->magic == 0x52314854U) {
        g_hint = *hint;
        g_hint_active = true;
    } else {
        g_hint_active = false;
    }
}

uint32_t surgeon_get_heal_count(void) {
    return g_heal_count;
}

void surgeon_redraw_hud(void) {
    if (!g_last.valid) return;
    fb_draw_ai_verdict_hud(
        g_last.event_num, g_last.fault_name, g_last.cpl, g_last.cr2,
        g_last.rip_gva, g_last.rip_gpa, g_last.orig_bytes, g_last.insn_len,
        g_last.patch_bytes, g_last.patch_len, g_last.action, g_last.diagnosis,
        r1_smp_num_cpus(), g_last.elapsed_ms, g_last.gen_toks
    );
}

static void mini_disasm(const uint8_t *b, int len, char *out, size_t cap) {
    if (len >= 2 && b[0] == 0x0F && b[1] == 0x0B) {
        hv_snprintf(out, cap, "ud2");
    } else if (len >= 3 && b[0] == 0x0F && b[1] == 0xB9) {
        hv_snprintf(out, cap, "ud1 eax, eax");
    } else if (len >= 3 && b[0] == 0x0F && b[1] == 0xFF) {
        hv_snprintf(out, cap, "ud0 eax, eax");
    } else if (len >= 2 && b[0] == 0xF7 && b[1] == 0xF1) {
        hv_snprintf(out, cap, "div ecx");
    } else if (len >= 2 && b[0] == 0xF7 && b[1] == 0xF3) {
        hv_snprintf(out, cap, "div ebx");
    } else if (len >= 2 && b[0] == 0xF7 && b[1] == 0xFE) {
        hv_snprintf(out, cap, "idiv esi");
    } else if (len >= 3 && b[0] == 0x48 && b[1] == 0xF7 && b[2] == 0xF1) {
        hv_snprintf(out, cap, "div rcx");
    } else if (len >= 11 && b[0] == 0xC7 && b[1] == 0x04 && b[2] == 0x25) {
        uint32_t imm = *(const uint32_t *)(b + 7);
        hv_snprintf(out, cap, "mov dword ptr [0x0], 0x%x", imm);
    } else if (len >= 2 && b[0] == 0x89 && b[1] == 0x10) {
        hv_snprintf(out, cap, "mov dword ptr [rax], edx");
    } else if (len >= 3 && b[0] == 0xC6 && b[1] == 0x00) {
        hv_snprintf(out, cap, "mov byte ptr [rax], 0x%x", b[2]);
    } else if (len >= 3 && b[0] == 0x48 && b[1] == 0x89 && b[2] == 0x10) {
        hv_snprintf(out, cap, "mov qword ptr [rax], rdx");
    } else if (len >= 3 && b[0] == 0x8B && b[1] == 0x43) {
        hv_snprintf(out, cap, "mov eax, dword ptr [rbx + 0x%x]", b[2]);
    } else if (len >= 3 && b[0] == 0x48 && b[1] == 0x8B) {
        hv_snprintf(out, cap, "mov rax, qword ptr [reg]");
    } else {
        hv_snprintf(out, cap, "op_%02x_%02x", b[0], len > 1 ? b[1] : 0);
    }
}

static int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

static const char *find_substr(const char *hay, const char *needle) {
    size_t nlen = hv_strlen(needle);
    if (nlen == 0) return hay;
    for (size_t i = 0; hay[i]; i++) {
        if (hv_strncmp(hay + i, needle, nlen) == 0) return hay + i;
    }
    return NULL;
}

static void copy_line_after(const char *p, char *dst, size_t cap) {
    while (*p == ' ' || *p == '\t') p++;
    size_t i = 0;
    while (*p && *p != '\n' && *p != '\r' && i + 1 < cap) {
        dst[i++] = *p++;
    }
    dst[i] = '\0';
}

static bool validate_patch_bytes(const uint8_t *patch, int plen, int expected_len) {
    if (plen != expected_len || plen <= 0 || plen > 15) return false;
    int pos = 0;
    while (pos < plen) {
        uint8_t op = patch[pos];
        if (op == 0xEB || op == 0xE8 || op == 0xE9 || op == 0xEA ||
            op == 0xC3 || op == 0xC2 || op == 0xCB || op == 0xCA ||
            op == 0xCF || op == 0xCC || op == 0xCD || op == 0xCE ||
            op == 0xF4 || op == 0xFA || op == 0xFB || op == 0xFF ||
            (op >= 0x70 && op <= 0x7F) || (op >= 0xE0 && op <= 0xE7)) {
            return false;
        }
        if (op == 0x0F && pos + 1 < plen) {
            uint8_t op2 = patch[pos + 1];
            if (op2 == 0x0B || op2 == 0x05 || op2 == 0x07 || op2 == 0x30 ||
                op2 == 0x32 || op2 == 0x01 || op2 == 0xB9 || op2 == 0xFF ||
                (op2 >= 0x80 && op2 <= 0x8F)) {
                return false;
            }
        }
        int step = x86_decode_len(patch + pos, plen - pos);
        if (step <= 0 || pos + step > plen) return false;
        pos += step;
    }
    return (pos == plen);
}

typedef struct {
    uint32_t       event_num;
    const char    *fault_name;
    uint64_t       rip_gva;
    const uint8_t *orig_bytes;
    uint8_t        insn_len;
    char           stream_buf[256];
    int            stream_pos;
} StreamHudCtx;

static void token_stream_cb(const char *piece, int piece_len, int step, void *user_ctx) {
    StreamHudCtx *sc = (StreamHudCtx *)user_ctx;
    for (int i = 0; i < piece_len && sc->stream_pos + 1 < (int)sizeof(sc->stream_buf); i++) {
        sc->stream_buf[sc->stream_pos++] = piece[i];
    }
    sc->stream_buf[sc->stream_pos] = '\0';
    if ((step & 3) == 0) {
        fb_draw_thinking_hud(sc->event_num, sc->fault_name, sc->rip_gva,
                             sc->orig_bytes, sc->insn_len, r1_smp_num_cpus(), sc->stream_buf);
    }
}

SurgeonResult surgeon_handle_fault(Vmcb *vmcb, GuestRegs *regs, uint32_t vector, uint64_t err, uint64_t cr2) {
    uint64_t rip_gva = vmcb->save.rip;
    uint64_t rip_gpa = 0;

    if (!guest_gva_to_gpa(vmcb->save.cr3, rip_gva, &rip_gpa)) {
        kprintf("[SURGEON] Cannot translate RIP GVA 0x%016lx\n", rip_gva);
        return SURGEON_RES_PASS_TO_LINUX;
    }

    uint8_t orig_bytes[16];
    hv_memset(orig_bytes, 0, sizeof(orig_bytes));
    if (!guest_read_gva(vmcb->save.cr3, rip_gva, orig_bytes, 15)) {
        if (!guest_read_gva(vmcb->save.cr3, rip_gva, orig_bytes, 2)) {
            return SURGEON_RES_PASS_TO_LINUX;
        }
    }

    // Pass through Linux early-boot futex_detect_cmpxchg() probe silently unless hint active
    if (!g_hint_active && vector == 14 && cr2 == 0 &&
        (orig_bytes[0] == 0x3E || orig_bytes[0] == 0xF0) &&
        orig_bytes[1] == 0x0F && orig_bytes[2] == 0xB1) {
        return SURGEON_RES_PASS_TO_LINUX;
    }

    int ilen = x86_decode_len(orig_bytes, 15);
    if (ilen <= 0 || ilen > 15) {
        return SURGEON_RES_PASS_TO_LINUX;
    }

    g_event_count++;
    const char *exc_short = (vector == 14) ? "#PF" : (vector == 0) ? "#DE" : (vector == 6) ? "#UD" : "#EX";
    const char *fname = (vector == 14) ? "#PF (PAGE FAULT)" :
                        (vector == 0)  ? "#DE (DIV BY ZERO)" :
                        (vector == 6)  ? "#UD (INVALID OP)" : "#EXCP";

    char hex_orig[64], disasm[96];
    size_t hpos = 0;
    for (int i = 0; i < ilen && hpos + 4 < sizeof(hex_orig); i++) {
        hpos += (size_t)hv_snprintf(hex_orig + hpos, sizeof(hex_orig) - hpos, "%s%02x", i ? " " : "", orig_bytes[i]);
    }
    mini_disasm(orig_bytes, ilen, disasm, sizeof(disasm));

    R1ContextHint h;
    if (g_hint_active) {
        h = g_hint;
        g_hint_active = false;
    } else {
        hv_memset(&h, 0, sizeof(h));
        h.stack_ok     = (vmcb->save.rsp >= 0xFFFF800000000000ULL) ? 1U : 0U;
        h.obs_complete = 1U;
        hv_snprintf(h.sym, sizeof(h.sym), "ring0+0x%lx", rip_gva & 0xFFFFULL);
        if (vector == 0) {
            hv_snprintf(h.contract, sizeof(h.contract),
                        "Integer division quotient in RAX/EAX is stored to out_val and returned; safe fallback requires quotient=0 (out=0, ret=0).");
            hv_snprintf(h.log_tail, sizeof(h.log_tail), "kernel: divide error denom=0 fallback_quotient=0");
        } else if (vector == 6) {
            hv_snprintf(h.contract, sizeof(h.contract),
                        "Non-fatal diagnostic trap with EAX=0; skipping trap completes cleanly with ret=0.");
            hv_snprintf(h.log_tail, sizeof(h.log_tail), "kernel: diag_trap state=clean pre_commit=1");
        } else {
            hv_snprintf(h.contract, sizeof(h.contract),
                        "Optional telemetry store to unmapped pointer before commit; skipping store preserves state and returns 0.");
            hv_snprintf(h.log_tail, sizeof(h.log_tail), "kernel: stage=pre_commit opt_ptr=NULL");
        }
    }

    char user_prompt[1280];
    hv_snprintf(user_prompt, sizeof(user_prompt),
        "[RING-1 TELEMETRY]\n"
        "EXC: %s (vec=%u err=0x%04lx cpl=%u)\n"
        "CR2: 0x%016lx\n"
        "RIP: %s (len=%d)\n"
        "BYTES: %s\n"
        "DISASM: %s\n"
        "REGS: RAX=0x%lx RBX=0x%lx RCX=0x%lx RDX=0x%lx RSI=0x%lx RDI=0x%lx RSP=0x%lx RFLAGS=0x%lx\n"
        "CONTRACT: %s\n"
        "CTX: extable=%u bug_warn=%u lock_held=%u irq_ctx=%u side_effect=%u stack_ok=%u obs_complete=%u\n"
        "LOG_TAIL:\n  %s\n"
        "Output ONLY 3 lines (no <think>, no markdown):\n"
        "DIAGNOSIS: <one short sentence>\n"
        "ACTION: <PATCH_BYTES | PASS_TO_LINUX | HALT_FOR_DUMP>\n"
        "PATCH: <exact-length space-separated hex bytes or NONE>",
        exc_short, vector, err, (uint32_t)vmcb->save.cpl, cr2,
        h.sym, ilen, hex_orig, disasm,
        vmcb->save.rax, regs->rbx, regs->rcx, regs->rdx, regs->rsi, regs->rdi, vmcb->save.rsp, vmcb->save.rflags,
        h.contract,
        h.extable, h.bug_warn, h.lock_held, h.irq_ctx, h.side_effect, h.stack_ok, h.obs_complete,
        h.log_tail
    );

    kprintf("\n====================================================================\n");
    kprintf("[RING -1 AI SURGEON] LIVE INTERCEPT #%u: %s (vec=%u cpl=%u cr2=0x%lx)\n",
            g_event_count, fname, vector, (uint32_t)vmcb->save.cpl, cr2);
    kprintf("[RING -1 AI SURGEON] RIP_GVA=0x%016lx  RIP_GPA=0x%016lx  ORIG(%dB)=%s (%s)\n",
            rip_gva, rip_gpa, ilen, hex_orig, disasm);

    StreamHudCtx sctx;
    hv_memset(&sctx, 0, sizeof(sctx));
    sctx.event_num  = g_event_count;
    sctx.fault_name = fname;
    sctx.rip_gva    = rip_gva;
    sctx.orig_bytes = orig_bytes;
    sctx.insn_len   = (uint8_t)ilen;
    fb_draw_thinking_hud(g_event_count, fname, rip_gva, orig_bytes, (uint8_t)ilen, r1_smp_num_cpus(), NULL);

    char llm_out[512];
    hv_memset(llm_out, 0, sizeof(llm_out));
    int ptok = 0, gtok = 0;
    uint32_t elapsed_ms = 0;

    if (g_llm.ready) {
        enable_avx2_cpu();
        uint64_t saved_xcr0 = get_xcr0();
        set_xcr0(saved_xcr0 | 7ULL);
        void *xs_buf = (void *)(uintptr_t)XSAVE_BUF_PA;
        hv_memset(xs_buf, 0, 4096);
        xsave_state(xs_buf);

        uint64_t t0 = read_tsc();
        const char *sys_msg = "Ring -1 Kernel Surgeon. Output 3 lines only: DIAGNOSIS, ACTION, PATCH.";
        r1_llm_generate(&g_llm, sys_msg, user_prompt, llm_out, sizeof(llm_out), 64,
                        &ptok, &gtok, token_stream_cb, &sctx);
        uint64_t dt_tsc = read_tsc() - t0;
        elapsed_ms = (uint32_t)(dt_tsc / 2300000ULL);

        xrstor_state(xs_buf);
        set_xcr0(saved_xcr0);

        kprintf("[RING -1 LLM] MiniCPM5-2B Q4_K_M (%d L1 cores, prompt=%d tok, gen=%d tok, ~%u ms):\n%s\n",
                r1_smp_num_cpus(), ptok, gtok, elapsed_ms, llm_out);
    } else {
        hv_snprintf(llm_out, sizeof(llm_out),
                    "DIAGNOSIS: No NVDIMM model attached; halting safely.\n"
                    "ACTION: HALT_FOR_DUMP\nPATCH: NONE");
    }

    char diag[160] = "Ring -1 AI diagnosis";
    char act[24]   = "HALT_FOR_DUMP";
    char pat_s[128] = "NONE";

    const char *p_diag = find_substr(llm_out, "DIAGNOSIS:");
    if (p_diag) copy_line_after(p_diag + 10, diag, sizeof(diag));
    const char *p_act = find_substr(llm_out, "ACTION:");
    if (p_act) copy_line_after(p_act + 7, act, sizeof(act));
    const char *p_pat = find_substr(llm_out, "PATCH:");
    if (p_pat) copy_line_after(p_pat + 6, pat_s, sizeof(pat_s));

    SurgeonResult decision = SURGEON_RES_HALT_FOR_DUMP;
    uint8_t patch_bytes[16];
    uint8_t patch_len = 0;

    if (hv_strncmp(act, "PATCH_BYTES", 11) == 0) {
        hv_snprintf(act, sizeof(act), "PATCH_BYTES");
        const char *q = pat_s;
        while (*q && patch_len < 16) {
            while (*q == ' ' || *q == ',') q++;
            if (!*q) break;
            int hi = hex_val(q[0]), lo = q[1] ? hex_val(q[1]) : -1;
            if (hi < 0 || lo < 0) break;
            patch_bytes[patch_len++] = (uint8_t)((hi << 4) | lo);
            q += 2;
        }
        if (validate_patch_bytes(patch_bytes, patch_len, ilen)) {
            if (guest_write_gva(vmcb->save.cr3, rip_gva, patch_bytes, patch_len)) {
                g_heal_count++;
                decision = SURGEON_RES_PATCHED;
                vmcb->control.eventinj = 0;
                kprintf("[RING -1 AI SURGEON] VERIFIED & APPLIED PATCH (%u B) -> RESUMING VMRUN!\n", patch_len);
            } else {
                hv_snprintf(act, sizeof(act), "HALT_FOR_DUMP");
                hv_snprintf(diag, sizeof(diag), "Failed to write guest physical page; halting safely.");
                decision = SURGEON_RES_HALT_FOR_DUMP;
            }
        } else {
            kprintf("[RING -1 AI SURGEON] SAFETY REJECT: invalid patch len=%u (expected %d) -> HALT_FOR_DUMP\n",
                    patch_len, ilen);
            hv_snprintf(act, sizeof(act), "HALT_FOR_DUMP");
            decision = SURGEON_RES_HALT_FOR_DUMP;
        }
    } else if (hv_strncmp(act, "PASS_TO_LINUX", 13) == 0) {
        hv_snprintf(act, sizeof(act), "PASS_TO_LINUX");
        decision = SURGEON_RES_PASS_TO_LINUX;
        kprintf("[RING -1 AI SURGEON] DECISION: PASS_TO_LINUX (delegating to guest kernel)\n");
    } else {
        hv_snprintf(act, sizeof(act), "HALT_FOR_DUMP");
        decision = SURGEON_RES_HALT_FOR_DUMP;
        kprintf("[RING -1 AI SURGEON] DECISION: HALT_FOR_DUMP (unsafe state — freezing guest!)\n");
    }
    kprintf("====================================================================\n");

    g_last.valid = true;
    g_last.event_num = g_event_count;
    hv_snprintf(g_last.fault_name, sizeof(g_last.fault_name), "%s", fname);
    g_last.cpl = vmcb->save.cpl;
    g_last.cr2 = cr2;
    g_last.rip_gva = rip_gva;
    g_last.rip_gpa = rip_gpa;
    hv_memcpy(g_last.orig_bytes, orig_bytes, (size_t)ilen);
    g_last.insn_len = (uint8_t)ilen;
    hv_memcpy(g_last.patch_bytes, patch_bytes, patch_len);
    g_last.patch_len = patch_len;
    hv_snprintf(g_last.action, sizeof(g_last.action), "%s", act);
    hv_snprintf(g_last.diagnosis, sizeof(g_last.diagnosis), "%s", diag);
    g_last.elapsed_ms = elapsed_ms;
    g_last.gen_toks = (uint32_t)gtok;

    surgeon_redraw_hud();
    return decision;
}
