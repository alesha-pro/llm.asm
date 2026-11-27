#!/usr/bin/env python3
"""
Ring -1 Kernel Surgeon — Full 600-Scenario Benchmark & Training Suite (Stage 2.2)
Balanced 4-family corpus (160 #PF + 150 #DE + 140 #UD + 75 PASS_TO_LINUX + 75 HALT_FOR_DUMP = 600)
Family-level split (356 train / 122 val / 122 test) — ZERO family overlap!
Every executable stub is real x86-64 System V ABI verified in single-level QEMU/KVM Ring 0.
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from dataclasses import asdict
from typing import List

import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from dataset.scenarios import Scenario, disasm_hex, fmt_spaced_hex

def build_600_scenarios() -> List[Scenario]:
    S: List[Scenario] = []

    def add_exec(
        sid, fam, cat, split, exc_name, vec, err, cr2, rip_sym,
        prefix_hex, fault_hex, suffix_hex, regs, contract,
        expect_out, expect_ret, ref_patch, log_tail,
        lock_held=0, irq_ctx=0, side_effect=0, stack_ok=1, obs_complete=1,
        expected_action="PATCH_BYTES"
    ):
        p = prefix_hex.replace(" ", "").lower()
        f = fault_hex.replace(" ", "").lower()
        sf = suffix_hex.replace(" ", "").lower()
        full = p + f + sf
        poff = len(p) // 2
        flen = len(f) // 2
        S.append(Scenario(
            scenario_id=sid, family_id=fam, category=cat, split=split,
            exc_name=exc_name, exc_vec=vec, err_code=err, cpl=0, cr2=cr2,
            rip_sym=rip_sym, fault_hex=fmt_spaced_hex(f), fault_len=flen,
            disasm=disasm_hex(f), regs_summary=regs, contract_hint=contract,
            extable=0, bug_warn=0, lock_held=lock_held, irq_ctx=irq_ctx,
            side_effect=side_effect, stack_ok=stack_ok, obs_complete=obs_complete,
            log_tail=log_tail, expected_action=expected_action,
            ref_patch_hex=fmt_spaced_hex(ref_patch) if ref_patch != "NONE" else "NONE",
            kvm_mode="exec" if expected_action == "PATCH_BYTES" else "policy_only",
            stub_hex=full, patch_off=poff,
            expect_out_hex=expect_out, expect_ret_hex=expect_ret,
        ))

    def le32(v: int) -> str:
        return v.to_bytes(4, "little").hex()

    # =========================================================================
    # 1. PATCHABLE RING 0 #PF (160 scenarios: 20 families x 8 variants)
    #    Split by family: 12 train (96), 4 val (32), 4 test (32)
    # =========================================================================
    pf_store_fams = [
        # (fam_name, split, fault_builder, prefix_hex, regs_fmt, mod_prefix, len_bytes)
        ("pf_abs_u32_imm",  "train", lambda i: "c7042500000000" + le32(0xdead0001 + i*17), "31c0", "RAX=0x0 RBX=0x3 RCX=0x0 RDX=0x0", "kdrv_tel", 11),
        ("pf_abs_u8_imm",   "train", lambda i: "c6042500000000" + f"{(i+1)&0xff:02x}",      "31c0", "RAX=0x0 RBX=0x1 RCX=0x0 RDX=0x0", "kdrv_mark", 8),
        ("pf_reg_rax_edx",  "train", lambda i: "8910", lambda i: "31c0ba" + le32(0x10+i),             "RAX=0x0 RDX=0x{v:x} RCX=0x0", "kdrv_ring", 2),
        ("pf_reg_rax_ecx",  "train", lambda i: "8908", lambda i: "31c0b9" + le32(0x20+i),             "RAX=0x0 RCX=0x{v:x} RDX=0x0", "kdrv_stat", 2),
        ("pf_reg_rsi_edx",  "train", lambda i: "8916", lambda i: "31c031f6ba" + le32(0x30+i),         "RAX=0x0 RSI=0x0 RDX=0x{v:x}", "kdrv_slot", 2),
        ("pf_byte_rax",     "train", lambda i: "c600" + f"{(i+1)&0xff:02x}", "31c0",        "RAX=0x0 RBX=0x0 RDX=0x1", "kdrv_flag", 3),
        ("pf_byte_rsi",     "train", lambda i: "c606" + f"{(i+5)&0xff:02x}", "31c031f6",    "RAX=0x0 RSI=0x0 RDX=0x1", "kdrv_tag", 3),
        ("pf_imm32_rax",    "train", lambda i: "c700" + le32(0x100+i*13), "31c0",           "RAX=0x0 RBX=0x0 RCX=0x0", "kdrv_dbg", 6),
        ("pf_qword_rax_rdx","val",   lambda i: "488910", lambda i: "31c0ba" + le32(0x40+i),           "RAX=0x0 RDX=0x{v:x} RCX=0x0", "kdrv_aud", 3),
        ("pf_qword_rax_rcx","val",   lambda i: "488908", lambda i: "31c0b9" + le32(0x50+i),           "RAX=0x0 RCX=0x{v:x} RDX=0x0", "kdrv_log", 3),
        ("pf_byte_rcx",     "test",  lambda i: "c601" + f"{(i+9)&0xff:02x}", "31c031c9",    "RAX=0x0 RCX=0x0 RDX=0x0", "kdrv_evt", 3),
        ("pf_qword_rsi_rdx","test",  lambda i: "488916", lambda i: "31c031f6ba" + le32(0x60+i),       "RAX=0x0 RSI=0x0 RDX=0x{v:x}", "kdrv_trc", 3),
    ]
    for fam, spl, ffn, pref, rfmt, mpref, flen in pf_store_fams:
        for i in range(8):
            out_v = 0x100 + i * 0x10
            suf = "48c74708" + le32(out_v) + "c3"
            fhex = ffn(i)
            regs = rfmt.format(v=0x10+i) + f" RDI=0xffff888004100000 RSP=0xffffc90000013{0xd00+i*8:03x} RFLAGS=0x246"
            add_exec(
                f"{fam}_{i:02d}", fam, "pf_patchable", spl,
                "#PF", 14, 0x0002, "0x0000000000000000", f"{mpref}+0x0{i+1:02x}0",
                pref(i) if callable(pref) else pref, fhex, suf, regs,
                f"Optional telemetry store to unmapped pointer before commit; skipping store preserves out=0x{out_v:x} and returns 0.",
                f"{out_v:x}", "0", "90" * flen,
                f"{mpref}: stage=pre_commit opt_ptr=NULL seq={i}"
            )

    # 8 families of NULL struct-offset loads needing zero fallback in EAX/RAX (64 scenarios: 4 train, 2 val, 2 test)
    pf_load_fams = [
        # (fam, split, pref_base, fault_hex_fn, off_base, reg_base_name, ref_patch, mod_pref)
        ("pf_load_rbx_d8",  "train", "53b8{imm}31db", lambda off: f"8b43{off:02x}", 0x08, "RBX=0x0", "31c090",   "kdrv_cfg", "5b48894708c3"),
        ("pf_load_rcx_d8",  "train", "b8{imm}31c9",   lambda off: f"8b41{off:02x}", 0x10, "RCX=0x0", "31c090",   "kdrv_opt", "48894708c3"),
        ("pf_load_rdx_d8",  "train", "b8{imm}31d2",   lambda off: f"8b42{off:02x}", 0x14, "RDX=0x0", "31c090",   "kdrv_env", "48894708c3"),
        ("pf_load_rcx_0",   "train", "b8{imm}31c9",   lambda off: "8b01",           0x00, "RCX=0x0", "31c0",     "kdrv_val", "48894708c3"),
        ("pf_load64_rbx_d8","val",   "53b8{imm}31db", lambda off: f"488b43{off:02x}", 0x08, "RBX=0x0", "4831c090", "kdrv_tbl", "5b48894708c3"),
        ("pf_load_rdx_0",   "val",   "b8{imm}31d2",   lambda off: "8b02",           0x00, "RDX=0x0", "31c0",     "kdrv_ref", "48894708c3"),
        ("pf_load_rsi_d8",  "test",  "b8{imm}31f6",   lambda off: f"8b46{off:02x}", 0x08, "RSI=0x0", "31c090",   "kdrv_hdr", "48894708c3"),
        ("pf_load64_rsi_d8","test",  "b8{imm}31f6",   lambda off: f"488b46{off:02x}", 0x10, "RSI=0x0", "4831c090", "kdrv_ctx", "48894708c3"),
    ]
    for fam, spl, ptmpl, ffn, obase, rbase, rpatch, mpref, suf in pf_load_fams:
        for i in range(8):
            stale = 0x55 + i * 0x11
            off = (obase + i * 4) if obase > 0 else 0
            pref = ptmpl.format(imm=le32(stale))
            fhex = ffn(off)
            cr2 = f"0x{off:016x}"
            regs = f"RAX=0x{stale:x} {rbase} RDI=0xffff888004100000 RSP=0xffffc90000013{0xc00+i*8:03x} RFLAGS=0x246"
            add_exec(
                f"{fam}_{i:02d}", fam, "pf_patchable", spl,
                "#PF", 14, 0x0000, cr2, f"{mpref}+0x0{i+2:02x}4",
                pref(i) if callable(pref) else pref, fhex, suf, regs,
                f"Struct field load into EAX/RAX then written to out_val and returned; stale RAX=0x{stale:x}, safe fallback requires RAX=0 (out=0, ret=0).",
                "0", "0", rpatch,
                f"{mpref}: ptr=NULL field_off=0x{off:x} fallback=zero seq={i}"
            )

    # =========================================================================
    # 2. PATCHABLE RING 0 #DE DIVIDE-BY-ZERO (150 scenarios: 15 families x 10)
    #    Split by family: 9 train (90), 3 val (30), 3 test (30)
    # =========================================================================
    de_fams = [
        ("de_div_ecx",    "train", "b8{imm}31d231c9",     "f7f1",   "48894708c3",   "RCX=0x0", "31c0",   "kdrv_rate"),
        ("de_idiv_ecx",   "train", "b8{imm}9931c9",       "f7f9",   "48894708c3",   "RCX=0x0", "31c0",   "kdrv_srate"),
        ("de_div_esi",    "train", "b8{imm}31d231f6",     "f7f6",   "48894708c3",   "RSI=0x0", "31c0",   "kdrv_avg"),
        ("de_idiv_esi",   "train", "b8{imm}9931f6",       "f7fe",   "48894708c3",   "RSI=0x0", "31c0",   "kdrv_savg"),
        ("de_div_ebx",    "train", "53b8{imm}31d231db",   "f7f3",   "5b48894708c3", "RBX=0x0", "31c0",   "kdrv_scale"),
        ("de_idiv_ebx",   "train", "53b8{imm}9931db",     "f7fb",   "5b48894708c3", "RBX=0x0", "31c0",   "kdrv_sscale"),
        ("de_div64_rcx",  "train", "b8{imm}31d231c9",     "48f7f1", "48894708c3",   "RCX=0x0", "4831c0", "kdrv_bw"),
        ("de_idiv64_rcx", "train", "b8{imm}489931c9",     "48f7f9", "48894708c3",   "RCX=0x0", "4831c0", "kdrv_sbw"),
        ("de_div64_rsi",  "train", "b8{imm}31d231f6",     "48f7f6", "48894708c3",   "RSI=0x0", "4831c0", "kdrv_lat"),
        ("de_div_r8d",    "val",   "b8{imm}31d24531c0",   "41f7f0", "48894708c3",   "R8=0x0",  "31c090", "kdrv_hist"),
        ("de_idiv_r8d",   "val",   "b8{imm}994531c0",     "41f7f8", "48894708c3",   "R8=0x0",  "31c090", "kdrv_shist"),
        ("de_div64_r8",   "val",   "b8{imm}31d24531c0",   "49f7f0", "48894708c3",   "R8=0x0",  "4831c0", "kdrv_thru"),
        ("de_div_r9d",    "test",  "b8{imm}31d24531c9",   "41f7f1", "48894708c3",   "R9=0x0",  "31c090", "kdrv_ema"),
        ("de_idiv_r9d",   "test",  "b8{imm}994531c9",     "41f7f9", "48894708c3",   "R9=0x0",  "31c090", "kdrv_sema"),
        ("de_div_r10d",   "test",  "b8{imm}31d24531d2",   "41f7f2", "48894708c3",   "R10=0x0", "31c090", "kdrv_norm"),
    ]
    for fam, spl, ptmpl, fhex, suf, dreg, rpatch, mpref in de_fams:
        for i in range(10):
            num = 0x2a + i * 0x19
            pref = ptmpl.format(imm=le32(num))
            regs = f"RAX=0x{num:x} {dreg} RDX=0x0 RDI=0xffff888004100000 RSP=0xffffc90000013{0xb00+i*8:03x} RFLAGS=0x246"
            add_exec(
                f"{fam}_{i:02d}", fam, "de_patchable", spl,
                "#DE", 0, 0x0000, "0x0000000000000000", f"{mpref}+0x0{i+1:02x}8",
                pref(i) if callable(pref) else pref, fhex, suf, regs,
                f"Integer division quotient in RAX/EAX is stored to out_val and returned; numerator=0x{num:x}, divisor=0, safe fallback requires quotient=0 (out=0, ret=0).",
                "0", "0", rpatch,
                f"{mpref}: denom=0 fallback_quotient=0 sample={i}"
            )

    # =========================================================================
    # 3. PATCHABLE RING 0 #UD TRAPS (140 scenarios: 14 families x 10)
    #    Split by family: 8 train (80), 3 val (30), 3 test (30)
    # =========================================================================
    ud_fams = [
        # (fam, split, fhex, dirty_eax, rpatch, mpref)
        ("ud2_clean_keep",   "train", "0f0b",   False, "9090",   "kdrv_guard"),
        ("ud2_clean_zero",   "train", "0f0b",   False, "9090",   "kdrv_sync"),
        ("ud2_dirty_ret0",   "train", "0f0b",   True,  "31c0",   "kdrv_gate"),
        ("ud2_dirty_zero",   "train", "0f0b",   True,  "31c0",   "kdrv_chk"),
        ("ud1_eax_clean",    "train", "0fb9c0", False, "909090", "kdrv_p1"),
        ("ud1_ecx_clean",    "train", "0fb9c9", False, "909090", "kdrv_p2"),
        ("ud1_eax_dirty",    "train", "0fb9c0", True,  "31c090", "kdrv_g1"),
        ("ud1_ecx_dirty",    "train", "0fb9c9", True,  "31c090", "kdrv_g2"),
        ("ud1_edx_clean",    "val",   "0fb9d2", False, "909090", "kdrv_v1"),
        ("ud1_edx_dirty",    "val",   "0fb9d2", True,  "31c090", "kdrv_v2"),
        ("ud2_val_keep",     "val",   "0f0b",   False, "9090",   "kdrv_v3"),
        ("ud0_eax_clean",    "test",  "0fffc0", False, "909090", "kdrv_t1"),
        ("ud0_eax_dirty",    "test",  "0fffc0", True,  "31c090", "kdrv_t2"),
        ("ud0_ecx_clean",    "test",  "0fffc9", False, "909090", "kdrv_t3"),
    ]
    for fam, spl, fhex, dirty, rpatch, mpref in ud_fams:
        for i in range(10):
            if dirty:
                stale = 0x1300 + i * 0x11
                pref = "b8" + le32(stale)
                suf = "48894708c3"
                exp_out = "0"
                regs = f"RAX=0x{stale:x} RBX=0x0 RCX=0x0 RDX=0x0 RDI=0xffff888004100000 RSP=0xffffc90000013{0xa00+i*8:03x} RFLAGS=0x246"
                chint = f"Stale EAX=0x{stale:x} precedes diagnostic trap before storing RAX to out_val and returning; safe fallback requires EAX=0 (out=0, ret=0)."
                ltail = f"{mpref}: diag_trap fallback_ret=0 state=clean seq={i}"
            else:
                ov = 0x40 + i * 7
                pref = "31c048c74708" + le32(ov)
                suf = "c3"
                exp_out = f"{ov:x}"
                regs = f"RAX=0x0 RBX=0x{i} RCX=0x0 RDX=0x0 RDI=0xffff888004100000 RSP=0xffffc90000013{0xa00+i*8:03x} RFLAGS=0x246"
                chint = f"Non-fatal diagnostic trap with EAX=0 and out_val=0x{ov:x} already prepared; skipping trap completes cleanly with ret=0."
                ltail = f"{mpref}: diag_trap state=clean pre_commit=1 seq={i}"
            add_exec(
                f"{fam}_{i:02d}", fam, "ud_patchable", spl,
                "#UD", 6, 0x0000, "0x0000000000000000", f"{mpref}+0x0{i+1:02x}a",
                pref(i) if callable(pref) else pref, fhex, suf, regs, chint, exp_out, "0", rpatch, ltail
            )

    # =========================================================================
    # 4A. CONTROL PASS_TO_LINUX (75 scenarios: 15 families x 5)
    #     Split by family: 9 train (45), 3 val (15), 3 test (15)
    # =========================================================================
    pass_fams = [
        ("ctrl_ext_get_user8",  "train", "#PF", 14, 0x0000, 0, "0x0000000000000010", "__get_user_8",       "488b11",   1, 0, "extable_test", "User-pointer read covered by Linux _ASM_EXTABLE fixup (-EFAULT)."),
        ("ctrl_ext_get_user4",  "train", "#PF", 14, 0x0000, 0, "0x0000000000000020", "__get_user_4",       "8b11",     1, 0, "extable_test", "32-bit get_user read covered by _ASM_EXTABLE fixup (-EFAULT)."),
        ("ctrl_ext_put_user8",  "train", "#PF", 14, 0x0002, 0, "0x0000000000000008", "__put_user_8",       "488901",   1, 0, "extable_test", "User-pointer write covered by Linux _ASM_EXTABLE fixup (-EFAULT)."),
        ("ctrl_ext_futex",      "train", "#PF", 14, 0x0002, 0, "0x0000000000000000", "futex_detect_cmpxchg","3e0fb10e",1, 0, "extable_test", "Linux futex_detect_cmpxchg probe covered by _ASM_EXTABLE."),
        ("ctrl_ext_strncpy",    "train", "#PF", 14, 0x0000, 0, "0x00007f0000001000", "strncpy_from_user",  "488b06",   1, 0, "extable_test", "String copy from user memory covered by _ASM_EXTABLE."),
        ("ctrl_warn_driver",    "train", "#UD",  6, 0x0000, 0, "0x0000000000000000", "kdrv_warn_check",    "0f0b",     0, 1, "warn_test",    "Linux WARN_ON_ONCE entry (bug_warn=1); kernel logs warning and continues."),
        ("ctrl_cpl3_pf_read",   "train", "#PF", 14, 0x0004, 3, "0x00007ffe91a00000", "user_app",           "8b00",     0, 0, "policy_only",  "Ring 3 userspace page fault (demand paging / COW handled by Linux mm)."),
        ("ctrl_cpl3_pf_write",  "train", "#PF", 14, 0x0006, 3, "0x00007ffe91b00000", "user_heap",          "8910",     0, 0, "policy_only",  "Ring 3 userspace COW write fault handled by Linux mm."),
        ("ctrl_cpl3_de",        "train", "#DE",  0, 0x0000, 3, "0x0000000000000000", "user_calc",          "f7f1",     0, 0, "policy_only",  "Ring 3 userspace divide-by-zero (Linux delivers SIGFPE)."),
        ("ctrl_ext_copy_from",  "val",   "#PF", 14, 0x0000, 0, "0x00007f1012340000", "copy_user_fast",     "f3a4",     1, 0, "extable_test", "Demand-paged copy_from_user fault covered by _ASM_EXTABLE."),
        ("ctrl_warn_net",       "val",   "#UD",  6, 0x0000, 0, "0x0000000000000000", "net_warn_once",      "0f0b",     0, 1, "warn_test",    "Network subsystem WARN_ON_ONCE (bug_warn=1); handled by Linux."),
        ("ctrl_cpl3_pf_null",   "val",   "#PF", 14, 0x0006, 3, "0x0000000000000000", "user_segv",          "c70001000000",0,0,"policy_only",  "Ring 3 userspace NULL write (Linux delivers SIGSEGV cleanly)."),
        ("ctrl_ext_probe_rd",   "test",  "#PF", 14, 0x0000, 0, "0x0000000000000040", "copy_from_kernel_nofault","8b06",1, 0, "extable_test", "Kernel nofault memory probe covered by _ASM_EXTABLE."),
        ("ctrl_warn_blk",       "test",  "#UD",  6, 0x0000, 0, "0x0000000000000000", "blk_warn_ratelimit", "0f0b",     0, 1, "warn_test",    "Block layer WARN_ON (bug_warn=1); handled cleanly by Linux."),
        ("ctrl_cpl3_ud",        "test",  "#UD",  6, 0x0000, 3, "0x0000000000000000", "user_sigill",        "0f0b",     0, 0, "policy_only",  "Ring 3 userspace invalid opcode (Linux delivers SIGILL)."),
    ]
    for fam, spl, en, vec, err, cpl, cr2, sym, fhex, ext, bw, kmode, chint in pass_fams:
        for i in range(5):
            flen = len(fhex) // 2
            rsp_hex = "7ffe919ff000" if cpl == 3 else "ffffc90000013b80"
            path_kind = "ring3" if cpl == 3 else ("extable" if ext else "warn")
            S.append(Scenario(
                scenario_id=f"{fam}_{i:02d}", family_id=fam, category="pass_to_linux", split=spl,
                exc_name=en, exc_vec=vec, err_code=err, cpl=cpl, cr2=cr2,
                rip_sym=f"{sym}+0x0{i+1:02x}0", fault_hex=fmt_spaced_hex(fhex), fault_len=flen,
                disasm=disasm_hex(fhex),
                regs_summary=f"RAX=0x{i} RBX=0x0 RCX=0x{16+i*8:x} RSP=0x{rsp_hex} RFLAGS=0x246",
                contract_hint=chint, extable=ext, bug_warn=bw,
                lock_held=0, irq_ctx=0, side_effect=0, stack_ok=1, obs_complete=1,
                log_tail=f"linux: standard {path_kind} path seq={i}",
                expected_action="PASS_TO_LINUX", ref_patch_hex="NONE",
                kvm_mode=kmode, stub_hex="", patch_off=0,
                expect_out_hex="ffffffffffffffff", expect_ret_hex="ffffffffffffffff",
            ))

    # =========================================================================
    # 4B. CONTROL HALT_FOR_DUMP (75 counter-examples: 15 families x 5)
    #     IDENTICAL opcodes to Families 1-3, differing ONLY in safety context!
    #     Split by family: 9 train (45), 3 val (15), 3 test (15)
    # =========================================================================
    halt_fams = [
        ("halt_pf_spinlock",   "train", "#PF", 14, 0x0002, "0x0000000000000000", "kdrv_tx",     "c7042500000000efbeadde", 1, 0, 0, 1, 1, "Ring 0 NULL write while holding spinlock (lock_held=1)."),
        ("halt_pf_dma_dirty",  "train", "#PF", 14, 0x0002, "0x0000000000000000", "kdrv_blk",    "8910",                   0, 0, 1, 1, 1, "Ring 0 NULL store after DMA descriptor already submitted (side_effect=1)."),
        ("halt_pf_hardirq",    "train", "#PF", 14, 0x0002, "0x0000000000000000", "kdrv_isr",    "c60001",                 0, 1, 0, 1, 1, "Ring 0 NULL byte store inside hard IRQ handler (irq_ctx=1)."),
        ("halt_pf_load_lock",  "train", "#PF", 14, 0x0000, "0x0000000000000010", "kdrv_cfg_lk", "8b4310",                 1, 0, 0, 1, 1, "Ring 0 NULL struct load while holding mutex/spinlock (lock_held=1)."),
        ("halt_de_journal",    "train", "#DE",  0, 0x0000, "0x0000000000000000", "kdrv_jnl",    "f7f1",                   0, 0, 1, 1, 1, "Ring 0 divide-by-zero after journal header already flushed (side_effect=1)."),
        ("halt_de_rwlock",     "train", "#DE",  0, 0x0000, "0x0000000000000000", "kdrv_sched",  "f7fe",                   1, 0, 0, 1, 1, "Ring 0 signed divide-by-zero while holding rwlock (lock_held=1)."),
        ("halt_ud_bug_dirty",  "train", "#UD",  6, 0x0000, "0x0000000000000000", "kdrv_fs_bug", "0f0b",                   0, 0, 1, 1, 1, "Hard BUG_ON invariant failure after metadata write (side_effect=1)."),
        ("halt_ud_lock",       "train", "#UD",  6, 0x0000, "0x0000000000000000", "kdrv_mm_bug", "0fb9c0",                 1, 0, 0, 1, 1, "Ring 0 UD1 trap while holding page-table spinlock (lock_held=1)."),
        ("halt_stack_pf",      "train", "#PF", 14, 0x0002, "0x0000000000000000", "kdrv_ovf",    "8908",                   0, 0, 0, 0, 1, "Ring 0 fault with corrupted kernel stack pointer (stack_ok=0)."),
        ("halt_val_de_irq",    "val",   "#DE",  0, 0x0000, "0x0000000000000000", "kdrv_timer",  "41f7f0",                 0, 1, 0, 1, 1, "Ring 0 divide-by-zero inside timer IRQ context (irq_ctx=1)."),
        ("halt_val_obs_pf",    "val",   "#PF", 14, 0x0002, "0x0000000000000000", "kdrv_unk_pf", "488910",                 0, 0, 0, 1, 0, "Incomplete Ring -1 telemetry snapshot (obs_complete=0); unsafe to patch."),
        ("halt_val_multi",     "val",   "#UD",  6, 0x0000, "0x0000000000000000", "kdrv_net_tx", "0f0b",                   1, 0, 1, 1, 1, "Ring 0 BUG_ON with spinlock held and TX ring mutated (lock_held=1, side_effect=1)."),
        ("halt_test_stack_de", "test",  "#DE",  0, 0x0000, "0x0000000000000000", "kdrv_stk_de", "41f7f9",                 0, 0, 0, 0, 1, "Ring 0 divide-by-zero with invalid kernel RSP (stack_ok=0)."),
        ("halt_test_obs_ud",   "test",  "#UD",  6, 0x0000, "0x0000000000000000", "kdrv_unk_ud", "0fffc0",                 0, 0, 0, 1, 0, "Truncated telemetry on Ring 0 #UD (obs_complete=0); halt for dump."),
        ("halt_test_side_pf",  "test",  "#PF", 14, 0x0000, "0x0000000000000008", "kdrv_nvme",   "8b4608",                 0, 0, 1, 1, 1, "Ring 0 NULL struct read after NVMe doorbell already rung (side_effect=1)."),
    ]
    for fam, spl, en, vec, err, cr2, sym, fhex, lh, ic, se, sok, oc, chint in halt_fams:
        for i in range(5):
            rsp_h = "0000000000000008" if not sok else "ffffc90000013900"
            add_exec(
                f"{fam}_{i:02d}", fam, "halt_for_dump", spl,
                en, vec, err, cr2, f"{sym}+0x0{i+1:02x}c",
                "", fhex, "",
                f"RAX=0x{0x20+i:x} RBX=0x0 RCX=0x0 RSP=0x{rsp_h} RFLAGS=0x246",
                chint, "ffffffffffffffff", "ffffffffffffffff", "NONE",
                f"{sym}: hazard lock={lh} irq={ic} dirty={se} stack_ok={sok} obs_ok={oc} seq={i}",
                lock_held=lh, irq_ctx=ic, side_effect=se, stack_ok=sok, obs_complete=oc,
                expected_action="HALT_FOR_DUMP"
            )

    return S

if __name__ == "__main__":
    items = build_600_scenarios()
    print(f"Total scenarios: {len(items)}")
    print("By split:", dict(Counter(x.split for x in items)))
    print("By category:", dict(Counter(x.category for x in items)))
    print("By expected_action:", dict(Counter(x.expected_action for x in items)))
    sf = {}
    for x in items:
        sf.setdefault(x.split, set()).add(x.family_id)
    assert not (sf["train"] & sf["val"]), "train/val family leak!"
    assert not (sf["train"] & sf["test"]), "train/test family leak!"
    assert not (sf["val"] & sf["test"]), "val/test family leak!"
    assert len(items) == 600, f"Expected 600, got {len(items)}"
    print("Family-level split isolation: OK", {k: len(v) for k, v in sf.items()})
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios_600.jsonl")
    with open(out_path, "w") as f:
        for x in items:
            f.write(json.dumps(asdict(x)) + "\n")
    print(f"Saved 600 scenarios -> {out_path}")
