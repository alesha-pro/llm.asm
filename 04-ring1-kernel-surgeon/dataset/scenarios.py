#!/usr/bin/env python3
"""
Ring -1 Kernel Surgeon — Stage 2.1 Scenario Suite (48 scenarios across 4 families)
Strict family-level 60/20/20 split (train / val / test) with matched counter-examples
having identical faulting opcodes to prevent regex shortcuts.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import List, Optional
from capstone import Cs, CS_ARCH_X86, CS_MODE_64

_CS = Cs(CS_ARCH_X86, CS_MODE_64)

def disasm_hex(hex_str: str) -> str:
    raw = bytes.fromhex(hex_str.replace(" ", ""))
    parts = [f"{ins.mnemonic} {ins.op_str}".strip() for ins in _CS.disasm(raw, 0)]
    return "; ".join(parts) if parts else "<invalid>"

def fmt_spaced_hex(hex_str: str) -> str:
    s = hex_str.replace(" ", "").lower()
    return " ".join(s[i:i+2] for i in range(0, len(s), 2))

@dataclass
class Scenario:
    scenario_id: str
    family_id: str
    category: str          # pf_patchable | de_patchable | ud_patchable | pass_to_linux | halt_for_dump
    split: str             # train | val | test
    # Telemetry visible to model (zero label leakage!)
    exc_name: str          # #PF | #DE | #UD
    exc_vec: int
    err_code: int
    cpl: int
    cr2: str
    rip_sym: str
    fault_hex: str         # spaced hex of faulting instruction
    fault_len: int
    disasm: str
    regs_summary: str
    contract_hint: str     # neutral ABI/contract description of the routine (no answer!)
    extable: int
    bug_warn: int
    lock_held: int
    irq_ctx: int
    side_effect: int
    stack_ok: int
    obs_complete: int
    log_tail: str
    # Private verification harness spec (NEVER sent to model!)
    expected_action: str   # PATCH_BYTES | PASS_TO_LINUX | HALT_FOR_DUMP
    ref_patch_hex: str     # spaced hex or "NONE"
    kvm_mode: str          # exec | extable_test | warn_test | policy_only
    stub_hex: str          # full Ring 0 x86-64 stub hex (for kvm_mode == "exec")
    patch_off: int
    expect_out_hex: str    # u64 hex or ffffffffffffffff (any)
    expect_ret_hex: str    # u64 hex or ffffffffffffffff (any)

    def render_prompt(self) -> str:
        bytes_field = self.fault_hex if self.obs_complete else "<TRUNCATED>"
        disasm_field = self.disasm if self.obs_complete else "<UNAVAILABLE>"
        regs_field = self.regs_summary if self.obs_complete else "RAX=<UNAVAILABLE> RSP=<UNAVAILABLE>"
        return (
            "[RING-1 TELEMETRY]\n"
            f"EXC: {self.exc_name} (vec={self.exc_vec} err=0x{self.err_code:04x} cpl={self.cpl})\n"
            f"CR2: {self.cr2}\n"
            f"RIP: {self.rip_sym} (len={self.fault_len})\n"
            f"BYTES: {bytes_field}\n"
            f"DISASM: {disasm_field}\n"
            f"REGS: {regs_field}\n"
            f"CONTRACT: {self.contract_hint}\n"
            f"CTX: extable={self.extable} bug_warn={self.bug_warn} lock_held={self.lock_held} "
            f"irq_ctx={self.irq_ctx} side_effect={self.side_effect} stack_ok={self.stack_ok} "
            f"obs_complete={self.obs_complete}\n"
            f"LOG_TAIL:\n  {self.log_tail}\n"
            "Output ONLY 3 lines (no <think>, no markdown):\n"
            "DIAGNOSIS: <one short sentence>\n"
            "ACTION: <PATCH_BYTES | PASS_TO_LINUX | HALT_FOR_DUMP>\n"
            "PATCH: <exact-length space-separated hex bytes or NONE>"
        )

def build_pilot_scenarios() -> List[Scenario]:
    """
    48 scenarios with strict family-level 60/20/20 split (28 train / 10 val / 10 test).
    Every stub_hex is verified real x86-64 System V ABI (rdi = struct bench_ctx *):
      [rdi+0x00] = canary_before
      [rdi+0x08] = out_val (initially 0x12345678)
      [rdi+0x10] = canary_after
    """
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

    # =========================================================================
    # FAMILY 1: Patchable Ring 0 #PF (12 scenarios, 6 sub-families)
    # =========================================================================
    # 1a. Absolute NULL dword store (11B): c7 04 25 00 00 00 00 ef be ad de
    # Stub: xor eax,eax (31 c0); [FAULT 11B]; mov qword [rdi+8], 0x100 (48 c7 47 08 00 01 00 00); ret (c3)
    for idx, (imm, tag, spl) in enumerate([
        ("efbead de", "v1", "train"),
        ("010000 00", "v2", "train"),
    ]):
        fhex = "c7042500000000" + imm.replace(" ", "")
        add_exec(
            f"pf_null_abs_u32_{tag}", "fam_pf_abs_u32", "pf_patchable", spl,
            "#PF", 14, 0x0002, "0x0000000000000000", f"kdrv_telemetry+0x01{idx}0",
            "31c0", fhex, "48c7470800010000c3",
            "RAX=0x0 RBX=0x3 RCX=0x0 RDX=0x0 RSI=0x0 RDI=0xffff888004100000 RSP=0xffffc90000013e20 RFLAGS=0x246",
            "Optional debug probe write before commit; skipping fault leaves status=0x100 and returns 0.",
            "100", "0", "90" * 11,
            "kdrv: stage=pre_commit debug_slot=unmapped"
        )

    # 1b. Indirect NULL register dword store (2B): 89 10 (mov [rax], edx) / 89 08 (mov [rax], ecx)
    # Stub: xor eax,eax (31 c0); mov edx, 0x55 (ba 55 00 00 00); [FAULT 2B]; mov qword [rdi+8], 0 (48 c7 47 08 00 00 00 00); ret (c3)
    for idx, (fhex, reg_s, tag, spl) in enumerate([
        ("8910", "RAX=0x0 RDX=0x55", "rax_edx", "train"),
        ("8908", "RAX=0x0 RCX=0x77", "rax_ecx", "train"),
    ]):
        add_exec(
            f"pf_null_reg_store32_{tag}", "fam_pf_reg_store32", "pf_patchable", spl,
            "#PF", 14, 0x0002, "0x0000000000000000", f"kdrv_ring+0x02{idx}4",
            "31c0ba55000000b977000000", fhex, "48c7470800000000c3",
            f"{reg_s} RBX=0x0 RSI=0x0 RDI=0xffff888004100000 RSP=0xffffc90000013e10 RFLAGS=0x246",
            "Best-effort counter store to optional pointer; skipping store keeps out=0, ret=0.",
            "0", "0", "9090",
            "kdrv_ring: optional_stat_ptr=NULL pre_commit=1"
        )

    # 1c. Indirect NULL byte store (3B): c6 00 01 (mov byte [rax], 1) / c6 06 01 (mov byte [rsi], 1)
    for idx, (fhex, pref, reg_s, tag, spl) in enumerate([
        ("c60001", "31c0", "RAX=0x0", "rax", "train"),
        ("c60601", "31c031f6", "RAX=0x0 RSI=0x0", "rsi", "train"),
    ]):
        add_exec(
            f"pf_null_byte_store_{tag}", "fam_pf_byte_store", "pf_patchable", spl,
            "#PF", 14, 0x0002, "0x0000000000000000", f"kdrv_flag+0x00{idx}c",
            pref, fhex, "48c7470800000000c3",
            f"{reg_s} RBX=0x0 RCX=0x0 RDX=0x1 RDI=0xffff888004100000 RSP=0xffffc90000013df0 RFLAGS=0x246",
            "Optional flag byte write before commit; skipping write preserves out=0 and returns 0.",
            "0", "0", "909090",
            "kdrv_flag: trace_marker=NULL state=clean"
        )

    # 1d. Indirect NULL qword store (3B): 48 89 10 (mov [rax], rdx) / 48 89 08 (mov [rax], rcx)
    for idx, (fhex, tag, spl) in enumerate([
        ("488910", "qword_rdx", "val"),
        ("488908", "qword_rcx", "val"),
    ]):
        add_exec(
            f"pf_null_qword_store_{tag}", "fam_pf_qword_store", "pf_patchable", spl,
            "#PF", 14, 0x0002, "0x0000000000000000", f"kdrv_audit+0x03{idx}8",
            "31c0ba11000000b922000000", fhex, "48c7470800000000c3",
            "RAX=0x0 RBX=0x0 RCX=0x22 RDX=0x11 RSI=0x0 RDI=0xffff888004100000 RSP=0xffffc90000013d90 RFLAGS=0x246",
            "Audit timestamp write to uninitialized slot before commit; skipping preserves out=0, ret=0.",
            "0", "0", "909090",
            "kdrv_audit: audit_slot=NULL pre_commit=1"
        )

    # 1e. NULL struct-offset read needing zero fallback (3B): 8b 43 10 (mov eax, [rbx+0x10])
    # Notice EAX starts as 0x77! Blind NOP leaves EAX=0x77 (FAILS!); xor eax,eax; nop (31 c0 90) sets EAX=0 (PASSES!)
    for idx, (fhex, off_s, tag, spl) in enumerate([
        ("8b4310", "0x10", "off10", "train"),
        ("8b4320", "0x20", "off20", "train"),
    ]):
        add_exec(
            f"pf_offset_load_zero_{tag}", "fam_pf_offset_load", "pf_patchable", spl,
            "#PF", 14, 0x0000, f"0x00000000000000{off_s[2:]}", f"kdrv_cfg+0x04{idx}0",
            "b87700000031db", fhex, "48894708c3",
            "RAX=0x77 RBX=0x0 RCX=0x0 RDX=0x0 RSI=0x0 RDI=0xffff888004100000 RSP=0xffffc90000013d40 RFLAGS=0x246",
            "Config field read into EAX then stored to out_val and returned; safe fallback requires EAX=0 (out=0, ret=0).",
            "0", "0", "31c090",
            f"kdrv_cfg: cfg_ptr=NULL field_off={off_s} fallback=zero"
        )

    # 1f. NULL struct-offset read into EAX (test split, unseen family!): 8b 46 08 (mov eax, [rsi+0x08])
    for idx, (fhex, off_s, tag, spl) in enumerate([
        ("8b4608", "0x08", "rsi_off08", "test"),
        ("8b4618", "0x18", "rsi_off18", "test"),
    ]):
        add_exec(
            f"pf_rsi_load_zero_{tag}", "fam_pf_rsi_load", "pf_patchable", spl,
            "#PF", 14, 0x0000, f"0x00000000000000{off_s[2:]}", f"kdrv_hdr+0x05{idx}0",
            "b89900000031f6", fhex, "48894708c3",
            "RAX=0x99 RBX=0x0 RCX=0x0 RDX=0x0 RSI=0x0 RDI=0xffff888004100000 RSP=0xffffc90000013d20 RFLAGS=0x246",
            "Header field read into EAX then stored to out_val and returned; safe fallback requires EAX=0 (out=0, ret=0).",
            "0", "0", "31c090",
            f"kdrv_hdr: hdr_ptr=NULL field_off={off_s} fallback=zero"
        )

    # =========================================================================
    # FAMILY 2: Patchable Ring 0 #DE Divide-by-Zero (12 scenarios, 6 sub-families)
    # =========================================================================
    # All #DE scenarios set EAX=nonzero before div, then store RAX->[rdi+8] and ret!
    # So blind NOP FAILS postcondition (out_val!=0); zeroing EAX/RAX (31 c0 / 31 c0 90 / 48 31 c0) PASSES!
    de_defs = [
        ("de_div_ecx_v1",  "fam_de_div_ecx",  "train", "b82a00000031d231c9", "f7f1", "48894708c3", "RAX=0x2a RCX=0x0 RDX=0x0", "31c0",   "kdrv_rate: denom=0 fallback_quotient=0"),
        ("de_div_ecx_v2",  "fam_de_div_ecx",  "train", "b86400000031d231c9", "f7f1", "48894708c3", "RAX=0x64 RCX=0x0 RDX=0x0", "31c0",   "kdrv_rate: samples=0 fallback_quotient=0"),
        ("de_idiv_esi_v1", "fam_de_idiv_esi", "train", "b8500000009931f6",   "f7fe", "48894708c3", "RAX=0x50 RSI=0x0 RDX=0x0", "31c0",   "kdrv_avg: count=0 fallback_quotient=0"),
        ("de_idiv_esi_v2", "fam_de_idiv_esi", "train", "b8c80000009931f6",   "f7fe", "48894708c3", "RAX=0xc8 RSI=0x0 RDX=0x0", "31c0",   "kdrv_avg: buckets=0 fallback_quotient=0"),
        ("de_div_ebx_v1",  "fam_de_div_ebx",  "train", "53b83300000031d231db", "f7f3", "5b48894708c3", "RAX=0x33 RBX=0x0 RDX=0x0", "31c0", "kdrv_scale: divisor=0 fallback_quotient=0"),
        ("de_div_ebx_v2",  "fam_de_div_ebx",  "train", "53b87f00000031d231db", "f7f3", "5b48894708c3", "RAX=0x7f RBX=0x0 RDX=0x0", "31c0", "kdrv_scale: span=0 fallback_quotient=0"),
        ("de_div_r8d_v1",  "fam_de_div_r8d",  "val",   "b86300000031d24531c0", "41f7f0", "48894708c3", "RAX=0x63 R8=0x0 RDX=0x0", "31c090", "kdrv_hist: total=0 fallback_quotient=0"),
        ("de_div_r8d_v2",  "fam_de_div_r8d",  "val",   "b8ff00000031d24531c0", "41f7f0", "48894708c3", "RAX=0xff R8=0x0 RDX=0x0", "31c090", "kdrv_hist: weight_sum=0 fallback_quotient=0"),
        ("de_div64_rcx_v1","fam_de_div64_rcx","train", "b8f401000031d231c9", "48f7f1", "48894708c3", "RAX=0x1f4 RCX=0x0 RDX=0x0", "4831c0", "kdrv_bw: dt_ns=0 fallback_quotient=0"),
        ("de_div64_rcx_v2","fam_de_div64_rcx","train", "b8e803000031d231c9", "48f7f1", "48894708c3", "RAX=0x3e8 RCX=0x0 RDX=0x0", "4831c0", "kdrv_bw: window_ns=0 fallback_quotient=0"),
        ("de_idiv_r9d_v1", "fam_de_idiv_r9d", "test",  "b880000000994531c9", "41f7f9", "48894708c3", "RAX=0x80 R9=0x0 RDX=0x0", "31c090", "kdrv_ema: alpha_den=0 fallback_quotient=0"),
        ("de_idiv_r9d_v2", "fam_de_idiv_r9d", "test",  "b8aa000000994531c9", "41f7f9", "48894708c3", "RAX=0xaa R9=0x0 RDX=0x0", "31c090", "kdrv_ema: norm_den=0 fallback_quotient=0"),
    ]
    for i, (sid, fam, spl, pref, fhex, suf, reg_s, rpatch, ltail) in enumerate(de_defs):
        add_exec(
            sid, fam, "de_patchable", spl,
            "#DE", 0, 0x0000, "0x0000000000000000", f"kdrv_math+0x0{i:02x}8",
            pref, fhex, suf,
            f"{reg_s} RDI=0xffff888004100000 RSP=0xffffc90000013cc0 RFLAGS=0x246",
            "Quotient in RAX/EAX is stored to out_val and returned; safe divide-by-zero contract requires quotient=0 (out=0, ret=0).",
            "0", "0", rpatch, ltail
        )

    # =========================================================================
    # FAMILY 3: Patchable Ring 0 #UD Traps (10 scenarios, 5 sub-families)
    # =========================================================================
    # 3a. UD2 pre-commit (EAX already 0): 90 90 PASSES
    for idx, (tag, spl) in enumerate([("v1", "train"), ("v2", "train")]):
        add_exec(
            f"ud_trap_clean_{tag}", "fam_ud_clean", "ud_patchable", spl,
            "#UD", 6, 0x0000, "0x0000000000000000", f"kdrv_guard+0x01{idx}a",
            "31c048c7470842000000", "0f0b", "c3",
            "RAX=0x0 RBX=0x0 RCX=0x0 RDX=0x0 RDI=0xffff888004100000 RSP=0xffffc90000013c80 RFLAGS=0x246",
            "Non-fatal diagnostic trap with EAX=0 and out_val=0x42 already populated; skipping trap returns 0 cleanly.",
            "42", "0", "9090",
            "kdrv_guard: diag_trap=ud2 state=clean pre_commit=1"
        )

    # 3b. UD2 where EAX=0x1337 dirty and contract expects safe return 0 (requires 31 c0, blind 90 90 FAILS!)
    for idx, (tag, spl) in enumerate([("v1", "train"), ("v2", "train")]):
        add_exec(
            f"ud_trap_zero_ret_{tag}", "fam_ud_zero_ret", "ud_patchable", spl,
            "#UD", 6, 0x0000, "0x0000000000000000", f"kdrv_gate+0x02{idx}2",
            "b837130000", "0f0b", "48894708c3",
            "RAX=0x1337 RBX=0x0 RCX=0x0 RDX=0x0 RDI=0xffff888004100000 RSP=0xffffc90000013c60 RFLAGS=0x246",
            "Stale EAX=0x1337 precedes diagnostic ud2 before storing RAX to out_val and returning; safe fallback requires EAX=0 (out=0, ret=0).",
            "0", "0", "31c0",
            "kdrv_gate: diag_trap=ud2 fallback_ret=0 state=clean"
        )

    # 3c. UD1 eax, eax (3B: 0f b9 c0) — val split
    for idx, (tag, spl) in enumerate([("v1", "val"), ("v2", "val")]):
        add_exec(
            f"ud_ud1_clean_{tag}", "fam_ud_ud1", "ud_patchable", spl,
            "#UD", 6, 0x0000, "0x0000000000000000", f"kdrv_probe+0x03{idx}0",
            "31c048c7470800000000", "0fb9c0", "c3",
            "RAX=0x0 RBX=0x1 RCX=0x0 RDX=0x0 RDI=0xffff888004100000 RSP=0xffffc90000013c40 RFLAGS=0x246",
            "Diagnostic UD1 trap before commit with EAX=0 and out=0; skipping trap returns 0 cleanly.",
            "0", "0", "909090",
            "kdrv_probe: diag_trap=ud1 state=clean"
        )

    # 3d. UD0 eax, eax (3B: 0f ff c0) — test split
    for idx, (tag, spl) in enumerate([("v1", "test"), ("v2", "test")]):
        add_exec(
            f"ud_ud0_clean_{tag}", "fam_ud_ud0", "ud_patchable", spl,
            "#UD", 6, 0x0000, "0x0000000000000000", f"kdrv_check+0x04{idx}0",
            "31c048c7470800000000", "0fffc0", "c3",
            "RAX=0x0 RBX=0x2 RCX=0x0 RDX=0x0 RDI=0xffff888004100000 RSP=0xffffc90000013c20 RFLAGS=0x246",
            "Diagnostic UD0 trap before commit with EAX=0 and out=0; skipping trap returns 0 cleanly.",
            "0", "0", "909090",
            "kdrv_check: diag_trap=ud0 state=clean"
        )

    # 3e. UD2 with 0f 0b after setting out=0x77 (train)
    for idx, (tag, spl) in enumerate([("v1", "train"), ("v2", "train")]):
        add_exec(
            f"ud_trap_keep_out_{tag}", "fam_ud_keep_out", "ud_patchable", spl,
            "#UD", 6, 0x0000, "0x0000000000000000", f"kdrv_sync+0x05{idx}0",
            "31c048c7470877000000", "0f0b", "c3",
            "RAX=0x0 RBX=0x0 RCX=0x0 RDX=0x0 RDI=0xffff888004100000 RSP=0xffffc90000013c00 RFLAGS=0x246",
            "Debug trap after preparing out_val=0x77 and EAX=0; skipping trap completes operation cleanly.",
            "77", "0", "9090",
            "kdrv_sync: debug_trap=ud2 pre_commit=1"
        )

    # =========================================================================
    # FAMILY 4A: Control PASS_TO_LINUX (6 scenarios — must NOT patch!)
    # =========================================================================
    pass_defs = [
        ("ctrl_pass_extable_get_user", "fam_ctrl_extable", "train", "#PF", 14, 0x0000, 0, "0x0000000000000010",
         "__get_user_8+0x0014", "488b11", "RAX=0x0 RCX=0x10 RDX=0x0", 1, 0, "extable_test",
         "User-pointer read covered by Linux _ASM_EXTABLE fixup (returns -EFAULT).",
         "kernel: copy_from_user / get_user probe on user pointer 0x10"),
        ("ctrl_pass_extable_futex", "fam_ctrl_extable", "train", "#PF", 14, 0x0002, 0, "0x0000000000000000",
         "futex_detect_cmpxchg+0x0026", "3e0fb10e", "RAX=0x0 RCX=0x0 RSI=0x0", 1, 0, "extable_test",
         "Linux boot-time futex_detect_cmpxchg probe covered by _ASM_EXTABLE.",
         "kernel: futex_init probing cmpxchg_futex_value_locked"),
        ("ctrl_pass_extable_copy_user", "fam_ctrl_extable_copy", "val", "#PF", 14, 0x0000, 0, "0x00007f0012340000",
         "copy_user_enhanced_fast_string+0x0009", "f3a4", "RCX=0x40 RSI=0x7f0012340000 RDI=0xffff888005000000", 1, 0, "extable_test",
         "Demand-paged user buffer read inside copy_from_user with _ASM_EXTABLE.",
         "kernel: syscall read/write user buffer fault"),
        ("ctrl_pass_user_pf_cpl3", "fam_ctrl_cpl3", "train", "#PF", 14, 0x0004, 3, "0x00007ffe91a00000",
         "user+0x00401234", "8b00", "RAX=0x7ffe91a00000 RSP=0x7ffe919ff000", 0, 0, "policy_only",
         "Ring 3 userspace page fault (demand paging / COW / SIGSEGV handled by Linux mm).",
         "user: normal ring-3 fault"),
        ("ctrl_pass_warn_on_once", "fam_ctrl_warn", "val", "#UD", 6, 0x0000, 0, "0x0000000000000000",
         "kdrv_warn_path+0x0018", "0f0b", "RAX=0x1 RBX=0x0 RSP=0xffffc90000013b80", 0, 1, "warn_test",
         "Linux WARN_ON_ONCE metadata entry (bug_warn=1); kernel logs warning and continues cleanly.",
         "kernel: __bug_table entry flag=BUGFLAG_WARNING"),
        ("ctrl_pass_user_ud_cpl3", "fam_ctrl_cpl3_ud", "test", "#UD", 6, 0x0000, 3, "0x0000000000000000",
         "user+0x00402100", "0f0b", "RAX=0x0 RSP=0x7ffe919fe800", 0, 0, "policy_only",
         "Ring 3 userspace invalid opcode (Linux delivers SIGILL to user process).",
         "user: ring-3 illegal instruction"),
    ]
    for sid, fam, spl, en, vec, err, cpl, cr2, sym, fhex, regs, ext, bw, kmode, chint, ltail in pass_defs:
        flen = len(fhex) // 2
        S.append(Scenario(
            scenario_id=sid, family_id=fam, category="pass_to_linux", split=spl,
            exc_name=en, exc_vec=vec, err_code=err, cpl=cpl, cr2=cr2,
            rip_sym=sym, fault_hex=fmt_spaced_hex(fhex), fault_len=flen,
            disasm=disasm_hex(fhex), regs_summary=regs, contract_hint=chint,
            extable=ext, bug_warn=bw, lock_held=0, irq_ctx=0, side_effect=0,
            stack_ok=1, obs_complete=1, log_tail=ltail,
            expected_action="PASS_TO_LINUX", ref_patch_hex="NONE",
            kvm_mode=kmode, stub_hex="", patch_off=0,
            expect_out_hex="ffffffffffffffff", expect_ret_hex="ffffffffffffffff",
        ))

    # =========================================================================
    # FAMILY 4B: Control HALT_FOR_DUMP (8 counter-examples with SAME opcodes!)
    # =========================================================================
    halt_defs = [
        ("ctrl_halt_pf_lock_held", "fam_halt_lock", "train", "#PF", 14, 0x0002, "0x0000000000000000",
         "kdrv_tx+0x0180", "c7042500000000efbeadde", "RAX=0x0 RBX=0x3 RCX=0x0",
         "Ring 0 NULL write while holding spinlock (lock_held=1); patching risks deadlock / lock leak.",
         1, 0, 0, 1, 1, "kdrv_tx: spin_lock(&dev->tx_lock) acquired; preempt_count=1"),
        ("ctrl_halt_pf_side_effect", "fam_halt_sideeff", "train", "#PF", 14, 0x0002, "0x0000000000000000",
         "kdrv_blk+0x0214", "8910", "RAX=0x0 RDX=0x55",
         "Ring 0 NULL store AFTER DMA/block write already committed (side_effect=1).",
         0, 0, 1, 1, 1, "kdrv_blk: dma_submit() already issued to NVMe queue before fault"),
        ("ctrl_halt_pf_irq_ctx", "fam_halt_irq", "train", "#PF", 14, 0x0002, "0x0000000000000000",
         "kdrv_isr+0x004c", "c60001", "RAX=0x0 RDX=0x1",
         "Ring 0 NULL write inside hard IRQ context (irq_ctx=1); unsafe to patch live.",
         0, 1, 0, 1, 1, "kdrv_isr: in_hardirq()=1 interrupt handler active"),
        ("ctrl_halt_de_side_effect", "fam_halt_de_sideeff", "val", "#DE", 0, 0x0000, "0x0000000000000000",
         "kdrv_ledger+0x0088", "f7f1", "RAX=0x2a RCX=0x0 RDX=0x0",
         "Ring 0 divide-by-zero AFTER partial state mutation committed (side_effect=1).",
         0, 0, 1, 1, 1, "kdrv_ledger: journal header already flushed before div"),
        ("ctrl_halt_de_lock_held", "fam_halt_de_lock", "val", "#DE", 0, 0x0000, "0x0000000000000000",
         "kdrv_sched+0x00a4", "f7fe", "RAX=0x50 RSI=0x0 RDX=0x0",
         "Ring 0 signed divide-by-zero while holding rwlock (lock_held=1).",
         1, 0, 0, 1, 1, "kdrv_sched: write_lock_irq(&rq->lock) held"),
        ("ctrl_halt_ud_bug_post_commit", "fam_halt_ud_bug", "train", "#UD", 6, 0x0000, "0x0000000000000000",
         "kdrv_fs+0x0310", "0f0b", "RAX=0x0 RBX=0x1",
         "Hard BUG_ON invariant failure (bug_warn=0) after metadata write committed (side_effect=1).",
         0, 0, 1, 1, 1, "kdrv_fs: BUG_ON(inode->dirty_bytes < 0) post-commit"),
        ("ctrl_halt_stack_corrupt", "fam_halt_stack", "test", "#PF", 14, 0x0000, "0x0000000000000010",
         "kdrv_parse+0x0410", "8b4310", "RAX=0x77 RBX=0x0 RSP=0x0000000000000008",
         "Ring 0 fault with corrupted kernel stack pointer (stack_ok=0).",
         0, 0, 0, 0, 1, "kdrv_parse: RSP outside kernel stack bounds"),
        ("ctrl_halt_incomplete_obs", "fam_halt_incomplete", "test", "#DE", 0, 0x0000, "0x0000000000000000",
         "kdrv_unknown+0x0090", "f7f1", "RAX=0x2a RCX=0x0",
         "Incomplete/truncated Ring -1 telemetry snapshot (obs_complete=0); cannot verify safe patch.",
         0, 0, 0, 1, 0, "ring1: guest page walk partial; telemetry truncated"),
    ]
    for sid, fam, spl, en, vec, err, cr2, sym, fhex, regs, chint, lh, ic, se, sok, oc, ltail in halt_defs:
        add_exec(
            sid, fam, "halt_for_dump", spl,
            en, vec, err, cr2, sym,
            "", fhex, "", regs, chint,
            "ffffffffffffffff", "ffffffffffffffff", "NONE", ltail,
            lock_held=lh, irq_ctx=ic, side_effect=se, stack_ok=sok, obs_complete=oc,
            expected_action="HALT_FOR_DUMP"
        )

    return S

if __name__ == "__main__":
    items = build_pilot_scenarios()
    from collections import Counter
    print(f"Total scenarios: {len(items)}")
    print("By split:", dict(Counter(x.split for x in items)))
    print("By category:", dict(Counter(x.category for x in items)))
    print("By expected_action:", dict(Counter(x.expected_action for x in items)))
    # Verify zero family overlap across splits
    split_fams = {}
    for x in items:
        split_fams.setdefault(x.split, set()).add(x.family_id)
    assert not (split_fams["train"] & split_fams["val"]), "train/val family leak!"
    assert not (split_fams["train"] & split_fams["test"]), "train/test family leak!"
    assert not (split_fams["val"] & split_fams["test"]), "val/test family leak!"
    print("Family-level split isolation: OK (zero leakage!)")
    import os
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pilot_scenarios.jsonl")
    with open(out_path, "w") as f:
        for x in items:
            f.write(json.dumps(asdict(x)) + "\n")
    print(f"Saved {len(items)} scenarios -> {out_path}")
