#!/usr/bin/env python3
"""
Qwen3.5-2B golden check for the bare-metal kernel (needs KVM, AVX2+FMA+F16C, 4G of guest RAM).
Builds the qwen35 kernel (make os-qwen35), boots it with the 2B .asmbin v3 module at each SMP count and checks
PROMPT_TOKENS == golden prompt ids and GEN_TOKENS == golden ids up to the first EOS (scripts/qwen35/golden_2b.json,
HF float32 reference). Exit 0 = all checks passed.
usage: python3 scripts/verify_qwen35.py [--model models/qwen35-2b-q8_0-v3.asmbin] [--smp 1,4,8] [--no-build]
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ELF = ROOT / "build/asm-llm-os-qwen35.elf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen35-2b-q8_0-v3.asmbin")
    ap.add_argument("--golden", default="scripts/qwen35/golden_2b.json")
    ap.add_argument("--smp", default="1,8", help="comma separated vCPU counts")
    ap.add_argument("--ram", default="4G")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    g = json.loads((ROOT / args.golden).read_text())
    want_prompt = g["prompt_token_ids"]
    want_gen = g["generated_token_ids"][: g["first_eos_index_in_generated"]]
    prompt = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", g["prompt_text"], re.S).group(1)

    if not args.no_build:
        subprocess.run(["make", "-C", str(ROOT), "os-qwen35"], check=True, stdout=subprocess.DEVNULL)
    ok = True
    for smp in (int(x) for x in args.smp.split(",")):
        cmdline = f'prompt="{prompt}" max_tokens={len(want_gen) + 2} think=0 print_tokens=1 shutdown=1'
        cmd = ["qemu-system-x86_64", "-enable-kvm", "-cpu", "host", "-m", args.ram, "-smp", str(smp),
               "-kernel", str(ELF), "-initrd", str(ROOT / args.model), "-append", cmdline, "-nographic"]
        out = subprocess.run(cmd, capture_output=True, timeout=args.timeout, cwd=ROOT).stdout.decode("utf-8", "replace")
        mp = re.search(r"^PROMPT_TOKENS:([0-9,]+)", out, re.M)
        mg = re.search(r"^GEN_TOKENS:([0-9,]+)", out, re.M)
        got_p = [int(x) for x in mp.group(1).split(",")] if mp else None
        got_g = [int(x) for x in mg.group(1).split(",")] if mg else None
        p_ok, g_ok = got_p == want_prompt, got_g is not None and got_g[: len(want_gen)] == want_gen
        ok &= p_ok and g_ok
        print(f"smp={smp}: prompt_tokens {'OK' if p_ok else 'FAIL ' + str(got_p)}  "
              f"gen_tokens {'OK' if g_ok else 'FAIL ' + str(got_g)} (want {want_gen})")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
