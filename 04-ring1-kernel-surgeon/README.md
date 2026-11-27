# Ring -1 Kernel Surgeon

Bare-metal Type-1 AMD SVM hypervisor (`Ring -1`) running beneath an unmodified Linux kernel (`Ring 0`) with an embedded freestanding **MiniCPM5-2B (`Q4_K_M`)** diagnostic and live-patching engine.

When Ring 0 triggers a fatal exception (`#PF` NULL-pointer dereference, `#DE` divide-by-zero, or `#UD` trap), AMD SVM hardware traps into Ring `-1` **before** Linux `oops()` / `panic()` can execute. Ring `-1` walks guest page tables, reads the faulting machine instruction and registers, runs **MiniCPM5-2B `Q4_K_M`** across 24 bare-metal LAPIC SMP cores inside an NPT-isolated arena, streams a surgical HUD directly onto the live `efifb` framebuffer, statically validates the proposed x86-64 machine-code patch, writes it into guest physical memory (bypassing guest `CR0.WP` and read-only PTEs), and resumes Linux via `VMRUN` without rebooting — or safely delegates benign exceptions (`PASS_TO_LINUX`) or freezes unrecoverable hazards (`HALT_FOR_DUMP`).

---

## Architecture Highlights

- **Bare-Metal Hypervisor (`build/surgeon.elf`, 75,424 B):** NASM (`boot.asm`, `svm_asm.asm`, `ap_trampoline.asm`) + freestanding zero-libc, zero-libm C (`-ffreestanding -nostdlib`). Fast-path `#VMEXIT` handlers compile with `-mno-mmx -mno-sse -mno-sse2 -mno-avx -mno-red-zone`; `XSAVE`/`XRSTOR` isolates guest FPU/SSE/AVX state around Ring `-1` AVX2+FMA+F16C inference (`hv/llm_engine.c`).
- **24-Core Bare-Metal SMP Worker Pool (`hv/smp.c`):** Ring `-1` boots up to 24 physical/virtual LAPIC cores (`INIT-SIPI-SIPI`) for parallel quantized matmul while Linux runs on BSP CPU 0 (`nr_cpus=1` or isolated APs).
- **Hardware AMD NPT Isolation (`hv/npt.c`):** Hypervisor ELF (`0x200000..0x3FFFFF`), 64 MiB KV/scratch arena (`0x70000000..0x73FFFFFF`), and the 2 GiB read-only NVDIMM model (`.r1m` @ `0x100000000`) are all **unmapped (`Present=0`) in AMD NPT** and hidden from the Linux E820 map.
- **3-Action Safety Policy:** Every fault yields a strict 3-line decision (`DIAGNOSIS`, `ACTION`, `PATCH`) with no `<think>` output:
  - `PATCH_BYTES` — exact-length x86-64 patch bytes verified by static instruction decoding (`hv/surgeon.c`, `hv/x86_decode.c`) before writing guest RAM.
  - `PASS_TO_LINUX` — re-injects userspace (`CPL=3`) faults, `_ASM_EXTABLE` probes (`copy_from_user` / `get_user`), and `WARN_ON_ONCE` natively into Linux.
  - `HALT_FOR_DUMP` — freezes the guest safely when `lock_held=1`, `irq_ctx=1`, `side_effect=1`, `stack_ok=0`, or `obs_complete=0`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the physical memory map, VMCB setup, and intercept protocol.

---

## Evaluation & Verification Results

### 1. Held-Out Unseen Scenarios (`dataset/scenarios_600.jsonl`: `356 train / 122 val / 122 test`, zero family overlap)

Evaluated with `training/eval_model.py` + `verifier/kvm_verifier.py` (3-tier verifier: L1 schema + Capstone x86-64 safety check, L2 safety contract policy, L3 isolated single-level QEMU/KVM Linux 5.15 Ring 0 execution via `guest/bench_mod/surgeon_bench.ko`):

| Model (`MiniCPM5-2B`) | Test Pass@1 (`n=122`) | Val Pass@1 (`n=122`) | Avg Reward (Test) | Dangerous Continue | Excess Halt | `<think>` Leaks |
|---|---:|---:|---:|---:|---:|---:|
| **Base** (`bf16`) | `16.39%` *(20/122)* | `16.39%` *(20/122)* | `-0.5984` | `8.20%` *(10)* | `7.38%` *(9)* | `0` |
| **SFT** (LoRA merged) | `95.90%` *(117/122)* | `97.54%` *(119/122)* | `+0.9180` | `0.00%` *(0)* | `0.00%` *(0)* | `0` |
| **GRPO** (KVM-verified RL) | **`97.54%`** *(119/122)* | **`98.36%`** *(120/122)* | **`+0.9508`** | **`0.00%`** *(0)* | **`0.00%`** *(0)* | **`0`** |

Per-category test breakdown (`n=122`, GRPO):
- `pf_patchable`: `29/32` pass (`0` dangerous)
- `de_patchable`: `30/30` pass (`0` dangerous)
- `ud_patchable`: `30/30` pass (`0` dangerous)
- `pass_to_linux`: `15/15` pass (`0` dangerous)
- `halt_for_dump`: `15/15` pass (`0` dangerous)

### 2. End-to-End Bare-Metal Ring `-1` Nested-SVM Verification

Verified with `scripts/verify_stage3.py` under QEMU/KVM (`-machine q35,accel=kvm,nvdimm=on -cpu EPYC,+svm -smp 24`) running `MiniCPM5-2B Q4_K_M` (`minicpm5-2b-surgeon.r1m`, 2 GiB NVDIMM) inside Ring `-1` beneath Linux `5.15`:
- `echo pf > /proc/surgeon_crash`: intercepts `#PF` (`c7 04 25 00 00 00 00 ef be ad de`, 11 B) -> applies `11x 0x90` NOP sled -> Linux survives (`SURVIVED NULL-DEREF!`).
- `echo de > /proc/surgeon_crash`: intercepts `#DE` (`f7 f1`, `div ecx`, 2 B) -> applies `31 c0` (`xor eax, eax`) -> Linux survives with `quotient=0`.
- `echo ud > /proc/surgeon_crash`: intercepts `#UD` (`0f 0b`, `ud2`, 2 B) -> applies `90 90` -> Linux survives (`SURVIVED UD2 TRAP!`).
- `echo extable > /proc/surgeon_crash`: intercepts `_ASM_EXTABLE` `get_user(0x10)` -> outputs `PASS_TO_LINUX` -> Linux handles natively (`rc=-14`).
- Continuous `uptime` and `boot_id` verified across all recoveries (`VMMCALL` reports `heals=3 l1_cores=24 llm_ready=1`).
- `echo danger > /proc/surgeon_crash`: intercepts spinlock-held `#PF` (`lock_held=1`) -> outputs `HALT_FOR_DUMP` -> guest frozen cleanly without corruption.

---

## Repository Layout

- `ARCHITECTURE.md` — technical design, physical memory map, NPT isolation, SVM intercepts.
- `hv/` — bare-metal Ring `-1` AMD SVM hypervisor (`boot.asm`, `svm_asm.asm`, `ap_trampoline.asm`, `main.c`, `npt.c`, `smp.c`, `surgeon.c`, `llm_engine.c`, `x86_decode.c`, `fb.c`, `serial.c`, `linux_boot.c`).
- `guest/crash_mod/` — **Ring 0 fault-injection test module** (`surgeon_crash.c`, builds `surgeon_crash.ko`, exposes `/proc/surgeon_crash` with `pf | de | ud | extable | danger | status | hud` triggers).
- `guest/bench_mod/` — **single-level KVM Ring 0 postcondition verifier module** (`surgeon_bench.c`, builds `surgeon_bench.ko`, exposes `/proc/surgeon_bench` for isolated canary/lock/return/repeat-call checks).
- `dataset/` — 48-scenario pilot (`scenarios.py`, `pilot_scenarios.jsonl`), 600-scenario corpus (`scenarios_600.py`, `scenarios_600.jsonl`), and DeepSeek API distillation pipeline (`generate_deepseek.py`).
- `verifier/` — 3-tier single-level KVM verifier (`kvm_verifier.py`) with Capstone static checks, policy enforcement, QEMU/KVM execution, and SQLite cache.
- `training/` — LoRA SFT (`sft_minicpm5.py`), GRPO (`grpo_minicpm5.py`), model evaluation (`eval_model.py`), TensorBoard logger (`tb_logger.py`), and YAML configs (`training/configs/`).
- `dashboard/` — FastAPI live training & KVM-verifier dashboard (`server.py`).
- `scripts/` — initramfs/ISO builders (`build_initramfs.sh`, `build_verifier_initramfs.sh`, `mkiso.sh`), QEMU/VNC launchers (`run_qemu.sh`, `run_demo_vnc.sh`), GGUF/`.r1m` packers (`export_gguf.sh`, `pack_r1model.py`), host LLM test (`test_r1_llm.c`), and end-to-end verification (`verify_stage1.py`, `verify_stage3.py`).
- `Makefile` — builds `build/surgeon.elf`, `guest/crash_mod/surgeon_crash.ko`, `build/initramfs.cpio.gz`, and `build/ring1-surgeon.iso`.

---

## Prerequisites

- **Hardware / Host:** x86-64 AMD CPU with SVM + NPT (`nested=1` in `kvm_amd`) and `/dev/kvm` access for nested QEMU/KVM runs.
- **Build Tools:** `gcc`, `nasm`, `ld`, `grub-mkstandalone`, `xorriso`, `busybox`, `cpio`, `gzip`, `python3`, OVMF UEFI firmware (`/usr/share/OVMF/OVMF_CODE.fd`), and target Linux kernel headers (`KDIR ?= /lib/modules/5.15.0-185-generic/build`) plus `guest/vmlinuz`.
- **Python (for verifier / training / dashboard):** `capstone`, `fastapi`, `uvicorn`, `pyyaml`, `httpx`, `torch`, `transformers`, `trl`, `peft`, `datasets`, `tensorboard`, `gguf`.

---

## Build & Run

### 1. Build Hypervisor, Fault-Injection Test Module, and Bootable ISO

```bash
cd 04-ring1-kernel-surgeon
# Place or symlink the guest kernel bzImage at guest/vmlinuz
make hv          # Build only build/surgeon.elf (75 KB Ring -1 hypervisor)
make all         # Build surgeon.elf, guest/crash_mod/surgeon_crash.ko, initramfs, and ISO
```

### 2. Launch Interactive QEMU Nested-SVM + VNC

```bash
# Default VNC binds 127.0.0.1:5901 (set VNC_BIND=0.0.0.0 for remote access)
./scripts/run_qemu.sh

# Or run the Stage 3 demo with a packed .r1m NVDIMM model:
MODEL_R1M=models/minicpm5-2b-surgeon.r1m ./scripts/run_demo_vnc.sh surgeon
```

Inside the guest shell (`tty0` / `ttyS0`):

```sh
echo pf      > /proc/surgeon_crash   # Ring 0 NULL-pointer dereference (#PF)
echo de      > /proc/surgeon_crash   # Ring 0 divide-by-zero (#DE)
echo ud      > /proc/surgeon_crash   # Ring 0 invalid opcode (#UD / ud2)
echo extable > /proc/surgeon_crash   # Benign _ASM_EXTABLE probe (PASS_TO_LINUX)
echo danger  > /proc/surgeon_crash   # Spinlock-held #PF (HALT_FOR_DUMP)
echo status  > /proc/surgeon_crash   # Query Ring -1 VMMCALL counters
uptime && dmesg | tail -n 15
```

### 3. Verifier, Distillation, SFT/GRPO & Dashboard

```bash
# 1. Verify single-level KVM verifier (48 golden + adversarial checks)
python3 verifier/kvm_verifier.py --selftest

# 2. Run DeepSeek API distillation (expects DEEPSEEK_API_KEY in env or .env)
python3 dataset/generate_deepseek.py --model deepseek-flash --limit 600 --out-dir runs/dataset_600

# 3. Validate SFT & GRPO configs (dry-run)
python3 training/sft_minicpm5.py --dry-run
python3 training/grpo_minicpm5.py --dry-run

# 4. Evaluate a checkpoint on the held-out test split
python3 training/eval_model.py --model-path runs/grpo-minicpm5-2b/merged --tag grpo --split test --out-json runs/eval_grpo_test.json

# 5. Launch TensorBoard (:6006) + Live Web Dashboard (:8780)
./scripts/run_dashboard.sh
```
