# Ring -1 Kernel Surgeon — Architecture

> **Target Platform:** x86-64 with AMD SVM + NPT (verified under nested `kvm_amd=1` on QEMU/KVM + OVMF UEFI + VNC)  
> **Safety Boundary:** All tests run inside **nested AMD SVM under QEMU/KVM** (`-cpu EPYC,+svm`). The host OS is never modified or rebooted.

---

## 1. Core Concept

**Ring -1 Kernel Surgeon** is a bare-metal Type-1 AMD SVM hypervisor (`Ring -1`) that sits transparently underneath an unmodified Linux kernel (`Ring 0`).

When Ring 0 triggers a fatal exception (e.g., `#PF` NULL-pointer dereference or `#UD` trap in kernel mode), AMD SVM hardware traps into Ring `-1` **before** Linux `do_page_fault` / `oops` / `panic()` can execute:
1. Ring `-1` walks the guest 4-level page tables (`CR3`) to resolve the faulting `RIP` to a guest physical address (GPA).
2. Ring `-1` reads the raw faulting machine instruction and captures full guest CPU state (GPRs, CRs, RFLAGS, stack).
3. Ring `-1` renders a live **Surgical HUD Overlay** directly into the shared linear `efifb` / `vesafb` framebuffer (visible over VNC, completely bypassing guest DRM/VT locks).
4. Ring `-1` decides a live machine-code patch:
   - **Stage 1 (Proof of Ring `-1`):** Deterministic x86-64 instruction-length decode (`hv/x86_decode.c`) + live physical-memory patching of the faulting instruction in guest `.text` (bypassing guest `CR0.WP` and read-only PTEs) with an exact-length `0x90` NOP sled.
   - **Stage 2/3 (AI Surgeon):** `surgeon_handle_fault()` (`hv/surgeon.c`) invokes a freestanding bare-metal local LLM (`MiniCPM5-2B Q4_K_M` packed in `.r1m` over a 2 GiB read-only NVDIMM, executed by `hv/llm_engine.c` across a 24-core LAPIC SMP pool with `XSAVE`/`XRSTOR` FPU isolation) and statically validates the proposed machine-code patch before writing guest physical memory.
5. Ring `-1` resumes the guest via `VMRUN` (or cleanly delegates via `PASS_TO_LINUX` or freezes safely via `HALT_FOR_DUMP`). When patched, Linux never panics and continues running with its original `uptime` and processes intact.

---

## 2. Boot Chain & Execution Flow

```text
QEMU (-enable-kvm -cpu EPYC,+svm -vnc :1)
  └── OVMF UEFI Firmware (sets GOP linear 32-bit BGRX framebuffer)
        └── GRUB2 UEFI (Multiboot2)
              ├── Kernel:   /boot/surgeon.elf       (Ring -1 Hypervisor, 64-bit ELF)
              ├── Module 1: /boot/vmlinuz           (Unmodified Linux 5.15 bzImage)
              ├── Module 2: /boot/initramfs.cpio.gz (Minimal BusyBox + surgeon_crash.ko)
              └── [Stage 2] /boot/model.asmbin      (Quantized LLM weights in reserved RAM)
```

Inside `surgeon.elf`:
1. **`hv/boot.asm` (32-bit -> 64-bit Long Mode):**
   - Multiboot2 header requests linear 32-bit framebuffer (`1280x720x32` preferred; accepts GOP native resolution) and page-aligned modules.
   - Sets up 4-level identity-mapped host page tables (`0..4 GiB` via 2 MiB pages, plus dynamic mapping if framebuffer > 4 GiB), GDT with 64-bit code/data + TSS, transitions to 64-bit Long Mode (`hv_main`).
   - Compiled with `-mno-mmx -mno-sse -mno-sse2 -mno-avx -mno-red-zone` so Ring `-1` never clobbers guest FPU/XMM/AVX state during `#VMEXIT`.
2. **Multiboot2 Parsing & Hardware Init (`hv/main.c`, `hv/serial.c`, `hv/fb.c`):**
   - Initializes COM1 (`0x3F8`, 115200 8N1) serial logging.
   - Parses Multiboot2 tags: framebuffer (`addr`, `width`, `height`, `pitch`, `bpp`, masks), E820 memory map, ACPI RSDP (v1/v2), and modules (`vmlinuz`, `initramfs`).
   - Renders initial Ring `-1` boot banner onto the linear framebuffer.
3. **Nested Page Table (NPT) & Memory Isolation (`hv/npt.c`):**
   - Builds a 4-level AMD NPT covering `0..4 GiB` (plus any >4 GiB framebuffer BAR) with `P|W|U` (`0x07` / `0x87`, since AMD NPT treats all guest accesses as User-mode).
   - **Ring `-1` Protection:** The hypervisor physical arena (`0x00200000 .. 0x003FFFFF`, 2 MiB) is:
     1. Marked `E820_RESERVED` (type 2) in Linux `boot_params.e820_table`.
     2. **Unmapped (`Present = 0`) in NPT**, so even a rogue Ring 0 kernel write cannot corrupt Ring `-1` code, VMCB, HSAVE, or future LLM weights.
4. **Linux 64-bit Direct Boot Loader (`hv/linux_boot.c`):**
   - Inspects `vmlinuz` bzImage `setup_header` (`0x01F1`), copies the protected-mode kernel payload to `pref_address` (`0x01000000`), relocates `initramfs.cpio.gz` to `0x08000000` (using bounce-safe copy to avoid overlap with GRUB module placement).
   - Builds `boot_params` ("zero page") at `0x00600000` with:
     - `screen_info` populated with `VIDEO_TYPE_EFI` (`0x70`) + exact GOP framebuffer geometry (`lfb_base`, `lfb_width`, `lfb_height`, `lfb_depth=32`, `lfb_linelength`, `lfb_size`, RGB bit positions/sizes) so Linux built-in `efifb` binds immediately.
     - `acpi_rsdp_addr` copied from Multiboot2 ACPI tag.
     - `e820_table` copied from Multiboot2 mmap with `0x00200000..0x003FFFFF` carved out as `E820_RESERVED`.
     - Command line at `0x00601000`: `console=tty0 console=ttyS0,115200 earlyprintk=serial,ttyS0,115200 nokaslr panic=0`.
   - Builds initial 64-bit identity guest page tables (`0..4 GiB`) at `0x00610000` for `startup_64`.
5. **AMD SVM Activation & `VMRUN` Loop (`hv/svm.c`, `hv/svm_asm.asm`):**
   - Verifies CPUID `0x80000001` ECX[2] (`SVM`) and `MSR_VM_CR` (`0xC0010114`) SVMDIS=0.
   - Enables `MSR_EFER.SVME` (bit 12), programs `MSR_VM_HSAVE_PA` (`0xC0010117`) with 4KB-aligned host save area.
   - Configures 4KB-aligned `VMCB`:
     - `guest_asid = 1`, `np_enable = 1`, `n_cr3 = npt_pml4_pa`, `g_pat = 0x0007040600070406ULL`.
     - Intercepts: `VMRUN` (mandatory), `VMMCALL`, `CPUID`, `MSR_PROT` (MSRPM intercepts ONLY `RDMSR/WRMSR` on `MSR_EFER` `0xC0000080`), `SHUTDOWN`, and exceptions `#UD` (vec 6), `#DF` (vec 8), `#PF` (vec 14).
     - Guest state initialized for Linux 64-bit entry: `RIP = 0x01000200` (`startup_64`), `RSI = 0x00600000` (`&boot_params`), `CR3 = guest_init_pml4`, `EFER = LME|LMA|NXE|SCE|SVME`.
   - Runs `CLGI -> VMSAVE host -> VMLOAD guest -> VMRUN -> VMSAVE guest -> VMLOAD host` loop.

---

## 3. Physical Memory Map (Stage 1, 2 GiB Guest RAM)

| Physical Range | Size | NPT Mapping | Guest E820 | Owner / Usage |
|---|---:|---|---|---|
| `0x0000_0000 .. 0x000F_FFFF` | 1 MiB | 1:1 `P\|W\|U` | BIOS/Low | Low RAM / EBDA / ACPI |
| `0x0010_0000 .. 0x001F_FFFF` | 1 MiB | 1:1 `P\|W\|U` | Usable | Guest low extended RAM |
| `0x0020_0000 .. 0x003F_FFFF` | 2 MiB | **UNMAPPED (`0`)** | **`E820_RESERVED`** | **Ring `-1` Hypervisor (ELF, stacks, VMCB, HSAVE, MSRPM, NPT)** |
| `0x0060_0000 .. 0x0063_FFFF` | 256 KiB | 1:1 `P\|W\|U` | Usable | `boot_params`, cmdline, initial guest PT |
| `0x0100_0000 .. 0x03FF_FFFF` | 48 MiB | 1:1 `P\|W\|U` | Usable | Linux `vmlinuz` kernel payload & decompression arena |
| `0x0800_0000 .. 0x09FF_FFFF` | 32 MiB | 1:1 `P\|W\|U` | Usable | `initramfs.cpio.gz` |
| `0x0A00_0000 .. 0x7DFF_FFFF` | ~1.8 GiB | 1:1 `P\|W\|U` | Usable | Guest Linux RAM |
| `0x8000_0000 .. 0xFFFF_FFFF` | 2 GiB | 1:1 `P\|W\|U` (+MMIO) | Reserved/MMIO | GOP Linear Framebuffer, PCI MMIO, IOAPIC, LAPIC |

*(In Stage 2, QEMU RAM increases to 6–8 GiB and an additional high physical arena is carved out of E820 and unmapped from NPT to house quantized LLM weights + KV cache.)*

---

## 4. Surgical Intercept & Live-Patch Protocol

### Fast-Path vs. Surgeon-Path Decision
- **`CPUID` (`0x72`):** Execute host `cpuid`, clear `SVM` bit (`ECX[2]`) on `EAX == 0x80000001` so Linux stays unaware of SVM; advertise `"Ring1Surgeon"` on leaf `0x40000000`; advance `RIP = next_rip`.
- **`MSR` (`0x7C` on `0xC0000080` `EFER`):**
  - `RDMSR`: return `vmcb.save.efer & ~EFER_SVME` (transparently hides `SVME`).
  - `WRMSR`: store `val | EFER_SVME` into `vmcb.save.efer` (prevents guest from clearing `SVME` and triggering `VMEXIT_INVALID`).
- **`#PF` (`0x4E`, Vector 14):**
  - If `CPL != 0` **or** `CR2 >= 0x1000` (normal userspace page fault, demand paging, COW, `copy_from_user`): fast re-inject via `VMCB.eventinj` (`vector=14, type=3, EV=1, V=1, err=exitinfo1`) and `vmcb.save.cr2 = exitinfo2`.
  - If `CPL == 0` **and** `CR2 < 0x1000` (Ring 0 NULL-pointer dereference, e.g. `surgeon_crash.ko` or `sysrq-trigger`): **invoke Surgeon pipeline**.
- **`#UD` (`0x46`, Vector 6):**
  - If `CPL != 0`: fast re-inject `#UD` to guest.
  - If `CPL == 0`: check if faulting bytes are `0F 0B` (`ud2`) from our crash target / Ring 0 fault -> **invoke Surgeon pipeline** (if Linux `BUG()`/`WARN_ON` outside test or configured passthrough, can optionally re-inject; in Stage 1 Ring 0 `#UD` is healed cleanly when triggered via `surgeon_crash` or test).
- **`VMMCALL` (`0x81`):** Diagnostic hypercall (`RAX = 0x53555247` `"SURG"` returns heal count & hypervisor status; `RAX = 0x53555248` forces HUD redraw).

### Stage 2 Extension Point (`hv/surgeon.h`)
```c
typedef struct {
    uint32_t vector;           // 14 (#PF) or 6 (#UD)
    uint64_t error_code;
    uint64_t cr2;
    uint64_t rip_gva;
    uint64_t rip_gpa;
    uint8_t  orig_bytes[16];
    uint8_t  insn_len;
    GuestRegs *regs;
    Vmcb *vmcb;
} SurgeonContext;

typedef struct {
    uint8_t  patch_bytes[16];
    uint8_t  patch_len;
    const char *reason;        // Shown in Ring -1 HUD overlay
} SurgeonPatch;

bool surgeon_decide_patch(const SurgeonContext *ctx, SurgeonPatch *out);
```
Stage 1 implements `surgeon_decide_patch()` deterministically (exact-length NOP sled verified by `x86_decode_len()`). Stage 2 swaps only this function to call the bare-metal LLM + static machine-code verifier.

---

## 5. Verification Plan (Stage 1)

1. Build `surgeon.elf`, `surgeon_crash.ko`, `initramfs.cpio.gz`, and UEFI bootable `ring1-surgeon.iso` via `make`.
2. Boot under QEMU nested AMD SVM (`scripts/run_qemu.sh` with `-enable-kvm -cpu EPYC,+svm -vnc :1`).
3. Confirm in serial log + VNC framebuffer:
   - Ring `-1` initializes AMD SVM + NPT and launches Linux `5.15.0-185-generic` via `VMRUN`.
   - Linux mounts `efifb` on the shared linear framebuffer and reaches shell.
   - `insmod /lib/modules/surgeon_crash.ko` loads cleanly.
   - `echo pf > /proc/surgeon_crash` triggers a real Ring 0 NULL-pointer dereference (`*(volatile uint32_t *)0 = 0xDEADBEEF`).
   - Ring `-1` intercepts `#PF`, resolves `RIP` via guest PT walk, draws the **Ring `-1` Surgical HUD Overlay** directly onto the live VNC framebuffer, live-patches the faulting instruction bytes in guest `.text` physical RAM to `NOP`s, and resumes `VMRUN`.
   - Linux **survives without panicking**, prints confirmation to both VNC console and `ttyS0`, `uptime` continues ticking, and `echo ud > /proc/surgeon_crash` similarly heals a Ring 0 `#UD` (`ud2`) trap live.
