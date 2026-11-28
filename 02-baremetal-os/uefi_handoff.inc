; ============================================================================
; os/uefi_handoff.inc — Contract between uefi_loader.asm (BOOTX64.EFI) and the
; kernel's 64-bit UEFI entry (boot.asm: uefi_entry64).
;
; Boot flow:
;   firmware -> BOOTX64.EFI -> loads \LLM\KERNEL.ELF at its ELF paddr (1 MiB),
;   loads \LLM\MODEL.000.. as ONE contiguous buffer anywhere below
;   UEFI_MAP_LIMIT (no 32-bit limit), ExitBootServices, then jumps to
;   uefi_entry64 in 64-bit long mode with RDI = handoff struct (< 4 GiB).
;
; The handoff region also contains a synthesized Multiboot-1-format info block
; (cmdline + memory map converted from the final UEFI map) so that the
; existing kernel code paths (parse_cmdline, mem_init, highmem_test) run
; unchanged. The MB1 module fields are NOT used: the model base/size are
; passed as full 64-bit values in the handoff struct.
; ============================================================================

; --- Kernel-side header, embedded in .multiboot right after the MB1 header ---
; Loader scans the first 8 KiB of the loaded kernel image for this.
;   +0  dq UEFI_KHDR_MAGIC
;   +8  dq uefi_entry64 (absolute, identity-mapped)
;   +16 dd UEFI_HANDOFF_VERSION
;   +20 dd 0
%define UEFI_KHDR_MAGIC         0x31494645554D4C4C  ; "LLMUEFI1"
%define UEFI_HANDOFF_VERSION    1

; --- Handoff struct (at region + 0) ---
%define HANDOFF_MAGIC           0x31444E41484D4C4C  ; "LLMHAND1"
%define HO_MAGIC                0       ; u64
%define HO_VERSION              8       ; u32
%define HO_STRUCT_SIZE          12      ; u32
%define HO_MBI_PTR              16      ; u64  synthesized MB1 info (< 4 GiB)
%define HO_MODEL_BASE           24      ; u64  physical base of .asmbin
%define HO_MODEL_SIZE           32      ; u64  exact byte size of .asmbin
%define HO_RSDP                 40      ; u64  ACPI RSDP from UEFI config table (0 if none)
%define HO_KERNEL_MIN           48      ; u64
%define HO_KERNEL_MAX           56      ; u64
%define HO_CTX_LEN              64      ; u64  ctx length used for validation
%define HO_ARENA_BYTES          72      ; u64  runtime arena (KV + scratch) validated
%define HO_UEFI_MMAP_PTR        80      ; u64  raw final UEFI memory map
%define HO_UEFI_MMAP_SIZE       88      ; u64
%define HO_UEFI_DESC_SIZE       96      ; u64
%define HO_SIZEOF               104

; --- Handoff region layout (one allocation, LoaderData, below 4 GiB) ---
%define HO_REGION_PAGES         32                  ; 128 KiB
%define HO_OFF_MBI              0x0100              ; MB1 info block (128 B)
%define HO_OFF_CMDLINE          0x0200              ; ASCII cmdline, NUL-terminated
%define HO_CMDLINE_MAX          1024
%define HO_OFF_MB_MMAP          0x1000              ; MB1 mmap entries (24 B each)
%define HO_MB_MMAP_CAP          0xF000
%define HO_OFF_UEFI_MMAP        0x10000             ; raw UEFI memory map buffer
%define HO_UEFI_MMAP_CAP        0x10000

; --- Physical address policy (must match boot.asm identity map + mem_init) ---
%define UEFI_MAP_LIMIT          0x400000000         ; 16 GiB identity-mapped by boot.asm
%define UEFI_HEAP_FLOOR         0x00200000          ; mem_init minimum heap floor
%define UEFI_HEAP_MIN           0x01000000          ; mem_init requires >= 16 MiB region

; Kernel-side boot path identifiers (g_boot_path)
%define BOOT_PATH_MULTIBOOT1    0
%define BOOT_PATH_UEFI          1
