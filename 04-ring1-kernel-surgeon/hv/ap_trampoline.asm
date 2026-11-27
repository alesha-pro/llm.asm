; ============================================================================
; hv/ap_trampoline.asm — Relocatable 16->32->64 AP Startup Trampoline @ 0x8000
; ============================================================================
section .rodata
align 16

global ap_trampoline_start
global ap_trampoline_end

TRAMP_BASE equ 0x8000
; Control block placed at 0x8F00 by BSP before SIPI:
;   [0x8F00] u64 host_pml4_pa
;   [0x8F08] u64 ap_entry64_addr
;   [0x8F10] u64 stack_top_pa
;   [0x8F18] u64 tid
;   [0x8F20] u16 gdt_limit / [0x8F22] u64 gdt_base (10-byte GDTR)

bits 16
ap_trampoline_start:
    cli
    cld
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax

    ; Load temporary 32-bit flat GDT embedded in trampoline
    lgdt [TRAMP_BASE + (tramp_gdtr32 - ap_trampoline_start)]

    ; Enter 32-bit protected mode
    mov eax, cr0
    or  eax, 1
    mov cr0, eax
    jmp 0x08:(TRAMP_BASE + (ap_pm32 - ap_trampoline_start))

bits 32
ap_pm32:
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov ss, ax

    ; Load BSP host_pml4 into CR3
    mov eax, [0x8F00]
    mov cr3, eax

    ; Enable PAE
    mov eax, cr4
    or  eax, (1 << 5)
    mov cr4, eax

    ; Enable EFER.LME | EFER.NXE
    mov ecx, 0xC0000080
    rdmsr
    or  eax, (1 << 8) | (1 << 11)
    wrmsr

    ; Enable paging + FPU MP
    mov eax, cr0
    or  eax, (1 << 31) | (1 << 0) | (1 << 5) | (1 << 1)
    and eax, ~((1 << 29) | (1 << 30) | (1 << 2) | (1 << 3))
    mov cr0, eax

    ; Load BSP 64-bit GDT and jump to 64-bit mode
    lgdt [0x8F20]
    jmp 0x08:(TRAMP_BASE + (ap_lm64 - ap_trampoline_start))

bits 64
ap_lm64:
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax

    mov rsp, [0x8F10]
    and rsp, ~0xF
    mov rdi, [0x8F18]       ; tid
    mov rax, [0x8F08]       ; ap_c_entry(uint64_t tid)
    call rax

.ap_hlt:
    cli
    hlt
    jmp .ap_hlt

align 8
tramp_gdt32:
    dq 0x0000000000000000
    dq 0x00CF9A000000FFFF   ; 0x08: 32-bit flat code
    dq 0x00CF92000000FFFF   ; 0x10: 32-bit flat data
tramp_gdtr32:
    dw (tramp_gdtr32 - tramp_gdt32 - 1)
    dd TRAMP_BASE + (tramp_gdt32 - ap_trampoline_start)

ap_trampoline_end:
