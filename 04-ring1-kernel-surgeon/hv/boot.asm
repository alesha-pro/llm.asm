; ============================================================================
; hv/boot.asm — Multiboot2 Entry -> 64-bit Long Mode (0..8 GiB identity map)
; ============================================================================
bits 32

MB2_MAGIC equ 0xE85250D6
MB2_ARCH  equ 0

section .multiboot align=8
mb2_hdr_start:
    dd MB2_MAGIC
    dd MB2_ARCH
    dd mb2_hdr_end - mb2_hdr_start
    dd -(MB2_MAGIC + MB2_ARCH + (mb2_hdr_end - mb2_hdr_start))

    align 8
    dw 5            ; framebuffer request
    dw 0
    dd 20
    dd 1280
    dd 720
    dd 32

    align 8
    dw 6            ; module alignment
    dw 0
    dd 8

    align 8
    dw 0            ; end tag
    dw 0
    dd 8
mb2_hdr_end:

section .text
global _start
extern hv_main
extern __bss_start
extern __bss_end

_start:
    cli
    mov esp, stack32_top

    mov [mb2_magic_saved], eax
    mov [mb2_info_saved], ebx

    ; Zero .bss
    mov edi, __bss_start
    mov ecx, __bss_end
    sub ecx, edi
    shr ecx, 2
    xor eax, eax
    rep stosd

    ; Build 4-level identity page tables covering 0 .. 8 GiB via 2 MiB pages
    ; (includes low RAM, MMIO/FB 2..4G, and 2 GiB NVDIMM model at 4..6G!)
    mov eax, host_pdpt
    or  eax, 0x03
    mov [host_pml4], eax

    mov ecx, 0
.pdpt_loop:
    mov eax, ecx
    shl eax, 12
    add eax, host_pd
    or  eax, 0x03
    mov [host_pdpt + ecx * 8], eax
    mov dword [host_pdpt + ecx * 8 + 4], 0
    inc ecx
    cmp ecx, 8
    jl .pdpt_loop

    ; Fill 8 * 512 = 4096 2-MiB page entries (0x00000000 .. 0x1FFFFFFFF)
    mov edi, host_pd
    xor ebx, ebx            ; low 32 of physical address
    xor edx, edx            ; high 32 of physical address
    mov ecx, 4096
.pd_loop:
    mov eax, ebx
    or  eax, 0x83           ; Present | Writable | PageSize (2 MiB)
    mov [edi], eax
    mov [edi + 4], edx
    add ebx, 0x00200000
    adc edx, 0
    add edi, 8
    dec ecx
    jnz .pd_loop

    mov eax, host_pml4
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

    ; Enable CR0.PG | PE | NE | MP (clear NW, CD, EM, TS)
    mov eax, cr0
    or  eax, (1 << 31) | (1 << 0) | (1 << 5) | (1 << 1)
    and eax, ~((1 << 29) | (1 << 30) | (1 << 2) | (1 << 3))
    mov cr0, eax

    lgdt [gdt64_ptr]
    jmp 0x08:_start64

bits 64
_start64:
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax

    mov rsp, stack64_top
    and rsp, ~0xF

    ; Populate 64-bit TSS descriptor in GDT[3..4] (selector 0x18) and load TR
    lea rax, [rel host_tss]
    mov rbx, rax
    mov [rel gdt64_tss + 2], bx
    shr rbx, 16
    mov [rel gdt64_tss + 4], bl
    mov [rel gdt64_tss + 7], bh
    shr rbx, 16
    mov [rel gdt64_tss + 8], ebx
    mov ax, 0x18
    ltr ax

    mov edi, [rel mb2_magic_saved]
    mov esi, [rel mb2_info_saved]
    call hv_main

.halt:
    cli
    hlt
    jmp .halt

section .data
align 8
mb2_magic_saved: dd 0
mb2_info_saved:  dd 0

global gdt64_ptr
align 16
gdt64:
    dq 0x0000000000000000   ; 0x00: null
    dq 0x00AF9A000000FFFF   ; 0x08: 64-bit code
    dq 0x00CF92000000FFFF   ; 0x10: 64-bit data
gdt64_tss:
    dw 0x0067
    dw 0x0000
    db 0x00
    db 0x89
    db 0x00
    db 0x00
    dd 0x00000000
    dd 0x00000000
gdt64_end:

gdt64_ptr:
    dw gdt64_end - gdt64 - 1
    dq gdt64

section .bss
global host_pml4
global host_pdpt
global host_pd
global host_pd_high

align 4096
host_pml4:    resb 4096
host_pdpt:    resb 4096
host_pd:      resb 4096 * 8   ; 8 GiB identity map
host_pd_high: resb 4096

align 64
host_tss:  resb 128

align 16
stack32_bottom: resb 4096
stack32_top:

align 16
stack64_bottom: resb 65536
stack64_top:
