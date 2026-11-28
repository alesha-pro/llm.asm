; ============================================================================
; os/uefi_loader.asm — BOOTX64.EFI: 64-bit UEFI model + kernel loader
; Pure x86-64 NASM (nasm -f bin), hand-built PE32+ image, no linker, no C.
;
; Purpose: load a multi-GiB .asmbin (Qwen3-8B Q8_0, ~8.7 GB) into one
; contiguous physical buffer ABOVE 4 GiB — impossible through Multiboot 1,
; whose module fields are 32-bit — and hand a 64-bit base/size to the
; existing kernel (boot.asm: uefi_entry64 -> long_mode_start -> kernel_main
; -> model_init). The Multiboot/GRUB path is untouched and remains the
; fallback.
;
; Volume layout (the volume this image was loaded from, FAT32 ESP):
;   \EFI\BOOT\BOOTX64.EFI     this loader
;   \LLM\KERNEL.ELF           build/os/asm-llm-os.elf64 (unchanged kernel)
;   \LLM\CMDLINE.TXT          optional kernel cmdline (same keys as GRUB path)
;   \LLM\MODEL.000, .001, ... the .asmbin split into chunks (FAT32 caps a
;                             single file at 4 GiB - 1); concatenated in order
;
; Sequence:
;   1. Locate volume, ACPI RSDP, allocate handoff region (< 4 GiB)
;   2. Load + validate kernel ELF64 at its physical link address
;   3. Enumerate model chunks, validate .asmbin header and total size
;   4. Compute memory requirements (weights + KV cache + scratch arena) for
;      the ctx length the kernel will use, and check them against the UEFI
;      memory map BEFORE allocating and AGAIN after allocating the weights
;   5. AllocatePages (below the kernel's 16 GiB identity map) and read chunks
;   6. Final memory map -> ExitBootServices -> convert map to MB1 format
;   7. Jump to uefi_entry64 with RDI = handoff
;
; ABI: UEFI x64 (Microsoft): RCX, RDX, R8, R9, stack; 32 B shadow space,
; RSP 16-byte aligned at every call. The code is fully RIP-relative, so the
; image needs no base relocations (an empty .reloc block is still provided
; so firmware treats it as relocatable).
; ============================================================================

bits 64
default rel

%include "constants.inc"
%include "os_constants.inc"
%include "uefi_handoff.inc"

; --- UEFI table offsets (x64) ---
%define ST_CONOUT               64
%define ST_BOOTSERVICES         96
%define ST_NUM_TABLES           104
%define ST_CONFIG_TABLE         112
%define CONOUT_OUTPUTSTRING     8
%define BS_ALLOCATE_PAGES       40
%define BS_FREE_PAGES           48
%define BS_GET_MEMORY_MAP       56
%define BS_ALLOCATE_POOL        64
%define BS_HANDLE_PROTOCOL      152
%define BS_EXIT_BOOT_SERVICES   232
%define BS_STALL                248
%define BS_SET_WATCHDOG         256
%define LI_DEVICE_HANDLE        24
%define SFS_OPEN_VOLUME         8
%define FILE_OPEN               8
%define FILE_CLOSE              16
%define FILE_READ               32
%define FILE_GET_POSITION       48
%define FILE_SET_POSITION       56
%define MD_TYPE                 0
%define MD_PHYS_START           8
%define MD_NUM_PAGES            24

%define ALLOC_ANY               0
%define ALLOC_MAX_ADDRESS       1
%define ALLOC_ADDRESS           2
%define EFI_LOADER_DATA         2
%define EFI_CONVENTIONAL        7
%define EFI_FILE_MODE_READ      1
%define EFI_LOAD_ERROR          0x8000000000000001
%define EFI_OUT_OF_RESOURCES    0x8000000000000009

%define READ_SLICE              (128 << 20)        ; progress dot every 128 MiB
%define MAX_CHUNKS              64
%define KERNEL_MAX_FILE         (16 << 20)
%define KERNEL_MIN_PADDR        0x00100000
%define KERNEL_MAX_PADDR        0x01000000
%define DEFAULT_CTX             2048               ; == kernel opt_ctx_len default
%define RAM_SAFETY              (64 << 20)         ; slack on the total-RAM check

%define IMAGE_BASE              0x10000000
%define SECT_ALIGN              0x1000

org 0                                   ; offsets == RVAs (FileAlign == SectAlign)

; ============================================================================
; PE32+ headers
; ============================================================================
dos_header:
    db      'MZ'
    times 0x3C - ($ - $$) db 0
    dd      pe_header

pe_header:
    db      'PE', 0, 0
    dw      0x8664                      ; Machine: x86-64
    dw      3                           ; NumberOfSections
    dd      0, 0, 0                     ; TimeDateStamp, SymTab, NumSyms
    dw      opt_header_end - opt_header ; SizeOfOptionalHeader
    dw      0x0022                      ; EXECUTABLE_IMAGE | LARGE_ADDRESS_AWARE
opt_header:
    dw      0x020B                      ; PE32+
    db      0, 0
    dd      text_end - text_start       ; SizeOfCode
    dd      (data_end - data_start) + (reloc_end - reloc_start)
    dd      0                           ; SizeOfUninitializedData
    dd      efi_main                    ; AddressOfEntryPoint
    dd      text_start                  ; BaseOfCode
    dq      IMAGE_BASE
    dd      SECT_ALIGN                  ; SectionAlignment
    dd      SECT_ALIGN                  ; FileAlignment
    dw      0, 0, 0, 0, 0, 0
    dd      0                           ; Win32VersionValue
    dd      image_end                   ; SizeOfImage
    dd      text_start                  ; SizeOfHeaders
    dd      0                           ; CheckSum
    dw      10                          ; Subsystem: EFI_APPLICATION
    dw      0                           ; DllCharacteristics
    dq      0x100000, 0x1000, 0x100000, 0x1000
    dd      0                           ; LoaderFlags
    dd      16                          ; NumberOfRvaAndSizes
    dq      0, 0, 0, 0, 0               ; export, import, resource, exception, security
    dd      reloc_start, reloc_end - reloc_start ; [5] base relocations
    times 10 dq 0
opt_header_end:

    db      '.text', 0, 0, 0
    dd      text_end - text_start, text_start, data_start - text_start, text_start
    dd      0, 0
    dw      0, 0
    dd      0x60000020                  ; CODE | EXECUTE | READ

    db      '.data', 0, 0, 0
    dd      data_end - data_start, data_start, reloc_start - data_start, data_start
    dd      0, 0
    dw      0, 0
    dd      0xC0000040                  ; INITIALIZED_DATA | READ | WRITE

    db      '.reloc', 0, 0
    dd      reloc_end - reloc_start, reloc_start, image_end - reloc_start, reloc_start
    dd      0, 0
    dw      0, 0
    dd      0x42000040                  ; INITIALIZED_DATA | DISCARDABLE | READ

    align   SECT_ALIGN, db 0

; ============================================================================
; .text
; ============================================================================
text_start:

; Die on EFI error (RAX high bit set) with message %1
%macro DIE_IF_ERR 1
    test    rax, rax
    jns     %%ok
    mov     [rel g_last_status], rax
    lea     rcx, [rel %1]
    jmp     die
%%ok:
%endmacro

%macro DIE 1
    lea     rcx, [rel %1]
    jmp     die
%endmacro

%macro PRINT 1
    lea     rcx, [rel %1]
    call    print_str
%endmacro

; Print "label" + value (MiB) + CRLF
%macro PRINT_MIB 2
    lea     rcx, [rel %1]
    mov     rdx, %2
    call    print_kv_mib
%endmacro

; Print "label" + decimal + CRLF
%macro PRINT_DEC 2
    lea     rcx, [rel %1]
    mov     rdx, %2
    call    print_kv_dec
%endmacro

; ----------------------------------------------------------------------------
; EFI_STATUS efi_main(EFI_HANDLE image, EFI_SYSTEM_TABLE *st)
; ----------------------------------------------------------------------------
efi_main:
    push    rbx
    push    rbp
    push    rsi
    push    rdi
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 0x58                   ; 8 pushes + 0x58 -> 16-byte aligned
    mov     [rel g_saved_rsp], rsp

    mov     [rel g_image], rcx
    mov     [rel g_st], rdx
    mov     rax, [rdx + ST_CONOUT]
    mov     [rel g_conout], rax
    mov     rax, [rdx + ST_BOOTSERVICES]
    mov     [rel g_bs], rax

    PRINT   s_banner

    ; Loading ~9 GB can exceed the 5-minute UEFI watchdog: disable it
    xor     ecx, ecx
    xor     edx, edx
    xor     r8d, r8d
    xor     r9d, r9d
    mov     rax, [rel g_bs]
    call    [rax + BS_SET_WATCHDOG]

    ; ---- 1. Volume this image was loaded from ------------------------------
    mov     rcx, [rel g_image]
    lea     rdx, [rel guid_loaded_image]
    lea     r8, [rel g_loaded_image]
    mov     rax, [rel g_bs]
    call    [rax + BS_HANDLE_PROTOCOL]
    DIE_IF_ERR s_e_loaded_image

    mov     rax, [rel g_loaded_image]
    mov     rcx, [rax + LI_DEVICE_HANDLE]
    lea     rdx, [rel guid_sfs]
    lea     r8, [rel g_sfs]
    mov     rax, [rel g_bs]
    call    [rax + BS_HANDLE_PROTOCOL]
    DIE_IF_ERR s_e_sfs

    mov     rcx, [rel g_sfs]
    lea     rdx, [rel g_root]
    call    [rcx + SFS_OPEN_VOLUME]
    DIE_IF_ERR s_e_volume

    ; ---- ACPI RSDP (2.0 preferred) -----------------------------------------
    lea     rcx, [rel guid_acpi20]
    call    find_config_table
    test    rax, rax
    jnz     .have_rsdp
    lea     rcx, [rel guid_acpi10]
    call    find_config_table
.have_rsdp:
    mov     [rel g_rsdp], rax
    PRINT   s_rsdp
    mov     rcx, [rel g_rsdp]
    call    print_hex
    PRINT   s_crlf

    ; ---- Handoff region (below 4 GiB: MB1 info fields are 32-bit) ---------
    mov     eax, 0xFFFFFFFF             ; zero-extended (imm32 would sign-extend)
    mov     [rel g_handoff], rax
    mov     ecx, ALLOC_MAX_ADDRESS
    mov     edx, EFI_LOADER_DATA
    mov     r8d, HO_REGION_PAGES
    lea     r9, [rel g_handoff]
    mov     rax, [rel g_bs]
    call    [rax + BS_ALLOCATE_PAGES]
    DIE_IF_ERR s_e_handoff
    mov     rdi, [rel g_handoff]
    mov     ecx, HO_REGION_PAGES * 4096
    xor     eax, eax
    rep     stosb

    ; ---- Optional \LLM\CMDLINE.TXT -----------------------------------------
    call    load_cmdline

    ; ---- 2. Kernel ELF64 ---------------------------------------------------
    call    load_kernel

    ; Keep the SMP trampoline page (TRAMP_BASE) away from firmware use.
    ; Failure is not fatal: it is normally boot-services memory that becomes
    ; free at ExitBootServices.
    mov     qword [rel g_tmp_addr], TRAMP_BASE
    mov     ecx, ALLOC_ADDRESS
    mov     edx, EFI_LOADER_DATA
    mov     r8d, 1
    lea     r9, [rel g_tmp_addr]
    mov     rax, [rel g_bs]
    call    [rax + BS_ALLOCATE_PAGES]
    test    rax, rax
    jns     .tramp_ok
    PRINT   s_w_tramp
.tramp_ok:

    ; ---- 3. Model chunks + header ------------------------------------------
    call    enumerate_model
    call    validate_header

    ; ---- 4. Memory requirements --------------------------------------------
    call    compute_requirements

    call    mmap_scan
    DIE_IF_ERR s_e_mmap
    PRINT_MIB s_free_total, [rel g_free_total]
    PRINT_MIB s_free_largest, [rel g_free_largest]

    mov     rax, [rel g_model_pages]
    shl     rax, 12
    add     rax, [rel g_heap_need]
    add     rax, RAM_SAFETY
    cmp     [rel g_free_total], rax
    jae     .ram_ok
    DIE     s_e_ram_total
.ram_ok:

    ; ---- 5. Allocate weights: one contiguous range below the 16 GiB map ----
    ; AllocateMaxAddress returns the highest fitting range, leaving low/mid
    ; RAM for the kernel heap (KV cache + scratch).
    mov     rax, UEFI_MAP_LIMIT - 1
    mov     [rel g_model_base], rax
    mov     ecx, ALLOC_MAX_ADDRESS
    mov     edx, EFI_LOADER_DATA
    mov     r8, [rel g_model_pages]
    lea     r9, [rel g_model_base]
    mov     rax, [rel g_bs]
    call    [rax + BS_ALLOCATE_PAGES]
    DIE_IF_ERR s_e_model_alloc

    PRINT   s_model_at
    mov     rcx, [rel g_model_base]
    call    print_hex
    PRINT   s_dash
    mov     rcx, [rel g_model_base]
    add     rcx, [rel g_model_size]
    call    print_hex
    PRINT   s_crlf

    ; Re-check with the weights in place: the kernel heap is ONE region
    ; (mem_init picks the largest usable range), so it must fit there.
    call    mmap_scan
    DIE_IF_ERR s_e_mmap
    PRINT_MIB s_free_after, [rel g_free_largest]
    mov     rax, [rel g_heap_need]
    cmp     [rel g_free_largest], rax
    jae     .heap_ok
    DIE     s_e_heap_region
.heap_ok:
    PRINT   s_ok_fits

    ; Read chunks sequentially into the contiguous buffer
    call    read_model

    mov     rax, [rel g_model_base]
    cmp     dword [rax + HDR_MAGIC], ASMBIN_MAGIC
    je      .loaded_ok
    DIE     s_e_verify
.loaded_ok:

    ; ---- Fill handoff + synthesized MB1 info -------------------------------
    call    fill_handoff

    PRINT   s_jump
    mov     rcx, [rel g_kentry]
    call    print_hex
    PRINT   s_crlf

    ; ---- 6. ExitBootServices (retry on stale map key) ----------------------
    call    get_mmap
    DIE_IF_ERR s_e_mmap
    mov     r12d, 8
.ebs_try:
    mov     rcx, [rel g_image]
    mov     rdx, [rel g_mmap_key]
    mov     rax, [rel g_bs]
    call    [rax + BS_EXIT_BOOT_SERVICES]
    test    rax, rax
    jz      .ebs_ok
    dec     r12d
    jz      .ebs_dead
    call    get_mmap                    ; allowed after a failed EBS
    jmp     .ebs_try
.ebs_dead:
    cli
    hlt
    jmp     .ebs_dead

.ebs_ok:
    cli
    ; Boot services are gone: no more firmware calls from here on.
    call    convert_mmap

    ; ---- 7. Enter the kernel ----------------------------------------------
    mov     rdi, [rel g_handoff]
    mov     rax, [rel g_kentry]
    jmp     rax

; die(rcx = message): print, wait, return EFI_LOAD_ERROR to the firmware
die:
    mov     rbx, rcx
    mov     rsp, [rel g_saved_rsp]
    PRINT   s_err
    mov     rcx, rbx
    call    print_str
    mov     rcx, [rel g_last_status]
    test    rcx, rcx
    jz      .no_status
    PRINT   s_status
    mov     rcx, [rel g_last_status]
    call    print_hex
.no_status:
    PRINT   s_crlf
    mov     ecx, 10000000               ; 10 s so the message can be read
    mov     rax, [rel g_bs]
    call    [rax + BS_STALL]
    mov     rax, EFI_LOAD_ERROR
    mov     rsp, [rel g_saved_rsp]
    add     rsp, 0x58
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    pop     rbp
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; load_cmdline: \LLM\CMDLINE.TXT -> handoff cmdline (control chars -> spaces),
; then parse "ctx=" exactly as the kernel would (default 2048, 0 = header).
; ----------------------------------------------------------------------------
load_cmdline:
    push    rbx
    push    rsi
    sub     rsp, 0x28
    mov     qword [rel g_ctx_req], DEFAULT_CTX

    lea     rcx, [rel p_cmdline]
    call    open_file
    test    rax, rax
    jz      .done
    mov     rbx, rax
    mov     rcx, rbx
    call    file_size
    mov     r8, rax
    cmp     r8, HO_CMDLINE_MAX - 1
    jbe     .sz_ok
    mov     r8d, HO_CMDLINE_MAX - 1
.sz_ok:
    mov     rsi, r8
    mov     rcx, rbx
    mov     rdx, [rel g_handoff]
    add     rdx, HO_OFF_CMDLINE
    call    file_read
    mov     rcx, rbx
    call    [rbx + FILE_CLOSE]

    ; Sanitize: control characters (CR/LF/TAB) -> space
    mov     rdx, [rel g_handoff]
    add     rdx, HO_OFF_CMDLINE
    xor     ecx, ecx
.san:
    cmp     rcx, rsi
    jae     .san_done
    cmp     byte [rdx + rcx], 0x20
    jae     .san_next
    mov     byte [rdx + rcx], 0x20
.san_next:
    inc     rcx
    jmp     .san
.san_done:
    mov     byte [rdx + rsi], 0
    mov     dword [rel g_has_cmdline], 1

    PRINT   s_cmdline
    mov     rcx, [rel g_handoff]
    add     rcx, HO_OFF_CMDLINE
    call    print_ascii
    PRINT   s_crlf

    ; Find "ctx=" at a token start
    mov     rdx, [rel g_handoff]
    add     rdx, HO_OFF_CMDLINE
    mov     rcx, rdx
.scan:
    cmp     byte [rcx], 0
    je      .done
    cmp     rcx, rdx
    je      .tok_start
    cmp     byte [rcx - 1], ' '
    jne     .scan_next
.tok_start:
    cmp     dword [rcx], 'ctx='
    jne     .scan_next
    add     rcx, 4
    xor     eax, eax
.dig:
    movzx   r8d, byte [rcx]
    sub     r8d, '0'
    cmp     r8d, 9
    ja      .dig_done
    imul    rax, rax, 10
    add     rax, r8
    inc     rcx
    jmp     .dig
.dig_done:
    mov     [rel g_ctx_req], rax
    jmp     .done
.scan_next:
    inc     rcx
    jmp     .scan
.done:
    add     rsp, 0x28
    pop     rsi
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; load_kernel: read \LLM\KERNEL.ELF, validate ELF64 x86-64, claim the exact
; physical range of its PT_LOAD segments, copy/zero them, locate uefi_entry64.
; ----------------------------------------------------------------------------
load_kernel:
    push    rbx
    push    rsi
    push    rdi
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 0x20                   ; 7 pushes -> aligned after sub 0x20

    lea     rcx, [rel p_kernel]
    call    open_file
    test    rax, rax
    jnz     .opened
    DIE     s_e_kernel_open
.opened:
    mov     r12, rax
    mov     rcx, r12
    call    file_size
    mov     r13, rax
    cmp     r13, 64
    jb      .bad
    cmp     r13, KERNEL_MAX_FILE
    ja      .bad

    mov     ecx, EFI_LOADER_DATA
    mov     rdx, r13
    lea     r8, [rel g_kbuf]
    mov     rax, [rel g_bs]
    call    [rax + BS_ALLOCATE_POOL]
    DIE_IF_ERR s_e_oom

    mov     rcx, r12
    mov     rdx, [rel g_kbuf]
    mov     r8, r13
    call    file_read
    DIE_IF_ERR s_e_kernel_read
    mov     rcx, r12
    call    [r12 + FILE_CLOSE]

    mov     rsi, [rel g_kbuf]
    cmp     dword [rsi], 0x464C457F     ; "\x7FELF"
    jne     .bad
    cmp     byte [rsi + 4], 2           ; ELFCLASS64
    jne     .bad
    cmp     word [rsi + 18], 0x3E       ; EM_X86_64
    jne     .bad
    movzx   eax, word [rsi + 54]        ; e_phentsize
    cmp     eax, 56
    jb      .bad
    mov     [rel g_phentsize], rax
    movzx   eax, word [rsi + 56]        ; e_phnum
    test    eax, eax
    jz      .bad
    mov     [rel g_phnum], rax
    imul    rax, [rel g_phentsize]
    add     rax, [rsi + 32]             ; phoff + phnum*phentsize
    jc      .bad
    cmp     rax, r13
    ja      .bad

    ; Pass 1: physical span of PT_LOAD segments, with bounds checks
    mov     r14, -1                     ; min paddr
    xor     r15d, r15d                  ; max paddr + memsz
    mov     rbx, [rsi + 32]
    add     rbx, rsi                    ; phdr cursor
    mov     rdi, [rel g_phnum]
.p1:
    cmp     dword [rbx], 1              ; PT_LOAD
    jne     .p1_next
    mov     rax, [rbx + 40]             ; memsz
    test    rax, rax
    jz      .p1_next
    mov     rcx, [rbx + 32]             ; filesz
    cmp     rcx, rax
    ja      .bad
    mov     rdx, [rbx + 8]              ; offset
    add     rdx, rcx
    jc      .bad
    cmp     rdx, r13
    ja      .bad
    mov     rdx, [rbx + 24]             ; paddr
    cmp     rdx, r14
    jae     .p1_min
    mov     r14, rdx
.p1_min:
    add     rdx, rax
    jc      .bad
    cmp     rdx, r15
    jbe     .p1_next
    mov     r15, rdx
.p1_next:
    add     rbx, [rel g_phentsize]
    dec     rdi
    jnz     .p1

    test    r15, r15
    jz      .bad
    and     r14, -4096
    add     r15, 4095
    and     r15, -4096
    cmp     r14, KERNEL_MIN_PADDR
    jb      .bad
    cmp     r15, KERNEL_MAX_PADDR
    ja      .bad
    mov     [rel g_kmin], r14
    mov     [rel g_kmax], r15

    ; Claim exactly [kmin, kmax) — the kernel is not relocatable
    mov     [rel g_tmp_addr], r14
    mov     ecx, ALLOC_ADDRESS
    mov     edx, EFI_LOADER_DATA
    mov     r8, r15
    sub     r8, r14
    shr     r8, 12
    lea     r9, [rel g_tmp_addr]
    mov     rax, [rel g_bs]
    call    [rax + BS_ALLOCATE_PAGES]
    DIE_IF_ERR s_e_kernel_range

    ; Zero the whole span (covers .bss), then copy file-backed bytes
    mov     rdi, r14
    mov     rcx, r15
    sub     rcx, r14
    xor     eax, eax
    rep     stosb

    mov     rsi, [rel g_kbuf]
    mov     rbx, [rsi + 32]
    add     rbx, rsi
    mov     r12, [rel g_phnum]
.p2:
    cmp     dword [rbx], 1
    jne     .p2_next
    mov     rcx, [rbx + 32]             ; filesz
    test    rcx, rcx
    jz      .p2_next
    mov     rdi, [rbx + 24]             ; paddr
    push    rsi
    add     rsi, [rbx + 8]              ; file offset
    rep     movsb
    pop     rsi
.p2_next:
    add     rbx, [rel g_phentsize]
    dec     r12
    jnz     .p2

    ; Locate the UEFI kernel header in the first 8 KiB of the image
    mov     rbx, r14
    lea     rdx, [r14 + 8192 - 24]
    cmp     rdx, r15
    jbe     .scan_lim_ok
    lea     rdx, [r15 - 24]
.scan_lim_ok:
    mov     rax, UEFI_KHDR_MAGIC
.hscan:
    cmp     rbx, rdx
    ja      .no_hdr
    cmp     [rbx], rax
    je      .hfound
    add     rbx, 4
    jmp     .hscan
.no_hdr:
    DIE     s_e_kernel_hdr
.hfound:
    cmp     dword [rbx + 16], UEFI_HANDOFF_VERSION
    jne     .no_hdr
    mov     rax, [rbx + 8]
    cmp     rax, r14
    jb      .no_hdr
    cmp     rax, r15
    jae     .no_hdr
    mov     [rel g_kentry], rax

    PRINT   s_kernel
    mov     rcx, r14
    call    print_hex
    PRINT   s_dash
    mov     rcx, r15
    call    print_hex
    PRINT   s_entry
    mov     rcx, [rel g_kentry]
    call    print_hex
    PRINT   s_crlf

    add     rsp, 0x20
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    pop     rbx
    ret
.bad:
    DIE     s_e_kernel_bad

; ----------------------------------------------------------------------------
; enumerate_model: open \LLM\MODEL.000 .. until one is missing; sum sizes.
; ----------------------------------------------------------------------------
enumerate_model:
    push    rbx
    push    rsi
    push    rdi
    push    r12
    push    r13
    sub     rsp, 0x20                   ; 5 pushes -> aligned

    xor     ebx, ebx                    ; chunk index
    xor     r13d, r13d                  ; total bytes
.lp:
    cmp     ebx, MAX_CHUNKS
    jae     .done
    ; name digits: "\LLM\MODEL.ddd" -> UCS-2 chars 11..13
    mov     eax, ebx
    xor     edx, edx
    mov     ecx, 100
    div     ecx
    add     eax, '0'
    mov     [rel p_model + 22], ax
    mov     eax, edx
    xor     edx, edx
    mov     ecx, 10
    div     ecx
    add     eax, '0'
    mov     [rel p_model + 24], ax
    add     edx, '0'
    mov     [rel p_model + 26], dx

    lea     rcx, [rel p_model]
    call    open_file
    test    rax, rax
    jz      .done
    lea     rdx, [rel g_chunk_handle]
    mov     [rdx + rbx * 8], rax
    mov     rcx, rax
    call    file_size
    test    rax, rax
    jnz     .nonempty
    DIE     s_e_chunk_empty
.nonempty:
    lea     rdx, [rel g_chunk_size]
    mov     [rdx + rbx * 8], rax
    add     r13, rax
    jc      .too_big

    PRINT   s_chunk
    lea     rcx, [rel p_model]
    call    print_str
    PRINT   s_colon
    lea     rdx, [rel g_chunk_size]
    mov     rcx, [rdx + rbx * 8]
    call    print_mib
    PRINT   s_crlf

    inc     ebx
    jmp     .lp
.done:
    test    ebx, ebx
    jnz     .have
    DIE     s_e_no_model
.have:
    mov     [rel g_nchunks], rbx
    mov     [rel g_model_size], r13
    mov     rax, r13
    add     rax, 4095
    shr     rax, 12
    mov     [rel g_model_pages], rax
    PRINT_MIB s_model_total, r13

    add     rsp, 0x20
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    pop     rbx
    ret
.too_big:
    DIE     s_e_chunk_sum

; ----------------------------------------------------------------------------
; validate_header: read the 128-byte .asmbin header from chunk 0, check it
; against the chunk set, print the architecture.
; ----------------------------------------------------------------------------
validate_header:
    push    rbx
    sub     rsp, 0x20

    mov     rbx, [rel g_chunk_handle]
    cmp     qword [rel g_chunk_size], 128
    jb      .bad
    mov     rcx, rbx
    lea     rdx, [rel g_hdr]
    mov     r8d, 128
    call    file_read
    DIE_IF_ERR s_e_model_read
    mov     rcx, rbx
    xor     edx, edx
    call    [rbx + FILE_SET_POSITION]

    lea     rbx, [rel g_hdr]
    cmp     dword [rbx + HDR_MAGIC], ASMBIN_MAGIC
    jne     .bad
    mov     eax, [rbx + HDR_VERSION]
    cmp     eax, ASMBIN_VERSION
    je      .ver_ok
    cmp     eax, ASMBIN_VERSION_V2
    jne     .bad
.ver_ok:
    ; Header file_size must equal the concatenated chunks (catches a missing
    ; or truncated chunk before 9 GB are read)
    mov     rax, [rbx + HDR_FILE_SIZE]
    cmp     rax, [rel g_model_size]
    je      .size_ok
    DIE     s_e_size_mismatch
.size_ok:
    ; Sanity bounds so the requirement arithmetic cannot overflow
    mov     eax, [rbx + HDR_DIM]
    test    eax, eax
    jz      .bad
    cmp     eax, 65536
    ja      .bad
    mov     eax, [rbx + HDR_HIDDEN_DIM]
    test    eax, eax
    jz      .bad
    cmp     eax, 262144
    ja      .bad
    mov     eax, [rbx + HDR_N_LAYERS]
    test    eax, eax
    jz      .bad
    cmp     eax, 512
    ja      .bad
    mov     eax, [rbx + HDR_N_HEADS]
    test    eax, eax
    jz      .bad
    cmp     eax, 1024
    ja      .bad
    mov     ecx, [rbx + HDR_N_KV_HEADS]
    test    ecx, ecx
    jz      .bad
    xor     edx, edx
    div     ecx                         ; GQA: n_heads % n_kv_heads == 0
    test    edx, edx
    jnz     .bad
    mov     eax, [rbx + HDR_HEAD_DIM]
    test    eax, eax
    jz      .bad
    cmp     eax, 1024
    ja      .bad
    mov     eax, [rbx + HDR_VOCAB_SIZE]
    test    eax, eax
    jz      .bad
    cmp     eax, 1048576
    ja      .bad
    mov     eax, [rbx + HDR_MAX_SEQ_LEN]
    test    eax, eax
    jz      .bad
    cmp     eax, 1048576
    ja      .bad

    PRINT   s_arch
    PRINT_DEC s_dim, [rbx + HDR_DIM]
    PRINT_DEC s_hidden, [rbx + HDR_HIDDEN_DIM]
    PRINT_DEC s_layers, [rbx + HDR_N_LAYERS]
    PRINT_DEC s_heads, [rbx + HDR_N_HEADS]
    PRINT_DEC s_kvheads, [rbx + HDR_N_KV_HEADS]
    PRINT_DEC s_headdim, [rbx + HDR_HEAD_DIM]
    PRINT_DEC s_vocab, [rbx + HDR_VOCAB_SIZE]
    PRINT_DEC s_maxseq, [rbx + HDR_MAX_SEQ_LEN]

    test    dword [rbx + HDR_FLAGS], HDR_FLAG_TIED_EMB
    jnz     .tied
    PRINT   s_w_untied
.tied:
    add     rsp, 0x20
    pop     rbx
    ret
.bad:
    DIE     s_e_bad_header

; ----------------------------------------------------------------------------
; compute_requirements: mirror model_init's arena layout (transformer.asm)
; for the ctx length the kernel will select, plus mem_init's constraints.
;   ctx   = (ctx_req == 0 || ctx_req > hdr.max_seq) ? hdr.max_seq : ctx_req
;   kv    = 2 * n_layers * ctx * kv_dim * 4                (FP32 K + V cache)
;   arena = kv + 4*(2*dim + 2*q_dim + 2*kv_dim + q_dim + n_heads*ctx
;                   + 2*hidden + 2*vocab + 2*head_dim)
;           + max_K + max_K/8 + 2*8192*4 + 24*64
;   heap  = max(arena + 16 MiB, 16 MiB)    (mem_init minimum + diagnostics)
; ----------------------------------------------------------------------------
compute_requirements:
    push    rbx
    sub     rsp, 0x20
    lea     rbx, [rel g_hdr]

    mov     eax, [rbx + HDR_MAX_SEQ_LEN]
    mov     rcx, [rel g_ctx_req]
    test    rcx, rcx
    jz      .use_hdr
    cmp     rcx, rax
    ja      .use_hdr
    mov     rax, rcx
.use_hdr:
    mov     [rel g_ctx], rax

    mov     eax, [rbx + HDR_N_HEADS]
    mov     ecx, [rbx + HDR_HEAD_DIM]
    imul    rax, rcx
    mov     r8, rax                     ; q_dim
    mov     eax, [rbx + HDR_N_KV_HEADS]
    imul    rax, rcx
    mov     r9, rax                     ; kv_dim

    ; KV cache
    mov     eax, [rbx + HDR_N_LAYERS]
    imul    rax, [rel g_ctx]
    imul    rax, r9
    shl     rax, 3                      ; * 4 bytes * 2 (K and V)
    mov     [rel g_kv_bytes], rax

    ; Scratch (float32 element count)
    mov     eax, [rbx + HDR_DIM]
    lea     r10, [rax * 2]              ; x, xb
    lea     r10, [r10 + r8 * 2]         ; q, att_out
    add     r10, r8                     ; (slack: one extra q_dim)
    lea     r10, [r10 + r9 * 2]         ; k, v
    mov     eax, [rbx + HDR_N_HEADS]
    imul    rax, [rel g_ctx]
    add     r10, rax                    ; att scores
    mov     eax, [rbx + HDR_HIDDEN_DIM]
    lea     r10, [r10 + rax * 2]        ; gate, up
    mov     eax, [rbx + HDR_VOCAB_SIZE]
    lea     r10, [r10 + rax * 2]        ; logits, probs
    mov     eax, [rbx + HDR_HEAD_DIM]
    lea     r10, [r10 + rax * 2]        ; inv_freq (8B * hd/2) + cos + sin
    shl     r10, 2                      ; -> bytes

    mov     eax, [rbx + HDR_HIDDEN_DIM] ; max_K = max(hidden, q_dim)
    cmp     rax, r8
    jae     .kmax
    mov     rax, r8
.kmax:
    add     r10, rax                    ; xq (int8)
    shr     rax, 3
    add     r10, rax                    ; xs (scales)
    add     r10, 2 * 8192 * 4 + 24 * 64 ; prompt/gen token buffers + alignment

    mov     rax, [rel g_kv_bytes]
    add     rax, r10
    mov     [rel g_arena_bytes], rax
    add     rax, UEFI_HEAP_MIN
    mov     [rel g_heap_need], rax

    PRINT   s_req
    PRINT_MIB s_req_weights, [rel g_model_size]
    PRINT_DEC s_req_ctx, [rel g_ctx]
    PRINT_MIB s_req_kv, [rel g_kv_bytes]
    PRINT_MIB s_req_arena, [rel g_arena_bytes]
    PRINT_MIB s_req_heap, [rel g_heap_need]
    mov     rax, [rel g_model_pages]
    shl     rax, 12
    add     rax, [rel g_heap_need]
    add     rax, [rel g_kmax]
    PRINT_MIB s_req_total, rax

    add     rsp, 0x20
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; read_model: read every chunk back-to-back into [g_model_base ...)
; ----------------------------------------------------------------------------
read_model:
    push    rbx
    push    rsi
    push    rdi
    push    r12
    push    r13
    sub     rsp, 0x20

    PRINT   s_reading
    mov     dword [rel g_progress], 1
    mov     rdi, [rel g_model_base]
    xor     ebx, ebx
.lp:
    cmp     rbx, [rel g_nchunks]
    jae     .done
    lea     rax, [rel g_chunk_handle]
    mov     r12, [rax + rbx * 8]
    lea     rax, [rel g_chunk_size]
    mov     r13, [rax + rbx * 8]
    mov     rcx, r12
    mov     rdx, rdi
    mov     r8, r13
    call    file_read
    DIE_IF_ERR s_e_model_read
    mov     rcx, r12
    call    [r12 + FILE_CLOSE]
    add     rdi, r13
    inc     rbx
    jmp     .lp
.done:
    mov     dword [rel g_progress], 0
    PRINT   s_crlf
    PRINT   s_read_ok

    add     rsp, 0x20
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; fill_handoff: handoff struct + MB1 info (mmap length set after EBS)
; ----------------------------------------------------------------------------
fill_handoff:
    mov     rdx, [rel g_handoff]
    mov     rax, HANDOFF_MAGIC
    mov     [rdx + HO_MAGIC], rax
    mov     dword [rdx + HO_VERSION], UEFI_HANDOFF_VERSION
    mov     dword [rdx + HO_STRUCT_SIZE], HO_SIZEOF
    lea     rax, [rdx + HO_OFF_MBI]
    mov     [rdx + HO_MBI_PTR], rax
    mov     rax, [rel g_model_base]
    mov     [rdx + HO_MODEL_BASE], rax
    mov     rax, [rel g_model_size]
    mov     [rdx + HO_MODEL_SIZE], rax
    mov     rax, [rel g_rsdp]
    mov     [rdx + HO_RSDP], rax
    mov     rax, [rel g_kmin]
    mov     [rdx + HO_KERNEL_MIN], rax
    mov     rax, [rel g_kmax]
    mov     [rdx + HO_KERNEL_MAX], rax
    mov     rax, [rel g_ctx]
    mov     [rdx + HO_CTX_LEN], rax
    mov     rax, [rel g_arena_bytes]
    mov     [rdx + HO_ARENA_BYTES], rax
    lea     rax, [rdx + HO_OFF_UEFI_MMAP]
    mov     [rdx + HO_UEFI_MMAP_PTR], rax

    lea     rcx, [rdx + HO_OFF_MBI]
    mov     eax, 0x40                   ; mmap valid
    cmp     dword [rel g_has_cmdline], 0
    je      .no_cmd
    or      eax, 0x04                   ; cmdline valid
.no_cmd:
    mov     [rcx + MBI_FLAGS], eax
    lea     rax, [rdx + HO_OFF_CMDLINE]
    mov     [rcx + MBI_CMDLINE], eax
    lea     rax, [rdx + HO_OFF_MB_MMAP]
    mov     [rcx + MBI_MMAP_ADDR], eax
    ret

; ----------------------------------------------------------------------------
; convert_mmap (after ExitBootServices, no firmware calls): UEFI descriptors
; -> MB1 mmap entries, merging physically adjacent same-type ranges so
; mem_init's "largest region" sees real contiguous free RAM.
;   Conventional/BootServices{Code,Data} -> 1 (usable)
;   ACPIReclaim -> 3, ACPINVS -> 4, Unusable -> 5, everything else -> 2.
; LoaderCode/LoaderData (this image, kernel, model, handoff) stay reserved.
; ----------------------------------------------------------------------------
convert_mmap:
    mov     rdx, [rel g_handoff]
    lea     rsi, [rdx + HO_OFF_UEFI_MMAP]
    mov     rbx, rsi
    add     rbx, [rel g_mmap_size]
    lea     rdi, [rdx + HO_OFF_MB_MMAP]
    lea     r11, [rdi + HO_MB_MMAP_CAP - 24]
    xor     r10d, r10d                  ; previous MB entry (0 = none)
.lp:
    cmp     rsi, rbx
    jae     .done
    mov     eax, [rsi + MD_TYPE]
    mov     ecx, 1
    cmp     eax, 3
    je      .typed
    cmp     eax, 4
    je      .typed
    cmp     eax, EFI_CONVENTIONAL
    je      .typed
    mov     ecx, 3
    cmp     eax, 9
    je      .typed
    mov     ecx, 4
    cmp     eax, 10
    je      .typed
    mov     ecx, 5
    cmp     eax, 8
    je      .typed
    mov     ecx, 2
.typed:
    mov     r8, [rsi + MD_PHYS_START]
    mov     r9, [rsi + MD_NUM_PAGES]
    shl     r9, 12
    test    r10, r10
    jz      .emit
    cmp     [r10 + 20], ecx
    jne     .emit
    mov     rax, [r10 + 4]
    add     rax, [r10 + 12]
    cmp     rax, r8
    jne     .emit
    add     [r10 + 12], r9              ; merge
    jmp     .next
.emit:
    cmp     rdi, r11
    ja      .next                       ; capacity guard (cannot happen: 2560 slots)
    mov     dword [rdi], 20
    mov     [rdi + 4], r8
    mov     [rdi + 12], r9
    mov     [rdi + 20], ecx
    mov     r10, rdi
    add     rdi, 24
.next:
    add     rsi, [rel g_desc_size]
    jmp     .lp
.done:
    mov     rdx, [rel g_handoff]
    lea     rax, [rdx + HO_OFF_MB_MMAP]
    sub     rdi, rax
    mov     [rdx + HO_OFF_MBI + MBI_MMAP_LENGTH], edi
    mov     rax, [rel g_mmap_size]
    mov     [rdx + HO_UEFI_MMAP_SIZE], rax
    mov     rax, [rel g_desc_size]
    mov     [rdx + HO_UEFI_DESC_SIZE], rax
    ret

; ----------------------------------------------------------------------------
; get_mmap -> rax status; fills raw map at handoff + HO_OFF_UEFI_MMAP
; ----------------------------------------------------------------------------
get_mmap:
    sub     rsp, 0x38
    mov     qword [rel g_mmap_size], HO_UEFI_MMAP_CAP
    lea     rcx, [rel g_mmap_size]
    mov     rdx, [rel g_handoff]
    add     rdx, HO_OFF_UEFI_MMAP
    lea     r8, [rel g_mmap_key]
    lea     r9, [rel g_desc_size]
    lea     rax, [rel g_desc_ver]
    mov     [rsp + 0x20], rax
    mov     rax, [rel g_bs]
    call    [rax + BS_GET_MEMORY_MAP]
    test    rax, rax
    js      .ret
    cmp     qword [rel g_desc_size], 40
    jae     .ret
    mov     rax, EFI_LOAD_ERROR
.ret:
    add     rsp, 0x38
    ret

; ----------------------------------------------------------------------------
; mmap_scan -> rax status; g_free_total / g_free_largest = EfiConventional
; bytes clipped to [UEFI_HEAP_FLOOR, UEFI_MAP_LIMIT) (what the kernel can use)
; ----------------------------------------------------------------------------
mmap_scan:
    push    rbx
    push    rsi
    sub     rsp, 0x28
    call    get_mmap
    test    rax, rax
    js      .ret
    xor     r10d, r10d
    xor     r11d, r11d
    mov     rsi, [rel g_handoff]
    add     rsi, HO_OFF_UEFI_MMAP
    mov     rbx, rsi
    add     rbx, [rel g_mmap_size]
.lp:
    cmp     rsi, rbx
    jae     .done
    cmp     dword [rsi + MD_TYPE], EFI_CONVENTIONAL
    jne     .next
    mov     rax, [rsi + MD_PHYS_START]
    mov     rdx, [rsi + MD_NUM_PAGES]
    shl     rdx, 12
    add     rdx, rax
    mov     rcx, UEFI_MAP_LIMIT
    cmp     rdx, rcx
    jbe     .e_ok
    mov     rdx, rcx
.e_ok:
    cmp     rax, UEFI_HEAP_FLOOR
    jae     .s_ok
    mov     eax, UEFI_HEAP_FLOOR
.s_ok:
    cmp     rdx, rax
    jbe     .next
    sub     rdx, rax
    add     r10, rdx
    cmp     rdx, r11
    jbe     .next
    mov     r11, rdx
.next:
    add     rsi, [rel g_desc_size]
    jmp     .lp
.done:
    mov     [rel g_free_total], r10
    mov     [rel g_free_largest], r11
    xor     eax, eax
.ret:
    add     rsp, 0x28
    pop     rsi
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; find_config_table(rcx = GUID*) -> rax = VendorTable or 0
; ----------------------------------------------------------------------------
find_config_table:
    mov     rdx, [rel g_st]
    mov     r8, [rdx + ST_NUM_TABLES]
    mov     r9, [rdx + ST_CONFIG_TABLE]
    mov     r10, [rcx]
    mov     r11, [rcx + 8]
.lp:
    test    r8, r8
    jz      .none
    cmp     [r9], r10
    jne     .next
    cmp     [r9 + 8], r11
    jne     .next
    mov     rax, [r9 + 16]
    ret
.next:
    add     r9, 24
    dec     r8
    jmp     .lp
.none:
    xor     eax, eax
    ret

; ----------------------------------------------------------------------------
; File helpers (EFI_FILE_PROTOCOL)
; ----------------------------------------------------------------------------
; open_file(rcx = UCS-2 path) -> rax = EFI_FILE* or 0
open_file:
    sub     rsp, 0x38
    mov     r8, rcx
    mov     rcx, [rel g_root]
    lea     rdx, [rel g_tmp_handle]
    mov     r9d, EFI_FILE_MODE_READ
    mov     qword [rsp + 0x20], 0
    call    [rcx + FILE_OPEN]
    test    rax, rax
    js      .fail
    mov     rax, [rel g_tmp_handle]
    jmp     .ret
.fail:
    xor     eax, eax
.ret:
    add     rsp, 0x38
    ret

; file_size(rcx = file) -> rax = size (position reset to 0)
file_size:
    push    rbx
    sub     rsp, 0x20
    mov     rbx, rcx
    mov     rdx, -1                     ; end of file
    call    [rbx + FILE_SET_POSITION]
    mov     rcx, rbx
    lea     rdx, [rel g_tmp_u64]
    call    [rbx + FILE_GET_POSITION]
    mov     rcx, rbx
    xor     edx, edx
    call    [rbx + FILE_SET_POSITION]
    mov     rax, [rel g_tmp_u64]
    add     rsp, 0x20
    pop     rbx
    ret

; file_read(rcx = file, rdx = buf, r8 = size) -> rax status
; Reads in READ_SLICE pieces; prints a dot per slice when g_progress != 0.
file_read:
    push    rbx
    push    rsi
    push    rdi
    push    r12
    sub     rsp, 0x28
    mov     rbx, rcx
    mov     rsi, rdx
    mov     rdi, r8
.lp:
    test    rdi, rdi
    jz      .ok
    mov     r12, READ_SLICE
    cmp     rdi, r12
    jae     .slice
    mov     r12, rdi
.slice:
    mov     [rel g_tmp_u64], r12
    mov     rcx, rbx
    lea     rdx, [rel g_tmp_u64]
    mov     r8, rsi
    call    [rbx + FILE_READ]
    test    rax, rax
    js      .ret
    cmp     [rel g_tmp_u64], r12
    jne     .short
    add     rsi, r12
    sub     rdi, r12
    cmp     dword [rel g_progress], 0
    je      .lp
    PRINT   s_dot
    jmp     .lp
.short:
    mov     rax, EFI_LOAD_ERROR
    jmp     .ret
.ok:
    xor     eax, eax
.ret:
    add     rsp, 0x28
    pop     r12
    pop     rdi
    pop     rsi
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; Console helpers (ConOut->OutputString, UCS-2)
; ----------------------------------------------------------------------------
print_str:                              ; rcx = UCS-2 string
    sub     rsp, 0x28
    mov     rdx, rcx
    mov     rcx, [rel g_conout]
    call    [rcx + CONOUT_OUTPUTSTRING]
    add     rsp, 0x28
    ret

print_ascii:                            ; rcx = ASCII string (widened in chunks)
    push    rbx
    sub     rsp, 0x20
    mov     rbx, rcx
.chunk:
    lea     rdx, [rel g_wbuf]
    xor     ecx, ecx
.cp:
    movzx   eax, byte [rbx]
    test    eax, eax
    jz      .flush
    mov     [rdx + rcx * 2], ax
    inc     rbx
    inc     ecx
    cmp     ecx, 127
    jb      .cp
.flush:
    mov     word [rdx + rcx * 2], 0
    push    rax
    sub     rsp, 8
    lea     rcx, [rel g_wbuf]
    call    print_str
    add     rsp, 8
    pop     rax
    test    eax, eax
    jnz     .chunk
    add     rsp, 0x20
    pop     rbx
    ret

print_hex:                              ; rcx = value -> "0x%016X"
    sub     rsp, 0x28
    lea     rdx, [rel g_hexbuf + 2 * 17]
    lea     r9, [rel hexdigits]
    mov     r8d, 16
.lp:
    mov     eax, ecx
    and     eax, 15
    movzx   eax, byte [r9 + rax]
    mov     [rdx], ax
    sub     rdx, 2
    shr     rcx, 4
    dec     r8d
    jnz     .lp
    lea     rcx, [rel g_hexbuf]
    call    print_str
    add     rsp, 0x28
    ret

print_dec:                              ; rcx = value
    sub     rsp, 0x28
    mov     rax, rcx
    lea     r8, [rel g_decbuf_end]
    mov     r9d, 10
.lp:
    xor     edx, edx
    div     r9
    add     edx, '0'
    sub     r8, 2
    mov     [r8], dx
    test    rax, rax
    jnz     .lp
    mov     rcx, r8
    call    print_str
    add     rsp, 0x28
    ret

print_mib:                              ; rcx = bytes -> "N MiB" (rounded up)
    sub     rsp, 0x28
    add     rcx, (1 << 20) - 1
    shr     rcx, 20
    call    print_dec
    PRINT   s_mib
    add     rsp, 0x28
    ret

print_kv_mib:                           ; rcx = label, rdx = bytes
    push    rbx
    sub     rsp, 0x20
    mov     rbx, rdx
    call    print_str
    mov     rcx, rbx
    call    print_mib
    PRINT   s_crlf
    add     rsp, 0x20
    pop     rbx
    ret

print_kv_dec:                           ; rcx = label, rdx = value (u32 zero-ext ok)
    push    rbx
    sub     rsp, 0x20
    mov     ebx, edx
    call    print_str
    mov     rcx, rbx
    call    print_dec
    PRINT   s_crlf
    add     rsp, 0x20
    pop     rbx
    ret

text_end:
    align   SECT_ALIGN, db 0

; ============================================================================
; .data
; ============================================================================
data_start:

guid_loaded_image:  dd 0x5B1B31A1
                    dw 0x9562, 0x11D2
                    db 0x8E, 0x3F, 0x00, 0xA0, 0xC9, 0x69, 0x72, 0x3B
guid_sfs:           dd 0x964E5B22
                    dw 0x6459, 0x11D2
                    db 0x8E, 0x39, 0x00, 0xA0, 0xC9, 0x69, 0x72, 0x3B
guid_acpi20:        dd 0x8868E871
                    dw 0xE4F1, 0x11D3
                    db 0xBC, 0x22, 0x00, 0x80, 0xC7, 0x3C, 0x88, 0x81
guid_acpi10:        dd 0xEB9D2D30
                    dw 0x2D88, 0x11D3
                    db 0x9A, 0x16, 0x00, 0x90, 0x27, 0x3F, 0xC1, 0x4D

hexdigits:          db '0123456789ABCDEF'

p_kernel:           dw __utf16__('\LLM\KERNEL.ELF'), 0
p_cmdline:          dw __utf16__('\LLM\CMDLINE.TXT'), 0
p_model:            dw __utf16__('\LLM\MODEL.000'), 0      ; digits patched

s_banner:           dw __utf16__(`\r\nasm-llm UEFI loader (BOOTX64.EFI) - 64-bit model handoff\r\n`), 0
s_rsdp:             dw __utf16__('ACPI RSDP        : '), 0
s_cmdline:          dw __utf16__('cmdline          : '), 0
s_kernel:           dw __utf16__('kernel           : '), 0
s_entry:            dw __utf16__(' entry '), 0
s_chunk:            dw __utf16__('chunk            : '), 0
s_colon:            dw __utf16__(' = '), 0
s_model_total:      dw __utf16__('model file total : '), 0
s_arch:             dw __utf16__(`model header     : .asmbin OK\r\n`), 0
s_dim:              dw __utf16__('  dim            : '), 0
s_hidden:           dw __utf16__('  hidden_dim     : '), 0
s_layers:           dw __utf16__('  n_layers       : '), 0
s_heads:            dw __utf16__('  n_heads        : '), 0
s_kvheads:          dw __utf16__('  n_kv_heads     : '), 0
s_headdim:          dw __utf16__('  head_dim       : '), 0
s_vocab:            dw __utf16__('  vocab          : '), 0
s_maxseq:           dw __utf16__('  max_seq        : '), 0
s_req:              dw __utf16__(`memory requirement (validated before allocation):\r\n`), 0
s_req_weights:      dw __utf16__('  weights        : '), 0
s_req_ctx:          dw __utf16__('  ctx tokens     : '), 0
s_req_kv:           dw __utf16__('  KV cache fp32  : '), 0
s_req_arena:        dw __utf16__('  runtime arena  : '), 0
s_req_heap:         dw __utf16__('  heap region    : '), 0
s_req_total:        dw __utf16__('  total          : '), 0
s_free_total:       dw __utf16__('free RAM < 16GiB : '), 0
s_free_largest:     dw __utf16__('largest free     : '), 0
s_free_after:       dw __utf16__('largest free now : '), 0
s_model_at:         dw __utf16__('model buffer     : '), 0
s_ok_fits:          dw __utf16__(`memory check     : OK (weights + heap fit)\r\n`), 0
s_reading:          dw __utf16__('reading model    : '), 0
s_read_ok:          dw __utf16__(`model loaded     : magic OK\r\n`), 0
s_jump:             dw __utf16__('ExitBootServices -> kernel '), 0
s_dash:             dw __utf16__(' - '), 0
s_dot:              dw __utf16__('.'), 0
s_mib:              dw __utf16__(' MiB'), 0
s_crlf:             dw __utf16__(`\r\n`), 0
s_err:              dw __utf16__(`\r\n[loader] FATAL: `), 0
s_status:           dw __utf16__(' status='), 0
s_w_tramp:          dw __utf16__(`warning          : SMP trampoline page 0x8000 not reservable\r\n`), 0
s_w_untied:         dw __utf16__(`warning          : untied lm_head flag (engine uses embed for logits)\r\n`), 0

s_e_loaded_image:   dw __utf16__('LoadedImage protocol unavailable'), 0
s_e_sfs:            dw __utf16__('boot volume has no SimpleFileSystem'), 0
s_e_volume:         dw __utf16__('OpenVolume failed'), 0
s_e_handoff:        dw __utf16__('cannot allocate handoff region below 4 GiB'), 0
s_e_kernel_open:    dw __utf16__('\LLM\KERNEL.ELF not found'), 0
s_e_kernel_read:    dw __utf16__('reading KERNEL.ELF failed'), 0
s_e_kernel_bad:     dw __utf16__('KERNEL.ELF is not a valid x86-64 ELF64 kernel'), 0
s_e_kernel_range:   dw __utf16__('kernel physical range (1 MiB) is not free'), 0
s_e_kernel_hdr:     dw __utf16__('KERNEL.ELF has no UEFI entry header (rebuild kernel)'), 0
s_e_oom:            dw __utf16__('AllocatePool failed'), 0
s_e_no_model:       dw __utf16__('\LLM\MODEL.000 not found'), 0
s_e_chunk_empty:    dw __utf16__('empty model chunk'), 0
s_e_chunk_sum:      dw __utf16__('model chunk sizes overflow'), 0
s_e_bad_header:     dw __utf16__('model is not a valid .asmbin (magic/version/dims)'), 0
s_e_size_mismatch:  dw __utf16__('chunk total != header file_size (missing/truncated chunk)'), 0
s_e_mmap:           dw __utf16__('GetMemoryMap failed'), 0
s_e_ram_total:      dw __utf16__('not enough RAM below 16 GiB for weights + KV cache + arena'), 0
s_e_model_alloc:    dw __utf16__('no contiguous range below 16 GiB for model weights'), 0
s_e_heap_region:    dw __utf16__('no single free region left for KV cache + arena (lower ctx=)'), 0
s_e_model_read:     dw __utf16__('reading model chunk failed'), 0
s_e_verify:         dw __utf16__('model magic mismatch after load'), 0

align 8
g_hexbuf:           dw '0', 'x'
                    times 16 dw '0'
                    dw 0
g_decbuf:           times 24 dw 0
g_decbuf_end:       dw 0
g_wbuf:             times 128 dw 0

align 8
g_saved_rsp:        dq 0
g_image:            dq 0
g_st:               dq 0
g_bs:               dq 0
g_conout:           dq 0
g_loaded_image:     dq 0
g_sfs:              dq 0
g_root:             dq 0
g_rsdp:             dq 0
g_handoff:          dq 0
g_last_status:      dq 0
g_tmp_handle:       dq 0
g_tmp_u64:          dq 0
g_tmp_addr:         dq 0
g_kbuf:             dq 0
g_phnum:            dq 0
g_phentsize:        dq 0
g_kmin:             dq 0
g_kmax:             dq 0
g_kentry:           dq 0
g_nchunks:          dq 0
g_model_size:       dq 0
g_model_pages:      dq 0
g_model_base:       dq 0
g_ctx_req:          dq 0
g_ctx:              dq 0
g_kv_bytes:         dq 0
g_arena_bytes:      dq 0
g_heap_need:        dq 0
g_free_total:       dq 0
g_free_largest:     dq 0
g_mmap_size:        dq 0
g_mmap_key:         dq 0
g_desc_size:        dq 0
g_desc_ver:         dq 0
g_has_cmdline:      dd 0
g_progress:         dd 0
g_chunk_handle:     times MAX_CHUNKS dq 0
g_chunk_size:       times MAX_CHUNKS dq 0
align 16
g_hdr:              times 128 db 0

data_end:
    align   SECT_ALIGN, db 0

; ============================================================================
; .reloc — one block of ABSOLUTE (no-op) entries: image is position-independent
; ============================================================================
reloc_start:
    dd      text_start                  ; PageRVA
    dd      12                          ; SizeOfBlock
    dw      0, 0                        ; IMAGE_REL_BASED_ABSOLUTE padding
reloc_end:
    align   SECT_ALIGN, db 0
image_end:
