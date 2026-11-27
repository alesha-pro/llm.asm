; ============================================================================
; hv/svm_asm.asm — AMD SVM VMRUN / VMLOAD / VMSAVE + AVX2 / XSAVE Helpers
; ============================================================================
bits 64
section .text

global svm_vmrun
svm_vmrun:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15

    push rdi                ; [rsp + 16] = guest_vmcb_pa
    push rsi                ; [rsp + 8]  = host_vmcb_pa
    push rdx                ; [rsp + 0]  = GuestRegs *regs

    clgi

    mov rax, rsi
    vmsave

    mov rax, rdi
    vmload

    mov rbx, [rdx + 0]
    mov rcx, [rdx + 8]
    mov rsi, [rdx + 24]
    mov rdi, [rdx + 32]
    mov rbp, [rdx + 40]
    mov r8,  [rdx + 48]
    mov r9,  [rdx + 56]
    mov r10, [rdx + 64]
    mov r11, [rdx + 72]
    mov r12, [rdx + 80]
    mov r13, [rdx + 88]
    mov r14, [rdx + 96]
    mov r15, [rdx + 104]
    mov rax, [rsp + 16]
    mov rdx, [rdx + 16]

    vmrun

    mov rax, [rsp + 0]
    mov [rax + 0],   rbx
    mov [rax + 8],   rcx
    mov [rax + 16],  rdx
    mov [rax + 24],  rsi
    mov [rax + 32],  rdi
    mov [rax + 40],  rbp
    mov [rax + 48],  r8
    mov [rax + 56],  r9
    mov [rax + 64],  r10
    mov [rax + 72],  r11
    mov [rax + 80],  r12
    mov [rax + 88],  r13
    mov [rax + 96],  r14
    mov [rax + 104], r15

    mov rax, [rsp + 16]
    vmsave

    mov rax, [rsp + 8]
    vmload

    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

global read_msr
read_msr:
    mov ecx, edi
    rdmsr
    shl rdx, 32
    or  rax, rdx
    ret

global write_msr
write_msr:
    mov ecx, edi
    mov eax, esi
    mov rdx, rsi
    shr rdx, 32
    wrmsr
    ret

global read_tsc
read_tsc:
    rdtsc
    shl rdx, 32
    or  rax, rdx
    ret

global cpuid_raw
cpuid_raw:
    push rbx
    mov r10, rdx
    mov r11, rcx
    mov eax, edi
    mov ecx, esi
    cpuid
    mov [r10], eax
    mov [r11], ebx
    mov [r8],  ecx
    mov [r9],  edx
    pop rbx
    ret

global flush_host_tlb
flush_host_tlb:
    mov rax, cr3
    mov cr3, rax
    ret

; Enable SSE + AVX2 + FMA + F16C on current CPU (CR0.EM=0, TS=0, MP=1; CR4.OSFXSR|OSXMMEXCPT|OSXSAVE; XCR0=7)
global enable_avx2_cpu
enable_avx2_cpu:
    mov rax, cr0
    and rax, ~((1 << 2) | (1 << 3))     ; clear EM, TS
    or  rax, (1 << 1)                   ; set MP
    mov cr0, rax

    mov rax, cr4
    or  rax, (1 << 9) | (1 << 10) | (1 << 18) ; OSFXSR | OSXMMEXCPT | OSXSAVE
    mov cr4, rax

    xor ecx, ecx
    xgetbv
    or  eax, 7                          ; x87 (1) | SSE (2) | AVX (4)
    xor edx, edx
    xsetbv
    ret

; Save/restore guest FPU+SSE+AVX state cleanly around Ring -1 AVX2 LLM inference
; void xsave_state(void *buf_64aligned)
global xsave_state
xsave_state:
    mov eax, 7
    xor edx, edx
    xsave [rdi]
    ret

; void xrstor_state(const void *buf_64aligned)
global xrstor_state
xrstor_state:
    vzeroupper
    mov eax, 7
    xor edx, edx
    xrstor [rdi]
    ret

; uint64_t get_xcr0(void) / void set_xcr0(uint64_t v)
global get_xcr0
get_xcr0:
    xor ecx, ecx
    xgetbv
    shl rdx, 32
    or  rax, rdx
    ret

global set_xcr0
set_xcr0:
    xor ecx, ecx
    mov eax, edi
    mov rdx, rdi
    shr rdx, 32
    xsetbv
    ret
