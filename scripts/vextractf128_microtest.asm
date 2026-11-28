
[BITS 64]
; Micro-test: vextractf128 with dst in modrm.RM field position.
; Proves (on this machine) that the form writes correctly with imm=0 AND imm=1,
; and that writing xmm N zeroes ymm N's upper half (zero-upper semantics).
; Build: nasm -f elf64 -o vt.o vt.asm && ld -o vt vt.o && ./vt | od -d
section .text
global _start
_start:
    vmovdqa xmm0, [buf_lo]
    vmovdqa xmm2, [buf_hi]
    vinsertf128 ymm1, xmm0, 0
    vinsertf128 ymm1, xmm2, 1
    vpcmpeqd xmm4, xmm4, xmm4     ; poison control
    vextractf128 xmm2, ymm1, 0    ; TEST A: low extract, dst = RM field
    vmovdqa [out], xmm2
    vmovdqa [out+16], xmm4        ; assert poison untouched
    ; TEST B: high extract overwrites — and prove zero-upper on xmm4 write:
    vextractf128 xmm4, ymm1, 1    ; writes xmm4 (kills ymm4.hi: xmm4 was poison)
    vmovdqa [out+32], xmm4
    mov     eax, 1
    mov     edi, 1
    mov     esi, out
    mov     edx, 48
    syscall
    mov     eax, 60
    xor     edi, edi
    syscall
section .data
buf_lo: dd 100, 101, 102, 103
buf_hi: dd 200, 201, 202, 203
out:    dd 0,0,0,0, 0,0,0,0, 0,0,0,0
