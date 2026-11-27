; ============================================================================
; os/fbcon.asm — Linear framebuffer text console (UEFI GOP via GRUB Multiboot 1)
;
; GRUB on UEFI has no VGA text mode, so the 80x25 console at 0xB8000 shows
; nothing. The Multiboot header asks for EGA text first; GRUB-EFI then falls
; back to a GOP framebuffer and reports it in the Multiboot info (flags bit 12).
;
; When a direct-RGB framebuffer is offered, fb_init points the VGA text
; driver at an 80x25 cell shadow buffer in .bss and this module draws each cell
; (24 or 32 bpp) with an 8x8 public-domain font, doubled vertically to 8x16, scaled 1x or 2x
; and centred. Otherwise nothing changes: BIOS boots keep real VGA text mode.
; ============================================================================

%include "os_constants.inc"

extern g_mb_magic
extern g_mb_info
extern g_vga_base

global fb_init
global fb_draw_cell
global fb_redraw
global g_fb_on

section .text

; void fb_init(void)
; Enables framebuffer mode if GRUB passed a usable framebuffer. Preserves
; callee-saved registers (SysV).
fb_init:
    push    rbx
    cmp     dword [rel g_mb_magic], MB_BOOTLOADER_MAGIC
    jne     .done
    mov     esi, [rel g_mb_info]
    test    dword [rsi + MBI_FLAGS], MBI_FLAG_FRAMEBUFFER
    jz      .done
    cmp     byte [rsi + MBI_FB_TYPE], MBI_FB_TYPE_RGB
    jne     .done
    movzx   eax, byte [rsi + MBI_FB_BPP]    ; 24 or 32 bpp (GOP modes vary)
    cmp     eax, 32
    je      .bpp_ok
    cmp     eax, 24
    jne     .done
.bpp_ok:
    shr     eax, 3
    mov     [rel fb_bpp_bytes], eax
    cmp     byte [rsi + MBI_FB_RED_SIZE], 8
    jne     .done
    cmp     byte [rsi + MBI_FB_GREEN_SIZE], 8
    jne     .done
    cmp     byte [rsi + MBI_FB_BLUE_SIZE], 8
    jne     .done

    mov     r8d, [rsi + MBI_FB_WIDTH]
    mov     r9d, [rsi + MBI_FB_HEIGHT]
    mov     r10d, [rsi + MBI_FB_PITCH]
    cmp     r8d, FB_GRID_W
    jb      .done
    cmp     r9d, FB_GRID_H
    jb      .done
    mov     eax, r8d                    ; pitch must hold a full row of pixels
    imul    eax, [rel fb_bpp_bytes]
    cmp     r10d, eax
    jb      .done

    ; Framebuffer must sit inside the 0..4 GiB identity map built in boot.asm
    mov     rdi, [rsi + MBI_FB_ADDR]
    test    rdi, rdi
    jz      .done
    mov     eax, r9d
    imul    rax, r10                    ; rax = pitch * height (bytes)
    mov     rbx, rax
    add     rax, rdi
    jc      .done
    mov     rdx, FB_MAP_LIMIT
    cmp     rax, rdx
    ja      .done

    ; scale = min(width / 640, height / 400), clamped to 1..FB_MAX_SCALE
    xor     edx, edx
    mov     eax, r8d
    mov     ecx, FB_GRID_W
    div     ecx
    mov     r11d, eax
    xor     edx, edx
    mov     eax, r9d
    mov     ecx, FB_GRID_H
    div     ecx
    cmp     eax, r11d
    cmovb   r11d, eax
    cmp     r11d, FB_MAX_SCALE
    jbe     .scale_ok
    mov     r11d, FB_MAX_SCALE
.scale_ok:
    mov     [rel fb_scale], r11d
    lea     eax, [r11 * 2]
    mov     [rel fb_line_reps], eax     ; each 8x8 glyph row covers 2*scale lines
    mov     [rel fb_pitch], r10
    mov     eax, r11d
    shl     eax, 3                      ; 8 px * bytes per pixel * scale
    imul    eax, [rel fb_bpp_bytes]
    mov     [rel fb_col_bytes], rax
    mov     eax, r11d
    shl     eax, 4                      ; 16 lines * scale
    imul    rax, r10
    mov     [rel fb_row_bytes], rax

    ; origin = addr + ((h - 400*s) / 2) * pitch + ((w - 640*s) / 2) * Bpp
    imul    eax, r11d, FB_GRID_H
    mov     ecx, r9d
    sub     ecx, eax
    shr     ecx, 1
    imul    rcx, r10
    imul    eax, r11d, FB_GRID_W
    mov     edx, r8d
    sub     edx, eax
    shr     edx, 1
    imul    edx, [rel fb_bpp_bytes]
    add     rcx, rdx
    add     rcx, rdi
    mov     [rel fb_origin], rcx

    ; Clear the whole framebuffer (GRUB leaves its menu there)
    ; rdi = framebuffer base, rbx = pitch * height
    mov     rcx, rbx
    shr     rcx, 2
    xor     eax, eax
    rep     stosd

    ; Build the 16-colour CGA palette in the framebuffer's pixel layout
    movzx   r8d, byte [rsi + MBI_FB_RED_POS]
    movzx   r9d, byte [rsi + MBI_FB_GREEN_POS]
    movzx   r10d, byte [rsi + MBI_FB_BLUE_POS]
    lea     rsi, [rel cga_rgb]
    lea     r11, [rel fb_palette]
    xor     edx, edx
.pal:
    movzx   eax, byte [rsi]
    mov     ecx, r8d
    shl     eax, cl
    movzx   ebx, byte [rsi + 1]
    mov     ecx, r9d
    shl     ebx, cl
    or      eax, ebx
    movzx   ebx, byte [rsi + 2]
    mov     ecx, r10d
    shl     ebx, cl
    or      eax, ebx
    mov     [r11 + rdx * 4], eax
    add     rsi, 3
    inc     edx
    cmp     edx, 16
    jb      .pal

    ; The cleared screen equals all-blank cells: record that so the first
    ; redraw only paints cells that hold something.
    lea     rdi, [rel fb_drawn]
    mov     eax, FB_BLANK_CELL | (FB_BLANK_CELL << 16)
    mov     ecx, (VGA_COLS * VGA_ROWS) / 2
    rep     stosd

    lea     rax, [rel fb_shadow]
    mov     [rel g_vga_base], rax
    mov     byte [rel g_fb_on], 1
.done:
    pop     rbx
    ret

; void fb_draw_cell(uint32 index in ecx)
; Draws one 80x25 shadow cell. Preserves every general-purpose register.
fb_draw_cell:
    push    rax
    push    rbx
    push    rcx
    push    rdx
    push    rsi
    push    rdi
    push    r8
    push    r9
    push    r10
    push    r11
    push    r12
    push    r13
    push    r14
    push    r15

    mov     r15d, [rel fb_bpp_bytes]
    lea     rsi, [rel fb_shadow]
    movzx   ebx, word [rsi + rcx * 2]   ; bl = char, bh = attribute
    lea     rsi, [rel fb_drawn]
    mov     [rsi + rcx * 2], bx

    mov     eax, ecx
    xor     edx, edx
    mov     ecx, VGA_COLS
    div     ecx                         ; eax = row, edx = column
    imul    rax, [rel fb_row_bytes]
    imul    rdx, [rel fb_col_bytes]
    mov     rdi, [rel fb_origin]
    add     rdi, rax
    add     rdi, rdx

    lea     r11, [rel fb_palette]
    movzx   eax, bh
    and     eax, 0x0F
    mov     r8d, [r11 + rax * 4]        ; foreground
    movzx   eax, bh
    shr     eax, 4
    and     eax, 0x0F
    mov     r9d, [r11 + rax * 4]        ; background

    movzx   eax, bl
    sub     eax, 0x20
    cmp     eax, 0x7E - 0x20
    jbe     .glyph_ok
    xor     eax, eax                    ; anything else draws as a space
.glyph_ok:
    lea     rsi, [rel fb_font]
    lea     rsi, [rsi + rax * 8]

    mov     r10d, 8                     ; glyph rows
.row:
    movzx   r11d, byte [rsi]
    inc     rsi
    mov     ebx, [rel fb_line_reps]
.line:
    mov     rdx, rdi
    mov     eax, r11d
    mov     ecx, 8                      ; pixels per glyph row
.px:
    mov     r12d, r9d
    test    eax, 1                      ; bit 0 is the leftmost pixel
    cmovnz  r12d, r8d
    shr     eax, 1
    mov     r14d, r12d
    shr     r14d, 16                    ; third byte for 24 bpp
    mov     r13d, [rel fb_scale]
.rep:
    cmp     r15d, 4
    jne     .rgb24
    mov     [rdx], r12d
    jmp     .next
.rgb24:
    mov     [rdx], r12w
    mov     [rdx + 2], r14b
.next:
    add     rdx, r15
    dec     r13d
    jnz     .rep
    dec     ecx
    jnz     .px
    add     rdi, [rel fb_pitch]
    dec     ebx
    jnz     .line
    dec     r10d
    jnz     .row

    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     r11
    pop     r10
    pop     r9
    pop     r8
    pop     rdi
    pop     rsi
    pop     rdx
    pop     rcx
    pop     rbx
    pop     rax
    ret

; void fb_redraw(void) — repaint every cell whose content differs from what
; is on screen (after clear or scroll). Preserves every GPR.
fb_redraw:
    push    rax
    push    rcx
    push    rsi
    push    rdi
    lea     rsi, [rel fb_shadow]
    lea     rdi, [rel fb_drawn]
    xor     ecx, ecx
.lp:
    movzx   eax, word [rsi + rcx * 2]
    cmp     ax, [rdi + rcx * 2]
    je      .next
    call    fb_draw_cell
.next:
    inc     ecx
    cmp     ecx, VGA_COLS * VGA_ROWS
    jb      .lp
    pop     rdi
    pop     rsi
    pop     rcx
    pop     rax
    ret

section .rodata
; CGA/VGA text palette as R, G, B bytes for attributes 0..15
cga_rgb:
    db 0x00,0x00,0x00, 0x00,0x00,0xAA, 0x00,0xAA,0x00, 0x00,0xAA,0xAA
    db 0xAA,0x00,0x00, 0xAA,0x00,0xAA, 0xAA,0x55,0x00, 0xAA,0xAA,0xAA
    db 0x55,0x55,0x55, 0x55,0x55,0xFF, 0x55,0xFF,0x55, 0x55,0xFF,0xFF
    db 0xFF,0x55,0x55, 0xFF,0x55,0xFF, 0xFF,0xFF,0x55, 0xFF,0xFF,0xFF

fb_font:
%include "font8x8.inc"

section .bss
alignb 16
fb_shadow:      resw VGA_COLS * VGA_ROWS    ; cells as the console wrote them
fb_drawn:       resw VGA_COLS * VGA_ROWS    ; cells as currently painted
fb_palette:     resd 16
fb_origin:      resq 1
fb_pitch:       resq 1
fb_row_bytes:   resq 1
fb_col_bytes:   resq 1
fb_scale:       resd 1
fb_bpp_bytes:   resd 1
fb_line_reps:   resd 1
g_fb_on:        resb 1
