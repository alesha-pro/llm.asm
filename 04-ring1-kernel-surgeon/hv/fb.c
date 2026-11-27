#include "fb.h"
#include "font8x16.h"

FbInfo g_fb;

static inline uint32_t pack_rgb(uint32_t rgb) {
    uint32_t r = (rgb >> 16) & 0xFF;
    uint32_t g = (rgb >> 8) & 0xFF;
    uint32_t b = rgb & 0xFF;
    return (r << g_fb.red_pos) | (g << g_fb.green_pos) | (b << g_fb.blue_pos);
}

void fb_init(const FbInfo *info) {
    g_fb = *info;
    if (g_fb.bpp == 32 && g_fb.addr != 0 && g_fb.width > 0 && g_fb.height > 0) {
        if (g_fb.red_size == 0 && g_fb.green_size == 0 && g_fb.blue_size == 0) {
            g_fb.red_pos = 16; g_fb.red_size = 8;
            g_fb.green_pos = 8; g_fb.green_size = 8;
            g_fb.blue_pos = 0; g_fb.blue_size = 8;
        }
        g_fb.ready = true;
    } else {
        g_fb.ready = false;
    }
}

void fb_put_pixel(int x, int y, uint32_t rgb) {
    if (!g_fb.ready) return;
    if (x < 0 || y < 0 || (uint32_t)x >= g_fb.width || (uint32_t)y >= g_fb.height) return;
    volatile uint32_t *row = (volatile uint32_t *)(uintptr_t)(g_fb.addr + (uint64_t)y * g_fb.pitch);
    row[x] = pack_rgb(rgb);
}

void fb_fill_rect(int x, int y, int w, int h, uint32_t rgb) {
    if (!g_fb.ready) return;
    uint32_t px = pack_rgb(rgb);
    for (int dy = 0; dy < h; dy++) {
        int yy = y + dy;
        if (yy < 0 || (uint32_t)yy >= g_fb.height) continue;
        volatile uint32_t *row = (volatile uint32_t *)(uintptr_t)(g_fb.addr + (uint64_t)yy * g_fb.pitch);
        for (int dx = 0; dx < w; dx++) {
            int xx = x + dx;
            if (xx < 0 || (uint32_t)xx >= g_fb.width) continue;
            row[xx] = px;
        }
    }
}

void fb_draw_border(int x, int y, int w, int h, int t, uint32_t rgb) {
    fb_fill_rect(x, y, w, t, rgb);
    fb_fill_rect(x, y + h - t, w, t, rgb);
    fb_fill_rect(x, y, t, h, rgb);
    fb_fill_rect(x + w - t, y, t, h, rgb);
}

void fb_draw_char(int x, int y, char c, uint32_t fg, uint32_t bg, bool transparent_bg) {
    if (!g_fb.ready) return;
    uint8_t uc = (uint8_t)c;
    if (uc >= 128) uc = '?';
    const uint8_t *glyph = FONT_8X16[uc];
    uint32_t fg_px = pack_rgb(fg);
    uint32_t bg_px = pack_rgb(bg);
    for (int row = 0; row < 16; row++) {
        int yy = y + row;
        if (yy < 0 || (uint32_t)yy >= g_fb.height) continue;
        volatile uint32_t *line = (volatile uint32_t *)(uintptr_t)(g_fb.addr + (uint64_t)yy * g_fb.pitch);
        uint8_t bits = glyph[row];
        for (int col = 0; col < 8; col++) {
            int xx = x + col;
            if (xx < 0 || (uint32_t)xx >= g_fb.width) continue;
            if (bits & (0x80 >> col)) line[xx] = fg_px;
            else if (!transparent_bg) line[xx] = bg_px;
        }
    }
}

void fb_draw_str(int x, int y, const char *s, uint32_t fg, uint32_t bg, bool transparent_bg) {
    int cx = x;
    while (*s) {
        if (*s == '\n') break;
        fb_draw_char(cx, y, *s++, fg, bg, transparent_bg);
        cx += 8;
    }
}

static void fb_draw_str_clip(int x, int y, const char *s, int max_chars, uint32_t fg) {
    int cx = x, n = 0;
    while (*s && n < max_chars) {
        char c = *s++;
        if (c == '\n' || c == '\r') c = ' ';
        fb_draw_char(cx, y, c, fg, 0, true);
        cx += 8;
        n++;
    }
}

static void fmt_hex_bytes(char *out, size_t cap, const uint8_t *b, uint8_t n) {
    size_t pos = 0;
    for (uint8_t i = 0; i < n && pos + 4 < cap; i++) {
        pos += (size_t)hv_snprintf(out + pos, cap - pos, "%s%02x", i ? " " : "", b[i]);
    }
    if (pos == 0 && cap > 0) out[0] = '\0';
}

void fb_draw_boot_banner(int n_cores, bool llm_ready) {
    if (!g_fb.ready) return;
    fb_fill_rect(0, 0, (int)g_fb.width, (int)g_fb.height, 0x0B0F19);
    int bw = 680, bh = 168;
    int bx = ((int)g_fb.width - bw) / 2;
    int by = ((int)g_fb.height - bh) / 2;
    if (bx < 8) bx = 8;
    if (by < 8) by = 8;
    fb_fill_rect(bx, by, bw, bh, 0x111827);
    fb_draw_border(bx, by, bw, bh, 2, 0x22D3EE);
    fb_draw_str(bx + 20, by + 18, "RING -1 KERNEL SURGEON  //  BARE-METAL AMD SVM + MiniCPM5-2B", 0x22D3EE, 0, true);
    char line[128];
    hv_snprintf(line, sizeof(line), "Engine : %s (%d Bare-Metal AVX2/FMA Cores, Zero OS/libc)",
                llm_ready ? "MiniCPM5-2B GRPO Q4_K_M ACTIVE" : "Stage1 Fallback", n_cores);
    fb_draw_str(bx + 20, by + 48, line, llm_ready ? 0x34D399 : 0xFBBF24, 0, true);
    hv_snprintf(line, sizeof(line), "Isolat : 2 GiB Weights + 64 MiB KV/Arena UNMAPPED in AMD NPT");
    fb_draw_str(bx + 20, by + 74, line, 0xE5E7EB, 0, true);
    hv_snprintf(line, sizeof(line), "FB     : %ux%ux%u @ 0x%lx (shared with guest efifb)",
                g_fb.width, g_fb.height, g_fb.bpp, g_fb.addr);
    fb_draw_str(bx + 20, by + 100, line, 0x9CA3AF, 0, true);
    fb_draw_str(bx + 20, by + 130, "Launching Linux 5.15 guest via VMRUN (intercepting #PF #DE #UD)...", 0x60A5FA, 0, true);
}

void fb_draw_thinking_hud(uint32_t event_num, const char *fault_name, uint64_t rip_gva,
                          const uint8_t *orig_bytes, uint8_t insn_len, int n_cores,
                          const char *live_stream) {
    if (!g_fb.ready) return;
    int pw = 660, ph = 248;
    if ((int)g_fb.width < pw + 24) pw = (int)g_fb.width - 16;
    int px = (int)g_fb.width - pw - 16;
    int py = 16;
    if (px < 8) px = 8;

    fb_fill_rect(px, py, pw, ph, 0x0B101B);
    fb_fill_rect(px, py, pw, 28, 0xB45309);
    fb_draw_border(px, py, pw, ph, 2, 0xFBBF24);

    char buf[160], hex_o[64];
    fmt_hex_bytes(hex_o, sizeof(hex_o), orig_bytes, insn_len);
    hv_snprintf(buf, sizeof(buf), " [ RING -1 AI SURGEON ]  LIVE INFERENCE #%u (%d BARE-METAL CORES)", event_num, n_cores);
    fb_draw_str_clip(px + 8, py + 6, buf, 80, 0xFFFFFF);

    hv_snprintf(buf, sizeof(buf), "MODEL  : MiniCPM5-2B GRPO Q4_K_M (Isolated L1 RAM, Zero OS)");
    fb_draw_str_clip(px + 12, py + 38, buf, 78, 0x22D3EE);

    hv_snprintf(buf, sizeof(buf), "FAULT  : %s   RIP=0x%016lx", fault_name, rip_gva);
    fb_draw_str_clip(px + 12, py + 60, buf, 78, 0xFCA5A5);

    hv_snprintf(buf, sizeof(buf), "BYTES  : [%u B] %s", insn_len, hex_o);
    fb_draw_str_clip(px + 12, py + 82, buf, 78, 0xFDE68A);

    fb_draw_str_clip(px + 12, py + 112, "STREAMING RING -1 LLM OUTPUT:", 78, 0x9CA3AF);
    if (live_stream && live_stream[0]) {
        // Print up to 3 lines of streamed tokens
        const char *p = live_stream;
        int ly = py + 134;
        for (int line_i = 0; line_i < 3 && *p; line_i++) {
            int k = 0;
            char lbuf[80];
            while (*p && *p != '\n' && k < 76) lbuf[k++] = *p++;
            lbuf[k] = '\0';
            if (*p == '\n') p++;
            fb_draw_str_clip(px + 12, ly, lbuf, 78, 0x6EE7B7);
            ly += 20;
        }
    } else {
        fb_draw_str_clip(px + 12, py + 134, "Prefilling Ring -1 telemetry prompt across AVX2 cores...", 78, 0xFBBF24);
    }

    fb_fill_rect(px + 12, py + ph - 32, pw - 24, 22, 0x78350F);
    fb_draw_str_clip(px + 18, py + ph - 29, "GUEST FROZEN IN RING -1  //  COMPUTING SAFE MACHINE-CODE VERDICT...", 76, 0xFEF3C7);
}

void fb_draw_ai_verdict_hud(uint32_t event_num, const char *fault_name, uint32_t cpl,
                            uint64_t cr2, uint64_t rip_gva, uint64_t rip_gpa,
                            const uint8_t *orig_bytes, uint8_t insn_len,
                            const uint8_t *patch_bytes, uint8_t patch_len,
                            const char *action_str, const char *diagnosis,
                            int n_cores, uint32_t elapsed_ms, uint32_t gen_toks) {
    if (!g_fb.ready) return;
    int pw = 660, ph = 248;
    if ((int)g_fb.width < pw + 24) pw = (int)g_fb.width - 16;
    int px = (int)g_fb.width - pw - 16;
    int py = 16;
    if (px < 8) px = 8;

    bool is_patch = (hv_strcmp(action_str, "PATCH_BYTES") == 0);
    bool is_pass  = (hv_strcmp(action_str, "PASS_TO_LINUX") == 0);

    uint32_t hdr_bg  = is_patch ? 0x065F46 : (is_pass ? 0x1D4ED8 : 0x991B1B);
    uint32_t bdr_col = is_patch ? 0x34D399 : (is_pass ? 0x60A5FA : 0xF87171);

    fb_fill_rect(px, py, pw, ph, 0x0B101B);
    fb_fill_rect(px, py, pw, 28, hdr_bg);
    fb_draw_border(px, py, pw, ph, 2, bdr_col);

    char buf[160];
    hv_snprintf(buf, sizeof(buf), " [ RING -1 AI SURGEON ]  EVENT #%u  |  %s", event_num, action_str);
    fb_draw_str_clip(px + 8, py + 6, buf, 80, 0xFFFFFF);

    int y = py + 36;
    uint32_t secs = elapsed_ms / 1000U;
    uint32_t tenths = (elapsed_ms % 1000U) / 100U;
    hv_snprintf(buf, sizeof(buf), "MODEL  : MiniCPM5-2B Q4_K_M @ %d L1 Cores (%u.%us, %u tok)", n_cores, secs, tenths, gen_toks);
    fb_draw_str_clip(px + 12, y, buf, 78, 0x22D3EE);
    y += 22;

    hv_snprintf(buf, sizeof(buf), "FAULT  : %s  CPL=%u  CR2=0x%016lx", fault_name, cpl, cr2);
    fb_draw_str_clip(px + 12, y, buf, 78, 0xFCA5A5);
    y += 22;

    hv_snprintf(buf, sizeof(buf), "RIP GVA: 0x%016lx  GPA: 0x%016lx", rip_gva, rip_gpa);
    fb_draw_str_clip(px + 12, y, buf, 78, 0x93C5FD);
    y += 22;

    hv_snprintf(buf, sizeof(buf), "DIAG   : %s", diagnosis && diagnosis[0] ? diagnosis : "(none)");
    fb_draw_str_clip(px + 12, y, buf, 78, 0xF9FAFB);
    y += 22;

    char hex_orig[64], hex_patch[64];
    fmt_hex_bytes(hex_orig, sizeof(hex_orig), orig_bytes, insn_len);
    if (is_patch && patch_len > 0) fmt_hex_bytes(hex_patch, sizeof(hex_patch), patch_bytes, patch_len);
    else hv_snprintf(hex_patch, sizeof(hex_patch), "NONE");

    hv_snprintf(buf, sizeof(buf), "BEFORE : [%u B] %s", insn_len, hex_orig);
    fb_draw_str_clip(px + 12, y, buf, 78, 0xFDE68A);
    y += 22;

    hv_snprintf(buf, sizeof(buf), "PATCH  : [%u B] %s", is_patch ? patch_len : 0, hex_patch);
    fb_draw_str_clip(px + 12, y, buf, 78, is_patch ? 0x6EE7B7 : 0xFCA5A5);
    y += 28;

    fb_fill_rect(px + 12, y - 4, pw - 24, 24, hdr_bg);
    const char *status_line = is_patch
        ? "STATUS : LIVE-PATCHED IN GUEST .TEXT VIA NPT -> RESUMING VMRUN"
        : (is_pass
            ? "STATUS : BENIGN LINUX EXCEPTION -> RE-INJECTED TO GUEST KERNEL"
            : "STATUS : UNSAFE STATE (LOCK/DIRTY) -> HALTED FOR FORENSIC DUMP");
    fb_draw_str_clip(px + 18, y, status_line, 76, 0xFFFFFF);
}
