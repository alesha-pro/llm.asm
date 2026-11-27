#pragma once
#include "types.h"

typedef struct {
    uint64_t addr;
    uint32_t pitch;
    uint32_t width;
    uint32_t height;
    uint8_t  bpp;
    uint8_t  red_pos, red_size;
    uint8_t  green_pos, green_size;
    uint8_t  blue_pos, blue_size;
    bool     ready;
} FbInfo;

extern FbInfo g_fb;

void fb_init(const FbInfo *info);
void fb_put_pixel(int x, int y, uint32_t rgb);
void fb_fill_rect(int x, int y, int w, int h, uint32_t rgb);
void fb_draw_border(int x, int y, int w, int h, int thickness, uint32_t rgb);
void fb_draw_char(int x, int y, char c, uint32_t fg, uint32_t bg, bool transparent_bg);
void fb_draw_str(int x, int y, const char *s, uint32_t fg, uint32_t bg, bool transparent_bg);
void fb_draw_boot_banner(int n_cores, bool llm_ready);
void fb_draw_thinking_hud(uint32_t event_num, const char *fault_name, uint64_t rip_gva,
                          const uint8_t *orig_bytes, uint8_t insn_len, int n_cores,
                          const char *live_stream);
void fb_draw_ai_verdict_hud(uint32_t event_num, const char *fault_name, uint32_t cpl,
                            uint64_t cr2, uint64_t rip_gva, uint64_t rip_gpa,
                            const uint8_t *orig_bytes, uint8_t insn_len,
                            const uint8_t *patch_bytes, uint8_t patch_len,
                            const char *action_str, const char *diagnosis,
                            int n_cores, uint32_t elapsed_ms, uint32_t gen_toks);
