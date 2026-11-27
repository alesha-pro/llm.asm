#include "linux_boot.h"

#define HV_ARENA_START 0x00200000ULL
#define HV_ARENA_END   0x00400000ULL

#define BOOT_PARAMS_PA 0x00600000ULL
#define CMDLINE_PA     0x00601000ULL
#define RSDP_COPY_PA   0x00602000ULL
#define GUEST_GDT_PA   0x00603000ULL
#define GUEST_STACK_PA 0x0060F000ULL
#define GUEST_PT_BASE  0x00610000ULL  // PML4 + PDPT + 4x PD = 6 pages (24 KiB)

#define INITRD_DEST_PA 0x08000000ULL  // 128 MiB
#define BOUNCE_BASE_PA 0x18000000ULL  // 384 MiB (safe temporary bounce area)

typedef struct __attribute__((packed)) {
    uint32_t type;
    uint32_t size;
} Mb2Tag;

typedef struct __attribute__((packed)) {
    uint32_t type;
    uint32_t size;
    uint32_t mod_start;
    uint32_t mod_end;
    char     cmdline[];
} Mb2TagModule;

typedef struct __attribute__((packed)) {
    uint32_t type;
    uint32_t size;
    uint32_t entry_size;
    uint32_t entry_version;
} Mb2TagMmap;

typedef struct __attribute__((packed)) {
    uint64_t addr;
    uint64_t len;
    uint32_t type;
    uint32_t zero;
} Mb2MmapEntry;

typedef struct __attribute__((packed)) {
    uint32_t type;
    uint32_t size;
    uint64_t fb_addr;
    uint32_t fb_pitch;
    uint32_t fb_width;
    uint32_t fb_height;
    uint8_t  fb_bpp;
    uint8_t  fb_type;
    uint16_t reserved;
    // RGB fields when fb_type == 1
    uint8_t  red_pos;
    uint8_t  red_mask_size;
    uint8_t  green_pos;
    uint8_t  green_mask_size;
    uint8_t  blue_pos;
    uint8_t  blue_mask_size;
} Mb2TagFb;

static bool str_contains(const char *hay, const char *needle) {
    size_t nlen = hv_strlen(needle);
    if (nlen == 0) return true;
    for (size_t i = 0; hay[i]; i++) {
        if (hv_strncmp(hay + i, needle, nlen) == 0) return true;
    }
    return false;
}

bool mb2_parse(uint64_t magic, uint64_t mbi_addr, BootAssets *out) {
    hv_memset(out, 0, sizeof(*out));
    if ((uint32_t)magic != MB2_BOOTLOADER_MAGIC) {
        kprintf("[MB2] ERROR: bad magic 0x%lx (expected 0x%x)\n", magic, MB2_BOOTLOADER_MAGIC);
        return false;
    }

    uint32_t total_size = *(const uint32_t *)(uintptr_t)mbi_addr;
    uint64_t cur = mbi_addr + 8;
    uint64_t end = mbi_addr + total_size;
    int mod_idx = 0;

    while (cur < end) {
        const Mb2Tag *tag = (const Mb2Tag *)(uintptr_t)cur;
        if (tag->type == 0) break;

        if (tag->type == 3) {
            // Module
            const Mb2TagModule *m = (const Mb2TagModule *)tag;
            uint32_t sz = m->mod_end - m->mod_start;
            kprintf("[MB2] Module %d: 0x%x..0x%x (%u bytes) \"%s\"\n",
                    mod_idx, m->mod_start, m->mod_end, sz, m->cmdline);
            if (str_contains(m->cmdline, "vmlinuz") || str_contains(m->cmdline, "kernel") ||
                (mod_idx == 0 && out->vmlinuz_addr == 0)) {
                out->vmlinuz_addr = m->mod_start;
                out->vmlinuz_size = sz;
            } else if (str_contains(m->cmdline, "initr") || mod_idx == 1) {
                out->initrd_addr = m->mod_start;
                out->initrd_size = sz;
            }
            mod_idx++;
        } else if (tag->type == 6) {
            // Memory map
            const Mb2TagMmap *mm = (const Mb2TagMmap *)tag;
            uint64_t ecur = cur + sizeof(Mb2TagMmap);
            uint64_t eend = cur + tag->size;
            while (ecur + sizeof(Mb2MmapEntry) <= eend && out->mmap_count < 64) {
                const Mb2MmapEntry *e = (const Mb2MmapEntry *)(uintptr_t)ecur;
                out->mmap[out->mmap_count].addr = e->addr;
                out->mmap[out->mmap_count].len  = e->len;
                out->mmap[out->mmap_count].type = e->type;
                out->mmap_count++;
                ecur += mm->entry_size;
            }
        } else if (tag->type == 8) {
            // Framebuffer
            const Mb2TagFb *fb = (const Mb2TagFb *)tag;
            out->fb.addr       = fb->fb_addr;
            out->fb.pitch      = fb->fb_pitch;
            out->fb.width      = fb->fb_width;
            out->fb.height     = fb->fb_height;
            out->fb.bpp        = fb->fb_bpp;
            if (fb->fb_type == 1) {
                out->fb.red_pos    = fb->red_pos;
                out->fb.red_size   = fb->red_mask_size;
                out->fb.green_pos  = fb->green_pos;
                out->fb.green_size = fb->green_mask_size;
                out->fb.blue_pos   = fb->blue_pos;
                out->fb.blue_size  = fb->blue_mask_size;
            }
            kprintf("[MB2] Framebuffer: %ux%ux%u pitch=%u @ 0x%lx (R:%u/%u G:%u/%u B:%u/%u)\n",
                    out->fb.width, out->fb.height, out->fb.bpp, out->fb.pitch, out->fb.addr,
                    out->fb.red_pos, out->fb.red_size,
                    out->fb.green_pos, out->fb.green_size,
                    out->fb.blue_pos, out->fb.blue_size);
        } else if (tag->type == 14 || tag->type == 15) {
            // ACPI RSDP (14 = v1 20B, 15 = v2 36B; prefer v2 if both present)
            uint32_t plen = tag->size - 8;
            if (tag->type == 15 || out->acpi_rsdp_src == 0) {
                out->acpi_rsdp_src = cur + 8;
                out->acpi_rsdp_len = plen;
                kprintf("[MB2] ACPI RSDP (tag %u, %u bytes) @ 0x%lx\n", tag->type, plen, out->acpi_rsdp_src);
            }
        }

        cur += (tag->size + 7U) & ~7U;
    }

    return out->vmlinuz_addr != 0 && out->vmlinuz_size > 1024;
}

typedef struct __attribute__((packed)) {
    uint64_t addr;
    uint64_t size;
    uint32_t type;
} E820Entry;

static void e820_push(E820Entry *tbl, uint8_t *cnt, uint64_t addr, uint64_t size, uint32_t type) {
    if (size == 0 || *cnt >= 126) return;
    tbl[*cnt].addr = addr;
    tbl[*cnt].size = size;
    tbl[*cnt].type = type;
    (*cnt)++;
}

static void e820_add_carved(E820Entry *tbl, uint8_t *cnt, uint64_t addr, uint64_t size, uint32_t type,
                            uint64_t cut_start, uint64_t cut_end, uint32_t cut_type) {
    uint64_t end = addr + size;
    if (type != 1 || end <= cut_start || addr >= cut_end) {
        e820_push(tbl, cnt, addr, size, type);
        return;
    }
    if (addr < cut_start) {
        e820_push(tbl, cnt, addr, cut_start - addr, type);
    }
    uint64_t cs = (addr > cut_start) ? addr : cut_start;
    uint64_t ce = (end < cut_end) ? end : cut_end;
    e820_push(tbl, cnt, cs, ce - cs, cut_type);
    if (end > cut_end) {
        e820_push(tbl, cnt, cut_end, end - cut_end, type);
    }
}

bool linux_prepare_guest(const BootAssets *assets, LinuxGuestInit *init_out) {
    // 1. Copy both Multiboot2 modules into a safe bounce area first so nothing overlaps
    uint64_t vml_bounce = BOUNCE_BASE_PA;
    uint64_t ird_bounce = (BOUNCE_BASE_PA + assets->vmlinuz_size + 0xFFFULL) & ~0xFFFULL;
    hv_memmove((void *)(uintptr_t)vml_bounce, (const void *)(uintptr_t)assets->vmlinuz_addr, assets->vmlinuz_size);
    if (assets->initrd_addr && assets->initrd_size) {
        hv_memmove((void *)(uintptr_t)ird_bounce, (const void *)(uintptr_t)assets->initrd_addr, assets->initrd_size);
    }

    const uint8_t *bz = (const uint8_t *)(uintptr_t)vml_bounce;
    if (*(const uint32_t *)(bz + 0x202) != 0x53726448U) { // "HdrS"
        kprintf("[LINUX] ERROR: invalid bzImage magic (expected HdrS)\n");
        return false;
    }

    uint16_t boot_proto = *(const uint16_t *)(bz + 0x206);
    uint8_t  setup_sects = bz[0x1F1];
    if (setup_sects == 0) setup_sects = 4;
    uint32_t payload_off = ((uint32_t)setup_sects + 1U) * 512U;
    uint32_t payload_len = assets->vmlinuz_size - payload_off;
    uint64_t pref_addr   = *(const uint64_t *)(bz + 0x258);
    uint32_t init_size   = *(const uint32_t *)(bz + 0x260);
    if (pref_addr == 0) pref_addr = 0x01000000ULL;

    kprintf("[LINUX] bzImage proto=0x%x setup_sects=%u payload=%u B pref_addr=0x%lx init_size=0x%x\n",
            boot_proto, setup_sects, payload_len, pref_addr, init_size);

    // 2. Copy protected-mode kernel payload to pref_addr (0x01000000) and initramfs to 0x08000000
    hv_memmove((void *)(uintptr_t)pref_addr, bz + payload_off, payload_len);
    if (assets->initrd_size) {
        hv_memmove((void *)(uintptr_t)INITRD_DEST_PA, (const void *)(uintptr_t)ird_bounce, assets->initrd_size);
    }

    // 3. Build boot_params ("zero page") at BOOT_PARAMS_PA (0x00600000)
    uint8_t *bp = (uint8_t *)(uintptr_t)BOOT_PARAMS_PA;
    hv_memset(bp, 0, 4096);

    // Copy setup_header [0x1F1 .. hdr_end)
    uint32_t hdr_end = 0x202U + bz[0x201];
    if (hdr_end < 0x268U) hdr_end = 0x268U;
    if (hdr_end > 0x290U) hdr_end = 0x290U;
    hv_memcpy(bp + 0x1F1, bz + 0x1F1, hdr_end - 0x1F1);

    // Command line at CMDLINE_PA (0x00601000)
    const char *cmdline =
        "console=tty0 console=ttyS0,115200 earlyprintk=serial,ttyS0,115200 "
        "nokaslr nr_cpus=1 nosmp noapic_debug rdinit=/init";
    char *cmd_dst = (char *)(uintptr_t)CMDLINE_PA;
    hv_memset(cmd_dst, 0, 4096);
    hv_memcpy(cmd_dst, cmdline, hv_strlen(cmdline) + 1);

    // Patch boot_params.hdr fields (verified against linux-headers-5.15.0-185-generic)
    bp[0x210] = 0xFF;                                     // type_of_loader = undefined/custom
    bp[0x211] = (uint8_t)((bp[0x211] | 0x81) & ~0x20);    // LOADED_HIGH | CAN_USE_HEAP, clear QUIET
    *(uint32_t *)(bp + 0x218) = (uint32_t)INITRD_DEST_PA; // ramdisk_image
    *(uint32_t *)(bp + 0x21C) = assets->initrd_size;      // ramdisk_size
    *(uint32_t *)(bp + 0x228) = (uint32_t)CMDLINE_PA;     // cmd_line_ptr

    // 4. Populate screen_info (offset 0x000) for Linux built-in efifb!
    if (assets->fb.ready) {
        bp[0x07] = 80;                                    // orig_video_cols
        bp[0x0E] = 25;                                    // orig_video_lines
        bp[0x0F] = 0x70;                                  // orig_video_isVGA = VIDEO_TYPE_EFI
        *(uint16_t *)(bp + 0x12) = (uint16_t)assets->fb.width;
        *(uint16_t *)(bp + 0x14) = (uint16_t)assets->fb.height;
        *(uint16_t *)(bp + 0x16) = (uint16_t)assets->fb.bpp;
        *(uint32_t *)(bp + 0x18) = (uint32_t)assets->fb.addr;
        *(uint32_t *)(bp + 0x1C) = assets->fb.pitch * assets->fb.height; // lfb_size in bytes (EFI)
        *(uint16_t *)(bp + 0x24) = (uint16_t)assets->fb.pitch;
        bp[0x26] = assets->fb.red_size;
        bp[0x27] = assets->fb.red_pos;
        bp[0x28] = assets->fb.green_size;
        bp[0x29] = assets->fb.green_pos;
        bp[0x2A] = assets->fb.blue_size;
        bp[0x2B] = assets->fb.blue_pos;
        bp[0x2C] = 8;
        bp[0x2D] = 24;
        uint32_t caps = 0x01; // VIDEO_CAPABILITY_SKIP_QUIRKS
        if (assets->fb.addr >> 32) {
            caps |= 0x02;     // VIDEO_CAPABILITY_64BIT_BASE
            *(uint32_t *)(bp + 0x3A) = (uint32_t)(assets->fb.addr >> 32);
        }
        *(uint32_t *)(bp + 0x36) = caps;
    }

    // 5. Copy ACPI RSDP to safe guest-accessible page (RSDP_COPY_PA = 0x00602000)
    if (assets->acpi_rsdp_src && assets->acpi_rsdp_len > 0 && assets->acpi_rsdp_len <= 4096) {
        hv_memset((void *)(uintptr_t)RSDP_COPY_PA, 0, 4096);
        hv_memcpy((void *)(uintptr_t)RSDP_COPY_PA,
                  (const void *)(uintptr_t)assets->acpi_rsdp_src,
                  assets->acpi_rsdp_len);
        *(uint64_t *)(bp + 0x070) = RSDP_COPY_PA; // boot_params.acpi_rsdp_addr
    }

    // 6. Populate E820 table (offset 0x2D0), carving out Ring -1 arena (0x00200000..0x00600000)
    //    as E820_RESERVED (type 2) and RSDP copy (0x00602000..0x00603000) as E820_ACPI (type 3)
    E820Entry tmp1[128], tmp2[128];
    uint8_t tmp1_cnt = 0, tmp2_cnt = 0;
    for (uint32_t i = 0; i < assets->mmap_count; i++) {
        // Hide >= 4 GiB (2 GiB NVDIMM holding MiniCPM5-2B weights!) completely from Linux E820
        if (assets->mmap[i].addr >= 0x100000000ULL) continue;
        e820_add_carved(tmp1, &tmp1_cnt,
                        assets->mmap[i].addr, assets->mmap[i].len, assets->mmap[i].type,
                        HV_ARENA_START, HV_ARENA_END, 2);
    }
    // Also carve 64 MiB Ring -1 LLM runtime arena (0x70000000..0x73FFFFFF) as E820_RESERVED
    for (uint8_t i = 0; i < tmp1_cnt; i++) {
        e820_add_carved(tmp2, &tmp2_cnt,
                        tmp1[i].addr, tmp1[i].size, tmp1[i].type,
                        0x70000000ULL, 0x74000000ULL, 2);
    }
    E820Entry *e820 = (E820Entry *)(bp + 0x2D0);
    uint8_t final_cnt = 0;
    for (uint8_t i = 0; i < tmp2_cnt; i++) {
        e820_add_carved(e820, &final_cnt,
                        tmp2[i].addr, tmp2[i].size, tmp2[i].type,
                        RSDP_COPY_PA, RSDP_COPY_PA + 0x1000ULL, 3);
    }
    bp[0x1E8] = final_cnt;
    kprintf("[LINUX] E820 entries=%u (Ring -1 0x%lx..0x%lx marked RESERVED)\n",
            final_cnt, HV_ARENA_START, HV_ARENA_END);

    // 7. Build clean guest GDT at GUEST_GDT_PA (0x00603000)
    uint64_t *ggdt = (uint64_t *)(uintptr_t)GUEST_GDT_PA;
    hv_memset(ggdt, 0, 4096);
    ggdt[0] = 0x0000000000000000ULL; // 0x00 null
    ggdt[1] = 0x00AF9A000000FFFFULL; // 0x08 64-bit code
    ggdt[2] = 0x00CF92000000FFFFULL; // 0x10 64-bit data
    ggdt[3] = 0x00008B0000000067ULL; // 0x18 64-bit TSS (busy)
    ggdt[4] = 0x0000000000000000ULL;

    // 8. Build initial 64-bit identity guest page tables (0..4 GiB via 2 MiB pages) at GUEST_PT_BASE
    uint64_t *gpml4 = (uint64_t *)(uintptr_t)(GUEST_PT_BASE + 0x0000);
    uint64_t *gpdpt = (uint64_t *)(uintptr_t)(GUEST_PT_BASE + 0x1000);
    uint64_t *gpd   = (uint64_t *)(uintptr_t)(GUEST_PT_BASE + 0x2000);
    hv_memset(gpml4, 0, 6 * 4096);

    gpml4[0] = (uint64_t)(uintptr_t)gpdpt | 0x03ULL;
    for (uint64_t g = 0; g < 4; g++) {
        uint64_t *pd = gpd + g * 512;
        gpdpt[g] = (uint64_t)(uintptr_t)pd | 0x03ULL;
        for (uint64_t i = 0; i < 512; i++) {
            pd[i] = ((g << 30) | (i << 21)) | 0x83ULL;
        }
    }

    init_out->entry_rip        = pref_addr + 0x200ULL; // Linux 64-bit startup_64!
    init_out->boot_params_pa   = BOOT_PARAMS_PA;
    init_out->guest_cr3        = (uint64_t)(uintptr_t)gpml4;
    init_out->guest_gdtr_base  = GUEST_GDT_PA;
    init_out->guest_gdtr_limit = 5 * 8 - 1;
    init_out->guest_rsp        = GUEST_STACK_PA;
    return true;
}
