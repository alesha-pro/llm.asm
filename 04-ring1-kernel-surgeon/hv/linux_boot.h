#pragma once
#include "types.h"
#include "fb.h"

#define MB2_BOOTLOADER_MAGIC 0x36D76289U

typedef struct {
    FbInfo   fb;
    uint64_t vmlinuz_addr;
    uint32_t vmlinuz_size;
    uint64_t initrd_addr;
    uint32_t initrd_size;
    uint64_t acpi_rsdp_src;
    uint32_t acpi_rsdp_len;
    // Parsed Multiboot2 E820 entries
    struct {
        uint64_t addr;
        uint64_t len;
        uint32_t type;
    } mmap[64];
    uint32_t mmap_count;
} BootAssets;

typedef struct {
    uint64_t entry_rip;
    uint64_t boot_params_pa;
    uint64_t guest_cr3;
    uint64_t guest_gdtr_base;
    uint32_t guest_gdtr_limit;
    uint64_t guest_rsp;
} LinuxGuestInit;

bool mb2_parse(uint64_t magic, uint64_t mbi_addr, BootAssets *out);
bool linux_prepare_guest(const BootAssets *assets, LinuxGuestInit *init_out);
