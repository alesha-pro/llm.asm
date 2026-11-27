#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

static inline void outb(uint16_t port, uint8_t val) {
    __asm__ volatile ("outb %0, %1" : : "a"(val), "Nd"(port));
}
static inline uint8_t inb(uint16_t port) {
    uint8_t val;
    __asm__ volatile ("inb %1, %0" : "=a"(val) : "Nd"(port));
    return val;
}
static inline void cpu_pause(void) {
    __asm__ volatile ("pause");
}

void *hv_memset(void *dst, int c, size_t n);
void *hv_memcpy(void *dst, const void *src, size_t n);
void *hv_memmove(void *dst, const void *src, size_t n);
size_t hv_strlen(const char *s);
int hv_strcmp(const char *a, const char *b);
int hv_strncmp(const char *a, const char *b, size_t n);
int hv_snprintf(char *buf, size_t cap, const char *fmt, ...);

void serial_init(void);
void serial_putc(char c);
void serial_puts(const char *s);
void kprintf(const char *fmt, ...);

extern uint64_t read_msr(uint32_t msr);
extern void write_msr(uint32_t msr, uint64_t val);
extern uint64_t read_tsc(void);
extern void cpuid_raw(uint32_t leaf, uint32_t subleaf,
                      uint32_t *a, uint32_t *b, uint32_t *c, uint32_t *d);
extern void flush_host_tlb(void);
