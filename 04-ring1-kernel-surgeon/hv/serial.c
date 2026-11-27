#include "types.h"
#include <stdarg.h>

#define COM1 0x3F8

void *hv_memset(void *dst, int c, size_t n) {
    uint8_t *p = (uint8_t *)dst;
    while (n--) *p++ = (uint8_t)c;
    return dst;
}

void *hv_memcpy(void *dst, const void *src, size_t n) {
    uint8_t *d = (uint8_t *)dst;
    const uint8_t *s = (const uint8_t *)src;
    while (n--) *d++ = *s++;
    return dst;
}

void *hv_memmove(void *dst, const void *src, size_t n) {
    uint8_t *d = (uint8_t *)dst;
    const uint8_t *s = (const uint8_t *)src;
    if (d < s) {
        while (n--) *d++ = *s++;
    } else if (d > s) {
        d += n;
        s += n;
        while (n--) *--d = *--s;
    }
    return dst;
}

size_t hv_strlen(const char *s) {
    size_t n = 0;
    while (s[n]) n++;
    return n;
}

int hv_strcmp(const char *a, const char *b) {
    while (*a && *a == *b) { a++; b++; }
    return (int)(uint8_t)*a - (int)(uint8_t)*b;
}

int hv_strncmp(const char *a, const char *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i] || a[i] == 0)
            return (int)(uint8_t)a[i] - (int)(uint8_t)b[i];
    }
    return 0;
}

void serial_init(void) {
    outb(COM1 + 1, 0x00); // Disable interrupts
    outb(COM1 + 3, 0x80); // Enable DLAB
    outb(COM1 + 0, 0x01); // 115200 baud lo
    outb(COM1 + 1, 0x00); // 115200 baud hi
    outb(COM1 + 3, 0x03); // 8N1
    outb(COM1 + 2, 0xC7); // FIFO
    outb(COM1 + 4, 0x0B); // IRQs enabled, RTS/DSR set
}

void serial_putc(char c) {
    if (c == '\n') {
        while ((inb(COM1 + 5) & 0x20) == 0) {}
        outb(COM1, '\r');
    }
    while ((inb(COM1 + 5) & 0x20) == 0) {}
    outb(COM1, (uint8_t)c);
}

void serial_puts(const char *s) {
    while (*s) serial_putc(*s++);
}

typedef struct {
    char *buf;
    size_t cap;
    size_t pos;
    bool to_serial;
} FmtOut;

static void fmt_emit(FmtOut *o, char c) {
    if (o->to_serial) {
        serial_putc(c);
    } else if (o->buf && o->pos + 1 < o->cap) {
        o->buf[o->pos] = c;
    }
    o->pos++;
}

static void fmt_uint(FmtOut *o, uint64_t val, int base, int width, char pad, bool upper) {
    char tmp[32];
    const char *digs = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    int n = 0;
    if (val == 0) {
        tmp[n++] = '0';
    } else {
        while (val) {
            tmp[n++] = digs[val % (uint64_t)base];
            val /= (uint64_t)base;
        }
    }
    while (n < width) {
        fmt_emit(o, pad);
        width--;
    }
    while (n > 0) fmt_emit(o, tmp[--n]);
}

static int hv_vformat(FmtOut *o, const char *fmt, va_list ap) {
    for (; *fmt; fmt++) {
        if (*fmt != '%') {
            fmt_emit(o, *fmt);
            continue;
        }
        fmt++;
        char pad = ' ';
        if (*fmt == '0') { pad = '0'; fmt++; }
        int width = 0;
        while (*fmt >= '0' && *fmt <= '9') {
            width = width * 10 + (*fmt - '0');
            fmt++;
        }
        bool is_long = false;
        while (*fmt == 'l') { is_long = true; fmt++; }
        switch (*fmt) {
            case 's': {
                const char *s = va_arg(ap, const char *);
                if (!s) s = "(null)";
                while (*s) fmt_emit(o, *s++);
                break;
            }
            case 'c':
                fmt_emit(o, (char)va_arg(ap, int));
                break;
            case 'd': {
                int64_t v = is_long ? va_arg(ap, int64_t) : (int64_t)va_arg(ap, int32_t);
                if (v < 0) { fmt_emit(o, '-'); v = -v; }
                fmt_uint(o, (uint64_t)v, 10, width, pad, false);
                break;
            }
            case 'u': {
                uint64_t v = is_long ? va_arg(ap, uint64_t) : (uint64_t)va_arg(ap, uint32_t);
                fmt_uint(o, v, 10, width, pad, false);
                break;
            }
            case 'x':
            case 'X': {
                uint64_t v = is_long ? va_arg(ap, uint64_t) : (uint64_t)va_arg(ap, uint32_t);
                fmt_uint(o, v, 16, width, pad, *fmt == 'X');
                break;
            }
            case 'p': {
                uint64_t v = (uint64_t)(uintptr_t)va_arg(ap, void *);
                fmt_emit(o, '0'); fmt_emit(o, 'x');
                fmt_uint(o, v, 16, 16, '0', false);
                break;
            }
            case '%':
                fmt_emit(o, '%');
                break;
            default:
                fmt_emit(o, '%');
                fmt_emit(o, *fmt);
                break;
        }
    }
    if (!o->to_serial && o->buf && o->cap > 0) {
        size_t end = o->pos < o->cap ? o->pos : (o->cap - 1);
        o->buf[end] = '\0';
    }
    return (int)o->pos;
}

void kprintf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    FmtOut o = { NULL, 0, 0, true };
    hv_vformat(&o, fmt, ap);
    va_end(ap);
}

int hv_snprintf(char *buf, size_t cap, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    FmtOut o = { buf, cap, 0, false };
    int r = hv_vformat(&o, fmt, ap);
    va_end(ap);
    return r;
}
