#ifndef HOLYC_PRINT_H_
#define HOLYC_PRINT_H_

/* print - split out of base.h (was base.h's own "print" section, moved
 * out per explicit user direction so base.h stays scoped to base types/
 * bit/SIMD, not print too - the same move mathx.h's vec/quat/mat split
 * already made). Same PRINT_IMPLEMENTATION guard name as before the
 * move - every existing `#define PRINT_IMPLEMENTATION` call site keeps
 * working unchanged, consumers just need `#include "print.h"` instead
 * of relying on it being bundled inside base.h. */

#include "base.h"

/* ============================================================
 * print (print.h) - "wlasny print bez printf, tylko write itoa": no
 * stdio, no format-string parsing, a small line buffer flushed
 * through write(2).
 * ============================================================ */

#ifndef PRINT_BUF_SIZE
#define PRINT_BUF_SIZE 4096
#endif

void print_flush(void);
void print_buf(const char *buf, size_t n);
void print_char(char c);
void print_cstr(const char *s);
void print_u64_base(u64 v, u32 base);
void print_i64_base(i64 v, u32 base);
void print_u64(u64 v);
void print_i64(i64 v);
#endif /* HOLYC_PRINT_H_ */

#ifdef PRINT_IMPLEMENTATION
#ifndef PRINT_IMPLEMENTATION_
#define PRINT_IMPLEMENTATION_

/* raw Linux write(2) syscall, replacing unistd.h's wrapper - own local
 * copy (not shared with log's own copy below) matching this project's
 * standing "no cross-IMPLEMENTATION-guard dependency" rule. */
#if defined(__x86_64__)
static i64 print_sys_write_(i32 fd, const void *buf, size_t n)
{
    i64 ret;
    __asm__ volatile (
        "syscall"
        : "=a"(ret)
        : "a"(1), "D"(fd), "S"(buf), "d"(n)
        : "rcx", "r11", "memory"
    );
    return ret;
}
#elif defined(__i386__)
static i64 print_sys_write_(i32 fd, const void *buf, size_t n)
{
    i32 ret;
    __asm__ volatile (
        "int $0x80"
        : "=a"(ret)
        : "a"(4), "b"(fd), "c"(buf), "d"(n)
        : "memory"
    );
    return (i64)ret;
}
#else
#error "print.h's raw write() syscall is only implemented for x86_64/i386 Linux"
#endif

static char print_buffer_[PRINT_BUF_SIZE];
static size_t print_len_ = 0;

void print_flush(void)
{
    if (print_len_ > 0) {
        print_sys_write_(1, print_buffer_, print_len_);
        print_len_ = 0;
    }
}

void print_buf(const char *buf, size_t n)
{
    size_t i;
    for (i = 0; i < n; i++) {
        if (print_len_ >= PRINT_BUF_SIZE) print_flush();
        print_buffer_[print_len_++] = buf[i];
    }
}

void print_char(char c) { print_buf(&c, 1); }
void print_cstr(const char *s) { while (*s) print_char(*s++); }

static const char print_digit_pairs_[] =
    "00010203040506070809"
    "10111213141516171819"
    "20212223242526272829"
    "30313233343536373839"
    "40414243444546474849"
    "50515253545556575859"
    "60616263646566676869"
    "70717273747576777879"
    "80818283848586878889"
    "90919293949596979899";

void print_u64_base(u64 v, u32 base)
{
    static const char digits[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    char tmp[64];
    size_t n = 0;

    if (v == 0) {
        print_char('0');
        return;
    }

    if (base == 10) {
        while (v >= 100) {
            u32 pair = (u32)(v % 100) * 2;
            v /= 100;
            tmp[n++] = print_digit_pairs_[pair + 1];
            tmp[n++] = print_digit_pairs_[pair];
        }
        if (v >= 10) {
            u32 pair = (u32)v * 2;
            tmp[n++] = print_digit_pairs_[pair + 1];
            tmp[n++] = print_digit_pairs_[pair];
        } else {
            tmp[n++] = digits[v];
        }
    } else if (base != 0 && (base & (base - 1)) == 0) {
        u32 shift = (u32)__builtin_ctz(base);
        u32 mask = base - 1;
        while (v > 0) {
            tmp[n++] = digits[v & mask];
            v >>= shift;
        }
    } else {
        while (v > 0) {
            tmp[n++] = digits[v % base];
            v /= base;
        }
    }
    while (n > 0) print_char(tmp[--n]);
}

void print_i64_base(i64 v, u32 base)
{
    u64 uv;
    if (v < 0) {
        print_char('-');
        uv = (u64)(-(v + 1)) + 1; /* safe negation, avoids overflow at i64 min */
    } else {
        uv = (u64)v;
    }
    print_u64_base(uv, base);
}

void print_u64(u64 v) { print_u64_base(v, 10); }
void print_i64(i64 v) { print_i64_base(v, 10); }

#endif /* PRINT_IMPLEMENTATION_ */
#endif /* PRINT_IMPLEMENTATION */
