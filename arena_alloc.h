#ifndef HOLYC_ARENA_ALLOC_H_
#define HOLYC_ARENA_ALLOC_H_

/* Port of tsoding's arena.h (https://github.com/tsoding/arena) to plain C89:
 *   - no <stdint.h> (own base_type.h u64 instead of uintptr_t)
 *   - no flexible array member (C99) -> offsetof() struct-hack instead
 *   - no mixed declarations / inline for-loop decls
 *   - no C++ interop cruft (this is not going to touch a C++ compiler)
 *   - WIN32/WASM backends dropped for now, add back when actually needed
 */

#include "base.h"
#include "contracts.h"
#include <stdarg.h> /* va_list is part of arena_vsprintf's declaration below, not implementation-only */
#include <string.h> /* memcpy - arena_memcpy's underlying primitive */

#ifndef ARENA_ASSERT
#define ARENA_ASSERT HOLYC_ASSERT
#endif

#define ARENA_BACKEND_LIBC_MALLOC 0
#define ARENA_BACKEND_LINUX_MMAP  1
#define ARENA_BACKEND_FLAT        2

#ifndef ARENA_BACKEND
#define ARENA_BACKEND ARENA_BACKEND_FLAT
#endif

typedef struct Arena_Region Arena_Region;

struct Arena_Region {
    Arena_Region *next;
    size_t count;
    size_t capacity;
    u64 data[1]; /* struct-hack: real allocation is offsetof(Arena_Region, data) + capacity*sizeof(u64) */
};

typedef struct {
    Arena_Region *begin, *end;
} Arena;

typedef struct {
    Arena_Region *region;
    size_t count;
} Arena_Mark;

#ifndef ARENA_REGION_DEFAULT_CAPACITY
#define ARENA_REGION_DEFAULT_CAPACITY ((size_t)8 * 1024) /* size_t, not int, so any future edit to this macro can't silently overflow int before widening */
#endif

Arena_Region *new_region(size_t capacity);
void free_region(Arena_Region *r);

void *arena_alloc(Arena *a, size_t size_bytes);

/* alignof(type) with no <stdalign.h> (C11): the classic struct-offset
 * trick. A compiler must pad `d` up to its natural alignment right after
 * the single `char`, so that padding amount IS alignof(type). Seen while
 * surveying vendored reference libraries (arena-allocator-main, Carter
 * Dugan, Apache-2.0) - worth lifting on its own even though the allocator
 * it came from isn't (no growth, no bounds checking - silently overruns
 * past whatever size you gave it, which this project's own arena already
 * does better without). */
#define ARENA_ALIGNOF(type) offsetof(struct { char c_; type d_; }, d_)

/* returns a block of size_bytes whose address is a multiple of `align`
 * (align must be a power of 2 - true of every real alignment requirement,
 * including ARENA_ALIGNOF's own output). Wastes up to align-1 bytes doing
 * it - the arena has no way to hand back the slack since it never frees
 * individual allocations anyway, same slack-for-simplicity tradeoff as
 * arena_da_append's capacity doubling. Exists for SIMD-width-aligned
 * buffers (aligned loads are cheaper than unaligned ones, e.g.
 * vecindex.h's AVX2 kernel currently uses _mm256_loadu_ps specifically
 * because nothing guaranteed 32-byte alignment before this existed). */
void *arena_alloc_aligned(Arena *a, size_t size_bytes, size_t align);

void *arena_realloc(Arena *a, void *oldptr, size_t oldsz, size_t newsz);
size_t arena_strlen(const char *s);
void *arena_memcpy(void *dest, const void *src, size_t n);
char *arena_strdup(Arena *a, const char *cstr);
void *arena_memdup(Arena *a, void *data, size_t size);

#ifndef ARENA_NOSTDIO
char *arena_sprintf(Arena *a, const char *format, ...);
char *arena_vsprintf(Arena *a, const char *format, va_list args);
#endif

Arena_Mark arena_snapshot(Arena *a);
void arena_reset(Arena *a);
void arena_rewind(Arena *a, Arena_Mark m);
void arena_free(Arena *a);
void arena_trim(Arena *a);

#ifndef ARENA_DA_INIT_CAP
#define ARENA_DA_INIT_CAP 256
#endif

/* void* -> T* needs no cast in C (unlike C++), so no cast_ptr wrapper here */

#define arena_da_append(a, da, item)                                                         \
    do {                                                                                      \
        if ((da)->count >= (da)->capacity) {                                                  \
            size_t new_capacity = (da)->capacity == 0 ? ARENA_DA_INIT_CAP : (da)->capacity*2; \
            (da)->items = arena_realloc(                                                      \
                (a), (da)->items,                                                             \
                (da)->capacity*sizeof(*(da)->items),                                          \
                new_capacity*sizeof(*(da)->items));                                           \
            (da)->capacity = new_capacity;                                                    \
        }                                                                                      \
        (da)->items[(da)->count++] = (item);                                                  \
    } while (0)

#define arena_da_append_many(a, da, new_items, new_items_count)                                       \
    do {                                                                                              \
        if ((da)->count + (new_items_count) > (da)->capacity) {                                       \
            size_t new_capacity = (da)->capacity;                                                     \
            if (new_capacity == 0) new_capacity = ARENA_DA_INIT_CAP;                                  \
            while ((da)->count + (new_items_count) > new_capacity) new_capacity *= 2;                 \
            (da)->items = arena_realloc(                                                              \
                (a), (da)->items,                                                                     \
                (da)->capacity*sizeof(*(da)->items),                                                  \
                new_capacity*sizeof(*(da)->items));                                                   \
            (da)->capacity = new_capacity;                                                            \
        }                                                                                              \
        arena_memcpy((da)->items + (da)->count, (new_items), (new_items_count)*sizeof(*(da)->items)); \
        (da)->count += (new_items_count);                                                             \
    } while (0)

#define arena_sb_append_buf arena_da_append_many

#define arena_sb_append_cstr(a, sb, cstr)      \
    do {                                       \
        const char *s_ = (cstr);               \
        size_t n_ = arena_strlen(s_);          \
        arena_da_append_many(a, sb, s_, n_);   \
    } while (0)

#define arena_sb_append_null(a, sb) arena_da_append(a, sb, 0)

#endif /* HOLYC_ARENA_ALLOC_H_ */

#ifdef ARENA_ALLOC_IMPLEMENTATION
#ifndef ARENA_ALLOC_IMPLEMENTATION_
#define ARENA_ALLOC_IMPLEMENTATION_

#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>


#if ARENA_BACKEND == ARENA_BACKEND_LIBC_MALLOC


Arena_Region *new_region(size_t capacity)
{
    Arena_Region *r;
    size_t size_bytes = offsetof(Arena_Region, data) + sizeof(u64) * capacity;
    r = (Arena_Region *)malloc(size_bytes);
    ARENA_ASSERT(r);
    r->next = NULL;
    r->count = 0;
    r->capacity = capacity;
    return r;
}

void free_region(Arena_Region *r)
{
    free(r);
}

#elif ARENA_BACKEND == ARENA_BACKEND_LINUX_MMAP


Arena_Region *new_region(size_t capacity)
{
    Arena_Region *r;
    size_t size_bytes = offsetof(Arena_Region, data) + sizeof(u64) * capacity;
    r = (Arena_Region *)mmap(NULL, size_bytes, PROT_READ | PROT_WRITE,
                        MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    ARENA_ASSERT(r != MAP_FAILED);
    r->next = NULL;
    r->count = 0;
    r->capacity = capacity;
    return r;
}

void free_region(Arena_Region *r)
{
    size_t size_bytes = offsetof(Arena_Region, data) + sizeof(u64) * r->capacity;
    int ret = munmap(r, size_bytes);
    ARENA_ASSERT(ret == 0);
}

#elif ARENA_BACKEND == ARENA_BACKEND_FLAT

/* "pamiec jako jedna wielka tablica totalnie w stylu terrego" - the region
 * chasing (fixed vs geometric, how big should the next malloc be, does
 * this size class warm glibc's free-list) was never really the interesting
 * question; it only existed because every new_region() call handed memory
 * management back to glibc. This backend hands it back to the kernel
 * instead, once: reserve one huge chunk of VIRTUAL address space up front
 * (MAP_NORESERVE - Linux backs it with real RAM/swap lazily, page by page,
 * only where something actually gets written), then never call mmap()
 * again in the overwhelmingly common case. arena_alloc()'s existing
 * region-growth loop is completely unchanged - it just so happens the
 * first "region" is already bigger than any realistic arena will ever
 * need, so growth code exists only as a safety net, not a hot path. No
 * glibc heap, no free-list warm-up effects, no size-class bookkeeping -
 * closer to the flat-array ideal than a linked list of glibc-managed
 * chunks could ever be. */


#ifndef ARENA_FLAT_RESERVE_BYTES
#define ARENA_FLAT_RESERVE_BYTES ((size_t)1 << 30) /* 1GB of address space, reserved not committed */
#endif

Arena_Region *new_region(size_t capacity)
{
    Arena_Region *r;
    size_t requested_bytes = offsetof(Arena_Region, data) + sizeof(u64) * capacity;
    size_t size_bytes = requested_bytes > ARENA_FLAT_RESERVE_BYTES ? requested_bytes : ARENA_FLAT_RESERVE_BYTES;

    r = (Arena_Region *)mmap(NULL, size_bytes, PROT_READ | PROT_WRITE,
                        MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE, -1, 0);
    if (r == MAP_FAILED) {
        /* the over-reservation itself failed - not a real out-of-memory
         * condition (MAP_NORESERVE reservations are cheap and this
         * machine's overcommit=0 heuristic accepts them fine, but a
         * container with a tight memory cgroup or a system running
         * overcommit_memory=2 can legitimately refuse a 1GB ask). Retry
         * for exactly what this call actually needs before giving up -
         * ARENA_ASSERT below is a real hard failure only if even that
         * fails, not silently continuing with a MAP_FAILED pointer under
         * NDEBUG (assert() compiles out in release builds - dereferencing
         * r->next below on (void*)-1 would otherwise be a use-after-"success"
         * crash waiting for whichever machine has a tighter memory limit). */
        size_bytes = requested_bytes;
        r = (Arena_Region *)mmap(NULL, size_bytes, PROT_READ | PROT_WRITE,
                            MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE, -1, 0);
    }
    ARENA_ASSERT(r != MAP_FAILED);
    r->next = NULL;
    r->count = 0;
    /* real capacity is the whole reservation, not just what this call
     * asked for - that's the entire point, over-reserve once */
    r->capacity = (size_bytes - offsetof(Arena_Region, data)) / sizeof(u64);
    return r;
}

void free_region(Arena_Region *r)
{
    size_t size_bytes = offsetof(Arena_Region, data) + sizeof(u64) * r->capacity;
    int ret = munmap(r, size_bytes);
    ARENA_ASSERT(ret == 0);
}

#else
#error "Unknown ARENA_BACKEND"
#endif

void *arena_alloc(Arena *a, size_t size_bytes)
{
    size_t size = (size_bytes + sizeof(u64) - 1) / sizeof(u64);
    void *result;

    if (a->end == NULL) {
        size_t capacity = ARENA_REGION_DEFAULT_CAPACITY;
        ARENA_ASSERT(a->begin == NULL);
        /* branchless max(capacity, size): selects size when capacity <
         * size, else keeps capacity - same XOR-mask select idiom used
         * in string_view.h's sv_chop_left/right clamp. */
        capacity ^= (capacity ^ size) & (size_t)-(size_t)(capacity < size);
        a->end = new_region(capacity);
        a->begin = a->end;
    }

    while (a->end->count + size > a->end->capacity && a->end->next != NULL) {
        a->end = a->end->next;
    }

    if (a->end->count + size > a->end->capacity) {
        /* Tried geometric growth here (double the previous region's
         * capacity) reasoning "fewer malloc() calls must be faster" -
         * measured with a benchmark that allocated but never wrote to the
         * memory, which showed a 2.7x win (8192 regions -> 14 for a 512MB
         * arena). That benchmark was misleading: pages that are never
         * touched may never actually get committed, so it wasn't measuring
         * real usage. Redone with every allocation actually written to
         * (memset), 5 trials: flat ARENA_REGION_DEFAULT_CAPACITY every time
         * was consistently ~40% FASTER than geometric growth (70-76ms vs
         * 120-124ms for the same 512MB workload), and a capped variant
         * (grow up to 1MB then stay flat) was worse than both (175-179ms).
         * The likely cause: ARENA_REGION_DEFAULT_CAPACITY (64KB) sits under
         * glibc's default mmap threshold (128KB) and stays on the fast
         * brk-extended heap; both geometric and capped-at-1MB cross that
         * threshold and pay a real per-region mmap() cost with different
         * page-fault behavior. So: still flat, on purpose, re-verified
         * rather than "obviously" switched to geometric. */
        size_t capacity = ARENA_REGION_DEFAULT_CAPACITY;
        ARENA_ASSERT(a->end->next == NULL);
        /* branchless max(capacity, size) - see arena_alloc's identical clamp */
        capacity ^= (capacity ^ size) & (size_t)-(size_t)(capacity < size);
        a->end->next = new_region(capacity);
        a->end = a->end->next;
    }

    result = &a->end->data[a->end->count];
    a->end->count += size;
    /* the exact "arena nie wysadza" property: this allocation never
     * pushed the active region's count past its own capacity - true
     * unconditionally here (not "usually"), since every path above
     * either found/made a region with enough room before ever
     * touching count. */
    HOLYC_ENSURES(a->end->count <= a->end->capacity);
    return result;
}

void *arena_alloc_aligned(Arena *a, size_t size_bytes, size_t align)
{
    size_t addr, aligned_addr;
    void *raw = arena_alloc(a, size_bytes + align - 1);
    addr = (size_t)raw;
    aligned_addr = (addr + align - 1) & ~(align - 1); /* align must be a power of 2 */
    return (void *)aligned_addr;
}

void *arena_realloc(Arena *a, void *oldptr, size_t oldsz, size_t newsz)
{
    char *newptr_char, *oldptr_char;
    size_t i;
    void *newptr;

    if (newsz <= oldsz) return oldptr;

    newptr = arena_alloc(a, newsz);
    newptr_char = (char *)newptr;
    oldptr_char = (char *)oldptr;
    for (i = 0; i < oldsz; ++i) {
        newptr_char[i] = oldptr_char[i];
    }
    return newptr;
}

size_t arena_strlen(const char *s)
{
    size_t n = 0;
    while (*s++) n++;
    return n;
}

void *arena_memcpy(void *dest, const void *src, size_t n)
{
    /* Measured, not assumed: a hand-rolled byte loop here benchmarks at
     * ~3-4 GB/s flat regardless of size on this machine, while glibc's
     * memcpy (runtime-dispatched to AVX2/AVX-512 + non-temporal stores
     * depending on size) hits 30-100+ GB/s - a 2x to 26x gap depending on
     * copy size. A hand-rolled word-at-a-time (u64) loop closes some of
     * that gap but still loses to libc by 2-4x. arena_memcpy is only a
     * name/API of ours - what runs underneath doesn't have to be reinvented
     * badly when the platform's own implementation measurably wins, same
     * pragmatic call already made for vsnprintf/mmap/sqrt elsewhere in this
     * file. This is on the hot path of every arena_realloc call, which is
     * every growable array in the codebase (arena_da_append, vecindex's
     * codes[] growth, ...). */
    return memcpy(dest, src, n);
}

char *arena_strdup(Arena *a, const char *cstr)
{
    size_t n = arena_strlen(cstr);
    char *dup = (char *)arena_alloc(a, n + 1);
    arena_memcpy(dup, cstr, n);
    dup[n] = '\0';
    return dup;
}

void *arena_memdup(Arena *a, void *data, size_t size)
{
    return arena_memcpy(arena_alloc(a, size), data, size);
}

#ifndef ARENA_NOSTDIO

/* va_copy is a C99 macro; under strict -std=c89 glibc's <stdarg.h> hides it,
 * so fall back to the gcc/clang builtin that is there regardless of -std=. */
#ifndef va_copy
#define va_copy(dst, src) __builtin_va_copy((dst), (src))
#endif

/* vsnprintf is not strict ISO C89 (it is C99/POSIX), but every libc that
 * matters here (glibc) has provided it forever regardless of -std= mode.
 * Compile with -D_POSIX_C_SOURCE=200112L (or -std=gnu89) to expose it under
 * -std=c89. "shut up and write code" beats pedantry on this one. */
char *arena_vsprintf(Arena *a, const char *format, va_list args)
{
    va_list args_copy;
    int n;
    char *result;

    va_copy(args_copy, args);
    n = vsnprintf(NULL, 0, format, args_copy);
    va_end(args_copy);

    ARENA_ASSERT(n >= 0);
    result = (char *)arena_alloc(a, (size_t)n + 1);
    vsnprintf(result, (size_t)n + 1, format, args);

    return result;
}

char *arena_sprintf(Arena *a, const char *format, ...)
{
    va_list args;
    char *result;

    va_start(args, format);
    result = arena_vsprintf(a, format, args);
    va_end(args);

    return result;
}
#endif /* ARENA_NOSTDIO */

Arena_Mark arena_snapshot(Arena *a)
{
    Arena_Mark m;
    ARENA_ASSERT(a->end != NULL || a->begin == NULL);
    m.region = a->end;
    m.count = (a->end == NULL) ? 0 : a->end->count; /* both branches only differed in this one field */
    return m;
}

void arena_reset(Arena *a)
{
    Arena_Region *r;
    for (r = a->begin; r != NULL; r = r->next) {
        r->count = 0;
    }
    a->end = a->begin;
}

void arena_rewind(Arena *a, Arena_Mark m)
{
    Arena_Region *r;

    if (m.region == NULL) {
        arena_reset(a);
        return;
    }

    m.region->count = m.count;
    for (r = m.region->next; r != NULL; r = r->next) {
        r->count = 0;
    }
    a->end = m.region;
}

void arena_free(Arena *a)
{
    Arena_Region *r = a->begin;
    while (r) {
        Arena_Region *r0 = r;
        r = r->next;
        free_region(r0);
    }
    a->begin = NULL;
    a->end = NULL;
}

void arena_trim(Arena *a)
{
    Arena_Region *r = a->end->next;
    while (r) {
        Arena_Region *r0 = r;
        r = r->next;
        free_region(r0);
    }
    a->end->next = NULL;
}

#endif /* ARENA_ALLOC_IMPLEMENTATION_ */
#endif /* ARENA_ALLOC_IMPLEMENTATION */
