/* mem.h - where memory comes from: pages from the OS, handed out through arenas. No malloc and no free of single
 * objects: a piece of memory belongs to an arena, and an arena is freed whole when what it served is over - a
 * build's working memory when the build returns, an index's arrays when the index is replaced, a call's scratch
 * when the call ends.
 *
 *   Gen g; gen_init_(&g);
 *   p = gen_alloc_(&g, n);            n bytes, zeroed, on a cache line; NULL when there is no memory
 *   gen_free_(&g);                    everything at once
 *
 * A Gen may be used from several threads (a lock around the bump). Its blocks come from mem_pages_ and go back to
 * a small cache of free blocks, so a function that opens an arena for its temporaries pays no system call after
 * the first few.
 *
 * Under AddressSanitizer (or -DMEM_DEBUG) every gen_alloc_ is its own heap block, so the sanitizer still sees each
 * piece's bounds and a use after gen_free_; that build is the only one that calls the C allocator, with WASM, which
 * has no mmap. */
#ifndef VECINDEX_MEM_H
#define VECINDEX_MEM_H

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__) || defined(__SANITIZE_MEMORY__)
#ifndef MEM_DEBUG
#define MEM_DEBUG 1
#endif
#endif
#if defined(__has_feature) && !defined(MEM_DEBUG)
#if __has_feature(address_sanitizer) || __has_feature(memory_sanitizer) || __has_feature(thread_sanitizer)
#define MEM_DEBUG 1
#endif
#endif

#if defined(__wasi__) || defined(__EMSCRIPTEN__) || defined(MEM_DEBUG)
#define MEM_LIBC 1                  /* pages from the C allocator: no mmap here, or the sanitizer wants to see them */
#else
#include <sys/mman.h>
#endif

/* n bytes from the OS, zeroed; NULL: none to be had */
static void *mem_pages_(size_t n)
{
#ifdef MEM_LIBC
    return calloc(1, n ? n : 1);
#else
    void *p = mmap((void *)0, n ? n : 1, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? (void *)0 : p;
#endif
}

static void mem_unpages_(void *p, size_t n)
{
    if (!p) return;
#ifdef MEM_LIBC
    (void)n; free(p);
#else
    munmap(p, n ? n : 1);
#endif
}

#define MEM_BLOCK ((size_t)1 << 20)                 /* an arena's ordinary block; a larger piece gets a block of its own */
#define MEM_CACHE 16                                /* free ordinary blocks kept for the next arena */

typedef struct GenBlk { struct GenBlk *next; size_t cap, used; } GenBlk;      /* the header sits at the block's start */
typedef struct { GenBlk *head; pthread_mutex_t mu; i32 live; i64 bytes; } Gen;

static struct { GenBlk *b[MEM_CACHE]; i32 n; pthread_mutex_t mu; } mem_cache_ = { { (GenBlk *)0 }, 0, PTHREAD_MUTEX_INITIALIZER };
static i64 mem_oom_;                                /* allocations that found no memory (mem_oom_count_) */

static void gen_init_(Gen *g) { g->head = (GenBlk *)0; g->live = 1; g->bytes = 0; pthread_mutex_init(&g->mu, (const pthread_mutexattr_t *)0); }

static size_t mem_up_(size_t n) { return (n + 63) & ~(size_t)63; }

/* n bytes of g, zeroed, on a 64-byte boundary; NULL when there is no memory (counted) */
static void *gen_alloc_(Gen *g, size_t n)
{
    GenBlk *b;
    void *p = (void *)0;
    size_t hdr = mem_up_(sizeof(GenBlk));
    n = mem_up_(n ? n : 1);
    pthread_mutex_lock(&g->mu);
#ifdef MEM_DEBUG
    b = (GenBlk *)calloc(1, hdr + n);                            /* one heap block a piece: the sanitizer's view */
    if (b) { b->cap = hdr + n; b->used = b->cap; b->next = g->head; g->head = b; p = (u8 *)b + hdr; }
#else
    b = g->head;
    if (!b || b->used + n > b->cap) {
        size_t cap = hdr + n > MEM_BLOCK ? hdr + n : MEM_BLOCK;
        b = (GenBlk *)0;
        if (cap == MEM_BLOCK) {                                  /* a block someone handed back: zero what it used */
            pthread_mutex_lock(&mem_cache_.mu);
            if (mem_cache_.n > 0) b = mem_cache_.b[--mem_cache_.n];
            pthread_mutex_unlock(&mem_cache_.mu);
            if (b) memset((u8 *)b + hdr, 0, b->used - hdr);
        }
        if (!b) b = (GenBlk *)mem_pages_(cap);
        if (b) { b->cap = cap; b->used = hdr; b->next = g->head; g->head = b; }
    }
    if (b) { p = (u8 *)b + b->used; b->used += n; }
#endif
    if (p) g->bytes += (i64)n;
    pthread_mutex_unlock(&g->mu);
    if (!p) { pthread_mutex_lock(&mem_cache_.mu); ++mem_oom_; pthread_mutex_unlock(&mem_cache_.mu); }
    return p;
}

/* everything g handed out, at once; g can be used again after gen_init_ */
static void gen_free_(Gen *g)
{
    GenBlk *b, *nx;
    if (!g->live) return;
    for (b = g->head; b; b = nx) {
        nx = b->next;
#ifdef MEM_DEBUG
        free(b);
#else
        if (b->cap == MEM_BLOCK) {
            i32 kept = 0;
            pthread_mutex_lock(&mem_cache_.mu);
            if (mem_cache_.n < MEM_CACHE) { mem_cache_.b[mem_cache_.n++] = b; kept = 1; }
            pthread_mutex_unlock(&mem_cache_.mu);
            if (kept) continue;
        }
        mem_unpages_(b, b->cap);
#endif
    }
    g->head = (GenBlk *)0; g->live = 0; g->bytes = 0;
    pthread_mutex_destroy(&g->mu);
}

/* A region of its own for one array that grows (or is dropped) while its arena lives on: growing copies it into a
 * larger region and gives the old one straight back, so nothing is left behind as it would be in an arena. */
typedef struct { void *p; size_t cap; } Reg;

/* *pp ends up in r with room for want bytes, its first `have` bytes kept (*pp may come from elsewhere - an arena, a
 * file mapping - the first time). 0, or -1 when there is no memory (*pp and r are as they were). */
static i32 reg_fit_(Reg *r, void **pp, size_t have, size_t want)
{
    void *q;
    if (r->p && r->p == *pp && r->cap >= want) return 0;
    if (!want) want = 1;
    if (!(q = mem_pages_(want))) { pthread_mutex_lock(&mem_cache_.mu); ++mem_oom_; pthread_mutex_unlock(&mem_cache_.mu); return -1; }
    if (*pp && have) memcpy(q, *pp, have < want ? have : want);
    if (r->p) mem_unpages_(r->p, r->cap);
    r->p = q; r->cap = want; *pp = q;
    return 0;
}

static void reg_free_(Reg *r) { if (r->p) mem_unpages_(r->p, r->cap); r->p = (void *)0; r->cap = 0; }

/* allocations of this process that found no memory */
static i64 mem_oom_count_(void)
{
    i64 n;
    pthread_mutex_lock(&mem_cache_.mu); n = mem_oom_; pthread_mutex_unlock(&mem_cache_.mu);
    return n;
}

#endif
