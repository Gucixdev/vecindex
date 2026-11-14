/* scratch.h - a query's working memory, from the OS's pages (mem.h) and never malloc: a pool of buffers, one taken for the length of
 * a call and handed back. A buffer is allocated once, at the largest size any call asked of it, and reused for
 * every later query; inside a call the memory is handed out by bumping an offset (an arena), all freed at once by
 * scr_put_. SCR_SLOTS calls may hold one at the same time; past that a call gets a buffer of its own, freed on put.
 * Out of memory is an answer (-1), never a NULL the caller forgot to check: scr_take_ on a buffer too small returns
 * NULL only when the caller asked scr_get_ for less than it then takes, which is a bug, not a condition. */
#ifndef SCRATCH_H
#define SCRATCH_H

#include <pthread.h>
#include <stdlib.h>

#ifndef SCR_SLOTS
#define SCR_SLOTS 128
#endif

typedef struct { u8 *mem; size_t cap, used; i32 slot; } Scr;

static struct { u8 *mem; size_t cap; i32 busy; } scr_pool_[SCR_SLOTS];
static pthread_mutex_t scr_mu_ = PTHREAD_MUTEX_INITIALIZER;
static i64 scr_oom_;                /* calls that found no memory (scr_oom_count) */

/* n rounded up to a multiple of 64: every piece starts on a cache line (and is aligned for any SIMD load) */
static size_t scr_up_(size_t n) { return (n + 63) & ~(size_t)63; }

/* the bytes scr_take_(n) uses of a buffer: what a caller adds up for scr_get_ */
static size_t scr_need_(size_t n) { return scr_up_(n ? n : 1); }

/* a buffer of at least size bytes into a. 0, or -1 when there is no memory (a holds nothing then). */
static i32 scr_get_(Scr *a, size_t size)
{
    i32 i, got = -1;
    size += 64;                                                  /* room to align the start */
    pthread_mutex_lock(&scr_mu_);
    for (i = 0; i < SCR_SLOTS; ++i) if (!scr_pool_[i].busy) { scr_pool_[i].busy = 1; got = i; break; }
    pthread_mutex_unlock(&scr_mu_);
    a->slot = got; a->used = 0;
    if (got < 0) {                                               /* every slot taken: a buffer for this call alone */
        a->mem = (u8 *)mem_pages_(size); a->cap = size;
    } else {
        if (scr_pool_[got].cap < size) {                         /* grown once, kept */
            u8 *m = (u8 *)mem_pages_(size + size / 4);
            if (m) { mem_unpages_(scr_pool_[got].mem, scr_pool_[got].cap); scr_pool_[got].mem = m; scr_pool_[got].cap = size + size / 4; }
            else { pthread_mutex_lock(&scr_mu_); scr_pool_[got].busy = 0; pthread_mutex_unlock(&scr_mu_); a->mem = (u8 *)0; a->slot = -1; a->cap = 0; }
        }
        if (a->slot >= 0) { a->mem = scr_pool_[got].mem; a->cap = scr_pool_[got].cap; }
    }
    if (!a->mem) { pthread_mutex_lock(&scr_mu_); ++scr_oom_; pthread_mutex_unlock(&scr_mu_); a->cap = 0; return -1; }
    a->used = (size_t)(64 - ((size_t)a->mem & 63)) & 63;
    return 0;
}

/* n bytes of a, on a cache line */
static void *scr_take_(Scr *a, size_t n)
{
    void *p;
    n = scr_need_(n);
    if (a->used + n > a->cap) return (void *)0;
    p = a->mem + a->used;
    a->used += n;
    return p;
}

/* a back to the pool (a no-op for one that holds nothing) */
static void scr_put_(Scr *a)
{
    if (!a->mem) return;
    if (a->slot < 0) mem_unpages_(a->mem, a->cap);
    else { pthread_mutex_lock(&scr_mu_); scr_pool_[a->slot].busy = 0; pthread_mutex_unlock(&scr_mu_); }
    a->mem = (u8 *)0; a->cap = 0;
}

/* calls that failed for want of memory since the process started */
static i64 scr_oom_count_(void)
{
    i64 n;
    pthread_mutex_lock(&scr_mu_); n = scr_oom_; pthread_mutex_unlock(&scr_mu_);
    return n;
}

#endif
