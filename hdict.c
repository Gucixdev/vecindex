/* hdict.c - hash dictionaries for vecindex.
 *
 * Two open-addressing tables (linear probing, power-of-two capacity, load
 * kept at or below 1/2, tombstones purged on growth):
 *
 *   H64   i64 key -> i32 value.  doc id -> row, in store.c: O(1) delete,
 *         lookup and duplicate detection instead of a scan of every row.
 *   HStr  byte-string key -> i32 value. Keys are copied into one pool, the
 *         64-bit hash is stored beside the slot so a probe compares the hash
 *         first and only then the bytes. term -> term id in lex.c, title ->
 *         node id and tag -> node id in vecdb.c.
 *
 * Values are i32 >= 0; -1 means "absent". No arena: the tables grow with
 * realloc-style rehashing, so they may be used from any layer.
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -Werror -O2 -I. hdict.c \
 *       -DHDICT_DEMO -o hdict && ./hdict
 */
#ifndef HDICT_C
#define HDICT_C

#include "base.h"
#include <stdlib.h>
#include <string.h>

/* 64-bit constants without the C99 `ull` suffix (the build is -std=c89 -pedantic) */
#define HD_C64(hi, lo) (((u64)(hi##u) << 32) | (u64)(lo##u))

/* ------------------------------------------------------------------ */
/* H64: i64 -> i32                                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    i64 *key;
    i32 *val;
    u8  *st;        /* 0 empty, 1 full, 2 deleted */
    i32  cap;       /* power of two */
    i32  n;         /* full slots */
    i32  used;      /* full + deleted */
} H64;

static u64 h64_mix_(u64 x)      /* splitmix64 finaliser */
{
    x += HD_C64(0x9e3779b9, 0x7f4a7c15);
    x = (x ^ (x >> 30)) * HD_C64(0xbf58476d, 0x1ce4e5b9);
    x = (x ^ (x >> 27)) * HD_C64(0x94d049bb, 0x133111eb);
    return x ^ (x >> 31);
}

static void h64_alloc_(H64 *h, i32 cap)
{
    h->key = (i64 *)malloc((size_t)cap * sizeof(i64));
    h->val = (i32 *)malloc((size_t)cap * sizeof(i32));
    h->st  = (u8  *)calloc((size_t)cap, 1);
    h->cap = cap;
    h->n = h->used = 0;
}

void h64_init(H64 *h, i32 hint)
{
    i32 cap = 16;
    while (cap < hint * 2) cap <<= 1;
    h64_alloc_(h, cap);
}

void h64_free(H64 *h)
{
    free(h->key); free(h->val); free(h->st);
    memset(h, 0, sizeof(*h));
}

void h64_clear(H64 *h)
{
    memset(h->st, 0, (size_t)h->cap);
    h->n = h->used = 0;
}

static void h64_put_raw_(H64 *h, i64 k, i32 v)   /* key known to be absent */
{
    u32 i = (u32)h64_mix_((u64)k) & (u32)(h->cap - 1);
    while (h->st[i] == 1) i = (i + 1) & (u32)(h->cap - 1);
    if (h->st[i] == 0) ++h->used;
    h->st[i] = 1; h->key[i] = k; h->val[i] = v; ++h->n;
}

static void h64_grow_(H64 *h)
{
    H64 old = *h;
    i32 i, cap = old.cap;
    if (old.n * 4 > cap) cap <<= 1;     /* really full: double; else just purge tombstones */
    h64_alloc_(h, cap);
    for (i = 0; i < old.cap; ++i)
        if (old.st[i] == 1) h64_put_raw_(h, old.key[i], old.val[i]);
    free(old.key); free(old.val); free(old.st);
}

i32 h64_get(const H64 *h, i64 k)
{
    u32 i = (u32)h64_mix_((u64)k) & (u32)(h->cap - 1);
    while (h->st[i] != 0) {
        if (h->st[i] == 1 && h->key[i] == k) return h->val[i];
        i = (i + 1) & (u32)(h->cap - 1);
    }
    return -1;
}

/* store k -> v; returns the previous value or -1 if k was new */
i32 h64_put(H64 *h, i64 k, i32 v)
{
    u32 i;
    i32 tomb = -1;
    if ((h->used + 1) * 2 > h->cap) h64_grow_(h);
    i = (u32)h64_mix_((u64)k) & (u32)(h->cap - 1);
    while (h->st[i] != 0) {
        if (h->st[i] == 1 && h->key[i] == k) {
            i32 old = h->val[i];
            h->val[i] = v;
            return old;
        }
        if (h->st[i] == 2 && tomb < 0) tomb = (i32)i;
        i = (i + 1) & (u32)(h->cap - 1);
    }
    if (tomb >= 0) i = (u32)tomb; else ++h->used;
    h->st[i] = 1; h->key[i] = k; h->val[i] = v; ++h->n;
    return -1;
}

/* remove k; returns its value or -1 if absent */
i32 h64_del(H64 *h, i64 k)
{
    u32 i = (u32)h64_mix_((u64)k) & (u32)(h->cap - 1);
    while (h->st[i] != 0) {
        if (h->st[i] == 1 && h->key[i] == k) {
            h->st[i] = 2; --h->n;
            return h->val[i];
        }
        i = (i + 1) & (u32)(h->cap - 1);
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/* HStr: byte string -> i32                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    u64 *hash;
    i32 *off, *len, *val;   /* key bytes = pool[off .. off+len) */
    u8  *st;                /* 0 empty, 1 full, 2 deleted */
    i32  cap, n, used;
    char *pool;
    size_t plen, pcap;
} HStr;

static u64 hs_hash_(const char *s, i32 len)      /* FNV-1a 64, then a final avalanche */
{
    u64 h = HD_C64(0xcbf29ce4, 0x84222325);
    i32 i;
    for (i = 0; i < len; ++i) { h ^= (u8)s[i]; h *= HD_C64(0x00000100, 0x000001b3); }
    h ^= h >> 32;
    h *= HD_C64(0xd6e8feb8, 0x6659fd93);
    return h ^ (h >> 32);
}

static void hs_alloc_(HStr *h, i32 cap)
{
    h->hash = (u64 *)malloc((size_t)cap * sizeof(u64));
    h->off  = (i32 *)malloc((size_t)cap * sizeof(i32));
    h->len  = (i32 *)malloc((size_t)cap * sizeof(i32));
    h->val  = (i32 *)malloc((size_t)cap * sizeof(i32));
    h->st   = (u8  *)calloc((size_t)cap, 1);
    h->cap = cap;
    h->n = h->used = 0;
}

void hs_init(HStr *h, i32 hint)
{
    i32 cap = 16;
    while (cap < hint * 2) cap <<= 1;
    hs_alloc_(h, cap);
    h->pcap = 4096;
    h->pool = (char *)malloc(h->pcap);
    h->plen = 0;
}

void hs_free(HStr *h)
{
    free(h->hash); free(h->off); free(h->len); free(h->val); free(h->st); free(h->pool);
    memset(h, 0, sizeof(*h));
}

static i32 hs_pool_add_(HStr *h, const char *s, i32 len)
{
    i32 off;
    if (h->plen + (size_t)len + 1 > h->pcap) {
        while (h->plen + (size_t)len + 1 > h->pcap) h->pcap *= 2;
        h->pool = (char *)realloc(h->pool, h->pcap);
    }
    off = (i32)h->plen;
    memcpy(h->pool + h->plen, s, (size_t)len);
    h->pool[h->plen + (size_t)len] = 0;         /* keys are also NUL terminated */
    h->plen += (size_t)len + 1;
    return off;
}

static void hs_grow_(HStr *h)
{
    HStr old = *h;
    i32 i, cap = old.cap;
    if (old.n * 4 > cap) cap <<= 1;
    hs_alloc_(h, cap);
    h->pool = old.pool; h->plen = old.plen; h->pcap = old.pcap;   /* the pool is kept as is */
    for (i = 0; i < old.cap; ++i) {
        if (old.st[i] == 1) {
            u32 j = (u32)old.hash[i] & (u32)(cap - 1);
            while (h->st[j] == 1) j = (j + 1) & (u32)(cap - 1);
            h->st[j] = 1; h->hash[j] = old.hash[i];
            h->off[j] = old.off[i]; h->len[j] = old.len[i]; h->val[j] = old.val[i];
            ++h->n; ++h->used;
        }
    }
    free(old.hash); free(old.off); free(old.len); free(old.val); free(old.st);
}

static i32 hs_find_(const HStr *h, const char *s, i32 len, u64 hv)
{
    u32 i = (u32)hv & (u32)(h->cap - 1);
    while (h->st[i] != 0) {
        if (h->st[i] == 1 && h->hash[i] == hv && h->len[i] == len &&
            memcmp(h->pool + h->off[i], s, (size_t)len) == 0)
            return (i32)i;
        i = (i + 1) & (u32)(h->cap - 1);
    }
    return -1;
}

i32 hs_get(const HStr *h, const char *s, i32 len)
{
    i32 slot = hs_find_(h, s, len, hs_hash_(s, len));
    return slot < 0 ? -1 : h->val[slot];
}

/* store s -> v; returns the previous value or -1 if s was new */
i32 hs_put(HStr *h, const char *s, i32 len, i32 v)
{
    u64 hv = hs_hash_(s, len);
    i32 slot, tomb = -1;
    u32 i;
    if ((slot = hs_find_(h, s, len, hv)) >= 0) {
        i32 old = h->val[slot];
        h->val[slot] = v;
        return old;
    }
    if ((h->used + 1) * 2 > h->cap) hs_grow_(h);
    i = (u32)hv & (u32)(h->cap - 1);
    while (h->st[i] != 0) {
        if (h->st[i] == 2 && tomb < 0) tomb = (i32)i;
        i = (i + 1) & (u32)(h->cap - 1);
    }
    if (tomb >= 0) i = (u32)tomb; else ++h->used;
    h->st[i] = 1; h->hash[i] = hv;
    h->off[i] = hs_pool_add_(h, s, len); h->len[i] = len; h->val[i] = v;
    ++h->n;
    return -1;
}

/* remove s; returns its value or -1. The key bytes stay in the pool. */
i32 hs_del(HStr *h, const char *s, i32 len)
{
    i32 slot = hs_find_(h, s, len, hs_hash_(s, len)), v;
    if (slot < 0) return -1;
    v = h->val[slot];
    h->st[slot] = 2; --h->n;
    return v;
}

/* iteration: start with *cursor = 0; returns 0 when done. */
i32 hs_next(const HStr *h, i32 *cursor, const char **key, i32 *len, i32 *val)
{
    while (*cursor < h->cap) {
        i32 i = (*cursor)++;
        if (h->st[i] == 1) {
            *key = h->pool + h->off[i]; *len = h->len[i]; *val = h->val[i];
            return 1;
        }
    }
    return 0;
}

#endif /* HDICT_C */

/* ------------------------------------------------------------------ */

#ifdef HDICT_DEMO
#include <stdio.h>
#include <time.h>

int main(void)
{
    enum { N = 200000 };
    H64 h; HStr d;
    i32 i, ok = 1, seen = 0, cur = 0, vlen, vv;
    const char *kp;
    char key[32];
    clock_t t0;
    double t_h, t_scan;
    i64 *scanids;

    /* H64: insert, get, update, delete, tombstone reuse, growth */
    h64_init(&h, 4);
    for (i = 0; i < N; ++i) if (h64_put(&h, (i64)i * 7919 + 13, i) != -1) ok = 0;
    for (i = 0; i < N; ++i) if (h64_get(&h, (i64)i * 7919 + 13) != i) ok = 0;
    if (h.n != N) ok = 0;
    if (h64_put(&h, 13, 42) != 0 || h64_get(&h, 13) != 42) ok = 0;         /* update returns old */
    for (i = 1; i < N; i += 2) if (h64_del(&h, (i64)i * 7919 + 13) != i) ok = 0;
    for (i = 1; i < N; i += 2) if (h64_get(&h, (i64)i * 7919 + 13) != -1) ok = 0;
    for (i = 0; i < N; i += 2) if (h64_get(&h, (i64)i * 7919 + 13) != (i == 0 ? 42 : i)) ok = 0;
    if (h.n != N / 2) ok = 0;
    for (i = 1; i < N; i += 2) h64_put(&h, (i64)i * 7919 + 13, -i - 1 < 0 ? i : 0);   /* reinsert over tombstones */
    if (h.n != N) ok = 0;
    if (h64_get(&h, -5) != -1 || h64_del(&h, -5) != -1) ok = 0;            /* absent */
    printf("H64   n=%d cap=%d used=%d  %s\n", h.n, h.cap, h.used, ok ? "ok" : "FAIL");

    /* HStr: interning, update, delete, iteration, empty and long keys */
    hs_init(&d, 4);
    for (i = 0; i < N; ++i) { sprintf(key, "term-%d", i); if (hs_put(&d, key, (i32)strlen(key), i) != -1) ok = 0; }
    for (i = 0; i < N; ++i) { sprintf(key, "term-%d", i); if (hs_get(&d, key, (i32)strlen(key)) != i) ok = 0; }
    if (hs_get(&d, "term-1", 6) != 1 || hs_get(&d, "term-1x", 7) != -1 || hs_get(&d, "", 0) != -1) ok = 0;
    if (hs_put(&d, "", 0, 7) != -1 || hs_get(&d, "", 0) != 7) ok = 0;
    if (hs_put(&d, "term-5", 6, 555) != 5 || hs_get(&d, "term-5", 6) != 555) ok = 0;
    for (i = 0; i < N; i += 3) { sprintf(key, "term-%d", i); if (hs_del(&d, key, (i32)strlen(key)) < 0) ok = 0; }
    for (i = 0; i < N; ++i) { sprintf(key, "term-%d", i); if ((hs_get(&d, key, (i32)strlen(key)) == -1) != (i % 3 == 0)) ok = 0; }
    while (hs_next(&d, &cur, &kp, &vlen, &vv)) { ++seen; if ((i32)strlen(kp) != vlen) ok = 0; }
    if (seen != d.n) ok = 0;
    printf("HStr  n=%d cap=%d pool=%lu B  iterated=%d  %s\n", d.n, d.cap, (unsigned long)d.plen, seen, ok ? "ok" : "FAIL");

    /* the point of it: doc id -> row by hash vs by scanning every row */
    scanids = (i64 *)malloc(sizeof(i64) * N);
    for (i = 0; i < N; ++i) scanids[i] = (i64)i * 7919 + 13;
    t0 = clock();
    for (i = 0; i < N; i += 100) { i32 r; i64 want = (i64)(N - 1 - i) * 7919 + 13; for (r = 0; r < N; ++r) if (scanids[r] == want) break; if (r == N) ok = 0; }
    t_scan = (double)(clock() - t0) / CLOCKS_PER_SEC;
    t0 = clock();
    for (i = 0; i < N; i += 100) if (h64_get(&h, (i64)(N - 1 - i) * 7919 + 13) < 0) ok = 0;
    t_h = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("lookup of %d ids among %d: scan %.4f s, hash %.6f s  (%s)\n", N / 100, N, t_scan, t_h, t_h < t_scan ? "hash faster" : "?");
    free(scanids);
    h64_free(&h); hs_free(&d);
    puts(ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
#endif
