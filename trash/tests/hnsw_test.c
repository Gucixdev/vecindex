/* hnsw_test.c - the self-test of src/hnsw.c; ./run builds and runs it. */
#define ARENA_ALLOC_IMPLEMENTATION
#include "hnsw.c"
#define PRINT_IMPLEMENTATION
#include "print.h"
#include <stdio.h>
#include <time.h>

static f64 now_s(void)
{ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return (f64)t.tv_sec + (f64)t.tv_nsec * 1e-9; }

static u32 dr_(u32 *s){ *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static f32 df_(u32 *s){ return (f32)dr_(s) / (f32)0xffffffffu - 0.5f; }

/* brute-force nprobe-nearest, for the recall baseline */
static void brute_(const f32 *pts, i32 n, i32 dim, const f32 *q, i32 kp, i64 *out)
{
    f32 bd[64]; i32 have = 0, c, j;
    if (kp > 64) kp = 64;
    for (c = 0; c < n; ++c) {
        f32 d = vs_sqdist(q, pts + (size_t)c * dim, dim);
        if (have < kp) { j = have; bd[j] = d; out[j] = c;
            while (j > 0 && bd[j-1] > bd[j]) { f32 t = bd[j-1]; i64 ti = out[j-1];
                bd[j-1]=bd[j]; out[j-1]=out[j]; bd[j]=t; out[j]=ti; --j; } ++have; }
        else if (d < bd[kp-1]) { j = kp-1; bd[j] = d; out[j] = c;
            while (j > 0 && bd[j-1] > bd[j]) { f32 t = bd[j-1]; i64 ti = out[j-1];
                bd[j-1]=bd[j]; out[j-1]=out[j]; bd[j]=t; out[j]=ti; --j; } }
    }
}

int main(void)
{
    static Arena arena;
    Arena *a = &arena;
    i32 dim = 96, n = 4000, nq = 500, kp = 10, i, q, c, topics = 60;
    f32 *pts = (f32 *)arena_alloc(a, (size_t)n * dim * sizeof(f32));
    f32 *qs  = (f32 *)arena_alloc(a, (size_t)nq * dim * sizeof(f32));
    f32 *ctr = (f32 *)arena_alloc(a, (size_t)topics * dim * sizeof(f32));
    u32 st = 777u;
    Hnsw *h;
    HnswParams p;
    i64 g[64];
    i32 r[64];
    i32 hit1 = 0, hitk = 0;
    f64 t0, tb, tr;

    /* clustered, like the coarse centroids the router actually indexes:
     * structure to navigate, not uniform-random worst case. */
    for (i = 0; i < topics * dim; ++i) ctr[i] = df_(&st) * 6.0f;
    for (i = 0; i < n; ++i) { c = (i32)(dr_(&st) % (u32)topics);
        for (q = 0; q < dim; ++q) pts[(size_t)i*dim+q] = ctr[c*dim+q] + df_(&st) * 1.2f; }
    for (i = 0; i < nq; ++i) { c = (i32)(dr_(&st) % (u32)topics);
        for (q = 0; q < dim; ++q) qs[(size_t)i*dim+q] = ctr[c*dim+q] + df_(&st) * 1.4f; }

    p.M = 16; p.ef_construction = 200; p.seed = 42u;
    t0 = now_s(); h = hnsw_build(a, pts, n, dim, p); tb = (now_s() - t0) * 1e3;

    t0 = now_s();
    for (q = 0; q < nq; ++q) {
        const f32 *qq = qs + (size_t)q * dim;
        i32 got = hnsw_route(h, qq, kp, 48, r), x, y;
        brute_(pts, n, dim, qq, kp, g);
        if (got > 0 && (i64)r[0] == g[0]) ++hit1;
        for (x = 0; x < got; ++x)
            for (y = 0; y < kp; ++y) if ((i64)r[x] == g[y]) { ++hitk; break; }
    }
    tr = (now_s() - t0) / nq * 1e6;

    print_cstr("hnsw  n="); print_i64(n); print_cstr(" dim="); print_i64(dim);
    print_cstr("  M="); print_i64(h->M); print_cstr("  maxl="); print_i64(h->maxl);
    print_cstr("\n");
    printf("  build %.1f ms   route %.2f us/q\n", tb, tr);
    printf("  vs brute nprobe=%d:  nearest-1 = %.1f%%   recall@%d = %.1f%%\n",
           kp, 100.0 * hit1 / nq, kp, 100.0 * hitk / (nq * kp));
    print_flush();
    arena_free(a);
    return (hit1 * 100 / nq >= 95 && hitk * 100 / (nq * kp) >= 90) ? 0 : 1;
}
