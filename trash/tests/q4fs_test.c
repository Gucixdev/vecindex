/* q4fs_test.c - the self-test of src/q4fs.c; ./run builds and runs it. */
#define ARENA_ALLOC_IMPLEMENTATION
#include "q4fs.c"
#define PRINT_IMPLEMENTATION
#include "print.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static f64 now_s(void)
{ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return (f64)t.tv_sec + (f64)t.tv_nsec * 1e-9; }

static u32 fs_rng_(u32 *s){ *s^=*s<<13; *s^=*s>>17; *s^=*s<<5; return *s; }
static f32 fs_f_(u32 *s){ return (f32)fs_rng_(s)/(f32)0xffffffffu - 0.5f; }

int main(void)
{
    static Arena arena;
    Arena *a = &arena;
    i32 dim = 384, n = 20000, nq = 200, k = 10, i, d, qi;
    f32 *v = (f32 *)arena_alloc(a, (size_t)n * dim * sizeof(f32));
    u32 st = 12345u;
    Q4Index *base;
    Q4FSIndex *fs;
    i64 a1[16], a2[16]; f32 s1[16], s2[16];
    i32 agree = 0, tot = 0;
    f64 t0, t_plain, t_fast;

    for (i = 0; i < n * dim; ++i) v[i] = fs_f_(&st) * (1.0f + (f32)((i % dim) < dim / 4) * 3.0f);
    base = q4_build(a, v, n, dim);
    fs = q4fs_wrap(a, base);

    t0 = now_s();
    for (qi = 0; qi < nq; ++qi) q4_search_l2(base, v + (size_t)(qi * (n / nq)) * dim, k, a1, s1);
    t_plain = (now_s() - t0) / nq * 1e6;

    t0 = now_s();
    for (qi = 0; qi < nq; ++qi) q4fs_search_l2(fs, v + (size_t)(qi * (n / nq)) * dim, k, a2, s2);
    t_fast = (now_s() - t0) / nq * 1e6;

    for (qi = 0; qi < nq; ++qi) {
        i32 x, y;
        q4_search_l2(base, v + (size_t)(qi * (n / nq)) * dim, k, a1, s1);
        q4fs_search_l2(fs,  v + (size_t)(qi * (n / nq)) * dim, k, a2, s2);
        for (x = 0; x < k; ++x) for (y = 0; y < k; ++y)
            if (a1[x] == a2[y]) { ++agree; break; }
        tot += k;
    }
    (void)d;
    print_cstr("q4fs  dim="); print_i64(dim); print_cstr(" n="); print_i64(n);
    print_cstr("  code_bytes="); print_i64(fs->code_bytes); print_cstr("\n");
    printf("  q4 plain L2 = %.1f us/q     fast-scan L2 = %.1f us/q     speedup = %.2fx\n",
           t_plain, t_fast, t_plain / t_fast);
    printf("  top-%d agreement fast-scan vs plain q4 = %.1f%%\n",
           k, 100.0 * agree / tot);
    if (!Q4FS_SIMD) printf("  (no AVX2 here: the fast-scan runs its scalar fallback, so the speed is not checked)\n");
    print_flush();
    arena_free(a);
    return (agree * 100 / tot >= 95 && (!Q4FS_SIMD || t_fast < t_plain)) ? 0 : 1;
}
