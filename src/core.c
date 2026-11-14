/* core.c - the few helpers every layer shares: the square root and squared length the store normalises with, the
 * top-k insertion (proved in lean4/Vecindex/TopK.lean), and little-endian file I/O (a big-endian host swaps on the
 * way in and out, so a file is the same everywhere). Part of store.c, which includes it first; not compiled on its
 * own. These came out of the old adaptive engine (engine.c, now in trash/), which nothing else of it survives. */
#define ARENA_ALLOC_IMPLEMENTATION
#include "base.h"
#include "arena_alloc.h"
#include "vsimd.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static f64 eng_sqrt_(f64 x)
{
    union { f32 f; u32 i; } u;
    f64 g, sc = 1.0; i32 i;
    if (x <= 0.0) return 0.0;
    if (!(x <= 1.7e308)) return x;              /* inf (and NaN) as they are: Newton would make inf/inf a NaN */
    while (x > 1e30) { x *= 1e-30; sc *= 1e15; }    /* the first guess is an f32's: x into its range */
    while (x < 1e-30) { x *= 1e30; sc *= 1e-15; }
    u.f = (f32)x; u.i = (u.i >> 1) + 0x1fbd1df5u; g = (f64)u.f;
    for (i = 0; i < 3; ++i) g = 0.5 * (g + x / g);
    return g * sc;
}

/* the squared length of v: the SIMD dot product, or again in f64 when that overflowed - a component past about
 * 1.8e19 squares past f32's range, and its inf length made the store's query and rows NaN (tests/fuzz_proto.c) */
static f64 eng_ssq_(const f32 *v, i32 dim)
{
    f32 s = vs_dot(v, v, dim);
    f64 d = 0.0;
    i32 i;
    if (s <= 3.4e38f) return (f64)s;
    for (i = 0; i < dim; ++i) d += (f64)v[i] * (f64)v[i];
    return d;
}

static void eng_topk_(i64 *ids, f32 *sc, i32 k, i32 *have, i64 id, f32 score)
{
    i32 j;
    if (*have < k) {
        j = *have; ids[j] = id; sc[j] = score;
        while (j > 0 && sc[j - 1] < sc[j]) {
            f32 ts = sc[j-1]; i64 ti = ids[j-1];
            sc[j-1] = sc[j]; ids[j-1] = ids[j]; sc[j] = ts; ids[j] = ti; --j;
        }
        ++(*have);
    } else if (score > sc[k - 1]) {
        j = k - 1; ids[j] = id; sc[j] = score;
        while (j > 0 && sc[j - 1] < sc[j]) {
            f32 ts = sc[j-1]; i64 ti = ids[j-1];
            sc[j-1] = sc[j]; ids[j-1] = ids[j]; sc[j] = ts; ids[j] = ti; --j;
        }
    }
}

static int eng_host_be_(void) { u32 x = 1u; return *(const u8 *)&x == 0; }

static void eng_bswap_(void *p, size_t esz, size_t n)
{
    u8 *b = (u8 *)p;
    size_t i, j;
    for (i = 0; i < n; ++i) {
        u8 *e = b + i * esz;
        for (j = 0; j < esz / 2; ++j) {
            u8 t = e[j]; e[j] = e[esz - 1 - j]; e[esz - 1 - j] = t;
        }
    }
}

/* write / read n elements of esz bytes as little-endian */
static int engs_wle_(FILE *f, const void *p, size_t esz, size_t n)
{
    if (esz == 1 || !eng_host_be_())
        return fwrite(p, esz, n, f) == n ? 0 : -1;
    {
        void *tmp = malloc(esz * (n ? n : 1));
        int rc;
        if (!tmp) return -1;
        memcpy(tmp, p, esz * n);
        eng_bswap_(tmp, esz, n);
        rc = fwrite(tmp, esz, n, f) == n ? 0 : -1;
        free(tmp);
        return rc;
    }
}
static int engs_rle_(FILE *f, void *p, size_t esz, size_t n)
{
    if (fread(p, esz, n, f) != n) return -1;
    if (esz > 1 && eng_host_be_()) eng_bswap_(p, esz, n);
    return 0;
}
