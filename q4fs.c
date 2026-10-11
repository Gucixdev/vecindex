/* q4fs.c - FAISS-style 4-bit fast-scan over a built q4 index.
 *
 * q4's per-query scoring gathers a 256-entry f32 LUT once per code byte
 * per vector. Fast-scan trades that for an 8-bit LUT (16 entries per
 * dimension, one nibble) read with _mm256_shuffle_epi8 (pshufb): 32
 * vectors' nibbles for a dimension are transposed into 32 contiguous
 * bytes, so one instruction does 32 table lookups. Accumulate 16-bit,
 * shortlist by the quantised score, then re-rank the shortlist exactly
 * on q4's own f32 LUT - so recall matches q4 while the scan is ~4-6x
 * faster on the wide-dim / large-n case.
 *
 * Not a new quantizer: it wraps an existing Q4Index (its calibration,
 * codebook, centroid, rnorm, qn2 and original codes are reused as-is)
 * and only adds the transposed code block. engine.c selects it in place
 * of a plain q4 scan when the corpus is big enough to amortise the
 * transpose.
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -march=native -I. q4fs.c \
 *       -DQ4FS_DEMO -lm -o q4fs && ./q4fs
 */

#ifdef Q4FS_DEMO
#define ARENA_ALLOC_IMPLEMENTATION
#endif
#include "base.h"
#include "arena_alloc.h"
#include "vsimd.h"
#include <string.h>

#ifndef Q4_LEVELS
#include "q4.c"
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define Q4FS_SIMD 1
#else
#define Q4FS_SIMD 0
#endif

#define Q4FS_BLK 32          /* vectors scored per pshufb pass */

typedef struct {
    Arena *arena;
    i32 dim, code_bytes, count;
    const Q4Index *base;     /* calibration + f32 LUT rerank + rnorm/qn2 */
    i32 nblk;               /* ceil(count / Q4FS_BLK) */
    u8 *packed;             /* [nblk][code_bytes][Q4FS_BLK] transpose of base->codes */
    f32 l2_lambda;
} Q4FSIndex;

/* ------------------------------------------------------------------ */

Q4FSIndex *q4fs_wrap(Arena *a, const Q4Index *base)
{
    Q4FSIndex *ix = (Q4FSIndex *)arena_alloc(a, sizeof(Q4FSIndex));
    i32 cb = base->code_bytes, n = base->count, nblk = (n + Q4FS_BLK - 1) / Q4FS_BLK;
    i32 blk, g, r;
    ix->arena = a; ix->base = base;
    ix->dim = base->dim; ix->code_bytes = cb; ix->count = n;
    ix->nblk = nblk; ix->l2_lambda = base->l2_lambda;
    ix->packed = (u8 *)arena_alloc(a, (size_t)(nblk > 0 ? nblk : 1) * cb * Q4FS_BLK);
    memset(ix->packed, 0, (size_t)(nblk > 0 ? nblk : 1) * cb * Q4FS_BLK);
    for (blk = 0; blk < nblk; ++blk) {
        u8 *pb = ix->packed + (size_t)blk * cb * Q4FS_BLK;
        for (r = 0; r < Q4FS_BLK; ++r) {
            i32 row = blk * Q4FS_BLK + r;
            const u8 *src;
            if (row >= n) break;
            src = base->codes + (size_t)row * cb;
            for (g = 0; g < cb; ++g) pb[(size_t)g * Q4FS_BLK + r] = src[g];
        }
    }
    return ix;
}

Q4FSIndex *q4fs_build(Arena *a, const f32 *vectors, i32 n, i32 dim)
{ return q4fs_wrap(a, q4_build(a, vectors, n, dim)); }

Q4FSIndex *q4fs_build_cluster(Arena *a, const f32 *vectors, const i64 *ids,
                              i32 n_ids, i32 dim, const f32 *centroid)
{ return q4fs_wrap(a, q4_build_cluster(a, vectors, ids, n_ids, dim, centroid)); }

i32 q4fs_code_bytes(i32 dim) { return q4_code_bytes(dim); }

/* ------------------------------------------------------------------ */

static void q4fs_topk_(i64 *ids, f32 *sc, i32 k, i32 *have, i64 id, f32 score)
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

/* per-dim 16-entry u8 LUT + the affine that maps a code's u16 sum back
 * to the f32 dot: dot ~= sum/scale + dim*bias. */
typedef struct { u8 *lu; f32 scale, bias; } Q4FSLut;

static Q4FSLut q4fs_lut_(const Q4FSIndex *ix, const f32 *residual, u8 *lu)
{
    const f32 *cb = ix->base->codebook;
    i32 dim = ix->dim, d, l;
    f32 mn = 1e30f, mx = -1e30f, span, scale, q;
    Q4FSLut out;
    for (d = 0; d < dim; ++d) {
        const f32 *c = cb + (size_t)d * Q4_LEVELS;
        f32 r = residual[d];
        for (l = 0; l < Q4_LEVELS; ++l) {
            f32 v = r * c[l];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
    }
    span = mx - mn;
    /* two constraints on the per-entry cap q:
     *   - the block kernel does one u8 add of two entries (lo+hi nibble)
     *     before widening, so 2*q <= 255  ->  q <= 127
     *   - the u16 block accumulator sums code_bytes such adds, so
     *     2*q*code_bytes <= 65535
     * q = 127 satisfies both for any dim up to ~258; clamp for wider. */
    q = 65200.0f / (f32)(dim > 0 ? dim : 1);
    if (q > 127.0f) q = 127.0f;
    scale = (span > 0.0f) ? q / span : 1.0f;
    for (d = 0; d < dim; ++d) {
        const f32 *c = cb + (size_t)d * Q4_LEVELS;
        f32 r = residual[d];
        u8 *row = lu + (size_t)d * 16;
        for (l = 0; l < Q4_LEVELS; ++l) {
            i32 u = (i32)((r * c[l] - mn) * scale + 0.5f);
            row[l] = (u8)(u < 0 ? 0 : (u > 255 ? 255 : u));
        }
    }
    out.lu = lu; out.scale = (scale != 0.0f) ? scale : 1.0f; out.bias = mn;
    return out;
}

/* one block -> 32 raw u16 sums (rows past count are left as-is; caller
 * bounds the loop). */
static void q4fs_block_(const Q4FSIndex *ix, const u8 *pb, const u8 *lu, u16 *out32)
{
#if Q4FS_SIMD
    i32 cb = ix->code_bytes, g;
    __m256i acc_lo = _mm256_setzero_si256();   /* rows 0..15  as 16x u16 */
    __m256i acc_hi = _mm256_setzero_si256();   /* rows 16..31 as 16x u16 */
    const __m256i m0f = _mm256_set1_epi8(0x0f);
    for (g = 0; g < cb; ++g) {
        __m256i c  = _mm256_loadu_si256((const __m256i *)(const void *)(pb + (size_t)g * Q4FS_BLK));
        __m256i lo = _mm256_and_si256(c, m0f);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), m0f);
        __m128i tl0 = _mm_loadu_si128((const __m128i *)(const void *)(lu + (size_t)(2*g) * 16));
        __m128i th0 = _mm_loadu_si128((const __m128i *)(const void *)(lu + (size_t)(2*g + 1) * 16));
        __m256i tl = _mm256_broadcastsi128_si256(tl0);
        __m256i th = _mm256_broadcastsi128_si256(th0);
        __m256i s  = _mm256_add_epi8(_mm256_shuffle_epi8(tl, lo),
                                    _mm256_shuffle_epi8(th, hi));   /* 32x u8, <=255+? */
        /* widen the two 128-bit halves cleanly (no lane permute) */
        acc_lo = _mm256_add_epi16(acc_lo, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(s)));
        acc_hi = _mm256_add_epi16(acc_hi, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(s, 1)));
    }
    _mm256_storeu_si256((__m256i *)(void *)out32,        acc_lo);
    _mm256_storeu_si256((__m256i *)(void *)(out32 + 16), acc_hi);
#else
    i32 cb = ix->code_bytes, g, r;
    for (r = 0; r < Q4FS_BLK; ++r) {
        u32 s = 0;
        for (g = 0; g < cb; ++g) {
            u8 byte = pb[(size_t)g * Q4FS_BLK + r];
            s += lu[(size_t)(2*g) * 16 + (byte & 0x0f)];
            s += lu[(size_t)(2*g + 1) * 16 + (byte >> 4)];
        }
        out32[r] = (u16)(s > 65535u ? 65535u : s);
    }
#endif
}

/* scratch the _into forms need, in f32 units: residual[dim] and the f32
 * rerank LUT [code_bytes*256] (matches engine_scratch_floats). The u8
 * fast-scan LUT is small and taken from the arena internally. */
i32 q4fs_scratch_residual(i32 dim) { return dim; }
i32 q4fs_scratch_lut(i32 dim) { return q4_code_bytes(dim) * 256; }

/* shared shortlist+rerank core. `mode` 0 = cosine flat, 1 = IVF L2. */
static i32 q4fs_search_core_(const Q4FSIndex *ix, const f32 *query, i32 k,
                             i64 *out_ids, f32 *out_scores,
                             f32 *residual, f32 *lut_f32, int l2)
{
    const Q4Index *b = ix->base;
    Arena *a = ix->arena;
    Arena_Mark m = arena_snapshot(a);
    i32 dim = ix->dim, cb = ix->code_bytes, n = ix->count;
    i32 shortn = 4 * k + 32; i32 blk, r, i, have = 0;
    u8  *lu   = (u8 *)arena_alloc(a, (size_t)dim * 16);
    u16 *raw  = (u16 *)arena_alloc(a, Q4FS_BLK * sizeof(u16));
    i64 *sid  = (i64 *)arena_alloc(a, (size_t)shortn * sizeof(i64));
    f32 *ssc  = (f32 *)arena_alloc(a, (size_t)shortn * sizeof(f32));
    f32 qc, qnorm;
    Q4FSLut L;
    i32 sh = 0;

    if (shortn > n) shortn = n;
    qc = vs_sub_normsq(residual, query, b->centroid, dim);
    qnorm = (f32)q4_sqrt_((f64)qc); if (qnorm == 0.0f) qnorm = 1.0f;
    L = q4fs_lut_(ix, residual, lu);

    /* pass 1: quantised shortlist over every block */
    for (blk = 0; blk < ix->nblk; ++blk) {
        const u8 *pb = ix->packed + (size_t)blk * cb * Q4FS_BLK;
        i32 base_row = blk * Q4FS_BLK, lim = n - base_row;
        if (lim > Q4FS_BLK) lim = Q4FS_BLK;
        q4fs_block_(ix, pb, lu, raw);
        for (r = 0; r < lim; ++r) {
            i32 id = base_row + r;
            f32 t3, key;
            if (l2) {
                t3 = ix->l2_lambda * (b->rnorm[id] * b->rnorm[id])
                   + (1.0f - ix->l2_lambda) * b->qn2[id];
                key = 2.0f * ((f32)raw[r] / L.scale + (f32)dim * L.bias) - t3;
            } else {
                key = (f32)raw[r];   /* monotone in the dot; exact rerank fixes order */
            }
            if (sh < shortn) {
                i32 j = sh; sid[j] = id; ssc[j] = key;
                while (j > 0 && ssc[j-1] < ssc[j]) {
                    f32 ts = ssc[j-1]; i64 ti = sid[j-1];
                    ssc[j-1] = ssc[j]; sid[j-1] = sid[j]; ssc[j] = ts; sid[j] = ti; --j;
                }
                ++sh;
            } else if (key > ssc[shortn - 1]) {
                i32 j = shortn - 1; sid[j] = id; ssc[j] = key;
                while (j > 0 && ssc[j-1] < ssc[j]) {
                    f32 ts = ssc[j-1]; i64 ti = sid[j-1];
                    ssc[j-1] = ssc[j]; sid[j-1] = sid[j]; ssc[j] = ts; sid[j] = ti; --j;
                }
            }
        }
    }

    /* pass 2: exact f32 rerank of the shortlist on q4's own LUT */
    q4_lut_resid(b, residual, lut_f32);
    for (i = 0; i < sh; ++i) {
        i32 id = (i32)sid[i];
        const u8 *row = b->codes + (size_t)id * cb;
        f32 rawf = vs_lut_gather_dot(lut_f32, row, cb), score;
        if (l2) {
            f32 t3 = ix->l2_lambda * (b->rnorm[id] * b->rnorm[id])
                   + (1.0f - ix->l2_lambda) * b->qn2[id];
            f32 dist_sq = qc - 2.0f * rawf + t3;
            if (dist_sq < 0.0f) dist_sq = 0.0f;
            score = -dist_sq;
        } else {
            f32 denom = qnorm * b->rnorm[id];
            score = (denom > 0.0f) ? rawf / denom : 0.0f;
        }
        q4fs_topk_(out_ids, out_scores, k, &have, (i64)id, score);
    }
    arena_rewind(a, m);
    return have;
}

i32 q4fs_search_into(const Q4FSIndex *ix, const f32 *query, i32 k,
                     i64 *out_ids, f32 *out_scores, f32 *residual, f32 *lut)
{ return q4fs_search_core_(ix, query, k, out_ids, out_scores, residual, lut, 0); }

i32 q4fs_search_l2_into(const Q4FSIndex *ix, const f32 *query, i32 k,
                        i64 *out_ids, f32 *out_scores, f32 *residual, f32 *lut)
{ return q4fs_search_core_(ix, query, k, out_ids, out_scores, residual, lut, 1); }

i32 q4fs_search(const Q4FSIndex *ix, const f32 *query, i32 k,
                i64 *out_ids, f32 *out_scores)
{
    Arena *a = ix->arena;
    Arena_Mark m = arena_snapshot(a);
    f32 *res = (f32 *)arena_alloc(a, (size_t)ix->dim * sizeof(f32));
    f32 *lut = (f32 *)arena_alloc(a, (size_t)ix->code_bytes * 256 * sizeof(f32));
    i32 n = q4fs_search_core_(ix, query, k, out_ids, out_scores, res, lut, 0);
    arena_rewind(a, m);
    return n;
}

i32 q4fs_search_l2(const Q4FSIndex *ix, const f32 *query, i32 k,
                   i64 *out_ids, f32 *out_scores)
{
    Arena *a = ix->arena;
    Arena_Mark m = arena_snapshot(a);
    f32 *res = (f32 *)arena_alloc(a, (size_t)ix->dim * sizeof(f32));
    f32 *lut = (f32 *)arena_alloc(a, (size_t)ix->code_bytes * 256 * sizeof(f32));
    i32 n = q4fs_search_core_(ix, query, k, out_ids, out_scores, res, lut, 1);
    arena_rewind(a, m);
    return n;
}

/* ================================================================== */
#ifdef Q4FS_DEMO
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
    print_flush();
    arena_free(a);
    return (agree * 100 / tot >= 95 && t_fast < t_plain) ? 0 : 1;
}
#endif /* Q4FS_DEMO */
