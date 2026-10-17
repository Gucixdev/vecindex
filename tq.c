/* tq.c - TurboQuant-style flat index: random rotation + 4-bit Lloyd-Max scalar codes + SIMD fast-scan.
 *
 * What it takes from Google's TurboQuant (as turbovec ships it): a data-oblivious quantizer.
 *  - a fixed random rotation (random signs + Walsh-Hadamard, applied twice around a permutation) makes
 *    every coordinate of a unit vector ~ N(0, 1/d), whatever the data looks like;
 *  - so one fixed Lloyd-Max codebook for the Gaussian quantises every coordinate optimally: no training,
 *    no k-means, adding a vector is O(d log d) and never invalidates anything;
 *  - codes are stored in the "fast-scan" layout (blocks of 32 vectors, 16 bytes per dimension) and a query
 *    is scored with byte shuffles: one PSHUFB looks up 32 table entries at once, no per-code gather.
 * The scan only proposes candidates; the caller re-scores them exactly (the store keeps the f32 rows).
 *
 * Large stores group the rows into clusters (IVF) and a query scans only the closest ones. The clustering is
 * built to scale: a sample that grows with the cluster count (capped in bytes), two-level k-means once there are
 * more than TQ_FLAT_MAX clusters, centroid scoring 16 centroids per instruction, and assignment and encoding
 * spread over threads.
 *
 *   Tq *tq_build(rows, n, dim, seed, nc, nthreads)   rows: n x dim f32 (unit length), possibly in chunks
 *   tq_search(tq, q, R, rows, filter..., nthreads)    the R best rows by approximate inner product
 *   tq_free
 * C89, AVX2 / AVX-512 kernels when compiled with -march=native, scalar otherwise; threads via pthreads. */
#ifndef TQ_C
#define TQ_C

#include "base.h"
#include "vsimd.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* Rows of a column in chunks of 2^shift rows (one chunk with a large shift = a plain array). A vector column holds
 * f32 rows, or int8 rows (i8 = 1) whose f32 scale sits in a second column. */
typedef struct TqCol { u8 *const *chunk; i32 shift; size_t esz; i32 i8, dim; const struct TqCol *scale; } TqCol;
#define TQ_AT(c, r) ((c)->chunk[(size_t)(r) >> (c)->shift] + ((size_t)(r) & (((size_t)1 << (c)->shift) - 1u)) * (c)->esz)
#define TQ_ROW(c, r) ((const f32 *)(const void *)TQ_AT(c, r))

/* a TqCol over one contiguous array (holder keeps the one chunk pointer) */
TqCol tq_col_flat(const void *base, size_t esz, u8 **holder)
{
    TqCol c;
    *holder = (u8 *)base;
    c.chunk = holder; c.shift = 31; c.esz = esz; c.i8 = 0; c.dim = 0; c.scale = (const TqCol *)0;
    return c;
}

/* row r as f32: a pointer into the column, or (int8 rows) dequantised into buf[dim] */
static const f32 *tq_row_(const TqCol *c, i64 r, f32 *buf)
{
    const i8 *x;
    f32 s;
    i32 d = 0;
    if (!c->i8) return TQ_ROW(c, r);
    x = (const i8 *)(const void *)TQ_AT(c, r);
    s = *(const f32 *)(const void *)TQ_AT(c->scale, r);
#if defined(__AVX512F__)
    for (; d + 16 <= c->dim; d += 16)
        _mm512_storeu_ps(buf + d, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(x + d)))), _mm512_set1_ps(s)));
#endif
    for (; d < c->dim; ++d) buf[d] = s * (f32)x[d];
    return buf;
}

typedef struct {
    i32   dim, dpad, n, nblk;
    /* optional IVF: rows are grouped by cluster (blocks of 32 slots per cluster, padded), a query scans only the
     * blocks of its nprobe closest clusters. nc == 0: flat, slot s holds row s. */
    i32   nc, nprobe;
    f32  *cent;             /* [nc x dim] unit centroids */
    f32  *ct;               /* the centroids again, transposed in groups of 16 for the scoring kernel (always malloc'd) */
    i32  *cstart;           /* [nc + 1] first block of each cluster */
    i32  *slot_row;         /* [nblk * 32] row held by each slot, -1 = padding */
    f32  *s1, *s2;          /* random +-1 signs, [dpad] */
    i32  *perm;             /* [dpad] */
    u8   *codes;            /* nblk * vd * 16 bytes */
    i32   mapped;           /* 1: the arrays point into a file mapping the caller owns (store_load) */
    i32   bits;             /* 4: one 4-bit code a coordinate; 2: one 4-bit code a pair of coordinates (2 bits each) */
    i32   vd;               /* code columns a row: dpad (4-bit) or dpad/2 rounded up to even (2-bit) */
} Tq;

static i32 tq_vd_(i32 dpad, i32 bits) { return bits == 2 ? ((dpad / 2) + 1) & ~1 : dpad; }

#define TQ_FLAT_MAX   1024              /* more clusters than this: two-level k-means */
#define TQ_SAMPLE_MAX ((size_t)512 << 20)   /* bytes of training sample at most (100M rows: ~25 a centroid) */

/* Lloyd-Max centroids of the standard normal, 16 levels ascending */
static const f32 TQ_CENT[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9424f, -0.6568f, -0.3881f, -0.1284f,
     0.1284f,  0.3881f,  0.6568f,  0.9424f,  1.2562f,  1.6180f,  2.0690f,  2.7326f
};

static u32 tq_rng_(u32 *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }

#if defined(__AVX512F__)
/* butterflies of span h (1, 2, 4 or 8) inside one 16-lane vector: lanes without bit h get x + partner, lanes with
 * it get partner - x */
static __m512 tq_bfly16_(__m512 x, i32 h, __mmask16 hi)
{
    __m512 p = _mm512_permutexvar_ps(_mm512_xor_si512(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                                                      _mm512_set1_epi32(h)), x);
    return _mm512_mask_sub_ps(_mm512_add_ps(x, p), hi, p, x);
}
#endif

/* Walsh-Hadamard on each power-of-two chunk of x[dpad] (binary decomposition of dpad), orthonormal.
 * AVX-512: spans below 16 are lane permutes inside a register, the rest are whole-vector butterflies; AVX2:
 * spans of 8 and more run 8 lanes at a time. */
static void tq_wht_(f32 *x, i32 dpad)
{
    i32 base = 0, len, h, i, j;
    for (len = 1; len * 2 <= dpad; len *= 2) {}
    while (base < dpad) {
        while (len > dpad - base) len /= 2;
#if defined(__AVX512F__)
        if (len >= 16) {
            f32 *y = x + base;
            __m512 sc = _mm512_set1_ps(1.0f / (f32)sqrt((f64)len));
            for (i = 0; i < len; i += 16) {
                __m512 v = _mm512_loadu_ps(y + i);
                v = tq_bfly16_(v, 1, (__mmask16)0xAAAA); v = tq_bfly16_(v, 2, (__mmask16)0xCCCC);
                v = tq_bfly16_(v, 4, (__mmask16)0xF0F0); v = tq_bfly16_(v, 8, (__mmask16)0xFF00);
                _mm512_storeu_ps(y + i, v);
            }
            for (h = 16; h < len; h *= 2)
                for (i = 0; i < len; i += 2 * h)
                    for (j = i; j < i + h; j += 16) {
                        __m512 a = _mm512_loadu_ps(y + j), b = _mm512_loadu_ps(y + j + h);
                        _mm512_storeu_ps(y + j, _mm512_add_ps(a, b)); _mm512_storeu_ps(y + j + h, _mm512_sub_ps(a, b));
                    }
            for (i = 0; i < len; i += 16) _mm512_storeu_ps(y + i, _mm512_mul_ps(_mm512_loadu_ps(y + i), sc));
            base += len;
            continue;
        }
#endif
        for (h = 1; h < len; h *= 2)
            for (i = base; i < base + len; i += 2 * h) {
                j = i;
#if defined(__AVX2__)
                for (; j + 8 <= i + h; j += 8) {
                    __m256 a = _mm256_loadu_ps(x + j), b = _mm256_loadu_ps(x + j + h);
                    _mm256_storeu_ps(x + j, _mm256_add_ps(a, b)); _mm256_storeu_ps(x + j + h, _mm256_sub_ps(a, b));
                }
#endif
                for (; j < i + h; ++j) { f32 a = x[j], b = x[j + h]; x[j] = a + b; x[j + h] = a - b; }
            }
        { f32 sc = 1.0f / (f32)sqrt((f64)len); for (i = base; i < base + len; ++i) x[i] *= sc; }
        base += len;
    }
}

/* out[dpad] = R * in (in has dim entries, the rest is zero), scaled so coordinates are ~ N(0,1) */
static void tq_rot_(const Tq *t, const f32 *in, f32 *out, f32 *tmp)
{
    i32 i = 0;
    f32 sc = (f32)sqrt((f64)t->dim);
#if defined(__AVX512F__)
    for (; i < t->dpad; i += 16) {                                       /* signs, zero padding */
        __mmask16 mi = (__mmask16)(t->dim - i >= 16 ? 0xffff : (t->dim > i ? (1u << (t->dim - i)) - 1u : 0u));
        __mmask16 mp = (__mmask16)(t->dpad - i >= 16 ? 0xffff : (1u << (t->dpad - i)) - 1u);
        _mm512_mask_storeu_ps(tmp + i, mp, _mm512_mul_ps(_mm512_maskz_loadu_ps(mi, in + i), _mm512_maskz_loadu_ps(mp, t->s1 + i)));
    }
    tq_wht_(tmp, t->dpad);
    for (i = 0; i < t->dpad; i += 16) {                                  /* permutation as a gather, second signs */
        __mmask16 mp = (__mmask16)(t->dpad - i >= 16 ? 0xffff : (1u << (t->dpad - i)) - 1u);
        __m512 g = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mp, _mm512_maskz_loadu_epi32(mp, t->perm + i), tmp, 4);
        _mm512_mask_storeu_ps(out + i, mp, _mm512_mul_ps(g, _mm512_maskz_loadu_ps(mp, t->s2 + i)));
    }
    tq_wht_(out, t->dpad);
    for (i = 0; i < t->dpad; i += 16) {
        __mmask16 mp = (__mmask16)(t->dpad - i >= 16 ? 0xffff : (1u << (t->dpad - i)) - 1u);
        _mm512_mask_storeu_ps(out + i, mp, _mm512_mul_ps(_mm512_maskz_loadu_ps(mp, out + i), _mm512_set1_ps(sc)));
    }
#else
    for (i = 0; i < t->dpad; ++i) tmp[i] = (i < t->dim ? in[i] : 0.0f) * t->s1[i];
    tq_wht_(tmp, t->dpad);
    for (i = 0; i < t->dpad; ++i) out[i] = tmp[t->perm[i]] * t->s2[i];
    tq_wht_(out, t->dpad);
    for (i = 0; i < t->dpad; ++i) out[i] *= sc;
#endif
}

/* nearest of the 16 levels: boundaries are the midpoints; four branch-free steps of a binary search */
static const f32 TQ_B[16] = { -2.4008f, -1.8435f, -1.4371f, -1.0993f, -0.7996f, -0.5225f, -0.2583f, 0.0f,
                               0.2583f, 0.5225f, 0.7996f, 1.0993f, 1.4371f, 1.8435f, 2.4008f, 1e30f };
static u8 tq_code_(f32 v)
{
    i32 c = (v > TQ_B[7]) << 3;
    c += (v > TQ_B[c + 3]) << 2;
    c += (v > TQ_B[c + 1]) << 1;
    c += (v > TQ_B[c]);
    return (u8)c;
}

/* codes[d] for every coordinate of a rotated row; AVX-512: the same binary search on 16 lanes, the boundary of
 * each step fetched with one lane permute */
static void tq_codes_(const f32 *rot, i32 dpad, u8 *codes)
{
    i32 d = 0;
#if defined(__AVX512F__)
    const __m512 B = _mm512_loadu_ps(TQ_B), b7 = _mm512_set1_ps(TQ_B[7]);
    const __m512i one = _mm512_set1_epi32(1), two = _mm512_set1_epi32(2), three = _mm512_set1_epi32(3), four = _mm512_set1_epi32(4);
    for (; d + 16 <= dpad; d += 16) {
        __m512 x = _mm512_loadu_ps(rot + d);
        __m512i c = _mm512_maskz_mov_epi32(_mm512_cmp_ps_mask(x, b7, _CMP_GT_OQ), _mm512_set1_epi32(8));
        c = _mm512_mask_add_epi32(c, _mm512_cmp_ps_mask(x, _mm512_permutexvar_ps(_mm512_add_epi32(c, three), B), _CMP_GT_OQ), c, four);
        c = _mm512_mask_add_epi32(c, _mm512_cmp_ps_mask(x, _mm512_permutexvar_ps(_mm512_add_epi32(c, one), B), _CMP_GT_OQ), c, two);
        c = _mm512_mask_add_epi32(c, _mm512_cmp_ps_mask(x, _mm512_permutexvar_ps(c, B), _CMP_GT_OQ), c, one);
        _mm_storeu_si128((__m128i *)(void *)(codes + d), _mm512_cvtepi32_epi8(c));
    }
#endif
    for (; d < dpad; ++d) codes[d] = tq_code_(rot[d]);
}

/* 2-bit: Lloyd-Max levels of the standard normal (4 of them) and the decision boundaries between them */
static const f32 TQ_CENT2[4] = { -1.5104f, -0.4528f, 0.4528f, 1.5104f };
static const f32 TQ_B2[3] = { -0.9816f, 0.0f, 0.9816f };

/* code columns of a rotated row: 4-bit, one per coordinate; 2-bit, coordinates 2v and 2v+1 share column v as
 * (code of 2v) | (code of 2v+1) << 2, so the 16-entry tables and the scan kernel stay exactly the same */
static void tq_encode_row_(const Tq *t, const f32 *rot, u8 *cd)
{
    i32 v;
    if (t->bits != 2) { tq_codes_(rot, t->dpad, cd); return; }
    for (v = 0; v < t->vd; ++v) {
        f32 a = 2 * v < t->dpad ? rot[2 * v] : 0.0f, b = 2 * v + 1 < t->dpad ? rot[2 * v + 1] : 0.0f;
        i32 ca = (a > TQ_B2[0]) + (a > TQ_B2[1]) + (a > TQ_B2[2]), cb = (b > TQ_B2[0]) + (b > TQ_B2[1]) + (b > TQ_B2[2]);
        cd[v] = (u8)(ca | (cb << 2));
    }
}

/* ---- centroid scoring: 16 centroids per step, centroids transposed in groups ([dim][16] per group) ---- */

static f32 *tq_transpose_(const f32 *cent, i32 nc, i32 dim)
{
    i32 ng = (nc + 15) / 16, g, d, l;
    f32 *ct = (f32 *)calloc((size_t)(ng ? ng : 1) * (size_t)dim * 16, sizeof(f32));
    for (g = 0; g < ng; ++g)
        for (l = 0; l < 16 && g * 16 + l < nc; ++l)
            for (d = 0; d < dim; ++d) ct[((size_t)g * dim + (size_t)d) * 16 + (size_t)l] = cent[(size_t)(g * 16 + l) * dim + (size_t)d];
    return ct;
}

/* out[16 * ngroups] = dots of v with every centroid of the groups (padding lanes score 0) */
static void tq_score_(const f32 *v, const f32 *ct, i32 ngroups, i32 dim, f32 *out)
{
    i32 g = 0, d;
    size_t gs = (size_t)dim * 16;
#if defined(__AVX512F__)
    for (; g + 8 <= ngroups; g += 8) {               /* 8 independent accumulators keep both FMA ports busy */
        const f32 *c = ct + (size_t)g * gs;
        __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
        for (d = 0; d < dim; ++d) {
            __m512 x = _mm512_set1_ps(v[d]);
            const f32 *p = c + (size_t)d * 16;
            a0 = _mm512_fmadd_ps(x, _mm512_loadu_ps(p), a0);          a1 = _mm512_fmadd_ps(x, _mm512_loadu_ps(p + gs), a1);
            a2 = _mm512_fmadd_ps(x, _mm512_loadu_ps(p + 2 * gs), a2); a3 = _mm512_fmadd_ps(x, _mm512_loadu_ps(p + 3 * gs), a3);
            a4 = _mm512_fmadd_ps(x, _mm512_loadu_ps(p + 4 * gs), a4); a5 = _mm512_fmadd_ps(x, _mm512_loadu_ps(p + 5 * gs), a5);
            a6 = _mm512_fmadd_ps(x, _mm512_loadu_ps(p + 6 * gs), a6); a7 = _mm512_fmadd_ps(x, _mm512_loadu_ps(p + 7 * gs), a7);
        }
        _mm512_storeu_ps(out + (size_t)g * 16, a0);       _mm512_storeu_ps(out + (size_t)g * 16 + 16, a1);
        _mm512_storeu_ps(out + (size_t)g * 16 + 32, a2);  _mm512_storeu_ps(out + (size_t)g * 16 + 48, a3);
        _mm512_storeu_ps(out + (size_t)g * 16 + 64, a4);  _mm512_storeu_ps(out + (size_t)g * 16 + 80, a5);
        _mm512_storeu_ps(out + (size_t)g * 16 + 96, a6);  _mm512_storeu_ps(out + (size_t)g * 16 + 112, a7);
    }
    for (; g < ngroups; ++g) {
        const f32 *c = ct + (size_t)g * gs;
        __m512 a0 = _mm512_setzero_ps(), a1 = a0;
        for (d = 0; d + 2 <= dim; d += 2) {
            a0 = _mm512_fmadd_ps(_mm512_set1_ps(v[d]), _mm512_loadu_ps(c + (size_t)d * 16), a0);
            a1 = _mm512_fmadd_ps(_mm512_set1_ps(v[d + 1]), _mm512_loadu_ps(c + (size_t)d * 16 + 16), a1);
        }
        if (d < dim) a0 = _mm512_fmadd_ps(_mm512_set1_ps(v[d]), _mm512_loadu_ps(c + (size_t)d * 16), a0);
        _mm512_storeu_ps(out + (size_t)g * 16, _mm512_add_ps(a0, a1));
    }
#elif defined(__AVX2__) && defined(__FMA__)
    for (; g + 2 <= ngroups; g += 2) {
        const f32 *c = ct + (size_t)g * gs;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        for (d = 0; d < dim; ++d) {
            __m256 x = _mm256_set1_ps(v[d]);
            const f32 *p = c + (size_t)d * 16;
            a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(p), a0);      a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(p + 8), a1);
            a2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(p + gs), a2); a3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(p + gs + 8), a3);
        }
        _mm256_storeu_ps(out + (size_t)g * 16, a0); _mm256_storeu_ps(out + (size_t)g * 16 + 8, a1);
        _mm256_storeu_ps(out + (size_t)g * 16 + 16, a2); _mm256_storeu_ps(out + (size_t)g * 16 + 24, a3);
    }
    for (; g < ngroups; ++g) {
        const f32 *c = ct + (size_t)g * gs;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0;
        for (d = 0; d < dim; ++d) {
            __m256 x = _mm256_set1_ps(v[d]);
            a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c + (size_t)d * 16), a0); a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(c + (size_t)d * 16 + 8), a1);
        }
        _mm256_storeu_ps(out + (size_t)g * 16, a0); _mm256_storeu_ps(out + (size_t)g * 16 + 8, a1);
    }
#else
    for (; g < ngroups; ++g) {
        i32 l;
        for (l = 0; l < 16; ++l) out[(size_t)g * 16 + (size_t)l] = 0.0f;
        for (d = 0; d < dim; ++d) {
            f32 x = v[d];
            const f32 *p = ct + (size_t)g * gs + (size_t)d * 16;
            for (l = 0; l < 16; ++l) out[(size_t)g * 16 + (size_t)l] += x * p[l];
        }
    }
#endif
}

/* the same for 4 rows at once: every centroid load feeds 4 FMAs, so the kernel runs on the FMA units instead of
 * the cache bandwidth (16 accumulators: 4 rows x 4 groups). out + r * stride gets row r's scores. */
static void tq_score4_(const f32 *const *v, const f32 *ct, i32 ngroups, i32 dim, f32 *out, size_t stride)
{
#if defined(__AVX512F__)
    i32 g = 0, d;
    size_t gs = (size_t)dim * 16;
    for (; g + 4 <= ngroups; g += 4) {
        const f32 *c = ct + (size_t)g * gs;
        __m512 a00 = _mm512_setzero_ps(), a01 = a00, a02 = a00, a03 = a00, a10 = a00, a11 = a00, a12 = a00, a13 = a00;
        __m512 a20 = a00, a21 = a00, a22 = a00, a23 = a00, a30 = a00, a31 = a00, a32 = a00, a33 = a00;
        for (d = 0; d < dim; ++d) {
            const f32 *p = c + (size_t)d * 16;
            __m512 c0 = _mm512_loadu_ps(p), c1 = _mm512_loadu_ps(p + gs), c2 = _mm512_loadu_ps(p + 2 * gs), c3 = _mm512_loadu_ps(p + 3 * gs);
            __m512 x = _mm512_set1_ps(v[0][d]);
            a00 = _mm512_fmadd_ps(x, c0, a00); a01 = _mm512_fmadd_ps(x, c1, a01); a02 = _mm512_fmadd_ps(x, c2, a02); a03 = _mm512_fmadd_ps(x, c3, a03);
            x = _mm512_set1_ps(v[1][d]);
            a10 = _mm512_fmadd_ps(x, c0, a10); a11 = _mm512_fmadd_ps(x, c1, a11); a12 = _mm512_fmadd_ps(x, c2, a12); a13 = _mm512_fmadd_ps(x, c3, a13);
            x = _mm512_set1_ps(v[2][d]);
            a20 = _mm512_fmadd_ps(x, c0, a20); a21 = _mm512_fmadd_ps(x, c1, a21); a22 = _mm512_fmadd_ps(x, c2, a22); a23 = _mm512_fmadd_ps(x, c3, a23);
            x = _mm512_set1_ps(v[3][d]);
            a30 = _mm512_fmadd_ps(x, c0, a30); a31 = _mm512_fmadd_ps(x, c1, a31); a32 = _mm512_fmadd_ps(x, c2, a32); a33 = _mm512_fmadd_ps(x, c3, a33);
        }
        { f32 *o = out + (size_t)g * 16;
          _mm512_storeu_ps(o, a00); _mm512_storeu_ps(o + 16, a01); _mm512_storeu_ps(o + 32, a02); _mm512_storeu_ps(o + 48, a03);
          o += stride;
          _mm512_storeu_ps(o, a10); _mm512_storeu_ps(o + 16, a11); _mm512_storeu_ps(o + 32, a12); _mm512_storeu_ps(o + 48, a13);
          o += stride;
          _mm512_storeu_ps(o, a20); _mm512_storeu_ps(o + 16, a21); _mm512_storeu_ps(o + 32, a22); _mm512_storeu_ps(o + 48, a23);
          o += stride;
          _mm512_storeu_ps(o, a30); _mm512_storeu_ps(o + 16, a31); _mm512_storeu_ps(o + 32, a32); _mm512_storeu_ps(o + 48, a33); }
    }
    for (; g < ngroups; ++g) {
        const f32 *c = ct + (size_t)g * gs;
        __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        for (d = 0; d < dim; ++d) {
            __m512 c0 = _mm512_loadu_ps(c + (size_t)d * 16);
            a0 = _mm512_fmadd_ps(_mm512_set1_ps(v[0][d]), c0, a0); a1 = _mm512_fmadd_ps(_mm512_set1_ps(v[1][d]), c0, a1);
            a2 = _mm512_fmadd_ps(_mm512_set1_ps(v[2][d]), c0, a2); a3 = _mm512_fmadd_ps(_mm512_set1_ps(v[3][d]), c0, a3);
        }
        _mm512_storeu_ps(out + (size_t)g * 16, a0); _mm512_storeu_ps(out + stride + (size_t)g * 16, a1);
        _mm512_storeu_ps(out + 2 * stride + (size_t)g * 16, a2); _mm512_storeu_ps(out + 3 * stride + (size_t)g * 16, a3);
    }
#else
    i32 r;
    for (r = 0; r < 4; ++r) tq_score_(v[r], ct, ngroups, dim, out + (size_t)r * stride);
#endif
}

static i32 tq_argmax_(const f32 *sc, i32 n)
{
    i32 i, best = 0;
    f32 bs = sc[0];
    for (i = 1; i < n; ++i) if (sc[i] > bs) { bs = sc[i]; best = i; }
    return best;
}

/* nearest centroid of m (<= 4) rows; sc: scratch of 4 * 16 * ceil(nc/16) floats */
static void tq_nearest4_(const f32 *const *v, i32 m, const f32 *ct, i32 nc, i32 dim, f32 *sc, i32 *out)
{
    const f32 *vv[4];
    i32 ng = (nc + 15) / 16, r;
    size_t st = (size_t)ng * 16;
    for (r = 0; r < 4; ++r) vv[r] = v[r < m ? r : m - 1];               /* pad a short group with its last row */
    tq_score4_(vv, ct, ng, dim, sc, st);
    for (r = 0; r < m; ++r) out[r] = tq_argmax_(sc + (size_t)r * st, nc);
}

/* index of the closest of nc centroids (ct transposed); sc: scratch of 16 * ceil(nc/16) floats */
static VS_UNUSED i32 tq_nearest_(const f32 *v, const f32 *ct, i32 nc, i32 dim, f32 *sc)
{
    i32 i, best = 0;
    f32 bs;
    tq_score_(v, ct, (nc + 15) / 16, dim, sc);
    bs = sc[0];
    for (i = 1; i < nc; ++i) if (sc[i] > bs) { bs = sc[i]; best = i; }
    return best;
}

/* ---- a tiny parallel for: fn(ctx, a, b, tid) over [0, n) split in nth ranges ---- */
typedef void (*TqFn)(void *ctx, i64 a, i64 b, i32 tid);
typedef struct { TqFn fn; void *ctx; i64 a, b; i32 tid; } TqPar;
static void *tq_par_job_(void *p) { TqPar *j = (TqPar *)p; j->fn(j->ctx, j->a, j->b, j->tid); return (void *)0; }
static void tq_par_(TqFn fn, void *ctx, i64 n, i32 nth)
{
    TqPar j[64]; pthread_t th[64];
    i32 t;
    if (nth < 1) nth = 1;
    if (nth > 64) nth = 64;
    if (n < 4096) nth = 1;
    for (t = 0; t < nth; ++t) { j[t].fn = fn; j[t].ctx = ctx; j[t].a = n * t / nth; j[t].b = n * (t + 1) / nth; j[t].tid = t; }
    for (t = 1; t < nth; ++t) pthread_create(&th[t], (const pthread_attr_t *)0, tq_par_job_, &j[t]);
    tq_par_job_(&j[0]);
    for (t = 1; t < nth; ++t) pthread_join(th[t], (void **)0);
}

/* ---- spherical k-means on a contiguous sample ---- */
typedef struct { const f32 *x; const f32 *ct; i32 dim, nc; i32 *asg; } TqKm;

static void tq_km_asg_(void *p, i64 a, i64 b, i32 tid)
{
    TqKm *k = (TqKm *)p;
    f32 *sc = (f32 *)malloc((size_t)((k->nc + 15) / 16) * 64 * sizeof(f32));
    i64 i;
    (void)tid;
    for (i = a; i < b; i += 4) {
        const f32 *v[4];
        i32 m = b - i < 4 ? (i32)(b - i) : 4, r;
        for (r = 0; r < m; ++r) v[r] = k->x + (size_t)(i + r) * k->dim;
        tq_nearest4_(v, m, k->ct, k->nc, k->dim, sc, k->asg + i);
    }
    free(sc);
}

/* nc unit centroids of the m rows of x into cent[nc x dim]; asg (optional, [m]) gets the final assignment */
static void tq_kmeans_(const f32 *x, i32 m, i32 dim, i32 nc, u32 *rng, f32 *cent, i32 *asg_out, i32 nth)
{
    f32 *sum = (f32 *)malloc((size_t)nc * (size_t)dim * sizeof(f32));
    i32 *asg = asg_out ? asg_out : (i32 *)malloc((size_t)m * sizeof(i32)), *cnt = (i32 *)calloc((size_t)nc, sizeof(i32));
    i32 i, c, d, it;
    TqKm k;
    for (c = 0; c < nc; ++c) memcpy(cent + (size_t)c * dim, x + (size_t)((i64)m * c / nc + (tq_rng_(rng) % (u32)(m / nc > 0 ? m / nc : 1))) * dim, (size_t)dim * sizeof(f32));
    k.x = x; k.dim = dim; k.nc = nc; k.asg = asg;
    for (it = 0; it < 8; ++it) {
        k.ct = tq_transpose_(cent, nc, dim);
        tq_par_(tq_km_asg_, &k, m, nth);
        free((void *)k.ct);
        memset(sum, 0, (size_t)nc * (size_t)dim * sizeof(f32)); memset(cnt, 0, (size_t)nc * sizeof(i32));
        for (i = 0; i < m; ++i) { f32 *s = sum + (size_t)asg[i] * dim; const f32 *v = x + (size_t)i * dim; for (d = 0; d < dim; ++d) s[d] += v[d]; ++cnt[asg[i]]; }
        for (c = 0; c < nc; ++c) {
            f32 nrm = 0.0f, *s = sum + (size_t)c * dim;
            if (cnt[c] == 0) memcpy(s, x + (size_t)(tq_rng_(rng) % (u32)m) * dim, (size_t)dim * sizeof(f32));   /* empty cluster: reseed */
            for (d = 0; d < dim; ++d) nrm += s[d] * s[d];
            nrm = nrm > 0.0f ? 1.0f / (f32)sqrt((f64)nrm) : 1.0f;
            for (d = 0; d < dim; ++d) cent[(size_t)c * dim + d] = s[d] * nrm;
        }
    }
    if (asg_out) { k.ct = tq_transpose_(cent, nc, dim); tq_par_(tq_km_asg_, &k, m, nth); free((void *)k.ct); }
    free(sum); free(cnt);
    if (!asg_out) free(asg);
}

/* The clustering of a whole build. Flat (one level) up to TQ_FLAT_MAX clusters; above, k1 ~ sqrt(nc) top cells
 * each split into sub-clusters in proportion to its share of the sample, and a row is assigned to the nearest
 * sub-cluster of its nearest top cell (k1 + nc/k1 dot products instead of nc). */
typedef struct {
    i32 nc, dim;
    f32 *cent;              /* [nc x dim] final centroids */
    i32 k1;                 /* 0 = flat */
    f32 *top_ct;            /* [k1] top cells, transposed */
    i32 *sub0, *subn;       /* cell j owns centroids sub0[j] .. sub0[j] + subn[j] */
    f32 **sub_ct;           /* [k1] each cell's centroids, transposed */
    f32 *flat_ct;           /* flat: all centroids transposed */
} TqClus;

static void tq_clus_free_(TqClus *cl)
{
    i32 j;
    free(cl->top_ct); free(cl->flat_ct); free(cl->sub0); free(cl->subn);
    if (cl->sub_ct) for (j = 0; j < cl->k1; ++j) free(cl->sub_ct[j]);
    free(cl->sub_ct);
}

static void tq_clus_train_(TqClus *cl, const TqCol *rows, i32 n, i32 dim, i32 nc, u32 seed, i32 nth)
{
    size_t cap = TQ_SAMPLE_MAX / ((size_t)dim * sizeof(f32));
    i32 m, i;
    f32 *x;
    u32 r = seed ? seed : 99u;
    memset(cl, 0, sizeof(*cl));
    cl->dim = dim;
    m = nc * 64 > 40000 ? nc * 64 : 40000;
    if ((size_t)m > cap) m = (i32)cap;
    if (m > n) m = n;
    if (nc > m) nc = m;
    x = (f32 *)malloc((size_t)m * (size_t)dim * sizeof(f32));
    for (i = 0; i < m; ++i) {                                            /* an even spread of rows */
        f32 *dst = x + (size_t)i * dim;
        const f32 *src = tq_row_(rows, (i64)n * i / m, dst);
        if (src != dst) memcpy(dst, src, (size_t)dim * sizeof(f32));
    }
    if (nc <= TQ_FLAT_MAX) {
        cl->nc = nc;
        cl->cent = (f32 *)malloc((size_t)nc * (size_t)dim * sizeof(f32));
        tq_kmeans_(x, m, dim, nc, &r, cl->cent, (i32 *)0, nth);
        cl->flat_ct = tq_transpose_(cl->cent, nc, dim);
    } else {
        i32 k1 = (i32)sqrt((f64)nc), j, tot = 0, *asg = (i32 *)malloc((size_t)m * sizeof(i32)), *cnt, *ord, *at;
        f32 *top = (f32 *)malloc((size_t)k1 * (size_t)dim * sizeof(f32)), *buf;
        tq_kmeans_(x, m, dim, k1, &r, top, asg, nth);
        cnt = (i32 *)calloc((size_t)k1 + 1, sizeof(i32));
        for (i = 0; i < m; ++i) ++cnt[asg[i] + 1];
        for (j = 0; j < k1; ++j) cnt[j + 1] += cnt[j];                      /* cnt[j] = first member of cell j in ord */
        ord = (i32 *)malloc((size_t)m * sizeof(i32)); at = (i32 *)malloc((size_t)k1 * sizeof(i32));
        memcpy(at, cnt, (size_t)k1 * sizeof(i32));
        for (i = 0; i < m; ++i) ord[at[asg[i]]++] = i;
        cl->k1 = k1;
        cl->sub0 = (i32 *)malloc((size_t)k1 * sizeof(i32)); cl->subn = (i32 *)malloc((size_t)k1 * sizeof(i32));
        for (j = 0; j < k1; ++j) {                                          /* sub-clusters in proportion to the cell's share */
            i32 cn = cnt[j + 1] - cnt[j], kj = (i32)(((i64)nc * cn + m / 2) / m);
            if (kj < 1) kj = 1;
            if (kj > cn && cn > 0) kj = cn;
            cl->sub0[j] = tot; cl->subn[j] = kj; tot += kj;
        }
        cl->nc = tot;
        cl->cent = (f32 *)malloc((size_t)tot * (size_t)dim * sizeof(f32));
        cl->sub_ct = (f32 **)calloc((size_t)k1, sizeof(f32 *));
        { i32 big = 1; for (j = 0; j < k1; ++j) if (cnt[j + 1] - cnt[j] > big) big = cnt[j + 1] - cnt[j];
          buf = (f32 *)malloc((size_t)big * (size_t)dim * sizeof(f32)); }        /* the largest cell, not the whole sample */
        for (j = 0; j < k1; ++j) {
            i32 cn = cnt[j + 1] - cnt[j], z;
            f32 *dst = cl->cent + (size_t)cl->sub0[j] * dim;
            if (cn == 0) memcpy(dst, top + (size_t)j * dim, (size_t)dim * sizeof(f32));          /* an empty cell keeps its own centre */
            else {
                for (z = 0; z < cn; ++z) memcpy(buf + (size_t)z * dim, x + (size_t)ord[cnt[j] + z] * dim, (size_t)dim * sizeof(f32));
                if (cl->subn[j] == 1) memcpy(dst, top + (size_t)j * dim, (size_t)dim * sizeof(f32));
                else tq_kmeans_(buf, cn, dim, cl->subn[j], &r, dst, (i32 *)0, nth);
            }
            cl->sub_ct[j] = tq_transpose_(dst, cl->subn[j], dim);
        }
        cl->top_ct = tq_transpose_(top, k1, dim);
        free(buf); free(top); free(asg); free(cnt); free(ord); free(at);
    }
    free(x);
}

/* ---- the build ---- */
typedef struct { const TqCol *rows; const TqClus *cl; i32 *asg; i32 *cnt; i32 nc; } TqAsgJob;

#define TQ_ABATCH 1024
static void tq_asg_rows_(void *p, i64 a, i64 b, i32 tid)
{
    TqAsgJob *j = (TqAsgJob *)p;
    const TqClus *cl = j->cl;
    i32 wide = cl->nc > cl->k1 ? cl->nc : cl->k1, *top = (i32 *)malloc(TQ_ABATCH * sizeof(i32));
    i32 *ord = (i32 *)malloc(TQ_ABATCH * sizeof(i32)), *cst = (i32 *)malloc(((size_t)cl->k1 + 2) * sizeof(i32));
    f32 *sc = (f32 *)malloc(((size_t)wide + 16) * 4 * sizeof(f32)), *rb = (f32 *)malloc((size_t)cl->dim * 4 * sizeof(f32));
    i32 *cnt = j->cnt + (size_t)tid * j->nc;
    i64 x;
    for (x = a; x < b; x += TQ_ABATCH) {
        i32 m = b - x < TQ_ABATCH ? (i32)(b - x) : TQ_ABATCH, i, r, c;
        const f32 *v[4];
        /* stage 1: the flat centroids, or the top cells */
        for (i = 0; i < m; i += 4) {
            i32 g = m - i < 4 ? m - i : 4;
            for (r = 0; r < g; ++r) v[r] = tq_row_(j->rows, x + i + r, rb + (size_t)r * cl->dim);
            if (cl->k1) tq_nearest4_(v, g, cl->top_ct, cl->k1, cl->dim, sc, top + i);
            else tq_nearest4_(v, g, cl->flat_ct, cl->nc, cl->dim, sc, j->asg + x + i);
        }
        if (!cl->k1) { for (i = 0; i < m; ++i) ++cnt[j->asg[x + i]]; continue; }
        /* stage 2: rows grouped by top cell, so four rows share each cell's sub-centroids */
        memset(cst, 0, ((size_t)cl->k1 + 2) * sizeof(i32));
        for (i = 0; i < m; ++i) ++cst[top[i] + 2];
        for (c = 0; c < cl->k1; ++c) cst[c + 2] += cst[c + 1];
        for (i = 0; i < m; ++i) ord[cst[top[i] + 1]++] = i;           /* cst[c + 1] now ends cell c, cst[c] starts it */
        for (c = 0; c < cl->k1; ++c) {
            i32 s0 = cst[c], s1 = cst[c + 1], q;
            for (q = s0; q < s1; q += 4) {
                i32 g = s1 - q < 4 ? s1 - q : 4, best[4];
                for (r = 0; r < g; ++r) v[r] = tq_row_(j->rows, x + ord[q + r], rb + (size_t)r * cl->dim);
                if (cl->subn[c] > 1) tq_nearest4_(v, g, cl->sub_ct[c], cl->subn[c], cl->dim, sc, best);
                else for (r = 0; r < g; ++r) best[r] = 0;
                for (r = 0; r < g; ++r) { i32 asg = cl->sub0[c] + best[r]; j->asg[x + ord[q + r]] = asg; ++cnt[asg]; }
            }
        }
    }
    free(top); free(ord); free(cst); free(sc); free(rb);
}

typedef struct { const TqCol *rows; Tq *t; const i32 *slot; i32 n, nth; } TqEncJob;

/* thread tid encodes exactly the rows whose slot falls in its share of the blocks: no two threads ever write
 * the same byte (a byte holds the codes of two rows of one block) */
static void tq_enc_rows_(void *p, i64 a, i64 b, i32 tid)
{
    TqEncJob *j = (TqEncJob *)p;
    Tq *t = j->t;
    i64 lo = (i64)t->nblk * tid / j->nth * 32, hi = (i64)t->nblk * (tid + 1) / j->nth * 32, i;
    f32 *rot = (f32 *)malloc((size_t)t->dpad * sizeof(f32)), *tmp = (f32 *)malloc((size_t)t->dpad * sizeof(f32));
    f32 *rb = (f32 *)malloc((size_t)t->dim * sizeof(f32));
    u8 *cd = (u8 *)malloc((size_t)t->dpad + 16);
    i32 d, vd = t->vd;
    (void)a; (void)b;
    for (i = j->slot ? 0 : lo; i < (j->slot ? j->n : (hi < j->n ? hi : j->n)); ++i) {   /* flat: slot == row, walk only our range */
        i64 s = j->slot ? j->slot[i] : i;
        i32 r;
        u8 *blk;
        if (s < lo || s >= hi) continue;
        blk = t->codes + (size_t)(s / 32) * (size_t)vd * 16;
        r = (i32)(s % 32);
        tq_rot_(t, tq_row_(j->rows, i, rb), rot, tmp);
        tq_encode_row_(t, rot, cd);
        if (r < 16) for (d = 0; d < vd; ++d) { u8 *q = blk + (size_t)d * 16 + r; *q = (u8)((*q & 0xf0) | cd[d]); }
        else for (d = 0; d < vd; ++d) { u8 *q = blk + (size_t)d * 16 + (r - 16); *q = (u8)((*q & 0x0f) | (cd[d] << 4)); }
    }
    free(rot); free(tmp); free(cd); free(rb);
}

static i32 tq_default_probe_(i32 nc)
{
    i32 a = nc * 16 / 100, b = (i32)(4.0 * sqrt((f64)nc) + 0.5), np = a < b ? a : b;
    if (np < 4) np = nc < 4 ? nc : 4;
    return np;
}

/* nc <= 1: flat scan of every row. nc > 1: IVF with about nc clusters (nprobe defaults to min(16%, 4 sqrt nc)). */
static f64 tq_now_(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (f64)ts.tv_sec + 1e-9 * (f64)ts.tv_nsec; }

Tq *tq_build(const TqCol *rows, i32 n, i32 dim, u32 seed, i32 nc, i32 nth, i32 bits)
{
    Tq *t = (Tq *)calloc(1, sizeof(Tq));
    f64 t0 = tq_now_();
    i32 verbose = getenv("TQ_VERBOSE") != (char *)0;
    i32 i, nslots, *slot = (i32 *)0;
    u32 r = seed ? seed : 0x7154u;
    TqEncJob ej;
    if (nth < 1) nth = 1;
    t->dim = dim; t->dpad = dim + (dim & 1); t->n = n;
    t->bits = bits == 2 ? 2 : 4; t->vd = tq_vd_(t->dpad, t->bits);
    t->s1 = (f32 *)malloc((size_t)t->dpad * sizeof(f32)); t->s2 = (f32 *)malloc((size_t)t->dpad * sizeof(f32));
    t->perm = (i32 *)malloc((size_t)t->dpad * sizeof(i32));
    for (i = 0; i < t->dpad; ++i) { t->s1[i] = (tq_rng_(&r) & 1u) ? 1.0f : -1.0f; t->s2[i] = (tq_rng_(&r) & 1u) ? 1.0f : -1.0f; t->perm[i] = i; }
    for (i = t->dpad - 1; i > 0; --i) { i32 k = (i32)(tq_rng_(&r) % (u32)(i + 1)), x = t->perm[i]; t->perm[i] = t->perm[k]; t->perm[k] = x; }

    if (nc > 1 && n >= nc * 8) {                                 /* group the rows by cluster, each cluster padded to whole blocks */
        TqClus cl;
        TqAsgJob aj;
        i32 c, blk = 0, *pos, tt;
        tq_clus_train_(&cl, rows, n, dim, nc, seed, nth);
        if (verbose) fprintf(stderr, "tq_build: train %d clusters (k1=%d) %.2fs\n", cl.nc, cl.k1, tq_now_() - t0);
        nc = cl.nc;
        t->nc = nc;
        t->cent = cl.cent; cl.cent = (f32 *)0;
        slot = (i32 *)malloc((size_t)n * sizeof(i32));           /* first the cluster of each row, then its slot */
        aj.rows = rows; aj.cl = &cl; aj.asg = slot; aj.nc = nc;
        aj.cnt = (i32 *)calloc((size_t)nth * (size_t)nc, sizeof(i32));
        tq_par_(tq_asg_rows_, &aj, n, nth);
        if (verbose) fprintf(stderr, "tq_build: assign %d rows %.2fs\n", n, tq_now_() - t0);
        tq_clus_free_(&cl);
        for (tt = 1; tt < nth; ++tt) for (c = 0; c < nc; ++c) aj.cnt[c] += aj.cnt[(size_t)tt * nc + c];
        t->cstart = (i32 *)malloc((size_t)(nc + 1) * sizeof(i32));
        for (c = 0; c < nc; ++c) { t->cstart[c] = blk; blk += (aj.cnt[c] + 31) / 32; }
        t->cstart[nc] = blk;
        t->nblk = blk; nslots = blk * 32;
        t->slot_row = (i32 *)malloc((size_t)nslots * sizeof(i32));
        for (i = 0; i < nslots; ++i) t->slot_row[i] = -1;
        pos = aj.cnt;                                            /* reuse: next free slot of each cluster */
        for (c = 0; c < nc; ++c) pos[c] = t->cstart[c] * 32;
        for (i = 0; i < n; ++i) { i32 s = pos[slot[i]]++; t->slot_row[s] = i; slot[i] = s; }
        free(aj.cnt);
        t->nprobe = tq_default_probe_(nc);
        t->ct = tq_transpose_(t->cent, nc, dim);
    } else {
        t->nblk = (n + 31) / 32; nslots = t->nblk * 32;
    }
    t->codes = (u8 *)calloc((size_t)(t->nblk ? t->nblk : 1) * (size_t)t->vd * 16, 1);
    ej.rows = rows; ej.t = t; ej.slot = slot; ej.n = n; ej.nth = n >= 4096 ? nth : 1;
    if (ej.nth > 64) ej.nth = 64;
    tq_par_(tq_enc_rows_, &ej, (i64)ej.nth * 4096, ej.nth);     /* every thread walks all rows, keeps its own blocks */
    free(slot);
    (void)nslots;
    if (verbose) fprintf(stderr, "tq_build: encode done %.2fs\n", tq_now_() - t0);
    return t;
}

/* after a load: the transposed centroids are derived, not saved */
static void tq_prepare_(Tq *t)
{
    if (t->nc > 1 && !t->ct) t->ct = tq_transpose_(t->cent, t->nc, t->dim);
}

void tq_free(Tq *t)
{
    if (!t) return;
    if (!t->mapped) { free(t->s1); free(t->s2); free(t->perm); free(t->codes); free(t->cent); free(t->cstart); free(t->slot_row); }
    free(t->ct);
    free(t);
}

/* bytes of RAM the index holds (0 for the parts that live in a file mapping) */
static i64 tq_ram_(const Tq *t)
{
    i64 b = 0;
    if (!t) return 0;
    if (t->nc > 1) b += (i64)((t->nc + 15) / 16) * 16 * t->dim * 4;
    if (!t->mapped) {
        b += (i64)t->nblk * t->vd * 16 + (i64)t->dpad * 12;
        if (t->slot_row) b += (i64)t->nblk * 32 * 4;
        if (t->nc > 1) b += (i64)t->nc * t->dim * 4 + (i64)(t->nc + 1) * 4;
    }
    return b;
}

/* ---- scanning ---- */

/* Scores of the 32 vectors of one block into out[32] (u16, bigger = closer). Returns a bit mask of the rows
 * scoring above thr, so the caller only touches those (most rows of a scan never get near the top R).
 * AVX-512BW: 4 dimensions per 64-byte step; AVX2: 2 per step; the lane sums and the threshold test stay in
 * registers. */
static u32 tq_block_(const u8 *blk, const u8 *lut, i32 dpad, u16 thr, u16 *out)
{
#if defined(__AVX2__)
    __m256i a0, a1, b0, b1;
    const __m256i m0f = _mm256_set1_epi8(0x0f), m00ff = _mm256_set1_epi16(0x00ff);
    __m128i x0, x1, y0, y1, r0q, r1q, r2q, r3q;
    i32 d = 0;
#if defined(__AVX512BW__)
    {
        __m512i A0 = _mm512_setzero_si512(), A1 = A0, B0 = A0, B1 = A0;
        const __m512i M0F = _mm512_set1_epi8(0x0f), M00FF = _mm512_set1_epi16(0x00ff);
        for (; d + 4 <= dpad; d += 4) {
            __m512i codes = _mm512_loadu_si512((const void *)(blk + (size_t)d * 16));
            __m512i lv = _mm512_loadu_si512((const void *)(lut + (size_t)d * 16));
            __m512i lo = _mm512_and_si512(codes, M0F);
            __m512i hi = _mm512_and_si512(_mm512_srli_epi16(codes, 4), M0F);
            __m512i r0 = _mm512_shuffle_epi8(lv, lo), r1 = _mm512_shuffle_epi8(lv, hi);
            A0 = _mm512_add_epi16(A0, _mm512_and_si512(r0, M00FF)); A1 = _mm512_add_epi16(A1, _mm512_srli_epi16(r0, 8));
            B0 = _mm512_add_epi16(B0, _mm512_and_si512(r1, M00FF)); B1 = _mm512_add_epi16(B1, _mm512_srli_epi16(r1, 8));
        }
        a0 = _mm256_add_epi16(_mm512_castsi512_si256(A0), _mm512_extracti64x4_epi64(A0, 1));
        a1 = _mm256_add_epi16(_mm512_castsi512_si256(A1), _mm512_extracti64x4_epi64(A1, 1));
        b0 = _mm256_add_epi16(_mm512_castsi512_si256(B0), _mm512_extracti64x4_epi64(B0, 1));
        b1 = _mm256_add_epi16(_mm512_castsi512_si256(B1), _mm512_extracti64x4_epi64(B1, 1));
    }
#else
    a0 = _mm256_setzero_si256(); a1 = a0; b0 = a0; b1 = a0;
#endif
    for (; d < dpad; d += 2) {
        __m256i codes = _mm256_loadu_si256((const __m256i *)(blk + (size_t)d * 16));
        __m256i lv = _mm256_loadu_si256((const __m256i *)(lut + (size_t)d * 16));
        __m256i lo = _mm256_and_si256(codes, m0f);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(codes, 4), m0f);
        __m256i r0 = _mm256_shuffle_epi8(lv, lo), r1 = _mm256_shuffle_epi8(lv, hi);
        a0 = _mm256_add_epi16(a0, _mm256_and_si256(r0, m00ff)); a1 = _mm256_add_epi16(a1, _mm256_srli_epi16(r0, 8));
        b0 = _mm256_add_epi16(b0, _mm256_and_si256(r1, m00ff)); b1 = _mm256_add_epi16(b1, _mm256_srli_epi16(r1, 8));
    }
    /* fold the two dimension lanes; a0/a1 hold the even/odd rows of 0-15, b0/b1 those of 16-31 */
    x0 = _mm_add_epi16(_mm256_castsi256_si128(a0), _mm256_extracti128_si256(a0, 1));
    x1 = _mm_add_epi16(_mm256_castsi256_si128(a1), _mm256_extracti128_si256(a1, 1));
    y0 = _mm_add_epi16(_mm256_castsi256_si128(b0), _mm256_extracti128_si256(b0, 1));
    y1 = _mm_add_epi16(_mm256_castsi256_si128(b1), _mm256_extracti128_si256(b1, 1));
    r0q = _mm_unpacklo_epi16(x0, x1); r1q = _mm_unpackhi_epi16(x0, x1);     /* rows 0-7, 8-15 */
    r2q = _mm_unpacklo_epi16(y0, y1); r3q = _mm_unpackhi_epi16(y0, y1);     /* rows 16-23, 24-31 */
    _mm_storeu_si128((__m128i *)(void *)out, r0q); _mm_storeu_si128((__m128i *)(void *)(out + 8), r1q);
    _mm_storeu_si128((__m128i *)(void *)(out + 16), r2q); _mm_storeu_si128((__m128i *)(void *)(out + 24), r3q);
    {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
        __m128i t = _mm_set1_epi16((short)thr);
        return (u32)_mm_cmpgt_epu16_mask(r0q, t) | ((u32)_mm_cmpgt_epu16_mask(r1q, t) << 8)
             | ((u32)_mm_cmpgt_epu16_mask(r2q, t) << 16) | ((u32)_mm_cmpgt_epu16_mask(r3q, t) << 24);
#else
        /* unsigned v > thr: flip the sign bits, compare signed, pack the 0/-1 lanes to bytes */
        __m128i bias = _mm_set1_epi16((short)0x8000), t = _mm_set1_epi16((short)(thr ^ 0x8000));
        u32 m0 = (u32)_mm_movemask_epi8(_mm_packs_epi16(_mm_cmpgt_epi16(_mm_xor_si128(r0q, bias), t),
                                                        _mm_cmpgt_epi16(_mm_xor_si128(r1q, bias), t)));
        u32 m1 = (u32)_mm_movemask_epi8(_mm_packs_epi16(_mm_cmpgt_epi16(_mm_xor_si128(r2q, bias), t),
                                                        _mm_cmpgt_epi16(_mm_xor_si128(r3q, bias), t)));
        return m0 | (m1 << 16);
#endif
    }
#else
    i32 d, j;
    u32 m = 0;
    for (j = 0; j < 32; ++j) out[j] = 0;
    for (d = 0; d < dpad; ++d)
        for (j = 0; j < 32; ++j) {
            u8 byte = blk[(size_t)d * 16 + (j & 15)];
            u8 c = j < 16 ? (u8)(byte & 15) : (u8)(byte >> 4);
            out[j] = (u16)(out[j] + lut[(size_t)d * 16 + c]);
        }
    for (j = 0; j < 32; ++j) if (out[j] > thr) m |= 1u << j;
    return m;
#endif
}

/* index of the lowest set bit of a non-zero mask */
static i32 tq_ctz_(u32 m)
{
#if defined(__GNUC__)
    return __builtin_ctz(m);
#else
    i32 k = 0;
    while (!(m & 1u)) { m >>= 1; ++k; }
    return k;
#endif
}

typedef struct { u64 *h; i32 n, cap; } TqHeap;   /* min-heap of (score << 32 | row) */

static void tq_heap_push_(TqHeap *h, u64 v)
{
    i32 i;
    if (h->n < h->cap) {
        i = h->n++;
        while (i > 0 && h->h[(i - 1) / 2] > v) { h->h[i] = h->h[(i - 1) / 2]; i = (i - 1) / 2; }
        h->h[i] = v;
    } else if (v > h->h[0]) {
        i = 0;
        for (;;) {
            i32 c = 2 * i + 1;
            if (c >= h->n) break;
            if (c + 1 < h->n && h->h[c + 1] < h->h[c]) ++c;
            if (h->h[c] >= v) break;
            h->h[i] = h->h[c]; i = c;
        }
        h->h[i] = v;
    }
}

typedef struct {
    const Tq *t; const u8 *lut;
    i32 b0, b1, R;
    const i32 *blks;                 /* IVF: the blocks to scan (b0..b1 index into this list); NULL = blocks b0..b1 */
    const u64 *dead; const TqCol *meta; i64 all, any; i32 filtered;
    TqHeap heap;
} TqJob;

static void *tq_job_(void *arg)
{
    TqJob *j = (TqJob *)arg;
    const Tq *t = j->t;
    i32 b, k;
    u16 sc[32];
    for (b = j->b0; b < j->b1; ++b) {
        i32 blk = j->blks ? j->blks[b] : b, base = blk * 32;
        i32 full = j->heap.n == j->heap.cap;
        u32 m = tq_block_(t->codes + (size_t)blk * (size_t)t->vd * 16, j->lut, t->vd,
                          full ? (u16)(j->heap.h[0] >> 32) : (u16)0, sc);
        if (!full) m = 0xffffffffu;                              /* still filling the heap: every row counts */
        while (m) {
            i32 row;
            u64 v;
            k = tq_ctz_(m); m &= m - 1u;
            row = t->slot_row ? t->slot_row[base + k] : base + k;
            if (row < 0 || row >= t->n) continue;                /* padding */
            if (j->dead && ((j->dead[(u32)row >> 6] >> (row & 63)) & 1u)) continue;
            if (j->filtered) {
                i64 tag = *(const i64 *)(const void *)TQ_AT(j->meta, row);
                if ((tag & j->all) != j->all) continue;
                if (j->any && !(tag & j->any)) continue;
            }
            v = ((u64)sc[k] << 32) | (u64)(u32)row;
            tq_heap_push_(&j->heap, v);
        }
    }
    return (void *)0;
}

typedef struct { f32 s; i32 c; } TqCs;

static int tq_cs_desc_(const void *a, const void *b)
{
    f32 x = ((const TqCs *)a)->s, y = ((const TqCs *)b)->s;
    return x < y ? 1 : (x > y ? -1 : 0);
}

/* moves the k largest entries (by s) to a[0..k), in no particular order (Hoare quickselect, O(n)) */
static void tq_select_(TqCs *a, i32 n, i32 k)
{
    i32 lo = 0, hi = n - 1;
    while (lo < hi) {
        f32 piv = a[lo + (hi - lo) / 2].s;
        i32 i = lo, j = hi;
        while (i <= j) {
            while (a[i].s > piv) ++i;
            while (a[j].s < piv) --j;
            if (i <= j) { TqCs x = a[i]; a[i] = a[j]; a[j] = x; ++i; --j; }
        }
        if (k - 1 <= j) hi = j;
        else if (k - 1 >= i) lo = i;
        else break;
    }
}

static int tq_u64_desc_(const void *a, const void *b)
{
    u64 x = *(const u64 *)a, y = *(const u64 *)b;
    return x < y ? 1 : (x > y ? -1 : 0);
}

/* The R best rows (by the quantised inner product) that are alive and pass the meta filter; returns how
 * many were found, rows[] best first. dead (a bitset) / meta may be NULL. nthreads > 1 splits the blocks.
 * probe: IVF clusters to scan (0 = the index default, >= nc = all). */
i32 tq_search(const Tq *t, const f32 *q, i32 R, i32 *rows, const u64 *dead, const TqCol *meta, i64 all, i64 any, i32 filtered, i32 nthreads, i32 probe)
{
    i32 *blks = (i32 *)0, nblk_scan;
    f32 *rot = (f32 *)malloc((size_t)t->dpad * sizeof(f32)), *tmp = (f32 *)malloc((size_t)t->dpad * sizeof(f32));
    f32 *lf = (f32 *)malloc((size_t)t->vd * 16 * sizeof(f32)), maxr = 0.0f, scale;
    u8 *lut = (u8 *)malloc((size_t)t->vd * 16 + 32);
    TqJob jobs[16];
    pthread_t th[16];
    i32 d, c, i, nj, per, cnt = 0, lutmax;
    u64 *all_h;

    if (t->n == 0 || R <= 0) { free(rot); free(tmp); free(lf); free(lut); return 0; }
    tq_rot_(t, q, rot, tmp);
    for (d = 0; d < t->vd; ++d) {                                /* one 16-entry table per code column */
        f32 mn = 1e30f, mx = -1e30f;
        if (t->bits == 2) {                                      /* column d = coordinates 2d, 2d+1, 2 bits each */
            f32 a = 2 * d < t->dpad ? rot[2 * d] : 0.0f, b = 2 * d + 1 < t->dpad ? rot[2 * d + 1] : 0.0f;
            for (c = 0; c < 16; ++c) { f32 v = a * TQ_CENT2[c & 3] + b * TQ_CENT2[c >> 2]; lf[d * 16 + c] = v; if (v < mn) mn = v; if (v > mx) mx = v; }
        } else
            for (c = 0; c < 16; ++c) { f32 v = rot[d] * TQ_CENT[c]; lf[d * 16 + c] = v; if (v < mn) mn = v; if (v > mx) mx = v; }
        for (c = 0; c < 16; ++c) lf[d * 16 + c] -= mn;
        if (mx - mn > maxr) maxr = mx - mn;
    }
    lutmax = 65535 / t->vd; if (lutmax > 255) lutmax = 255;
    scale = maxr > 0.0f ? (f32)lutmax / maxr : 1.0f;
    for (i = 0; i < t->vd * 16; ++i) { i32 v = (i32)(lf[i] * scale + 0.5f); lut[i] = (u8)(v > lutmax ? lutmax : v); }

    if (t->nc > 1) {                                             /* IVF: the blocks of the nprobe clusters closest to the query */
        i32 np = probe > 0 ? probe : t->nprobe, nb = 0, ng = (t->nc + 15) / 16;
        TqCs *cs = (TqCs *)malloc((size_t)ng * 16 * sizeof(TqCs));
        f32 *cd = (f32 *)malloc((size_t)ng * 16 * sizeof(f32));
        if (np > t->nc) np = t->nc;
        tq_score_(q, t->ct, ng, t->dim, cd);                     /* every centroid, 16 per instruction */
        for (c = 0; c < t->nc; ++c) { cs[c].s = cd[c]; cs[c].c = c; }
        free(cd);
        tq_select_(cs, t->nc, np);                               /* the np best centroids, O(nc) */
        qsort(cs, (size_t)np, sizeof(TqCs), tq_cs_desc_);        /* closest first: the heap threshold rises early */
        for (i = 0; i < np; ++i) nb += t->cstart[cs[i].c + 1] - t->cstart[cs[i].c];
        blks = (i32 *)malloc((size_t)(nb ? nb : 1) * sizeof(i32));
        for (nb = 0, i = 0; i < np; ++i) { i32 b; for (b = t->cstart[cs[i].c]; b < t->cstart[cs[i].c + 1]; ++b) blks[nb++] = b; }
        nblk_scan = nb;
        free(cs);
    } else nblk_scan = t->nblk;
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 16) nthreads = 16;
    if (nblk_scan < 64) nthreads = 1;
    nj = nthreads; per = (nblk_scan + nj - 1) / nj;
    for (i = 0; i < nj; ++i) {
        jobs[i].t = t; jobs[i].lut = lut; jobs[i].blks = blks; jobs[i].b0 = i * per; jobs[i].b1 = (i + 1) * per < nblk_scan ? (i + 1) * per : nblk_scan;
        if (jobs[i].b0 > jobs[i].b1) jobs[i].b0 = jobs[i].b1;
        jobs[i].R = R; jobs[i].dead = dead; jobs[i].meta = meta; jobs[i].all = all; jobs[i].any = any; jobs[i].filtered = filtered && meta;
        jobs[i].heap.h = (u64 *)malloc((size_t)R * sizeof(u64)); jobs[i].heap.n = 0; jobs[i].heap.cap = R;
    }
    for (i = 1; i < nj; ++i) pthread_create(&th[i], (const pthread_attr_t *)0, tq_job_, &jobs[i]);
    tq_job_(&jobs[0]);
    for (i = 1; i < nj; ++i) pthread_join(th[i], (void **)0);

    { TqHeap m; m.h = (u64 *)malloc((size_t)R * sizeof(u64)); m.n = 0; m.cap = R;
      for (i = 0; i < nj; ++i) { i32 k; for (k = 0; k < jobs[i].heap.n; ++k) tq_heap_push_(&m, jobs[i].heap.h[k]); free(jobs[i].heap.h); }
      all_h = m.h; cnt = m.n; }
    qsort(all_h, (size_t)cnt, sizeof(u64), tq_u64_desc_);      /* heap array -> best first */
    for (i = 0; i < cnt; ++i) rows[i] = (i32)(u32)(all_h[i] & 0xffffffffu);
    free(all_h); free(rot); free(tmp); free(lf); free(lut); free(blks);
    return cnt;
}

#endif /* TQ_C */
