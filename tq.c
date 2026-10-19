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
 * more than TQ_FLAT_MAX clusters, rows placed by looking in their 4 closest top cells, a polish round, centroid
 * scoring 16 centroids per instruction, and assignment and encoding spread over threads.
 *
 * A query picks its clusters with an int8 copy of the centroids (VNNI: 16 centroids x 4 dims per instruction, a
 * quarter of the bytes) and exact scores for the few near the cut; the scan keeps its candidates in a reservoir
 * (append, cut to the best R with a quickselect now and then) and prefetches two blocks ahead.
 *
 * Rows added after the build go straight into the index: each into its closest cluster, first into the free slots of
 * the cluster's last block, then into extra blocks (arrays of their own, chained cluster by cluster), so the built
 * arrays - possibly a mapped file - never move. The rotation and the codebook need no training, so an added row is
 * coded exactly like a built one; only the clusters age, and tq_add_put keeps the numbers that say when to rebuild.
 *
 *   Tq *tq_build(rows, n, dim, seed, nc, nthreads)   rows: n x dim f32 (unit length), possibly in chunks
 *   tq_search(tq, q, R, rows, filter..., nthreads)    the R best rows by approximate inner product
 *   tq_search_many(tq, Q, nq, R, ...)                 the same for a batch, cluster by cluster
 *   tq_add_prep / tq_add_put                          rows n, n+1, ... into the index (prep reads only: no lock)
 *   tq_free
 * C89, AVX2 / AVX-512 (VNNI) kernels when compiled with -march=native, scalar otherwise; threads via pthreads. */
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
    /* the centroids again as int8 for picking the probes with VNNI (derived, malloc'd; NULL without VNNI):
     * c8 is [ngroups][d4][16 centroids][4 dims], c8b the bias of each centroid (128 x its byte sum, since the query
     * bytes carry +128), c8s a per-dimension scale folded into the query before it is quantised */
    i8   *c8;
    i32  *c8b;
    f32  *c8s;
    i32   c8d4;             /* dims / 4, rounded up */
    /* rows added after the build (tq_add_put): extra blocks are numbered from nblk on and live in arrays of their own */
    i32   xn, xcap;         /* extra blocks in use / allocated */
    u8   *xcodes;           /* [xcap x vd x 16] */
    i32  *xslot;            /* [xcap x 32] the row in each slot, -1 = free */
    i32  *xnext;            /* [xcap] the cluster's next extra block, -1 = none */
    i32  *xhead, *xtail;    /* [max(nc, 1)] a cluster's first and last extra block, -1 = none */
    i32   slot_own;         /* slot_row was malloc'd for a mapped index (tq_remap): tq_free frees it */
    i32   nbuilt;           /* rows the build placed */
    i32   nadd;             /* rows added since */
    f32   fit0;             /* mean squared distance of a built row to its centroid (on a sample) */
    f64   fit_add;          /* the squared distances of the added rows to theirs, summed */
    i32  *csize;            /* [nc] rows in each cluster, added ones too (NULL: not counted yet - a loaded index) */
    f64   csum, csq;        /* the sum of csize and of its squares */
    f64   skew0;            /* tq_skew when the clusters were made (or first counted) */
} Tq;

#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512F__)
#define TQ_VNNI 1
#else
#define TQ_VNNI 0
#endif

static i32 tq_vd_(i32 dpad, i32 bits) { return bits == 2 ? ((dpad / 2) + 1) & ~1 : dpad; }

#define TQ_FLAT_MAX   1024              /* more clusters than this: two-level k-means */
#define TQ_SAMPLE_MAX ((size_t)512 << 20)   /* bytes of training sample at most (100M rows: ~25 a centroid) */

/* Lloyd-Max centroids of the standard normal, 16 levels ascending */
static const f32 TQ_CENT[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9424f, -0.6568f, -0.3881f, -0.1284f,
     0.1284f,  0.3881f,  0.6568f,  0.9424f,  1.2562f,  1.6180f,  2.0690f,  2.7326f
};

static u32 tq_rng_(u32 *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static f64 tq_now_(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (f64)ts.tv_sec + 1e-9 * (f64)ts.tv_nsec; }

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
    for (; g < ngroups; ++g) {                       /* a leftover group: two partial sums a row, so 8 chains overlap */
        const f32 *c = ct + (size_t)g * gs;
        __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, e0 = a0, e1 = a0, e2 = a0, e3 = a0;
        for (d = 0; d + 2 <= dim; d += 2) {
            __m512 c0 = _mm512_loadu_ps(c + (size_t)d * 16), c1 = _mm512_loadu_ps(c + (size_t)d * 16 + 16);
            a0 = _mm512_fmadd_ps(_mm512_set1_ps(v[0][d]), c0, a0); a1 = _mm512_fmadd_ps(_mm512_set1_ps(v[1][d]), c0, a1);
            a2 = _mm512_fmadd_ps(_mm512_set1_ps(v[2][d]), c0, a2); a3 = _mm512_fmadd_ps(_mm512_set1_ps(v[3][d]), c0, a3);
            e0 = _mm512_fmadd_ps(_mm512_set1_ps(v[0][d + 1]), c1, e0); e1 = _mm512_fmadd_ps(_mm512_set1_ps(v[1][d + 1]), c1, e1);
            e2 = _mm512_fmadd_ps(_mm512_set1_ps(v[2][d + 1]), c1, e2); e3 = _mm512_fmadd_ps(_mm512_set1_ps(v[3][d + 1]), c1, e3);
        }
        if (d < dim) {
            __m512 c0 = _mm512_loadu_ps(c + (size_t)d * 16);
            a0 = _mm512_fmadd_ps(_mm512_set1_ps(v[0][d]), c0, a0); a1 = _mm512_fmadd_ps(_mm512_set1_ps(v[1][d]), c0, a1);
            a2 = _mm512_fmadd_ps(_mm512_set1_ps(v[2][d]), c0, a2); a3 = _mm512_fmadd_ps(_mm512_set1_ps(v[3][d]), c0, a3);
        }
        _mm512_storeu_ps(out + (size_t)g * 16, _mm512_add_ps(a0, e0)); _mm512_storeu_ps(out + stride + (size_t)g * 16, _mm512_add_ps(a1, e1));
        _mm512_storeu_ps(out + 2 * stride + (size_t)g * 16, _mm512_add_ps(a2, e2)); _mm512_storeu_ps(out + 3 * stride + (size_t)g * 16, _mm512_add_ps(a3, e3));
    }
#else
    i32 r;
    for (r = 0; r < 4; ++r) tq_score_(v[r], ct, ngroups, dim, out + (size_t)r * stride);
#endif
}

/* index of the largest of sc[0..n) (the first, on a tie); AVX-512: a max reduction, then a compare to find it */
static i32 tq_argmax_(const f32 *sc, i32 n)
{
    i32 i, best = 0;
    f32 bs = sc[0];
#if defined(__AVX512F__)
    if (n >= 16) {
        __m512 mv = _mm512_loadu_ps(sc), bv;
        for (i = 16; i + 16 <= n; i += 16) mv = _mm512_max_ps(mv, _mm512_loadu_ps(sc + i));
        bs = _mm512_reduce_max_ps(mv);
        for (; i < n; ++i) if (sc[i] > bs) bs = sc[i];
        bv = _mm512_set1_ps(bs);
        for (i = 0; i + 16 <= n; i += 16) {
            __mmask16 k = _mm512_cmp_ps_mask(_mm512_loadu_ps(sc + i), bv, _CMP_EQ_OQ);
            if (k) return i + __builtin_ctz((u32)k);
        }
        for (; i < n; ++i) if (sc[i] == bs) return i;
        return 0;
    }
#endif
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
static void tq_par2_(TqFn fn, void *ctx, i64 n, i32 nth);
static void tq_par_(TqFn fn, void *ctx, i64 n, i32 nth)
{
    if (n < 4096) nth = 1;                                       /* row loops: not worth a thread below this */
    tq_par2_(fn, ctx, n, nth);
}

/* the same, without the minimum size (units of work that are each big, such as queries) */
static void tq_par2_(TqFn fn, void *ctx, i64 n, i32 nth)
{
    TqPar j[64]; pthread_t th[64];
    i32 t;
    if (nth < 1) nth = 1;
    if (nth > 64) nth = 64;
    if (n < nth) nth = n > 0 ? (i32)n : 1;
    for (t = 0; t < nth; ++t) { j[t].fn = fn; j[t].ctx = ctx; j[t].a = n * t / nth; j[t].b = n * (t + 1) / nth; j[t].tid = t; }
    for (t = 1; t < nth; ++t) pthread_create(&th[t], (const pthread_attr_t *)0, tq_par_job_, &j[t]);
    tq_par_job_(&j[0]);
    for (t = 1; t < nth; ++t) pthread_join(th[t], (void **)0);
}

/* ---- spherical k-means on a contiguous sample ---- */

/* s[dim] += v[dim] */
static void tq_acc_(f32 *s, const f32 *v, i32 dim)
{
    i32 d = 0;
#if defined(__AVX512F__)
    for (; d + 16 <= dim; d += 16) _mm512_storeu_ps(s + d, _mm512_add_ps(_mm512_loadu_ps(s + d), _mm512_loadu_ps(v + d)));
#elif defined(__AVX2__)
    for (; d + 8 <= dim; d += 8) _mm256_storeu_ps(s + d, _mm256_add_ps(_mm256_loadu_ps(s + d), _mm256_loadu_ps(v + d)));
#endif
    for (; d < dim; ++d) s[d] += v[d];
}

/* nth slices of per-thread sums and counts (thread t: sum + t * nc * dim, cnt + t * nc) folded into slice 0 */
static void tq_fold_(f32 *sum, i32 *cnt, i32 nc, i32 dim, i32 nth)
{
    i32 t, c;
    for (t = 1; t < nth; ++t) {
        for (c = 0; c < nc; ++c) { tq_acc_(sum + (size_t)c * dim, sum + ((size_t)t * nc + c) * dim, dim); cnt[c] += cnt[(size_t)t * nc + c]; }
    }
}

/* the centroids from their rows' sums: normalised; an empty one is left where it was (reseed: NULL) or moved onto a
 * random row of x */
static void tq_update_(f32 *cent, const f32 *sum, const i32 *cnt, i32 nc, i32 dim, const f32 *x, i32 m, u32 *reseed)
{
    i32 c, d;
    for (c = 0; c < nc; ++c) {
        const f32 *s = sum + (size_t)c * dim;
        f32 nrm = 0.0f, *o = cent + (size_t)c * dim;
        if (cnt[c] == 0) { if (reseed) memcpy(o, x + (size_t)(tq_rng_(reseed) % (u32)m) * dim, (size_t)dim * sizeof(f32)); continue; }
        for (d = 0; d < dim; ++d) nrm += s[d] * s[d];
        nrm = nrm > 0.0f ? 1.0f / (f32)sqrt((f64)nrm) : 1.0f;
        for (d = 0; d < dim; ++d) o[d] = s[d] * nrm;
    }
}

typedef struct { const f32 *x; const f32 *ct; i32 dim, nc; i32 *asg; f32 *sum; i32 *cnt; } TqKm;   /* sum, cnt: per thread, or NULL */

static void tq_km_asg_(void *p, i64 a, i64 b, i32 tid)
{
    TqKm *k = (TqKm *)p;
    f32 *sc = (f32 *)malloc((size_t)((k->nc + 15) / 16) * 64 * sizeof(f32));
    f32 *sum = k->sum ? k->sum + (size_t)tid * k->nc * k->dim : (f32 *)0;
    i32 *cnt = k->cnt ? k->cnt + (size_t)tid * k->nc : (i32 *)0;
    i64 i;
    for (i = a; i < b; i += 4) {
        const f32 *v[4];
        i32 m = b - i < 4 ? (i32)(b - i) : 4, r;
        for (r = 0; r < m; ++r) v[r] = k->x + (size_t)(i + r) * k->dim;
        tq_nearest4_(v, m, k->ct, k->nc, k->dim, sc, k->asg + i);
        if (sum) for (r = 0; r < m; ++r) { i32 c = k->asg[i + r]; tq_acc_(sum + (size_t)c * k->dim, v[r], k->dim); ++cnt[c]; }
    }
    free(sc);
}

/* nc unit centroids of the m rows of x into cent[nc x dim]; asg (optional, [m]) gets the final assignment. Every
 * thread sums the rows it assigned, so an iteration is one parallel pass. */
static void tq_kmeans_(const f32 *x, i32 m, i32 dim, i32 nc, u32 *rng, f32 *cent, i32 *asg_out, i32 nth)
{
    i32 nt = nth < 1 ? 1 : (nth > 64 ? 64 : nth), c, it;
    f32 *sum = (f32 *)malloc((size_t)nt * (size_t)nc * (size_t)dim * sizeof(f32));
    i32 *asg = asg_out ? asg_out : (i32 *)malloc((size_t)m * sizeof(i32)), *cnt = (i32 *)malloc((size_t)nt * (size_t)nc * sizeof(i32));
    TqKm k;
    for (c = 0; c < nc; ++c) memcpy(cent + (size_t)c * dim, x + (size_t)((i64)m * c / nc + (tq_rng_(rng) % (u32)(m / nc > 0 ? m / nc : 1))) * dim, (size_t)dim * sizeof(f32));
    k.x = x; k.dim = dim; k.nc = nc; k.asg = asg; k.sum = sum; k.cnt = cnt;
    for (it = 0; it < 8; ++it) {
        memset(sum, 0, (size_t)nt * (size_t)nc * (size_t)dim * sizeof(f32)); memset(cnt, 0, (size_t)nt * (size_t)nc * sizeof(i32));
        k.ct = tq_transpose_(cent, nc, dim);
        tq_par_(tq_km_asg_, &k, m, nt);
        free((void *)k.ct);
        tq_fold_(sum, cnt, nc, dim, nt);
        tq_update_(cent, sum, cnt, nc, dim, x, m, rng);            /* an empty cluster is reseeded on a random row */
    }
    if (asg_out) { k.sum = (f32 *)0; k.cnt = (i32 *)0; k.ct = tq_transpose_(cent, nc, dim); tq_par_(tq_km_asg_, &k, m, nt); free((void *)k.ct); }
    free(sum); free(cnt);
    if (!asg_out) free(asg);
}

/* The clustering of a whole build. Flat (one level) up to TQ_FLAT_MAX clusters; above, k1 ~ sqrt(nc) top cells, each
 * split into sub-clusters in proportion to its share of the sample. A row goes to the closest sub-cluster among those
 * of its T = 4 closest top cells: k1 + 4 nc/k1 dot products instead of nc, and the truly closest centroid for nearly
 * every row (with only the closest cell, a quarter of GloVe's rows sat in the wrong cluster - rows a query then looks
 * for in vain). A Lloyd round over the whole sample (polish) lets the sub-clusters of neighbouring cells settle their
 * shared borders, which k-means inside one cell cannot see. On GloVe-100 these three (a sample of 256 rows a centroid
 * instead of 64, 4 cells, the polish) take the true top 10 inside the 96 closest of 4350 clusters from 87% to 90%,
 * while scanning 10% fewer rows. */
typedef struct {
    i32 nc, dim;
    f32 *cent;              /* [nc x dim] final centroids */
    i32 k1;                 /* 0 = flat */
    f32 *top_ct;            /* [k1] top cells, transposed */
    i32 *sub0, *subn;       /* cell j owns centroids sub0[j] .. sub0[j] + subn[j] */
    f32 **sub_ct;           /* [k1] each cell's centroids, transposed */
    f32 *flat_ct;           /* flat: all centroids transposed */
    i32 T, wide;            /* top cells a row looks in, longest score row */
} TqClus;

#define TQ_TMAX 4           /* top cells a row looks in (2 above 128 top cells: past ~16M rows, where each one costs more) */
#define TQ_SAMPLE_PER 256   /* training rows per centroid (FAISS uses as many), capped by TQ_SAMPLE_MAX */
#define TQ_POLISH 1         /* Lloyd rounds over the whole sample after the per-cell k-means */
#define TQ_ABATCH 1024      /* rows assigned per batch */

static void tq_clus_free_(TqClus *cl)
{
    i32 j;
    free(cl->top_ct); free(cl->flat_ct); free(cl->sub0); free(cl->subn);
    if (cl->sub_ct) for (j = 0; j < cl->k1; ++j) free(cl->sub_ct[j]);
    free(cl->sub_ct);
}

/* the n largest of sc[0..cnt) into out[n] as indices, best first (n <= cnt): n vector argmax passes, each pick then
 * knocked out of sc (sc is scratch) */
static void tq_topn_(f32 *sc, i32 cnt, i32 n, i32 *out)
{
    i32 j;
    for (j = 0; j < n; ++j) { i32 b = tq_argmax_(sc, cnt); out[j] = b; sc[b] = -3e38f; }
}

/* scratch bytes of tq_asg_core_ */
static size_t tq_asg_scr_(const TqClus *cl)
{
    size_t st = (size_t)((cl->wide + 15) / 16) * 16;
    return (4 * st + (size_t)TQ_ABATCH * (2 + 2 * TQ_TMAX) + (size_t)cl->k1 + 2 + TQ_TMAX) * 4;
}

/* the closest centroid of each of the m (<= TQ_ABATCH) rows of x (contiguous) into asg[m]; w: tq_asg_scr_ bytes */
static void tq_asg_core_(const TqClus *cl, const f32 *x, i32 m, i32 *asg, u8 *w)
{
    i32 dim = cl->dim, i, r, c, p, T = cl->T, st = (cl->wide + 15) / 16 * 16;
    const f32 *v[4];
    f32 *sc = (f32 *)(void *)w, *bs = sc + 4 * (size_t)st;
    i32 *bi = (i32 *)(void *)(bs + m), *tops = bi + m, *ord = tops + (size_t)m * T, *cst = ord + (size_t)m * T;
    if (!cl->k1) {                                               /* flat: every centroid */
        for (i = 0; i < m; i += 4) {
            i32 g = m - i < 4 ? m - i : 4;
            for (r = 0; r < g; ++r) v[r] = x + (size_t)(i + r) * dim;
            tq_nearest4_(v, g, cl->flat_ct, cl->nc, dim, sc, asg + i);
        }
        return;
    }
    for (i = 0; i < m; i += 4) {                                 /* the T closest top cells of every row */
        i32 g = m - i < 4 ? m - i : 4;
        for (r = 0; r < 4; ++r) v[r] = x + (size_t)(i + (r < g ? r : g - 1)) * dim;
        tq_score4_(v, cl->top_ct, (cl->k1 + 15) / 16, dim, sc, (size_t)st);
        for (r = 0; r < g; ++r) tq_topn_(sc + (size_t)r * st, cl->k1, T, tops + (size_t)(i + r) * T);
    }
    memset(cst, 0, ((size_t)cl->k1 + 2) * sizeof(i32));        /* (row, cell) pairs grouped by cell */
    for (p = 0; p < m * T; ++p) ++cst[tops[p] + 2];
    for (c = 0; c < cl->k1; ++c) cst[c + 2] += cst[c + 1];
    for (p = 0; p < m * T; ++p) ord[cst[tops[p] + 1]++] = p;     /* cell c's pairs: ord[cst[c] .. cst[c + 1]) */
    for (i = 0; i < m; ++i) { bs[i] = -1e30f; bi[i] = 0; }
    for (c = 0; c < cl->k1; ++c) {                               /* four rows at a time against the cell's sub-clusters */
        i32 q, ns = cl->subn[c];
        for (q = cst[c]; q < cst[c + 1]; q += 4) {
            i32 g = cst[c + 1] - q < 4 ? cst[c + 1] - q : 4;
            for (r = 0; r < 4; ++r) v[r] = x + (size_t)(ord[q + (r < g ? r : g - 1)] / T) * dim;
            tq_score4_(v, cl->sub_ct[c], (ns + 15) / 16, dim, sc, (size_t)st);
            for (r = 0; r < g; ++r) {
                i32 row = ord[q + r] / T, k = tq_argmax_(sc + (size_t)r * st, ns);
                if (sc[(size_t)r * st + k] > bs[row]) { bs[row] = sc[(size_t)r * st + k]; bi[row] = cl->sub0[c] + k; }
            }
        }
    }
    memcpy(asg, bi, (size_t)m * sizeof(i32));
}

/* Lloyd rounds over the whole sample with the full assignment: sub-clusters move freely across their cells' borders */
typedef struct { const TqClus *cl; const f32 *x; i32 *asg; f32 *sum; i32 *cnt; } TqPolJob;

static void tq_pol_rows_(void *p, i64 a, i64 b, i32 tid)
{
    TqPolJob *j = (TqPolJob *)p;
    i32 dim = j->cl->dim, nc = j->cl->nc;
    u8 *w = (u8 *)malloc(tq_asg_scr_(j->cl));
    f32 *sum = j->sum + (size_t)tid * nc * dim;
    i32 *cnt = j->cnt + (size_t)tid * nc;
    i64 i;
    for (i = a; i < b; i += TQ_ABATCH) {
        i32 m = b - i < TQ_ABATCH ? (i32)(b - i) : TQ_ABATCH, r;
        tq_asg_core_(j->cl, j->x + (size_t)i * dim, m, j->asg + i, w);
        for (r = 0; r < m; ++r) { i32 c = j->asg[i + r]; tq_acc_(sum + (size_t)c * dim, j->x + (size_t)(i + r) * dim, dim); ++cnt[c]; }
    }
    free(w);
}

static void tq_clus_polish_(TqClus *cl, const f32 *x, i32 m, i32 rounds, i32 nth)
{
    i32 dim = cl->dim, nt = nth < 1 ? 1 : (nth > 64 ? 64 : nth), it, j;
    TqPolJob pj;
    pj.cl = cl; pj.x = x; pj.asg = (i32 *)malloc((size_t)m * sizeof(i32));
    pj.sum = (f32 *)malloc((size_t)nt * (size_t)cl->nc * (size_t)dim * sizeof(f32));
    pj.cnt = (i32 *)malloc((size_t)nt * (size_t)cl->nc * sizeof(i32));
    for (it = 0; it < rounds; ++it) {
        memset(pj.sum, 0, (size_t)nt * (size_t)cl->nc * (size_t)dim * sizeof(f32)); memset(pj.cnt, 0, (size_t)nt * (size_t)cl->nc * sizeof(i32));
        tq_par_(tq_pol_rows_, &pj, m, nt);
        tq_fold_(pj.sum, pj.cnt, cl->nc, dim, nt);
        tq_update_(cl->cent, pj.sum, pj.cnt, cl->nc, dim, x, m, (u32 *)0);   /* an empty one keeps its place */
        if (cl->k1) for (j = 0; j < cl->k1; ++j) { free(cl->sub_ct[j]); cl->sub_ct[j] = tq_transpose_(cl->cent + (size_t)cl->sub0[j] * dim, cl->subn[j], dim); }
        else { free(cl->flat_ct); cl->flat_ct = tq_transpose_(cl->cent, cl->nc, dim); }
    }
    free(pj.asg); free(pj.sum); free(pj.cnt);
}

static void tq_clus_train_(TqClus *cl, const TqCol *rows, i32 n, i32 dim, i32 nc, u32 seed, i32 nth)
{
    size_t cap = TQ_SAMPLE_MAX / ((size_t)dim * sizeof(f32));
    i32 m, i;
    f32 *x;
    u32 r = seed ? seed : 99u;
    memset(cl, 0, sizeof(*cl));
    cl->dim = dim;
    m = nc * TQ_SAMPLE_PER > 40000 ? nc * TQ_SAMPLE_PER : 40000;
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
        {   /* the top cells need far fewer rows than the sub-clusters: k-means on every st-th sample row (at most 1024
             * a cell), then one pass puts every sample row in its cell */
            i32 st = m / (k1 * 1024) > 1 ? m / (k1 * 1024) : 1, ms = (m + st - 1) / st;
            if (st == 1) tq_kmeans_(x, m, dim, k1, &r, top, asg, nth);
            else {
                f32 *xs = (f32 *)malloc((size_t)ms * (size_t)dim * sizeof(f32));
                TqKm k;
                for (i = 0; i < ms; ++i) memcpy(xs + (size_t)i * dim, x + (size_t)i * st * dim, (size_t)dim * sizeof(f32));
                tq_kmeans_(xs, ms, dim, k1, &r, top, (i32 *)0, nth);
                free(xs);
                k.x = x; k.dim = dim; k.nc = k1; k.asg = asg; k.sum = (f32 *)0; k.cnt = (i32 *)0; k.ct = tq_transpose_(top, k1, dim);
                tq_par_(tq_km_asg_, &k, m, nth);
                free((void *)k.ct);
            }
        }
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
    cl->T = cl->k1 > 128 ? 2 : TQ_TMAX;
    if (cl->T > cl->k1) cl->T = cl->k1 ? cl->k1 : 1;
    cl->wide = cl->nc;
    if (cl->k1) { i32 j; cl->wide = cl->k1; for (j = 0; j < cl->k1; ++j) if (cl->subn[j] > cl->wide) cl->wide = cl->subn[j]; }
    if (cl->k1) tq_clus_polish_(cl, x, m, TQ_POLISH, nth);
    free(x);
}

/* ---- the build ---- */
typedef struct { const TqCol *rows; const TqClus *cl; i32 *asg; i32 *cnt; i32 nc; } TqAsgJob;

static void tq_asg_rows_(void *p, i64 a, i64 b, i32 tid)
{
    TqAsgJob *j = (TqAsgJob *)p;
    const TqClus *cl = j->cl;
    i32 dim = cl->dim, *cnt = j->cnt + (size_t)tid * j->nc;
    f32 *xb = (f32 *)malloc((size_t)TQ_ABATCH * (size_t)dim * sizeof(f32));
    u8 *w = (u8 *)malloc(tq_asg_scr_(cl));
    i64 x;
    for (x = a; x < b; x += TQ_ABATCH) {
        i32 m = b - x < TQ_ABATCH ? (i32)(b - x) : TQ_ABATCH, i;
        for (i = 0; i < m; ++i) {                                /* the batch side by side (int8 rows dequantised) */
            f32 *dst = xb + (size_t)i * dim;
            const f32 *src = tq_row_(j->rows, x + i, dst);
            if (src != dst) memcpy(dst, src, (size_t)dim * sizeof(f32));
        }
        tq_asg_core_(cl, xb, m, j->asg + x, w);
        for (i = 0; i < m; ++i) ++cnt[j->asg[x + i]];
    }
    free(xb); free(w);
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

/* How unevenly the rows sit in the clusters: the rows a probe scans when queries land where rows do (sum s^2 / sum s),
 * over the plain mean cluster size - 1 when every cluster is the same size. Added rows that pile into a few clusters
 * raise it, and every query near them pays for those big clusters. */
f64 tq_skew(const Tq *t)
{
    if (t->nc <= 1 || !t->csize || t->csum <= 0.0) return 1.0;
    return (t->csq / t->csum) / (t->csum / (f64)t->nc);
}

static i32 tq_cblocks_(const Tq *t, i32 c, i32 *out);
static i32 tq_srow_(const Tq *t, i32 b, i32 k);

/* csize, csum, csq counted from the slots (an index from a file, or after tq_remap) */
static void tq_count_(Tq *t)
{
    i32 c, b, k, *bl;
    if (t->nc <= 1) return;
    if (!t->csize) t->csize = (i32 *)malloc((size_t)t->nc * sizeof(i32));
    bl = (i32 *)malloc((size_t)(t->nblk + t->xn > 0 ? t->nblk + t->xn : 1) * sizeof(i32));
    t->csum = t->csq = 0.0;
    for (c = 0; c < t->nc; ++c) {
        i32 nb = tq_cblocks_(t, c, bl), z = 0;
        for (b = 0; b < nb; ++b) for (k = 0; k < 32; ++k) { i32 r = tq_srow_(t, bl[b], k); z += r >= 0 && r < t->n; }
        t->csize[c] = z; t->csum += z; t->csq += (f64)z * z;
    }
    free(bl);
}

/* nc <= 1: flat scan of every row. nc > 1: IVF with about nc clusters (nprobe defaults to min(16%, 4 sqrt nc)).
 * skip (a bitset, IVF only, may be NULL): rows that get no slot - the store's tombstones, left out of a rebuild. */
static void tq_c8_(Tq *t);
#define TQ_SKIP(s, i) ((s) && (((s)[(size_t)(i) >> 6] >> ((u32)(i) & 63u)) & 1u))

Tq *tq_build_skip(const TqCol *rows, i32 n, i32 dim, u32 seed, i32 nc, i32 nth, i32 bits, const u64 *skip)
{
    Tq *t = (Tq *)calloc(1, sizeof(Tq));
    f64 t0 = tq_now_();
    i32 verbose = getenv("TQ_VERBOSE") != (char *)0;
    i32 i, nslots, *slot = (i32 *)0;
    u32 r = seed ? seed : 0x7154u;
    TqEncJob ej;
    if (nth < 1) nth = 1;
    t->dim = dim; t->dpad = dim + (dim & 1); t->n = n; t->nbuilt = n;
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
        t->cent = cl.cent;                                       /* the clustering keeps reading it until it is freed */
        slot = (i32 *)malloc((size_t)n * sizeof(i32));           /* first the cluster of each row, then its slot */
        aj.rows = rows; aj.cl = &cl; aj.asg = slot; aj.nc = nc;
        aj.cnt = (i32 *)calloc((size_t)nth * (size_t)nc, sizeof(i32));
        tq_par_(tq_asg_rows_, &aj, n, nth);
        if (verbose) fprintf(stderr, "tq_build: assign %d rows %.2fs\n", n, tq_now_() - t0);
        tq_clus_free_(&cl);
        for (tt = 1; tt < nth; ++tt) for (c = 0; c < nc; ++c) aj.cnt[c] += aj.cnt[(size_t)tt * nc + c];
        if (skip) for (i = 0; i < n; ++i) if (TQ_SKIP(skip, i)) { --aj.cnt[slot[i]]; slot[i] = -1; --t->nbuilt; }
        t->csize = (i32 *)malloc((size_t)nc * sizeof(i32));
        for (c = 0; c < nc; ++c) { t->csize[c] = aj.cnt[c]; t->csum += aj.cnt[c]; t->csq += (f64)aj.cnt[c] * aj.cnt[c]; }
        {   /* how far a row sits from its centroid, on up to 4096 rows: what added rows are held against (tq_add_put) */
            f64 sum = 0.0;
            i32 got = 0, step = n > 4096 ? n / 4096 : 1;
            f32 *rb = (f32 *)malloc((size_t)dim * sizeof(f32));
            for (i = 0; i < n; i += step) {
                if (slot[i] < 0) continue;
                sum += 2.0 - 2.0 * (f64)vs_dot(tq_row_(rows, i, rb), t->cent + (size_t)slot[i] * dim, dim);
                ++got;
            }
            t->fit0 = got ? (f32)(sum / got) : 0.0f;
            free(rb);
        }
        t->cstart = (i32 *)malloc((size_t)(nc + 1) * sizeof(i32));
        for (c = 0; c < nc; ++c) { t->cstart[c] = blk; blk += (aj.cnt[c] + 31) / 32; }
        t->cstart[nc] = blk;
        t->nblk = blk; nslots = blk * 32;
        t->slot_row = (i32 *)malloc((size_t)nslots * sizeof(i32));
        for (i = 0; i < nslots; ++i) t->slot_row[i] = -1;
        pos = aj.cnt;                                            /* reuse: next free slot of each cluster */
        for (c = 0; c < nc; ++c) pos[c] = t->cstart[c] * 32;
        for (i = 0; i < n; ++i) { i32 s; if (slot[i] < 0) continue; s = pos[slot[i]]++; t->slot_row[s] = i; slot[i] = s; }
        free(aj.cnt);
        t->nprobe = tq_default_probe_(nc);
        t->skew0 = tq_skew(t);
        t->ct = tq_transpose_(t->cent, nc, dim);
        tq_c8_(t);
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

Tq *tq_build(const TqCol *rows, i32 n, i32 dim, u32 seed, i32 nc, i32 nth, i32 bits)
{ return tq_build_skip(rows, n, dim, seed, nc, nth, bits, (const u64 *)0); }

/* the int8 copy of the centroids that picks the probes (VNNI builds): each dimension scaled by its largest |value|
 * over all centroids, so every dimension uses the full 8 bits */
static void tq_c8_(Tq *t)
{
    i32 nc = t->nc, dim = t->dim, d4 = (dim + 3) / 4, ng = (nc + 15) / 16, c, d;
    f32 *mx;
    if (!TQ_VNNI || nc <= 1 || t->c8) return;
    t->c8d4 = d4;
    t->c8 = (i8 *)calloc((size_t)ng * (size_t)d4 * 64, 1);
    t->c8b = (i32 *)calloc((size_t)ng * 16, sizeof(i32));
    t->c8s = (f32 *)calloc((size_t)d4 * 4, sizeof(f32));
    mx = t->c8s;                                                 /* first the largest |value| of each dimension */
    for (c = 0; c < nc; ++c)
        for (d = 0; d < dim; ++d) { f32 v = (f32)fabs((f64)t->cent[(size_t)c * dim + d]); if (v > mx[d]) mx[d] = v; }
    for (c = 0; c < nc; ++c) {
        i32 sum = 0;
        for (d = 0; d < dim; ++d) {
            f32 x = mx[d] > 0.0f ? t->cent[(size_t)c * dim + d] * 127.0f / mx[d] : 0.0f;
            i32 v = (i32)(x >= 0.0f ? x + 0.5f : x - 0.5f);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            t->c8[((size_t)(c / 16) * d4 + (size_t)(d / 4)) * 64 + (size_t)(c % 16) * 4 + (size_t)(d % 4)] = (i8)v;
            sum += v;
        }
        t->c8b[c] = 128 * sum;
    }
    for (d = 0; d < dim; ++d) mx[d] /= 127.0f;                   /* now the factor for the query: q'[d] = q[d] * c8s[d] */
}

/* after a load: the transposed and int8 centroids are derived, not saved */
static void tq_prepare_(Tq *t)
{
    if (t->nc > 1 && !t->ct) t->ct = tq_transpose_(t->cent, t->nc, t->dim);
    tq_c8_(t);
}

void tq_free(Tq *t)
{
    if (!t) return;
    if (!t->mapped) { free(t->s1); free(t->s2); free(t->perm); free(t->codes); free(t->cent); free(t->cstart); free(t->slot_row); }
    else if (t->slot_own) free(t->slot_row);
    free(t->ct); free(t->c8); free(t->c8b); free(t->c8s);
    free(t->xcodes); free(t->xslot); free(t->xnext); free(t->xhead); free(t->xtail); free(t->csize);
    free(t);
}

/* bytes of RAM the index holds (0 for the parts that live in a file mapping) */
static i64 tq_ram_(const Tq *t)
{
    i64 b = 0;
    if (!t) return 0;
    if (t->nc > 1) b += (i64)((t->nc + 15) / 16) * 16 * t->dim * 4;
    if (t->c8) b += (i64)((t->nc + 15) / 16) * (t->c8d4 * 64 + 64) + (i64)t->c8d4 * 16;
    if (!t->mapped) {
        b += (i64)t->nblk * t->vd * 16 + (i64)t->dpad * 12;
        if (t->slot_row) b += (i64)t->nblk * 32 * 4;
        if (t->nc > 1) b += (i64)t->nc * t->dim * 4 + (i64)(t->nc + 1) * 4;
    } else if (t->slot_own) b += (i64)t->nblk * 32 * 4;
    if (t->xcap) b += (i64)t->xcap * ((i64)t->vd * 16 + 32 * 4 + 4) + (i64)(t->nc > 1 ? t->nc : 1) * 8;
    return b;
}

/* block b of the scan: built (b < nblk) or added later (an extra block) */
static const u8 *tq_bptr_(const Tq *t, i32 b)
{
    size_t bb = (size_t)t->vd * 16;
    return b < t->nblk ? t->codes + (size_t)b * bb : t->xcodes + (size_t)(b - t->nblk) * bb;
}

/* the row in slot k of block b; -1 or >= t->n: none */
static i32 tq_srow_(const Tq *t, i32 b, i32 k)
{
    if (b >= t->nblk) return t->xslot[(size_t)(b - t->nblk) * 32 + (size_t)k];
    return t->slot_row ? t->slot_row[(size_t)b * 32 + (size_t)k] : b * 32 + k;
}

/* blocks of cluster c (flat: c = 0) - its built range, then its extra blocks - into out (NULL: just count them) */
static i32 tq_cblocks_(const Tq *t, i32 c, i32 *out)
{
    i32 b, nb = 0, b0 = t->nc > 1 ? t->cstart[c] : 0, b1 = t->nc > 1 ? t->cstart[c + 1] : t->nblk;
    for (b = b0; b < b1; ++b) { if (out) out[nb] = b; ++nb; }
    for (b = t->xhead ? t->xhead[c] : -1; b >= 0; b = t->xnext[b]) { if (out) out[nb] = t->nblk + b; ++nb; }
    return nb;
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
    const __m256i m0f = _mm256_set1_epi8(0x0f);
    __m128i x0, x1, y0, y1, r0q, r1q, r2q, r3q;
    i32 d = 0;
#if defined(__AVX512BW__)
    {
        __m512i A0 = _mm512_setzero_si512(), A1 = A0, B0 = A0, B1 = A0;
        const __m512i M0F = _mm512_set1_epi8(0x0f);
        for (; d + 4 <= dpad; d += 4) {
            __m512i codes = _mm512_loadu_si512((const void *)(blk + (size_t)d * 16));
            __m512i lv = _mm512_loadu_si512((const void *)(lut + (size_t)d * 16));
            __m512i lo = _mm512_and_si512(codes, M0F);
            __m512i hi = _mm512_and_si512(_mm512_srli_epi16(codes, 4), M0F);
            __m512i r0 = _mm512_shuffle_epi8(lv, lo), r1 = _mm512_shuffle_epi8(lv, hi);
            A0 = _mm512_add_epi16(A0, r0); A1 = _mm512_add_epi16(A1, _mm512_srli_epi16(r0, 8));
            B0 = _mm512_add_epi16(B0, r1); B1 = _mm512_add_epi16(B1, _mm512_srli_epi16(r1, 8));
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
        a0 = _mm256_add_epi16(a0, r0); a1 = _mm256_add_epi16(a1, _mm256_srli_epi16(r0, 8));
        b0 = _mm256_add_epi16(b0, r1); b1 = _mm256_add_epi16(b1, _mm256_srli_epi16(r1, 8));
    }
    /* fold the two dimension lanes; a1/b1 hold the odd rows of 0-15 / 16-31, and a0/b0 every 16-bit word whole (even
     * row + 256 x odd row, mod 2^16 - FAISS's trick: no mask per step), so the even rows are a0 - (a1 << 8) */
    x0 = _mm_add_epi16(_mm256_castsi256_si128(a0), _mm256_extracti128_si256(a0, 1));
    x1 = _mm_add_epi16(_mm256_castsi256_si128(a1), _mm256_extracti128_si256(a1, 1));
    y0 = _mm_add_epi16(_mm256_castsi256_si128(b0), _mm256_extracti128_si256(b0, 1));
    y1 = _mm_add_epi16(_mm256_castsi256_si128(b1), _mm256_extracti128_si256(b1, 1));
    x0 = _mm_sub_epi16(x0, _mm_slli_epi16(x1, 8));
    y0 = _mm_sub_epi16(y0, _mm_slli_epi16(y1, 8));
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

/* moves the k largest keys of a[0..n) to a[0..k), in no particular order (quickselect, median of three, O(n)) */
static void tq_sel64_(u64 *a, i32 n, i32 k)
{
    i32 lo = 0, hi = n - 1;
    if (k <= 0 || k >= n) return;
    while (lo < hi) {
        i32 i = lo, j = hi;
        u64 x = a[lo], y = a[lo + (hi - lo) / 2], z = a[hi], piv;
        piv = x < y ? (y < z ? y : (x < z ? z : x)) : (x < z ? x : (y < z ? z : y));
        while (i <= j) {
            while (a[i] > piv) ++i;
            while (a[j] < piv) --j;
            if (i <= j) { u64 w = a[i]; a[i] = a[j]; a[j] = w; ++i; --j; }
        }
        if (k - 1 <= j) hi = j;
        else if (k - 1 >= i) lo = i;
        else break;
    }
}

/* sorts a[0..n) by bytes b0 .. b0 + nb - 1 of the high half (key >> 32), biggest first; everything else rides along.
 * LSD radix (b0 = 0, nb = 2: u16 scores; b0 = 2, nb = 2: the top 16 bits of an f32 key, close enough to order the
 * probes); tmp: n entries. Stable, O(n), no comparisons. */
static void tq_rsort_(u64 *a, i32 n, u64 *tmp, i32 b0, i32 nb)
{
    i32 c[256], p, i, k, s;
    u64 *src = a, *dst = tmp, *x;
    if (n < 2) return;
    for (p = b0; p < b0 + nb; ++p) {
        i32 sh = 32 + 8 * p;
        memset(c, 0, sizeof c);
        for (i = 0; i < n; ++i) ++c[255 - (i32)((src[i] >> sh) & 255u)];
        for (k = 0, s = 0; k < 256; ++k) { i32 w = c[k]; c[k] = s; s += w; }
        for (i = 0; i < n; ++i) dst[c[255 - (i32)((src[i] >> sh) & 255u)]++] = src[i];
        x = src; src = dst; dst = x;
    }
    if (src != a) memcpy(a, src, (size_t)n * sizeof(u64));
}

/* an f32 as a u32 that sorts the same way */
static u32 tq_fkey_(f32 f)
{
    u32 u;
    memcpy(&u, &f, sizeof u);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/* The R best (score << 32 | row) keys seen so far. A key that beats thr is appended to a buffer of 2R; a full buffer
 * is cut back to its R best (quickselect), which raises thr to the R-th best key. An append is a store and a cut is
 * O(R) once per R appends - a heap paid a sift of log R steps on every row that got in. */
typedef struct { u64 *v; u64 thr; i32 n, R, cut; } TqRes;

static void tq_res_init_(TqRes *r, u64 *buf, i32 R) { r->v = buf; r->thr = 0; r->n = 0; r->R = R; r->cut = 0; }

static void tq_res_cut_(TqRes *r)
{
    i32 i;
    u64 m;
    if (r->n < r->R) return;
    if (r->n > r->R) { tq_sel64_(r->v, r->n, r->R); r->n = r->R; }
    for (m = r->v[0], i = 1; i < r->R; ++i) if (r->v[i] < m) m = r->v[i];
    r->thr = m; r->cut = 1;
}

static void tq_res_push_(TqRes *r, u64 key)
{
    if (r->cut && key <= r->thr) return;
    r->v[r->n++] = key;
    if (r->n == 2 * r->R) tq_res_cut_(r);
}

/* The kernel's threshold for the next block. A row that ties the R-th best score still gets the full key comparison,
 * so the R best keys come out the same whatever order the blocks are scanned in (one query or a batch, IVF or flat).
 * tq_res_all_: no threshold yet, every row counts. */
static u16 tq_res_kthr_(const TqRes *r) { u16 f = r->cut ? (u16)(r->thr >> 32) : (u16)0; return f ? (u16)(f - 1) : (u16)0; }
static i32 tq_res_all_(const TqRes *r) { return !r->cut || !(r->thr >> 32); }

/* the reservoir's rows, best key first, into rows[]; returns how many (at most R) */
static i32 tq_res_rows_(TqRes *r, i32 *rows, u64 *tmp)
{
    i32 i, j;
    u64 *a = r->v;
    if (r->n > r->R) { tq_sel64_(a, r->n, r->R); r->n = r->R; }
    tq_rsort_(a, r->n, tmp, 0, 2);
    for (i = 1; i < r->n; ++i) {                                 /* equal scores: bigger row first, as the whole key says */
        u64 v = a[i];
        for (j = i; j > 0 && a[j - 1] < v; --j) a[j] = a[j - 1];
        a[j] = v;
    }
    for (i = 0; i < r->n; ++i) rows[i] = (i32)(u32)(a[i] & 0xffffffffu);
    return r->n;
}

/* pulls a block of codes toward L1 (vd * 16 bytes) */
static void tq_prefetch_blk_(const u8 *p, i32 bytes)
{
#if defined(__GNUC__)
    i32 o;
    for (o = 0; o < bytes; o += 64) __builtin_prefetch((const void *)(p + o), 0, 3);
#else
    (void)p; (void)bytes;
#endif
}

typedef struct {
    const Tq *t; const u8 *lut;
    i32 b0, b1, R;
    const i32 *blks;                 /* IVF: the blocks to scan (b0..b1 index into this list); NULL = blocks b0..b1 */
    const u64 *dead; const TqCol *meta; i64 all, any; i32 filtered;
    TqRes res;
} TqJob;

static void *tq_job_(void *arg)
{
    TqJob *j = (TqJob *)arg;
    const Tq *t = j->t;
    i32 b, k, bb = t->vd * 16;
    u16 sc[32];
    for (b = j->b0; b < j->b1; ++b) {
        i32 blk = j->blks ? j->blks[b] : b;
        u32 m;
        if (b + 2 < j->b1)                                       /* two blocks ahead: the hardware prefetcher stops at every
                                                                  * 4 KB page and every jump to another cluster */
            tq_prefetch_blk_(tq_bptr_(t, j->blks ? j->blks[b + 2] : b + 2), bb);
        m = tq_block_(tq_bptr_(t, blk), j->lut, t->vd, tq_res_kthr_(&j->res), sc);
        if (tq_res_all_(&j->res)) m = 0xffffffffu;               /* no threshold yet: every row counts */
        while (m) {
            i32 row;
            k = tq_ctz_(m); m &= m - 1u;
            row = tq_srow_(t, blk, k);
            if (row < 0 || row >= t->n) continue;                /* padding */
            if (j->dead && ((j->dead[(u32)row >> 6] >> (row & 63)) & 1u)) continue;
            if (j->filtered) {
                i64 tag = *(const i64 *)(const void *)TQ_AT(j->meta, row);
                if ((tag & j->all) != j->all) continue;
                if (j->any && !(tag & j->any)) continue;
            }
            tq_res_push_(&j->res, ((u64)sc[k] << 32) | (u64)(u32)row);
        }
    }
    return (void *)0;
}

/* the query's 16-entry tables, one per code column, quantised to u8 (lut: vd * 16 + 32 bytes; rot, tmp: dpad
 * floats; lf: vd * 16 floats) */
static void tq_lut_(const Tq *t, const f32 *q, f32 *rot, f32 *tmp, f32 *lf, u8 *lut)
{
    f32 maxr = 0.0f, scale;
    i32 d, c, i, lutmax;
    tq_rot_(t, q, rot, tmp);
    for (d = 0; d < t->vd; ++d) {
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
}

#if TQ_VNNI
/* the query for VNNI: q[d] * c8s[d] scaled to +-127, plus 128 (unsigned bytes), packed four dims to a word */
static void tq_q8_(const Tq *t, const f32 *q, u32 *qw)
{
    i32 d, n4 = t->c8d4 * 4;
    f32 mx = 0.0f, sc;
    for (d = 0; d < t->dim; ++d) { f32 v = (f32)fabs((f64)(q[d] * t->c8s[d])); if (v > mx) mx = v; }
    sc = mx > 0.0f ? 127.0f / mx : 0.0f;
    for (d = 0; d < t->c8d4; ++d) qw[d] = 0;
    for (d = 0; d < n4; ++d) {
        f32 x = d < t->dim ? q[d] * t->c8s[d] * sc : 0.0f;
        i32 v = (i32)(x >= 0.0f ? x + 0.5f : x - 0.5f);
        v = v > 127 ? 127 : (v < -127 ? -127 : v);
        qw[d / 4] |= (u32)(128 + v) << (8 * (d % 4));
    }
}

/* int scores of the query against every centroid, bigger = closer, out[16 * ngroups] (padding lanes: INT_MIN). One
 * VNNI instruction scores 16 centroids on 4 dimensions; 8 groups at a time keep the unit busy. 4x fewer bytes than
 * the f32 centroids (GloVe-100: 0.44 MB, which stays in L2) and 4x the work per instruction. */
static void tq_score8_(const Tq *t, const u32 *qw, i32 *out)
{
    i32 ng = (t->nc + 15) / 16, d4 = t->c8d4, g = 0, k;
    size_t gs = (size_t)d4 * 64;
    for (; g + 8 <= ng; g += 8) {
        const i8 *c = t->c8 + (size_t)g * gs;
        __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
        for (k = 0; k < d4; ++k) {
            __m512i x = _mm512_set1_epi32((int)qw[k]);
            const i8 *p = c + (size_t)k * 64;
            a0 = _mm512_dpbusd_epi32(a0, x, _mm512_loadu_si512((const void *)p));
            a1 = _mm512_dpbusd_epi32(a1, x, _mm512_loadu_si512((const void *)(p + gs)));
            a2 = _mm512_dpbusd_epi32(a2, x, _mm512_loadu_si512((const void *)(p + 2 * gs)));
            a3 = _mm512_dpbusd_epi32(a3, x, _mm512_loadu_si512((const void *)(p + 3 * gs)));
            a4 = _mm512_dpbusd_epi32(a4, x, _mm512_loadu_si512((const void *)(p + 4 * gs)));
            a5 = _mm512_dpbusd_epi32(a5, x, _mm512_loadu_si512((const void *)(p + 5 * gs)));
            a6 = _mm512_dpbusd_epi32(a6, x, _mm512_loadu_si512((const void *)(p + 6 * gs)));
            a7 = _mm512_dpbusd_epi32(a7, x, _mm512_loadu_si512((const void *)(p + 7 * gs)));
        }
        { i32 *o = out + (size_t)g * 16; const i32 *bz = t->c8b + (size_t)g * 16;
          _mm512_storeu_si512((void *)o, _mm512_sub_epi32(a0, _mm512_loadu_si512((const void *)bz)));
          _mm512_storeu_si512((void *)(o + 16), _mm512_sub_epi32(a1, _mm512_loadu_si512((const void *)(bz + 16))));
          _mm512_storeu_si512((void *)(o + 32), _mm512_sub_epi32(a2, _mm512_loadu_si512((const void *)(bz + 32))));
          _mm512_storeu_si512((void *)(o + 48), _mm512_sub_epi32(a3, _mm512_loadu_si512((const void *)(bz + 48))));
          _mm512_storeu_si512((void *)(o + 64), _mm512_sub_epi32(a4, _mm512_loadu_si512((const void *)(bz + 64))));
          _mm512_storeu_si512((void *)(o + 80), _mm512_sub_epi32(a5, _mm512_loadu_si512((const void *)(bz + 80))));
          _mm512_storeu_si512((void *)(o + 96), _mm512_sub_epi32(a6, _mm512_loadu_si512((const void *)(bz + 96))));
          _mm512_storeu_si512((void *)(o + 112), _mm512_sub_epi32(a7, _mm512_loadu_si512((const void *)(bz + 112)))); }
    }
    for (; g < ng; ++g) {
        const i8 *c = t->c8 + (size_t)g * gs;
        __m512i a0 = _mm512_setzero_si512(), a1 = a0;
        for (k = 0; k + 2 <= d4; k += 2) {
            a0 = _mm512_dpbusd_epi32(a0, _mm512_set1_epi32((int)qw[k]), _mm512_loadu_si512((const void *)(c + (size_t)k * 64)));
            a1 = _mm512_dpbusd_epi32(a1, _mm512_set1_epi32((int)qw[k + 1]), _mm512_loadu_si512((const void *)(c + (size_t)k * 64 + 64)));
        }
        if (k < d4) a0 = _mm512_dpbusd_epi32(a0, _mm512_set1_epi32((int)qw[k]), _mm512_loadu_si512((const void *)(c + (size_t)k * 64)));
        _mm512_storeu_si512((void *)(out + (size_t)g * 16),
                            _mm512_sub_epi32(_mm512_add_epi32(a0, a1), _mm512_loadu_si512((const void *)(t->c8b + (size_t)g * 16))));
    }
    for (k = t->nc; k < ng * 16; ++k) out[k] = -2147483647 - 1;
}

/* the same for 8 queries at once (qw[r] -> out[r]): every 64 bytes of centroids loaded feed 8 queries, so a batch
 * reads the int8 centroids once per 8 queries instead of once per query */
#define TQ_D8(r) { __m512i x = _mm512_set1_epi32((int)qw[r][k]); \
                   a##r##0 = _mm512_dpbusd_epi32(a##r##0, x, c0); a##r##1 = _mm512_dpbusd_epi32(a##r##1, x, c1); }
#define TQ_S8(r) { _mm512_storeu_si512((void *)(out[r] + (size_t)g * 16), _mm512_sub_epi32(a##r##0, b0)); \
                   _mm512_storeu_si512((void *)(out[r] + (size_t)g * 16 + 16), _mm512_sub_epi32(a##r##1, b1)); }
static void tq_score8x8_(const Tq *t, u32 *const *qw, i32 *const *out)
{
    i32 ng = (t->nc + 15) / 16, d4 = t->c8d4, g = 0, k, r;
    size_t gs = (size_t)d4 * 64;
    for (; g + 2 <= ng; g += 2) {
        const i8 *c = t->c8 + (size_t)g * gs;
        __m512i a00 = _mm512_setzero_si512(), a01 = a00, a10 = a00, a11 = a00, a20 = a00, a21 = a00, a30 = a00, a31 = a00;
        __m512i a40 = a00, a41 = a00, a50 = a00, a51 = a00, a60 = a00, a61 = a00, a70 = a00, a71 = a00, b0, b1;
        for (k = 0; k < d4; ++k) {
            __m512i c0 = _mm512_loadu_si512((const void *)(c + (size_t)k * 64)), c1 = _mm512_loadu_si512((const void *)(c + gs + (size_t)k * 64));
            TQ_D8(0) TQ_D8(1) TQ_D8(2) TQ_D8(3) TQ_D8(4) TQ_D8(5) TQ_D8(6) TQ_D8(7)
        }
        b0 = _mm512_loadu_si512((const void *)(t->c8b + (size_t)g * 16)); b1 = _mm512_loadu_si512((const void *)(t->c8b + (size_t)g * 16 + 16));
        TQ_S8(0) TQ_S8(1) TQ_S8(2) TQ_S8(3) TQ_S8(4) TQ_S8(5) TQ_S8(6) TQ_S8(7)
    }
    for (; g < ng; ++g) {                                        /* an odd last group */
        const i8 *c = t->c8 + (size_t)g * gs;
        for (r = 0; r < 8; ++r) {
            __m512i a = _mm512_setzero_si512();
            for (k = 0; k < d4; ++k) a = _mm512_dpbusd_epi32(a, _mm512_set1_epi32((int)qw[r][k]), _mm512_loadu_si512((const void *)(c + (size_t)k * 64)));
            _mm512_storeu_si512((void *)(out[r] + (size_t)g * 16), _mm512_sub_epi32(a, _mm512_loadu_si512((const void *)(t->c8b + (size_t)g * 16))));
        }
    }
    for (r = 0; r < 8; ++r) for (k = t->nc; k < ng * 16; ++k) out[r][k] = -2147483647 - 1;
}
#undef TQ_D8
#undef TQ_S8

/* how many of s[0..n16) are >= T */
static i32 tq_count_ge_(const i32 *s, i32 n16, i32 T)
{
    __m512i tv = _mm512_set1_epi32(T);
    i32 i, c = 0;
    for (i = 0; i < n16; i += 16) c += __builtin_popcount((u32)_mm512_cmpge_epi32_mask(_mm512_loadu_si512((const void *)(s + i)), tv));
    return c;
}

/* From the int8 scores sc of query q to candidates: the largest threshold that keeps at least K = np + np/4 + 16 of
 * them (a binary search on a vector count, stopping once at most 1.5 K pass - no sort), then exact f32 scores for
 * those few as keys. Returns how many keys. */
static i32 tq_cands_(const Tq *t, const f32 *q, i32 np, const i32 *sc, u64 *key, i32 *cand)
{
    i32 nc = t->nc, ng16 = (nc + 15) / 16 * 16, K = np + np / 4 + 16, K2, lo, hi, i, nk = 0;
    __m512i tv, idx = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    if (K > nc) K = nc;
    K2 = K + K / 2;
    lo = hi = sc[0];
    for (i = 1; i < nc; ++i) { if (sc[i] < lo) lo = sc[i]; if (sc[i] > hi) hi = sc[i]; }
    while (lo < hi) {                                            /* the largest T with at least K scores >= T */
        i32 mid = lo + (i32)(((i64)hi - (i64)lo + 1) / 2), c = tq_count_ge_(sc, ng16, mid);
        if (c >= K) { lo = mid; if (c <= K2) break; }
        else hi = mid - 1;
    }
    tv = _mm512_set1_epi32(lo);
    for (i = 0; i < ng16; i += 16) {
        __mmask16 m = _mm512_cmpge_epi32_mask(_mm512_loadu_si512((const void *)(sc + i)), tv);
        _mm512_mask_compressstoreu_epi32((void *)(cand + nk), m, idx);
        nk += __builtin_popcount((u32)m);
        idx = _mm512_add_epi32(idx, _mm512_set1_epi32(16));
    }
    for (i = 0; i < nk; ++i) {                                    /* exact scores for the few that are left */
        i32 c = cand[i] < nc ? cand[i] : 0;
        key[i] = ((u64)tq_fkey_(vs_dot(q, t->cent + (size_t)c * t->dim, t->dim)) << 32) | (u32)c;
    }
    return nk;
}
#endif

/* scratch bytes for tq_probe_ */
static size_t tq_probe_scr_(const Tq *t)
{
    size_t ng16 = (size_t)((t->nc + 15) / 16) * 16;
    return ng16 * 24 + (size_t)(t->dim + 3) / 4 * 4 + 64;
}

/* the np best of nk keys into out[np]: sorted closest first (by the top 16 bits of the score - enough to make the
 * threshold rise early), or (sorted = 0) only the closest moved first */
static void tq_pick_(u64 *key, i32 nk, i32 np, u64 *tmp, i32 *out, i32 sorted)
{
    i32 i;
    if (nk > np) tq_sel64_(key, nk, np);
    else np = nk;
    if (sorted) tq_rsort_(key, np, tmp, 2, 2);
    else { i32 b = 0; u64 w; for (i = 1; i < np; ++i) if (key[i] > key[b]) b = i; w = key[0]; key[0] = key[b]; key[b] = w; }
    for (i = 0; i < np; ++i) out[i] = (i32)(u32)key[i];
}

/* The np clusters closest to q into out[np], closest first. VNNI: int8 scores of every centroid, a threshold that
 * keeps the best np + np/4 + 16 or so, exact f32 scores for only those - so the probes are the f32 ones. Otherwise:
 * f32 scores of every centroid. scr: tq_probe_scr_ bytes. */
static void tq_probe_(const Tq *t, const f32 *q, i32 np, u8 *scr, i32 *out)
{
    i32 nc = t->nc, ng16 = (nc + 15) / 16 * 16, i, nk;
    u64 *key = (u64 *)(void *)scr, *tmp = key + ng16;
    i32 *sc = (i32 *)(void *)(tmp + ng16);
    if (np > nc) np = nc;
    if (np <= 0) return;
#if TQ_VNNI
    if (t->c8) {
        i32 *cand = sc + ng16;
        u32 *qw = (u32 *)(void *)(cand + ng16);
        tq_q8_(t, q, qw);
        tq_score8_(t, qw, sc);
        nk = tq_cands_(t, q, np, sc, key, cand);
    } else
#endif
    {
        f32 *fs = (f32 *)(void *)sc;
        tq_score_(q, t->ct, ng16 / 16, t->dim, fs);              /* every centroid, 16 per instruction */
        for (i = 0; i < nc; ++i) key[i] = ((u64)tq_fkey_(fs[i]) << 32) | (u32)i;
        nk = nc;
    }
    tq_pick_(key, nk, np, tmp, out, 1);
}

/* The R best rows (by the quantised inner product) that are alive and pass the meta filter; returns how
 * many were found, rows[] best first. dead (a bitset) / meta may be NULL. nthreads > 1 splits the blocks.
 * probe: IVF clusters to scan (0 = the index default, >= nc = all). */
i32 tq_search(const Tq *t, const f32 *q, i32 R, i32 *rows, const u64 *dead, const TqCol *meta, i64 all, i64 any, i32 filtered, i32 nthreads, i32 probe)
{
    i32 *blks = (i32 *)0, nblk_scan, i, nj, per, tot;
    f32 *rot, *tmp, *lf;
    u8 *lut;
    u64 *buf, *mt;
    TqJob jobs[16];
    TqRes all_r;
    pthread_t th[16];

    if (t->n == 0 || R <= 0) return 0;
    rot = (f32 *)malloc((size_t)t->dpad * sizeof(f32)); tmp = (f32 *)malloc((size_t)t->dpad * sizeof(f32));
    lf = (f32 *)malloc((size_t)t->vd * 16 * sizeof(f32)); lut = (u8 *)malloc((size_t)t->vd * 16 + 32);
    tq_lut_(t, q, rot, tmp, lf, lut);
    if (t->nc > 1) {                                             /* IVF: the blocks of the nprobe clusters closest to the query */
        i32 np = probe > 0 ? probe : t->nprobe, nb = 0, *pc;
        u8 *scr = (u8 *)malloc(tq_probe_scr_(t));
        if (np > t->nc) np = t->nc;
        pc = (i32 *)malloc((size_t)np * sizeof(i32));
        tq_probe_(t, q, np, scr, pc);
        for (i = 0; i < np; ++i) nb += tq_cblocks_(t, pc[i], (i32 *)0);
        blks = (i32 *)malloc((size_t)(nb ? nb : 1) * sizeof(i32));
        for (nb = 0, i = 0; i < np; ++i) nb += tq_cblocks_(t, pc[i], blks + nb);
        nblk_scan = nb;
        free(scr); free(pc);
    } else nblk_scan = t->nblk + t->xn;                          /* flat: added blocks hold rows in order too */
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 16) nthreads = 16;
    if (nblk_scan < 64) nthreads = 1;
    nj = nthreads; per = (nblk_scan + nj - 1) / nj;
    buf = (u64 *)malloc((size_t)nj * 2 * (size_t)R * sizeof(u64));
    mt = (u64 *)malloc((size_t)R * sizeof(u64));
    for (i = 0; i < nj; ++i) {
        jobs[i].t = t; jobs[i].lut = lut; jobs[i].blks = blks; jobs[i].b0 = i * per; jobs[i].b1 = (i + 1) * per < nblk_scan ? (i + 1) * per : nblk_scan;
        if (jobs[i].b0 > jobs[i].b1) jobs[i].b0 = jobs[i].b1;
        jobs[i].R = R; jobs[i].dead = dead; jobs[i].meta = meta; jobs[i].all = all; jobs[i].any = any; jobs[i].filtered = filtered && meta;
        tq_res_init_(&jobs[i].res, buf + (size_t)i * 2 * (size_t)R, R);
    }
    for (i = 1; i < nj; ++i) pthread_create(&th[i], (const pthread_attr_t *)0, tq_job_, &jobs[i]);
    tq_job_(&jobs[0]);
    for (i = 1; i < nj; ++i) pthread_join(th[i], (void **)0);

    for (tot = 0, i = 0; i < nj; ++i) {                          /* every thread's keys side by side, then the R best */
        if (buf + tot != jobs[i].res.v) memmove(buf + tot, jobs[i].res.v, (size_t)jobs[i].res.n * sizeof(u64));
        tot += jobs[i].res.n;
    }
    tq_res_init_(&all_r, buf, R); all_r.n = tot;
    tot = tq_res_rows_(&all_r, rows, mt);
    free(buf); free(mt); free(rot); free(tmp); free(lf); free(lut); free(blks);
    return tot;
}

/* ---- many queries at once: the scan reads more code bytes than the cache holds, so turn it around - each thread
 * takes a share of the queries, and every block of codes is scored for all of its queries that probe it while the
 * block sits in L1: the codes come in from memory once per batch instead of once per query. No cross-thread merge:
 * a query's candidates live in one reservoir. ---- */
#define TQ_MANY_SUB 256              /* queries a thread batches at a time (bounds the reservoirs: 256 x 2R x 8 bytes) */

typedef struct {
    const Tq *t;
    const f32 *Q;                    /* the batch's queries, unit */
    i32 nq, R, np, lutb;
    u8 *luts;                        /* nq x lutb */
    i32 *probes;                     /* nq x np (IVF) */
    i32 *cq0, *cq;                   /* cluster c's queries: cq[cq0[c] .. cq0[c + 1]) (IVF) */
    const u64 *dead;
    TqRes *res;                      /* nq; each is cut right after the query's closest cluster, so the rest of the scan
                                      * starts with a threshold */
} TqMany;

static void tq_many_push_(const TqMany *m, TqRes *rs, const u8 *blk, i32 blkno, const u8 *lut)
{
    const Tq *t = m->t;
    u16 sc[32];
    i32 k;
    u32 msk = tq_block_(blk, lut, t->vd, tq_res_kthr_(rs), sc);
    if (tq_res_all_(rs)) msk = 0xffffffffu;
    while (msk) {
        i32 row;
        k = tq_ctz_(msk); msk &= msk - 1u;
        row = tq_srow_(t, blkno, k);
        if (row < 0 || row >= t->n) continue;
        if (m->dead && ((m->dead[(u32)row >> 6] >> (row & 63)) & 1u)) continue;
        tq_res_push_(rs, ((u64)sc[k] << 32) | (u64)(u32)row);
    }
}

#define TQ_MANY_QG 32                /* flat scan: queries a tile */
#define TQ_MANY_BG 64                /* flat scan: blocks a tile (about 100 KB of codes at d = 100) */

/* one thread, nq (<= TQ_MANY_SUB) queries: rows nq x R, cnt nq */
static void tq_many_one_(const Tq *t, const f32 *Q, i32 nq, i32 R, i32 np, i32 *rows, i32 *cnt, const u64 *dead)
{
    TqMany m;
    i32 i, c, q;
    size_t bb = (size_t)t->vd * 16;
    f32 *rot = (f32 *)malloc((size_t)t->dpad * sizeof(f32)), *tmp = (f32 *)malloc((size_t)t->dpad * sizeof(f32));
    f32 *lf = (f32 *)malloc((size_t)t->vd * 16 * sizeof(f32));
    u8 *scr = np ? (u8 *)malloc(tq_probe_scr_(t)) : (u8 *)0;
    u64 *buf = (u64 *)malloc((size_t)nq * 2 * (size_t)R * sizeof(u64)), *mt = (u64 *)malloc((size_t)R * sizeof(u64));
    memset(&m, 0, sizeof(m));
    m.t = t; m.Q = Q; m.nq = nq; m.R = R; m.np = np; m.dead = dead; m.lutb = t->vd * 16 + 32;
    m.luts = (u8 *)malloc((size_t)nq * (size_t)m.lutb);
    m.res = (TqRes *)malloc((size_t)nq * sizeof(TqRes));
    for (i = 0; i < nq; ++i) tq_res_init_(&m.res[i], buf + (size_t)i * 2 * (size_t)R, R);
    if (np) m.probes = (i32 *)malloc((size_t)nq * (size_t)np * sizeof(i32));
    for (q = 0; q < nq; ++q) tq_lut_(t, Q + (size_t)q * t->dim, rot, tmp, lf, m.luts + (size_t)q * m.lutb);
#if TQ_VNNI
    if (np && t->c8) {                                           /* probes, 8 queries per pass over the int8 centroids */
        i32 ng16 = (t->nc + 15) / 16 * 16, d4 = t->c8d4, r;
        u32 *qws = (u32 *)malloc((size_t)nq * (size_t)d4 * sizeof(u32)), *qp[8];
        i32 *s8 = (i32 *)malloc((size_t)8 * (size_t)ng16 * sizeof(i32)), *sp[8];
        u64 *key = (u64 *)(void *)scr, *ktmp = key + ng16;
        i32 *cand = (i32 *)(void *)(ktmp + ng16) + ng16;
        for (q = 0; q < nq; ++q) tq_q8_(t, Q + (size_t)q * t->dim, qws + (size_t)q * d4);
        for (r = 0; r < 8; ++r) sp[r] = s8 + (size_t)r * ng16;
        for (q = 0; q < nq; q += 8) {
            i32 g = nq - q < 8 ? nq - q : 8;
            if (g == 8) { for (r = 0; r < 8; ++r) qp[r] = qws + (size_t)(q + r) * d4; tq_score8x8_(t, qp, sp); }
            else for (r = 0; r < g; ++r) tq_score8_(t, qws + (size_t)(q + r) * d4, sp[r]);
            for (r = 0; r < g; ++r)                              /* only the closest has to come first: clusters are */
                tq_pick_(key, tq_cands_(t, Q + (size_t)(q + r) * t->dim, np, sp[r], key, cand), np, ktmp,   /* scanned in order */
                         m.probes + (size_t)(q + r) * np, 0);
        }
        free(qws); free(s8);
    } else
#endif
    for (q = 0; q < nq && np; ++q) tq_probe_(t, Q + (size_t)q * t->dim, np, scr, m.probes + (size_t)q * np);
    for (q = 0; q < nq && np; ++q) {                             /* each query's closest cluster first: a threshold early */
        i32 b2;
        c = m.probes[(size_t)q * np];
        for (b2 = t->cstart[c]; b2 < t->cstart[c + 1]; ++b2) tq_many_push_(&m, &m.res[q], t->codes + (size_t)b2 * bb, b2, m.luts + (size_t)q * m.lutb);
        for (b2 = t->xhead ? t->xhead[c] : -1; b2 >= 0; b2 = t->xnext[b2])
            tq_many_push_(&m, &m.res[q], t->xcodes + (size_t)b2 * bb, t->nblk + b2, m.luts + (size_t)q * m.lutb);
        tq_res_cut_(&m.res[q]);
    }
    if (np) {                                                    /* cluster -> the queries that still have to scan it */
        i32 nx;
        m.cq0 = (i32 *)calloc((size_t)t->nc + 2, sizeof(i32));
        m.cq = (i32 *)malloc((size_t)nq * (size_t)np * sizeof(i32));
        for (i = 0; i < nq * np; ++i) if (i % np) ++m.cq0[m.probes[i] + 2];
        for (c = 0; c < t->nc; ++c) m.cq0[c + 2] += m.cq0[c + 1];
        for (i = 0; i < nq * np; ++i) if (i % np) m.cq[m.cq0[m.probes[i] + 1]++] = i / np;
        for (c = 0; c < t->nc; c = nx) {                         /* cluster by cluster: its blocks x its queries */
            i32 b2, x;
            for (nx = c + 1; nx < t->nc && m.cq0[nx] == m.cq0[nx + 1]; ++nx) {}
            if (m.cq0[c] == m.cq0[c + 1]) continue;
            for (b2 = t->cstart[c]; b2 < t->cstart[c + 1]; ++b2) {
                const u8 *blk = t->codes + (size_t)b2 * bb;
                i32 pf = b2 + 2 - t->cstart[c + 1];                  /* two blocks ahead, into the next probed cluster */
                if (pf < 0) tq_prefetch_blk_(blk + 2 * bb, (i32)bb);
                else if (nx < t->nc && t->cstart[nx] + pf < t->cstart[nx + 1]) tq_prefetch_blk_(t->codes + (size_t)(t->cstart[nx] + pf) * bb, (i32)bb);
                for (x = m.cq0[c]; x < m.cq0[c + 1]; ++x) { q = m.cq[x]; tq_many_push_(&m, &m.res[q], blk, b2, m.luts + (size_t)q * m.lutb); }
            }
            for (b2 = t->xhead ? t->xhead[c] : -1; b2 >= 0; b2 = t->xnext[b2]) {   /* rows added since the build */
                const u8 *blk = t->xcodes + (size_t)b2 * bb;
                if (t->xnext[b2] >= 0) tq_prefetch_blk_(t->xcodes + (size_t)t->xnext[b2] * bb, (i32)bb);
                for (x = m.cq0[c]; x < m.cq0[c + 1]; ++x) { q = m.cq[x]; tq_many_push_(&m, &m.res[q], blk, t->nblk + b2, m.luts + (size_t)q * m.lutb); }
            }
        }
    } else {                                                     /* flat: tiles of blocks x groups of queries */
        i32 b0, q0, b2, nbt = t->nblk + t->xn;
        for (b0 = 0; b0 < nbt; b0 += TQ_MANY_BG)
            for (q0 = 0; q0 < nq; q0 += TQ_MANY_QG)
                for (b2 = b0; b2 < nbt && b2 < b0 + TQ_MANY_BG; ++b2)
                    for (q = q0; q < nq && q < q0 + TQ_MANY_QG; ++q)
                        tq_many_push_(&m, &m.res[q], tq_bptr_(t, b2), b2, m.luts + (size_t)q * m.lutb);
    }
    for (q = 0; q < nq; ++q) cnt[q] = tq_res_rows_(&m.res[q], rows + (size_t)q * R, mt);   /* best first */
    free(m.res); free(m.luts); free(m.probes); free(m.cq0); free(m.cq);
    free(buf); free(mt); free(rot); free(tmp); free(lf); free(scr);
}

typedef struct { const Tq *t; const f32 *Q; i32 R, np; i32 *rows, *cnt; const u64 *dead; } TqManyJob;

static void tq_many_range_(void *p, i64 a, i64 b, i32 tid)
{
    TqManyJob *j = (TqManyJob *)p;
    i64 q;
    (void)tid;
    for (q = a; q < b; q += TQ_MANY_SUB) {
        i32 m = b - q < TQ_MANY_SUB ? (i32)(b - q) : TQ_MANY_SUB;
        tq_many_one_(j->t, j->Q + (size_t)q * j->t->dim, m, j->R, j->np, j->rows + (size_t)q * j->R, j->cnt + q, j->dead);
    }
}

/* tq_search for nq queries (no meta filter): rows is nq x R, cnt[q] how many rows query q got; each thread batches
 * its share of the queries. The rows found are the same as tq_search's up to ties in the quantised score. */
void tq_search_many(const Tq *t, const f32 *Q, i32 nq, i32 R, i32 *rows, i32 *cnt, const u64 *dead, i32 nthreads, i32 probe)
{
    TqManyJob j;
    i32 i;
    if (t->n == 0 || nq <= 0 || R <= 0) { for (i = 0; i < nq; ++i) cnt[i] = 0; return; }
    j.t = t; j.Q = Q; j.R = R; j.rows = rows; j.cnt = cnt; j.dead = dead;
    j.np = t->nc > 1 ? (probe > 0 ? probe : t->nprobe) : 0;
    if (j.np > t->nc) j.np = t->nc;
    tq_par2_(tq_many_range_, &j, nq, nthreads);
}

/* ---- rows added after the build ---- */

/* rows prepared for tq_add_put: the closest cluster of each (0 when flat), its squared distance to that centroid, and
 * its code bytes (vd a row) */
typedef struct { i32 m; i32 *cl; f32 *d2; u8 *cd; } TqAdd;

typedef struct { const Tq *t; const TqCol *rows; i64 r0; TqAdd *a; } TqPrepJob;

static void tq_prep_rows_(void *p, i64 a, i64 b, i32 tid)
{
    TqPrepJob *j = (TqPrepJob *)p;
    const Tq *t = j->t;
    i32 dim = t->dim, r, *cl = j->a->cl;
    f32 *x = (f32 *)malloc((size_t)8 * (size_t)dim * sizeof(f32));
    f32 *rot = (f32 *)malloc((size_t)t->dpad * sizeof(f32)), *tmp = (f32 *)malloc((size_t)t->dpad * sizeof(f32));
    u8 *scr = t->nc > 1 ? (u8 *)malloc(tq_probe_scr_(t)) : (u8 *)0;
#if TQ_VNNI
    i32 ng16 = (t->nc + 15) / 16 * 16;
    u32 *qws = t->c8 ? (u32 *)malloc((size_t)8 * (size_t)t->c8d4 * sizeof(u32)) : (u32 *)0, *qp[8];
    i32 *s8 = t->c8 ? (i32 *)malloc((size_t)8 * (size_t)ng16 * sizeof(i32)) : (i32 *)0, *sp[8];
#endif
    i64 i;
    (void)tid;
    for (i = a; i < b; i += 8) {
        i32 g = b - i < 8 ? (i32)(b - i) : 8;
        for (r = 0; r < g; ++r) {
            f32 *dst = x + (size_t)r * dim;
            const f32 *src = tq_row_(j->rows, j->r0 + i + r, dst);
            if (src != dst) memcpy(dst, src, (size_t)dim * sizeof(f32));
        }
        if (t->nc > 1) {
#if TQ_VNNI
            if (t->c8 && g == 8) {                               /* 8 rows per pass over the int8 centroids, like a batch */
                u64 *key = (u64 *)(void *)scr, *ktmp = key + ng16;
                i32 *cand = (i32 *)(void *)(ktmp + ng16) + ng16;
                for (r = 0; r < 8; ++r) { qp[r] = qws + (size_t)r * t->c8d4; sp[r] = s8 + (size_t)r * ng16; tq_q8_(t, x + (size_t)r * dim, qp[r]); }
                tq_score8x8_(t, qp, sp);
                for (r = 0; r < 8; ++r) tq_pick_(key, tq_cands_(t, x + (size_t)r * dim, 1, sp[r], key, cand), 1, ktmp, cl + i + r, 0);
            } else
#endif
            for (r = 0; r < g; ++r) tq_probe_(t, x + (size_t)r * dim, 1, scr, cl + i + r);
            for (r = 0; r < g; ++r) j->a->d2[i + r] = 2.0f - 2.0f * vs_dot(x + (size_t)r * dim, t->cent + (size_t)cl[i + r] * dim, dim);
        } else for (r = 0; r < g; ++r) { cl[i + r] = 0; j->a->d2[i + r] = 0.0f; }
        for (r = 0; r < g; ++r) { tq_rot_(t, x + (size_t)r * dim, rot, tmp); tq_encode_row_(t, rot, j->a->cd + (size_t)(i + r) * (size_t)t->vd); }
    }
    free(x); free(rot); free(tmp); free(scr);
#if TQ_VNNI
    free(qws); free(s8);
#endif
}

void tq_add_free(TqAdd *a) { free(a->cl); free(a->d2); free(a->cd); memset(a, 0, sizeof(*a)); }

/* Rows r0 .. r0 + m - 1 of `rows` (unit length) prepared for tq_add_put, on nth threads. Only reads t, so it can run
 * while searches do, as long as nothing adds to t or replaces it meanwhile. 0, or -1 when memory runs out. */
i32 tq_add_prep(const Tq *t, const TqCol *rows, i64 r0, i32 m, i32 nth, TqAdd *a)
{
    TqPrepJob j;
    size_t n1 = (size_t)(m > 0 ? m : 1);
    a->m = m > 0 ? m : 0;
    a->cl = (i32 *)malloc(n1 * sizeof(i32)); a->d2 = (f32 *)malloc(n1 * sizeof(f32)); a->cd = (u8 *)malloc(n1 * (size_t)t->vd);
    if (!a->cl || !a->d2 || !a->cd) { tq_add_free(a); return -1; }
    j.t = t; j.rows = rows; j.r0 = r0; j.a = a;
    if (a->m) tq_par_(tq_prep_rows_, &j, a->m, nth);
    return 0;
}

/* a new extra block at the end of cluster c's chain: 0, or -1 when memory runs out */
static i32 tq_xblock_(Tq *t, i32 c)
{
    i32 b, k, nh = t->nc > 1 ? t->nc : 1;
    if (!t->xhead) {
        t->xhead = (i32 *)malloc((size_t)nh * sizeof(i32)); t->xtail = (i32 *)malloc((size_t)nh * sizeof(i32));
        if (!t->xhead || !t->xtail) { free(t->xhead); free(t->xtail); t->xhead = t->xtail = (i32 *)0; return -1; }
        for (k = 0; k < nh; ++k) t->xhead[k] = t->xtail[k] = -1;
    }
    if (t->xn == t->xcap) {                                      /* grow by half: an add copies O(1) bytes on average */
        i32 cap = t->xcap + t->xcap / 2 + 64;
        u8 *xc = (u8 *)realloc(t->xcodes, (size_t)cap * (size_t)t->vd * 16);
        i32 *xs, *xx;
        if (!xc) return -1;
        t->xcodes = xc;
        if (!(xs = (i32 *)realloc(t->xslot, (size_t)cap * 32 * sizeof(i32)))) return -1;
        t->xslot = xs;
        if (!(xx = (i32 *)realloc(t->xnext, (size_t)cap * sizeof(i32)))) return -1;
        t->xnext = xx;
        t->xcap = cap;
    }
    b = t->xn++;
    memset(t->xcodes + (size_t)b * (size_t)t->vd * 16, 0, (size_t)t->vd * 16);
    for (k = 0; k < 32; ++k) t->xslot[(size_t)b * 32 + (size_t)k] = -1;
    t->xnext[b] = -1;
    if (t->xtail[c] >= 0) t->xnext[t->xtail[c]] = b; else t->xhead[c] = b;
    t->xtail[c] = b;
    return 0;
}

/* Rows r0 .. r0 + m - 1, prepared by tq_add_prep against this same t, into the index: r0 must be t->n (rows go in in
 * order). A row takes a free slot of its cluster's last block, or opens an extra block. Writes t: no search may run
 * meanwhile. 0, or -1 (r0 out of order, or memory ran out - the rows placed until then stay). */
i32 tq_add_put(Tq *t, const TqAdd *a, i64 r0)
{
    i32 i, d, vd = t->vd;
    if (r0 != (i64)t->n) return -1;
    for (i = 0; i < a->m; ++i) {
        i32 row = (i32)(r0 + i), c = t->nc > 1 ? a->cl[i] : 0, b = -1, k = 32;
        const u8 *cd = a->cd + (size_t)i * (size_t)vd;
        u8 *blk;
        if (t->xtail && t->xtail[c] >= 0) {                      /* the cluster already spilled into extra blocks */
            i32 xb = t->xtail[c];
            for (k = 0; k < 32 && t->xslot[(size_t)xb * 32 + (size_t)k] >= 0; ++k) {}
            b = t->nblk + xb;
        } else {                                                 /* else the padding of its last built block */
            i32 b0 = t->nc > 1 ? t->cstart[c] : 0, last = t->nc > 1 ? t->cstart[c + 1] - 1 : t->nblk - 1;
            if (last >= b0) {
                if (t->slot_row) for (k = 0; k < 32 && t->slot_row[(size_t)last * 32 + (size_t)k] >= 0; ++k) {}
                else k = row - last * 32;                        /* flat, slot = row */
                b = last;
            }
        }
        if (b < 0 || k < 0 || k >= 32) {                         /* full, or no block yet: open an extra one */
            if (tq_xblock_(t, c) != 0) return -1;
            b = t->nblk + t->xtail[c]; k = 0;
        }
        blk = (u8 *)tq_bptr_(t, b);
        if (k < 16) for (d = 0; d < vd; ++d) { u8 *q = blk + (size_t)d * 16 + (size_t)k; *q = (u8)((*q & 0xf0) | cd[d]); }
        else for (d = 0; d < vd; ++d) { u8 *q = blk + (size_t)d * 16 + (size_t)(k - 16); *q = (u8)((*q & 0x0f) | (cd[d] << 4)); }
        if (b >= t->nblk) t->xslot[(size_t)(b - t->nblk) * 32 + (size_t)k] = row;
        else if (t->slot_row) t->slot_row[(size_t)b * 32 + (size_t)k] = row;
        t->n = row + 1;
        t->fit_add += (f64)a->d2[i];
        ++t->nadd;
        if (t->csize && t->nc > 1) { t->csq += 2.0 * t->csize[c] + 1.0; ++t->csize[c]; t->csum += 1.0; }
    }
    return 0;
}

/* what an index that came from a file lacks (neither is saved): fit0 - the mean squared distance of up to 4096 of its
 * rows to their centroids, read through rows - and the cluster sizes, with skew0 taken from them */
void tq_fit0(Tq *t, const TqCol *rows)
{
    i64 ns = (i64)t->nblk * 32, sl, step = ns > 4096 ? ns / 4096 : 1;
    f64 sum = 0.0;
    i32 got = 0, c = 0;
    f32 *rb;
    if (t->nc <= 1 || !t->slot_row || ns == 0) return;
    rb = (f32 *)malloc((size_t)t->dim * sizeof(f32));
    for (sl = 0; sl < ns; sl += step) {
        i32 r = t->slot_row[sl], b = (i32)(sl / 32);
        if (r < 0 || r >= t->n) continue;
        while (c < t->nc - 1 && t->cstart[c + 1] <= b) ++c;    /* slots in order: the cluster only moves forward */
        sum += 2.0 - 2.0 * (f64)vs_dot(tq_row_(rows, r, rb), t->cent + (size_t)c * t->dim, t->dim);
        ++got;
    }
    t->fit0 = got ? (f32)(sum / got) : 0.0f;
    free(rb);
    if (!t->csize) { tq_count_(t); t->skew0 = tq_skew(t); }
}

static i32 tq_pop64_(u64 x)
{
#if defined(__GNUC__)
    return __builtin_popcountl((unsigned long)x);
#else
    i32 c = 0;
    for (; x; x &= x - 1u) ++c;
    return c;
#endif
}

/* The caller dropped its dead rows and moved the rest down, order kept. Every slot follows its row: row r becomes
 * r minus the dead rows before it, a dead row's slot goes free, and the index covers nlive rows. dead: the bitset
 * from before (rows [0, t->n)); pre[w]: dead rows in its words [0, w). A flat index gets an explicit slot_row. */
void tq_remap(Tq *t, const u64 *dead, const i32 *pre, i32 nlive)
{
    i64 s, ns = (i64)t->nblk * 32;
    if (!t->slot_row) {
        t->slot_row = (i32 *)malloc((size_t)(ns ? ns : 1) * sizeof(i32));
        for (s = 0; s < ns; ++s) t->slot_row[s] = s < (i64)t->n ? (i32)s : -1;
        t->slot_own = t->mapped;
    }
    for (s = 0; s < ns + (i64)t->xn * 32; ++s) {
        i32 *p = s < ns ? &t->slot_row[s] : &t->xslot[s - ns], r = *p;
        if (r < 0 || r >= t->n || TQ_SKIP(dead, r)) { *p = -1; continue; }
        *p = r - pre[r >> 6] - tq_pop64_(dead[r >> 6] & (((u64)1 << (r & 63)) - 1u));
    }
    t->n = nlive;
    if (t->csize) tq_count_(t);                                  /* the dead rows left their clusters */
}

#endif /* TQ_C */
