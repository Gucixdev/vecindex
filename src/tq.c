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
 * built to scale: a sample that grows with the cluster count (capped in bytes), e-means (mini-batch k-means with
 * running averages, Meta FAIR 2026) at two levels once there are more than TQ_FLAT_MAX clusters, the cells of the
 * second level on threads of their own, rows placed by looking in their 4 closest top cells, a polish round,
 * centroid scoring 16 centroids per instruction, and assignment and encoding spread over threads.
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
 *   tq_search(tq, q, R, rows, dead, filt, nthreads)      the R best rows by approximate inner product
 *   tq_search_many(tq, Q, nq, R, ...)                 the same for a batch, cluster by cluster
 *   tq_add_prep / tq_add_put                          rows n, n+1, ... into the index (prep reads only: no lock)
 *   tq_free
 * C89, AVX2 / AVX-512 (VNNI) kernels when compiled with -march=native, NEON on ARM, scalar otherwise; threads via
 * pthreads. */
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
/* what a scan keeps: rows whose i64 tag (meta) has every bit of all and one of any (any 0: no condition), whose
 * i64 number (num) lies in [lo, hi], and whose i64 namespace (ns) is nsv - or, when nns > 0, one of nss[0, nns), sorted
 * and without repeats; a column NULL: no such condition */
typedef struct { const TqCol *meta; i64 all, any; const TqCol *num; i64 lo, hi; const TqCol *ns; i64 nsv; const i64 *nss; i32 nns; } TqFilt;

/* v in set[0, n), sorted ascending without repeats (n >= 1): a lower bound without branches on the data, log2(n) steps.
 * lean4/Vecindex/NsSet.lean proves it equals membership. */
static i32 tq_in_set_(const i64 *set, i32 n, i64 v)
{
    const i64 *b = set;
    i32 len = n;
    while (len > 1) {
        i32 h = len >> 1;
        b += (b[h - 1] < v) ? h : 0;                             /* a cmov: the step does not depend on a branch */
        len -= h;
    }
    return *b == v;
}
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
    if (!c->i8) return TQ_ROW(c, r);
    vs_i8_scale(buf, (const i8 *)(const void *)TQ_AT(c, r), c->dim, *(const f32 *)(const void *)TQ_AT(c->scale, r));
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
    i32   bits;             /* 4: one 4-bit code a coordinate; 2: one 4-bit code a pair of coordinates (2 bits each); 1: one
                             * 4-bit code four coordinates (their signs) */
    i32   vd;               /* code columns a row: dpad (4-bit) or dpad/2 rounded up to even (2-bit) */
    /* the cascade (tq_cascade_on, 4-bit indexes): the 4-bit codes' top bits - the signs of the coordinates - as 1-bit
     * codes in the same slots, scanned first; the 4-bit codes then re-score only the shortlist (tq_search_cascade) */
    u8   *codes1;           /* nblk * vd1 * 16 bytes; NULL: no cascade */
    u8   *xcodes1;          /* [xcap x vd1 x 16], beside xcodes */
    i32   vd1;              /* 1-bit code columns a row */
    i32   c1_own;           /* codes1 was malloc'd (tq_free frees it), not mapped from a file */
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
    i32   cold;             /* 1: mapped codes larger than RAM can be counted on to hold, so the load did not read
                             * them in: a search pages in its probed clusters (measured: bench/cold.c) */
    /* clusters split in place (tq_split.c) */
    i32   cent_own;         /* cent and cstart were copied out of a file mapping to grow: tq_free frees them */
    i32   holes;            /* slots a split emptied (rows moved to another cluster): scanned, never returned */
    i32   nsplit;           /* clusters split since the build */
    u32   sgen;             /* bumped whenever a built slot loses its row: what a cache of slot-order facts checks */
    i32   pnc, pnp;         /* nc and nprobe when the probe count was last set: nprobe follows nc as clusters split */
    /* a second level over the centroids (tq_top.c); NULL below tq_top_min_ clusters */
    f32  *top;              /* [nk1 * dim] top cells, unit */
    i32  *tstart, *tlist;   /* cell g holds the clusters tlist[tstart[g] .. tstart[g + 1]) */
    i32   nk1;
} Tq;

#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512F__)
#define TQ_VNNI 1
#else
#define TQ_VNNI 0
#endif

/* code columns a row: 4-bit one per coordinate; 2-bit one per pair; 1-bit one per four (an even count of columns) */
static i32 tq_vd_(i32 dpad, i32 bits) { return bits == 2 ? ((dpad / 2) + 1) & ~1 : bits == 1 ? ((dpad + 3) / 4 + 1) & ~1 : dpad; }

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
#elif defined(VS_NEON)
                for (; j + 4 <= i + h; j += 4) {
                    float32x4_t a = vld1q_f32(x + j), b = vld1q_f32(x + j + h);
                    vst1q_f32(x + j, vaddq_f32(a, b)); vst1q_f32(x + j + h, vsubq_f32(a, b));
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

/* 1-bit: the Lloyd-Max level of the standard normal for one bit, +-sqrt(2 / pi) */
#define TQ_L1 0.7978846f

/* code columns of a rotated row: 4-bit, one per coordinate; 2-bit, coordinates 2v and 2v+1 share column v as
 * (code of 2v) | (code of 2v+1) << 2; 1-bit, the signs of coordinates 4v..4v+3 as bits 0-3 of column v. The 16-entry
 * tables and the scan kernel stay exactly the same for all three. */
static void tq_encode_row_(const Tq *t, const f32 *rot, u8 *cd)
{
    i32 v;
    if (t->bits == 1) {
        for (v = 0; v < t->vd; ++v) {
            i32 i, c = 0;
            for (i = 0; i < 4; ++i) if (4 * v + i < t->dpad && rot[4 * v + i] > 0.0f) c |= 1 << i;
            cd[v] = (u8)c;
        }
        return;
    }
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
#elif defined(VS_NEON)
    for (; g + 2 <= ngroups; g += 2) {               /* 8 accumulators: two groups of 16 centroids */
        const f32 *c = ct + (size_t)g * gs;
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
        f32 *o = out + (size_t)g * 16;
        for (d = 0; d < dim; ++d) {
            float32x4_t x = vdupq_n_f32(v[d]);
            const f32 *p = c + (size_t)d * 16, *p2 = p + gs;
            a0 = VS_FMAQ(a0, x, vld1q_f32(p));      a1 = VS_FMAQ(a1, x, vld1q_f32(p + 4));
            a2 = VS_FMAQ(a2, x, vld1q_f32(p + 8));  a3 = VS_FMAQ(a3, x, vld1q_f32(p + 12));
            a4 = VS_FMAQ(a4, x, vld1q_f32(p2));     a5 = VS_FMAQ(a5, x, vld1q_f32(p2 + 4));
            a6 = VS_FMAQ(a6, x, vld1q_f32(p2 + 8)); a7 = VS_FMAQ(a7, x, vld1q_f32(p2 + 12));
        }
        vst1q_f32(o, a0); vst1q_f32(o + 4, a1); vst1q_f32(o + 8, a2); vst1q_f32(o + 12, a3);
        vst1q_f32(o + 16, a4); vst1q_f32(o + 20, a5); vst1q_f32(o + 24, a6); vst1q_f32(o + 28, a7);
    }
    for (; g < ngroups; ++g) {
        const f32 *c = ct + (size_t)g * gs;
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
        f32 *o = out + (size_t)g * 16;
        for (d = 0; d < dim; ++d) {
            float32x4_t x = vdupq_n_f32(v[d]);
            const f32 *p = c + (size_t)d * 16;
            a0 = VS_FMAQ(a0, x, vld1q_f32(p));      a1 = VS_FMAQ(a1, x, vld1q_f32(p + 4));
            a2 = VS_FMAQ(a2, x, vld1q_f32(p + 8));  a3 = VS_FMAQ(a3, x, vld1q_f32(p + 12));
        }
        vst1q_f32(o, a0); vst1q_f32(o + 4, a1); vst1q_f32(o + 8, a2); vst1q_f32(o + 12, a3);
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

/* ---- spherical e-means: the same job as tq_kmeans_ in fewer passes ----
 * e-means (Meta FAIR, 2026; arXiv 2611.09501; reference code faiss/contrib/e_means, MIT) keeps Lloyd's
 * assign-then-average step but takes it on mini-batches, folding each batch into an exponential moving average of
 * every cluster's count N and sum S (the centroid is S / N):
 *   N <- (1 - a) N + a n,   S <- (1 - a) S + a s     (n, s: the batch's count and sum for the cluster)
 * kept here as mass N and position c = S / N, which the reference does too (lean4/Vecindex/Emeans.lean proves the two
 * forms equal, and that a = 1 on a whole pass is one Lloyd round). The batch size B and the first step a0 come from
 * the paper's fitted laws (no tuning), a falls along a cosine to a0 / 10, a cluster whose mass drops under a tenth of
 * the mean is revived every 5 steps by splitting a busy one (picked in proportion to its mass), and the last pass's
 * exact means are the result. Spherical, as the reference's spherical mode: a centroid is used rescaled to norm 1,
 * rows go to the largest dot product. Epochs: VECINDEX_KMEANS=emeans:T (default 3); VECINDEX_KMEANS=lloyd picks
 * tq_kmeans_. */
#define TQ_EM_PI          3.14159265358979323846
#define TQ_EM_EPOCHS      3.0
#define TQ_EM_INIT_MASS   (1.0f / 65536.0f)  /* the reference's 2^-16 */
#define TQ_EM_REVIVE      0.1f               /* revive below this share of the mean mass */
#define TQ_EM_REVIVE_EVERY 5
#define TQ_EM_SPLIT_EPS   1e-4f

/* nearest power of two of v (1 below 1), as the reference rounds the batch */
static f64 tq_em_pow2_(f64 v)
{
    f64 lo;
    if (v <= 1.0) return 1.0;
    lo = pow(2.0, floor(log(v) / log(2.0)));
    if (lo * 2.0 <= v) lo *= 2.0;                           /* log rounding on either side of an exact power */
    if (lo > v) lo *= 0.5;
    return v / lo < (2.0 * lo) / v ? lo : 2.0 * lo;
}

/* the batch size and first step of n rows, k clusters, T passes in dim dimensions (the reference's shipped fits) */
static void tq_em_recipe_(i32 n, i32 k, f64 T, i32 dim, i32 *bout, f64 *a0out)
{
    f64 b, a0;
    if (dim > 16) {                                         /* "throughput" law, capped at a power of two near n / 2 */
        b = tq_em_pow2_(0.035134 * pow((f64)k, 0.134749) * pow((f64)n, 0.876473) * pow(T, 0.813344));
        if (b > tq_em_pow2_((f64)n / 2.0)) b = tq_em_pow2_((f64)n / 2.0);
    } else {                                                /* "quality" law (low dimension), capped at n / 2 */
        b = tq_em_pow2_(0.00439175 * pow((f64)k, 0.134749) * pow((f64)n, 0.876473) * pow(T, 0.813344));
        if (b > (f64)(n / 2)) b = (f64)(n / 2);
    }
    if (b > (f64)n) b = (f64)n;
    if (b < 1.0) b = 1.0;
    a0 = 0.8244 * pow((f64)k, -0.4617) * pow(b, 0.9003) * pow((f64)n, -0.4498) * pow(T, -0.1057);
    *bout = (i32)b;
    *a0out = a0 < 1e-4 ? 1e-4 : (a0 > 1.0 ? 1.0 : a0);
}

/* the rows of each pass of T passes over m rows into rows[] (T <= 60), their count returned: whole passes, then a pass
 * over a share of the rows for a fractional T, then a whole closing pass (the reference's _epoch_rows) */
static i32 tq_em_passes_(i32 m, f64 T, i64 *rows)
{
    i32 whole = (i32)floor(T), rem = (i32)floor((T - whole) * m + 0.5), n = 0, e;
    if (whole == 0) { rows[n++] = rem > 0 ? rem : 1; return n; }
    for (e = 0; e < whole - 1; ++e) rows[n++] = m;
    if (rem > 0) rows[n++] = rem;
    rows[n++] = m;
    return n;
}

/* the step a of step s of total: a0 down a cosine to a0 / 10 */
static f64 tq_em_alpha_(f64 a0, i64 s, i64 total)
{
    f64 f = (f64)(s < total ? s : total) / (f64)(total > 1 ? total : 1);
    return a0 * (0.1 + 0.9 * 0.5 * (1.0 + cos(TQ_EM_PI * f)));
}

/* out[nc x dim] = c rescaled to norm 1 (a zero row stays zero) */
static void tq_em_unit_(const f32 *c, i32 nc, i32 dim, f32 *out)
{
    i32 j, d;
    for (j = 0; j < nc; ++j) {
        const f32 *s = c + (size_t)j * dim;
        f32 *o = out + (size_t)j * dim;
        f64 q = 0.0;
        for (d = 0; d < dim; ++d) q += (f64)s[d] * s[d];
        q = q > 0.0 ? 1.0 / sqrt(q) : 1.0;
        for (d = 0; d < dim; ++d) o[d] = (f32)(s[d] * q);
    }
}

typedef struct { f32 key; i32 j; } TqEmKey;
static int tq_em_key_cmp_(const void *a, const void *b)
{
    f32 x = ((const TqEmKey *)a)->key, y = ((const TqEmKey *)b)->key;
    return x > y ? -1 : (x < y ? 1 : 0);                    /* largest first */
}

/* Revive the clusters whose mass fell under TQ_EM_REVIVE of the mean: each takes half of a busy cluster, picked
 * without replacement in proportion to its mass (Gumbel keys, as the reference), the two pushed 1e-4 apart. */
static i32 tq_em_revive_(f32 *c, f32 *mass, i32 nc, i32 dim, u32 *rng, TqEmKey *key)
{
    f64 tot = 0.0;
    f32 thr;
    i32 j, d, nu = 0, nd = 0, ns, t;
    for (j = 0; j < nc; ++j) tot += mass[j];
    thr = (f32)(tot * TQ_EM_REVIVE / nc);
    for (j = 0; j < nc; ++j) {
        if (mass[j] < thr) ++nu;
        else if (mass[j] > 0.0f) {
            f64 u = ((f64)tq_rng_(rng) + 1.0) / 4294967296.0;    /* (0, 1] */
            key[nd].key = (f32)(log((f64)mass[j]) - log(-log(u) > 1e-300 ? -log(u) : 1e-300)); key[nd].j = j; ++nd;
        }
    }
    ns = nu < nd ? nu : nd;
    if (ns == 0) return 0;
    qsort(key, (size_t)nd, sizeof(TqEmKey), tq_em_key_cmp_);
    for (j = 0, t = 0; j < nc && t < ns; ++j) {
        f32 *src, *dst;
        i32 same = 1;
        if (mass[j] >= thr) continue;
        src = c + (size_t)key[t].j * dim; dst = c + (size_t)j * dim;
        for (d = 0; d < dim; ++d) {
            f32 e = (d & 1) ? -TQ_EM_SPLIT_EPS : TQ_EM_SPLIT_EPS, v = src[d];
            src[d] = v * (1.0f - e); dst[d] = v * (1.0f + e);
            same &= src[d] == dst[d];
        }
        if (same) dst[0] = dst[0] > 0.0f ? dst[0] * (1.0f + 1e-6f) + 1e-30f : dst[0] + 1e-30f;   /* a zero donor */
        mass[key[t].j] *= 0.5f; mass[j] = mass[key[t].j];
        ++t;
    }
    return ns;
}

/* One step, the paper's line 8: every mass decays by 1 - a (not at a = 1, where a touched cluster's mass is just its
 * count), a cluster the batch touched (cnt rows summing to sum) moves to c + w (sum / cnt - c) with w = a cnt / N', its
 * new mass N' = (1 - a) N + a cnt: the sum-and-count average written for position and mass. psum, pcnt (NULL psum: not
 * the last pass) collect the exact sums for the closing means. */
static void tq_em_update_(f32 *pos, f32 *mass, const f32 *sum, const i32 *cnt, i32 nc, i32 dim, f64 a, f64 *psum, f64 *pcnt)
{
    i32 c, d;
    if (a < 1.0) for (c = 0; c < nc; ++c) mass[c] = (f32)(mass[c] * (1.0 - a));
    for (c = 0; c < nc; ++c) {
        f32 *p = pos + (size_t)c * dim;
        const f32 *s = sum + (size_t)c * dim;
        f64 mm, w, inv;
        if (cnt[c] == 0) continue;
        if (psum) { for (d = 0; d < dim; ++d) psum[(size_t)c * dim + d] += s[d]; pcnt[c] += cnt[c]; }
        mm = (a < 1.0 ? (f64)mass[c] : 0.0) + a * cnt[c];
        w = a * cnt[c] / mm; inv = 1.0 / cnt[c];
        if (w >= 1.0) for (d = 0; d < dim; ++d) p[d] = (f32)(s[d] * inv);
        else for (d = 0; d < dim; ++d) p[d] = (f32)(p[d] + w * (s[d] * inv - p[d]));
        mass[c] = (f32)mm;
    }
}

/* nc unit centroids of the m rows of x into cent[nc x dim], as tq_kmeans_ (same seeding, same optional final
 * assignment), in T passes of mini-batches */
static void tq_emeans_(const f32 *x, i32 m, i32 dim, i32 nc, u32 *rng, f32 *cent, i32 *asg_out, i32 nth, f64 T)
{
    i32 nt = nth < 1 ? 1 : (nth > 64 ? 64 : nth), B, c, d, ep, nep, *perm, *asg, *cnt, revive = dim > 2;
    i64 total = 0, step = 0, rows_ep[64];
    f64 a0, *psum, *pcnt;
    f32 *pos = (f32 *)malloc((size_t)nc * (size_t)dim * sizeof(f32)), *unit = (f32 *)malloc((size_t)nc * (size_t)dim * sizeof(f32));
    f32 *mass = (f32 *)malloc((size_t)nc * sizeof(f32)), *xb, *sum;
    TqEmKey *key = (TqEmKey *)malloc((size_t)nc * sizeof(TqEmKey));
    TqKm k;
    if (T <= 0.0) T = TQ_EM_EPOCHS;
    if (T > 60.0) T = 60.0;
    tq_em_recipe_(m, nc, T, dim, &B, &a0);
    nep = tq_em_passes_(m, T, rows_ep);
    for (ep = 0; ep < nep; ++ep) total += (rows_ep[ep] + B - 1) / B;
    for (c = 0; c < nc; ++c) memcpy(pos + (size_t)c * dim, x + (size_t)((i64)m * c / nc + (tq_rng_(rng) % (u32)(m / nc > 0 ? m / nc : 1))) * dim, (size_t)dim * sizeof(f32));
    for (c = 0; c < nc; ++c) mass[c] = TQ_EM_INIT_MASS;
    perm = (i32 *)malloc((size_t)m * sizeof(i32)); asg = (i32 *)malloc((size_t)B * sizeof(i32));
    xb = (f32 *)malloc((size_t)B * (size_t)dim * sizeof(f32));
    sum = (f32 *)malloc((size_t)nt * (size_t)nc * (size_t)dim * sizeof(f32)); cnt = (i32 *)malloc((size_t)nt * (size_t)nc * sizeof(i32));
    psum = (f64 *)calloc((size_t)nc * (size_t)dim, sizeof(f64)); pcnt = (f64 *)calloc((size_t)nc, sizeof(f64));
    k.dim = dim; k.nc = nc; k.asg = asg; k.sum = sum; k.cnt = cnt; k.x = xb;
    for (ep = 0; ep < nep; ++ep) {
        i32 i, last = ep == nep - 1;
        i64 b0;
        for (i = 0; i < m; ++i) perm[i] = i;                 /* a fresh shuffle every pass */
        for (i = m - 1; i > 0; --i) { i32 j = (i32)(tq_rng_(rng) % (u32)(i + 1)), z = perm[i]; perm[i] = perm[j]; perm[j] = z; }
        for (b0 = 0; b0 < rows_ep[ep]; b0 += B, ++step) {
            i32 bn = (i32)(rows_ep[ep] - b0 < B ? rows_ep[ep] - b0 : B);
            f64 a = tq_em_alpha_(a0, step, total);
            for (i = 0; i < bn; ++i) memcpy(xb + (size_t)i * dim, x + (size_t)perm[b0 + i] * dim, (size_t)dim * sizeof(f32));
            memset(sum, 0, (size_t)nt * (size_t)nc * (size_t)dim * sizeof(f32)); memset(cnt, 0, (size_t)nt * (size_t)nc * sizeof(i32));
            tq_em_unit_(pos, nc, dim, unit);                     /* every use sees the centroids at norm 1 */
            k.ct = tq_transpose_(unit, nc, dim);
            tq_par_(tq_km_asg_, &k, bn, nt);
            free((void *)k.ct);
            tq_fold_(sum, cnt, nc, dim, nt);
            tq_em_update_(pos, mass, sum, cnt, nc, dim, a, last ? psum : (f64 *)0, pcnt);
            if (revive && (step + 1) % TQ_EM_REVIVE_EVERY == 0) tq_em_revive_(pos, mass, nc, dim, rng, key);
        }
    }
    for (c = 0; c < nc; ++c) if (pcnt[c] > 0.0)               /* the closing means: the last pass's exact averages */
        for (d = 0; d < dim; ++d) pos[(size_t)c * dim + d] = (f32)(psum[(size_t)c * dim + d] / pcnt[c]);
    tq_em_unit_(pos, nc, dim, cent);
    free(xb); free(asg); free(perm); free(psum); free(pcnt);
    if (asg_out) {
        k.x = x; k.asg = asg_out; k.sum = (f32 *)0; k.cnt = (i32 *)0; k.ct = tq_transpose_(cent, nc, dim);
        tq_par_(tq_km_asg_, &k, m, nt); free((void *)k.ct);
    }
    free(sum); free(cnt); free(pos); free(unit); free(mass); free(key);
}

/* the clustering every build uses: e-means by default, Lloyd's 8 rounds with VECINDEX_KMEANS=lloyd; emeans:T sets
 * the passes (read on every call, so a test or a benchmark can switch it per build) */
static void tq_cluster_(const f32 *x, i32 m, i32 dim, i32 nc, u32 *rng, f32 *cent, i32 *asg_out, i32 nth)
{
    const char *e = getenv("VECINDEX_KMEANS");
    f64 T = TQ_EM_EPOCHS;
    if (e && !strncmp(e, "lloyd", 5)) { tq_kmeans_(x, m, dim, nc, rng, cent, asg_out, nth); return; }
    if (e && !strncmp(e, "emeans:", 7)) T = atof(e + 7);
    if (nc < 2 || m < 2 * nc) { tq_kmeans_(x, m, dim, nc, rng, cent, asg_out, nth); return; }   /* nothing to batch */
    tq_emeans_(x, m, dim, nc, rng, cent, asg_out, nth, T);
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

/* the sub-clusters of a cell: its rows side by side (buf: the largest cell), clustered on one thread */
typedef struct { TqClus *cl; const f32 *x, *top; const i32 *cnt, *ord; i32 dim, k1, big, nth; u32 seed; } TqCellJob;

static void tq_cell_job_(void *p, i64 a, i64 b, i32 tid)
{
    TqCellJob *cj = (TqCellJob *)p;
    TqClus *cl = cj->cl;
    i32 dim = cj->dim, j;
    f32 *buf = (f32 *)malloc((size_t)cj->big * (size_t)dim * sizeof(f32));
    (void)a; (void)b;
    for (j = tid; j < cj->k1; j += cj->nth) {
        i32 cn = cj->cnt[j + 1] - cj->cnt[j], z;
        u32 rr = cj->seed ^ ((u32)(j + 1) * 0x9e3779b9u);
        f32 *dst = cl->cent + (size_t)cl->sub0[j] * dim;
        if (!rr) rr = 1u;
        if (cn == 0 || cl->subn[j] == 1) memcpy(dst, cj->top + (size_t)j * dim, (size_t)dim * sizeof(f32));   /* an empty or single cell: its own centre */
        else {
            for (z = 0; z < cn; ++z) memcpy(buf + (size_t)z * dim, cj->x + (size_t)cj->ord[cj->cnt[j] + z] * dim, (size_t)dim * sizeof(f32));
            tq_cluster_(buf, cn, dim, cl->subn[j], &rr, dst, (i32 *)0, 1);
        }
        cl->sub_ct[j] = tq_transpose_(dst, cl->subn[j], dim);
    }
    free(buf);
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
        tq_cluster_(x, m, dim, nc, &r, cl->cent, (i32 *)0, nth);
        cl->flat_ct = tq_transpose_(cl->cent, nc, dim);
    } else {
        i32 k1 = (i32)sqrt((f64)nc), j, tot = 0, *asg = (i32 *)malloc((size_t)m * sizeof(i32)), *cnt, *ord, *at;
        f64 t0 = tq_now_(), tsub;
        f32 *top = (f32 *)malloc((size_t)k1 * (size_t)dim * sizeof(f32));
        {   /* the top cells need far fewer rows than the sub-clusters: k-means on every st-th sample row (at most 1024
             * a cell), then one pass puts every sample row in its cell */
            i32 st = m / (k1 * 1024) > 1 ? m / (k1 * 1024) : 1, ms = (m + st - 1) / st;
            if (st == 1) tq_cluster_(x, m, dim, k1, &r, top, asg, nth);
            else {
                f32 *xs = (f32 *)malloc((size_t)ms * (size_t)dim * sizeof(f32));
                TqKm k;
                for (i = 0; i < ms; ++i) memcpy(xs + (size_t)i * dim, x + (size_t)i * st * dim, (size_t)dim * sizeof(f32));
                tq_cluster_(xs, ms, dim, k1, &r, top, (i32 *)0, nth);
                free(xs);
                k.x = x; k.dim = dim; k.nc = k1; k.asg = asg; k.sum = (f32 *)0; k.cnt = (i32 *)0; k.ct = tq_transpose_(top, k1, dim);
                tq_par_(tq_km_asg_, &k, m, nth);
                free((void *)k.ct);
            }
        }
        tsub = tq_now_();
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
        {   /* the cells are independent: the threads take every nth-th one, each cell with a generator of its own,
             * so the result does not depend on the thread count */
            TqCellJob cj;
            cj.cl = cl; cj.x = x; cj.top = top; cj.cnt = cnt; cj.ord = ord; cj.dim = dim; cj.k1 = k1; cj.seed = r;
            cj.nth = nth < 1 ? 1 : (nth > 64 ? 64 : nth); cj.big = 1;
            for (j = 0; j < k1; ++j) if (cnt[j + 1] - cnt[j] > cj.big) cj.big = cnt[j + 1] - cnt[j];
            tq_par2_(tq_cell_job_, &cj, cj.nth, cj.nth);
        }
        cl->top_ct = tq_transpose_(top, k1, dim);
        free(top); free(asg); free(cnt); free(ord); free(at);
        if (getenv("TQ_VERBOSE")) fprintf(stderr, "tq_clus_train_: top cells %.2fs, sub-clusters %.2fs\n", tsub - t0, tq_now_() - tsub);
    }
    cl->T = cl->k1 > 128 ? 2 : TQ_TMAX;
    if (cl->T > cl->k1) cl->T = cl->k1 ? cl->k1 : 1;
    cl->wide = cl->nc;
    if (cl->k1) { i32 j; cl->wide = cl->k1; for (j = 0; j < cl->k1; ++j) if (cl->subn[j] > cl->wide) cl->wide = cl->subn[j]; }
    if (cl->k1) { f64 tp = tq_now_(); tq_clus_polish_(cl, x, m, TQ_POLISH, nth); if (getenv("TQ_VERBOSE")) fprintf(stderr, "tq_clus_train_: polish %.2fs\n", tq_now_() - tp); }
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
    t->bits = bits == 2 || bits == 1 ? bits : 4; t->vd = tq_vd_(t->dpad, t->bits);
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
#include "tq_top.c"

static void tq_prepare_(Tq *t)
{
    if (t->nc > 1 && !t->ct) t->ct = tq_transpose_(t->cent, t->nc, t->dim);
    tq_c8_(t);
    if (t->nc > 1) tq_top_build_(t);
}

void tq_free(Tq *t)
{
    if (!t) return;
    if (!t->mapped) { free(t->s1); free(t->s2); free(t->perm); free(t->codes); free(t->cent); free(t->cstart); free(t->slot_row); }
    else {
        if (t->slot_own) free(t->slot_row);
        if (t->cent_own) { free(t->cent); free(t->cstart); }
    }
    free(t->ct); free(t->c8); free(t->c8b); free(t->c8s);
    tq_top_free_(t);
    free(t->xcodes); free(t->xslot); free(t->xnext); free(t->xhead); free(t->xtail); free(t->csize);
    if (t->c1_own) free(t->codes1);
    free(t->xcodes1);
    free(t);
}

/* bytes of RAM the index holds (0 for the parts that live in a file mapping) */
static i64 tq_ram_(const Tq *t)
{
    i64 b = 0;
    if (!t) return 0;
    if (t->nc > 1) b += (i64)((t->nc + 15) / 16) * 16 * t->dim * 4;
    if (t->top) b += (i64)t->nk1 * t->dim * 4 + (i64)(t->nk1 + 1 + t->nc) * 4;
    if (t->c8) b += (i64)((t->nc + 15) / 16) * (t->c8d4 * 64 + 64) + (i64)t->c8d4 * 16;
    if (!t->mapped) {
        b += (i64)t->nblk * t->vd * 16 + (i64)t->dpad * 12;
        if (t->slot_row) b += (i64)t->nblk * 32 * 4;
        if (t->nc > 1) b += (i64)t->nc * t->dim * 4 + (i64)(t->nc + 1) * 4;
    } else {
        if (t->slot_own) b += (i64)t->nblk * 32 * 4;
        if (t->cent_own) b += (i64)t->nc * t->dim * 4 + (i64)(t->nc + 1) * 4;
    }
    if (t->xcap) b += (i64)t->xcap * ((i64)t->vd * 16 + 32 * 4 + 4) + (i64)(t->nc > 1 ? t->nc : 1) * 8;
    if (t->c1_own) b += (i64)t->nblk * t->vd1 * 16;
    if (t->xcodes1) b += (i64)t->xcap * t->vd1 * 16;
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

/* ---- the cascade: 1-bit codes derived from the 4-bit ones, in the same slots. The top bit of a 4-bit code is its
 * coordinate's sign (TQ_B[7] = 0), exactly what a 1-bit code keeps, so nothing has to be rotated again. ---- */

/* lane k's codes (vd columns) out of a block, and into one */
static void tq_get_lane_(const u8 *blk, i32 vd, i32 k, u8 *cd)
{
    i32 d;
    if (k < 16) for (d = 0; d < vd; ++d) cd[d] = (u8)(blk[(size_t)d * 16 + (size_t)k] & 15u);
    else for (d = 0; d < vd; ++d) cd[d] = (u8)(blk[(size_t)d * 16 + (size_t)(k - 16)] >> 4);
}
static void tq_put_lane_(u8 *blk, i32 vd, i32 k, const u8 *cd)
{
    i32 d;
    if (k < 16) for (d = 0; d < vd; ++d) { u8 *q = blk + (size_t)d * 16 + (size_t)k; *q = (u8)((*q & 0xf0) | cd[d]); }
    else for (d = 0; d < vd; ++d) { u8 *q = blk + (size_t)d * 16 + (size_t)(k - 16); *q = (u8)((*q & 0x0f) | (cd[d] << 4)); }
}

/* a row's 1-bit columns from its 4-bit codes: bit i of column v is the top bit of coordinate 4v + i's code */
static void tq_c1_row_(const u8 *cd4, i32 dpad, i32 vd1, u8 *cd1)
{
    i32 v, i;
    for (v = 0; v < vd1; ++v) {
        i32 c = 0;
        for (i = 0; i < 4; ++i) if (4 * v + i < dpad && (cd4[4 * v + i] & 8u)) c |= 1 << i;
        cd1[v] = (u8)c;
    }
}

/* block b of the 1-bit codes: built (b < nblk) or added later */
static u8 *tq_bptr1_(const Tq *t, i32 b)
{
    size_t bb = (size_t)t->vd1 * 16;
    return b < t->nblk ? t->codes1 + (size_t)b * bb : t->xcodes1 + (size_t)(b - t->nblk) * bb;
}

typedef struct { Tq *t; i32 fail[64]; } TqC1Job;
static void tq_c1_blocks_(void *p, i64 a, i64 b, i32 tid)
{
    TqC1Job *j = (TqC1Job *)p;
    Tq *t = j->t;
    u8 *cd4 = (u8 *)malloc((size_t)t->vd + 16), *cd1 = (u8 *)malloc((size_t)t->vd1 + 16);
    i64 i;
    i32 k;
    if (!cd4 || !cd1) j->fail[tid] = 1;
    else for (i = a; i < b; ++i) {
        u8 *b1 = tq_bptr1_(t, (i32)i);
        memset(b1, 0, (size_t)t->vd1 * 16);
        for (k = 0; k < 32; ++k) {
            tq_get_lane_(tq_bptr_(t, (i32)i), t->vd, k, cd4);
            tq_c1_row_(cd4, t->dpad, t->vd1, cd1);
            tq_put_lane_(b1, t->vd1, k, cd1);
        }
    }
    free(cd4); free(cd1);
}

/* drops the 1-bit codes: the index scans its 4-bit codes again */
void tq_cascade_off(Tq *t)
{
    if (t->c1_own) free(t->codes1);
    free(t->xcodes1);
    t->codes1 = t->xcodes1 = (u8 *)0;
    t->c1_own = 0;
}

/* Gives a 4-bit index its 1-bit codes, derived from the 4-bit ones in nth threads; rows added later get theirs as
 * they come (tq_add_put). 0, or -1: not a 4-bit index, or no memory. Writes t: no search may run meanwhile. */
i32 tq_cascade_on(Tq *t, i32 nth)
{
    TqC1Job j;
    i32 i;
    if (t->bits != 4) return -1;
    if (t->codes1) return 0;
    t->vd1 = tq_vd_(t->dpad, 1);
    t->codes1 = (u8 *)malloc((size_t)(t->nblk > 0 ? t->nblk : 1) * (size_t)t->vd1 * 16);
    if (t->xcap) t->xcodes1 = (u8 *)malloc((size_t)t->xcap * (size_t)t->vd1 * 16);
    t->c1_own = 1;
    if (!t->codes1 || (t->xcap && !t->xcodes1)) { tq_cascade_off(t); return -1; }
    j.t = t;
    memset(j.fail, 0, sizeof(j.fail));
    tq_par_(tq_c1_blocks_, &j, (i64)t->nblk + t->xn, nth);
    for (i = 0; i < 64; ++i) if (j.fail[i]) { tq_cascade_off(t); return -1; }
    return 0;
}

#include "tq_scan.c"

#include "tq_add.c"

#include "tq_split.c"

#endif /* TQ_C */
