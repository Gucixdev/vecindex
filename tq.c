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
 *   Tq *tq_build(vecs, n, dim, seed)      vecs: n x dim f32 (unit length)
 *   tq_search(tq, q, R, rows, filter..., nthreads)   the R best rows by approximate inner product
 *   tq_free
 * C89, AVX2 kernel when compiled with -mavx2 (-march=native), scalar otherwise; threads via pthreads. */
#ifndef TQ_C
#define TQ_C

#include "base.h"
#include "vsimd.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

typedef struct {
    i32   dim, dpad, n, nblk;
    /* optional IVF: rows are grouped by cluster (blocks of 32 slots per cluster, padded), a query scans only the
     * blocks of its nprobe closest clusters. nc == 0: flat, slot s holds row s. */
    i32   nc, nprobe;
    f32  *cent;             /* [nc x dim] unit centroids */
    i32  *cstart;           /* [nc + 1] first block of each cluster */
    i32  *slot_row;         /* [nblk * 32] row held by each slot, -1 = padding */
    f32  *s1, *s2;          /* random +-1 signs, [dpad] */
    i32  *perm;             /* [dpad] */
    u8   *codes;            /* nblk * dpad * 16 bytes */
} Tq;

/* Lloyd-Max centroids of the standard normal, 16 levels ascending */
static const f32 TQ_CENT[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9424f, -0.6568f, -0.3881f, -0.1284f,
     0.1284f,  0.3881f,  0.6568f,  0.9424f,  1.2562f,  1.6180f,  2.0690f,  2.7326f
};

static u32 tq_rng_(u32 *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }

/* Walsh-Hadamard on each power-of-two chunk of x[dpad] (binary decomposition of dpad), orthonormal */
static void tq_wht_(f32 *x, i32 dpad)
{
    i32 base = 0, len, h, i, j;
    for (len = 1; len * 2 <= dpad; len *= 2) {}
    while (base < dpad) {
        while (len > dpad - base) len /= 2;
        for (h = 1; h < len; h *= 2)
            for (i = base; i < base + len; i += 2 * h)
                for (j = i; j < i + h; ++j) { f32 a = x[j], b = x[j + h]; x[j] = a + b; x[j + h] = a - b; }
        { f32 sc = 1.0f / (f32)sqrt((f64)len); for (i = base; i < base + len; ++i) x[i] *= sc; }
        base += len;
    }
}

/* out[dpad] = R * in (in has dim entries, the rest is zero), scaled so coordinates are ~ N(0,1) */
static void tq_rot_(const Tq *t, const f32 *in, f32 *out, f32 *tmp)
{
    i32 i;
    for (i = 0; i < t->dpad; ++i) tmp[i] = (i < t->dim ? in[i] : 0.0f) * t->s1[i];
    tq_wht_(tmp, t->dpad);
    for (i = 0; i < t->dpad; ++i) out[i] = tmp[t->perm[i]] * t->s2[i];
    tq_wht_(out, t->dpad);
    { f32 sc = (f32)sqrt((f64)t->dim); for (i = 0; i < t->dpad; ++i) out[i] *= sc; }
}

static u8 tq_code_(f32 v)
{
    /* nearest centroid: boundaries are the midpoints; centroids are symmetric */
    static const f32 B[15] = { -2.4008f, -1.8435f, -1.4371f, -1.0993f, -0.7996f, -0.5225f, -0.2583f, 0.0f,
                                0.2583f, 0.5225f, 0.7996f, 1.0993f, 1.4371f, 1.8435f, 2.4008f };
    i32 c = 0;
    while (c < 15 && v > B[c]) ++c;
    return (u8)c;
}

/* ---- IVF training (spherical k-means on unit vectors), threaded assignment ---- */
typedef struct { const f32 *vecs; const f32 *cent; i32 dim, nc, a, b; i32 *asg; } TqAsg;

static void *tq_asg_job_(void *arg)
{
    TqAsg *j = (TqAsg *)arg;
    i32 i, c;
    for (i = j->a; i < j->b; ++i) {
        const f32 *v = j->vecs + (size_t)i * j->dim;
        f32 best = -1e30f; i32 bc = 0;
        for (c = 0; c < j->nc; ++c) { f32 s = vs_dot(v, j->cent + (size_t)c * j->dim, j->dim); if (s > best) { best = s; bc = c; } }
        j->asg[i] = bc;
    }
    return (void *)0;
}

static void tq_assign_(const f32 *vecs, i32 n, i32 dim, const f32 *cent, i32 nc, i32 *asg, i32 nthreads)
{
    TqAsg jobs[16]; pthread_t th[16];
    i32 i;
    if (nthreads > 16) nthreads = 16;
    if (nthreads < 1 || n < 20000) nthreads = 1;
    for (i = 0; i < nthreads; ++i) {
        jobs[i].vecs = vecs; jobs[i].cent = cent; jobs[i].dim = dim; jobs[i].nc = nc; jobs[i].asg = asg;
        jobs[i].a = (i32)((i64)n * i / nthreads); jobs[i].b = (i32)((i64)n * (i + 1) / nthreads);
    }
    for (i = 1; i < nthreads; ++i) pthread_create(&th[i], (const pthread_attr_t *)0, tq_asg_job_, &jobs[i]);
    tq_asg_job_(&jobs[0]);
    for (i = 1; i < nthreads; ++i) pthread_join(th[i], (void **)0);
}

/* k-means on a sample: nc unit centroids in cent[nc x dim] */
static void tq_kmeans_(const f32 *vecs, i32 n, i32 dim, i32 nc, u32 seed, f32 *cent, i32 nthreads)
{
    i32 m = n < 40000 ? n : 40000, i, c, d, it;
    f32 *sample = (f32 *)malloc((size_t)m * (size_t)dim * sizeof(f32)), *sum = (f32 *)malloc((size_t)nc * (size_t)dim * sizeof(f32));
    i32 *asg = (i32 *)malloc((size_t)m * sizeof(i32)), *cnt = (i32 *)calloc((size_t)nc, sizeof(i32));
    u32 r = seed ? seed : 99u;
    for (i = 0; i < m; ++i) memcpy(sample + (size_t)i * dim, vecs + (size_t)((i64)n * i / m) * dim, (size_t)dim * sizeof(f32));
    for (c = 0; c < nc; ++c) memcpy(cent + (size_t)c * dim, sample + (size_t)(tq_rng_(&r) % (u32)m) * dim, (size_t)dim * sizeof(f32));
    for (it = 0; it < 8; ++it) {
        tq_assign_(sample, m, dim, cent, nc, asg, nthreads);
        memset(sum, 0, (size_t)nc * (size_t)dim * sizeof(f32)); memset(cnt, 0, (size_t)nc * sizeof(i32));
        for (i = 0; i < m; ++i) { f32 *s = sum + (size_t)asg[i] * dim; const f32 *v = sample + (size_t)i * dim; for (d = 0; d < dim; ++d) s[d] += v[d]; ++cnt[asg[i]]; }
        for (c = 0; c < nc; ++c) {
            f32 nrm = 0.0f, *s = sum + (size_t)c * dim;
            if (cnt[c] == 0) { memcpy(s, sample + (size_t)(tq_rng_(&r) % (u32)m) * dim, (size_t)dim * sizeof(f32)); }      /* empty cluster: reseed */
            for (d = 0; d < dim; ++d) nrm += s[d] * s[d];
            nrm = nrm > 0.0f ? 1.0f / (f32)sqrt((f64)nrm) : 1.0f;
            for (d = 0; d < dim; ++d) cent[(size_t)c * dim + d] = s[d] * nrm;
        }
    }
    free(sample); free(sum); free(asg); free(cnt);
}

/* nc <= 1: flat scan of every row. nc > 1: IVF with nc clusters (nprobe defaults to ~16% of them). */
Tq *tq_build(const f32 *vecs, i32 n, i32 dim, u32 seed, i32 nc)
{
    Tq *t = (Tq *)calloc(1, sizeof(Tq));
    f32 *rot, *tmp;
    i32 i, d, j, nslots, *order = (i32 *)0, *asg = (i32 *)0;
    u32 r = seed ? seed : 0x7154u;
    t->dim = dim; t->dpad = dim + (dim & 1); t->n = n;
    t->s1 = (f32 *)malloc((size_t)t->dpad * sizeof(f32)); t->s2 = (f32 *)malloc((size_t)t->dpad * sizeof(f32));
    t->perm = (i32 *)malloc((size_t)t->dpad * sizeof(i32));
    for (i = 0; i < t->dpad; ++i) { t->s1[i] = (tq_rng_(&r) & 1u) ? 1.0f : -1.0f; t->s2[i] = (tq_rng_(&r) & 1u) ? 1.0f : -1.0f; t->perm[i] = i; }
    for (i = t->dpad - 1; i > 0; --i) { i32 k = (i32)(tq_rng_(&r) % (u32)(i + 1)), x = t->perm[i]; t->perm[i] = t->perm[k]; t->perm[k] = x; }

    if (nc > 1 && n >= nc * 8) {                                 /* group the rows by cluster, each cluster padded to whole blocks */
        i32 c, blk = 0, *cnt = (i32 *)calloc((size_t)nc + 1, sizeof(i32)), *pos;
        i32 nth = 4;
        t->nc = nc;
        t->cent = (f32 *)malloc((size_t)nc * (size_t)dim * sizeof(f32));
        tq_kmeans_(vecs, n, dim, nc, seed, t->cent, nth);
        asg = (i32 *)malloc((size_t)n * sizeof(i32));
        tq_assign_(vecs, n, dim, t->cent, nc, asg, nth);
        for (i = 0; i < n; ++i) ++cnt[asg[i]];
        t->cstart = (i32 *)malloc((size_t)(nc + 1) * sizeof(i32));
        for (c = 0; c < nc; ++c) { t->cstart[c] = blk; blk += (cnt[c] + 31) / 32; }
        t->cstart[nc] = blk;
        t->nblk = blk; nslots = blk * 32;
        t->slot_row = (i32 *)malloc((size_t)nslots * sizeof(i32));
        for (i = 0; i < nslots; ++i) t->slot_row[i] = -1;
        pos = (i32 *)calloc((size_t)nc, sizeof(i32));
        for (i = 0; i < n; ++i) { c = asg[i]; t->slot_row[(size_t)t->cstart[c] * 32 + (size_t)pos[c]++] = i; }
        free(pos); free(cnt); free(asg);
        t->nprobe = nc * 16 / 100 > 4 ? nc * 16 / 100 : (nc < 4 ? nc : 4);      /* ~16% of the clusters: 98.5% recall@10 on real e5 vectors, 3x less data read */
    } else {
        t->nblk = (n + 31) / 32; nslots = t->nblk * 32;
    }
    (void)order;
    t->codes = (u8 *)calloc((size_t)(t->nblk ? t->nblk : 1) * (size_t)t->dpad * 16, 1);
    rot = (f32 *)malloc((size_t)t->dpad * sizeof(f32)); tmp = (f32 *)malloc((size_t)t->dpad * sizeof(f32));
    for (i = 0; i < nslots; ++i) {
        i32 row = t->slot_row ? t->slot_row[i] : (i < n ? i : -1);
        u8 *blk;
        if (row < 0) continue;
        blk = t->codes + (size_t)(i / 32) * (size_t)t->dpad * 16;
        j = i % 32;
        tq_rot_(t, vecs + (size_t)row * dim, rot, tmp);
        for (d = 0; d < t->dpad; ++d) {
            u8 c = tq_code_(rot[d]);
            u8 *p = blk + (size_t)d * 16 + (j & 15);
            if (j < 16) *p = (u8)((*p & 0xf0) | c); else *p = (u8)((*p & 0x0f) | (c << 4));
        }
    }
    free(rot); free(tmp);
    return t;
}

void tq_free(Tq *t)
{
    if (!t) return;
    free(t->s1); free(t->s2); free(t->perm); free(t->codes); free(t->cent); free(t->cstart); free(t->slot_row); free(t);
}

/* ---- scanning ---- */

/* score of the 32 vectors of one block: out[32] u16 (bigger = closer) */
static void tq_block_(const u8 *blk, const u8 *lut, i32 dpad, u16 *out)
{
#if defined(__AVX2__)
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, b0 = a0, b1 = a0;
    const __m256i m0f = _mm256_set1_epi8(0x0f), m00ff = _mm256_set1_epi16(0x00ff);
    u16 A0[16], A1[16], B0[16], B1[16];
    i32 d, k;
    for (d = 0; d < dpad; d += 2) {
        __m256i codes = _mm256_loadu_si256((const __m256i *)(blk + (size_t)d * 16));
        __m256i lv = _mm256_loadu_si256((const __m256i *)(lut + (size_t)d * 16));
        __m256i lo = _mm256_and_si256(codes, m0f);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(codes, 4), m0f);
        __m256i r0 = _mm256_shuffle_epi8(lv, lo), r1 = _mm256_shuffle_epi8(lv, hi);
        a0 = _mm256_add_epi16(a0, _mm256_and_si256(r0, m00ff)); a1 = _mm256_add_epi16(a1, _mm256_srli_epi16(r0, 8));
        b0 = _mm256_add_epi16(b0, _mm256_and_si256(r1, m00ff)); b1 = _mm256_add_epi16(b1, _mm256_srli_epi16(r1, 8));
    }
    _mm256_storeu_si256((__m256i *)A0, a0); _mm256_storeu_si256((__m256i *)A1, a1);
    _mm256_storeu_si256((__m256i *)B0, b0); _mm256_storeu_si256((__m256i *)B1, b1);
    for (k = 0; k < 8; ++k) {
        out[2 * k] = (u16)(A0[k] + A0[8 + k]);          out[2 * k + 1] = (u16)(A1[k] + A1[8 + k]);
        out[16 + 2 * k] = (u16)(B0[k] + B0[8 + k]);     out[16 + 2 * k + 1] = (u16)(B1[k] + B1[8 + k]);
    }
#else
    i32 d, j;
    for (j = 0; j < 32; ++j) out[j] = 0;
    for (d = 0; d < dpad; ++d)
        for (j = 0; j < 32; ++j) {
            u8 byte = blk[(size_t)d * 16 + (j & 15)];
            u8 c = j < 16 ? (u8)(byte & 15) : (u8)(byte >> 4);
            out[j] = (u16)(out[j] + lut[(size_t)d * 16 + c]);
        }
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
    const u8 *dead; const i64 *meta; i64 all, any; i32 filtered;
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
        tq_block_(t->codes + (size_t)blk * (size_t)t->dpad * 16, j->lut, t->dpad, sc);
        for (k = 0; k < 32; ++k) {
            i32 row = t->slot_row ? t->slot_row[base + k] : base + k;
            u64 v;
            if (row < 0 || row >= t->n) continue;                /* padding */
            if (j->heap.n == j->heap.cap && (u64)sc[k] << 32 <= j->heap.h[0]) continue;
            if (j->dead && j->dead[row]) continue;
            if (j->filtered) {
                i64 m = j->meta[row];
                if ((m & j->all) != j->all) continue;
                if (j->any && !(m & j->any)) continue;
            }
            v = ((u64)sc[k] << 32) | (u64)(u32)row;
            tq_heap_push_(&j->heap, v);
        }
    }
    return (void *)0;
}

/* The R best rows (by the quantised inner product) that are alive and pass the meta filter; returns how
 * many were found, rows[] best first. dead/meta may be NULL. nthreads > 1 splits the blocks. */
i32 tq_search(const Tq *t, const f32 *q, i32 R, i32 *rows, const u8 *dead, const i64 *meta, i64 all, i64 any, i32 filtered, i32 nthreads, i32 probe)
{
    i32 *blks = (i32 *)0, nblk_scan;
    f32 *rot = (f32 *)malloc((size_t)t->dpad * sizeof(f32)), *tmp = (f32 *)malloc((size_t)t->dpad * sizeof(f32));
    f32 *lf = (f32 *)malloc((size_t)t->dpad * 16 * sizeof(f32)), maxr = 0.0f, scale;
    u8 *lut = (u8 *)malloc((size_t)t->dpad * 16 + 32);
    TqJob jobs[16];
    pthread_t th[16];
    i32 d, c, i, nj, per, cnt = 0, lutmax;
    u64 *all_h;

    if (t->n == 0 || R <= 0) { free(rot); free(tmp); free(lf); free(lut); return 0; }
    tq_rot_(t, q, rot, tmp);
    for (d = 0; d < t->dpad; ++d) {
        f32 mn = 1e30f, mx = -1e30f;
        for (c = 0; c < 16; ++c) { f32 v = rot[d] * TQ_CENT[c]; lf[d * 16 + c] = v; if (v < mn) mn = v; if (v > mx) mx = v; }
        for (c = 0; c < 16; ++c) lf[d * 16 + c] -= mn;
        if (mx - mn > maxr) maxr = mx - mn;
    }
    lutmax = 65535 / t->dpad; if (lutmax > 255) lutmax = 255;
    scale = maxr > 0.0f ? (f32)lutmax / maxr : 1.0f;
    for (i = 0; i < t->dpad * 16; ++i) { i32 v = (i32)(lf[i] * scale + 0.5f); lut[i] = (u8)(v > lutmax ? lutmax : v); }

    if (t->nc > 1) {                                             /* IVF: the blocks of the nprobe clusters closest to the query */
        i32 np = probe > 0 ? probe : t->nprobe, c, nb = 0, *top = (i32 *)malloc((size_t)np * sizeof(i32));
        f32 *ts = (f32 *)malloc((size_t)np * sizeof(f32));
        i32 got = 0;
        if (np > t->nc) np = t->nc;
        for (c = 0; c < t->nc; ++c) {                            /* the np best centroids by inner product with the query */
            f32 s = vs_dot(q, t->cent + (size_t)c * t->dim, t->dim);
            if (got == np && s <= ts[np - 1]) continue;
            { i32 k = got < np ? got++ : np - 1; while (k > 0 && ts[k - 1] < s) { ts[k] = ts[k - 1]; top[k] = top[k - 1]; --k; } ts[k] = s; top[k] = c; }
        }
        for (i = 0; i < got; ++i) nb += t->cstart[top[i] + 1] - t->cstart[top[i]];
        blks = (i32 *)malloc((size_t)(nb ? nb : 1) * sizeof(i32));
        for (nb = 0, i = 0; i < got; ++i) { i32 b; for (b = t->cstart[top[i]]; b < t->cstart[top[i] + 1]; ++b) blks[nb++] = b; }
        nblk_scan = nb;
        free(top); free(ts);
    } else nblk_scan = t->nblk;
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 16) nthreads = 16;
    if (nblk_scan < 64) nthreads = 1;
    nj = nthreads; per = (nblk_scan + nj - 1) / nj;
    for (i = 0; i < nj; ++i) {
        jobs[i].t = t; jobs[i].lut = lut; jobs[i].blks = blks; jobs[i].b0 = i * per; jobs[i].b1 = (i + 1) * per < nblk_scan ? (i + 1) * per : nblk_scan;
        if (jobs[i].b0 > jobs[i].b1) jobs[i].b0 = jobs[i].b1;
        jobs[i].R = R; jobs[i].dead = dead; jobs[i].meta = meta; jobs[i].all = all; jobs[i].any = any; jobs[i].filtered = filtered;
        jobs[i].heap.h = (u64 *)malloc((size_t)R * sizeof(u64)); jobs[i].heap.n = 0; jobs[i].heap.cap = R;
    }
    for (i = 1; i < nj; ++i) pthread_create(&th[i], (const pthread_attr_t *)0, tq_job_, &jobs[i]);
    tq_job_(&jobs[0]);
    for (i = 1; i < nj; ++i) pthread_join(th[i], (void **)0);

    { TqHeap m; m.h = (u64 *)malloc((size_t)R * sizeof(u64)); m.n = 0; m.cap = R;
      for (i = 0; i < nj; ++i) { i32 k; for (k = 0; k < jobs[i].heap.n; ++k) tq_heap_push_(&m, jobs[i].heap.h[k]); free(jobs[i].heap.h); }
      all_h = m.h; cnt = m.n; }
    /* heap array -> descending order */
    for (i = 0; i < cnt; ++i) { i32 k, best = i; for (k = i + 1; k < cnt; ++k) if (all_h[k] > all_h[best]) best = k; { u64 x = all_h[i]; all_h[i] = all_h[best]; all_h[best] = x; } rows[i] = (i32)(u32)(all_h[i] & 0xffffffffu); }
    free(all_h); free(rot); free(tmp); free(lf); free(lut); free(blks);
    return cnt;
}

#endif /* TQ_C */
