/* qmix.c - per-dimension mixed-depth quantized flat vector index.
 *
 * The five uniform quantizers (q1..q5) spend the same number of bits on
 * every dimension. Real embedding matrices are nowhere near that flat:
 * on a transformer encoder a handful of dimensions carry most of the
 * variance and the rest are near-noise. qmix bins the dimensions by
 * variance and gives each bin its own depth -
 *
 *     top    QMIX_HI_FRAC  of dims -> 4 bits   (reuses q4's machinery)
 *     next  (1-hi-lo)      of dims -> 2 bits   (q2)
 *     bottom QMIX_LO_FRAC  of dims -> 1 bit    (q1)
 *
 * so the average is ~2.15 bits/dim (tunable) but the bits land where
 * they reduce reconstruction error most. It is a non-uniform product
 * quantizer built by composition: three real sub-indexes over disjoint,
 * permuted dimension slabs, their per-query LUT gathers summed. Because
 * the slabs are disjoint the combined ||q-c||^2, ||vhat-c||^2 and
 * ||v-c||^2 are just the sums of the parts, so the same consistent
 * squared-L2 estimate (and the l2_lambda term-3 blend) carries over
 * unchanged.
 *
 * Single file, one caller-supplied Arena, no free - same contract as
 * q1..q5. engine.c can select it as one more candidate topology.
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -I. qmix.c -DQMIX_DEMO -lm -o qmix && ./qmix
 */

#ifdef QMIX_DEMO
#define ARENA_ALLOC_IMPLEMENTATION
#endif
#include "base.h"
#include "arena_alloc.h"
#include "vsimd.h"
#include <stdlib.h>          /* the transient column slab is malloc/free'd,
                             * never left as dead arena space */

/* pull in the three sub-quantizers unless a consumer (engine.c) already
 * did - their headers #define Q1_LEVELS / Q2_LEVELS / Q4_LEVELS. */
#ifndef Q1_LEVELS
#include "q1.c"
#endif
#ifndef Q2_LEVELS
#include "q2.c"
#endif
#ifndef Q4_LEVELS
#include "q4.c"
#endif

#ifndef QMIX_HI_FRAC
#define QMIX_HI_FRAC 0.25f      /* fraction of dims at 4 bits */
#endif
#ifndef QMIX_LO_FRAC
#define QMIX_LO_FRAC 0.35f      /* fraction of dims at 1 bit */
#endif

typedef struct {
    Arena *arena;
    i32 dim;
    i32 code_bytes;             /* hi->code_bytes + mid->code_bytes + lo->code_bytes */
    i32 count;

    i32 *perm;                  /* [dim] storage-order -> original dim index */
    i32 n_hi, n_mid, n_lo;      /* perm split points: [0,n_hi) [n_hi,n_hi+n_mid) ... */

    Q4Index *hi;                /* over perm[0 .. n_hi)          */
    Q2Index *mid;               /* over perm[n_hi .. n_hi+n_mid) */
    Q1Index *lo;                /* over perm[.. dim)             */

    f32 *rn2;                   /* [count] ||v - c||^2  (sum of the three)   */
    f32 *qn2;                   /* [count] ||vhat - c||^2 (sum of the three) */
    f32  l2_lambda;             /* term-3 blend, same meaning as in q1..q5   */
} QMixIndex;

/* ------------------------------------------------------------------ */

static f64 qmix_sqrt_(f64 x)
{
    union { f32 f; u32 i; } u;
    f64 g; i32 i;
    if (x <= 0.0) return 0.0;
    u.f = (f32)x; u.i = (u.i >> 1) + 0x1fbd1df5u; g = (f64)u.f;
    for (i = 0; i < 3; ++i) g = 0.5 * (g + x / g);
    return g;
}

#define QMIX_GID(ids, i) ((ids) ? (size_t)(ids)[i] : (size_t)(i))

/* descending-by-key insertion index sort of 0..dim-1 (dim is small and
 * this runs once per build). */
static void qmix_argsort_desc_(i32 *idx, const f32 *key, i32 dim)
{
    i32 i, j;
    for (i = 0; i < dim; ++i) idx[i] = i;
    for (i = 1; i < dim; ++i) {
        i32 ki = idx[i]; f32 kv = key[ki]; j = i - 1;
        while (j >= 0 && key[idx[j]] < kv) { idx[j + 1] = idx[j]; --j; }
        idx[j + 1] = ki;
    }
}

/* gather the perm[lo..hi) columns of the selected rows into `slab`
 * ([count * (hi-lo)], caller-owned - reused across the three sub-builds). */
static void qmix_slab_(f32 *slab, const f32 *vectors, const i64 *ids, i32 count,
                       i32 dim, const i32 *perm, i32 lo, i32 hi)
{
    i32 sd = hi - lo, i, d;
    for (i = 0; i < count; ++i) {
        const f32 *v = vectors + QMIX_GID(ids, i) * dim;
        f32 *row = slab + (size_t)i * sd;
        for (d = 0; d < sd; ++d) row[d] = v[perm[lo + d]];
    }
}

/* Shared core. ids==NULL -> whole 0..n-1; otherwise the n rows named by
 * ids[] (one IVF cluster). Each sub-quantizer centres on its own slab
 * mean, so no coarse centroid is needed here.
 *
 * Arena discipline: everything kept (perm, the 3 sub-indexes, rn2, qn2)
 * is allocated before the scratch cursor is snapshotted or after it is
 * rewound; the variance workspace and the (single, reused) column slab
 * live strictly between and are handed back. No trailing dead slab -
 * unlike engine_build_cascade. */
QMixIndex *qmix_build_(Arena *a, const f32 *vectors, const i64 *ids,
                       i32 n, i32 dim, f32 hi_frac, f32 lo_frac)
{
    QMixIndex *ix = (QMixIndex *)arena_alloc(a, sizeof(QMixIndex));
    i32 cnt = n > 0 ? n : 1, i, d, nh, nl, nm, maxsub;

    ix->arena = a; ix->dim = dim; ix->count = n;
    ix->l2_lambda = 1.0f;
    ix->perm = (i32 *)arena_alloc(a, (size_t)(dim > 0 ? dim : 1) * sizeof(i32));
    ix->rn2  = (f32 *)arena_alloc(a, (size_t)cnt * sizeof(f32));
    ix->qn2  = (f32 *)arena_alloc(a, (size_t)cnt * sizeof(f32));

    {   /* ---- variance-ranked bit allocation (scratch) ---- */
        Arena_Mark m = arena_snapshot(a);
        f32 *var  = (f32 *)arena_alloc(a, (size_t)(dim > 0 ? dim : 1) * sizeof(f32));
        f32 *mean = (f32 *)arena_alloc(a, (size_t)(dim > 0 ? dim : 1) * sizeof(f32));
        i32 *ord  = (i32 *)arena_alloc(a, (size_t)(dim > 0 ? dim : 1) * sizeof(i32));
        for (d = 0; d < dim; ++d) { mean[d] = 0.0f; var[d] = 0.0f; }
        for (i = 0; i < n; ++i) {
            const f32 *v = vectors + QMIX_GID(ids, i) * dim;
            for (d = 0; d < dim; ++d) mean[d] += v[d];
        }
        if (n > 0) for (d = 0; d < dim; ++d) mean[d] /= (f32)n;
        for (i = 0; i < n; ++i) {
            const f32 *v = vectors + QMIX_GID(ids, i) * dim;
            for (d = 0; d < dim; ++d) { f32 e = v[d] - mean[d]; var[d] += e * e; }
        }
        qmix_argsort_desc_(ord, var, dim);   /* n cancels in the ranking */
        nh = (i32)(hi_frac * (f32)dim + 0.5f);
        nl = (i32)(lo_frac * (f32)dim + 0.5f);
        if (nh < 0) nh = 0;
        if (nh > dim) nh = dim;
        if (nl < 0) nl = 0;
        if (nl > dim - nh) nl = dim - nh;
        nm = dim - nh - nl;
        for (d = 0; d < dim; ++d) ix->perm[d] = ord[d];  /* [hi | mid | lo] */
        arena_rewind(a, m);
    }
    ix->n_hi = nh; ix->n_mid = nm; ix->n_lo = nl;

    {   /* ---- three sub-indexes, each over a transient column slab ---- *
         * the slab is malloc'd so the arena holds only the kept index;
         * qN_build_cluster copies out everything it needs. */
        f32 *slab;
        maxsub = nh > nm ? nh : nm; if (nl > maxsub) maxsub = nl;
        slab = (f32 *)malloc((size_t)cnt * (maxsub > 0 ? maxsub : 1) * sizeof(f32));
        if (!slab) return (QMixIndex *)0;

        qmix_slab_(slab, vectors, ids, n, dim, ix->perm, 0, nh);
        ix->hi  = q4_build_cluster(a, slab, (const i64 *)0, n, nh, (const f32 *)0);
        qmix_slab_(slab, vectors, ids, n, dim, ix->perm, nh, nh + nm);
        ix->mid = q2_build_cluster(a, slab, (const i64 *)0, n, nm, (const f32 *)0);
        qmix_slab_(slab, vectors, ids, n, dim, ix->perm, nh + nm, dim);
        ix->lo  = q1_build_cluster(a, slab, (const i64 *)0, n, nl, (const f32 *)0);
        free(slab);
    }
    ix->code_bytes = ix->hi->code_bytes + ix->mid->code_bytes + ix->lo->code_bytes;

    for (i = 0; i < n; ++i) {
        f32 rh = ix->hi->rnorm[i], rm = ix->mid->rnorm[i], rl = ix->lo->rnorm[i];
        ix->rn2[i] = rh * rh + rm * rm + rl * rl;
        ix->qn2[i] = ix->hi->qn2[i] + ix->mid->qn2[i] + ix->lo->qn2[i];
    }
    return ix;
}

QMixIndex *qmix_build(Arena *a, const f32 *vectors, i32 n, i32 dim)
{ return qmix_build_(a, vectors, (const i64 *)0, n, dim, QMIX_HI_FRAC, QMIX_LO_FRAC); }

QMixIndex *qmix_build_cluster(Arena *a, const f32 *vectors, const i64 *ids,
                              i32 n_ids, i32 dim, const f32 *centroid)
{ (void)centroid; return qmix_build_(a, vectors, ids, n_ids, dim, QMIX_HI_FRAC, QMIX_LO_FRAC); }

i32 qmix_code_bytes(i32 dim)
{
    i32 nh = (i32)(QMIX_HI_FRAC * (f32)dim + 0.5f);
    i32 nl = (i32)(QMIX_LO_FRAC * (f32)dim + 0.5f);
    i32 nm;
    if (nh > dim) nh = dim;
    if (nl > dim - nh) nl = dim - nh;
    nm = dim - nh - nl;
    return q4_code_bytes(nh) + q2_code_bytes(nm) + q1_code_bytes(nl);
}

/* ------------------------------------------------------------------ */

static void qmix_topk_(i64 *ids, f32 *sc, i32 k, i32 *have, i64 id, f32 score)
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

/* fill the per-sub residual (permuted query minus that sub's centroid)
 * and LUT; return the summed ||q - c||^2 over all three slabs. res is
 * [dim] laid out [hi | mid | lo]; lut is [code_bytes*256] likewise. */
static f32 qmix_prep_(const QMixIndex *ix, const f32 *query, f32 *res, f32 *lut)
{
    i32 nh = ix->n_hi, nm = ix->n_mid, nl = ix->n_lo, d;
    f32 *rh = res, *rm = res + nh, *rl = res + nh + nm;
    f32 *lh = lut;
    f32 *lm = lut + (size_t)ix->hi->code_bytes * 256;
    f32 *ll = lm  + (size_t)ix->mid->code_bytes * 256;
    f32 s = 0.0f;

    for (d = 0; d < nh; ++d) rh[d] = query[ix->perm[d]];
    for (d = 0; d < nm; ++d) rm[d] = query[ix->perm[nh + d]];
    for (d = 0; d < nl; ++d) rl[d] = query[ix->perm[nh + nm + d]];

    if (nh) { s += vs_sub_normsq(rh, rh, ix->hi->centroid,  nh); q4_lut_resid(ix->hi,  rh, lh); }
    if (nm) { s += vs_sub_normsq(rm, rm, ix->mid->centroid, nm); q2_lut_resid(ix->mid, rm, lm); }
    if (nl) { s += vs_sub_normsq(rl, rl, ix->lo->centroid,  nl); q1_lut_resid(ix->lo,  rl, ll); }
    return s;
}

static f32 qmix_raw_(const QMixIndex *ix, const f32 *lut, i32 id)
{
    const f32 *lh = lut;
    const f32 *lm = lut + (size_t)ix->hi->code_bytes * 256;
    const f32 *ll = lm  + (size_t)ix->mid->code_bytes * 256;
    f32 r = 0.0f;
    if (ix->n_hi)  r += vs_lut_gather_dot(lh, ix->hi->codes  + (size_t)id * ix->hi->code_bytes,  ix->hi->code_bytes);
    if (ix->n_mid) r += vs_lut_gather_dot(lm, ix->mid->codes + (size_t)id * ix->mid->code_bytes, ix->mid->code_bytes);
    if (ix->n_lo)  r += vs_lut_gather_dot(ll, ix->lo->codes  + (size_t)id * ix->lo->code_bytes,  ix->lo->code_bytes);
    return r;
}

i32 qmix_scratch_residual(i32 dim) { return dim; }
i32 qmix_scratch_lut(i32 dim) { return qmix_code_bytes(dim) * 256; }

/* cosine-style flat search (one global centroid per slab; comparable
 * scale). Higher = closer. */
i32 qmix_search_into(const QMixIndex *ix, const f32 *query, i32 k,
                     i64 *out_ids, f32 *out_scores, f32 *res, f32 *lut)
{
    f32 qn = (f32)qmix_sqrt_((f64)qmix_prep_(ix, query, res, lut));
    i32 id, have = 0;
    if (qn == 0.0f) qn = 1.0f;
    for (id = 0; id < ix->count; ++id) {
        f32 raw = qmix_raw_(ix, lut, id);
        f32 rn = (f32)qmix_sqrt_((f64)ix->rn2[id]);
        f32 denom = qn * rn;
        qmix_topk_(out_ids, out_scores, k, &have, (i64)id,
                   (denom > 0.0f) ? raw / denom : 0.0f);
    }
    return have;
}

i32 qmix_search(const QMixIndex *ix, const f32 *query, i32 k,
                i64 *out_ids, f32 *out_scores)
{
    Arena *a = ix->arena;
    Arena_Mark m = arena_snapshot(a);
    f32 *res = (f32 *)arena_alloc(a, (size_t)ix->dim * sizeof(f32));
    f32 *lut = (f32 *)arena_alloc(a, (size_t)ix->code_bytes * 256 * sizeof(f32));
    i32 n = qmix_search_into(ix, query, k, out_ids, out_scores, res, lut);
    arena_rewind(a, m);
    return n;
}

/* IVFPQ squared-L2 estimate, consistent term 3 with the l2_lambda blend
 * (identical maths to q1..q5, summed over the three disjoint slabs).
 * Returned as -dist^2. */
i32 qmix_search_l2_into(const QMixIndex *ix, const f32 *query, i32 k,
                        i64 *out_ids, f32 *out_scores, f32 *res, f32 *lut)
{
    f32 qc = qmix_prep_(ix, query, res, lut);
    i32 id, have = 0;
    for (id = 0; id < ix->count; ++id) {
        f32 raw = qmix_raw_(ix, lut, id), dist_sq;
        dist_sq = qc - 2.0f * raw
                + ix->l2_lambda * ix->rn2[id] + (1.0f - ix->l2_lambda) * ix->qn2[id];
        if (dist_sq < 0.0f) dist_sq = 0.0f;
        qmix_topk_(out_ids, out_scores, k, &have, (i64)id, -dist_sq);
    }
    return have;
}

i32 qmix_search_l2(const QMixIndex *ix, const f32 *query, i32 k,
                   i64 *out_ids, f32 *out_scores)
{
    Arena *a = ix->arena;
    Arena_Mark m = arena_snapshot(a);
    f32 *res = (f32 *)arena_alloc(a, (size_t)ix->dim * sizeof(f32));
    f32 *lut = (f32 *)arena_alloc(a, (size_t)ix->code_bytes * 256 * sizeof(f32));
    i32 n = qmix_search_l2_into(ix, query, k, out_ids, out_scores, res, lut);
    arena_rewind(a, m);
    return n;
}

/* dequantize one row back to an approximate vector in ORIGINAL dim order. */
void qmix_reconstruct_into(const QMixIndex *ix, i32 id, f32 *out)
{
    i32 nh = ix->n_hi, nm = ix->n_mid, nl = ix->n_lo, d;
    Arena *a = ix->arena;
    Arena_Mark m = arena_snapshot(a);
    f32 *tmp = (f32 *)arena_alloc(a, (size_t)(ix->dim > 0 ? ix->dim : 1) * sizeof(f32));
    if (nh) q4_reconstruct_into(ix->hi,  ix->hi->codes  + (size_t)id * ix->hi->code_bytes,  tmp);
    for (d = 0; d < nh; ++d) out[ix->perm[d]] = tmp[d];
    if (nm) q2_reconstruct_into(ix->mid, ix->mid->codes + (size_t)id * ix->mid->code_bytes, tmp);
    for (d = 0; d < nm; ++d) out[ix->perm[nh + d]] = tmp[d];
    if (nl) q1_reconstruct_into(ix->lo,  ix->lo->codes  + (size_t)id * ix->lo->code_bytes,  tmp);
    for (d = 0; d < nl; ++d) out[ix->perm[nh + nm + d]] = tmp[d];
    arena_rewind(a, m);
}

/* ================================================================== */
#ifdef QMIX_DEMO
#define PRINT_IMPLEMENTATION
#include "print.h"

static u32 qm_rng_(u32 *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static f32 qm_f_(u32 *s) { return (f32)qm_rng_(s) / (f32)0xffffffffu - 0.5f; }

/* corpus with a deliberately skewed per-dim variance: the first ~20% of
 * dims swing hard, the tail barely moves - the shape qmix is built for. */
static f32 *qm_corpus_(Arena *a, i32 n, i32 dim, u32 seed)
{
    f32 *o = (f32 *)arena_alloc(a, (size_t)n * dim * sizeof(f32));
    u32 st = seed | 1u; i32 i, d;
    for (i = 0; i < n; ++i)
        for (d = 0; d < dim; ++d) {
            f32 amp = (d < dim / 5) ? 3.0f : 0.15f;
            o[(size_t)i * dim + d] = qm_f_(&st) * amp;
        }
    return o;
}

static i32 qm_recall_(const QMixIndex *ix, const f32 *v, i32 n, i32 dim, i32 nq)
{
    i32 qi, hit = 0, tot = 0;
    for (qi = 0; qi < n; qi += (n / nq ? n / nq : 1)) {
        const f32 *q = v + (size_t)qi * dim;
        i64 ap[10], ex[10]; f32 as[10], bs[10]; i32 na, r, x, bh = 0;
        na = qmix_search(ix, q, 10, ap, as);
        for (r = 0; r < n; ++r) {
            const f32 *row = v + (size_t)r * dim;
            f32 dot = vs_dot(q, row, dim), qq = vs_dot(q, q, dim), rr = vs_dot(row, row, dim);
            f32 cs = (qq > 0.0f && rr > 0.0f)
                   ? dot / (f32)(qmix_sqrt_(qq) * qmix_sqrt_(rr)) : 0.0f;
            if (bh < 10) { i32 h = bh; ex[h] = r; bs[h] = cs;
                while (h > 0 && bs[h-1] < bs[h]) { f32 t = bs[h-1]; i64 ti = ex[h-1];
                    bs[h-1]=bs[h]; ex[h-1]=ex[h]; bs[h]=t; ex[h]=ti; --h; } ++bh; }
            else if (cs > bs[9]) { i32 h = 9; ex[h] = r; bs[h] = cs;
                while (h > 0 && bs[h-1] < bs[h]) { f32 t = bs[h-1]; i64 ti = ex[h-1];
                    bs[h-1]=bs[h]; ex[h-1]=ex[h]; bs[h]=t; ex[h]=ti; --h; } }
        }
        for (r = 0; r < na; ++r)
            for (x = 0; x < 10; ++x) if (ap[r] == ex[x]) { ++hit; break; }
        tot += 10;
    }
    return (i32)(100L * hit / (tot ? tot : 1));
}

int main(void)
{
    static Arena arena;
    Arena *a = &arena;
    i32 dim = 128, n = 2000, i, hit1 = 0;
    f32 *v = qm_corpus_(a, n, dim, 7u);
    QMixIndex *ix = qmix_build(a, v, n, dim);
    Q2Index *q2 = q2_build(a, v, n, dim);
    i64 id[4]; f32 sc[4];
    i32 rm, r2;

    for (i = 0; i < n; ++i) {
        i32 h = qmix_search(ix, v + (size_t)i * dim, 1, id, sc);
        if (h > 0 && id[0] == (i64)i) ++hit1;
    }
    print_cstr("qmix  dim="); print_i64(dim); print_cstr(" n="); print_i64(n);
    print_cstr("  split hi/mid/lo="); print_i64(ix->n_hi); print_cstr("/");
    print_i64(ix->n_mid); print_cstr("/"); print_i64(ix->n_lo);
    print_cstr("  code_bytes="); print_i64(ix->code_bytes);
    print_cstr(" (q2 uniform = "); print_i64(q2_code_bytes(dim)); print_cstr(")\n");
    print_cstr("  self-retrieval@1 = "); print_i64((i64)(100L * hit1 / n)); print_cstr("%\n");

    rm = qm_recall_(ix, v, n, dim, 60);
    {
        i32 qi, hit = 0, tot = 0;
        for (qi = 0; qi < n; qi += n / 60) {
            const f32 *q = v + (size_t)qi * dim;
            i64 ap[10], ex[10]; f32 as[10], bs[10]; i32 na, r, x, bh = 0;
            na = q2_search(q2, q, 10, ap, as);
            for (r = 0; r < n; ++r) {
                const f32 *row = v + (size_t)r * dim;
                f32 dot = vs_dot(q, row, dim), qq = vs_dot(q,q,dim), rr = vs_dot(row,row,dim);
                f32 cs = (qq>0.0f&&rr>0.0f)? dot/(f32)(qmix_sqrt_(qq)*qmix_sqrt_(rr)):0.0f;
                if (bh < 10) { i32 h=bh; ex[h]=r; bs[h]=cs;
                    while(h>0&&bs[h-1]<bs[h]){f32 t=bs[h-1];i64 ti=ex[h-1];bs[h-1]=bs[h];ex[h-1]=ex[h];bs[h]=t;ex[h]=ti;--h;} ++bh; }
                else if (cs>bs[9]) { i32 h=9; ex[h]=r; bs[h]=cs;
                    while(h>0&&bs[h-1]<bs[h]){f32 t=bs[h-1];i64 ti=ex[h-1];bs[h-1]=bs[h];ex[h-1]=ex[h];bs[h]=t;ex[h]=ti;--h;} }
            }
            for (r=0;r<na;++r) for(x=0;x<10;++x) if(ap[r]==ex[x]){++hit;break;}
            tot += 10;
        }
        r2 = (i32)(100L * hit / tot);
    }
    print_cstr("  recall@10 vs exact f32:  qmix = "); print_i64(rm);
    print_cstr("%   uniform q2 = "); print_i64(r2); print_cstr("%\n");
    print_flush();
    arena_free(a);
    return (hit1 * 100 / n >= 95 && rm + 2 >= r2) ? 0 : 1;
}
#endif /* QMIX_DEMO */
