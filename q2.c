/* q2.c - 2-bit (4-level) quantized flat vector index, one file.
 *
 * The leanest of vecindex's five independent single-file quantizers
 * (q1/q2/q3/q4/q5, LEVELS = 2/4/3/16/256). Bit-packed: 8 dimensions per
 * byte, 1 bit each. Effective cost 1 bit/dim (48 code bytes at
 * dim=384) - half of q3's radix-3.
 *
 * Calibration is equal-POPULATION: the single per-dimension boundary
 * is the residual median, the two codebook values are the real means
 * of the below-median and above-median halves - a 1-bit Lloyd-Max
 * scalar quantizer, not a fixed sign() of the residual.
 *
 * Same shape as the other four: everything in one caller-supplied
 * Arena, no free; per-query LUT (256 entries per byte-group) rebuilt
 * once per search; brute-force cosine-style top-k over the residuals.
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -I. q1.c -DQ2_DEMO -lm -o q1 && ./q1
 */

#ifdef Q2_DEMO
#define ARENA_ALLOC_IMPLEMENTATION
#endif
#include "base.h"
#include "arena_alloc.h"
#include "vsimd.h"

#define Q2_LEVELS 4
#define Q2_BITS   2
#define Q2_DPB    4
#define Q2_MASK   ((1 << Q2_BITS) - 1)

typedef struct {
    Arena *arena;
    i32 dim;
    i32 code_bytes;                /* ceil(dim / Q2_DPB) */
    i32 count;
    f32 *centroid;                /* [dim]                 */
    f32 *codebook;               /* [dim * Q2_LEVELS]     bucket means */
    f32 *bounds;                 /* [dim * (Q2_LEVELS-1)] ascending    */
    u8  *codes;                  /* [count * code_bytes]  */
    f32 *rnorm;                  /* [count] ||v - centroid||       (cosine path) */
    f32 *qn2;                    /* [count] ||vhat - centroid||^2  (IVF L2 path) */
    f32  l2_lambda;              /* IVF term-3 blend: lambda*||v-c||^2 + (1-lambda)*qn2 */
} Q2Index;

/* ------------------------------------------------------------------ */

/* sqrt with no libm: a bit-hack seed (~5% of the true root via one
 * integer add on the float's bits) then 3 Newton-Raphson iterations -
 * converges to full f32 precision, ~8x fewer divides than seeding with
 * x itself. */
static f64 q2_sqrt_(f64 x)
{
    union { f32 f; u32 i; } u;
    f64 g;
    i32 i;
    if (x <= 0.0) return 0.0;
    u.f = (f32)x;
    u.i = (u.i >> 1) + 0x1fbd1df5u;
    g = (f64)u.f;
    for (i = 0; i < 3; ++i) g = 0.5 * (g + x / g);
    return g;
}

static void q2_isort_(f32 *a, i32 n)
{
    i32 i, j;
    for (i = 1; i < n; ++i) {
        f32 key = a[i];
        for (j = i - 1; j >= 0 && a[j] > key; --j) a[j + 1] = a[j];
        a[j + 1] = key;
    }
}

static void q2_qsort_(f32 *a, i32 lo, i32 hi)
{
    while (hi - lo > 16) {
        f32 pivot = a[lo + (hi - lo) / 2];
        i32 i = lo, j = hi;
        for (;;) {
            while (a[i] < pivot) ++i;
            while (a[j] > pivot) --j;
            if (i >= j) break;
            { f32 t = a[i]; a[i] = a[j]; a[j] = t; }
            ++i; --j;
        }
        if (j - lo < hi - j) { q2_qsort_(a, lo, j); lo = j + 1; }
        else                 { q2_qsort_(a, j + 1, hi); hi = j; }
    }
}

static void q2_sort_(f32 *a, i32 n)
{
    if (n > 1) { q2_qsort_(a, 0, n - 1); q2_isort_(a, n); }
}

/* byte value -> its Q2_DPB packed level digits, built once, shared. */
static const u8 *q2_digits_(void)
{
    static u8 tbl[256][Q2_DPB];
    static i32 built = 0;
    if (!built) {
        i32 b, s;
        for (b = 0; b < 256; ++b)
            for (s = 0; s < Q2_DPB; ++s)
                tbl[b][s] = (u8)((b >> (Q2_BITS * s)) & Q2_MASK);
        built = 1;
    }
    return &tbl[0][0];
}

/* ------------------------------------------------------------------ */

#define Q2_GID(ids, i) ((ids) ? (size_t)(ids)[i] : (size_t)(i))

static void q2_calibrate_(Arena *a, const f32 *vectors, const i64 *ids,
                          i32 n, i32 dim, const f32 *centroid,
                          f32 *codebook, f32 *bounds)
{
    f32 *col = (f32 *)arena_alloc(a, (size_t)(n > 0 ? n : 1) * sizeof(f32));
    i32 j, k, l;
    for (j = 0; j < dim; ++j) {
        for (k = 0; k < n; ++k)
            col[k] = vectors[Q2_GID(ids, k) * dim + j] - centroid[j];
        q2_sort_(col, n);
        for (l = 0; l < Q2_LEVELS; ++l) {
            i32 start = (i32)((i64)l * n / Q2_LEVELS);
            i32 end   = (i32)((i64)(l + 1) * n / Q2_LEVELS);
            f32 sum = 0.0f;
            i32 m;
            if (end < start) end = start;
            for (m = start; m < end; ++m) sum += col[m];
            codebook[j * Q2_LEVELS + l] = (end > start) ? sum / (f32)(end - start) : 0.0f;
        }
        for (l = 0; l < Q2_LEVELS - 1; ++l) {
            i32 idx = (i32)((i64)(l + 1) * n / Q2_LEVELS);
            if (idx > n - 1) idx = n - 1;
            bounds[j * (Q2_LEVELS - 1) + l] = (n > 0) ? col[idx] : 0.0f;
        }
    }
}

/* level for one residual: largest i in [0, Q2_LEVELS-1] with
 * residual >= bounds[i-1] (bounds[] is Q2_LEVELS-1 ascending). Linear
 * for small level counts (branch predictor wins), binary once the bound
 * list is long (q5's 255). */
static i32 q2_level_(f32 residual, const f32 *bounds)
{
#if Q2_LEVELS > 16
    i32 lo = 0, hi = Q2_LEVELS - 1;
    while (lo < hi) {
        i32 mid = (lo + hi) >> 1;
        if (residual >= bounds[mid]) lo = mid + 1;
        else hi = mid;
    }
    return lo;
#else
    i32 lvl = 0, b;
    for (b = 0; b < Q2_LEVELS - 1; ++b) {
        if (residual >= bounds[b]) ++lvl;
        else break;
    }
    return lvl;
#endif
}

/* ------------------------------------------------------------------ */

/* Shared core. `ids == NULL` -> the whole 0..n-1 dataset in order;
 * `centroid == NULL` -> use the mean of the selected rows (flat scheme).
 * Otherwise calibrate the `n` rows named by `ids[]` against `centroid`
 * (one IVF cluster). Local id i always maps to row Q2_GID(ids, i). */
static Q2Index *q2_build_(Arena *a, const f32 *vectors, const i64 *ids,
                          i32 n, i32 dim, const f32 *centroid)
{
    Q2Index *ix = (Q2Index *)arena_alloc(a, sizeof(Q2Index));
    i32 i, d;

    ix->arena = a;
    ix->dim = dim;
    ix->code_bytes = (dim + Q2_DPB - 1) / Q2_DPB;
    ix->count = n;
    ix->centroid = (f32 *)arena_alloc(a, (size_t)dim * sizeof(f32));
    ix->codebook = (f32 *)arena_alloc(a, (size_t)dim * Q2_LEVELS * sizeof(f32));
    ix->bounds   = (f32 *)arena_alloc(a, (size_t)dim * (Q2_LEVELS - 1) * sizeof(f32));
    ix->codes    = (u8  *)arena_alloc(a, (size_t)(n > 0 ? n : 1) * ix->code_bytes);
    ix->rnorm    = (f32 *)arena_alloc(a, (size_t)(n > 0 ? n : 1) * sizeof(f32));
    ix->qn2      = (f32 *)arena_alloc(a, (size_t)(n > 0 ? n : 1) * sizeof(f32));
    ix->l2_lambda = 1.0f;   /* engine_build may retune for an nprobe>1 merge */

    if (centroid) {
        for (d = 0; d < dim; ++d) ix->centroid[d] = centroid[d];
    } else {
        f64 inv_n = (n > 0) ? 1.0 / (f64)n : 0.0;
        for (d = 0; d < dim; ++d) ix->centroid[d] = 0.0f;
        for (i = 0; i < n; ++i) {
            const f32 *v = vectors + Q2_GID(ids, i) * dim;
            for (d = 0; d < dim; ++d) ix->centroid[d] += v[d];
        }
        for (d = 0; d < dim; ++d) ix->centroid[d] = (f32)(ix->centroid[d] * inv_n);
    }

    q2_calibrate_(a, vectors, ids, n, dim, ix->centroid, ix->codebook, ix->bounds);

    for (i = 0; i < n; ++i) {
        const f32 *v = vectors + Q2_GID(ids, i) * dim;
        u8 *row = ix->codes + (size_t)i * ix->code_bytes;
        f32 sum_sq = 0.0f, qsq = 0.0f;
        for (d = 0; d < ix->code_bytes; ++d) row[d] = 0;
        for (d = 0; d < dim; ++d) {
            f32 r = v[d] - ix->centroid[d], cbv;
            i32 lvl = q2_level_(r, ix->bounds + d * (Q2_LEVELS - 1));
            cbv = ix->codebook[d * Q2_LEVELS + lvl];
            sum_sq += r * r;
            qsq += cbv * cbv;
            row[d / Q2_DPB] = (u8)(row[d / Q2_DPB] | ((u32)lvl << (Q2_BITS * (d % Q2_DPB))));
        }
        ix->rnorm[i] = (f32)q2_sqrt_((f64)sum_sq);
        ix->qn2[i] = qsq;
    }
    return ix;
}

/* Flat index: calibrate against the whole dataset's own mean. */
Q2Index *q2_build(Arena *a, const f32 *vectors, i32 n, i32 dim)
{
    return q2_build_(a, vectors, (const i64 *)0, n, dim, (const f32 *)0);
}

/* One IVF cluster: the `n_ids` rows named by `ids[]`, calibrated against
 * `centroid` (that cluster's coarse centroid). Local id i -> ids[i]. */
Q2Index *q2_build_cluster(Arena *a, const f32 *vectors, const i64 *ids,
                          i32 n_ids, i32 dim, const f32 *centroid)
{
    return q2_build_(a, vectors, ids, n_ids, dim, centroid);
}

/* ------------------------------------------------------------------ */

static void q2_topk_insert_(i64 *ids, f32 *scores, i32 k, i32 *have,
                            i64 id, f32 score)
{
    i32 j;
    if (*have < k) {
        j = *have;
        ids[j] = id; scores[j] = score;
        while (j > 0 && scores[j - 1] < scores[j]) {
            f32 ts = scores[j - 1]; i64 ti = ids[j - 1];
            scores[j - 1] = scores[j]; ids[j - 1] = ids[j];
            scores[j] = ts; ids[j] = ti; --j;
        }
        ++(*have);
    } else if (score > scores[k - 1]) {
        j = k - 1;
        ids[j] = id; scores[j] = score;
        while (j > 0 && scores[j - 1] < scores[j]) {
            f32 ts = scores[j - 1]; i64 ti = ids[j - 1];
            scores[j - 1] = scores[j]; ids[j - 1] = ids[j];
            scores[j] = ts; ids[j] = ti; --j;
        }
    }
}

/* Fill residual[dim] = query - centroid and lut[code_bytes*256] (the
 * per-byte-group dot contribution). Returns ||query - centroid||^2. */
/* Build lut[code_bytes*256] from a caller-supplied residual (query minus
 * whatever centroid the caller means). Per byte-group: precompute
 * contrib[slot][level] = residual * codebook, then each of the 256 LUT
 * entries is Q2_DPB adds through the digit table - no multiply, no
 * branch, in the 256-wide loop. */
void q2_lut_resid(const Q2Index *ix, const f32 *residual, f32 *lut)
{
    const u8 *digits = q2_digits_();
    i32 g, bv, s;
    for (g = 0; g < ix->code_bytes; ++g) {
        f32 contrib[Q2_DPB][Q2_LEVELS];
        i32 base = g * Q2_DPB, l;
        f32 *L = lut + g * 256;
        for (s = 0; s < Q2_DPB; ++s) {
            i32 d = base + s;
            if (d < ix->dim) {
                f32 r = residual[d];
                const f32 *cb = ix->codebook + d * Q2_LEVELS;
                for (l = 0; l < Q2_LEVELS; ++l) contrib[s][l] = r * cb[l];
            } else {
                for (l = 0; l < Q2_LEVELS; ++l) contrib[s][l] = 0.0f;
            }
        }
        for (bv = 0; bv < 256; ++bv) {
            const u8 *slots = digits + bv * Q2_DPB;
            f32 acc = 0.0f;
            for (s = 0; s < Q2_DPB; ++s) acc += contrib[s][slots[s]];
            L[bv] = acc;
        }
    }
}

static f32 q2_prep_(const Q2Index *ix, const f32 *query, f32 *residual, f32 *lut)
{
    f32 sum_sq = vs_sub_normsq(residual, query, ix->centroid, ix->dim);
    q2_lut_resid(ix, residual, lut);
    return sum_sq;
}

/* Raw LUT dot for every stored code (higher = closer), top-k by that -
 * the primitive a residual cascade sums two of. Local ids. */
i32 q2_scan_dot(const Q2Index *ix, const f32 *lut, i32 k,
                i64 *out_ids, f32 *out_scores)
{
    i32 id, have = 0;
    for (id = 0; id < ix->count; ++id) {
        const u8 *row = ix->codes + (size_t)id * ix->code_bytes;
        f32 raw = vs_lut_gather_dot(lut, row, ix->code_bytes);
        q2_topk_insert_(out_ids, out_scores, k, &have, (i64)id, raw);
    }
    return have;
}

/* Raw LUT dot for one stored code by local id - the cascade rerank
 * primitive (score a shortlist against the second layer's LUT). */
f32 q2_code_dot(const Q2Index *ix, i32 id, const f32 *lut)
{
    const u8 *row = ix->codes + (size_t)id * ix->code_bytes;
    f32 raw = vs_lut_gather_dot(lut, row, ix->code_bytes);
    return raw;
}

/* Dequantize one stored code back to an approximate vector:
 * out[d] = centroid[d] + codebook[d][level(code_row, d)]. */
void q2_reconstruct_into(const Q2Index *ix, const u8 *code_row, f32 *out)
{
    const u8 *digits = q2_digits_();
    i32 d;
    for (d = 0; d < ix->dim; ++d) {
        i32 lvl = digits[(size_t)code_row[d / Q2_DPB] * Q2_DPB + (d % Q2_DPB)];
        out[d] = ix->centroid[d] + ix->codebook[d * Q2_LEVELS + lvl];
    }
}

/* Flat search: cosine-style similarity of (query - centroid) vs
 * (vector - centroid) residuals. Higher score = closer. Returns the
 * number written (<= k). Use when this index was built with q2_build
 * (one global centroid, so every score shares a scale). */
/* scratch sizes a caller must supply to the *_into forms (in floats).
 * The _into forms touch no arena, so a batch driver can run many across
 * threads on the same index. */
i32 q2_scratch_residual(i32 dim) { return dim; }
i32 q2_scratch_lut(i32 dim) { return ((dim + Q2_DPB - 1) / Q2_DPB) * 256; }

i32 q2_search_into(const Q2Index *ix, const f32 *query, i32 k,
                   i64 *out_ids, f32 *out_scores, f32 *residual, f32 *lut)
{
    f32 qnorm = (f32)q2_sqrt_((f64)q2_prep_(ix, query, residual, lut));
    i32 id, have = 0;
    if (qnorm == 0.0f) qnorm = 1.0f;
    for (id = 0; id < ix->count; ++id) {
        const u8 *row = ix->codes + (size_t)id * ix->code_bytes;
        f32 raw = vs_lut_gather_dot(lut, row, ix->code_bytes), denom, cos_est;
        denom = qnorm * ix->rnorm[id];
        cos_est = (denom > 0.0f) ? raw / denom : 0.0f;
        q2_topk_insert_(out_ids, out_scores, k, &have, (i64)id, cos_est);
    }
    return have;
}

i32 q2_search(const Q2Index *ix, const f32 *query, i32 k,
              i64 *out_ids, f32 *out_scores)
{
    Arena *a = ix->arena;
    Arena_Mark mark = arena_snapshot(a);
    f32 *residual = (f32 *)arena_alloc(a, (size_t)q2_scratch_residual(ix->dim) * sizeof(f32));
    f32 *lut = (f32 *)arena_alloc(a, (size_t)q2_scratch_lut(ix->dim) * sizeof(f32));
    i32 n = q2_search_into(ix, query, k, out_ids, out_scores, residual, lut);
    arena_rewind(a, mark);
    return n;
}

/* IVFPQ search: the squared-L2 estimate, computed CONSISTENTLY as
 *   dist(q,v)^2 ~= ||q - vhat||^2
 *              = ||q-c||^2 - 2*(q-c).(vhat-c) + ||vhat-c||^2
 * where (q-c).(vhat-c) is the exact LUT gather and ||vhat-c||^2 is the
 * stored qn2[id] (the RECONSTRUCTION residual norm, not the true one).
 * Every term is then in the same reconstructed metric space, so scores
 * from clusters calibrated against different centroids are directly
 * comparable - which is what an nprobe>1 merge needs. Using the true
 * ||v-c||^2 here instead (a "partial correction") re-introduces a
 * per-cluster bias that tracks that cluster's quantisation error and
 * costs ~10 recall points at nprobe=16. Returned as -dist^2 (higher =
 * closer). Local ids; the caller remaps to global. */
i32 q2_search_l2_into(const Q2Index *ix, const f32 *query, i32 k,
                      i64 *out_ids, f32 *out_scores, f32 *residual, f32 *lut)
{
    f32 q_c_dist_sq = q2_prep_(ix, query, residual, lut);
    i32 id, have = 0;
    for (id = 0; id < ix->count; ++id) {
        const u8 *row = ix->codes + (size_t)id * ix->code_bytes;
        f32 raw = vs_lut_gather_dot(lut, row, ix->code_bytes), rn = ix->rnorm[id], dist_sq;
        /* term 3 blends the reconstruction norm ||vhat-c||^2 (qn2, keeps
         * the estimate a proper cross-cluster-comparable metric) with the
         * true ||v-c||^2 (rn*rn, an unbiased partial correction): the
         * half-and-half point is the lower-variance estimator for an IVF
         * merge without the positive bias a pure ||q-vhat||^2 puts on
         * poorly-quantised true neighbours. */
        dist_sq = q_c_dist_sq - 2.0f * raw
                + ix->l2_lambda * (rn * rn) + (1.0f - ix->l2_lambda) * ix->qn2[id];
        if (dist_sq < 0.0f) dist_sq = 0.0f;
        q2_topk_insert_(out_ids, out_scores, k, &have, (i64)id, -dist_sq);
    }
    return have;
}

i32 q2_search_l2(const Q2Index *ix, const f32 *query, i32 k,
                 i64 *out_ids, f32 *out_scores)
{
    Arena *a = ix->arena;
    Arena_Mark mark = arena_snapshot(a);
    f32 *residual = (f32 *)arena_alloc(a, (size_t)q2_scratch_residual(ix->dim) * sizeof(f32));
    f32 *lut = (f32 *)arena_alloc(a, (size_t)q2_scratch_lut(ix->dim) * sizeof(f32));
    i32 n = q2_search_l2_into(ix, query, k, out_ids, out_scores, residual, lut);
    arena_rewind(a, mark);
    return n;
}

i32 q2_code_bytes(i32 dim) { return (dim + Q2_DPB - 1) / Q2_DPB; }

/* Encode one vector against an already-built index's calibration into
 * `out_row` (>= code_bytes) and its residual norm - the pure per-vector
 * cost, with no calibration/sort in the way. */
void q2_encode(const Q2Index *ix, const f32 *v, u8 *out_row, f32 *out_rnorm)
{
    f32 sum_sq = 0.0f;
    i32 d;
    for (d = 0; d < ix->code_bytes; ++d) out_row[d] = 0;
    for (d = 0; d < ix->dim; ++d) {
        f32 r = v[d] - ix->centroid[d];
        i32 lvl = q2_level_(r, ix->bounds + d * (Q2_LEVELS - 1));
        sum_sq += r * r;
        out_row[d / Q2_DPB] = (u8)(out_row[d / Q2_DPB] |
                                   ((u32)lvl << (Q2_BITS * (d % Q2_DPB))));
    }
    *out_rnorm = (f32)q2_sqrt_((f64)sum_sq);
}

/* ================================================================== */
#ifdef Q2_DEMO
#define PRINT_IMPLEMENTATION
#include "print.h"

static u32 q2d_rng_(u32 *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static f32 q2d_f_(u32 *s) { return (f32)q2d_rng_(s) / (f32)0xffffffffu - 0.5f; }

static f32 *q2d_corpus_(Arena *a, i32 n, i32 dim, u32 seed)
{
    f32 *out = (f32 *)arena_alloc(a, (size_t)n * dim * sizeof(f32));
    u32 st = seed | 1u;
    i32 i;
    for (i = 0; i < n * dim; ++i) out[i] = q2d_f_(&st) * 2.0f;
    return out;
}

int main(void)
{
    static Arena arena;
    Arena *a = &arena;
    i32 dim = 64, n = 500;
    f32 *vecs;
    Q2Index *ix;
    i64 ids[16]; f32 sc[16];
    i32 i, hit1 = 0, overlap = 0, qn = 0, exq;

    vecs = q2d_corpus_(a, n, dim, 42u);
    ix = q2_build(a, vecs, n, dim);

    for (i = 0; i < n; ++i) {
        i32 h = q2_search(ix, vecs + (size_t)i * dim, 1, ids, sc);
        if (h > 0 && ids[0] == (i64)i) ++hit1;
    }
    print_cstr("q2  dim="); print_i64(dim);
    print_cstr(" n=");      print_i64(n);
    print_cstr("  code_bytes/vec="); print_i64(ix->code_bytes);
    print_cstr(" (+8 rnorm+qn2)  self-retrieval@1=");
    print_i64((i64)(100.0 * hit1 / n)); print_cstr("%\n");

    for (exq = 0; exq < n; exq += n / 20) {
        const f32 *q = vecs + (size_t)exq * dim;
        i64 approx[10]; f32 as[10];
        i32 na = q2_search(ix, q, 10, approx, as), r, j2;
        f32 best[10]; i64 bid[10]; i32 bh = 0;
        for (r = 0; r < n; ++r) {
            const f32 *row = vecs + (size_t)r * dim;
            f32 dot = 0.0f, qq = 0.0f, rr = 0.0f, cs; i32 d;
            for (d = 0; d < dim; ++d) { dot += q[d]*row[d]; qq += q[d]*q[d]; rr += row[d]*row[d]; }
            cs = (qq > 0.0f && rr > 0.0f) ? dot / (f32)(q2_sqrt_(qq) * q2_sqrt_(rr)) : 0.0f;
            { i32 hh = bh;
              if (hh < 10) { bid[hh]=r; best[hh]=cs; while(hh>0&&best[hh-1]<best[hh]){f32 t=best[hh-1];i64 ti=bid[hh-1];best[hh-1]=best[hh];bid[hh-1]=bid[hh];best[hh]=t;bid[hh]=ti;--hh;} ++bh; }
              else if (cs > best[9]) { hh=9; bid[hh]=r; best[hh]=cs; while(hh>0&&best[hh-1]<best[hh]){f32 t=best[hh-1];i64 ti=bid[hh-1];best[hh-1]=best[hh];bid[hh-1]=bid[hh];best[hh]=t;bid[hh]=ti;--hh;} } }
        }
        for (r = 0; r < na; ++r)
            for (j2 = 0; j2 < 10; ++j2)
                if (approx[r] == bid[j2]) { ++overlap; break; }
        ++qn;
    }
    print_cstr("     recall@10 vs exact f32 = ");
    print_i64((i64)(100.0 * overlap / (qn * 10))); print_cstr("%\n");
    print_flush();

    arena_free(a);
    return (hit1 * 100 / n >= 92) ? 0 : 1;
}
#endif /* Q2_DEMO */
