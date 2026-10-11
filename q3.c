/* q3.c - ternary ({-1, 0, +1}) quantized flat vector index, one file.
 *
 * The BitNet b1.58 packing scheme: NOT a literal 3-bit field (a byte
 * holds no whole number of them), but radix-3 positional packing - 5
 * dimensions per byte, since 3^5 = 243 <= 256. Effective cost
 * log2(3) ~= 1.585 bits/dim (77 code bytes at dim=384).
 *
 * One of vecindex's five independent single-file quantizers
 * (q1/q2/q3/q4/q5, LEVELS = 2/4/3/16/256). Each is standalone: it owns
 * its calibration, packing, per-query LUT scoring and brute-force
 * top-k. engine.c is the optional layer that picks among them.
 *
 * Calibration is equal-POPULATION (empirical 1/3 and 2/3 residual
 * percentiles per dimension, codebook = the real mean of each bucket's
 * members), a 1-D Lloyd-Max-style scalar quantizer - not a fixed
 * +-3-sigma range cut into equal-width slices. On near-Gaussian
 * per-dimension residuals the equal-width middle bucket otherwise
 * swallows the common case and wastes 2 of the 3 levels on outliers.
 *
 * Everything lives in one caller-supplied Arena; there is no free -
 * drop the arena. Depends only on the project base (base.h, arena).
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -I. q3.c -DQ3_DEMO -o q3 && ./q3
 */

/* Single-file libs here follow the project's IMPLEMENTATION-guard
 * pattern: a consumer (engine.c, a test) defines the *_IMPLEMENTATION
 * macros once. The standalone Q3_DEMO build is its own only TU, so it
 * pulls the base implementations in itself. */
#ifdef Q3_DEMO
#define ARENA_ALLOC_IMPLEMENTATION
#endif
#include "base.h"
#include "arena_alloc.h"
#include "vsimd.h"

#define Q3_LEVELS 3
#define Q3_DPB    5              /* dims per byte: 3^5 = 243 <= 256 */

typedef struct {
    Arena *arena;
    i32 dim;
    i32 code_bytes;             /* ceil(dim / Q3_DPB) */
    i32 count;
    f32 *centroid;             /* [dim]                 */
    f32 *codebook;             /* [dim * Q3_LEVELS]     bucket means */
    f32 *bounds;               /* [dim * (Q3_LEVELS-1)] ascending    */
    u8  *codes;                /* [count * code_bytes]  */
    f32 *rnorm;                /* [count] ||v - centroid||       (cosine path) */
    f32 *qn2;                  /* [count] ||vhat - centroid||^2  (IVF L2 path) */
    f32  l2_lambda;            /* IVF term-3 blend: lambda*||v-c||^2 + (1-lambda)*qn2 */
} Q3Index;

/* ------------------------------------------------------------------ *
 * local math - own copies, not shared (project convention)          *
 * ------------------------------------------------------------------ */

/* sqrt with no libm: a bit-hack seed (~5% of the true root via one
 * integer add on the float's bits) then 3 Newton-Raphson iterations -
 * converges to full f32 precision, ~8x fewer divides than seeding with
 * x itself. */
static f64 q3_sqrt_(f64 x)
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

/* in-place ascending sort of a small f32 column (quicksort with an
 * insertion-sort tail) - keeps q3.c free of qsort/libc. */
static void q3_isort_(f32 *a, i32 n)
{
    i32 i, j;
    for (i = 1; i < n; ++i) {
        f32 key = a[i];
        for (j = i - 1; j >= 0 && a[j] > key; --j) a[j + 1] = a[j];
        a[j + 1] = key;
    }
}

static void q3_qsort_(f32 *a, i32 lo, i32 hi)
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
        if (j - lo < hi - j) { q3_qsort_(a, lo, j); lo = j + 1; }
        else                 { q3_qsort_(a, j + 1, hi); hi = j; }
    }
}

static void q3_sort_(f32 *a, i32 n)
{
    if (n > 1) { q3_qsort_(a, 0, n - 1); q3_isort_(a, n); }
}

/* byte value -> its Q3_DPB radix-3 digits, built once, shared. */
static const u8 *q3_digits_(void)
{
    static u8 tbl[256][Q3_DPB];
    static i32 built = 0;
    if (!built) {
        i32 b, s;
        for (b = 0; b < 256; ++b) {
            i32 rem = b;
            for (s = 0; s < Q3_DPB; ++s) { tbl[b][s] = (u8)(rem % Q3_LEVELS); rem /= Q3_LEVELS; }
        }
        built = 1;
    }
    return &tbl[0][0];
}

static const u32 q3_radix_pow_[Q3_DPB] = { 1u, 3u, 9u, 27u, 81u };

/* ------------------------------------------------------------------ *
 * calibration                                                        *
 * ------------------------------------------------------------------ */

#define Q3_GID(ids, i) ((ids) ? (size_t)(ids)[i] : (size_t)(i))

static void q3_calibrate_(Arena *a, const f32 *vectors, const i64 *ids,
                          i32 n, i32 dim, const f32 *centroid,
                          f32 *codebook, f32 *bounds)
{
    f32 *col = (f32 *)arena_alloc(a, (size_t)(n > 0 ? n : 1) * sizeof(f32));
    i32 j, k, l;
    for (j = 0; j < dim; ++j) {
        for (k = 0; k < n; ++k)
            col[k] = vectors[Q3_GID(ids, k) * dim + j] - centroid[j];
        q3_sort_(col, n);
        for (l = 0; l < Q3_LEVELS; ++l) {
            i32 start = (i32)((i64)l * n / Q3_LEVELS);
            i32 end   = (i32)((i64)(l + 1) * n / Q3_LEVELS);
            f32 sum = 0.0f;
            i32 m;
            if (end < start) end = start;
            for (m = start; m < end; ++m) sum += col[m];
            codebook[j * Q3_LEVELS + l] = (end > start) ? sum / (f32)(end - start) : 0.0f;
        }
        for (l = 0; l < Q3_LEVELS - 1; ++l) {
            i32 idx = (i32)((i64)(l + 1) * n / Q3_LEVELS);
            if (idx > n - 1) idx = n - 1;
            bounds[j * (Q3_LEVELS - 1) + l] = (n > 0) ? col[idx] : 0.0f;
        }
    }
}

static i32 q3_level_(f32 residual, const f32 *bounds)
{
    i32 lvl = 0, b;
    for (b = 0; b < Q3_LEVELS - 1; ++b) {
        if (residual >= bounds[b]) ++lvl;
        else break;
    }
    return lvl;
}

/* ------------------------------------------------------------------ *
 * build                                                              *
 * ------------------------------------------------------------------ */

/* Shared core. `ids == NULL` -> the whole 0..n-1 dataset in order;
 * `centroid == NULL` -> use the mean of the selected rows (flat scheme).
 * Otherwise calibrate the `n` rows named by `ids[]` against `centroid`
 * (one IVF cluster). Local id i always maps to row Q3_GID(ids, i). */
static Q3Index *q3_build_(Arena *a, const f32 *vectors, const i64 *ids,
                          i32 n, i32 dim, const f32 *centroid)
{
    Q3Index *ix = (Q3Index *)arena_alloc(a, sizeof(Q3Index));
    i32 i, d;

    ix->arena = a;
    ix->dim = dim;
    ix->code_bytes = (dim + Q3_DPB - 1) / Q3_DPB;
    ix->count = n;
    ix->centroid = (f32 *)arena_alloc(a, (size_t)dim * sizeof(f32));
    ix->codebook = (f32 *)arena_alloc(a, (size_t)dim * Q3_LEVELS * sizeof(f32));
    ix->bounds   = (f32 *)arena_alloc(a, (size_t)dim * (Q3_LEVELS - 1) * sizeof(f32));
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
            const f32 *v = vectors + Q3_GID(ids, i) * dim;
            for (d = 0; d < dim; ++d) ix->centroid[d] += v[d];
        }
        for (d = 0; d < dim; ++d) ix->centroid[d] = (f32)(ix->centroid[d] * inv_n);
    }

    q3_calibrate_(a, vectors, ids, n, dim, ix->centroid, ix->codebook, ix->bounds);

    for (i = 0; i < n; ++i) {
        const f32 *v = vectors + Q3_GID(ids, i) * dim;
        u8 *row = ix->codes + (size_t)i * ix->code_bytes;
        f32 sum_sq = 0.0f, qsq = 0.0f;
        for (d = 0; d < ix->code_bytes; ++d) row[d] = 0;
        for (d = 0; d < dim; ++d) {
            f32 r = v[d] - ix->centroid[d], cbv;
            i32 lvl = q3_level_(r, ix->bounds + d * (Q3_LEVELS - 1));
            cbv = ix->codebook[d * Q3_LEVELS + lvl];
            sum_sq += r * r;
            qsq += cbv * cbv;
            row[d / Q3_DPB] = (u8)(row[d / Q3_DPB] + (u32)lvl * q3_radix_pow_[d % Q3_DPB]);
        }
        ix->rnorm[i] = (f32)q3_sqrt_((f64)sum_sq);
        ix->qn2[i] = qsq;
    }
    return ix;
}

/* Flat index: calibrate against the whole dataset's own mean. */
Q3Index *q3_build(Arena *a, const f32 *vectors, i32 n, i32 dim)
{
    return q3_build_(a, vectors, (const i64 *)0, n, dim, (const f32 *)0);
}

/* One IVF cluster: the `n_ids` rows named by `ids[]`, calibrated against
 * `centroid` (that cluster's coarse centroid). Local id i -> ids[i]. */
Q3Index *q3_build_cluster(Arena *a, const f32 *vectors, const i64 *ids,
                          i32 n_ids, i32 dim, const f32 *centroid)
{
    return q3_build_(a, vectors, ids, n_ids, dim, centroid);
}

/* ------------------------------------------------------------------ *
 * search                                                             *
 * ------------------------------------------------------------------ */

static void q3_topk_insert_(i64 *ids, f32 *scores, i32 k, i32 *have,
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
void q3_lut_resid(const Q3Index *ix, const f32 *residual, f32 *lut)
{
    const u8 *digits = q3_digits_();
    i32 g, bv, s;
    for (g = 0; g < ix->code_bytes; ++g) {
        f32 contrib[Q3_DPB][Q3_LEVELS];
        i32 base = g * Q3_DPB, l;
        f32 *L = lut + g * 256;
        for (s = 0; s < Q3_DPB; ++s) {
            i32 d = base + s;
            if (d < ix->dim) {
                f32 r = residual[d];
                const f32 *cb = ix->codebook + d * Q3_LEVELS;
                for (l = 0; l < Q3_LEVELS; ++l) contrib[s][l] = r * cb[l];
            } else {
                for (l = 0; l < Q3_LEVELS; ++l) contrib[s][l] = 0.0f;
            }
        }
        for (bv = 0; bv < 256; ++bv) {
            const u8 *slots = digits + bv * Q3_DPB;
            f32 acc = 0.0f;
            for (s = 0; s < Q3_DPB; ++s) acc += contrib[s][slots[s]];
            L[bv] = acc;
        }
    }
}

static f32 q3_prep_(const Q3Index *ix, const f32 *query, f32 *residual, f32 *lut)
{
    f32 sum_sq = vs_sub_normsq(residual, query, ix->centroid, ix->dim);
    q3_lut_resid(ix, residual, lut);
    return sum_sq;
}

i32 q3_scan_dot(const Q3Index *ix, const f32 *lut, i32 k,
                i64 *out_ids, f32 *out_scores)
{
    i32 id, have = 0;
    for (id = 0; id < ix->count; ++id) {
        const u8 *row = ix->codes + (size_t)id * ix->code_bytes;
        f32 raw = vs_lut_gather_dot(lut, row, ix->code_bytes);
        q3_topk_insert_(out_ids, out_scores, k, &have, (i64)id, raw);
    }
    return have;
}

f32 q3_code_dot(const Q3Index *ix, i32 id, const f32 *lut)
{
    const u8 *row = ix->codes + (size_t)id * ix->code_bytes;
    f32 raw = vs_lut_gather_dot(lut, row, ix->code_bytes);
    return raw;
}

void q3_reconstruct_into(const Q3Index *ix, const u8 *code_row, f32 *out)
{
    const u8 *digits = q3_digits_();
    i32 d;
    for (d = 0; d < ix->dim; ++d) {
        i32 lvl = digits[(size_t)code_row[d / Q3_DPB] * Q3_DPB + (d % Q3_DPB)];
        out[d] = ix->centroid[d] + ix->codebook[d * Q3_LEVELS + lvl];
    }
}

/* Flat search: cosine-style similarity of (query - centroid) vs
 * (vector - centroid) residuals. Higher score = closer. Returns the
 * number written (<= k). Use when this index was built with q3_build
 * (one global centroid, so every score shares a scale). */
/* scratch sizes a caller must supply to the *_into forms (in floats).
 * The _into forms touch no arena, so a batch driver can run many across
 * threads on the same index. */
i32 q3_scratch_residual(i32 dim) { return dim; }
i32 q3_scratch_lut(i32 dim) { return ((dim + Q3_DPB - 1) / Q3_DPB) * 256; }

i32 q3_search_into(const Q3Index *ix, const f32 *query, i32 k,
                   i64 *out_ids, f32 *out_scores, f32 *residual, f32 *lut)
{
    f32 qnorm = (f32)q3_sqrt_((f64)q3_prep_(ix, query, residual, lut));
    i32 id, have = 0;
    if (qnorm == 0.0f) qnorm = 1.0f;
    for (id = 0; id < ix->count; ++id) {
        const u8 *row = ix->codes + (size_t)id * ix->code_bytes;
        f32 raw = vs_lut_gather_dot(lut, row, ix->code_bytes), denom, cos_est;
        denom = qnorm * ix->rnorm[id];
        cos_est = (denom > 0.0f) ? raw / denom : 0.0f;
        q3_topk_insert_(out_ids, out_scores, k, &have, (i64)id, cos_est);
    }
    return have;
}

i32 q3_search(const Q3Index *ix, const f32 *query, i32 k,
              i64 *out_ids, f32 *out_scores)
{
    Arena *a = ix->arena;
    Arena_Mark mark = arena_snapshot(a);
    f32 *residual = (f32 *)arena_alloc(a, (size_t)q3_scratch_residual(ix->dim) * sizeof(f32));
    f32 *lut = (f32 *)arena_alloc(a, (size_t)q3_scratch_lut(ix->dim) * sizeof(f32));
    i32 n = q3_search_into(ix, query, k, out_ids, out_scores, residual, lut);
    arena_rewind(a, mark);
    return n;
}

/* IVFPQ search: the squared-L2 estimate, computed CONSISTENTLY as
 *   dist(q,v)^2 ~= ||q - vhat||^2
 *              = ||q-c||^2 - 2*(q-c).(vhat-c) + ||vhat-c||^2
 * with (q-c).(vhat-c) the exact LUT gather and ||vhat-c||^2 the stored
 * qn2[id] (reconstruction residual norm, not the true one). Every term
 * then lives in the same reconstructed metric space, so scores from
 * clusters calibrated against different centroids are directly
 * comparable - what an nprobe>1 merge needs. Returned as -dist^2
 * (higher = closer). Local ids; the caller remaps to global. */
i32 q3_search_l2_into(const Q3Index *ix, const f32 *query, i32 k,
                      i64 *out_ids, f32 *out_scores, f32 *residual, f32 *lut)
{
    f32 q_c_dist_sq = q3_prep_(ix, query, residual, lut);
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
        q3_topk_insert_(out_ids, out_scores, k, &have, (i64)id, -dist_sq);
    }
    return have;
}

i32 q3_search_l2(const Q3Index *ix, const f32 *query, i32 k,
                 i64 *out_ids, f32 *out_scores)
{
    Arena *a = ix->arena;
    Arena_Mark mark = arena_snapshot(a);
    f32 *residual = (f32 *)arena_alloc(a, (size_t)q3_scratch_residual(ix->dim) * sizeof(f32));
    f32 *lut = (f32 *)arena_alloc(a, (size_t)q3_scratch_lut(ix->dim) * sizeof(f32));
    i32 n = q3_search_l2_into(ix, query, k, out_ids, out_scores, residual, lut);
    arena_rewind(a, mark);
    return n;
}

i32 q3_code_bytes(i32 dim) { return (dim + Q3_DPB - 1) / Q3_DPB; }

/* Encode one vector against an already-built index's calibration into
 * `out_row` (>= code_bytes) and its residual norm - the pure per-vector
 * cost, with no calibration/sort in the way. */
void q3_encode(const Q3Index *ix, const f32 *v, u8 *out_row, f32 *out_rnorm)
{
    f32 sum_sq = 0.0f;
    i32 d;
    for (d = 0; d < ix->code_bytes; ++d) out_row[d] = 0;
    for (d = 0; d < ix->dim; ++d) {
        f32 r = v[d] - ix->centroid[d];
        i32 lvl = q3_level_(r, ix->bounds + d * (Q3_LEVELS - 1));
        sum_sq += r * r;
        out_row[d / Q3_DPB] = (u8)(out_row[d / Q3_DPB] +
                                   (u32)lvl * q3_radix_pow_[d % Q3_DPB]);
    }
    *out_rnorm = (f32)q3_sqrt_((f64)sum_sq);
}

/* ================================================================== *
 * demo / self-test                                                   *
 * ================================================================== */
#ifdef Q3_DEMO
#define PRINT_IMPLEMENTATION
#include "print.h"

static u32 q3d_rng_(u32 *s)
{
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s;
}
static f32 q3d_f_(u32 *s) { return (f32)q3d_rng_(s) / (f32)0xffffffffu - 0.5f; }

/* Uniform [-1, 1] noise - the corpus shape the FLAT q3 scheme is for.
 * (On genuinely clustered data one global codebook spends its 3 levels
 * on "which cluster" rather than "where within it" - that case is
 * engine.c's per-cluster IVFPQ path, not flat q3.) */
static f32 *q3d_corpus_(Arena *a, i32 n, i32 dim, u32 seed)
{
    f32 *out = (f32 *)arena_alloc(a, (size_t)n * dim * sizeof(f32));
    u32 st = seed | 1u;
    i32 i;
    for (i = 0; i < n * dim; ++i) out[i] = q3d_f_(&st) * 2.0f;
    return out;
}

int main(void)
{
    static Arena arena;        /* statics are zero-initialised */
    Arena *a = &arena;
    i32 dim = 64, n = 500;
    f32 *vecs;
    Q3Index *ix;
    i64 ids[16]; f32 sc[16];
    i32 i, hit1 = 0;
    i32 overlap = 0, qn = 0, exq;

    vecs = q3d_corpus_(a, n, dim, 42u);
    ix = q3_build(a, vecs, n, dim);

    for (i = 0; i < n; ++i) {
        i32 h = q3_search(ix, vecs + (size_t)i * dim, 1, ids, sc);
        if (h > 0 && ids[0] == (i64)i) ++hit1;
    }
    print_cstr("q3  dim="); print_i64(dim);
    print_cstr(" n=");      print_i64(n);
    print_cstr("  code_bytes/vec="); print_i64(ix->code_bytes);
    print_cstr(" (+8 rnorm+qn2)  self-retrieval@1=");
    print_i64((i64)(100.0 * hit1 / n)); print_cstr("%\n");

    /* recall@10 vs exact cosine on a sample of queries */
    for (exq = 0; exq < n; exq += n / 20) {
        const f32 *q = vecs + (size_t)exq * dim;
        i64 approx[10]; f32 as[10];
        i32 na = q3_search(ix, q, 10, approx, as), r, j2;
        f32 best[10]; i64 bid[10]; i32 bh = 0;
        for (r = 0; r < n; ++r) {
            const f32 *row = vecs + (size_t)r * dim;
            f32 dot = 0.0f, qq = 0.0f, rr = 0.0f, cs; i32 d;
            for (d = 0; d < dim; ++d) { dot += q[d]*row[d]; qq += q[d]*q[d]; rr += row[d]*row[d]; }
            cs = (qq > 0.0f && rr > 0.0f) ? dot / (f32)(q3_sqrt_(qq) * q3_sqrt_(rr)) : 0.0f;
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
    return (hit1 * 100 / n >= 95) ? 0 : 1;
}
#endif /* Q3_DEMO */
