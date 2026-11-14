/* engine.c - the production adaptive layer over vecindex.
 *
 * Combines the two real levers:
 *   - ivf.c   coarse k-means router  -> O(n) becomes O(routed subset)
 *   - q1..q5  fine scalar quantizers -> 1 / 1.585 / 2 / 4 / 8 bits/dim
 *
 * engine_build() constructs every candidate - the 5 flat quantizers and
 * IVF + each of the 5, at a small sweep of nprobe - measures each one's
 * real recall@k against an exact f32 baseline on a sample of the corpus
 * itself, and keeps the configuration that meets the caller's goal:
 *
 *   ENG_MIN_SIZE  smallest code (bytes/vector) that reaches target_recall
 *   ENG_MAX_ACC   best recall that fits within max_code_bytes
 *
 * So topology (flat vs IVF), fine scheme (q1..q5) and nprobe are all
 * picked from the data, not guessed. One arena owns everything; there is
 * no engine_free - drop the arena.
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -D_POSIX_C_SOURCE=200112L -O2 \
 *       -Isrc tests/engine_test.c -lm -o engine_test && ./engine_test
 */

#define ARENA_ALLOC_IMPLEMENTATION
#include "base.h"
#include "arena_alloc.h"

#include "ivf.c"
#include "hnsw.c"     /* HNSW graph coarse router over the IVF centroids */
#include "q1.c"
#include "q2.c"
#include "q3.c"
#include "q4.c"
#include "q5.c"
#include "qmix.c"     /* per-dimension mixed-depth (uses q1/q2/q4) */
#include "q4fs.c"     /* FAISS 4-bit fast-scan wrapper over a Q4Index   */

/* engine_build measures the cell count too: sqrt(n) vs 4*sqrt(n), each
 * q4-probed, better kept. More cells lift recall on weakly-clustered
 * data but fragment tight clusters, so it is a measured choice. A graph
 * router is built once nc >= ENG_HNSW_MIN_NC (below that the linear
 * centroid scan is already exhaustive and cheap); it keeps the
 * many-cell option's routing sub-linear. k-means training stays bounded
 * by ENG_IVF_TRAIN_CAP. */
#define ENG_HNSW_MIN_NC   48
#define ENG_IVF_TRAIN_CAP 50000

/* qmix rides in the scheme sweep as a 6th candidate: variance-ranked
 * per-dim bit depth (4/2/1) instead of one uniform width. It has no
 * uniform "levels" and is not a cascade layer, so the 4 cascade-only
 * dispatchers below leave it unhandled by design.
 *
 * ENG_Q4FS sits AFTER ENG_NS: never a sweep candidate, only a
 * post-selection swap-in for a chosen q4 config once the corpus is big
 * enough for the pshufb scan to pay for the code transpose. Identical
 * calibration and recall as q4 (it exact-reranks its shortlist). */
typedef enum { ENG_Q1 = 0, ENG_Q3, ENG_Q2, ENG_Q4, ENG_Q5, ENG_QMIX,
               ENG_NS, ENG_Q4FS } EngScheme;

/* per-index corpus size below which the q4 transpose is not worth it */
#define ENG_Q4FS_MIN 2048

/* ENG_HNSW: a full HNSW graph over the f32 vectors themselves (exact
 * distances, no quantisation) - the highest-recall topology, for small
 * n where dim*4 bytes/vec is acceptable. Tried as a candidate only
 * within these bounds. */
#define ENG_VHNSW_MAXN     200000
#define ENG_VHNSW_MAXBYTES ((size_t)256 << 20)

typedef enum { ENG_FLAT = 0, ENG_IVF = 1, ENG_CASCADE = 2, ENG_HNSW = 3 } EngTopo;
typedef enum { ENG_MIN_SIZE = 0, ENG_MAX_ACC = 1 } EngMode;

/* the sweep probes every scheme at a coarse nprobe set; the chosen IVF
 * scheme is then refined over the full set (finds the smallest nprobe
 * that saturates recall, and honours a scan budget). */
#define ENG_NNPROBE 3
static const i32 ENG_NPROBES[ENG_NNPROBE] = { 1, 4, 16 };
#define ENG_NPROBE_FINE_N 6
static const i32 ENG_NPROBE_FINE[ENG_NPROBE_FINE_N] = { 1, 2, 4, 8, 16, 32 };

/* coarse-quantizer training subsample floor - k-means trains on
 * min(n, max(40*n_clusters, this)); the rest is a single assignment
 * pass. */
#define ENG_IVF_TRAIN_MIN 25000

typedef struct {
    EngMode mode;
    i32     k;                 /* recall@k to optimise (default 10)     */
    f32     target_recall;     /* ENG_MIN_SIZE                          */
    i32     max_code_bytes;    /* ENG_MAX_ACC                           */
    i32     eval_queries;      /* recall-probe sample size (default 96) */
    i32     n_iters;           /* IVF Lloyd iterations (default 12)     */
    u32     seed;              /* IVF seed (default 0x5eed)             */
    i32     max_candidates;    /* IVF nprobe budget: largest nprobe whose
                               * expected scanned vectors (nprobe * n/nc)
                               * stays <= this. 0 = pick nprobe purely by
                               * recall. A machine-independent stand-in
                               * for a query-latency cap. */
    const f32 *eval_q;        /* optional [n_eval_q * dim] representative
                               * QUERIES to tune against - use for an
                               * asymmetric workload (RAG: questions, not
                               * corpus rows). NULL -> tune on corpus rows. */
    i32     n_eval_q;
} EngGoal;

typedef struct {
    Arena    *arena;
    i32       dim;
    i32       count;

    EngTopo   topo;
    EngScheme scheme;
    i32       nprobe;          /* IVF only */
    i32       code_bytes;      /* fine code, per vector */
    f32       recall;          /* measured, of the chosen config */
    f32       l2_lambda;       /* IVF term-3 blend actually chosen (1.0 = default) */
    i32       est_candidates;  /* IVF: expected vectors scanned per query at the chosen nprobe */

    void     *flat_ix;         /* FLAT / CASCADE(lo): QNIndex* (scheme) */
    Ivf      *ivf;             /* IVF */
    Hnsw     *hrouter;         /* IVF: graph router over ivf->centroids (NULL -> brute) */
    u32       router_seed;     /* HNSW build seed, kept so load rebuilds the same graph */
    Hnsw     *vhnsw;           /* ENG_HNSW: graph over the vectors themselves */
    f32      *vhnsw_pts;       /* ENG_HNSW: the f32 vectors the graph borrows */
    u32       vhnsw_seed;
    void    **cluster_ix;      /* IVF: [ivf->n_clusters] QNIndex* (NULL if empty) */
    void     *casc_hi;         /* CASCADE: the residual layer's QNIndex* */
    EngScheme casc_hi_s;

    /* diagnostics */
    i32       cand_bytes[ENG_NS];
    f32       flat_recall[ENG_NS];
    f32       ivf_recall[ENG_NS][ENG_NNPROBE];
} EngIndex;

const char *engine_scheme_name(EngScheme s)
{
    switch (s) {
    case ENG_Q1: return "q1"; case ENG_Q3: return "q3"; case ENG_Q2: return "q2";
    case ENG_Q4: return "q4"; case ENG_QMIX: return "qmix";
    case ENG_Q4FS: return "q4fs"; default: return "q5";
    }
}

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

/* ---- per-scheme dispatch -------------------------------------- */

static void *eng_flat_build_(Arena *a, EngScheme s, const f32 *v, i32 n, i32 dim)
{
    switch (s) {
    case ENG_Q1: return (void *)q1_build(a, v, n, dim);
    case ENG_Q3: return (void *)q3_build(a, v, n, dim);
    case ENG_Q2: return (void *)q2_build(a, v, n, dim);
    case ENG_Q4: return (void *)q4_build(a, v, n, dim);
    case ENG_QMIX: return (void *)qmix_build(a, v, n, dim);
    default:     return (void *)q5_build(a, v, n, dim);
    }
}

static void *eng_cluster_build_(Arena *a, EngScheme s, const f32 *v,
                                const i64 *ids, i32 nid, i32 dim, const f32 *c)
{
    switch (s) {
    case ENG_Q1: return (void *)q1_build_cluster(a, v, ids, nid, dim, c);
    case ENG_Q3: return (void *)q3_build_cluster(a, v, ids, nid, dim, c);
    case ENG_Q2: return (void *)q2_build_cluster(a, v, ids, nid, dim, c);
    case ENG_Q4: return (void *)q4_build_cluster(a, v, ids, nid, dim, c);
    case ENG_QMIX: return (void *)qmix_build_cluster(a, v, ids, nid, dim, c);
    default:     return (void *)q5_build_cluster(a, v, ids, nid, dim, c);
    }
}

static i32 eng_flat_search_(EngScheme s, void *ix, const f32 *q, i32 k,
                            i64 *ids, f32 *sc)
{
    switch (s) {
    case ENG_Q1: return q1_search((Q1Index *)ix, q, k, ids, sc);
    case ENG_Q3: return q3_search((Q3Index *)ix, q, k, ids, sc);
    case ENG_Q2: return q2_search((Q2Index *)ix, q, k, ids, sc);
    case ENG_Q4: return q4_search((Q4Index *)ix, q, k, ids, sc);
    case ENG_QMIX: return qmix_search((QMixIndex *)ix, q, k, ids, sc);
    case ENG_Q4FS: return q4fs_search((Q4FSIndex *)ix, q, k, ids, sc);
    default:     return q5_search((Q5Index *)ix, q, k, ids, sc);
    }
}

static i32 eng_l2_search_(EngScheme s, void *ix, const f32 *q, i32 k,
                          i64 *ids, f32 *sc)
{
    switch (s) {
    case ENG_Q1: return q1_search_l2((Q1Index *)ix, q, k, ids, sc);
    case ENG_Q3: return q3_search_l2((Q3Index *)ix, q, k, ids, sc);
    case ENG_Q2: return q2_search_l2((Q2Index *)ix, q, k, ids, sc);
    case ENG_Q4: return q4_search_l2((Q4Index *)ix, q, k, ids, sc);
    case ENG_QMIX: return qmix_search_l2((QMixIndex *)ix, q, k, ids, sc);
    case ENG_Q4FS: return q4fs_search_l2((Q4FSIndex *)ix, q, k, ids, sc);
    default:     return q5_search_l2((Q5Index *)ix, q, k, ids, sc);
    }
}

static i32 eng_bytes_(EngScheme s, i32 dim)
{
    switch (s) {
    case ENG_Q1: return q1_code_bytes(dim);
    case ENG_Q3: return q3_code_bytes(dim);
    case ENG_Q2: return q2_code_bytes(dim);
    case ENG_Q4: return q4_code_bytes(dim);
    case ENG_QMIX: return qmix_code_bytes(dim);
    case ENG_Q4FS: return q4fs_code_bytes(dim);
    default:     return q5_code_bytes(dim);
    }
}

/* All five uniform QNIndex structs share this leading layout (q1.c is
 * the template, q2/q4/q5 are sed-generated from it, q3.c matches field
 * for field) - so the cascade can walk the lo layer's code rows, and
 * the term-3 retune can reach l2_lambda, without the concrete type.
 * qmix's struct is NOT one of these (it holds three sub-indexes). */
typedef struct {
    void *arena;
    i32 dim, code_bytes, count;
    f32 *centroid, *codebook, *bounds;
    u8  *codes;
    f32 *rnorm;
    f32 *qn2;
    f32  l2_lambda;
} EngQHead;

/* set the IVF term-3 blend on one built index. */
static void eng_set_lambda_(EngScheme s, void *ix, f32 v)
{
    if (!ix) return;
    if (s == ENG_QMIX) ((QMixIndex *)ix)->l2_lambda = v;
    else if (s == ENG_Q4FS) ((Q4FSIndex *)ix)->l2_lambda = v;
    else               ((EngQHead *)ix)->l2_lambda = v;
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

/* ---- exact f32 cosine top-k (ground truth) ------------------- */

static void eng_exact_(const f32 *vecs, i32 n, i32 dim, const f32 *q, i32 k, i64 *out)
{
    f32 sc[64]; i32 have = 0, r; f32 qn;
    if (k > 64) k = 64;
    qn = (f32)eng_sqrt_(eng_ssq_(q, dim)); if (qn == 0.0f) qn = 1.0f;
    for (r = 0; r < n; ++r) {
        const f32 *row = vecs + (size_t)r * dim;
        f32 dot = vs_dot(q, row, dim), rn, cs;
        rn = (f32)eng_sqrt_(eng_ssq_(row, dim));
        cs = (rn > 0.0f) ? dot / (qn * rn) : 0.0f;
        eng_topk_(out, sc, k, &have, (i64)r, cs);
    }
}

/* ---- search ------------------------------------------------- */

/* cell selection: the HNSW graph when one was built, else the plain
 * nearest-centroid scan. Same contract as ivf_route (out[<=nprobe],
 * nearest first, returns count). */
static i32 eng_route_(const EngIndex *e, const f32 *query, i32 *out_clusters)
{
    if (e->hrouter) {
        i32 ef = e->nprobe * 4;
        if (ef < 32) ef = 32;
        return hnsw_route(e->hrouter, query, e->nprobe, ef, out_clusters);
    }
    return ivf_route(e->ivf, query, e->nprobe, out_clusters);
}

/* ENG_HNSW: graph over the L2-NORMALISED vectors, so a squared-L2
 * nearest-neighbour is a cosine nearest-neighbour (matches the flat /
 * IVF paths' scoring). The query is normalised the same way; score is
 * -d2 (larger = closer). Stack-only scratch, thread-safe. */
#define ENG_VHNSW_QMAX 8192
static i32 eng_vhnsw_search_(const EngIndex *e, const f32 *query, i32 k,
                             i64 *out_ids, f32 *out_scores)
{
    i32 lids[64]; f32 ld2[64];
    f32 qn[ENG_VHNSW_QMAX];
    const f32 *q = query;
    i32 ef, got, i, d = e->dim;
    if (k > 64) k = 64;
    ef = k * 16; if (ef < 192) ef = 192;
    if (d <= ENG_VHNSW_QMAX) {
        f64 s = 0.0; f32 inv;
        for (i = 0; i < d; ++i) s += (f64)query[i] * query[i];
        inv = (s > 0.0) ? (f32)(1.0 / eng_sqrt_(s)) : 1.0f;
        for (i = 0; i < d; ++i) qn[i] = query[i] * inv;
        q = qn;
    }
    got = hnsw_search_topk(e->vhnsw, q, k, ef, lids, ld2);
    for (i = 0; i < got; ++i) { out_ids[i] = (i64)lids[i]; out_scores[i] = -ld2[i]; }
    return got;
}

static i32 eng_ivf_search_(const EngIndex *e, const f32 *query, i32 k,
                           i64 *out_ids, f32 *out_scores)
{
    Arena *a = e->arena;
    Arena_Mark m = arena_snapshot(a);
    i32 probe[64];
    i64 *lids = (i64 *)arena_alloc(a, (size_t)k * sizeof(i64));
    f32 *lsc  = (f32 *)arena_alloc(a, (size_t)k * sizeof(f32));
    i32 np = eng_route_(e, query, probe);
    i32 have = 0, pi, j;

    for (pi = 0; pi < np; ++pi) {
        i32 c = probe[pi], mc, got;
        const i64 *mem = ivf_members(e->ivf, c, &mc);
        if (mc <= 0 || e->cluster_ix[c] == (void *)0) continue;
        got = eng_l2_search_(e->scheme, e->cluster_ix[c], query, k, lids, lsc);
        for (j = 0; j < got; ++j)
            eng_topk_(out_ids, out_scores, k, &have, mem[lids[j]], lsc[j]);
    }
    arena_rewind(a, m);
    return have;
}

/* ---- residual cascade (ENG_CASCADE) ------------------------- *
 * lo layer quantizes v against the global centroid; hi layer quantizes
 * the reconstruction residual (v - v_hat_lo) against zero. Combined
 * code = lo_bytes + hi_bytes. Search: lo shortlist, then re-rank the
 * shortlist by lo_dot + hi_dot (both against the same query residual
 * q - c_lo). Beats a single scheme at equal total bytes on data with a
 * fat per-dimension error tail (diffuse, not tightly clustered - that
 * case is still IVF's).                                                */

static void eng_lut_resid_(EngScheme s, void *ix, const f32 *resid, f32 *lut)
{
    switch (s) {
    case ENG_Q1: q1_lut_resid((Q1Index *)ix, resid, lut); break;
    case ENG_Q3: q3_lut_resid((Q3Index *)ix, resid, lut); break;
    case ENG_Q2: q2_lut_resid((Q2Index *)ix, resid, lut); break;
    case ENG_Q4: q4_lut_resid((Q4Index *)ix, resid, lut); break;
    default:     q5_lut_resid((Q5Index *)ix, resid, lut); break;
    }
}
static i32 eng_scan_dot_(EngScheme s, void *ix, const f32 *lut, i32 k,
                         i64 *ids, f32 *sc)
{
    switch (s) {
    case ENG_Q1: return q1_scan_dot((Q1Index *)ix, lut, k, ids, sc);
    case ENG_Q3: return q3_scan_dot((Q3Index *)ix, lut, k, ids, sc);
    case ENG_Q2: return q2_scan_dot((Q2Index *)ix, lut, k, ids, sc);
    case ENG_Q4: return q4_scan_dot((Q4Index *)ix, lut, k, ids, sc);
    default:     return q5_scan_dot((Q5Index *)ix, lut, k, ids, sc);
    }
}
static f32 eng_code_dot_(EngScheme s, void *ix, i32 id, const f32 *lut)
{
    switch (s) {
    case ENG_Q1: return q1_code_dot((Q1Index *)ix, id, lut);
    case ENG_Q3: return q3_code_dot((Q3Index *)ix, id, lut);
    case ENG_Q2: return q2_code_dot((Q2Index *)ix, id, lut);
    case ENG_Q4: return q4_code_dot((Q4Index *)ix, id, lut);
    default:     return q5_code_dot((Q5Index *)ix, id, lut);
    }
}
static void eng_reconstruct_(EngScheme s, void *ix, const u8 *row, f32 *out)
{
    switch (s) {
    case ENG_Q1: q1_reconstruct_into((Q1Index *)ix, row, out); break;
    case ENG_Q3: q3_reconstruct_into((Q3Index *)ix, row, out); break;
    case ENG_Q2: q2_reconstruct_into((Q2Index *)ix, row, out); break;
    case ENG_Q4: q4_reconstruct_into((Q4Index *)ix, row, out); break;
    default:     q5_reconstruct_into((Q5Index *)ix, row, out); break;
    }
}

EngIndex *engine_build_cascade(Arena *a, const f32 *vecs, i32 n, i32 dim,
                               EngScheme lo_s, EngScheme hi_s)
{
    EngIndex *e = (EngIndex *)arena_alloc(a, sizeof(EngIndex));
    void *lo;
    const EngQHead *lh;
    f32 *resid, *vhat;
    i32 i, d;

    e->arena = a; e->dim = dim; e->count = n;
    e->topo = ENG_CASCADE; e->scheme = lo_s; e->casc_hi_s = hi_s;
    e->ivf = (Ivf *)0; e->cluster_ix = (void **)0;

    lo = eng_flat_build_(a, lo_s, vecs, n, dim);
    e->flat_ix = lo;
    lh = (const EngQHead *)lo;

    /* the hi layer needs the full n*dim residual matrix to calibrate
     * against - it is built from it, so it cannot be rewound afterwards.
     * engine_build_cascade therefore leaves an n*dim*4 f32 residual slab
     * on the arena for the life of the index. Fine for the diffuse /
     * moderate-n corpora the cascade is for; use IVF for large n. */
    vhat = (f32 *)arena_alloc(a, (size_t)dim * sizeof(f32));
    resid = (f32 *)arena_alloc(a, (size_t)n * dim * sizeof(f32));
    for (i = 0; i < n; ++i) {
        const u8 *row = lh->codes + (size_t)i * lh->code_bytes;
        eng_reconstruct_(lo_s, lo, row, vhat);
        for (d = 0; d < dim; ++d)
            resid[(size_t)i * dim + d] = vecs[(size_t)i * dim + d] - vhat[d];
    }
    e->casc_hi = eng_flat_build_(a, hi_s, resid, n, dim);

    e->code_bytes = eng_bytes_(lo_s, dim) + eng_bytes_(hi_s, dim);
    e->recall = 0.0f;
    return e;
}

static i32 eng_cascade_search_(const EngIndex *e, const f32 *query, i32 k,
                               i64 *out_ids, f32 *out_scores)
{
    Arena *a = e->arena;
    Arena_Mark m = arena_snapshot(a);
    const EngQHead *lh = (const EngQHead *)e->flat_ix;
    i32 dim = e->dim, shortn = (8 * k < e->count) ? 8 * k : e->count;
    f32 *rq  = (f32 *)arena_alloc(a, (size_t)dim * sizeof(f32));
    f32 *llo = (f32 *)arena_alloc(a, (size_t)eng_bytes_(e->scheme, dim) * 256 * sizeof(f32));
    f32 *lhi = (f32 *)arena_alloc(a, (size_t)eng_bytes_(e->casc_hi_s, dim) * 256 * sizeof(f32));
    i64 *sids = (i64 *)arena_alloc(a, (size_t)shortn * sizeof(i64));
    f32 *ssc  = (f32 *)arena_alloc(a, (size_t)shortn * sizeof(f32));
    i32 got, j, have = 0;

    (void)vs_sub_normsq(rq, query, lh->centroid, dim);
    eng_lut_resid_(e->scheme, e->flat_ix, rq, llo);
    eng_lut_resid_(e->casc_hi_s, e->casc_hi, rq, lhi);

    got = eng_scan_dot_(e->scheme, e->flat_ix, llo, shortn, sids, ssc);
    for (j = 0; j < got; ++j) {
        f32 combined = ssc[j] + eng_code_dot_(e->casc_hi_s, e->casc_hi,
                                              (i32)sids[j], lhi);
        eng_topk_(out_ids, out_scores, k, &have, sids[j], combined);
    }
    arena_rewind(a, m);
    return have;
}

i32 engine_search(const EngIndex *e, const f32 *query, i32 k,
                  i64 *out_ids, f32 *out_scores)
{
    if (e->topo == ENG_FLAT)
        return eng_flat_search_(e->scheme, e->flat_ix, query, k, out_ids, out_scores);
    if (e->topo == ENG_CASCADE)
        return eng_cascade_search_(e, query, k, out_ids, out_scores);
    if (e->topo == ENG_HNSW)
        return eng_vhnsw_search_(e, query, k, out_ids, out_scores);
    return eng_ivf_search_(e, query, k, out_ids, out_scores);
}

/* ---- arena-free search + threaded batch --------------------- *
 * engine_search allocates per call; the *_into path takes caller
 * scratch so a thread pool can run many queries at once on one index.  */

typedef struct {
    f32 *residual;   /* engine_scratch_floats(e) worth of f32, laid out: */
    f32 *lut;        /*   residual[dim] then lut[code_bytes*256]          */
    i64 *lids;       /* k */
    f32 *lsc;        /* k */
    i32 *probe;      /* 64 */
} EngScratch;

i32 engine_scratch_floats(const EngIndex *e)
{
    return e->dim + e->code_bytes * 256;
}

static i32 eng_flat_search_into_(EngScheme s, void *ix, const f32 *q, i32 k,
                                 i64 *ids, f32 *sc, f32 *res, f32 *lut)
{
    switch (s) {
    case ENG_Q1: return q1_search_into((Q1Index *)ix, q, k, ids, sc, res, lut);
    case ENG_Q3: return q3_search_into((Q3Index *)ix, q, k, ids, sc, res, lut);
    case ENG_Q2: return q2_search_into((Q2Index *)ix, q, k, ids, sc, res, lut);
    case ENG_Q4: return q4_search_into((Q4Index *)ix, q, k, ids, sc, res, lut);
    case ENG_QMIX: return qmix_search_into((QMixIndex *)ix, q, k, ids, sc, res, lut);
    case ENG_Q4FS: return q4fs_search_into((Q4FSIndex *)ix, q, k, ids, sc, res, lut);
    default:     return q5_search_into((Q5Index *)ix, q, k, ids, sc, res, lut);
    }
}

static i32 eng_l2_search_into_(EngScheme s, void *ix, const f32 *q, i32 k,
                               i64 *ids, f32 *sc, f32 *res, f32 *lut)
{
    switch (s) {
    case ENG_Q1: return q1_search_l2_into((Q1Index *)ix, q, k, ids, sc, res, lut);
    case ENG_Q3: return q3_search_l2_into((Q3Index *)ix, q, k, ids, sc, res, lut);
    case ENG_Q2: return q2_search_l2_into((Q2Index *)ix, q, k, ids, sc, res, lut);
    case ENG_Q4: return q4_search_l2_into((Q4Index *)ix, q, k, ids, sc, res, lut);
    case ENG_QMIX: return qmix_search_l2_into((QMixIndex *)ix, q, k, ids, sc, res, lut);
    case ENG_Q4FS: return q4fs_search_l2_into((Q4FSIndex *)ix, q, k, ids, sc, res, lut);
    default:     return q5_search_l2_into((Q5Index *)ix, q, k, ids, sc, res, lut);
    }
}

i32 engine_search_into(const EngIndex *e, const f32 *query, i32 k,
                               i64 *out_ids, f32 *out_scores, EngScratch *sc)
{
    i32 have = 0, np, pi, j;
    if (e->topo == ENG_FLAT)
        return eng_flat_search_into_(e->scheme, e->flat_ix, query, k,
                                     out_ids, out_scores, sc->residual, sc->lut);
    if (e->topo == ENG_CASCADE)   /* not batched - uses the arena internally */
        return eng_cascade_search_(e, query, k, out_ids, out_scores);
    if (e->topo == ENG_HNSW)      /* stack-only scratch, thread-safe */
        return eng_vhnsw_search_(e, query, k, out_ids, out_scores);
    np = eng_route_(e, query, sc->probe);
    for (pi = 0; pi < np; ++pi) {
        i32 c = sc->probe[pi], mc, got;
        const i64 *mem = ivf_members(e->ivf, c, &mc);
        if (mc <= 0 || e->cluster_ix[c] == (void *)0) continue;
        got = eng_l2_search_into_(e->scheme, e->cluster_ix[c], query, k,
                                  sc->lids, sc->lsc, sc->residual, sc->lut);
        for (j = 0; j < got; ++j)
            eng_topk_(out_ids, out_scores, k, &have, mem[sc->lids[j]], sc->lsc[j]);
    }
    return have;
}

#ifdef ENGINE_THREADS
/* one query per job is fairly coarse already (~200us of LUT build +
 * cluster scans); claim them a few at a time so 200-query batches
 * actually spread across all workers rather than one grabbing the
 * default 256-wide chunk. */
#define FGJOB_CHUNK 4
#define FGJOB_IMPLEMENTATION
#include "fine_grained.h"

typedef struct {
    const EngIndex *e;
    const f32 *queries;
    i32 dim, k;
    i64 *out_ids;
    f32 *out_scores;
    EngScratch *scr;         /* [nq], one per query row */
} EngBatchCtx;

static void engine_batch_job_(i64 qi, void *ctx_)
{
    EngBatchCtx *c = (EngBatchCtx *)ctx_;
    engine_search_into(c->e, c->queries + (size_t)qi * c->dim, c->k,
                        c->out_ids + (size_t)qi * c->k,
                        c->out_scores + (size_t)qi * c->k, &c->scr[qi]);
}

/* Top-k for `nq` queries at once, one worker thread per chunk of query
 * rows. out_ids / out_scores are [nq * k], row-major. Spawns and joins a
 * pool per call (~190us) - a hot caller should keep its own Thread_Pool
 * and drive engine_search_into directly. */
void engine_search_batch(const EngIndex *e, const f32 *queries, i32 nq, i32 k,
                         i64 *out_ids, f32 *out_scores)
{
    Arena *a = e->arena;
    Arena_Mark m = arena_snapshot(a);
    i32 sf = engine_scratch_floats(e), i;
    EngScratch *scr = (EngScratch *)arena_alloc(a, (size_t)nq * sizeof(EngScratch));
    f32 *fslab = (f32 *)arena_alloc(a, (size_t)nq * sf * sizeof(f32));
    i64 *islab = (i64 *)arena_alloc(a, (size_t)nq * k * sizeof(i64));
    f32 *sslab = (f32 *)arena_alloc(a, (size_t)nq * k * sizeof(f32));
    i32 *pslab = (i32 *)arena_alloc(a, (size_t)nq * 64 * sizeof(i32));
    Thread_Pool pool;
    EngBatchCtx c;

    for (i = 0; i < nq; ++i) {
        scr[i].residual = fslab + (size_t)i * sf;
        scr[i].lut      = scr[i].residual + e->dim;
        scr[i].lids     = islab + (size_t)i * k;
        scr[i].lsc      = sslab + (size_t)i * k;
        scr[i].probe    = pslab + (size_t)i * 64;
    }
    c.e = e; c.queries = queries; c.dim = e->dim; c.k = k;
    c.out_ids = out_ids; c.out_scores = out_scores; c.scr = scr;

    thread_pool_init(&pool, -1);
    thread_pool_run_range(&pool, 0, nq, engine_batch_job_, &c);
    thread_pool_destroy(&pool);
    arena_rewind(a, m);
}
#endif /* ENGINE_THREADS */

/* ---- recall probe ----------------------------------------- */

/* one config's recall@k over the pre-computed ground truth. `topo`
 * selects flat vs ivf; for ivf `ix_or_ivf` is unused and the EngIndex
 * fields (ivf, cluster_ix, scheme, nprobe) are read instead. */
static f32 eng_probe_(EngIndex *tmp, EngTopo topo, EngScheme s, void *flat_ix,
                      const f32 *vecs, i32 dim, const i32 *qrows, i32 nq,
                      const i64 *gt, i32 k)
{
    Arena *a = tmp->arena;
    Arena_Mark m = arena_snapshot(a);
    i64 *got = (i64 *)arena_alloc(a, (size_t)k * sizeof(i64));
    f32 *gsc = (f32 *)arena_alloc(a, (size_t)k * sizeof(f32));
    i32 hits = 0, total = 0, qi, e, x, ng;
    EngTopo save_t = tmp->topo; EngScheme save_s = tmp->scheme;

    tmp->topo = topo; tmp->scheme = s; tmp->flat_ix = flat_ix;
    for (qi = 0; qi < nq; ++qi) {
        const f32 *q = vecs + (size_t)qrows[qi] * dim;
        const i64 *truth = gt + (size_t)qi * k;
        ng = engine_search(tmp, q, k, got, gsc);
        for (e = 0; e < ng; ++e)
            for (x = 0; x < k; ++x)
                if (got[e] == truth[x]) { ++hits; break; }
        total += k;
    }
    tmp->topo = save_t; tmp->scheme = save_s;
    arena_rewind(a, m);
    return (total > 0) ? (f32)hits / (f32)total : 0.0f;
}

/* how big a probe-recall gain must be to be real and not eval-set noise:
 * ~1 binomial standard error of a rate-p estimate over `samples` trials,
 * floored so a tiny probe set cannot wave anything through. Used by the
 * cell-count and term-3-blend adopt decisions. */
static f32 eng_noise_margin_(f32 p, i32 samples)
{
    f32 se;
    if (samples < 1) samples = 1;
    if (p < 0.0f) p = 0.0f; else if (p > 1.0f) p = 1.0f;
    se = (f32)eng_sqrt_((f64)(p * (1.0f - p) / (f32)samples));
    return se > 0.01f ? se : 0.01f;
}

/* ---- build ---------------------------------------------- */

EngIndex *engine_build(Arena *a, const f32 *vecs, i32 n, i32 dim, EngGoal goal)
{
    EngIndex *e = (EngIndex *)arena_alloc(a, sizeof(EngIndex));
    i32 k      = (goal.k > 0) ? goal.k : 10;
    i32 want   = (goal.eval_queries > 0) ? goal.eval_queries : 96;
    i32 niters = (goal.n_iters > 0) ? goal.n_iters : 12;
    u32 seed   = goal.seed ? goal.seed : 0x5eedu;
    /* tune against caller-supplied queries when given (asymmetric
     * workloads), else against corpus rows. `qsrc` is the array `qrows`
     * indexes for QUERIES; the index is always built over `vecs`. */
    const f32 *qsrc = (goal.eval_q && goal.n_eval_q > 0) ? goal.eval_q : vecs;
    i32 qsrc_n = (qsrc == vecs) ? n : goal.n_eval_q;
    i32 nq     = (want < qsrc_n) ? want : qsrc_n;
    i32 step   = (nq > 0) ? qsrc_n / nq : 1;
    i32 *qrows;
    i64 *gt;
    void *flat_ix[ENG_NS];
    void **all_cix[ENG_NS];
    void **reuse_q4_cix = (void **)0;   /* bake-off's q4 clusters, if reusable */
    i32 nc, s, pj, i, c;
    /* selection state */
    i32  best_bytes = 0; f32 best_rec = -1.0f;
    EngTopo best_topo = ENG_FLAT; EngScheme best_s = ENG_Q5; i32 best_np = 1;
    i32 have_best = 0;
    f32 vhnsw_rec = -1.0f; i32 vhnsw_bytes = dim * (i32)sizeof(f32);

    if (k > 64) k = 64;
    if (step < 1) step = 1;

    e->arena = a; e->dim = dim; e->count = n;
    e->topo = ENG_FLAT; e->scheme = ENG_Q5; e->nprobe = 1;
    e->flat_ix = (void *)0; e->ivf = (Ivf *)0; e->cluster_ix = (void **)0;
    e->hrouter = (Hnsw *)0; e->vhnsw = (Hnsw *)0; e->vhnsw_pts = (f32 *)0;

    /* eval query rows + their exact ground truth over `vecs` (once) */
    qrows = (i32 *)arena_alloc(a, (size_t)(nq > 0 ? nq : 1) * sizeof(i32));
    for (i = 0; i < nq; ++i) qrows[i] = (i * step) % (qsrc_n > 0 ? qsrc_n : 1);
    gt = (i64 *)arena_alloc(a, (size_t)(nq > 0 ? nq : 1) * k * sizeof(i64));
    for (i = 0; i < nq; ++i)
        eng_exact_(vecs, n, dim, qsrc + (size_t)qrows[i] * dim, k, gt + (size_t)i * k);

    /* 6 flat candidates (q1-q5 + qmix) */
    for (s = 0; s < ENG_NS; ++s) {
        flat_ix[s] = eng_flat_build_(a, (EngScheme)s, vecs, n, dim);
        e->cand_bytes[s] = eng_bytes_((EngScheme)s, dim);
        e->flat_recall[s] = eng_probe_(e, ENG_FLAT, (EngScheme)s, flat_ix[s],
                                       qsrc, dim, qrows, nq, gt, k);
    }

    /* IVF: train the coarse quantizer on a subsample, assign the full
     * set to it (the standard FAISS "train small, add everything" split).
     * The cell count is itself measured: sqrt(n) and 4*sqrt(n) are each
     * trained + q4-probed at a mid nprobe, and the better one is kept.
     * More cells lift recall on weakly-clustered data but fragment
     * tightly-clustered data, so the choice is left to the probe, not a
     * constant. The graph router (built when nc >= ENG_HNSW_MIN_NC) is
     * what keeps the many-cell option's routing cost sub-linear. */
    e->router_seed = seed ^ 0x484e5357u;   /* 'HNSW' */
    {
        i32 sq = ivf_suggest_clusters(n);
        i32 lim = (n >= 32) ? n / 16 : n;
        i32 fine = 4 * sq;
        i32 hh = nq / 2, ho = nq - hh;        /* probe / held-out split */
        Ivf *cand_iv[2]; Hnsw *cand_hr[2]; void **cand_cix[2]; i32 cand_nc[2], ncand = 0;
        f32 pr[2], hr_[2];
        i32 ci, win, keep_q4;

        if (fine >= lim) fine = 0;            /* no room for a finer grid */

        for (ci = 0; ci < (fine ? 2 : 1); ++ci) {
            i32 tnc = ci == 0 ? sq : fine;
            i32 full = (ci == 0);            /* incumbent trains at full quality;
                                             * challenger is a cheap comparison */
            i32 it = full ? niters : (niters + 1) / 2;
            i32 cap = 40 * tnc, tn, c;
            Ivf *iv; Hnsw *hr = (Hnsw *)0; void **cix;
            if (tnc > lim) tnc = lim;
            if (tnc < 1) tnc = 1;
            if (cap < ENG_IVF_TRAIN_MIN) cap = ENG_IVF_TRAIN_MIN;
            if (cap > ENG_IVF_TRAIN_CAP) cap = ENG_IVF_TRAIN_CAP;
            if (!full && cap > 12000) cap = 12000;
            tn = (n < cap) ? n : cap;
            iv = ivf_train(a, vecs, tn, dim, tnc, it, seed);
            if (tn < n) ivf_assign_full(iv, vecs, n);
            tnc = iv->n_clusters;
            if (tnc >= ENG_HNSW_MIN_NC) {
                HnswParams hp;
                hp.M = 16; hp.ef_construction = 96; hp.seed = e->router_seed;
                hr = hnsw_build(a, iv->centroids, tnc, dim, hp);
            }
            cix = (void **)arena_alloc(a, (size_t)tnc * sizeof(void *));
            for (c = 0; c < tnc; ++c) {
                i32 mc; const i64 *mem = ivf_members(iv, c, &mc);
                cix[c] = (mc > 0)
                    ? (void *)q4_build_cluster(a, vecs, mem, mc, dim,
                                               iv->centroids + (size_t)c * dim)
                    : (void *)0;
            }
            e->ivf = iv; e->hrouter = hr; e->cluster_ix = cix;
            e->scheme = ENG_Q4; e->nprobe = (8 < tnc) ? 8 : tnc;
            pr[ci]  = eng_probe_(e, ENG_IVF, ENG_Q4, (void *)0, qsrc, dim,
                                 qrows, hh, gt, k);
            hr_[ci] = eng_probe_(e, ENG_IVF, ENG_Q4, (void *)0, qsrc, dim,
                                 qrows + hh, ho, gt + (size_t)hh * k, k);
            cand_iv[ci] = iv; cand_hr[ci] = hr; cand_cix[ci] = cix; cand_nc[ci] = tnc;
            ncand = ci + 1;
        }

        /* adopt the finer grid only if it beats sqrt(n) by more than the
         * eval-set noise on the probe half AND does not regress on the
         * held-out half (guards the q4-proxy overfit). */
        win = 0;
        if (ncand == 2) {
            f32 marg = eng_noise_margin_(pr[0], hh * k);
            if (pr[1] >= pr[0] + marg && hr_[1] >= hr_[0] - marg) win = 1;
        }
        e->ivf = cand_iv[win]; e->hrouter = cand_hr[win]; nc = cand_nc[win];

        /* the winner's q4 clusters are already built - hand them to the
         * scheme sweep so q4 is not rebuilt. If the challenger won it was
         * trained cheap; retrain it at full quality now (rare path). */
        keep_q4 = 1;
        if (win == 1) {
            i32 cap = 40 * nc, tn;
            Ivf *iv2;
            if (cap < ENG_IVF_TRAIN_MIN) cap = ENG_IVF_TRAIN_MIN;
            if (cap > ENG_IVF_TRAIN_CAP) cap = ENG_IVF_TRAIN_CAP;
            tn = (n < cap) ? n : cap;
            iv2 = ivf_train(a, vecs, tn, dim, nc, niters, seed);
            if (tn < n) ivf_assign_full(iv2, vecs, n);
            nc = iv2->n_clusters;
            e->ivf = iv2;
            if (nc >= ENG_HNSW_MIN_NC) {
                HnswParams hp;
                hp.M = 16; hp.ef_construction = 96; hp.seed = e->router_seed;
                e->hrouter = hnsw_build(a, iv2->centroids, nc, dim, hp);
            } else {
                e->hrouter = (Hnsw *)0;
            }
            keep_q4 = 0;   /* centroids changed - cannot reuse the bake-off clusters */
        }
        reuse_q4_cix = keep_q4 ? cand_cix[win] : (void **)0;
    }
    for (s = 0; s < ENG_NS; ++s) {
        void **cix;
        if (s == ENG_Q4 && reuse_q4_cix) {
            cix = reuse_q4_cix;
        } else {
            cix = (void **)arena_alloc(a, (size_t)nc * sizeof(void *));
            for (c = 0; c < nc; ++c) {
                i32 mc; const i64 *mem = ivf_members(e->ivf, c, &mc);
                cix[c] = (mc > 0)
                    ? eng_cluster_build_(a, (EngScheme)s, vecs, mem, mc, dim,
                                         e->ivf->centroids + (size_t)c * dim)
                    : (void *)0;
            }
        }
        all_cix[s] = cix;
        e->cluster_ix = cix; e->scheme = (EngScheme)s;
        for (pj = 0; pj < ENG_NNPROBE; ++pj) {
            e->nprobe = (ENG_NPROBES[pj] < nc) ? ENG_NPROBES[pj] : nc;
            e->ivf_recall[s][pj] = eng_probe_(e, ENG_IVF, (EngScheme)s, (void *)0,
                                              qsrc, dim, qrows, nq, gt, k);
        }
    }

    /* ENG_HNSW candidate (exact f32 graph over unit-norm vectors). The
     * highest-recall topology but dim*4 bytes/vec and a slow build, so
     * it is only constructed when it could actually be selected: a
     * MAX_ACC budget that admits f32, or a MIN_SIZE target no quantised
     * config reached. */
    {
        f32 bq = 0.0f;
        i32 want_vh, s2, p2;
        for (s2 = 0; s2 < ENG_NS; ++s2) {
            if (e->flat_recall[s2] > bq) bq = e->flat_recall[s2];
            for (p2 = 0; p2 < ENG_NNPROBE; ++p2)
                if (e->ivf_recall[s2][p2] > bq) bq = e->ivf_recall[s2][p2];
        }
        want_vh = (goal.mode == ENG_MAX_ACC)
                ? (vhnsw_bytes <= goal.max_code_bytes)
                : (bq < goal.target_recall);
        if (want_vh && n >= 64 && (i64)n <= ENG_VHNSW_MAXN &&
            (size_t)n * dim * sizeof(f32) <= ENG_VHNSW_MAXBYTES) {
            f32 *pc = (f32 *)arena_alloc(a, (size_t)n * dim * sizeof(f32));
            HnswParams hp;
            i32 ri, rrd;
            for (ri = 0; ri < n; ++ri) {
                const f32 *src = vecs + (size_t)ri * dim;
                f32 *dst = pc + (size_t)ri * dim;
                f64 ss = 0.0; f32 inv;
                for (rrd = 0; rrd < dim; ++rrd) ss += (f64)src[rrd] * src[rrd];
                inv = (ss > 0.0) ? (f32)(1.0 / eng_sqrt_(ss)) : 1.0f;
                for (rrd = 0; rrd < dim; ++rrd) dst[rrd] = src[rrd] * inv;
            }
            hp.M = 24; hp.seed = seed ^ 0x56484e57u; /* 'VHNW' */
            hp.ef_construction = (n <= 12000) ? 256 : 128;   /* build stays tractable */
            e->vhnsw = hnsw_build(a, pc, n, dim, hp);
            e->vhnsw_pts = pc; e->vhnsw_seed = hp.seed;
            vhnsw_rec = eng_probe_(e, ENG_HNSW, ENG_Q5, (void *)0,
                                   qsrc, dim, qrows, nq, gt, k);
        }
    }

    /* ---- select over all 5 flat + 5*NNPROBE ivf configs ---- */
    for (s = 0; s < ENG_NS; ++s) {
        i32 by = e->cand_bytes[s];
        /* flat */
        {
            f32 rc = e->flat_recall[s];
            i32 take = 0;
            if (goal.mode == ENG_MIN_SIZE) {
                if (rc >= goal.target_recall &&
                    (!have_best || by < best_bytes ||
                     (by == best_bytes && rc > best_rec)))
                    take = 1;
            } else {
                if (by <= goal.max_code_bytes &&
                    (!have_best || rc > best_rec ||
                     (rc == best_rec && by < best_bytes)))
                    take = 1;
            }
            if (take) { have_best = 1; best_bytes = by; best_rec = rc;
                        best_topo = ENG_FLAT; best_s = (EngScheme)s; best_np = 0; }
        }
        /* ivf, each nprobe */
        for (pj = 0; pj < ENG_NNPROBE; ++pj) {
            f32 rc = e->ivf_recall[s][pj];
            i32 np = (ENG_NPROBES[pj] < nc) ? ENG_NPROBES[pj] : nc;
            i32 take = 0;
            if (goal.mode == ENG_MIN_SIZE) {
                if (rc >= goal.target_recall &&
                    (!have_best || by < best_bytes ||
                     (by == best_bytes && rc > best_rec) ||
                     (by == best_bytes && rc == best_rec &&
                      best_topo == ENG_IVF && np < best_np)))
                    take = 1;
            } else {
                if (by <= goal.max_code_bytes &&
                    (!have_best || rc > best_rec ||
                     (rc == best_rec && by < best_bytes)))
                    take = 1;
            }
            if (take) { have_best = 1; best_bytes = by; best_rec = rc;
                        best_topo = ENG_IVF; best_s = (EngScheme)s; best_np = np; }
        }
    }

    /* ENG_HNSW: exact f32 graph, dim*4 bytes/vec */
    if (e->vhnsw) {
        i32 take = 0;
        if (goal.mode == ENG_MIN_SIZE) {
            if (vhnsw_rec >= goal.target_recall &&
                (!have_best || vhnsw_bytes < best_bytes ||
                 (vhnsw_bytes == best_bytes && vhnsw_rec > best_rec)))
                take = 1;
        } else {
            if (vhnsw_bytes <= goal.max_code_bytes &&
                (!have_best || vhnsw_rec > best_rec ||
                 (vhnsw_rec == best_rec && vhnsw_bytes < best_bytes)))
                take = 1;
        }
        if (take) { have_best = 1; best_bytes = vhnsw_bytes; best_rec = vhnsw_rec;
                    best_topo = ENG_HNSW; best_s = ENG_Q5; best_np = 0; }
    }

    if (!have_best) {
        /* nothing met the goal: MIN_SIZE -> most accurate; MAX_ACC -> smallest */
        best_topo = ENG_FLAT;
        best_s = (goal.mode == ENG_MIN_SIZE) ? ENG_Q5 : ENG_Q1;
        best_bytes = e->cand_bytes[best_s];
        best_rec = e->flat_recall[best_s];
        best_np = 0;
    }

    e->topo = best_topo; e->scheme = best_s; e->nprobe = best_np ? best_np : 1;
    e->code_bytes = best_bytes; e->recall = best_rec; e->l2_lambda = 1.0f;
    if (best_topo == ENG_FLAT) { e->flat_ix = flat_ix[best_s]; e->cluster_ix = (void **)0; }
    else                       { e->cluster_ix = all_cix[best_s]; e->flat_ix = (void *)0; }

    /* nprobe refinement: the coarse sweep found IVF+best_s at best_np; now
     * probe the full {1,2,4,8,16,32} set for that one scheme and take the
     * SMALLEST nprobe whose recall is within the probe noise of the best
     * (do not scan more cells than needed), further capped by
     * goal.max_candidates (a machine-independent query-cost budget). */
    if (best_topo == ENG_IVF) {
        i32 per = (nc > 0) ? (n / nc) : n;
        i32 fi, bi = 0; f32 fr[ENG_NPROBE_FINE_N], top = -1.0f, marg;
        e->cluster_ix = all_cix[best_s]; e->scheme = best_s;
        for (fi = 0; fi < ENG_NPROBE_FINE_N; ++fi) {
            i32 np = (ENG_NPROBE_FINE[fi] < nc) ? ENG_NPROBE_FINE[fi] : nc;
            e->nprobe = np;
            fr[fi] = eng_probe_(e, ENG_IVF, best_s, (void *)0, qsrc, dim,
                                qrows, nq, gt, k);
            if (fr[fi] > top) top = fr[fi];
        }
        marg = eng_noise_margin_(top, nq * k);
        for (fi = 0; fi < ENG_NPROBE_FINE_N; ++fi) {
            i32 np = (ENG_NPROBE_FINE[fi] < nc) ? ENG_NPROBE_FINE[fi] : nc;
            if (goal.max_candidates > 0 && np > 1 &&
                (i64)np * per > (i64)goal.max_candidates) continue;
            if (fr[fi] >= top - marg) { bi = fi; break; }
            bi = fi;   /* budget rejected everything above - keep the largest allowed */
        }
        e->nprobe = (ENG_NPROBE_FINE[bi] < nc) ? ENG_NPROBE_FINE[bi] : nc;
        e->recall = fr[bi];
    }
    e->est_candidates = (best_topo == ENG_IVF && nc > 0)
                      ? e->nprobe * (n / nc) : n;

    /* IVF term-3 blend retune. The per-cluster L2 estimate's third term
     * can be the true ||v-c||^2 (lambda=1: unbiased, but its variance
     * tracks each cluster's quantisation error, so an nprobe>1 merge of
     * heterogeneous clusters mis-ranks), the reconstruction ||vhat-c||^2
     * (lambda=0: a proper cross-cluster metric, but positively biased on
     * poorly-quantised true neighbours), or a blend. Homogeneous data
     * keeps lambda=1 (no change); only when a lower lambda measurably
     * beats it on the real recall probe is it adopted. nprobe==1 has no
     * merge, so it is left at the default. */
    if (best_topo == ENG_IVF && e->nprobe > 1) {
        static const f32 LAM[] = { 0.0f, 0.25f, 0.5f, 0.75f };
        void **cix = all_cix[best_s];
        i32 hh = nq / 2, ho = nq - hh;
        const i64 *gth = gt + (size_t)hh * k;
        f32 lam1_p, lam1_h, best_p = -1.0f, best_h = 0.0f, best_alt = 1.0f, marg;
        i32 li, cc;
        /* baseline: the default true-norm estimate (lambda = 1) */
        for (cc = 0; cc < nc; ++cc) eng_set_lambda_(best_s, cix[cc], 1.0f);
        lam1_p = eng_probe_(e, ENG_IVF, best_s, (void *)0, qsrc, dim, qrows, hh, gt, k);
        lam1_h = eng_probe_(e, ENG_IVF, best_s, (void *)0, qsrc, dim, qrows + hh, ho, gth, k);
        for (li = 0; li < (i32)(sizeof(LAM) / sizeof(LAM[0])); ++li) {
            f32 lp, lh;
            for (cc = 0; cc < nc; ++cc) eng_set_lambda_(best_s, cix[cc], LAM[li]);
            lp = eng_probe_(e, ENG_IVF, best_s, (void *)0, qsrc, dim, qrows, hh, gt, k);
            lh = eng_probe_(e, ENG_IVF, best_s, (void *)0, qsrc, dim, qrows + hh, ho, gth, k);
            if (lp > best_p) { best_p = lp; best_h = lh; best_alt = LAM[li]; }
        }
        /* adopt a blend only if it clears lambda=1 by more than the
         * probe-set noise AND does not regress on the held-out half -
         * otherwise the sweep is fitting eval noise. */
        marg = eng_noise_margin_(lam1_p, hh * k);
        if (best_p >= lam1_p + marg && best_h >= lam1_h - marg) {
            for (cc = 0; cc < nc; ++cc) eng_set_lambda_(best_s, cix[cc], best_alt);
            e->recall = 0.5f * (best_p + best_h); e->l2_lambda = best_alt;
        } else {
            for (cc = 0; cc < nc; ++cc) eng_set_lambda_(best_s, cix[cc], 1.0f);
            e->recall = 0.5f * (lam1_p + lam1_h); e->l2_lambda = 1.0f;
        }
    }

    /* q4 -> q4fs swap-in: same calibration, pshufb scan + exact rerank,
     * so recall is unchanged; only worthwhile once an index is large. */
    if (best_s == ENG_Q4) {
        if (e->topo == ENG_FLAT && e->count >= ENG_Q4FS_MIN) {
            e->flat_ix = q4fs_wrap(a, (const Q4Index *)e->flat_ix);
            e->scheme = ENG_Q4FS;
        } else if (e->topo == ENG_IVF && e->count >= ENG_Q4FS_MIN) {
            i32 cc;
            for (cc = 0; cc < nc; ++cc)
                if (e->cluster_ix[cc])
                    e->cluster_ix[cc] = q4fs_wrap(a, (const Q4Index *)e->cluster_ix[cc]);
            e->scheme = ENG_Q4FS;
        }
    }
    return e;
}

/* Fast build: no bake-off. IVF (sqrt(n) cells) + q4, nprobe grown from 8 (max 32) until a
 * 32-query probe reaches `target` recall@k. Costs one
 * k-means and one q4 pass instead of ~20 candidate builds; tiny sets stay flat q5. */
EngIndex *engine_build_fast(Arena *a, const f32 *vecs, i32 n, i32 dim, i32 k, f32 target, u32 seed)
{
    EngIndex *e = (EngIndex *)arena_alloc(a, sizeof(EngIndex));
    i32 nc, cap, tn, c, nq, step, i, *qrows, np;
    i64 *gt;
    Ivf *iv;
    void **cix;
    f32 rec = 0.0f;

    memset(e, 0, sizeof(*e));
    e->arena = a; e->dim = dim; e->count = n; e->l2_lambda = 1.0f; e->nprobe = 1;
    if (k < 1) k = 10;
    if (k > 64) k = 64;
    if (n < 2048) {                                   /* small: exact-ish flat q5 is cheap */
        e->topo = ENG_FLAT; e->scheme = ENG_Q5;
        e->flat_ix = eng_flat_build_(a, ENG_Q5, vecs, n, dim);
        e->code_bytes = eng_bytes_(ENG_Q5, dim); e->recall = 1.0f; e->est_candidates = n;
        return e;
    }
    nc = ivf_suggest_clusters(n);
    cap = 40 * nc;
    if (cap < ENG_IVF_TRAIN_MIN) cap = ENG_IVF_TRAIN_MIN;
    if (cap > ENG_IVF_TRAIN_CAP) cap = ENG_IVF_TRAIN_CAP;
    tn = (n < cap) ? n : cap;
    iv = ivf_train(a, vecs, tn, dim, nc, 8, seed ? seed : 0x5eedu);
    if (tn < n) ivf_assign_full(iv, vecs, n);
    nc = iv->n_clusters;
    e->router_seed = (seed ? seed : 0x5eedu) ^ 0x484e5357u;
    if (nc >= ENG_HNSW_MIN_NC) {
        HnswParams hp;
        hp.M = 16; hp.ef_construction = 96; hp.seed = e->router_seed;
        e->hrouter = hnsw_build(a, iv->centroids, nc, dim, hp);
    }
    cix = (void **)arena_alloc(a, (size_t)nc * sizeof(void *));
    for (c = 0; c < nc; ++c) {
        i32 mc; const i64 *mem = ivf_members(iv, c, &mc);
        cix[c] = (mc > 0) ? (void *)q4_build_cluster(a, vecs, mem, mc, dim, iv->centroids + (size_t)c * dim) : (void *)0;
    }
    e->ivf = iv; e->cluster_ix = cix; e->topo = ENG_IVF; e->scheme = ENG_Q4;
    e->code_bytes = eng_bytes_(ENG_Q4, dim);

    nq = 32; if (nq > n) nq = n;
    step = n / nq; if (step < 1) step = 1;
    qrows = (i32 *)arena_alloc(a, (size_t)nq * sizeof(i32));
    gt = (i64 *)arena_alloc(a, (size_t)nq * k * sizeof(i64));
    for (i = 0; i < nq; ++i) { qrows[i] = (i * step) % n; eng_exact_(vecs, n, dim, vecs + (size_t)qrows[i] * dim, k, gt + (size_t)i * k); }
    for (np = 8; ; np *= 2) {
        e->nprobe = (np < nc) ? np : nc;
        rec = eng_probe_(e, ENG_IVF, ENG_Q4, (void *)0, vecs, dim, qrows, nq, gt, k);
        if (rec >= target || e->nprobe >= nc || np >= 32) break;   /* search scratch is sized for <= 32 probes */
    }
    e->recall = rec;
    e->est_candidates = e->nprobe * (n / nc);
    if (e->count >= ENG_Q4FS_MIN) {
        for (c = 0; c < nc; ++c)
            if (cix[c]) cix[c] = q4fs_wrap(a, (const Q4Index *)cix[c]);
        e->scheme = ENG_Q4FS;
    }
    return e;
}

#include "engine_disk.c"

/* ================================================================== */

/* ENGINE TODO (further modes):
 *  - nprobe auto-tuned to a latency budget, not a fixed {1,4,16} sweep.
 *  - a full HNSW over the vectors (not just the centroids) as its own
 *    topology, for the small-n / highest-recall regime.
 * DONE: residual cascade (engine_build_cascade); per-dim-block depth
 *   (qmix, ENG_QMIX in the sweep); 4-bit fast-scan (q4fs, swapped in for
 *   a chosen q4 once n >= ENG_Q4FS_MIN); consistent cross-cluster L2
 *   term-3 with the l2_lambda retune; HNSW graph coarse router (hnsw.c)
 *   with a measured cell count (sqrt(n) vs 4*sqrt(n)).
 */
