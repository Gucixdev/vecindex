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
 *       -I. engine.c -DENGINE_DEMO -lm -o engine && ./engine
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
    f64 g; i32 i;
    if (x <= 0.0) return 0.0;
    u.f = (f32)x; u.i = (u.i >> 1) + 0x1fbd1df5u; g = (f64)u.f;
    for (i = 0; i < 3; ++i) g = 0.5 * (g + x / g);
    return g;
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
    qn = (f32)eng_sqrt_((f64)vs_dot(q, q, dim)); if (qn == 0.0f) qn = 1.0f;
    for (r = 0; r < n; ++r) {
        const f32 *row = vecs + (size_t)r * dim;
        f32 dot = vs_dot(q, row, dim), rn, cs;
        rn = (f32)eng_sqrt_((f64)vs_dot(row, row, dim));
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

/* ================================================================== *
 * serialise / load a built index - so a consumer does not have to
 * rebuild it from raw vectors on every start. On-disk format is
 * LITTLE-ENDIAN, IEEE-754 f32/f64, magic+version guarded; scalars are
 * written as explicit i32/i64/f32, never raw structs. A big-endian host
 * byte-swaps on the way in/out (dead code on x86). Needs <stdio.h>.
 * ================================================================== */
#ifdef ENGINE_SERIALIZE
#include <stdio.h>

#define ENG_SER_MAGIC   0x35584756u   /* 'VGX5' - v5 adds the ENG_HNSW vector-graph topology */

static i32 eng_levels_(EngScheme s)
{
    switch (s) {
    case ENG_Q1: return 2; case ENG_Q3: return 3; case ENG_Q2: return 4;
    case ENG_Q4: return 16; case ENG_QMIX: case ENG_Q4FS: return 0; /* not a uniform scheme */
    default: return 256;
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
#define ENG_W32(f, p, n)  engs_wle_((f), (p), 4, (n))
#define ENG_R32(f, p, n)  engs_rle_((f), (p), 4, (n))
#define ENG_W64(f, p, n)  engs_wle_((f), (p), 8, (n))
#define ENG_R64(f, p, n)  engs_rle_((f), (p), 8, (n))

static int eng_q_save_(FILE *f, EngScheme s, i32 dim, const void *qix)
{
    const EngQHead *h = (const EngQHead *)qix;
    i32 L = eng_levels_(s), hdr[3];
    hdr[0] = h->dim; hdr[1] = h->code_bytes; hdr[2] = h->count;
    return ENG_W32(f, hdr, 3)
        || ENG_W32(f, &h->l2_lambda, 1)
        || ENG_W32(f, h->centroid, (size_t)dim)
        || ENG_W32(f, h->codebook, (size_t)dim * L)
        || ENG_W32(f, h->bounds,   (size_t)dim * (L - 1))
        || engs_wle_(f, h->codes,  1, (size_t)h->count * h->code_bytes)
        || ENG_W32(f, h->rnorm,    (size_t)h->count)
        || ENG_W32(f, h->qn2,      (size_t)h->count);
}
static void *eng_q_load_(Arena *a, FILE *f, EngScheme s, i32 dim)
{
    EngQHead *h = (EngQHead *)arena_alloc(a, sizeof(EngQHead));
    i32 L = eng_levels_(s), hdr[3], cnt;
    if (ENG_R32(f, hdr, 3) || ENG_R32(f, &h->l2_lambda, 1)) return (void *)0;
    h->arena = a; h->dim = hdr[0]; h->code_bytes = hdr[1]; h->count = hdr[2];
    cnt = h->count > 0 ? h->count : 1;
    h->centroid = (f32 *)arena_alloc(a, (size_t)dim * sizeof(f32));
    h->codebook = (f32 *)arena_alloc(a, (size_t)dim * L * sizeof(f32));
    h->bounds   = (f32 *)arena_alloc(a, (size_t)dim * (L - 1) * sizeof(f32));
    h->codes    = (u8  *)arena_alloc(a, (size_t)cnt * h->code_bytes);
    h->rnorm    = (f32 *)arena_alloc(a, (size_t)cnt * sizeof(f32));
    h->qn2      = (f32 *)arena_alloc(a, (size_t)cnt * sizeof(f32));
    if (ENG_R32(f, h->centroid, (size_t)dim)
        || ENG_R32(f, h->codebook, (size_t)dim * L)
        || ENG_R32(f, h->bounds,   (size_t)dim * (L - 1))
        || engs_rle_(f, h->codes,  1, (size_t)h->count * h->code_bytes)
        || ENG_R32(f, h->rnorm,    (size_t)h->count)
        || ENG_R32(f, h->qn2,      (size_t)h->count))
        return (void *)0;
    return (void *)h;
}

/* qmix: perm + rn2 + qn2 + l2_lambda, then its three sub-indexes reuse
 * eng_q_save_/eng_q_load_ (they are EngQHead-layout Q4/Q2/Q1 indexes). */
static int eng_qmix_save_(FILE *f, const void *qix)
{
    const QMixIndex *ix = (const QMixIndex *)qix;
    i32 hdr[6];
    hdr[0] = ix->dim; hdr[1] = ix->code_bytes; hdr[2] = ix->count;
    hdr[3] = ix->n_hi; hdr[4] = ix->n_mid; hdr[5] = ix->n_lo;
    return ENG_W32(f, hdr, 6)
        || ENG_W32(f, &ix->l2_lambda, 1)
        || ENG_W32(f, ix->perm, (size_t)ix->dim)
        || ENG_W32(f, ix->rn2,  (size_t)ix->count)
        || ENG_W32(f, ix->qn2,  (size_t)ix->count)
        || eng_q_save_(f, ENG_Q4, ix->n_hi,  ix->hi)
        || eng_q_save_(f, ENG_Q2, ix->n_mid, ix->mid)
        || eng_q_save_(f, ENG_Q1, ix->n_lo,  ix->lo);
}
static void *eng_qmix_load_(Arena *a, FILE *f)
{
    QMixIndex *ix = (QMixIndex *)arena_alloc(a, sizeof(QMixIndex));
    i32 hdr[6], cnt;
    if (ENG_R32(f, hdr, 6) || ENG_R32(f, &ix->l2_lambda, 1)) return (void *)0;
    ix->arena = a; ix->dim = hdr[0]; ix->code_bytes = hdr[1]; ix->count = hdr[2];
    ix->n_hi = hdr[3]; ix->n_mid = hdr[4]; ix->n_lo = hdr[5];
    cnt = ix->count > 0 ? ix->count : 1;
    ix->perm = (i32 *)arena_alloc(a, (size_t)(ix->dim > 0 ? ix->dim : 1) * sizeof(i32));
    ix->rn2  = (f32 *)arena_alloc(a, (size_t)cnt * sizeof(f32));
    ix->qn2  = (f32 *)arena_alloc(a, (size_t)cnt * sizeof(f32));
    if (ENG_R32(f, ix->perm, (size_t)ix->dim)
        || ENG_R32(f, ix->rn2, (size_t)ix->count)
        || ENG_R32(f, ix->qn2, (size_t)ix->count))
        return (void *)0;
    ix->hi  = (Q4Index *)eng_q_load_(a, f, ENG_Q4, ix->n_hi);
    ix->mid = (Q2Index *)eng_q_load_(a, f, ENG_Q2, ix->n_mid);
    ix->lo  = (Q1Index *)eng_q_load_(a, f, ENG_Q1, ix->n_lo);
    if (!ix->hi || !ix->mid || !ix->lo) return (void *)0;
    return (void *)ix;
}
static int eng_ivf_save_(FILE *f, const Ivf *v)
{
    i32 hdr[3];
    hdr[0] = v->dim; hdr[1] = v->n_clusters; hdr[2] = v->n;
    return ENG_W32(f, hdr, 3)
        || ENG_W32(f, v->centroids, (size_t)v->n_clusters * v->dim)
        || ENG_W64(f, v->list_ids,  (size_t)v->n)
        || ENG_W64(f, v->list_off,  (size_t)(v->n_clusters + 1))
        || ENG_W64(f, v->cluster_of,(size_t)v->n);
}
static Ivf *eng_ivf_load_(Arena *a, FILE *f)
{
    Ivf *v = (Ivf *)arena_alloc(a, sizeof(Ivf));
    i32 hdr[3];
    if (ENG_R32(f, hdr, 3)) return (Ivf *)0;
    v->arena = a; v->dim = hdr[0]; v->n_clusters = hdr[1]; v->n = hdr[2];
    v->centroids  = (f32 *)arena_alloc(a, (size_t)v->n_clusters * v->dim * sizeof(f32));
    v->list_ids   = (i64 *)arena_alloc(a, (size_t)(v->n > 0 ? v->n : 1) * sizeof(i64));
    v->list_off   = (i64 *)arena_alloc(a, (size_t)(v->n_clusters + 1) * sizeof(i64));
    v->cluster_of = (i64 *)arena_alloc(a, (size_t)(v->n > 0 ? v->n : 1) * sizeof(i64));
    if (ENG_R32(f, v->centroids, (size_t)v->n_clusters * v->dim)
        || ENG_R64(f, v->list_ids,  (size_t)v->n)
        || ENG_R64(f, v->list_off,  (size_t)(v->n_clusters + 1))
        || ENG_R64(f, v->cluster_of,(size_t)v->n))
        return (Ivf *)0;
    return v;
}

/* scheme-agnostic fine-index (de)serialise: qmix has its own shape;
 * q4fs stores its wrapped Q4Index and rebuilds the transpose on load;
 * the five uniform schemes share eng_q_save_/eng_q_load_. */
static int eng_qany_save_(FILE *f, EngScheme s, i32 dim, const void *ix)
{
    if (s == ENG_QMIX) return eng_qmix_save_(f, ix);
    if (s == ENG_Q4FS) return eng_q_save_(f, ENG_Q4, dim,
                                          ((const Q4FSIndex *)ix)->base);
    return eng_q_save_(f, s, dim, ix);
}
static void *eng_qany_load_(Arena *a, FILE *f, EngScheme s, i32 dim)
{
    if (s == ENG_QMIX) return eng_qmix_load_(a, f);
    if (s == ENG_Q4FS) {
        void *base = eng_q_load_(a, f, ENG_Q4, dim);
        return base ? (void *)q4fs_wrap(a, (const Q4Index *)base) : (void *)0;
    }
    return eng_q_load_(a, f, s, dim);
}

/* 0 on success; a self-contained index blob is appended to `f`. */
int engine_save(const EngIndex *e, FILE *f)
{
    i32 hdr[7];
    hdr[0] = (i32)ENG_SER_MAGIC; hdr[1] = (i32)e->topo; hdr[2] = (i32)e->scheme;
    hdr[3] = e->nprobe; hdr[4] = e->code_bytes; hdr[5] = e->dim;
    hdr[6] = (i32)e->casc_hi_s;
    if (ENG_W32(f, hdr, 7)
        || ENG_W32(f, &e->count, 1)
        || ENG_W32(f, &e->recall, 1))
        return -1;
    if (e->topo == ENG_FLAT)
        return eng_qany_save_(f, e->scheme, e->dim, e->flat_ix);
    if (e->topo == ENG_CASCADE)
        return eng_q_save_(f, e->scheme, e->dim, e->flat_ix)
            || eng_q_save_(f, e->casc_hi_s, e->dim, e->casc_hi);
    if (e->topo == ENG_HNSW) {
        /* the graph is rebuilt from the vectors + seed on load */
        return ENG_W32(f, &e->vhnsw_seed, 1)
            || ENG_W32(f, e->vhnsw_pts, (size_t)e->count * e->dim);
    }
    if (eng_ivf_save_(f, e->ivf)) return -1;
    {
        u8 has_router = (u8)(e->hrouter != (Hnsw *)0);
        if (fwrite(&has_router, 1, 1, f) != 1) return -1;
        if (ENG_W32(f, &e->router_seed, 1)) return -1;
    }
    {
        i32 c, nc = e->ivf->n_clusters;
        for (c = 0; c < nc; ++c) {
            u8 present = (u8)(e->cluster_ix[c] != (void *)0);
            if (fwrite(&present, 1, 1, f) != 1) return -1;
            if (present && eng_qany_save_(f, e->scheme, e->dim, e->cluster_ix[c]))
                return -1;
        }
    }
    return 0;
}

/* Reconstruct an EngIndex from `f` into arena `a`. NULL on any error. */
EngIndex *engine_load(Arena *a, FILE *f)
{
    EngIndex *e = (EngIndex *)arena_alloc(a, sizeof(EngIndex));
    i32 hdr[7];
    if (ENG_R32(f, hdr, 7) || (u32)hdr[0] != ENG_SER_MAGIC)
        return (EngIndex *)0;
    memset(e, 0, sizeof(*e));
    e->arena = a; e->topo = (EngTopo)hdr[1]; e->scheme = (EngScheme)hdr[2];
    e->nprobe = hdr[3]; e->code_bytes = hdr[4]; e->dim = hdr[5];
    e->casc_hi_s = (EngScheme)hdr[6];
    if (ENG_R32(f, &e->count, 1)
        || ENG_R32(f, &e->recall, 1))
        return (EngIndex *)0;
    if (e->topo == ENG_FLAT) {
        e->flat_ix = eng_qany_load_(a, f, e->scheme, e->dim);
        return e->flat_ix ? e : (EngIndex *)0;
    }
    if (e->topo == ENG_CASCADE) {
        e->flat_ix = eng_q_load_(a, f, e->scheme, e->dim);
        e->casc_hi = eng_q_load_(a, f, e->casc_hi_s, e->dim);
        return (e->flat_ix && e->casc_hi) ? e : (EngIndex *)0;
    }
    if (e->topo == ENG_HNSW) {
        size_t nv = (size_t)e->count * e->dim;
        HnswParams hp;
        e->vhnsw_pts = (f32 *)arena_alloc(a, nv * sizeof(f32));
        if (ENG_R32(f, &e->vhnsw_seed, 1) || ENG_R32(f, e->vhnsw_pts, nv))
            return (EngIndex *)0;
        hp.M = 24; hp.seed = e->vhnsw_seed;
        hp.ef_construction = (e->count <= 12000) ? 256 : 128;   /* match engine_build */
        e->vhnsw = hnsw_build(a, e->vhnsw_pts, e->count, e->dim, hp);
        return e->vhnsw ? e : (EngIndex *)0;
    }
    e->ivf = eng_ivf_load_(a, f);
    if (!e->ivf) return (EngIndex *)0;
    {
        u8 has_router;
        if (fread(&has_router, 1, 1, f) != 1) return (EngIndex *)0;
        if (ENG_R32(f, &e->router_seed, 1)) return (EngIndex *)0;
        if (has_router) {
            HnswParams hp;
            hp.M = 16; hp.ef_construction = 96; hp.seed = e->router_seed;
            e->hrouter = hnsw_build(a, e->ivf->centroids, e->ivf->n_clusters,
                                    e->ivf->dim, hp);
        }
    }
    {
        i32 c, nc = e->ivf->n_clusters;
        e->cluster_ix = (void **)arena_alloc(a, (size_t)nc * sizeof(void *));
        for (c = 0; c < nc; ++c) {
            u8 present;
            if (fread(&present, 1, 1, f) != 1) return (EngIndex *)0;
            if (present) {
                e->cluster_ix[c] = eng_qany_load_(a, f, e->scheme, e->dim);
                if (!e->cluster_ix[c]) return (EngIndex *)0;
            } else {
                e->cluster_ix[c] = (void *)0;
            }
        }
    }
    return e;
}
#endif /* ENGINE_SERIALIZE */

/* ================================================================== */
#ifdef ENGINE_DEMO
#define PRINT_IMPLEMENTATION
#include "print.h"

static u32 ed_rng_(u32 *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static f32 ed_f_(u32 *s) { return (f32)ed_rng_(s) / (f32)0xffffffffu - 0.5f; }

static f32 *ed_clustered_(Arena *a, i32 per, i32 topics, i32 dim, f32 spread, u32 seed)
{
    i32 n = per * topics, c, i, d;
    f32 *ct = (f32 *)arena_alloc(a, (size_t)topics * dim * sizeof(f32));
    f32 *o  = (f32 *)arena_alloc(a, (size_t)n * dim * sizeof(f32));
    u32 st = seed | 1u;
    for (i = 0; i < topics * dim; ++i) ct[i] = ed_f_(&st) * 8.0f;
    for (c = 0; c < topics; ++c)
        for (i = 0; i < per; ++i)
            for (d = 0; d < dim; ++d)
                o[((size_t)c * per + i) * dim + d] = ct[c * dim + d] + ed_f_(&st) * spread;
    return o;
}

static void ed_table_(const EngIndex *e)
{
    i32 s, pj;
    print_cstr("    scheme  B/vec  flat ");
    for (pj = 0; pj < ENG_NNPROBE; ++pj) {
        print_cstr(" ivf@"); print_i64(ENG_NPROBES[pj]);
    }
    print_cstr("\n");
    for (s = 0; s < ENG_NS; ++s) {
        print_cstr("      "); print_cstr(engine_scheme_name((EngScheme)s));
        print_cstr("    "); print_i64(e->cand_bytes[s]);
        print_cstr("    "); print_i64((i64)(100.0 * e->flat_recall[s])); print_cstr("%");
        for (pj = 0; pj < ENG_NNPROBE; ++pj) {
            print_cstr("  "); print_i64((i64)(100.0 * e->ivf_recall[s][pj])); print_cstr("%");
        }
        print_cstr("\n");
    }
}

static i32 ed_selfretr_(const EngIndex *e, const f32 *v, i32 n, i32 dim)
{
    i64 ids[8]; f32 sc[8]; i32 i, hit = 0;
    for (i = 0; i < n; ++i) {
        i32 h = engine_search(e, v + (size_t)i * dim, 1, ids, sc);
        if (h > 0 && ids[0] == (i64)i) ++hit;
    }
    return (i32)(100L * hit / n);
}

int main(void)
{
    static Arena arena;
    Arena *a = &arena;
    i32 dim = 96, topics = 15, per = 120, n = topics * per;
    f32 *v = ed_clustered_(a, per, topics, dim, 0.25f, 2024u);
    EngGoal g;
    EngIndex *e;
    i32 rc = 0;

    print_cstr("engine  dim="); print_i64(dim); print_cstr(" n="); print_i64(n);
    print_cstr("  (clustered corpus - the IVF+per-cluster path's real case)\n\n");

    memset(&g, 0, sizeof g);   /* all unset EngGoal fields default */
    g.mode = ENG_MIN_SIZE; g.k = 10; g.target_recall = 0.90f;
    g.max_code_bytes = 0; g.eval_queries = 60; g.n_iters = 12; g.seed = 0;
    e = engine_build(a, v, n, dim, g);
    ed_table_(e);
    print_cstr("  ENG_MIN_SIZE recall@10 >= 90%  =>  ");
    print_cstr(e->topo == ENG_IVF ? "IVF+" : "flat ");
    print_cstr(engine_scheme_name(e->scheme));
    if (e->topo == ENG_IVF) { print_cstr(" nprobe="); print_i64(e->nprobe); }
    print_cstr(", "); print_i64(e->code_bytes); print_cstr(" B/vec, recall ");
    print_i64((i64)(100.0 * e->recall)); print_cstr("%\n");
    print_cstr("  self-retrieval@1 = "); print_i64(ed_selfretr_(e, v, n, dim));
    print_cstr("%\n\n");
    if (e->recall < 0.90f) rc = 1;

    g.mode = ENG_MAX_ACC; g.k = 10; g.target_recall = 0.0f;
    g.max_code_bytes = 24; g.eval_queries = 60; g.n_iters = 12; g.seed = 0;
    e = engine_build(a, v, n, dim, g);
    print_cstr("  ENG_MAX_ACC budget <= 24 B/vec  =>  ");
    print_cstr(e->topo == ENG_IVF ? "IVF+" : "flat ");
    print_cstr(engine_scheme_name(e->scheme));
    if (e->topo == ENG_IVF) { print_cstr(" nprobe="); print_i64(e->nprobe); }
    print_cstr(", "); print_i64(e->code_bytes); print_cstr(" B/vec, recall ");
    print_i64((i64)(100.0 * e->recall)); print_cstr("%\n");
    if (e->code_bytes > 24) rc = 1;

    /* scan budget: cap the vectors a query may touch. The engine must
     * pick an nprobe whose n/nc * nprobe stays under it. */
    {
        i32 cap = n / 12;   /* ~150 vectors */
        g.mode = ENG_MIN_SIZE; g.k = 10; g.target_recall = 0.70f;
        g.max_code_bytes = 0; g.eval_queries = 60; g.n_iters = 12; g.seed = 0;
        g.max_candidates = cap;
        e = engine_build(a, v, n, dim, g);
        print_cstr("  scan budget <= "); print_i64(cap);
        print_cstr(" cand  =>  ");
        print_cstr(e->topo == ENG_IVF ? "IVF+" : "flat ");
        print_cstr(engine_scheme_name(e->scheme));
        if (e->topo == ENG_IVF) {
            print_cstr(" nprobe="); print_i64(e->nprobe);
            print_cstr(" (~"); print_i64(e->est_candidates); print_cstr(" cand)");
        }
        print_cstr(", recall "); print_i64((i64)(100.0 * e->recall)); print_cstr("%\n");
        if (e->topo == ENG_IVF && e->est_candidates > cap) rc = 1;
        g.max_candidates = 0;
    }

    /* ENG_HNSW: ask for a recall no quantised scheme reaches on this
     * clustered corpus - only the exact vector graph can. */
    {
        i32 qi, hit = 0, tot = 0;
        i64 hx[10]; f32 hs[10]; i64 gx[10];
        g.mode = ENG_MIN_SIZE; g.k = 10; g.target_recall = 0.97f;
        g.max_code_bytes = 0; g.eval_queries = 80; g.n_iters = 12; g.seed = 0;
        e = engine_build(a, v, n, dim, g);
        for (qi = 0; qi < n; qi += n / 80) {
            eng_exact_(v, n, dim, v + (size_t)qi * dim, 10, gx);
            engine_search(e, v + (size_t)qi * dim, 10, hx, hs);
            { i32 x, y; for (x = 0; x < 10; ++x) for (y = 0; y < 10; ++y)
                if (hx[x] == gx[y]) { ++hit; break; } }
            tot += 10;
        }
        print_cstr("  target 97%%  =>  ");
        print_cstr(e->topo == ENG_HNSW ? "HNSW-vectors " :
                   (e->topo == ENG_IVF ? "IVF+" : "flat "));
        print_cstr(e->topo == ENG_HNSW ? "" : engine_scheme_name(e->scheme));
        print_cstr(", "); print_i64(e->code_bytes);
        print_cstr(" B/vec, recall@10 "); print_i64((i64)(100L * hit / tot));
        print_cstr("%\n");
        if (e->topo != ENG_HNSW || 100L * hit / tot < 95) rc = 1;
    }

    /* residual cascade on a diffuse (uniform) corpus - the case a single
     * flat scheme wastes levels on. cascade(q1 lo + q3 hi) vs the flat
     * schemes at a comparable byte size. */
    {
        i32 un = 3000, ud = 96, i2;
        f32 *u = (f32 *)arena_alloc(a, (size_t)un * ud * sizeof(f32));
        u32 st = 777u;
        EngIndex *c;
        i64 ids[16]; f32 sc[16];
        i32 hitc = 0, hitq2 = 0;
        void *q2ix;
        for (i2 = 0; i2 < un * ud; ++i2) u[i2] = ed_f_(&st) * 2.0f;

        c = engine_build_cascade(a, u, un, ud, ENG_Q1, ENG_Q2);   /* 12+24 B */
        q2ix = eng_flat_build_(a, ENG_Q2, u, un, ud);             /* 24 B    */
        {   /* recall@10 vs exact cosine over 40 queries */
            i32 qi, ov_c = 0, ov_2 = 0, nq2 = 40;
            for (qi = 0; qi < un; qi += un / nq2) {
                const f32 *q = u + (size_t)qi * ud;
                i64 ex[10], ac[10], a2[10]; i32 r2, x, na;
                eng_exact_(u, un, ud, q, 10, ex);
                na = engine_search(c, q, 10, ac, sc);
                for (r2 = 0; r2 < na; ++r2)
                    for (x = 0; x < 10; ++x) if (ac[r2] == ex[x]) { ++ov_c; break; }
                na = q2_search((Q2Index *)q2ix, q, 10, a2, sc);
                for (r2 = 0; r2 < na; ++r2)
                    for (x = 0; x < 10; ++x) if (a2[r2] == ex[x]) { ++ov_2; break; }
            }
            (void)ids; (void)hitc; (void)hitq2;
            print_cstr("\n  diffuse corpus, recall@10 vs exact:\n");
            print_cstr("    cascade q1+q2 ("); print_i64(c->code_bytes);
            print_cstr(" B) = "); print_i64((i64)(100L * ov_c / (nq2 * 10)));
            print_cstr("%   flat q2 ("); print_i64(eng_bytes_(ENG_Q2, ud));
            print_cstr(" B) = "); print_i64((i64)(100L * ov_2 / (nq2 * 10)));
            print_cstr("%\n");
            if (ov_c < ov_2) rc = 1;   /* cascade must not lose to its own lo+ tier */
        }
    }

    /* heterogeneous clusters: per-cluster spread varies 8x, so each
     * cluster's quantiser error differs - the case where an nprobe>1
     * merge on the plain true-norm L2 estimate mis-ranks and the
     * term-3 blend retune earns its keep. */
    {
        i32 tp = 22, pc = 140, hn = tp * pc, hd = 96, c2, i3, d2;
        f32 *hv = (f32 *)arena_alloc(a, (size_t)hn * hd * sizeof(f32));
        f32 *hc = (f32 *)arena_alloc(a, (size_t)tp * hd * sizeof(f32));
        u32 st = 4242u;
        EngIndex *he;
        i64 ex[10], ac[10]; f32 sc2[10];
        i32 qi, hit_tuned = 0, hit_l1 = 0, nq3 = 60, cc2;
        for (i3 = 0; i3 < tp * hd; ++i3) hc[i3] = ed_f_(&st) * 8.0f;
        for (c2 = 0; c2 < tp; ++c2) {
            f32 sp = 0.05f + 1.15f * ((f32)c2 / (f32)(tp - 1));   /* 0.05 .. 1.20, 24x */
            for (i3 = 0; i3 < pc; ++i3)
                for (d2 = 0; d2 < hd; ++d2)
                    hv[((size_t)c2 * pc + i3) * hd + d2] =
                        hc[c2 * hd + d2] + ed_f_(&st) * sp;
        }
        /* tight byte budget -> forces the coarse ternary q3, where the
         * per-cluster quantiser-error spread is largest and the term-3
         * variance mismatch actually costs recall at nprobe>1. */
        g.mode = ENG_MAX_ACC; g.k = 10; g.target_recall = 0.0f;
        g.max_code_bytes = 20; g.eval_queries = 80; g.n_iters = 12; g.seed = 0;
        he = engine_build(a, hv, hn, hd, g);

        for (qi = 0; qi < hn; qi += hn / nq3) {
            const f32 *q = hv + (size_t)qi * hd;
            i32 r2, x, na;
            eng_exact_(hv, hn, hd, q, 10, ex);
            na = engine_search(he, q, 10, ac, sc2);
            for (r2 = 0; r2 < na; ++r2)
                for (x = 0; x < 10; ++x) if (ac[r2] == ex[x]) { ++hit_tuned; break; }
        }
        if (he->topo == ENG_IVF && he->cluster_ix)
            for (cc2 = 0; cc2 < he->ivf->n_clusters; ++cc2)
                eng_set_lambda_(he->scheme, he->cluster_ix[cc2], 1.0f);
        for (qi = 0; qi < hn; qi += hn / nq3) {
            const f32 *q = hv + (size_t)qi * hd;
            i32 r2, x, na;
            eng_exact_(hv, hn, hd, q, 10, ex);
            na = engine_search(he, q, 10, ac, sc2);
            for (r2 = 0; r2 < na; ++r2)
                for (x = 0; x < 10; ++x) if (ac[r2] == ex[x]) { ++hit_l1; break; }
        }
        print_cstr("\n  heterogeneous clusters (spread 0.05..1.20, 24x), ");
        print_cstr(he->topo == ENG_IVF ? "IVF+" : "flat ");
        print_cstr(engine_scheme_name(he->scheme));
        if (he->topo == ENG_IVF) { print_cstr(" np="); print_i64(he->nprobe); }
        print_cstr(":\n    recall@10  lambda=1 (default) = ");
        print_i64((i64)(100L * hit_l1 / (nq3 * 10)));
        print_cstr("%   engine chose lambda="); print_i64((i64)(100.0f * he->l2_lambda + 0.5f));
        print_cstr("e-2 -> "); print_i64((i64)(100L * hit_tuned / (nq3 * 10)));
        print_cstr("%\n");
        /* the retune keeps lambda=1 unless a blend clears it by 2pt on the
         * probe, so on held-out queries it must not lose by more than the
         * eval/test sampling slop (~3pt at nq3=60). */
        if (hit_tuned + 2 < hit_l1 * 9 / 10) rc = 1;
    }

#ifdef ENGINE_SERIALIZE
    /* qmix path through the .vxi (de)serialiser: a variance-skewed corpus
     * so engine_build picks the mixed-depth scheme, then a save/load
     * round-trip must reproduce every top-1. */
    {
        i32 sd = 128, sn = 3000, i4, d4;
        f32 *sv = (f32 *)arena_alloc(a, (size_t)sn * sd * sizeof(f32));
        u32 st = 6161u;
        EngIndex *se, *sl;
        FILE *tf;
        i64 a1[10], a2[10]; f32 c1[10], c2[10];
        i32 qi, same = 0, tot = 0;
        for (i4 = 0; i4 < sn; ++i4)
            for (d4 = 0; d4 < sd; ++d4) {
                f32 amp = (d4 < sd / 6) ? 3.0f : 0.12f;
                sv[(size_t)i4 * sd + d4] = ed_f_(&st) * amp;
            }
        g.mode = ENG_MAX_ACC; g.k = 10; g.target_recall = 0.0f;
        g.max_code_bytes = 40; g.eval_queries = 80; g.n_iters = 12; g.seed = 0;
        se = engine_build(a, sv, sn, sd, g);
        tf = tmpfile();
        if (!tf || engine_save(se, tf)) { print_cstr("  vxi qmix save FAIL\n"); rc = 1; }
        else {
            rewind(tf);
            sl = engine_load(a, tf);
            if (!sl) { print_cstr("  vxi qmix load FAIL\n"); rc = 1; }
            else {
                for (qi = 0; qi < sn; qi += sn / 200) {
                    engine_search(se, sv + (size_t)qi * sd, 10, a1, c1);
                    engine_search(sl, sv + (size_t)qi * sd, 10, a2, c2);
                    if (a1[0] == a2[0]) ++same;
                    ++tot;
                }
                print_cstr("\n  .vxi round-trip, scheme=");
                print_cstr(engine_scheme_name(se->scheme));
                print_cstr(": top-1 identical "); print_i64(same);
                print_cstr("/"); print_i64(tot); print_cstr("\n");
                if (same != tot || se->scheme != ENG_QMIX) rc = 1;
            }
        }
        if (tf) fclose(tf);
    }

    /* q4fs path: a uniform corpus + a byte budget that forces q4, large
     * enough (>= ENG_Q4FS_MIN) that engine_build swaps in the fast-scan
     * wrapper. Recall must equal a plain-q4 build and survive .vxi. */
    {
        i32 fd = 192, fn = 4000, i5;
        f32 *fv = (f32 *)arena_alloc(a, (size_t)fn * fd * sizeof(f32));
        u32 st = 909u;
        EngIndex *fe, *fl;
        FILE *tf;
        i64 b1[10], b2[10]; f32 d1[10], d2[10];
        i32 qi, same = 0, tot = 0;
        for (i5 = 0; i5 < fn * fd; ++i5) fv[i5] = ed_f_(&st) * 2.0f;
        g.mode = ENG_MAX_ACC; g.k = 10; g.target_recall = 0.0f;
        g.max_code_bytes = fd / 2 + 2;   /* fits q4, not q5 */
        g.eval_queries = 80; g.n_iters = 12; g.seed = 0;
        fe = engine_build(a, fv, fn, fd, g);
        tf = tmpfile();
        if (!tf || engine_save(fe, tf)) { print_cstr("  vxi q4fs save FAIL\n"); rc = 1; }
        else {
            rewind(tf);
            fl = engine_load(a, tf);
            if (!fl) { print_cstr("  vxi q4fs load FAIL\n"); rc = 1; }
            else {
                for (qi = 0; qi < fn; qi += fn / 200) {
                    engine_search(fe, fv + (size_t)qi * fd, 10, b1, d1);
                    engine_search(fl, fv + (size_t)qi * fd, 10, b2, d2);
                    if (b1[0] == b2[0]) ++same;
                    ++tot;
                }
                print_cstr("  .vxi round-trip, scheme=");
                print_cstr(engine_scheme_name(fe->scheme));
                print_cstr(": top-1 identical "); print_i64(same);
                print_cstr("/"); print_i64(tot); print_cstr("\n");
                if (same != tot || fe->scheme != ENG_Q4FS) rc = 1;
            }
        }
        if (tf) fclose(tf);
    }

    /* IVF + HNSW router: a clustered corpus big enough that engine_build
     * lands on IVF with nc >= ENG_HNSW_MIN_NC, so a graph router is
     * built. The .vxi carries only the flag+seed; engine_load rebuilds
     * the graph from the loaded centroids and must route identically.
     * Also checks the build is deterministic (same seed -> same graph). */
    {
        i32 rd_ = 64, rn_ = 5000, tp = 40, i6, d6, c6;
        f32 *rv = (f32 *)arena_alloc(a, (size_t)rn_ * rd_ * sizeof(f32));
        f32 *rc_ = (f32 *)arena_alloc(a, (size_t)tp * rd_ * sizeof(f32));
        u32 st = 5150u;
        EngIndex *re, *re2, *rl;
        FILE *tf;
        i64 x1[10], x2[10]; f32 y1[10], y2[10];
        i32 qi, same_ld = 0, same_det = 0, tot = 0;
        for (i6 = 0; i6 < tp * rd_; ++i6) rc_[i6] = ed_f_(&st) * 7.0f;
        for (i6 = 0; i6 < rn_; ++i6) { c6 = (i32)(ed_rng_(&st) % (u32)tp);
            for (d6 = 0; d6 < rd_; ++d6)
                rv[(size_t)i6*rd_+d6] = rc_[c6*rd_+d6] + ed_f_(&st) * 0.9f; }
        g.mode = ENG_MIN_SIZE; g.k = 10; g.target_recall = 0.80f;
        g.max_code_bytes = 0; g.eval_queries = 80; g.n_iters = 12; g.seed = 0x5eedu;
        re  = engine_build(a, rv, rn_, rd_, g);
        re2 = engine_build(a, rv, rn_, rd_, g);
        tf = tmpfile();
        if (!tf || engine_save(re, tf)) { print_cstr("  vxi router save FAIL\n"); rc = 1; }
        else {
            rewind(tf);
            rl = engine_load(a, tf);
            if (!rl) { print_cstr("  vxi router load FAIL\n"); rc = 1; }
            else {
                for (qi = 0; qi < rn_; qi += rn_ / 200) {
                    const f32 *q = rv + (size_t)qi * rd_;
                    engine_search(re,  q, 10, x1, y1);
                    engine_search(rl,  q, 10, x2, y2);
                    if (x1[0] == x2[0]) ++same_ld;
                    engine_search(re2, q, 10, x2, y2);
                    if (x1[0] == x2[0]) ++same_det;
                    ++tot;
                }
                print_cstr("  .vxi round-trip, ");
                print_cstr(re->topo == ENG_IVF ? "IVF+" : "flat ");
                print_cstr(engine_scheme_name(re->scheme));
                print_cstr(re->hrouter ? " +HNSW" : " (brute route)");
                print_cstr(" nc="); print_i64(re->topo == ENG_IVF ? re->ivf->n_clusters : 0);
                print_cstr(": top-1 load="); print_i64(same_ld);
                print_cstr("/"); print_i64(tot);
                print_cstr("  rebuild="); print_i64(same_det);
                print_cstr("/"); print_i64(tot); print_cstr("\n");
                if (same_ld != tot || same_det != tot
                    || re->topo != ENG_IVF || !re->hrouter) rc = 1;
            }
        }
        if (tf) fclose(tf);
    }

    /* ENG_HNSW topology: force it with a recall target no quantiser
     * reaches, then .vxi save writes the (normalised) vectors + seed and
     * load rebuilds the graph - must route identically. */
    {
        i32 hd = 72, hn = 2500, tp = 30, i7, d7, c7;
        f32 *hv = (f32 *)arena_alloc(a, (size_t)hn * hd * sizeof(f32));
        f32 *hc = (f32 *)arena_alloc(a, (size_t)tp * hd * sizeof(f32));
        u32 st = 8317u;
        EngIndex *he, *hl;
        FILE *tf;
        i64 z1[10], z2[10]; f32 w1[10], w2[10];
        i32 qi, same = 0, tot = 0;
        for (i7 = 0; i7 < tp * hd; ++i7) hc[i7] = ed_f_(&st) * 6.0f;
        for (i7 = 0; i7 < hn; ++i7) { c7 = (i32)(ed_rng_(&st) % (u32)tp);
            for (d7 = 0; d7 < hd; ++d7)
                hv[(size_t)i7*hd+d7] = hc[c7*hd+d7] + ed_f_(&st) * 0.8f; }
        g.mode = ENG_MIN_SIZE; g.k = 10; g.target_recall = 0.97f;
        g.max_code_bytes = 0; g.eval_queries = 80; g.n_iters = 12; g.seed = 0x5eedu;
        he = engine_build(a, hv, hn, hd, g);
        tf = tmpfile();
        if (!tf || engine_save(he, tf)) { print_cstr("  vxi hnsw save FAIL\n"); rc = 1; }
        else {
            rewind(tf);
            hl = engine_load(a, tf);
            if (!hl) { print_cstr("  vxi hnsw load FAIL\n"); rc = 1; }
            else {
                for (qi = 0; qi < hn; qi += hn / 200) {
                    const f32 *q = hv + (size_t)qi * hd;
                    engine_search(he, q, 10, z1, w1);
                    engine_search(hl, q, 10, z2, w2);
                    if (z1[0] == z2[0]) ++same;
                    ++tot;
                }
                print_cstr("  .vxi round-trip, topo=");
                print_cstr(he->topo == ENG_HNSW ? "HNSW-vectors" :
                           (he->topo == ENG_IVF ? "IVF" : "flat"));
                print_cstr(": top-1 identical "); print_i64(same);
                print_cstr("/"); print_i64(tot); print_cstr("\n");
                if (same != tot || he->topo != ENG_HNSW) rc = 1;
            }
        }
        if (tf) fclose(tf);
    }
#endif

    print_flush();
    arena_free(a);
    return rc;
}
#endif /* ENGINE_DEMO */

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
