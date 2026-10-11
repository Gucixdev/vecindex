/* ivf.c - IVF coarse quantizer / inverted file for vecindex, one file.
 *
 * The fine quantizers (q1..q5) shrink the cost PER vector; they do not
 * change the O(n) search - every candidate is still touched. This layer
 * makes the search sub-linear: cluster the dataset once with Lloyd's
 * k-means over full-precision f32 (no compression here - it only ever
 * compares to a handful of centroids), then at query time scan only the
 * `nprobe` nearest clusters' members through a fine quantizer.
 *
 * Port of the Rust vecindex (src/ivf.rs). Arena-owned, no free.
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -I. ivf.c -DIVF_DEMO -lm -o ivf && ./ivf
 */

#ifdef IVF_DEMO
#define ARENA_ALLOC_IMPLEMENTATION
#endif
#include "base.h"
#include "arena_alloc.h"
#include "vsimd.h"

typedef struct {
    Arena *arena;
    i32 dim;
    i32 n_clusters;
    i32 n;
    f32 *centroids;    /* [n_clusters * dim]                        */
    i64 *list_ids;     /* [n]   member row ids, grouped by cluster  */
    i64 *list_off;     /* [n_clusters + 1]  CSR row pointers        */
    i64 *cluster_of;   /* [n]   row id -> its cluster               */
} Ivf;

/* ------------------------------------------------------------------ */

static u32 ivf_xs32_(u32 *s)
{
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s;
}

static f32 ivf_sqdist_(const f32 *a, const f32 *b, i32 dim)
{
    return vs_sqdist(a, b, dim);   /* AVX2+FMA when built for it */
}

static i32 ivf_nearest_(const f32 *v, const f32 *centroids, i32 nc, i32 dim)
{
    i32 best = 0, c;
    f32 bd = ivf_sqdist_(v, centroids, dim);
    for (c = 1; c < nc; ++c) {
        f32 d = ivf_sqdist_(v, centroids + (size_t)c * dim, dim);
        if (d < bd) { bd = d; best = c; }
    }
    return best;
}

/* counting-pass + placement-pass CSR build over a settled assignment. */
static void ivf_build_csr_(Arena *a, i32 nc, const i64 *cluster_of, i32 n,
                           i64 **out_ids, i64 **out_off)
{
    i64 *off = (i64 *)arena_alloc(a, (size_t)(nc + 1) * sizeof(i64));
    i64 *ids = (i64 *)arena_alloc(a, (size_t)(n > 0 ? n : 1) * sizeof(i64));
    i64 *cursor = (i64 *)arena_alloc(a, (size_t)(nc + 1) * sizeof(i64));
    i32 c, i;

    for (c = 0; c <= nc; ++c) off[c] = 0;
    for (i = 0; i < n; ++i) off[(i32)cluster_of[i] + 1] += 1;
    for (c = 0; c < nc; ++c) off[c + 1] += off[c];
    for (c = 0; c <= nc; ++c) cursor[c] = off[c];
    for (i = 0; i < n; ++i) {
        i32 cl = (i32)cluster_of[i];
        ids[cursor[cl]++] = i;
    }
    *out_ids = ids;
    *out_off = off;
}

/* ------------------------------------------------------------------ */

i32 ivf_suggest_clusters(i32 n)
{
    i32 c = 1;
    while ((i64)(c + 1) * (c + 1) <= (i64)n) ++c;   /* floor(sqrt(n)) */
    /* round to nearest */
    if ((i64)2 * c + 1 <= (i64)n - (i64)c * c) ++c;   /* c is bounded ~sqrt(n) so this never actually overflows either way - cast moved to match the pattern above/right of it, not because it was reachable */
    return c < 1 ? 1 : c;
}

/* Real Lloyd's k-means: centroids seeded from random (possibly
 * repeated) rows, then `n_iters` rounds of assign-then-recompute; an
 * empty cluster keeps its previous centroid rather than going NaN. */
Ivf *ivf_train(Arena *a, const f32 *vecs, i32 n, i32 dim,
               i32 n_clusters, i32 n_iters, u32 seed)
{
    Ivf *ix = (Ivf *)arena_alloc(a, sizeof(Ivf));
    u32 st = seed ? seed : 1u;
    i64 *cluster_of = (i64 *)arena_alloc(a, (size_t)(n > 0 ? n : 1) * sizeof(i64));
    f32 *sums;
    i32 *counts;
    i32 c, i, d, it;

    if (n_clusters < 1) n_clusters = 1;
    if (n_clusters > n && n > 0) n_clusters = n;
    if (n_iters < 1) n_iters = 1;

    ix->arena = a; ix->dim = dim; ix->n_clusters = n_clusters; ix->n = n;
    ix->centroids = (f32 *)arena_alloc(a, (size_t)n_clusters * dim * sizeof(f32));
    sums   = (f32 *)arena_alloc(a, (size_t)n_clusters * dim * sizeof(f32));
    counts = (i32 *)arena_alloc(a, (size_t)n_clusters * sizeof(i32));

    for (c = 0; c < n_clusters; ++c) {
        i32 src = (n > 0) ? (i32)(ivf_xs32_(&st) % (u32)n) : 0;
        for (d = 0; d < dim; ++d)
            ix->centroids[(size_t)c * dim + d] = vecs[(size_t)src * dim + d];
    }

    for (it = 0; it < n_iters; ++it) {
        for (i = 0; i < n; ++i)
            cluster_of[i] = ivf_nearest_(vecs + (size_t)i * dim,
                                         ix->centroids, n_clusters, dim);
        for (c = 0; c < n_clusters * dim; ++c) sums[c] = 0.0f;
        for (c = 0; c < n_clusters; ++c) counts[c] = 0;
        for (i = 0; i < n; ++i) {
            i32 cl = (i32)cluster_of[i];
            const f32 *v = vecs + (size_t)i * dim;
            counts[cl] += 1;
            for (d = 0; d < dim; ++d) sums[(size_t)cl * dim + d] += v[d];
        }
        for (c = 0; c < n_clusters; ++c) {
            f32 inv;
            if (counts[c] == 0) continue;
            inv = 1.0f / (f32)counts[c];
            for (d = 0; d < dim; ++d)
                ix->centroids[(size_t)c * dim + d] = sums[(size_t)c * dim + d] * inv;
        }
    }

    for (i = 0; i < n; ++i)
        cluster_of[i] = ivf_nearest_(vecs + (size_t)i * dim,
                                     ix->centroids, n_clusters, dim);

    ivf_build_csr_(a, n_clusters, cluster_of, n, &ix->list_ids, &ix->list_off);
    ix->cluster_of = cluster_of;
    return ix;
}

/* Re-assign a (typically larger) full dataset to the already-trained
 * centroids and rebuild the CSR - the "train on a subsample, assign
 * everything" trick. */
void ivf_assign_full(Ivf *ix, const f32 *full_vecs, i32 full_n)
{
    i64 *cluster_of = (i64 *)arena_alloc(ix->arena,
                        (size_t)(full_n > 0 ? full_n : 1) * sizeof(i64));
    i32 i;
    for (i = 0; i < full_n; ++i)
        cluster_of[i] = ivf_nearest_(full_vecs + (size_t)i * ix->dim,
                                     ix->centroids, ix->n_clusters, ix->dim);
    ivf_build_csr_(ix->arena, ix->n_clusters, cluster_of, full_n,
                   &ix->list_ids, &ix->list_off);
    ix->cluster_of = cluster_of;
    ix->n = full_n;
}

/* The `nprobe` nearest clusters to `query`, nearest first. Writes into
 * out_clusters[<= nprobe] and returns the count. */
i32 ivf_route(const Ivf *ix, const f32 *query, i32 nprobe, i32 *out_clusters)
{
    f32 bd[64];
    i32 have = 0, c, j;
    if (nprobe > ix->n_clusters) nprobe = ix->n_clusters;
    if (nprobe > 64) nprobe = 64;
    if (nprobe < 1) nprobe = 1;

    for (c = 0; c < ix->n_clusters; ++c) {
        f32 d = ivf_sqdist_(query, ix->centroids + (size_t)c * ix->dim, ix->dim);
        if (have < nprobe) {
            j = have; bd[j] = d; out_clusters[j] = c;
            while (j > 0 && bd[j - 1] > bd[j]) {
                f32 td = bd[j-1]; i32 tc = out_clusters[j-1];
                bd[j-1] = bd[j]; out_clusters[j-1] = out_clusters[j];
                bd[j] = td; out_clusters[j] = tc; --j;
            }
            ++have;
        } else if (d < bd[nprobe - 1]) {
            j = nprobe - 1; bd[j] = d; out_clusters[j] = c;
            while (j > 0 && bd[j - 1] > bd[j]) {
                f32 td = bd[j-1]; i32 tc = out_clusters[j-1];
                bd[j-1] = bd[j]; out_clusters[j-1] = out_clusters[j];
                bd[j] = td; out_clusters[j] = tc; --j;
            }
        }
    }
    return have;
}

const i64 *ivf_members(const Ivf *ix, i32 cluster, i32 *out_count)
{
    i64 start = ix->list_off[cluster];
    i64 end   = ix->list_off[cluster + 1];
    *out_count = (i32)(end - start);
    return ix->list_ids + start;
}

/* ================================================================== */
#ifdef IVF_DEMO
#define PRINT_IMPLEMENTATION
#include "print.h"

static u32 ivfd_rng_(u32 *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static f32 ivfd_f_(u32 *s) { return (f32)ivfd_rng_(s) / (f32)0xffffffffu - 0.5f; }

static f32 *ivfd_clustered_(Arena *a, i32 per, i32 topics, i32 dim, f32 spread, u32 seed)
{
    i32 n = per * topics, c, i, d;
    f32 *centers = (f32 *)arena_alloc(a, (size_t)topics * dim * sizeof(f32));
    f32 *out = (f32 *)arena_alloc(a, (size_t)n * dim * sizeof(f32));
    u32 st = seed | 1u;
    for (i = 0; i < topics * dim; ++i) centers[i] = ivfd_f_(&st) * 10.0f;
    for (c = 0; c < topics; ++c)
        for (i = 0; i < per; ++i)
            for (d = 0; d < dim; ++d)
                out[((size_t)c * per + i) * dim + d] = centers[c * dim + d] + ivfd_f_(&st) * spread;
    return out;
}

int main(void)
{
    static Arena arena;
    Arena *a = &arena;
    i32 dim = 32, topics = 8, per = 50, n = topics * per;
    f32 *v = ivfd_clustered_(a, per, topics, dim, 0.3f, 7u);
    i32 nc = ivf_suggest_clusters(n);
    Ivf *ix = ivf_train(a, v, n, dim, nc, 12, 123u);
    i32 probe[8], i, found = 0, checked = 0;

    print_cstr("ivf  n="); print_i64(n); print_cstr(" dim="); print_i64(dim);
    print_cstr("  n_clusters="); print_i64(nc);
    print_cstr("  (suggest = round(sqrt(n)))\n");

    /* last CSR offset must be n */
    if (ix->list_off[nc] != (i64)n) { print_cstr("  BAD: csr total != n\n"); print_flush(); return 1; }

    /* every row's own vector must route into a probed cluster whose
     * member list contains it, at nprobe=1 */
    for (i = 0; i < n; i += 3) {
        i32 np = ivf_route(ix, v + (size_t)i * dim, 1, probe), mc, m, hit = 0;
        const i64 *mem = ivf_members(ix, probe[0], &mc);
        (void)np;
        for (m = 0; m < mc; ++m) if (mem[m] == (i64)i) { hit = 1; break; }
        found += hit; ++checked;
    }
    print_cstr("  self-vector routes into own cluster: ");
    print_i64((i64)(100 * found / checked)); print_cstr("%\n");
    print_flush();
    arena_free(a);
    return (found == checked) ? 0 : 1;
}
#endif /* IVF_DEMO */
