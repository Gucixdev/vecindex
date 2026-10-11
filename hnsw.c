/* hnsw.c - Hierarchical Navigable Small World graph over f32 vectors,
 * squared-L2. vecindex builds it on the IVF coarse centroids and uses it
 * as the ROUTER: greedy descent through the graph picks the nprobe cells
 * to scan, instead of a raw nearest-centroid ranking. A query sitting on
 * a Voronoi boundary then still reaches the cell that holds its true
 * neighbour - the fix for the nprobe>1 recall plateau, which is
 * routing-bound, not a scoring problem.
 *
 * Built on ~sqrt(n) nodes (or a small multiple), so it is tiny; seeded
 * RNG so a rebuild from the same centroids gives the same graph (engine
 * reconstructs it on .vxi load rather than serialising the adjacency).
 * One caller-supplied Arena, no free. No libc.
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -march=native -I. hnsw.c \
 *       -DHNSW_DEMO -lm -o hnsw && ./hnsw
 */

#ifdef HNSW_DEMO
#define ARENA_ALLOC_IMPLEMENTATION
#endif
#include "base.h"
#include "arena_alloc.h"
#include "vsimd.h"

#define HNSW_MAXL       10          /* level cap; P(level>=L) = (1/M)^L   */
#define HNSW_MAXN       262144      /* node-count ceiling (stack bitset)  */
#define HNSW_VISIT_CAP  4096        /* per-search expansion budget        */
#define HNSW_EF_CAP     256         /* per-search beam ceiling            */

typedef struct { i32 M; i32 ef_construction; u32 seed; } HnswParams;

typedef struct {
    Arena     *arena;
    const f32 *pts;                /* borrowed: [n * dim] centroid array   */
    i32        n, dim, M, Mmax0, efc;
    i32        entry, maxl;
    i16       *level;             /* [n] */
    i32      **adj;               /* [n] -> flat per-level neighbour ids   */
    i16      **cnt;               /* [n] -> [level+1] live counts          */
} Hnsw;

/* ---- small parallel-array heaps (id[], d[]) ---------------------- */

static void hn_swap_(i32 *id, f32 *d, i32 i, i32 j)
{ i32 ti = id[i]; f32 td = d[i]; id[i] = id[j]; d[i] = d[j]; id[j] = ti; d[j] = td; }

static void hn_up_min_(i32 *id, f32 *d, i32 i)
{
    while (i > 0) {
        i32 p = (i - 1) >> 1;
        if (d[p] <= d[i]) break;
        hn_swap_(id, d, p, i); i = p;
    }
}
static void hn_down_min_(i32 *id, f32 *d, i32 n, i32 i)
{
    for (;;) {
        i32 l = 2*i+1, r = l+1, m = i;
        if (l < n && d[l] < d[m]) m = l;
        if (r < n && d[r] < d[m]) m = r;
        if (m == i) break;
        hn_swap_(id, d, m, i); i = m;
    }
}

static void hn_up_max_(i32 *id, f32 *d, i32 i)
{
    while (i > 0) {
        i32 p = (i - 1) >> 1;
        if (d[p] >= d[i]) break;
        hn_swap_(id, d, p, i); i = p;
    }
}
static void hn_down_max_(i32 *id, f32 *d, i32 n, i32 i)
{
    for (;;) {
        i32 l = 2*i+1, r = l+1, m = i;
        if (l < n && d[l] > d[m]) m = l;
        if (r < n && d[r] > d[m]) m = r;
        if (m == i) break;
        hn_swap_(id, d, m, i); i = m;
    }
}

/* ---- adjacency layout ------------------------------------------- *
 * per node u: one i32 block of Mmax0 + level[u]*M ids, plus a small
 * i16 count per level. layer lc lives at offset
 *   lc == 0 ? 0 : Mmax0 + (lc - 1) * M      with cap
 *   lc == 0 ? Mmax0 : M                                            */

static i32 hn_off_(const Hnsw *h, i32 lc) { return lc == 0 ? 0 : h->Mmax0 + (lc - 1) * h->M; }
static i32 hn_cap_(const Hnsw *h, i32 lc) { return lc == 0 ? h->Mmax0 : h->M; }

static i32 *hn_neigh_(const Hnsw *h, i32 u, i32 lc, i32 *out_cnt)
{
    *out_cnt = h->cnt[u][lc];
    return h->adj[u] + hn_off_(h, lc);
}

/* ---- one layer's ef-beam search ------------------------------- *
 * entrypoints eps[neps]; fills the max-heap (rid,rd) with up to ef
 * nearest, returns its size. vbits is the caller's zeroed visited
 * bitset (n bits). cid/cd (HNSW_VISIT_CAP) and rid/rd (ef<=HNSW_EF_CAP)
 * are caller scratch. */
static i32 hn_search_layer_(const Hnsw *h, const f32 *q, const i32 *eps, i32 neps,
                            i32 ef, i32 lc, u8 *vbits,
                            i32 *cid, f32 *cd, i32 *rid, f32 *rd)
{
    i32 cn = 0, rn = 0, visits = 0, i;
    for (i = 0; i < neps; ++i) {
        i32 e = eps[i]; f32 de;
        if (vbits[e >> 3] & (u8)(1u << (e & 7))) continue;
        vbits[e >> 3] |= (u8)(1u << (e & 7));
        de = vs_sqdist(q, h->pts + (size_t)e * h->dim, h->dim);
        cid[cn] = e; cd[cn] = de; hn_up_min_(cid, cd, cn); ++cn;
        rid[rn] = e; rd[rn] = de; hn_up_max_(rid, rd, rn); ++rn;
    }
    while (cn > 0) {
        i32 c = cid[0]; f32 dc = cd[0];
        i32 *nb, nbc, j;
        if (rn >= ef && dc > rd[0]) break;
        --cn; cid[0] = cid[cn]; cd[0] = cd[cn]; hn_down_min_(cid, cd, cn, 0);
        if (++visits > HNSW_VISIT_CAP) break;
        nb = hn_neigh_(h, c, lc, &nbc);
        for (j = 0; j < nbc; ++j) {
            i32 x = nb[j]; f32 dx;
            if (vbits[x >> 3] & (u8)(1u << (x & 7))) continue;
            vbits[x >> 3] |= (u8)(1u << (x & 7));
            dx = vs_sqdist(q, h->pts + (size_t)x * h->dim, h->dim);
            if (rn < ef || dx < rd[0]) {
                if (cn < HNSW_VISIT_CAP) { cid[cn] = x; cd[cn] = dx; hn_up_min_(cid, cd, cn); ++cn; }
                if (rn < ef) { rid[rn] = x; rd[rn] = dx; hn_up_max_(rid, rd, rn); ++rn; }
                else if (dx < rd[0]) { rid[0] = x; rd[0] = dx; hn_down_max_(rid, rd, rn, 0); }
            }
        }
    }
    return rn;
}

/* ---- neighbour-selection heuristic (HNSW alg. 4) --------------- *
 * cand sorted ascending by distance to the new node q; keep c only if
 * it is closer to q than to every already-kept neighbour (spreads links
 * along the manifold, not in a ball). Fills sel[] (<= M), returns count;
 * back-fills from the pruned set to reach M for connectivity. */
static i32 hn_select_(const Hnsw *h, const i32 *cand, const f32 *cand_d, i32 nc,
                      i32 M, i32 *sel)
{
    i32 ns = 0, i, j;
    for (i = 0; i < nc && ns < M; ++i) {
        i32 c = cand[i]; i32 keep = 1;
        for (j = 0; j < ns; ++j) {
            f32 dcs = vs_sqdist(h->pts + (size_t)c * h->dim,
                                h->pts + (size_t)sel[j] * h->dim, h->dim);
            if (dcs < cand_d[i]) { keep = 0; break; }
        }
        if (keep) sel[ns++] = c;
    }
    for (i = 0; i < nc && ns < M; ++i) {
        i32 c = cand[i], dup = 0;
        for (j = 0; j < ns; ++j) if (sel[j] == c) { dup = 1; break; }
        if (!dup) sel[ns++] = c;
    }
    return ns;
}

/* sorted (ascending) copy of a max-heap's contents into id/d. */
static void hn_heap_sorted_(const i32 *hid, const f32 *hd, i32 n, i32 *id, f32 *d)
{
    i32 i, j;
    for (i = 0; i < n; ++i) { id[i] = hid[i]; d[i] = hd[i]; }
    for (i = 1; i < n; ++i) {           /* insertion sort, n <= efc */
        i32 ti = id[i]; f32 td = d[i]; j = i - 1;
        while (j >= 0 && d[j] > td) { id[j+1] = id[j]; d[j+1] = d[j]; --j; }
        id[j+1] = ti; d[j+1] = td;
    }
}

/* ---- build ---------------------------------------------------- */

static u32 hn_xs_(u32 *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }

static i32 hn_gen_level_(u32 *rng, i32 M)
{
    i32 l = 0;
    /* geometric with ratio 1/M, capped: same shape as -ln(U)/ln(M). */
    while (l < HNSW_MAXL - 1) {
        u32 r = hn_xs_(rng);
        if ((r % (u32)M) != 0u) break;
        ++l;
    }
    return l;
}

static void hn_connect_(Hnsw *h, i32 u, i32 lc, const i32 *sel, i32 ns)
{
    i32 *au = h->adj[u] + hn_off_(h, lc);
    i32 i, cap = hn_cap_(h, lc), c = ns > cap ? cap : ns;
    for (i = 0; i < c; ++i) au[i] = sel[i];
    h->cnt[u][lc] = (i16)c;
}

Hnsw *hnsw_build(Arena *a, const f32 *pts, i32 n, i32 dim, HnswParams p)
{
    Hnsw *h = (Hnsw *)arena_alloc(a, sizeof(Hnsw));
    i32 M   = p.M > 1 ? p.M : 16;
    i32 efc = p.ef_construction > 8 ? p.ef_construction : 128;
    u32 rng = p.seed ? p.seed : 0x9e3779b9u;
    i32 i, lc;
    /* per-insert scratch (reused; freed at the end) */
    Arena_Mark scratch;
    u8  *vbits;
    i32 *cid, *rid, *sel, *wq_id, *nb2;
    f32 *cd, *rd, *wq_d;

    if (efc > HNSW_EF_CAP) efc = HNSW_EF_CAP;
    if (n > HNSW_MAXN) n = HNSW_MAXN;      /* router centroid counts are tiny */

    h->arena = a; h->pts = pts; h->n = n; h->dim = dim;
    h->M = M; h->Mmax0 = 2 * M; h->efc = efc;
    h->entry = 0; h->maxl = 0;
    h->level = (i16 *)arena_alloc(a, (size_t)(n > 0 ? n : 1) * sizeof(i16));
    h->adj   = (i32 **)arena_alloc(a, (size_t)(n > 0 ? n : 1) * sizeof(i32 *));
    h->cnt   = (i16 **)arena_alloc(a, (size_t)(n > 0 ? n : 1) * sizeof(i16 *));
    if (n <= 0) { h->n = 0; return h; }

    /* pass 1: draw every node's level and allocate its (permanent)
     * adjacency block up front, so the reusable insert scratch can sit
     * strictly on top and be rewound without touching the graph. */
    for (i = 0; i < n; ++i) {
        i32 li = hn_gen_level_(&rng, M);
        h->level[i] = (i16)li;
        h->adj[i] = (i32 *)arena_alloc(a, (size_t)(h->Mmax0 + li * M) * sizeof(i32));
        h->cnt[i] = (i16 *)arena_alloc(a, (size_t)(li + 1) * sizeof(i16));
        for (lc = 0; lc <= li; ++lc) h->cnt[i][lc] = 0;
    }

    scratch = arena_snapshot(a);
    vbits = (u8  *)arena_alloc(a, (size_t)((n + 7) >> 3) + 1);
    cid   = (i32 *)arena_alloc(a, HNSW_VISIT_CAP * sizeof(i32));
    cd    = (f32 *)arena_alloc(a, HNSW_VISIT_CAP * sizeof(f32));
    rid   = (i32 *)arena_alloc(a, (size_t)(efc + 1) * sizeof(i32));
    rd    = (f32 *)arena_alloc(a, (size_t)(efc + 1) * sizeof(f32));
    wq_id = (i32 *)arena_alloc(a, (size_t)(efc + 1) * sizeof(i32));
    wq_d  = (f32 *)arena_alloc(a, (size_t)(efc + 1) * sizeof(f32));
    sel   = (i32 *)arena_alloc(a, (size_t)(2 * M + 1) * sizeof(i32));
    nb2   = (i32 *)arena_alloc(a, (size_t)(2 * M + 1) * sizeof(i32));

    h->maxl = h->level[0]; h->entry = 0;

    /* pass 2: link each node into the graph in id order */
    for (i = 1; i < n; ++i) {
        i32 li = h->level[i];
        i32 ep = h->entry, top = h->maxl, l, rn, ns, j;
        f32 epd;
        const f32 *qi = pts + (size_t)i * dim;

        /* descend layers above li with a width-1 beam */
        epd = vs_sqdist(qi, pts + (size_t)ep * dim, dim);
        for (l = top; l > li; --l) {
            i32 changed = 1;
            while (changed) {
                i32 *nb, nbc;
                changed = 0;
                nb = hn_neigh_(h, ep, l, &nbc);
                for (j = 0; j < nbc; ++j) {
                    f32 dd = vs_sqdist(qi, pts + (size_t)nb[j] * dim, dim);
                    if (dd < epd) { epd = dd; ep = nb[j]; changed = 1; }
                }
            }
        }
        /* insert from min(li, top) down to 0 */
        for (l = (li < top ? li : top); l >= 0; --l) {
            i32 eps1 = ep, e;
            memset(vbits, 0, (size_t)((n + 7) >> 3) + 1);
            rn = hn_search_layer_(h, qi, &eps1, 1, efc, l, vbits, cid, cd, rid, rd);
            hn_heap_sorted_(rid, rd, rn, wq_id, wq_d);
            ns = hn_select_(h, wq_id, wq_d, rn, M, sel);
            hn_connect_(h, i, l, sel, ns);
            for (e = 0; e < ns; ++e) {
                i32 t = sel[e], tc, cap = hn_cap_(h, l);
                i32 *ta = hn_neigh_(h, t, l, &tc);
                if (tc < cap) { ta[tc] = i; h->cnt[t][l] = (i16)(tc + 1); }
                else {
                    /* t is full: re-select its neighbours from {current + i} */
                    i32 m, cnt2 = 0; f32 *d2 = wq_d;
                    for (m = 0; m < tc; ++m) {
                        nb2[cnt2] = ta[m];
                        d2[cnt2] = vs_sqdist(pts + (size_t)t * dim,
                                             pts + (size_t)ta[m] * dim, dim);
                        ++cnt2;
                    }
                    nb2[cnt2] = i;
                    d2[cnt2] = vs_sqdist(pts + (size_t)t * dim, pts + (size_t)i * dim, dim);
                    ++cnt2;
                    /* insertion-sort the (small) candidate set by distance */
                    for (m = 1; m < cnt2; ++m) {
                        i32 ti = nb2[m]; f32 td = d2[m]; j = m - 1;
                        while (j >= 0 && d2[j] > td) { nb2[j+1] = nb2[j]; d2[j+1] = d2[j]; --j; }
                        nb2[j+1] = ti; d2[j+1] = td;
                    }
                    { i32 keep = hn_select_(h, nb2, d2, cnt2, cap, sel);
                      hn_connect_(h, t, l, sel, keep); }
                }
            }
            if (rn > 0) ep = wq_id[0];   /* nearest, as entry for the next layer down */
        }
        if (li > h->maxl) { h->maxl = li; h->entry = i; }
    }
    arena_rewind(a, scratch);
    return h;
}

/* ---- route -------------------------------------------------- *
 * up to `nprobe` nearest node ids to `query`, nearest first. `ef` is the
 * layer-0 beam (clamped to [nprobe, HNSW_EF_CAP]); a good default is
 * max(nprobe*4, 32). Thread-safe: all scratch is on the stack. */
i32 hnsw_route(const Hnsw *h, const f32 *query, i32 nprobe, i32 ef, i32 *out_ids)
{
    u8  vbits[(HNSW_MAXN >> 3) + 1];
    i32 cid[HNSW_VISIT_CAP]; f32 cd[HNSW_VISIT_CAP];
    i32 rid[HNSW_EF_CAP + 1]; f32 rd[HNSW_EF_CAP + 1];
    i32 wq_id[HNSW_EF_CAP + 1]; f32 wq_d[HNSW_EF_CAP + 1];
    i32 ep, l, rn, i, out = 0;
    f32 epd;

    if (h->n <= 0) return 0;
    if (nprobe < 1) nprobe = 1;
    if (nprobe > h->n) nprobe = h->n;
    if (ef < nprobe) ef = nprobe;
    if (ef > HNSW_EF_CAP) ef = HNSW_EF_CAP;

    ep = h->entry;
    epd = vs_sqdist(query, h->pts + (size_t)ep * h->dim, h->dim);
    for (l = h->maxl; l >= 1; --l) {
        i32 changed = 1;
        while (changed) {
            i32 *nb, nbc, j;
            changed = 0;
            nb = hn_neigh_(h, ep, l, &nbc);
            for (j = 0; j < nbc; ++j) {
                f32 dd = vs_sqdist(query, h->pts + (size_t)nb[j] * h->dim, h->dim);
                if (dd < epd) { epd = dd; ep = nb[j]; changed = 1; }
            }
        }
    }
    memset(vbits, 0, (size_t)((h->n + 7) >> 3) + 1);
    rn = hn_search_layer_(h, query, &ep, 1, ef, 0, vbits, cid, cd, rid, rd);
    hn_heap_sorted_(rid, rd, rn, wq_id, wq_d);
    for (i = 0; i < rn && out < nprobe; ++i) out_ids[out++] = wq_id[i];
    return out;
}

/* like hnsw_route but also returns the squared-L2 distance of each hit
 * (out_d2 may be NULL). Used when the graph indexes the vectors
 * themselves, not just centroids. */
i32 hnsw_search_topk(const Hnsw *h, const f32 *query, i32 k, i32 ef,
                     i32 *out_ids, f32 *out_d2)
{
    u8  vbits[(HNSW_MAXN >> 3) + 1];
    i32 cid[HNSW_VISIT_CAP]; f32 cd[HNSW_VISIT_CAP];
    i32 rid[HNSW_EF_CAP + 1]; f32 rd[HNSW_EF_CAP + 1];
    i32 wq_id[HNSW_EF_CAP + 1]; f32 wq_d[HNSW_EF_CAP + 1];
    i32 ep, l, rn, i, out = 0;
    f32 epd;

    if (h->n <= 0) return 0;
    if (k < 1) k = 1;
    if (k > h->n) k = h->n;
    if (ef < k) ef = k;
    if (ef > HNSW_EF_CAP) ef = HNSW_EF_CAP;

    ep = h->entry;
    epd = vs_sqdist(query, h->pts + (size_t)ep * h->dim, h->dim);
    for (l = h->maxl; l >= 1; --l) {
        i32 changed = 1;
        while (changed) {
            i32 *nb, nbc, j;
            changed = 0;
            nb = hn_neigh_(h, ep, l, &nbc);
            for (j = 0; j < nbc; ++j) {
                f32 dd = vs_sqdist(query, h->pts + (size_t)nb[j] * h->dim, h->dim);
                if (dd < epd) { epd = dd; ep = nb[j]; changed = 1; }
            }
        }
    }
    memset(vbits, 0, (size_t)((h->n + 7) >> 3) + 1);
    rn = hn_search_layer_(h, query, &ep, 1, ef, 0, vbits, cid, cd, rid, rd);
    hn_heap_sorted_(rid, rd, rn, wq_id, wq_d);
    for (i = 0; i < rn && out < k; ++i) {
        out_ids[out] = wq_id[i];
        if (out_d2) out_d2[out] = wq_d[i];
        ++out;
    }
    return out;
}

/* ================================================================== */
#ifdef HNSW_DEMO
#define PRINT_IMPLEMENTATION
#include "print.h"
#include <stdio.h>
#include <time.h>

static f64 now_s(void)
{ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return (f64)t.tv_sec + (f64)t.tv_nsec * 1e-9; }

static u32 dr_(u32 *s){ *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static f32 df_(u32 *s){ return (f32)dr_(s) / (f32)0xffffffffu - 0.5f; }

/* brute-force nprobe-nearest, for the recall baseline */
static void brute_(const f32 *pts, i32 n, i32 dim, const f32 *q, i32 kp, i64 *out)
{
    f32 bd[64]; i32 have = 0, c, j;
    if (kp > 64) kp = 64;
    for (c = 0; c < n; ++c) {
        f32 d = vs_sqdist(q, pts + (size_t)c * dim, dim);
        if (have < kp) { j = have; bd[j] = d; out[j] = c;
            while (j > 0 && bd[j-1] > bd[j]) { f32 t = bd[j-1]; i64 ti = out[j-1];
                bd[j-1]=bd[j]; out[j-1]=out[j]; bd[j]=t; out[j]=ti; --j; } ++have; }
        else if (d < bd[kp-1]) { j = kp-1; bd[j] = d; out[j] = c;
            while (j > 0 && bd[j-1] > bd[j]) { f32 t = bd[j-1]; i64 ti = out[j-1];
                bd[j-1]=bd[j]; out[j-1]=out[j]; bd[j]=t; out[j]=ti; --j; } }
    }
}

int main(void)
{
    static Arena arena;
    Arena *a = &arena;
    i32 dim = 96, n = 4000, nq = 500, kp = 10, i, q, c, topics = 60;
    f32 *pts = (f32 *)arena_alloc(a, (size_t)n * dim * sizeof(f32));
    f32 *qs  = (f32 *)arena_alloc(a, (size_t)nq * dim * sizeof(f32));
    f32 *ctr = (f32 *)arena_alloc(a, (size_t)topics * dim * sizeof(f32));
    u32 st = 777u;
    Hnsw *h;
    HnswParams p;
    i64 g[64];
    i32 r[64];
    i32 hit1 = 0, hitk = 0;
    f64 t0, tb, tr;

    /* clustered, like the coarse centroids the router actually indexes:
     * structure to navigate, not uniform-random worst case. */
    for (i = 0; i < topics * dim; ++i) ctr[i] = df_(&st) * 6.0f;
    for (i = 0; i < n; ++i) { c = (i32)(dr_(&st) % (u32)topics);
        for (q = 0; q < dim; ++q) pts[(size_t)i*dim+q] = ctr[c*dim+q] + df_(&st) * 1.2f; }
    for (i = 0; i < nq; ++i) { c = (i32)(dr_(&st) % (u32)topics);
        for (q = 0; q < dim; ++q) qs[(size_t)i*dim+q] = ctr[c*dim+q] + df_(&st) * 1.4f; }

    p.M = 16; p.ef_construction = 200; p.seed = 42u;
    t0 = now_s(); h = hnsw_build(a, pts, n, dim, p); tb = (now_s() - t0) * 1e3;

    t0 = now_s();
    for (q = 0; q < nq; ++q) {
        const f32 *qq = qs + (size_t)q * dim;
        i32 got = hnsw_route(h, qq, kp, 48, r), x, y;
        brute_(pts, n, dim, qq, kp, g);
        if (got > 0 && (i64)r[0] == g[0]) ++hit1;
        for (x = 0; x < got; ++x)
            for (y = 0; y < kp; ++y) if ((i64)r[x] == g[y]) { ++hitk; break; }
    }
    tr = (now_s() - t0) / nq * 1e6;

    print_cstr("hnsw  n="); print_i64(n); print_cstr(" dim="); print_i64(dim);
    print_cstr("  M="); print_i64(h->M); print_cstr("  maxl="); print_i64(h->maxl);
    print_cstr("\n");
    printf("  build %.1f ms   route %.2f us/q\n", tb, tr);
    printf("  vs brute nprobe=%d:  nearest-1 = %.1f%%   recall@%d = %.1f%%\n",
           kp, 100.0 * hit1 / nq, kp, 100.0 * hitk / (nq * kp));
    print_flush();
    arena_free(a);
    return (hit1 * 100 / nq >= 95 && hitk * 100 / (nq * kp) >= 90) ? 0 : 1;
}
#endif /* HNSW_DEMO */
