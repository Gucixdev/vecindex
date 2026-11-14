/* tq_top.c - a second level over the centroids, so a query does not compare itself with every one of them.
 * Part of tq.c, which includes it at this point; not compiled on its own.
 *
 * With nc clusters a query scored all nc centroids before it scanned a single row: 40,000 at 100M rows, 4 million
 * at 10^12 - that part grows as sqrt(rows) and at scale it is the whole query. Here the centroids themselves are
 * grouped into k1 ~ sqrt(nc) top cells (a spherical k-means over the centroids, made when the index is prepared -
 * nothing new in the file). A query scores the k1 top cells, takes the closest ones until they hold enough centroids
 * (tq_top_want_), and scores only those: k1 + a few thousand dot products instead of nc. Indexes below tq_top_min_
 * clusters keep the full comparison, which is cheap there and exact. */

#ifndef VI_TOP_MIN
#define VI_TOP_MIN 16384
#endif
static i32 tq_top_min_ = VI_TOP_MIN;     /* clusters from which the second level is made (~16M rows); tests lower it */

#define TQ_TOP_ROUNDS 5
#define TQ_TOP_KMAX   4096

/* centroids a query with np probes wants to choose among */
static i32 tq_top_want_(i32 np) { i32 w = np * 8; return w < 1024 ? 1024 : w; }

static void tq_top_free_(Tq *t)
{
    free(t->top); free(t->tstart); free(t->tlist);
    t->top = (f32 *)0; t->tstart = (i32 *)0; t->tlist = (i32 *)0; t->nk1 = 0;
}

/* the top cell closest to v */
static i32 tq_top_near_(const Tq *t, const f32 *v)
{
    i32 g, b = 0;
    f32 bs = -3.0f;
    for (g = 0; g < t->nk1; ++g) { f32 s = vs_dot(v, t->top + (size_t)g * t->dim, t->dim); if (s > bs) { bs = s; b = g; } }
    return b;
}

/* The top cells of t's centroids (again, when there were some). Leaves t without them below tq_top_min_ clusters
 * or when memory runs out: then every centroid is compared, as before. */
static void tq_top_build_(Tq *t)
{
    i32 nc = t->nc, dim = t->dim, k1, g, c, d, it, *of, *cnt;
    f32 *top, *sum;
    tq_top_free_(t);
    if (nc < tq_top_min_ || nc < 64) return;
    k1 = (i32)(sqrt((f64)nc) + 0.5);
    if (k1 > TQ_TOP_KMAX) k1 = TQ_TOP_KMAX;
    top = (f32 *)malloc((size_t)k1 * dim * sizeof(f32)); sum = (f32 *)malloc((size_t)k1 * dim * sizeof(f32));
    of = (i32 *)malloc((size_t)nc * sizeof(i32)); cnt = (i32 *)malloc((size_t)(k1 + 1) * sizeof(i32));
    t->tstart = (i32 *)malloc((size_t)(k1 + 1) * sizeof(i32)); t->tlist = (i32 *)malloc((size_t)nc * sizeof(i32));
    if (!top || !sum || !of || !cnt || !t->tstart || !t->tlist) { free(top); free(sum); free(of); free(cnt); tq_top_free_(t); return; }
    for (g = 0; g < k1; ++g)                                     /* the start: every (nc / k1)-th centroid */
        memcpy(top + (size_t)g * dim, t->cent + (size_t)((i64)g * nc / k1) * dim, (size_t)dim * sizeof(f32));
    t->top = top; t->nk1 = k1;
    for (it = 0; it <= TQ_TOP_ROUNDS; ++it) {
        for (c = 0; c < nc; ++c) of[c] = tq_top_near_(t, t->cent + (size_t)c * dim);
        if (it == TQ_TOP_ROUNDS) break;                          /* the last pass only assigns */
        memset(sum, 0, (size_t)k1 * dim * sizeof(f32));
        memset(cnt, 0, (size_t)k1 * sizeof(i32));
        for (c = 0; c < nc; ++c) { f32 *to = sum + (size_t)of[c] * dim; const f32 *v = t->cent + (size_t)c * dim; for (d = 0; d < dim; ++d) to[d] += v[d]; ++cnt[of[c]]; }
        for (g = 0; g < k1; ++g) {
            f32 *v = sum + (size_t)g * dim;
            f64 n = 0.0;
            if (!cnt[g]) continue;                               /* an empty cell keeps its place */
            for (d = 0; d < dim; ++d) n += (f64)v[d] * v[d];
            if (n <= 0.0) continue;
            n = 1.0 / sqrt(n);
            for (d = 0; d < dim; ++d) top[(size_t)g * dim + d] = (f32)(v[d] * n);
        }
    }
    memset(cnt, 0, (size_t)(k1 + 1) * sizeof(i32));
    for (c = 0; c < nc; ++c) ++cnt[of[c]];
    t->tstart[0] = 0;
    for (g = 0; g < k1; ++g) t->tstart[g + 1] = t->tstart[g] + cnt[g];
    for (g = 0; g < k1; ++g) cnt[g] = t->tstart[g];
    for (c = 0; c < nc; ++c) t->tlist[cnt[of[c]]++] = c;
    free(sum); free(of); free(cnt);
}

/* Top cells read from a file: copied in when the lists are whole (starts rising from 0 to nc, every cluster in
 * exactly one cell); else t stays without them and tq_prepare_ builds them. */
void tq_top_set(Tq *t, i32 k1, const f32 *top, const i32 *ts, const i32 *tl)
{
    i32 g, i, ok = ts[0] == 0 && ts[k1] == t->nc;
    u8 *seen;
    tq_top_free_(t);
    for (g = 0; ok && g < k1; ++g) ok = ts[g] <= ts[g + 1];
    if (!ok || !(seen = (u8 *)calloc((size_t)t->nc, 1))) return;
    for (i = 0; ok && i < t->nc; ++i) { ok = tl[i] >= 0 && tl[i] < t->nc && !seen[tl[i]]; if (ok) seen[tl[i]] = 1; }
    free(seen);
    if (!ok) return;
    t->top = (f32 *)malloc((size_t)k1 * t->dim * sizeof(f32));
    t->tstart = (i32 *)malloc((size_t)(k1 + 1) * sizeof(i32)); t->tlist = (i32 *)malloc((size_t)t->nc * sizeof(i32));
    if (!t->top || !t->tstart || !t->tlist) { tq_top_free_(t); return; }
    memcpy(t->top, top, (size_t)k1 * t->dim * sizeof(f32));
    memcpy(t->tstart, ts, (size_t)(k1 + 1) * sizeof(i32)); memcpy(t->tlist, tl, (size_t)t->nc * sizeof(i32));
    t->nk1 = k1;
}

/* cluster c (new: t->nc already counts it) into its closest top cell. On no memory the second level is dropped. */
static void tq_top_add_(Tq *t, i32 c)
{
    i32 g, i, *tl;
    if (!t->top) return;
    if (!(tl = (i32 *)realloc(t->tlist, (size_t)t->nc * sizeof(i32)))) { tq_top_free_(t); return; }
    t->tlist = tl;
    g = tq_top_near_(t, t->cent + (size_t)c * t->dim);
    for (i = t->tstart[t->nk1]; i > t->tstart[g + 1]; --i) tl[i] = tl[i - 1];
    tl[t->tstart[g + 1]] = c;
    for (i = g + 1; i <= t->nk1; ++i) ++t->tstart[i];
}
