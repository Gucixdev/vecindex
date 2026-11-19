/* store_rag.c - what a RAG pipeline does with the hits before a model reads them, done in the store: diverse hits
 * (MMR) with near-duplicates dropped, the chunks around a hit (small-to-big, parent-child), and a context built from
 * both to a byte budget. Ids are taken as document << shift | chunk, as store_range and store_search_grouped take
 * them. Every buffer comes from the scratch pool (scratch.h): no malloc on any of these paths.
 * Part of store.c, which includes it after store_text.c; not compiled on its own. */

/* the window of ids around id inside its group (the ids sharing id >> shift): from id - before to id + after, cut at
 * the group's edges (before or after below 0: to that edge). Unsigned, so negative ids and shift 63 work too. */
static void store_win_(i64 id, i32 shift, i32 before, i32 after, i64 *lo, i64 *hi)
{
    u64 u = (u64)id, mask = shift > 0 ? (((u64)1 << shift) - 1) : 0;
    u64 glo = u & ~mask, ghi = u | mask;
    *lo = (i64)(before < 0 || u - glo < (u64)before ? glo : u - (u64)before);
    *hi = (i64)(after < 0 || ghi - u < (u64)after ? ghi : u + (u64)after);
}

/* The chunks around id: the ids of its window (store_win_) that have a row, ascending, at most max. -1 out of
 * memory. */
i32 store_neighbors(RagStore *s, i64 id, i32 shift, i32 before, i32 after, i64 *out, i32 max)
{
    i64 lo, hi;
    store_win_(id, shift, before, after, &lo, &hi);
    return store_range(s, lo, hi, out, max);
}

/* MMR over n candidates cid / csc (best first): up to k of them into ids / scores, chosen one at a time for
 * lambda x relevance - (1 - lambda) x its highest cosine to a candidate already chosen (0 while none is, and never
 * below 0: an opposite direction is no bonus). Relevance is the candidate's score scaled to [0, 1] over the list, so
 * cosines, inner products, distances and fused ranks all work. dup > 0: a candidate whose cosine to a chosen one is
 * dup or more is dropped as a near-duplicate. A candidate gone since the search is skipped. Scores out are the
 * candidates' own. Returns how many, -1 out of memory. */
i32 store_mmr_pick(RagStore *s, const i64 *cid, const f32 *csc, i32 n, i32 k, f32 lambda, f32 dup, i64 *ids, f32 *scores)
{
    i32 dim = s->dim, i, t, have = 0;
    f32 *v, *ms, lo, span;
    u8 *st;                                                      /* 0 open, 1 chosen, 2 dropped or gone */
    Scr a;
    if (n <= 0 || k <= 0) return 0;
    if (scr_get_(&a, scr_need_((size_t)n * (size_t)dim * sizeof(f32)) + scr_need_((size_t)n * sizeof(f32)) + scr_need_((size_t)n)) != 0) return -1;
    v = (f32 *)scr_take_(&a, (size_t)n * (size_t)dim * sizeof(f32));
    ms = (f32 *)scr_take_(&a, (size_t)n * sizeof(f32));
    st = (u8 *)scr_take_(&a, (size_t)n);
    lo = csc[n - 1]; span = csc[0] - lo;
    for (i = 0; i < n; ++i) {
        f32 *x = v + (size_t)i * dim;
        f64 l;
        ms[i] = 0.0f; st[i] = 0;
        if (store_get_vec(s, cid[i], x) != 0) { st[i] = 2; continue; }
        l = eng_sqrt_(eng_ssq_(x, dim));
        if (l > 0.0) { i32 d; f32 r = (f32)(1.0 / l); for (d = 0; d < dim; ++d) x[d] *= r; }
    }
    for (t = 0; t < k; ++t) {
        i32 best = -1;
        f32 bv = 0.0f;
        const f32 *b;
        for (i = 0; i < n; ++i) {
            f32 rel, m;
            if (st[i]) continue;
            rel = span > 0.0f ? (csc[i] - lo) / span : 1.0f;
            m = lambda * rel - (1.0f - lambda) * ms[i];
            if (best < 0 || m > bv) { best = i; bv = m; }
        }
        if (best < 0) break;
        st[best] = 1; ids[have] = cid[best]; scores[have] = csc[best]; ++have;
        b = v + (size_t)best * dim;
        for (i = 0; i < n; ++i) {
            f32 c;
            if (st[i]) continue;
            c = vs_dot(v + (size_t)i * dim, b, dim);
            if (dup > 0.0f && c >= dup) { st[i] = 2; continue; }
            if (c > ms[i]) ms[i] = c;
        }
    }
    scr_put_(&a);
    return have;
}

/* store_search over the rows that pass the fields' filter flt (store_meta.c), tested inside the scan */
i32 store_search_flt(RagStore *s, const f32 *query, i32 k, const StoreFlt *flt, i64 *ids, f32 *scores)
{
    StoreCnf c;
    memset(&c, 0, sizeof(c));
    c.flt = flt;
    return store_search_cnf(s, query, k, ids, scores, &c);
}

/* the n best candidates: the vector and the words fused when there are words (store_search_hybrid: at most 250 a
 * side), else the vector's; flt (may be 0): only rows that pass it - inside the scan for a vector alone, the fused
 * list kept to them for words (which can leave fewer) */
static i32 store_cands_(RagStore *s, const f32 *q, const char *text, i32 tlen, const StoreFlt *flt, i32 n, i64 *ids, f32 *sc)
{
    i32 m, i, w;
    if (!(text && tlen > 0)) return !q ? 0 : flt ? store_search_flt(s, q, n, flt, ids, sc) : store_search(s, q, n, ids, sc);
    m = store_search_hybrid(s, q, text, tlen, n, ids, sc);
    if (!flt) return m;
    for (i = w = 0; i < m; ++i) if (store_flt_match(s, ids[i], flt) == 1) { ids[w] = ids[i]; sc[w] = sc[i]; ++w; }
    return w;
}

/* Diverse hits: k x VI_MMR_POOL candidates (store_cands_), then store_mmr_pick. Scores are the search's own (the
 * internal score for a vector alone: store_score_out turns it into what callers see). -1 out of memory. */
i32 store_search_mmr(RagStore *s, const f32 *q, const char *text, i32 tlen, const StoreFlt *flt, i32 k, f32 lambda, f32 dup, i64 *ids, f32 *scores)
{
    i32 n, c, r;
    i64 *ci;
    f32 *cs;
    Scr a;
    if (k <= 0) return 0;
    c = k > 65536 / VI_MMR_POOL ? 65536 : k * VI_MMR_POOL;
    if (c < k) c = k;
    if (scr_get_(&a, scr_need_((size_t)c * sizeof(i64)) + scr_need_((size_t)c * sizeof(f32))) != 0) return -1;
    ci = (i64 *)scr_take_(&a, (size_t)c * sizeof(i64));
    cs = (f32 *)scr_take_(&a, (size_t)c * sizeof(f32));
    n = store_cands_(s, q, text, tlen, flt, c, ci, cs);
    r = store_mmr_pick(s, ci, cs, n, k, lambda, dup, ids, scores);
    scr_put_(&a);
    return r;
}

/* A span of a context: chunks first to last of one document (every id between them that has a row), the best hit
 * among them, and their text in buf[off, off + len). */
typedef struct { i64 doc, first, last, hit; f32 score; i32 chunks, off, len; } StoreSpan;

typedef struct { i64 id, lo, hi; f32 score; i32 rank, win, pos; } StoreHit;   /* pos: in the chunk list, -1 none */
typedef struct { i64 lo, hi; i32 off, n; } StoreWin;                         /* chunks cl[off, off + n) */

static int store_hit_lo_cmp_(const void *a, const void *b)
{
    u64 x = (u64)((const StoreHit *)a)->lo, y = (u64)((const StoreHit *)b)->lo;
    return x < y ? -1 : x > y;
}

typedef struct { i32 z, y, best, hit; } StoreRun;                           /* chunks cl[z, y), its best hit */

static int store_run_cmp_(const void *a, const void *b)
{
    i32 x = ((const StoreRun *)a)->best, y = ((const StoreRun *)b)->best;
    return x < y ? -1 : x > y;
}

/* A context for a model: the k hits of store_search_mmr - or, given non-0, the caller's k hits given[0, k) with
 * their scores gsc (0: 1 / (rank + 1)), in that order, repeats once (flt: of the rows that pass it; the neighbours are the hits'
 * own document's chunks, filter or not); around each hit its window of before / after chunks
 * (store_win_; below 0: the whole document, at most VI_CTX_SIDE_MAX ids a side); windows of one document that
 * overlap or touch merged. Then the budget, cap bytes of buf: every hit's own chunk first, best hit first; then the
 * neighbours one ring at a time around every hit, best hit first, each while it fits and only next to what is
 * already in, so a span is always unbroken text. Spans (at most maxsp, best hit first) get their chunks' texts as
 * stored, one after another, nothing added between them. *used: bytes written. Returns how many spans, -1 out of
 * memory. */
i32 store_context(RagStore *s, const f32 *q, const char *text, i32 tlen, const StoreFlt *flt, const i64 *given, const f32 *gsc,
                  i32 k, i32 shift, i32 before, i32 after, f32 lambda, f32 dup, StoreSpan *sp, i32 maxsp, char *buf, i32 cap, i32 *used)
{
    i64 *hid, *cl;
    f32 *hsc;
    i32 m, i, j, nw = 0, w1, tot = 0, nsp = 0, d, off = 0, *ln, *ord;
    u8 *inc;                                                     /* 0 out, 1 in, 2 no row any more */
    StoreHit *h;
    StoreWin *w;
    StoreRun *run;
    i64 budget = cap > 0 ? cap : 0, spent = 0;
    Scr a, b;
    *used = 0;
    if (k <= 0 || maxsp <= 0) return 0;
    if (before < 0 || before > VI_CTX_SIDE_MAX) before = VI_CTX_SIDE_MAX;
    if (after < 0 || after > VI_CTX_SIDE_MAX) after = VI_CTX_SIDE_MAX;
    if (shift == 0) before = after = 0;                          /* every row its own document: no neighbours */
    w1 = before + after + 1;
    if (scr_get_(&a, scr_need_((size_t)k * sizeof(i64)) + scr_need_((size_t)k * sizeof(f32)) + scr_need_((size_t)k * sizeof(StoreHit))
                   + scr_need_((size_t)k * sizeof(StoreWin)) + scr_need_((size_t)k * sizeof(i32)) + scr_need_((size_t)k * sizeof(StoreRun))) != 0) return -1;
    hid = (i64 *)scr_take_(&a, (size_t)k * sizeof(i64));
    hsc = (f32 *)scr_take_(&a, (size_t)k * sizeof(f32));
    h = (StoreHit *)scr_take_(&a, (size_t)k * sizeof(StoreHit));
    w = (StoreWin *)scr_take_(&a, (size_t)k * sizeof(StoreWin));
    ord = (i32 *)scr_take_(&a, (size_t)k * sizeof(i32));
    run = (StoreRun *)scr_take_(&a, (size_t)k * sizeof(StoreRun));
    if (given) {                                                 /* the caller's hits, in its order, each once */
        for (i = m = 0; i < k; ++i) {
            for (j = 0; j < m && hid[j] != given[i]; ++j) {}
            if (j < m) continue;
            hid[m] = given[i]; hsc[m] = gsc ? gsc[i] : 1.0f / (f32)(i + 1); ++m;
        }
    } else if ((m = store_search_mmr(s, q, text, tlen, flt, k, lambda, dup, hid, hsc)) <= 0) { scr_put_(&a); return m; }
    for (i = 0; i < m; ++i) {
        h[i].id = hid[i]; h[i].score = hsc[i]; h[i].rank = i; h[i].pos = -1;
        store_win_(hid[i], shift, before, after, &h[i].lo, &h[i].hi);
    }
    /* merge the windows: by start, one document's that overlap or touch become one */
    qsort(h, (size_t)m, sizeof(StoreHit), store_hit_lo_cmp_);
    for (i = 0; i < m; ++i) {
        u64 lo = (u64)h[i].lo, hi = (u64)h[i].hi;
        if (nw > 0 && ((u64)w[nw - 1].lo >> shift) == (lo >> shift) && (lo <= (u64)w[nw - 1].hi || lo - (u64)w[nw - 1].hi == 1)) {
            if (hi > (u64)w[nw - 1].hi) w[nw - 1].hi = (i64)hi;
            w[nw - 1].n += w1;                                   /* room: one window's worth a hit */
        } else { w[nw].lo = (i64)lo; w[nw].hi = (i64)hi; w[nw].n = w1; ++nw; }
        h[i].win = nw - 1;
    }
    for (j = 0; j < nw; ++j) { w[j].off = tot; tot += w[j].n; }
    if (scr_get_(&b, scr_need_((size_t)tot * sizeof(i64)) + scr_need_((size_t)tot * sizeof(i32)) + scr_need_((size_t)tot)) != 0) { scr_put_(&a); return -1; }
    cl = (i64 *)scr_take_(&b, (size_t)tot * sizeof(i64));
    ln = (i32 *)scr_take_(&b, (size_t)tot * sizeof(i32));
    inc = (u8 *)scr_take_(&b, (size_t)tot);
    for (j = 0; j < nw; ++j) {                                   /* each window's chunks and their lengths */
        i32 n = store_range(s, w[j].lo, w[j].hi, cl + w[j].off, w[j].n), z;
        w[j].n = n > 0 ? n : 0;
        for (z = w[j].off; z < w[j].off + w[j].n; ++z) { ln[z] = store_get_text(s, cl[z], (char *)0, 0); inc[z] = ln[z] < 0 ? 2 : 0; }
    }
    for (i = 0; i < m; ++i) {                                    /* each hit's place in its window's chunks */
        const StoreWin *x = &w[h[i].win];
        i32 lo = x->off, hi = x->off + x->n;
        while (lo < hi) { i32 c = lo + (hi - lo) / 2; if (cl[c] < h[i].id) lo = c + 1; else hi = c; }
        if (lo < x->off + x->n && cl[lo] == h[i].id && inc[lo] != 2) h[i].pos = lo;
        ord[h[i].rank] = i;
    }
    /* the budget: the hits' own chunks, best first */
    for (i = 0; i < m; ++i) {
        i32 p = h[ord[i]].pos;
        if (p < 0 || inc[p]) continue;
        if (spent + ln[p] <= budget) { inc[p] = 1; spent += ln[p]; }
    }
    /* then ring d around every hit that is in, best hit first, only next to what is in */
    for (d = 1; d <= before || d <= after; ++d) {
        i32 reach = 0;
        for (i = 0; i < m; ++i) {
            const StoreHit *x = &h[ord[i]];
            i32 p = x->pos, lo = w[x->win].off, hi = w[x->win].off + w[x->win].n, c;
            if (p < 0 || inc[p] != 1) continue;
            c = p - d;
            if (d <= before && c >= lo) { reach = 1; if (!inc[c] && inc[c + 1] == 1 && spent + ln[c] <= budget) { inc[c] = 1; spent += ln[c]; } }
            c = p + d;
            if (d <= after && c < hi) { reach = 1; if (!inc[c] && inc[c - 1] == 1 && spent + ln[c] <= budget) { inc[c] = 1; spent += ln[c]; } }
        }
        if (!reach) break;
    }
    /* the spans: unbroken runs of chunks that are in; each holds a hit (it grew from one) */
    for (j = 0; j < nw; ++j) {
        i32 z = w[j].off, e = w[j].off + w[j].n;
        while (z < e) {
            i32 y, bi = -1;
            if (inc[z] != 1) { ++z; continue; }
            for (y = z; y < e && inc[y] == 1; ++y) {}
            for (i = 0; i < m; ++i) if (h[i].pos >= z && h[i].pos < y && (bi < 0 || h[i].rank < h[bi].rank)) bi = i;
            if (bi >= 0 && nsp < m) {
                run[nsp].z = z; run[nsp].y = y; run[nsp].best = h[bi].rank; run[nsp].hit = bi;
                ++nsp;
            }
            z = y;
        }
    }
    qsort(run, (size_t)nsp, sizeof(StoreRun), store_run_cmp_);
    if (nsp > maxsp) nsp = maxsp;
    for (i = 0; i < nsp; ++i) {                                  /* the texts, best span first */
        StoreSpan *o = &sp[i];
        const StoreHit *x = &h[run[i].hit];
        i32 z;
        o->first = cl[run[i].z]; o->last = cl[run[i].y - 1]; o->hit = x->id; o->score = x->score;
        o->doc = (i64)((u64)x->id >> shift); o->chunks = run[i].y - run[i].z; o->off = off;
        for (z = run[i].z; z < run[i].y; ++z) {
            i32 r = off < cap ? store_get_text(s, cl[z], buf + off, cap - off) : 0;
            if (r > 0) off += r < cap - off ? r : cap - off;     /* a text grown since it was measured: cut at cap */
        }
        o->len = off - o->off;
    }
    *used = off;
    scr_put_(&b);
    scr_put_(&a);
    return nsp;
}
