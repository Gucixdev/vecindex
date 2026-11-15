/* exact_test.c - search with a proof (src/store_exact.c) against a plain scan of every row.
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -D_POSIX_C_SOURCE=200112L -O2 -Isrc tests/exact_test.c -lm -lpthread
 *
 * Two kinds of check, on stores of every kind (flat and clustered index; 4-, 2- and 1-bit codes; f32 and int8 rows;
 * cosine, dot and L2; odd dimensions; rows deleted, rows pending after the build, rows added live, numbers and
 * namespaces, a compaction by save and load):
 *  - the bound itself: for every row and query, the true cosine lies within store_ex_b_ of the estimate from the
 *    row's code - the inequality the proof rests on, checked on every row, not only on the ones that matter;
 *  - the result: store_search_exact returns exactly what scoring every row returns (same scores, same rows up to
 *    ties), and re-scores only a small part of the store. */
#include "store.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static i32 fails = 0;
#define CHECK(c, msg) do { if (c) printf("  ok   %s\n", msg); else { printf("  FAIL %s (line %d)\n", msg, __LINE__); ++fails; } } while (0)

static u32 xr_(u32 *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static f32 xf_(u32 *s) { return (f32)((f64)xr_(s) / 4294967296.0 - 0.5); }

/* n rows around 40 topics, lengths varying (for dot and L2) */
static f32 *ex_data_(i32 n, i32 dim, u32 seed)
{
    f32 *x = (f32 *)malloc((size_t)n * dim * sizeof(f32)), *c = (f32 *)malloc((size_t)40 * dim * sizeof(f32));
    u32 s = seed;
    i32 i, d;
    for (i = 0; i < 40 * dim; ++i) c[i] = xf_(&s) * 4.0f;
    for (i = 0; i < n; ++i) {
        f32 len = 0.5f + (f32)(xr_(&s) % 1000) / 500.0f;
        for (d = 0; d < dim; ++d) x[(size_t)i * dim + d] = (c[(size_t)(i % 40) * dim + d] + xf_(&s) * 2.5f) * len;
    }
    free(c);
    return x;
}

/* the k best by scoring every row the store would serve */
static i32 ex_brute_(RagStore *s, const f32 *q, i32 k, const StoreCnf *cnf, i64 *ids, f32 *sc)
{
    f32 *qn = (f32 *)malloc((size_t)s->dim * sizeof(f32)), qlen;
    i32 i, have = 0;
    memcpy(qn, q, (size_t)s->dim * sizeof(f32));
    qlen = (f32)eng_sqrt_(eng_ssq_(qn, s->dim));
    store_normalise_(qn, s->dim);
    for (i = 0; i < s->count; ++i) {
        if (STORE_DEAD(s, i) || (cnf && !store_cnf_ok_(s, cnf, i)) || store_row_bad_(s, i)) continue;
        eng_topk_(ids, sc, k, &have, (i64)i, store_score_(s, qn, qlen, i));
    }
    for (i = 0; i < have; ++i) ids[i] = STORE_ID(s, ids[i]);
    free(qn);
    return have;
}

/* a row's levels as the index stores them, read back from its block (not re-encoded): the codes' column layout of
 * tq_block_ - 16 bytes a column, lanes 0-15 in the low nibbles, 16-31 in the high ones */
static void ex_levels_(const Tq *t, i32 slot, f64 *lv)
{
    const u8 *blk = tq_bptr_(t, slot / 32);
    i32 lane = slot % 32, v, i;
    for (i = 0; i < t->dpad; ++i) lv[i] = 0.0;
    for (v = 0; v < t->vd; ++v) {
        u8 byte = blk[(size_t)v * 16 + (lane & 15)], c = lane < 16 ? (u8)(byte & 15) : (u8)(byte >> 4);
        if (t->bits == 4) { if (v < t->dpad) lv[v] = TQ_CENT[c]; }
        else if (t->bits == 2) { if (2 * v < t->dpad) lv[2 * v] = TQ_CENT2[c & 3]; if (2 * v + 1 < t->dpad) lv[2 * v + 1] = TQ_CENT2[c >> 2]; }
        else for (i = 0; i < 4; ++i) if (4 * v + i < t->dpad) lv[4 * v + i] = ((c >> i) & 1) ? TQ_L1 : -TQ_L1;
    }
}

/* The bound for one query, piece by piece, on every indexed row: the store's score is the line a cos + b; the u16 sum
 * gives q' . x^ to within the tables' rounding (x^ read back from the stored codes); the row's error byte covers
 * |x' - x^| and is no more than a step above it; and the whole: |cos - est| <= store_ex_b_. Returns how many rows broke
 * any of them; *worst: the largest |cos - est| / b. */
static i32 ex_bound_(RagStore *s, const f32 *q, f64 *worst)
{
    const Tq *t = s->tq;
    i32 nb = t->nblk + t->xn, i, j, bad = 0;
    u16 *sums = (u16 *)malloc((size_t)(nb ? nb : 1) * 32 * sizeof(u16));
    u8 *lut = (u8 *)malloc((size_t)t->vd * 16 + 32);
    f32 *qn = (f32 *)malloc((size_t)s->dim * sizeof(f32)), *qr = (f32 *)malloc((size_t)t->dpad * sizeof(f32));
    f32 *xr = (f32 *)malloc((size_t)t->dpad * sizeof(f32)), *tmp = (f32 *)malloc((size_t)t->dpad * sizeof(f32)), *rb = (f32 *)malloc((size_t)s->dim * sizeof(f32)), qlen;
    f64 *lv = (f64 *)malloc((size_t)t->dpad * sizeof(f64));
    u8 emax = 0;
    const u8 *serr = (const u8 *)0, *err = store_ex_errs_(s, s->built_upto, &emax, &serr);
    static const f64 CUTC[5] = { -0.2, 0.1, 0.4, 0.7, 0.95 };
    i32 cut[5];
    TqCol sc, col = store_vcol_(s, &sc);
    StoreExQ eq;
    memcpy(qn, q, (size_t)s->dim * sizeof(f32));
    qlen = (f32)eng_sqrt_(eng_ssq_(qn, s->dim));
    store_normalise_(qn, s->dim);
    store_ex_query_(t, qn, lut, &eq);
    tq_rot_(t, qn, qr, tmp);
    {   f64 n2 = 0.0; for (j = 0; j < t->dpad; ++j) n2 += (f64)qr[j] * qr[j];
        if (fabs(eq.qr - sqrt(n2)) > 1e-9 * sqrt(n2)) ++bad; }                         /* |q'| is the rotated query's length */
    for (i = 0; i < nb; ++i) tq_block_(tq_bptr_(t, i), lut, t->vd, (u16)0xffff, sums + (size_t)i * 32);
    for (i = 0; i < nb; ++i) {                                      /* the kernel's lane mask is store_ex_below_, at the edges too */
        i32 cuts[7], z;
        u16 sm[32];
        cuts[0] = 0; cuts[1] = 1; cuts[2] = 65535; cuts[3] = 65536;
        cuts[4] = sums[(size_t)i * 32 + 5]; cuts[5] = cuts[4] + 1; cuts[6] = sums[(size_t)i * 32 + 30];
        for (z = 0; z < 7; ++z) {
            u32 m = store_ex_lanes_(tq_bptr_(t, i), lut, t->vd, cuts[z], sm), want = 0;
            for (j = 0; j < 32; ++j) want |= (u32)!store_ex_below_(sums[(size_t)i * 32 + j], cuts[z]) << j;
            if (m != want || memcmp(sm, sums + (size_t)i * 32, sizeof(sm)) != 0) ++bad;
        }
    }
    for (j = 0; j < 5; ++j) {                                       /* the cut: the least sum whose bound reaches tau */
        f64 tau = 2.0 * CUTC[j] - 2.0;
        cut[j] = store_ex_cut_(&eq, emax, tau);
        if (cut[j] < 65536 && store_ex_ub_(&eq, 2.0, -2.0, (u16)cut[j], emax) < tau) ++bad;
        if (cut[j] > 0 && store_ex_ub_(&eq, 2.0, -2.0, (u16)(cut[j] - 1), emax) >= tau) ++bad;
    }
    for (i = 0; i < s->built_upto; ++i) if (err[i] > emax) ++bad;            /* emax: the largest byte */
    {   u8 *once = (u8 *)calloc((size_t)(t->n > 0 ? t->n : 1), 1);            /* a row in one slot at most: the pass keeps no seen set */
        for (i = 0; i < nb * 32; ++i) { i32 row = tq_srow_(t, i / 32, i % 32); if (row >= 0 && row < t->n && once[row]++) ++bad; }
        free(once); }
    for (i = 0; i < nb * 32; ++i) {
        i32 row = tq_srow_(t, i / 32, i % 32);
        f64 c, est, b, la, lb, qx = 0.0, e = 0.0, sco;
        if (row < 0 || row >= t->n || row >= s->built_upto) continue;
        c = store_dot_(s, qn, row);
        store_ex_line_(s, qlen, row, &la, &lb);
        sco = store_score_(s, qn, qlen, row);
        if (fabs(sco - (la * c + lb)) > 1e-4 * (1.0 + fabs(sco))) ++bad;               /* the score's line */
        ex_levels_(t, i, lv);
        tq_rot_(t, tq_row_(&col, row, rb), xr, tmp);
        for (j = 0; j < t->dpad; ++j) { qx += (f64)qr[j] * lv[j]; e += ((f64)xr[j] - lv[j]) * ((f64)xr[j] - lv[j]); }
        e = sqrt(e);
        if (fabs((f64)sums[i] / eq.sc + eq.base - qx) > eq.lutb + 1e-3) ++bad;          /* the tables' rounding */
        if (err[row] != STORE_EX_NONE && ((f64)err[row] * eq.step < e * (1.0 - 1e-6) || (f64)err[row] * eq.step > e + eq.step * 1.000001)) ++bad;   /* the byte */
        if (err[row] == STORE_EX_NONE && e < 254.0 * eq.step) ++bad;                    /* given up only when too big */
        est = store_ex_est_(&eq, sums[i]); b = store_ex_b_(&eq, err[row]);
        if (fabs(c - est) > b) ++bad;                                                   /* the whole */
        if (i < t->nblk * 32 && serr[i] != err[row] && serr[i] != STORE_EX_NONE) ++bad; /* slot order: the same bytes, or none */
        if (!s->metric) for (j = 0; j < 5; ++j)                                         /* below the cut: out by its own byte */
            if ((i32)sums[i] < cut[j] && store_ex_ub_(&eq, 2.0, -2.0, sums[i], err[row]) >= 2.0 * CUTC[j] - 2.0) ++bad;
        if (err[row] != STORE_EX_NONE && fabs(c - est) / b > *worst) *worst = fabs(c - est) / b;
    }
    free(sums); free(lut); free(qn); free(qr); free(xr); free(tmp); free(rb); free(lv);
    return bad;
}

/* nq queries (rows of x and fresh points): exact == brute force; returns mismatches, adds rows re-scored */
static i32 ex_same_(RagStore *s, const f32 *x, i32 n, i32 dim, i32 nq, i32 k, const StoreCnf *cnf, u32 seed, i64 *checked, i32 *bound_bad, f64 *worst)
{
    i64 ia[64], ib[64];
    f32 sa[64], sb[64], *q = (f32 *)malloc((size_t)dim * sizeof(f32));
    u32 r = seed;
    i32 qi, j, bad = 0;
    for (qi = 0; qi < nq; ++qi) {
        i64 ch = 0;
        i32 na, nb2, d;
        if (qi & 1) for (d = 0; d < dim; ++d) q[d] = x[(size_t)(xr_(&r) % (u32)n) * dim + d] + xf_(&r) * 0.3f;
        else for (d = 0; d < dim; ++d) q[d] = xf_(&r) * 3.0f;
        na = store_search_exact(s, q, k, ia, sa, cnf, &ch);
        nb2 = ex_brute_(s, q, k, cnf, ib, sb);
        *checked += ch;
        if (na != nb2 || ch < na) { ++bad; continue; }                  /* every row returned was scored, and counted */
        for (j = 0; j < na; ++j) if (sa[j] != sb[j] || (ia[j] != ib[j] && (j + 1 >= na || sa[j] != sa[j + 1]) && (j == 0 || sa[j] != sa[j - 1]))) { ++bad; break; }
        if (s->tq && (qi & 3) == 0) *bound_bad += ex_bound_(s, q, worst);
    }
    free(q);
    return bad;
}

typedef struct { const char *name; i32 dim, n, ivf, bits, i8, metric, del, pend, live, filt, reload, maxpct, late; } ExCase;   /* late: rows added and indexed after the first exact searches */   /* maxpct: rows re-scored, % at most (0: any) */

/* 21 queries in one store_search_exact_batch call (two passes, one short) == one call each: hits and counts */
static i32 ex_batch_(RagStore *s, const f32 *x, i32 n, i32 dim, const StoreCnf *cnf)
{
    enum { NQ = 21, K = 10 };
    i64 bi[NQ * K], si[K], bc[NQ], c;
    f32 bs[NQ * K], ss[K], *q = (f32 *)malloc((size_t)NQ * dim * sizeof(f32));
    i32 bn[NQ], i, j, sn, bad = 0;
    u32 r = 99u;
    for (i = 0; i < NQ * dim; ++i) q[i] = x[(size_t)((i / dim) * 37 % n) * dim + i % dim] + xf_(&r) * 0.2f;
    if (store_search_exact_batch(s, q, NQ, K, bi, bs, bn, cnf, bc) != 0) bad = NQ;
    for (i = 0; !bad && i < NQ; ++i) {
        c = 0;
        sn = store_search_exact(s, q + (size_t)i * dim, K, si, ss, cnf, &c);
        if (sn != bn[i]) { ++bad; continue; }
        for (j = 0; j < sn; ++j) if (si[j] != bi[i * K + j] || ss[j] != bs[i * K + j]) { ++bad; break; }
    }
    free(q);
    return bad;
}

static void ex_case_(const ExCase *c)
{
    static Arena arena;
    RagStore *s;
    f32 *x = ex_data_(c->n + c->pend + c->late, c->dim, 2026u + (u32)c->dim);
    i32 i, keep = store_ivf_min_, bad, nq = 120, bb = 0;
    i64 chk = 0;
    f64 worst = 0.0;
    StoreCnf cf, *cnf = (StoreCnf *)0;
    char msg[300];
    store_ivf_min_ = c->ivf ? 1000 : STORE_IVF_MIN;
    s = store_new(&arena, c->dim);
    if (c->metric) store_set_metric(s, c->metric);
    if (c->i8) store_set_vec_bytes(s, 1);
    if (c->bits) store_set_codes(s, c->bits);
    if (c->live) store_set_live(s, 1, 1.0f);
    for (i = 0; i < c->n; ++i) store_add(s, x + (size_t)i * c->dim, 1000 + i);
    store_build(s);
    if (c->del) for (i = 0; i < c->n; i += 7) store_delete(s, 1000 + i);
    if (c->filt) {
        for (i = 0; i < c->n; ++i) { store_set_num(s, 1000 + i, i % 100); store_set_ns(s, 1000 + i, (i % 3) + 1); }
        memset(&cf, 0, sizeof cf); cf.num = 1; cf.lo = 10; cf.hi = 60; cf.ns_on = 1; cf.ns = 2; cnf = &cf;
    }
    for (i = c->n; i < c->n + c->pend; ++i) store_add(s, x + (size_t)i * c->dim, 1000 + i);
    if (c->live) {
        i32 w;
        for (w = 0; w < 20000 && store_pending(s) > 0; ++w) { struct timespec ts; ts.tv_sec = 0; ts.tv_nsec = 1000000L; nanosleep(&ts, (struct timespec *)0); }
        store_set_live(s, 0, 1.0f);
        for (w = 0; w < 20 && store_maintain(s); ++w) {}
    }
    if (c->reload) {
        RagStore *l;
        bad = ex_same_(s, x, c->n, c->dim, 8, 10, cnf, 4u, &chk, &bb, &worst);   /* the error bytes built before the save */
        if (store_save(s, "/tmp/vecindex_exact_test.vxi") != 0) printf("  (save failed)\n");
        l = store_load(&arena, "/tmp/vecindex_exact_test.vxi");
        bad += l ? ex_same_(s, x, c->n, c->dim, 40, 10, cnf, 5u, &chk, &bb, &worst) : 1;  /* the saved one: compacted in place */
        if (l) { bad += ex_same_(l, x, c->n, c->dim, 40, 10, cnf, 6u, &chk, &bb, &worst); store_destroy(l); }
        remove("/tmp/vecindex_exact_test.vxi"); remove("/tmp/vecindex_exact_test.vxi.j"); remove("/tmp/vecindex_exact_test.vxi.jn");
        nq = 88;
    } else bad = ex_same_(s, x, c->n, c->dim, nq, 10, cnf, 7u, &chk, &bb, &worst) + ex_same_(s, x, c->n, c->dim, 20, 40, cnf, 8u, &chk, &bb, &worst), nq += 20;
    bad += ex_batch_(s, x, c->n, c->dim, cnf);
    if (c->late) {                                                  /* into the padding of built blocks and extra ones */
        for (i = c->n; i < c->n + c->late; ++i) store_add(s, x + (size_t)i * c->dim, 1000 + i);
        store_maintain(s);
        bad += ex_same_(s, x, c->n + c->late, c->dim, 60, 10, cnf, 9u, &chk, &bb, &worst);
        nq += 60;
    }
    sprintf(msg, "%-34s exact == every row scored, batch == one by one: %s; re-scored %.1f%% of %d rows a query%s; bound held on every row (%d broken, tightest %.2f)",
            c->name, bad ? "NO" : "yes", 100.0 * (f64)chk / nq / (c->n + c->pend + c->late), c->n + c->pend + c->late,
            s->tq ? (s->tq->nc > 1 ? ", clustered" : ", flat") : ", no index", bb, worst);
    if (s->tq && s->tq->nblk + s->tq->xn >= 256) sprintf(msg + strlen(msg), "; the pass on %d threads", store_ncpu_(s) > 4 ? 4 : store_ncpu_(s));
    CHECK(bad == 0 && bb == 0 && (!c->maxpct || 100.0 * (f64)chk / nq / (c->n + c->pend + c->late) <= c->maxpct), msg);
    store_destroy(s);
    store_ivf_min_ = keep;
    free(x);
}

/* readers on four threads while rows come in and a build runs: the error bytes are built and extended under
 * concurrent searches (run under TSan by ./run) */
typedef struct { RagStore *s; const f32 *x; i32 dim, n, bad; } ExThr;
static void *ex_reader_(void *p)
{
    ExThr *t = (ExThr *)p;
    i64 ids[10]; f32 sc[10];
    i32 q;
    for (q = 0; q < 40; ++q) if (store_search_exact(t->s, t->x + (size_t)(q * 37 % t->n) * t->dim, 10, ids, sc, (const StoreCnf *)0, (i64 *)0) < 1) ++t->bad;
    return (void *)0;
}

static void ex_threads_(void)
{
    static Arena arena;
    i32 dim = 32, n = 3000, i, keep = store_ivf_min_, bad = 0;
    f32 *x = ex_data_(n * 2, dim, 77u);
    RagStore *s;
    ExThr th[4];
    pthread_t id[4];
    store_ivf_min_ = 1000;
    s = store_new(&arena, dim);
    for (i = 0; i < n; ++i) store_add(s, x + (size_t)i * dim, i);
    store_build(s);
    for (i = 0; i < 4; ++i) { th[i].s = s; th[i].x = x; th[i].dim = dim; th[i].n = n; th[i].bad = 0; pthread_create(&id[i], (const pthread_attr_t *)0, ex_reader_, &th[i]); }
    for (i = n; i < 2 * n; ++i) { store_add(s, x + (size_t)i * dim, i); if (i == n + n / 2) store_build(s); }
    for (i = 0; i < 4; ++i) { pthread_join(id[i], (void **)0); bad += th[i].bad; }
    CHECK(bad == 0, "4 threads searching with a proof while rows come in and the index is rebuilt");
    {   i64 ia[10], ib[10]; f32 sa[10], sb[10]; i32 q, j, same = 1;
        for (q = 0; q < 30; ++q) {
            i32 na = store_search_exact(s, x + (size_t)(q * 101 % (2 * n)) * dim, 10, ia, sa, (const StoreCnf *)0, (i64 *)0), nb = ex_brute_(s, x + (size_t)(q * 101 % (2 * n)) * dim, 10, (const StoreCnf *)0, ib, sb);
            if (na != nb) same = 0; else for (j = 0; j < na; ++j) if (sa[j] != sb[j]) same = 0;
        }
        CHECK(same, "after it, still the same answers as scoring every row");
    }
    store_destroy(s); free(x);
    store_ivf_min_ = keep;
}

int main(void)
{
    static const ExCase cases[] = {
        { "flat, 4-bit, cosine",              64, 6000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0 },
        { "clustered, 4-bit, cosine",         64, 6000, 1, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0 },
        { "clustered, odd dim 33",            33, 5000, 1, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0 },
        { "2-bit codes",                      64, 5000, 1, 2, 0, 0, 0, 0, 0, 0, 0, 80, 0 },
        { "1-bit codes",                      48, 4000, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { "int8 rows",                        64, 5000, 1, 0, 1, 0, 0, 0, 0, 0, 0, 18, 0 },
        { "dot product",                      64, 5000, 1, 0, 0, 1, 0, 0, 0, 0, 0, 35, 0 },
        { "L2",                               64, 5000, 0, 0, 0, 2, 0, 0, 0, 0, 0, 8, 0 },
        { "deleted + pending rows",           64, 5000, 1, 0, 0, 0, 1, 700, 0, 0, 0, 34, 0 },
        { "live adds",                        64, 4000, 1, 0, 0, 0, 0, 3000, 1, 0, 0, 15, 0 },
        { "numbers + namespace filter",       64, 6000, 1, 0, 0, 0, 0, 0, 0, 1, 0, 8, 0 },
        { "deletes, save, load",              64, 5000, 1, 0, 0, 0, 1, 0, 0, 0, 1, 15, 0 },
        { "12000 rows: the pass on threads",  32, 12000, 1, 0, 0, 0, 1, 0, 0, 0, 0, 7, 0 },
        { "rows indexed after the first search", 64, 5000, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1500 }
    };
    i32 i;
    printf("exact_test\n");
    {   /* an error byte too big to keep means no bound, and is used only when it is */
        StoreExQ eq;
        eq.sc = 1.0; eq.base = 0.0; eq.qr = 8.0; eq.lutb = 0.5; eq.inv = 1.0 / 64.0; eq.step = 8.0 / 512.0;
        CHECK(store_ex_b_(&eq, STORE_EX_NONE) >= 1e29 && store_ex_b_(&eq, 254) < 1.0, "error byte 255: no bound at all (the row is always re-scored)");
        CHECK(store_ex_byte_(254.0 * eq.step, eq.step) == 254 && store_ex_byte_(254.0 * eq.step * 1.0001, eq.step) == STORE_EX_NONE
              && store_ex_byte_(0.1 * eq.step, eq.step) == 1 && store_ex_byte_(0.0, eq.step) == 0, "error bytes round up: 0.1 step -> 1, a bit past 254 -> 255 (none)");
        {   /* the cut is the least sum that reaches tau: that sum and above stay in, below it is out */
            f64 tau = store_ex_ub_(&eq, 2.0, -2.0, (u16)1000, 7);
            i32 c = store_ex_cut_(&eq, 7, tau);
            CHECK(c == 1000 && !store_ex_below_(c, c) && store_ex_below_(c - 1, c)
                  && store_ex_cut_(&eq, STORE_EX_NONE, tau) == 0 && store_ex_cut_(&eq, 7, 1e9) == 65536,
                  "the cut: a sum equal to it stays in, one below is out; no bound: nothing out");
        }
    }
    for (i = 0; i < (i32)(sizeof cases / sizeof cases[0]); ++i) ex_case_(&cases[i]);
    ex_threads_();
    {   /* no index at all: every row scored */
        static Arena arena;
        RagStore *s = store_new(&arena, 16);
        f32 *x = ex_data_(300, 16, 3u);
        i64 ch = 0;
        i32 bb = 0, j;
        f64 w = 0.0;
        for (j = 0; j < 300; ++j) store_add(s, x + (size_t)j * 16, j);
        CHECK(ex_same_(s, x, 300, 16, 20, 10, (const StoreCnf *)0, 9u, &ch, &bb, &w) == 0 && ch == 20 * 300, "no index yet: every row scored, the same answer");
        store_destroy(s); free(x);
    }
    printf("%s\n", fails ? "FAILED" : "all ok");
    return fails ? 1 : 0;
}
