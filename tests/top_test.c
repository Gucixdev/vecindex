/* top_test.c - the second level over the centroids (src/tq_top.c) against comparing a query with every centroid.
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -D_POSIX_C_SOURCE=200112L -O2 -Isrc tests/top_test.c -lm -lpthread
 *
 * 400,000 rows of 32 dimensions in ~1500 clusters, the second level forced on (tq_top_min_ lowered). Checked:
 *  - its lists: every cluster in exactly one top cell;
 *  - the probes of 300 queries through it against the full comparison: the closest cluster the same on nearly
 *    all, and nearly all of the np probes the same;
 *  - recall@10 of store_search with it within 0.01 of without it;
 *  - a cluster split in place lands in a top cell (the lists stay whole). */
#include "store.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static i32 fails = 0;
#define CHECK(c, msg) do { if (c) printf("  ok   %s\n", msg); else { printf("  FAIL %s (line %d)\n", msg, __LINE__); ++fails; } } while (0)

static u32 tr_(u32 *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static f32 tf_(u32 *s) { return (f32)((f64)tr_(s) / 4294967296.0 - 0.5); }

#define DIM 32
#define N 400000
#define NQ 300

static void tp_row_(i32 i, f32 *v)
{
    u32 s = 1234u + (u32)(i % 3000) * 7919u, r = 0x9e37u + (u32)i * 2654435761u;
    i32 d;
    for (d = 0; d < DIM; ++d) v[d] = tf_(&s) * 4.0f;
    for (d = 0; d < DIM; ++d) v[d] += tf_(&r) * 2.0f;
}

/* every cluster in one top cell */
static i32 tp_lists_(const Tq *t)
{
    i32 *seen = (i32 *)calloc((size_t)t->nc, sizeof(i32)), g, i, bad = 0;
    if (!t->top || t->tstart[0] != 0 || t->tstart[t->nk1] != t->nc) { free(seen); return 1; }
    for (g = 0; g < t->nk1; ++g) for (i = t->tstart[g]; i < t->tstart[g + 1]; ++i) { if (t->tlist[i] < 0 || t->tlist[i] >= t->nc) ++bad; else ++seen[t->tlist[i]]; }
    for (i = 0; i < t->nc; ++i) if (seen[i] != 1) ++bad;
    free(seen);
    return bad;
}

static f64 tp_recall_(RagStore *s)
{
    i64 ia[10], ib[10];
    f32 sa[10], sb[10], q[DIM], qn[DIM];
    i32 qi, i, j, hit = 0, have, d, na;
    for (qi = 0; qi < NQ; ++qi) {
        tp_row_(N + 17 * qi, q);
        for (d = 0; d < DIM; ++d) qn[d] = q[d];
        store_normalise_(qn, DIM);
        have = 0;
        for (i = 0; i < s->count; ++i) eng_topk_(ib, sb, 10, &have, (i64)i, store_score_(s, qn, 1.0f, i));
        for (i = 0; i < have; ++i) ib[i] = STORE_ID(s, ib[i]);
        na = store_search(s, q, 10, ia, sa);
        for (i = 0; i < na; ++i) for (j = 0; j < have; ++j) if (ia[i] == ib[j]) { ++hit; break; }
    }
    return (f64)hit / (NQ * 10.0);
}

int main(void)
{
    static Arena arena;
    RagStore *s;
    Tq *t;
    i32 i, keep = store_ivf_min_, keept = tq_top_min_, np, first = 0, both = 0, qi, a, b, *pa, *pb;
    f32 v[DIM], qn[DIM], *top;
    f64 r2, r1;
    u8 *scr;
    char msg[300];
    printf("top_test\n");
    store_ivf_min_ = 1000; tq_top_min_ = 64;
    s = store_new(&arena, DIM);
    for (i = 0; i < N; ++i) { tp_row_(i, v); store_add(s, v, i); }
    store_build(s);
    t = s->tq;
    np = t->nprobe < 32 ? t->nprobe : 32;                           /* (so that 8 np stays below the cluster count) */
    sprintf(msg, "%d rows in %d clusters under %d top cells: every cluster in one cell (%d broken)", N, t->nc, t->nk1, tp_lists_(t));
    CHECK(t->top && t->nc > tq_top_want_(np) && tp_lists_(t) == 0, msg);
    scr = (u8 *)malloc(tq_probe_scr_(t)); pa = (i32 *)malloc((size_t)np * 2 * sizeof(i32)); pb = pa + np;
    top = t->top;
    for (qi = 0; qi < NQ; ++qi) {
        tp_row_(N + 17 * qi, qn);
        store_normalise_(qn, DIM);
        tq_probe_(t, qn, np, scr, pa);
        t->top = (f32 *)0; tq_probe_(t, qn, np, scr, pb); t->top = top;   /* every centroid */
        first += pa[0] == pb[0];
        for (a = 0; a < np; ++a) for (b = 0; b < np; ++b) if (pa[a] == pb[b]) { ++both; break; }
    }
    sprintf(msg, "%d probes of %d queries: the closest cluster the same on %d, %.1f%% of all probes the same", np, NQ, first, 100.0 * both / (NQ * (f64)np));
    CHECK(first >= NQ * 97 / 100 && both >= (i32)(0.95 * NQ * np), msg);
    store_set_probe(s, np);
    r2 = tp_recall_(s);
    t->top = (f32 *)0; r1 = tp_recall_(s); t->top = top;
    sprintf(msg, "recall@10 at %d probes: %.4f through the top cells, %.4f comparing every centroid", np, r2, r1);
    CHECK(r2 >= r1 - 0.01, msg);
    {   TqCol sc, col = store_vcol_(s, &sc);
        i32 nc0 = t->nc, done;
        pthread_rwlock_wrlock(&s->lock);
        done = tq_split_big(t, &col, s->dead, 64, 3);                /* any 3 clusters over 64 rows */
        pthread_rwlock_unlock(&s->lock);
        sprintf(msg, "%d clusters split: %d clusters, every one in a top cell (%d broken)", done, t->nc, tp_lists_(t));
        CHECK(done == 3 && t->nc == nc0 + 3 && tp_lists_(t) == 0, msg);
    }
    free(scr); free(pa);
    store_destroy(s);
    store_ivf_min_ = keep; tq_top_min_ = keept;
    printf(fails ? "FAILED\n" : "all ok\n");
    return fails ? 1 : 0;
}
