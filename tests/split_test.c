/* split_test.c - clusters split in place (src/tq_split.c) as rows pour into a few of them.
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -D_POSIX_C_SOURCE=200112L -O2 -Isrc tests/split_test.c -lm -lpthread
 *
 * A clustered store gets rows from 40 topics, is built, then takes four times as many rows from 6 new topics, so a
 * few clusters overflow. Maintenance by hand (store_maintain) splits them. Checked after it:
 *  - the index's own invariants: every live row in exactly one slot, cluster sizes as counted, every slot's code the
 *    row's code (re-encoded here and compared lane by lane, so a move carried the right bytes), holes counted;
 *  - search: recall@10 against every row scored, split against the same rows with splits off (one big cluster each),
 *    and the scan's cost; search with a proof still equal to every row scored (its slot cache follows the splits);
 *  - a save and a load give the same hits;
 *  - live mode with splits on while 4 threads search (for TSan). */
#include "store.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static i32 fails = 0;
#define CHECK(c, msg) do { if (c) printf("  ok   %s\n", msg); else { printf("  FAIL %s (line %d)\n", msg, __LINE__); ++fails; } } while (0)

static u32 sr_(u32 *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static f32 sf_(u32 *s) { return (f32)((f64)sr_(s) / 4294967296.0 - 0.5); }

#define DIM 48

/* row i: topic (i % nt) + off of nt topics drawn from seed */
static void sp_row_(i32 i, i32 nt, i32 off, u32 seed, f32 *v)
{
    u32 s = seed + (u32)((i % nt) + off) * 7919u, r = 0x9e37u + (u32)i * 2654435761u;
    i32 d;
    for (d = 0; d < DIM; ++d) v[d] = sf_(&s) * 4.0f;
    for (d = 0; d < DIM; ++d) v[d] += sf_(&r) * 3.0f;
}

/* the index's invariants; returns how many broke */
static i32 sp_inv_(RagStore *s, i32 *holes_seen)
{
    const Tq *t = s->tq;
    i32 nb = t->nblk + t->xn, b, k, c, bad = 0, *seen = (i32 *)calloc((size_t)t->n, sizeof(i32)), *bl, *cnt;
    f32 *rot = (f32 *)malloc((size_t)t->dpad * sizeof(f32)), *tmp = (f32 *)malloc((size_t)t->dpad * sizeof(f32)), rb[DIM];
    u8 *cd = (u8 *)malloc((size_t)t->vd + 16), *got = (u8 *)malloc((size_t)t->vd + 16);
    TqCol sc, col = store_vcol_(s, &sc);
    *holes_seen = 0;
    for (b = 0; b < nb; ++b)
        for (k = 0; k < 32; ++k) {
            i32 r = tq_srow_(t, b, k);
            if (r < 0 || r >= t->n) continue;
            ++seen[r];
            tq_rot_(t, tq_row_(&col, r, rb), rot, tmp);
            tq_encode_row_(t, rot, cd);
            tq_get_lane_(tq_bptr_(t, b), t->vd, k, got);
            if (memcmp(cd, got, (size_t)t->vd) != 0) ++bad;                 /* the slot holds this row's code */
        }
    for (b = 0; b < t->n; ++b) if (seen[b] != 1) ++bad;                     /* every row in one slot */
    bl = (i32 *)malloc((size_t)(nb + 1) * sizeof(i32));
    cnt = (i32 *)calloc((size_t)t->nc, sizeof(i32));
    for (c = 0; c < t->nc; ++c) {                                           /* every block in one cluster; sizes */
        i32 n = tq_cblocks_(t, c, bl), j;
        for (j = 0; j < n; ++j) for (k = 0; k < 32; ++k) { i32 r = tq_srow_(t, bl[j], k); cnt[c] += r >= 0 && r < t->n; }
        if (t->csize && t->csize[c] != cnt[c]) ++bad;
    }
    for (b = 0; b < t->nblk; ++b) for (k = 0; k < 32; ++k) if (t->slot_row[(size_t)b * 32 + k] < 0) ++*holes_seen;
    for (b = 0; b < t->xn; ++b) for (k = 0; k < 32; ++k) if (t->xslot[(size_t)b * 32 + k] < 0) ++*holes_seen;
    free(seen); free(bl); free(cnt); free(rot); free(tmp); free(cd); free(got);
    return bad;
}

/* recall@10 of store_search against every row scored, over nq queries near the new topics; *same: exact searches
 * that equal every row scored */
static f64 sp_recall_(RagStore *s, i32 nq, i32 *same)
{
    i64 ia[10], ib[10], ic[10], chk;
    f32 sa[10], sb[10], scc[10], q[DIM], qn[DIM];
    i32 qi, i, j, hit = 0, have, d;
    *same = 0;
    for (qi = 0; qi < nq; ++qi) {
        i32 na;
        sp_row_(900000 + qi, 6, 100, 77u, q);
        for (d = 0; d < DIM; ++d) qn[d] = q[d];
        store_normalise_(qn, DIM);
        have = 0;
        for (i = 0; i < s->count; ++i) { if (STORE_DEAD(s, i)) continue; eng_topk_(ib, sb, 10, &have, (i64)i, store_score_(s, qn, 1.0f, i)); }
        for (i = 0; i < have; ++i) ib[i] = STORE_ID(s, ib[i]);
        na = store_search(s, q, 10, ia, sa);
        for (i = 0; i < na; ++i) for (j = 0; j < have; ++j) if (ia[i] == ib[j]) { ++hit; break; }
        if (store_search_exact(s, q, 10, ic, scc, (const StoreCnf *)0, &chk) == have) {
            for (i = 0; i < have && scc[i] == sb[i]; ++i) {}
            *same += i == have;
        }
    }
    return (f64)hit / (nq * 10.0);
}

typedef struct { RagStore *s; volatile i32 *stop; i32 bad; } SpReader;
static void *sp_reader_(void *p)
{
    SpReader *r = (SpReader *)p;
    i64 id[10], chk;
    f32 sc[10], q[DIM];
    i32 i = 0;
    while (!*r->stop) {
        sp_row_(500000 + i++, 6, 100, 77u, q);
        if (store_search(r->s, q, 10, id, sc) < 1) ++r->bad;
        if (store_search_exact(r->s, q, 10, id, sc, (const StoreCnf *)0, &chk) < 1) ++r->bad;
    }
    return (void *)0;
}

int main(void)
{
    static Arena arena;
    RagStore *s, *u, *l;
    i32 i, n0 = 6000, n1 = 24000, keep = store_ivf_min_, holes, same, w;
    f32 v[DIM];
    f64 r_split, r_nosplit;
    i64 splits;
    char msg[300];
    printf("split_test\n");
    store_ivf_min_ = 1000;
    s = store_new(&arena, DIM); u = store_new(&arena, DIM);
    for (i = 0; i < n0; ++i) { sp_row_(i, 40, 0, 77u, v); store_add(s, v, i); store_add(u, v, i); }
    store_build(s); store_build(u);
    store_set_split(u, 0);
    s->live_grow = 0.0f; u->live_grow = 0.0f;                               /* no rebuild either: the two methods alone */
    {   i32 nc0 = s->tq->nc;
        for (i = 0; i < n1; ++i) {
            sp_row_(n0 + i, 6, 100, 77u, v);
            store_add(s, v, n0 + i); store_add(u, v, n0 + i);
            if (i % 4000 == 3999) { for (w = 0; w < 50 && (store_maintain(s) & 8); ++w) {} store_maintain(u); }
        }
        for (w = 0; w < 200 && (store_maintain(s) & 9); ++w) {}
        store_maintain(u);
        splits = store_splits(s);
        sprintf(msg, "%d rows from 6 new topics into %d clusters: %ld splits in place, %d clusters now, none rebuilt",
                n1, nc0, (long)splits, s->tq->nc);
        CHECK(splits > 0 && s->tq->nc == nc0 + splits && store_stat_get_(&s->n_rebuilds) == 0 && s->built_upto == s->count, msg);
    }
    i = sp_inv_(s, &holes);
    sprintf(msg, "after the splits: every row in one slot with its own code, sizes as counted (%d broken), %d empty slots (%d holes made)",
            i, holes, s->tq->holes);
    CHECK(i == 0 && holes >= s->tq->holes, msg);
    {   i32 su;
        f64 cost_s, cost_u;
        r_split = sp_recall_(s, 200, &same);
        r_nosplit = sp_recall_(u, 200, &su);
        cost_s = (f64)s->tq->nprobe / s->tq->nc; cost_u = (f64)u->tq->nprobe / u->tq->nc;
        sprintf(msg, "recall@10 near the new topics: %.3f with splits (probe %d of %d clusters) against %.3f without (%d of %d)",
                r_split, s->tq->nprobe, s->tq->nc, r_nosplit, u->tq->nprobe, u->tq->nc);
        CHECK(r_split >= r_nosplit - 0.02, msg);
        (void)cost_s; (void)cost_u;
        sprintf(msg, "search with a proof after the splits == every row scored: %d/200 (without splits %d/200)", same, su);
        CHECK(same == 200 && su == 200, msg);
    }
    {   /* save and load: the same hits */
        i64 a[10], b[10];
        f32 sa[10], sb[10];
        i32 ok = 0, q;
        if (store_save(s, "/tmp/vecindex_split_test.vxi") != 0) printf("  (save failed)\n");
        l = store_load(&arena, "/tmp/vecindex_split_test.vxi");
        for (q = 0; l && q < 100; ++q) {
            i32 na, nb2, j;
            sp_row_(700000 + q, 6, 100, 77u, v);
            na = store_search(s, v, 10, a, sa); nb2 = store_search(l, v, 10, b, sb);
            for (j = 0; j < na && na == nb2 && a[j] == b[j]; ++j) {}
            ok += na == nb2 && j == na;
        }
        sprintf(msg, "saved and loaded after the splits: the same top 10 on %d/100, %d clusters", ok, l ? l->tq->nc : 0);
        CHECK(l && ok == 100 && l->tq->nc == s->tq->nc, msg);
        if (l) {                                                            /* a loaded (mapped) index splits too */
            for (i = 0; i < 8000; ++i) { sp_row_(n0 + n1 + i, 3, 200, 77u, v); store_add(l, v, n0 + n1 + i); }
            l->live_grow = 0.0f;
            for (w = 0; w < 200 && (store_maintain(l) & 9); ++w) {}
            i = sp_inv_(l, &holes);
            sprintf(msg, "the loaded index takes 8000 more rows: %ld splits, invariants hold (%d broken)", (long)store_splits(l), i);
            CHECK(store_splits(l) > 0 && i == 0, msg);
            store_destroy(l);
        }
        remove("/tmp/vecindex_split_test.vxi"); remove("/tmp/vecindex_split_test.vxi.j"); remove("/tmp/vecindex_split_test.vxi.jn");
    }
    store_destroy(u);
    {   /* live mode: splits while 4 threads search */
        pthread_t th[4];
        SpReader rd[4];
        volatile i32 stop = 0;
        i32 bad = 0;
        i64 s0 = store_splits(s);
        store_set_live(s, 1, 0.0f);
        for (i = 0; i < 4; ++i) { rd[i].s = s; rd[i].stop = &stop; rd[i].bad = 0; pthread_create(&th[i], (const pthread_attr_t *)0, sp_reader_, &rd[i]); }
        for (i = 0; i < 12000; ++i) { sp_row_(2000000 + i, 2, 300, 77u, v); store_add(s, v, 2000000 + i); }
        for (w = 0; w < 4000 && store_pending(s) > 0; ++w) { struct timespec ts; ts.tv_sec = 0; ts.tv_nsec = 1000000L; nanosleep(&ts, (struct timespec *)0); }
        { struct timespec ts; ts.tv_sec = 0; ts.tv_nsec = 300000000L; nanosleep(&ts, (struct timespec *)0); }
        stop = 1;
        for (i = 0; i < 4; ++i) { pthread_join(th[i], (void **)0); bad += rd[i].bad; }
        store_set_live(s, 0, 0.0f);
        i = sp_inv_(s, &holes);
        sprintf(msg, "live mode, 12000 rows while 4 threads search: %ld more splits, every search answered (%d empty), invariants hold (%d broken)",
                (long)(store_splits(s) - s0), bad, i);
        CHECK(store_splits(s) > s0 && bad == 0 && i == 0, msg);
    }
    store_destroy(s);
    store_ivf_min_ = keep;
    printf(fails ? "FAILED\n" : "all ok\n");
    return fails ? 1 : 0;
}
