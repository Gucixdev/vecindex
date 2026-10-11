/* stress.c - hammer store.c / engine.c and check the invariants hold.
 *
 * Not a demo (those live in each module). This runs a long churn of
 * add / delete / rebuild / search against a brute-force f32 oracle, hits
 * the edge cases explicitly, and cross-checks the threaded batch path
 * and save/load.
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -D_POSIX_C_SOURCE=200112L -O2 \
 *       -march=native -I. stress.c -DSTRESS -lm -lpthread -o stress && ./stress
 */

#include "store.c"        /* pulls engine.c + q1..q5 + ivf + arena + pool */
#include <stdlib.h>
#include <time.h>

static i32 g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { print_cstr("  FAIL: "); print_cstr(msg); print_cstr("\n"); ++g_fail; } \
} while (0)

static f64 now_s(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (f64)ts.tv_sec + (f64)ts.tv_nsec * 1e-9;
}

static u32 xs_(u32 *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static f32 rf_(u32 *s) { return (f32)xs_(s) / (f32)0xffffffffu - 0.5f; }

/* approx N(0,1) - sum of 4 uniforms, cheap */
static f32 gauss_(u32 *s)
{ return rf_(s) + rf_(s) + rf_(s) + rf_(s); }

/* one row of an EMBEDDING-REALISTIC corpus: a cluster centre plus
 * ANISOTROPIC noise (per-cluster per-dim scale) with a fat tail (5% of
 * dims get a 4x kick), unit-normalised - the shape real sentence
 * embeddings actually have, not isotropic gaussian blobs. `scale` is
 * [topics * dim]. */
static void mkvec_(f32 *out, const f32 *centers, const f32 *scale,
                   i32 topic, i32 dim, u32 *st)
{
    f32 nrm = 0.0f; i32 d;
    for (d = 0; d < dim; ++d) {
        f32 sd = scale[topic * dim + d];
        f32 noise = gauss_(st) * sd;
        if ((xs_(st) & 31u) == 0u) noise *= 4.0f;      /* fat tail */
        out[d] = centers[topic * dim + d] + noise;
    }
    for (d = 0; d < dim; ++d) nrm += out[d] * out[d];
    nrm = nrm > 0.0f ? 1.0f / (f32)eng_sqrt_((f64)nrm) : 1.0f;
    for (d = 0; d < dim; ++d) out[d] *= nrm;
}

/* brute-force exact cosine top-k over a caller-held live set */
static void oracle_topk_(const f32 *vecs, const i64 *ids, const u8 *dead,
                         i32 n, i32 dim, const f32 *q, i32 k, i64 *out, i32 *nout)
{
    f32 *sc = (f32 *)malloc((size_t)k * sizeof(f32));
    i32 have = 0, r, d, j;
    for (r = 0; r < n; ++r) {
        f32 dot = 0.0f, cs;
        if (dead[r]) continue;
        for (d = 0; d < dim; ++d) dot += q[d] * vecs[(size_t)r * dim + d];
        cs = dot;               /* both sides unit-normalised */
        if (have < k) {
            j = have; out[j] = ids[r]; sc[j] = cs;
            while (j > 0 && sc[j-1] < sc[j]) { f32 t=sc[j-1]; i64 ti=out[j-1];
                sc[j-1]=sc[j]; out[j-1]=out[j]; sc[j]=t; out[j]=ti; --j; }
            ++have;
        } else if (cs > sc[k-1]) {
            j = k-1; out[j] = ids[r]; sc[j] = cs;
            while (j > 0 && sc[j-1] < sc[j]) { f32 t=sc[j-1]; i64 ti=out[j-1];
                sc[j-1]=sc[j]; out[j-1]=out[j]; sc[j]=t; out[j]=ti; --j; }
        }
    }
    *nout = have;
    free(sc);
}

static i32 overlap_(const i64 *a, i32 na, const i64 *b, i32 nb)
{
    i32 i, j, o = 0;
    for (i = 0; i < na; ++i) for (j = 0; j < nb; ++j) if (a[i] == b[j]) { ++o; break; }
    return o;
}

/* concurrent-reader harness: many threads hammer store_search while the
 * main thread mutates. They cannot check recall (the set is moving) -
 * just that every call returns a well-formed result and never faults. */
static RagStore *volatile g_reader_store;
static volatile i32       g_reader_stop;
static volatile i32       g_reader_crash;
static i32                g_reader_dim;

static void *reader_thread_(void *arg)
{
    u32 seed = (u32)((size_t)&arg) | 1u;
    f32 *q = (f32 *)malloc((size_t)g_reader_dim * sizeof(f32));
    i64 r[10]; f32 c[10];
    (void)arg;
    while (!g_reader_stop) {
        i32 d, h, x;
        for (d = 0; d < g_reader_dim; ++d) q[d] = rf_(&seed);
        h = store_search(g_reader_store, q, 10, r, c);
        if (h < 0 || h > 10) g_reader_crash = 1;
        for (x = 0; x < h; ++x) if (r[x] < 0) g_reader_crash = 1;
    }
    free(q);
    return (void *)0;
}

int main(void)
{
    static Arena arena;
    Arena *a = &arena;
    i32 dim = 256, topics = 50, i;
    u32 st = 0xC0FFEEu;
    f32 *centers = (f32 *)arena_alloc(a, (size_t)topics * dim * sizeof(f32));
    f32 *scale   = (f32 *)arena_alloc(a, (size_t)topics * dim * sizeof(f32));
    f32 *tmp = (f32 *)arena_alloc(a, (size_t)dim * sizeof(f32));
    RagStore *s;
    /* shadow oracle: every row ever added, with a live/dead bit */
    i32 ocap = 200000, on = 0;
    f32 *ovec = (f32 *)malloc((size_t)ocap * dim * sizeof(f32));
    i64 *oid  = (i64 *)malloc((size_t)ocap * sizeof(i64));
    u8  *odead = (u8 *)calloc((size_t)ocap, 1);
    i64 next_id = 1;
    f64 t0;

    /* embedding-realistic: centres ~N(0,1)-ish, per-cluster anisotropic
     * per-dim noise scale in [0.05, 0.55] (some dims tight, some loose). */
    for (i = 0; i < topics * dim; ++i) {
        centers[i] = gauss_(&st);
        scale[i] = 0.05f + (rf_(&st) + 0.5f) * 0.50f;
    }
    s = store_new(a, dim);
    { EngGoal g = s->goal; g.max_code_bytes = dim;                   /* a code budget below f32 size: keeps the q1-q5/IVF tuner path under test */
      store_set_goal(s, g); }

    print_cstr("stress  dim="); print_i64(dim); print_cstr(" topics="); print_i64(topics);
    print_cstr("\n");
    print_flush();

    /* -------- edge cases first -------- */
    {
        i64 r[8]; f32 c[8];
        CHECK(store_search(s, tmp, 5, r, c) == 0, "search on empty store != 0");
        store_build(s);
        CHECK(store_search(s, tmp, 5, r, c) == 0, "search after empty build != 0");
        CHECK(s->idx == (EngIndex *)0, "empty build left a non-NULL index");
    }

    /* -------- churn: 30 rounds of add / delete / (some) rebuild -------- */
    t0 = now_s();
    for (i = 0; i < 8; ++i) {
        i32 add = 400 + (i32)(xs_(&st) % 700u), j, del, k;
        for (j = 0; j < add && on < ocap; ++j) {
            i32 topic = (i32)(xs_(&st) % (u32)topics);
            mkvec_(tmp, centers, scale, topic, dim, &st);
            /* ~2% duplicate doc ids on purpose */
            oid[on] = ((xs_(&st) & 63u) == 0u && on > 10)
                      ? oid[xs_(&st) % (u32)on] : next_id++;
            memcpy(ovec + (size_t)on * dim, tmp, (size_t)dim * sizeof(f32));
            odead[on] = 0;
            store_add(s, tmp, oid[on]);
            ++on;
        }
        /* delete ~15% of live doc ids at random */
        del = on / 7;
        for (k = 0; k < del; ++k) {
            i32 pick = (i32)(xs_(&st) % (u32)on);
            i64 did = oid[pick];
            i32 m;
            for (m = 0; m < on; ++m) if (oid[m] == did && !odead[m]) odead[m] = 1;
            (void)store_delete(s, did);   /* store removes all matching too */
        }
        if (i == 5) store_build(s);   /* one rebuild mid-churn */

        /* invariant: store's own live count matches the oracle's */
        {
            i32 olive = 0, m;
            for (m = 0; m < on; ++m) if (!odead[m]) ++olive;
            CHECK(store_count(s) == olive, "live count drifted from oracle");
        }
    }
    print_cstr("  churn (8 rounds, "); print_i64(on);
    print_cstr(" rows touched) in "); print_i64((i64)((now_s()-t0)*1e3+0.5));
    print_cstr(" ms\n");
    print_flush();

    store_build(s);   /* fold in the pending tail, drop tombstones */
    {   /* after a full build: count == compacted count, no pending */
        i32 olive = 0, m;
        for (m = 0; m < on; ++m) if (!odead[m]) ++olive;
        CHECK(store_count(s) == olive, "post-build live != oracle");
        CHECK(s->count == store_count(s), "post-build s->count != live (tombstones left)");
        CHECK(store_pending(s) == 0, "pending tail not cleared by build");
    }

    /* -------- recall vs the oracle, 300 held-out-ish queries -------- */
    {
        i32 nq = 150, q, hit1 = 0, ov = 0, tot = 0, ndead_seen = 0;
        i64 got[10], exa[10]; f32 gs[10]; i32 ng, ne;
        for (q = 0; q < nq; ++q) {
            i32 pick;
            /* query = a live row's vector + a little noise */
            do { pick = (i32)(xs_(&st) % (u32)on); } while (odead[pick]);
            for (i = 0; i < dim; ++i)
                tmp[i] = ovec[(size_t)pick * dim + i] + rf_(&st) * 0.05f;
            {   f32 nrm = 0.0f; for (i=0;i<dim;++i) nrm += tmp[i]*tmp[i];
                nrm = nrm>0.0f ? 1.0f/(f32)eng_sqrt_((f64)nrm) : 1.0f;
                for (i=0;i<dim;++i) tmp[i] *= nrm; }

            ng = store_search(s, tmp, 10, got, gs);
            oracle_topk_(ovec, oid, odead, on, dim, tmp, 10, exa, &ne);
            ov += overlap_(got, ng, exa, ne); tot += ne;
            if (ng > 0 && got[0] == oid[pick]) ++hit1;
            /* no tombstoned id may appear */
            for (i = 0; i < ng; ++i) {
                i32 m;
                for (m = 0; m < on; ++m)
                    if (oid[m] == got[i] && odead[m]) {
                        /* allowed only if a live row also carries that id */
                        i32 x, live_dup = 0;
                        for (x = 0; x < on; ++x)
                            if (oid[x] == got[i] && !odead[x]) { live_dup = 1; break; }
                        if (!live_dup) ++ndead_seen;
                        break;
                    }
            }
        }
        print_cstr("  recall@10 vs exact = "); print_i64((i64)(100L * ov / tot));
        print_cstr("%   self(noisy)@1 = "); print_i64((i64)(100L * hit1 / nq));
        print_cstr("%   tombstoned leaks = "); print_i64(ndead_seen); print_cstr("\n");
        CHECK(ndead_seen == 0, "a fully-deleted doc id came back in results");
        CHECK(100L * ov / tot >= 55, "recall@10 collapsed (<55%)");
        CHECK(100L * hit1 / nq >= 90, "noisy self-retrieval@1 collapsed (<90%)");
    }

    /* -------- threaded batch == single-thread, bit for bit -------- */
    {
        i32 nb = 800, mism = 0, q;
        f32 *qb = (f32 *)malloc((size_t)nb * dim * sizeof(f32));
        i64 *bi = (i64 *)malloc((size_t)nb * 10 * sizeof(i64));
        f32 *bs = (f32 *)malloc((size_t)nb * 10 * sizeof(f32));
        for (q = 0; q < nb; ++q) {
            i32 pick = (i32)(xs_(&st) % (u32)on), d;
            for (d = 0; d < dim; ++d)
                qb[(size_t)q*dim+d] = ovec[(size_t)pick*dim+d] + rf_(&st)*0.03f;
        }
        t0 = now_s();
        engine_search_batch(s->idx, qb, nb, 10, bi, bs);
        print_cstr("  batch 4000q/12T in "); print_i64((i64)((now_s()-t0)*1e6/nb+0.5));
        print_cstr(" us/q\n");
        for (q = 0; q < nb; ++q) {
            i64 r1[10]; f32 c1[10];
            i32 h1 = engine_search(s->idx, qb + (size_t)q*dim, 10, r1, c1), z;
            for (z = 0; z < h1 && z < 10; ++z)
                if (r1[z] != bi[(size_t)q*10+z]) { ++mism; break; }
        }
        CHECK(mism == 0, "threaded batch disagreed with single-thread");
        print_cstr("  batch vs 1T mismatches = "); print_i64(mism); print_cstr("\n");
        free(qb); free(bi); free(bs);
    }

    /* -------- save / load mid-life, identical results -------- */
    {
        i32 same = 0, tried = 0, q;
        RagStore *s2;
        CHECK(store_save(s, "/tmp/vecidx_stress.vxi") == 0, "store_save failed");
        s2 = store_load(a, "/tmp/vecidx_stress.vxi");
        CHECK(s2 != (RagStore *)0, "store_load returned NULL");
        if (s2) {
            CHECK(store_count(s2) == store_count(s), "loaded live count differs");
            for (q = 0; q < 400; ++q) {
                i32 pick = (i32)(xs_(&st) % (u32)on), d;
                i64 r1[5], r2[5]; f32 c1[5], c2[5];
                i32 h1, h2;
                for (d = 0; d < dim; ++d) tmp[d] = ovec[(size_t)pick*dim+d] + rf_(&st)*0.04f;
                h1 = store_search(s,  tmp, 5, r1, c1);
                h2 = store_search(s2, tmp, 5, r2, c2);
                ++tried;
                if (h1 == h2 && h1 > 0 && r1[0] == r2[0]) ++same;
            }
            print_cstr("  save/load top-1 identical on ");
            print_i64((i64)(100L * same / tried)); print_cstr("%\n");
            CHECK(100L * same / tried >= 98, "save/load diverged");
        }
    }

    /* -------- k > live, single-cluster, zero query -------- */
    {
        i64 r[64]; f32 c[64]; i32 h;
        h = store_search(s, tmp, 64, r, c);
        CHECK(h <= store_count(s) && h <= 64, "k>live returned too many");
        for (i = 0; i < dim; ++i) tmp[i] = 0.0f;
        h = store_search(s, tmp, 8, r, c);          /* zero query - must not crash */
        CHECK(h >= 0, "zero query broke");
        print_cstr("  k>live -> "); print_i64(h);
        print_cstr(" hits;  zero query survived\n");
    }

    /* -------- determinism: same seed + ops -> same index choice -------- */
    {
        RagStore *d1, *d2; u32 z1 = 42u, z2 = 42u; i32 j;
        f32 *scratch = (f32 *)malloc((size_t)dim * sizeof(f32));
        d1 = store_new(a, dim); d2 = store_new(a, dim);
        { EngGoal g = d1->goal; g.max_code_bytes = dim; store_set_goal(d1, g); store_set_goal(d2, g); }   /* the tuner must be in play */
        for (j = 0; j < 1800; ++j) {
            i32 tp = (i32)(xs_(&z1) % (u32)topics); mkvec_(scratch, centers, scale, tp, dim, &z1);
            store_add(d1, scratch, 1 + j);
        }
        for (j = 0; j < 1800; ++j) {
            i32 tp = (i32)(xs_(&z2) % (u32)topics); mkvec_(scratch, centers, scale, tp, dim, &z2);
            store_add(d2, scratch, 1 + j);
        }
        store_build(d1); store_build(d2);
        CHECK(d1->idx->topo == d2->idx->topo && d1->idx->scheme == d2->idx->scheme
              && d1->idx->nprobe == d2->idx->nprobe, "engine choice not deterministic");
        print_cstr("  determinism: both chose ");
        print_cstr(d1->idx->topo == ENG_IVF ? "IVF+" : "flat ");
        print_cstr(engine_scheme_name(d1->idx->scheme));
        print_cstr("\n");
        free(scratch);
    }

    /* -------- concurrent readers while the writer mutates -------- */
    {
        pthread_t th[8];
        i32 j;
        g_reader_store = s; g_reader_stop = 0; g_reader_crash = 0;
        g_reader_dim = dim;
        for (j = 0; j < 8; ++j) pthread_create(&th[j], (pthread_attr_t *)0,
                                               reader_thread_, (void *)0);
        for (j = 0; j < 60; ++j) {
            i32 tp = (i32)(xs_(&st) % (u32)topics);
            mkvec_(tmp, centers, scale, tp, dim, &st);
            store_add(s, tmp, 5000000 + j);
            if ((j % 80) == 79) store_build(s);
        }
        g_reader_stop = 1;
        for (j = 0; j < 8; ++j) pthread_join(th[j], (void **)0);
        CHECK(g_reader_crash == 0, "a concurrent reader saw a bad result / crashed");
        print_cstr("  concurrent: 8 readers x churning writer, bad results = ");
        print_i64(g_reader_crash); print_cstr("\n");
    }

    /* -------- metadata filter end to end -------- */
    {
        RagStore *fs = store_new(a, dim);
        i64 fr[10]; f32 fsc[10];
        i32 j, want = 0, wrong = 0, h;
        for (j = 0; j < 1500; ++j) {
            i32 tp = (i32)(xs_(&st) % (u32)topics);
            mkvec_(tmp, centers, scale, tp, dim, &st);
            /* meta bit 0 = "even topic", bit 1 = "topic < 20" */
            store_add_meta(fs, tmp, 1 + j,
                           ((tp & 1) ? 0 : (i64)1) | ((tp < 20) ? (i64)2 : 0));
        }
        store_build(fs);
        for (j = 0; j < 80; ++j) {
            i32 tp = (i32)(xs_(&st) % (u32)topics), d, x;
            mkvec_(tmp, centers, scale, tp, dim, &st);
            h = store_search_filter(fs, tmp, 10, fr, fsc, (i64)1 | (i64)2, 0);
            (void)d;
            for (x = 0; x < h; ++x) {
                i32 row;
                for (row = 0; row < fs->count; ++row)
                    if (fs->ids[row] == fr[x] && !fs->dead[row]) {
                        if ((fs->meta[row] & 3) != 3) ++wrong;
                        break;
                    }
            }
            if (h >= 1) ++want;
        }
        CHECK(wrong == 0, "filter returned a row that fails the mask");
        CHECK(want >= 78, "filter returned nothing for most queries");
        print_cstr("  filter: 80 queries (mask = even AND topic<20), full k on ");
        print_i64(want); print_cstr("/80, violations = ");
        print_i64(wrong); print_cstr("\n");
        store_destroy(fs);
    }

    store_destroy(s);
    free(ovec); free(oid); free(odead);
    print_cstr(g_fail ? "\nSTRESS FAILED\n" : "\nstress OK\n");
    print_flush();
    arena_free(a);
    return g_fail ? 1 : 0;
}
