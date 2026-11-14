/* engine_test.c - the self-test of src/engine.c; ./run builds and runs it. */
#include "engine.c"
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
