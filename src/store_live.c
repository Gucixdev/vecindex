/* store_live.c - live mode: rows indexed as they come, the index rebuilt as it ages.
 * Part of store.c, which includes it at this point; not compiled on its own. */

/* ---- live mode: rows indexed as they come, the index rebuilt as it ages ---- */

#ifndef STORE_LIVE_CHUNK
#define STORE_LIVE_CHUNK 4096   /* rows coded per step of live indexing */
#endif
#ifndef STORE_LIVE_MIN
#define STORE_LIVE_MIN 1024     /* rows before a live store builds its first index */
#endif
#ifndef STORE_LIVE_DRIFT
#define STORE_LIVE_DRIFT 1.25   /* rebuild once added rows sit this much farther from their centroids than built ones */
#endif
#ifndef STORE_LIVE_SKEW
#define STORE_LIVE_SKEW 1.2     /* ... or once the cluster sizes are this much more uneven than at the build (tq_skew) */
#endif

/* How well the rows going in fit the clusters, a window of max(1000, a twentieth of the build) rows at a time: a
 * window whose rows sit STORE_LIVE_DRIFT times farther from their centroids than the built rows did means the data
 * has moved away from the clusters (one far window is enough; the old rows added since cannot water it down). Under
 * the write lock, mt_mu held. */
static void store_fit_(RagStore *s, const Tq *t, const TqAdd *ad, i32 m)
{
    i32 i, win = t->nbuilt / 20 > 1000 ? t->nbuilt / 20 : 1000;
    if (t->nc <= 1 || !(t->fit0 > 1e-6f)) return;
    for (i = 0; i < m; ++i) {
        s->win_sum += (f64)ad->d2[i];
        if (++s->win_n >= win) {
            if (s->win_sum > STORE_LIVE_DRIFT * (f64)t->fit0 * (f64)s->win_n) ++s->drifted;
            s->win_sum = 0.0; s->win_n = 0;
        }
    }
}

/* Index the pending tail, STORE_LIVE_CHUNK rows at a time: coded on a snapshot with no lock held (searches and adds go
 * on), then placed under the write lock (a few stores a row). mt_mu held. Returns the rows indexed. */
static i32 store_index_pending_(RagStore *s, i32 nth)
{
    i32 done = 0;
    for (;;) {
        StoreSnap sn;
        TqAdd ad;
        Tq *t;
        i32 r0, m, ok;
        pthread_rwlock_rdlock(&s->lock);
        t = s->tq; r0 = s->built_upto; m = s->count - r0;
        if (!t || s->idx || m <= 0 || t->n != r0) { pthread_rwlock_unlock(&s->lock); break; }
        if (m > STORE_LIVE_CHUNK) m = STORE_LIVE_CHUNK;
        store_snap_(s, r0 + m, 0, &sn);
        pthread_rwlock_unlock(&s->lock);
        if (t->nc > 1 && !(t->fit0 > 0.0f)) tq_fit0(t, &sn.col);   /* a loaded index: what its rows fit like */
        ok = tq_add_prep(t, &sn.col, r0, m, nth, &ad) == 0;
        store_snap_free_(&sn);
        if (!ok) break;
        pthread_rwlock_wrlock(&s->lock);
        if (s->tq == t && s->built_upto == r0) {
            ok = tq_add_put(t, &ad, r0) == 0;
            store_fit_(s, t, &ad, t->n - r0);
            done += t->n - r0;
            s->built_upto = t->n;                                /* what went in: all of it, unless memory ran out */
        } else ok = 0;
        pthread_rwlock_unlock(&s->lock);
        tq_add_free(&ad);
        if (!ok) break;
    }
    store_stat_add_(&s->n_indexed, done);
    return done;
}

/* the same with the write lock held: store_save indexes what came in after its store_index_pending_ */
static void store_index_locked_(RagStore *s)
{
    TqCol sc, col;
    TqAdd ad;
    i32 r0 = s->built_upto, m = s->count - r0;
    if (!s->tq || s->idx || m <= 0 || s->tq->n != r0) return;
    col = store_vcol_(s, &sc);
    if (s->tq->nc > 1 && !(s->tq->fit0 > 0.0f)) tq_fit0(s->tq, &col);
    if (tq_add_prep(s->tq, &col, r0, m, store_ncpu_(s), &ad) != 0) return;
    tq_add_put(s->tq, &ad, r0);
    store_fit_(s, s->tq, &ad, s->tq->n - r0);
    store_stat_add_(&s->n_indexed, s->tq->n - r0);
    s->built_upto = s->tq->n;
    tq_add_free(&ad);
}

#ifndef STORE_SPLIT_X
#define STORE_SPLIT_X 2.0       /* a cluster is split once it holds this many times the store's cluster size */
#endif
#ifndef STORE_SPLIT_STEP
#define STORE_SPLIT_STEP 4      /* clusters split per maintenance step (each holds the write lock a millisecond or two) */
#endif

/* The clusters that outgrew STORE_SPLIT_X times the store's cluster size (its rows over store_nclusters_), split in
 * place (tq_split_big), STORE_SPLIT_STEP at most. Under the write lock, mt_mu held. Returns how many. */
static i32 store_split_(RagStore *s)
{
    Tq *t = s->tq;
    i32 nc = store_nclusters_(s->live), lim, done;
    TqCol sc, col;
    if (!s->split_on || !t || t->nc <= 1 || s->idx || nc <= 1 || t->n != s->built_upto) return 0;
    lim = (i32)(STORE_SPLIT_X * (f64)s->live / (f64)nc);
    col = store_vcol_(s, &sc);
    done = tq_split_big(t, &col, s->dead, lim, STORE_SPLIT_STEP);
    if (done) { s->tuned_n = 0; store_stat_add_(&s->n_splits, done); }   /* re-tune on the new clusters */
    return done;
}

/* Rebuild the scan index over every row there is, without stopping anything: the clustering and the codes are
 * computed on a snapshot of the rows (tombstones left out) while searches go on with the old index and the live thread
 * keeps adding new rows to it; then the new index catches up on those rows privately and is swapped in under the
 * write lock once only a few are left - so the exact-scan tail does not grow while a rebuild runs. Holds rb_mu
 * throughout (no compaction may move rows under it), mt_mu only for the swap. The caller set s->rebuilding; it is
 * cleared here. 0, or -1 when there was nothing to build. */
static i32 store_rebuild_(RagStore *s, i32 nth)
{
    StoreSnap sn;
    Tq *nt, *old;
    i32 n, nc, live, dead, bits, round, stuck = 0, unpin, cas;
    u32 seed;
    f64 t0 = tq_now_();
    pthread_mutex_lock(&s->rb_mu);
    pthread_rwlock_rdlock(&s->lock);
    n = s->count; live = s->live; dead = n - live; bits = s->codebits; seed = s->goal.seed; cas = s->cascade;
    if (live <= 0 || s->idx) {
        pthread_rwlock_unlock(&s->lock); pthread_mutex_unlock(&s->rb_mu); store_stat_set_(&s->rebuilding, 0); return -1;
    }
    store_snap_(s, n, 1, &sn);
    store_stat_set_(&s->rebuilt_mods, s->mods);
    pthread_rwlock_unlock(&s->lock);
    nc = store_nclusters_(live);
    nt = tq_build_skip(&sn.col, n, s->dim, seed, nc, nth, bits, nc > 1 ? sn.dead : (const u64 *)0);
    store_snap_free_(&sn);
    if (nt && cas && tq_cascade_on(nt, nth) != 0) { tq_free(nt); nt = (Tq *)0; }   /* rows added later get theirs in tq_add_put */
    if (!nt) { pthread_mutex_unlock(&s->rb_mu); store_stat_set_(&s->rebuilding, 0); return -1; }
    for (round = 0; round < 8 && !stuck; ++round) {             /* the rows added meanwhile, into the private index */
        i32 c;
        pthread_rwlock_rdlock(&s->lock);
        c = s->count;
        if (c - nt->n > 256) store_snap_(s, c, 0, &sn);
        pthread_rwlock_unlock(&s->lock);
        if (c - nt->n <= 256) break;
        while (nt->n < c && !stuck) {
            TqAdd ad;
            i32 m = c - nt->n < STORE_LIVE_CHUNK ? c - nt->n : STORE_LIVE_CHUNK;
            if (tq_add_prep(nt, &sn.col, nt->n, m, nth, &ad) != 0) { stuck = 1; break; }
            if (tq_add_put(nt, &ad, nt->n) != 0) stuck = 1;
            tq_add_free(&ad);
        }
        store_snap_free_(&sn);
    }
    pthread_mutex_lock(&s->mt_mu);                               /* the live thread, between two chunks */
    pthread_rwlock_wrlock(&s->lock);
    if (!stuck && s->count > nt->n) {                            /* the last few, under the lock (at most a chunk) */
        TqCol sc, col = store_vcol_(s, &sc);
        TqAdd ad;
        i32 m = s->count - nt->n < STORE_LIVE_CHUNK ? s->count - nt->n : STORE_LIVE_CHUNK;
        if (tq_add_prep(nt, &col, nt->n, m, nth, &ad) == 0) { tq_add_put(nt, &ad, nt->n); tq_add_free(&ad); }
    }
    old = s->tq;
    if (old && old->nc > 1 && nt->nc > 1 && s->tuned_n > 0) {   /* the tuned share of the clusters, until re-tuned */
        i32 np = (i32)((f64)old->nprobe * (f64)nt->nc / (f64)old->nc + 0.5);
        nt->nprobe = np < 1 ? 1 : (np > nt->nc ? nt->nc : np);
    }
    if (nt->codes1 && !s->cascade) tq_cascade_off(nt);         /* turned off while this was building */
    store_ex_reap_(s);
    s->tq = nt;
    s->built_upto = nt->n;                                       /* rows beyond it (if any) pend, as usual */
    s->tuned_n = 0;
    s->dead_built = nt->nc > 1 ? dead : 0;                       /* a flat index keeps its tombstones' slots */
    s->win_sum = 0.0; s->win_n = 0; s->drifted = 0;
    unpin = s->mem_tier >= 1;
    pthread_rwlock_unlock(&s->lock);
    pthread_mutex_unlock(&s->mt_mu);
    if (unpin) store_tq_pin_(s, old, 0, 0);                      /* a loaded index's locked pages go back */
    tq_free(old);
    pthread_mutex_unlock(&s->rb_mu);
    store_stat_add_(&s->n_rebuilds, 1);
    store_stat_set_(&s->rebuild_us, (i64)((tq_now_() - t0) * 1e6));
    store_stat_set_(&s->rebuilding, 0);
    return 0;
}

/* Why the index should be rebuilt now, 0 if it should not: 1 there is none yet, 2 the store grew into IVF size, 3 it
 * grew by live_grow since the build, 4 a window of added rows sat STORE_LIVE_DRIFT times farther from their centroids
 * than the built rows (store_fit_: the data moved), 5 a quarter of the index is tombstones, 6 the added rows piled into
 * a few clusters (tq_skew up by STORE_LIVE_SKEW: every query near them scans those big clusters; not with splits, which
 * cut such clusters in place), 7 splits left a quarter of the slots as holes. With splits on, growth (3) rebuilds only
 * at 8 times live_grow. Nothing changed since the last rebuild: 0. Under the read lock. */
static i32 store_due_(const RagStore *s)
{
    const Tq *t = s->tq;
    i32 dead = s->count - s->live - s->dead_built;
    if (s->idx || s->mods == store_stat_get_(&s->rebuilt_mods)) return 0;   /* an engine index: store_build's business */
    if (!t) return s->live >= STORE_LIVE_MIN ? 1 : 0;
    if (s->live_grow <= 0.0f) return 0;
    if (t->nc <= 1) return store_nclusters_(s->live) > 1 ? 2 : 0;
    /* with splits the clusters keep their size as rows come, so only a far larger growth rebuilds */
    if ((f64)t->nadd >= (f64)s->live_grow * (s->split_on ? 8.0 : 1.0) * (f64)(t->nbuilt > 0 ? t->nbuilt : 1)) return 3;
    if (s->drifted) return 4;
    if ((i64)dead * 4 > (i64)t->n) return 5;
    if (!s->split_on && t->csize && t->skew0 > 0.0 && tq_skew(t) > STORE_LIVE_SKEW * t->skew0) return 6;
    if ((i64)t->holes * 4 > (i64)t->n) return 7;                 /* splits left a quarter of the slots empty */
    return 0;
}

/* re-tune: a target is set and the index was rebuilt, grew by a tenth, or real traffic arrived to tune on */
static i32 store_tune_due_(RagStore *s)
{
    const Tq *t = s->tq;
    i32 nlog;
    if (s->tune_target <= 0.0f || !t || t->nc <= 1 || s->built_upto < 1000 || t->n != s->built_upto) return 0;
    if (s->tuned_n <= 0 || (i64)t->n * 10 >= (i64)s->tuned_n * 11) return 1;
    pthread_mutex_lock(&s->qlog_mu); nlog = s->qlog_n; pthread_mutex_unlock(&s->qlog_mu);
    return !s->tuned_log && nlog >= 64;
}

/* One maintenance step: index the pending rows, rebuild if the index has aged (store_due_) - async: on a thread of
 * its own, so this one goes on indexing meanwhile; else right here - and re-tune if a target recall is set and the
 * index changed. Returns what it did: 1 indexed, 2 rebuilt (or started to), 4 tuned. */
static void *store_rb_main_(void *p)
{
    RagStore *s = (RagStore *)p;
    store_bg_prio_(s);
    store_rebuild_(s, store_bg_threads_(s));
    return (void *)0;
}

static i32 store_step_(RagStore *s, i32 async)
{
    i32 did = 0, tune = 0, go = 0, nth = store_bg_threads_(s);
    pthread_mutex_lock(&s->mt_mu);
    if (store_index_pending_(s, nth) > 0) did |= 1;
    if (s->split_on && !store_stat_get_(&s->rebuilding)) {     /* not while a rebuild may swap the index */
        pthread_rwlock_wrlock(&s->lock);
        if (store_split_(s) > 0) did |= 8;
        pthread_rwlock_unlock(&s->lock);
    }
    pthread_rwlock_rdlock(&s->lock);
    if (!store_stat_get_(&s->rebuilding)) {
        if (store_due_(s)) { store_stat_set_(&s->rebuilding, 1); go = 1; }
        else tune = store_tune_due_(s);                          /* not while the index is about to be replaced */
    }
    pthread_rwlock_unlock(&s->lock);
    if (tune && store_tune_run_(s, nth) >= 0.0f) did |= 4;
    pthread_mutex_unlock(&s->mt_mu);
    if (go) {
        did |= 2;
        if (async && s->rb_live) { pthread_join(s->rb_th, (void **)0); s->rb_live = 0; }   /* the last one has finished */
        if (!async || pthread_create(&s->rb_th, (const pthread_attr_t *)0, store_rb_main_, s) != 0) store_rebuild_(s, nth);
        else s->rb_live = 1;
    }
    return did;
}

/* One maintenance step, now, by hand (what the live thread runs again and again, here with the rebuild in this
 * thread): index the pending rows, split clusters that outgrew their size, rebuild if the index has aged, re-tune if a
 * target recall is set and the index changed. Searches go on throughout (the work takes half the cores); an add waits
 * at most while a chunk of rows is placed, a few clusters split or an index swapped. Returns what it did: 1 indexed,
 * 2 rebuilt, 4 tuned, 8 split. */
i32 store_maintain(RagStore *s) { return store_step_(s, 0); }

static void *store_live_main_(void *p)
{
    RagStore *s = (RagStore *)p;
    store_bg_prio_(s);                                           /* searches and adds come first */
    for (;;) {
        i32 did, stop;
        pthread_mutex_lock(&s->mt_cmu); stop = s->mt_stop; pthread_mutex_unlock(&s->mt_cmu);
        if (stop) break;
        did = store_step_(s, 1);
        if (s->rb_live && !store_stat_get_(&s->rebuilding)) { pthread_join(s->rb_th, (void **)0); s->rb_live = 0; }
        pthread_mutex_lock(&s->mt_cmu);
        if (!did && !s->mt_stop) {                               /* nothing to do: sleep until an add, at most 50 ms */
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 50000000L;
            if (ts.tv_nsec >= 1000000000L) { ts.tv_sec += 1; ts.tv_nsec -= 1000000000L; }
            pthread_cond_timedwait(&s->mt_cv, &s->mt_cmu, &ts);
        }
        pthread_mutex_unlock(&s->mt_cmu);
    }
    if (s->rb_live) { pthread_join(s->rb_th, (void **)0); s->rb_live = 0; }   /* a rebuild in flight finishes first */
    return (void *)0;
}

static void store_live_stop_(RagStore *s)
{
    if (!s->live_on) return;
    pthread_mutex_lock(&s->mt_cmu); s->mt_stop = 1; pthread_cond_signal(&s->mt_cv); pthread_mutex_unlock(&s->mt_cmu);
    pthread_join(s->mt, (void **)0);
    pthread_rwlock_wrlock(&s->lock); s->live_on = 0; pthread_rwlock_unlock(&s->lock);
}

/* Live mode on or off. On: a thread of the store's own indexes rows as they are added - within ~50 ms; until then the
 * exact scan of the pending tail finds them, as always - and rebuilds the index off the lock as it ages: grown by
 * `grow` since the last build (1.0: doubled; 0: never rebuild), the added rows sitting STORE_LIVE_DRIFT times farther
 * from their centroids than the built ones (the data moved), or a quarter of it tombstones. With a target recall
 * (store_set_target_recall) it re-tunes the probes after every rebuild and every tenth more rows. A store with no
 * index yet gets one at STORE_LIVE_MIN rows. */
void store_set_live(RagStore *s, i32 on, f32 grow)
{
    pthread_mutex_lock(&s->mt_mu);
    s->live_grow = grow > 0.0f ? grow : 0.0f;
    pthread_mutex_unlock(&s->mt_mu);
    if (on && !s->live_on) {
        s->mt_stop = 0;
        if (pthread_create(&s->mt, (const pthread_attr_t *)0, store_live_main_, s) == 0) {
            pthread_rwlock_wrlock(&s->lock); s->live_on = 1; pthread_rwlock_unlock(&s->lock);
            pthread_cond_signal(&s->mt_cv);
        }
    } else if (!on) store_live_stop_(s);
}

/* splits on (default) or off: whether live mode cuts outgrown clusters in place (store_split_) or rebuilds instead */
void store_set_split(RagStore *s, i32 on)
{
    pthread_mutex_lock(&s->mt_mu);
    s->split_on = on ? 1 : 0;
    pthread_mutex_unlock(&s->mt_mu);
}

/* query-aware probe pruning (tq_prune_), for every store of the process: eps 0 off */
void store_set_prune(f32 eps) { tq_prune_eps_ = eps > 0.0f ? eps : 0.0f; }

/* clusters split in place so far */
i64 store_splits(RagStore *s) { return store_stat_get_(&s->n_splits); }

/* what maintenance has done: rows indexed after a build, rebuilds, seconds the last rebuild took (any may be NULL) */
void store_live_stats(RagStore *s, i64 *indexed, i64 *rebuilds, f64 *last_rebuild_s)
{
    if (indexed) *indexed = store_stat_get_(&s->n_indexed);
    if (rebuilds) *rebuilds = store_stat_get_(&s->n_rebuilds);
    if (last_rebuild_s) *last_rebuild_s = 1e-6 * (f64)store_stat_get_(&s->rebuild_us);
}
