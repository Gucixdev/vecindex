/* store_tune.c - exact re-scoring, the query log, recall measurement and self-tuning.
 * Part of store.c, which includes it at this point; not compiled on its own. */

/* ------------------------------------------------------------------ *
 * search - read-locked, own scratch, engine_search_into (arena-free)  *
 * ------------------------------------------------------------------ */

static i32 store_meta_ok_(i64 tag, i64 all_bits, i64 any_bits)
{
    if ((tag & all_bits) != all_bits) return 0;
    if (any_bits && !(tag & any_bits)) return 0;
    return 1;
}

static i64 store_majflt_(void)
{
#if defined(RUSAGE_THREAD)
    struct rusage u;
    return getrusage(RUSAGE_THREAD, &u) == 0 ? (i64)u.ru_majflt : 0;
#else
    return 0;
#endif
}

/* ask the kernel for the pages of every candidate row at once (they come in parallel instead of one blocking
 * fault after another); only for rows that live in a file */
static void store_willneed_(const RagStore *s, const i32 *rows, i32 m)
{
    i32 i;
    for (i = 0; i < m; ++i) {
        i32 r = rows[i];
        if (r < 0 || r >= s->count || s->vec.how[(size_t)r >> STORE_CSHIFT] == STORE_ANON) continue;
        store_madv_(s, SCOL_AT(&s->vec, r), s->vec.esz, MADV_WILLNEED);
        if (s->vbytes == 1) store_madv_(s, SCOL_AT(&s->vsc, r), 4, MADV_WILLNEED);
    }
}

/* exact re-score of candidate rows into the (row, score) top-k */
static void store_rescore_(RagStore *s, const f32 *qn, f32 qlen, const StoreCnf *cnf, const i32 *rows, i32 m, i32 k, i64 *out_rows, f32 *out_scores, i32 *have)
{
    i32 i, pf;
    i64 f0 = 0;
#if defined(__GNUC__)
    pf = __atomic_load_n(&s->pf_cold, __ATOMIC_RELAXED);
    {   u32 t = __atomic_add_fetch(&s->pf_tick, 1u, __ATOMIC_RELAXED);
        if (pf && (t & 63u) == 0) pf = 0; }                      /* now and then try without, to notice a warm cache */
#else
    pf = s->pf_cold;
#endif
    if (pf) store_willneed_(s, rows, m);
    else if (store_vec_filed_(s)) f0 = store_majflt_();
    for (i = 0; i < m; ++i) {
        i32 row = rows[i];
        if (i + STORE_PF_AHEAD < m && rows[i + STORE_PF_AHEAD] >= 0 && rows[i + STORE_PF_AHEAD] < s->built_upto)
            store_prefetch_row_((const f32 *)(const void *)SCOL_AT(&s->vec, rows[i + STORE_PF_AHEAD]), (i32)(s->vec.esz / 4));
        if (row < 0 || row >= s->built_upto || (cnf && !store_cnf_ok_(s, cnf, row)) || store_row_bad_(s, row)) continue;   /* a damaged row is never served */
        eng_topk_(out_rows, out_scores, k, have, (i64)row, store_score_(s, qn, qlen, row));
    }
    if (!pf && store_vec_filed_(s)) {
        i64 df = store_majflt_() - f0;
        i32 want = df > 4 ? 1 : 0;                               /* faults went to the disk: prefetch from now on */
#if defined(__GNUC__)
        __atomic_store_n(&s->pf_cold, want, __ATOMIC_RELAXED);
#else
        s->pf_cold = want;
#endif
    }
}

/* ---- self-tuning: the fewest probes that keep recall@10 at the target ---- */

/* a query into the reservoir (s->qlog): try-locked, so a busy reservoir skips the query and nobody waits */
static void store_qlog_(RagStore *s, const f32 *qn)
{
    i64 slot;
    if (pthread_mutex_trylock(&s->qlog_mu) != 0) return;
    if (!s->qlog) s->qlog = (f32 *)malloc((size_t)STORE_QLOG * (size_t)s->dim * sizeof(f32));
    if (s->qlog) {
        ++s->qlog_seen;
        if (s->qlog_n < STORE_QLOG) slot = s->qlog_n++;
        else slot = (i64)(tq_rng_(&s->qlog_rng) % (u32)(s->qlog_seen < 0x7fffffff ? s->qlog_seen : 0x7fffffff));
        if (slot < STORE_QLOG) memcpy(s->qlog + (size_t)slot * s->dim, qn, (size_t)s->dim * sizeof(f32));
    }
    pthread_mutex_unlock(&s->qlog_mu);
}

/* ---- reading rows without the lock: what the live thread builds, codes and measures on ---- */

/* Rows [0, n) as they are now: the chunk pointers of the vector (and scale) columns and, if asked, the tombstones,
 * copied - adds may grow the columns meanwhile, but no row below n moves while mt_mu is held (only a compaction moves
 * rows, and it runs under mt_mu), and no chunk is ever freed before store_destroy. Taken under the read lock. */
typedef struct { i32 n; u8 **vch, **sch; u64 *dead; TqCol col, scol; } StoreSnap;

static void store_snap_(const RagStore *s, i32 n, i32 with_dead, StoreSnap *sn)
{
    i32 nch = (i32)(((i64)n + STORE_CROWS - 1) >> STORE_CSHIFT);
    size_t nw = ((size_t)n + 63) >> 6;
    memset(sn, 0, sizeof(*sn));
    sn->n = n;
    sn->vch = (u8 **)malloc((size_t)(nch ? nch : 1) * sizeof(u8 *));
    if (nch) memcpy(sn->vch, s->vec.chunk, (size_t)nch * sizeof(u8 *));
    sn->col = store_tqcol_(&s->vec); sn->col.chunk = sn->vch;
    if (s->vbytes == 1) {
        sn->sch = (u8 **)malloc((size_t)(nch ? nch : 1) * sizeof(u8 *));
        if (nch) memcpy(sn->sch, s->vsc.chunk, (size_t)nch * sizeof(u8 *));
        sn->scol = store_tqcol_(&s->vsc); sn->scol.chunk = sn->sch;
        sn->col.i8 = 1; sn->col.dim = s->dim; sn->col.scale = &sn->scol;
    }
    if (with_dead) {
        sn->dead = (u64 *)calloc(nw ? nw : 1, sizeof(u64));
        if (nw) memcpy(sn->dead, s->dead, nw * sizeof(u64));
    }
}

static void store_snap_free_(StoreSnap *sn) { free(sn->vch); free(sn->sch); free(sn->dead); memset(sn, 0, sizeof(*sn)); }

/* q . row r of a column (q unit length; int8 rows times their scale) */
static f32 store_cdot_(const TqCol *c, const f32 *q, i64 r, i32 dim)
{
    if (!c->i8) return vs_dot(q, TQ_ROW(c, r), dim);
    return vs_dot_i8(q, (const i8 *)(const void *)TQ_AT(c, r), dim) * *(const f32 *)(const void *)TQ_AT(c->scale, r);
}

/* maintenance counters, read without the lock (store_live_stats) */
static void store_stat_add_(i64 *p, i64 v)
{
#if defined(__GNUC__)
    __atomic_add_fetch(p, v, __ATOMIC_RELAXED);
#else
    *p += v;
#endif
}
static i64 store_stat_get_(const i64 *p)
{
#if defined(__GNUC__)
    return __atomic_load_n(p, __ATOMIC_RELAXED);
#else
    return *p;
#endif
}
static void store_stat_set_(i64 *p, i64 v)
{
#if defined(__GNUC__)
    __atomic_store_n(p, v, __ATOMIC_RELAXED);
#else
    *p = v;
#endif
}

/* the exact top k rows of every query among rows [0, n) of a column (tombstones skipped, a query's own row left out
 * when self[q] >= 0): the rows are read once, each thread takes a range and keeps its own top k, and the ranges are
 * merged at the end */
typedef struct { const TqCol *col; const u64 *dead; i32 dim; const f32 *Q; const i32 *self; i32 nq, k; f32 *ts; i32 *tr; } StoreTruth;

static void store_topk_put_(f32 *bs, i32 *br, i32 k, f32 d, i32 r)
{
    i32 j = k - 1;
    if (!(d > bs[k - 1])) return;
    while (j > 0 && bs[j - 1] < d) { bs[j] = bs[j - 1]; br[j] = br[j - 1]; --j; }
    bs[j] = d; br[j] = r;
}

static void store_truth_rows_(void *p, i64 a, i64 b, i32 tid)
{
    StoreTruth *j = (StoreTruth *)p;
    f32 *ts = j->ts + (size_t)tid * j->nq * j->k;
    i32 *tr = j->tr + (size_t)tid * j->nq * j->k, q;
    i64 r0, r;
    for (r0 = a; r0 < b; r0 += 256)                              /* 256 rows stay in cache across all the queries */
        for (q = 0; q < j->nq; ++q) {
            const f32 *qv = j->Q + (size_t)q * j->dim;
            for (r = r0; r < b && r < r0 + 256; ++r) {
                if (TQ_SKIP(j->dead, r) || (j->self && j->self[q] == (i32)r)) continue;
                store_topk_put_(ts + (size_t)q * j->k, tr + (size_t)q * j->k, j->k, store_cdot_(j->col, qv, r, j->dim), (i32)r);
            }
        }
}

static void store_truth_(const TqCol *col, const u64 *dead, i32 n, i32 dim, const f32 *Q, const i32 *self, i32 nq, i32 k, i32 *truth, i32 nth)
{
    StoreTruth j;
    i32 t, q, x;
    size_t per = (size_t)nq * (size_t)k;
    if (nth > 64) nth = 64;
    j.col = col; j.dead = dead; j.dim = dim; j.Q = Q; j.self = self; j.nq = nq; j.k = k;
    j.ts = (f32 *)malloc((size_t)nth * per * sizeof(f32)); j.tr = (i32 *)malloc((size_t)nth * per * sizeof(i32));
    for (x = 0; x < (i32)((size_t)nth * per); ++x) { j.ts[x] = -1e30f; j.tr[x] = -1; }
    tq_par_(store_truth_rows_, &j, n, nth);
    for (q = 0; q < nq; ++q)                                     /* fold every thread's top k into thread 0's */
        for (t = 1; t < nth; ++t)
            for (x = 0; x < k; ++x)
                if (j.tr[(size_t)t * per + (size_t)q * k + x] >= 0)
                    store_topk_put_(j.ts + (size_t)q * k, j.tr + (size_t)q * k, k, j.ts[(size_t)t * per + (size_t)q * k + x], j.tr[(size_t)t * per + (size_t)q * k + x]);
    memcpy(truth, j.tr, per * sizeof(i32));
    free(j.ts); free(j.tr);
}

/* recall@10 of the index at `probe` clusters against the exact truth (the batch path, re-scored exactly as a search) */
static f32 store_recall_at_(RagStore *s, const f32 *Q, const i32 *self, i32 nq, const i32 *truth, i32 probe, i32 nth)
{
    i32 R = s->tq_rerank > 0 ? s->tq_rerank : 256, q, x, z, hits = 0;
    i32 *rows = (i32 *)malloc((size_t)nq * (size_t)R * sizeof(i32)), *cnt = (i32 *)malloc((size_t)nq * sizeof(i32));
    tq_search_many(s->tq, Q, nq, R, rows, cnt, s->dead, nth, probe);
    for (q = 0; q < nq; ++q) {
        i64 got[11];
        f32 sc[11];
        i32 have = 0, kept = 0;
        store_rescore_(s, Q + (size_t)q * s->dim, 1.0f, (const StoreCnf *)0, rows + (size_t)q * R, cnt[q], 11, got, sc, &have);
        for (x = 0; x < have && kept < 10; ++x) {
            if (self && got[x] == self[q]) continue;             /* a row asking for itself finds itself: not a hit */
            ++kept;
            for (z = 0; z < 10; ++z) if (truth[(size_t)q * 10 + z] == (i32)got[x]) { ++hits; break; }
        }
    }
    free(rows); free(cnt);
    return (f32)hits / (f32)(nq * 10);
}

/* the same under the read lock (the caller holds none): one measurement is all a search ever waits for */
static f32 store_recall_rd_(RagStore *s, const f32 *Q, const i32 *self, i32 nq, const i32 *truth, i32 probe, i32 nth)
{
    f32 r;
    pthread_rwlock_rdlock(&s->lock);
    r = store_recall_at_(s, Q, self, nq, truth, probe, nth);
    pthread_rwlock_unlock(&s->lock);
    return r;
}

/* The queries to tune on, into Q[nq x dim]: the logged traffic when 64 or more were kept (*self = NULL), else a
 * spread of the store's own live rows below n (*self[q] = that row, left out of its own truth). Returns how many. */
static i32 store_tune_queries_(RagStore *s, const TqCol *col, const u64 *dead, i32 n, f32 *Q, i32 nq, i32 **self, i32 *from_log)
{
    i32 q;
    *self = (i32 *)0;
    pthread_mutex_lock(&s->qlog_mu);
    *from_log = s->qlog_n >= 64;
    if (*from_log) {                                             /* real traffic: the queries this store answers */
        if (nq > s->qlog_n) nq = s->qlog_n;
        memcpy(Q, s->qlog, (size_t)nq * (size_t)s->dim * sizeof(f32));
    }
    pthread_mutex_unlock(&s->qlog_mu);
    if (*from_log) return nq;
    *self = (i32 *)malloc((size_t)nq * sizeof(i32));             /* no traffic yet: rows of its own, spread out */
    for (q = 0; q < nq; ++q) {
        i32 r = (i32)((i64)n * q / nq) + q % 7;
        f32 *dst = Q + (size_t)q * s->dim;
        const f32 *v;
        if (r >= n) r = n - 1;
        while (r < n - 1 && TQ_SKIP(dead, r)) ++r;
        v = tq_row_(col, r, dst);
        if (v != dst) memcpy(dst, v, (size_t)s->dim * sizeof(f32));
        (*self)[q] = r;
    }
    return nq;
}

/* The fewest IVF probes whose recall@10 reaches s->tune_target (plus a margin for the sample's own error), measured
 * against exact search on the queries the store has been answering (when 64 or more were kept) or else on a spread of
 * its own rows (their own row left out of the truth). Up to STORE_QLOG queries, fewer on huge stores (the exact search
 * reads every row once per query: about 3e10 multiply-adds at most, ~3 s on 2 cores). The probe count becomes the
 * index's default (saved with it). Returns the recall reached, -1 with no IVF index to tune.
 * Called with mt_mu held and no lock: the exact truth runs on a snapshot with no lock at all, each recall measurement
 * under the read lock, and the result goes in under the write lock - searches go on throughout. nth: threads. */
static f32 store_tune_run_(RagStore *s, i32 nth)
{
    i32 nq, nc, n, lo, hi, p, *self = (i32 *)0, *truth, from_log = 0, np0;
    f32 *Q, rec, rec_hi, goal, target;
    f64 budget;
    StoreSnap sn;
    Tq *t;
    pthread_rwlock_rdlock(&s->lock);
    t = s->tq; n = s->built_upto; target = s->tune_target;
    if (!t || t->nc <= 1 || target <= 0.0f || n < 1000 || t->n != n) { pthread_rwlock_unlock(&s->lock); return -1.0f; }
    nc = t->nc; np0 = t->nprobe;
    store_snap_(s, n, 1, &sn);
    pthread_rwlock_unlock(&s->lock);
    budget = 3e10 / ((f64)n * (f64)s->dim);
    nq = budget > STORE_QLOG ? STORE_QLOG : (budget < 32.0 ? 32 : (i32)budget);
    Q = (f32 *)malloc((size_t)nq * (size_t)s->dim * sizeof(f32));
    nq = store_tune_queries_(s, &sn.col, sn.dead, n, Q, nq, &self, &from_log);
    truth = (i32 *)malloc((size_t)nq * 10 * sizeof(i32));
    store_truth_(&sn.col, sn.dead, n, s->dim, Q, self, nq, 10, truth, nth);
    /* aim 1.5 standard errors above the target, so fresh queries meet it too, not just half of the time */
    goal = target + 1.5f * (f32)sqrt((f64)target * (1.0 - (f64)target) / (10.0 * (f64)nq));
    if (goal > 0.999f) goal = 0.999f;
    /* recall grows with the probes: double from the default until the target is met, then halve the gap */
    hi = np0 > 0 ? np0 : 1;
    rec_hi = store_recall_rd_(s, Q, self, nq, truth, hi, nth);
    lo = 0;
    while (rec_hi < goal && hi < nc) { lo = hi; hi = hi * 2 < nc ? hi * 2 : nc; rec_hi = store_recall_rd_(s, Q, self, nq, truth, hi, nth); }
    if (rec_hi >= goal)
        while (hi - lo > 1) {                                    /* the goal is met at hi and not at lo */
            p = lo + (hi - lo) / 2;
            rec = store_recall_rd_(s, Q, self, nq, truth, p, nth);
            if (rec >= goal) { hi = p; rec_hi = rec; } else lo = p;
        }
    pthread_rwlock_wrlock(&s->lock);
    if (s->tq == t) {                                            /* the index's default from now on (and on disk) */
        t->nprobe = hi; s->tq_probe = 0; s->tune_recall = rec_hi; s->tuned_n = t->n; s->tuned_log = from_log;
    }
    pthread_rwlock_unlock(&s->lock);
    store_snap_free_(&sn);
    free(Q); free(self); free(truth);
    return rec_hi;
}

void store_set_target_recall(RagStore *s, f32 target)
{
    pthread_rwlock_wrlock(&s->lock);
    s->tune_target = target > 0.0f && target < 1.0f ? target : 0.0f;
    pthread_rwlock_unlock(&s->lock);
}

/* tune now (store_build tuned on what it had): returns the recall@10 reached, -1 without an IVF index or a target.
 * Searches go on meanwhile (a few seconds on a million rows). */
f32 store_tune(RagStore *s)
{
    f32 r;
    pthread_mutex_lock(&s->mt_mu);
    r = store_tune_run_(s, store_ncpu_(s));
    pthread_mutex_unlock(&s->mt_mu);
    return r;
}

f32 store_tuned_recall(RagStore *s) { return s->tune_recall; }
