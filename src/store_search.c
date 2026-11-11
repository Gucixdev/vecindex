/* store_search.c - search: one query, a filtered one, a batch.
 * Part of store.c, which includes it at this point; not compiled on its own. */

#define STORE_CAS_SHORT 8   /* the cascade's shortlist: this many times the rows the exact re-score reads */

/* the scan's R best rows: through the cascade when the index has one (1-bit codes first, a shortlist of
 * STORE_CAS_SHORT x R re-scored with the 4-bit codes) */
static i32 store_scan_(const RagStore *s, const f32 *qn, i32 R, i32 *rows, const TqFilt *f, i32 nth, i32 probe)
{
    if (s->tq->codes1) return tq_search_cascade(s->tq, qn, STORE_CAS_SHORT * R, R, rows, s->dead, f, nth, probe);
    return tq_search(s->tq, qn, R, rows, s->dead, f, nth, probe);
}

static i32 store_search_impl_(RagStore *s, const f32 *query, i32 k,
                              i64 *out_doc_ids, f32 *out_scores,
                              i64 all_bits, i64 any_bits, int filtered, const StoreCnf *cnf)
{
    i32 dim = s->dim, over = filtered ? (k * 4 + 16) : k;
    f32 *qn  = (f32 *)malloc((size_t)dim * sizeof(f32));
    i64 *rid = (i64 *)malloc((size_t)over * sizeof(i64));
    f32 *rsc = (f32 *)malloc((size_t)over * sizeof(f32));
    EngScratch sc;
    f32 *slut = (f32 *)0;
    i32 have = 0, i, got, tail;
    TqCol mcol, ncol, scol;
    TqFilt tf;
    i32 exact;
    f32 qlen;

    memcpy(qn, query, (size_t)dim * sizeof(f32));
    qlen = (f32)eng_sqrt_(eng_ssq_(qn, dim));
    store_normalise_(qn, dim);

    pthread_rwlock_rdlock(&s->lock);
    if (s->tune_target > 0.0f) store_qlog_(s, qn);
    mcol = store_tqcol_(&s->metac);
    memset(&tf, 0, sizeof(tf));
    if (filtered) { tf.meta = &mcol; tf.all = all_bits; tf.any = any_bits; }
    if (cnf && cnf->num && s->has_num) { ncol = store_tqcol_(&s->numc); tf.num = &ncol; tf.lo = cnf->lo; tf.hi = cnf->hi; }
    if (cnf && cnf->ns_on && s->has_ns) { scol = store_tqcol_(&s->nsc); tf.ns = &scol; tf.nsv = cnf->ns; }

    exact = store_ns_exact_(s, cnf, qn, qlen, k, out_doc_ids, out_scores, &have);   /* a small namespace: its own rows */
    if (exact) {}
    else if (cnf && cnf->num && !s->has_num && (cnf->lo > 0 || cnf->hi < 0)) {}   /* no numbers yet: every row's is 0 */
    else if (s->tq) {
        /* candidates re-scored exactly: 20 k, at least 256 - four times that with 1-bit codes, which rank coarsely */
        i32 R = filtered ? k * 40 + 256
                         : (s->tq_rerank > 0 ? s->tq_rerank : (k * 20 > 256 ? k * 20 : 256) * (s->tq->bits == 1 ? 4 : 1) * (s->metric ? 4 : 1)), m, nth = 1;
        i32 *rows = (i32 *)malloc((size_t)R * sizeof(i32));
        if (s->tq_threads > 0) nth = s->tq_threads;
        else if (s->tq->n >= 150000) { i32 c = store_ncpu_(s); nth = c > 4 ? 4 : c; }   /* the scan is memory-bound: 4 threads already saturate the bus */
        m = store_scan_(s, qn, R, rows, &tf, nth, s->tq_probe);
        store_rescore_(s, qn, qlen, cnf, rows, m, k, out_doc_ids, out_scores, &have);
        /* a selective filter can leave the probed clusters short of k matches: then scan every cluster's codes
         * (RAM, 4 bits a row) for matches, never every f32 row */
        if (filtered && have < k && s->tq->nc > 1) {
            i32 j, z, n2 = store_scan_(s, qn, R, rows, &tf, nth, s->tq->nc);
            for (j = z = 0; j < n2; ++j) {
                i32 x, dup = 0;
                for (x = 0; x < have; ++x) if (out_doc_ids[x] == rows[j]) { dup = 1; break; }
                if (!dup) rows[z++] = rows[j];
            }
            store_rescore_(s, qn, qlen, cnf, rows, z, k, out_doc_ids, out_scores, &have);
        }
        free(rows);
    } else if (s->idx) {
        i32 sf = engine_scratch_floats(s->idx);
        slut = (f32 *)malloc((size_t)sf * sizeof(f32));
        sc.residual = slut;
        sc.lut      = slut + dim;
        sc.lids     = (i64 *)malloc((size_t)over * sizeof(i64));
        sc.lsc      = (f32 *)malloc((size_t)over * sizeof(f32));
        sc.probe    = (i32 *)malloc(64 * sizeof(i32));
        got = engine_search_into(s->idx, qn, over, rid, rsc, &sc);
        for (i = 0; i < got; ++i) {
            i32 row = (i32)rid[i];
            if (row < 0 || row >= s->built_upto || STORE_DEAD(s, row)) continue;
            if (filtered && !store_meta_ok_(STORE_META(s, row), all_bits, any_bits)) continue;
            if (cnf && !store_cnf_ok_(s, cnf, row)) continue;
            eng_topk_(out_doc_ids, out_scores, k, &have, (i64)row, s->metric ? store_score_(s, qn, qlen, row) : rsc[i]);
        }
        free(sc.lids); free(sc.lsc); free(sc.probe);
        /* engine path: a very selective filter can exclude the whole neighbourhood - exact filtered scan */
        if (filtered && have < k) {
            for (i = 0; i < s->built_upto; ++i) {
                i32 j, dup = 0;
                if (STORE_DEAD(s, i) || !store_meta_ok_(STORE_META(s, i), all_bits, any_bits)) continue;
                if (cnf && !store_cnf_ok_(s, cnf, i)) continue;
                for (j = 0; j < have; ++j) if (out_doc_ids[j] == i) { dup = 1; break; }
                if (dup || store_row_bad_(s, i)) continue;
                eng_topk_(out_doc_ids, out_scores, k, &have, (i64)i, store_score_(s, qn, qlen, i));
            }
        }
    }

    tail = exact || (cnf && cnf->num && !s->has_num && (cnf->lo > 0 || cnf->hi < 0)) ? 0 : s->count;
    for (i = s->built_upto; i < tail; ++i) {                    /* the pending tail: exact */
        if (STORE_DEAD(s, i)) continue;
        if (filtered && !store_meta_ok_(STORE_META(s, i), all_bits, any_bits)) continue;
        if (cnf && !store_cnf_ok_(s, cnf, i)) continue;
        if (store_row_bad_(s, i)) continue;
        eng_topk_(out_doc_ids, out_scores, k, &have, (i64)i, store_score_(s, qn, qlen, i));
    }
    for (i = 0; i < have; ++i) out_doc_ids[i] = STORE_ID(s, out_doc_ids[i]);   /* rows -> doc ids, only for the k kept */

    pthread_rwlock_unlock(&s->lock);
    free(qn); free(rid); free(rsc);
    if (slut) free(slut);
    return have;
}

/* A query with a NaN or an infinity has no direction: no hits. vserv's line protocol passed one on ("hybrid 3 0.1 -1
 * 0.1 nan"), and its NaNs reached tq_lut_'s conversion to int (tests/fuzz_proto.c); the C API refuses them sooner. */
i32 store_search(RagStore *s, const f32 *query, i32 k,
                 i64 *out_doc_ids, f32 *out_scores)
{
    if (!store_finite_(query, s->dim)) return 0;
    return store_search_impl_(s, query, k, out_doc_ids, out_scores, 0, 0, 0, (const StoreCnf *)0);
}

i32 store_search_filter(RagStore *s, const f32 *query, i32 k,
                        i64 *out_doc_ids, f32 *out_scores,
                        i64 all_bits, i64 any_bits)
{
    if (!store_finite_(query, s->dim)) return 0;
    return store_search_impl_(s, query, k, out_doc_ids, out_scores, all_bits, any_bits, 1, (const StoreCnf *)0);
}

/* ---- many queries at once ---- */
typedef struct { RagStore *s; const f32 *qn, *qlen; i32 k, R; const i32 *rows, *cnt; i64 *ids; f32 *sc; } StoreBatch;

static void store_batch_fin_(void *p, i64 a, i64 b, i32 tid)
{
    StoreBatch *j = (StoreBatch *)p;
    RagStore *s = j->s;
    i64 q;
    (void)tid;
    for (q = a; q < b; ++q) {
        const f32 *qn = j->qn + (size_t)q * s->dim;
        i64 *ids = j->ids + (size_t)q * j->k;
        f32 *sc = j->sc + (size_t)q * j->k;
        i32 have = 0, i;
        store_rescore_(s, qn, j->qlen[q], (const StoreCnf *)0, j->rows + (size_t)q * j->R, j->cnt[q], j->k, ids, sc, &have);
        for (i = s->built_upto; i < s->count; ++i)               /* the pending tail: exact */
            if (!STORE_DEAD(s, i) && !store_row_bad_(s, i)) eng_topk_(ids, sc, j->k, &have, (i64)i, store_score_(s, qn, j->qlen[q], i));
        for (i = 0; i < have; ++i) ids[i] = STORE_ID(s, ids[i]);
        for (; i < j->k; ++i) { ids[i] = -1; sc[i] = -4.0f; }
    }
}

typedef struct { RagStore *s; const f32 *Q; i32 k; i64 *ids; f32 *sc; } StoreEach;
static void store_each_(void *p, i64 a, i64 b, i32 tid)
{
    StoreEach *j = (StoreEach *)p;
    i64 q;
    (void)tid;
    for (q = a; q < b; ++q) {
        i32 got = store_search(j->s, j->Q + (size_t)q * j->s->dim, j->k, j->ids + (size_t)q * j->k, j->sc + (size_t)q * j->k);
        for (; got < j->k; ++got) { j->ids[(size_t)q * j->k + got] = -1; j->sc[(size_t)q * j->k + got] = -4.0f; }
    }
}

/* nq queries at once, no filter: out_ids / out_scores are nq x k (a slot a query could not fill: id -1). The scan
 * runs cluster by cluster for the whole batch, so codes come in from memory once per batch instead of once per
 * query; the exact re-score then runs query by query on nthreads threads. Same answers as store_search per query.
 * For throughput (bulk jobs, benchmarks); a server answering one query at a time uses store_search. */
void store_search_batch(RagStore *s, const f32 *Q, i32 nq, i32 k, i64 *out_ids, f32 *out_scores, i32 nthreads)
{
    i32 dim = s->dim, R, off, i;
    f32 *qn, *ql;
    if (nq <= 0 || k <= 0) return;
    if (nthreads < 1) nthreads = 1;
    if (!s->tq) {                                                /* no scan index: one query at a time, in parallel */
        StoreEach e;
        e.s = s; e.Q = Q; e.k = k; e.ids = out_ids; e.sc = out_scores;
        tq_par2_(store_each_, &e, nq, nthreads);
        return;
    }
    qn = (f32 *)malloc((size_t)nq * (size_t)dim * sizeof(f32));
    ql = (f32 *)malloc((size_t)nq * sizeof(f32));
    memcpy(qn, Q, (size_t)nq * (size_t)dim * sizeof(f32));
    for (i = 0; i < nq; ++i) {                                   /* a query that is not finite: scanned as zeros */
        f32 *q = qn + (size_t)i * dim;
        if (!store_finite_(q, dim)) { memset(q, 0, (size_t)dim * sizeof(f32)); ql[i] = 0.0f; continue; }
        ql[i] = (f32)eng_sqrt_(eng_ssq_(q, dim)); store_normalise_(q, dim);
    }
    pthread_rwlock_rdlock(&s->lock);
    if (s->tune_target > 0.0f) for (i = 0; i < nq; i += 1 + nq / STORE_QLOG) store_qlog_(s, qn + (size_t)i * dim);
    R = s->tq_rerank > 0 ? s->tq_rerank : (k * 20 > 256 ? k * 20 : 256) * (s->tq->bits == 1 ? 4 : 1) * (s->metric ? 4 : 1);   /* as one query */
    for (off = 0; off < nq; off += 1024) {                       /* batches of 1024 bound the per-query candidates */
        i32 m = nq - off < 1024 ? nq - off : 1024;
        i32 *rows = (i32 *)malloc((size_t)m * (size_t)R * sizeof(i32)), *cnt = (i32 *)malloc((size_t)m * sizeof(i32));
        StoreBatch j;
        if (s->tq->codes1) tq_search_many_cascade(s->tq, qn + (size_t)off * dim, m, STORE_CAS_SHORT * R, R, rows, cnt, s->dead, nthreads, s->tq_probe);
        else tq_search_many(s->tq, qn + (size_t)off * dim, m, R, rows, cnt, s->dead, nthreads, s->tq_probe);
        j.s = s; j.qn = qn + (size_t)off * dim; j.qlen = ql + off; j.k = k; j.R = R; j.rows = rows; j.cnt = cnt;
        j.ids = out_ids + (size_t)off * k; j.sc = out_scores + (size_t)off * k;
        tq_par2_(store_batch_fin_, &j, m, nthreads);
        free(rows); free(cnt);
    }
    pthread_rwlock_unlock(&s->lock);
    free(qn); free(ql);
}

#define STORE_GROUP_MAX 65536   /* the most rows a grouped search ranks before it gives up on finding k groups */

/* store_search (cnf: store_search_cnf) with at most one hit a group, the group of an id being (u64)id >> shift (0 to 63): best first from
 * store_search, so a group's first hit is its best - keep it, skip the rest of the group; too few groups, ask for
 * four times as many rows, up to STORE_GROUP_MAX or the whole store. Returns how many groups, -1 out of memory. */
i32 store_search_cnf(RagStore *s, const f32 *query, i32 k, i64 *out_doc_ids, f32 *out_scores, const StoreCnf *c);
i32 store_search_grouped(RagStore *s, const f32 *query, i32 k, i32 shift, const StoreCnf *cnf, i64 *ids, f32 *scores)
{
    i64 total, *ci;
    f32 *cs;
    i32 want, n, have, i, j;
    if (shift <= 0) return cnf ? store_search_cnf(s, query, k, ids, scores, cnf) : store_search(s, query, k, ids, scores);
    total = store_count(s);
    want = k > STORE_GROUP_MAX / 4 ? STORE_GROUP_MAX : k * 4;
    if (want < 64) want = 64;
    for (;;) {
        if ((i64)want > total) want = total > 0 ? (i32)total : 1;
        ci = (i64 *)malloc((size_t)want * sizeof(i64));
        cs = (f32 *)malloc((size_t)want * sizeof(f32));
        if (!ci || !cs) { free(ci); free(cs); return -1; }
        n = cnf ? store_search_cnf(s, query, want, ci, cs, cnf) : store_search(s, query, want, ci, cs);
        for (i = have = 0; i < n && have < k; ++i) {
            u64 g = (u64)ci[i] >> shift;
            for (j = 0; j < have && ((u64)ids[j] >> shift) != g; ++j) {}
            if (j == have) { ids[have] = ci[i]; scores[have] = cs[i]; ++have; }
        }
        free(ci); free(cs);
        if (have >= k || n < want || (i64)want >= total || want >= STORE_GROUP_MAX) break;
        want = want > STORE_GROUP_MAX / 4 ? STORE_GROUP_MAX : want * 4;
    }
    return have;
}
