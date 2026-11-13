/* store_exact.c - search with a proof: the k best rows of the whole store, not an index's guess at them.
 * Part of store.c, which includes it at this point; not compiled on its own.
 *
 * The scan index holds a 4-bit (2-, 1-bit) code of every row. From the code a row's cosine with the query is known to
 * within a margin that can be computed: the code is the row rotated and rounded to levels, so
 *
 *     cos(q, x) = (q' . x^ + q' . e) / dim       q' the rotated query, x^ the row's levels, e = x' - x^ its rounding
 *
 * and the scan's u16 sum S of the query's u8 tables gives q' . x^ = S / sc + base - rho / sc with |rho| <= vd / 2
 * (each table entry rounded to the nearest). Cauchy-Schwarz bounds the rest: |q' . e| <= |q'| |e|. A byte a row
 * (store_ex_errs_) keeps |e| rounded up. So every row's cosine lies in [est - b, est + b].
 *
 * The search: one pass over every code (the u16 sums kept, the R best estimates noted), the R best re-scored from
 * their stored vectors, tau = the k-th best of those. A row whose score at est + b stays under tau cannot be among the
 * k best (k rows already reach tau); every other row is re-scored. What comes out is the exact k best by the store's
 * own score - lean4/Vecindex/Exact.lean proves the cut. With the 4-bit codes of GloVe-100 (1.18M rows) a query
 * re-scores a median 277 rows (0.05%); the price is reading every code, so the time grows with the store (no cluster
 * is skipped: in 100 dimensions no bound on a whole cluster excludes it, see docs/benchmarks.md).
 *
 * For the cosine the line is the same for every row, so the cut is a u16 sum per error byte (store_ex_cut_): the
 * second pass skips a block whose biggest sum is below the cut for the store's largest byte, then a row below the cut
 * for its own byte (kept in slot order beside the sums), and runs the per-row test on the rest.
 *
 * Scores other than the cosine (dot, L2) are a * cos + b per row with a >= 0 (the row's norm, the query's length), so
 * the same bound carries over. */

#define STORE_EX_R      64          /* rows re-scored first, for tau: k at least, this many at most k > 64 */
#define STORE_EX_MARGIN 1e-4        /* cosine units: f32 rounding of the rotation, the tables and the dots (~1e-6) */
#define STORE_EX_NONE   255         /* an error byte too big to keep: the row is always re-scored */

/* the unit of an error byte: |e| of a unit row is about 0.1 sqrt(dim) at 4 bits, 0.34 at 2, 0.6 at 1 */
static f64 store_ex_step_(const Tq *t)
{
    return sqrt((f64)t->dim) / (t->bits == 4 ? 512.0 : t->bits == 2 ? 256.0 : 128.0);
}

/* |x' - x^| of one row: rotated, coded as the index codes it, against the levels the scan's tables use */
static f64 store_ex_rowerr_(const Tq *t, const f32 *x, f32 *rot, f32 *tmp, u8 *cd)
{
    f64 e = 0.0;
    i32 i;
    tq_rot_(t, x, rot, tmp);
    tq_encode_row_(t, rot, cd);
    for (i = 0; i < t->dpad; ++i) {
        f64 lv;
        if (t->bits == 4) lv = TQ_CENT[cd[i] & 15];
        else if (t->bits == 2) lv = TQ_CENT2[(cd[i / 2] >> (2 * (i & 1))) & 3];
        else lv = ((cd[i / 4] >> (i & 3)) & 1) ? TQ_L1 : -TQ_L1;
        e += ((f64)rot[i] - lv) * ((f64)rot[i] - lv);
    }
    return sqrt(e);
}

/* |e| rounded UP to whole steps, checked: b * step >= e, or STORE_EX_NONE */
static u8 store_ex_byte_(f64 e, f64 step)
{
    f64 b = ceil(e / step);
    if (b < 0.0) b = 0.0;
    while (b * step < e) b += 1.0;
    return b >= (f64)STORE_EX_NONE ? (u8)STORE_EX_NONE : (u8)b;
}

/* the error bytes of rows off + [a, b) (a thread's share) */
typedef struct { const RagStore *s; const Tq *t; u8 *err; f64 step; i64 off; } StoreExJob;
static void store_ex_job_(void *p, i64 a, i64 b, i32 tid)
{
    StoreExJob *j = (StoreExJob *)p;
    const Tq *t = j->t;
    TqCol sc, col = store_vcol_(j->s, &sc);
    f32 *rot = (f32 *)malloc((size_t)t->dpad * sizeof(f32)), *tmp = (f32 *)malloc((size_t)t->dpad * sizeof(f32));
    f32 *rb = (f32 *)malloc((size_t)t->dim * sizeof(f32));
    u8 *cd = (u8 *)malloc((size_t)t->dpad + 16);
    i64 r;
    (void)tid;
    for (r = j->off + a; r < j->off + b; ++r) j->err[r] = store_ex_byte_(store_ex_rowerr_(t, tq_row_(&col, r, rb), rot, tmp, cd), j->step);
    free(rot); free(tmp); free(rb); free(cd);
}

/* The error bytes of rows [0, upto) for the current index, built or extended on demand and kept until the index
 * changes (a rebuild, a compaction: store_ex_reap_). A reader may still hold the previous array when another extends
 * it, so an outgrown array is kept on the side until the next store_ex_reap_ (under the write lock). *emax: the
 * largest of them; *serr (NULL: not wanted): the same bytes in the order of the built index's slots (nblk x 32, a slot
 * without a row 0), so the scan's second pass reads them in step with its sums. Called under the read lock. */
static const u8 *store_ex_errs_(RagStore *s, i32 upto, u8 *emax, const u8 **serr)
{
    StoreCert *c;
    const u8 *out;
    pthread_mutex_lock(&s->cert_mu);
    c = s->cert;
    if (c && (c->tq != s->tq || c->gen != s->cert_gen)) {          /* stale: start over */
        c->n = 0; c->emax = 0; c->tq = s->tq; c->gen = s->cert_gen;
        free(c->serr); c->serr = (u8 *)0;                          /* no reader holds it: it is only made for the current gen */
    }
    if (!c) {
        c = (StoreCert *)calloc(1, sizeof(StoreCert));
        if (c) { c->tq = s->tq; c->gen = s->cert_gen; s->cert = c; }
    }
    if (c && c->n < upto) {
        i32 cap = c->cap >= upto ? c->cap : upto + upto / 2 + 1024, nth = store_ncpu_(s);
        u8 *e = c->cap >= upto ? c->err : (u8 *)malloc((size_t)cap);
        StoreExJob j;
        if (!e) { pthread_mutex_unlock(&s->cert_mu); return (const u8 *)0; }
        if (e != c->err) {
            if (c->n) memcpy(e, c->err, (size_t)c->n);
            if (c->err) { StoreCertOld *o = (StoreCertOld *)malloc(sizeof(StoreCertOld)); if (o) { o->err = c->err; o->next = c->old; c->old = o; } else free(c->err); }
            c->err = e; c->cap = cap;
        }
        j.s = s; j.t = s->tq; j.err = e; j.step = store_ex_step_(s->tq); j.off = c->n;
        tq_par_(store_ex_job_, &j, (i64)upto - c->n, nth);          /* one thread below 4096 rows */
        for (; c->n < upto; ++c->n) if (e[c->n] > c->emax) c->emax = e[c->n];
    }
    if (c && serr && !c->serr && s->tq->nblk > 0) {
        const Tq *t = s->tq;
        i32 i, row;
        c->serr = (u8 *)malloc((size_t)t->nblk * 32);
        if (c->serr) for (i = 0; i < t->nblk * 32; ++i) {
            row = tq_srow_(t, i / 32, i % 32);
            c->serr[i] = row < 0 || row >= t->n ? (u8)0 : row < c->n ? c->err[row] : (u8)STORE_EX_NONE;
        }
    }
    out = c ? c->err : (const u8 *)0;
    *emax = c ? c->emax : (u8)STORE_EX_NONE;
    if (serr) *serr = c ? c->serr : (const u8 *)0;
    pthread_mutex_unlock(&s->cert_mu);
    return out;
}

/* every error array, the current one too (it describes an index or a row order that is going away); under the
 * write lock, so no reader holds one */
static void store_ex_reap_(RagStore *s)
{
    StoreCert *c = s->cert;
    ++s->cert_gen;
    if (!c) return;
    while (c->old) { StoreCertOld *o = c->old; c->old = o->next; free(o->err); free(o); }
    free(c->err); c->err = (u8 *)0; c->cap = 0; c->n = 0; c->emax = 0;
    free(c->serr); c->serr = (u8 *)0;
}

static void store_ex_free_(RagStore *s)
{
    if (!s->cert) return;
    store_ex_reap_(s);
    free(s->cert); s->cert = (StoreCert *)0;
}

/* score = a * cos + b for row r: the store's own score as a line in the cosine (a >= 0) */
static void store_ex_line_(const RagStore *s, f32 qlen, i32 r, f64 *a, f64 *b)
{
    f64 vl;
    if (!s->metric) { *a = 2.0; *b = -2.0; return; }
    vl = STORE_NRM(s, r);
    if (s->metric == 1) { *a = (f64)qlen * vl; *b = 0.0; return; }
    *a = 2.0 * (f64)qlen * vl; *b = -((f64)qlen * qlen + vl * vl);
}

/* A query's side of the bound: what turns a row's u16 sum into its estimated cosine, and the margin around it. */
typedef struct { f64 sc, base, qr, lutb, inv, step; } StoreExQ;

/* the query's tables (lut: vd * 16 + 32 bytes) and its side of the bound */
static void store_ex_query_(const Tq *t, const f32 *qn, u8 *lut, StoreExQ *eq)
{
    f32 *rot = (f32 *)malloc((size_t)t->dpad * sizeof(f32)), *tmp = (f32 *)malloc((size_t)t->dpad * sizeof(f32));
    f32 *lf = (f32 *)malloc((size_t)t->vd * 16 * sizeof(f32)), sc = 1.0f;
    f64 base = 0.0, qr = 0.0;
    i32 i;
    tq_lut_ex_(t, qn, rot, tmp, lf, lut, &sc, &base);
    for (i = 0; i < t->dpad; ++i) qr += (f64)rot[i] * rot[i];
    eq->sc = sc; eq->base = base; eq->qr = sqrt(qr);                 /* |q'|: sqrt(dim) for a unit query, up to rounding */
    eq->lutb = (f64)t->vd * 0.5 / (f64)sc;                           /* the tables' rounding, in q' . x^ units */
    eq->inv = 1.0 / (f64)t->dim; eq->step = store_ex_step_(t);
    free(rot); free(tmp); free(lf);
}

/* the estimated cosine of a row from its u16 sum */
static f64 store_ex_est_(const StoreExQ *eq, u16 sum) { return ((f64)sum / eq->sc + eq->base) * eq->inv; }

/* how far the true cosine can be from the estimate, for a row with error byte e (STORE_EX_NONE: no bound) */
static f64 store_ex_b_(const StoreExQ *eq, u8 e)
{
    if (e == STORE_EX_NONE) return 1e30;
    return (eq->lutb + eq->qr * (f64)e * eq->step) * eq->inv + STORE_EX_MARGIN;
}

/* a row's score at its estimate plus bound: the most it can be (la, lb: store_ex_line_) */
static f64 store_ex_ub_(const StoreExQ *eq, f64 la, f64 lb, u16 sum, u8 e) { return la * (store_ex_est_(eq, sum) + store_ex_b_(eq, e)) + lb; }

/* The cosine's line is the same for every row, so the cut is one number: the least u16 sum whose bound with byte e
 * reaches tau (65536: none does). store_ex_ub_ only grows with the sum and with e (each f64 step rounds monotonically),
 * so a row with sum below the cut for the largest byte of the store is out by its own byte too: the per-row test then
 * runs on the few rows at or above it. */
static i32 store_ex_cut_(const StoreExQ *eq, u8 e, f64 tau)
{
    i32 lo = 0, hi = 65536;
    while (lo < hi) { i32 m = lo + (hi - lo) / 2; if (store_ex_ub_(eq, 2.0, -2.0, (u16)m, e) < tau) lo = m + 1; else hi = m; }
    return lo;
}

/* out by the cut: a sum below it (a block, by its biggest sum - every sum of the block is then below too). One test
 * for both, checked at the boundary by tests/exact_test.c: a sum equal to the cut reaches tau and stays in. */
static i32 store_ex_below_(i32 sum, i32 cut) { return sum < cut; }

/* the biggest of a block's 32 sums (a loop the compiler turns into a few vector max) */
static i32 store_ex_bmax_(const u16 *sm)
{
    u16 m = 0;
    i32 l;
    for (l = 0; l < 32; ++l) m = sm[l] > m ? sm[l] : m;
    return (i32)m;
}

/* the cut for byte e at the current tau, computed on first use after tau moved (gen: tau's version) */
static i32 store_ex_cutat_(const StoreExQ *eq, i32 *ct, u32 *ctg, u32 gen, u8 e, f64 tau)
{
    if (ctg[e] != gen) { ct[e] = store_ex_cut_(eq, e, tau); ctg[e] = gen; }
    return ct[e];
}

/* a row the search may return: live, inside the index, through the filter */
static i32 store_ex_ok_(const RagStore *s, const StoreCnf *cnf, i32 row)
{
    if (row < 0 || row >= s->built_upto || STORE_DEAD(s, row)) return 0;
    return !cnf || store_cnf_ok_((RagStore *)s, cnf, row);
}

/* Pass one, a range of blocks: every u16 sum into sums[block * 32 + lane], and (R > 0) the thread's R biggest sums
 * of rows the search may return into rsum/rrow + tid * R, nh[tid] of them, biggest first - read while the block's
 * sums are still in cache. */
typedef struct { const Tq *t; const u8 *lut; u16 *sums; const RagStore *s; const StoreCnf *cnf; i32 R; i32 *rsum, *rrow, *nh; } StoreExScan;
static void store_ex_scan_(void *p, i64 a, i64 b, i32 tid)
{
    StoreExScan *j = (StoreExScan *)p;
    const Tq *t = j->t;
    i32 blk, bb = t->vd * 16, R = j->R, n = 0, l, *rs = j->rsum + (size_t)tid * R, *rr = j->rrow + (size_t)tid * R;
    for (blk = (i32)a; blk < (i32)b; ++blk) {
        u16 *sm = j->sums + (size_t)blk * 32;
        if (blk + 2 < (i32)b) tq_prefetch_blk_(tq_bptr_(t, blk + 2), bb);
        tq_block_(tq_bptr_(t, blk), j->lut, t->vd, (u16)0xffff, sm);
        if (R > 0 && n == R && store_ex_bmax_(sm) <= rs[R - 1]) continue;   /* no sum of the block makes the list */
        if (R > 0) for (l = 0; l < 32; ++l) {
            i32 row, z;
            if (n == R && (i32)sm[l] <= rs[R - 1]) continue;
            row = tq_srow_(t, blk, l);
            if (row >= t->n || !store_ex_ok_(j->s, j->cnf, row)) continue;
            for (z = n < R ? n++ : R - 1; z > 0 && rs[z - 1] < (i32)sm[l]; --z) { rs[z] = rs[z - 1]; rr[z] = rr[z - 1]; }
            rs[z] = sm[l]; rr[z] = row;
        }
    }
    if (R > 0) j->nh[tid] = n;
}

/* The exact k best rows by the store's score among the indexed rows [0, built_upto) that cnf lets through, into
 * rows/scores (have of them); *checked += the rows re-scored from their vectors. -1: out of memory (nothing written).
 * Under the read lock. */
static i32 store_ex_tq_(RagStore *s, const f32 *qn, f32 qlen, const StoreCnf *cnf, i32 k, i64 *out_rows, f32 *out_scores, i32 *have, i64 *checked)
{
    const Tq *t = s->tq;
    i32 nb = t->nblk + t->xn, nslot = nb * 32, i, z, R = k > STORE_EX_R ? k : STORE_EX_R, nr = 0, nh = 0, cut = 0;
    i32 nth = nb >= 256 ? (store_ncpu_(s) > 4 ? 4 : store_ncpu_(s)) : 1, nt[4] = { 0, 0, 0, 0 };
    u8 *lut = (u8 *)malloc((size_t)t->vd * 16 + 32), *seen = (u8 *)calloc((size_t)(s->built_upto > 0 ? s->built_upto : 1), 1), emax = 0;
    u16 *sums = (u16 *)malloc((size_t)(nslot ? nslot : 1) * sizeof(u16));
    i32 *rrow = (i32 *)malloc((size_t)R * 5 * sizeof(i32)), *rsum = (i32 *)malloc((size_t)R * 5 * sizeof(i32));
    const u8 *serr = (const u8 *)0, *err = store_ex_errs_(s, s->built_upto, &emax, &serr);
    i32 ct[256];
    u32 ctg[256], cgen = 1;
    f64 tau;
    StoreExQ eq;
    StoreExScan sj;
    if (!lut || !sums || !rsum || !rrow || !err || !seen || (!serr && t->nblk > 0)) { free(lut); free(sums); free(rsum); free(rrow); free(seen); return -1; }
    memset(ctg, 0, sizeof(ctg));
    if (nth < 1) nth = 1;
    store_ex_query_(t, qn, lut, &eq);
    sj.t = t; sj.lut = lut; sj.sums = sums; sj.s = s; sj.cnf = cnf; sj.R = R; sj.rsum = rsum + R; sj.rrow = rrow + R; sj.nh = nt;
    tq_par2_(store_ex_scan_, &sj, nb, nth);                          /* every code once; each thread's R biggest sums */
    for (i = 0; i < nth; ++i)                                        /* merged: the R best estimates */
        for (z = 0; z < nt[i]; ++z) {
            i32 v = rsum[R + i * R + z], w = rrow[R + i * R + z], y;
            if (nh == R && v <= rsum[R - 1]) break;                  /* each thread's list is biggest first */
            for (y = nh < R ? nh++ : R - 1; y > 0 && rsum[y - 1] < v; --y) { rsum[y] = rsum[y - 1]; rrow[y] = rrow[y - 1]; }
            rsum[y] = v; rrow[y] = w;
        }
    for (i = 0; i < nh; ++i) {                                       /* re-scored: tau, the k-th best of them */
        seen[rrow[i]] = 1;
        if (store_row_bad_(s, rrow[i])) continue;
        eng_topk_(out_rows, out_scores, k, have, (i64)rrow[i], store_score_(s, qn, qlen, rrow[i]));
        ++nr;
    }
    tau = *have >= k ? (f64)out_scores[k - 1] : -3e38;
    if (!s->metric && *have >= k) cut = store_ex_cut_(&eq, emax, tau);
    for (i = 0; i < nslot; ++i) {                                    /* every row whose best case reaches tau */
        i32 row;
        if ((i & 31) == 0 && store_ex_below_(store_ex_bmax_(sums + i), cut)) { i += 31; continue; }   /* the whole block is out */
        if (store_ex_below_(sums[i], cut)) continue;                 /* out for the largest byte, so for its own */
        if (!s->metric && *have >= k) {                              /* the cosine: its own byte's cut, in slot order */
            u8 e = i < t->nblk * 32 ? serr[i] : STORE_EX_NONE;
            if (store_ex_below_(sums[i], store_ex_cutat_(&eq, ct, ctg, cgen, e, tau))) continue;
        }
        row = tq_srow_(t, i / 32, i % 32);
        if (row >= t->n || !store_ex_ok_(s, cnf, row) || seen[row]) continue;
        if (*have >= k) {
            f64 la, lb;
            store_ex_line_(s, qlen, row, &la, &lb);
            if (store_ex_ub_(&eq, la, lb, sums[i], err[row]) < tau) continue;   /* proved out */
        }
        seen[row] = 1;
        if (store_row_bad_(s, row)) continue;
        ++nr;
        eng_topk_(out_rows, out_scores, k, have, (i64)row, store_score_(s, qn, qlen, row));
        if (*have >= k && (f64)out_scores[k - 1] > tau) {            /* a better k-th: a higher bar for the rest */
            tau = (f64)out_scores[k - 1];
            ++cgen;
            if (!s->metric) cut = store_ex_cut_(&eq, emax, tau);
        }
    }
    *checked += nr;
    free(lut); free(sums); free(rsum); free(rrow); free(seen);
    return 0;
}

/* store_search with a proof: the exact k best of every row the filter lets through. Without a scan index (or with an
 * engine index) every row is re-scored, which is exact too. *checked (may be NULL): the rows re-scored from their
 * vectors. Returns how many (k at most), -1 on a bad query or out of memory. */
i32 store_search_exact(RagStore *s, const f32 *query, i32 k, i64 *out_ids, f32 *out_scores, const StoreCnf *cnf, i64 *checked)
{
    i32 dim = s->dim, have = 0, i, from = 0, rc = 0;
    f32 *qn, qlen;
    i64 chk = 0;
    if (k < 1 || !store_finite_(query, dim)) return k < 1 ? 0 : -1;
    qn = (f32 *)malloc((size_t)dim * sizeof(f32));
    if (!qn) return -1;
    memcpy(qn, query, (size_t)dim * sizeof(f32));
    qlen = (f32)eng_sqrt_(eng_ssq_(qn, dim));
    store_normalise_(qn, dim);
    pthread_rwlock_rdlock(&s->lock);
    if (s->tq && s->built_upto > 0) {
        rc = store_ex_tq_(s, qn, qlen, cnf, k, out_ids, out_scores, &have, &chk);
        from = s->built_upto;
    }
    if (rc == 0)
        for (i = from; i < s->count; ++i) {                          /* the rest (no scan index, or the pending tail): every row */
            if (STORE_DEAD(s, i) || (cnf && !store_cnf_ok_(s, cnf, i)) || store_row_bad_(s, i)) continue;
            ++chk;
            eng_topk_(out_ids, out_scores, k, &have, (i64)i, store_score_(s, qn, qlen, i));
        }
    if (rc == 0) for (i = 0; i < have; ++i) out_ids[i] = STORE_ID(s, out_ids[i]);
    pthread_rwlock_unlock(&s->lock);
    free(qn);
    if (checked) *checked = chk;
    return rc < 0 ? -1 : have;
}
