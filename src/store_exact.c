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
 * The search starts from the store's own approximate answer (the scan of the probed clusters, its best re-scored):
 * k rows scored exactly, so a bar tau stands before the first block. Then one pass over every code: a row whose score
 * at est + b stays under tau cannot be among the k best (k rows already reach tau); every other row is re-scored and
 * may raise the bar. The pass runs on up to 4 threads over ranges of blocks, each with its own bar (any k rows scored
 * exactly make a valid bar), and the threads' lists are merged. What comes out is the exact k best by the store's own
 * score - lean4/Vecindex/Exact.lean proves the cut and the answer. With the 4-bit codes of GloVe-100 (1.18M rows) a
 * query re-scores 0.16% of the rows; the price is reading every code, so the time grows with the store (no cluster is
 * skipped: in 100 dimensions no bound on a whole cluster excludes it, see docs/benchmarks.md).
 *
 * For the cosine the line is the same for every row, so the cut is a u16 sum per error byte (store_ex_cut_): the
 * scan kernel's own threshold mask drops the lanes below the cut for the store's largest byte, then a lane below the
 * cut for its own byte (kept in slot order) is out, and the per-row test runs on the rest. A batch reads each block
 * once for up to STORE_EX_QB queries while it is in cache.
 *
 * Scores other than the cosine (dot, L2) are a * cos + b per row with a >= 0 (the row's norm, the query's length), so
 * the same bound carries over. */

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

/* out by the cut: a sum below it. Checked at the boundary by tests/exact_test.c: a sum equal to the cut reaches tau
 * and stays in. */
static i32 store_ex_below_(i32 sum, i32 cut) { return sum < cut; }

/* A block's 32 sums into sm, and the lanes not out by the cut as bits: the kernel's own test (sum > thr, in
 * registers) with thr = cut - 1; cut 0 keeps every lane. tests/exact_test.c holds it to store_ex_below_ lane by lane. */
static u32 store_ex_lanes_(const u8 *cb, const u8 *lut, i32 vd, i32 cut, u16 *sm)
{
    if (cut <= 0) { tq_block_(cb, lut, vd, (u16)0xffff, sm); return 0xffffffffu; }
    return tq_block_(cb, lut, vd, (u16)(cut - 1), sm);
}

/* the lowest set bit of m (m != 0) */
static i32 store_ex_ctz_(u32 m)
{
#if defined(__GNUC__)
    return __builtin_ctz(m);
#else
    i32 i = 0;
    while (!(m & 1u)) { m >>= 1; ++i; }
    return i;
#endif
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

/* already among the rows a query started from (k at most: a linear look) */
static i32 store_ex_in_(const i64 *rows, i32 n, i32 row)
{
    i32 i;
    for (i = 0; i < n; ++i) if (rows[i] == (i64)row) return 1;
    return 0;
}

/* One query of a pass: its tables, its side of the bound, and the rows it starts from - the store's own approximate
 * answer re-scored, so a bar stands from the first block on. */
typedef struct { const f32 *qn; f32 qlen; StoreExQ eq; u8 *lut; i64 *irow; f32 *isc; i32 ni; i64 nchk; } StoreExQy;

/* one thread's state for one query: its k best so far (the start rows among them), its bar, its cuts */
typedef struct { i64 *row; f32 *sc; i32 have; i32 cut; f64 tau; u32 cgen; i64 nr; i32 ct[256]; u32 ctg[256]; } StoreExTh;

typedef struct { RagStore *s; const StoreCnf *cnf; StoreExQy *qy; StoreExTh *th; i32 nq, k; const u8 *err, *serr; u8 emax; } StoreExPass;

/* a bar of k rows scored exactly: tau and (the cosine) the cut for the store's largest byte */
static void store_ex_bar_(const RagStore *s, const StoreExQy *y, StoreExTh *h, i32 k, u8 emax)
{
    if (h->have < k || (f64)h->sc[k - 1] <= h->tau) return;
    h->tau = (f64)h->sc[k - 1];
    ++h->cgen;
    if (!s->metric) h->cut = store_ex_cut_(&y->eq, emax, h->tau);
}

/* The pass, a thread's range of blocks: each block's codes read once for every query of the pass, while they are in
 * cache. A block is out by its biggest sum, a row by its sum against the cut for the largest byte, then for its own
 * byte (the cosine), then by its own bound; what is left is scored from its vector. */
static void store_ex_pass_(void *p, i64 a, i64 b, i32 tid)
{
    StoreExPass *P = (StoreExPass *)p;
    RagStore *s = P->s;
    const Tq *t = s->tq;
    i32 blk, q, l, k = P->k, bb = t->vd * 16;
    StoreExTh *th = P->th + (size_t)tid * P->nq;
    u16 sm[32];
    for (blk = (i32)a; blk < (i32)b; ++blk) {
        const u8 *cb = tq_bptr_(t, blk);
        if (blk + 2 < (i32)b) tq_prefetch_blk_(tq_bptr_(t, blk + 2), bb);
        for (q = 0; q < P->nq; ++q) {
            StoreExQy *y = P->qy + q;
            StoreExTh *h = th + q;
            u32 in;
            for (in = store_ex_lanes_(cb, y->lut, t->vd, h->cut, sm); in; in &= in - 1) {   /* the lanes not out for the largest byte */
                i32 row;
                l = store_ex_ctz_(in);
                if (!s->metric && h->have >= k) {                                /* the cosine: its own byte's cut, in slot order */
                    u8 e = blk < t->nblk ? P->serr[(size_t)blk * 32 + l] : (u8)STORE_EX_NONE;
                    if (store_ex_below_(sm[l], store_ex_cutat_(&y->eq, h->ct, h->ctg, h->cgen, e, h->tau))) continue;
                }
                row = tq_srow_(t, blk, l);
                if (row >= t->n || !store_ex_ok_(s, P->cnf, row)) continue;
                if (h->have >= k) {
                    f64 la, lb;
                    store_ex_line_(s, y->qlen, row, &la, &lb);
                    if (store_ex_ub_(&y->eq, la, lb, sm[l], P->err[row]) < h->tau) continue;   /* proved out */
                }
                if (store_ex_in_(y->irow, y->ni, row) || store_row_bad_(s, row)) continue;
                ++h->nr;
                eng_topk_(h->row, h->sc, k, &h->have, (i64)row, store_score_(s, y->qn, y->qlen, row));
                store_ex_bar_(s, y, h, k, P->emax);                              /* a better k-th: a higher bar */
            }
        }
    }
}

/* the rows a query starts from: the store's scan of its probed clusters, the best re-scored (none for a flat
 * index, where that scan would read every code a second time) */
typedef struct { RagStore *s; const StoreCnf *cnf; StoreExQy *qy; i32 k, nth; } StoreExAnn;
static void store_ex_ann_(void *p, i64 a, i64 b, i32 tid)
{
    StoreExAnn *A = (StoreExAnn *)p;
    RagStore *s = A->s;
    const StoreCnf *cnf = A->cnf;
    i32 R = s->tq_rerank > 0 ? s->tq_rerank : (A->k * 20 > 256 ? A->k * 20 : 256) * (s->tq->bits == 1 ? 4 : 1) * (s->metric ? 4 : 1);
    i32 *rows = (i32 *)malloc((size_t)R * sizeof(i32)), m, i;
    i64 q;
    TqCol ncol, scol;
    TqFilt tf;
    (void)tid;
    memset(&tf, 0, sizeof(tf));
    if (cnf && cnf->num && s->has_num) { ncol = store_tqcol_(&s->numc); tf.num = &ncol; tf.lo = cnf->lo; tf.hi = cnf->hi; }
    if (cnf && cnf->ns_on && s->has_ns) { scol = store_tqcol_(&s->nsc); tf.ns = &scol; tf.nsv = cnf->ns; tf.nss = cnf->nss; tf.nns = cnf->nns; }
    for (q = a; q < b; ++q) {
        StoreExQy *y = A->qy + q;
        y->ni = 0; y->nchk = 0;
        if (!rows || s->tq->nc <= 1) continue;
        m = store_scan_(s, y->qn, R, rows, &tf, A->nth, s->tq_probe);
        for (i = 0; i < m; ++i)                                              /* what store_rescore_ scores */
            if (rows[i] >= 0 && rows[i] < s->built_upto && (!cnf || store_cnf_ok_(s, cnf, rows[i]))) ++y->nchk;
        store_rescore_(s, y->qn, y->qlen, cnf, rows, m, A->k, y->irow, y->isc, &y->ni);
        for (i = 0; i < y->ni; ++i) if (!store_ex_ok_(s, cnf, (i32)y->irow[i])) break;
        if (i < y->ni) y->ni = 0;                                            /* never: the scan keeps dead rows out; a bar of rows the search may return only */
    }
    free(rows);
}

#define STORE_EX_QB 16              /* queries a pass: each block's codes read once for all of them (1.3x, docs) */

/* The exact k best rows by the store's score among the indexed rows [0, built_upto) that cnf lets through, for nq
 * queries (qy: qn, qlen, lut, eq set; irow, isc room for k): into rows/scores + q * k (have[q] of them, sorted);
 * chk[q] += the rows scored from their vectors. -1: out of memory (nothing written). Under the read lock. */
static i32 store_ex_tq_(RagStore *s, StoreExQy *qy, i32 nq, const StoreCnf *cnf, i32 k, i64 *out_rows, f32 *out_scores, i32 *have, i64 *chk)
{
    const Tq *t = s->tq;
    i32 nb = t->nblk + t->xn, i, j, q, z, c = store_ncpu_(s), nth = nb >= 256 ? (c > 4 ? 4 : c) : 1;
    u8 emax = 0;
    const u8 *serr = (const u8 *)0, *err = store_ex_errs_(s, s->built_upto, &emax, &serr);
    StoreExTh *th;
    i64 *trow;
    f32 *tsc;
    StoreExPass P;
    StoreExAnn A;
    if (nth < 1) nth = 1;
    if (nth > nb) nth = nb > 0 ? nb : 1;
    th = (StoreExTh *)calloc((size_t)nth * nq, sizeof(StoreExTh));
    trow = (i64 *)malloc((size_t)nth * nq * k * sizeof(i64));
    tsc = (f32 *)malloc((size_t)nth * nq * k * sizeof(f32));
    if (!th || !trow || !tsc || !err || (!serr && t->nblk > 0)) { free(th); free(trow); free(tsc); return -1; }
    A.s = s; A.cnf = cnf; A.qy = qy; A.k = k;
    A.nth = nq == 1 ? (s->tq->n >= 150000 ? nth : 1) : 1;
    tq_par2_(store_ex_ann_, &A, nq, nq == 1 ? 1 : c);               /* one query: its scan on threads; many: a query a thread */
    for (j = 0; j < nth; ++j)
        for (q = 0; q < nq; ++q) {
            StoreExTh *h = th + (size_t)j * nq + q;
            h->row = trow + ((size_t)j * nq + q) * k; h->sc = tsc + ((size_t)j * nq + q) * k;
            h->have = qy[q].ni; h->tau = -3e38; h->cgen = 1;
            memcpy(h->row, qy[q].irow, (size_t)qy[q].ni * sizeof(i64)); memcpy(h->sc, qy[q].isc, (size_t)qy[q].ni * sizeof(f32));
            store_ex_bar_(s, qy + q, h, k, emax);
        }
    P.s = s; P.cnf = cnf; P.qy = qy; P.th = th; P.nq = nq; P.k = k; P.err = err; P.serr = serr; P.emax = emax;
    tq_par2_(store_ex_pass_, &P, nb, nth);                          /* every code once, for every query of the pass */
    for (q = 0; q < nq; ++q) {                                       /* each thread's k best, merged */
        i64 *ro = out_rows + (size_t)q * k;
        f32 *so = out_scores + (size_t)q * k;
        have[q] = 0;
        chk[q] += qy[q].nchk;
        for (j = 0; j < nth; ++j) {
            StoreExTh *h = th + (size_t)j * nq + q;
            chk[q] += h->nr;
            for (i = 0; i < h->have; ++i) {
                for (z = 0; z < have[q] && ro[z] != h->row[i]; ++z) {}
                if (z == have[q]) eng_topk_(ro, so, k, have + q, h->row[i], h->sc[i]);   /* a start row is in every thread's list */
            }
        }
    }
    free(th); free(trow); free(tsc);
    return 0;
}

/* store_search_exact for nq queries (queries: nq x dim): the exact k best of each into ids/scores + q * k, n[q] of
 * them (NULL: not wanted), checked[q] (NULL: not wanted) the rows scored from their vectors for it. Each block of
 * codes is read once for up to STORE_EX_QB queries. 0, or -1 (a query with a NaN or an infinity, out of memory). */
i32 store_search_exact_batch(RagStore *s, const f32 *queries, i32 nq, i32 k, i64 *out_ids, f32 *out_scores, i32 *n, const StoreCnf *cnf, i64 *checked)
{
    i32 dim = s->dim, q0, q, i, rc = 0, nqb, *hv;
    f32 *qn, *isc;
    i64 *irow, *chk;
    u8 *lut;
    StoreExQy *qy;
    if (k < 1 || nq < 1) return 0;
    for (q = 0; q < nq; ++q) if (!store_finite_(queries + (size_t)q * dim, dim)) return -1;
    nqb = nq < STORE_EX_QB ? nq : STORE_EX_QB;
    qn = (f32 *)malloc((size_t)nq * dim * sizeof(f32));
    qy = (StoreExQy *)calloc((size_t)nqb, sizeof(StoreExQy));
    irow = (i64 *)malloc((size_t)nqb * k * sizeof(i64)); isc = (f32 *)malloc((size_t)nqb * k * sizeof(f32));
    hv = (i32 *)calloc((size_t)nq, sizeof(i32)); chk = (i64 *)calloc((size_t)nq, sizeof(i64));
    pthread_rwlock_rdlock(&s->lock);
    lut = s->tq ? (u8 *)malloc((size_t)nqb * (s->tq->vd * 16 + 32)) : (u8 *)malloc(1);
    if (!qn || !qy || !irow || !isc || !hv || !chk || !lut) rc = -1;
    for (q = 0; rc == 0 && q < nq; ++q) {
        f32 *v = qn + (size_t)q * dim;
        memcpy(v, queries + (size_t)q * dim, (size_t)dim * sizeof(f32));
        store_normalise_(v, dim);
    }
    for (q0 = 0; rc == 0 && q0 < nq; q0 += nqb) {
        i32 m = nq - q0 < nqb ? nq - q0 : nqb;
        for (q = 0; q < m; ++q) {
            const f32 *v = queries + (size_t)(q0 + q) * dim;
            qy[q].qn = qn + (size_t)(q0 + q) * dim;
            qy[q].qlen = (f32)eng_sqrt_(eng_ssq_(v, dim));
            qy[q].irow = irow + (size_t)q * k; qy[q].isc = isc + (size_t)q * k;
            if (s->tq && s->built_upto > 0) { qy[q].lut = lut + (size_t)q * (s->tq->vd * 16 + 32); store_ex_query_(s->tq, qy[q].qn, qy[q].lut, &qy[q].eq); }
        }
        if (s->tq && s->built_upto > 0) rc = store_ex_tq_(s, qy, m, cnf, k, out_ids + (size_t)q0 * k, out_scores + (size_t)q0 * k, hv + q0, chk + q0);
        for (q = 0; rc == 0 && q < m; ++q) {                     /* the rest (no scan index, or the pending tail): every row */
            i64 *ro = out_ids + (size_t)(q0 + q) * k;
            f32 *so = out_scores + (size_t)(q0 + q) * k;
            for (i = s->tq && s->built_upto > 0 ? s->built_upto : 0; i < s->count; ++i) {
                if (STORE_DEAD(s, i) || (cnf && !store_cnf_ok_(s, cnf, i)) || store_row_bad_(s, i)) continue;
                ++chk[q0 + q];
                eng_topk_(ro, so, k, hv + q0 + q, (i64)i, store_score_(s, qy[q].qn, qy[q].qlen, i));
            }
            for (i = 0; i < hv[q0 + q]; ++i) ro[i] = STORE_ID(s, ro[i]);
        }
    }
    pthread_rwlock_unlock(&s->lock);
    if (rc == 0) for (q = 0; q < nq; ++q) { if (n) n[q] = hv[q]; if (checked) checked[q] = chk[q]; }
    free(qn); free(qy); free(irow); free(isc); free(hv); free(chk); free(lut);
    return rc;
}

/* store_search with a proof: the exact k best of every row the filter lets through. Without a scan index (or with an
 * engine index) every row is re-scored, which is exact too. *checked (may be NULL): the rows re-scored from their
 * vectors. Returns how many (k at most), -1 on a bad query or out of memory. */
i32 store_search_exact(RagStore *s, const f32 *query, i32 k, i64 *out_ids, f32 *out_scores, const StoreCnf *cnf, i64 *checked)
{
    i32 n = 0;
    i64 c = 0;
    if (k < 1) return 0;
    if (store_search_exact_batch(s, query, 1, k, out_ids, out_scores, &n, cnf, &c) != 0) return -1;
    if (checked) *checked = c;
    return n;
}
