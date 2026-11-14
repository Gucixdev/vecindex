/* store_text.c - a text per row: put with text, get it back, the ids in a range (small-to-big retrieval), and
 * hybrid search (the vector and BM25 over the texts, fused by rank).
 * Part of store.c, which includes it at this point; not compiled on its own. */

/* 'Q' id meta vec[dim] len(u32) text[len]: an upsert with its text, one framed record, so a torn tail drops both */
static void store_jrec_text_(RagStore *s, i64 id, i64 meta, const f32 *vec, const char *text, i32 len)
{
    const void *pt[6];
    size_t pl[6];
    u8 op = 'Q';
    u32 n = (u32)len;
    pt[0] = &op; pl[0] = 1; pt[1] = &id; pl[1] = 8; pt[2] = &meta; pl[2] = 8;
    pt[3] = vec; pl[3] = sizeof(f32) * (size_t)s->dim; pt[4] = &n; pl[4] = 4; pt[5] = text; pl[5] = len > 0 ? (size_t)len : 0;
    store_jframe_(s, pt, pl, 6);
}

/* Upsert with a text (len bytes, at most 16 MB - 1; any bytes, UTF-8 for the word index): every row with doc_id
 * goes, the new one comes in with its text. 0, or -1 (a non-finite vector, a bad length, out of memory: nothing
 * changed). A text of length 0 stores the row without one. */
i32 store_put_text(RagStore *s, const f32 *vec, i64 doc_id, i64 meta, const char *text, i32 len)
{
    u64 v;
    if (len == 0) return store_put(s, vec, doc_id, meta);
    if (!store_finite_(vec, s->dim) || len < 0 || len > STORE_TXMAX || !text) return -1;
    pthread_rwlock_wrlock(&s->lock);
    store_idmap_locked_(s);
    s->has_text = 1;                                             /* from now on the column has a slot for every row */
    if (store_reserve_(s, (i64)s->count + 1) != 0 || !(v = store_tx_put_(s, &s->tx, text, len))) {
        pthread_rwlock_unlock(&s->lock);
        return -1;
    }
    store_kill_id_locked_(s, doc_id);
    store_append_locked_(s, vec, doc_id, meta);
    STORE_TXO(s, s->count - 1) = v;
    s->txlive += len;
    if (s->lex) lex_add(s->lex, doc_id, text, len);
    store_jrec_text_(s, doc_id, meta, s->addbuf, text, len);
    pthread_rwlock_unlock(&s->lock);
    return 0;
}

/* The text of doc_id: its length (0: the row has none), with the first min(length, cap) bytes copied into buf (no
 * terminating 0 added); -1 when the id is not live. A length above cap means buf was too small. */
i32 store_get_text(RagStore *s, i64 doc_id, char *buf, i32 cap)
{
    i32 r, n = -1;
    store_need_idmap_(s);
    pthread_rwlock_rdlock(&s->lock);
    r = store_live_row_(s, doc_id);
    if (r >= 0) {
        u64 v = s->has_text ? STORE_TXO(s, r) : 0;
        n = STORE_TXLEN(v);
        if (n > 0 && buf && cap > 0) store_tx_copy_(&s->tx, v, buf, n < cap ? n : cap);
    }
    pthread_rwlock_unlock(&s->lock);
    return n;
}

static int store_i64_cmp_(const void *a, const void *b)
{
    i64 x = *(const i64 *)a, y = *(const i64 *)b;
    return x < y ? -1 : x > y;
}

/* sort u[0, *nu) and keep its max smallest */
static void store_keep_low_(i64 *u, i32 *nu, i32 max)
{
    qsort(u, (size_t)*nu, sizeof(i64), store_i64_cmp_);
    if (*nu > max) *nu = max;
}

/* The live ids in [lo, hi], ascending: the smallest max of them into out; returns how many were written. With ids
 * made as document << bits | chunk, this lists a document's chunks (fetch each with store_get_text); to page through
 * a long range, call again from the last id + 1. Rows added in increasing id order are found by binary search; the
 * rest by looking each id of a short range up, or by one pass over them when the range is wide. -1: out of memory. */
i32 store_range(RagStore *s, i64 lo, i64 hi, i64 *out, i32 max)
{
    i64 *u = (i64 *)0;
    i32 nu = 0, cap = 0, n = 0, a = 0, b, r, j = 0;
    if (lo > hi || max <= 0) return 0;
    store_need_idmap_(s);
    pthread_rwlock_rdlock(&s->lock);
    if (s->nsorted < s->count) {                                 /* rows added out of order: collected, then sorted */
        cap = max < 32 ? 64 : 2 * max;
        if (!(u = (i64 *)malloc((size_t)cap * sizeof(i64)))) { pthread_rwlock_unlock(&s->lock); return -1; }
        if ((u64)hi - (u64)lo < (u64)(s->count - s->nsorted)) {
            i64 id;
            for (id = lo; ; ++id) {                              /* short range: one lookup per id */
                if (store_live_row_(s, id) >= s->nsorted) { if (nu == cap) store_keep_low_(u, &nu, max); u[nu++] = id; }
                if (id == hi) break;
            }
        } else
            for (r = s->nsorted; r < s->count; ++r) {
                i64 id = STORE_ID(s, r);
                if (STORE_DEAD(s, r) || id < lo || id > hi) continue;
                if (nu == cap) store_keep_low_(u, &nu, max);
                u[nu++] = id;
            }
        store_keep_low_(u, &nu, max);
    }
    b = s->nsorted;                                              /* rows [0, nsorted): ids strictly increasing */
    while (a < b) { i32 m = a + (b - a) / 2; if (STORE_ID(s, m) < lo) a = m + 1; else b = m; }
    for (r = a; n < max; ) {                                     /* merge the two ascending lists */
        while (r < s->nsorted && STORE_DEAD(s, r)) ++r;
        if (r < s->nsorted && STORE_ID(s, r) <= hi && (j >= nu || STORE_ID(s, r) <= u[j])) { out[n++] = STORE_ID(s, r); ++r; }
        else if (j < nu) out[n++] = u[j++];
        else break;
    }
    pthread_rwlock_unlock(&s->lock);
    free(u);
    return n;
}

/* the word index over every live text, built on first use (under the read lock: writers wait, readers go on) */
static void store_lex_ready_(RagStore *s)
{
    Lex *x;
    char *buf;
    i32 r;
#if defined(__GNUC__)
    if (__atomic_load_n(&s->lex, __ATOMIC_ACQUIRE)) return;
#else
    if (s->lex) return;
#endif
    pthread_mutex_lock(&s->lex_mu);
    if (s->lex || !s->has_text) { pthread_mutex_unlock(&s->lex_mu); return; }
    x = (Lex *)malloc(sizeof(Lex));
    buf = (char *)malloc((size_t)STORE_TXMAX + 1);
    if (!x || !buf) { free(x); free(buf); pthread_mutex_unlock(&s->lex_mu); return; }
    lex_init(x);
    for (r = 0; r < s->count; ++r) {
        u64 v = STORE_TXO(s, r);
        if (!v || STORE_DEAD(s, r)) continue;
        store_tx_copy_(&s->tx, v, buf, STORE_TXLEN(v));
        lex_add(x, STORE_ID(s, r), buf, STORE_TXLEN(v));
    }
    free(buf);
#if defined(__GNUC__)
    __atomic_store_n(&s->lex, x, __ATOMIC_RELEASE);
#else
    s->lex = x;
#endif
    pthread_mutex_unlock(&s->lex_mu);
}

#define STORE_RRF_K 60.0f

/* Hybrid search: the k best of the vector query q (unit length or not; NULL: words only) and BM25 over the texts for
 * the words of text (tlen bytes; NULL or 0: vector only), fused by rank. Each side fetches 3k + 10 candidates (at
 * most 250); a candidate scores w / (60 + rank) on each side it appears in. The weights were measured on
 * Wikipedia with two embedders: words 1.5, the vector 0.6, plus up to 2 for a side whose best hit stands out from its
 * runner-up, so an exact name or a near-duplicate wins either way. out_scores get the fused scores (bigger =
 * better). Returns the number of hits. */
i32 store_search_hybrid(RagStore *s, const f32 *q, const char *text, i32 tlen, i32 k, i64 *out_ids, f32 *out_scores)
{
    i64 vid[250], lid[250], cid[500];
    f32 vsc[250], lsc[250], fused[500], wv = 0.6f, wl = 1.5f;
    i32 over = 3 * k + 10, nv = 0, nl = 0, nc = 0, i, j, n;
    if (k <= 0) return 0;
    if (over > 250) over = 250;
    if (q) nv = store_search(s, q, over, vid, vsc);
    if (text && tlen > 0) {
        pthread_rwlock_rdlock(&s->lock);
        store_lex_ready_(s);
        if (s->lex) nl = lex_search(s->lex, text, tlen, over, lid, lsc);
        pthread_rwlock_unlock(&s->lock);
    }
    if (nv > 1 && vsc[0] < 0.0f && vsc[1] < 0.0f) { f32 m = 1.0f - vsc[0] / vsc[1]; wv += 2.0f * (m > 0.0f ? (m < 1.0f ? m : 1.0f) : 0.0f); }
    if (nl == 1) wl = 3.0f;
    else if (nl > 1 && lsc[0] > 0.0f) wl += 2.0f * (1.0f - lsc[1] / lsc[0]);
    for (i = 0; i < nv + nl; ++i) {
        i64 id = i < nv ? vid[i] : lid[i - nv];
        f32 add = i < nv ? wv / (STORE_RRF_K + (f32)i + 1.0f) : wl / (STORE_RRF_K + (f32)(i - nv) + 1.0f);
        for (j = 0; j < nc && cid[j] != id; ++j) {}
        if (j == nc) { cid[nc] = id; fused[nc] = 0.0f; ++nc; }
        fused[j] += add;
    }
    for (n = 0; n < k && n < nc; ++n) {                          /* selection of the top k */
        i32 best = n;
        for (j = n + 1; j < nc; ++j) if (fused[j] > fused[best]) best = j;
        if (best != n) { i64 t = cid[n]; f32 f = fused[n]; cid[n] = cid[best]; fused[n] = fused[best]; cid[best] = t; fused[best] = f; }
        out_ids[n] = cid[n];
        if (out_scores) out_scores[n] = fused[n];
    }
    return n;
}
