/* store_ns.c - namespaces: many tenants in one store, each row in at most one namespace (an i64; 0: none), and a
 * search in a namespace that sees only its rows. A small namespace - most tenants - is searched exactly over a list
 * of its own rows, so its few rows scattered over every IVF cluster cost no scan of the others; a large one goes
 * through the scan with the namespace as a filter tested inside it. The lists are built on the first namespace
 * search and kept up to date by the writes after it (under the write lock): a row that moves leaves its old list, a
 * dead row stays on it until the next compaction (searches skip it), so no list holds a row twice. Ids stay the store's: one id is one row, whatever its
 * namespace. Part of store.c, which includes it at this point; not compiled on its own. */

static i32 store_ns_exact_max_ = 20000;   /* rows a namespace may have and still be searched over its list, exactly
                                             * (the self-test lowers it to reach the scan) */

typedef struct { i64 key; i32 n, cap; i32 *rows; } StoreNsL;   /* cap 0: an empty slot */
typedef struct { StoreNsL *t; u32 mask; i32 used; } StoreNs;

static u32 store_ns_h_(i64 k) { u64 x = (u64)k * STORE_U64(0x9e3779b9u, 0x7f4a7c15u); return (u32)(x >> 32); }

static void store_ns_drop_(StoreNs *x)
{
    u32 i;
    if (!x) return;
    for (i = 0; x->t && i <= x->mask; ++i) free(x->t[i].rows);
    free(x->t);
    free(x);
}

static void store_ns_free_(RagStore *s) { store_ns_drop_((StoreNs *)s->nsx); s->nsx = (void *)0; }

/* the slot of namespace k (k != 0): its list, or with make a new empty one when missing; NULL out of memory or not
 * found */
static StoreNsL *store_ns_slot_(StoreNs *x, i64 k, i32 make)
{
    u32 i;
    if (make && (u32)(x->used + 1) * 2 > x->mask + 1) {           /* half full: twice the slots */
        StoreNsL *o = x->t, *t = (StoreNsL *)calloc(((size_t)x->mask + 1) * 2, sizeof(StoreNsL));
        u32 om = x->mask, j;
        if (!t) return (StoreNsL *)0;
        x->t = t; x->mask = om * 2 + 1;
        for (j = 0; j <= om; ++j) {
            if (!o[j].cap) continue;
            for (i = store_ns_h_(o[j].key) & x->mask; t[i].cap; i = (i + 1) & x->mask) {}
            t[i] = o[j];
        }
        free(o);
    }
    for (i = store_ns_h_(k) & x->mask; x->t[i].cap; i = (i + 1) & x->mask) if (x->t[i].key == k) return &x->t[i];
    if (!make) return (StoreNsL *)0;
    if (!(x->t[i].rows = (i32 *)malloc(8 * sizeof(i32)))) return (StoreNsL *)0;
    x->t[i].key = k; x->t[i].n = 0; x->t[i].cap = 8;
    ++x->used;
    return &x->t[i];
}

/* row r onto its namespace's list; out of memory, the lists go (the next namespace search builds them again) */
static i32 store_ns_push_(StoreNs *x, i64 k, i32 r)
{
    StoreNsL *l = store_ns_slot_(x, k, 1);
    if (!l) return -1;
    if (l->n == l->cap) {
        i32 *nr = (i32 *)realloc(l->rows, (size_t)l->cap * 2 * sizeof(i32));
        if (!nr) return -1;
        l->rows = nr; l->cap *= 2;
    }
    l->rows[l->n++] = r;
    return 0;
}

/* row r leaves namespace k's list (a move to another namespace), so no list ever holds a row twice */
static void store_ns_unlist_(RagStore *s, i64 k, i32 r)
{
    StoreNsL *l = s->nsx && k ? store_ns_slot_((StoreNs *)s->nsx, k, 0) : (StoreNsL *)0;
    i32 i;
    if (!l) return;
    for (i = 0; i < l->n; ++i) if (l->rows[i] == r) { l->rows[i] = l->rows[--l->n]; return; }
}

/* a write under the write lock put row r into a namespace: onto its list, when the lists exist */
static void store_ns_row_(RagStore *s, i32 r)
{
    StoreNs *x = (StoreNs *)s->nsx;
    if (!x || !s->has_ns || !STORE_NS(s, r)) return;
    if (store_ns_push_(x, STORE_NS(s, r), r) != 0) store_ns_free_(s);
}

/* the lists, under the read lock (no write runs meanwhile): built once, then published */
static StoreNs *store_ns_get_(RagStore *s)
{
    StoreNs *x;
    i32 r;
#if defined(__GNUC__)
    if ((x = (StoreNs *)__atomic_load_n(&s->nsx, __ATOMIC_ACQUIRE)) != (StoreNs *)0) return x;
#else
    if ((x = (StoreNs *)s->nsx) != (StoreNs *)0) return x;
#endif
    if (!s->has_ns) return (StoreNs *)0;
    pthread_mutex_lock(&s->ns_mu);
    if (!(x = (StoreNs *)s->nsx)) {
        x = (StoreNs *)calloc(1, sizeof(StoreNs));
        if (x) { x->mask = 63; x->t = (StoreNsL *)calloc(64, sizeof(StoreNsL)); }
        for (r = 0; x && x->t && r < s->count; ++r)
            if (!STORE_DEAD(s, r) && STORE_NS(s, r) && store_ns_push_(x, STORE_NS(s, r), r) != 0) break;
        if (!x || !x->t || r < s->count) { store_ns_drop_(x); x = (StoreNs *)0; }
#if defined(__GNUC__)
        else __atomic_store_n(&s->nsx, (void *)x, __ATOMIC_RELEASE);
#else
        else s->nsx = (void *)x;
#endif
    }
    pthread_mutex_unlock(&s->ns_mu);
    return x;
}

/* Under the read lock: when cnf asks for a namespace small enough for its list, its k best rows by an exact score into
 * rows[] (row numbers) / sc[], *have of them, and 1; else 0 - the caller scans with the namespace as a filter */
static i32 store_ns_exact_(RagStore *s, const StoreCnf *cnf, const f32 *qn, f32 qlen, i32 k, i64 *rows, f32 *sc, i32 *have)
{
    StoreNs *x;
    StoreNsL *l;
    i32 i;
    if (!cnf || !cnf->ns_on || !cnf->ns) return 0;
    if (!s->has_ns) { *have = 0; return 1; }                     /* no namespaces yet: none has a row */
    if (!(x = store_ns_get_(s))) return 0;
    if (!(l = store_ns_slot_(x, cnf->ns, 0))) { *have = 0; return 1; }
    if (l->n > store_ns_exact_max_) return 0;
    for (i = 0; i < l->n; ++i) {
        i32 r = l->rows[i];
        if (r < 0 || r >= s->count || STORE_DEAD(s, r) || !store_cnf_ok_(s, cnf, r) || store_row_bad_(s, r)) continue;
        eng_topk_(rows, sc, k, have, (i64)r, store_score_(s, qn, qlen, r));
    }
    return 1;
}

/* every row of namespace ns (not 0) goes; journaled as one record. Returns how many, -1 for ns 0 */
i32 store_delete_ns(RagStore *s, i64 ns)
{
    i32 r, hit = 0;
    StoreNsL *l;
    if (!ns) return -1;
    pthread_rwlock_wrlock(&s->lock);
    if (s->has_ns) {
        l = s->nsx ? store_ns_slot_((StoreNs *)s->nsx, ns, 0) : (StoreNsL *)0;
        if (s->nsx && !l) {}                                     /* the lists know it has no rows */
        else for (r = 0; r < (l ? l->n : s->count); ++r) {
            i32 q = l ? l->rows[r] : r;
            if (q < 0 || q >= s->count || STORE_DEAD(s, q) || STORE_NS(s, q) != ns) continue;
            STORE_KILL(s, q); ++hit; --s->live; ++s->mods; store_tx_drop_(s, q);
        }
    }
    store_jrec_(s, 'Y', ns, 0, (const f32 *)0);
    pthread_rwlock_unlock(&s->lock);
    return hit;
}

/* the live rows of namespace ns (0: the rows in none) */
i64 store_count_ns(RagStore *s, i64 ns)
{
    StoreNs *x;
    StoreNsL *l = (StoreNsL *)0;
    i64 n = 0;
    i32 r;
    pthread_rwlock_rdlock(&s->lock);
    if (ns && (x = store_ns_get_(s)) != (StoreNs *)0) l = store_ns_slot_(x, ns, 0);
    if (ns && s->nsx && !l) {}
    else for (r = 0; r < (l ? l->n : s->count); ++r) {
        i32 q = l ? l->rows[r] : r;
        if (q < 0 || q >= s->count || STORE_DEAD(s, q) || STORE_NSOF(s, q) != ns) continue;
        ++n;
    }
    pthread_rwlock_unlock(&s->lock);
    return n;
}
