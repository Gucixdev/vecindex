/* vecindex.c - the C API of include/vecindex.h over store.c.
 *
 * Built as a shared library with -fvisibility=hidden, so only the VI_API functions are exported; everything else in
 * the translation unit (the store, the indexes) stays internal. ./run lib builds .build/libvecindex.so and
 * .build/libvecindex.a and runs tests/api_test.c against both. By hand:
 *
 *   gcc -std=c89 -pedantic -O2 -march=native -fPIC -shared -fvisibility=hidden -D_POSIX_C_SOURCE=200112L \
 *       -Isrc -Iinclude src/vecindex.c -o libvecindex.so -lm -lpthread
 *
 * Or compile it into a program like any other part of vecindex: one translation unit, -Isrc -Iinclude. */
#include "store.c"
#include "vecindex.h"

struct vi_db {
    Arena arena;                /* the store lives in it */
    RagStore *s;
    char *path;                 /* the file it was opened from; NULL: in memory */
};

/* ---- why the last call failed: a message per thread, vi_error() ---- */

static pthread_key_t vi_ekey_;
static pthread_once_t vi_eonce_ = PTHREAD_ONCE_INIT;
static void vi_ekey_init_(void) { pthread_key_create(&vi_ekey_, free); }
#define VI_EMAX 320

static char *vi_ebuf_(void)
{
    char *b;
    pthread_once(&vi_eonce_, vi_ekey_init_);
    b = (char *)pthread_getspecific(vi_ekey_);
    if (!b && (b = (char *)calloc(1, VI_EMAX)) != (char *)0 && pthread_setspecific(vi_ekey_, b) != 0) { free(b); b = (char *)0; }
    return b;
}

/* the calling thread's reason, a then b then c (NULL ones skipped, the whole cut to fit); returns r */
static int vi_fail_(int r, const char *a, const char *b, const char *c)
{
    char *e = vi_ebuf_();
    const char *part[3];
    size_t n = 0, m;
    int i;
    if (!e) return r;
    part[0] = a; part[1] = b; part[2] = c;
    for (i = 0; i < 3; ++i) {
        if (!part[i]) continue;
        m = strlen(part[i]);
        if (m > VI_EMAX - 1 - n) m = VI_EMAX - 1 - n;
        memcpy(e + n, part[i], m);
        n += m;
    }
    e[n] = 0;
    return r;
}

/* v in decimal, exactly (ids are any 64-bit value) */
static const char *vi_i64s_(char *buf, vi_i64 v)
{
    char t[24];
    int n = 0, neg = v < 0;
    vi_i64 q = v;
    do { int d = (int)(q % 10); t[n++] = (char)('0' + (d < 0 ? -d : d)); q /= 10; } while (q);
    if (neg) t[n++] = '-';
    {   int i; for (i = 0; i < n; ++i) buf[i] = t[n - 1 - i]; buf[n] = 0; }
    return buf;
}

#define VI_NODB "db is NULL (did vi_open fail? vi_error() said why)"

/* 0 when vec holds dim finite values that are not all zero, else -1 with the reason (`what`: "the vector", "the query") */
static int vi_vec_ok_(const vi_db *db, const float *v, const char *what)
{
    char nb[24];
    int i, any = 0;
    if (!v) return vi_fail_(-1, what, " is NULL", (const char *)0);
    for (i = 0; i < db->s->dim; ++i) {
        if (!(v[i] - v[i] == 0.0f)) { sprintf(nb, "%d", i); return vi_fail_(-1, what, " has a NaN or an infinity at index ", nb); }
        if (v[i] != 0.0f) any = 1;
    }
    return any ? 0 : vi_fail_(-1, what, " is all zeros: it has no direction to compare", (const char *)0);
}

/* why a row could not be read: none with that id, or it failed its check */
static int vi_norow_(vi_db *db, vi_i64 id)
{
    char nb[24];
    vi_i64s_(nb, id);
    if (store_has(db->s, (i64)id))
        return vi_fail_(-1, "the row with id ", nb, " is damaged: it failed its check and could not be rebuilt (vi_integrity)");
    return vi_fail_(-1, "no row with id ", nb, (const char *)0);
}

/* scores as callers expect them: the cosine (1 same direction, -1 opposite), the inner product, or the L2 distance */
static void vi_cos_(const vi_db *db, float *sc, int n) { int i; for (i = 0; i < n; ++i) sc[i] = store_score_out(db->s, sc[i]); }

static char *vi_strdup_(const char *p)
{
    char *d = (char *)malloc(strlen(p) + 1);
    if (d) strcpy(d, p);
    return d;
}

VI_API const char *vi_version(void) { return VI_VERSION; }

VI_API const char *vi_error(void)
{
    char *e = vi_ebuf_();
    return e ? e : "out of memory";
}

VI_API const char *vi_open_error(void) { return vi_error(); }

VI_API vi_db *vi_open(const char *path, int dim, int flags)
{
    vi_db *db;
    struct stat st;
    int exists = path && stat(path, &st) == 0;
    if (dim < 0) { vi_fail_(0, "dim must be 0 (open a file that exists) or more", (const char *)0, (const char *)0); return (vi_db *)0; }
    if (exists && S_ISDIR(st.st_mode)) { vi_fail_(0, path, " is a directory (vi_dir_open opens a directory of stores)", (const char *)0); return (vi_db *)0; }
    if (!exists && dim == 0) {
        vi_fail_(0, path ? path : "a store in memory", path ? " does not exist, and dim 0 cannot create it: give the dimensions" : " needs dim of 1 or more", (const char *)0);
        return (vi_db *)0;
    }
    if (!(db = (vi_db *)calloc(1, sizeof(vi_db)))) { vi_fail_(0, "out of memory", (const char *)0, (const char *)0); return (vi_db *)0; }
    if (exists) {
        db->s = store_load(&db->arena, path);                    /* there is a file: load it or fail, never replace it */
        if (!db->s) {
            vi_fail_(0, path, " cannot be opened: ", *store_load_error() ? store_load_error() : "not a vecindex store");
            arena_free(&db->arena); free(db); return (vi_db *)0;
        }
        if (dim > 0 && db->s->dim != dim) {
            char b[64];
            sprintf(b, "%d dimensions, not %d", (int)db->s->dim, dim);
            vi_fail_(0, path, " holds vectors of ", b);
            store_destroy(db->s); arena_free(&db->arena); free(db); return (vi_db *)0;
        }
    } else
        db->s = store_new(&db->arena, (i32)dim);
    {   int want = (flags & VI_DOT) ? 1 : ((flags & VI_L2) ? 2 : 0);   /* the metric: asked for, or the file's */
        static const char *const mn[3] = { "cosine", "dot", "L2" };
        if ((flags & VI_DOT) && (flags & VI_L2)) {
            vi_fail_(0, "VI_DOT and VI_L2 together: a store has one metric", (const char *)0, (const char *)0);
            store_destroy(db->s); arena_free(&db->arena); free(db->path); free(db); return (vi_db *)0;
        }
        if (exists && want && want != db->s->metric) {
            vi_fail_(0, path, " is a ", mn[db->s->metric]);
            {   char *e = vi_ebuf_(); if (e && strlen(e) + 40 < VI_EMAX) { strcat(e, " store, not "); strcat(e, mn[want]); } }
            store_destroy(db->s); arena_free(&db->arena); free(db->path); free(db); return (vi_db *)0;
        }
        if (!exists && want) store_set_metric(db->s, want);
    }
    if (path) {
        db->path = vi_strdup_(path);
        if (!db->path || (!db->s->jf && store_save(db->s, path) != 0)) {   /* a new file exists from the start, */
            int en = errno;                                              /* with its journal */
            vi_fail_(0, "cannot create ", path, en ? ": " : "");
            if (en) { char *e = vi_ebuf_(); if (e && strlen(e) + strlen(strerror(en)) < VI_EMAX) strcat(e, strerror(en)); }
            store_destroy(db->s); arena_free(&db->arena); free(db->path); free(db);
            return (vi_db *)0;
        }
    }
    if (!(flags & VI_MANUAL)) store_set_live(db->s, 1, 1.0f);   /* rows are indexed as they come: no vi_build to forget */
    {   char *e = vi_ebuf_(); if (e) *e = 0; }                   /* an open that worked leaves no reason from an earlier failure */
    return db;
}

VI_API void vi_close(vi_db *db)
{
    if (!db) return;
    if (db->path) store_sync(db->s, db->path);
    store_destroy(db->s);                                        /* waits for a running snapshot */
    arena_free(&db->arena);
    free(db->path);
    free(db);
}

/* ---- tenants without a server: a directory of stores, opened on first use ---- */

typedef struct vi_ten { char name[65]; vi_db *db; struct vi_ten *next; } vi_ten;
struct vi_dir {
    char *path;
    int dim, flags;
    pthread_mutex_t mu;         /* the list, and opening a store into it */
    vi_ten *head;
};

VI_API vi_dir *vi_dir_open(const char *path, int dim, int flags)
{
    struct stat st;
    vi_dir *d;
    if (!path) { vi_fail_(0, "path is NULL", (const char *)0, (const char *)0); return (vi_dir *)0; }
    if (stat(path, &st) != 0 || !S_ISDIR(st.st_mode)) { vi_fail_(0, path, " is not a directory (make it first)", (const char *)0); return (vi_dir *)0; }
    d = (vi_dir *)calloc(1, sizeof(vi_dir));
    if (!d || !(d->path = vi_strdup_(path))) { free(d); vi_fail_(0, "out of memory", (const char *)0, (const char *)0); return (vi_dir *)0; }
    d->dim = dim; d->flags = flags;
    pthread_mutex_init(&d->mu, (const pthread_mutexattr_t *)0);
    return d;
}

VI_API vi_db *vi_dir_get(vi_dir *d, const char *name)
{
    vi_ten *t;
    size_t n = name ? strlen(name) : 0, i;
    if (!d) { vi_fail_(0, "dir is NULL (did vi_dir_open fail? vi_error() said why)", (const char *)0, (const char *)0); return (vi_db *)0; }
    for (i = 0; i < n; ++i) {
        char ch = name[i];
        if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_' || ch == '-')) break;
    }
    if (n < 1 || n > 64 || i < n) {
        vi_fail_(0, "a tenant name is 1 to 64 of A-Z a-z 0-9 _ - (so it never leaves the directory)", (const char *)0, (const char *)0);
        return (vi_db *)0;
    }
    pthread_mutex_lock(&d->mu);
    for (t = d->head; t; t = t->next) if (!strcmp(t->name, name)) break;
    if (!t) {                                                    /* first use: open it, or create it there */
        char *p = (char *)malloc(strlen(d->path) + n + 6);
        t = (vi_ten *)calloc(1, sizeof(vi_ten));
        if (p && t) {
            sprintf(p, "%s/%s.vxi", d->path, name);
            t->db = vi_open(p, d->dim, d->flags);                /* its reason stays in vi_error() when it fails */
        } else vi_fail_(0, "out of memory", (const char *)0, (const char *)0);
        free(p);
        if (t && t->db) { memcpy(t->name, name, n + 1); t->next = d->head; d->head = t; }
        else { free(t); t = (vi_ten *)0; }
    }
    pthread_mutex_unlock(&d->mu);
    return t ? t->db : (vi_db *)0;
}

VI_API void vi_dir_close(vi_dir *d)
{
    vi_ten *t, *nx;
    if (!d) return;
    for (t = d->head; t; t = nx) { nx = t->next; vi_close(t->db); free(t); }
    pthread_mutex_destroy(&d->mu);
    free(d->path);
    free(d);
}

VI_API int vi_dim(const vi_db *db) { return db ? (int)db->s->dim : vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0); }

VI_API vi_i64 vi_count(vi_db *db) { return db ? (vi_i64)store_count(db->s) : (vi_i64)vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0); }

VI_API int vi_add(vi_db *db, vi_i64 id, const float *vec)
{
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (vi_vec_ok_(db, vec, "the vector")) return -1;
    return store_put(db->s, vec, (i64)id, 0) == 0 ? 0 : vi_fail_(-1, "out of memory", (const char *)0, (const char *)0);
}

VI_API int vi_add_batch(vi_db *db, int n, const vi_i64 *ids, const float *vecs)
{
    int i, ok = 0, first = 1;
    char nb[24], ib[24], why[VI_EMAX];
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (n < 0 || (n > 0 && (!ids || !vecs))) return vi_fail_(-1, "n must be 0 or more, with ids and vecs not NULL", (const char *)0, (const char *)0);
    for (i = 0; i < n; ++i) {
        const float *v = vecs + (size_t)i * (size_t)db->s->dim;
        if (vi_add(db, ids[i], v) == 0) { ++ok; continue; }
        if (first) {                                             /* the first refusal is the one vi_error() names */
            first = 0;
            strcpy(why, vi_error());
            sprintf(nb, "%d", i);
            vi_fail_(0, "row ", nb, " (id ");
            {   char *e = vi_ebuf_();
                if (e && strlen(e) + strlen(vi_i64s_(ib, ids[i])) + strlen(why) + 4 < VI_EMAX) { strcat(e, ib); strcat(e, "): "); strcat(e, why); } }
        }
    }
    return ok;
}

VI_API int vi_delete(vi_db *db, vi_i64 id) { return db ? store_delete(db->s, (i64)id) : vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0); }

VI_API int vi_delete_range(vi_db *db, vi_i64 lo, vi_i64 hi)
{
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (lo > hi) return vi_fail_(-1, "lo is above hi", (const char *)0, (const char *)0);
    return store_delete_range(db->s, (i64)lo, (i64)hi);
}

VI_API int vi_get(vi_db *db, vi_i64 id, float *out)
{
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (!out) return vi_fail_(-1, "out is NULL (it takes dim floats)", (const char *)0, (const char *)0);
    return store_get_vec(db->s, (i64)id, out) == 0 ? 0 : vi_norow_(db, id);
}

static i32 vi_len_(const char *text, int len) { return len < 0 ? (text ? (i32)strlen(text) : 0) : (i32)len; }

VI_API void vi_set_aggression(vi_db *db, float a) { if (db) store_set_aggression(db->s, a); }

VI_API int vi_set_memory(vi_db *db, int tier)
{
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (tier < 0 || tier > 2) return vi_fail_(-1, "tier is 0, 1 or 2", (const char *)0, (const char *)0);
    return store_set_memory(db->s, (i32)tier) == 0 ? 0
         : vi_fail_(-1, "the memlock limit refused part of it (ulimit -l, systemd's LimitMEMLOCK=): that part is only read ahead", (const char *)0, (const char *)0);
}
VI_API int vi_set_parity(vi_db *db, int data, int parity)
{
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    return store_set_parity(db->s, (i32)data, (i32)parity) == 0 ? 0 : vi_fail_(-1, "data is 1 to 128 blocks and parity 0 to 64", (const char *)0, (const char *)0);
}
VI_API void vi_set_verify(int level) { store_set_verify((i32)level); }
VI_API void vi_set_check_reads(vi_db *db, int on) { if (db) store_set_check_reads(db->s, (i32)on); }
VI_API int vi_integrity(vi_db *db, vi_health *h)
{
    i64 b, d, f, rb, rh;
    i32 j;
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (!h) return vi_fail_(-1, "h is NULL (it takes a vi_health)", (const char *)0, (const char *)0);
    store_integrity(db->s, &b, &d, &f, &rb, &rh, &j);
    h->blocks = (vi_i64)b; h->bad = (vi_i64)d; h->fixed = (vi_i64)f;
    h->rows_bad = (vi_i64)rb; h->rows_healed = (vi_i64)rh; h->journal_damaged = (int)j;
    return 0;
}
VI_API vi_i64 vi_scrub(const char *path, int repair, vi_i64 *fixed)
{
    i64 f = 0, r;
    if (!path) return vi_fail_(-1, "path is NULL", (const char *)0, (const char *)0);
    r = store_scrub(path, (i32)repair, &f);
    if (fixed) *fixed = (vi_i64)f;
    if (r == -1) vi_fail_(0, path, " cannot be read", (const char *)0);
    else if (r == -2) vi_fail_(0, path, " has no checksums: it was written before v9 (vi_save writes it again with them)", (const char *)0);
    else if (r == -3) vi_fail_(0, path, ": neither copy of its table of digests can be read", (const char *)0);
    return (vi_i64)r;
}

VI_API void vi_memory(vi_db *db, vi_i64 *hot, vi_i64 *cold)
{
    i64 h = 0, c = 0;
    if (db) store_tiers(db->s, &h, &c);
    if (hot) *hot = (vi_i64)h;
    if (cold) *cold = (vi_i64)c;
}

VI_API int vi_add_text(vi_db *db, vi_i64 id, const float *vec, const char *text, int len)
{
    i32 n;
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (vi_vec_ok_(db, vec, "the vector")) return -1;
    if (!text && len != 0) return vi_fail_(-1, "text is NULL (len 0 stores the row without one)", (const char *)0, (const char *)0);
    n = vi_len_(text, len);
    if (n > STORE_TXMAX) return vi_fail_(-1, "a text is at most 16 MB - 1 bytes: cut it into chunks", (const char *)0, (const char *)0);
    return store_put_text(db->s, vec, (i64)id, 0, text, n) == 0 ? 0 : vi_fail_(-1, "out of memory", (const char *)0, (const char *)0);
}

VI_API int vi_get_text(vi_db *db, vi_i64 id, char *buf, int cap)
{
    int r;
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (cap < 0 || (cap > 0 && !buf)) return vi_fail_(-1, "buf is NULL or cap below 0", (const char *)0, (const char *)0);
    r = store_get_text(db->s, (i64)id, buf, (i32)cap);
    return r >= 0 ? r : vi_norow_(db, id);
}

struct vi_batch { StoreWb w; };

VI_API vi_batch *vi_batch_begin(vi_db *db)
{
    vi_batch *b;
    if (!db) { vi_fail_(0, VI_NODB, (const char *)0, (const char *)0); return (vi_batch *)0; }
    if (!(b = (vi_batch *)malloc(sizeof(vi_batch)))) { vi_fail_(0, "out of memory", (const char *)0, (const char *)0); return (vi_batch *)0; }
    store_batch_init(&b->w, db->s);
    return b;
}

#define VI_NOB "batch is NULL (did vi_batch_begin fail? vi_error() said why)"
#define VI_BADB " - the whole batch is refused at its commit"

VI_API int vi_batch_add(vi_batch *b, vi_i64 id, const float *vec)
{
    vi_db t;
    if (!b) return vi_fail_(-1, VI_NOB, (const char *)0, (const char *)0);
    t.s = b->w.s;
    if (vi_vec_ok_(&t, vec, "the vector")) { b->w.bad = 1; { char *e = vi_ebuf_(); if (e && strlen(e) + strlen(VI_BADB) < VI_EMAX) strcat(e, VI_BADB); } return -1; }
    return store_batch_put(&b->w, vec, (i64)id, 0) == 0 ? 0 : vi_fail_(-1, "out of memory, or the batch is over 1 GB", VI_BADB, (const char *)0);
}

VI_API int vi_batch_add_text(vi_batch *b, vi_i64 id, const float *vec, const char *text, int len)
{
    vi_db t;
    if (!b) return vi_fail_(-1, VI_NOB, (const char *)0, (const char *)0);
    t.s = b->w.s;
    if (vi_vec_ok_(&t, vec, "the vector")) { b->w.bad = 1; { char *e = vi_ebuf_(); if (e && strlen(e) + strlen(VI_BADB) < VI_EMAX) strcat(e, VI_BADB); } return -1; }
    if ((!text && len != 0) || vi_len_(text, len) > STORE_TXMAX) { b->w.bad = 1; return vi_fail_(-1, "text is NULL, or longer than 16 MB - 1 bytes", VI_BADB, (const char *)0); }
    return store_batch_put_text(&b->w, vec, (i64)id, 0, text, vi_len_(text, len)) == 0 ? 0
         : vi_fail_(-1, "out of memory, or the batch is over 1 GB", VI_BADB, (const char *)0);
}

VI_API int vi_batch_delete(vi_batch *b, vi_i64 id) { return b ? store_batch_delete(&b->w, (i64)id) : vi_fail_(-1, VI_NOB, (const char *)0, (const char *)0); }

VI_API int vi_batch_delete_range(vi_batch *b, vi_i64 lo, vi_i64 hi)
{
    if (!b) return vi_fail_(-1, VI_NOB, (const char *)0, (const char *)0);
    if (lo > hi) { b->w.bad = 1; return vi_fail_(-1, "lo is above hi", VI_BADB, (const char *)0); }
    return store_batch_delete_range(&b->w, (i64)lo, (i64)hi);
}

VI_API int vi_batch_set_num(vi_batch *b, vi_i64 id, vi_i64 num) { return b ? store_batch_set_num(&b->w, (i64)id, (i64)num) : vi_fail_(-1, VI_NOB, (const char *)0, (const char *)0); }

VI_API int vi_batch_commit(vi_batch *b, int sync)
{
    i32 r;
    if (!b) return vi_fail_(-1, VI_NOB, (const char *)0, (const char *)0);
    r = store_batch_commit(&b->w, (i32)sync);
    store_batch_free(&b->w);
    free(b);
    if (r == -1) vi_fail_(0, "the batch was refused, nothing applied: one of its calls returned -1 (vi_error() said why then)", (const char *)0, (const char *)0);
    else if (r == -2) vi_fail_(0, "the batch was applied, but the fsync failed: it is not on the disk yet (vi_sync tries again)", (const char *)0, (const char *)0);
    return r;
}

VI_API void vi_batch_abort(vi_batch *b)
{
    if (!b) return;
    store_batch_free(&b->w);
    free(b);
}

VI_API int vi_range(vi_db *db, vi_i64 lo, vi_i64 hi, vi_i64 *ids, int max)
{
    int r;
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (max < 0 || (max > 0 && !ids)) return vi_fail_(-1, "ids is NULL or max below 0", (const char *)0, (const char *)0);
    if (lo > hi) return 0;
    r = store_range(db->s, (i64)lo, (i64)hi, (i64 *)ids, (i32)max);
    return r >= 0 ? r : vi_fail_(-1, "out of memory", (const char *)0, (const char *)0);
}

/* the checks every search makes: 0, or -1 with the reason */
static int vi_search_ok_(vi_db *db, const float *query, int k, const vi_i64 *ids, const float *scores)
{
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (k < 1) return vi_fail_(-1, "k must be 1 or more", (const char *)0, (const char *)0);
    if (!ids || !scores) return vi_fail_(-1, "ids and scores must hold k values each", (const char *)0, (const char *)0);
    return query ? vi_vec_ok_(db, query, "the query") : 0;
}

VI_API int vi_search_hybrid(vi_db *db, const float *query, const char *text, int tlen, int k, vi_i64 *ids, float *scores)
{
    if (vi_search_ok_(db, query, k, ids, scores)) return -1;
    if (!query && (!text || vi_len_(text, tlen) == 0)) return vi_fail_(-1, "give the query vector, the words, or both", (const char *)0, (const char *)0);
    return store_search_hybrid(db->s, query, text, vi_len_(text, tlen), (i32)k, (i64 *)ids, scores);
}

VI_API int vi_set_tag(vi_db *db, vi_i64 id, vi_i64 tag)
{
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    return store_set_meta(db->s, (i64)id, (i64)tag) >= 0 ? 0 : vi_norow_(db, id);
}

VI_API int vi_get_tag(vi_db *db, vi_i64 id, vi_i64 *tag)
{
    i64 m = 0;
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (store_get_meta(db->s, (i64)id, &m) != 0) return vi_norow_(db, id);
    if (tag) *tag = (vi_i64)m;
    return 0;
}

VI_API int vi_set_num(vi_db *db, vi_i64 id, vi_i64 num)
{
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    return store_set_num(db->s, (i64)id, (i64)num) == 0 ? 0 : vi_norow_(db, id);
}

VI_API int vi_get_num(vi_db *db, vi_i64 id, vi_i64 *num)
{
    i64 v = 0;
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (store_get_num(db->s, (i64)id, &v) != 0) return vi_norow_(db, id);
    if (num) *num = (vi_i64)v;
    return 0;
}

VI_API int vi_search_range(vi_db *db, const float *query, int k, vi_i64 lo, vi_i64 hi, vi_i64 *ids, float *scores)
{
    StoreCnf c;
    int n;
    if (vi_search_ok_(db, query, k, ids, scores)) return -1;
    if (!query) return vi_fail_(-1, "the query is NULL", (const char *)0, (const char *)0);
    if (lo > hi) return 0;
    memset(&c, 0, sizeof(c));
    c.num = 1; c.lo = (i64)lo; c.hi = (i64)hi;
    n = store_search_cnf(db->s, query, (i32)k, (i64 *)ids, scores, &c);
    vi_cos_(db, scores, n);
    return n;
}

VI_API int vi_search(vi_db *db, const float *query, int k, vi_i64 *ids, float *scores)
{
    int n;
    if (vi_search_ok_(db, query, k, ids, scores)) return -1;
    if (!query) return vi_fail_(-1, "the query is NULL", (const char *)0, (const char *)0);
    n = store_search(db->s, query, (i32)k, (i64 *)ids, scores);
    vi_cos_(db, scores, n);
    return n;
}

#define VI_GROUP_MAX 65536   /* the most rows a grouped search re-ranks before it gives up on finding k groups */

/* best first from store_search, so the first hit of a group is its best: keep it, skip the rest of the group; too
 * few groups, ask for four times as many rows, up to VI_GROUP_MAX or the whole store */
VI_API int vi_search_grouped(vi_db *db, const float *query, int k, int shift, vi_i64 *ids, float *scores)
{
    i64 total, *ci;
    f32 *cs;
    i32 want, n, have, i, j;
    if (vi_search_ok_(db, query, k, ids, scores)) return -1;
    if (!query) return vi_fail_(-1, "the query is NULL", (const char *)0, (const char *)0);
    if (shift < 0 || shift > 63) return vi_fail_(-1, "shift must be 0 to 63", (const char *)0, (const char *)0);
    if (shift == 0) return vi_search(db, query, k, ids, scores);
    total = store_count(db->s);
    want = k > VI_GROUP_MAX / 4 ? VI_GROUP_MAX : k * 4;
    if (want < 64) want = 64;
    for (;;) {
        if ((i64)want > total) want = total > 0 ? (i32)total : 1;
        ci = (i64 *)malloc((size_t)want * sizeof(i64));
        cs = (f32 *)malloc((size_t)want * sizeof(f32));
        if (!ci || !cs) { free(ci); free(cs); return vi_fail_(-1, "out of memory", (const char *)0, (const char *)0); }
        n = store_search(db->s, query, want, ci, cs);
        for (i = have = 0; i < n && have < k; ++i) {
            u64 g = (u64)ci[i] >> shift;
            for (j = 0; j < have && ((u64)ids[j] >> shift) != g; ++j) {}
            if (j == have) { ids[have] = (vi_i64)ci[i]; scores[have] = cs[i]; ++have; }
        }
        free(ci); free(cs);
        if (have >= k || n < want || (i64)want >= total || want >= VI_GROUP_MAX) break;
        want = want > VI_GROUP_MAX / 4 ? VI_GROUP_MAX : want * 4;
    }
    vi_cos_(db, scores, have);
    return have;
}

VI_API int vi_search_tagged(vi_db *db, const float *query, int k, vi_i64 all, vi_i64 any, vi_i64 *ids, float *scores)
{
    int n;
    if (vi_search_ok_(db, query, k, ids, scores)) return -1;
    if (!query) return vi_fail_(-1, "the query is NULL", (const char *)0, (const char *)0);
    n = store_search_filter(db->s, query, (i32)k, (i64 *)ids, scores, (i64)all, (i64)any);
    vi_cos_(db, scores, n);
    return n;
}

VI_API int vi_search_batch(vi_db *db, const float *queries, int nq, int k, vi_i64 *ids, float *scores, int threads)
{
    int q;
    if (vi_search_ok_(db, (const float *)0, k, ids, scores)) return -1;
    if (nq < 0 || (nq > 0 && !queries)) return vi_fail_(-1, "queries is NULL or nq below 0", (const char *)0, (const char *)0);
    for (q = 0; q < nq; ++q) if (vi_vec_ok_(db, queries + (size_t)q * (size_t)db->s->dim, "a query")) {
        char nb[24], why[VI_EMAX];
        strcpy(why, vi_error());
        sprintf(nb, "%d: ", q);
        return vi_fail_(-1, "query ", nb, why);
    }
    if (nq > 0) store_search_batch(db->s, queries, (i32)nq, (i32)k, (i64 *)ids, scores, (i32)threads);
    vi_cos_(db, scores, nq * k);
    return 0;
}

VI_API void vi_build(vi_db *db) { if (db) store_build(db->s); }

VI_API int vi_set_recall(vi_db *db, float target)
{
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (!(target >= 0.0f && target < 1.0f)) return vi_fail_(-1, "target is a recall below 1, such as 0.95 (0 turns it off)", (const char *)0, (const char *)0);
    store_set_target_recall(db->s, target);
    return 0;
}

VI_API int vi_set_layout(vi_db *db, int code_bits, int row_bytes)
{
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (code_bits != 4 && code_bits != 2 && code_bits != 1) return vi_fail_(-1, "code_bits is 4, 2 or 1", (const char *)0, (const char *)0);
    if (row_bytes != 4 && row_bytes != 1) return vi_fail_(-1, "row_bytes is 4 (f32 rows) or 1 (int8 rows)", (const char *)0, (const char *)0);
    if (store_count(db->s) > 0 || store_set_vec_bytes(db->s, (i32)row_bytes) != 0)
        return vi_fail_(-1, "the layout is set on an empty store, before the first add", (const char *)0, (const char *)0);
    store_set_codes(db->s, (i32)code_bits);
    return 0;
}

VI_API int vi_set_cascade(vi_db *db, int on)
{
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    return store_set_cascade(db->s, (i32)on) == 0 ? 0
         : vi_fail_(-1, "the cascade needs 4-bit codes (vi_set_layout(db, 4, ...)), or memory ran out", (const char *)0, (const char *)0);
}

/* the file a save, sync or snapshot goes to: path, else the one the store came from */
static const char *vi_file_(vi_db *db, const char *path)
{
    const char *p = path ? path : db->path;
    if (!p) vi_fail_(0, "a store in memory has no file: give a path", (const char *)0, (const char *)0);
    return p;
}

static int vi_wfail_(const char *what, const char *p)
{
    int en = errno;
    vi_fail_(-1, what, p, en ? ": " : " (disk full, no permission, or a read-only file system)");
    if (en) { char *e = vi_ebuf_(); if (e && strlen(e) + strlen(strerror(en)) < VI_EMAX) strcat(e, strerror(en)); }
    return -1;
}

VI_API int vi_save(vi_db *db, const char *path)
{
    const char *p;
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (!(p = vi_file_(db, path))) return -1;
    errno = 0;
    return store_save(db->s, p) == 0 ? 0 : vi_wfail_("cannot write ", p);
}

VI_API int vi_sync(vi_db *db)
{
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (!db->path) return vi_fail_(-1, "a store in memory has no file to sync: vi_save it to a path", (const char *)0, (const char *)0);
    errno = 0;
    return store_sync(db->s, db->path) == 0 ? 0 : vi_wfail_("cannot sync ", db->path);
}

VI_API int vi_snapshot(vi_db *db, const char *path)
{
    const char *p;
    int r;
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (!(p = vi_file_(db, path))) return -1;
    errno = 0;
    r = store_snapshot(db->s, p);
    return r >= 0 ? r : vi_wfail_("cannot start a snapshot of ", p);
}

VI_API int vi_snapshot_wait(vi_db *db, int block) { return db ? store_snapshot_poll(db->s, block) : vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0); }

/* ---- string keys, labels ---- */

VI_API vi_i64 vi_key(const char *s, int len)
{
    u8 h[32];
    u64 v = 0;
    i32 i;
    if (!s) return (vi_i64)vi_fail_(-1, "the key is NULL", (const char *)0, (const char *)0);
    b3_hash(s, (size_t)vi_len_(s, len), h);
    for (i = 7; i >= 0; --i) v = v << 8 | (u64)h[i];
    return (vi_i64)(v & ~((u64)1 << 63));                       /* never negative: -1 stays the error */
}

/* "the label NAME ..." for the name a list holds at p (up to its comma or space) */
static int vi_badlabel_(int r, const char *p, const char *why)
{
    char nm[STORE_LABEL_NAME + 8];
    size_t n = 0;
    if (p) while (p[n] && p[n] != ',' && p[n] != ' ' && n < STORE_LABEL_NAME + 4) { nm[n] = p[n]; ++n; }
    nm[n] = 0;
    return vi_fail_(r, "label \"", nm, why);
}

VI_API int vi_label(vi_db *db, const char *name, const char *parent)
{
    i32 r;
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (!name) return vi_fail_(-1, "the name is NULL", (const char *)0, (const char *)0);
    r = store_label(db->s, name, parent);
    if (r == -2) return vi_fail_(-1, "that link would make a cycle: the parent is below the label already", (const char *)0, (const char *)0);
    if (r < 0) {
        if (!store_label_name_ok_(name, strlen(name)) || (parent && !store_label_name_ok_(parent, strlen(parent))))
            return vi_fail_(-1, "a label is 1 to 63 of A-Z a-z 0-9 _ - . / :", (const char *)0, (const char *)0);
        return vi_fail_(-1, "all 64 labels are in use", (const char *)0, (const char *)0);
    }
    return 0;
}

static int vi_label_row_(vi_db *db, vi_i64 id, const char *labels, int on)
{
    const char *bad = (const char *)0;
    i32 r;
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (!labels) return vi_fail_(-1, "the labels are NULL", (const char *)0, (const char *)0);
    r = store_label_row(db->s, (i64)id, labels, (i32)on, &bad);
    if (r == -1) return vi_norow_(db, id);
    if (r == -2) return vi_badlabel_(-1, bad, "\" is not a name (1 to 63 of A-Z a-z 0-9 _ - . / :), or all 64 labels are in use");
    return 0;
}
VI_API int vi_label_add(vi_db *db, vi_i64 id, const char *labels) { return vi_label_row_(db, id, labels, 1); }
VI_API int vi_label_remove(vi_db *db, vi_i64 id, const char *labels) { return vi_label_row_(db, id, labels, 0); }

VI_API int vi_search_labeled(vi_db *db, const float *query, int k, const char *all, const char *any, const char *none,
                             vi_i64 *ids, float *scores)
{
    StoreCnf c;
    const char *bad = (const char *)0;
    i32 r;
    int n;
    if (vi_search_ok_(db, query, k, ids, scores)) return -1;
    if (!query) return vi_fail_(-1, "the query is NULL", (const char *)0, (const char *)0);
    r = store_label_cnf(db->s, all, any, none, &c, &bad);
    if (r == -1) return vi_badlabel_(-1, bad, "\" does not exist (vi_label or vi_label_add makes it)");
    if (r == -2) return vi_fail_(-1, "at most 15 labels in all", (const char *)0, (const char *)0);
    n = store_search_cnf(db->s, query, (i32)k, (i64 *)ids, scores, &c);
    vi_cos_(db, scores, n);
    return n;
}
