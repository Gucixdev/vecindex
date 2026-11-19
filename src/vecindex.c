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
#include <dirent.h>

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
    if (!any) return vi_fail_(-1, what, " is all zeros: it has no direction to compare", (const char *)0);
    if (db->s->metric == 3) {                                        /* a point of the hyperboloid's upper sheet */
        f64 sp = 0.0;
        for (i = 1; i < db->s->dim; ++i) sp += (f64)v[i] * v[i];
        if (!(v[0] > 0.0f) || !((f64)v[0] * v[0] > sp))
            return vi_fail_(-1, what, " is not a point of the Lorentz model: x[0] > 0 and x0^2 > x1^2 + ... are needed", (const char *)0);
    }
    return 0;
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

/* The recall@10 a store opened without VI_MANUAL keeps by default: for RAG over many documents speed matters more
 * than the last few hits (GloVe-100: about 1.6 times the queries a second of recall 0.95, docs/benchmarks.md);
 * vi_set_recall changes it, 0 turns the tuning off (the build's default probes: about 0.96 there). */
#ifndef VI_RECALL_DEFAULT
#define VI_RECALL_DEFAULT 0.90f
#endif

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
    {   int want = (flags & VI_DOT) ? 1 : ((flags & VI_L2) ? 2 : ((flags & VI_LORENTZ) ? 3 : 0));   /* asked for, or the file's */
        int have = db->s->metric;
        static const char *const mn[4] = { "cosine", "dot", "L2", "Lorentz" };
        if (((flags & VI_DOT) != 0) + ((flags & VI_L2) != 0) + ((flags & VI_LORENTZ) != 0) > 1) {
            vi_fail_(0, "VI_DOT, VI_L2 and VI_LORENTZ: one of them, a store has one metric", (const char *)0, (const char *)0);
            store_destroy(db->s); arena_free(&db->arena); free(db->path); free(db); return (vi_db *)0;
        }
        if (exists && want && want != have) {
            vi_fail_(0, path, " is a ", mn[have]);
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
    if (!(flags & VI_MANUAL)) {
        store_set_live(db->s, 1, 1.0f);                          /* rows are indexed as they come: no vi_build to forget */
        store_set_target_recall(db->s, VI_RECALL_DEFAULT);       /* the fewest probes that keep recall@10 at 0.90 */
    }
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

/* the k best of every store in the directory: each searched as vi_search, the hits merged by score (smaller first in
 * an L2 store, bigger first otherwise); stores of another dimension or metric than the first are an error */
VI_API int vi_dir_search(vi_dir *d, const float *query, int k, vi_i64 *ids, float *scores)
{
    DIR *dd;
    struct dirent *e;
    vi_i64 *ci;
    float *cs;
    int have = 0, n, i, j, metric = -1, dim = -1, r = 0;
    if (!d) return vi_fail_(-1, "dir is NULL (did vi_dir_open fail? vi_error() said why)", (const char *)0, (const char *)0);
    if (k < 1) return vi_fail_(-1, "k must be 1 or more", (const char *)0, (const char *)0);
    if (!query || !ids || !scores) return vi_fail_(-1, "the query is NULL, or ids and scores do not hold k values each", (const char *)0, (const char *)0);
    if (!(dd = opendir(d->path))) return vi_fail_(-1, d->path, ": cannot list it", (const char *)0);
    ci = (vi_i64 *)malloc((size_t)k * sizeof(vi_i64));
    cs = (float *)malloc((size_t)k * sizeof(float));
    if (!ci || !cs) { free(ci); free(cs); closedir(dd); return vi_fail_(-1, "out of memory", (const char *)0, (const char *)0); }
    while (r == 0 && (e = readdir(dd)) != (struct dirent *)0) {
        char name[72];
        size_t L = strlen(e->d_name);
        vi_db *db;
        if (L < 5 || L > 68 || strcmp(e->d_name + L - 4, ".vxi") != 0) continue;   /* PATH.vxi.j and the rest are not stores */
        memcpy(name, e->d_name, L - 4); name[L - 4] = 0;
        if (!(db = vi_dir_get(d, name))) { r = -1; break; }      /* vi_error() says why */
        if (metric < 0) { metric = db->s->metric; dim = db->s->dim; }
        else if (db->s->metric != metric || db->s->dim != dim) {
            r = vi_fail_(-1, e->d_name, " has another dimension or metric than the other stores", (const char *)0);
            break;
        }
        if ((n = vi_search(db, query, k, ci, cs)) < 0) { r = -1; break; }
        for (i = 0; i < n; ++i) {                                /* into the merged top k, best first */
            float x = cs[i];
            if (have == k && (metric >= 2 ? x >= scores[k - 1] : x <= scores[k - 1])) break;   /* the rest are worse */
            for (j = have < k ? have++ : k - 1; j > 0 && (metric >= 2 ? scores[j - 1] > x : scores[j - 1] < x); --j) {
                ids[j] = ids[j - 1]; scores[j] = scores[j - 1];
            }
            ids[j] = ci[i]; scores[j] = x;
        }
    }
    closedir(dd);
    free(ci); free(cs);
    return r ? r : have;
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

VI_API int vi_update(vi_db *db, vi_i64 id, const float *vec)
{
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (vi_vec_ok_(db, vec, "the vector")) return -1;
    return store_update(db->s, vec, (i64)id) == 0 ? 0 : vi_norow_(db, id);
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
VI_API int vi_lens(vi_db *db, vi_i64 *done, vi_i64 *total)
{
    i64 d, t;
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    store_lens_info(db->s, &d, &t);
    if (done) *done = (vi_i64)d;
    if (total) *total = (vi_i64)t;
    return 0;
}
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

VI_API int vi_batch_update(vi_batch *b, vi_i64 id, const float *vec)
{
    vi_db t;
    if (!b) return vi_fail_(-1, VI_NOB, (const char *)0, (const char *)0);
    t.s = b->w.s;
    if (vi_vec_ok_(&t, vec, "the vector")) { b->w.bad = 1; { char *e = vi_ebuf_(); if (e && strlen(e) + strlen(VI_BADB) < VI_EMAX) strcat(e, VI_BADB); } return -1; }
    return store_batch_update(&b->w, vec, (i64)id) == 0 ? 0 : vi_fail_(-1, "out of memory, or the batch is over 1 GB", VI_BADB, (const char *)0);
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

VI_API int vi_batch_set_ns(vi_batch *b, vi_i64 id, vi_i64 ns) { return b ? store_batch_set_ns(&b->w, (i64)id, (i64)ns) : vi_fail_(-1, VI_NOB, (const char *)0, (const char *)0); }
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

/* ---- payload (store_meta.c) ---- */

typedef char vi_filter_fits_[sizeof(vi_filter) >= sizeof(StoreFlt) ? 1 : -1];   /* the caller's value holds the compiled one */

static int vi_fname_(vi_db *db, const char *name)
{
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (!name) return vi_fail_(-1, "the field's name is NULL", (const char *)0, (const char *)0);
    return 0;
}

static int vi_fset_(vi_db *db, vi_i64 id, const char *name, i32 type, u64 code)
{
    i32 r;
    if (vi_fname_(db, name)) return -1;
    if (db->s->ro) return vi_fail_(-1, "the store is open read-only", (const char *)0, (const char *)0);
    r = store_field_set(db->s, (i64)id, name, (i32)strlen(name), type, code);
    if (r == -1) return vi_norow_(db, id);
    if (r == -2) return vi_fail_(-1, "the field \"", name, "\" exists with another type");
    if (r == -3) return vi_fail_(-1, "the store has 16 fields already (STORE_FIELDS): no room for \"", name, "\"");
    if (r == -4) return vi_fail_(-1, "not a field name (1 to 31 of A-Z a-z 0-9 _, no digit first, not and/or/not/in/has/true/false): \"", name, "\"");
    if (r == -5) return vi_fail_(-1, "out of memory", (const char *)0, (const char *)0);
    return 0;
}

static int vi_bfset_(vi_batch *b, vi_i64 id, const char *name, i32 type, u64 code)
{
    i32 r;
    if (!b) return vi_fail_(-1, VI_NOB, (const char *)0, (const char *)0);
    if (!name) return vi_fail_(-1, "the field's name is NULL", (const char *)0, (const char *)0);
    r = store_batch_set_field(&b->w, (i64)id, name, (i32)strlen(name), type, code);
    if (r == -2) return vi_fail_(-1, "the field \"", name, "\" exists with another type");
    if (r == -3) return vi_fail_(-1, "the store has 16 fields already (STORE_FIELDS): no room for \"", name, "\"");
    if (r == -4) return vi_fail_(-1, "not a field name (1 to 31 of A-Z a-z 0-9 _, no digit first, not and/or/not/in/has/true/false): \"", name, "\"");
    if (r < 0) return vi_fail_(-1, "out of memory", (const char *)0, (const char *)0);
    return 0;
}
VI_API int vi_batch_set_int(vi_batch *b, vi_i64 id, const char *name, vi_i64 v)
{
    if ((u64)v == ((u64)1 << 63)) return vi_fail_(-1, "INT64_MIN cannot be stored in a field (its code means \"not set\")", (const char *)0, (const char *)0);
    return vi_bfset_(b, id, name, STORE_FT_INT, store_enc_int_((i64)v));
}
VI_API int vi_batch_set_float(vi_batch *b, vi_i64 id, const char *name, double v)
{
    if (v != v) return vi_fail_(-1, "NaN cannot be stored in a field", (const char *)0, (const char *)0);
    return vi_bfset_(b, id, name, STORE_FT_FLOAT, store_enc_f64_(v));
}
VI_API int vi_batch_set_str(vi_batch *b, vi_i64 id, const char *name, const char *v, int len)
{
    if (!v) return vi_fail_(-1, "the string is NULL", (const char *)0, (const char *)0);
    return vi_bfset_(b, id, name, STORE_FT_STR, store_enc_str_(v, vi_len_(v, len)));
}

VI_API int vi_field(vi_db *db, const char *name, int type)
{
    i32 f;
    if (vi_fname_(db, name)) return -1;
    if (type != VI_INT && type != VI_FLOAT && type != VI_STR) return vi_fail_(-1, "the type is VI_INT, VI_FLOAT or VI_STR", (const char *)0, (const char *)0);
    if (db->s->ro) return vi_fail_(-1, "the store is open read-only", (const char *)0, (const char *)0);
    f = store_field_decl(db->s, name, (i32)strlen(name), (i32)type);
    if (f == -2) return vi_fail_(-1, "the field \"", name, "\" exists with another type");
    if (f == -3) return vi_fail_(-1, "the store has 16 fields already (STORE_FIELDS): no room for \"", name, "\"");
    if (f == -4) return vi_fail_(-1, "not a field name (1 to 31 of A-Z a-z 0-9 _, no digit first, not and/or/not/in/has/true/false): \"", name, "\"");
    if (f < 0) return vi_fail_(-1, "out of memory", (const char *)0, (const char *)0);
    return 0;
}

VI_API int vi_set_int(vi_db *db, vi_i64 id, const char *name, vi_i64 v)
{
    if ((u64)v == ((u64)1 << 63)) return vi_fail_(-1, "INT64_MIN cannot be stored in a field (its code means \"not set\")", (const char *)0, (const char *)0);
    return vi_fset_(db, id, name, STORE_FT_INT, store_enc_int_((i64)v));
}

VI_API int vi_set_float(vi_db *db, vi_i64 id, const char *name, double v)
{
    if (v != v) return vi_fail_(-1, "NaN cannot be stored in a field", (const char *)0, (const char *)0);
    return vi_fset_(db, id, name, STORE_FT_FLOAT, store_enc_f64_(v));
}

VI_API int vi_set_str(vi_db *db, vi_i64 id, const char *name, const char *v, int len)
{
    if (!v) return vi_fail_(-1, "the string is NULL", (const char *)0, (const char *)0);
    return vi_fset_(db, id, name, STORE_FT_STR, store_enc_str_(v, vi_len_(v, len)));
}

VI_API int vi_unset(vi_db *db, vi_i64 id, const char *name)
{
    i32 t;
    u64 c;
    if (vi_fname_(db, name)) return -1;
    if (store_field_get(db->s, (i64)id, name, (i32)strlen(name), &t, &c) == -2) t = STORE_FT_INT;   /* no such field: nothing to unset */
    return vi_fset_(db, id, name, t, 0);
}

static int vi_fget_(vi_db *db, vi_i64 id, const char *name, i32 want, u64 *c)
{
    i32 t, r;
    if (vi_fname_(db, name)) return -1;
    r = store_field_get(db->s, (i64)id, name, (i32)strlen(name), &t, c);
    if (r == -1) return vi_norow_(db, id);
    if (r == -2) return vi_fail_(-1, "no field \"", name, "\"");
    if (t != want) return vi_fail_(-1, "the field \"", name, "\" has another type");
    return r;
}

VI_API int vi_get_int(vi_db *db, vi_i64 id, const char *name, vi_i64 *v)
{
    u64 c = 0;
    int r = vi_fget_(db, id, name, STORE_FT_INT, &c);
    if (r == 0 && v) *v = (vi_i64)store_dec_int_(c);
    return r;
}

VI_API int vi_get_float(vi_db *db, vi_i64 id, const char *name, double *v)
{
    u64 c = 0;
    int r = vi_fget_(db, id, name, STORE_FT_FLOAT, &c);
    if (r == 0 && v) *v = store_dec_f64_(c);
    return r;
}

VI_API int vi_get_str_key(vi_db *db, vi_i64 id, const char *name, vi_i64 *key)
{
    u64 c = 0;
    int r = vi_fget_(db, id, name, STORE_FT_STR, &c);
    if (r == 0 && key) *key = (vi_i64)c;
    return r;
}

VI_API int vi_field_info(vi_db *db, int i, char *name, int *type)
{
    char nm[STORE_FIELD_NAME];
    i32 t;
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (store_field_info(db->s, (i32)i, nm, &t) != 0) return -1;
    if (name) memcpy(name, nm, STORE_FIELD_NAME);
    if (type) *type = (int)t;
    return 0;
}

static const StoreFlt *vi_flt_(vi_db *db, const vi_filter *f)
{
    const StoreFlt *x = (const StoreFlt *)(const void *)f;
    if (!f) return x;
    if (x->sid != (u64)(size_t)db->s || x->nop < 1 || x->nop > STORE_FLT_OPS) {
        vi_fail_(-1, "the filter was not compiled for this store (vi_filter_parse)", (const char *)0, (const char *)0);
        return (const StoreFlt *)(const void *)-1;
    }
    return x;
}
#define VI_FLT_BAD(x) ((const void *)(x) == (const void *)-1)

VI_API int vi_filter_parse(vi_db *db, const char *expr, vi_filter *f)
{
    const char *why;
    i32 at;
    char where[64];
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (!expr || !f) return vi_fail_(-1, "the filter or its expression is NULL", (const char *)0, (const char *)0);
    if (store_flt_parse(db->s, expr, (StoreFlt *)(void *)f, &why, &at) == 0) return 0;
    memset(f, 0, sizeof(*f));                                    /* never usable half-made */
    sprintf(where, " (at byte %d)", (int)at);
    return vi_fail_(-1, "filter: ", why ? why : "not a filter", where);
}

VI_API int vi_match(vi_db *db, vi_i64 id, const vi_filter *f)
{
    const StoreFlt *x;
    i32 r;
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (!f) return vi_fail_(-1, "the filter is NULL", (const char *)0, (const char *)0);
    if (VI_FLT_BAD(x = vi_flt_(db, f))) return -1;
    r = store_flt_match(db->s, (i64)id, x);
    return r < 0 ? vi_norow_(db, id) : r;
}

VI_API int vi_search_filter(vi_db *db, const float *query, int k, const vi_filter *f, vi_i64 *ids, float *scores)
{
    const StoreFlt *x;
    int n;
    if (vi_search_ok_(db, query, k, ids, scores)) return -1;
    if (!query) return vi_fail_(-1, "the query is NULL", (const char *)0, (const char *)0);
    if (!f) return vi_fail_(-1, "the filter is NULL (vi_search searches every row)", (const char *)0, (const char *)0);
    if (VI_FLT_BAD(x = vi_flt_(db, f))) return -1;
    n = store_search_flt(db->s, query, (i32)k, x, (i64 *)ids, scores);
    vi_cos_(db, scores, n);
    return n;
}

/* ---- vi_chunk ---- */

static int vi_ws_(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v'; }

/* the best end of a chunk in [lo, hi): after a blank line, a line end, a sentence end, a space; 0 none */
static i32 vi_cut_(const char *t, i32 lo, i32 hi)
{
    i32 i, k;
    for (k = 0; k < 4; ++k)
        for (i = hi - 1; i > lo; --i) {
            if (k == 0 && t[i] == '\n' && t[i - 1] == '\n') return i + 1;
            if (k == 1 && t[i] == '\n') return i + 1;
            if (k == 2 && t[i] == ' ' && (t[i - 1] == '.' || t[i - 1] == '?' || t[i - 1] == '!')) return i + 1;
            if (k == 3 && vi_ws_(t[i])) return i + 1;
        }
    return 0;
}

VI_API int vi_chunk(const char *text, int len, int max_bytes, int overlap, int *offs, int *lens, int *pages, int max)
{
    i32 L, pos = 0, page = 0, n = 0;
    if (!text || max_bytes < 16 || overlap < 0 || overlap > max_bytes / 2 || max < 0 || (max > 0 && (!offs || !lens)))
        return vi_fail_(-1, "text, max_bytes >= 16, overlap 0 to max_bytes / 2, offs and lens for max chunks", (const char *)0, (const char *)0);
    L = vi_len_(text, len);
    while (pos < L) {
        i32 pe = pos, end, a, b;
        if (text[pos] == '\f') { ++pos; ++page; continue; }
        while (pe < L && text[pe] != '\f') ++pe;                /* the end of this page */
        while (pos < pe && vi_ws_(text[pos])) ++pos;
        while (pos < pe) {                                       /* the chunks of the page */
            if (pe - pos <= max_bytes) end = pe;
            else {
                end = vi_cut_(text, pos + max_bytes * 2 / 3, pos + max_bytes);
                if (!end) { end = pos + max_bytes; while (end > pos + 1 && ((u8)text[end] & 0xc0u) == 0x80u) --end; }
            }
            a = pos; b = end;
            while (b > a && vi_ws_(text[b - 1])) --b;
            if (b > a) {
                if (n < max) { offs[n] = (int)a; lens[n] = (int)(b - a); if (pages) pages[n] = (int)page; }
                ++n;
            }
            if (end >= pe) break;
            a = end - overlap;                                   /* the next starts overlap bytes back, on a character */
            while (a < end && ((u8)text[a] & 0xc0u) == 0x80u) ++a;
            pos = a > pos ? a : end;
            while (pos < pe && vi_ws_(text[pos])) ++pos;
        }
        pos = pe;
    }
    return (int)n;
}

/* ---- RAG (store_rag.c) ---- */

static int vi_rag_args_(vi_db *db, const float *query, const char *text, int tlen, float lambda, float dup)
{
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (query && vi_vec_ok_(db, query, "the query")) return -1;
    if (!query && (!text || vi_len_(text, tlen) == 0)) return vi_fail_(-1, "give the query vector, the words, or both", (const char *)0, (const char *)0);
    if (!(lambda >= 0.0f && lambda <= 1.0f)) return vi_fail_(-1, "lambda must be 0 to 1 (1: relevance alone)", (const char *)0, (const char *)0);
    if (!(dup >= 0.0f && dup <= 1.0f)) return vi_fail_(-1, "dup must be 0 to 1 (0: keep near-duplicates)", (const char *)0, (const char *)0);
    return 0;
}

/* a vector alone: the internal scores as callers see them (vi_search); with words: the fused rank as it is */
static void vi_rag_scores_(vi_db *db, const char *text, int tlen, float *sc, int n)
{
    if (!text || vi_len_(text, tlen) == 0) vi_cos_(db, sc, n);
}

VI_API int vi_search_mmr(vi_db *db, const float *query, const char *text, int tlen, const vi_filter *filter, int k,
                         float lambda, float dup, vi_i64 *ids, float *scores)
{
    const StoreFlt *x;
    i32 n;
    if (vi_rag_args_(db, query, text, tlen, lambda, dup)) return -1;
    if (VI_FLT_BAD(x = vi_flt_(db, filter))) return -1;
    if (k < 1) return vi_fail_(-1, "k must be 1 or more", (const char *)0, (const char *)0);
    if (!ids || !scores) return vi_fail_(-1, "ids and scores must hold k values each", (const char *)0, (const char *)0);
    if ((n = store_search_mmr(db->s, query, text, vi_len_(text, tlen), x, (i32)k, lambda, dup, (i64 *)ids, scores)) < 0)
        return vi_fail_(-1, "out of memory", (const char *)0, (const char *)0);
    vi_rag_scores_(db, text, tlen, scores, n);
    return n;
}

VI_API int vi_neighbors(vi_db *db, vi_i64 id, int shift, int before, int after, vi_i64 *ids, int max)
{
    i32 r;
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (shift < 0 || shift > 63) return vi_fail_(-1, "shift must be 0 to 63", (const char *)0, (const char *)0);
    if (!ids || max < 1) return vi_fail_(-1, "ids is NULL or max below 1", (const char *)0, (const char *)0);
    if (store_get_text(db->s, (i64)id, (char *)0, 0) < 0) return vi_norow_(db, id);
    r = store_neighbors(db->s, (i64)id, (i32)shift, (i32)before, (i32)after, (i64 *)ids, (i32)max);
    return r >= 0 ? r : vi_fail_(-1, "out of memory", (const char *)0, (const char *)0);
}

VI_API void vi_ctx_defaults(vi_ctx_opts *o)
{
    if (!o) return;
    o->k = 8; o->shift = 20; o->before = 1; o->after = 1; o->lambda = 0.7f; o->dup = 0.97f; o->filter = (const vi_filter *)0;
}

VI_API int vi_context(vi_db *db, const float *query, const char *text, int tlen, const vi_ctx_opts *o,
                      vi_ctx_span *spans, int max_spans, char *buf, int cap, int *used)
{
    vi_ctx_opts d;
    const StoreFlt *x;
    StoreSpan sp[VI_CTX_HITS_MAX];
    i32 n, i, u = 0;
    if (used) *used = 0;
    vi_ctx_defaults(&d);
    if (o) d = *o;
    if (vi_rag_args_(db, query, text, tlen, d.lambda, d.dup)) return -1;
    if (VI_FLT_BAD(x = vi_flt_(db, d.filter))) return -1;
    if (d.k < 1 || d.k > VI_CTX_HITS_MAX) return vi_fail_(-1, "k must be 1 to 256 (VI_CTX_HITS_MAX)", (const char *)0, (const char *)0);
    if (d.shift < 0 || d.shift > 63) return vi_fail_(-1, "shift must be 0 to 63", (const char *)0, (const char *)0);
    if (d.before < -1 || d.after < -1) return vi_fail_(-1, "before and after are 0 or more, or -1 (the whole document)", (const char *)0, (const char *)0);
    if (!spans || max_spans < 1) return vi_fail_(-1, "spans is NULL or max_spans below 1", (const char *)0, (const char *)0);
    if (cap < 0 || (cap > 0 && !buf)) return vi_fail_(-1, "buf is NULL or cap below 0", (const char *)0, (const char *)0);
    if (max_spans > VI_CTX_HITS_MAX) max_spans = VI_CTX_HITS_MAX;   /* a span holds a hit: never more than k */
    n = store_context(db->s, query, text, vi_len_(text, tlen), x, (const i64 *)0, (const f32 *)0, (i32)d.k, (i32)d.shift, (i32)d.before, (i32)d.after,
                      d.lambda, d.dup, sp, (i32)max_spans, buf, (i32)cap, &u);
    if (n < 0) return vi_fail_(-1, "out of memory", (const char *)0, (const char *)0);
    for (i = 0; i < n; ++i) {
        spans[i].doc = (vi_i64)sp[i].doc; spans[i].first = (vi_i64)sp[i].first; spans[i].last = (vi_i64)sp[i].last;
        spans[i].hit = (vi_i64)sp[i].hit; spans[i].score = sp[i].score; spans[i].chunks = (int)sp[i].chunks;
        spans[i].off = (int)sp[i].off; spans[i].len = (int)sp[i].len;
    }
    if (!text || vi_len_(text, tlen) == 0) for (i = 0; i < n; ++i) spans[i].score = store_score_out(db->s, spans[i].score);
    if (used) *used = (int)u;
    return (int)n;
}

VI_API int vi_context_ids(vi_db *db, const vi_i64 *ids, const float *scores, int n, const vi_ctx_opts *o,
                          vi_ctx_span *spans, int max_spans, char *buf, int cap, int *used)
{
    vi_ctx_opts d;
    StoreSpan sp[VI_CTX_HITS_MAX];
    i32 m, i, u = 0;
    if (used) *used = 0;
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    vi_ctx_defaults(&d);
    if (o) d = *o;
    if (n < 0 || n > VI_CTX_HITS_MAX || (n > 0 && !ids)) return vi_fail_(-1, "ids holds 0 to 256 hits (VI_CTX_HITS_MAX)", (const char *)0, (const char *)0);
    if (d.shift < 0 || d.shift > 63) return vi_fail_(-1, "shift must be 0 to 63", (const char *)0, (const char *)0);
    if (d.before < -1 || d.after < -1) return vi_fail_(-1, "before and after are 0 or more, or -1 (the whole document)", (const char *)0, (const char *)0);
    if (!spans || max_spans < 1) return vi_fail_(-1, "spans is NULL or max_spans below 1", (const char *)0, (const char *)0);
    if (cap < 0 || (cap > 0 && !buf)) return vi_fail_(-1, "buf is NULL or cap below 0", (const char *)0, (const char *)0);
    if (n == 0) return 0;
    if (max_spans > VI_CTX_HITS_MAX) max_spans = VI_CTX_HITS_MAX;
    m = store_context(db->s, (const f32 *)0, (const char *)0, 0, (const StoreFlt *)0, (const i64 *)ids, scores, (i32)n, (i32)d.shift,
                      (i32)d.before, (i32)d.after, 1.0f, 0.0f, sp, (i32)max_spans, buf, (i32)cap, &u);
    if (m < 0) return vi_fail_(-1, "out of memory", (const char *)0, (const char *)0);
    for (i = 0; i < m; ++i) {
        spans[i].doc = (vi_i64)sp[i].doc; spans[i].first = (vi_i64)sp[i].first; spans[i].last = (vi_i64)sp[i].last;
        spans[i].hit = (vi_i64)sp[i].hit; spans[i].score = sp[i].score; spans[i].chunks = (int)sp[i].chunks;
        spans[i].off = (int)sp[i].off; spans[i].len = (int)sp[i].len;
    }
    if (used) *used = (int)u;
    return (int)m;
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
    int n;
    if (vi_search_ok_(db, query, k, ids, scores)) return -1;
    if (!query) return vi_fail_(-1, "the query is NULL", (const char *)0, (const char *)0);
    n = store_search_num(db->s, query, (i32)k, (i64)lo, (i64)hi, (i64 *)ids, scores);
    vi_cos_(db, scores, n);
    return n;
}

VI_API int vi_add_ns(vi_db *db, vi_i64 ns, vi_i64 id, const float *vec, const char *text, int len)
{
    StoreWb b;
    i32 r, n;
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (vi_vec_ok_(db, vec, "the vector")) return -1;
    n = text ? vi_len_(text, len) : 0;
    if (n < 0 || n > STORE_TXMAX) return vi_fail_(-1, "the text is longer than 16 MB - 1 bytes", (const char *)0, (const char *)0);
    store_batch_init(&b, db->s);                                 /* the row and its namespace: one journal record */
    if (n > 0) store_batch_put_text(&b, vec, (i64)id, 0, text, n); else store_batch_put(&b, vec, (i64)id, 0);
    store_batch_set_ns(&b, (i64)id, (i64)ns);
    r = store_batch_commit(&b, 0);
    store_batch_free(&b);
    return r >= 0 ? 0 : vi_fail_(-1, "out of memory", (const char *)0, (const char *)0);
}

VI_API int vi_set_ns(vi_db *db, vi_i64 id, vi_i64 ns)
{
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    return store_set_ns(db->s, (i64)id, (i64)ns) == 0 ? 0 : vi_norow_(db, id);
}

VI_API int vi_get_ns(vi_db *db, vi_i64 id, vi_i64 *ns)
{
    i64 v = 0;
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (store_get_ns(db->s, (i64)id, &v) != 0) return vi_norow_(db, id);
    if (ns) *ns = (vi_i64)v;
    return 0;
}

VI_API int vi_search_ns(vi_db *db, vi_i64 ns, const float *query, int k, vi_i64 *ids, float *scores)
{
    int n;
    if (vi_search_ok_(db, query, k, ids, scores)) return -1;
    if (!query) return vi_fail_(-1, "the query is NULL", (const char *)0, (const char *)0);
    n = store_search_ns(db->s, (i64)ns, query, (i32)k, (i64 *)ids, scores);
    vi_cos_(db, scores, n);
    return n;
}

VI_API int vi_search_in(vi_db *db, const vi_i64 *ns, int n, const float *query, int k, vi_i64 *ids, float *scores)
{
    int r;
    if (vi_search_ok_(db, query, k, ids, scores)) return -1;
    if (!query) return vi_fail_(-1, "the query is NULL", (const char *)0, (const char *)0);
    if (n < 1 || !ns) return vi_fail_(-1, "no namespace to search in", (const char *)0, (const char *)0);
    r = store_search_nss(db->s, (const i64 *)ns, (i32)n, query, (i32)k, (i64 *)ids, scores);
    if (r < 0) return vi_fail_(-1, "out of memory", (const char *)0, (const char *)0);
    vi_cos_(db, scores, r);
    return r;
}

VI_API vi_i64 vi_delete_ns(vi_db *db, vi_i64 ns)
{
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    if (!ns) return vi_fail_(-1, "namespace 0 is the rows in none: delete those by id", (const char *)0, (const char *)0);
    return (vi_i64)store_delete_ns(db->s, (i64)ns);
}

VI_API vi_i64 vi_count_ns(vi_db *db, vi_i64 ns)
{
    if (!db) return vi_fail_(-1, VI_NODB, (const char *)0, (const char *)0);
    return (vi_i64)store_count_ns(db->s, (i64)ns);
}

VI_API int vi_search_exact(vi_db *db, const float *query, int k, vi_i64 *ids, float *scores, vi_i64 *checked)
{
    int n;
    i64 c = 0;
    if (vi_search_ok_(db, query, k, ids, scores)) return -1;
    if (!query) return vi_fail_(-1, "the query is NULL", (const char *)0, (const char *)0);
    n = store_search_exact(db->s, query, (i32)k, (i64 *)ids, scores, (const StoreCnf *)0, &c);
    if (n < 0) return vi_fail_(-1, "out of memory", (const char *)0, (const char *)0);
    vi_cos_(db, scores, n);
    if (checked) *checked = (vi_i64)c;
    return n;
}

VI_API int vi_search_exact_batch(vi_db *db, const float *queries, int nq, int k, vi_i64 *ids, float *scores, vi_i64 *checked)
{
    int q, j;
    i32 *n;
    if (vi_search_ok_(db, (const float *)0, k, ids, scores)) return -1;
    if (nq < 0 || (nq > 0 && !queries)) return vi_fail_(-1, "queries is NULL or nq below 0", (const char *)0, (const char *)0);
    for (q = 0; q < nq; ++q) if (vi_vec_ok_(db, queries + (size_t)q * (size_t)db->s->dim, "a query")) {
        char nb[24], why[VI_EMAX];
        strcpy(why, vi_error());
        sprintf(nb, "%d: ", q);
        return vi_fail_(-1, "query ", nb, why);
    }
    if (nq == 0) return 0;
    n = (i32 *)malloc((size_t)nq * sizeof(i32));
    if (!n || store_search_exact_batch(db->s, queries, (i32)nq, (i32)k, (i64 *)ids, scores, n, (const StoreCnf *)0, (i64 *)checked) != 0) {
        free(n);
        return vi_fail_(-1, "out of memory", (const char *)0, (const char *)0);
    }
    for (q = 0; q < nq; ++q) {
        vi_cos_(db, scores + (size_t)q * k, n[q]);
        for (j = n[q]; j < k; ++j) { ids[(size_t)q * k + j] = -1; scores[(size_t)q * k + j] = -1.0f; }
    }
    free(n);
    return 0;
}

VI_API int vi_search(vi_db *db, const float *query, int k, vi_i64 *ids, float *scores)
{
    int n;
    if (vi_search_ok_(db, query, k, ids, scores)) return -1;
    if (!query) return vi_fail_(-1, "the query is NULL", (const char *)0, (const char *)0);
    {   i64 oom = store_oom_count();
        n = store_search(db->s, query, (i32)k, (i64 *)ids, scores);
        if (n == 0 && store_oom_count() != oom) return vi_fail_(-1, "out of memory", (const char *)0, (const char *)0); }
    vi_cos_(db, scores, n);
    return n;
}

VI_API int vi_search_grouped(vi_db *db, const float *query, int k, int shift, vi_i64 *ids, float *scores)
{
    i32 n;
    if (vi_search_ok_(db, query, k, ids, scores)) return -1;
    if (!query) return vi_fail_(-1, "the query is NULL", (const char *)0, (const char *)0);
    if (shift < 0 || shift > 63) return vi_fail_(-1, "shift must be 0 to 63", (const char *)0, (const char *)0);
    if ((n = store_search_grouped(db->s, query, (i32)k, (i32)shift, (const StoreCnf *)0, (i64 *)ids, scores)) < 0)
        return vi_fail_(-1, "out of memory", (const char *)0, (const char *)0);
    vi_cos_(db, scores, n);
    return n;
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

VI_API int vi_search_where(vi_db *db, const float *query, int k, const char *all, const char *any, const char *none,
                           vi_i64 lo, vi_i64 hi, vi_i64 *ids, float *scores)
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
    if (lo > hi) return 0;
    c.num = 1; c.lo = (i64)lo; c.hi = (i64)hi;
    n = store_search_cnf(db->s, query, (i32)k, (i64 *)ids, scores, &c);
    vi_cos_(db, scores, n);
    return n;
}

/* the graph beside a store (include/graph.h): a module of its own, built into the same library */
#include "graph.c"

/* documents into a store (include/ingest.h): Markdown and JSON, on the public API above */
#include "ingest.c"
