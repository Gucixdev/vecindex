/* vserv_keys.c - API keys (-K FILE): many users of one server, each seeing only its own data, with nothing to set
 * up but a file of keys. A line of the file is KEY TENANT [ROWS [QPS]] (tabs or spaces between; # starts a comment):
 *
 *   k_9f2c41d07e3a4b55a1f0   7        one store: tenant 7 - a namespace of the store (store_ns.c), ids 0 to 2^44 - 1
 *   k_1b7e0c9da2f84c3e8d61   acme     a tenant directory: the tenant acme, DIR/acme.vxi
 *   k_4c0d9e2b7a1f3e6d5c8b   8  100000  50     at most 100000 rows and 50 requests a second (0: no limit)
 *   k_admin_7e3a4b55a1f0c9   *        the operator: everything, housekeeping and /metrics included
 *
 * The row limit counts the tenant's rows once a second, so it may be passed by what one second of adds brings; a
 * replacement of a row the tenant has is never refused. The request limit is a bucket of QPS tokens refilled QPS a
 * second; past it a request is answered "rate limit" (lines), status 7 (binary) or 429 (HTTP).
 *
 * A connection on the TCP port shows its key before anything else (line: auth KEY; binary: op 20; HTTP: an
 * Authorization: Bearer KEY header on every request), and from then on reaches only its tenant: in one store, the
 * server puts its rows in the tenant's namespace and gives them ids of the tenant's own (tenant << 44 | id), so two
 * tenants may both have an id 1 and neither can name the other's rows; searches see only the namespace. Building,
 * saving, snapshots and scrubs stay with whoever runs the server. The Unix socket needs no key (who may connect to it
 * is the socket file's mode): there, auth is optional and the rest is the whole store.
 * Part of vserv.c, which includes it at this point; not compiled on its own. */

#define VS_KEYMIN  16                             /* a key: 16 to 128 printable characters, no spaces */
#define VS_KEYMAX  128
#define VS_TSHIFT  44                             /* a tenant's ids are [0, 2^44): 17.6 trillion, doc << 20 | chunk fits */
#define VS_TMASK   (((i64)1 << VS_TSHIFT) - 1)
#define VS_TMAXN   (((i64)1 << (63 - VS_TSHIFT)) - 1)   /* tenants 1 to 524287 in one store */

typedef struct VsKey VsKey;                       /* key, klen, tn, name, admin, rows, qps: defined in vserv.c */
static VsKey *vs_keys;
static i32 vs_nkeys;
typedef struct { pthread_mutex_t mu; f64 tokens, last, rows_at; i64 rows; } VsKeySt;   /* a key's limits as they run */
static VsKeySt *vs_kst;

/* a whole number in t[0, n); -1 when it is not one */
static i64 vs_knum_(const char *t, size_t n)
{
    i64 v = 0;
    size_t i;
    if (!n || n > 15) return -1;
    for (i = 0; i < n; ++i) { if (t[i] < '0' || t[i] > '9') return -1; v = v * 10 + (t[i] - '0'); }
    return v;
}

/* the key file at path: 0, or -1 with the line that is wrong on stderr */
static i32 vs_keys_load_(const char *path)
{
    FILE *f = fopen(path, "r");
    char line[512];
    i32 ln = 0, cap = 0;
    if (!f) { fprintf(stderr, "vserv: cannot open the key file %s: %s\n", path, strerror(errno)); return -1; }
    while (fgets(line, sizeof(line), f)) {
        char *p = line, *k, *t, *f3, *f4;
        size_t kl, tl, l3, l4, i;
        VsKey *e;
        ++ln;
        if (!strchr(line, '\n') && !feof(f)) {                   /* the rest would read as a line of its own */
            fprintf(stderr, "vserv: %s:%d: a line is at most %d bytes\n", path, (int)ln, (int)sizeof(line) - 2);
            fclose(f); return -1;
        }
        while (*p == ' ' || *p == '\t') ++p;
        if (!*p || *p == '\n' || *p == '#') continue;
        for (k = p; *p && *p != ' ' && *p != '\t' && *p != '\n'; ++p) {}
        kl = (size_t)(p - k);
        while (*p == ' ' || *p == '\t') ++p;
        for (t = p; *p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '#'; ++p) {}
        tl = (size_t)(p - t);
        while (*p == ' ' || *p == '\t') ++p;
        for (f3 = p; *p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '#'; ++p) {}
        l3 = (size_t)(p - f3);
        while (*p == ' ' || *p == '\t') ++p;
        for (f4 = p; *p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '#'; ++p) {}
        l4 = (size_t)(p - f4);
        while (*p == ' ' || *p == '\t') ++p;
        if (kl < VS_KEYMIN || kl > VS_KEYMAX || !tl || (*p && *p != '\n' && *p != '#')
            || (l3 && vs_knum_(f3, l3) < 0) || (l4 && vs_knum_(f4, l4) < 0)) {
            fprintf(stderr, "vserv: %s:%d: a line is KEY TENANT [ROWS [QPS]], the key 16 to 128 characters without spaces\n", path, (int)ln);
            fclose(f); return -1;
        }
        for (i = 0; i < kl; ++i) if ((unsigned char)k[i] < 0x21 || (unsigned char)k[i] > 0x7e) break;
        if (i < kl) { fprintf(stderr, "vserv: %s:%d: a key is printable ASCII\n", path, (int)ln); fclose(f); return -1; }
        if (vs_nkeys == cap) {
            VsKey *nk = (VsKey *)realloc(vs_keys, (size_t)(cap = cap ? cap * 2 : 16) * sizeof(VsKey));
            if (!nk) { fclose(f); return -1; }
            vs_keys = nk;
        }
        e = &vs_keys[vs_nkeys];
        memset(e, 0, sizeof(*e));
        memcpy(e->key, k, kl); e->klen = (i32)kl;
        e->rows = l3 ? vs_knum_(f3, l3) : 0; e->qps = l4 ? vs_knum_(f4, l4) : 0;
        if (tl == 1 && *t == '*') e->admin = 1;                  /* the operator */
        else if (vs_dir) {                                            /* a tenant directory: the tenant's name */
            for (i = 0; i < tl && vs_tname_ok_(t[i]); ++i) {}
            if (tl > VS_TNAME || i < tl) { fprintf(stderr, "vserv: %s:%d: a tenant's name is 1 to 64 of A-Z a-z 0-9 _ -\n", path, (int)ln); fclose(f); return -1; }
            memcpy(e->name, t, tl);
        } else {                                                 /* one store: the tenant's number, its namespace */
            i64 n = 0;
            for (i = 0; i < tl && t[i] >= '0' && t[i] <= '9' && n <= VS_TMAXN; ++i) n = n * 10 + (t[i] - '0');
            if (i < tl || n < 1 || n > VS_TMAXN) {
                fprintf(stderr, "vserv: %s:%d: with one store a tenant is a number, 1 to 524287\n", path, (int)ln);
                fclose(f); return -1;
            }
            e->tn = n;
        }
        for (i = 0; i < (size_t)vs_nkeys; ++i)
            if (vs_keys[i].klen == e->klen && !memcmp(vs_keys[i].key, e->key, kl)) {
                fprintf(stderr, "vserv: %s:%d: the same key twice\n", path, (int)ln); fclose(f); return -1;
            }
        ++vs_nkeys;
    }
    fclose(f);
    if (!vs_nkeys) { fprintf(stderr, "vserv: %s holds no keys\n", path); return -1; }
    if (!(vs_kst = (VsKeySt *)calloc((size_t)vs_nkeys, sizeof(VsKeySt)))) return -1;
    for (ln = 0; ln < vs_nkeys; ++ln) {
        pthread_mutex_init(&vs_kst[ln].mu, (const pthread_mutexattr_t *)0);
        vs_kst[ln].tokens = (f64)vs_keys[ln].qps; vs_kst[ln].last = vs_now_(); vs_kst[ln].rows_at = -1.0;
    }
    return 0;
}

/* the key k[0, n) is - compared against every key the same way, so the time says nothing of how close a guess came */
static const VsKey *vs_key_find_(const char *k, i32 n)
{
    const VsKey *hit = (const VsKey *)0;
    i32 i, j;
    for (i = 0; i < vs_nkeys; ++i) {
        u32 d = (u32)(vs_keys[i].klen ^ n);
        for (j = 0; j < VS_KEYMAX; ++j) d |= (u32)(u8)((j < n ? k[j] : 0) ^ vs_keys[i].key[j]);
        if (!d) hit = &vs_keys[i];
    }
    return hit;
}

/* the connection shows key k[0, n): NULL, or why not (for the client) */
static const char *vs_auth_(VsConn *c, const char *k, i32 n)
{
    const VsKey *e = n >= VS_KEYMIN && n <= VS_KEYMAX ? vs_key_find_(k, n) : (const VsKey *)0;
    if (!vs_nkeys) return "this server has no keys (-K)";
    if (!e) { VS_M_ADD(vs_m_refused, 1); vs_log_(c, 1, "auth refused: an unknown key"); return "unknown key"; }
    if (c->key == e) return (const char *)0;                     /* the same key again (HTTP shows it every request) */
    if (c->wb && c->key != e) return "a batch is open: commit or abort it first";
    c->key = e;
    if (e->admin) c->tn = 0;                                     /* the operator: the whole store, or any tenant */
    else if (vs_dir) {
        const char *why;
        c->tn = 0;
        if ((why = vs_use_(c, e->name, (i32)strlen(e->name))) != (const char *)0) { c->key = (const VsKey *)0; return why; }
    } else c->tn = e->tn;
    vs_log_(c, 0, "auth tenant=%s%.0f", e->name, vs_dir ? 0.0 : (double)e->tn);
    return (const char *)0;
}

/* a TCP connection to a server with keys that has not shown one: only ping, help, quit and auth answer */
static i32 vs_need_key_(const VsConn *c) { return vs_nkeys > 0 && c->tcp && !c->key; }

/* a keyed connection may not do the server's housekeeping (build, save, snapshot, scrub), unless the operator's */
static i32 vs_admin_(const VsConn *c) { return !c->key || c->key->admin; }

/* The token bucket of one key at time now: refilled qps a second, never above qps; a request takes one token. 1 it
 * may go on, 0 over the limit. lean4/Vecindex/Rate.lean proves that it lets through at most qps + qps * t requests
 * in any t seconds. */
static i32 vs_bucket_(f64 *tokens, f64 *last, f64 now, f64 qps)
{
    i32 ok;
    *tokens += (now - *last) * qps;
    if (*tokens > qps) *tokens = qps;
    *last = now;
    ok = *tokens >= 1.0;
    if (ok) *tokens -= 1.0;
    return ok;
}

/* one request of a keyed connection against its key's QPS: 1 it may go on, 0 over the limit */
static i32 vs_key_take_(const VsConn *c)
{
    VsKeySt *k;
    f64 now;
    i32 ok;
    if (!c->key || !c->key->qps) return 1;
    k = &vs_kst[c->key - vs_keys];
    pthread_mutex_lock(&k->mu);
    now = vs_now_();
    ok = vs_bucket_(&k->tokens, &k->last, now, (f64)c->key->qps);
    pthread_mutex_unlock(&k->mu);
    if (!ok) VS_M_ADD(vs_m_refused, 1);
    return ok;
}

/* room for a new row (internal id) under the key's row limit: 1 yes (or a row it replaces), 0 the limit is reached */
static i32 vs_key_room_(const VsConn *c, i64 id)
{
    VsKeySt *k;
    i64 n;
    if (!c->key || !c->key->rows || !c->tn) return 1;
    if (store_has(c->t->s, id)) return 1;
    k = &vs_kst[c->key - vs_keys];
    pthread_mutex_lock(&k->mu);
    if (k->rows_at < 0.0 || vs_now_() - k->rows_at > 1.0) { k->rows = store_count_ns(c->t->s, c->tn); k->rows_at = vs_now_(); }
    n = k->rows++;                                               /* counted until the next look */
    pthread_mutex_unlock(&k->mu);
    return n < c->key->rows;
}

/* ids on the wire <-> in the store: a tenant's id gains its tenant above bit 44 coming in, loses it going out. 0, or
 * -1 for an id outside [0, 2^44) */
static i32 vs_in_(const VsConn *c, i64 *id)
{
    if (!c->tn) return 0;
    if (*id < 0 || *id > VS_TMASK) return -1;
    *id |= c->tn << VS_TSHIFT;
    return 0;
}
static void vs_out_(const VsConn *c, i64 *ids, i32 n) { i32 i; if (c->tn) for (i = 0; i < n; ++i) ids[i] &= VS_TMASK; }
/* a range of ids [lo, hi] as the tenant's: clipped to [0, 2^44) and moved up; -1 when nothing of it is left */
static i32 vs_rin_(const VsConn *c, i64 *lo, i64 *hi)
{
    if (!c->tn) return 0;
    if (*hi < 0 || *lo > VS_TMASK || *lo > *hi) return -1;
    if (*lo < 0) *lo = 0;
    if (*hi > VS_TMASK) *hi = VS_TMASK;
    *lo |= c->tn << VS_TSHIFT; *hi |= c->tn << VS_TSHIFT;
    return 0;
}
