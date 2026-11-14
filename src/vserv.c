/* vserv.c - vecindex over a socket: libc + pthreads + Linux epoll, and TLS through the vendored BearSSL
 * (third_party/bearssl, MIT, no malloc inside). Nothing else.
 *
 *   vserv [-s SOCK] [-p PORT] [-b ADDR] [-t THREADS] [-g AGGRESSION] [-m TIER] [-C] [-d DIM] [-x] [-L] [-r RECALL]
 *         [-f SECONDS] [-S SECONDS] [-l syslog|-|FILE] [-A] [-c CERT.pem -k KEY.pem [-a CA.pem] [-P] [-T]] [-U]
 *         STORE.vxi | DIR
 *     STORE.vxi one store; DIR: a store per tenant, DIR/NAME.vxi, opened on a client's first "use NAME" (created
 *               there when it does not exist yet; -d gives every tenant's dimension and is required) and kept open
 *     -s PATH   Unix socket, created mode 0600 (default ./vecindex.sock when -p is not given)
 *     -p PORT   TCP port, bound to 127.0.0.1 unless -b gives another address
 *     -t N      worker threads (default: the CPUs the process may use - its affinity and cgroup quota - times -g)
 *     -g A      how much of the machine to take, 0..1 (default 1, or $VECINDEX_AGGRESSION): worker threads and the
 *               store's build and scan threads scale with it; background work (live indexing, rebuilds, snapshots)
 *               runs at nice 10 (1) to 19 (0) and on the disk's lowest priority (store_set_aggression)
 *     -m TIER   what stays in RAM whatever the memory pressure (store_set_memory): 0 the kernel decides (default),
 *               1 what every search scans (the index's codes, slot table, centroids) - rows stay on the disk and are
 *               read only for the candidates a query re-scores; 2 the rows, ids and tags too. Needs a memlock limit
 *               that large (ulimit -l, systemd's LimitMEMLOCK=); what it refuses is only read ahead
 *     -C        the cascade (store_set_cascade): 1-bit codes beside the 4-bit ones, scanned first, the 4-bit codes
 *               re-scoring a shortlist - a quarter of the code bytes a query reads; -m 1 then locks just those
 *     -M METRIC the metric of a new store: cosine (default), dot (inner product of the vectors as they come) or l2
 *               (Euclidean distance); scores come back in it. A store keeps the metric it was made with
 *     -d DIM    dimension of a new, empty store when STORE.vxi does not exist yet (it is created at once, with its
 *               journal, so every write is journaled from the start)
 *     -x        save the store to STORE.vxi on SIGINT / SIGTERM
 *     -L        live: added rows are indexed as they come and the index is rebuilt in the background as it ages
 *               (grown 2x, the data drifted, a quarter deleted) - no build calls needed (store_set_live)
 *     -r R      keep recall@10 at R (e.g. 0.95): the probes are tuned on the queries the server answers, after every
 *               build or rebuild (store_set_target_recall)
 *     -f S      flush the journal to the disk every S seconds (default 1; 0: leave it to the kernel). A crash loses
 *               at most the last S seconds of writes
 *     -S S      a background snapshot every S seconds when something changed (default 0: only when the journal has
 *               grown past the file, and at least 64 MB). Either way the journal is folded into the file without
 *               stopping searches or writes
 *     -l DEST   log connections, every write and every refusal, with who made it (vserv_log.c): syslog, - (stderr)
 *               or a file
 *     -A        log every search as well
 *     -c, -k    TLS: certificate chain and private key (PEM, EC or RSA). The TCP port then takes only TLS: HTTPS,
 *               or the binary protocol inside TLS (-P lets plaintext in too). The Unix socket takes TLS as well and
 *               still accepts plaintext from local clients, unless -T
 *     -a FILE   mutual TLS: a client must present a certificate that chains to one of the CA certificates in FILE
 *               (or is one of them), else the handshake fails; its name goes into the log
 *     -U        with DIR and -a: a client certificate's name is the one tenant it may use - the CA, not the server,
 *               decides who sees which tenant (clients without a certificate, on the Unix socket, may use any)
 *
 * The store is opened with store_load (mmap: ready in milliseconds whatever its size). Every socket speaks three
 * protocols, told apart by the first byte of a request (vserv_proto.c has the details):
 *
 *   binary (first byte 0xB5): fixed little-endian frames, nothing to parse - search, add, delete, delete a range
 *     of ids, get, build, save, snapshot, stats; texts: add with a text, get a text, the ids in a range, hybrid
 *   lines (a lowercase letter): "search 10 0.1 0.2 ..." -> "ok 3 42 -0.01 17 -0.12 9 -0.31", one line each way,
 *     for nc, socat and a readline in any language; texts: addt, text, range, hybrid, words
 *   HTTP/1.1 (anything else), for curl and friends; keep-alive, Content-Length bodies:
 *     GET /health | GET /stats | GET /get?id=N | GET /text?id=N | GET /range?lo=A&hi=B | POST /search?k=10 |
 *     POST /hybrid?k=10 | POST /add?id=N | POST /delete?id=N (or ?lo=A&hi=B) | POST /build | POST /save |
 *     POST /snapshot; vectors as a JSON array ([0.1, -0.2, ...]), {"vector": [...], "text": "..."}, or raw
 *     little-endian f32 with Content-Type: application/octet-stream; replies are JSON.
 * An add replaces the row with the same id in every protocol: one row per id. A tenant server first needs the
 * tenant: "use NAME" (lines), op 14 (binary), or a path under /t/NAME/ (HTTP: /t/acme/search?k=10); tenants share
 * the workers, the log and the housekeeping, and nothing else - each has its own file, journal, index and snapshots.
 *
 * Who may connect is the operating system's business (the socket file's mode and group, the address the port is
 * bound to) and TLS client certificates'; what each connection did goes to the log. Durability: the journal of
 * changes is flushed every second (-f) and folded into the file by background snapshots (store_snapshot), so a
 * restart replays at most what came in since the last snapshot. Replication is a copy of the file and its journal.
 *
 * TLS (when -c/-k are given): a connection is TLS when its first byte is a handshake record (0x16). TLS 1.2 only,
 * ECDHE key exchange (forward secrecy), AEAD suites only (AES-GCM - AES-NI when the CPU has it - and
 * ChaCha20-Poly1305), sessions resumable from one cache shared by every worker. Client certificates with -a. On
 * the Unix socket TLS is optional unless -T (the socket never leaves the machine, and its 0600 mode decides who
 * may connect at all).
 *
 * Memory: an idle connection is one small struct. Its request buffer (one request of this store's
 * dimension) is allocated on first read and reused for every later request on it; responses are built in a
 * per-thread buffer, so a request allocates nothing (a client that stops reading gets its unsent tail
 * buffered until it catches up). TLS adds about 37 KB a connection: the engine state and room for one full
 * 16 KB record in each direction, which the protocol requires.
 *
 *   ./run builds it (BearSSL first, into .build/libbearssl.a); by hand:
 *   gcc -std=c89 -pedantic -Wall -Wextra -O2 -march=native -Isrc -isystem third_party/bearssl/inc src/vserv.c \
 *       .build/libbearssl.a -lm -lpthread -o vserv          (-DVSERV_NO_TLS: plaintext only, no BearSSL)
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "store.c"
#include <errno.h>
#include <signal.h>
#include <stdarg.h>
#include <syslog.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#if defined(__SSE2__)
#include <emmintrin.h>
#endif
#if !defined(VSERV_NO_TLS)
#define VSERV_TLS 1
#if defined(__GNUC__) && (!defined(__STDC_VERSION__) || __STDC_VERSION__ < 199901L)
#define inline __inline__                       /* BearSSL's headers use C99 static inline; GCC has it in C89 mode as __inline__ */
#include "bearssl.h"
#undef inline
#else
#include "bearssl.h"
#endif
#endif

#define VS_MAGIC     0xB5
#define VS_MAXK      1000
#define VS_EVENTS    128
#define VS_IDLE_S    60.0
#define VS_OUT_CAP   (64 + VS_MAXK * 48)          /* the largest reply: JSON for VS_MAXK hits */
#define VS_REQ_TEXT  (8 << 20)                   /* a request may carry this much text (escaped) beyond a vector */

typedef struct VsTls VsTls;

typedef struct VsConn {
    i32 fd, len, cap, out_len, out_off, closing;
    u8 tcp, checked;                            /* accepted on the TCP port; its first byte looked at (TLS or not) */
    char *buf;                                  /* request bytes (plaintext), allocated on first read */
    char *out;                                  /* reply bytes not yet out (under TLS: not yet in the engine) */
    double last;
    VsTls *tls;                                 /* NULL: plaintext */
    struct VsConn *prev, *next;                 /* the worker's list, for the idle sweep */
    unsigned long id, nreq;                     /* for the log: its number, requests answered */
    u8 named;                                   /* TLS: its certificate's name is in peer */
    char peer[112];                             /* who is on the other end (vserv_log.c) */
    struct VsTen *t;                            /* the store its requests go to; NULL: a tenant server, none chosen */
    StoreWb *wb;                                /* a batch it began: its writes wait here until commit (or abort) */
} VsConn;

typedef struct { i32 fd, tcp; } VsListen;       /* epoll data for a listening socket */

typedef struct {
    i32 ep;
    pthread_t th;
    VsConn *conns;
    char *out;                                  /* reply scratch: VS_OUT_CAP, or one vector as text if that is more */
    f32 *vec;                                   /* dim floats: an aligned copy of the query */
    i64 *ids;
    f32 *sc;
    char *txt;                                  /* a decoded or fetched text (vserv_text.c), grown as needed */
    i32 txtcap;
} VsWorker;

#ifdef VSERV_TLS
struct VsTls {
    br_ssl_server_context sc;                   /* first: the session cache gets this pointer and finds the rest */
    br_x509_minimal_context xc;                 /* validates client certificates (-a) */
    br_name_element cn;                         /* the client certificate's common name, for the log */
    char cnbuf[64];
    i32 resumed;                                /* the session came from the cache (no certificate this time) */
    unsigned char io[BR_SSL_BUFSIZE_BIDI];      /* one full record each way */
};
static br_x509_certificate *vs_chain;
static size_t vs_nchain;
static br_skey_decoder_context vs_skey;         /* the decoded private key, for the server's lifetime */
static i32 vs_keytype, vs_issuer;
static br_x509_trust_anchor *vs_tas;
static size_t vs_ntas;

/* One session cache for every worker: a client that comes back is handed to whichever worker the kernel wakes,
 * so a per-thread cache would miss about half the resumptions. BearSSL's LRU is not thread-safe; a mutex around
 * its two calls is all it needs (a lookup is a short tree walk). */
typedef struct {
    const br_ssl_session_cache_class *vtable;
    br_ssl_session_cache_lru lru;
    pthread_mutex_t mu;
} VsCache;
#define VS_CACHE_BYTES (256 * 1024)             /* ~2500 sessions */
static VsCache vs_cache;
static unsigned char *vs_cache_mem;

/* A resumed session shows no certificate, so the name of the one that opened it is kept with its id: a ring of the
 * latest VS_NAMES sessions, searched on resumption, under the cache's mutex */
#define VS_NAMES 2048
typedef struct { unsigned char id[32]; u8 idlen; char cn[64]; } VsName;
static VsName *vs_names;
static i32 vs_names_next;

static void vs_cache_save_(const br_ssl_session_cache_class **ctx, br_ssl_server_context *sc, const br_ssl_session_parameters *p)
{
    VsCache *c = (VsCache *)(void *)ctx;
    VsTls *t = (VsTls *)(void *)sc;
    pthread_mutex_lock(&c->mu);
    c->lru.vtable->save(&c->lru.vtable, sc, p);
    if (vs_names && t->cn.status > 0 && p->session_id_len <= 32) {
        VsName *n = &vs_names[vs_names_next];
        vs_names_next = (vs_names_next + 1) % VS_NAMES;
        memcpy(n->id, p->session_id, p->session_id_len); n->idlen = (u8)p->session_id_len;
        strncpy(n->cn, t->cnbuf, sizeof(n->cn) - 1); n->cn[sizeof(n->cn) - 1] = 0;
    }
    pthread_mutex_unlock(&c->mu);
}
static int vs_cache_load_(const br_ssl_session_cache_class **ctx, br_ssl_server_context *sc, br_ssl_session_parameters *p)
{
    VsCache *c = (VsCache *)(void *)ctx;
    VsTls *t = (VsTls *)(void *)sc;
    int r, i;
    pthread_mutex_lock(&c->mu);
    r = c->lru.vtable->load(&c->lru.vtable, sc, p);
    if (r) {
        t->resumed = 1;
        for (i = 0; vs_names && i < VS_NAMES; ++i)
            if (vs_names[i].idlen == p->session_id_len && !memcmp(vs_names[i].id, p->session_id, p->session_id_len)) {
                strcpy(t->cnbuf, vs_names[i].cn); t->cn.status = 1;
                break;
            }
    }
    pthread_mutex_unlock(&c->mu);
    return r;
}
static const br_ssl_session_cache_class vs_cache_class = { sizeof(VsCache), vs_cache_save_, vs_cache_load_ };
#else
struct VsTls { i32 unused; };
#endif
static i32 vs_tls_on, vs_plain_ok, vs_unix_tls_only;

/* A store and what the main thread keeps about it. A server of one store has one (vs_one); a tenant server (a
 * directory given instead of a file) opens DIR/NAME.vxi on a client's first "use NAME" and keeps it open, newest
 * first in vs_tens: a tenant is added under vs_ten_mu and never removed, so whoever read the head may walk the
 * rest without the lock. */
#define VS_TNAME   64                             /* a tenant's name: 1 to 64 of A-Z a-z 0-9 _ - */
#define VS_TENANTS 4096                           /* open at once, at most */
typedef struct VsTen {
    char name[VS_TNAME + 1];
    const char *path;                             /* its file */
    RagStore *s;
    f64 last_snap;                                /* housekeeping: when the last timed snapshot was considered */
    i64 mods_snap, done;                          /* changes when the last snapshot started; snapshots finished */
    struct VsTen *next;
} VsTen;
static VsTen vs_one;
static VsTen *vs_tens;
static const char *vs_dir;                        /* a tenant server: the directory of the tenants' stores */
static i32 vs_ntens, vs_ten_cn;                   /* -U: a client certificate's name is the one tenant it may use */
static pthread_mutex_t vs_ten_mu = PTHREAD_MUTEX_INITIALIZER;
static Arena vs_arena;
static f32 vs_aggr = 1.0f, vs_recall;             /* applied to every store opened (-g, -r, -L, -m) */
static i32 vs_live, vs_mem, vs_cascade, vs_metric;   /* vs_metric: of new stores, 0 cosine, 1 dot, 2 L2 */
static i32 vs_dim, vs_reqcap, vs_reqmax;          /* a request's buffer starts at reqcap, grows to reqmax for texts */
static VsListen vs_ls[2];
static i32 vs_nls;
static volatile sig_atomic_t vs_stop;

static double vs_now_(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec; }
static void vs_on_signal_(int sig) { (void)sig; vs_stop = 1; }

#include "vserv_log.c"

/* ---- stores and tenants ---- */

/* the settings every store gets once it is open (-t's one query a thread, -g, -r, -L, -m); 0, or -1 when the
 * memlock limit refused part of -m */
static i32 vs_ten_setup_(VsTen *t)
{
    i32 r = 0;
    store_set_threads(t->s, 1);                  /* one query per thread: the workers are the parallelism */
    store_set_aggression(t->s, vs_aggr);
    if (vs_recall > 0.0f) store_set_target_recall(t->s, vs_recall);
    if (vs_live) store_set_live(t->s, 1, 1.0f);
    if (vs_cascade && store_set_cascade(t->s, 1) != 0)            /* before -m: the 1-bit codes are what it locks */
        vs_log_((const VsConn *)0, 1, "cascade refused file=%s: not 4-bit codes, or no memory", t->path);
    if (vs_mem > 0) r = store_set_memory(t->s, vs_mem);         /* reads the locked part in before it serves */
    t->last_snap = vs_now_();
    t->mods_snap = store_changes(t->s);
    t->done = store_snapshots_done(t->s);
    return r;
}

static i32 vs_tname_ok_(char ch)
{
    return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_' || ch == '-';
}

/* opens tenant name[0, n) from DIR/NAME.vxi, or creates it there with the server's dimension; under vs_ten_mu.
 * NULL with *err the reason for the client. */
static VsTen *vs_ten_open_(const char *name, i32 n, const char **err)
{
    VsTen *t;
    struct stat st;
    char *path;
    i32 created = 0;
    if (vs_ntens >= VS_TENANTS) { *err = "too many tenants open"; return (VsTen *)0; }
    t = (VsTen *)calloc(1, sizeof(VsTen));
    path = t ? (char *)malloc(strlen(vs_dir) + (size_t)n + 6) : (char *)0;
    if (!path) { free(t); *err = "out of memory"; return (VsTen *)0; }
    memcpy(t->name, name, (size_t)n); t->name[n] = 0;
    sprintf(path, "%s/%s.vxi", vs_dir, t->name);
    t->path = path;
    t->s = store_load(&vs_arena, t->path);
    if (!t->s && stat(t->path, &st) != 0) {                      /* none yet: a new, empty store, journaled at once */
        t->s = store_new(&vs_arena, vs_dim);
        if (t->s) store_set_metric(t->s, vs_metric);
        if (t->s && store_save(t->s, t->path) != 0) { store_destroy(t->s); t->s = (RagStore *)0; }
        created = 1;
    }
    if (t->s && t->s->dim != vs_dim) { store_destroy(t->s); t->s = (RagStore *)0; *err = "the tenant's store has another dimension"; }
    else if (!t->s) *err = created ? "cannot create the tenant's store" : "the tenant's store cannot be read";
    if (!t->s) {
        vs_log_((const VsConn *)0, 1, "tenant name=%s file=%s: %s%s%s", t->name, t->path, *err,
                *store_load_error() ? ": " : "", store_load_error());
        free(path); free(t);
        return (VsTen *)0;
    }
    if (vs_ten_setup_(t) != 0) vs_log_((const VsConn *)0, 1, "tenant name=%s: the memlock limit refused part of -m %d", t->name, (int)vs_mem);
    {   i64 nb, bad, fx;                                         /* what the load's checks found, when anything */
        i32 jd;
        store_integrity(t->s, &nb, &bad, &fx, (i64 *)0, (i64 *)0, &jd);
        if (bad || jd) vs_log_((const VsConn *)0, bad > fx, "check file=%s blocks=%.0f damaged=%.0f rebuilt=%.0f journal=%s", t->path,
                               (double)nb, (double)bad, (double)fx, jd ? "damaged (copy kept as .j.damaged)" : "ok"); }
    t->next = vs_tens;
    vs_tens = t;
    ++vs_ntens;
    vs_log_((const VsConn *)0, 0, "tenant %s name=%s rows=%d", created ? "created" : "opened", t->name, store_count(t->s));
    return t;
}

/* Points the connection's requests at tenant name[0, n), opening or creating it: NULL, or why not (for the client).
 * With -U a client that showed a certificate may use only the tenant its certificate names. */
static const char *vs_use_(VsConn *c, const char *name, i32 n)
{
    VsTen *t;
    const char *err = (const char *)0;
    i32 i;
    if (!vs_dir) return "this server has one store, no tenants";
    if (c->wb && (!c->t || strlen(c->t->name) != (size_t)n || memcmp(c->t->name, name, (size_t)n) != 0))
        return "a batch is open: commit or abort it first";
    if (n < 1 || n > VS_TNAME) return "a tenant's name is 1 to 64 of A-Z a-z 0-9 _ -";
    for (i = 0; i < n; ++i) if (!vs_tname_ok_(name[i])) return "a tenant's name is 1 to 64 of A-Z a-z 0-9 _ -";
#ifdef VSERV_TLS
    if (vs_ten_cn && c->tls && c->tls->cn.status > 0
        && (strlen(c->tls->cnbuf) != (size_t)n || memcmp(c->tls->cnbuf, name, (size_t)n) != 0)) {
        vs_log_(c, 1, "use %.*s refused: the certificate names another tenant", (int)n, name);
        return "your certificate names another tenant";
    }
#endif
    pthread_mutex_lock(&vs_ten_mu);
    for (t = vs_tens; t; t = t->next) if (!strncmp(t->name, name, (size_t)n) && !t->name[n]) break;
    if (!t) t = vs_ten_open_(name, n, &err);
    pthread_mutex_unlock(&vs_ten_mu);
    if (!t) return err;
    if (c->t != t) { c->t = t; vs_log_(c, 0, "use"); }
    return (const char *)0;
}

/* ---- small formatting helpers (no printf on the hot path) ---- */

static char *vs_i64_(char *p, i64 v)
{
    char tmp[24];
    i32 n = 0;
    u64 u = v < 0 ? (u64)(-(v + 1)) + 1u : (u64)v;
    if (v < 0) *p++ = '-';
    do { tmp[n++] = (char)('0' + (i32)(u % 10u)); u /= 10u; } while (u);
    while (n) *p++ = tmp[--n];
    return p;
}

/* a number with 6 decimals: a cosine, a fused score, a vector's value */
static char *vs_f32_(char *p, f32 v)
{
    i64 x;
    i32 i;
    char frac[6];
    if (v != v) { memcpy(p, "null", 4); return p + 4; }
    if (v < 0.0f) { *p++ = '-'; v = -v; }
    if (v > 1e15f) v = 1e15f;
    x = (i64)(v * 1e6f + 0.5f);
    p = vs_i64_(p, x / 1000000);
    *p++ = '.';
    x %= 1000000;
    for (i = 5; i >= 0; --i) { frac[i] = (char)('0' + (i32)(x % 10)); x /= 10; }
    memcpy(p, frac, 6);
    return p + 6;
}

static char *vs_str_(char *p, const char *s) { size_t n = strlen(s); memcpy(p, s, n); return p + n; }

/* ---- sending ---- */

static void vs_ep_mod_(VsWorker *w, VsConn *c, u32 ev)
{
    struct epoll_event e;
    e.events = ev; e.data.ptr = c;
    epoll_ctl(w->ep, EPOLL_CTL_MOD, c->fd, &e);
}

/* keep bytes for later (after what is already queued) and wait for EPOLLOUT */
static void vs_queue_(VsWorker *w, VsConn *c, const char *p, i32 n)
{
    i32 rest = c->out ? c->out_len - c->out_off : 0;
    char *o = (char *)malloc((size_t)rest + (size_t)n);
    if (!o) { c->closing = 1; return; }
    if (rest) memcpy(o, c->out + c->out_off, (size_t)rest);
    memcpy(o + rest, p, (size_t)n);
    free(c->out);
    c->out = o; c->out_len = rest + n; c->out_off = 0;
    vs_ep_mod_(w, c, EPOLLIN | EPOLLOUT | EPOLLRDHUP | EPOLLET);
}

#ifdef VSERV_TLS
/* encrypted records to the socket: 1 = all out, 0 = the socket is full, -1 = error */
static i32 vs_tls_tx_(VsConn *c)
{
    br_ssl_engine_context *e = &c->tls->sc.eng;
    for (;;) {
        size_t len;
        unsigned char *b;
        ssize_t k;
        if (!(br_ssl_engine_current_state(e) & BR_SSL_SENDREC)) return 1;
        b = br_ssl_engine_sendrec_buf(e, &len);
        k = send(c->fd, b, len, MSG_NOSIGNAL);
        if (k > 0) { br_ssl_engine_sendrec_ack(e, (size_t)k); continue; }
        if (k < 0 && errno == EINTR) continue;
        if (k < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return 0;
        return -1;
    }
}

/* plaintext into the engine, as much as it takes now; returns the bytes taken */
static i32 vs_tls_put_(VsConn *c, const char *p, i32 n)
{
    br_ssl_engine_context *e = &c->tls->sc.eng;
    i32 done = 0;
    while (done < n && (br_ssl_engine_current_state(e) & BR_SSL_SENDAPP)) {
        size_t len;
        unsigned char *b = br_ssl_engine_sendapp_buf(e, &len);
        if (!len) break;
        if (len > (size_t)(n - done)) len = (size_t)(n - done);
        memcpy(b, p + done, len);
        br_ssl_engine_sendapp_ack(e, len);
        done += (i32)len;
    }
    return done;
}

static void vs_tls_send_(VsWorker *w, VsConn *c, const char *p, i32 n)
{
    br_ssl_engine_context *e = &c->tls->sc.eng;
    i32 r;
    while (n > 0 && !c->out) {
        i32 k = vs_tls_put_(c, p, n);
        p += k; n -= k;
        if (n == 0) break;
        br_ssl_engine_flush(e, 0);                /* the engine is full: records out first */
        r = vs_tls_tx_(c);
        if (r < 0) { c->closing = 1; return; }
        if (r == 0 || (k == 0 && !(br_ssl_engine_current_state(e) & BR_SSL_SENDAPP))) break;
    }
    if (n > 0) vs_queue_(w, c, p, n);
    br_ssl_engine_flush(e, 0);
    r = vs_tls_tx_(c);
    if (r < 0) c->closing = 1;
    else if (r == 0) vs_ep_mod_(w, c, EPOLLIN | EPOLLOUT | EPOLLRDHUP | EPOLLET);
}
#endif

/* send what we can; keep the rest on the connection and wait for EPOLLOUT */
static void vs_send_(VsWorker *w, VsConn *c, const char *p, i32 n)
{
#ifdef VSERV_TLS
    if (c->tls) { vs_tls_send_(w, c, p, n); return; }
#endif
    if (c->out) { vs_queue_(w, c, p, n); return; }      /* never ahead of bytes already waiting */
    while (n > 0) {
        ssize_t k = send(c->fd, p, (size_t)n, MSG_NOSIGNAL);
        if (k > 0) { p += k; n -= (i32)k; continue; }
        if (k < 0 && errno == EINTR) continue;
        if (k < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
        c->closing = 1;
        return;
    }
    if (n > 0) vs_queue_(w, c, p, n);
}

/* reply bytes still on their way out */
static i32 vs_pending_(const VsConn *c)
{
    if (c->out) return 1;
#ifdef VSERV_TLS
    if (c->tls && (br_ssl_engine_current_state(&c->tls->sc.eng) & BR_SSL_SENDREC)) return 1;
#endif
    return 0;
}

static void vs_flush_(VsWorker *w, VsConn *c)
{
#ifdef VSERV_TLS
    if (c->tls) {
        br_ssl_engine_context *e = &c->tls->sc.eng;
        i32 r = vs_tls_tx_(c);
        if (r < 0) { c->closing = 1; return; }
        if (r == 0) return;
        while (c->out && c->out_off < c->out_len) {
            i32 k = vs_tls_put_(c, c->out + c->out_off, c->out_len - c->out_off);
            c->out_off += k;
            br_ssl_engine_flush(e, 0);
            r = vs_tls_tx_(c);
            if (r < 0) { c->closing = 1; return; }
            if (r == 0 || k == 0) return;         /* the next EPOLLOUT goes on */
        }
        if (c->out) { free(c->out); c->out = (char *)0; c->out_len = c->out_off = 0; }
        vs_ep_mod_(w, c, EPOLLIN | EPOLLRDHUP | EPOLLET);
        return;
    }
#endif
    while (c->out && c->out_off < c->out_len) {
        ssize_t k = send(c->fd, c->out + c->out_off, (size_t)(c->out_len - c->out_off), MSG_NOSIGNAL);
        if (k > 0) { c->out_off += (i32)k; continue; }
        if (k < 0 && errno == EINTR) continue;
        if (k < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return;
        c->closing = 1;
        return;
    }
    free(c->out); c->out = (char *)0; c->out_len = c->out_off = 0;
    vs_ep_mod_(w, c, EPOLLIN | EPOLLRDHUP | EPOLLET);
}

/* ---- the store operations both protocols share ---- */

/* the k nearest rows; their scores as the store's metric gives them: the cosine, the inner product, the L2 distance */
static i32 vs_search_(VsWorker *w, VsConn *c, const f32 *q, i32 k)
{
    i32 n = store_search(c->t->s, q, k, w->ids, w->sc), i;
    for (i = 0; i < n; ++i) w->sc[i] = store_score_out(c->t->s, w->sc[i]);
    return n;
}

static void vs_stats_(VsConn *c, i64 *v)
{
    v[0] = store_count(c->t->s); v[1] = vs_dim; v[2] = store_pending(c->t->s); v[3] = store_built(c->t->s);
}

#include "vserv_proto.c"

/* ---- connection lifecycle ---- */

static void vs_close_(VsWorker *w, VsConn *c)
{
    if (c->wb) {                                                 /* begun, never committed: none of it happened */
        vs_log_(c, 1, "batch of %d changes dropped: the connection closed", (int)c->wb->n);
        store_batch_free(c->wb); free(c->wb); c->wb = (StoreWb *)0;
    }
    vs_log_(c, 0, "close requests=%lu", c->nreq);
#ifdef VSERV_TLS
    if (c->tls) {
        br_ssl_engine_context *e = &c->tls->sc.eng;
        if (!(br_ssl_engine_current_state(e) & BR_SSL_CLOSED)) { br_ssl_engine_close(e); vs_tls_tx_(c); }   /* close_notify, best effort */
        explicit_bzero(c->tls, sizeof(VsTls));     /* session keys do not outlive the connection */
        free(c->tls);
    }
#endif
    epoll_ctl(w->ep, EPOLL_CTL_DEL, c->fd, (struct epoll_event *)0);
    {   /* closing over unread input makes the kernel send a reset that can overtake the last reply: say FIN first,
         * then drop whatever the peer already sent */
        char sink[4096];
        i32 n = 16;
        shutdown(c->fd, SHUT_WR);
        while (n-- > 0 && read(c->fd, sink, sizeof(sink)) > 0) {}
    }
    close(c->fd);
    if (c->prev) c->prev->next = c->next; else w->conns = c->next;
    if (c->next) c->next->prev = c->prev;
    free(c->buf); free(c->out); free(c);
}

/* parse and answer every complete request in the buffer */
static void vs_process_(VsWorker *w, VsConn *c)
{
    while (c->len > 0 && !c->closing && !c->out) {
        u8 b0 = (u8)c->buf[0];
        i32 used = b0 == VS_MAGIC ? vs_binary_(w, c) : (b0 >= 'a' && b0 <= 'z' ? vs_line_(w, c) : vs_http_(w, c));
        if (used < 0) { c->closing = 1; break; }
        if (used == 0) break;
        ++c->nreq;
        c->len -= used;
        if (c->len) memmove(c->buf, c->buf + used, (size_t)c->len);
    }
    if (c->len == 0 && c->cap > vs_reqcap) { free(c->buf); c->buf = (char *)0; c->cap = 0; }   /* a big request went */
}

/* room for more of a request: the buffer doubles, up to vs_reqmax; 0, or -1 when it may not grow */
static i32 vs_grow_(VsConn *c)
{
    i32 nc = c->cap > vs_reqmax / 2 ? vs_reqmax : c->cap * 2;
    char *b;
    if (c->cap >= vs_reqmax || !(b = (char *)realloc(c->buf, (size_t)nc))) return -1;
    c->buf = b; c->cap = nc;
    return 0;
}

#ifdef VSERV_TLS
static VsTls *vs_tls_new_(VsWorker *w)
{
    static const uint16_t ec_suites[] = { BR_TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256, BR_TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
                                          BR_TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384 };
    static const uint16_t rsa_suites[] = { BR_TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256, BR_TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
                                           BR_TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384 };
    VsTls *t = (VsTls *)malloc(sizeof(VsTls));
    if (!t) return (VsTls *)0;
    if (vs_keytype == BR_KEYTYPE_EC) {
        br_ssl_server_init_full_ec(&t->sc, vs_chain, vs_nchain, (unsigned)vs_issuer, br_skey_decoder_get_ec(&vs_skey));
        br_ssl_engine_set_suites(&t->sc.eng, ec_suites, 3);
    } else {
        br_ssl_server_init_full_rsa(&t->sc, vs_chain, vs_nchain, br_skey_decoder_get_rsa(&vs_skey));
        br_ssl_engine_set_suites(&t->sc.eng, rsa_suites, 3);
    }
    br_ssl_engine_set_versions(&t->sc.eng, BR_TLS12, BR_TLS12);
    t->cnbuf[0] = 0; t->cn.status = 0; t->resumed = 0;
    if (vs_ntas) {                                 /* mutual TLS: the client's chain must end at one of our CAs */
        static const unsigned char oid_cn[] = { 3, 0x55, 0x04, 0x03 };   /* 2.5.4.3, commonName */
        br_x509_minimal_init_full(&t->xc, vs_tas, vs_ntas);
        t->cn.oid = oid_cn; t->cn.buf = t->cnbuf; t->cn.len = sizeof(t->cnbuf);
        br_x509_minimal_set_name_elements(&t->xc, &t->cn, 1);
        br_ssl_engine_set_default_rsavrfy(&t->sc.eng);
        br_ssl_engine_set_default_ecdsa(&t->sc.eng);
        br_ssl_server_set_trust_anchor_names_alt(&t->sc, vs_tas, vs_ntas);
        br_ssl_engine_set_x509(&t->sc.eng, &t->xc.vtable);
    }
    br_ssl_server_set_cache(&t->sc, &vs_cache.vtable);
    (void)w;
    br_ssl_engine_set_buffer(&t->sc.eng, t->io, sizeof(t->io), 1);
    if (!br_ssl_server_reset(&t->sc)) { free(t); return (VsTls *)0; }
    return t;
}

/* move records in, plaintext up to the parsers, replies back down, until nothing moves */
static void vs_tls_readable_(VsWorker *w, VsConn *c)
{
    br_ssl_engine_context *e = &c->tls->sc.eng;
    for (;;) {
        unsigned st = br_ssl_engine_current_state(e);
        if (st & BR_SSL_CLOSED) {
            i32 err = br_ssl_engine_last_error(e);
            if (err != BR_ERR_OK && !c->named) vs_log_(c, 1, "tls handshake failed err=%d", err);   /* never got in */
            c->closing = 1;
            return;
        }
        if ((st & (BR_SSL_SENDAPP | BR_SSL_RECVAPP)) && !c->named) {   /* the handshake is done: say who it is */
            c->named = 1;
            if (vs_ntas && c->tls->cn.status > 0) vs_peer_name_(c, c->tls->cnbuf, c->tls->resumed);
            else vs_log_(c, 0, "tls %s", c->tls->resumed ? "resumed" : "established");
        }
        if (st & BR_SSL_SENDREC) {                /* handshake messages and replies */
            i32 r = vs_tls_tx_(c);
            if (r < 0) { c->closing = 1; return; }
            if (r == 0) vs_ep_mod_(w, c, EPOLLIN | EPOLLOUT | EPOLLRDHUP | EPOLLET);
        }
        if ((st & BR_SSL_RECVAPP) && !c->out) {   /* requests wait while replies are queued */
            size_t len;
            unsigned char *b = br_ssl_engine_recvapp_buf(e, &len);
            i32 room;
            if (!c->buf) { if (!(c->buf = (char *)malloc((size_t)vs_reqcap))) { c->closing = 1; return; } c->cap = vs_reqcap; }
            if (c->len == c->cap && vs_grow_(c)) { c->closing = 1; return; }
            room = c->cap - c->len;
            if (len > (size_t)room) len = (size_t)room;
            memcpy(c->buf + c->len, b, len);
            br_ssl_engine_recvapp_ack(e, len);
            c->len += (i32)len;
            c->last = vs_now_();
            vs_process_(w, c);
            if (c->closing) return;
            continue;
        }
        if (st & BR_SSL_RECVREC) {
            size_t len;
            unsigned char *b = br_ssl_engine_recvrec_buf(e, &len);
            ssize_t k = read(c->fd, b, len);
            if (k > 0) { br_ssl_engine_recvrec_ack(e, (size_t)k); c->last = vs_now_(); continue; }
            if (k == 0) { c->closing = 1; return; }
            if (errno == EINTR) continue;
            if (errno != EAGAIN && errno != EWOULDBLOCK) c->closing = 1;
            return;
        }
        return;
    }
}
#endif

static void vs_readable_(VsWorker *w, VsConn *c)
{
    if (!c->buf) {
        c->buf = (char *)malloc((size_t)vs_reqcap);
        if (!c->buf) { c->closing = 1; return; }
        c->cap = vs_reqcap;
    }
    if (vs_tls_on && !c->checked) {                /* a TLS ClientHello starts with a handshake record, 0x16 */
        u8 b0;
        ssize_t k = recv(c->fd, &b0, 1, MSG_PEEK);
        if (k == 0) { c->closing = 1; return; }
        if (k < 0) { if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) c->closing = 1; return; }
        c->checked = 1;
#ifdef VSERV_TLS
        if (b0 == 0x16) {
            c->tls = vs_tls_new_(w);
            if (!c->tls) { c->closing = 1; return; }
        } else
#endif
        if (c->tcp ? !vs_plain_ok : vs_unix_tls_only) {
            static const char msg[] = "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: 32\r\n"
                                      "Connection: close\r\n\r\n{\"error\":\"this port speaks TLS\"}";
            vs_send_(w, c, msg, (i32)sizeof(msg) - 1);
            c->closing = 1;
            return;
        }
    }
#ifdef VSERV_TLS
    if (c->tls) { vs_tls_readable_(w, c); return; }
#endif
    for (;;) {
        ssize_t k;
        if (!c->buf) { if (!(c->buf = (char *)malloc((size_t)vs_reqcap))) { c->closing = 1; return; } c->cap = vs_reqcap; }
        if (c->len == c->cap && vs_grow_(c)) { c->closing = 1; return; }   /* the parsers refuse what is too big */
        k = read(c->fd, c->buf + c->len, (size_t)(c->cap - c->len));
        if (k > 0) {
            c->len += (i32)k;
            c->last = vs_now_();
            vs_process_(w, c);
            if (c->closing || c->out) return;
            continue;
        }
        if (k == 0) { c->closing = 1; return; }
        if (errno == EINTR) continue;
        if (errno != EAGAIN && errno != EWOULDBLOCK) c->closing = 1;
        return;
    }
}

static void vs_accept_(VsWorker *w, const VsListen *l)
{
    for (;;) {
        struct epoll_event e;
        VsConn *c;
        i32 fd = accept4(l->fd, (struct sockaddr *)0, (socklen_t *)0, SOCK_NONBLOCK | SOCK_CLOEXEC), one = 1;
        if (fd < 0) return;                      /* EAGAIN: another worker took it, or none left */
        if (l->tcp) setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        c = (VsConn *)calloc(1, sizeof(VsConn));
        if (!c) { close(fd); continue; }
        c->fd = fd; c->last = vs_now_(); c->tcp = (u8)l->tcp;
        c->t = vs_dir ? (VsTen *)0 : &vs_one;     /* a tenant server: none until the client names one */
#if defined(__GNUC__)
        c->id = __atomic_add_fetch(&vs_conn_seq, 1UL, __ATOMIC_RELAXED);
#else
        c->id = ++vs_conn_seq;
#endif
        if (vs_logging_()) { vs_peer_(c, l); vs_log_(c, 0, "open"); }
        c->next = w->conns; if (w->conns) w->conns->prev = c; w->conns = c;
        e.events = EPOLLIN | EPOLLRDHUP | EPOLLET; e.data.ptr = c;
        if (epoll_ctl(w->ep, EPOLL_CTL_ADD, fd, &e) != 0) { vs_close_(w, c); continue; }
    }
}

static void *vs_worker_(void *arg)
{
    VsWorker *w = (VsWorker *)arg;
    struct epoll_event ev[VS_EVENTS];
    double sweep = vs_now_();
    while (!vs_stop) {
        i32 n = epoll_wait(w->ep, ev, VS_EVENTS, 1000), i;
        for (i = 0; i < n; ++i) {
            void *p = ev[i].data.ptr;
            VsConn *c;
            if (p == (void *)&vs_ls[0] || p == (void *)&vs_ls[1]) { vs_accept_(w, (const VsListen *)p); continue; }
            c = (VsConn *)p;
            if (ev[i].events & (EPOLLERR | EPOLLHUP)) { vs_close_(w, c); continue; }
            if (ev[i].events & EPOLLOUT) {
                vs_flush_(w, c);
                if (!c->closing && !vs_pending_(c)) vs_readable_(w, c);   /* requests that waited behind the backlog */
            }
            if (!c->closing && (ev[i].events & (EPOLLIN | EPOLLRDHUP))) vs_readable_(w, c);
            if (c->closing) {                            /* a last reply still going out keeps it open until EPOLLOUT */
                if (vs_pending_(c)) vs_flush_(w, c);
                if (!vs_pending_(c)) vs_close_(w, c);
            }
        }
        if (vs_now_() - sweep > 1.0) {           /* drop connections idle for VS_IDLE_S */
            VsConn *c = w->conns, *nx;
            double now = vs_now_();
            sweep = now;
            for (; c; c = nx) { nx = c->next; if (now - c->last > VS_IDLE_S) vs_close_(w, c); }
        }
    }
    while (w->conns) vs_close_(w, w->conns);
    return (void *)0;
}

/* ---- setup ---- */

static i32 vs_listen_unix_(const char *path)
{
    struct sockaddr_un a;
    struct stat st;
    mode_t old;
    i32 fd;
    if (strlen(path) >= sizeof(a.sun_path)) { fprintf(stderr, "vserv: socket path too long\n"); return -1; }
    if (lstat(path, &st) == 0) {
        if (!S_ISSOCK(st.st_mode)) { fprintf(stderr, "vserv: %s exists and is not a socket\n", path); return -1; }
        unlink(path);                            /* a stale socket from an earlier run */
    }
    fd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;
    memset(&a, 0, sizeof(a)); a.sun_family = AF_UNIX; strcpy(a.sun_path, path);
    old = umask(077);                            /* the socket file gets 0600: owner only */
    if (bind(fd, (struct sockaddr *)&a, sizeof(a)) != 0 || listen(fd, 1024) != 0) { umask(old); close(fd); return -1; }
    umask(old);
    return fd;
}

static i32 vs_listen_tcp_(const char *addr, i32 port)
{
    struct sockaddr_in a;
    i32 fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0), one = 1;
    if (fd < 0) return -1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    memset(&a, 0, sizeof(a)); a.sin_family = AF_INET; a.sin_port = htons((u16)port);
    if (inet_pton(AF_INET, addr, &a.sin_addr) != 1) { close(fd); return -1; }
    if (bind(fd, (struct sockaddr *)&a, sizeof(a)) != 0 || listen(fd, 1024) != 0) { close(fd); return -1; }
    return fd;
}

static i32 vs_ntas_(void)
{
#ifdef VSERV_TLS
    return vs_ntas > 0;
#else
    return 0;
#endif
}

#include "vserv_pem.c"

static i64 vs_fsize_(const char *path) { struct stat st; return stat(path, &st) == 0 ? (i64)st.st_size : 0; }

/* every open store: the one, or each tenant (the list only grows at its head, so it is walked without the lock) */
static VsTen *vs_first_(void)
{
    VsTen *t;
    if (!vs_dir) return &vs_one;
    pthread_mutex_lock(&vs_ten_mu);
    t = vs_tens;
    pthread_mutex_unlock(&vs_ten_mu);
    return t;
}

/* one store's turn: reap and log a finished snapshot, flush the journal when sync is due, start a snapshot */
static void vs_keep_(VsTen *t, i32 sync, f64 snap_s)
{
    i64 us;
    i32 run, last;
    store_snapshot_info(t->s, &run, &last, &us);
    if (run) store_snapshot_poll(t->s, 0);                       /* reaps it once the child is done */
    store_snapshot_info(t->s, &run, &last, &us);
    if (store_snapshots_done(t->s) != t->done) {                 /* finished since the last look, whoever reaped it */
        t->done = store_snapshots_done(t->s);
        vs_log_((const VsConn *)0, last != 0, "snapshot %s file=%s ms=%ld", last == 0 ? "ok" : "failed", t->path, (long)(us / 1000));
    }
    if (sync && store_journal_bytes(t->s, t->path) >= 0 && store_sync(t->s, t->path) != 0)
        vs_log_((const VsConn *)0, 1, "journal flush failed file=%s", t->path);
    if (!run) {
        i64 jb = store_journal_bytes(t->s, t->path), m = store_changes(t->s);
        i32 by_time = snap_s > 0 && vs_now_() - t->last_snap >= snap_s && m != t->mods_snap;
        i32 by_size = jb > ((i64)64 << 20) && jb > vs_fsize_(t->path);
        if (by_time || by_size) {
            if (store_snapshot(t->s, t->path) == 0) {
                t->mods_snap = m;
                vs_log_((const VsConn *)0, 0, "snapshot started file=%s (%s)", t->path, by_size ? "the journal outgrew the file" : "timer");
            }
            t->last_snap = vs_now_();
        }
    }
}

/* The main thread, while the workers serve, for every open store: reaps finished snapshots (and logs them), flushes
 * the journal every sync_s seconds, and starts a background snapshot every snap_s seconds when something changed, or
 * whenever the journal has outgrown the file (and 64 MB): the file stays the bulk of the data, a restart replays
 * little. */
static void vs_housekeep_(f64 sync_s, f64 snap_s)
{
    f64 last_sync = vs_now_();
    while (!vs_stop) {
        struct timespec ts;
        VsTen *t;
        i32 sync = sync_s > 0 && vs_now_() - last_sync >= sync_s;
        ts.tv_sec = 0; ts.tv_nsec = 100000000L;
        nanosleep(&ts, (struct timespec *)0);
        for (t = vs_first_(); t; t = t->next) vs_keep_(t, sync, snap_s);
        if (sync) last_sync = vs_now_();
    }
}

int main(int argc, char **argv)
{
    const char *sock = (const char *)0, *bind_addr = "127.0.0.1", *cert = (const char *)0, *key = (const char *)0, *ca = (const char *)0;
    const char *logdest = (const char *)0, *path = (const char *)0;
    i32 port = 0, nth = 0, dim = 0, save_on_exit = 0, created = 0, memrc = 0, i, j;
    i64 hot = 0, cold = 0;
    f32 aggr = -1.0f;
    f64 sync_s = 1.0, snap_s = 0.0;
    size_t outcap;
    VsWorker *ws;
    VsTen *t;
    struct sigaction sa;
    struct stat st;
    char tail[320];
    double t0;

    for (i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-s") && i + 1 < argc) sock = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-b") && i + 1 < argc) bind_addr = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) nth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-g") && i + 1 < argc) aggr = (f32)atof(argv[++i]);
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) vs_mem = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-x")) save_on_exit = 1;
        else if (!strcmp(argv[i], "-L")) vs_live = 1;
        else if (!strcmp(argv[i], "-C")) vs_cascade = 1;
        else if (!strcmp(argv[i], "-M") && i + 1 < argc) {
            const char *m = argv[++i];
            if (!strcmp(m, "cosine")) vs_metric = 0; else if (!strcmp(m, "dot")) vs_metric = 1; else if (!strcmp(m, "l2")) vs_metric = 2;
            else { fprintf(stderr, "vserv: -M is cosine, dot or l2, not %s\n", m); return 2; }
        }
        else if (!strcmp(argv[i], "-U")) vs_ten_cn = 1;
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) vs_recall = (f32)atof(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) cert = argv[++i];
        else if (!strcmp(argv[i], "-k") && i + 1 < argc) key = argv[++i];
        else if (!strcmp(argv[i], "-a") && i + 1 < argc) ca = argv[++i];
        else if (!strcmp(argv[i], "-P")) vs_plain_ok = 1;
        else if (!strcmp(argv[i], "-T")) vs_unix_tls_only = 1;
        else if (!strcmp(argv[i], "-f") && i + 1 < argc) sync_s = atof(argv[++i]);
        else if (!strcmp(argv[i], "-S") && i + 1 < argc) snap_s = atof(argv[++i]);
        else if (!strcmp(argv[i], "-l") && i + 1 < argc) logdest = argv[++i];
        else if (!strcmp(argv[i], "-A")) vs_log_all = 1;
        else if (argv[i][0] != '-' && !path) path = argv[i];
        else {
            fprintf(stderr, "usage: vserv [-s SOCK] [-p PORT] [-b ADDR] [-t THREADS] [-g A] [-m TIER] [-C] [-d DIM] [-x] [-L] [-r RECALL]\n"
                            "             [-f SECONDS] [-S SECONDS] [-l syslog|-|FILE] [-A] [-c CERT -k KEY [-a CA] [-P] [-T]] [-U]\n"
                            "             STORE.vxi | DIR\n");
            return 2;
        }
    }
    if (logdest && vs_log_open_(logdest) != 0) { fprintf(stderr, "vserv: cannot open the log %s: %s\n", logdest, strerror(errno)); return 1; }
    if (!path) { fprintf(stderr, "vserv: no store file or tenant directory given\n"); return 2; }
    if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) {           /* a directory: a store per tenant in it */
        char *d = (char *)malloc(strlen(path) + 1);
        size_t n;
        if (!d) return 1;
        strcpy(d, path);
        for (n = strlen(d); n > 1 && d[n - 1] == '/'; --n) d[n - 1] = 0;
        vs_dir = d;
        if (dim <= 0) { fprintf(stderr, "vserv: a tenant directory needs -d DIM, the dimension of its stores\n"); return 2; }
        vs_dim = dim;
    }
    if (vs_ten_cn && !vs_dir) { fprintf(stderr, "vserv: -U binds clients to tenants: give a directory\n"); return 2; }
    if ((cert || key || ca || vs_unix_tls_only) && !(cert && key)) { fprintf(stderr, "vserv: TLS needs both -c CERT and -k KEY\n"); return 2; }
#ifdef VSERV_TLS
    if (cert) {
        if (vs_tls_load_(cert, key, ca)) return 1;
        vs_tls_on = 1;
        vs_cache_mem = (unsigned char *)malloc(VS_CACHE_BYTES);
        br_ssl_session_cache_lru_init(&vs_cache.lru, vs_cache_mem, VS_CACHE_BYTES);
        pthread_mutex_init(&vs_cache.mu, (const pthread_mutexattr_t *)0);
        vs_cache.vtable = &vs_cache_class;
    }
#else
    if (cert) { fprintf(stderr, "vserv: built without TLS (VSERV_NO_TLS)\n"); return 2; }
#endif
    if (!vs_tls_on) vs_plain_ok = 1;
    if (!sock && !port) sock = "vecindex.sock";
    if (aggr < 0.0f) { const char *ag = getenv("VECINDEX_AGGRESSION"); aggr = ag && *ag ? (f32)atof(ag) : 1.0f; }
    if (aggr < 0.0f) aggr = 0.0f;
    if (aggr > 1.0f) aggr = 1.0f;
    vs_aggr = aggr;
    if (nth <= 0) { nth = (i32)(aggr * (f32)store_cpus_avail_() + 0.5f); if (nth < 1) nth = 1; }

    t0 = vs_now_();
    if (!vs_dir) {                                               /* one store */
        vs_one.path = path;
        vs_one.s = store_load(&vs_arena, path);
        if (!vs_one.s) {
            if (stat(path, &st) == 0) {
                fprintf(stderr, "vserv: %s exists but cannot be read as a store%s%s\n", path, *store_load_error() ? ": " : "", store_load_error());
                return 1;
            }
            if (dim <= 0) { fprintf(stderr, "vserv: cannot load %s (give -d DIM to start a new store)\n", path); return 1; }
            vs_one.s = store_new(&vs_arena, dim);
            store_set_metric(vs_one.s, vs_metric);
            if (store_save(vs_one.s, path) != 0) { fprintf(stderr, "vserv: cannot create %s: %s\n", path, strerror(errno)); return 1; }
            created = 1;                                         /* the file and its journal exist from the first write on */
        }
        vs_dim = vs_one.s->dim;
        memrc = vs_ten_setup_(&vs_one);
        {   i64 nb, bad, fx;
            i32 jd;
            store_integrity(vs_one.s, &nb, &bad, &fx, (i64 *)0, (i64 *)0, &jd);
            if (bad || jd) fprintf(stderr, "vserv: %s: %.0f of %.0f blocks were damaged, %.0f rebuilt%s\n", path, (double)bad, (double)nb,
                                   (double)fx, jd ? "; the journal had a damaged record (its copy: .j.damaged)" : ""); }
        store_tiers(vs_one.s, &hot, &cold);
    }
    /* one request: headers + a JSON vector (~24 bytes a number) or the binary frame */
    vs_reqcap = 2048 + vs_dim * 24;
    if (vs_reqcap < 4096) vs_reqcap = 4096;
    vs_reqmax = vs_reqcap + VS_REQ_TEXT;
    outcap = (size_t)VS_OUT_CAP;                                 /* replies: VS_MAXK hits, or one vector as text */
    if (outcap < 512 + (size_t)vs_dim * 16) outcap = 512 + (size_t)vs_dim * 16;
#ifdef VSERV_TLS
    if (vs_ntas_()) vs_names = (VsName *)calloc(VS_NAMES, sizeof(VsName));
#endif

    if (sock) { vs_ls[vs_nls].fd = vs_listen_unix_(sock); if (vs_ls[vs_nls].fd < 0) { fprintf(stderr, "vserv: cannot listen on %s: %s\n", sock, strerror(errno)); return 1; } ++vs_nls; }
    if (port) { vs_ls[vs_nls].tcp = 1; vs_ls[vs_nls].fd = vs_listen_tcp_(bind_addr, port); if (vs_ls[vs_nls].fd < 0) { fprintf(stderr, "vserv: cannot listen on %s:%d: %s\n", bind_addr, port, strerror(errno)); return 1; } ++vs_nls; }

    memset(&sa, 0, sizeof(sa)); sa.sa_handler = vs_on_signal_;
    sigaction(SIGINT, &sa, (struct sigaction *)0); sigaction(SIGTERM, &sa, (struct sigaction *)0);
    signal(SIGPIPE, SIG_IGN);

    ws = (VsWorker *)calloc((size_t)nth, sizeof(VsWorker));
    for (i = 0; i < nth; ++i) {
        VsWorker *w = &ws[i];
        w->ep = epoll_create1(EPOLL_CLOEXEC);
        w->out = (char *)malloc(outcap);
        w->vec = (f32 *)malloc((size_t)vs_dim * sizeof(f32));
        w->ids = (i64 *)malloc(VS_MAXK * sizeof(i64));
        w->sc = (f32 *)malloc(VS_MAXK * sizeof(f32));
        for (j = 0; j < vs_nls; ++j) {               /* every worker waits on every listener; the kernel wakes one */
            struct epoll_event e;
            e.events = EPOLLIN | EPOLLEXCLUSIVE; e.data.ptr = &vs_ls[j];
            epoll_ctl(w->ep, EPOLL_CTL_ADD, vs_ls[j].fd, &e);
        }
    }
    snprintf(tail, sizeof(tail), "%d threads;%s%s%s%s%s%s%s", nth, vs_live ? " live;" : "", sock ? " unix " : "", sock ? sock : "",
             port ? " tcp " : "", port ? bind_addr : "", vs_tls_on ? (vs_ntas_() ? " (TLS, client certificates required)" : " (TLS)") : "",
             vs_tls_on ? (vs_plain_ok ? " + plaintext on tcp" : (vs_unix_tls_only ? "" : "; plaintext on unix only")) : "");
    if (vs_dir) {
        fprintf(stderr, "vserv: %s: a store per tenant (NAME.vxi, %d dimensions), opened on first use%s; %s\n", vs_dir, vs_dim,
                vs_ten_cn ? ", a client certificate names its tenant" : "", tail);
        vs_log_((const VsConn *)0, 0, "start dir=%s dim=%d threads=%d%s%s%s memory=%d", vs_dir, vs_dim, nth, vs_live ? " live" : "",
                vs_tls_on ? " tls" : "", vs_ten_cn ? " tenant=certificate" : "", (int)vs_mem);
    } else {
        fprintf(stderr, "vserv: %s: %d rows x %d, loaded in %.1f ms; %s\n", path, store_count(vs_one.s), vs_dim, (vs_now_() - t0) * 1e3, tail);
        vs_log_((const VsConn *)0, 0, "start file=%s rows=%d dim=%d threads=%d%s%s%s journal=%s memory=%d hot=%.0f cold=%.0f", path,
                store_count(vs_one.s), vs_dim, nth, created ? " created" : "", vs_live ? " live" : "", vs_tls_on ? " tls" : "",
                store_journal_bytes(vs_one.s, path) >= 0 ? "on" : "off", (int)vs_mem, (double)hot, (double)cold);
        if (memrc != 0)
            fprintf(stderr, "vserv: the memlock limit refused part of -m %d; that part is only read ahead (raise it: ulimit -l, "
                            "or LimitMEMLOCK=infinity in the systemd unit)\n", (int)vs_mem);
        if (store_journal_bytes(vs_one.s, path) < 0)
            fprintf(stderr, "vserv: no journal for %s: writes are durable only once saved or snapshotted\n", path);
    }
    for (i = 0; i < nth; ++i) pthread_create(&ws[i].th, (const pthread_attr_t *)0, vs_worker_, &ws[i]);
    vs_housekeep_(sync_s, snap_s);
    for (i = 0; i < nth; ++i) pthread_join(ws[i].th, (void **)0);

    for (j = 0; j < vs_nls; ++j) close(vs_ls[j].fd);
    if (sock) unlink(sock);
    for (t = vs_first_(); t; t = t->next) {
        if (save_on_exit) {
            if (store_save(t->s, t->path) == 0) fprintf(stderr, "vserv: saved %s\n", t->path);
            else fprintf(stderr, "vserv: saving %s failed\n", t->path);
        } else if (store_journal_bytes(t->s, t->path) >= 0)
            store_sync(t->s, t->path);                           /* everything written so far is on the disk */
        store_snapshot_poll(t->s, 1);
        if (vs_dir) vs_log_((const VsConn *)0, 0, "stop name=%s rows=%d", t->name, store_count(t->s));
        else vs_log_((const VsConn *)0, 0, "stop rows=%d", store_count(t->s));
    }
    for (i = 0; i < nth; ++i) {
        close(ws[i].ep); free(ws[i].out); free(ws[i].vec); free(ws[i].ids); free(ws[i].sc);
    }
#ifdef VSERV_TLS
    explicit_bzero(&vs_skey, sizeof(vs_skey));
    if (vs_cache_mem) { explicit_bzero(vs_cache_mem, VS_CACHE_BYTES); free(vs_cache_mem); }   /* master secrets */
#endif
    free(ws);
    return 0;
}
