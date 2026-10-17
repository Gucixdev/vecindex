/* vserv.c - vecindex over a socket: libc + pthreads + Linux epoll, and TLS through the vendored BearSSL
 * (third_party/bearssl, MIT, no malloc inside). Nothing else.
 *
 *   vserv [-s SOCK] [-p PORT] [-b ADDR] [-t THREADS] [-d DIM] [-x] [-c CERT.pem -k KEY.pem [-a CA.pem] [-P]] STORE.vxi
 *     -s PATH   Unix socket, created mode 0600 (default ./vecindex.sock when -p is not given)
 *     -p PORT   TCP port, bound to 127.0.0.1 unless -b gives another address
 *     -t N      worker threads (default: online CPUs)
 *     -d DIM    dimension of a new, empty store when STORE.vxi does not exist yet
 *     -x        save the store to STORE.vxi on SIGINT / SIGTERM
 *     -c, -k    TLS on the TCP port: certificate chain and private key (PEM, EC or RSA). The port then takes only
 *               TLS: HTTPS, or the binary protocol inside TLS (-P lets plaintext in too)
 *     -a FILE   mutual TLS: a client must present a certificate that chains to one of the CA certificates in FILE
 *               (or is one of them), else the handshake fails
 *
 * The store is opened with store_load (mmap: ready in milliseconds whatever its size). Every socket speaks
 * two protocols, told apart by the first byte of a request:
 *
 *   binary (first byte 0xB5): fixed little-endian frames, nothing to parse
 *     request  u8 0xB5 | u8 op | u16 k | u32 nbytes | payload[nbytes]
 *     response u8 0xB5 | u8 status (0 ok) | u16 count | u32 nbytes | payload[nbytes]
 *     op 1 search  payload: dim x f32            reply: count x { i64 id, f32 score }
 *     op 2 add     payload: i64 id, dim x f32
 *     op 3 delete  payload: i64 id               reply count: rows removed
 *     op 4 build   op 5 save   op 6 stats        stats reply: i64 live, dim, pending, built
 *
 *   HTTP/1.1 (anything else), for curl and friends; keep-alive, Content-Length bodies:
 *     GET /health | GET /stats | POST /search?k=10 | POST /add?id=N | POST /delete?id=N | POST /build | POST /save
 *     vectors as a JSON array ([0.1, -0.2, ...]) or raw little-endian f32 with
 *     Content-Type: application/octet-stream; replies are JSON.
 *
 * TLS (when -c/-k are given): a connection is TLS when its first byte is a handshake record (0x16). TLS 1.2 only,
 * ECDHE key exchange (forward secrecy), AEAD suites only (AES-GCM - AES-NI when the CPU has it - and
 * ChaCha20-Poly1305), sessions resumable from a per-thread cache. Client certificates with -a. The Unix socket
 * stays plaintext: it never leaves the machine and its 0600 mode decides who may connect.
 *
 * Memory: an idle connection is one small struct. Its request buffer (one request of this store's
 * dimension) is allocated on first read and reused for every later request on it; responses are built in a
 * per-thread buffer, so a request allocates nothing (a client that stops reading gets its unsent tail
 * buffered until it catches up). TLS adds about 37 KB a connection: the engine state and room for one full
 * 16 KB record in each direction, which the protocol requires.
 *
 *   ./run builds it (BearSSL first, into .build/libbearssl.a); by hand:
 *   gcc -std=c89 -pedantic -Wall -Wextra -O2 -march=native -I. -isystem third_party/bearssl/inc vserv.c \
 *       .build/libbearssl.a -lm -lpthread -o vserv          (-DVSERV_NO_TLS: plaintext only, no BearSSL)
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "store.c"
#include <errno.h>
#include <signal.h>
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

typedef struct VsTls VsTls;

typedef struct VsConn {
    i32 fd, len, cap, out_len, out_off, closing;
    u8 tcp, checked;                            /* accepted on the TCP port; its first byte looked at (TLS or not) */
    char *buf;                                  /* request bytes (plaintext), allocated on first read */
    char *out;                                  /* reply bytes not yet out (under TLS: not yet in the engine) */
    double last;
    VsTls *tls;                                 /* NULL: plaintext */
    struct VsConn *prev, *next;                 /* the worker's list, for the idle sweep */
} VsConn;

typedef struct { i32 fd, tcp; } VsListen;       /* epoll data for a listening socket */

typedef struct {
    i32 ep;
    pthread_t th;
    VsConn *conns;
    char *out;                                  /* VS_OUT_CAP reply scratch */
    f32 *vec;                                   /* dim floats: an aligned copy of the query */
    i64 *ids;
    f32 *sc;
#ifdef VSERV_TLS
    br_ssl_session_cache_lru lru;               /* resumable sessions of this thread's connections */
    unsigned char *lru_mem;
#endif
} VsWorker;

#ifdef VSERV_TLS
struct VsTls {
    br_ssl_server_context sc;
    br_x509_minimal_context xc;                 /* validates client certificates (-a) */
    unsigned char io[BR_SSL_BUFSIZE_BIDI];      /* one full record each way */
};
static br_x509_certificate *vs_chain;
static size_t vs_nchain;
static br_skey_decoder_context vs_skey;         /* the decoded private key, for the server's lifetime */
static i32 vs_keytype, vs_issuer;
static br_x509_trust_anchor *vs_tas;
static size_t vs_ntas;
#else
struct VsTls { i32 unused; };
#endif
static i32 vs_tls_on, vs_plain_ok;

static RagStore *vs_store;
static const char *vs_path;
static i32 vs_dim, vs_reqcap;
static VsListen vs_ls[2];
static i32 vs_nls;
static volatile sig_atomic_t vs_stop;

static double vs_now_(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec; }
static void vs_on_signal_(int sig) { (void)sig; vs_stop = 1; }

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

/* a score with 6 decimals; scores are -||q-v||^2 on unit vectors, so within [-4, 0] */
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

static i32 vs_search_(VsWorker *w, const f32 *q, i32 k) { return store_search(vs_store, q, k, w->ids, w->sc); }

static void vs_stats_(i64 *v)
{
    v[0] = store_count(vs_store); v[1] = vs_dim; v[2] = store_pending(vs_store); v[3] = store_built(vs_store);
}

/* ---- binary protocol ---- */

static void vs_bin_reply_(VsWorker *w, VsConn *c, u8 status, u16 count, const char *payload, u32 nbytes)
{
    char *o = w->out;
    o[0] = (char)VS_MAGIC; o[1] = (char)status;
    o[2] = (char)(count & 0xff); o[3] = (char)(count >> 8);
    o[4] = (char)(nbytes & 0xff); o[5] = (char)((nbytes >> 8) & 0xff); o[6] = (char)((nbytes >> 16) & 0xff); o[7] = (char)(nbytes >> 24);
    if (nbytes && payload != o + 8) memmove(o + 8, payload, nbytes);
    vs_send_(w, c, o, (i32)(8 + nbytes));
}

static u32 vs_le32_(const u8 *p) { return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24); }
static i64 vs_le64_(const u8 *p) { return (i64)((u64)vs_le32_(p) | ((u64)vs_le32_(p + 4) << 32)); }

/* returns bytes consumed, 0 if the frame is incomplete, -1 to close */
static i32 vs_binary_(VsWorker *w, VsConn *c)
{
    const u8 *b = (const u8 *)c->buf;
    u32 nb;
    i32 op, k, i, got;
    if (c->len < 8) return 0;
    op = b[1]; k = (i32)(b[2] | (b[3] << 8)); nb = vs_le32_(b + 4);
    if (nb > (u32)vs_reqcap - 8) return -1;
    if ((u32)c->len < 8 + nb) return 0;
    b += 8;
    switch (op) {
    case 1:
        if (nb != (u32)vs_dim * 4) { vs_bin_reply_(w, c, 2, 0, (const char *)0, 0); break; }
        memcpy(w->vec, b, nb);
        if (k < 1) k = 10;
        if (k > VS_MAXK) k = VS_MAXK;
        got = vs_search_(w, w->vec, k);
        {   char *o = w->out + 8;
            for (i = 0; i < got; ++i) { memcpy(o, &w->ids[i], 8); memcpy(o + 8, &w->sc[i], 4); o += 12; }
            vs_bin_reply_(w, c, 0, (u16)got, w->out + 8, (u32)got * 12u); }
        break;
    case 2:
        if (nb != 8 + (u32)vs_dim * 4) { vs_bin_reply_(w, c, 2, 0, (const char *)0, 0); break; }
        memcpy(w->vec, b + 8, (size_t)vs_dim * 4);
        vs_bin_reply_(w, c, store_add(vs_store, w->vec, vs_le64_(b)) == 0 ? 0 : 1, 0, (const char *)0, 0);
        break;
    case 3:
        if (nb != 8) { vs_bin_reply_(w, c, 2, 0, (const char *)0, 0); break; }
        vs_bin_reply_(w, c, 0, (u16)store_delete(vs_store, vs_le64_(b)), (const char *)0, 0);
        break;
    case 4: store_build(vs_store); vs_bin_reply_(w, c, 0, 0, (const char *)0, 0); break;
    case 5: vs_bin_reply_(w, c, (u8)(vs_path && store_save(vs_store, vs_path) == 0 ? 0 : 1), 0, (const char *)0, 0); break;
    case 6: { i64 v[4]; vs_stats_(v); vs_bin_reply_(w, c, 0, 4, (const char *)v, sizeof(v)); } break;
    default: vs_bin_reply_(w, c, 3, 0, (const char *)0, 0); break;
    }
    return (i32)(8 + nb);
}

/* ---- HTTP/1.1 ---- */

/* offset just past "\r\n\r\n" in p[0..n), or -1. SSE2: 16 bytes per step looking for '\r' (the
 * picohttpparser idea, reduced to what this server needs); scalar check of each hit. */
static i32 vs_head_end_(const char *p, i32 n)
{
    i32 i = 0;
#if defined(__SSE2__)
    const __m128i cr = _mm_set1_epi8('\r');
    for (; i + 16 <= n; i += 16) {
        i32 m = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)(const void *)(p + i)), cr));
        while (m) {
            i32 j = i + tq_ctz_((u32)m);
            m &= m - 1;
            if (j + 3 < n && p[j + 1] == '\n' && p[j + 2] == '\r' && p[j + 3] == '\n') return j + 4;
        }
    }
    if (i >= 3) i -= 3;                          /* a terminator may straddle the last full block */
#endif
    for (; i + 3 < n; ++i)
        if (p[i] == '\r' && p[i + 1] == '\n' && p[i + 2] == '\r' && p[i + 3] == '\n') return i + 4;
    return -1;
}

static i32 vs_ieq_(const char *a, const char *b, i32 n)
{
    i32 i;
    for (i = 0; i < n; ++i) {
        char x = a[i], y = b[i];
        if (x >= 'A' && x <= 'Z') x = (char)(x + 32);
        if (x != y) return 0;
    }
    return 1;
}

/* value of query parameter name= in the request target [t, t+tn), or def */
static i64 vs_qparam_(const char *t, i32 tn, const char *name, i64 def)
{
    i32 nl = (i32)strlen(name), i;
    for (i = 0; i < tn; ++i) {
        if ((t[i] == '?' || t[i] == '&') && i + 1 + nl < tn && !memcmp(t + i + 1, name, (size_t)nl) && t[i + 1 + nl] == '=') {
            i64 v = 0, sg = 1;
            i32 j = i + 2 + nl;
            if (j < tn && t[j] == '-') { sg = -1; ++j; }
            for (; j < tn && t[j] >= '0' && t[j] <= '9'; ++j) v = v * 10 + (t[j] - '0');
            return v * sg;
        }
    }
    return def;
}

/* JSON array of exactly vs_dim numbers -> w->vec; 0 ok, -1 bad */
static i32 vs_json_vec_(VsWorker *w, const char *p, i32 n)
{
    const char *e = p + n;
    char num[64];
    i32 d = 0;
    while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
    if (p >= e || *p != '[') return -1;
    ++p;
    for (;;) {
        i32 k = 0;
        while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
        if (p < e && *p == ']' && d == 0) break;
        while (p < e && k < 63 && ((*p >= '0' && *p <= '9') || *p == '-' || *p == '+' || *p == '.' || *p == 'e' || *p == 'E')) num[k++] = *p++;
        if (k == 0 || d >= vs_dim) return -1;
        num[k] = 0;
        w->vec[d++] = (f32)strtod(num, (char **)0);
        while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
        if (p < e && *p == ',') { ++p; continue; }
        if (p < e && *p == ']') break;
        return -1;
    }
    return d == vs_dim ? 0 : -1;
}

static void vs_http_reply_(VsWorker *w, VsConn *c, const char *status, const char *body, i32 blen, i32 keep)
{
    char *o = w->out + VS_OUT_CAP - 256, *p = o;   /* headers go in front of the body, built in place */
    char head[256];
    i32 hn;
    p = head;
    p = vs_str_(p, "HTTP/1.1 "); p = vs_str_(p, status);
    p = vs_str_(p, "\r\nContent-Type: application/json\r\nContent-Length: ");
    p = vs_i64_(p, blen);
    p = vs_str_(p, keep ? "\r\n\r\n" : "\r\nConnection: close\r\n\r\n");
    hn = (i32)(p - head);
    (void)o;
    if (body == w->out + 256) {                  /* body already sits after a 256-byte gap: prepend in place */
        memcpy(w->out + 256 - hn, head, (size_t)hn);
        vs_send_(w, c, w->out + 256 - hn, hn + blen);
    } else {
        vs_send_(w, c, head, hn);
        if (blen) vs_send_(w, c, body, blen);
    }
    if (!keep) c->closing = 1;
}

/* returns bytes consumed, 0 if incomplete, -1 to close */
static i32 vs_http_(VsWorker *w, VsConn *c)
{
    const char *b = c->buf, *line, *t, *body;
    i32 he = vs_head_end_(b, c->len), i, mlen, tn, clen = 0, keep = 1, raw = 0, got, k;
    char *o;
    if (he < 0) {
        if (c->len >= vs_reqcap) { vs_http_reply_(w, c, "431 Request Header Fields Too Large", "{\"error\":\"headers too large\"}", 29, 0); return -1; }
        return 0;
    }
    /* request line: METHOD SP TARGET SP HTTP/1.x */
    for (mlen = 0; mlen < he && b[mlen] != ' '; ++mlen) {}
    t = b + mlen + 1;
    for (tn = 0; mlen + 1 + tn < he && t[tn] != ' ' && t[tn] != '\r'; ++tn) {}
    if (mlen >= he || tn == 0) { vs_http_reply_(w, c, "400 Bad Request", "{\"error\":\"bad request line\"}", 28, 0); return -1; }
    if (mlen + 1 + tn + 9 <= he && !memcmp(t + tn, " HTTP/1.0", 9)) keep = 0;
    /* headers */
    for (line = b; ; ) {
        const char *nl = (const char *)memchr(line, '\n', (size_t)(b + he - line));
        i32 ll;
        if (!nl) break;
        line = nl + 1;
        ll = (i32)((const char *)memchr(line, '\n', (size_t)(b + he - line + 1)) ? 0 : 0);
        (void)ll;
        if (line >= b + he - 2) break;
        if (b + he - line > 15 && vs_ieq_(line, "content-length:", 15)) {
            const char *v = line + 15;
            while (*v == ' ') ++v;
            for (clen = 0; *v >= '0' && *v <= '9'; ++v) { clen = clen * 10 + (*v - '0'); if (clen > vs_reqcap) break; }
        } else if (b + he - line > 13 && vs_ieq_(line, "content-type:", 13)) {
            const char *v = line + 13;
            while (*v == ' ') ++v;
            if (b + he - v > 24 && vs_ieq_(v, "application/octet-stream", 24)) raw = 1;
        } else if (b + he - line > 11 && vs_ieq_(line, "connection:", 11)) {
            const char *v = line + 11;
            while (*v == ' ') ++v;
            if (b + he - v >= 5 && vs_ieq_(v, "close", 5)) keep = 0;
            if (b + he - v >= 10 && vs_ieq_(v, "keep-alive", 10)) keep = 1;
        }
    }
    if (clen > vs_reqcap - he) { vs_http_reply_(w, c, "413 Content Too Large", "{\"error\":\"body too large\"}", 26, 0); return -1; }
    if (c->len < he + clen) return 0;
    body = b + he;
    o = w->out + 256;                             /* bodies are written here; vs_http_reply_ prepends headers */

#define VS_IS(m, path) (mlen == (i32)strlen(m) && !memcmp(b, m, (size_t)mlen) && tn >= (i32)strlen(path) \
                        && !memcmp(t, path, strlen(path)) && (tn == (i32)strlen(path) || t[strlen(path)] == '?'))
    if (VS_IS("GET", "/health")) {
        vs_http_reply_(w, c, "200 OK", "{\"ok\":true}", 11, keep);
    } else if (VS_IS("GET", "/stats")) {
        i64 v[4]; char *p = o;
        vs_stats_(v);
        p = vs_str_(p, "{\"count\":"); p = vs_i64_(p, v[0]); p = vs_str_(p, ",\"dim\":"); p = vs_i64_(p, v[1]);
        p = vs_str_(p, ",\"pending\":"); p = vs_i64_(p, v[2]); p = vs_str_(p, ",\"built\":"); p = vs_str_(p, v[3] ? "true}" : "false}");
        vs_http_reply_(w, c, "200 OK", o, (i32)(p - o), keep);
    } else if (VS_IS("POST", "/search")) {
        char *p = o;
        if (raw ? clen != vs_dim * 4 : vs_json_vec_(w, body, clen) != 0) {
            vs_http_reply_(w, c, "400 Bad Request", "{\"error\":\"expected a vector of the store's dimension\"}", 54, keep);
        } else {
            if (raw) memcpy(w->vec, body, (size_t)vs_dim * 4);
            k = (i32)vs_qparam_(t, tn, "k", 10);
            if (k < 1) k = 1;
            if (k > VS_MAXK) k = VS_MAXK;
            got = vs_search_(w, w->vec, k);
            p = vs_str_(p, "{\"ids\":[");
            for (i = 0; i < got; ++i) { if (i) *p++ = ','; p = vs_i64_(p, w->ids[i]); }
            p = vs_str_(p, "],\"scores\":[");
            for (i = 0; i < got; ++i) { if (i) *p++ = ','; p = vs_f32_(p, w->sc[i]); }
            p = vs_str_(p, "]}");
            vs_http_reply_(w, c, "200 OK", o, (i32)(p - o), keep);
        }
    } else if (VS_IS("POST", "/add")) {
        i64 id = vs_qparam_(t, tn, "id", -1);
        if (id < 0 || (raw ? clen != vs_dim * 4 : vs_json_vec_(w, body, clen) != 0))
            vs_http_reply_(w, c, "400 Bad Request", "{\"error\":\"need ?id=N and a vector of the store's dimension\"}", 60, keep);
        else {
            if (raw) memcpy(w->vec, body, (size_t)vs_dim * 4);
            if (store_add(vs_store, w->vec, id) == 0) vs_http_reply_(w, c, "200 OK", "{\"ok\":true}", 11, keep);
            else vs_http_reply_(w, c, "400 Bad Request", "{\"error\":\"vector has NaN or Inf\"}", 33, keep);
        }
    } else if (VS_IS("POST", "/delete")) {
        char *p = o;
        p = vs_str_(p, "{\"removed\":"); p = vs_i64_(p, store_delete(vs_store, vs_qparam_(t, tn, "id", -1))); *p++ = '}';
        vs_http_reply_(w, c, "200 OK", o, (i32)(p - o), keep);
    } else if (VS_IS("POST", "/build")) {
        store_build(vs_store);
        vs_http_reply_(w, c, "200 OK", "{\"ok\":true}", 11, keep);
    } else if (VS_IS("POST", "/save")) {
        if (vs_path && store_save(vs_store, vs_path) == 0) vs_http_reply_(w, c, "200 OK", "{\"ok\":true}", 11, keep);
        else vs_http_reply_(w, c, "500 Internal Server Error", "{\"error\":\"save failed\"}", 23, keep);
    } else {
        vs_http_reply_(w, c, "404 Not Found", "{\"error\":\"not found\"}", 21, keep);
    }
#undef VS_IS
    return he + clen;
}

/* ---- connection lifecycle ---- */

static void vs_close_(VsWorker *w, VsConn *c)
{
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
        i32 used = (u8)c->buf[0] == VS_MAGIC ? vs_binary_(w, c) : vs_http_(w, c);
        if (used < 0) { c->closing = 1; break; }
        if (used == 0) break;
        c->len -= used;
        if (c->len) memmove(c->buf, c->buf + used, (size_t)c->len);
    }
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
    if (vs_ntas) {                                 /* mutual TLS: the client's chain must end at one of our CAs */
        br_x509_minimal_init_full(&t->xc, vs_tas, vs_ntas);
        br_ssl_engine_set_default_rsavrfy(&t->sc.eng);
        br_ssl_engine_set_default_ecdsa(&t->sc.eng);
        br_ssl_server_set_trust_anchor_names_alt(&t->sc, vs_tas, vs_ntas);
        br_ssl_engine_set_x509(&t->sc.eng, &t->xc.vtable);
    }
    br_ssl_server_set_cache(&t->sc, &w->lru.vtable);
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
        if (st & BR_SSL_CLOSED) { c->closing = 1; return; }
        if (st & BR_SSL_SENDREC) {                /* handshake messages and replies */
            i32 r = vs_tls_tx_(c);
            if (r < 0) { c->closing = 1; return; }
            if (r == 0) vs_ep_mod_(w, c, EPOLLIN | EPOLLOUT | EPOLLRDHUP | EPOLLET);
        }
        if ((st & BR_SSL_RECVAPP) && !c->out) {   /* requests wait while replies are queued */
            size_t len;
            unsigned char *b = br_ssl_engine_recvapp_buf(e, &len);
            i32 room = c->cap - c->len;
            if (room <= 0) { c->closing = 1; return; }
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
    if (vs_tls_on && c->tcp && !c->checked) {      /* a TLS ClientHello starts with a handshake record, 0x16 */
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
        if (!vs_plain_ok) {
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
        if (c->len == c->cap) { c->closing = 1; return; }  /* the parsers always make room or close */
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

#ifdef VSERV_TLS
/* ---- certificates and keys (PEM), decoded with BearSSL ---- */
typedef struct { unsigned char *p; size_t n, cap; } VsBuf;

static void vs_buf_add_(void *ctx, const void *d, size_t n)
{
    VsBuf *b = (VsBuf *)ctx;
    if (b->n + n > b->cap) { b->cap = (b->n + n) * 2 + 256; b->p = (unsigned char *)realloc(b->p, b->cap); }
    memcpy(b->p + b->n, d, n);
    b->n += n;
}

/* DER of every PEM object in the file whose name contains `want` ("CERTIFICATE", "PRIVATE KEY"); -1 on error */
static i32 vs_pem_(const char *path, const char *want, VsBuf **objs, i32 *nobj)
{
    FILE *f = fopen(path, "rb");
    VsBuf file, cur;
    br_pem_decoder_context pc;
    unsigned char chunk[4096], *p;
    size_t len, k;
    i32 in = 0, nl = 0;
    if (!f) return -1;
    memset(&file, 0, sizeof(file));
    while ((k = fread(chunk, 1, sizeof(chunk), f)) > 0) vs_buf_add_(&file, chunk, k);
    fclose(f);
    *objs = (VsBuf *)0; *nobj = 0;
    memset(&cur, 0, sizeof(cur));
    br_pem_decoder_init(&pc);
    p = file.p; len = file.n;
    for (;;) {
        if (len == 0 && !nl) { p = (unsigned char *)"\n"; len = 1; nl = 1; }   /* the last line needs its newline */
        if (len == 0) break;
        k = br_pem_decoder_push(&pc, p, len);
        p += k; len -= k;
        switch (br_pem_decoder_event(&pc)) {
        case BR_PEM_BEGIN_OBJ:
            in = strstr(br_pem_decoder_name(&pc), want) != (char *)0;
            memset(&cur, 0, sizeof(cur));
            br_pem_decoder_setdest(&pc, in ? vs_buf_add_ : (void (*)(void *, const void *, size_t))0, &cur);
            break;
        case BR_PEM_END_OBJ:
            if (in) { *objs = (VsBuf *)realloc(*objs, (size_t)(*nobj + 1) * sizeof(VsBuf)); (*objs)[(*nobj)++] = cur; }
            in = 0;
            break;
        case BR_PEM_ERROR:
            free(file.p);
            return -1;
        default:
            break;
        }
    }
    explicit_bzero(file.p, file.n);              /* the file may hold a private key */
    free(file.p);
    return 0;
}

static i32 vs_tls_load_(const char *cert, const char *key, const char *ca)
{
    VsBuf *o;
    i32 n, i;
    br_x509_decoder_context dc;
    if (vs_pem_(cert, "CERTIFICATE", &o, &n) || n == 0) { fprintf(stderr, "vserv: no certificate in %s\n", cert); return -1; }
    vs_chain = (br_x509_certificate *)calloc((size_t)n, sizeof(br_x509_certificate));
    for (i = 0; i < n; ++i) { vs_chain[i].data = o[i].p; vs_chain[i].data_len = o[i].n; }
    vs_nchain = (size_t)n;
    free(o);
    br_x509_decoder_init(&dc, (void (*)(void *, const void *, size_t))0, (void *)0);   /* who signed our certificate */
    br_x509_decoder_push(&dc, vs_chain[0].data, vs_chain[0].data_len);
    vs_issuer = (i32)br_x509_decoder_get_signer_key_type(&dc);
    if (vs_issuer == 0) vs_issuer = BR_KEYTYPE_EC;
    if (vs_pem_(key, "PRIVATE KEY", &o, &n) || n == 0) { fprintf(stderr, "vserv: no private key in %s\n", key); return -1; }
    br_skey_decoder_init(&vs_skey);
    br_skey_decoder_push(&vs_skey, o[0].p, o[0].n);
    for (i = 0; i < n; ++i) { explicit_bzero(o[i].p, o[i].n); free(o[i].p); }
    free(o);
    if (br_skey_decoder_last_error(&vs_skey) != 0) { fprintf(stderr, "vserv: cannot decode the key in %s\n", key); return -1; }
    vs_keytype = br_skey_decoder_key_type(&vs_skey);
    if (ca) {                                    /* trust anchors for client certificates */
        if (vs_pem_(ca, "CERTIFICATE", &o, &n) || n == 0) { fprintf(stderr, "vserv: no certificate in %s\n", ca); return -1; }
        vs_tas = (br_x509_trust_anchor *)calloc((size_t)n, sizeof(br_x509_trust_anchor));
        for (i = 0; i < n; ++i) {
            VsBuf dn;
            br_x509_pkey *pk;
            br_x509_trust_anchor *ta = &vs_tas[i];
            memset(&dn, 0, sizeof(dn));
            br_x509_decoder_init(&dc, vs_buf_add_, &dn);
            br_x509_decoder_push(&dc, o[i].p, o[i].n);
            pk = br_x509_decoder_get_pkey(&dc);
            if (!pk) { fprintf(stderr, "vserv: cannot decode certificate %d of %s\n", i + 1, ca); return -1; }
            ta->dn.data = dn.p; ta->dn.len = dn.n;
            ta->flags = br_x509_decoder_isCA(&dc) ? BR_X509_TA_CA : 0;   /* a non-CA anchor trusts that one certificate */
            ta->pkey.key_type = pk->key_type;
            if (pk->key_type == BR_KEYTYPE_RSA) {
                ta->pkey.key.rsa.n = (unsigned char *)malloc(pk->key.rsa.nlen); memcpy(ta->pkey.key.rsa.n, pk->key.rsa.n, pk->key.rsa.nlen);
                ta->pkey.key.rsa.nlen = pk->key.rsa.nlen;
                ta->pkey.key.rsa.e = (unsigned char *)malloc(pk->key.rsa.elen); memcpy(ta->pkey.key.rsa.e, pk->key.rsa.e, pk->key.rsa.elen);
                ta->pkey.key.rsa.elen = pk->key.rsa.elen;
            } else {
                ta->pkey.key.ec.curve = pk->key.ec.curve;
                ta->pkey.key.ec.q = (unsigned char *)malloc(pk->key.ec.qlen); memcpy(ta->pkey.key.ec.q, pk->key.ec.q, pk->key.ec.qlen);
                ta->pkey.key.ec.qlen = pk->key.ec.qlen;
            }
            free(o[i].p);
        }
        free(o);
        vs_ntas = (size_t)n;
    }
    return 0;
}
#endif

int main(int argc, char **argv)
{
    static Arena arena;
    const char *sock = (const char *)0, *bind_addr = "127.0.0.1", *cert = (const char *)0, *key = (const char *)0, *ca = (const char *)0;
    i32 port = 0, nth = 0, dim = 0, save_on_exit = 0, i, j;
    VsWorker *ws;
    struct sigaction sa;
    double t0;

    for (i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-s") && i + 1 < argc) sock = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-b") && i + 1 < argc) bind_addr = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) nth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-x")) save_on_exit = 1;
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) cert = argv[++i];
        else if (!strcmp(argv[i], "-k") && i + 1 < argc) key = argv[++i];
        else if (!strcmp(argv[i], "-a") && i + 1 < argc) ca = argv[++i];
        else if (!strcmp(argv[i], "-P")) vs_plain_ok = 1;
        else if (argv[i][0] != '-' && !vs_path) vs_path = argv[i];
        else { fprintf(stderr, "usage: vserv [-s SOCK] [-p PORT] [-b ADDR] [-t THREADS] [-d DIM] [-x] [-c CERT -k KEY [-a CA] [-P]] STORE.vxi\n"); return 2; }
    }
    if (!vs_path) { fprintf(stderr, "vserv: no store file given\n"); return 2; }
    if ((cert || key || ca) && !(cert && key)) { fprintf(stderr, "vserv: TLS needs both -c CERT and -k KEY\n"); return 2; }
#ifdef VSERV_TLS
    if (cert) { if (vs_tls_load_(cert, key, ca)) return 1; vs_tls_on = 1; }
#else
    if (cert) { fprintf(stderr, "vserv: built without TLS (VSERV_NO_TLS)\n"); return 2; }
#endif
    if (!vs_tls_on) vs_plain_ok = 1;
    if (!sock && !port) sock = "vecindex.sock";
    if (nth <= 0) { long c = sysconf(_SC_NPROCESSORS_ONLN); nth = c > 0 ? (i32)c : 1; }

    t0 = vs_now_();
    vs_store = store_load(&arena, vs_path);
    if (!vs_store) {
        if (dim <= 0) { fprintf(stderr, "vserv: cannot load %s (give -d DIM to start a new store)\n", vs_path); return 1; }
        vs_store = store_new(&arena, dim);
    }
    vs_dim = vs_store->dim;
    store_set_threads(vs_store, 1);              /* one query per thread: the workers are the parallelism */
    /* one request: headers + a JSON vector (~24 bytes a number) or the binary frame */
    vs_reqcap = 2048 + vs_dim * 24;
    if (vs_reqcap < 4096) vs_reqcap = 4096;

    if (sock) { vs_ls[vs_nls].fd = vs_listen_unix_(sock); if (vs_ls[vs_nls].fd < 0) { fprintf(stderr, "vserv: cannot listen on %s: %s\n", sock, strerror(errno)); return 1; } ++vs_nls; }
    if (port) { vs_ls[vs_nls].tcp = 1; vs_ls[vs_nls].fd = vs_listen_tcp_(bind_addr, port); if (vs_ls[vs_nls].fd < 0) { fprintf(stderr, "vserv: cannot listen on %s:%d: %s\n", bind_addr, port, strerror(errno)); return 1; } ++vs_nls; }

    memset(&sa, 0, sizeof(sa)); sa.sa_handler = vs_on_signal_;
    sigaction(SIGINT, &sa, (struct sigaction *)0); sigaction(SIGTERM, &sa, (struct sigaction *)0);
    signal(SIGPIPE, SIG_IGN);

    ws = (VsWorker *)calloc((size_t)nth, sizeof(VsWorker));
    for (i = 0; i < nth; ++i) {
        VsWorker *w = &ws[i];
        w->ep = epoll_create1(EPOLL_CLOEXEC);
        w->out = (char *)malloc(VS_OUT_CAP);
        w->vec = (f32 *)malloc((size_t)vs_dim * sizeof(f32));
        w->ids = (i64 *)malloc(VS_MAXK * sizeof(i64));
        w->sc = (f32 *)malloc(VS_MAXK * sizeof(f32));
#ifdef VSERV_TLS
        w->lru_mem = (unsigned char *)malloc(32768);   /* ~300 resumable sessions a thread */
        br_ssl_session_cache_lru_init(&w->lru, w->lru_mem, 32768);
#endif
        for (j = 0; j < vs_nls; ++j) {               /* every worker waits on every listener; the kernel wakes one */
            struct epoll_event e;
            e.events = EPOLLIN | EPOLLEXCLUSIVE; e.data.ptr = &vs_ls[j];
            epoll_ctl(w->ep, EPOLL_CTL_ADD, vs_ls[j].fd, &e);
        }
    }
    fprintf(stderr, "vserv: %s: %d rows x %d, loaded in %.1f ms; %d threads;%s%s%s%s%s%s\n", vs_path, store_count(vs_store), vs_dim,
            (vs_now_() - t0) * 1e3, nth, sock ? " unix " : "", sock ? sock : "", port ? " tcp " : "", port ? bind_addr : "",
            vs_tls_on ? (vs_ntas_() ? " (TLS, client certificates required)" : " (TLS)") : "", vs_tls_on && vs_plain_ok ? " + plaintext" : "");
    for (i = 0; i < nth; ++i) pthread_create(&ws[i].th, (const pthread_attr_t *)0, vs_worker_, &ws[i]);
    for (i = 0; i < nth; ++i) pthread_join(ws[i].th, (void **)0);

    for (j = 0; j < vs_nls; ++j) close(vs_ls[j].fd);
    if (sock) unlink(sock);
    if (save_on_exit) {
        if (store_save(vs_store, vs_path) == 0) fprintf(stderr, "vserv: saved %s\n", vs_path);
        else fprintf(stderr, "vserv: saving %s failed\n", vs_path);
    }
    for (i = 0; i < nth; ++i) {
        close(ws[i].ep); free(ws[i].out); free(ws[i].vec); free(ws[i].ids); free(ws[i].sc);
#ifdef VSERV_TLS
        explicit_bzero(ws[i].lru_mem, 32768); free(ws[i].lru_mem);
#endif
    }
#ifdef VSERV_TLS
    explicit_bzero(&vs_skey, sizeof(vs_skey));
#endif
    free(ws);
    return 0;
}
