/* vserv.c - vecindex over a socket. One file, libc + pthreads, Linux epoll; no other dependency.
 *
 *   vserv [-s SOCK] [-p PORT] [-b ADDR] [-t THREADS] [-d DIM] [-x] STORE.vxi
 *     -s PATH   Unix socket, created mode 0600 (default ./vecindex.sock when -p is not given)
 *     -p PORT   TCP port, bound to 127.0.0.1 unless -b gives another address
 *     -t N      worker threads (default: online CPUs)
 *     -d DIM    dimension of a new, empty store when STORE.vxi does not exist yet
 *     -x        save the store to STORE.vxi on SIGINT / SIGTERM
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
 * Memory: an idle connection is one small struct. Its request buffer (one request of this store's
 * dimension) is allocated on first read and reused for every later request on it; responses are built in a
 * per-thread buffer, so a request allocates nothing (a client that stops reading gets its unsent tail
 * buffered until it catches up).
 *
 * No authentication and no TLS: keep the Unix socket's 0600 mode, or bind TCP to localhost (the default) and
 * put a TLS proxy in front when it has to face a network.
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -O2 -march=native -I. vserv.c -lm -lpthread -o vserv
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

#define VS_MAGIC     0xB5
#define VS_MAXK      1000
#define VS_EVENTS    128
#define VS_IDLE_S    60.0
#define VS_OUT_CAP   (64 + VS_MAXK * 48)          /* the largest reply: JSON for VS_MAXK hits */

typedef struct VsConn {
    i32 fd, len, cap, out_len, out_off, closing;
    char *buf;                                  /* request bytes, allocated on first read */
    char *out;                                  /* unsent reply tail, only when the socket was full */
    double last;
    struct VsConn *prev, *next;                 /* the worker's list, for the idle sweep */
} VsConn;

typedef struct { i32 fd; } VsListen;            /* epoll data for a listening socket */

typedef struct {
    i32 ep;
    pthread_t th;
    VsConn *conns;
    char *out;                                  /* VS_OUT_CAP reply scratch */
    f32 *vec;                                   /* dim floats: an aligned copy of the query */
    i64 *ids;
    f32 *sc;
} VsWorker;

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

/* send what we can; keep the rest on the connection and wait for EPOLLOUT */
static void vs_send_(VsWorker *w, VsConn *c, const char *p, i32 n)
{
    while (n > 0) {
        ssize_t k = send(c->fd, p, (size_t)n, MSG_NOSIGNAL);
        if (k > 0) { p += k; n -= (i32)k; continue; }
        if (k < 0 && errno == EINTR) continue;
        if (k < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
        c->closing = 1;
        return;
    }
    if (n > 0) {
        char *o = (char *)malloc((size_t)c->out_len - (size_t)c->out_off + (size_t)n);
        if (!o) { c->closing = 1; return; }
        memcpy(o, c->out + c->out_off, (size_t)(c->out_len - c->out_off));
        memcpy(o + (c->out_len - c->out_off), p, (size_t)n);
        free(c->out);
        c->out_len = c->out_len - c->out_off + n; c->out_off = 0; c->out = o;
        vs_ep_mod_(w, c, EPOLLIN | EPOLLOUT | EPOLLRDHUP | EPOLLET);
    }
}

static void vs_flush_(VsWorker *w, VsConn *c)
{
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
    epoll_ctl(w->ep, EPOLL_CTL_DEL, c->fd, (struct epoll_event *)0);
    close(c->fd);
    if (c->prev) c->prev->next = c->next; else w->conns = c->next;
    if (c->next) c->next->prev = c->prev;
    free(c->buf); free(c->out); free(c);
}

static void vs_readable_(VsWorker *w, VsConn *c)
{
    for (;;) {
        ssize_t k;
        if (!c->buf) {
            c->buf = (char *)malloc((size_t)vs_reqcap);
            if (!c->buf) { c->closing = 1; return; }
            c->cap = vs_reqcap;
        }
        if (c->len == c->cap) { c->closing = 1; return; }  /* the parsers below always make room or close */
        k = read(c->fd, c->buf + c->len, (size_t)(c->cap - c->len));
        if (k > 0) {
            c->len += (i32)k;
            c->last = vs_now_();
            while (c->len > 0 && !c->closing && !c->out) {
                i32 used = (u8)c->buf[0] == VS_MAGIC ? vs_binary_(w, c) : vs_http_(w, c);
                if (used < 0) { c->closing = 1; break; }
                if (used == 0) break;
                c->len -= used;
                if (c->len) memmove(c->buf, c->buf + used, (size_t)c->len);
            }
            if (c->closing || c->out) return;
            continue;
        }
        if (k == 0) { c->closing = 1; return; }
        if (errno == EINTR) continue;
        if (errno != EAGAIN && errno != EWOULDBLOCK) c->closing = 1;
        return;
    }
}

static void vs_accept_(VsWorker *w, i32 lfd)
{
    for (;;) {
        struct epoll_event e;
        VsConn *c;
        i32 fd = accept4(lfd, (struct sockaddr *)0, (socklen_t *)0, SOCK_NONBLOCK | SOCK_CLOEXEC), one = 1;
        if (fd < 0) return;                      /* EAGAIN: another worker took it, or none left */
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));   /* fails harmlessly on a Unix socket */
        c = (VsConn *)calloc(1, sizeof(VsConn));
        if (!c) { close(fd); continue; }
        c->fd = fd; c->last = vs_now_();
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
            if (p == (void *)&vs_ls[0] || p == (void *)&vs_ls[1]) { vs_accept_(w, ((VsListen *)p)->fd); continue; }
            c = (VsConn *)p;
            if (ev[i].events & (EPOLLERR | EPOLLHUP)) c->closing = 1;
            if (!c->closing && (ev[i].events & EPOLLOUT)) {
                vs_flush_(w, c);
                if (!c->out && !c->closing) vs_readable_(w, c);   /* requests that waited behind the backlog */
            }
            if (!c->closing && (ev[i].events & (EPOLLIN | EPOLLRDHUP))) vs_readable_(w, c);
            if (c->closing && !c->out) vs_close_(w, c);
            else if (c->closing) { vs_flush_(w, c); if (!c->out) vs_close_(w, c); }
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

int main(int argc, char **argv)
{
    static Arena arena;
    const char *sock = (const char *)0, *bind_addr = "127.0.0.1";
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
        else if (argv[i][0] != '-' && !vs_path) vs_path = argv[i];
        else { fprintf(stderr, "usage: vserv [-s SOCK] [-p PORT] [-b ADDR] [-t THREADS] [-d DIM] [-x] STORE.vxi\n"); return 2; }
    }
    if (!vs_path) { fprintf(stderr, "vserv: no store file given\n"); return 2; }
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
    if (port) { vs_ls[vs_nls].fd = vs_listen_tcp_(bind_addr, port); if (vs_ls[vs_nls].fd < 0) { fprintf(stderr, "vserv: cannot listen on %s:%d: %s\n", bind_addr, port, strerror(errno)); return 1; } ++vs_nls; }

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
        for (j = 0; j < vs_nls; ++j) {               /* every worker waits on every listener; the kernel wakes one */
            struct epoll_event e;
            e.events = EPOLLIN | EPOLLEXCLUSIVE; e.data.ptr = &vs_ls[j];
            epoll_ctl(w->ep, EPOLL_CTL_ADD, vs_ls[j].fd, &e);
        }
    }
    fprintf(stderr, "vserv: %s: %d rows x %d, loaded in %.1f ms; %d threads;%s%s%s%s\n", vs_path, store_count(vs_store), vs_dim,
            (vs_now_() - t0) * 1e3, nth, sock ? " unix " : "", sock ? sock : "", port ? " tcp " : "", port ? bind_addr : "");
    for (i = 0; i < nth; ++i) pthread_create(&ws[i].th, (const pthread_attr_t *)0, vs_worker_, &ws[i]);
    for (i = 0; i < nth; ++i) pthread_join(ws[i].th, (void **)0);

    for (j = 0; j < vs_nls; ++j) close(vs_ls[j].fd);
    if (sock) unlink(sock);
    if (save_on_exit) {
        if (store_save(vs_store, vs_path) == 0) fprintf(stderr, "vserv: saved %s\n", vs_path);
        else fprintf(stderr, "vserv: saving %s failed\n", vs_path);
    }
    for (i = 0; i < nth; ++i) { close(ws[i].ep); free(ws[i].out); free(ws[i].vec); free(ws[i].ids); free(ws[i].sc); }
    free(ws);
    return 0;
}
