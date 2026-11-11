/* vserv_repl.c - a replica on a second machine that follows a primary (vserv -F), included by vserv.c.
 *
 * What it copies is what store_load reads: STORE.vxi and its journal STORE.vxi.j. The journal is copied byte for byte
 * up to what the primary's last fsync covered (store_jdur_), which always ends on a whole record: bytes the primary
 * may still lose in a crash never reach the replica. When the primary's file changes (a save or a snapshot gives it
 * a new stamp), the replica fetches the new file in 4 MB chunks to STORE.vxi.part, checks every block of it
 * (store_scrub), renames it into place and starts the journal over. So the replica is always a file and a journal
 * that match: `vserv STORE.vxi` on it starts a server with everything the primary had flushed - that is the
 * failover, by hand or by the init system.
 *
 * The primary's side, binary ops (operator only: the Unix socket, or the * key of -K):
 *   op 21 journal   payload: u64 stamp, u64 offset   reply: the journal's bytes from offset (at most 4 MB; none when
 *                   caught up). status 8 with a u64 payload, the stamp of the file now: the stamp no longer matches
 *                   (or offset is past what the primary has) - fetch that file
 *   op 22 file      payload: u64 stamp, u64 offset   reply: the file's bytes from offset (at most 4 MB), count 1 on
 *                   the last chunk. status 8 with the file's stamp: it changed meanwhile - start over
 *   op 23 tenants   reply: the tenants' names, a line each (a tenant server: every NAME.vxi in its directory) */

#include <dirent.h>
#include <netdb.h>

#define VS_RCHUNK  ((u32)4 << 20)                  /* bytes a journal or file reply carries at most */
#define VS_JHEAD   20                              /* a VXJ3 journal's header: magic, dim, stamp, CRC-32C */

static void vs_put64u_(char *o, u64 v) { vs_put32_(o, (u32)(v & 0xffffffffu)); vs_put32_(o + 4, (u32)(v >> 32)); }
static u64 vs_le64u_(const u8 *p) { return (u64)vs_le32_(p) | ((u64)vs_le32_(p + 4) << 32); }

static i32 vs_pread_all_(int fd, char *p, u32 n, u64 at)
{
    while (n) {
        ssize_t k = pread(fd, p, (size_t)n, (off_t)at);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return -1;
        p += k; n -= (u32)k; at += (u64)k;
    }
    return 0;
}

/* op 21 on the primary: status 0 with *n bytes in *buf (malloc'd), 8 (stamp: *cur), or 1 (no journal to follow) */
static i32 vs_repl_journal_(RagStore *s, u64 stamp, u64 off, char **buf, u32 *n, u64 *cur)
{
    char pf[48];
    u8 h[VS_JHEAD];
    int fd = -1;
    u64 st;
    i64 dur;
    *buf = (char *)0; *n = 0;
    pthread_rwlock_rdlock(&s->lock);                             /* the stamp, the journal and its durable size agree */
    st = s->stamp; dur = s->jdur;
    if (st && s->jf) { sprintf(pf, "/proc/self/fd/%d", fileno(s->jf)); fd = open(pf, O_RDONLY); }   /* whatever its name */
    pthread_rwlock_unlock(&s->lock);
    *cur = st;
    if (fd < 0) return 1;
    if (stamp != st || off > (u64)dur || vs_pread_all_(fd, (char *)h, VS_JHEAD, 0) != 0
        || vs_le32_(h) != STORE_JMAGIC3 || vs_le64u_(h + 8) != st) { close(fd); return 8; }
    *n = (u64)dur - off > VS_RCHUNK ? VS_RCHUNK : (u32)((u64)dur - off);
    if (*n && (!(*buf = (char *)malloc(*n)) || vs_pread_all_(fd, *buf, *n, off) != 0)) { free(*buf); *buf = (char *)0; *n = 0; close(fd); return 1; }
    close(fd);
    return 0;
}

/* op 22 on the primary: status 0 (*last: the chunk ends the file), 8 (another file now: its stamp in *cur), 1, 2 */
static i32 vs_repl_file_(const char *path, u64 stamp, u64 off, char **buf, u32 *n, i32 *last, u64 *cur)
{
    int fd = open(path, O_RDONLY);
    u8 h[32];
    struct stat sb;
    *buf = (char *)0; *n = 0; *last = 0; *cur = 0;
    if (fd < 0) return 1;
    if (vs_pread_all_(fd, (char *)h, 32, 0) != 0 || vs_le32_(h) != STORE_MAGIC || fstat(fd, &sb) != 0) { close(fd); return 1; }
    *cur = vs_le64u_(h + 24);
    if (*cur != stamp) { close(fd); return 8; }
    if (off > (u64)sb.st_size) { close(fd); return 2; }
    *n = (u64)sb.st_size - off > VS_RCHUNK ? VS_RCHUNK : (u32)((u64)sb.st_size - off);
    if (*n && (!(*buf = (char *)malloc(*n)) || vs_pread_all_(fd, *buf, *n, off) != 0)) { free(*buf); *buf = (char *)0; *n = 0; close(fd); return 1; }
    *last = off + *n == (u64)sb.st_size;
    close(fd);
    return 0;
}

/* op 23 on the primary: the tenants' names, a line each, in *buf (malloc'd); -1 not a tenant server */
static i32 vs_repl_tenants_(char **buf, u32 *n)
{
    DIR *d;
    struct dirent *e;
    u32 cap = 4096;
    *n = 0;
    if (!vs_dir || !(d = opendir(vs_dir))) return -1;
    if (!(*buf = (char *)malloc(cap))) { closedir(d); return -1; }
    while ((e = readdir(d)) != (struct dirent *)0) {
        size_t l = strlen(e->d_name), i;
        if (l < 5 || l - 4 > VS_TNAME || strcmp(e->d_name + l - 4, ".vxi") != 0) continue;
        for (i = 0; i < l - 4 && vs_tname_ok_(e->d_name[i]); ++i) ;
        if (i < l - 4) continue;
        if (*n + l > cap) { char *b = (char *)realloc(*buf, cap *= 2); if (!b) break; *buf = b; }
        memcpy(*buf + *n, e->d_name, l - 4); *n += (u32)(l - 4);
        (*buf)[(*n)++] = '\n';
    }
    closedir(d);
    return 0;
}

/* ---- the replica's side: vserv -F ADDR REPLICA ---- */

static i32 vs_sock_io_(int fd, char *p, size_t n, i32 wr)
{
    while (n) {
        ssize_t k = wr ? send(fd, p, n, MSG_NOSIGNAL) : recv(fd, p, n, 0);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return -1;
        p += k; n -= (size_t)k;
    }
    return 0;
}

#ifdef VSERV_TLS
/* The replica's TLS: the primary's certificate is not checked against a CA but against the key the replica saw the
 * first time (or $VSERV_PIN): the SHA-256 of its SubjectPublicKeyInfo, as vserv prints it at start. */
typedef struct {
    const br_x509_class *vt;
    br_x509_decoder_context dc;
    unsigned char cert[16384];
    size_t n;
    i32 depth, over;
    const char *want;                            /* the pinned key, or empty: remember the one shown */
    char got[64];
} VsPinX;

static void vs_px_start_chain_(const br_x509_class **ctx, const char *name)
{
    VsPinX *x = (VsPinX *)(void *)ctx;
    (void)name;
    x->depth = 0; x->n = 0; x->over = 0; x->got[0] = 0;
    br_x509_decoder_init(&x->dc, (void (*)(void *, const void *, size_t))0, (void *)0);
}
static void vs_px_start_cert_(const br_x509_class **ctx, uint32_t len) { (void)ctx; (void)len; }
static void vs_px_append_(const br_x509_class **ctx, const unsigned char *b, size_t n)
{
    VsPinX *x = (VsPinX *)(void *)ctx;
    if (x->depth) return;                        /* the server's own certificate is the first; the rest is its chain */
    br_x509_decoder_push(&x->dc, b, n);
    if (x->n + n > sizeof(x->cert)) { x->over = 1; return; }
    memcpy(x->cert + x->n, b, n); x->n += n;
}
static void vs_px_end_cert_(const br_x509_class **ctx) { ((VsPinX *)(void *)ctx)->depth++; }
static unsigned vs_px_end_chain_(const br_x509_class **ctx)
{
    VsPinX *x = (VsPinX *)(void *)ctx;
    if (x->over || !br_x509_decoder_get_pkey(&x->dc) || vs_pin_of_(x->cert, x->n, x->got) != 0) return BR_ERR_X509_BAD_SERVER_NAME;
    return !*x->want || !strcmp(x->want, x->got) ? 0 : BR_ERR_X509_NOT_TRUSTED;
}
static const br_x509_pkey *vs_px_pkey_(const br_x509_class *const *ctx, unsigned *usages)
{
    VsPinX *x = (VsPinX *)(void *)ctx;
    if (usages) *usages = BR_KEYTYPE_KEYX | BR_KEYTYPE_SIGN;
    return br_x509_decoder_get_pkey(&x->dc);
}
static const br_x509_class vs_px_class = {
    sizeof(VsPinX), vs_px_start_chain_, vs_px_start_cert_, vs_px_append_, vs_px_end_cert_, vs_px_end_chain_, vs_px_pkey_
};

typedef struct {
    br_ssl_client_context cc;
    br_x509_minimal_context xm;                  /* set up by init_full, replaced by px */
    VsPinX px;
    br_sslio_context io;
    unsigned char buf[BR_SSL_BUFSIZE_BIDI];
    int fd;
} VsRtls;

static int vs_rtls_rd_(void *c, unsigned char *b, size_t n)
{
    ssize_t k;
    do k = recv(*(int *)c, b, n, 0); while (k < 0 && errno == EINTR);
    return k <= 0 ? -1 : (int)k;
}
static int vs_rtls_wr_(void *c, const unsigned char *b, size_t n)
{
    ssize_t k;
    do k = send(*(int *)c, b, n, MSG_NOSIGNAL); while (k < 0 && errno == EINTR);
    return k <= 0 ? -1 : (int)k;
}
#endif

typedef struct { int fd; char *buf; u32 cap, nb; u8 status; u16 count; void *tls; } VsRc;

static i32 vs_rio_(VsRc *r, char *p, size_t n, i32 wr)
{
#ifdef VSERV_TLS
    if (r->tls) {
        VsRtls *t = (VsRtls *)r->tls;
        if (!n) return 0;
        return wr ? (br_sslio_write_all(&t->io, p, n) == 0 ? 0 : -1) : (br_sslio_read_all(&t->io, p, n) == 0 ? 0 : -1);
    }
#endif
    return vs_sock_io_(r->fd, p, n, wr);
}

static void vs_rclose_(VsRc *r)
{
#ifdef VSERV_TLS
    if (r->tls) { explicit_bzero(r->tls, sizeof(VsRtls)); free(r->tls); r->tls = (void *)0; }
#endif
    if (r->fd >= 0) close(r->fd);
    r->fd = -1;
}

/* HOST:PORT (TCP, plaintext) or the path of a Unix socket; -1 when it cannot connect */
static int vs_rconnect_(const char *addr)
{
    const char *colon = strrchr(addr, ':');
    struct timeval tv;
    int fd = -1;
    tv.tv_sec = 60; tv.tv_usec = 0;
    if (strchr(addr, '/') || !colon) {
        struct sockaddr_un un;
        if (strlen(addr) >= sizeof(un.sun_path)) return -1;
        memset(&un, 0, sizeof(un));
        un.sun_family = AF_UNIX;
        strcpy(un.sun_path, addr);
        if ((fd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) return -1;
        if (connect(fd, (struct sockaddr *)&un, sizeof(un)) != 0) { close(fd); return -1; }
    } else {
        struct addrinfo hint, *ai, *a;
        char host[256];
        size_t hl = (size_t)(colon - addr);
        if (hl >= sizeof(host)) return -1;
        memcpy(host, addr, hl); host[hl] = 0;
        if (hl > 1 && host[0] == '[' && host[hl - 1] == ']') { memmove(host, host + 1, hl - 2); host[hl - 2] = 0; }
        memset(&hint, 0, sizeof(hint));
        hint.ai_family = AF_UNSPEC; hint.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(host, colon + 1, &hint, &ai) != 0) return -1;
        for (a = ai; a; a = a->ai_next) {
            if ((fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol)) < 0) continue;
            if (connect(fd, a->ai_addr, a->ai_addrlen) == 0) break;
            close(fd); fd = -1;
        }
        freeaddrinfo(ai);
        if (fd < 0) return -1;
    }
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    return fd;
}

/* one request, its reply in r: 0, or -1 when the connection broke (or the reply is not one) */
static i32 vs_rcall_(VsRc *r, u8 op, const char *pl, u32 n)
{
    char h[8];
    h[0] = (char)VS_MAGIC; h[1] = (char)op; h[2] = h[3] = 0; vs_put32_(h + 4, n);
    if (vs_rio_(r, h, 8, 1) != 0 || (n && vs_rio_(r, (char *)pl, n, 1) != 0)) return -1;
#ifdef VSERV_TLS
    if (r->tls && br_sslio_flush(&((VsRtls *)r->tls)->io) != 0) return -1;
#endif
    if (vs_rio_(r, h, 8, 0) != 0) return -1;
    if ((u8)h[0] != VS_MAGIC) return -1;
    r->status = (u8)h[1]; r->count = (u16)((u8)h[2] | ((u8)h[3] << 8)); r->nb = vs_le32_((const u8 *)h + 4);
    if (r->nb > VS_RCHUNK + 4096) return -1;
    if (r->nb > r->cap) {
        char *b = (char *)realloc(r->buf, r->nb);
        if (!b) return -1;
        r->buf = b; r->cap = r->nb;
    }
    return r->nb && vs_rio_(r, r->buf, r->nb, 0) != 0 ? -1 : 0;
}

/* the replica's journal: the size of its good part, cut back to it (a torn tail from a crash of this process: those
 * bytes are fetched again); 0 when it is missing, damaged in its header, or for another file than the one beside it */
static u64 vs_rjtrim_(const char *path, const char *jn)
{
    FILE *f = fopen(jn, "rb");
    u8 h[VS_JHEAD], fh[9];
    u64 good = 0, fst = store_file_stamp_(path);
    u8 *g = (u8 *)0;
    size_t gcap = 0;
    if (!f) return 0;
    if (fread(h, 1, VS_JHEAD, f) == VS_JHEAD && vs_le32_(h) == STORE_JMAGIC3 && store_crc32c_(h + 4, 12) == vs_le32_(h + 16)
        && fst && vs_le64u_(h + 8) == fst) {
        good = VS_JHEAD;
        while (fread(fh, 1, 9, f) == 9 && fh[0] == 'G') {
            u32 n = vs_le32_(fh + 1);
            if ((size_t)n > STORE_BATCH_MAX) break;
            if (n > gcap) { u8 *t = (u8 *)realloc(g, n); if (!t) break; g = t; gcap = n; }
            if (n && fread(g, 1, n, f) != n) break;
            if (store_crc32c_(g, n) != vs_le32_(fh + 5)) break;
            good += 9 + (u64)n;
        }
    }
    fclose(f);
    free(g);
    if (good && truncate(jn, (off_t)good) != 0) good = 0;
    return good;
}

static i32 vs_rappend_(const char *jn, u64 off, const char *p, u32 n)
{
    FILE *f = fopen(jn, off ? "r+b" : "wb");
    i32 r;
    if (!f) return -1;
    r = fseeko(f, (off_t)off, SEEK_SET) == 0 && fwrite(p, 1, n, f) == n && store_fsync_(f) == 0 ? 0 : -1;
    if (fclose(f) != 0) r = -1;
    return r;
}

/* the primary's file with this stamp into path, through path.part, checked block by block: 0 done, 1 it changed
 * meanwhile (start over), -1 broken connection, -2 refused or not written */
static i32 vs_rfile_(VsRc *r, const char *path, u64 stamp)
{
    char pl[16], *part = store_jname_(path, ".part");
    FILE *f = fopen(part, "wb");
    u64 at = 0;
    i64 fixed = 0, bad;
    i32 rc = -2;
    if (!f) { free(part); return -2; }
    for (;;) {
        vs_put64u_(pl, stamp); vs_put64u_(pl + 8, at);
        if (vs_rcall_(r, 22, pl, 16) != 0) { rc = -1; break; }
        if (r->status == 8) { rc = 1; break; }
        if (r->status != 0 || (r->nb && fwrite(r->buf, 1, r->nb, f) != r->nb)) break;
        at += r->nb;
        if (r->count == 1) { rc = 0; break; }
        if (!r->nb) break;
    }
    if (store_fsync_(f) != 0) rc = rc == 0 ? -2 : rc;
    fclose(f);
    if (rc == 0) {
        bad = store_scrub(part, 1, &fixed);
        if (bad == -1 || bad == -3 || (bad > 0 && fixed < bad)) {
            fprintf(stderr, "vserv: %s: the copy of the primary's file fails its checksums (%ld blocks); fetching it again\n", path, (long)bad);
            rc = 1;
        } else if (rename(part, path) != 0) rc = -2;
        else store_fsync_dir_(path);
    }
    free(part);
    return rc;
}

/* brings the replica at path up to the store the connection points at: 0 caught up, 1 more to fetch, -1 the
 * connection broke, -2 refused (the reason on stderr). *off is the replica journal's size, kept between calls */
static i32 vs_rsync_(VsRc *r, const char *path, u64 *off)
{
    char pl[16], *jn = store_jname_(path, ".j");
    u64 fst = store_file_stamp_(path);
    i32 rc;
    vs_put64u_(pl, fst); vs_put64u_(pl + 8, fst ? *off : 0);
    if (vs_rcall_(r, 21, pl, 16) != 0) { free(jn); return -1; }
    if (r->status == 0) {
        if (!fst) rc = 0;
        else if (!r->nb) rc = 0;
        else if (vs_rappend_(jn, *off, r->buf, r->nb) != 0) { fprintf(stderr, "vserv: %s: %s\n", jn, strerror(errno)); rc = -2; }
        else { *off += r->nb; rc = 1; }
    } else if (r->status == 8 && r->nb == 8) {
        u64 st = vs_le64u_((const u8 *)r->buf);
        rc = vs_rfile_(r, path, st);
        if (rc == 0) {
            *off = 0;
            fprintf(stderr, "vserv: %s: the primary's file %016lx copied\n", path, (unsigned long)st);
            vs_log_((const VsConn *)0, 0, "replica file=%s stamp=%016lx", path, (unsigned long)st);
            rc = 1;                                              /* now its journal, from the start */
        }
    } else {
        fprintf(stderr, "vserv: %s: the primary refused (status %d%s%.*s)\n", path, (int)r->status, r->nb ? ": " : "",
                (int)(r->nb > 200 ? 200 : r->nb), r->nb ? r->buf : "");
        rc = -2;
    }
    free(jn);
    return rc;
}

typedef struct { char name[VS_TNAME + 1]; u64 off; i32 seen; } VsRst;

/* the session on a fresh connection: TLS for a TCP address (unless -P), the primary's key checked against the one
 * pinned in vserv.pin beside the replica (or $VSERV_PIN) and pinned there on the first connection, then the key of
 * $VSERV_KEY. 0; -1 the connection closed; -3 TLS failed; -2 give up (another key, the key refused) */
static i32 vs_ropen_(VsRc *r, const char *addr, const char *rep, i32 dir, const char *key)
{
    i32 tcp = !strchr(addr, '/') && strrchr(addr, ':') != (char *)0, ok;
#ifdef VSERV_TLS
    char pinf[4096], want[64];
    VsRtls *t = (VsRtls *)0;
    want[0] = 0;
    if (tcp && !vs_plain_ok) {
        const char *env = getenv("VSERV_PIN"), *sl = dir ? (const char *)0 : strrchr(rep, '/');
        size_t dl = dir ? strlen(rep) : (sl ? (size_t)(sl - rep) : 0);
        FILE *f;
        if (dl + 16 > sizeof(pinf)) return -2;
        if (dl) memcpy(pinf, rep, dl); else pinf[dl++] = '.';
        strcpy(pinf + dl, "/vserv.pin");
        if (env && *env) { strncpy(want, env, sizeof(want) - 1); want[sizeof(want) - 1] = 0; }
        else if ((f = fopen(pinf, "r")) != (FILE *)0) {
            if (!fgets(want, sizeof(want), f)) want[0] = 0;
            fclose(f);
            want[strcspn(want, " \t\r\n")] = 0;
        }
        if (!(t = (VsRtls *)calloc(1, sizeof(VsRtls)))) return -2;
        t->fd = r->fd;
        br_ssl_client_init_full(&t->cc, &t->xm, (const br_x509_trust_anchor *)0, 0);
        t->px.vt = &vs_px_class; t->px.want = want;
        br_ssl_engine_set_x509(&t->cc.eng, &t->px.vt);
        br_ssl_engine_set_buffer(&t->cc.eng, t->buf, sizeof(t->buf), 1);
        br_ssl_client_reset(&t->cc, (const char *)0, 0);
        br_sslio_init(&t->io, &t->cc.eng, vs_rtls_rd_, &t->fd, vs_rtls_wr_, &t->fd);
        r->tls = t;
    }
#else
    (void)rep; (void)dir; (void)tcp;
#endif
    if (key && *key) ok = vs_rcall_(r, 20, key, (u32)strlen(key)) == 0;
    else ok = vs_rcall_(r, 6, (const char *)0, 0) == 0;                 /* stats: the first exchange, the handshake */
#ifdef VSERV_TLS
    if (t && !ok) {
        if (br_ssl_engine_last_error(&t->cc.eng) == BR_ERR_X509_NOT_TRUSTED) {
            fprintf(stderr, "vserv: %s shows another TLS key: %s, pinned %s (%s). If the primary's key really changed, "
                            "move %s away; else someone is in between\n", addr, t->px.got, want, getenv("VSERV_PIN") && *getenv("VSERV_PIN") ? "$VSERV_PIN" : pinf, pinf);
            vs_log_((const VsConn *)0, 1, "replica tls key=%s pinned=%s refused", t->px.got, want);
            return -2;
        }
        return -3;
    }
    if (t && !*want && t->px.got[0]) {                           /* the first connection: remember the key */
        FILE *f = fopen(pinf, "w");
        if (!f || fprintf(f, "%s\n", t->px.got) < 0 || store_fsync_(f) != 0) {
            fprintf(stderr, "vserv: cannot write %s: %s\n", pinf, strerror(errno));
            if (f) fclose(f);
            return -2;
        }
        fclose(f);
        fprintf(stderr, "vserv: %s: TLS key %s, pinned in %s - check it against the one the primary printed\n", addr, t->px.got, pinf);
        vs_log_((const VsConn *)0, 0, "replica tls key=%s pinned", t->px.got);
    }
#endif
    if (!ok) return -1;
    if (key && *key && r->status != 0) { fprintf(stderr, "vserv: %s refused the key in VSERV_KEY\n", addr); return -2; }
    return 0;
}

/* vserv -F ADDR REPLICA: follows the primary at ADDR until SIGINT / SIGTERM, every `every` seconds. REPLICA is a
 * STORE.vxi for a one-store primary, or a directory for a tenant server (every tenant, NAME.vxi in it). $VSERV_KEY
 * is shown first when set (the operator's key of a primary with -K). Returns the exit code. */
static i32 vs_follow_(const char *addr, const char *rep, f64 every)
{
    struct stat sb;
    i32 dir = stat(rep, &sb) == 0 && S_ISDIR(sb.st_mode), nst = 0, cst = 0, i, rc, said = 0;
    VsRst *st = (VsRst *)0;
    VsRc r;
    const char *key = getenv("VSERV_KEY");
    memset(&r, 0, sizeof(r));
    r.fd = -1;
    if (every <= 0) every = 1;
    signal(SIGINT, vs_on_signal_); signal(SIGTERM, vs_on_signal_); signal(SIGPIPE, SIG_IGN);
    fprintf(stderr, "vserv: %s follows %s (%s)\n", rep, addr, dir ? "every tenant" : "one store");
    while (!vs_stop) {
        i32 more = 0, broke = 0;
        if (r.fd < 0) {
            if ((r.fd = vs_rconnect_(addr)) < 0) {
                if (!said) { fprintf(stderr, "vserv: cannot reach %s: %s; trying every %g s\n", addr, strerror(errno), every); said = 1; }
                goto wait;
            }
            rc = vs_ropen_(&r, addr, rep, dir, key);
            if (rc == -2) { free(st); free(r.buf); vs_rclose_(&r); return 1; }
            if (rc != 0) {
                if (!said) { fprintf(stderr, "vserv: %s: no session (%s); trying every %g s\n", addr, rc == -3 ? "TLS handshake failed" : "closed", every); said = 1; }
                vs_rclose_(&r);
                goto wait;
            }
            if (said) fprintf(stderr, "vserv: %s reached\n", addr);
            said = 0;
        }
        if (!dir) {
            if (!nst) {
                char *jn = store_jname_(rep, ".j");
                if (!(st = (VsRst *)calloc(1, sizeof(VsRst)))) return 1;
                nst = 1; st[0].off = vs_rjtrim_(rep, jn);
                free(jn);
            }
            rc = vs_rsync_(&r, rep, &st[0].off);
            if (rc == -1) broke = 1; else if (rc == -2) { free(st); free(r.buf); vs_rclose_(&r); return 1; } else more = rc;
        } else {
            char *names, *p, *e;
            if (vs_rcall_(&r, 23, (const char *)0, 0) != 0) { broke = 1; goto next; }
            if (r.status != 0) { fprintf(stderr, "vserv: %s is not a tenant server; give a STORE.vxi to follow it\n", addr); free(st); free(r.buf); vs_rclose_(&r); return 1; }
            if (!(names = (char *)malloc((size_t)r.nb + 1))) return 1;
            memcpy(names, r.buf, r.nb); names[r.nb] = 0;
            for (p = names; *p && !vs_stop && !broke; p = e + 1) {
                char *path;
                size_t l;
                if (!(e = strchr(p, '\n'))) break;
                l = (size_t)(e - p);
                if (l < 1 || l > VS_TNAME) continue;
                for (i = 0; i < nst; ++i) if (strlen(st[i].name) == l && !memcmp(st[i].name, p, l)) break;
                if (!(path = (char *)malloc(strlen(rep) + l + 6))) break;
                sprintf(path, "%s/%.*s.vxi", rep, (int)l, p);
                if (i == nst) {                                  /* first seen: its journal cut back to the good part */
                    char *jn = store_jname_(path, ".j");
                    if (nst == cst) { VsRst *t = (VsRst *)realloc(st, sizeof(VsRst) * (size_t)(cst = cst ? cst * 2 : 16)); if (!t) { free(path); break; } st = t; }
                    memset(&st[nst], 0, sizeof(VsRst));
                    memcpy(st[nst].name, p, l);
                    st[nst].off = vs_rjtrim_(path, jn);
                    ++nst;
                    free(jn);
                }
                if (vs_rcall_(&r, 14, p, (u32)l) != 0) broke = 1;
                else if (r.status == 0) {
                    rc = vs_rsync_(&r, path, &st[i].off);
                    if (rc == -1) broke = 1; else if (rc == 1) more = 1;
                }
                free(path);
            }
            free(names);
        }
    next:
        if (broke) { fprintf(stderr, "vserv: lost %s; reconnecting\n", addr); vs_rclose_(&r); }
        if (more && !broke) continue;                            /* behind: no pause until caught up */
    wait:
        {   f64 t = vs_now_() + every;
            while (!vs_stop && vs_now_() < t) { struct timespec ts; ts.tv_sec = 0; ts.tv_nsec = 50000000L; nanosleep(&ts, (struct timespec *)0); } }
    }
    vs_rclose_(&r);
    free(st); free(r.buf);
    return 0;
}
