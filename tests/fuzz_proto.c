/* fuzz_proto.c - libFuzzer target for vserv's request parsers: binary frames, the line protocol and HTTP/1.1
 * (vserv_proto.c), with a real store behind them, under ASan and UBSan.
 *
 * One input is everything one client sends on one connection, against a fresh store of dimension 4 that no other
 * input saw; the log is on (to /dev/null), so its formatting runs too. Beyond memory errors and undefined behaviour
 * the target checks what came back: the reply stream has to split into whole replies - binary frames (0xB5, status,
 * count, a 32-bit length and that many bytes), lines that start "ok" or "err" and end in \n, HTTP/1.1 replies whose
 * body is exactly Content-Length bytes - with nothing left over. A reply that breaks that would desynchronise a
 * client that pipelines. A snapshot's fork() fails here (as under memory pressure), so the store's refusal path runs
 * instead of a child process.
 *
 *   clang -std=gnu99 -g -O1 -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined -DVSERV_NO_TLS \
 *       -Isrc tests/fuzz_proto.c -lm -lpthread -o .build/fuzz_proto
 *   .build/fuzz_proto -dict=tests/fuzz/proto.dict -max_len=4096 CORPUS_DIR tests/fuzz/proto
 *
 * tools/fuzz builds and runs every target; ./run replays tests/fuzz/proto with gcc alone (-DFUZZ_MAIN, fuzz_main.h). */
#define main vserv_main_
#include "vserv.c"
#undef main

#define FZ_DIM 4

static Arena fz_arena;
static VsWorker fz_w;
static char fz_path[64];
static char *fz_rx;                               /* the replies of the current input */
static size_t fz_rxn, fz_rxcap;

/* A snapshot's child process: refused, as fork() is under memory pressure. */
pid_t fork(void)
{
    errno = EAGAIN;
    return -1;
}

static void fz_fail_(const char *what, size_t at)
{
    fprintf(stderr, "fuzz_proto: %s at byte %lu of %lu reply bytes\n", what, (unsigned long)at, (unsigned long)fz_rxn);
    abort();
}

/* everything the server sent so far, appended to fz_rx */
static void fz_drain_(int fd)
{
    for (;;) {
        ssize_t k;
        if (fz_rxcap - fz_rxn < 65536) {
            fz_rxcap = fz_rxcap ? fz_rxcap * 2 : 1 << 20;
            if (!(fz_rx = (char *)realloc(fz_rx, fz_rxcap))) abort();
        }
        k = read(fd, fz_rx + fz_rxn, fz_rxcap - fz_rxn);
        if (k > 0) { fz_rxn += (size_t)k; continue; }
        if (k < 0 && errno == EINTR) continue;
        return;
    }
}

static const char *fz_find_(const char *p, size_t n, const char *s)
{
    size_t m = strlen(s), i;
    for (i = 0; i + m <= n; ++i) if (!memcmp(p + i, s, m)) return p + i;
    return (const char *)0;
}

/* the reply stream splits into whole replies, nothing left over */
static void fz_check_(void)
{
    const char *p = fz_rx;
    size_t i = 0, n = fz_rxn;
    while (i < n) {
        if ((u8)p[i] == VS_MAGIC) {
            u32 nb;
            if (n - i < 8) fz_fail_("binary reply header cut short", i);
            nb = vs_le32_((const u8 *)p + i + 4);
            if (n - i - 8 < (size_t)nb) fz_fail_("binary reply shorter than its length field", i);
            i += 8 + (size_t)nb;
        } else if (n - i >= 9 && !memcmp(p + i, "HTTP/1.1 ", 9)) {
            const char *end = fz_find_(p + i, n - i, "\r\n\r\n"), *cl = fz_find_(p + i, n - i, "\r\nContent-Length: ");
            unsigned long blen;
            size_t body;
            if (!end) fz_fail_("HTTP reply without the end of its header", i);
            if (!cl || cl > end) fz_fail_("HTTP reply without Content-Length", i);
            blen = strtoul(cl + 18, (char **)0, 10);
            body = (size_t)(end + 4 - p);
            if (n - body < blen) fz_fail_("HTTP body shorter than Content-Length", i);
            i = body + blen;
        } else if ((n - i >= 2 && !memcmp(p + i, "ok", 2)) || (n - i >= 3 && !memcmp(p + i, "err", 3))) {
            const char *nl = (const char *)memchr(p + i, '\n', n - i);
            if (!nl) fz_fail_("line reply without its \\n", i);
            i = (size_t)(nl - p) + 1;
        } else fz_fail_("reply starts with neither 0xB5, HTTP/1.1 nor ok/err", i);
    }
}

int LLVMFuzzerInitialize(int *argc, char ***argv)
{
    char dir[] = "/tmp/fuzz_proto.XXXXXX";
    (void)argc; (void)argv;
    if (!mkdtemp(dir)) abort();
    snprintf(fz_path, sizeof(fz_path), "%s/f.vxi", dir);
    vs_dim = FZ_DIM;
    vs_reqcap = 2048 + vs_dim * 24;                /* as main() sets them */
    if (vs_reqcap < 4096) vs_reqcap = 4096;
    vs_reqmax = vs_reqcap + VS_REQ_TEXT;
    vs_log_fd = open("/dev/null", O_WRONLY | O_CLOEXEC);
    vs_log_all = 1;
    signal(SIGPIPE, SIG_IGN);
    fz_w.ep = epoll_create1(EPOLL_CLOEXEC);
    fz_w.out = (char *)malloc(VS_OUT_CAP);
    fz_w.vec = (f32 *)malloc((size_t)vs_dim * sizeof(f32));
    fz_w.ids = (i64 *)malloc(VS_MAXK * sizeof(i64));
    fz_w.sc = (f32 *)malloc(VS_MAXK * sizeof(f32));
    if (fz_w.ep < 0 || !fz_w.out || !fz_w.vec || !fz_w.ids || !fz_w.sc) abort();
    return 0;
}

int LLVMFuzzerTestOneInput(const unsigned char *d, size_t n)
{
    int sv[2];
    VsConn *c;
    if (n > (size_t)vs_reqmax) return 0;
    memset(&vs_one, 0, sizeof(vs_one));
    vs_one.path = fz_path;
    vs_one.s = store_new(&fz_arena, vs_dim);
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0, sv)) abort();
    c = (VsConn *)calloc(1, sizeof(VsConn));
    if (!c) abort();
    c->fd = sv[0];
    c->t = &vs_one;
    c->cap = n > (size_t)vs_reqcap ? (i32)n : vs_reqcap;
    if (!(c->buf = (char *)malloc((size_t)c->cap))) abort();
    if (n) memcpy(c->buf, d, n);
    c->len = (i32)n;
    fz_w.conns = c;
    fz_rxn = 0;
    for (;;) {                                     /* answer, let the replies out, until nothing moves */
        i32 before = c->len;
        vs_process_(&fz_w, c);
        fz_drain_(sv[1]);
        while (c->out && !c->closing) { vs_flush_(&fz_w, c); fz_drain_(sv[1]); }
        if (c->closing || c->len == 0 || c->len == before) break;
    }
    vs_close_(&fz_w, c);
    fz_drain_(sv[1]);
    close(sv[1]);
    fz_check_();
    store_destroy(vs_one.s);
    arena_free(&fz_arena);
    return 0;
}

#include "fuzz_main.h"
