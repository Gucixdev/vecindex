/* vserv_test.c - the server's line protocol, its log, its journal and its background snapshots, end to end.
 *   gcc -std=gnu99 -O2 tests/vserv_test.c -o .build/vserv_line_test && .build/vserv_line_test .build/vserv
 * Starts its own server on a fresh store in a temporary directory, talks to it over the Unix socket and TCP, and
 * checks: every line command and its errors; the binary ops for snapshot, get and range deletes; HTTP /snapshot,
 * /get and range deletes; that an add replaces the row with the same id; the log names the peer of every write; a
 * kill -9 loses nothing older than the journal flush (-f); timed snapshots (-S) fold the journal into the file;
 * batches (begin, commit [sync], abort, a dropped connection, a kill -9 right after a synced commit); the stats fields
 * of the memory tiers and -m; a tenant server (use NAME, HTTP /t/NAME/, binary op 14, isolation, the log, a restart,
 * -U with a tenant's certificate). The
 * protocol and TLS checks that predate these live in py/vserv_test.py and py/vserv_tls_test.py. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <time.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define DIM 8

static int fails;
static char dir[64], sock_path[128], store_path[128], log_path[128];
static const char *serve_path = store_path;    /* what start() serves: the store, or a tenant directory */
static int port;

static void check(int cond, const char *what)
{
    printf(cond ? "  ok   %s\n" : "  FAIL %s\n", what);
    if (!cond) ++fails;
}

static void msleep(int ms) { struct timespec ts; ts.tv_sec = ms / 1000; ts.tv_nsec = (long)(ms % 1000) * 1000000L; nanosleep(&ts, NULL); }

static int connect_unix(void)
{
    struct sockaddr_un a;
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    memset(&a, 0, sizeof(a)); a.sun_family = AF_UNIX; strcpy(a.sun_path, sock_path);
    if (connect(fd, (struct sockaddr *)&a, sizeof(a)) != 0) { close(fd); return -1; }
    return fd;
}

static int connect_tcp(void)
{
    struct sockaddr_in a;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    memset(&a, 0, sizeof(a)); a.sin_family = AF_INET; a.sin_port = htons((unsigned short)port);
    inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    if (connect(fd, (struct sockaddr *)&a, sizeof(a)) != 0) { close(fd); return -1; }
    return fd;
}

static pid_t start(const char *extra1, const char *extra2, const char *extra3)
{
    char ports[16];
    pid_t p;
    int i;
    sprintf(ports, "%d", port);
    p = fork();
    if (p == 0) {
        const char *argv[20];
        int n = 0;
        argv[n++] = getenv("VSERV"); argv[n++] = "-s"; argv[n++] = sock_path; argv[n++] = "-p"; argv[n++] = ports;
        argv[n++] = "-t"; argv[n++] = "2"; argv[n++] = "-d"; argv[n++] = "8"; argv[n++] = "-l"; argv[n++] = log_path;
        if (extra1) argv[n++] = extra1;
        if (extra2) argv[n++] = extra2;
        if (extra3) argv[n++] = extra3;
        argv[n++] = serve_path; argv[n] = NULL;
        execv(argv[0], (char *const *)argv);
        _exit(127);
    }
    for (i = 0; i < 300; ++i) {
        int fd = connect_unix();
        if (fd >= 0) { close(fd); return p; }
        msleep(20);
    }
    return p;
}

static void stop(pid_t p, int sig) { int st; kill(p, sig); waitpid(p, &st, 0); }

/* send a line, read one line back (without the newline) into out */
static int ask(int fd, const char *req, char *out, int cap)
{
    int n = 0;
    if (write(fd, req, strlen(req)) != (ssize_t)strlen(req)) return -1;
    while (n < cap - 1) {
        ssize_t k = read(fd, out + n, 1);
        if (k <= 0) return -1;
        if (out[n] == '\n') break;
        ++n;
    }
    out[n] = 0;
    return n;
}

static void vline(char *buf, const char *cmd, long id, int seed)
{
    int d, n = sprintf(buf, "%s %ld", cmd, id);
    unsigned s = (unsigned)seed * 2654435761u + 7u;
    for (d = 0; d < DIM; ++d) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; n += sprintf(buf + n, " %.4f", (double)(s % 2001u) / 1000.0 - 1.0); }
    strcpy(buf + n, "\n");
}

static int bin_call(int fd, int op, const void *payload, unsigned nb, unsigned char *st, unsigned *rnb, unsigned char *rbuf)
{
    unsigned char h[8];
    h[0] = 0xB5; h[1] = (unsigned char)op; h[2] = h[3] = 0;
    h[4] = (unsigned char)(nb & 0xff); h[5] = (unsigned char)((nb >> 8) & 0xff); h[6] = (unsigned char)((nb >> 16) & 0xff); h[7] = (unsigned char)(nb >> 24);
    if (write(fd, h, 8) != 8 || (nb && write(fd, payload, nb) != (ssize_t)nb)) return -1;
    {   unsigned got = 0, need = 8;
        unsigned char r[8];
        while (got < need) { ssize_t k = read(fd, r + got, need - got); if (k <= 0) return -1; got += (unsigned)k; }
        *st = r[1]; *rnb = (unsigned)r[4] | ((unsigned)r[5] << 8) | ((unsigned)r[6] << 16) | ((unsigned)r[7] << 24);
        for (got = 0; got < *rnb; ) { ssize_t k = read(fd, rbuf + got, *rnb - got); if (k <= 0) return -1; got += (unsigned)k; }
    }
    return 0;
}

static int http(const char *req, char *out, int cap)
{
    int fd = connect_tcp(), n = 0;
    ssize_t k;
    if (fd < 0) return -1;
    if (write(fd, req, strlen(req)) < 0) { close(fd); return -1; }
    while (n < cap - 1 && (k = read(fd, out + n, (size_t)(cap - 1 - n))) > 0) n += (int)k;
    out[n] = 0;
    close(fd);
    return n;
}

static int log_has(const char *what)
{
    FILE *f = fopen(log_path, "r");
    char line[2048];
    int hit = 0;
    if (!f) return 0;
    while (fgets(line, sizeof(line), f)) if (strstr(line, what)) { hit = 1; break; }
    fclose(f);
    return hit;
}

static int count_of(int fd)
{
    char r[512];
    const char *c;
    if (ask(fd, "stats\n", r, sizeof(r)) < 0 || !(c = strstr(r, "count="))) return -1;
    return atoi(c + 6);
}

int main(int argc, char **argv)
{
    char line[2048], r[65536], what[256];
    pid_t p;
    int fd, i, n;
    unsigned char st, rb[4096];
    unsigned rnb;
    if (argc < 2) { fprintf(stderr, "usage: vserv_test PATH/TO/VSERV\n"); return 2; }
    setenv("VSERV", argv[1], 1);
    strcpy(dir, "/tmp/vserv-line-XXXXXX");
    if (!mkdtemp(dir)) return 2;
    sprintf(sock_path, "%s/v.sock", dir); sprintf(store_path, "%s/v.vxi", dir); sprintf(log_path, "%s/v.log", dir);
    port = 22000 + (int)(getpid() % 20000);
    printf("vserv line protocol, log, journal and snapshots (%s)\n", argv[1]);

    p = start(NULL, NULL, NULL);
    fd = connect_unix();
    check(fd >= 0, "connects over the Unix socket");
    if (fd < 0) return 1;
    {   struct stat sb; check(stat(store_path, &sb) == 0, "a new store's file exists from the start"); }
    check(ask(fd, "ping\n", r, sizeof(r)) > 0 && !strcmp(r, "ok pong"), "ping -> ok pong");
    for (i = 0, n = 0; i < 300; ++i) { vline(line, "add", 100 + i, i); if (ask(fd, line, r, sizeof(r)) > 0 && !strcmp(r, "ok")) ++n; }
    sprintf(what, "add: 300 rows, one line each (%d ok)", n);
    check(n == 300, what);
    check(count_of(fd) == 300, "stats: count=300");
    {   char q[2048]; int d; unsigned s = 17u * 2654435761u + 7u; int m = sprintf(q, "search 3");
        for (d = 0; d < DIM; ++d) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; m += sprintf(q + m, " %.4f", (double)(s % 2001u) / 1000.0 - 1.0); }
        strcpy(q + m, "\n");
        check(ask(fd, q, r, sizeof(r)) > 0 && !strncmp(r, "ok 3 117 ", 9), "search 3 ... -> ok 3 117 -0.000000 ...");
        vline(line, "add", 117, 18);                             /* replace row 117 by the vector of row 118 */
        check(ask(fd, line, r, sizeof(r)) > 0 && !strcmp(r, "ok") && count_of(fd) == 300, "add of an existing id replaces it (count stays 300)");
        check(ask(fd, q, r, sizeof(r)) > 0 && strncmp(r, "ok 3 117 ", 9) != 0, "the old vector of 117 is gone");
    }
    check(ask(fd, "get 100\n", r, sizeof(r)) > 0 && !strncmp(r, "ok ", 3) && strchr(r + 3, ' '), "get 100 -> ok X1 .. X8");
    check(ask(fd, "get 99999\n", r, sizeof(r)) > 0 && !strncmp(r, "err ", 4), "get of an unknown id -> err");
    check(ask(fd, "del 100\n", r, sizeof(r)) > 0 && !strcmp(r, "ok 1") && count_of(fd) == 299, "del 100 -> ok 1");
    check(ask(fd, "del 200 249\n", r, sizeof(r)) > 0 && !strcmp(r, "ok 50") && count_of(fd) == 249, "del 200 249 -> ok 50 (a range)");
    check(ask(fd, "search 3 1 2\n", r, sizeof(r)) > 0 && !strncmp(r, "err ", 4), "search with too few numbers -> err");
    check(ask(fd, "add 5 1 2 3 4 5 6 7 x\n", r, sizeof(r)) > 0 && !strncmp(r, "err ", 4), "add with a word that is not a number -> err");
    check(ask(fd, "frobnicate\n", r, sizeof(r)) > 0 && !strncmp(r, "err unknown", 11), "unknown command -> err");
    check(ask(fd, "help\r\n", r, sizeof(r)) > 0 && strstr(r, "search") != NULL, "help, with a \\r\\n line ending");
    check(ask(fd, "build\n", r, sizeof(r)) > 0 && !strcmp(r, "ok"), "build -> ok");
    check(ask(fd, "snapshot\n", r, sizeof(r)) > 0 && !strcmp(r, "ok started"), "snapshot -> ok started");
    for (i = 0; i < 50; ++i) { msleep(20); if (ask(fd, "stats\n", r, sizeof(r)) > 0 && strstr(r, "snapshot=idle last=ok")) break; }
    check(strstr(r, "snapshot=idle last=ok") != NULL, "stats: snapshot=idle last=ok once it is written");
    {   int fd2 = connect_unix();
        check(ask(fd2, "quit\n", r, sizeof(r)) > 0 && !strcmp(r, "ok bye") && read(fd2, r, 1) == 0, "quit -> ok bye, then the connection closes");
        close(fd2);
    }

    /* the binary protocol: snapshot, get, range delete */
    {   long long id = 101;
        long long lohi[2];
        float v[DIM];
        check(bin_call(fd, 8, &id, 8, &st, &rnb, rb) == 0 && st == 0 && rnb == DIM * 4, "binary get (op 8): 8 floats");
        memcpy(v, rb, sizeof(v));
        {   double nn = 0; int d; for (d = 0; d < DIM; ++d) nn += (double)v[d] * v[d];
            check(nn > 0.999 && nn < 1.001, "binary get: the stored vector is unit length"); }
        lohi[0] = 250; lohi[1] = 259;
        check(bin_call(fd, 9, lohi, 16, &st, &rnb, rb) == 0 && st == 0 && rnb == 8 && rb[0] == 10, "binary range delete (op 9): 10 removed");
        check(bin_call(fd, 7, NULL, 0, &st, &rnb, rb) == 0 && (st == 0 || st == 4), "binary snapshot (op 7)");
    }
    /* HTTP: /snapshot, /get, a range delete */
    msleep(300);
    check(http("POST /snapshot HTTP/1.1\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", r, sizeof(r)) > 0 && (strstr(r, "202 Accepted") || strstr(r, "409")), "HTTP POST /snapshot");
    check(http("GET /get?id=102 HTTP/1.1\r\nConnection: close\r\n\r\n", r, sizeof(r)) > 0 && strstr(r, "{\"vector\":["), "HTTP GET /get?id=102");
    check(http("POST /delete?lo=260&hi=269 HTTP/1.1\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", r, sizeof(r)) > 0 && strstr(r, "{\"removed\":10}"), "HTTP POST /delete?lo=260&hi=269 -> removed 10");
    check(http("GET /stats HTTP/1.1\r\nConnection: close\r\n\r\n", r, sizeof(r)) > 0 && strstr(r, "\"snapshot\":") && strstr(r, "\"count\":229"), "HTTP GET /stats: count 229 and the snapshot state");

    /* durability: no -x, killed with SIGKILL - the journal (flushed every second) holds everything */
    for (i = 0; i < 40; ++i) { vline(line, "add", 1000 + i, 1000 + i); ask(fd, line, r, sizeof(r)); }
    close(fd);
    msleep(1600);
    stop(p, SIGKILL);
    p = start(NULL, NULL, NULL);
    fd = connect_unix();
    n = count_of(fd);
    sprintf(what, "kill -9, restart: %d rows of 269 (229 + 40 added a second before the kill)", n);
    check(n == 269, what);
    {   char q[2048]; int d; unsigned s = 1033u * 2654435761u + 7u; int m = sprintf(q, "search 1");
        for (d = 0; d < DIM; ++d) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; m += sprintf(q + m, " %.4f", (double)(s % 2001u) / 1000.0 - 1.0); }
        strcpy(q + m, "\n");
        check(ask(fd, q, r, sizeof(r)) > 0 && !strncmp(r, "ok 1 1033 ", 10), "a row added just before the kill is found");
    }
    close(fd);
    stop(p, SIGTERM);

    /* the log: who did what */
    check(log_has("peer=unix:uid="), "log: Unix peers named by uid, gid and pid");
    check(log_has("add id=117 ok"), "log: every add, with its id");
    check(log_has("del lo=200 hi=249 removed=50"), "log: a range delete and what it removed");
    check(log_has("peer=tcp:127.0.0.1:"), "log: TCP peers named by address and port");
    check(log_has(" snapshot started") && log_has("snapshot ok file="), "log: snapshots, started and finished");
    check(log_has(" open") && log_has(" close requests="), "log: connections opening and closing");

    /* -S 1: a snapshot every second when something changed folds the journal into the file */
    {   struct stat a, b;
        p = start("-S", "1", NULL);
        fd = connect_unix();
        stat(store_path, &a);
        for (i = 0; i < 20; ++i) { vline(line, "add", 5000 + i, 5000 + i); ask(fd, line, r, sizeof(r)); }
        msleep(2500);
        stat(store_path, &b);
        check(b.st_ino != a.st_ino && count_of(fd) == 289, "-S 1: a timed snapshot replaced the file, nothing lost");
        close(fd);
        stop(p, SIGTERM);
    }

    /* texts: a text per row over the three protocols, ids in a range (a document's chunks), hybrid search; a big text;
     * the journal keeps them across a kill -9 */
    {   char *big = (char *)malloc(400000), *bl = (char *)malloc(900000), *rr = (char *)malloc(900000);
        long base = 7L << 20;
        int d, m;
        unsigned char pay[4096];
        p = start(NULL, NULL, NULL);
        fd = connect_unix();
        vline(line, "addt", base, 1); line[strlen(line) - 1] = 0; strcat(line, " hello\\nworld \\\\ tab\\there\n");
        check(ask(fd, line, r, sizeof(r)) > 0 && !strcmp(r, "ok"), "addt ID X1 .. X8 TEXT -> ok");
        sprintf(line, "text %ld\n", base);
        check(ask(fd, line, r, sizeof(r)) > 0 && !strcmp(r, "ok 22 hello\\nworld \\\\ tab\\there"), "text ID -> ok 22 and the text, escaped the same way");
        for (i = 1; i < 3; ++i) {
            vline(line, "addt", base | i, 10 + i); line[strlen(line) - 1] = 0;
            strcat(line, i == 2 ? " a chunk about the zebrafish genome\n" : " a chunk about tea\n");
            ask(fd, line, r, sizeof(r));
        }
        sprintf(line, "range %ld %ld\n", base, base | 0xfffff);
        sprintf(what, "ok 3 %ld %ld %ld", base, base | 1, base | 2);
        check(ask(fd, line, r, sizeof(r)) > 0 && !strcmp(r, what), "range LO HI -> the document's 3 chunks, in order");
        sprintf(what, "ok 1 %ld ", base | 2);
        check(ask(fd, "words 3 zebrafish\n", r, sizeof(r)) > 0 && !strncmp(r, what, strlen(what)), "words 3 zebrafish -> the one chunk that says it");
        vline(line, "hybrid 3", 0, 1); line[strlen(line) - 1] = 0;
        memmove(line + 9, line + 11, strlen(line + 11) + 1);       /* "hybrid 3 0 X1..." -> "hybrid 3 X1..." */
        strcat(line, " zebrafish\n");
        sprintf(what, "ok 3 %ld ", base | 2);
        check(ask(fd, line, r, sizeof(r)) > 0 && !strncmp(r, what, strlen(what)), "hybrid 3 X1 .. X8 zebrafish -> the word wins over the vector of chunk 0");
        sprintf(line, "text %ld\n", 99999L);
        check(ask(fd, line, r, sizeof(r)) > 0 && !strcmp(r, "err no such id"), "text of an unknown id -> err");
        vline(line, "addt", base | 9, 1); line[strlen(line) - 1] = 0; strcat(line, " bad \\q escape\n");
        check(ask(fd, line, r, sizeof(r)) > 0 && !strncmp(r, "err bad escape", 14), "addt with a bad escape -> err");
        for (i = 0; i < 400000 - 1; ++i) big[i] = (char)('a' + i % 26);  /* a 400 kB text: the request buffer grows */
        big[399999] = 0;
        vline(bl, "addt", base | 5, 5); m = (int)strlen(bl) - 1; bl[m] = ' '; strcpy(bl + m + 1, big); strcat(bl, "\n");
        check(ask(fd, bl, r, sizeof(r)) > 0 && !strcmp(r, "ok"), "addt with a 400 kB text -> ok");
        sprintf(line, "text %ld\n", base | 5);
        check(ask(fd, line, rr, 900000) == 10 + 399999 && !strncmp(rr, "ok 399999 ", 10) && !memcmp(rr + 10, big, 399999), "text of it: all 399999 bytes back");
        /* binary: add with a text (op 10), get it (op 11), a range (op 12), words-only hybrid (op 13) */
        {   long long id = (long long)(8L << 20), lohi[2];
            float v[DIM];
            const char *bt = "binary text, caf\xc3\xa9 and zebrafish";
            for (d = 0; d < DIM; ++d) v[d] = (float)(d + 1);
            memcpy(pay, &id, 8); memcpy(pay + 8, v, sizeof(v)); memcpy(pay + 8 + sizeof(v), bt, strlen(bt));
            check(bin_call(fd, 10, pay, (unsigned)(8 + sizeof(v) + strlen(bt)), &st, &rnb, rb) == 0 && st == 0, "binary add with a text (op 10)");
            check(bin_call(fd, 11, &id, 8, &st, &rnb, rb) == 0 && st == 0 && rnb == strlen(bt) && !memcmp(rb, bt, rnb), "binary get text (op 11): the bytes back");
            lohi[0] = base; lohi[1] = base | 0xfffff;
            check(bin_call(fd, 12, lohi, 16, &st, &rnb, rb) == 0 && st == 0 && rnb == 4 * 8, "binary range (op 12): 4 ids (3 chunks + the big one)");
            memset(pay, 0, 4); memcpy(pay + 4, "zebrafish", 9);
            check(bin_call(fd, 13, pay, 13, &st, &rnb, rb) == 0 && st == 0 && rnb == 2 * 12, "binary hybrid with words only (op 13): 2 hits");
        }
        /* HTTP: add with a JSON text, get it back, a range, hybrid */
        sprintf(line, "POST /add?id=%ld HTTP/1.1\r\nContent-Length: %d\r\nConnection: close\r\n\r\n", 9L << 20,
                (int)strlen("{\"vector\":[1,2,3,4,5,6,7,8],\"text\":\"json \\\"quoted\\\" caf\\u00e9\"}"));
        strcat(line, "{\"vector\":[1,2,3,4,5,6,7,8],\"text\":\"json \\\"quoted\\\" caf\\u00e9\"}");
        check(http(line, r, sizeof(r)) > 0 && strstr(r, "{\"ok\":true}"), "HTTP POST /add with {\"vector\": [...], \"text\": \"...\"}");
        sprintf(line, "GET /text?id=%ld HTTP/1.1\r\nConnection: close\r\n\r\n", 9L << 20);
        check(http(line, r, sizeof(r)) > 0 && strstr(r, "{\"text\":\"json \\\"quoted\\\" caf\xc3\xa9\"}"), "HTTP GET /text: the text as a JSON string");
        sprintf(line, "GET /range?lo=%ld&hi=%ld HTTP/1.1\r\nConnection: close\r\n\r\n", base, base | 2);
        sprintf(what, "{\"ids\":[%ld,%ld,%ld]}", base, base | 1, base | 2);
        check(http(line, r, sizeof(r)) > 0 && strstr(r, what), "HTTP GET /range");
        strcpy(line, "POST /hybrid?k=2 HTTP/1.1\r\nContent-Length: 20\r\nConnection: close\r\n\r\n{\"text\":\"zebrafish\"}");
        sprintf(what, "{\"ids\":[%ld,", base | 2);
        check(http(line, r, sizeof(r)) > 0 && strstr(r, "\"scores\":[") && (strstr(r, what) || strstr(r, "{\"ids\":[8388608,")), "HTTP POST /hybrid with words only");
        /* the journal keeps the texts across a kill -9 */
        msleep(1600);
        close(fd);
        stop(p, SIGKILL);
        p = start(NULL, NULL, NULL);
        fd = connect_unix();
        sprintf(line, "text %ld\n", base | 2);
        check(ask(fd, line, r, sizeof(r)) > 0 && !strcmp(r, "ok 34 a chunk about the zebrafish genome"), "kill -9, restart: the texts are back");
        sprintf(line, "text %ld\n", base | 5);
        check(ask(fd, line, rr, 900000) == 10 + 399999, "kill -9, restart: the 400 kB text too");
        close(fd);
        stop(p, SIGTERM);
        free(big); free(bl); free(rr);
    }

    /* batches: begin, writes that wait, commit as one change; abort; a closed connection drops its batch; a synced
     * commit survives a kill -9 that comes right after it. And stats reports the memory tiers. */
    {
        int fd2, n0;
        p = start(NULL, NULL, NULL);
        fd = connect_unix(); fd2 = connect_unix();
        check(ask(fd, "stats\n", r, sizeof(r)) > 0 && strstr(r, " hot=") && strstr(r, " cold=") && strstr(r, " memory=0"),
              "stats: hot=, cold= and memory=0");
        check(strstr(r, " damaged=0 rebuilt=0 rows_bad=0 rows_healed=0 journal=ok") != NULL, "stats: damaged=0 rebuilt=0 rows_bad=0 rows_healed=0 journal=ok");
        check(ask(fd, "scrub\n", r, sizeof(r)) > 0 && !strncmp(r, "ok damaged=0 rebuilt=0", 22) && log_has("scrub file="),
              "scrub: ok damaged=0 rebuilt=0, and a line in the log");
        check(ask(fd, "use acme\n", r, sizeof(r)) > 0 && !strncmp(r, "err this server has one store", 29), "use on a server of one store -> err");
        for (i = 7001; i <= 7003; ++i) { vline(line, "add", i, i); ask(fd, line, r, sizeof(r)); }
        n0 = count_of(fd);
        check(ask(fd, "commit\n", r, sizeof(r)) > 0 && !strcmp(r, "err no batch: begin first"), "commit without begin -> err");
        check(ask(fd, "begin\n", r, sizeof(r)) > 0 && !strcmp(r, "ok"), "begin -> ok");
        check(ask(fd, "begin\n", r, sizeof(r)) > 0 && !strncmp(r, "err a batch is open", 19), "begin twice -> err");
        check(ask(fd, "del 7001\n", r, sizeof(r)) > 0 && !strcmp(r, "ok queued"), "del in a batch -> ok queued");
        vline(line, "add", 7004, 7004);
        check(ask(fd, line, r, sizeof(r)) > 0 && !strcmp(r, "ok queued"), "add in a batch -> ok queued");
        vline(line, "add", 7005, 7005); ask(fd, line, r, sizeof(r));
        check(ask(fd, "del 7002 7003\n", r, sizeof(r)) > 0 && !strcmp(r, "ok queued"), "a range delete in a batch -> ok queued");
        check(count_of(fd2) == n0 && ask(fd2, "get 7004\n", r, sizeof(r)) > 0 && !strncmp(r, "err", 3), "another connection sees none of it yet");
        check(ask(fd, "commit sync\n", r, sizeof(r)) > 0 && !strcmp(r, "ok 4"), "commit sync -> ok 4");
        check(count_of(fd2) == n0 - 1 && ask(fd2, "get 7004\n", r, sizeof(r)) > 0 && !strncmp(r, "ok ", 3)
              && ask(fd2, "get 7002\n", r, sizeof(r)) > 0 && !strncmp(r, "err", 3), "after the commit: all four changes");
        check(ask(fd, "begin\n", r, sizeof(r)) > 0 && !strcmp(r, "ok"), "begin again");
        vline(line, "add", 7006, 7006); ask(fd, line, r, sizeof(r));
        check(ask(fd, "abort\n", r, sizeof(r)) > 0 && !strcmp(r, "ok 0") && ask(fd, "get 7006\n", r, sizeof(r)) > 0 && !strncmp(r, "err", 3),
              "abort -> ok 0, the row never came");
        check(ask(fd2, "begin\n", r, sizeof(r)) > 0 && !strcmp(r, "ok"), "a batch on a second connection");
        vline(line, "add", 7007, 7007); ask(fd2, line, r, sizeof(r));
        close(fd2);
        msleep(200);
        check(ask(fd, "get 7007\n", r, sizeof(r)) > 0 && !strncmp(r, "err", 3), "closing the connection drops its batch");
        stop(p, SIGKILL);                                         /* right after the synced commit: no flush timer */
        p = start(NULL, NULL, NULL);
        close(fd);
        fd = connect_unix();
        check(ask(fd, "get 7005\n", r, sizeof(r)) > 0 && !strncmp(r, "ok ", 3) && ask(fd, "get 7001\n", r, sizeof(r)) > 0
              && !strncmp(r, "err", 3), "kill -9, restart: the synced batch is all there");
        close(fd);
        stop(p, SIGTERM);
        check(log_has("commit changes=4 sync ok"), "log: the commit, its size and sync");
        check(log_has("batch of 1 changes dropped"), "log: the batch a closed connection dropped");
        p = start("-m", "1", NULL);
        fd = connect_unix();
        check(fd >= 0 && ask(fd, "stats\n", r, sizeof(r)) > 0 && strstr(r, " memory=1"), "-m 1: stats says memory=1");
        if (fd >= 0) close(fd);
        stop(p, SIGTERM);
    }

    /* grouped search, numbers and number ranges, update keeping the rest of the row: line protocol, binary, HTTP,
     * a batch, and the journal across a kill -9 */
    {   long d0 = 1L << 20, d1 = 2L << 20, d2 = 3L << 20, id;
        int c2, ok, m;
        float v[DIM];
        unsigned char pay[256];
        p = start(NULL, NULL, NULL);
        fd = connect_unix();
        for (i = 0; i < 30; ++i) { vline(line, "add", (i < 10 ? d0 : (i < 20 ? d1 : d2)) | (i % 10), 9000 + i); ask(fd, line, r, sizeof(r)); }
        for (i = 0, ok = 1; i < 30; ++i) {
            sprintf(line, "num %ld %d\n", (i < 10 ? d0 : (i < 20 ? d1 : d2)) | (i % 10), i % 10);
            ok &= ask(fd, line, r, sizeof(r)) > 0 && !strcmp(r, "ok");
        }
        check(ok, "num ID N on 30 rows -> ok");
        sprintf(line, "num %ld\n", d1 | 7);
        check(ask(fd, line, r, sizeof(r)) > 0 && !strcmp(r, "ok 7"), "num ID -> ok N");
        check(ask(fd, "num 5\n", r, sizeof(r)) > 0 && !strcmp(r, "err no such id") && ask(fd, "num 5 1\n", r, sizeof(r)) > 0
              && !strcmp(r, "err no such id"), "num of an unknown id, read or set -> err no such id");
        vline(line, "group 3", 20, 9000 + 13);                   /* the vector of document 2, chunk 3 */
        sprintf(what, "ok 3 %ld ", d1 | 3);
        check(ask(fd, line, r, sizeof(r)) > 0 && !strncmp(r, what, strlen(what)), "group 3 20 -> 3 hits, the query's own row first");
        {   long g[3]; char *q = r + 5;
            for (m = 0; m < 3; ++m) { g[m] = strtol(q, &q, 10) >> 20; strtod(q, &q); }
            check(g[0] != g[1] && g[0] != g[2] && g[1] != g[2], "group 3 20 -> three different documents"); }
        vline(line, "group 3", 64, 1);
        check(ask(fd, line, r, sizeof(r)) > 0 && !strncmp(r, "err usage: group", 16), "group with SHIFT 64 -> err usage");
        vline(line, "between 5 3", 3, 9000 + 13);
        check(ask(fd, line, r, sizeof(r)) > 0 && !strncmp(r, "ok 3 ", 5) && strstr(r, " 1048579 ") && strstr(r, " 2097155 ") && strstr(r, " 3145731 "),
              "between 5 3 3 -> the 3 rows whose number is 3");
        vline(line, "between 5 4", 3, 1);
        check(ask(fd, line, r, sizeof(r)) > 0 && !strncmp(r, "err usage: between", 18), "between with LO > HI -> err usage");
        vline(line, "update", d0 | 4, 77777);                    /* a new vector for document 1 chunk 4: number 4 kept */
        check(ask(fd, line, r, sizeof(r)) > 0 && !strcmp(r, "ok"), "update ID X1 .. X8 -> ok");
        vline(line, "update", 4242, 1);
        check(ask(fd, line, r, sizeof(r)) > 0 && !strncmp(r, "err no such id", 14), "update of an unknown id -> err");
        sprintf(line, "num %ld\n", d0 | 4);
        vline(what, "between 1 4", 4, 77777); what[strlen(what)] = 0;
        check(ask(fd, line, r, sizeof(r)) > 0 && !strcmp(r, "ok 4") && ask(fd, what, r, sizeof(r)) > 0 && !strncmp(r, "ok 1 1048580 ", 13),
              "after update: the number kept, and the new vector found");
        check(ask(fd, "begin\n", r, sizeof(r)) > 0 && !strcmp(r, "ok"), "begin");
        sprintf(line, "num %ld 99\n", d2 | 9);
        check(ask(fd, line, r, sizeof(r)) > 0 && !strcmp(r, "ok queued"), "num in a batch -> ok queued");
        vline(line, "update", d2 | 9, 88888);
        check(ask(fd, line, r, sizeof(r)) > 0 && !strcmp(r, "ok queued"), "update in a batch -> ok queued");
        check(ask(fd, "commit sync\n", r, sizeof(r)) > 0 && !strcmp(r, "ok 2"), "commit sync -> ok 2");
        stop(p, SIGKILL);                                         /* the journal carries the numbers and the updates */
        p = start(NULL, NULL, NULL);
        close(fd);
        fd = connect_unix();
        sprintf(line, "num %ld\n", d2 | 9);
        vline(what, "between 1 99", 99, 88888);
        check(ask(fd, line, r, sizeof(r)) > 0 && !strcmp(r, "ok 99") && ask(fd, what, r, sizeof(r)) > 0 && !strncmp(r, "ok 1 3145737 ", 13),
              "kill -9, restart: the batch's number and update are there");
        id = d1 | 3;                                             /* binary: get its vector, then search with it */
        check(bin_call(fd, 8, &id, 8, &st, &rnb, rb) == 0 && st == 0 && rnb == DIM * 4, "binary get of a row for the queries");
        memcpy(v, rb, sizeof(v));
        {   unsigned sh = 20; memcpy(pay, &sh, 4); memcpy(pay + 4, v, sizeof(v)); }
        c2 = bin_call(fd, 15, pay, 4 + sizeof(v), &st, &rnb, rb);
        {   long long h0, g1, g2;
            memcpy(&h0, rb, 8);                                    /* k 0: 10, and the store holds earlier tests' rows */
            memcpy(&g1, rb + 12, 8); memcpy(&g2, rb + 24, 8);
            check(c2 == 0 && st == 0 && rnb >= 3 * 12 && rnb % 12 == 0 && h0 == (long long)(d1 | 3) && (g1 >> 20) != (h0 >> 20)
                  && (g2 >> 20) != (h0 >> 20) && (g1 >> 20) != (g2 >> 20), "binary grouped search (op 15): its own row first, one hit a document"); }
        {   long long lh[2]; lh[0] = 3; lh[1] = 3; memcpy(pay, lh, 16); memcpy(pay + 16, v, sizeof(v)); }
        check(bin_call(fd, 16, pay, 16 + sizeof(v), &st, &rnb, rb) == 0 && st == 0 && rnb == 3 * 12, "binary number range (op 16): the 3 rows numbered 3");
        {   long long q2[2]; q2[0] = d0 | 1; q2[1] = -5; memcpy(pay, q2, 16); }
        check(bin_call(fd, 18, pay, 16, &st, &rnb, rb) == 0 && st == 0, "binary set a number (op 18)");
        id = d0 | 1;
        {   long long nv = 0;
            c2 = bin_call(fd, 19, &id, 8, &st, &rnb, rb); memcpy(&nv, rb, 8);
            check(c2 == 0 && st == 0 && rnb == 8 && nv == -5, "binary get a number (op 19): -5"); }
        memcpy(pay, &id, 8); memcpy(pay + 8, v, sizeof(v));
        check(bin_call(fd, 17, pay, 8 + sizeof(v), &st, &rnb, rb) == 0 && st == 0, "binary update (op 17)");
        id = 4242; memcpy(pay, &id, 8);
        check(bin_call(fd, 17, pay, 8 + sizeof(v), &st, &rnb, rb) == 0 && st == 1, "binary update of no row -> status 1");
        {   char body[256], req[512]; int n = sprintf(body, "[");
            for (m = 0; m < DIM; ++m) n += sprintf(body + n, "%s%.6f", m ? "," : "", (double)v[m]);
            strcat(body, "]");
            sprintf(req, "POST /search?k=3&group=20 HTTP/1.1\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s", (int)strlen(body), body);
            {   char a1[32], a2[32];                             /* d0|1 has this vector now too: both documents lead */
                sprintf(a1, "%ld", d0 | 1); sprintf(a2, "%ld", d1 | 3);
                check(http(req, r, sizeof(r)) > 0 && strstr(r, "{\"ids\":[") && strstr(r, a1) && strstr(r, a2), "HTTP /search?k=3&group=20"); }
            sprintf(req, "POST /search?k=9&lo=-5&hi=-5 HTTP/1.1\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s", (int)strlen(body), body);
            sprintf(what, "{\"ids\":[%ld],", d0 | 1);
            check(http(req, r, sizeof(r)) > 0 && strstr(r, what), "HTTP /search?lo=-5&hi=-5: the one row numbered -5");
            sprintf(req, "POST /search?k=3&group=20&lo=1 HTTP/1.1\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s", (int)strlen(body), body);
            check(http(req, r, sizeof(r)) > 0 && strstr(r, "400 Bad Request"), "HTTP /search with group and lo -> 400");
            sprintf(req, "POST /update?id=%ld HTTP/1.1\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s", d2 | 0, (int)strlen(body), body);
            check(http(req, r, sizeof(r)) > 0 && strstr(r, "{\"ok\":true}"), "HTTP POST /update?id=");
        }
        sprintf(line, "POST /num?id=%ld&num=123 HTTP/1.1\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", d2 | 0);
        check(http(line, r, sizeof(r)) > 0 && strstr(r, "{\"ok\":true}"), "HTTP POST /num?id=&num=");
        sprintf(line, "GET /num?id=%ld HTTP/1.1\r\nConnection: close\r\n\r\n", d2 | 0);
        check(http(line, r, sizeof(r)) > 0 && strstr(r, "{\"num\":123}"), "HTTP GET /num?id= -> 123");
        check(http("GET /num?id=4242 HTTP/1.1\r\nConnection: close\r\n\r\n", r, sizeof(r)) > 0 && strstr(r, "404"), "HTTP GET /num of no row -> 404");
        close(fd);
        stop(p, SIGTERM);
        check(log_has("update id=") && log_has("num id="), "log: the updates and numbers");
    }

    /* API keys (-K): two tenants in one store, each its own ids - both have an id 1 - and none of the other's rows in
     * any protocol; housekeeping refused to a key; the Unix socket sees everything; the journal keeps it all */
    {   char kpath[160], kstore[160], hreq[1024], body[256];
        const char *k7 = "k_seven_0123456789ab", *k9 = "k_nine_0123456789abcd";
        const char *old = serve_path;
        int f7, f9, fa, ok, q, m;
        FILE *kf;
        sprintf(kpath, "%s/keys.tsv", dir); sprintf(kstore, "%s/keys.vxi", dir);
        kf = fopen(kpath, "w");
        if (kf) { fprintf(kf, "# key tenant\n%s\t7\n%s   9\n", k7, k9); fclose(kf); }
        serve_path = kstore;
        p = start("-K", kpath, NULL);
        f7 = connect_tcp(); f9 = connect_tcp(); fa = connect_unix();
        vline(line, "search 3", 0, 1); line[strlen(line) - 1] = 0; memmove(line + 9, line + 11, strlen(line + 11) + 1); strcat(line, "\n");
        check(ask(f7, line, r, sizeof(r)) > 0 && !strcmp(r, "err auth KEY first"), "-K: a TCP search before auth -> err auth KEY first");
        check(ask(f7, "ping\n", r, sizeof(r)) > 0 && !strcmp(r, "ok pong"), "-K: ping needs no key");
        check(ask(f7, "auth k_wrong_aaaaaaaaaaaa\n", r, sizeof(r)) > 0 && !strcmp(r, "err unknown key"), "auth with an unknown key -> err");
        sprintf(line, "auth %s\n", k7);
        check(ask(f7, line, r, sizeof(r)) > 0 && !strcmp(r, "ok"), "auth with tenant 7's key -> ok");
        sprintf(line, "auth %s\n", k9); ask(f9, line, r, sizeof(r));
        for (i = 0, ok = 1; i < 20; ++i) {
            vline(line, "add", 1 + i, 3000 + i); ok &= ask(f7, line, r, sizeof(r)) > 0 && !strcmp(r, "ok");
            vline(line, "add", i ? 100 + i : 1, 4000 + i); ok &= ask(f9, line, r, sizeof(r)) > 0 && !strcmp(r, "ok");
        }
        check(ok && count_of(f7) == 20 && count_of(f9) == 20, "two tenants add 20 rows each, both an id 1: each counts its own 20");
        check(count_of(fa) == 40, "the Unix socket, no key: all 40 rows");
        {   char a7[512], a9[512];
            ask(f7, "get 1\n", a7, sizeof(a7)); ask(f9, "get 1\n", a9, sizeof(a9));
            check(!strncmp(a7, "ok ", 3) && !strncmp(a9, "ok ", 3) && strcmp(a7, a9) != 0, "get 1: each tenant its own row 1");
        }
        check(ask(f7, "get 105\n", r, sizeof(r)) > 0 && !strcmp(r, "err no such id"), "tenant 7 cannot get tenant 9's row 105");
        for (q = 0, ok = 1; q < 20; ++q) {                       /* searches with tenant 9's own vectors: tenant 7 never sees them */
            char *z;
            vline(line, "search 10", 0, 4000 + q); line[strlen(line) - 1] = 0; memmove(line + 10, line + 12, strlen(line + 12) + 1); strcat(line, "\n");
            ask(f7, line, r, sizeof(r));
            if (strncmp(r, "ok 10 ", 6)) ok = 0;
            for (z = r + 6, m = 0; m < 10; ++m) { long id = strtol(z, &z, 10); strtod(z, &z); if (id < 1 || id > 20) ok = 0; }
        }
        check(ok, "tenant 7 searches with tenant 9's vectors: only its own ids 1-20 come back");
        check(ask(f7, "del 100 119\n", r, sizeof(r)) > 0 && !strcmp(r, "ok 0") && count_of(f9) == 20, "tenant 7 deleting 100-119: none of tenant 9's go");
        check(ask(f7, "build\n", r, sizeof(r)) > 0 && !strncmp(r, "err the server's housekeeping", 29), "build with a key -> refused");
        vline(line, "add", 17592186044416L, 1);
        check(ask(f7, line, r, sizeof(r)) > 0 && !strncmp(r, "err", 3), "an id of 2^44 from a tenant -> err");
        check(ask(f7, "stats\n", r, sizeof(r)) > 0 && !strcmp(r, "ok count=20 dim=8"), "stats with a key: its own count and the dimension, nothing else");
        {   unsigned char pay[64];
            int fb = connect_tcp();
            float v[DIM];
            memset(v, 0, sizeof(v)); v[0] = 1.0f;
            memcpy(pay, v, sizeof(v));
            check(bin_call(fb, 1, pay, sizeof(v), &st, &rnb, rb) == 0 && st == 6, "binary search before a key (op 1) -> status 6");
            check(bin_call(fb, 20, k9, (unsigned)strlen(k9), &st, &rnb, rb) == 0 && st == 0, "binary auth (op 20) -> status 0");
            {   int c2 = bin_call(fb, 1, pay, sizeof(v), &st, &rnb, rb), z; long long id;
                ok = c2 == 0 && st == 0 && rnb == 10 * 12;
                for (z = 0; ok && z < 10; ++z) { memcpy(&id, rb + 12 * z, 8); if (!(id == 1 || (id >= 101 && id <= 119))) ok = 0; }
                check(ok, "binary search with tenant 9's key: only tenant 9's ids"); }
            check(bin_call(fb, 4, NULL, 0, &st, &rnb, rb) == 0 && st == 1, "binary build with a key -> status 1");
            close(fb);
        }
        sprintf(body, "[1,0,0,0,0,0,0,0]");
        sprintf(hreq, "POST /search?k=5 HTTP/1.1\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s", (int)strlen(body), body);
        check(http(hreq, r, sizeof(r)) > 0 && strstr(r, "401 Unauthorized"), "HTTP without a key -> 401");
        sprintf(hreq, "POST /search?k=5 HTTP/1.1\r\nAuthorization: Bearer %s\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s", k7, (int)strlen(body), body);
        {   char *ids;
            ok = http(hreq, r, sizeof(r)) > 0 && (ids = strstr(r, "{\"ids\":[")) != NULL;
            if (ok) { char *z = ids + 8; for (m = 0; m < 5; ++m) { long id = strtol(z, &z, 10); if (id < 1 || id > 20) ok = 0; if (*z == ',') ++z; } }
            check(ok, "HTTP with tenant 7's key: only its ids");
        }
        sprintf(hreq, "GET /stats HTTP/1.1\r\nAuthorization: Bearer %s\r\nConnection: close\r\n\r\n", k9);
        check(http(hreq, r, sizeof(r)) > 0 && strstr(r, "{\"count\":20,\"dim\":8}"), "HTTP /stats with a key: its own count");
        sprintf(hreq, "POST /save HTTP/1.1\r\nAuthorization: Bearer %s\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", k9);
        check(http(hreq, r, sizeof(r)) > 0 && strstr(r, "403 Forbidden"), "HTTP /save with a key -> 403");
        check(http("GET /health HTTP/1.1\r\nConnection: close\r\n\r\n", r, sizeof(r)) > 0 && strstr(r, "200 OK"), "HTTP /health needs no key");
        close(f7); close(f9); close(fa);
        msleep(1500);                                             /* -f 1: on the disk within a second */
        stop(p, SIGKILL);                                         /* the journal: rows, ids and namespaces */
        p = start("-K", kpath, NULL);
        f7 = connect_tcp();
        sprintf(line, "auth %s\n", k7); ask(f7, line, r, sizeof(r));
        check(count_of(f7) == 20 && ask(f7, "get 1\n", r, sizeof(r)) > 0 && !strncmp(r, "ok ", 3), "kill -9, restart: tenant 7's 20 rows and its row 1");
        close(f7);
        stop(p, SIGTERM);
        check(log_has("auth tenant=") && log_has("auth refused: an unknown key"), "log: the keys shown, and the one refused");
        serve_path = old;
    }

    /* a tenant server: a directory, a store per tenant; the line protocol, HTTP and binary choose one; the log names it */
    {
        char tdir[160], tf[200];
        int fa, fb, fc;
        struct stat sb;
        sprintf(tdir, "%s/tenants", dir);
        mkdir(tdir, 0700);
        serve_path = tdir;
        p = start(NULL, NULL, NULL);
        fa = connect_unix(); fb = connect_unix();
        check(fa >= 0 && fb >= 0, "a tenant server: clients connect");
        check(ask(fa, "ping\n", r, sizeof(r)) > 0 && !strcmp(r, "ok pong"), "ping before choosing a tenant");
        check(ask(fa, "stats\n", r, sizeof(r)) > 0 && !strcmp(r, "err no tenant: use NAME first"), "stats before use -> err no tenant");
        check(ask(fa, "use ../evil\n", r, sizeof(r)) > 0 && !strncmp(r, "err a tenant's name", 19), "use ../evil -> err: a name never leaves the directory");
        check(ask(fa, "use acme extra\n", r, sizeof(r)) > 0 && !strcmp(r, "err usage: use NAME"), "use with two words -> err usage");
        check(ask(fa, "use acme\n", r, sizeof(r)) > 0 && !strcmp(r, "ok"), "use acme -> ok, created");
        sprintf(tf, "%s/acme.vxi", tdir);
        check(stat(tf, &sb) == 0, "the tenant's store is DIR/acme.vxi");
        vline(line, "add", 1, 1);
        check(ask(fa, line, r, sizeof(r)) > 0 && !strcmp(r, "ok") && count_of(fa) == 1, "acme: add, count 1");
        check(ask(fb, "use beta\n", r, sizeof(r)) > 0 && !strcmp(r, "ok") && count_of(fb) == 0, "beta: count 0 - tenants are apart");
        vline(line, "add", 2, 2); ask(fb, line, r, sizeof(r));
        vline(line, "add", 3, 3); ask(fb, line, r, sizeof(r));
        check(count_of(fb) == 2 && count_of(fa) == 1, "beta has 2 rows, acme still 1");
        check(http("GET /t/acme/stats HTTP/1.1\r\nConnection: close\r\n\r\n", r, sizeof(r)) > 0 && strstr(r, "\"count\":1"), "HTTP GET /t/acme/stats: count 1");
        check(http("GET /stats HTTP/1.1\r\nConnection: close\r\n\r\n", r, sizeof(r)) > 0 && strstr(r, "400 Bad Request") && strstr(r, "no tenant"),
              "HTTP without /t/NAME/ -> 400 no tenant");
        check(http("GET /health HTTP/1.1\r\nConnection: close\r\n\r\n", r, sizeof(r)) > 0 && strstr(r, "\"ok\":true"), "HTTP /health needs no tenant");
        check(http("GET /t/bad!name/stats HTTP/1.1\r\nConnection: close\r\n\r\n", r, sizeof(r)) > 0 && strstr(r, "403 Forbidden"), "HTTP /t/bad!name/ -> 403");
        fc = connect_unix();
        check(bin_call(fc, 6, NULL, 0, &st, &rnb, rb) == 0 && st == 5, "binary stats before choosing a tenant: status 5");
        check(bin_call(fc, 14, "beta", 4, &st, &rnb, rb) == 0 && st == 0, "binary use (op 14) of beta");
        check(bin_call(fc, 6, NULL, 0, &st, &rnb, rb) == 0 && st == 0 && rnb == 32 && rb[0] == 2, "binary stats on beta: 2 rows");
        check(bin_call(fc, 14, "x/y", 3, &st, &rnb, rb) == 0 && st == 1 && rnb > 0, "binary use of a bad name: status 1 and the reason");
        close(fc);
        check(ask(fa, "begin\n", r, sizeof(r)) > 0 && ask(fa, "use beta\n", r, sizeof(r)) > 0 && !strncmp(r, "err a batch is open", 19),
              "use while a batch is open -> err");
        ask(fa, "abort\n", r, sizeof(r));
        close(fa); close(fb);
        stop(p, SIGTERM);
        p = start(NULL, NULL, NULL);
        fa = connect_unix();
        check(ask(fa, "use acme\n", r, sizeof(r)) > 0 && count_of(fa) == 1, "restart: acme's row is back");
        close(fa);
        stop(p, SIGTERM);
        check(log_has("tenant created name=acme"), "log: a tenant created");
        check(log_has("tenant=acme add id=1 ok"), "log: a connection's lines name its tenant");
        {   /* keys on a tenant directory: a key names its tenant's file, and no other */
            char kp2[200];
            FILE *kf = (FILE *)0;
            int fk;
            sprintf(kp2, "%s/dkeys.tsv", dir);
            if ((kf = fopen(kp2, "w")) != (FILE *)0) { fprintf(kf, "k_acme_0123456789abcdef acme\n"); fclose(kf); }
            p = start("-K", kp2, NULL);
            fk = connect_tcp();
            check(ask(fk, "use acme\n", r, sizeof(r)) > 0 && !strcmp(r, "err auth KEY first"), "-K on a directory: use before auth -> err");
            check(ask(fk, "auth k_acme_0123456789abcdef\n", r, sizeof(r)) > 0 && !strcmp(r, "ok") && count_of(fk) == 1, "auth with acme's key: acme's store");
            check(ask(fk, "use beta\n", r, sizeof(r)) > 0 && !strcmp(r, "err your key is for another tenant"), "the acme key at use beta -> err");
            check(http("GET /stats HTTP/1.1\r\nAuthorization: Bearer k_acme_0123456789abcdef\r\nConnection: close\r\n\r\n", r, sizeof(r)) > 0
                  && strstr(r, "\"count\":1"), "HTTP with acme's key, no /t/NAME/: acme");
            check(http("GET /t/beta/stats HTTP/1.1\r\nAuthorization: Bearer k_acme_0123456789abcdef\r\nConnection: close\r\n\r\n", r, sizeof(r)) > 0
                  && strstr(r, "403"), "HTTP with acme's key at /t/beta/ -> 403");
            close(fk);
            stop(p, SIGTERM);
        }
        serve_path = store_path;
    }

    /* mutual TLS: the name in the client's certificate goes into the log (needs openssl and curl; skipped without) */
    if (system("command -v openssl >/dev/null && command -v curl >/dev/null") == 0) {
        char cmd[2048], ports[16];
        const char *ext = "subjectAltName=DNS:localhost,IP:127.0.0.1";
        sprintf(cmd, "cd %s && openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -keyout ca.key -out ca.pem -days 2 "
                     "-subj '/CN=test CA' -addext basicConstraints=critical,CA:TRUE -addext keyUsage=critical,keyCertSign >/dev/null 2>&1 && "
                     "printf '%s\n' > san.ext && "
                     "openssl req -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -keyout srv.key -out srv.csr -subj /CN=localhost >/dev/null 2>&1 && "
                     "openssl x509 -req -in srv.csr -CA ca.pem -CAkey ca.key -CAcreateserial -out srv.pem -days 2 -extfile san.ext >/dev/null 2>&1 && "
                     "openssl req -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -keyout cli.key -out cli.csr -subj '/CN=ingest worker 7' >/dev/null 2>&1 && "
                     "openssl x509 -req -in cli.csr -CA ca.pem -CAkey ca.key -CAcreateserial -out cli.pem -days 2 >/dev/null 2>&1 && "
                     "openssl req -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -keyout ten.key -out ten.csr -subj /CN=acme >/dev/null 2>&1 && "
                     "openssl x509 -req -in ten.csr -CA ca.pem -CAkey ca.key -CAcreateserial -out ten.pem -days 2 >/dev/null 2>&1", dir, ext);
        if (system(cmd) == 0) {
            char crt[160], key[160], ca[160];
            sprintf(crt, "%s/srv.pem", dir); sprintf(key, "%s/srv.key", dir); sprintf(ca, "%s/ca.pem", dir);
            port += 1;
            sprintf(ports, "%d", port);
            p = fork();
            if (p == 0) {
                execl(getenv("VSERV"), getenv("VSERV"), "-p", ports, "-s", sock_path, "-t", "2", "-d", "8", "-l", log_path,
                      "-c", crt, "-k", key, "-a", ca, store_path, (char *)0);
                _exit(127);
            }
            for (i = 0; i < 300 && (fd = connect_tcp()) < 0; ++i) msleep(20);
            if (fd >= 0) close(fd);
            sprintf(cmd, "curl -s --cacert %s/ca.pem --cert %s/cli.pem --key %s/cli.key https://localhost:%d/health > %s/curl.out 2>&1", dir, dir, dir, port, dir);
            n = system(cmd);
            sprintf(cmd, "grep -q '\\\"ok\\\":true' %s/curl.out", dir);
            check(n == 0 && system(cmd) == 0, "mTLS: curl with a client certificate gets in");
            sprintf(cmd, "curl -s --cacert %s/ca.pem https://localhost:%d/health > /dev/null 2>&1", dir, port);
            check(system(cmd) != 0, "mTLS: curl without one is refused");
            msleep(200);
            stop(p, SIGTERM);
            check(log_has("cn=\"ingest worker 7\""), "log: the client certificate's name (cn=\"ingest worker 7\")");
            check(log_has("tls handshake failed"), "log: the refused handshake");

            /* -U: a tenant server where a client certificate names the one tenant its holder may use */
            {
                char tdir[160];
                sprintf(tdir, "%s/tenants-u", dir);
                mkdir(tdir, 0700);
                port += 1;
                sprintf(ports, "%d", port);
                p = fork();
                if (p == 0) {
                    execl(getenv("VSERV"), getenv("VSERV"), "-p", ports, "-s", sock_path, "-t", "2", "-d", "8", "-l", log_path,
                          "-c", crt, "-k", key, "-a", ca, "-U", tdir, (char *)0);
                    _exit(127);
                }
                for (i = 0; i < 300 && (fd = connect_tcp()) < 0; ++i) msleep(20);
                if (fd >= 0) close(fd);
                sprintf(cmd, "curl -s --cacert %s/ca.pem --cert %s/ten.pem --key %s/ten.key https://localhost:%d/t/acme/stats > %s/curl.out 2>&1",
                        dir, dir, dir, port, dir);
                n = system(cmd);
                sprintf(cmd, "grep -q '\\\"count\\\":0' %s/curl.out", dir);
                check(n == 0 && system(cmd) == 0, "-U: the certificate for acme reaches /t/acme/");
                sprintf(cmd, "curl -s --cacert %s/ca.pem --cert %s/ten.pem --key %s/ten.key https://localhost:%d/t/beta/stats > %s/curl.out 2>&1",
                        dir, dir, dir, port, dir);
                n = system(cmd);
                sprintf(cmd, "grep -q 'another tenant' %s/curl.out", dir);
                check(n == 0 && system(cmd) == 0, "-U: the same certificate is refused at /t/beta/");
                msleep(200);
                stop(p, SIGTERM);
                check(log_has("use beta refused: the certificate names another tenant"), "log: the refused tenant");
            }
        } else
            printf("  (openssl could not make test certificates - mTLS log check skipped)\n");
    } else
        printf("  (openssl or curl missing - mTLS log check skipped)\n");

    {   char cmd[256]; sprintf(cmd, "rm -rf %s", dir); if (system(cmd) != 0) {} }
    printf("vserv line: %s\n", fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}
