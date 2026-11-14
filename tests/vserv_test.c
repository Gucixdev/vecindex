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
