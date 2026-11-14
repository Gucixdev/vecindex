/* scrash.c - kill -9 a store writer at random moments, with background snapshots running; the file must always load
 * with every synced change in it, exactly once.
 *   gcc -std=gnu99 -O2 -Isrc tests/scrash.c -lm -lpthread -o .build/scrash && .build/scrash [rounds]
 * The child opens (or creates) the store, adds rows with ids 0, 1, 2, ... (each id's vector is a function of the id),
 * deletes id i - 5 after adding every id i ending in 9, replaces id i - 2 by the same row after every id ending in 7
 * (store_put: one journal record), calls store_sync every 16 adds and reports the ids synced so
 * far over a pipe, and starts a background snapshot of its own file every 200 adds. The parent SIGKILLs it after a
 * random time: in even rounds its process group too (the snapshot writer dies with it, as in a power cut), in odd
 * rounds only the child (the snapshot writer finishes on its own, as when the server crashes). Then it loads the file
 * and checks: every synced id is there exactly once with its own vector, every synced delete stuck, search finds
 * rows by their vectors. */
#include "store.c"
#include <stdio.h>
#include <signal.h>
#include <sys/wait.h>
#include <time.h>

#define PATHX "/tmp/scrash.vxi"
#define DIM 16

static void vec_(i64 id, f32 *v)
{
    u32 s = (u32)id * 2654435761u + 12345u;
    i32 d;
    for (d = 0; d < DIM; ++d) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; v[d] = (f32)(s % 2001u) / 1000.0f - 1.0f; }
}

static i64 max_id_(RagStore *s)
{
    i64 m = -1;
    i32 r;
    for (r = 0; r < s->count; ++r) if (!STORE_DEAD(s, r) && STORE_ID(s, r) > m) m = STORE_ID(s, r);
    return m;
}

static void child_(int wfd)
{
    static Arena arena;
    RagStore *s = store_load(&arena, PATHX);
    i64 id;
    f32 v[DIM];
    i32 n = 0;
    setpgid(0, 0);
    if (!s) { s = store_new(&arena, DIM); store_set_live(s, 1, 1.0f); if (store_save(s, PATHX) != 0) _exit(4); }
    else store_set_live(s, 1, 1.0f);
    for (id = max_id_(s) + 1; ; ++id, ++n) {
        vec_(id, v);
        if (store_add(s, v, id) != 0) _exit(5);
        if (id % 10 == 9) store_delete(s, id - 5);
        if (id % 10 == 7) { vec_(id - 2, v); store_put(s, v, id - 2, 0); }   /* replaced by itself: still exactly once */
        if (n % 16 == 15 && store_sync(s, PATHX) == 0) {
            char line[32];
            int k = sprintf(line, "%lld\n", (long long)(id + 1));
            if (write(wfd, line, (size_t)k) < 0) _exit(3);
        }
        if (n % 200 == 199) store_snapshot(s, PATHX);
        store_snapshot_poll(s, 0);
    }
}

/* every process of group g is gone (bounded wait: 10 s) */
static void wait_group_(pid_t g)
{
    i32 i;
    for (i = 0; i < 1000 && kill(-g, 0) == 0; ++i) { struct timespec ts; ts.tv_sec = 0; ts.tv_nsec = 10000000L; nanosleep(&ts, NULL); }
}

int main(int argc, char **argv)
{
    int rounds = argc > 1 ? atoi(argv[1]) : 20, r, bad = 0;
    i64 confirmed = 0, stamps = 0, via_jn = 0;
    u64 last_stamp = 0;
    unsigned seed = (unsigned)time(NULL);
    remove(PATHX); remove(PATHX ".j"); remove(PATHX ".jn"); remove(PATHX ".tmp"); remove(PATHX ".snap");
    srand(seed);
    for (r = 0; r < rounds; ++r) {
        int fds[2], status, ms = 20 + rand() % 300;
        pid_t p;
        char buf[4096];
        ssize_t n;
        struct timespec ts;
        i64 got = 0;
        if (pipe(fds) != 0) return 2;
        p = fork();
        if (p == 0) { close(fds[0]); child_(fds[1]); _exit(0); }
        setpgid(p, p);
        close(fds[1]);
        ts.tv_sec = 0; ts.tv_nsec = (long)ms * 1000000L; nanosleep(&ts, NULL);
        if (r % 2 == 0) kill(-p, SIGKILL); else kill(p, SIGKILL);
        waitpid(p, &status, 0);
        wait_group_(p);                                           /* an orphaned snapshot writer finishes first */
        while ((n = read(fds[0], buf, sizeof buf - 1)) > 0) {
            char *a, *e;
            buf[n] = 0;
            for (a = buf; (e = strchr(a, '\n')); a = e + 1) { i64 x = atoll(a); if (x > got) got = x; }
        }
        close(fds[0]);
        if (got > confirmed) confirmed = got;
        {
            static Arena arena;
            u64 jn_stamp = 0;
            RagStore *s;
            {   /* a snapshot finished but its journal had not moved over yet: this load must replay PATH.jn */
                FILE *jf = fopen(PATHX ".jn", "rb");
                u32 m; i32 d;
                if (jf) { if (fread(&m, 4, 1, jf) != 1 || fread(&d, 4, 1, jf) != 1 || fread(&jn_stamp, 8, 1, jf) != 1) jn_stamp = 0; fclose(jf); }
            }
            s = store_load(&arena, PATHX);
            i64 id;
            i32 rr, *seen, missing = 0, dup = 0, undeleted = 0, wrong = 0, found = 0, tried = 0;
            f32 v[DIM], w[DIM];
            if (!s) {
                if (confirmed > 0) { printf("round %d: the store does not load after the kill (%lld ids synced)\n", r, (long long)confirmed); ++bad; }
                continue;
            }
            if (s->stamp != last_stamp) { ++stamps; last_stamp = s->stamp; }
            if (jn_stamp && s->stamp == jn_stamp) ++via_jn;
            seen = (i32 *)calloc((size_t)(confirmed + 1), sizeof(i32));
            for (rr = 0; rr < s->count; ++rr)
                if (!STORE_DEAD(s, rr) && STORE_ID(s, rr) < confirmed) ++seen[STORE_ID(s, rr)];
            for (id = 0; id < confirmed; ++id) {
                i32 must_go = id % 10 == 4 && id + 5 < confirmed, may_go = id % 10 == 4;
                if (seen[id] > 1) ++dup;
                if (must_go && seen[id]) ++undeleted;
                if (!may_go && seen[id] != 1) ++missing;
                if (seen[id] == 1 && (id % 97) == 0) {
                    f32 nn = 0.0f, err = 0.0f;
                    i32 d;
                    vec_(id, v);
                    for (d = 0; d < DIM; ++d) nn += v[d] * v[d];
                    nn = (f32)sqrt((f64)nn);
                    if (store_get_vec(s, id, w) != 0) ++wrong;
                    else { for (d = 0; d < DIM; ++d) err += (f32)fabs((f64)(w[d] - v[d] / nn)); if (err > 1e-3f) ++wrong; }
                }
            }
            for (id = 0; id < confirmed && tried < 30; id += confirmed / 30 + 1) {
                i64 hit[3];
                f32 sc[3];
                i32 h, q;
                if (id % 10 == 4) continue;
                vec_(id, v);
                ++tried;
                h = store_search(s, v, 3, hit, sc);
                for (q = 0; q < h; ++q) if (hit[q] == id) { ++found; break; }
            }
            if (missing || dup || undeleted || wrong || found < tried) {
                printf("round %d (%s): %lld ids synced, %d missing, %d twice, %d deletes lost, %d wrong vectors, found %d/%d by search\n",
                       r, r % 2 == 0 ? "group killed" : "child killed", (long long)confirmed, missing, dup, undeleted, wrong, found, tried);
                ++bad;
            }
            free(seen);
            store_destroy(s);
        }
    }
    printf("store crash test: %d rounds, %lld ids synced, %lld different files (snapshots done), %lld loads through PATH.jn, %s\n",
           rounds, (long long)confirmed, (long long)stamps, (long long)via_jn, bad ? "FAILURES" : "all consistent");
    remove(PATHX); remove(PATHX ".j"); remove(PATHX ".jn"); remove(PATHX ".tmp"); remove(PATHX ".snap");
    return bad ? 1 : 0;
}
