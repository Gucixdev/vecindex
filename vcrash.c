/* vcrash.c - kill -9 a writer at random moments; the database must always load and keep what was synced.
 *   gcc -std=gnu99 -O2 -I. vcrash.c -lm -lpthread -o .build/vcrash && .build/vcrash [rounds]
 * The child loads (or creates) the database, adds notes and calls vecdb_sync every few of them, reporting each
 * completed sync (note count) over a pipe. The parent SIGKILLs it after a random time, loads the files and
 * checks: it opens, holds at least the last reported count, every synced body is readable, search answers. */
#include "vecdb.c"
#include <stdio.h>
#include <signal.h>
#include <sys/wait.h>
#include <time.h>

#define DB "/tmp/vcrash.vdb"

static void child_(int wfd, int start)
{
    static Arena arena;
    VecDB *db = vecdb_load(&arena, DB);
    i32 i, added = start;
    char t[64], x[256];
    if (!db) db = vecdb_new(&arena, 64);
    for (i = 0; ; ++i) {
        sprintf(t, "note %d", added); sprintf(x, "body of note %d links [[note %d]] #t%d some words", added, added / 2, added % 5);
        vecdb_add_note(db, t, x); ++added;
        if (i % 4 == 3) {
            if (vecdb_sync(db, DB) == 0) { char line[32]; int n = sprintf(line, "%d\n", added); if (write(wfd, line, (size_t)n) < 0) _exit(3); }
        }
    }
}

int main(int argc, char **argv)
{
    static Arena arena;
    int rounds = argc > 1 ? atoi(argv[1]) : 30, r, confirmed = 0, bad = 0;
    unsigned seed = (unsigned)time(NULL);
    remove(DB); remove(DB ".vdt"); remove(DB ".vxi"); remove(DB ".vxi.j"); remove(DB ".tmp"); remove(DB ".vxi.tmp");
    srand(seed);
    for (r = 0; r < rounds; ++r) {
        int fds[2], status, got = 0, ms = 15 + rand() % 250;
        pid_t p;
        char buf[4096]; ssize_t n; struct timespec ts;
        if (pipe(fds) != 0) return 2;
        p = fork();
        if (p == 0) { close(fds[0]); child_(fds[1], confirmed); _exit(0); }
        close(fds[1]);
        ts.tv_sec = 0; ts.tv_nsec = (long)ms * 1000000L; nanosleep(&ts, NULL);
        kill(p, SIGKILL); waitpid(p, &status, 0);
        { long fl = 0; (void)fl; }
        while ((n = read(fds[0], buf, sizeof buf - 1)) > 0) { char *s, *e; buf[n] = 0; for (s = buf; (e = strchr(s, '\n')); s = e + 1) { int v = atoi(s); if (v > got) got = v; } }
        close(fds[0]);
        if (got > confirmed) confirmed = got;
        {   VecDB *db = vecdb_load(&arena, DB);
            i32 nn = 0, i, unreadable = 0;
            if (!db) { if (confirmed > 0) { printf("round %d: database does not load after kill (confirmed %d)\n", r, confirmed); ++bad; } continue; }
            nn = vecdb_count(db, VN_NOTE);
            for (i = 0; i < db->nn; ++i) if (db->nodes[i].kind == VN_NOTE) { char *tx = vecdb_text(db, i); if (!tx) ++unreadable; free(tx); }
            vecdb_build(db);
            { VHit h[3]; if (nn > 0 && vecdb_search(db, "body of note", 3, 0, -1, h) == 0) { printf("round %d: search returned nothing\n", r); ++bad; } }
            if (nn < confirmed) { printf("round %d: %d notes, %d were synced\n", r, nn, confirmed); ++bad; }
            if (unreadable) { printf("round %d: %d notes with unreadable bodies\n", r, unreadable); ++bad; }
            /* a later child starts from what really is on disk */
            confirmed = nn > confirmed ? confirmed : confirmed;
            vecdb_free(db);
        }
    }
    printf("crash test: %d rounds, %d notes confirmed, %s\n", rounds, confirmed, bad ? "FAILURES" : "all consistent");
    return bad ? 1 : 0;
}
