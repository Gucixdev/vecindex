/* mcrash.c - kill -9 a writer at random moments; mem.c's own side-table (PATH.mem) and cold
 * archive (PATH.mem.cold) must always reload consistently with whatever vecdb itself held onto
 * (see vcrash.c/vcrash2.c for the same discipline on vecdb's own files - this is the same bar,
 * one layer up). The child loads (or creates) db+Mem, adds events, corroborates/contradicts some
 * of them, periodically archives one to cold storage, and syncs both files (db first - mem_load
 * needs live vids to already exist), reporting each completed round over a pipe. The parent
 * SIGKILLs it after a random time and checks: everything loads, every meta entry's vid is a real
 * live vecdb note (no dangling bookkeeping), mem_fix finds nothing wrong, search still answers.
 *   gcc -std=gnu99 -O2 -I. mcrash.c -lm -lpthread -o .build/mcrash && .build/mcrash [rounds] */
#include "mem.c"
#include <stdio.h>
#include <signal.h>
#include <sys/wait.h>
#include <time.h>

#define DB "/tmp/mcrash.vdb"
#define MEMP "/tmp/mcrash.mem"

static void child_(int wfd, int start)
{
    static Arena arena;
    VecDB *db = vecdb_load(&arena, DB);
    Mem *m;
    i32 i, added = start;
    char t[64], x[256];
    if (!db) db = vecdb_new(&arena, 64);
    m = mem_open(db);
    mem_cold_open(m, DB);
    if (start > 0) mem_load(m, MEMP);                             /* best-effort: absent/stale is fine, matches mem_load's own contract */
    for (i = 0; ; ++i) {
        i32 vid;
        sprintf(t, "note %d", added); sprintf(x, "body of note %d links [[note %d]] #t%d some words", added, added / 2, added % 5);
        vid = mem_note_event(m, t, x); ++added;
        if (vid >= 0 && i % 3 == 0) { i32 other = vecdb_find(db, "note 1"); if (other >= 0) mem_corroborate(m, other, vid); }
        if (vid >= 0 && i % 11 == 0) { i32 other = vecdb_find(db, "note 2"); if (other >= 0) mem_contradict(m, other, vid); }
        if (i % 17 == 0) { i32 victim = vecdb_find(db, "note 5"); if (victim >= 0) mem_archive(m, victim); }
        if (i % 4 == 3) {
            if (vecdb_sync(db, DB) == 0 && mem_save(m, MEMP) == 0) {
                char line[32]; int n = sprintf(line, "%d\n", added);
                if (write(wfd, line, (size_t)n) < 0) _exit(3);
            }
        }
    }
}

int main(int argc, char **argv)
{
    static Arena arena;
    int rounds = argc > 1 ? atoi(argv[1]) : 20, r, confirmed = 0, bad = 0;
    unsigned seed = (unsigned)time(NULL);
    srand(seed);
    remove(DB); remove(DB ".vdt"); remove(DB ".vxi"); remove(DB ".vxi.j"); remove(DB ".tmp"); remove(DB ".vxi.tmp");
    remove(DB ".mem.cold"); remove(MEMP); remove(MEMP ".tmp");
    for (r = 0; r < rounds; ++r) {
        int fds[2], status, got = 0, ms = 15 + rand() % 300;
        pid_t p;
        char buf[4096]; ssize_t n; struct timespec ts;
        if (pipe(fds) != 0) return 2;
        p = fork();
        if (p == 0) { close(fds[0]); child_(fds[1], confirmed); _exit(0); }
        close(fds[1]);
        ts.tv_sec = 0; ts.tv_nsec = (long)ms * 1000000L; nanosleep(&ts, NULL);
        kill(p, SIGKILL); waitpid(p, &status, 0);
        while ((n = read(fds[0], buf, sizeof buf - 1)) > 0) { char *s, *e; buf[n] = 0; for (s = buf; (e = strchr(s, '\n')); s = e + 1) { int v = atoi(s); if (v > got) got = v; } }
        close(fds[0]);
        if (got > confirmed) confirmed = got;
        {   VecDB *db = vecdb_load(&arena, DB);
            Mem *m;
            i32 i, bad_meta = 0, fixed;
            if (!db) { if (confirmed > 0) { printf("round %d: database does not load after kill (confirmed %d)\n", r, confirmed); ++bad; } continue; }
            m = mem_open(db);
            if (mem_load(m, MEMP) != 0 && confirmed > 0) { printf("round %d: mem side-table does not load (confirmed %d)\n", r, confirmed); ++bad; }
            for (i = 0; i < m->n; ++i) {
                MemMeta *e = &m->meta[i];
                if (e->archived) continue;
                if (e->vid < 0 || e->vid >= db->nn || db->nodes[e->vid].kind != VN_NOTE) ++bad_meta;
            }
            if (bad_meta) { printf("round %d: %d meta entries point at a vid that is not a live note\n", r, bad_meta); ++bad; }
            fixed = mem_fix(m);
            if (fixed) { printf("round %d: mem_fix found %d issue(s) after reload (should have been none)\n", r, fixed); ++bad; }
            vecdb_build(db);
            { VHit h[3]; if (vecdb_count(db, VN_NOTE) > 0 && vecdb_search(db, "body of note", 3, 0, -1, h) == 0) { printf("round %d: search returned nothing\n", r); ++bad; } }
            mem_free(m); vecdb_free(db);
        }
    }
    printf("mcrash test: %d rounds, %d notes confirmed, %s\n", rounds, confirmed, bad ? "FAILURES" : "all consistent");
    remove(DB); remove(DB ".vdt"); remove(DB ".vxi"); remove(DB ".vxi.j"); remove(DB ".tmp"); remove(DB ".vxi.tmp");
    remove(DB ".mem.cold"); remove(MEMP); remove(MEMP ".tmp");
    return bad ? 1 : 0;
}
