/* vcrash2.c - the gap vcrash.c does not cover: re-indexing an EXISTING database from scratch (vecdb_new, not
 * vecdb_load - what the CLI's `index` command does when pointed at a path that already has a database), killed
 * mid-save. vdbl_save's body/vector-file replacement used to touch PATH.vdt/.vxi in place; a crash between the
 * two could pair an old node table with new (or vice versa) bodies - silently wrong reads, not just a crash.
 * The fix (a fresh generation-suffixed filename for the whole new file set, only the final node-table rename
 * switches to it - see VecDB.bodygen) is proven here: after the kill, the database must load as either the
 * complete OLD content or the complete NEW content, never a mix, never unreadable, never a load failure.
 *   gcc -std=gnu99 -O2 -I. vcrash2.c -lm -lpthread -o .build/vcrash2 && .build/vcrash2 [rounds] */
#include "vecdb.c"
#include <stdio.h>
#include <signal.h>
#include <sys/wait.h>
#include <time.h>

#define DB "/tmp/vcrash2.vdb"
#define NOLD 4000
#define NNEW 20000

static void cleanup_(void)
{
    /* bare names, generation-suffixed leftovers from an interrupted round, and their .tmp/.j siblings */
    system("rm -f " DB " " DB ".vdt " DB ".vdt.tmp " DB ".vxi " DB ".vxi.tmp " DB ".vxi.j " DB ".tmp "
           DB ".vdt.g* " DB ".vxi.g* " DB ".img.vxi* " DB ".tmp.* 2>/dev/null");
}

static void write_old_(void)
{
    static Arena arena;
    VecDB *db = vecdb_new(&arena, 64);
    i32 i;
    char t[64], x[256];
    for (i = 0; i < NOLD; ++i) { sprintf(t, "Old %d", i); sprintf(x, "old body %d some words here [[Old %d]] #old", i, i / 2); vecdb_add_note(db, t, x); }
    if (vecdb_save(db, DB) != 0) { fprintf(stderr, "initial save failed\n"); exit(2); }
    vecdb_free(db);
}

static void reindex_child_(void)
{
    static Arena arena;
    VecDB *db = vecdb_new(&arena, 64);                          /* NOT loaded: the vulnerable path (db->dbpath is NULL) */
    i32 i;
    char t[64], x[256];
    for (i = 0; i < NNEW; ++i) { sprintf(t, "New %d", i); sprintf(x, "new body %d different words entirely [[New %d]] #new", i, i / 2); vecdb_add_note(db, t, x); }
    vecdb_save(db, DB);                                         /* killed at a random point during this call */
    _exit(0);
}

int main(int argc, char **argv)
{
    static Arena arena;
    int rounds = argc > 1 ? atoi(argv[1]) : 20, r, bad = 0, saw_old = 0, saw_new = 0, saw_neither = 0;
    unsigned seed = (unsigned)time(NULL);
    srand(seed);
    for (r = 0; r < rounds; ++r) {
        pid_t p; int status, ms = 100 + rand() % 160;             /* brackets the ~150-250ms this save takes under load */
        struct timespec ts;
        cleanup_();
        write_old_();
        p = fork();
        if (p == 0) reindex_child_();
        ts.tv_sec = 0; ts.tv_nsec = (long)ms * 1000000L; nanosleep(&ts, NULL);
        kill(p, SIGKILL); waitpid(p, &status, 0);
        {   VecDB *db = vecdb_load(&arena, DB);
            i32 hasOld0, hasOldLast, hasNew0, hasNewLast, mixed;
            if (!db) { printf("round %d (kill at %dms): database will not load: %s\n", r, ms, vecdb_error()); ++bad; continue; }
            hasOld0 = vecdb_find(db, "Old 0") >= 0; hasOldLast = vecdb_find(db, "Old 3999") >= 0;
            hasNew0 = vecdb_find(db, "New 0") >= 0; hasNewLast = vecdb_find(db, "New 19999") >= 0;
            mixed = (hasOld0 != hasOldLast) || (hasNew0 != hasNewLast) || (hasOldLast && hasNewLast);
            if (mixed) { printf("round %d (kill at %dms): PARTIAL/mixed content (old0=%d oldlast=%d new0=%d newlast=%d)\n", r, ms, hasOld0, hasOldLast, hasNew0, hasNewLast); ++bad; }
            else if (hasOldLast) ++saw_old;
            else if (hasNewLast) ++saw_new;
            else { printf("round %d (kill at %dms): NEITHER old nor new content present\n", r, ms); ++saw_neither; ++bad; }
            {
                i32 i, unreadable = 0;
                vecdb_build(db);
                for (i = 0; i < db->nn; ++i) if (db->nodes[i].kind == VN_NOTE) { char *tx = vecdb_text(db, i); if (!tx) ++unreadable; free(tx); }
                if (unreadable) { printf("round %d: %d unreadable bodies\n", r, unreadable); ++bad; }
                vecdb_free(db);
            }
        }
    }
    printf("crash2 test: %d rounds, %d landed on the old copy, %d on the complete new copy, %d neither, %s\n",
           rounds, saw_old, saw_new, saw_neither, bad ? "FAILURES" : "all consistent (never a mix)");
    cleanup_();
    return bad ? 1 : 0;
}
