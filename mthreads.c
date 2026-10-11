/* mthreads.c - readers and a writer on one Mem at the same time (run it under -fsanitize=thread).
 * Mirrors vthreads.c one layer up: mem.c's own rwlock (Mem.lk) guards its side-table; the readers
 * exercise every read-ish mem_* entry point (search/touch, relevance, confidence, known, gaps,
 * hot list, provenance) while the writer does the structural ops (note, corroborate/contradict,
 * merge, prune, forget, consolidate - which itself alternates locking Mem.lk and vecdb's own
 * db->lk, see mem_consolidate's comment) and periodic mem_save.
 *   gcc -std=gnu99 -g -O1 -fsanitize=thread -I. mthreads.c -lm -lpthread -o .build/mthreads && .build/mthreads */
#include "mem.c"
#include <stdio.h>

static Mem *g_m;
static int g_stop;
static long g_reads[6];

static void *reader_(void *arg)
{
    long id = (long)arg;
    u32 s = 54321u + (u32)id * 7919u;
    VHit h[8]; i32 ids[32];
    while (!__atomic_load_n(&g_stop, __ATOMIC_RELAXED)) {
        char q[64]; i32 t = (i32)(vdb_rng_(&s) % 40);
        f32 conf; i32 vid;
        sprintf(q, "word%d alpha", t);
        switch (vdb_rng_(&s) % 10) {
        case 0: mem_search(g_m, q, 5, h); break;
        case 1: { i32 n = vecdb_find(g_m->db, "note 3"); if (n >= 0) { mem_relevance(g_m, n); mem_confidence(g_m, n); mem_provenance(g_m, n, ids, 32); } } break;
        case 2: mem_known(g_m, q, 0.5f, &conf, &vid); break;
        case 3: mem_gaps(g_m, 0.9f, ids, 8); break;
        case 4: mem_hot_ids(g_m, ids, 8); break;
        case 5: { i32 n = vecdb_find(g_m->db, "note 6"); if (n >= 0) mem_kind(g_m, n); } break;   /* NOT g_m->db->nn directly: an unguarded raw field read races vdb_newnode_'s writes */
        case 6: vecdb_search(g_m->db, q, 5, 2, -1, h); break;
        case 7: mem_set_lifetime(g_m, (MemKind)(vdb_rng_(&s) % 3), 3600.0f + (f32)(vdb_rng_(&s) % 1000));break;  /* writes shared state from a "reader" thread too */
        case 8: if (vdb_rng_(&s) % 6 == 0) mem_cold_count(g_m); break;   /* real file I/O under the lock - kept rare so TSan stays fast */
        default: { i32 n = vecdb_find(g_m->db, "note 3"); if (n >= 0) mem_touch(g_m, n); }
        }
        ++g_reads[id];
    }
    return (void *)0;
}

int main(void)
{
    static Arena arena;
    pthread_t th[4];
    long i;
    i32 w;
    VecDB *db = vecdb_new(&arena, 64);
    g_m = mem_open(db);
    mem_cold_open(g_m, "/tmp/mthreads.vdb");
    for (w = 0; w < 200; ++w) { char t[32], x[128]; sprintf(t, "note %d", w); sprintf(x, "alpha word%d [[note %d]] #tag%d", w % 40, w / 3, w % 3); mem_note_event(g_m, t, x); }
    vecdb_build(db);
    for (i = 0; i < 4; ++i) pthread_create(&th[i], (const pthread_attr_t *)0, reader_, (void *)i);
    for (w = 200; w < 700; ++w) {                                   /* the writer: note, corroborate, merge, prune, forget, consolidate, save */
        char t[32], x[128]; i32 vid;
        sprintf(t, "note %d", w % 260); sprintf(x, "alpha word%d [[note %d]] #tag%d", w % 40, w / 3 % 260, w % 3);
        vid = mem_note_event(g_m, t, x);
        if (vid >= 0 && w % 5 == 0) { i32 other = vecdb_find(db, "note 3"); if (other >= 0) mem_corroborate(g_m, other, vid); }
        if (w % 7 == 0) { i32 other = vecdb_find(db, "note 4"); if (other >= 0 && vid >= 0) mem_contradict(g_m, other, vid); }
        if (w % 23 == 0) { i32 a = vecdb_find(db, "note 5"), b = vecdb_find(db, "note 6"); if (a >= 0 && b >= 0) mem_merge(g_m, a, b); }
        if (w % 31 == 0) mem_prune(g_m, 0.02f);
        if (w % 41 == 0) { i32 n = vecdb_find(db, "note 7"); if (n >= 0) mem_forget(g_m, n); }
        if (w % 89 == 0) { i32 n = vecdb_find(db, "note 8"); if (n >= 0) { i32 cid = mem_archive(g_m, n); if (cid >= 0) mem_recall(g_m, cid); } }   /* real fsync'd I/O - kept rare, TSan makes each one costly */
        if (w % 97 == 0) mem_consolidate(g_m, "(rule promote-facts (and (kind event) (similar 0.55)) (promote fact 0.55))\n");
        if (w % 53 == 0) mem_consolidate_hot(g_m, "(rule promote-facts (and (kind event) (similar 0.55)) (promote fact 0.55))\n");
        if (w % 61 == 0) { i32 ids[3]; ids[0] = vecdb_find(db, "note 9"); ids[1] = vecdb_find(db, "note 10"); ids[2] = vid;
                            mem_consolidate_ids(g_m, "(rule promote-facts (and (kind event) (similar 0.55)) (promote fact 0.55))\n", ids, 3); }
        if (w % 150 == 0) vecdb_build(db);
        if (w % 251 == 0) mem_cold_open(g_m, "/tmp/mthreads.vdb");   /* re-point cold storage while readers may be mid-mem_cold_count */
        if (w % 300 == 0) mem_save(g_m, "/tmp/mthreads.mem");
    }
    __atomic_store_n(&g_stop, 1, __ATOMIC_RELAXED);
    for (i = 0; i < 4; ++i) pthread_join(th[i], (void **)0);
    printf("threads ok: %ld reads by 4 readers during 1000 writes, %d notes\n", g_reads[0] + g_reads[1] + g_reads[2] + g_reads[3], vecdb_count(db, VN_NOTE));
    mem_free(g_m); vecdb_free(db);
    remove("/tmp/mthreads.vdb"); remove("/tmp/mthreads.vdb.vdt"); remove("/tmp/mthreads.vdb.vxi"); remove("/tmp/mthreads.vdb.vxi.j");
    remove("/tmp/mthreads.vdb.mem.cold"); remove("/tmp/mthreads.mem");
    return 0;
}
