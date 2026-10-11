/* vthreads.c - readers and a writer on one database at the same time (run it under -fsanitize=thread).
 *   gcc -std=gnu99 -g -O1 -fsanitize=thread -I. vthreads.c -lm -lpthread -o .build/vthreads && .build/vthreads */
#include "vecdb.c"
#include <stdio.h>

static VecDB *g_db;
static int g_stop;
static long g_reads[6];

static void *reader_(void *arg)
{
    long id = (long)arg;
    u32 s = 12345u + (u32)id * 7919u;
    VHit h[8]; i32 ids[32];
    while (!__atomic_load_n(&g_stop, __ATOMIC_RELAXED)) {
        char q[64]; i32 t = (i32)(vdb_rng_(&s) % 40);
        sprintf(q, "word%d alpha", t);
        switch (vdb_rng_(&s) % 7) {
        case 0: vecdb_search(g_db, q, 5, 0, -1, h); break;
        case 1: vecdb_query(g_db, "alpha #tag1 -#tag2", 5, 0, h); break;
        case 2: vecdb_explore(g_db, q, 5, (const VExplore *)0, h); break;
        case 3: { i32 n = vecdb_find(g_db, "note 3"); if (n >= 0) { char *tx = vecdb_text(g_db, n); free(tx); vecdb_links(g_db, n, ids, 32); vecdb_backlinks(g_db, n, ids, 32); } } break;
        case 4: vecdb_count(g_db, VN_NOTE); break;
        case 5: vecdb_query(g_db, "to:[[note 2]]", 5, 1, h); break;
        default: vecdb_search(g_db, q, 5, 2, -1, h);
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
    g_db = vecdb_new(&arena, 64);
    for (w = 0; w < 200; ++w) { char t[32], x[128]; sprintf(t, "note %d", w); sprintf(x, "alpha word%d [[note %d]] #tag%d", w % 40, w / 3, w % 3); vecdb_add_note(g_db, t, x); }
    vecdb_build(g_db);
    for (i = 0; i < 4; ++i) pthread_create(&th[i], (const pthread_attr_t *)0, reader_, (void *)i);
    for (w = 200; w < 5200; ++w) {                              /* the writer: add, replace, remove, rebuild, sync */
        char t[32], x[128]; sprintf(t, "note %d", w % 260); sprintf(x, "alpha word%d [[note %d]] #tag%d", w % 40, w / 3 % 260, w % 3);
        vecdb_add_note(g_db, t, x);
        if (w % 17 == 0) vecdb_remove_note(g_db, w % 150);
        if (w % 150 == 0) vecdb_build(g_db);
        if (w % 1000 == 0) { vecdb_sync(g_db, "/tmp/vthreads.vdb"); }
    }
    __atomic_store_n(&g_stop, 1, __ATOMIC_RELAXED);
    for (i = 0; i < 4; ++i) pthread_join(th[i], (void **)0);
    printf("threads ok: %ld reads by 4 readers during 5000 writes, %d notes\n", g_reads[0] + g_reads[1] + g_reads[2] + g_reads[3], vecdb_count(g_db, VN_NOTE));
    remove("/tmp/vthreads.vdb"); remove("/tmp/vthreads.vdb.vdt"); remove("/tmp/vthreads.vdb.vxi"); remove("/tmp/vthreads.vdb.vxi.j");
    vecdb_free(g_db);
    return 0;
}
