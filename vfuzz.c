/* vfuzz.c - random input against the parser, the query language, replace/remove and save/load.
 *   gcc -std=gnu99 -g -O1 -fsanitize=address,undefined -I. vfuzz.c -lm -lpthread -o .build/vfuzz && .build/vfuzz [rounds] */
#include "vecdb.c"
#include <stdio.h>

static u32 r_ = 30000007u;
static u32 rnd_(void) { return vdb_rng_(&r_); }

static void junk_(char *b, i32 n)
{
    static const char *tok[] = { "[[", "]]", "#", "-#", "![[", "![", "](", ")", "|", "to:", "from:", "tag:", " ", "\n", "```", "`", "---", "tags:", ",", "/", "\\", "\xc3\xa9", "\xff", "\xe2\x82", "a", "B", "note", "x1", "http://", "*", "\t", "\r" };
    i32 i, w = 0;
    for (i = 0; i < n && w < 300; ++i) { const char *t = tok[rnd_() % (sizeof tok / sizeof *tok)]; i32 l = (i32)strlen(t); memcpy(b + w, t, (size_t)l); w += l; }
    b[w] = 0;
}

int main(int argc, char **argv)
{
    static Arena arena;
    i32 rounds = argc > 1 ? atoi(argv[1]) : 3000, i, ids[64];
    VecDB *db = vecdb_new(&arena, 64), *ld;
    char title[400], text[4000], q[400];
    VHit hits[32];
    for (i = 0; i < rounds; ++i) {
        u32 op = rnd_() % 10;
        junk_(title, 1 + (i32)(rnd_() % 6)); junk_(text, 5 + (i32)(rnd_() % 60)); junk_(q, 1 + (i32)(rnd_() % 10));
        if (op < 5) vecdb_add_note(db, title, text);
        else if (op == 5 && db->nn > 0) { i32 id = (i32)(rnd_() % (u32)db->nn); vecdb_remove_note(db, id); }
        else if (op == 6) { vecdb_build(db); vecdb_query(db, q, 5, (i32)(rnd_() & 1), hits); }
        else if (op == 7) { vecdb_search(db, q, 7, (i32)(rnd_() % 3), -1, hits); }
        else if (op == 8 && db->nn > 0) { i32 id = (i32)(rnd_() % (u32)db->nn); vecdb_links(db, id, ids, 64); vecdb_backlinks(db, id, ids, 64); vecdb_tags_of(db, id, ids, 64); }
        else if (op == 9) { vecdb_find(db, title); vecdb_find_tag(db, q); }
    }
    vecdb_build(db);
    if (vecdb_save(db, "/tmp/vfuzz.vdb") != 0) { fprintf(stderr, "save failed\n"); return 1; }
    ld = vecdb_load(&arena, "/tmp/vfuzz.vdb");
    if (!ld) { fprintf(stderr, "load failed\n"); return 1; }
    if (ld->nn != db->nn) { fprintf(stderr, "node count differs after reload: %d vs %d\n", ld->nn, db->nn); return 1; }
    vecdb_build(ld);
    for (i = 0; i < 200; ++i) { junk_(q, 1 + (i32)(rnd_() % 10)); vecdb_query(ld, q, 5, 0, hits); }
    remove("/tmp/vfuzz.vdb"); remove("/tmp/vfuzz.vdb.vdt"); remove("/tmp/vfuzz.vdb.vxi"); remove("/tmp/vfuzz.vdb.vxi.j");
    printf("fuzz ok: %d rounds, %d nodes (%d notes, %d ghosts, %d tags)\n", rounds, db->nn, vecdb_count(db, VN_NOTE), vecdb_count(db, VN_GHOST), vecdb_count(db, VN_TAG));
    vecdb_free(ld); vecdb_free(db);
    return 0;
}
