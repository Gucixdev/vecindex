/* make.c - writes the golden files of tests/compat: a store with every feature the build it is compiled against has,
 * saved, then more changes left in the journal only. Built once against each released vecindex (and against this
 * one), its files are committed; compat_test.c opens them with every later build and checks every value.
 *
 *   cc -Iinclude -Isrc tests/compat/make.c src/vecindex.c -lm -lpthread -o make     (add -DNOW for this build's features)
 *   ./make DIR NAME MODE        MODE: cos (f32 rows, cosine), l2i8 (int8 rows, Euclidean), plain (cos, no texts)
 *
 * The values are functions of the id (vec_, text_, tag_), so the checking side recomputes them; compat.h holds them. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "vecindex.h"
#include "compat.h"

int main(int argc, char **argv)
{
    char path[512], t[96];
    float v[CDIM];
    vi_db *db;
    vi_i64 id;
    int l2, tx;
    if (argc != 4) { fprintf(stderr, "usage: make DIR NAME cos|l2i8|plain\n"); return 2; }
    l2 = !strcmp(argv[3], "l2i8"); tx = strcmp(argv[3], "plain") != 0;
    sprintf(path, "%s/%s.vxi", argv[1], argv[2]);
    if (!(db = vi_open(path, CDIM, l2 ? VI_L2 : 0))) { fprintf(stderr, "%s\n", vi_error()); return 1; }
    if (l2 && vi_set_layout(db, 4, 1) != 0) return 1;
    vi_label(db, "pet", NULL); vi_label(db, "dog", "pet"); vi_label(db, "cat", "pet");
    for (id = 1; id <= CSAVED; ++id) {                           /* saved in the file */
        vec_(id, v);
        if (tx && id % 3 == 0) { text_(id, t); vi_add_text(db, id, v, t, -1); } else vi_add(db, id, v);
        vi_set_tag(db, id, tag_(id));
        if (id % 5 == 0) vi_label_add(db, id, id % 10 == 0 ? "dog" : "cat");
#ifdef NOW
        vi_set_num(db, id, num_(id));
        if (ns_(id)) vi_set_ns(db, id, ns_(id));
#endif
    }
    for (id = 7; id <= CSAVED; id += 50) vi_delete(db, id);
    vi_build(db);
    if (vi_save(db, path) != 0) { fprintf(stderr, "%s\n", vi_error()); return 1; }
    for (id = CSAVED + 1; id <= CALL; ++id) {                    /* in the journal only */
        vec_(id, v);
        if (tx && id % 3 == 0) { text_(id, t); vi_add_text(db, id, v, t, -1); } else vi_add(db, id, v);
        vi_set_tag(db, id, tag_(id));
#ifdef NOW
        vi_set_num(db, id, num_(id));
        if (ns_(id)) vi_set_ns(db, id, ns_(id));
#endif
    }
    vi_delete(db, 12);
    vec_(1000, v); vi_add(db, 13, v);                            /* a row replaced: now 1000's vector */
    vi_close(db);                                                /* flushes the journal, does not rewrite the file */
    return 0;
}
