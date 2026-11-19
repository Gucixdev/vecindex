/* graph_test.c - include/graph.h, src/graph.c: links, their lists and counts, the errors; vg_expand level by level;
 * vg_ppr against personalized PageRank by power iteration; the file - reopened, a torn tail cut, a damaged record
 * ending the log, a compaction; and a walk from a store's hits to the chunks they link to. ./run builds it against
 * libvecindex.a. */
#include "vecindex.h"
#include "graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <sys/stat.h>

static int fails;

static void check(int cond, const char *what)
{
    printf(cond ? "  ok   %s\n" : "  FAIL %s\n", what);
    if (!cond) { ++fails; printf("       (%s)\n", vg_error()); }
}

static int has_(const vi_i64 *ids, int n, vi_i64 id) { int i; for (i = 0; i < n; ++i) if (ids[i] == id) return i + 1; return 0; }
static long fsize_(const char *p) { struct stat st; return stat(p, &st) == 0 ? (long)st.st_size : -1L; }

#define CN 10                        /* two groups of CN ids, every pair inside a group linked, one bridge */

/* personalized PageRank by power iteration over the same walk (both ways, by weight): p = alpha sum (1 - alpha)^t s P^t */
static void ppr_exact_(double w[2 * CN][2 * CN], int seed, double alpha, double *p)
{
    double r[2 * CN], nr[2 * CN], W[2 * CN];
    int i, j, t;
    for (i = 0; i < 2 * CN; ++i) { W[i] = 0.0; for (j = 0; j < 2 * CN; ++j) W[i] += w[i][j]; r[i] = i == seed; p[i] = 0.0; }
    for (t = 0; t < 2000; ++t) {
        for (i = 0; i < 2 * CN; ++i) { p[i] += alpha * r[i]; nr[i] = 0.0; }
        for (i = 0; i < 2 * CN; ++i) for (j = 0; j < 2 * CN; ++j) if (w[i][j] > 0.0) nr[j] += (1.0 - alpha) * r[i] * w[i][j] / W[i];
        memcpy(r, nr, sizeof(r));
    }
}

int main(void)
{
    vg_graph *g;
    vg_expand_opts o;
    vg_ppr_opts po;
    vi_i64 ids[64], nn, nl, seed;
    float sc[64], wt[8];
    int dep[64], ty[8], n, i, j, ok;
    char path[128], line[256];
    static double w[2 * CN][2 * CN];

    printf("graph\n");
    g = vg_open((const char *)0);
    check(g != (vg_graph *)0, "vg_open in memory");
    if (!g) return 1;
    check(vg_link(g, 1, 2, 0, 1.0f) == 0 && vg_link(g, 1, 2, 0, 0.5f) == 0, "vg_link, then the same link again");
    n = vg_links(g, 1, VG_OUT, 0, ids, ty, wt, 8);
    check(n == 1 && ids[0] == 2 && wt[0] == 0.5f, "one link of that from, to and type: the second set its weight");
    vg_link(g, 1, 2, 3, 1.0f);
    vg_link(g, 3, 1, 0, 1.0f);
    check(vg_links(g, 1, VG_OUT, 0, ids, ty, wt, 8) == 2 && vg_links(g, 1, VG_IN, 0, ids, ty, wt, 8) == 1 && ids[0] == 3, "out and in");
    check(vg_links(g, 1, VG_BOTH, 1u << 3, ids, ty, wt, 8) == 1 && ty[0] == 3, "a mask of types");
    check(vg_links(g, 1, VG_BOTH, 0, ids, ty, wt, 1) == 3, "more than max: the count of all, max written");
    check(vg_link(g, 5, 5, 0, 1.0f) == -1 && vg_link(g, 5, 6, 32, 1.0f) == -1 && vg_link(g, 5, 6, 0, 0.0f) == -1, "refused: to itself, type 32, weight 0");
    {   volatile float z = 0.0f; check(vg_link(g, 5, 6, 0, z / z) == -1, "refused: a NaN weight"); }
    check(vg_unlink(g, 1, 2, 3) == 1 && vg_links(g, 1, VG_OUT, 0, ids, ty, wt, 8) == 1, "vg_unlink of one type");
    check(vg_drop(g, 1) == 2 && vg_links(g, 1, VG_BOTH, 0, ids, ty, wt, 8) == 0, "vg_drop: every link from or to it");
    vg_count(g, &nn, &nl);
    check(nn == 0 && nl == 0, "vg_count after: no ids with a link, no links");
    vg_close(g);

    /* vg_expand */
    g = vg_open((const char *)0);
    vg_link(g, 1, 2, 0, 1.0f); vg_link(g, 2, 3, 0, 1.0f); vg_link(g, 3, 4, 0, 1.0f); vg_link(g, 1, 10, 1, 1.0f);
    seed = 1;
    vg_expand_defaults(&o);
    n = vg_expand(g, &seed, (const float *)0, 1, &o, ids, sc, dep, 64);
    check(n == 4 && ids[0] == 1 && sc[0] == 1.0f && has_(ids, n, 3) && sc[has_(ids, n, 3) - 1] == 0.25f && !has_(ids, n, 4),
          "two levels from 1: itself, 2 and 10 at 0.5, 3 at 0.25, not 4");
    o.types = 1u << 0;
    n = vg_expand(g, &seed, (const float *)0, 1, &o, ids, sc, dep, 64);
    check(n == 3 && !has_(ids, n, 10), "types: only type 0 links");
    seed = 3; o.types = 0; o.dir = VG_IN;
    n = vg_expand(g, &seed, (const float *)0, 1, &o, ids, sc, dep, 64);
    check(n == 3 && has_(ids, n, 2) && has_(ids, n, 1) && !has_(ids, n, 4) && dep[has_(ids, n, 1) - 1] == 2, "VG_IN: against the links, 1 at depth 2");
    o.dir = VG_BOTH; o.max_nodes = 2;
    n = vg_expand(g, &seed, (const float *)0, 1, &o, ids, sc, dep, 64);
    check(n == 2, "max_nodes 2: the seed and one more");
    seed = 777; vg_expand_defaults(&o);
    n = vg_expand(g, &seed, (const float *)0, 1, &o, ids, sc, dep, 64);
    check(n == 1 && ids[0] == 777 && dep[0] == 0, "a seed with no links: itself, so it is one list");
    vg_close(g);

    /* vg_ppr against power iteration: two groups, a bridge */
    g = vg_open((const char *)0);
    for (i = 0; i < 2 * CN; ++i) for (j = 0; j < 2 * CN; ++j) w[i][j] = 0.0;
    for (i = 0; i < 2 * CN; ++i)
        for (j = i + 1; j < 2 * CN; ++j)
            if (i / CN == j / CN) { float x = 0.5f + 0.5f * (float)((i * 7 + j * 3) % 5) / 4.0f; vg_link(g, 100 + i, 100 + j, 0, x); w[i][j] = w[j][i] = x; }
    vg_link(g, 100 + CN - 1, 100 + CN, 1, 0.25f); w[CN - 1][CN] = w[CN][CN - 1] = 0.25;
    {   double p[2 * CN], err = 0.0, sum = 0.0;
        vg_ppr_defaults(&po);
        po.eps = 1e-8f;
        seed = 100;
        n = vg_ppr(g, &seed, (const float *)0, 1, &po, ids, sc, 64);
        ppr_exact_(w, 0, 0.15, p);
        for (i = 0; i < n; ++i) { err = fabs(sc[i] - p[ids[i] - 100]) > err ? fabs(sc[i] - p[ids[i] - 100]) : err; sum += sc[i]; }
        sprintf(line, "vg_ppr against power iteration: %d nodes, the largest error %.2e, the mass %.6f", n, err, sum);
        check(n == 2 * CN && err < 1e-4 && sum > 0.999 && sum <= 1.0 + 1e-6, line);
        for (i = 0, ok = 1; i < CN; ++i) ok &= ids[i] < 100 + CN;
        check(ok, "the seed's group first: all ten of it before any of the other");
        po.types = 1u << 0;
        n = vg_ppr(g, &seed, (const float *)0, 1, &po, ids, sc, 64);
        check(n == CN, "types: without the bridge's type the walk stays in its group");
        vg_ppr_defaults(&po); po.max_nodes = 5;
        n = vg_ppr(g, &seed, (const float *)0, 1, &po, ids, sc, 64);
        for (i = 0, sum = 0.0; i < n; ++i) sum += sc[i];
        check(n <= 5 && sum < 1.0, "max_nodes 5: at most five, the mass past them dropped");
        seed = 999;
        n = vg_ppr(g, &seed, (const float *)0, 1, &po, ids, sc, 64);
        check(n == 1 && ids[0] == 999 && sc[0] > 0.999f, "a seed with no links keeps its mass");
    }
    vg_close(g);

    /* the file */
    sprintf(path, "/tmp/vecindex_graph_%d.vxg", (int)getpid());
    remove(path);
    g = vg_open(path);
    check(g != (vg_graph *)0 && fsize_(path) == 16, "vg_open makes the file: its header");
    for (i = 0; i < 1000; ++i) vg_link(g, i, i + 1, i % 4, 1.0f);
    for (i = 0; i < 900; ++i) vg_unlink(g, i, i + 1, -1);
    vg_close(g);
    g = vg_open(path);
    vg_count(g, &nn, &nl);
    check(nl == 100 && nn == 101, "reopened: the links the log leaves, 100");
    vg_close(g);
    {   FILE *f = fopen(path, "ab"); fwrite("torn-in-the-mid", 1, 13, f); fclose(f); }
    g = vg_open(path);
    vg_count(g, &nn, &nl);
    check(g && nl == 100, "a torn tail (13 bytes): cut, the links before it kept");
    vg_link(g, 5000, 5001, 0, 1.0f);
    vg_close(g);
    g = vg_open(path);
    vg_count(g, &nn, &nl);
    check(nl == 101 && vg_links(g, 5000, VG_OUT, 0, ids, ty, wt, 8) == 1, "a link after the cut: there on the next open");
    check(vg_compact(g) == 0 && fsize_(path) == 16 + 32 * 101, "vg_compact: the file holds the 101 live links only");
    vg_close(g);
    g = vg_open(path);
    vg_count(g, &nn, &nl);
    check(nl == 101 && vg_links(g, 950, VG_OUT, 0, ids, ty, wt, 8) == 1 && ids[0] == 951, "reopened after it: the same links");
    vg_close(g);
    {   FILE *f = fopen(path, "r+b"); fseek(f, 16 + 32 * 2 + 9, SEEK_SET); fputc(0x5a, f); fclose(f); }   /* record 3 damaged */
    g = vg_open(path);
    vg_count(g, &nn, &nl);
    check(g && nl == 2, "a damaged record: the log ends before it (two links)");
    vg_close(g);
    remove(path);

    /* from a store's hits along the links */
    {   vi_db *db = vi_open((const char *)0, 4, 0);
        float v[5][4] = { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 }, { 1, 1, 0, 0 } };
        const char *tx[5] = { "alloc.c: the arena", "free lists", "mmap", "the page cache", "pools" };
        char b[64];
        vi_i64 hit;
        int len;
        g = vg_open((const char *)0);
        for (i = 0; i < 5; ++i) vi_add_text(db, i, v[i], tx[i], -1);
        vg_link(g, 0, 3, 1, 1.0f);                               /* the arena references the page cache */
        n = vi_search(db, v[0], 1, &hit, sc);
        n = vg_expand(g, &hit, (const float *)0, n, (const vg_expand_opts *)0, ids, sc, dep, 8);
        len = n == 2 ? vi_get_text(db, ids[1], b, (int)sizeof b) : -1;
        check(n == 2 && ids[0] == 0 && ids[1] == 3 && len == 14 && memcmp(b, "the page cache", 14) == 0,
              "vi_search's hit, then vg_expand: the chunk it references, its text from the store");
        vg_close(g); vi_close(db); }

    printf("graph: %s\n", fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}
