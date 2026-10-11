/* vbench.c - vecdb vs plain vecindex on a large synthetic note set.
 *   gcc -std=gnu99 -O2 -march=native -I. vbench.c -lm -lpthread -o .build/vbench && .build/vbench [N]
 * Notes come from T topics; each has topic words + Zipf noise words + one unique
 * token; 3 wikilinks inside its topic; a #topic tag. Queries:
 *   A  unique token of a random note        (exact-word precision: hit@1)
 *   B  4 topic words of a random topic      (topic precision@10)
 * "vecindex" = store_search on the very same embeddings. */
#include "vecdb.c"
#include <stdio.h>
#include <time.h>

static double now_(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec; }
static u32 r_ = 12345u;
static u32 rnd_(void) { return vdb_rng_(&r_); }

int main(int argc, char **argv)
{
    static Arena arena;
    i32 N = argc > 1 ? atoi(argv[1]) : 200000, T = N / 250 > 8 ? N / 250 : 8, i, q, dim = 256;
    i32 NQ = 300, W = 40, NOISE = 4000;
    VecDB *db = vecdb_new(&arena, dim);
    i32 *nid = (i32 *)malloc((size_t)N * sizeof(i32)), *topic = (i32 *)calloc((size_t)N + (size_t)T + 16, sizeof(i32)), *first = (i32 *)malloc((size_t)T * sizeof(i32));
    char *body = (char *)malloc(8192), title[64];
    double t0 = now_();
    i32 hitA_v = 0, hitA_h = 0, hitA_l = 0; double precB_v = 0, precB_h = 0, precB_e = 0;
    double tv = 0, th = 0, te = 0, ts;
    VHit hits[32]; i64 ids[32]; f32 sc[32], *qv = (f32 *)malloc((size_t)dim * sizeof(f32));

    for (i = 0; i < T; ++i) first[i] = -1;
    for (i = 0; i < N; ++i) {
        char *p = body; i32 k, t = (i32)(rnd_() % (u32)T);
        
        for (k = 0; k < 8; ++k) p += sprintf(p, "tw%d_%u ", t, rnd_() % (u32)W);
        for (k = 0; k < 17; ++k) { u32 z = rnd_() % (u32)NOISE; z = (z * z) / (u32)NOISE; p += sprintf(p, "nz%u ", z); }
        p += sprintf(p, "uid%d ", i);
        for (k = 0; k < 3; ++k) { i32 j = i > 0 ? (i32)(rnd_() % (u32)i) : 0; if (i > 0 && topic[nid[j]] == t) p += sprintf(p, "[[Note %d]] ", j); }
        sprintf(p, "#t%d", t);
        sprintf(title, "Note %d", i);
        nid[i] = vecdb_add_note(db, title, body); topic[nid[i]] = t;
        if (first[t] < 0) first[t] = i;
    }
    printf("N=%d notes, %d topics, dim %d\n", N, T, dim);
    printf("ingest (embed+parse+graph+BM25+store): %.2f s  (%.1f us/note)\n", now_() - t0, (now_() - t0) * 1e6 / N);
    t0 = now_(); vecdb_build(db); printf("build vector index: %.2f s\n", now_() - t0);

    for (q = 0; q < NQ; ++q) {                                       /* A: exact token */
        i32 tj = (i32)(rnd_() % (u32)N), target = nid[tj]; char query[64]; i32 n;
        sprintf(query, "uid%d", tj);
        vecdb_embed(query, (i32)strlen(query), qv, dim);
        ts = now_(); n = store_search(db->store, qv, 10, ids, sc); tv += now_() - ts;
        if (n > 0 && (i32)ids[0] == target) ++hitA_v;
        ts = now_(); n = vecdb_search(db, query, 10, 0, -1, hits); th += now_() - ts;
        if (n > 0 && hits[0].node == target) ++hitA_h;
        n = vecdb_search(db, query, 10, 2, -1, hits);
        if (n > 0 && hits[0].node == target) ++hitA_l;
    }
    printf("\nA  exact token query, hit@1 over %d queries\n   vecindex (vector only) %5.1f%%   vecdb hybrid %5.1f%%   (lexical alone %5.1f%%)\n",
           NQ, 100.0 * hitA_v / NQ, 100.0 * hitA_h / NQ, 100.0 * hitA_l / NQ);
    printf("   latency: vecindex %.3f ms   vecdb hybrid %.3f ms\n", tv * 1e3 / NQ, th * 1e3 / NQ);
    tv = th = 0;
    for (q = 0; q < NQ; ++q) {                                       /* B: topic */
        i32 t = (i32)(rnd_() % (u32)T), k, n, good; char query[256], *p = query;
        for (k = 0; k < 4; ++k) p += sprintf(p, "tw%d_%u ", t, rnd_() % (u32)W);
        vecdb_embed(query, (i32)strlen(query), qv, dim);
        ts = now_(); n = store_search(db->store, qv, 10, ids, sc); tv += now_() - ts;
        for (good = 0, k = 0; k < n; ++k) if (topic[(i32)ids[k]] == t) ++good;
        precB_v += n ? (double)good / 10 : 0;
        ts = now_(); n = vecdb_search(db, query, 10, 0, -1, hits); th += now_() - ts;
        for (good = 0, k = 0; k < n; ++k) if (topic[hits[k].node] == t) ++good;
        precB_h += (double)good / 10;
        ts = now_(); n = vecdb_explore(db, query, 10, (const VExplore *)0, hits); te += now_() - ts;
        for (good = 0, k = 0; k < n; ++k) if (topic[hits[k].node] == t) ++good;
        precB_e += (double)good / 10;
    }
    printf("\nB  topic query (4 words), precision@10 over %d queries\n   vecindex %5.1f%%   vecdb hybrid %5.1f%%   vecdb MCTS %5.1f%%\n",
           NQ, 100.0 * precB_v / NQ, 100.0 * precB_h / NQ, 100.0 * precB_e / NQ);
    printf("   latency: vecindex %.3f ms   hybrid %.3f ms   MCTS %.3f ms\n", tv * 1e3 / NQ, th * 1e3 / NQ, te * 1e3 / NQ);
    { VecDB *d2; VHit h2[16]; i32 n1, n2, same = 1, j;
      const char *qq = "tw3_1 tw3_2 tw3_3";
      t0 = now_(); vecdb_save(db, "/tmp/vbench.vdb"); printf("\nsave (notes + graph + vector index): %.2f s\n", now_() - t0);
      t0 = now_(); d2 = vecdb_load(&arena, "/tmp/vbench.vdb"); vecdb_build(d2); printf("load + ready to query: %.2f s (no reindex)\n", now_() - t0);
      n1 = vecdb_search(db, qq, 10, 0, -1, hits); n2 = vecdb_search(d2, qq, 10, 0, -1, h2);
      for (i = 0; i < n1 && i < n2; ++i) { i32 f = 0; for (j = 0; j < n2; ++j) if (h2[j].node == hits[i].node) f = 1; if (!f) same = 0; }
      printf("same results after reload: %s\n", same && n1 == n2 ? "yes" : "NO");
      remove("/tmp/vbench.vdb"); remove("/tmp/vbench.vdb.vxi"); }
    return 0;
}
