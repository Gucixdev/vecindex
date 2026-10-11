/* mbench.c - real numbers for mem.c, in particular the one honest gap its own README calls out:
 * mem_consolidate's (similar T)/promote/merge actions are O(nodes) vecdb_search calls per pass,
 * "not benchmarked at vecdb's own million-note scale" - this measures exactly how that scales,
 * plus ingest/search/touch/archive/recall/save/load throughput, so "O(nodes)" has actual numbers
 * next to it instead of just the shape of the curve.
 *   gcc -std=gnu99 -O2 -march=native -I. mbench.c -lm -lpthread -o .build/mbench && .build/mbench [N]
 * N (default 20000) sizes the ingest/search/save sections; the consolidate-scaling section always
 * runs its own fixed sweep (1000/4000/16000/64000) since that is the point of this file. */
#include "mem.c"
#include <stdio.h>
#include <time.h>

static double now_(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec; }

static void fill_(Mem *m, i32 n, i32 topics, u32 *r, i32 *ids)
{
    i32 i;
    for (i = 0; i < n; ++i) {
        char t[64], x[256];
        i32 topic, w1, w2, w3;
        *r = *r * 1664525u + 1013904223u; topic = (i32)(*r % (u32)topics);
        *r = *r * 1664525u + 1013904223u; w1 = (i32)(*r % 40);
        *r = *r * 1664525u + 1013904223u; w2 = (i32)(*r % 40);
        *r = *r * 1664525u + 1013904223u; w3 = (i32)(*r % 40);
        sprintf(t, "note %d", i);
        sprintf(x, "tw%d_%d tw%d_%d tw%d_%d note body %d %s", topic, w1, topic, w2, topic, w3, i,
                i > 0 ? "and a link" : "");
        if (i > 0) { char link[48]; sprintf(link, " [[note %d]]", (i32)((*r = *r * 1664525u + 1013904223u) % (u32)i)); strcat(x, link); }
        ids[i] = mem_note_event(m, t, x);
        if (i > 0 && i % 3 == 0) { i32 other = ids[(i32)((*r = *r * 1664525u + 1013904223u) % (u32)i)]; if (other >= 0) mem_corroborate(m, other, ids[i]); }
    }
}

int main(int argc, char **argv)
{
    static Arena arena;
    i32 N = argc > 1 ? atoi(argv[1]) : 20000, i;
    u32 r = 12345u;
    double t0;
    VecDB *db;
    Mem *m;
    i32 *ids;

    printf("mbench: mem.c on top of vecdb (hashed embedder, dim=128)\n\n");

    /* ---- ingest ---- */
    db = vecdb_new(&arena, 128);
    m = mem_open(db);
    ids = (i32 *)malloc((size_t)N * sizeof(i32));
    t0 = now_();
    fill_(m, N, N / 200 > 8 ? N / 200 : 8, &r, ids);
    printf("ingest %d events (mem_note_event + ~1/3 mem_corroborate): %.2f s (%.1f us/note)\n", N, now_() - t0, (now_() - t0) * 1e6 / N);
    t0 = now_(); vecdb_build(db); printf("vecdb_build: %.2f s\n\n", now_() - t0);

    /* ---- search / touch ---- */
    { i32 nq = 300; VHit hits[8];
      t0 = now_();
      for (i = 0; i < nq; ++i) { char q[64]; i32 topic = (i32)(r % 8); r = r * 1664525u + 1013904223u; sprintf(q, "tw%d_%u note body", topic, r % 40); mem_search(m, q, 5, hits); }
      printf("mem_search (vecdb_search + touch every hit): %.3f ms/query\n", (now_() - t0) * 1e3 / nq); }
    { i32 nq = 2000;
      t0 = now_();
      for (i = 0; i < nq; ++i) mem_touch(m, ids[(i32)((r = r * 1664525u + 1013904223u) % (u32)N)]);
      printf("mem_touch alone: %.3f us/call\n", (now_() - t0) * 1e6 / nq); }
    { i32 nq = 2000; f32 acc = 0.0f;
      t0 = now_();
      for (i = 0; i < nq; ++i) acc += mem_relevance(m, ids[(i32)((r = r * 1664525u + 1013904223u) % (u32)N)]);
      printf("mem_relevance: %.3f us/call\n", (now_() - t0) * 1e6 / nq);
      t0 = now_();
      for (i = 0; i < nq; ++i) acc += mem_confidence(m, ids[(i32)((r = r * 1664525u + 1013904223u) % (u32)N)]);
      printf("mem_confidence: %.3f us/call  (sink %.3f)\n\n", (now_() - t0) * 1e6 / nq, (double)acc); }

    /* ---- mem_gaps / mem_known at this N ---- */
    { i32 out[16]; t0 = now_(); mem_gaps(m, 0.9f, out, 16); printf("mem_gaps at N=%d: %.3f ms\n", N, (now_() - t0) * 1e3); }
    { f32 c; i32 vid; t0 = now_(); mem_known(m, "note body tw3", 0.5f, &c, &vid); printf("mem_known: %.3f ms\n\n", (now_() - t0) * 1e3); }

    /* ---- archive / recall round trip ---- */
    { i32 nn = 200, cid; t0 = now_();
      for (i = 0; i < nn; ++i) mem_cold_open(m, "/tmp/mbench.vdb");   /* only the last matters; measuring mem_archive itself below */
      mem_cold_open(m, "/tmp/mbench.vdb");
      t0 = now_();
      for (i = 0; i < nn; ++i) { cid = mem_archive(m, ids[N - 1 - i]); if (cid < 0) break; }
      printf("mem_archive (fsynced append + vecdb_remove_note): %.3f ms/call\n", (now_() - t0) * 1e3 / nn);
      t0 = now_();
      for (i = 0; i < nn; ++i) mem_recall(m, i);
      printf("mem_recall (linear cold-file scan to id %d avg): %.3f ms/call\n\n", nn / 2, (now_() - t0) * 1e3 / nn);
      remove("/tmp/mbench.vdb.mem.cold"); }

    /* ---- save / load at N ---- */
    t0 = now_(); vecdb_save(db, "/tmp/mbench.vdb"); printf("vecdb_save at N=%d: %.2f s\n", N, now_() - t0);
    t0 = now_(); mem_save(m, "/tmp/mbench.mem"); printf("mem_save at N=%d: %.3f s\n", N, now_() - t0);
    { static Arena a2; VecDB *db2 = vecdb_load(&a2, "/tmp/mbench.vdb"); Mem *m2 = mem_open(db2);
      t0 = now_(); mem_load(m2, "/tmp/mbench.mem"); printf("mem_load at N=%d: %.3f s (%d entries)\n\n", N, now_() - t0, m2->n);
      mem_free(m2); vecdb_free(db2); }
    remove("/tmp/mbench.vdb"); remove("/tmp/mbench.vdb.vdt"); remove("/tmp/mbench.vdb.vxi"); remove("/tmp/mbench.vdb.vxi.j");
    remove("/tmp/mbench.mem");
    mem_free(m); vecdb_free(db);

    /* ---- the actual point of this file: mem_consolidate cost vs N ---- */
    printf("mem_consolidate: cost per pass vs corpus size (the O(nodes) vecdb_search-per-node gap)\n");
    { i32 sizes[4] = { 1000, 4000, 16000, 64000 }, s;
      const char *rules = "(rule promote-facts (and (kind event) (similar 0.55)) (promote fact 0.55))\n"
                          "(rule promote-concepts (and (kind fact) (>= corroboration 1)) (promote concept 0.55))\n";
      for (s = 0; s < 4; ++s) {
          static Arena a3; VecDB *d3; Mem *m3; i32 *ids3, n3 = sizes[s], fired;
          u32 r3 = 777u;
          d3 = vecdb_new(&a3, 64); m3 = mem_open(d3);
          ids3 = (i32 *)malloc((size_t)n3 * sizeof(i32));
          fill_(m3, n3, n3 / 200 > 8 ? n3 / 200 : 8, &r3, ids3);
          vecdb_build(d3);
          t0 = now_();
          fired = mem_consolidate(m3, rules);
          printf("  N=%-6d  %8.3f s/pass  (%.1f us/node, %d actions fired)\n", n3, now_() - t0, (now_() - t0) * 1e6 / n3, fired);
          free(ids3); mem_free(m3); vecdb_free(d3);
      }

      /* ---- and the fix for that gap: mem_consolidate_hot, scoped to the (bounded) hot set instead
       * of every live node. Same rules, same N=64000 corpus size as the worst row above, so the two
       * numbers are directly comparable - this is the number that should replace a full sweep once
       * the corpus is past the low tens of thousands (see README). ---- */
      printf("\nmem_consolidate_hot: same rules, scoped to the hot set (compare to the N=64000 row above)\n");
      { static Arena a4; VecDB *d4; Mem *m4; i32 *ids4, n4 = 64000, fired4;
        u32 r4 = 777u;
        d4 = vecdb_new(&a4, 64); m4 = mem_open(d4);
        ids4 = (i32 *)malloc((size_t)n4 * sizeof(i32));
        fill_(m4, n4, n4 / 200 > 8 ? n4 / 200 : 8, &r4, ids4);
        vecdb_build(d4);
        t0 = now_();
        fired4 = mem_consolidate_hot(m4, rules);
        printf("  N=%-6d  hot-only   %8.3f s/pass  (%d actions fired, hot set bounded at %d)\n",
               n4, now_() - t0, fired4, m4->hot_cap);
        free(ids4); mem_free(m4); vecdb_free(d4);
      }
    }
    free(ids);
    return 0;
}
