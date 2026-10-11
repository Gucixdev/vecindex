/* vbench_wiki.c - vecdb vs plain vecindex on a real Wikipedia dump (see wikiprep.py).
 *   gcc -std=gnu99 -O2 -march=native -I. vbench_wiki.c -lm -lpthread -o .build/vbench_wiki
 *   .build/vbench_wiki wiki.rec [smooth_alpha]
 * Tasks (queries are drawn from articles, relevant sets from the REAL link graph):
 *   T  first 12 words of the article body -> is the article in the top 10?
 *   R  the article's title -> recall@10 of the articles it links to or that link to it
 *   K  a title-only query -> is the exact article first (title lookup)? */
#include "vecdb.c"
#include <stdio.h>
#include <time.h>

static double now_(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec; }

int main(int argc, char **argv)
{
    static Arena arena;
    const char *es = getenv("VECDB_EMBED_SERVER");
    VEmbedPool *ep = es ? vecdb_embpool_open(es) : (VEmbedPool *)0;
    VecDB *db = vecdb_new(&arena, ep ? ep->dim : 256);
    i32 maxn = argc > 3 ? atoi(argv[3]) : 1 << 30;
    FILE *f;
    char *title = (char *)malloc(4096), *buf = (char *)malloc(1 << 20), line[64];
    i32 N = 0, i, q, NQ = 300;
    i32 hM[3] = {0, 0, 0}, hT[3] = {0, 0, 0}, hK[3] = {0, 0, 0}, nR = 0;
    double rR[3] = {0, 0, 0}, tt[3] = {0, 0, 0}, t0, ts;
    u32 r = 99u;
    f32 alpha = argc > 2 ? (f32)atof(argv[2]) : 0.0f;
    VHit hits[16];
    char *qt[1024], *qx[1024];
    i32 nq = 0, nM = 0;
    i32 *ids;

    if (argc < 2 || !(f = fopen(argv[1], "rb"))) { fprintf(stderr, "usage: vbench_wiki wiki.rec [alpha]\n"); return 2; }
    if (ep) { vecdb_use_embpool(db, ep); if (getenv("VBENCH_CACHE")) vecdb_embed_cache_open(db, getenv("VBENCH_CACHE"), es); }
    t0 = now_();
    while (N < maxn && fgets(title, 4096, f) && fgets(line, sizeof line, f)) {
        i32 tl = (i32)strlen(title), len = atoi(line);
        if (tl > 0 && title[tl - 1] == '\n') title[--tl] = 0;
        if (len <= 0 || len >= (1 << 20) || fread(buf, 1, (size_t)len, f) != (size_t)len) break;
        buf[len] = 0; fgetc(f);
        qt[nq] = (char *)malloc(strlen(title) + 1); strcpy(qt[nq], title);
        qx[nq] = (char *)malloc((size_t)len + 1); memcpy(qx[nq], buf, (size_t)len + 1);
        if (++nq == 1024) { vecdb_add_notes(db, nq, (const char **)qt, (const char **)qx); for (i = 0; i < nq; ++i) { free(qt[i]); free(qx[i]); } nq = 0; }
        ++N;
        if (N % 2000 == 0) { fprintf(stderr, "  %d articles, %.0f s\n", N, now_() - t0); }
    }
    fclose(f);
    if (nq) { vecdb_add_notes(db, nq, (const char **)qt, (const char **)qx); for (i = 0; i < nq; ++i) { free(qt[i]); free(qx[i]); } }
    printf("embedder: %s\n", ep ? es : "hashed bag-of-words");
    printf("%d articles, %d ghost links, %d tags   ingest %.1f s\n", vecdb_count(db, VN_NOTE), vecdb_count(db, VN_GHOST), vecdb_count(db, VN_TAG), now_() - t0);
    if (alpha > 0.0f) { t0 = now_(); vecdb_smooth(db, alpha); printf("graph-smoothed embeddings (alpha %.2f): %.1f s\n", alpha, now_() - t0); }
    t0 = now_(); vecdb_build(db); printf("vector index build: %.1f s\n", now_() - t0);

    if (getenv("VBENCH_SAVE")) { t0 = now_(); vecdb_save(db, getenv("VBENCH_SAVE")); printf("saved to %s: %.1f s\n", getenv("VBENCH_SAVE"), now_() - t0); fflush(stdout); }
    ids = (i32 *)malloc((size_t)db->nn * sizeof(i32)); N = 0;
    for (i = 0; i < db->nn; ++i) if (db->nodes[i].kind == VN_NOTE && db->nodes[i].out.n + db->nodes[i].in.n >= 4) ids[N++] = i;
    printf("%d articles with >= 4 links are query candidates\n", N);
    for (q = 0; q < NQ; ++q) {
        i32 a = ids[vdb_rng_(&r) % (u32)N], m, k, w;
        char *txb = vecdb_text(db, a); const char *tx = txb ? txb : ""; char qt[512], *p = qt; i32 words = 0;
        VHit hv[16]; i64 rid[16]; f32 rsc[16], *qv = (f32 *)malloc((size_t)db->dim * sizeof(f32));
        /* T: first 12 words of the body */
        while (*tx && words < 12) { while (*tx == ' ' || *tx == '\n') ++tx; while (*tx && *tx != ' ' && *tx != '\n' && p - qt < 500) { if (*tx != '[' && *tx != ']') *p++ = *tx; ++tx; } *p++ = ' '; ++words; }
        *p = 0; free(txb);
        for (m = 0; m < 3; ++m) {                                    /* 0 vecindex, 1 hybrid, 2 MCTS */
            ts = now_();
            if (m == 0) { vdb_embed_(db, qt, (i32)strlen(qt), 1, qv); w = store_search(db->store, qv, 10, rid, rsc); for (k = 0; k < w; ++k) hv[k].node = (i32)rid[k]; }
            else if (m == 1) w = vecdb_search(db, qt, 10, 0, -1, hv);
            else w = vecdb_explore(db, qt, 10, (const VExplore *)0, hv);
            tt[m] += now_() - ts;
            for (k = 0; k < w; ++k) if (hv[k].node == a) { ++hT[m]; break; }
        }
        /* M: 12 words from deep in the body (beyond what a 128-token embedding saw) */
        { char *tb2 = vecdb_text(db, a); i32 tl2 = tb2 ? (i32)strlen(tb2) : 0;
          if (tl2 > 1800) {
              const char *z = tb2 + 1400; char qm[512], *pm = qm; i32 wd = 0;
              while (*z && *z != ' ') ++z;
              while (*z && wd < 12) { while (*z == ' ' || *z == '\n') ++z; while (*z && *z != ' ' && *z != '\n' && pm - qm < 500) { if (*z != '[' && *z != ']') *pm++ = *z; ++z; } *pm++ = ' '; ++wd; }
              *pm = 0; ++nM;
              for (m = 0; m < 3; ++m) {
                  i32 w2, k2;
                  if (m == 0) { vdb_embed_(db, qm, (i32)strlen(qm), 1, qv); w2 = store_search(db->store, qv, 10, rid, rsc); for (k2 = 0; k2 < w2; ++k2) hv[k2].node = (i32)rid[k2]; }
                  else if (m == 1) w2 = vecdb_search(db, qm, 10, 0, -1, hv);
                  else w2 = vecdb_explore(db, qm, 10, (const VExplore *)0, hv);
                  for (k2 = 0; k2 < w2; ++k2) if (hv[k2].node == a) { ++hM[m]; break; }
              }
          }
          free(tb2); }
        /* K: exact title */
        for (m = 0; m < 3; ++m) {
            const char *tq = db->nodes[a].name; i32 w2, k2;
            if (m == 0) { vdb_embed_(db, tq, (i32)strlen(tq), 1, qv); w2 = store_search(db->store, qv, 1, rid, rsc); if (w2 > 0 && (i32)rid[0] == a) ++hK[m]; }
            else { w2 = m == 1 ? vecdb_search(db, tq, 1, 0, -1, hv) : vecdb_explore(db, tq, 1, (const VExplore *)0, hv); (void)k2; if (w2 > 0 && hv[0].node == a) ++hK[m]; }
        }
        /* R: related = real neighbours (links + backlinks that are articles) */
        { i32 rel[4096], nrel = 0, got, j; const VNode *v = &db->nodes[a];
          for (j = 0; j < v->out.n + v->in.n && nrel < 4096; ++j) { i32 g = vdb_nb_(db, a, j); if (db->nodes[g].kind == VN_NOTE) rel[nrel++] = g; }
          if (nrel >= 3) {
              ++nR;
              for (m = 0; m < 3; ++m) {
                  const char *tq = db->nodes[a].name; i32 w3;
                  if (m == 0) { vdb_embed_(db, tq, (i32)strlen(tq), 1, qv); w3 = store_search(db->store, qv, 11, rid, rsc); for (k = 0; k < w3; ++k) hv[k].node = (i32)rid[k]; }
                  else if (m == 1) w3 = vecdb_search(db, tq, 11, 0, -1, hv);
                  else w3 = vecdb_explore(db, tq, 11, (const VExplore *)0, hv);
                  got = 0;
                  for (k = 0; k < w3; ++k) { if (hv[k].node == a) continue; for (j = 0; j < nrel; ++j) if (rel[j] == hv[k].node) { ++got; break; } }
                  rR[m] += (double)got / (nrel < 10 ? nrel : 10);
              }
          } }
        free(qv);
    }
    { static const char *nm[3] = { "vecindex (vector only)", "vecdb hybrid", "vecdb MCTS" }; i32 m;
      printf("\n%-24s %14s %14s %16s %10s\n", "", "T body->article", "K title->exact", "R related recall", "ms/query");
      for (m = 0; m < 3; ++m)
          printf("%-24s %13.1f%% %13.1f%% %15.1f%% %10.2f\n", nm[m], 100.0 * hT[m] / NQ, 100.0 * hK[m] / NQ, nR ? 100.0 * rR[m] / nR : 0.0, tt[m] * 1e3 / NQ);
      printf("M mid-body words -> article (top 10): vecindex %.1f%%  hybrid %.1f%%  MCTS %.1f%%  over %d queries\n", nM ? 100.0 * hM[0] / nM : 0.0, nM ? 100.0 * hM[1] / nM : 0.0, nM ? 100.0 * hM[2] / nM : 0.0, nM);
      printf("(T,K over %d queries, R over %d articles with >=3 linked neighbours; R = share of up to 10 real neighbours returned in the top 10)\n", NQ, nR); }
    vecdb_embpool_close(ep);
    return 0;
}
