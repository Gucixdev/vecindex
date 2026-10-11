/* vbench1m.c - find the real ceiling on this machine: N notes with graph+tags+BM25+vectors, ingest, build,
 * query, save/load, all through the public vecdb API (hashed embedder, no model server - the point here is
 * the C-side memory/time ceiling, not embedding quality, which vbench_wiki.c already measured).
 *   gcc -std=gnu99 -O2 -march=native -I. vbench1m.c -lm -lpthread -o .build/vbench1m && .build/vbench1m N [SAVEDIR]
 */
#include "vecdb.c"
#include <sys/statvfs.h>
#include <stdio.h>
#include <time.h>
#include <sys/resource.h>

static double now_(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec; }

static long rss_mb_(void)
{
    FILE *f = fopen("/proc/self/status", "r");
    char line[256]; long kb = 0;
    if (!f) return -1;
    while (fgets(line, sizeof line, f)) if (!strncmp(line, "VmRSS:", 6)) { sscanf(line + 6, "%ld", &kb); break; }
    fclose(f);
    return kb / 1024;
}

static void mark_(const char *what, double t0)
{
    printf("  [%6.1fs, RSS %4ld MB] %s\n", now_() - t0, rss_mb_(), what);
    fflush(stdout);
}

int main(int argc, char **argv)
{
    static Arena arena;
    i32 N = argc > 1 ? atoi(argv[1]) : 1000000, T = N / 300 > 16 ? N / 300 : 16, i, dim = 128;
    const char *savedir = argc > 2 ? argv[2] : (const char *)0;
    VecDB *db = vecdb_new(&arena, dim);
    double t0 = now_();
    char title[64], body[1024];
    u32 r = 20260922u;
    VHit hits[16];
    i32 *ids = (i32 *)malloc((size_t)N * sizeof(i32));

    printf("N=%d notes, %d topics, dim=%d\n", N, T, dim);
    for (i = 0; i < N; ++i) {
        char *p = body; i32 k, topic = (i32)((r = r * 1664525u + 1013904223u) % (u32)T);   /* unsigned mod: no negative topic ids */
        p += sprintf(p, "tw%d_%u tw%d_%u tw%d_%u ", topic, (r = r * 1664525u + 1013904223u) % 40, topic, (r = r * 1664525u + 1013904223u) % 40, topic, (r = r * 1664525u + 1013904223u) % 40);
        for (k = 0; k < 10; ++k) { r = r * 1664525u + 1013904223u; p += sprintf(p, "nz%u ", (r >> 8) % 3000); }
        p += sprintf(p, "uid%d ", i);
        for (k = 0; k < 2 && i > 0; ++k) { r = r * 1664525u + 1013904223u; i32 j = (i32)(r % (u32)i); p += sprintf(p, "[[Note %d]] ", j); }
        sprintf(p, "#t%d", topic);
        sprintf(title, "Note %d", i);
        ids[i] = vecdb_add_note(db, title, body);
        if (i > 0 && i % 100000 == 0) { char m[64]; sprintf(m, "%d notes added", i); mark_(m, t0); }
    }
    mark_("ingest done", t0);
    t0 = now_();
    vecdb_build(db);
    mark_("vector index built", t0);

    { i32 hit = 0, nq = 300, q;
      t0 = now_();
      for (q = 0; q < nq; ++q) {
          char qs[64]; i32 target = (i32)((r = r * 1664525u + 1013904223u) % (u32)N), n;
          sprintf(qs, "uid%d", target);
          n = vecdb_search(db, qs, 10, 0, -1, hits);
          if (n > 0 && hits[0].node == ids[target]) ++hit;
      }
      printf("  hybrid search: %d/%d exact-id queries hit@1, %.3f ms/query\n", hit, nq, (now_() - t0) * 1e3 / nq);
    }
    { i32 nq = 100, q, sum = 0;
      t0 = now_();
      for (q = 0; q < nq; ++q) { i32 seed = (i32)((r = r * 1664525u + 1013904223u) % (u32)N); char qs[64]; sprintf(qs, "Note %d", seed); sum += vecdb_explore(db, qs, 10, (const VExplore *)0, hits); }
      printf("  MCTS explore: %.3f ms/query (avg %d hits)\n", (now_() - t0) * 1e3 / nq, sum / nq);
    }
    mark_("after search benchmarks", t0);

    if (savedir) {
        char path[512]; struct statvfs sv;
        sprintf(path, "%s/x.vdb", savedir);
        t0 = now_();
        if (vecdb_save(db, path) != 0) { printf("  save FAILED\n"); return 1; }
        mark_("saved", t0);
        if (statvfs(savedir, &sv) == 0) printf("  free space at %s after save: %.0f MB\n", savedir, (double)sv.f_bavail * (double)sv.f_frsize / 1e6);
        vecdb_free(db);
        t0 = now_();
        db = vecdb_load(&arena, path);
        if (!db) { printf("  load FAILED: %s\n", vecdb_error()); return 1; }
        mark_("loaded (no reindex expected)", t0);
        t0 = now_(); vecdb_build(db); mark_("build after load (should be near-instant)", t0);
        { i64 vsz = 0, tsz = 0, xsz = 0; char p2[512]; FILE *f;
          sprintf(p2, "%s.vdt", path); f = fopen(p2, "rb"); if (f) { fseek(f, 0, SEEK_END); tsz = ftell(f); fclose(f); }
          sprintf(p2, "%s.vxi", path); f = fopen(p2, "rb"); if (f) { fseek(f, 0, SEEK_END); vsz = ftell(f); fclose(f); }
          f = fopen(path, "rb"); if (f) { fseek(f, 0, SEEK_END); xsz = ftell(f); fclose(f); }
          printf("  file sizes: nodes %lld KB, bodies %lld KB, vector index %lld KB\n", (long long)(xsz / 1024), (long long)(tsz / 1024), (long long)(vsz / 1024));
        }
        remove(path); { char p2[512]; sprintf(p2, "%s.vdt", path); remove(p2); sprintf(p2, "%s.vxi", path); remove(p2); sprintf(p2, "%s.vxi.j", path); remove(p2); }
    }
    mark_("done", t0);
    return 0;
}
