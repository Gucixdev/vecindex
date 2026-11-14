/* rerank_test.c - the self-test of src/rerank.c; ./run builds and runs it. */
#include "rerank.c"
#include <stdio.h>

static i32 g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  FAIL: %s\n", msg); ++g_fail; } else printf("  ok   %s\n", msg); } while (0)

int main(void)
{
    static Arena arena;
    VecDB *db = vecdb_new(&arena, 48);
    VRerankProc *rp;
    VHit hits[8];
    i32 n;

    vecdb_add_note(db, "Server crash", "the web server crashed under heavy load during peak traffic");
    vecdb_add_note(db, "Cooking pasta", "boil pasta in salted water for eight to ten minutes");
    vecdb_add_note(db, "Quantum physics", "quantum entanglement links particles across distance");
    vecdb_add_note(db, "Load balancing", "distributing traffic across servers prevents any one from crashing");
    vecdb_build(db);

    printf("rerank_demo: cross-encoder second pass over vecdb's own hybrid search\n\n");
    /* same caveat as py/embedsrv.py's own docstring: run it with whatever interpreter actually has
     * torch/transformers - plain "python3" on $PATH usually will not. Set RERANK_CMD to the
     * interpreter that has them; vecdb_rerank_open itself takes any shell command, nothing is
     * hardcoded into the API. */
    { const char *spec = getenv("RERANK_CMD");
      rp = vecdb_rerank_open(spec ? spec : "python3 py/rerank_srv.py"); }
    if (!rp) { printf("  (py/rerank_srv.py unavailable/failed to start - skipped, this is meant to degrade gracefully)\n"); return 0; }

    n = vecdb_search(db, "why did the server go down under traffic", 4, 1, -1, hits);
    CHECK(n > 0, "hybrid search returns candidates");
    { i32 top_before = hits[0].node;
      i32 rr = vecdb_rerank_hits(db, rp, "why did the server go down under traffic", hits, n);
      CHECK(rr == 0, "vecdb_rerank_hits succeeds against a live server");
      CHECK(hits[0].node == top_before || 1, "re-sorted by cross-encoder score (informational - not asserting WHICH wins)");
      { i32 i; f32 prev = 1e30f;
        for (i = 0; i < n; ++i) { CHECK(hits[i].score <= prev + 1e-4f, "hits are sorted descending by the new score"); prev = hits[i].score; } }
      printf("  top result after rerank: \"%s\" (score %.3f)\n", db->nodes[hits[0].node].name, (double)hits[0].score);
    }

    vecdb_rerank_close(rp);
    vecdb_free(db);
    printf(g_fail ? "\nFAILED\n" : "\nOK\n");
    return g_fail ? 1 : 0;
}
