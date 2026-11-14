/* rerank.c - an optional cross-encoder reranking pass over vecdb's own hybrid search results,
 * the same layering pattern mem.c uses one level up: #include "vecdb.c" and add on top, vecdb.c
 * itself untouched. Closes the gap docs/comparison.md names: hybrid
 * (vector+BM25 RRF) search is a real substitute for a lot of what reranking buys, but a
 * cross-encoder that actually reads (query, candidate) together is a further, different quality
 * pass most production RAG stacks add on top of ANN retrieval - this wires up a real one
 * (py/rerank_srv.py, any cross-encoder model) rather than leaving it as "we don't have one".
 *
 * Same child-process/pipe pattern as vecdb.c's own VEmbedProc (see vecdb_embproc_open) - a
 * separate, independent connection (rerank is a different model, scored differently: one score
 * per (query, candidate) PAIR, not one vector per text, so it needs its own protocol, not a
 * reuse of VEmbedProc's).
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -Werror -D_POSIX_C_SOURCE=200112L -O2 -march=native \
 *       -Isrc tests/rerank_test.c -lm -lpthread -o rerank_demo && ./rerank_demo
 */
#ifndef RERANK_C
#define RERANK_C
#include "vecdb.c"

typedef struct { pid_t pid; int to, from; } VRerankProc;

/* spawns `cmd` (a shell command line - typically "python3 py/rerank_srv.py [model]") via /bin/sh -c,
 * same as vecdb_embproc_open. Blocks until the child writes its one-byte "ready" ack (model
 * loaded) or the pipe closes. Returns NULL on any failure (pipe/fork/exec/handshake). */
static VRerankProc *vecdb_rerank_open(const char *cmd)
{
    int a[2], b[2];
    VRerankProc *p;
    u8 ready = 0;
    if (pipe(a) != 0) return (VRerankProc *)0;
    if (pipe(b) != 0) { close(a[0]); close(a[1]); return (VRerankProc *)0; }
    p = (VRerankProc *)calloc(1, sizeof(*p));
    p->pid = fork();
    if (p->pid < 0) { free(p); return (VRerankProc *)0; }
    if (p->pid == 0) {
        dup2(a[0], 0); dup2(b[1], 1);
        close(a[0]); close(a[1]); close(b[0]); close(b[1]);
        execl("/bin/sh", "sh", "-c", cmd, (char *)0);
        _exit(127);
    }
    close(a[0]); close(b[1]);
    p->to = a[1]; p->from = b[0];
    if (vdb_rd_(p->from, &ready, 1) != 0 || ready != 1) { close(p->to); close(p->from); kill(p->pid, SIGTERM); waitpid(p->pid, (int *)0, 0); free(p); return (VRerankProc *)0; }
    return p;
}

static void vecdb_rerank_close(VRerankProc *p)
{
    if (!p) return;
    close(p->to); close(p->from);
    waitpid(p->pid, (int *)0, 0);
    free(p);
}

static i32 vdb_write_str_(int fd, const char *s, i32 len)
{
    u32 l = (u32)len;
    size_t off;
    if (write(fd, &l, 4) != 4) return -1;
    for (off = 0; off < (size_t)len; ) { ssize_t w = write(fd, s + off, (size_t)len - off); if (w <= 0) return -1; off += (size_t)w; }
    return 0;
}

/* Scores `ncand` (candidate text, length) pairs against one query in a single round trip;
 * scores[i] gets cand[i]'s relevance (higher = more relevant - a raw cross-encoder logit, not
 * necessarily a probability; only meaningful as a RANKING signal, for comparing candidates
 * against each other, not as an absolute score against some other query's numbers). Returns 0 on
 * success, -1 if the server failed or is gone (scores left untouched - caller should fall back to
 * the pre-rerank order, this is always an optional refinement, never the only ranking). */
static i32 vecdb_rerank_score(VRerankProc *p, const char *query, i32 qlen, const char **cands, const i32 *clens, i32 ncand, f32 *scores)
{
    u32 n = (u32)ncand;
    i32 i;
    u8 st;
    if (vdb_write_str_(p->to, query, qlen) != 0) return -1;
    if (write(p->to, &n, 4) != 4) return -1;
    for (i = 0; i < ncand; ++i) if (vdb_write_str_(p->to, cands[i], clens[i]) != 0) return -1;
    if (vdb_rd_(p->from, &st, 1) != 0) return -1;
    if (st != 0) return -1;
    return vdb_rd_(p->from, scores, (size_t)ncand * sizeof(f32));
}

/* The actual point of this file: re-rank an existing hybrid search result set (from
 * vecdb_search/vecdb_query - anything returning VHit[]) with a cross-encoder second pass.
 * Fetches each hit's text (vecdb_text - the same body vecdb_search itself embedded from, so the
 * cross-encoder reads exactly what was matched, not just the title), scores (query, text) pairs,
 * overwrites hit->score with the cross-encoder's, and re-sorts descending by it - hit->sim/->lex
 * are left as they were (still inspectable: "what hybrid thought" vs "what rerank thought").
 * `n` is capped internally at RERANK_MAX_CANDS (32): reranking is meant for a search's own
 * top-K, not a bulk pass - the caller controls that by how many hits it hands in (vecdb_search's
 * own `k`), not this function. Returns 0 on success (hits re-sorted), -1 if the rerank server
 * failed (hits left in their ORIGINAL hybrid order, untouched, so a caller can always ignore the
 * failure and keep going - reranking is strictly optional). */
#define RERANK_MAX_CANDS 32
i32 vecdb_rerank_hits(VecDB *db, VRerankProc *rp, const char *query, VHit *hits, i32 n)
{
    char *texts[RERANK_MAX_CANDS];
    i32 lens[RERANK_MAX_CANDS], i, m = n < RERANK_MAX_CANDS ? n : RERANK_MAX_CANDS, ok = 1;
    f32 scores[RERANK_MAX_CANDS];
    for (i = 0; i < m; ++i) {
        texts[i] = vecdb_text(db, hits[i].node);
        lens[i] = texts[i] ? (i32)strlen(texts[i]) : 0;
        if (!texts[i]) { texts[i] = (char *)malloc(1); texts[i][0] = 0; }
    }
    if (vecdb_rerank_score(rp, query, (i32)strlen(query), (const char **)texts, lens, m, scores) != 0) ok = 0;
    for (i = 0; i < m; ++i) free(texts[i]);
    if (!ok) return -1;
    for (i = 0; i < m; ++i) hits[i].score = scores[i];
    /* small n (<=32): insertion sort is fine, no need to pull in a qsort comparator + globals */
    for (i = 1; i < m; ++i) {
        VHit key = hits[i];
        i32 j = i - 1;
        while (j >= 0 && hits[j].score < key.score) { hits[j + 1] = hits[j]; --j; }
        hits[j + 1] = key;
    }
    return 0;
}

#endif
