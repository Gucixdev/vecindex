/* rag_test.c - the RAG calls of the C API (store_rag.c): vi_neighbors, vi_search_mmr, vi_context, through the
 * library and the header alone. ./run builds it against libvecindex.a.
 *
 * The data: documents 1 and 2, ten chunks each, ids doc << 20 | chunk, texts "dDcC;". A document's chunks lie on
 * one plane of their own, chunk c at angle c / 20, so neighbouring chunks are neighbouring vectors and the two
 * documents are orthogonal. Document 3 holds one chunk, chunk 5 of document 1 with a trace of a third plane (cosine
 * 0.99995 to it): the near-duplicate, a hair further from a query at chunk 5. */
#include "vecindex.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

#define DIM 8
#define ID(d, c) (((vi_i64)(d) << 20) | (vi_i64)(c))

static int fails;

static void check(int cond, const char *what)
{
    printf(cond ? "  ok   %s\n" : "  FAIL %s\n", what);
    if (!cond) ++fails;
}

/* a unit vector at angle a on the plane of document d (dimensions 2d, 2d + 1); w > 0 adds that much of document
 * e's plane at angle 0 */
static void vec_(float *v, int d, double a, int e, double w)
{
    memset(v, 0, DIM * sizeof(float));
    v[2 * d] = (float)cos(a); v[2 * d + 1] = (float)sin(a);
    if (w > 0.0) v[2 * e] += (float)w;
}

static int has_(const vi_i64 *ids, int n, vi_i64 id) { int i; for (i = 0; i < n; ++i) if (ids[i] == id) return 1; return 0; }

int main(void)
{
    vi_db *db = vi_open((const char *)0, DIM, 0);
    float v[DIM], q[DIM], sc[16], sc2[16];
    vi_i64 ids[16], ids2[16];
    char t[32], buf[256];
    vi_ctx_opts o;
    vi_ctx_span sp[8];
    int d, c, n, n2, used, i, same;

    check(db != (vi_db *)0, "a store in memory");
    if (!db) return 1;
    for (d = 1; d <= 2; ++d)
        for (c = 0; c < 10; ++c) {
            vec_(v, d, c / 20.0, 0, 0.0);
            sprintf(t, "d%dc%d;", d, c);
            if (vi_add_text(db, ID(d, c), v, t, -1) != 0) check(0, vi_error());
        }
    vec_(v, 1, 5 / 20.0, 3, 0.01);
    vi_add_text(db, ID(3, 0), v, "d3c0;", -1);

    /* neighbours */
    n = vi_neighbors(db, ID(1, 5), 20, 2, 2, ids, 16);
    check(n == 5 && ids[0] == ID(1, 3) && ids[4] == ID(1, 7), "vi_neighbors: chunks 3 to 7 around chunk 5");
    n = vi_neighbors(db, ID(1, 0), 20, 2, 1, ids, 16);
    check(n == 2 && ids[0] == ID(1, 0) && ids[1] == ID(1, 1), "vi_neighbors: never before the document's first chunk");
    n = vi_neighbors(db, ID(1, 9), 20, 0, 5, ids, 16);
    check(n == 1 && ids[0] == ID(1, 9), "vi_neighbors: never into the next document");
    n = vi_neighbors(db, ID(2, 4), 20, -1, -1, ids, 16);
    check(n == 10 && ids[0] == ID(2, 0) && ids[9] == ID(2, 9), "vi_neighbors -1 -1: the whole document");
    check(vi_neighbors(db, ID(1, 42), 20, 1, 1, ids, 16) == -1, "vi_neighbors: an id without a row is an error");
    check(vi_neighbors(db, ID(1, 5), 64, 1, 1, ids, 16) == -1, "vi_neighbors: shift 64 is an error");

    /* MMR */
    vec_(q, 1, 5 / 20.0, 0, 0.0);
    n = vi_search_mmr(db, q, (const char *)0, 0, (const vi_filter *)0, 4, 1.0f, 0.0f, ids, sc);
    n2 = vi_search(db, q, 4, ids2, sc2);
    for (i = 0, same = n == n2; same && i < n; ++i) same = fabs((double)sc[i] - (double)sc2[i]) < 1e-5;
    check(same, "vi_search_mmr lambda 1, dup 0: the scores of vi_search");
    n = vi_search_mmr(db, q, (const char *)0, 0, (const vi_filter *)0, 4, 1.0f, 0.97f, ids, sc);
    check(n >= 1 && !(has_(ids, n, ID(1, 5)) && has_(ids, n, ID(3, 0))), "vi_search_mmr dup 0.97: the copy of the best hit is out");
    vec_(q, 1, 5 / 20.0, 2, 0.35);
    n = vi_search_mmr(db, q, (const char *)0, 0, (const vi_filter *)0, 3, 1.0f, 0.0f, ids, sc);
    check(n == 3 && !has_(ids, n, ID(2, 0)), "vi_search_mmr lambda 1: three hits of document 1 (or its copy)");
    n = vi_search_mmr(db, q, (const char *)0, 0, (const vi_filter *)0, 3, 0.3f, 0.9999f, ids, sc);
    check(n >= 2 && (has_(ids, n, ID(2, 0)) || has_(ids, n, ID(2, 1))), "vi_search_mmr lambda 0.3: document 2 comes in for variety");
    n = vi_search_mmr(db, (const float *)0, "d2c7", -1, (const vi_filter *)0, 2, 0.7f, 0.97f, ids, sc);
    check(n >= 1 && ids[0] == ID(2, 7), "vi_search_mmr with words alone: the chunk that has them");
    check(vi_search_mmr(db, q, (const char *)0, 0, (const vi_filter *)0, 3, 1.5f, 0.0f, ids, sc) == -1, "vi_search_mmr: lambda 1.5 is an error");

    /* the context */
    vec_(q, 1, 5 / 20.0, 0, 0.0);
    vi_ctx_defaults(&o);
    o.k = 1;
    n = vi_context(db, q, (const char *)0, 0, &o, sp, 8, buf, (int)sizeof buf, &used);
    check(n == 1 && sp[0].chunks == 3 && used == 15 && sp[0].off == 0 && sp[0].len == 15 && memcmp(buf, "d1c4;d1c5;d1c6;", 15) == 0,
          "vi_context k 1: the hit and a chunk each side, one span of text");
    check(n == 1 && sp[0].doc == 1 && sp[0].first == ID(1, 4) && sp[0].last == ID(1, 6) && sp[0].hit == ID(1, 5)
          && sp[0].score > 0.999f, "vi_context: the span's document, first, last, hit and cosine");
    o.dup = 0.9999f; o.k = 2;
    n = vi_context(db, q, (const char *)0, 0, &o, sp, 8, buf, (int)sizeof buf, &used);
    check(n == 1 && sp[0].doc == 1 && sp[0].chunks == 4, "vi_context k 2, the copy out: two hits of one document, one span of 4 chunks");
    o.dup = 0.0f;
    n = vi_context(db, q, (const char *)0, 0, &o, sp, 8, buf, (int)sizeof buf, &used);
    check(n == 2 && sp[0].doc == 1 && sp[1].doc == 3 && sp[1].off == sp[0].len && used == sp[0].len + 5,
          "vi_context k 2, the copy kept: a span of its own, after the best one in buf");
    o.dup = 0.97f;
    o.k = 1;
    n = vi_context(db, q, (const char *)0, 0, &o, sp, 8, buf, 5, &used);
    check(n == 1 && sp[0].chunks == 1 && used == 5, "vi_context, a budget of 5 bytes: the hit's chunk alone");
    n = vi_context(db, q, (const char *)0, 0, &o, sp, 8, buf, 12, &used);
    check(n == 1 && sp[0].chunks == 2 && used == 10 && sp[0].first == ID(1, 4), "vi_context, 12 bytes: the hit and one neighbour, never a gap");
    n = vi_context(db, q, (const char *)0, 0, &o, sp, 8, buf, 4, &used);
    check(n == 0 && used == 0, "vi_context, 4 bytes: not even the hit fits, no spans");
    o.before = o.after = -1;
    n = vi_context(db, q, (const char *)0, 0, &o, sp, 8, buf, (int)sizeof buf, &used);
    check(n == 1 && sp[0].chunks == 10 && used == 50 && memcmp(buf, "d1c0;", 5) == 0, "vi_context -1 -1: the whole document, in order");
    {   vi_filter fl;                                            /* a filter: hits only from document 2's chunks 4 to 6 */
        for (c = 0; c < 10; ++c) vi_set_int(db, ID(2, c), "doc", 2);
        vi_set_int(db, ID(1, 5), "doc", 1);
        check(vi_filter_parse(db, "doc = 2", &fl) == 0, "a filter for the context");
        vi_ctx_defaults(&o);
        o.k = 1; o.filter = &fl;
        n = vi_context(db, q, (const char *)0, 0, &o, sp, 8, buf, (int)sizeof buf, &used);
        check(n == 1 && sp[0].doc == 2, "vi_context with a filter: the best hit that passes it, document 2");
        n = vi_search_mmr(db, q, (const char *)0, 0, &fl, 3, 0.7f, 0.0f, ids, sc);   /* dup 0: its chunks lie close */
        for (i = 0, same = n == 3; same && i < n; ++i) same = (ids[i] >> 20) == 2;
        check(same, "vi_search_mmr with a filter: every hit passes");
        vi_ctx_defaults(&o); }
    {   vi_i64 given[4];                                         /* the caller's own hits: in its order, repeats once */
        given[0] = ID(2, 5); given[1] = ID(1, 5); given[2] = ID(2, 5); given[3] = ID(9, 9);
        vi_ctx_defaults(&o); o.before = o.after = 0;
        n = vi_context_ids(db, given, (const float *)0, 4, &o, sp, 8, buf, (int)sizeof buf, &used);
        check(n == 2 && sp[0].hit == ID(2, 5) && sp[1].hit == ID(1, 5) && used == 10 && memcmp(buf, "d2c5;d1c5;", 10) == 0,
              "vi_context_ids: the given hits in their order, a repeat and an id without a row passed over");
        vi_ctx_defaults(&o); }
    o.k = 0;
    check(vi_context(db, q, (const char *)0, 0, &o, sp, 8, buf, (int)sizeof buf, &used) == -1, "vi_context k 0 is an error");

    vi_close(db);
    printf("RAG: %s\n", fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}
