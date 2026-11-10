/* api_test.c - the C API (include/vecindex.h) through the library, as a program in another language would use it:
 * nothing but the header and the exported functions. ./run builds it against libvecindex.so and libvecindex.a. */
#include "vecindex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>

#define DIM 32
#define N 3000

static int fails;

static void check(int cond, const char *what)
{
    printf(cond ? "  ok   %s\n" : "  FAIL %s\n", what);
    if (!cond) ++fails;
}

static unsigned rng_(unsigned *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }

static void vec_(int i, float *v)
{
    unsigned s = (unsigned)i * 2654435761u + 99u;
    int d;
    for (d = 0; d < DIM; ++d) v[d] = (float)(rng_(&s) % 2001u) / 1000.0f - 1.0f;
}

/* how many of the sampled rows come back first for their own vector */
static int self_found_(vi_db *db, int from, int to, int step, int skip_mod)
{
    float v[DIM], sc[1];
    vi_i64 id[1];
    int i, hit = 0;
    for (i = from; i < to; i += step) {
        if (skip_mod && i % skip_mod == 0) continue;
        vec_(i, v);
        if (vi_search(db, v, 1, id, sc) == 1 && id[0] == 1000 + i) ++hit;
    }
    return hit;
}

int main(void)
{
    char path[64], line[200];
    float v[DIM], w[DIM], *qs, *bs, sc[10];
    vi_i64 ids[N], got[10], *bi, tag = 0;
    vi_db *db;
    int i, n, same, want, inner;
    FILE *f;

    sprintf(path, "/tmp/vecindex_api_%d.vxi", (int)getpid());
    printf("C API (vecindex %s)\n", vi_version());
    check(strlen(vi_version()) > 0, "vi_version");

    db = vi_open(path, DIM, 0);
    check(db != (vi_db *)0 && vi_dim(db) == DIM, "vi_open creates a new file");
    if (!db) return 1;
    check(access(path, F_OK) == 0, "the file exists right after vi_open");
    qs = (float *)malloc((size_t)N * DIM * sizeof(float));
    for (i = 0; i < N; ++i) { ids[i] = 1000 + i; vec_(i, qs + (size_t)i * DIM); }
    check(vi_add_batch(db, N, ids, qs) == N && vi_count(db) == N, "vi_add_batch: 3000 rows");
    n = self_found_(db, 0, N, 7, 0);
    sprintf(line, "vi_search before any build (exact scan of the new rows): %d/%d", n, (N + 6) / 7);
    check(n == (N + 6) / 7, line);
    vi_build(db);
    n = self_found_(db, 0, N, 7, 0);
    sprintf(line, "vi_search after vi_build: %d/%d", n, (N + 6) / 7);
    check(n * 100 >= ((N + 6) / 7) * 99, line);

    {   volatile float z = 0.0f; v[0] = z / z; }
    for (i = 1; i < DIM; ++i) v[i] = 1.0f;
    check(vi_add(db, 77, v) == -1 && vi_count(db) == N, "vi_add refuses a NaN");
    for (i = 0; i < N; i += 10) vi_delete(db, 1000 + i);
    check(vi_count(db) == N - N / 10, "vi_delete: 300 rows gone");
    vec_(20, v);
    n = vi_search(db, v, 10, got, sc);
    for (i = 0, same = 0; i < n; ++i) if (got[i] == 1020) same = 1;
    check(n == 10 && !same, "a deleted row is not found");

    vec_(7, v);
    check(vi_add(db, 1006, v) == 0 && vi_count(db) == N - N / 10, "vi_add of an existing id replaces its row");
    n = vi_search(db, v, 2, got, sc);
    check(n == 2 && got[0] + got[1] == 1006 + 1007 && got[0] != got[1] && sc[1] > 0.9999f && sc[0] <= 1.0001f, "the replaced row has the new vector, once (cosine 1)");
    vec_(6, v);
    vi_add(db, 1006, v);                                          /* back as it was */
    for (i = 0; i < 20; ++i) { vec_(5000 + i, v); vi_add(db, ((vi_i64)7 << 20) | i, v); }
    check(vi_delete_range(db, (vi_i64)7 << 20, ((vi_i64)8 << 20) - 1) == 20 && vi_count(db) == N - N / 10,
          "vi_delete_range: all 20 chunks of document 7");

    vec_(5, v);
    check(vi_get(db, 1005, w) == 0, "vi_get");
    {   float nn = 0.0f, err = 0.0f;
        for (i = 0; i < DIM; ++i) nn += v[i] * v[i];
        nn = (float)sqrt((double)nn);
        for (i = 0; i < DIM; ++i) err += (float)fabs((double)(w[i] - v[i] / nn));
        check(err < 1e-4f, "vi_get returns the row, unit-normalised");
    }
    check(vi_get(db, 999999, w) == -1, "vi_get of an unknown id: -1");

    for (i = 1; i < N; i += 3) vi_set_tag(db, 1000 + i, 5);
    check(vi_get_tag(db, 1004, &tag) == 0 && tag == 5, "vi_set_tag / vi_get_tag");
    vec_(9, v);
    n = vi_search_tagged(db, v, 10, 4, 0, got, sc);
    for (i = 0, same = 1; i < n; ++i) if ((got[i] - 1000 - 1) % 3 != 0) same = 0;
    check(n == 10 && same, "vi_search_tagged: only rows whose tag has bit 4");

    bi = (vi_i64 *)malloc((size_t)100 * 10 * sizeof(vi_i64));
    bs = (float *)malloc((size_t)100 * 10 * sizeof(float));
    vi_search_batch(db, qs, 100, 10, bi, bs, 0);
    for (i = 0, same = 0; i < 100; ++i) {
        int k, eq = 1;
        n = vi_search(db, qs + (size_t)i * DIM, 10, got, sc);
        for (k = 0; k < n; ++k) if (bi[i * 10 + k] != got[k]) eq = 0;
        same += eq;
    }
    sprintf(line, "vi_search_batch == vi_search on %d/100 queries", same);
    check(same == 100, line);

    check(vi_snapshot(db, (const char *)0) == 0, "vi_snapshot starts a background snapshot of the store's file");
    for (i = 0; i < 200; ++i) { vec_(N + i, v); vi_add(db, 1000 + N + i, v); }
    check(vi_snapshot_wait(db, 1) == 0, "vi_snapshot_wait: written");
    for (i = 200; i < 300; ++i) { vec_(N + i, v); vi_add(db, 1000 + N + i, v); }
    vec_(9, v);
    vi_add(db, 1008, v);                                          /* replaced after the snapshot: from the journal */
    for (i = 0; i < 5; ++i) { vec_(6000 + i, v); vi_add(db, ((vi_i64)9 << 20) | i, v); }
    vi_delete_range(db, (vi_i64)9 << 20, ((vi_i64)9 << 20) | 3);  /* 4 of the 5 go, also from the journal */
    want = (int)vi_count(db);
    check(vi_sync(db) == 0, "vi_sync");
    vi_close(db);

    db = vi_open(path, 0, 0);
    check(db != (vi_db *)0, "vi_open of the existing file (dim 0: it must exist)");
    if (!db) return 1;
    sprintf(line, "reopened: %d rows of %d (the snapshot plus its journal)", (int)vi_count(db), want);
    check(vi_count(db) == want, line);
    n = self_found_(db, 0, N + 300, 11, 10);
    inner = 0;
    for (i = 0; i < N + 300; i += 11) if (i % 10) ++inner;
    sprintf(line, "every row found again: %d/%d", n, inner);
    check(n * 100 >= inner * 99, line);
    vec_(9, v);
    n = vi_search(db, v, 2, got, sc);
    check(n == 2 && got[0] + got[1] == 1008 + 1009 && got[0] != got[1], "a replace replayed from the journal");
    vec_(6004, v);
    n = vi_search(db, v, 1, got, sc);
    check(n == 1 && got[0] == (((vi_i64)9 << 20) | 4), "a range delete replayed from the journal");
    check(vi_save(db, (const char *)0) == 0, "vi_save to its own file");
    vi_close(db);

    f = fopen(path, "wb");
    if (f) { fputs("not a vecindex file", f); fclose(f); }
    check(vi_open(path, DIM, 0) == (vi_db *)0, "a file that cannot be read is an error, not replaced");
    remove(path);
    sprintf(line, "%s.j", path); remove(line);
    sprintf(line, "%s.jn", path); remove(line);

    /* texts: chunks of documents (ids doc << 20 | chunk), the chunks around a hit by id range, hybrid search */
    db = vi_open(path, DIM, 0);
    check(db != (vi_db *)0, "vi_open for texts");
    for (i = 0; i < 40; ++i) {
        vi_i64 id = ((vi_i64)(i / 8) << 20) | (i % 8);
        char t[96];
        sprintf(t, "document %d, chunk %d: the %s of the matter%s", i / 8, i % 8, i % 2 ? "heart" : "core", i == 21 ? " xylophone" : "");
        vec_(i, v);
        if (vi_add_text(db, id, v, t, -1) != 0) check(0, "vi_add_text");
    }
    {
        char buf[96], small[8];
        vi_i64 ids[16];
        float s2[16];
        int k;
        n = vi_get_text(db, ((vi_i64)2 << 20) | 5, buf, (int)sizeof(buf));
        check(n == (int)strlen("document 2, chunk 5: the heart of the matter xylophone") && memcmp(buf, "document 2, chunk 5: the heart of the matter xylophone", (size_t)n) == 0,
              "vi_get_text: the bytes back");
        check(vi_get_text(db, ((vi_i64)2 << 20) | 5, small, (int)sizeof(small)) == n && memcmp(small, "document", 8) == 0,
              "vi_get_text with a small buffer: the full length, the first bytes");
        check(vi_get_text(db, (vi_i64)99 << 20, buf, (int)sizeof(buf)) == -1, "vi_get_text of no such row: -1");
        k = vi_range(db, (vi_i64)3 << 20, ((vi_i64)3 << 20) | 0xfffff, ids, 16);
        check(k == 8 && ids[0] == ((vi_i64)3 << 20) && ids[7] == (((vi_i64)3 << 20) | 7), "vi_range: a document's 8 chunks, in order");
        vec_(1000, v);
        k = vi_search_hybrid(db, v, "xylophone", -1, 3, ids, s2);
        check(k >= 1 && ids[0] == (((vi_i64)2 << 20) | 5), "vi_search_hybrid: a rare word wins over a vector pointing elsewhere");
        k = vi_search_hybrid(db, (const float *)0, "xylophone", -1, 3, ids, s2);
        check(k == 1 && ids[0] == (((vi_i64)2 << 20) | 5), "vi_search_hybrid with words only");
        vec_(17, v);
        k = vi_search_hybrid(db, v, (const char *)0, 0, 3, ids, s2);
        check(k == 3 && ids[0] == (((vi_i64)2 << 20) | 1), "vi_search_hybrid with the vector only");
        {
            vi_i64 all[40], best[5], gi[10];
            float as[40], gs[10];
            int na, j, ok = 1;
            for (j = 0; j < 5; ++j) best[j] = -1;
            na = vi_search(db, v, 40, all, as);                      /* every row ranked: each document's first hit */
            for (j = 0; j < na; ++j) if (best[all[j] >> 20] < 0) best[all[j] >> 20] = all[j];
            k = vi_search_grouped(db, v, 3, 20, gi, gs);
            for (j = 0; j < k; ++j) ok &= gi[j] == best[gi[j] >> 20] && (j == 0 || gs[j] <= gs[j - 1]);
            check(na == 40 && k == 3 && gi[0] == (((vi_i64)2 << 20) | 1) && ok && (gi[0] >> 20) != (gi[1] >> 20)
                  && (gi[0] >> 20) != (gi[2] >> 20) && (gi[1] >> 20) != (gi[2] >> 20),
                  "vi_search_grouped 3, shift 20: three documents, each by its best chunk, best first");
            k = vi_search_grouped(db, v, 10, 20, gi, gs);
            for (j = 0, ok = 1; j < k; ++j) ok &= gi[j] == best[gi[j] >> 20];
            check(k == 5 && ok, "vi_search_grouped 10 over 5 documents: the 5, each once");
            check(vi_search_grouped(db, v, 3, 0, gi, gs) == 3 && gi[0] == all[0] && gi[1] == all[1], "vi_search_grouped shift 0: vi_search");
            check(vi_search_grouped(db, v, 3, 64, gi, gs) == -1 && strstr(vi_error(), "shift") != 0, "vi_search_grouped shift 64: refused");
        }
        {   /* one document crowds the top: 600 chunks next to the query, three more documents further off */
            vi_db *m = vi_open((const char *)0, DIM, 0);
            vi_i64 gi[4];
            float gs[4], q[DIM];
            int j, d;
            vec_(5000, q);
            for (j = 0; m && j < 600; ++j) { for (d = 0; d < DIM; ++d) w[d] = q[d] + 0.001f * (float)(j % 7) * (d == j % DIM); vi_add(m, j, w); }
            for (j = 1; m && j <= 3; ++j) { vec_(5000 + j, w); for (d = 0; d < DIM; ++d) w[d] = 0.5f * w[d] + 0.5f * q[d]; vi_add(m, ((vi_i64)j << 20) | 3, w); }
            k = m ? vi_search_grouped(m, q, 4, 20, gi, gs) : 0;
            check(k == 4 && (gi[0] >> 20) == 0 && gi[1] != gi[2] && (gi[1] & 0xfffff) == 3 && (gi[2] & 0xfffff) == 3 && (gi[3] & 0xfffff) == 3,
                  "vi_search_grouped: past a document of 600 chunks to the next three");
            if (m) vi_close(m);
        }
        check(vi_add_text(db, ((vi_i64)2 << 20) | 5, v, "replaced text", -1) == 0 && vi_get_text(db, ((vi_i64)2 << 20) | 5, buf, (int)sizeof(buf)) == 13,
              "vi_add_text replaces the text with the row");
        vi_close(db);
        db = vi_open(path, 0, 0);
        check(db != (vi_db *)0 && vi_get_text(db, ((vi_i64)2 << 20) | 5, buf, (int)sizeof(buf)) == 13 && memcmp(buf, "replaced text", 13) == 0
              && vi_get_text(db, ((vi_i64)4 << 20) | 7, buf, (int)sizeof(buf)) > 0, "texts come back from the journal");
        check(vi_save(db, (const char *)0) == 0, "vi_save of a store with texts");
        vi_close(db);
        db = vi_open(path, 0, 0);
        check(db != (vi_db *)0 && vi_get_text(db, ((vi_i64)1 << 20) | 3, buf, (int)sizeof(buf)) == (int)strlen("document 1, chunk 3: the heart of the matter")
              && vi_search_hybrid(db, (const float *)0, "replaced", -1, 3, ids, s2) == 1, "texts come back from the file");
        vi_close(db);
    }
    remove(path);
    sprintf(line, "%s.j", path); remove(line);

    db = vi_open((const char *)0, 16, VI_LIVE);
    check(db != (vi_db *)0 && vi_set_layout(db, 2, 1) == 0, "an in-memory store, live, 2-bit codes + int8 rows");
    for (i = 0; i < 500; ++i) { vec_(i, v); vi_add(db, 1000 + i, v); }
    check(vi_set_layout(db, 4, 4) == -1, "vi_set_layout once there are rows: -1");
    check(vi_save(db, (const char *)0) == -1 && vi_sync(db) == -1, "vi_save / vi_sync without a file: -1");
    vi_close(db);
    check(vi_open((const char *)0, 0, 0) == (vi_db *)0, "vi_open of nothing with dim 0: NULL");

    /* write batches: a document replaced as one change, a batch refused whole, one aborted, the journal replays it */
    sprintf(path, "/tmp/vecindex_api_b_%d.vxi", (int)getpid());
    remove(path);
    db = vi_open(path, DIM, 0);
    check(db != (vi_db *)0, "vi_open for batches");
    for (i = 0; i < 6; ++i) { vec_(i, v); vi_add_text(db, ((vi_i64)1 << 20) | i, v, "old chunk", -1); }
    if (db) {
        vi_batch *b = vi_batch_begin(db);
        vi_i64 bids[16];
        char buf[32];
        union { float f; unsigned u; } nan_;
        int r;
        const vi_i64 lo = (vi_i64)1 << 20, hi = ((vi_i64)1 << 20) | 0xfffff;
        check(b != (vi_batch *)0, "vi_batch_begin");
        vi_batch_delete_range(b, lo, hi);
        for (i = 0; i < 2; ++i) { vec_(100 + i, v); vi_batch_add_text(b, lo | (10 + i), v, "new chunk", -1); }
        check(vi_range(db, lo, hi, bids, 16) == 6, "vi_batch: nothing shows before the commit");
        r = vi_batch_commit(b, 1);
        check(r == 3 && vi_range(db, lo, hi, bids, 16) == 2 && bids[0] == (lo | 10) && bids[1] == (lo | 11),
              "vi_batch_commit: the document's 6 chunks out, 2 new ones in, at once (3 changes)");
        b = vi_batch_begin(db);
        vi_batch_delete(b, lo | 10);
        nan_.u = 0x7fc00000u;
        vec_(7, v); v[2] = nan_.f;
        check(vi_batch_add(b, 99, v) == -1, "vi_batch_add of a vector with a NaN: -1");
        check(vi_batch_commit(b, 0) == -1 && vi_range(db, lo, hi, bids, 16) == 2, "vi_batch_commit of that batch: -1, its delete not done either");
        b = vi_batch_begin(db);
        vi_batch_delete_range(b, lo, hi);
        vi_batch_abort(b);
        check(vi_range(db, lo, hi, bids, 16) == 2, "vi_batch_abort: nothing changed");
        vi_close(db);
        db = vi_open(path, 0, 0);
        check(db != (vi_db *)0 && vi_range(db, lo, hi, bids, 16) == 2 && vi_get_text(db, lo | 11, buf, (int)sizeof(buf)) == 9
              && memcmp(buf, "new chunk", 9) == 0 && vi_count(db) == 2, "the batch comes back from the journal, whole");
        if (db) vi_close(db);
    }
    remove(path);
    sprintf(line, "%s.j", path); remove(line);

    /* memory tiers: the two sizes, and tier changes that searches do not notice */
    db = vi_open((const char *)0, DIM, 0);
    for (i = 0; i < 2000; ++i) { vec_(i, v); vi_add(db, 1000 + i, v); }
    vi_build(db);
    {
        vi_i64 hot = 0, cold = 0;
        vi_memory(db, &hot, &cold);
        check(hot > 0 && hot < cold && cold == (vi_i64)2000 * (DIM * 4 + 20), "vi_memory: hot (the codes) and cold (the rows, ids, tags and checks)");
        check(vi_set_memory(db, 1) >= -1 && vi_set_memory(db, 2) >= -1 && vi_set_memory(db, 0) == 0 && self_found_(db, 0, 2000, 97, 0) >= 20,
              "vi_set_memory 1, 2, 0: searches still find their rows");
    }
    vi_close(db);

    /* tenants without a server: a directory, a store per name */
    {
        char dpath[64], fn[160], longname[80];
        vi_dir *dir;
        vi_db *t1, *t2;
        sprintf(dpath, "/tmp/vecindex_api_dir_%d", (int)getpid());
        mkdir(dpath, 0700);
        memset(longname, 'a', 65); longname[65] = 0;
        check(vi_dir_open("/nonexistent/vecindex-dir", DIM, 0) == (vi_dir *)0, "vi_dir_open of no directory: NULL");
        dir = vi_dir_open(dpath, DIM, 0);
        check(dir != (vi_dir *)0, "vi_dir_open of a directory");
        if (dir) {
            t1 = vi_dir_get(dir, "acme");
            t2 = vi_dir_get(dir, "beta-2");
            check(t1 != (vi_db *)0 && t2 != (vi_db *)0 && t1 != t2, "vi_dir_get: a store per tenant");
            if (t1 && t2) {
                vec_(5, v); vi_add(t1, 1, v);
                check(vi_count(t1) == 1 && vi_count(t2) == 0, "tenants are apart: a row in one is not in the other");
            }
            check(vi_dir_get(dir, "acme") == t1, "vi_dir_get again: the same handle");
            check(!vi_dir_get(dir, "../x") && !vi_dir_get(dir, "") && !vi_dir_get(dir, "a b") && !vi_dir_get(dir, longname),
                  "vi_dir_get of a bad name (../x, empty, a space, 65 letters): NULL");
            sprintf(fn, "%s/acme.vxi", dpath);
            check(access(fn, F_OK) == 0, "a tenant's store is DIR/NAME.vxi");
            vi_dir_close(dir);
            dir = vi_dir_open(dpath, DIM, 0);
            t1 = dir ? vi_dir_get(dir, "acme") : (vi_db *)0;
            check(t1 != (vi_db *)0 && vi_count(t1) == 1, "a tenant's rows come back after vi_dir_close and vi_dir_open");
            vi_dir_close(dir);
        }
        sprintf(fn, "%s/acme.vxi", dpath); remove(fn);
        sprintf(fn, "%s/acme.vxi.j", dpath); remove(fn);
        sprintf(fn, "%s/beta-2.vxi", dpath); remove(fn);
        sprintf(fn, "%s/beta-2.vxi.j", dpath); remove(fn);
        rmdir(dpath);
    }

    /* metrics: dot and L2 rank by the vectors as they came and score in that metric; a file keeps its metric */
    {
        char fn[64], fn2[64], jn[72];
        float a[DIM], b[DIM], q[DIM];
        vi_i64 i2[2];
        float s2[2];
        sprintf(fn, "/tmp/vecindex_api_dot_%d.vxi", (int)getpid());
        sprintf(fn2, "/tmp/vecindex_api_l2_%d.vxi", (int)getpid());
        for (i = 0; i < DIM; ++i) { a[i] = 0.0f; b[i] = 0.0f; q[i] = 0.0f; }
        a[0] = 1.0f;                                              /* q's own direction, length 1 */
        b[0] = 3.0f; b[1] = 3.0f;                                 /* 45 degrees away, length 4.24: the larger inner product */
        q[0] = 1.0f;
        db = vi_open(fn, DIM, VI_DOT);
        if (db) { vi_add(db, 1, a); vi_add(db, 2, b); n = vi_search(db, q, 2, i2, s2); } else n = -1;
        check(n == 2 && i2[0] == 2 && s2[0] > 2.999f && s2[0] < 3.001f && i2[1] == 1 && s2[1] > 0.999f && s2[1] < 1.001f,
              "VI_DOT: ranked by the inner product (3 before 1), scores the inner product");
        check(db && vi_get(db, 2, v) == 0 && v[0] > 2.999f && v[0] < 3.001f, "VI_DOT: vi_get returns the vector as it came");
        if (db) vi_close(db);
        db = vi_open(fn, 0, 0);
        check(db && vi_search(db, q, 1, i2, s2) == 1 && i2[0] == 2, "a dot store opened again is still a dot store");
        if (db) vi_close(db);
        check(vi_open(fn, 0, VI_L2) == (vi_db *)0 && strstr(vi_error(), "dot store, not L2") != 0, "asking a dot store for L2: NULL, and why");
        q[0] = 2.5f; q[1] = 2.5f;                                 /* L2: b is 0.71 away, a 2.92 */
        db = vi_open(fn2, DIM, VI_L2);
        if (db) { vi_add(db, 1, a); vi_add(db, 2, b); n = vi_search(db, q, 2, i2, s2); } else n = -1;
        check(n == 2 && i2[0] == 2 && s2[0] > 0.706f && s2[0] < 0.708f && i2[1] == 1 && s2[1] > 2.914f && s2[1] < 2.917f,
              "VI_L2: nearest first, scores the Euclidean distance");
        if (db) vi_close(db);
        remove(fn); remove(fn2);
        sprintf(jn, "%s.j", fn); remove(jn);
        sprintf(jn, "%s.j", fn2); remove(jn);
    }

    /* string keys; labels in a hierarchy: a label stands for everything below it, a new hierarchy touches no row */
    {
        char fn[64], jn[72];
        vi_i64 k1, i3[3];
        float s3[3];
        int pass;
        sprintf(fn, "/tmp/vecindex_api_lab_%d.vxi", (int)getpid());
        sprintf(jn, "%s.j", fn);
        k1 = vi_key("docs/readme.md", -1);
        check(k1 >= 0 && k1 == vi_key("docs/readme.md", 14) && k1 != vi_key("docs/readme.mD", -1) && vi_key((const char *)0, 0) == -1,
              "vi_key: the same string, the same id; another string, another id");
        db = vi_open(fn, DIM, 0);
        check(db && vi_label(db, "dog", "animal") == 0 && vi_label(db, "cat", "animal") == 0
              && vi_label(db, "animal", "dog") == -1 && strstr(vi_error(), "cycle") != 0,
              "vi_label: dog and cat under animal; animal under dog refused as a cycle");
        check(db && vi_label(db, "bad name", (const char *)0) == -1, "vi_label of a name with a space: refused");
        for (i = 0; db && i < 3; ++i) { vec_(100 + i, v); vi_add(db, 1 + i, v); }
        check(db && vi_label_add(db, 1, "dog") == 0 && vi_label_add(db, 2, "cat,big") == 0 && vi_label_add(db, 3, "big") == 0
              && vi_label_add(db, 99, "dog") == -1, "vi_label_add: lists, new labels made, a missing row refused");
        vec_(100, v);
        check(db && vi_search_labeled(db, v, 3, "animal", (const char *)0, (const char *)0, i3, s3) == 2,
              "all=animal finds the dog and the cat, never labelled animal themselves");
        check(db && vi_search_labeled(db, v, 3, "animal", (const char *)0, "big", i3, s3) == 1 && i3[0] == 1, "all=animal none=big: the dog");
        check(db && vi_search_labeled(db, v, 3, (const char *)0, "dog,big", (const char *)0, i3, s3) == 3, "any=dog,big: all three");
        check(db && vi_label(db, "dog", "pet") == 0 && vi_search_labeled(db, v, 3, "pet", (const char *)0, (const char *)0, i3, s3) == 1
              && i3[0] == 1, "a new parent, pet over dog: found through it, no row changed");
        check(db && vi_search_labeled(db, v, 3, "nosuch", (const char *)0, (const char *)0, i3, s3) == -1
              && strstr(vi_error(), "\"nosuch\" does not exist") != 0, "an unknown label: -1, and its name");
        check(db && vi_label_remove(db, 2, "cat") == 0 && vi_search_labeled(db, v, 3, "animal", (const char *)0, (const char *)0, i3, s3) == 1,
              "vi_label_remove: the cat is no longer an animal");
        if (db) vi_close(db);
        for (pass = 0; pass < 2; ++pass) {                       /* from the journal, then from the file's table */
            db = vi_open(fn, 0, 0);
            check(db && vi_search_labeled(db, v, 3, "pet", (const char *)0, (const char *)0, i3, s3) == 1 && i3[0] == 1,
                  pass ? "labels come back from the file's table" : "labels come back from the journal");
            if (db && !pass) vi_save(db, (const char *)0);
            if (db) vi_close(db);
        }
        remove(fn); remove(jn);
    }

    /* errors say why; a NULL handle is an error, not a crash */
    {
        char fn[64];
        float bad[DIM], z[DIM];
        vi_i64 i2[2];
        float s2[2];
        sprintf(fn, "/tmp/vecindex_api_err_%d.vxi", (int)getpid());
        for (i = 0; i < DIM; ++i) { bad[i] = 0.5f; z[i] = 0.0f; }
        bad[3] = bad[3] / (z[0] * 0.0f + z[1]);                  /* 0.5 / 0: an infinity */
        db = vi_open(fn, DIM, 0);
        check(db && vi_add(db, 1, bad) == -1 && strstr(vi_error(), "infinity at index 3") != 0, "vi_add of an infinity: -1, and vi_error names its index");
        check(db && vi_add(db, 2, z) == -1 && strstr(vi_error(), "all zeros") != 0, "vi_add of a zero vector: -1, all zeros");
        vec_(1, v);
        check(db && vi_add(db, 3, v) == 0 && vi_get(db, 99, w) == -1 && strstr(vi_error(), "no row with id 99") != 0, "vi_get of no such id: no row with id 99");
        check(db && vi_search(db, v, 0, i2, s2) == -1 && strstr(vi_error(), "k must be") != 0, "vi_search with k 0: -1, k must be 1 or more");
        check(db && vi_search(db, v, 2, i2, s2) == 1 && i2[0] == 3 && s2[0] > 0.9999f, "vi_search: the cosine, 1 for the same direction");
        check(vi_add((vi_db *)0, 1, v) == -1 && strstr(vi_error(), "db is NULL") != 0, "a NULL handle: -1, db is NULL");
        if (db) vi_close(db);
        check(vi_open(fn, DIM + 1, 0) == (vi_db *)0 && strstr(vi_error(), "dimensions, not") != 0, "vi_open with another dim: NULL, and the dims in vi_error");
        check(vi_open("/nonexistent/x.vxi", 0, 0) == (vi_db *)0 && strstr(vi_error(), "dim 0 cannot create") != 0, "vi_open of no file with dim 0: NULL, and why");
        remove(fn);
        sprintf(fn, "/tmp/vecindex_api_err_%d.vxi.j", (int)getpid()); remove(fn);
    }

    /* integrity: digests and parity in every file, checked at open, a scrub of the file */
    {
        char fn[64], jn[72];
        vi_i64 blocks = 0, bad = -1, fixed = -1, rows_bad = -1, rows_healed = -1;
        int jd = -1;
        sprintf(fn, "/tmp/vecindex_api_sums_%d.vxi", (int)getpid());
        sprintf(jn, "%s.j", fn);
        db = vi_open(fn, DIM, 0);
        check(db && vi_set_parity(db, 8, 2) == 0 && vi_set_parity(db, 0, 2) == -1 && vi_set_parity(db, 8, 99) == -1,
              "vi_set_parity: 8 and 2 taken, 0 data blocks or 99 parity refused");
        if (db) { for (i = 0; i < 3000; ++i) { vec_(i, v); vi_add(db, 1000 + i, v); } vi_build(db); vi_save(db, (const char *)0); vi_close(db); }
        check(vi_scrub(fn, 0, &fixed) == 0 && fixed == 0, "vi_scrub of an intact file: 0 damaged");
        check(vi_scrub("/nonexistent/vecindex.vxi", 0, (vi_i64 *)0) == -1, "vi_scrub of no file: -1");
        db = vi_open(fn, 0, 0);
        if (db) { vi_health hh; if (vi_integrity(db, &hh) == 0) { blocks = hh.blocks; bad = hh.bad; fixed = hh.fixed; rows_bad = hh.rows_bad; rows_healed = hh.rows_healed; jd = hh.journal_damaged; } }
        check(db && blocks > 0 && bad == 0 && fixed == 0 && rows_bad == 0 && rows_healed == 0 && jd == 0 && !*vi_open_error(),
              "vi_integrity after an open: blocks checked, none damaged");
        if (db) { vec_(77, v); check(vi_get(db, 1078, v) == 0 && self_found_(db, 0, 3000, 97, 0) >= 20, "rows read with their checks on: found"); }
        if (db) vi_close(db);
        vi_set_verify(1);
        db = vi_open(fn, 0, 0);
        vi_set_verify(2);
        check(db && vi_count(db) == 3000, "vi_set_verify 1: opens, checking the header's blocks only");
        if (db) vi_close(db);
        vi_set_verify(3);
        db = vi_open(fn, 0, 0);
        vi_set_verify(2);
        {   vi_i64 ld = 0, lt = 0;
            int tries;
            for (tries = 0; db && tries < 400; ++tries) {          /* 16 MB/s: a small file is walked in a moment */
                vi_lens(db, &ld, &lt);
                if (lt > 0 && ld >= lt) break;
                { struct timespec ts; ts.tv_sec = 0; ts.tv_nsec = 10000000L; nanosleep(&ts, (struct timespec *)0); }
            }
            check(db && vi_count(db) == 3000 && lt > 0 && ld == lt, "vi_set_verify 3 (the lens): opens at once, vi_lens walks every block");
        }
        if (db) vi_close(db);
        remove(fn); remove(jn);
    }

    /* numbers on rows and range filters: in the scan, the file, the journal and batches */
    {
        char fn[64], jn[72];
        vi_i64 ri[10], all[N], num = -1;
        float rs[10], as[N];
        int j, na, ok, want_n, bad;
        vi_batch *b;
        vi_health hh;
        sprintf(fn, "/tmp/vecindex_api_num_%d.vxi", (int)getpid());
        sprintf(jn, "%s.j", fn);
        db = vi_open(fn, DIM, 0);
        for (i = 0; db && i < N; ++i) { vec_(i, v); vi_add(db, i, v); }
        vec_(1234, v);
        check(db && vi_search_range(db, v, 5, 0, 0, ri, rs) == 5 && vi_search_range(db, v, 5, 1, 9, ri, rs) == 0,
              "vi_search_range before any number: every row's is 0");
        for (i = 0, ok = 1; db && i < N; ++i) ok &= vi_set_num(db, i, (vi_i64)(i % 100) - 50) == 0;
        check(db && ok && vi_set_num(db, N + 5, 1) == -1 && vi_get_num(db, 1234, &num) == 0 && num == 34 - 50,
              "vi_set_num on every row, a missing row refused, vi_get_num back");
        if (db) vi_build(db);
        na = db ? vi_search(db, v, N, all, as) : 0;              /* every row ranked: the expected answer in order */
        for (want_n = 0, j = 0; j < na && want_n < 10; ++j) if (all[j] % 100 >= 80) ri[want_n++] = all[j];
        n = db ? vi_search_range(db, v, 10, 30, 49, got, sc) : 0;
        for (j = 0, ok = n == 10; j < n; ++j) ok &= got[j] % 100 >= 80 && got[j] == ri[j];
        check(ok, "vi_search_range [30, 49]: the 10 nearest rows with numbers in it, in order");
        for (i = 0, ok = 1; db && i < N; i += 3) ok &= vi_label_add(db, i, "legal") == 0;   /* a third of the rows */
        for (want_n = 0, j = 0; j < na && want_n < 10; ++j) if (all[j] % 100 >= 80 && all[j] % 3 == 0) ri[want_n++] = all[j];
        n = db ? vi_search_where(db, v, 10, "legal", (const char *)0, (const char *)0, 30, 49, got, sc) : 0;
        for (j = 0, ok &= n == 10; j < n; ++j) ok &= got[j] == ri[j];
        check(ok, "vi_search_where legal and [30, 49]: the 10 nearest rows with both, in order");
        n = db ? vi_search_where(db, v, 10, (const char *)0, (const char *)0, "legal", -16, -16, got, sc) : 0;
        for (j = 0, ok = n == 10; j < n; ++j) ok &= got[j] % 100 == 34 && got[j] % 3 != 0;
        check(ok && got[0] == 1234, "vi_search_where none=legal and one number: the query's own row first, no legal one");
        check(db && vi_search_where(db, v, 10, "nosuch", (const char *)0, (const char *)0, 0, 9, got, sc) == -1,
              "vi_search_where with a label that does not exist: -1");
        n = db ? vi_search_range(db, v, 10, -16, -16, got, sc) : 0;
        check(n == 10 && got[0] == 1234 && got[1] % 100 == 34, "vi_search_range of one value: the query's own row first");
        check(db && vi_search_range(db, v, 10, 50, 900, got, sc) == 0 && vi_search_range(db, v, 10, 5, 4, got, sc) == 0,
              "vi_search_range of no rows, and lo > hi: no hits");
        if (db) { vec_(7, w); vi_add(db, 7, w); }                /* a replaced row starts again at 0 */
        check(db && vi_get_num(db, 7, &num) == 0 && num == 0, "vi_add over a numbered row: its number is 0 again");
        if (db) vi_set_num(db, 7, -43);
        b = db ? vi_batch_begin(db) : (vi_batch *)0;
        vec_(N + 1, w);
        check(b && vi_batch_add(b, N + 1, w) == 0 && vi_batch_set_num(b, N + 1, 777) == 0 && vi_batch_set_num(b, 99, 778) == 0
              && vi_batch_set_num(b, N + 9, 1) == 0 && vi_batch_commit(b, 1) == 4,
              "vi_batch_set_num: on a row the batch adds, a row the store has, a missing id");
        check(db && vi_get_num(db, N + 1, &num) == 0 && num == 777 && vi_get_num(db, 99, &num) == 0 && num == 778
              && vi_search_range(db, w, 3, 777, 777, got, sc) == 1 && got[0] == N + 1, "the batch's numbers are in, and searched");
        if (db) vi_close(db);                                    /* no save: the journal carries every number */
        db = vi_open(fn, 0, 0);
        for (i = 0, ok = db != (vi_db *)0; db && i < N; ++i)
            ok &= vi_get_num(db, i, &num) == 0 && num == (i == 7 ? -43 : (i == 99 ? 778 : (vi_i64)(i % 100) - 50));
        check(ok && vi_get_num(db, N + 1, &num) == 0 && num == 777, "the journal replayed: every number back");
        check(db && vi_save(db, (const char *)0) == 0, "vi_save with numbers");
        if (db) vi_close(db);
        db = vi_open(fn, 0, 0);
        for (i = 0, ok = db != (vi_db *)0; db && i < N; ++i)
            ok &= vi_get_num(db, i, &num) == 0 && num == (i == 7 ? -43 : (i == 99 ? 778 : (vi_i64)(i % 100) - 50));
        bad = db && vi_integrity(db, &hh) == 0 ? (int)(hh.bad + hh.rows_bad) : -1;
        n = db ? vi_search_range(db, v, 10, -16, -16, got, sc) : 0;
        check(ok && bad == 0 && n == 10 && got[0] == 1234, "the saved file: numbers back, checks intact, searched");
        if (db) { vi_set_num(db, 1234, 5); vi_close(db); }       /* a number set on a row of the mapped file */
        db = vi_open(fn, 0, 0);
        check(db && vi_get_num(db, 1234, &num) == 0 && num == 5 && vi_search_range(db, v, 1, 5, 5, got, sc) == 1 && got[0] == 1234,
              "a number changed after the load: journaled, back after a reopen");
        if (db) vi_close(db);
        remove(fn); remove(jn);
    }

    /* vi_update: a new vector, the tag, number and text kept - live, through the journal, a batch and a save */
    {
        char fn[64], jn[72], tb[32];
        vi_i64 tag = 0, num = 0;
        float g[DIM];
        int pass, ok, j, k;
        vi_batch *b;
        sprintf(fn, "/tmp/vecindex_api_upd_%d.vxi", (int)getpid());
        sprintf(jn, "%s.j", fn);
        db = vi_open(fn, DIM, 0);
        for (i = 0; db && i < 200; ++i) { vec_(i, v); vi_add(db, i, v); }
        vec_(900, v);
        if (db) { vi_add_text(db, 5, v, "zebra quartz", -1); vi_set_tag(db, 5, 0x50); vi_set_num(db, 5, 1999); }
        vec_(901, w);
        check(db && vi_update(db, 5, w) == 0 && vi_update(db, 4444, w) == -1 && strstr(vi_error(), "4444") != 0,
              "vi_update of a row, and of no row: -1 naming the id");
        for (pass = 0; pass < 3; ++pass) {                       /* live, after a reopen (journal), after save + reopen */
            ok = db && vi_get(db, 5, g) == 0 && vi_count(db) == 200 && vi_get_tag(db, 5, &tag) == 0 && tag == 0x50
                 && vi_get_num(db, 5, &num) == 0 && num == 1999 && vi_get_text(db, 5, tb, sizeof tb) == 12 && !memcmp(tb, "zebra quartz", 12);
            {   double dw = 0.0, ww = 0.0;                       /* the new vector: a cosine store keeps its direction */
                for (j = 0; j < DIM; ++j) { dw += (double)g[j] * w[j]; ww += (double)w[j] * w[j]; }
                ok &= dw / sqrt(ww) > 0.999;
            }
            k = db ? vi_search_hybrid(db, (const float *)0, "quartz", -1, 1, got, sc) : 0;
            ok &= k == 1 && got[0] == 5;
            k = db ? vi_search_range(db, w, 1, 1999, 1999, got, sc) : 0;
            ok &= k == 1 && got[0] == 5;
            check(ok, pass == 0 ? "vi_update: the new vector, tag, number and text kept, found by words and range"
                      : (pass == 1 ? "vi_update after a reopen: the journal replayed it" : "vi_update after a save and a reopen"));
            if (db) { if (pass == 1) vi_save(db, (const char *)0); vi_close(db); }
            db = vi_open(fn, 0, 0);
        }
        b = db ? vi_batch_begin(db) : (vi_batch *)0;
        vec_(902, w);
        vec_(903, g);
        check(b && vi_batch_add(b, 300, g) == 0 && vi_batch_update(b, 300, w) == 0 && vi_batch_update(b, 5, g) == 0
              && vi_batch_update(b, 7777, g) == 0 && vi_batch_commit(b, 1) == 4, "vi_batch_update: on a row the batch adds, a row the store has, a missing id");
        check(db && vi_count(db) == 201 && vi_get_tag(db, 5, &tag) == 0 && tag == 0x50 && vi_get_num(db, 5, &num) == 0 && num == 1999
              && vi_search(db, g, 1, got, sc) == 1 && got[0] == 5 && vi_search(db, w, 1, got, sc) == 1 && got[0] == 300,
              "the batch's updates are in, the row's tag and number kept");
        if (db) vi_close(db);
        remove(fn); remove(jn);
    }

    /* namespaces: tenants in one store - a small one searched over its own rows, a big one through the scan with the
     * namespace inside it; nothing of another namespace ever comes back; moves, replacements, deletes, batches; the
     * journal and the file keep them */
    {
        char fn[64], jn[72];
        vi_i64 got2[64], all[33000];
        float sc2[64], as[33000];
        int pass, j, ok, iso, n1, na, want, hit, tot;
        vi_batch *b;
        sprintf(fn, "/tmp/vecindex_api_ns_%d.vxi", (int)getpid());
        sprintf(jn, "%s.j", fn);
        db = vi_open(fn, DIM, 0);
        for (i = 0; db && i < 50; ++i) { vec_(100000 + i, v); vi_add_ns(db, 1, 100000 + i, v, (const char *)0, 0); }
        for (i = 0; db && i < 30000; ++i) { vec_(200000 + i, v); vi_add_ns(db, 2, 200000 + i, v, i < 3 ? "a text of tenant two" : (const char *)0, -1); }
        for (i = 0; db && i < 10; ++i) { vec_(300000 + i, v); vi_add_ns(db, 3, 300000 + i, v, (const char *)0, 0); }
        for (i = 0; db && i < 100; ++i) { vec_(400000 + i, v); vi_add(db, 400000 + i, v); }
        check(db && vi_count_ns(db, 1) == 50 && vi_count_ns(db, 2) == 30000 && vi_count_ns(db, 3) == 10 && vi_count_ns(db, 0) == 100
              && vi_count_ns(db, 9) == 0, "vi_add_ns: three namespaces and the rows in none, counted");
        if (db) vi_build(db);
        for (pass = 0; pass < 3; ++pass) {                       /* live, after a reopen (journal), after save + reopen */
            for (j = 0, iso = 1, ok = 1, hit = tot = 0; db && j < 40; ++j) {
                int q, m;
                vec_(500000 + j, v);
                n1 = vi_search_ns(db, 1, v, 10, got2, sc2);
                for (q = 0; q < n1; ++q) iso &= got2[q] >= 100000 && got2[q] < 100050 && (pass == 0 || got2[q] != 100008);
                na = vi_search(db, v, 33000, all, as);           /* every row ranked: the exact answer inside each */
                for (q = want = 0; q < na && want < 10; ++q) if (all[q] >= 100000 && all[q] < 100050 && (pass == 0 || all[q] != 100008)) ok &= all[q] == got2[want++];
                ok &= n1 == 10 && want == 10;
                m = vi_search_ns(db, 2, v, 10, got2, sc2);
                for (q = 0; q < m; ++q) iso &= got2[q] >= 200000 && got2[q] < 230000;
                for (q = want = 0; q < na && want < 10; ++q) if (all[q] >= 200000 && all[q] < 230000) {
                    int z; ++want; ++tot;
                    for (z = 0; z < m; ++z) if (got2[z] == all[q]) { ++hit; break; }
                }
                m = vi_search_ns(db, 0, v, 10, got2, sc2);
                for (q = 0; q < m; ++q) iso &= got2[q] >= 400000;
            }
            check(iso, pass == 0 ? "vi_search_ns: never a row of another namespace (40 queries, namespaces 1, 2 and none)"
                       : (pass == 1 ? "vi_search_ns after a reopen: still none from another" : "vi_search_ns after a save: still none from another"));
            check(ok, "vi_search_ns, a namespace of 50: the exact 10 nearest of its own rows, in order");
            check(hit * 10 >= tot * 8, "vi_search_ns, a namespace of 30000: recall@10 against its own rows at least 0.8");
            vec_(300004, v);
            check(db && vi_search_ns(db, 3, v, 3, got2, sc2) == 3 && got2[0] == 300004, "vi_search_ns, a namespace of 10: the query's own row first");
            if (pass == 0) {
                vec_(100007, w); w[0] += 0.01f;
                check(db && vi_add(db, 100007, w) == 0 && vi_get_ns(db, 100007, &tag) == 0 && tag == 1 && vi_count_ns(db, 1) == 50,
                      "vi_add over a row of namespace 1: it stays in namespace 1");
                check(db && vi_set_ns(db, 100008, 3) == 0 && vi_count_ns(db, 1) == 49 && vi_count_ns(db, 3) == 11, "vi_set_ns: a row moves from 1 to 3");
                vec_(100008, v);
                check(db && vi_search_ns(db, 3, v, 1, got2, sc2) == 1 && got2[0] == 100008 && vi_search_ns(db, 1, v, 1, got2, sc2) == 1 && got2[0] != 100008,
                      "after the move: found in 3, not in 1");
                b = db ? vi_batch_begin(db) : (vi_batch *)0;
                vec_(600000, v);
                check(b && vi_batch_add(b, 600000, v) == 0 && vi_batch_set_ns(b, 600000, 4) == 0 && vi_batch_commit(b, 1) == 2
                      && vi_count_ns(db, 4) == 1 && vi_search_ns(db, 4, v, 5, got2, sc2) == 1 && got2[0] == 600000, "vi_batch_set_ns: a fourth namespace");
                check(db && vi_get_text(db, 200001, (char *)got2, 64) == 20, "vi_add_ns with a text");
            }
            if (db) { if (pass == 1) vi_save(db, (const char *)0); vi_close(db); }
            db = vi_open(fn, 0, 0);
            check(db && vi_count_ns(db, 1) == 49 && vi_count_ns(db, 3) == 11 && vi_count_ns(db, 4) == 1, "after the reopen: the namespaces counted the same");
        }
        check(db && vi_delete_ns(db, 3) == 11 && vi_count_ns(db, 3) == 0 && vi_search_ns(db, 3, v, 5, got2, sc2) == 0
              && vi_count_ns(db, 1) == 49 && vi_delete_ns(db, 0) == -1, "vi_delete_ns: all 11 rows of namespace 3, the rest stay; namespace 0 refused");
        if (db) vi_close(db);
        db = vi_open(fn, 0, 0);
        check(db && vi_count_ns(db, 3) == 0 && vi_count_ns(db, 2) == 30000, "vi_delete_ns replayed from the journal");
        if (db) vi_close(db);
        remove(fn); remove(jn);
    }

    /* shards: rows split over three stores of a directory, one search over all of them, the same hits as one store */
    {
        char dp[64], fp[96];
        const char *nm[3] = { "s0", "s1", "s2" };
        vi_dir *dir;
        vi_db *one = vi_open((const char *)0, DIM, 0), *l2;
        vi_i64 a[10], b[10];
        float as[10], bs2[10];
        int j, na, nb, same = 1, q;
        sprintf(dp, "/tmp/vecindex_api_shards_%d", (int)getpid());
        mkdir(dp, 0700);
        dir = vi_dir_open(dp, DIM, 0);
        for (i = 0; dir && one && i < N; ++i) { vec_(i, v); vi_add(vi_dir_get(dir, nm[i % 3]), 70000 + i, v); vi_add(one, 70000 + i, v); }
        sprintf(fp, "%s/notes.txt", dp); f = fopen(fp, "w"); if (f) { fputs("not a store\n", f); fclose(f); }
        for (q = 0; dir && one && q < 20; ++q) {
            vec_(q * 149 + 3, v);
            na = vi_dir_search(dir, v, 10, a, as);
            nb = vi_search(one, v, 10, b, bs2);
            same &= na == 10 && nb == 10;
            for (j = 0; j < na && j < nb; ++j) same &= a[j] == b[j] && (j == 0 || as[j] <= as[j - 1]);
        }
        check(dir && one && same, "vi_dir_search over 3 shards: the hits of one store with every row, in order");
        if (dir) vi_dir_close(dir);
        sprintf(fp, "%s/l2.vxi", dp);
        l2 = vi_open(fp, DIM, VI_L2);
        if (l2) { vec_(1, v); vi_add(l2, 1, v); vi_close(l2); }
        dir = vi_dir_open(dp, DIM, 0);
        vec_(3, v);
        check(dir && vi_dir_search(dir, v, 10, a, as) == -1 && strstr(vi_error(), "another dimension or metric") != 0,
              "vi_dir_search with a store of another metric: -1, and why");
        if (dir) vi_dir_close(dir);
        for (j = 0; j < 4; ++j) {
            sprintf(fp, "%s/%s.vxi", dp, j < 3 ? nm[j] : "l2"); remove(fp);
            sprintf(fp, "%s/%s.vxi.j", dp, j < 3 ? nm[j] : "l2"); remove(fp);
        }
        sprintf(fp, "%s/notes.txt", dp); remove(fp);
        rmdir(dp);
        if (one) vi_close(one);
    }

    free(qs); free(bi); free(bs);
    printf("C API: %s\n", fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}
