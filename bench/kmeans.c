/* kmeans.c - Lloyd's k-means against e-means, as the store's clustering, on real embeddings.
 *
 * The clustering only decides which rows sit together; a query then scans its nprobe closest clusters. So it is
 * judged by what a query loses: of the true 10 nearest rows, how many sit in the clusters the query probes
 * (probe recall), how many rows that costs, the recall@10 after the store's own scan and exact re-score, and the
 * k-means error (mean 2 - 2 cos of a row to its centroid). Speed is the build's wall time and the clustering's.
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -D_POSIX_C_SOURCE=200112L -O2 -march=native -Isrc bench/kmeans.c \
 *       -lm -lpthread -o .build/kmeans
 *   zcat glove-twitter-100.gz | .build/kmeans prep DIR      text vectors -> DIR/{train,query,gt}.bin
 *   VECINDEX_KMEANS=lloyd     .build/kmeans run DIR [seed]   one build, its numbers on one line
 *   VECINDEX_KMEANS=emeans:3  .build/kmeans run DIR [seed]
 *
 * prep: every vector scaled to norm 1 (the store's cosine), 10,000 held out as queries by a seeded draw (the
 * ann-benchmarks split of the same GloVe file is drawn by sklearn, so the rows differ), the exact top 10 of each by
 * dot product as ground truth. */
#include "store.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define KM_NQ   10000
#define KM_K    10
#define KM_DIM  100

static f64 km_now_(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (f64)ts.tv_sec + (f64)ts.tv_nsec * 1e-9;
}

static void *km_read_(const char *dir, const char *name, size_t *len)
{
    char p[1024];
    FILE *f;
    void *b;
    long n;
    sprintf(p, "%.900s/%s", dir, name);
    f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "kmeans: cannot open %s\n", p); exit(1); }
    fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET);
    b = malloc((size_t)n);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "kmeans: short read %s\n", p); exit(1); }
    fclose(f);
    *len = (size_t)n;
    return b;
}

static void km_write_(const char *dir, const char *name, const void *b, size_t n)
{
    char p[1024];
    FILE *f;
    sprintf(p, "%.900s/%s", dir, name);
    f = fopen(p, "wb");
    if (!f || fwrite(b, 1, n, f) != n) { fprintf(stderr, "kmeans: cannot write %s\n", p); exit(1); }
    fclose(f);
}

/* ---- prep ---- */
typedef struct { const f32 *x; i32 n; const f32 *q; i32 *gt; } KmGt;

static void km_gt_rows_(void *p, i64 a, i64 b, i32 tid)
{
    KmGt *g = (KmGt *)p;
    i64 qi;
    (void)tid;
    for (qi = a; qi < b; ++qi) {
        f32 best[KM_K];
        i32 id[KM_K], i, j;
        for (j = 0; j < KM_K; ++j) { best[j] = -3e38f; id[j] = -1; }
        for (i = 0; i < g->n; ++i) {
            f32 s = vs_dot(g->q + (size_t)qi * KM_DIM, g->x + (size_t)i * KM_DIM, KM_DIM);
            if (s <= best[KM_K - 1]) continue;
            for (j = KM_K - 1; j > 0 && best[j - 1] < s; --j) { best[j] = best[j - 1]; id[j] = id[j - 1]; }
            best[j] = s; id[j] = i;
        }
        memcpy(g->gt + (size_t)qi * KM_K, id, sizeof(id));
    }
}

static i32 km_prep_(const char *dir)
{
    static char line[1 << 16];
    i32 cap = 1300000, n = 0, i, d, nt = 0, nq = 0, *isq;
    f32 *all = (f32 *)malloc((size_t)cap * KM_DIM * sizeof(f32)), *tr, *qv;
    u32 st = 0x51ed2026u;
    KmGt g;
    f64 t0 = km_now_();
    while (fgets(line, sizeof line, stdin)) {
        char *p = line, *e;
        f32 v[KM_DIM];
        f64 q = 0.0;
        if (n == 0 && line[0] >= '0' && line[0] <= '9' && strchr(line, ' ') && strlen(line) < 32) continue;   /* "count dim" */
        while (*p && *p != ' ') ++p;                        /* the word */
        for (d = 0; d < KM_DIM; ++d) { v[d] = (f32)strtod(p, &e); if (e == p) break; p = e; }
        if (d < KM_DIM) continue;
        for (d = 0; d < KM_DIM; ++d) q += (f64)v[d] * v[d];
        if (q <= 0.0) continue;                             /* a zero vector has no direction */
        q = 1.0 / sqrt(q);
        if (n == cap) break;
        for (d = 0; d < KM_DIM; ++d) all[(size_t)n * KM_DIM + d] = (f32)(v[d] * q);
        ++n;
    }
    isq = (i32 *)calloc((size_t)n, sizeof(i32));
    for (i = 0; i < KM_NQ; ) { i32 r = (i32)(tq_rng_(&st) % (u32)n); if (!isq[r]) { isq[r] = 1; ++i; } }
    tr = (f32 *)malloc((size_t)(n - KM_NQ) * KM_DIM * sizeof(f32)); qv = (f32 *)malloc((size_t)KM_NQ * KM_DIM * sizeof(f32));
    for (i = 0; i < n; ++i) memcpy(isq[i] ? qv + (size_t)nq++ * KM_DIM : tr + (size_t)nt++ * KM_DIM, all + (size_t)i * KM_DIM, KM_DIM * sizeof(f32));
    free(all);
    fprintf(stderr, "prep: %d vectors read, %d train, %d queries (%.1fs)\n", n, nt, nq, km_now_() - t0);
    g.x = tr; g.n = nt; g.q = qv; g.gt = (i32 *)malloc((size_t)KM_NQ * KM_K * sizeof(i32));
    tq_par2_(km_gt_rows_, &g, KM_NQ, store_cpus_avail_());
    fprintf(stderr, "prep: ground truth %.1fs\n", km_now_() - t0);
    km_write_(dir, "train.bin", tr, (size_t)nt * KM_DIM * sizeof(f32));
    km_write_(dir, "query.bin", qv, (size_t)KM_NQ * KM_DIM * sizeof(f32));
    km_write_(dir, "gt.bin", g.gt, (size_t)KM_NQ * KM_K * sizeof(i32));
    return 0;
}

/* ---- run ---- */
static i32 km_run_(const char *dir, u32 seed)
{
    size_t lt, lq, lg;
    f32 *tr = (f32 *)km_read_(dir, "train.bin", &lt), *qv = (f32 *)km_read_(dir, "query.bin", &lq);
    i32 *gt = (i32 *)km_read_(dir, "gt.bin", &lg);
    i32 n = (i32)(lt / (KM_DIM * sizeof(f32))), nq = (i32)(lq / (KM_DIM * sizeof(f32))), nc = store_nclusters_(n);
    i32 nth = store_cpus_avail_(), i, b, c, pi, *blkc, *rowc, *probes, *got;
    static const i32 PROBES[] = { 8, 16, 32, 64, 128 };
    u8 *chunk[1];
    u8 *scr;
    TqCol col;
    Tq *t;
    f64 t0, tb, err = 0.0;
    const char *kind = getenv("VECINDEX_KMEANS");
    if (nc < 2) { fprintf(stderr, "kmeans: %d rows is below the store's IVF threshold (%d): no clustering\n", n, store_ivf_min_); return 1; }
    chunk[0] = (u8 *)tr;
    col.chunk = chunk; col.shift = 30; col.esz = KM_DIM * sizeof(f32); col.i8 = 0; col.dim = KM_DIM; col.scale = (const TqCol *)0;
    t0 = km_now_();
    t = tq_build(&col, n, KM_DIM, seed, nc, nth, 4);
    tb = km_now_() - t0;
    blkc = (i32 *)malloc((size_t)t->nblk * sizeof(i32)); rowc = (i32 *)malloc((size_t)n * sizeof(i32));
    for (c = 0; c < t->nc; ++c) for (b = t->cstart[c]; b < t->cstart[c + 1]; ++b) blkc[b] = c;
    for (i = 0; i < t->nblk * 32; ++i) if (t->slot_row[i] >= 0) rowc[t->slot_row[i]] = blkc[i / 32];
    for (i = 0; i < n; ++i) err += 2.0 - 2.0 * (f64)vs_dot(tr + (size_t)i * KM_DIM, t->cent + (size_t)rowc[i] * KM_DIM, KM_DIM);
    printf("%-10s seed=%u n=%d nc=%d build=%.2fs err=%.5f", kind ? kind : "default", seed, n, t->nc, tb, err / n);
    scr = (u8 *)malloc(tq_probe_scr_(t));
    probes = (i32 *)malloc(128 * sizeof(i32)); got = (i32 *)malloc(256 * sizeof(i32));
    for (pi = 0; pi < (i32)(sizeof PROBES / sizeof PROBES[0]); ++pi) {
        i32 np = PROBES[pi], hit = 0, hit2 = 0, j, z;
        f64 scanned = 0.0, ts;
        for (i = 0; i < nq; ++i) {                          /* probe recall: the true 10 inside the probed clusters */
            tq_probe_(t, qv + (size_t)i * KM_DIM, np, scr, probes);
            for (j = 0; j < np; ++j) scanned += t->csize[probes[j]];
            for (j = 0; j < KM_K; ++j) for (z = 0; z < np; ++z) if (rowc[gt[(size_t)i * KM_K + j]] == probes[z]) { ++hit; break; }
        }
        ts = km_now_();
        for (i = 0; i < nq; ++i) {                          /* the store's path: 4-bit scan, 100 candidates, exact re-score */
            const f32 *q = qv + (size_t)i * KM_DIM;
            f32 bs[KM_K];
            i32 bi[KM_K], m = tq_search(t, q, 100, got, (const u64 *)0, (const TqFilt *)0, 1, np), k2;
            for (j = 0; j < KM_K; ++j) { bs[j] = -3e38f; bi[j] = -1; }
            for (z = 0; z < m; ++z) {
                f32 s = vs_dot(q, tr + (size_t)got[z] * KM_DIM, KM_DIM);
                if (s <= bs[KM_K - 1]) continue;
                for (k2 = KM_K - 1; k2 > 0 && bs[k2 - 1] < s; --k2) { bs[k2] = bs[k2 - 1]; bi[k2] = bi[k2 - 1]; }
                bs[k2] = s; bi[k2] = got[z];
            }
            for (j = 0; j < KM_K; ++j) for (z = 0; z < KM_K; ++z) if (bi[j] == gt[(size_t)i * KM_K + z]) { ++hit2; break; }
        }
        ts = km_now_() - ts;
        printf(" | p%d probe=%.4f r10=%.4f scan=%.4f qps=%.0f", np, (f64)hit / ((f64)nq * KM_K),
               (f64)hit2 / ((f64)nq * KM_K), scanned / nq / n, nq / ts);
    }
    printf("\n");
    tq_free(t);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc >= 3 && !strcmp(argv[1], "prep")) return km_prep_(argv[2]);
    if (argc >= 3 && !strcmp(argv[1], "run")) return km_run_(argv[2], argc >= 4 ? (u32)strtoul(argv[3], (char **)0, 0) : 0x5eedu);
    fprintf(stderr, "usage: kmeans prep DIR < vectors.txt | kmeans run DIR [seed]\n");
    return 2;
}
