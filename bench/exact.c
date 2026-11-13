/* exact.c - search with a proof (vi_search_exact) against vi_search and against scoring every row, on real embeddings.
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -D_POSIX_C_SOURCE=200112L -O2 -march=native -Isrc -Iinclude bench/exact.c \
 *       -lm -lpthread -o .build/exact
 *   .build/exact DIR [nq]        DIR/{train,query,gt}.bin from `bench/kmeans prep` (GloVe-100, unit vectors)
 *
 * One store over every train row (vi_add_batch, vi_build), then per query: vi_search (the default probes and
 * re-score), vi_search_exact, and every row scored in f32 by the store's own SIMD dot (on the first 200 queries). Printed:
 * recall@10 of each against the ground truth, whether vi_search_exact's scores equal the f32 scan's, the latency
 * (median, p99), and how many rows vi_search_exact scored exactly. */
#include "vecindex.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define EX_K   10
#define EX_DIM 100
#define EX_NBRUTE 200

static double ex_now_(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *ex_read_(const char *dir, const char *name, size_t *len)
{
    char p[1024];
    FILE *f;
    void *b;
    long n;
    sprintf(p, "%.900s/%s", dir, name);
    if (!(f = fopen(p, "rb"))) { fprintf(stderr, "exact: no %s\n", p); exit(1); }
    fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET);
    b = malloc((size_t)n);
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "exact: short read %s\n", p); exit(1); }
    fclose(f);
    *len = (size_t)n;
    return b;
}

static int ex_cmpd_(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static int ex_cmpl_(const void *a, const void *b)
{
    long x = *(const long *)a, y = *(const long *)b;
    return x < y ? -1 : x > y;
}

static void ex_lat_(const char *name, double *t, int n)
{
    qsort(t, (size_t)n, sizeof(double), ex_cmpd_);
    printf("  %-22s median %8.3f ms   p99 %8.3f ms   (%d queries)\n", name, t[n / 2] * 1e3, t[(n * 99) / 100] * 1e3, n);
}

static int ex_hits_(const vi_i64 *got, int n, const int *gt)
{
    int j, z, h = 0;
    for (j = 0; j < n; ++j) for (z = 0; z < EX_K; ++z) if (got[j] == gt[z]) { ++h; break; }
    return h;
}

int main(int argc, char **argv)
{
    size_t lt, lq, lg;
    float *tr, *qv, sc[EX_K], es[EX_K], bs[EX_K];
    int *gt, n, nq, i, j, k, h1 = 0, h2 = 0, h3 = 0, same = 0, nb;
    vi_i64 *ids, got[EX_K], eg[EX_K], chk;
    long *chks;
    double *t1, *t2, *t3, t0, mean = 0.0;
    vi_db *db;
    if (argc < 2) { fprintf(stderr, "usage: exact DIR [nq]\n"); return 2; }
    tr = (float *)ex_read_(argv[1], "train.bin", &lt);
    qv = (float *)ex_read_(argv[1], "query.bin", &lq);
    gt = (int *)ex_read_(argv[1], "gt.bin", &lg);
    n = (int)(lt / (EX_DIM * sizeof(float)));
    nq = (int)(lq / (EX_DIM * sizeof(float)));
    if (argc >= 3 && atoi(argv[2]) > 0 && atoi(argv[2]) < nq) nq = atoi(argv[2]);
    nb = nq < EX_NBRUTE ? nq : EX_NBRUTE;
    ids = (vi_i64 *)malloc((size_t)n * sizeof(vi_i64));
    chks = (long *)malloc((size_t)nq * sizeof(long));
    t1 = (double *)malloc((size_t)nq * sizeof(double)); t2 = (double *)malloc((size_t)nq * sizeof(double));
    t3 = (double *)malloc((size_t)nb * sizeof(double));
    for (i = 0; i < n; ++i) ids[i] = i;
    db = vi_open((const char *)0, EX_DIM, VI_MANUAL);
    t0 = ex_now_();
    if (!db || vi_add_batch(db, n, ids, tr) != n) { fprintf(stderr, "exact: %s\n", vi_error()); return 1; }
    vi_build(db);
    printf("exact: %d rows, %d queries, dim %d, built in %.1f s\n", n, nq, EX_DIM, ex_now_() - t0);
    for (i = 0; i < nq; ++i) {
        const float *q = qv + (size_t)i * EX_DIM;
        t0 = ex_now_(); k = vi_search(db, q, EX_K, got, sc); t1[i] = ex_now_() - t0;
        h1 += ex_hits_(got, k, gt + (size_t)i * EX_K);
        t0 = ex_now_(); k = vi_search_exact(db, q, EX_K, eg, es, &chk); t2[i] = ex_now_() - t0;
        h2 += ex_hits_(eg, k, gt + (size_t)i * EX_K);
        chks[i] = (long)chk; mean += (double)chk;
        if (i < nb) {                                            /* every row, f32, the store's own vectors */
            vi_i64 bi[EX_K];
            int r, z, ok = 1;
            for (j = 0; j < EX_K; ++j) { bs[j] = -3.0f; bi[j] = -1; }
            t0 = ex_now_();
            for (r = 0; r < n; ++r) {
                float s = 0.0f;
                const float *x = tr + (size_t)r * EX_DIM;
                s = vs_dot(x, q, EX_DIM);
                if (s <= bs[EX_K - 1]) continue;
                for (z = EX_K - 1; z > 0 && bs[z - 1] < s; --z) { bs[z] = bs[z - 1]; bi[z] = bi[z - 1]; }
                bs[z] = s; bi[z] = r;
            }
            t3[i] = ex_now_() - t0;
            h3 += ex_hits_(bi, EX_K, gt + (size_t)i * EX_K);
            for (j = 0; j < EX_K; ++j) if (k != EX_K || fabs((double)(es[j] - bs[j])) > 1e-5) ok = 0;
            same += ok;
        }
    }
    qsort(chks, (size_t)nq, sizeof(long), ex_cmpl_);
    printf("  recall@10 vs gt        vi_search %.4f   vi_search_exact %.4f   every row (f32, %d q) %.4f\n",
           (double)h1 / ((double)nq * EX_K), (double)h2 / ((double)nq * EX_K), nb, (double)h3 / ((double)nb * EX_K));
    printf("  vi_search_exact scores == every row's top 10 scores (1e-5) on %d/%d queries\n", same, nb);
    printf("  rows scored exactly    median %ld   mean %.0f   p99 %ld   max %ld   of %d (%.3f%%)\n",
           chks[nq / 2], mean / nq, chks[(nq * 99) / 100], chks[nq - 1], n, 100.0 * mean / nq / n);
    ex_lat_("vi_search", t1, nq);
    ex_lat_("vi_search_exact", t2, nq);
    ex_lat_("every row, f32", t3, nb);
    vi_close(db);
    return 0;
}
