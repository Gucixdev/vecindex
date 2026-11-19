/* lorentz_test.c - VI_LORENTZ through the C API: points of the hyperboloid model, the score the Lorentz product,
 * the ranking against brute force by the true hyperbolic distance (exact tail, built index, certified search), the
 * vectors back as they came, refusals off the upper sheet, and the metric kept by the file. ./run builds it against
 * libvecindex.a.
 *
 * The data: N points x = (cosh r, sinh r u), u a random unit direction in 32 dimensions and r (the distance from the
 * origin) spread from 0.1 to 4 - near points general, far points specific, as a hierarchy lies in hyperbolic space. */
#include "vecindex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#define SD 32                       /* space dimensions */
#define DIM (SD + 1)                /* and the time coordinate */
#define N 20000
#define NQ 50
#define K 10

static int fails;

static void check(int cond, const char *what)
{
    printf(cond ? "  ok   %s\n" : "  FAIL %s\n", what);
    if (!cond) { ++fails; printf("       (%s)\n", vi_error()); }
}

static unsigned rng_ = 777u;
static double uni_(void) { rng_ = rng_ * 1103515245u + 12345u; return ((double)((rng_ >> 8) & 0xffffff) + 0.5) / 16777216.0; }
static double gau_(void) { return sqrt(-2.0 * log(uni_())) * cos(6.283185307179586 * uni_()); }

static void point_(float *x)
{
    double u[SD], l = 0.0, r = 0.1 + 3.9 * uni_();
    int i;
    for (i = 0; i < SD; ++i) { u[i] = gau_(); l += u[i] * u[i]; }
    l = sqrt(l);
    x[0] = (float)cosh(r);
    for (i = 0; i < SD; ++i) x[1 + i] = (float)(sinh(r) * u[i] / l);
}

static double lprod_(const float *a, const float *b)
{
    double s = -(double)a[0] * b[0];
    int i;
    for (i = 1; i < DIM; ++i) s += (double)a[i] * b[i];
    return s;
}

/* the K nearest by the hyperbolic distance acosh(-<q, x>_L), nearest first */
static void brute_(const float *xs, const float *q, vi_i64 *ids, double *d)
{
    int i, j, n = 0;
    for (i = 0; i < N; ++i) {
        double p = -lprod_(q, xs + (size_t)i * DIM), dist;
        if (p < 1.0) p = 1.0;
        dist = log(p + sqrt(p * p - 1.0));   /* acosh, which C89 lacks */
        if (n == K && dist >= d[K - 1]) continue;
        j = n < K ? n++ : K - 1;
        while (j > 0 && d[j - 1] > dist) { d[j] = d[j - 1]; ids[j] = ids[j - 1]; --j; }
        d[j] = dist; ids[j] = i;
    }
}

static int overlap_(const vi_i64 *a, int na, const vi_i64 *b, int nb)
{
    int i, j, h = 0;
    for (i = 0; i < na; ++i) for (j = 0; j < nb; ++j) if (a[i] == b[j]) { ++h; break; }
    return h;
}

int main(void)
{
    float *xs = (float *)malloc((size_t)N * DIM * sizeof(float)), *qs = (float *)malloc((size_t)NQ * DIM * sizeof(float));
    float sc[K], v[DIM], bad[DIM];
    vi_i64 ids[K], bid[K], chk;
    double bd[K];
    char path[128], line[256];
    vi_db *db;
    int i, q, n, ok, hit, same;

    if (!xs || !qs) return 1;
    for (i = 0; i < N; ++i) point_(xs + (size_t)i * DIM);
    for (q = 0; q < NQ; ++q) point_(qs + (size_t)q * DIM);
    sprintf(path, "/tmp/vecindex_lorentz_%d.vxi", (int)getpid());
    remove(path);
    db = vi_open(path, DIM, VI_LORENTZ | VI_MANUAL);
    check(db != (vi_db *)0, "vi_open with VI_LORENTZ");
    if (!db) return 1;

    memcpy(bad, xs, sizeof(bad)); bad[0] = -bad[0];
    check(vi_add(db, 1, bad) == -1, "a point of the lower sheet (x0 < 0): refused");
    memcpy(bad, xs, sizeof(bad)); bad[0] = 0.5f * (float)sqrt(lprod_(bad, bad) + (double)bad[0] * bad[0]);
    check(vi_add(db, 1, bad) == -1, "a spacelike vector (x0^2 < |x|^2): refused");
    for (i = 0, ok = 1; i < N; ++i) ok &= vi_add(db, i, xs + (size_t)i * DIM) == 0;
    check(ok, "20000 points of the hyperboloid added");
    check(vi_get(db, 7, v) == 0 && fabs(v[0] - xs[7 * DIM]) < 1e-4f * xs[7 * DIM] && fabs(v[3] - xs[7 * DIM + 3]) < 1e-4f * xs[7 * DIM],
          "vi_get: the vector as it came, x0 positive");

    /* not built: the exact scan of every row */
    for (q = 0, ok = 1, same = 1; q < NQ; ++q) {
        const float *qq = qs + (size_t)q * DIM;
        n = vi_search(db, qq, K, ids, sc);
        brute_(xs, qq, bid, bd);
        if (n != K || overlap_(ids, n, bid, K) < K - 1) ok = 0;   /* ties of float rounding aside */
        for (i = 0; i < n; ++i) {
            double p = -lprod_(qq, xs + (size_t)ids[i] * DIM), want;
            if (p < 1.0) p = 1.0;
            want = log(p + sqrt(p * p - 1.0));
            if (fabs(sc[i] - want) > 2e-3 * (want > 1.0 ? want : 1.0)) same = 0;
            if (i && sc[i] < sc[i - 1]) same = 0;
        }
    }
    check(ok, "before a build: the 10 nearest by hyperbolic distance (brute force), every query");
    check(same, "the scores are the hyperbolic distances, nearest first");

    /* the built index: candidates by direction over the negated rows, re-scored exactly */
    vi_build(db);
    for (q = 0, hit = 0; q < NQ; ++q) {
        const float *qq = qs + (size_t)q * DIM;
        n = vi_search(db, qq, K, ids, sc);
        brute_(xs, qq, bid, bd);
        hit += overlap_(ids, n, bid, K);
    }
    sprintf(line, "after vi_build: recall@10 against brute force %d/%d (measured, the index unchanged)", hit, NQ * K);
    check(hit >= NQ * K * 8 / 10, line);
    for (q = 0, hit = 0; q < NQ; ++q) {
        const float *qq = qs + (size_t)q * DIM;
        n = vi_search_exact(db, qq, K, ids, sc, &chk);
        brute_(xs, qq, bid, bd);
        hit += overlap_(ids, n, bid, K);
    }
    sprintf(line, "vi_search_exact: %d/%d of the brute-force nearest (its bound: the score a line in the cosine)", hit, NQ * K);
    check(hit >= NQ * (K - 1), line);

    /* the file keeps the metric */
    check(vi_save(db, (const char *)0) == 0, "vi_save");
    vi_close(db);
    check(vi_open(path, DIM, VI_DOT) == (vi_db *)0 && strstr(vi_error(), "Lorentz") != (const char *)0, "reopened as VI_DOT: refused, a Lorentz store");
    db = vi_open(path, DIM, 0);
    check(db != (vi_db *)0, "reopened without a flag: the file's metric");
    if (db) {
        n = vi_search(db, qs, K, ids, sc);
        brute_(xs, qs, bid, bd);
        check(n == K && overlap_(ids, n, bid, K) >= K - 2 && vi_get(db, 7, v) == 0 && v[0] > 0.0f, "and it searches and gives vectors as before");
        vi_close(db);
    }
    remove(path);
    sprintf(line, "%s.j", path); remove(line);
    free(xs); free(qs);
    printf("Lorentz: %s\n", fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}
