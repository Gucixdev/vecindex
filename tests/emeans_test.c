/* emeans_test.c - the store's e-means clustering (tq.c, tq_emeans_) against Lloyd's and against itself.
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -D_POSIX_C_SOURCE=200112L -O2 -Isrc tests/emeans_test.c -lm -lpthread
 *   ./emeans_test                  the checks below, one "ok" line each
 *   ./emeans_test recipe           reads "n k T d" lines, prints "B alpha0 a(s)..." for tests/emeans_ref.py
 *   ./emeans_test train FILE k T [seed]  e-means and Lloyd on FILE (int32 n, int32 d, n x d f32 unit rows), prints both errors
 *
 * The checks: the update in position form (what tq_emeans_ does) equals the sum-and-count EMA of the paper, a = 1 on a
 * whole pass is one Lloyd round (both also proved in lean4/Vecindex/Emeans.lean, over a field), revival halves a busy
 * cluster into an empty one, the same seed gives the same centroids, and on unevenly clustered data e-means in 3
 * passes ends below Lloyd's 8 rounds and finds no fewer of the true clusters. */
#include "store.c"           /* tq.c and the rest, as the store builds it */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static i32 fails = 0;
#define CHECK(c, msg) do { if (c) printf("  ok   %s\n", msg); else { printf("  FAIL %s (line %d)\n", msg, __LINE__); ++fails; } } while (0)

static f64 et_err_(const f32 *x, i32 n, i32 d, const f32 *c, i32 k)
{
    f64 e = 0.0;
    i32 i, j;
    for (i = 0; i < n; ++i) {
        f32 b = -3e38f;
        for (j = 0; j < k; ++j) { f32 s = vs_dot(x + (size_t)i * d, c + (size_t)j * d, d); if (s > b) b = s; }
        e += 2.0 - 2.0 * b;
    }
    return e / n;
}

/* k unit centres, per rows around each (noise sd sig), unit rows, rows in cluster order */
static f32 *et_data_(i32 k, i32 per, i32 d, f32 sig, u32 seed, f32 *centres)
{
    f32 *x = (f32 *)malloc((size_t)k * per * d * sizeof(f32));
    u32 s = seed;
    i32 c, i, j;
    for (c = 0; c < k; ++c) {
        f64 q = 0.0;
        for (j = 0; j < d; ++j) { f32 v = (f32)((f64)tq_rng_(&s) / 4294967296.0 - 0.5); centres[(size_t)c * d + j] = v; q += (f64)v * v; }
        for (j = 0; j < d; ++j) centres[(size_t)c * d + j] = (f32)(centres[(size_t)c * d + j] / sqrt(q));
    }
    for (c = 0; c < k; ++c) for (i = 0; i < per; ++i) {
        f32 *r = x + ((size_t)c * per + i) * d;
        f64 q = 0.0;
        for (j = 0; j < d; ++j) {                            /* a sum of 4 uniforms: roughly normal, sd sig */
            f64 g = ((f64)tq_rng_(&s) + tq_rng_(&s) + tq_rng_(&s) + tq_rng_(&s)) / 4294967296.0 - 2.0;
            r[j] = (f32)(centres[(size_t)c * d + j] + sig * g * 1.7320508);
            q += (f64)r[j] * r[j];
        }
        for (j = 0; j < d; ++j) r[j] = (f32)(r[j] / sqrt(q));
    }
    return x;
}

static void et_algebra_(void)
{
    /* tq_em_update_ on one cluster with mass N at position S / N, a batch of n rows summing to s: the result must be
     * the paper's sum-and-count average, ((1-a) S + a s) / ((1-a) N + a n), with mass (1-a) N + a n */
    u32 s = 7u;
    i32 t, bad = 0, badm = 0, badw = 0;
    for (t = 0; t < 10000; ++t) {
        f64 N = (f64)(tq_rng_(&s) % 1000) / 10.0 + 1e-3, S = ((f64)tq_rng_(&s) / 4294967296.0 - 0.5) * N;
        i32 n = (i32)(1 + tq_rng_(&s) % 500);
        f64 sb = ((f64)tq_rng_(&s) / 4294967296.0 - 0.5) * n, a = ((f64)(tq_rng_(&s) % 999) + 1.0) / 1000.0;
        f64 ref = ((1.0 - a) * S + a * sb) / ((1.0 - a) * N + a * n), lo, hi;
        f32 pos = (f32)(S / N), mass = (f32)N, sum = (f32)sb, c0 = pos, mean = (f32)(sb / n);
        tq_em_update_(&pos, &mass, &sum, &n, 1, 1, a, (f64 *)0, (f64 *)0);
        if (fabs(pos - ref) > 1e-4 * (1.0 + fabs(ref))) ++bad;
        if (fabs(mass - ((1.0 - a) * N + a * n)) > 1e-4 * (1.0 + mass)) ++badm;
        lo = c0 < mean ? c0 : mean; hi = c0 < mean ? mean : c0;
        if (pos < lo - 1e-6 || pos > hi + 1e-6) ++badw;
    }
    CHECK(bad == 0, "update = ((1-a) S + a s) / ((1-a) N + a n), the paper's average, 10000 random cases");
    CHECK(badm == 0, "update: the new mass is (1-a) N + a n");
    CHECK(badw == 0, "update: a centroid moves towards its batch mean and never past it");
}

static void et_lloyd_step_(void)
{
    /* one batch = every row, a = 1: tq_em_update_ makes each centroid the plain mean of its rows, one Lloyd round */
    i32 k = 6, per = 300, d = 24, n = k * per, i, j, c, ok = 1, cnt[6] = { 0, 0, 0, 0, 0, 0 };
    f32 *cen = (f32 *)malloc((size_t)k * d * sizeof(f32)), *x = et_data_(k, per, d, 0.05f, 11u, cen);
    f32 *pos = (f32 *)malloc((size_t)k * d * sizeof(f32)), *sum = (f32 *)calloc((size_t)k * d, sizeof(f32)), mass[6];
    for (c = 0; c < k; ++c) { memcpy(pos + (size_t)c * d, x + (size_t)(c * per + 17) * d, (size_t)d * sizeof(f32)); mass[c] = 3.0f + c; }
    for (i = 0; i < n; ++i) {
        i32 b = 0; f32 bs = -3e38f;
        for (c = 0; c < k; ++c) { f32 v = vs_dot(x + (size_t)i * d, pos + (size_t)c * d, d); if (v > bs) { bs = v; b = c; } }
        for (j = 0; j < d; ++j) sum[(size_t)b * d + j] += x[(size_t)i * d + j];
        ++cnt[b];
    }
    tq_em_update_(pos, mass, sum, cnt, k, d, 1.0, (f64 *)0, (f64 *)0);
    for (c = 0; c < k; ++c) for (j = 0; j < d; ++j) if (cnt[c] && pos[(size_t)c * d + j] != (f32)(sum[(size_t)c * d + j] * (1.0 / cnt[c]))) ok = 0;
    for (c = 0; c < k; ++c) if (cnt[c] && mass[c] != (f32)cnt[c]) ok = 0;
    CHECK(ok, "a = 1 on the whole set is one Lloyd round: each centroid the mean of its rows, its mass their count");
    free(cen); free(x); free(pos); free(sum);
}

static void et_revive_(void)
{
    i32 k = 8, d = 6, j, z, okm = 1, okp = 1, n;
    f32 c[48], mass[8], keep[48];
    u32 r = 3u;
    TqEmKey key[8];
    for (j = 0; j < k * d; ++j) c[j] = (f32)(j % 7) - 3.0f;
    memcpy(keep, c, sizeof c);
    mass[0] = 100.0f; mass[1] = 80.0f; mass[2] = 0.0f; mass[3] = 60.0f; mass[4] = 0.5f; mass[5] = 90.0f; mass[6] = 70.0f; mass[7] = 0.0f;
    n = tq_em_revive_(c, mass, k, d, &r, key);             /* mean mass 50: under 5 are 2, 4 and 7 */
    CHECK(n == 3, "revival: the three clusters under a tenth of the mean mass are revived");
    for (j = 0; j < k; ++j) if (j == 2 || j == 4 || j == 7) {
        i32 src = -1, q;
        for (q = 0; q < k; ++q) if (q != j && mass[q] == mass[j] && q != 2 && q != 4 && q != 7) src = q;
        if (src < 0 || !(mass[j] > 20.0f)) okm = 0;
        else for (z = 0; z < d; ++z) {
            f32 e = (z & 1) ? -TQ_EM_SPLIT_EPS : TQ_EM_SPLIT_EPS, v = keep[(size_t)src * d + z];
            if (c[(size_t)j * d + z] != v * (1.0f + e) || c[(size_t)src * d + z] != v * (1.0f - e)) okp = 0;
        }
    }
    CHECK(okm, "revival: each revived cluster takes half the mass of a busy one, and that one keeps the other half");
    CHECK(okp, "revival: the two halves sit 1e-4 on either side of the donor");
    mass[0] = mass[1] = mass[2] = mass[3] = mass[4] = mass[5] = mass[6] = mass[7] = 10.0f;
    CHECK(tq_em_revive_(c, mass, k, d, &r, key) == 0, "revival: equal masses, nothing to revive");
}

static void et_recipe_(void)
{
    i32 B; f64 a0, a;
    tq_em_recipe_(1000000, 1000, 3.0, 100, &B, &a0);
    CHECK((B & (B - 1)) == 0 && B > 1, "recipe: the batch is a power of two");
    tq_em_recipe_(100, 50, 3.0, 100, &B, &a0);
    CHECK(B <= 64 && B >= 1, "recipe: the batch stays at or under n / 2 rounded to a power of two");
    tq_em_recipe_(10000000, 2, 1.0, 100, &B, &a0);
    CHECK(a0 >= 1e-4 && a0 <= 1.0, "recipe: the first step is clamped to [1e-4, 1]");
    a = tq_em_alpha_(0.5, 0, 100);
    CHECK(fabs(a - 0.5) < 1e-12 && fabs(tq_em_alpha_(0.5, 100, 100) - 0.05) < 1e-12, "schedule: a0 at the start, a0 / 10 at the end");
}

static void et_cluster_(void)
{
    /* 40 clusters of uneven size (100 to 1300 rows), rows shuffled, so a seeding draws several rows of a big cluster
     * and none of some small ones: Lloyd's rounds keep such a start, revival is what fixes it */
    i32 k = 40, per = 1300, d = 64, c, j, i, n = 0, coverE = 0, coverL = 0, *perm;
    f32 *cen = (f32 *)malloc((size_t)k * d * sizeof(f32)), *all = et_data_(k, per, d, 0.06f, 2026u, cen), *x;
    f32 *ce = (f32 *)malloc((size_t)k * d * sizeof(f32)), *ce2 = (f32 *)malloc((size_t)k * d * sizeof(f32)), *cl = (f32 *)malloc((size_t)k * d * sizeof(f32));
    u32 r1 = 5u, r2 = 5u, r3 = 5u, sh = 99u;
    f64 ee, el, eo;
    char msg[240];
    x = (f32 *)malloc((size_t)k * per * d * sizeof(f32));
    for (c = 0; c < k; ++c) {                               /* cluster c keeps 100 + 30 c of its rows */
        i32 keep = 100 + 30 * c;
        memcpy(x + (size_t)n * d, all + (size_t)c * per * d, (size_t)keep * d * sizeof(f32)); n += keep;
    }
    perm = (i32 *)malloc((size_t)n * sizeof(i32));
    for (i = 0; i < n; ++i) perm[i] = i;
    for (i = n - 1; i > 0; --i) { i32 q = (i32)(tq_rng_(&sh) % (u32)(i + 1)), z = perm[i]; perm[i] = perm[q]; perm[q] = z; }
    for (i = 0; i < n; ++i) memcpy(all + (size_t)i * d, x + (size_t)perm[i] * d, (size_t)d * sizeof(f32));
    tq_emeans_(all, n, d, k, &r1, ce, (i32 *)0, 2, 3.0);
    tq_emeans_(all, n, d, k, &r2, ce2, (i32 *)0, 1, 3.0);
    CHECK(!memcmp(ce, ce2, (size_t)k * d * sizeof(f32)), "the same seed gives the same centroids, on 1 thread or 2");
    tq_kmeans_(all, n, d, k, &r3, cl, (i32 *)0, 2);
    ee = et_err_(all, n, d, ce, k); el = et_err_(all, n, d, cl, k); eo = et_err_(all, n, d, cen, k);
    for (c = 0; c < k; ++c) {                               /* true centres with a centroid close by */
        f32 be = -3e38f, bl = -3e38f;
        for (j = 0; j < k; ++j) {
            f32 s1 = vs_dot(cen + (size_t)c * d, ce + (size_t)j * d, d), s2 = vs_dot(cen + (size_t)c * d, cl + (size_t)j * d, d);
            if (s1 > be) be = s1;
            if (s2 > bl) bl = s2;
        }
        coverE += be >= 0.98f; coverL += bl >= 0.98f;
    }
    sprintf(msg, "uneven clusters, 3 passes: e-means error %.5f < Lloyd's 8 rounds %.5f (true centres %.5f)", ee, el, eo);
    CHECK(ee < el, msg);
    sprintf(msg, "e-means finds %d of the 40 true centres, Lloyd %d (within cos 0.98): no fewer", coverE, coverL);
    CHECK(coverE >= coverL && coverE >= k * 3 / 4, msg);
    free(cen); free(all); free(x); free(ce); free(ce2); free(cl); free(perm);
}

/* clusters of different spread (sd 0.02 to 0.06) and size, rows shuffled or in cluster order */
static f32 *et_spread_(i32 k, i32 d, u32 seed, i32 shuffled, i32 *nout, f32 *cen)
{
    i32 c, i, j, n = 0, *perm;
    f32 *x, *y;
    u32 s = seed;
    for (c = 0; c < k; ++c) n += 150 + 25 * c;
    x = (f32 *)malloc((size_t)n * d * sizeof(f32)); y = (f32 *)malloc((size_t)n * d * sizeof(f32));
    for (c = 0, n = 0; c < k; ++c) {
        f32 *one = et_data_(1, 150 + 25 * c, d, 0.02f + 0.04f * c / (k - 1), seed + 17u * c, cen + (size_t)c * d);
        memcpy(x + (size_t)n * d, one, (size_t)(150 + 25 * c) * d * sizeof(f32)); n += 150 + 25 * c;
        free(one);
    }
    perm = (i32 *)malloc((size_t)n * sizeof(i32));
    for (i = 0; i < n; ++i) perm[i] = i;
    if (shuffled) for (i = n - 1; i > 0; --i) { i32 q = (i32)(tq_rng_(&s) % (u32)(i + 1)), z = perm[i]; perm[i] = perm[q]; perm[q] = z; }
    for (i = 0; i < n; ++i) for (j = 0; j < d; ++j) y[(size_t)i * d + j] = x[(size_t)perm[i] * d + j];
    free(x); free(perm);
    *nout = n;
    return y;
}

/* how far one spherical Lloyd round (assign every row, mean, rescale) moves the centroids: 0 at a fixed point */
static f64 et_lloyd_move_(const f32 *x, i32 n, i32 d, const f32 *c, i32 k)
{
    f64 *sum = (f64 *)calloc((size_t)k * d, sizeof(f64)), mv = 0.0;
    i32 i, j, z, *cnt = (i32 *)calloc((size_t)k, sizeof(i32));
    for (i = 0; i < n; ++i) {
        i32 b = 0; f32 bs = -3e38f;
        for (j = 0; j < k; ++j) { f32 v = vs_dot(x + (size_t)i * d, c + (size_t)j * d, d); if (v > bs) { bs = v; b = j; } }
        for (z = 0; z < d; ++z) sum[(size_t)b * d + z] += x[(size_t)i * d + z];
        ++cnt[b];
    }
    for (j = 0; j < k; ++j) {
        f64 q = 0.0, dd = 0.0;
        if (!cnt[j]) continue;
        for (z = 0; z < d; ++z) q += sum[(size_t)j * d + z] * sum[(size_t)j * d + z];
        q = 1.0 / sqrt(q);
        for (z = 0; z < d; ++z) { f64 e = sum[(size_t)j * d + z] * q - c[(size_t)j * d + z]; dd += e * e; }
        if (sqrt(dd) > mv) mv = sqrt(dd);
    }
    free(sum); free(cnt);
    return mv;
}

static void et_closing_(void)
{
    /* the result is the last pass's exact means: on clusters it has settled into, one more Lloyd round barely moves
     * a centroid; the moving averages (no closing step), or sums over every pass, or centroids used at their raw
     * norm, are no fixed point. And rows given in cluster order are shuffled, not taken a cluster at a time. */
    i32 k = 30, d = 32, n, n2;
    f32 *cen = (f32 *)malloc((size_t)k * d * sizeof(f32)), *x = et_spread_(k, d, 77u, 1, &n, cen);
    f32 *y = et_spread_(k, d, 77u, 0, &n2, cen), *ce = (f32 *)malloc((size_t)k * d * sizeof(f32));
    u32 r = 9u;
    f64 mv, e1, e2;
    char msg[200];
    tq_emeans_(x, n, d, k, &r, ce, (i32 *)0, 1, 3.0);
    mv = et_lloyd_move_(x, n, d, ce, k); e1 = et_err_(x, n, d, ce, k);
    sprintf(msg, "closing means: one more Lloyd round moves a centroid by %.2g at most (< 0.012; without them 0.025)", mv);
    CHECK(mv < 0.012, msg);
    r = 9u;
    tq_emeans_(y, n2, d, k, &r, ce, (i32 *)0, 1, 3.0);
    e2 = et_err_(y, n2, d, ce, k);
    sprintf(msg, "rows in cluster order: error %.5f, within 20%% of the shuffled rows' %.5f (unshuffled batches: +75%%)", e2, e1);
    CHECK(e2 <= e1 * 1.2, msg);
    free(cen); free(x); free(y); free(ce);
}

static void et_golden_(void)
{
    /* batch and first step from Meta's reference (base._resolve_recipe, faiss d1d9789), copied here so the C build
     * checks them without Python; tests/emeans_ref.py compares a grid of 700 against the live reference */
    static const struct { i32 n, k; f64 T; i32 d, B; f64 a0; } g[] = {
        { 1110000, 4350, 3.0, 100, 65536, 0.6351939928447485 }, { 17345, 67, 3.0, 100, 1024, 0.6698845770553654 },
        { 65536, 1024, 2.5, 768, 4096, 0.37151403282223583 },   { 200001, 16, 5.0, 16, 1024, 0.409339297446246 },
        { 1000, 2, 1.0, 2, 2, 0.049976948939926616 },           { 66560, 65, 3.0, 100, 2048, 0.692457586098699 } };
    i32 i, bad = 0, B;
    f64 a0;
    for (i = 0; i < (i32)(sizeof g / sizeof g[0]); ++i) {
        tq_em_recipe_(g[i].n, g[i].k, g[i].T, g[i].d, &B, &a0);
        if (B != g[i].B || fabs(a0 - g[i].a0) > 1e-12 * g[i].a0) { ++bad; printf("       n=%d k=%d: B %d a0 %.17g\n", g[i].n, g[i].k, B, a0); }
    }
    CHECK(bad == 0, "recipe: batch and first step equal Meta's reference on 6 recorded cases");
    {   /* the rows of each pass, against the reference's _epoch_rows: 1000 rows, 2.5 / 3 / 0.5 / 1.25 passes */
        i64 r[64];
        i32 ok = tq_em_passes_(1000, 2.5, r) == 3 && r[0] == 1000 && r[1] == 500 && r[2] == 1000;
        ok &= tq_em_passes_(1000, 3.0, r) == 3 && r[0] == 1000 && r[1] == 1000 && r[2] == 1000;
        ok &= tq_em_passes_(1000, 0.5, r) == 1 && r[0] == 500;
        ok &= tq_em_passes_(1000, 1.25, r) == 2 && r[0] == 250 && r[1] == 1000;
        CHECK(ok, "passes: whole passes, the fractional share, then a whole closing pass (as the reference)");
    }
}

static i32 et_recipe_io_(void)
{
    char line[256];
    while (fgets(line, sizeof line, stdin)) {
        i64 n, k; f64 T; i32 d, B; f64 a0;
        if (sscanf(line, "%ld %ld %lf %d", (long *)&n, (long *)&k, &T, &d) != 4) continue;
        tq_em_recipe_((i32)n, (i32)k, T, d, &B, &a0);
        printf("%d %.17g %.17g %.17g\n", B, a0, tq_em_alpha_(a0, 7, 20), tq_em_alpha_(a0, 20, 20));
    }
    return 0;
}

static i32 et_train_io_(const char *file, i32 k, f64 T, u32 seed)
{
    FILE *f = fopen(file, "rb");
    i32 hd[2];
    f32 *x, *c;
    u32 r = seed;
    if (!f || fread(hd, sizeof(i32), 2, f) != 2) { fprintf(stderr, "emeans_test: cannot read %s\n", file); return 1; }
    x = (f32 *)malloc((size_t)hd[0] * hd[1] * sizeof(f32)); c = (f32 *)malloc((size_t)k * hd[1] * sizeof(f32));
    if (fread(x, sizeof(f32), (size_t)hd[0] * hd[1], f) != (size_t)hd[0] * hd[1]) { fprintf(stderr, "emeans_test: short %s\n", file); return 1; }
    fclose(f);
    tq_emeans_(x, hd[0], hd[1], k, &r, c, (i32 *)0, 2, T);
    printf("emeans %.6f", et_err_(x, hd[0], hd[1], c, k));
    r = seed; tq_kmeans_(x, hd[0], hd[1], k, &r, c, (i32 *)0, 2);
    printf(" lloyd %.6f\n", et_err_(x, hd[0], hd[1], c, k));
    return 0;
}

int main(int argc, char **argv)
{
    if (argc >= 2 && !strcmp(argv[1], "recipe")) return et_recipe_io_();
    if (argc >= 5 && !strcmp(argv[1], "train")) return et_train_io_(argv[2], atoi(argv[3]), atof(argv[4]), argc >= 6 ? (u32)atoi(argv[5]) : 1u);
    printf("emeans_test\n");
    et_algebra_();
    et_lloyd_step_();
    et_revive_();
    et_recipe_();
    et_golden_();
    et_cluster_();
    et_closing_();
    printf("%s\n", fails ? "FAILED" : "all ok");
    return fails ? 1 : 0;
}
