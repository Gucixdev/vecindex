/* graph_ablation.c - the same context budget, four ways to fill it: A the store alone (vi_context: MMR hits and their
 * chunks), B A's hits then the links they have (vg_context, VG_WALK_EXPAND), C personalized PageRank from them
 * (VG_WALK_PPR), D both walks fused by reciprocal rank (VG_WALK_BOTH).
 *
 * Synthetic, and built so that an answer needs one hop: P questions, each with two chunks of evidence - A_i, near
 * the question's vector, and B_i, a random vector anywhere, linked A_i -references-> B_i - among N chunks, every one of
 * which also has two weak links (weight 0.3) to random others: noise. A context of 4 chunks (256 bytes) is "complete"
 * when it holds both A_i and B_i. It measures what a walk costs and that the budget takes the hit's link before
 * weaker hits - not that a graph helps on real data, where the links are what a caller makes of its documents.
 *
 *   gcc -std=c89 -O2 -march=native -Isrc -Iinclude bench/graph_ablation.c src/vecindex.c -lm -lpthread -o ga && ./ga */
#include "vecindex.h"
#include "graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define DIM 64
#define N 20000
#define P 500
#define CH 64                       /* bytes a chunk's text */
#define BUDGET (4 * CH)

static unsigned rng_ = 4242u;
static double uni_(void) { rng_ = rng_ * 1103515245u + 12345u; return ((double)((rng_ >> 8) & 0xffffff) + 0.5) / 16777216.0; }
static float gau_(void) { return (float)(sqrt(-2.0 * log(uni_())) * cos(6.283185307179586 * uni_())); }
static void rvec_(float *v) { int d; for (d = 0; d < DIM; ++d) v[d] = gau_(); }

static double now_(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec; }
static int cmpd_(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }

/* chunk i's id: its own document (shift 20), chunk 0; the pairs are rows 0..2P-1: A_i = 2i, B_i = 2i + 1 */
#define ID(i) ((vi_i64)(i) << 20)

int main(void)
{
    static float vs[N][DIM];
    float q[DIM];
    char t[CH + 1], buf[BUDGET];
    vi_db *db = vi_open((const char *)0, DIM, 0);
    vg_graph *g = vg_open((const char *)0);
    vi_ctx_opts co;
    vg_ctx_opts go;
    vi_ctx_span sp[16];
    static double lat[4][P];
    const char *name[4] = { "A store alone (vi_context)", "B + expansion (VG_WALK_EXPAND)", "C + PPR (VG_WALK_PPR)", "D + both, fused (VG_WALK_BOTH)" };
    int i, d, v, k, n, used, full[4], half[4];
    long bytes[4];
    if (!db || !g) return 1;
    for (i = 0; i < N; ++i) {
        rvec_(vs[i]);
        sprintf(t, "chunk %06d", i);
        memset(t + strlen(t), '.', (size_t)(CH - strlen(t))); t[CH] = 0;
        vi_add_text(db, ID(i), vs[i], t, CH);
    }
    for (i = 0; i < P; ++i) vg_link(g, ID(2 * i), ID(2 * i + 1), 1, 1.0f);
    for (i = 0; i < N; ++i) for (k = 0; k < 2; ++k) {
        int j = (int)(uni_() * N) % N;
        if (j != i) vg_link(g, ID(i), ID(j), 0, 0.3f);
    }
    vi_build(db);
    vi_ctx_defaults(&co); co.k = 4; co.before = co.after = 0;
    for (v = 0; v < 4; ++v) { full[v] = half[v] = 0; bytes[v] = 0; }
    for (i = 0; i < P; ++i) {
        for (d = 0; d < DIM; ++d) q[d] = vs[2 * i][d] + 0.5f * gau_();
        for (v = 0; v < 4; ++v) {
            int a = 0, b = 0, s;
            double t0 = now_();
            if (v == 0) n = vi_context(db, q, (const char *)0, 0, &co, sp, 16, buf, BUDGET, &used);
            else { vg_ctx_defaults(&go); go.walk = v; go.hits = 8; n = vg_context(g, db, q, (const char *)0, 0, &co, &go, sp, 16, buf, BUDGET, &used); }
            lat[v][i] = now_() - t0;
            for (s = 0; s < n; ++s) { a |= sp[s].hit == ID(2 * i); b |= sp[s].hit == ID(2 * i + 1); }
            full[v] += a && b; half[v] += a + b; bytes[v] += used;
        }
    }
    printf("graph ablation: %d chunks of %d bytes, %d two-hop questions, a budget of %d bytes (4 chunks)\n", N, CH, P, BUDGET);
    printf("%-32s %10s %14s %8s %10s %10s\n", "variant", "complete", "evidence", "bytes", "p50 us", "p95 us");
    for (v = 0; v < 4; ++v) {
        qsort(lat[v], P, sizeof(double), cmpd_);
        printf("%-32s %9.1f%% %13.1f%% %8ld %10.0f %10.0f\n", name[v], 100.0 * full[v] / P, 50.0 * half[v] / P, bytes[v] / P,
               1e6 * lat[v][P / 2], 1e6 * lat[v][P * 95 / 100]);
    }
    vg_close(g); vi_close(db);
    return 0;
}
