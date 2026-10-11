/* cmp.c - head-to-head on the fixed dataset shared with the turbovec
 * (bench_turbovec_clustered.py) and Rust (examples/rawbench) benches:
 *   data_clustered.bin        dim=384 n=100000, 200 queries
 *   ground_truth_clustered.bin  cosine top-10
 *
 * Measures specific configs directly (not engine_build's 20-config
 * sweep, which is a one-time cost, not the comparison point):
 *   - flat q2 / q3 / q4        vs turbovec bit_width 2 / 3 / 4
 *   - IVF + q3 / q4 at nprobe {1, 8, 24}   vs Rust VecIndex
 * IVF is trained on a 20k subsample then assigned to all 100k (the
 * standard FAISS trick - ivf.c exposes ivf_assign_full).
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -D_POSIX_C_SOURCE=200112L -O2 \
 *       -I. cmp.c -lm -o cmp && ./cmp <data.bin> <gt.bin>
 */

#define PRINT_IMPLEMENTATION
#define ENGINE_THREADS
#include "engine.c"          /* q1..q5 + ivf + arena + thread pool */
#include "print.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static f64 now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (f64)ts.tv_sec + (f64)ts.tv_nsec * 1e-9;
}
static void die(const char *m) { fprintf(stderr, "cmp: %s\n", m); exit(1); }

static i32 g_dim, g_n, g_nq, g_k;
static f32 *g_vecs, *g_queries;
static i64 *g_gt;

static f32 recall_of_(i32 got, const i64 *ids, i32 qi)
{
    i32 e, x, hit = 0;
    for (e = 0; e < got; ++e)
        for (x = 0; x < g_k; ++x)
            if (ids[e] == g_gt[(size_t)qi * g_k + x]) { ++hit; break; }
    return (f32)hit;
}

static void row_flat_(Arena *a, EngScheme s)
{
    Arena_Mark m = arena_snapshot(a);
    void *ix; f64 tb, tq = 0.0, t0; i64 ids[16]; f32 sc[16];
    i32 qi; f32 hit = 0.0f;

    t0 = now_s(); ix = eng_flat_build_(a, s, g_vecs, g_n, g_dim);
    tb = (now_s() - t0) * 1e3;
    for (qi = 0; qi < g_nq; ++qi) {
        i32 got;
        t0 = now_s();
        got = eng_flat_search_(s, ix, g_queries + (size_t)qi * g_dim, g_k, ids, sc);
        tq += now_s() - t0;
        hit += recall_of_(got, ids, qi);
    }
    print_cstr("  flat "); print_cstr(engine_scheme_name(s));
    print_cstr("        "); print_i64(eng_bytes_(s, g_dim)); print_cstr(" B");
    print_cstr("   build "); print_i64((i64)(tb + 0.5)); print_cstr(" ms");
    print_cstr("   "); print_i64((i64)(tq / g_nq * 1e6 + 0.5)); print_cstr(" us/q");
    print_cstr("   recall@10 "); print_i64((i64)(100.0 * hit / (g_nq * g_k) + 0.5));
    print_cstr("%\n"); print_flush();
    arena_rewind(a, m);
}

static void rows_ivf_(Arena *a, EngScheme s, i32 train_n)
{
    Arena_Mark m = arena_snapshot(a);
    i32 nc = ivf_suggest_clusters(g_n), c, pj;
    Ivf *ivf;
    void **cix;
    f64 t0, tb;
    static const i32 NP[3] = { 1, 8, 24 };

    t0 = now_s();
    ivf = ivf_train(a, g_vecs, (train_n < g_n) ? train_n : g_n, g_dim, nc, 10, 0x5eedu);
    if (train_n < g_n) ivf_assign_full(ivf, g_vecs, g_n);
    cix = (void **)arena_alloc(a, (size_t)nc * sizeof(void *));
    for (c = 0; c < nc; ++c) {
        i32 mc; const i64 *mem = ivf_members(ivf, c, &mc);
        cix[c] = (mc > 0)
            ? eng_cluster_build_(a, s, g_vecs, mem, mc, g_dim,
                                 ivf->centroids + (size_t)c * g_dim)
            : (void *)0;
    }
    tb = (now_s() - t0) * 1e3;

    for (pj = 0; pj < 3; ++pj) {
        i32 nprobe = (NP[pj] < nc) ? NP[pj] : nc;
        EngIndex e;
        i64 *bids; f32 *bsc; f64 tq1 = 0.0, tqB;
        i32 qi; f32 hit1 = 0.0f, hitB = 0.0f;
        i64 s1_ids[16]; f32 s1_sc[16];

        e.arena = a; e.dim = g_dim; e.count = g_n;
        e.topo = ENG_IVF; e.scheme = s; e.nprobe = nprobe;
        e.code_bytes = eng_bytes_(s, g_dim); e.ivf = ivf; e.cluster_ix = cix;
        e.flat_ix = (void *)0;

        /* single-thread */
        for (qi = 0; qi < g_nq; ++qi) {
            i32 got; t0 = now_s();
            got = engine_search(&e, g_queries + (size_t)qi * g_dim, g_k, s1_ids, s1_sc);
            tq1 += now_s() - t0;
            hit1 += recall_of_(got, s1_ids, qi);
        }
        /* threaded batch (12 workers) */
        bids = (i64 *)arena_alloc(a, (size_t)g_nq * g_k * sizeof(i64));
        bsc  = (f32 *)arena_alloc(a, (size_t)g_nq * g_k * sizeof(f32));
        t0 = now_s();
        engine_search_batch(&e, g_queries, g_nq, g_k, bids, bsc);
        tqB = now_s() - t0;
        for (qi = 0; qi < g_nq; ++qi) {
            i32 x, e2, h = 0;
            for (e2 = 0; e2 < g_k; ++e2)
                for (x = 0; x < g_k; ++x)
                    if (bids[(size_t)qi * g_k + e2] == g_gt[(size_t)qi * g_k + x]) { ++h; break; }
            hitB += (f32)h;
        }

        print_cstr("  IVF+"); print_cstr(engine_scheme_name(s));
        print_cstr(" np="); print_i64(nprobe); if (nprobe < 10) print_cstr(" ");
        print_cstr("  "); print_i64(e.code_bytes); print_cstr(" B  build ");
        print_i64((i64)(tb + 0.5)); print_cstr(" ms   1T ");
        print_i64((i64)(tq1 / g_nq * 1e6 + 0.5)); print_cstr(" us/q   12T ");
        print_i64((i64)(tqB / g_nq * 1e6 + 0.5)); print_cstr(" us/q   recall@10 ");
        print_i64((i64)(100.0 * hitB / (g_nq * g_k) + 0.5)); print_cstr("%");
        if ((i64)(100.0 * hit1 / (g_nq * g_k) + 0.5) !=
            (i64)(100.0 * hitB / (g_nq * g_k) + 0.5)) print_cstr(" (!1T/12T recall differ)");
        print_cstr("\n"); print_flush();
    }
    arena_rewind(a, m);
}

int main(int argc, char **argv)
{
    static Arena arena;
    Arena *a = &arena;
    FILE *fd, *fg;
    u32 hdr[4], ghdr[2];
    i32 i;

    if (argc < 3) die("usage: cmp <data.bin> <ground_truth.bin>");
    fd = fopen(argv[1], "rb"); if (!fd) die("open data");
    fg = fopen(argv[2], "rb"); if (!fg) die("open gt");
    if (fread(hdr, 4, 4, fd) != 4) die("data header");
    g_dim = (i32)hdr[0]; g_n = (i32)hdr[1]; g_nq = (i32)hdr[2]; g_k = (i32)hdr[3];
    if (fread(ghdr, 4, 2, fg) != 2) die("gt header");
    if ((i32)ghdr[0] != g_nq || (i32)ghdr[1] != g_k) die("gt/data mismatch");

    g_vecs    = (f32 *)arena_alloc(a, (size_t)g_n * g_dim * sizeof(f32));
    g_queries = (f32 *)arena_alloc(a, (size_t)g_nq * g_dim * sizeof(f32));
    g_gt      = (i64 *)arena_alloc(a, (size_t)g_nq * g_k * sizeof(i64));
    if (fread(g_vecs, sizeof(f32), (size_t)g_n * g_dim, fd) != (size_t)g_n * g_dim) die("vecs");
    if (fread(g_queries, sizeof(f32), (size_t)g_nq * g_dim, fd) != (size_t)g_nq * g_dim) die("queries");
    {
        u64 *tmp = (u64 *)arena_alloc(a, (size_t)g_nq * g_k * sizeof(u64));
        if (fread(tmp, sizeof(u64), (size_t)g_nq * g_k, fg) != (size_t)g_nq * g_k) die("gt");
        for (i = 0; i < g_nq * g_k; ++i) g_gt[i] = (i64)tmp[i];
    }
    fclose(fd); fclose(fg);

    print_cstr("dataset  dim="); print_i64(g_dim); print_cstr(" n="); print_i64(g_n);
    print_cstr("  queries="); print_i64(g_nq);
    print_cstr("  (clustered, 200 blobs, cosine ground truth)\n\n");
    print_flush();

    row_flat_(a, ENG_Q2);
    row_flat_(a, ENG_Q3);
    row_flat_(a, ENG_Q4);
    print_cstr("\n"); print_flush();
    rows_ivf_(a, ENG_Q3, 20000);
    rows_ivf_(a, ENG_Q4, 20000);

    print_cstr("\nturbovec 0.8.0 TurboQuantIndex (flat, pip, multithreaded) same data:\n");
    print_cstr("  bit_width=2   96 B*  build 1026 ms   79 us/q   recall@10 13%\n");
    print_cstr("  bit_width=3  144 B*  build 1052 ms  162 us/q   recall@10 14%\n");
    print_cstr("  bit_width=4  192 B*  build 1296 ms  151 us/q   recall@10 14%\n");
    print_cstr("  (* nominal bit_width*dim/8)\n");
    print_cstr("\nRust vecindex VecIndex (IVF k-means + per-cluster ternary) same data:\n");
    print_cstr("  nprobe=1    81 B   build 118789 ms   274 us/q   recall@10 50%\n");
    print_cstr("  nprobe=8    81 B   build 118789 ms  1704 us/q   recall@10 56%\n");
    print_cstr("  nprobe=24   81 B   build 118789 ms  3662 us/q   recall@10 56%\n");
    print_flush();
    return 0;
}
