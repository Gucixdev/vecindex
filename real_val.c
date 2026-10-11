/* real_val.c - vecindex on a REAL RAG retrieval benchmark.
 *
 * Corpus: 18.9k unique SQuAD Wikipedia passages + 30k AG-News snippets,
 * embedded with sentence-transformers/all-MiniLM-L6-v2 (384-dim, unit
 * norm). Queries: 1000 SQuAD questions - a genuine asymmetric
 * question -> passage task, not synthetic near-duplicates.
 *
 * Built from emb/real_gen.py (see that script). Not part of `./run`
 * unless emb/real_data.bin is present.
 *
 *   real_data.bin  <IIII dim n nq k>  f32[n*dim]  f32[nq*dim]
 *   real_gt.bin    <II nq k>  u64[nq*k]      exact cosine top-k
 *   real_gold.bin  <I nq>     i32[nq]        gold passage row per query
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -Werror -O2 -march=native -I. \
 *       real_val.c -lm -lpthread -o real_val && ./real_val emb
 */
#define _GNU_SOURCE
#define ARENA_ALLOC_IMPLEMENTATION
#define ENGINE_THREADS
#define ENGINE_SERIALIZE
#define PRINT_IMPLEMENTATION
#include "store.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static f64 now_s(void)
{ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return (f64)t.tv_sec + (f64)t.tv_nsec * 1e-9; }

static i32 DIM, N, NQ, K;
static f32 *VEC, *QRY;
static i64 *GT;      /* [NQ*K] cosine top-k rows */
static i32 *GOLD;    /* [NQ]   gold passage row, -1 if absent */

static void load_(const char *dir)
{
    char p[1024];
    FILE *f;
    u32 h[4], g2[2], ng;
    size_t nv, nq;

    sprintf(p, "%s/real_data.bin", dir);
    f = fopen(p, "rb");
    if (!f || fread(h, 4, 4, f) != 4) { fprintf(stderr, "no %s\n", p); exit(2); }
    DIM = (i32)h[0]; N = (i32)h[1]; NQ = (i32)h[2]; K = (i32)h[3];
    nv = (size_t)N * DIM; nq = (size_t)NQ * DIM;
    VEC = (f32 *)malloc(nv * 4); QRY = (f32 *)malloc(nq * 4);
    if (fread(VEC, 4, nv, f) != nv || fread(QRY, 4, nq, f) != nq) exit(2);
    fclose(f);

    sprintf(p, "%s/real_gt.bin", dir);
    f = fopen(p, "rb");
    if (!f || fread(g2, 4, 2, f) != 2) exit(2);
    GT = (i64 *)malloc((size_t)NQ * K * 8);
    if (fread(GT, 8, (size_t)NQ * K, f) != (size_t)NQ * K) exit(2);
    fclose(f);

    sprintf(p, "%s/real_gold.bin", dir);
    f = fopen(p, "rb");
    if (!f || fread(&ng, 4, 1, f) != 1) exit(2);
    GOLD = (i32 *)malloc((size_t)NQ * 4);
    if (fread(GOLD, 4, (size_t)NQ, f) != (size_t)NQ) exit(2);
    fclose(f);
}

/* the first TUNE_Q real questions are handed to engine_build as eval_q;
 * scoring is on the rest, so tuning and scoring never share a query. */
#define TUNE_Q 200

/* recall@K vs exact cosine GT, and gold-passage-in-top-K rate, over the
 * held-out query slice [TUNE_Q, NQ). */
static void score_(EngIndex *e, f64 *out_rec, f64 *out_gold, f64 *out_us)
{
    i64 ids[64]; f32 sc[64];
    i32 q, x, y, hit = 0, ghit = 0, gtot = 0, nsc = 0;
    f64 t0 = now_s();
    for (q = TUNE_Q; q < NQ; ++q) {
        i32 got = engine_search(e, QRY + (size_t)q * DIM, K, ids, sc);
        ++nsc;
        for (x = 0; x < got; ++x)
            for (y = 0; y < K; ++y)
                if (ids[x] == GT[(size_t)q * K + y]) { ++hit; break; }
        if (GOLD[q] >= 0) {
            ++gtot;
            for (x = 0; x < got; ++x) if (ids[x] == (i64)GOLD[q]) { ++ghit; break; }
        }
    }
    *out_us  = (now_s() - t0) / (nsc ? nsc : 1) * 1e6;
    *out_rec = 100.0 * hit / ((f64)(nsc ? nsc : 1) * K);
    *out_gold = gtot ? 100.0 * ghit / gtot : 0.0;
}

static const char *topo_(const EngIndex *e)
{
    if (e->topo == ENG_HNSW) return "HNSW-vec";
    if (e->topo == ENG_IVF) return "IVF";
    if (e->topo == ENG_CASCADE) return "cascade";
    return "flat";
}

static void run_goal_(Arena *a, const char *tag, EngGoal g)
{
    EngIndex *e, *e2;
    f64 rec, gold, us, rec2, gold2, us2, tb;
    FILE *tf;
    tb = now_s();
    e = engine_build(a, VEC, N, DIM, g);
    tb = now_s() - tb;
    score_(e, &rec, &gold, &us);

    tf = tmpfile();
    e2 = (tf && engine_save(e, tf) == 0 && (rewind(tf), 1)) ? engine_load(a, tf) : (EngIndex *)0;
    if (tf) fclose(tf);
    if (e2) score_(e2, &rec2, &gold2, &us2);

    printf("  %-13s %-8s %-5s np=%-2d %4d B/vec  build %5.1fs  %6.1f us/q"
           "  recall@10 %5.1f%%  gold@10 %5.1f%%   .vxi %s\n",
           tag, topo_(e),
           e->topo == ENG_IVF ? engine_scheme_name(e->scheme) : "-",
           e->topo == ENG_IVF ? e->nprobe : 0,
           e->code_bytes, tb, us, rec, gold,
           (e2 && rec2 == rec && gold2 == gold) ? "identical" : "MISMATCH");
    fflush(stdout);
}

int main(int argc, char **argv)
{
    static Arena a, b;
    const char *dir = (argc > 1) ? argv[1] : "emb";
    EngGoal g;
    RagStore *s, *s2;
    i64 ids[16]; f32 sc[16];
    i32 q, x, y, hit = 0, ghit = 0, gtot = 0;
    f64 us;
    FILE *tf; (void)b;

    load_(dir);
    printf("real RAG bench: n=%d dim=%d queries=%d  (SQuAD passages + AG-News, MiniLM)\n",
           N, DIM, NQ);
    fflush(stdout);

    memset(&g, 0, sizeof g);
    g.k = 10; g.n_iters = 12; g.seed = 0x5eedu; g.eval_queries = 128;
    g.eval_q = QRY; g.n_eval_q = TUNE_Q;   /* tune on real questions, not corpus rows */

    g.mode = ENG_MIN_SIZE; g.target_recall = 0.85f; g.max_code_bytes = 0;
    run_goal_(&a, "min>=85%", g);
    g.mode = ENG_MIN_SIZE; g.target_recall = 0.95f;
    run_goal_(&a, "min>=95%", g);
    g.mode = ENG_MAX_ACC; g.target_recall = 0.0f; g.max_code_bytes = 96;
    run_goal_(&a, "max<=96B", g);
    g.mode = ENG_MAX_ACC; g.max_code_bytes = 1 << 20;
    run_goal_(&a, "max-acc", g);
    g.mode = ENG_MIN_SIZE; g.target_recall = 0.90f; g.max_code_bytes = 0;
    g.max_candidates = N / 40;
    run_goal_(&a, "budget n/40", g);
    g.max_candidates = 0;

    /* store round-trip on the same corpus (tuned on the same real
     * question slice), scored on the held-out slice. */
    s = store_new(&b, DIM);
    g.mode = ENG_MIN_SIZE; g.target_recall = 0.90f; g.max_code_bytes = 0;
    store_set_goal(s, g);
    for (q = 0; q < N; ++q)
        if (store_add(s, VEC + (size_t)q * DIM, (i64)q) < 0) { puts("  store add FAIL"); return 1; }
    store_build(s);
    {
        f64 t0 = now_s();
        i32 nsc = 0;
        for (q = TUNE_Q; q < NQ; ++q) {
            i32 got = store_search(s, QRY + (size_t)q * DIM, 10, ids, sc);
            ++nsc;
            for (x = 0; x < got; ++x)
                for (y = 0; y < 10; ++y)
                    if (ids[x] == GT[(size_t)q * 10 + y]) { ++hit; break; }
            if (GOLD[q] >= 0) { ++gtot;
                for (x = 0; x < got; ++x) if (ids[x] == (i64)GOLD[q]) { ++ghit; break; } }
        }
        us = (now_s() - t0) / (nsc ? nsc : 1) * 1e6;
        printf("  %-13s %-8s              %6.1f us/q  recall@10 %5.1f%%  gold@10 %5.1f%%\n",
               "store", "wrapper", us,
               100.0 * hit / ((f64)(nsc ? nsc : 1) * 10),
               gtot ? 100.0 * ghit / gtot : 0.0);
    }
    fflush(stdout);

    tf = tmpfile();
    if (!tf || store_save(s, "/tmp/real_val_store.vxi") != 0) { puts("  store_save FAIL"); return 1; }
    if (tf) fclose(tf);
    s2 = store_load(&b, "/tmp/real_val_store.vxi");
    remove("/tmp/real_val_store.vxi");
    if (!s2) { puts("  store_load FAIL"); return 1; }
    {
        i32 same = 0;
        for (q = 0; q < NQ; q += NQ / 300 ? NQ / 300 : 1) {
            i64 r1[1], r2[1]; f32 c1[1], c2[1];
            store_search(s,  QRY + (size_t)q * DIM, 1, r1, c1);
            store_search(s2, QRY + (size_t)q * DIM, 1, r2, c2);
            if (r1[0] == r2[0]) ++same;
        }
        printf("  store .vxi round-trip: top-1 identical on %d probes\n", same);
        fflush(stdout);
    }
    store_destroy(s); store_destroy(s2);
    puts("real_val OK");
    return 0;
}
