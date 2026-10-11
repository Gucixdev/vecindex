/* bench.c - timings for the flat quantizers and the full IVF engine,
 * next to the Rust port of vecindex.
 *
 * Now apples-to-apples: engine_build/engine_search go through the same
 * IVF k-means + per-cluster fine quantizer path as Rust's VecIndex.
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -D_POSIX_C_SOURCE=200112L -O2 \
 *       -I. bench.c -lm -o bench && ./bench
 */

#define PRINT_IMPLEMENTATION
#include "engine.c"          /* pulls in ivf.c + q1..q5.c + the arena impl */
#include "print.h"
#include <time.h>

static f64 now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (f64)ts.tv_sec + (f64)ts.tv_nsec * 1e-9;
}

static u32 rng_(u32 *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static f32 rf_(u32 *s) { return (f32)rng_(s) / (f32)0xffffffffu - 0.5f; }

static f32 *uniform_(Arena *a, i32 n, i32 dim, u32 seed)
{
    f32 *o = (f32 *)arena_alloc(a, (size_t)n * dim * sizeof(f32));
    u32 st = seed | 1u; i32 i;
    for (i = 0; i < n * dim; ++i) o[i] = rf_(&st) * 2.0f;
    return o;
}

static f32 *clustered_(Arena *a, i32 per, i32 topics, i32 dim, f32 spread, u32 seed)
{
    i32 n = per * topics, c, i, d;
    f32 *ct = (f32 *)arena_alloc(a, (size_t)topics * dim * sizeof(f32));
    f32 *o  = (f32 *)arena_alloc(a, (size_t)n * dim * sizeof(f32));
    u32 st = seed | 1u;
    for (i = 0; i < topics * dim; ++i) ct[i] = rf_(&st) * 8.0f;
    for (c = 0; c < topics; ++c)
        for (i = 0; i < per; ++i)
            for (d = 0; d < dim; ++d)
                o[((size_t)c * per + i) * dim + d] = ct[c * dim + d] + rf_(&st) * spread;
    return o;
}

static void us1(f64 sec)
{
    i64 t = (i64)(sec * 1e7 + 0.5);
    print_i64(t / 10); print_cstr("."); print_i64(t % 10);
}

/* --- flat quantizer micro-timings (encode is the cross-language ==) --- */

#define FLATROW(Q, T) do {                                              \
    Arena_Mark m = arena_snapshot(a);                                   \
    T *ix; u8 *tmp; f32 rn; i32 qi, rep;                                \
    i64 ids[16]; f32 sc[16]; f64 t0, tenc, tq;                          \
    ix = Q##_build(a, v, n, dim);                                       \
    tmp = (u8 *)arena_alloc(a, (size_t)Q##_code_bytes(dim));            \
    t0 = now_s();                                                       \
    for (rep = 0; rep < 20000; ++rep)                                   \
        Q##_encode(ix, v + (size_t)(rep % n) * dim, tmp, &rn);          \
    tenc = (now_s() - t0) / 20000.0;                                    \
    tq = 0.0;                                                           \
    for (qi = 0; qi < 200; ++qi) {                                      \
        const f32 *q = v + (size_t)((qi * 97) % n) * dim;               \
        t0 = now_s(); Q##_search(ix, q, 10, ids, sc); tq += now_s() - t0; \
    }                                                                  \
    print_cstr("  " #Q "  "); print_i64(Q##_code_bytes(dim));          \
    print_cstr(" B   encode "); print_i64((i64)(tenc * 1e9 + 0.5));    \
    print_cstr(" ns   flat-scan query "); us1(tq / 200.0);             \
    print_cstr(" us\n");                                                \
    arena_rewind(a, m);                                                 \
} while (0)

static void engine_row_(Arena *a, i32 n, i32 dim, i32 topics)
{
    f32 *v = clustered_(a, n / topics, topics, dim, 1.0f, 7u);
    EngGoal g;
    EngIndex *e;
    f64 t0, tq = 0.0;
    i64 ids[16]; f32 sc[16];
    i32 qi;

    memset(&g, 0, sizeof g);
    g.mode = ENG_MAX_ACC; g.k = 10; g.target_recall = 0.0f;
    g.max_code_bytes = 64; g.eval_queries = 32; g.n_iters = 10; g.seed = 0;

    t0 = now_s();
    e = engine_build(a, v, n, dim, g);
    print_cstr("  n="); print_i64(n);
    print_cstr("   build (20 configs, all measured) ");
    print_i64((i64)((now_s() - t0) * 1e3 + 0.5)); print_cstr(" ms\n");

    for (qi = 0; qi < 500; ++qi) {
        const f32 *q = v + (size_t)((qi * 97) % n) * dim;
        t0 = now_s();
        engine_search(e, q, 10, ids, sc);
        tq += now_s() - t0;
    }
    print_cstr("         chosen "); print_cstr(e->topo == ENG_IVF ? "IVF+" : "flat ");
    print_cstr(engine_scheme_name(e->scheme));
    if (e->topo == ENG_IVF) { print_cstr(" nprobe="); print_i64(e->nprobe); }
    print_cstr(", recall@10 "); print_i64((i64)(100.0 * e->recall));
    print_cstr("%   query "); us1(tq / 500.0); print_cstr(" us\n");
}

int main(void)
{
    static Arena arena;
    Arena *a = &arena;
    f32 *v;
    i32 dim, n;

    print_cstr("vecindex timings   gcc -O2, single thread\n\n");

    /* flat encode: 1:1 with Rust ternary_encode_one_vector_dim384 */
    dim = 384; n = 5000; v = uniform_(a, n, dim, 3u);
    print_cstr("flat quantizers, n="); print_i64(n); print_cstr(" dim="); print_i64(dim);
    print_cstr(" (uniform)\n");
    FLATROW(q1, Q1Index); FLATROW(q3, Q3Index); FLATROW(q2, Q2Index);
    FLATROW(q4, Q4Index); FLATROW(q5, Q5Index);

    /* the engine's IVF path: 1:1 with Rust VecIndex build/search */
    print_cstr("\nengine (IVF + per-cluster fine), clustered, dim=128\n");
    engine_row_(a, 5000, 128, 20);
    engine_row_(a, 20000, 128, 20);

    print_cstr("\nRust vecindex (criterion):\n");
    print_cstr("  ternary_encode_one_vector_dim384         ~1340 ns   (== flat encode)\n");
    print_cstr("  VecIndex build/5000  (dim128)            ~377 ms    (single config)\n");
    print_cstr("  VecIndex build/20000                     ~2.94 s\n");
    print_cstr("  VecIndex search nprobe 1 / 4 / 16 (20k)  56 / 186 / 668 us\n");
    print_cstr("  note: Rust build is ONE config; engine build above measures 20.\n");
    print_flush();
    arena_free(a);
    return 0;
}
