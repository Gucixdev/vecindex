/* live.c - rows streamed into a built index, on GloVe-100: what live mode costs and what it buys.
 *
 *   gcc -std=gnu99 -O2 -march=native -I.. live.c -lm -lpthread -o live && ./live DATA_DIR MODE
 *
 * MODE (each a separate run, same probe sweep: a batch of the 10k held-out queries on 2 threads, recall@10 against
 * the exact top 10 of all 1,183,514 rows):
 *   full    one build over every row - the reference
 *   tail    a build over 60% of the rows; the other 40% added with live mode off (they wait in the exact-scan tail,
 *           as before live mode), then live mode on: how long the store takes to index them, and the sweep after
 *   stream [RATE]  a build over 60%; the other 40% streamed in by a thread (RATE rows/s, default flat out) while this
 *           one searches one query at a time, live mode on with rebuilds at 50% growth: query latency before, while
 *           rows stream in, during the rebuild that growth triggers, after; then the sweep
 *   drift   a build over the 60% of rows least aligned with a direction in the data, the 40% most aligned streamed in
 *           (rows the clusters never saw): the sweep with them appended, then after the rebuild live mode decides on
 * DATA_DIR comes from prepare_glove.py. Doc ids are the row numbers of train.f32, so answers check against gt.i32. */
#include "store.c"

static Arena pa;
static f32 *X, *Q, *SC;
static i32 *GT, *ORD;
static i64 *IDS;
static i32 n = 1183514, nq = 10000, dim = 100;

static double now_(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static double recall_(const i64 *ids, i32 from, i32 cnt)
{
    i32 q, x, z, hit = 0;
    for (q = from; q < from + cnt; ++q)
        for (x = 0; x < 10; ++x)
            for (z = 0; z < 10; ++z) if (GT[(size_t)q * 100 + z] == ids[(size_t)(q - from) * 10 + x]) { ++hit; break; }
    return hit / (10.0 * cnt);
}

/* recall@10 and QPS (one batch of 10k queries on 2 threads) at each probe count */
static void sweep_(RagStore *s, const char *name)
{
    static const i32 P[] = { 32, 48, 64, 96, 128, 192, 256, 384, 512 };
    i32 i;
    for (i = 0; i < (i32)(sizeof(P) / sizeof(P[0])); ++i) {
        double t0;
        store_set_probe(s, P[i]);
        t0 = now_();
        store_search_batch(s, Q, nq, 10, IDS, SC, 2);
        printf("%-8s probe %4d: recall@10 %.4f  %6.0f QPS\n", name, P[i], recall_(IDS, 0, nq), nq / (now_() - t0));
        fflush(stdout);
    }
    store_set_probe(s, 0);
}

static void info_(RagStore *s, const char *what)
{
    i64 ix, rb; f64 rs;
    store_live_stats(s, &ix, &rb, &rs);
    printf("%s: %d rows, %d clusters, %d built + %d added (%d extra blocks), pending %d; indexed live %lld, rebuilds %lld (last %.1fs)\n",
           what, s->count, s->tq ? s->tq->nc : 0, s->tq ? s->tq->nbuilt : 0, s->tq ? s->tq->nadd : 0, s->tq ? s->tq->xn : 0,
           store_pending(s), (long long)ix, (long long)rb, rs);
    fflush(stdout);
}

static void wait_indexed_(RagStore *s)
{
    while (store_pending(s) > 0) { struct timespec ts = { 0, 1000000L }; nanosleep(&ts, 0); }
}

static RagStore *built_(i32 n0, double *bt)
{
    RagStore *s = store_new(&pa, dim);
    i32 i;
    double t0;
    store_set_spill(s, (const char *)0, 0);
    for (i = 0; i < n0; ++i) store_add(s, X + (size_t)ORD[i] * dim, ORD[i]);
    t0 = now_(); store_build(s); *bt = now_() - t0;
    return s;
}

typedef struct { RagStore *s; i32 from, to, done; double rate, t0, t1; } Feed;
static void *feed_(void *p)                                     /* rows from..to, at `rate` rows a second (0: flat out) */
{
    Feed *f = (Feed *)p;
    i32 i;
    f->t0 = now_();
    for (i = f->from; i < f->to; ++i) {
        store_add(f->s, X + (size_t)ORD[i] * dim, ORD[i]);
        __atomic_store_n(&f->done, i + 1, __ATOMIC_RELEASE);
        if (f->rate > 0 && (i & 255) == 255) {
            double ahead = (i + 1 - f->from) / f->rate - (now_() - f->t0);
            if (ahead > 0) { struct timespec ts = { (time_t)ahead, (long)((ahead - (time_t)ahead) * 1e9) }; nanosleep(&ts, 0); }
        }
    }
    f->t1 = now_();
    return 0;
}

static f32 *KEY;
static int cmpk_(const void *a, const void *b) { f32 x = KEY[*(const i32 *)a], y = KEY[*(const i32 *)b]; return x < y ? -1 : x > y; }
static int cmpd_(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static void lat_(double *l, i32 m, const char *what)
{
    qsort(l, (size_t)m, sizeof(double), cmpd_);
    printf("%s: %d queries, latency p50 %.2f ms, p99 %.2f ms, max %.1f ms\n", what, m, 1e3 * l[m / 2], 1e3 * l[(i32)(m * 0.99)], 1e3 * l[m - 1]);
}

int main(int argc, char **argv)
{
    FILE *f;
    char p[512];
    i32 i, n0 = (i32)(n * 0.6);
    const char *mode = argc > 2 ? argv[2] : "full";
    RagStore *s;
    double bt, t0;
    if (argc < 3) { fprintf(stderr, "usage: live DATA_DIR full|tail|stream|drift\n"); return 2; }
    X = malloc((size_t)n * dim * 4); Q = malloc((size_t)nq * dim * 4); GT = malloc((size_t)nq * 100 * 4);
    IDS = malloc((size_t)nq * 10 * 8); SC = malloc((size_t)nq * 10 * 4); ORD = malloc((size_t)n * 4);
    sprintf(p, "%s/train.f32", argv[1]); f = fopen(p, "rb"); if (!f || fread(X, 4, (size_t)n * dim, f) != (size_t)n * dim) return 1; fclose(f);
    sprintf(p, "%s/test.f32", argv[1]); f = fopen(p, "rb"); if (!f || fread(Q, 4, (size_t)nq * dim, f) != (size_t)nq * dim) return 1; fclose(f);
    sprintf(p, "%s/gt.i32", argv[1]); f = fopen(p, "rb"); if (!f || fread(GT, 4, (size_t)nq * 100, f) != (size_t)nq * 100) return 1; fclose(f);
    for (i = 0; i < n; ++i) ORD[i] = i;                          /* train.f32 is already in random order */

    if (!strcmp(mode, "full")) {
        s = built_(n, &bt);
        printf("full: build over %d rows %.2fs\n", n, bt);
        info_(s, "full"); sweep_(s, "full");
    } else if (!strcmp(mode, "tail")) {
        s = built_(n0, &bt);
        printf("tail: build over %d rows %.2fs\n", n0, bt);
        for (i = n0; i < n; ++i) store_add(s, X + (size_t)ORD[i] * dim, ORD[i]);
        store_set_probe(s, 128);
        t0 = now_(); store_search_batch(s, Q, 1000, 10, IDS, SC, 2);
        printf("tail   probe  128: recall@10 %.4f  %6.0f QPS (1000 queries, %d rows in the exact-scan tail)\n", recall_(IDS, 0, 1000), 1000 / (now_() - t0), store_pending(s));
        store_set_probe(s, 0);
        t0 = now_(); store_set_live(s, 1, 0.0f); wait_indexed_(s); store_set_live(s, 0, 0.0f);
        printf("tail: live mode indexed the %d tail rows in %.2fs (%.0f rows/s)\n", n - n0, now_() - t0, (n - n0) / (now_() - t0));
        info_(s, "live"); sweep_(s, "live");
    } else if (!strcmp(mode, "stream")) {
        Feed fd;
        pthread_t th;
        i32 cap = 4000000, q = 0, ph, mm[3] = { 0, 0, 0 };
        double *lat[3], tq0, tr0 = 0, tr = 0;
        static const char *pn[3] = { "while rows stream in and get indexed", "while the index is rebuilt", "after" };
        for (i = 0; i < 3; ++i) lat[i] = malloc((size_t)cap * sizeof(double));
        fd.rate = argc > 3 ? atof(argv[3]) : 0.0;
        s = built_(n0, &bt);
        printf("stream: build over %d rows %.2fs; then %d rows at %s rows/s\n", n0, bt, n - n0, argc > 3 ? argv[3] : "full speed");
        store_set_threads(s, 1);
        for (i = 0; i < 3000; ++i) { tq0 = now_(); store_search(s, Q + (size_t)(i % nq) * dim, 10, IDS, SC); lat[2][mm[2]++] = now_() - tq0; }
        lat_(lat[2], mm[2], "before, idle (1 thread, one query at a time)");
        mm[2] = 0;
        store_set_live(s, 1, 0.5f);                              /* rebuild once 50% more rows came in */
        fd.s = s; fd.from = n0; fd.to = n; fd.done = 0;
        pthread_create(&th, 0, feed_, &fd);
        for (;;) {                                               /* until everything is in, indexed, rebuilt + 1 s */
            i32 fed = __atomic_load_n(&fd.done, __ATOMIC_ACQUIRE) >= n, rb = (i32)store_stat_get_(&s->rebuilding);
            if (rb && !tr0) tr0 = now_();
            if (!rb && tr0 && !tr) tr = now_();
            if (fed && store_pending(s) == 0 && !rb && store_stat_get_(&s->n_rebuilds) > 0 && tr && now_() - tr > 1.0) break;
            ph = rb ? 1 : (tr ? 2 : 0);
            tq0 = now_(); store_search(s, Q + (size_t)(q++ % nq) * dim, 10, IDS, SC);
            if (mm[ph] < cap) lat[ph][mm[ph]++] = now_() - tq0;
        }
        pthread_join(th, 0);
        printf("stream: %d rows added in %.2fs (%.0f rows/s) while searching; rebuild ran %.2fs-%.2fs after the first add\n",
               n - n0, fd.t1 - fd.t0, (n - n0) / (fd.t1 - fd.t0), tr0 - fd.t0, tr - fd.t0);
        for (i = 0; i < 3; ++i) if (mm[i]) lat_(lat[i], mm[i], pn[i]);
        store_set_live(s, 0, 0.5f);
        for (i = 0; i < 20 && store_maintain(s); ++i) {}
        info_(s, "stream"); sweep_(s, "stream");
    } else if (!strcmp(mode, "drift")) {
        /* the direction: the mean of 1000 rows around one row (a region of the data); rows sorted by how much they lean
         * that way - the build gets the 60% that lean least, the stream the rest (in random order) */
        f32 *u = calloc((size_t)dim, 4), *key = malloc((size_t)n * 4);
        i32 j;
        u32 r = 12345u;
        const f32 *c0 = X + (size_t)777 * dim;
        {   /* the 1000 rows closest to row 777 */
            f32 *bs = malloc(1000 * 4); i32 *br = malloc(1000 * 4);
            for (i = 0; i < 1000; ++i) { bs[i] = -1e30f; br[i] = -1; }
            for (i = 0; i < n; ++i) store_topk_put_(bs, br, 1000, vs_dot(c0, X + (size_t)i * dim, dim), i);
            for (i = 0; i < 1000; ++i) for (j = 0; j < dim; ++j) u[j] += X[(size_t)br[i] * dim + j];
            free(bs); free(br);
        }
        for (i = 0; i < n; ++i) { key[i] = vs_dot(u, X + (size_t)i * dim, dim); ORD[i] = i; }
        KEY = key; qsort(ORD, (size_t)n, 4, cmpk_);           /* ascending */
        for (i = n - 1; i > n0; --i) { i32 k = n0 + (i32)(tq_rng_(&r) % (u32)(i - n0 + 1)), w = ORD[i]; ORD[i] = ORD[k]; ORD[k] = w; }
        {   /* how many of the queries' true neighbours are in the streamed part */
            i32 *in = calloc((size_t)n, 4), h = 0;
            for (i = n0; i < n; ++i) in[ORD[i]] = 1;
            for (i = 0; i < nq * 10; ++i) h += in[GT[(size_t)(i / 10) * 100 + i % 10]];
            printf("drift: %.1f%% of the queries' true top-10 rows are in the streamed 40%%\n", 100.0 * h / (nq * 10.0));
            free(in);
        }
        s = built_(n0, &bt);
        printf("drift: build over the %d least-aligned rows %.2fs\n", n0, bt);
        store_set_live(s, 1, 0.0f);                              /* index the stream, no rebuild yet */
        t0 = now_();
        for (i = n0; i < n; ++i) store_add(s, X + (size_t)ORD[i] * dim, ORD[i]);
        wait_indexed_(s);
        store_set_live(s, 0, 0.0f);
        printf("drift: %d rows added and indexed in %.2fs; far windows seen: %d (fit0 %.4f, added rows %.4f)\n", n - n0, now_() - t0,
               s->drifted, s->tq->fit0, s->tq->fit_add / (s->tq->nadd ? s->tq->nadd : 1));
        info_(s, "appended"); sweep_(s, "appended");
        store_set_live(s, 0, 1.0f);                              /* rebuilds allowed: live mode's own decision */
        t0 = now_(); i = store_maintain(s);
        printf("drift: maintenance step %.2fs, did %s%s%s\n", now_() - t0, i & 1 ? "index " : "", i & 2 ? "rebuild " : "", i & 4 ? "tune" : "");
        info_(s, "rebuilt"); sweep_(s, "rebuilt");
        free(u); free(key);
    } else { fprintf(stderr, "unknown mode %s\n", mode); return 2; }
    store_destroy(s);
    return 0;
}
