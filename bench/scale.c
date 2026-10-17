/* scale.c - how far does one machine go? Streams N synthetic vectors into a store, builds, searches, saves and
 * reloads, printing time and memory (RSS split into anonymous memory and file-backed pages) at every stage.
 *
 *   gcc -std=gnu99 -O2 -march=native -I.. scale.c -lm -lpthread -o scale
 *   ./scale N [dim=128] [--nq 200] [--max-rss-mb 6000] [--save DIR] [--probe a,b,c] [--spill DIR]
 *
 * The data: a deterministic mixture (1000 topics x 50 subtopics, Gaussian-ish noise), generated row by row from
 * the row number, so the benchmark itself holds no copy of the corpus and the exact ground truth is computed by
 * regenerating the rows. Synthetic, so recall here says nothing about real embeddings (bench.py measures those
 * on GloVe); time and memory are the point.
 *
 * --max-rss-mb stops ingesting once the process's RSS passes the limit: that is the ceiling, measured instead
 * of reached the hard way (an OOM kill or a swap storm). */
#include "store.c"
#include <pthread.h>
#include <sys/resource.h>

#define SC_TOPICS 1000
#define SC_SUBS   50

static double now_(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec; }

static u64 mix_(u64 x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static f32 *topic, *sub;
static int DIM;
static f32 NOISE = 1.6f;

/* uniform-ish in [-1, 1) from a 64-bit state */
static f32 u_(u64 *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return (f32)(i64)(*s >> 11) * (1.0f / 4503599627370496.0f) - 1.0f; }

static void gen_centres_(void)
{
    u64 s = 12345;
    size_t i;
    topic = (f32 *)malloc((size_t)SC_TOPICS * DIM * sizeof(f32));
    sub = (f32 *)malloc((size_t)SC_TOPICS * SC_SUBS * DIM * sizeof(f32));
    for (i = 0; i < (size_t)SC_TOPICS * DIM; ++i) topic[i] = u_(&s) + u_(&s) + u_(&s);
    for (i = 0; i < (size_t)SC_TOPICS * SC_SUBS * DIM; ++i) sub[i] = 0.7f * (u_(&s) + u_(&s) + u_(&s));
}

/* row i of stream `seed`, unit length */
static void gen_row_(u64 seed, i64 i, f32 *v)
{
    u64 h = mix_(seed * 0x100000001b3ULL ^ (u64)i), s = h | 1;
    i32 t = (i32)(h % SC_TOPICS), u = (i32)((h >> 24) % SC_SUBS), d;
    const f32 *a = topic + (size_t)t * DIM, *b = sub + ((size_t)t * SC_SUBS + (size_t)u) * DIM;
    f32 n = 0;
    for (d = 0; d < DIM; ++d) { v[d] = a[d] + b[d] + NOISE * (u_(&s) + u_(&s) + u_(&s)); n += v[d] * v[d]; }
    n = 1.0f / sqrtf(n);
    for (d = 0; d < DIM; ++d) v[d] *= n;
}

static long status_kb_(const char *key)
{
    FILE *f = fopen("/proc/self/status", "r");
    char line[256]; long kb = -1; size_t kl = strlen(key);
    if (!f) return -1;
    while (fgets(line, sizeof line, f)) if (!strncmp(line, key, kl)) { sscanf(line + kl, "%ld", &kb); break; }
    fclose(f);
    return kb;
}

static void mem_(const char *stage, i64 n, double secs, const char *extra)
{
    printf("{\"stage\":\"%s\",\"n\":%lld,\"secs\":%.2f,\"rss_mb\":%.0f,\"anon_mb\":%.0f,\"file_mb\":%.0f,\"hwm_mb\":%.0f%s%s}\n",
           stage, (long long)n, secs, status_kb_("VmRSS:") / 1024.0, status_kb_("RssAnon:") / 1024.0,
           status_kb_("RssFile:") / 1024.0, status_kb_("VmHWM:") / 1024.0, extra ? "," : "", extra ? extra : "");
    fflush(stdout);
}

/* ---- exact ground truth by regenerating every row ---- */
typedef struct { i64 a, b; i32 nq; const f32 *q; f32 *bs; i64 *bi; } GtJob;
#define GTK 10

static void *gt_job_(void *p)
{
    GtJob *j = (GtJob *)p;
    f32 *v = (f32 *)malloc((size_t)DIM * sizeof(f32));
    i64 i;
    i32 q, x;
    for (q = 0; q < j->nq * GTK; ++q) { j->bs[q] = -1e30f; j->bi[q] = -1; }
    for (i = j->a; i < j->b; ++i) {
        gen_row_(1, i, v);
        for (q = 0; q < j->nq; ++q) {
            f32 s = vs_dot(v, j->q + (size_t)q * DIM, DIM), *bs = j->bs + q * GTK;
            i64 *bi = j->bi + q * GTK;
            if (s <= bs[GTK - 1]) continue;
            for (x = GTK - 1; x > 0 && bs[x - 1] < s; --x) { bs[x] = bs[x - 1]; bi[x] = bi[x - 1]; }
            bs[x] = s; bi[x] = i;
        }
    }
    free(v);
    return (void *)0;
}

static void ground_truth_(i64 n, i32 nq, const f32 *q, i64 *gt, i32 nth)
{
    GtJob j[16]; pthread_t th[16];
    i32 t, qi, x;
    for (t = 0; t < nth; ++t) {
        j[t].a = n * t / nth; j[t].b = n * (t + 1) / nth; j[t].nq = nq; j[t].q = q;
        j[t].bs = (f32 *)malloc((size_t)nq * GTK * sizeof(f32)); j[t].bi = (i64 *)malloc((size_t)nq * GTK * sizeof(i64));
        pthread_create(&th[t], 0, gt_job_, &j[t]);
    }
    for (t = 0; t < nth; ++t) pthread_join(th[t], 0);
    for (qi = 0; qi < nq; ++qi) {                       /* merge the per-thread top lists */
        f32 bs[GTK]; i64 bi[GTK];
        for (x = 0; x < GTK; ++x) { bs[x] = -1e30f; bi[x] = -1; }
        for (t = 0; t < nth; ++t) {
            i32 y;
            for (y = 0; y < GTK; ++y) {
                f32 s = j[t].bs[qi * GTK + y]; i64 id = j[t].bi[qi * GTK + y];
                if (s <= bs[GTK - 1]) continue;
                for (x = GTK - 1; x > 0 && bs[x - 1] < s; --x) { bs[x] = bs[x - 1]; bi[x] = bi[x - 1]; }
                bs[x] = s; bi[x] = id;
            }
        }
        memcpy(gt + (size_t)qi * GTK, bi, sizeof(bi));
    }
    for (t = 0; t < nth; ++t) { free(j[t].bs); free(j[t].bi); }
}

/* ---- searching: one query per thread ---- */
typedef struct { RagStore *s; const f32 *q; i64 *res; i32 a, b; double *lat; } SJob;

static void *s_job_(void *p)
{
    SJob *j = (SJob *)p;
    i32 i;
    f32 sc[GTK];
    for (i = j->a; i < j->b; ++i) {
        double t0 = now_();
        i32 got = store_search(j->s, j->q + (size_t)i * DIM, GTK, j->res + (size_t)i * GTK, sc), x;
        for (x = got; x < GTK; ++x) j->res[(size_t)i * GTK + x] = -1;
        if (j->lat) j->lat[i] = now_() - t0;
    }
    return (void *)0;
}

static double run_queries_(RagStore *s, const f32 *q, i32 nq, i64 *res, i32 nth, double *lat)
{
    SJob j[16]; pthread_t th[16];
    i32 t;
    double t0 = now_();
    for (t = 0; t < nth; ++t) { j[t].s = s; j[t].q = q; j[t].res = res; j[t].a = nq * t / nth; j[t].b = nq * (t + 1) / nth; j[t].lat = lat; }
    for (t = 1; t < nth; ++t) pthread_create(&th[t], 0, s_job_, &j[t]);
    s_job_(&j[0]);
    for (t = 1; t < nth; ++t) pthread_join(th[t], 0);
    return now_() - t0;
}

static int dcmp_(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }

static void search_report_(const char *stage, RagStore *s, i64 n, const f32 *q, i32 nq, const i64 *gt, i32 probe, i32 ncpu)
{
    i64 *res = (i64 *)malloc((size_t)nq * GTK * sizeof(i64));
    double *lat = (double *)malloc((size_t)nq * sizeof(double)), t1, tn;
    i32 i, x, y, hit = 0;
    char extra[256];
    store_set_probe(s, probe);
    store_set_threads(s, 1);
    t1 = run_queries_(s, q, nq, res, 1, lat);
    for (i = 0; i < nq; ++i) for (x = 0; x < GTK; ++x) for (y = 0; y < GTK; ++y) if (res[(size_t)i * GTK + x] == gt[(size_t)i * GTK + y]) { ++hit; break; }
    tn = run_queries_(s, q, nq, res, ncpu, (double *)0);
    qsort(lat, (size_t)nq, sizeof(double), dcmp_);
    sprintf(extra, "\"probe\":%d,\"recall10\":%.4f,\"qps1\":%.1f,\"qps%d\":%.1f,\"p50_ms\":%.3f,\"p99_ms\":%.3f", probe,
            (double)hit / (nq * GTK), nq / t1, ncpu, nq / tn, lat[nq / 2] * 1e3, lat[nq * 99 / 100] * 1e3);
    mem_(stage, n, t1, extra);
    free(res); free(lat);
}

int main(int argc, char **argv)
{
    static Arena arena;
    i64 N = 1000000, i, added = 0;
    i32 nq = 200, ncpu = (i32)sysconf(_SC_NPROCESSORS_ONLN), np = 0, a, probes[16];
    long max_rss_mb = 0;
    i32 ingest_only = 0;
    const char *save = (const char *)0, *spill = (const char *)0;
    RagStore *s;
    f32 *v, *q;
    i64 *gt;
    double t0;
    DIM = 128;
    for (a = 1; a < argc; ++a) {
        if (!strcmp(argv[a], "--nq") && a + 1 < argc) nq = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--max-rss-mb") && a + 1 < argc) max_rss_mb = atol(argv[++a]);
        else if (!strcmp(argv[a], "--save") && a + 1 < argc) save = argv[++a];
        else if (!strcmp(argv[a], "--spill") && a + 1 < argc) spill = argv[++a];
        else if (!strcmp(argv[a], "--noise") && a + 1 < argc) NOISE = (f32)atof(argv[++a]);
        else if (!strcmp(argv[a], "--ingest-only")) ingest_only = 1;
        else if (!strcmp(argv[a], "--probe") && a + 1 < argc) {
            char *p = argv[++a];
            while (*p && np < 16) { probes[np++] = atoi(p); while (*p && *p != ',') ++p; if (*p) ++p; }
        }
        else if (a == 1) N = atoll(argv[a]);
        else if (a == 2) DIM = atoi(argv[a]);
    }
    if (ncpu > 16) ncpu = 16;
    gen_centres_();
    printf("{\"stage\":\"config\",\"n\":%lld,\"dim\":%d,\"nq\":%d,\"cpus\":%d,\"noise\":%.2f}\n", (long long)N, DIM, nq, ncpu, NOISE);
    s = store_new(&arena, DIM);
    store_set_fast(s, 0.97f);
#ifdef STORE_HAS_SPILL
    if (spill) store_set_spill(s, !strcmp(spill, "off") ? (const char *)0 : spill, 0);
#else
    (void)spill;
#endif
    v = (f32 *)malloc((size_t)DIM * sizeof(f32));
    t0 = now_();
    for (i = 0; i < N; ++i) {
        gen_row_(1, i, v);
        store_add(s, v, i);
        ++added;
        if ((i + 1) % 1000000 == 0) {
            long rss = status_kb_("VmRSS:") / 1024;
            mem_("ingest", added, now_() - t0, (const char *)0);
            if (max_rss_mb && rss > max_rss_mb) { printf("{\"stage\":\"ceiling\",\"n\":%lld,\"rss_mb\":%ld}\n", (long long)added, rss); break; }
        }
    }
    N = added;
    mem_("ingested", N, now_() - t0, (const char *)0);
    if (ingest_only) return 0;

    t0 = now_();
    store_build(s);
    mem_("built", N, now_() - t0, (const char *)0);

    q = (f32 *)malloc((size_t)nq * DIM * sizeof(f32));
    for (i = 0; i < nq; ++i) gen_row_(2, i, q + (size_t)i * DIM);
    gt = (i64 *)malloc((size_t)nq * GTK * sizeof(i64));
    t0 = now_();
    ground_truth_(N, nq, q, gt, ncpu);
    printf("{\"stage\":\"ground_truth\",\"secs\":%.1f}\n", now_() - t0);
    if (!np) probes[np++] = 0;
    for (a = 0; a < np; ++a) search_report_("search", s, N, q, nq, gt, probes[a], ncpu);

    if (save) {
        char path[1024];
        RagStore *l;
        struct stat st;
        sprintf(path, "%s/scale.vxi", save);
        t0 = now_();
        if (store_save(s, path) != 0) { printf("{\"stage\":\"save\",\"error\":1}\n"); return 1; }
        stat(path, &st);
        { char e[64]; sprintf(e, "\"file_mb\":%.0f", (double)st.st_size / 1048576.0); mem_("saved", N, now_() - t0, e); }
        store_destroy(s);
        mem_("destroyed", N, 0, (const char *)0);
        t0 = now_();
        l = store_load(&arena, path);
        if (!l) { printf("{\"stage\":\"load\",\"error\":1}\n"); return 1; }
        mem_("loaded", N, now_() - t0, (const char *)0);
        for (a = 0; a < np; ++a) search_report_("search_loaded", l, N, q, nq, gt, probes[a], ncpu);
        store_destroy(l);
        remove(path); { char j[1040]; sprintf(j, "%s.j", path); remove(j); }
    }
    return 0;
}
