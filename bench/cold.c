/* cold.c - searches on an index whose codes are not in the page cache: what a store larger than RAM sees on every
 * query (Tq.cold: the load leaves the codes on the disk, a search pages in the clusters it probes).
 *
 *   gcc -std=gnu99 -O2 -march=native -Isrc bench/cold.c -lm -lpthread -o .build/cold
 *   .build/cold make PATH N [DIM=96]   a store of N clustered rows, IVF built, saved to PATH
 *   VECINDEX_VERIFY=1 .build/cold query PATH [NQ=200]
 *                                      the file's pages dropped from the page cache (posix_fadvise), loaded cold,
 *                                      NQ queries one after another: median and p90 ms, major faults a query, and
 *                                      how much of the file the page cache holds after the load and after the queries
 *
 * VECINDEX_VERIFY=1: the default load checks every block's digest, which reads the whole file into the page cache
 * first. Tried and dropped (2026-11-10, this VM): asking for each probed cluster with MADV_WILLNEED before the scan
 * cut major faults 18.6 -> 8.6 a query but made the median 0.70 -> 1.0 ms - 300 madvise calls cost more here than
 * the kernel's own readahead saves. */
#include "store.c"
#include <sys/resource.h>
#include <time.h>

static double now_(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec; }
static f32 u_(u64 *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return (f32)(i64)(*s >> 11) * (1.0f / 4503599627370496.0f) - 1.0f; }

/* row i: one of 4096 centres plus noise, the same for the same i */
static void row_(i64 i, i32 dim, const f32 *ctr, f32 *v)
{
    u64 st = (u64)i * 0x9e3779b97f4a7c15ULL + 1;
    const f32 *c = ctr + (size_t)((u64)i % 4096u) * (size_t)dim;
    i32 d;
    for (d = 0; d < dim; ++d) v[d] = c[d] + 0.35f * u_(&st);
}

/* percent of the file's pages in the page cache (mincore over a fresh mapping) */
static long resident_(const char *p)
{
    int fd = open(p, O_RDONLY);
    struct stat st;
    size_t n, np, k;
    long c = 0;
    void *m;
    unsigned char *v;
    if (fd < 0 || fstat(fd, &st) != 0) return -1;
    n = (size_t)st.st_size; np = (n + 4095) / 4096;
    m = mmap((void *)0, n, PROT_READ, MAP_SHARED, fd, 0);
    v = (unsigned char *)malloc(np);
    if (m != MAP_FAILED && v && mincore(m, n, v) == 0) for (k = 0; k < np; ++k) c += v[k] & 1;
    if (m != MAP_FAILED) munmap(m, n);
    close(fd); free(v);
    return (long)(c * 100 / (long)(np ? np : 1));
}

static int cmpd_(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }

int main(int argc, char **argv)
{
    static Arena arena;
    i32 dim = argc > 4 ? atoi(argv[4]) : 96, d;
    f32 *ctr, *v;
    u64 st = 77;
    i64 i;
    if (argc < 3) { fprintf(stderr, "usage: cold make PATH N [DIM] | cold query PATH [NQ]\n"); return 2; }
    if (!strcmp(argv[1], "make")) {
        i64 n = argc > 3 ? atoll(argv[3]) : 2000000;
        RagStore *s = store_new(&arena, dim);
        double t0 = now_();
        ctr = (f32 *)malloc((size_t)4096 * dim * sizeof(f32)); v = (f32 *)malloc((size_t)dim * sizeof(f32));
        for (i = 0; i < (i64)4096 * dim; ++i) ctr[i] = u_(&st);
        for (i = 0; i < n; ++i) { row_(i, dim, ctr, v); store_add(s, v, i); }
        store_build(s);
        if (store_save(s, argv[2]) != 0) { fprintf(stderr, "save failed\n"); return 1; }
        printf("{\"rows\":%lld,\"dim\":%d,\"clusters\":%d,\"probe\":%d,\"code_bytes\":%lld,\"s\":%.1f}\n", (long long)n, (int)dim,
               s->tq ? (int)s->tq->nc : 0, s->tq ? (int)s->tq->nprobe : 0, s->tq ? (long long)s->tq->nblk * s->tq->vd * 16 : 0LL, now_() - t0);
        return 0;
    } else {
        i32 nq = argc > 3 ? atoi(argv[3]) : 200, q, fd;
        double *ms;
        long *mf;
        RagStore *s;
        i64 ids[10];
        f32 sc[10];
        struct rusage r0, r1;
        fd = open(argv[2], O_RDONLY);
        if (fd < 0) { perror(argv[2]); return 1; }
        fsync(fd);                                               /* dirty pages cannot be dropped: write them first */
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);            /* then the clean pages out of the page cache */
        close(fd);
        store_cold_min_ = 0;                                     /* cold, whatever the size */
        if (!(s = store_load(&arena, argv[2])) || !s->tq) { fprintf(stderr, "load failed\n"); return 1; }
        dim = s->dim;
        printf("{\"resident_after_load_pct\":%ld}\n", resident_(argv[2]));
        v = (f32 *)malloc((size_t)dim * sizeof(f32));
        ms = (double *)malloc((size_t)nq * sizeof(double)); mf = (long *)malloc((size_t)nq * sizeof(long));
        for (q = 0; q < nq; ++q) {
            double t0;
            for (d = 0; d < dim; ++d) v[d] = u_(&st);           /* anywhere: each query lands on clusters of its own */
            getrusage(RUSAGE_SELF, &r0);
            t0 = now_();
            store_search(s, v, 10, ids, sc);
            ms[q] = (now_() - t0) * 1e3;
            getrusage(RUSAGE_SELF, &r1);
            mf[q] = r1.ru_majflt - r0.ru_majflt;
        }
        {   long tot = 0; for (q = 0; q < nq; ++q) tot += mf[q];
            qsort(ms, (size_t)nq, sizeof(double), cmpd_);
            printf("{\"resident_after_pct\":%ld}\n", resident_(argv[2]));
            printf("{\"queries\":%d,\"median_ms\":%.2f,\"p90_ms\":%.2f,\"major_faults_per_query\":%.1f,\"cold\":%d}\n",
                   nq, ms[nq / 2], ms[nq * 9 / 10], (double)tot / nq, s->tq->cold); }
        return 0;
    }
}
