/* vi.c - a small C ABI over store.c for the benchmark harness (bench.py loads it with ctypes, the same way
 * every other library in the comparison is driven from Python).
 *
 *   gcc -std=gnu99 -O3 -march=native -fPIC -shared -I.. vi.c -o libvi.so -lm -lpthread
 *
 * Doc ids are the row numbers 0..n-1, so results compare directly with the ground truth. */
#include "store.c"
#include <pthread.h>

static Arena vi_arena;

void *vi_new(int dim) { return store_new(&vi_arena, dim); }

/* adds n rows (ids = first_id + i); returns how many were accepted */
int vi_add(void *s, const float *v, long n, long first_id)
{
    long i;
    int ok = 0;
    for (i = 0; i < n; ++i)
        if (store_add((RagStore *)s, v + (size_t)i * ((RagStore *)s)->dim, (i64)(first_id + i)) == 0) ++ok;
    return ok;
}

void vi_build(void *s) { store_build((RagStore *)s); }
void vi_set_probe(void *s, int nprobe) { store_set_probe((RagStore *)s, nprobe); }
void vi_set_threads(void *s, int nthreads) { store_set_threads((RagStore *)s, nthreads); }
int vi_save(void *s, const char *path) { return store_save((RagStore *)s, path); }
void *vi_load(const char *path) { return store_load(&vi_arena, path); }

/* bytes the store holds in RAM for searching: rows, ids, tags, tombstones, the id map and the scan index */
long long vi_mem_bytes(void *sv)
{
    RagStore *s = (RagStore *)sv;
    long long b = (long long)s->count * ((long long)s->dim * 4 + 8 + 8 + 1 + 4);
    b += (long long)s->idmap.cap * (8 + 4 + 1);
    if (s->tq) {
        const Tq *t = s->tq;
        b += (long long)t->nblk * t->dpad * 16;
        if (t->slot_row) b += (long long)t->nblk * 32 * 4;
        if (t->cent) b += (long long)t->nc * t->dim * 4 + (long long)(t->nc + 1) * 4;
        b += (long long)t->dpad * 12;
    }
    return b;
}

int vi_search(void *s, const float *q, int k, long long *ids)
{
    f32 sc[256];
    i64 tmp[256];
    int i, got;
    if (k > 256) k = 256;
    got = store_search((RagStore *)s, q, k, tmp, sc);
    for (i = 0; i < got; ++i) ids[i] = (long long)tmp[i];
    for (; i < k; ++i) ids[i] = -1;
    return got;
}

typedef struct { void *s; const float *q; int nq, dim, k, a, b; long long *ids; } ViJob;

static void *vi_job_(void *p)
{
    ViJob *j = (ViJob *)p;
    int i;
    for (i = j->a; i < j->b; ++i) vi_search(j->s, j->q + (size_t)i * j->dim, j->k, j->ids + (size_t)i * j->k);
    return (void *)0;
}

/* nq queries spread over nthreads threads, one query per thread at a time */
void vi_search_batch(void *s, const float *q, int nq, int k, long long *ids, int nthreads)
{
    ViJob jobs[64];
    pthread_t th[64];
    int i, dim = ((RagStore *)s)->dim;
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 64) nthreads = 64;
    for (i = 0; i < nthreads; ++i) {
        jobs[i].s = s; jobs[i].q = q; jobs[i].nq = nq; jobs[i].dim = dim; jobs[i].k = k; jobs[i].ids = ids;
        jobs[i].a = (int)((long)nq * i / nthreads); jobs[i].b = (int)((long)nq * (i + 1) / nthreads);
    }
    for (i = 1; i < nthreads; ++i) pthread_create(&th[i], (const pthread_attr_t *)0, vi_job_, &jobs[i]);
    vi_job_(&jobs[0]);
    for (i = 1; i < nthreads; ++i) pthread_join(th[i], (void **)0);
}
