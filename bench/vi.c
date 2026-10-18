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

/* codebits 4 or 2 (scan codes), vecbytes 4 (f32 rows) or 1 (int8 rows) */
void *vi_new2(int dim, int codebits, int vecbytes)
{
    RagStore *s = store_new(&vi_arena, dim);
    store_set_codes(s, codebits);
    store_set_vec_bytes(s, vecbytes);
    return s;
}
void vi_set_rerank(void *s, int r) { store_set_rerank((RagStore *)s, r); }
/* self-tuning: the build (and vi_tune) pick the fewest probes that keep recall@10 at r; vi_tune returns what it measured */
void vi_set_target_recall(void *s, float r) { store_set_target_recall((RagStore *)s, r); }
float vi_tune(void *s) { return store_tune((RagStore *)s); }

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

/* bytes the store holds: anonymous memory (ram) and file-backed pages (file: the loaded .vxi or the spill file,
 * resident only while there is room) */
long long vi_mem_bytes(void *sv)
{
    i64 ram = 0, file = 0;
    store_mem((RagStore *)sv, &ram, &file);
    return (long long)ram;
}
long long vi_file_bytes(void *sv)
{
    i64 ram = 0, file = 0;
    store_mem((RagStore *)sv, &ram, &file);
    return (long long)file;
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

/* nq queries on nthreads threads: the store's batch search (cluster-major scan, then the re-scoring in parallel) */
void vi_search_batch(void *s, const float *q, int nq, int k, long long *ids, int nthreads)
{
    f32 *sc = (f32 *)malloc((size_t)nq * (size_t)k * sizeof(f32));
    store_search_batch((RagStore *)s, q, nq, k, (i64 *)ids, sc, nthreads);
    free(sc);
}
