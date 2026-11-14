#ifndef HOLYC_FGJOB_H_
#define HOLYC_FGJOB_H_

/* "fine grained jobs - brak wielowatkowosci brak race conditions" means no
 * *locks around shared mutable state*, not "no threads" - the threads are
 * very much there. Mirrors ashcore's ThreadPool + TaskGraph (Mojo):
 *
 *   - a fixed pool of worker OS threads (pthreads)
 *   - workers claim disjoint chunks of a flat [0, n) index range through a
 *     single atomic counter (fetch_add) instead of a shared job queue + a
 *     mutex around it - there is nothing to lock because there is nothing
 *     shared except one integer
 *   - each dispatch(job_id) is expected to only touch memory indexed by
 *     job_id, so two workers never write the same location: no data race
 *     by construction, not by "please don't race" discipline
 *   - Task_Graph adds dependencies on top: jobs are topo-sorted (Kahn's
 *     algorithm) into levels, with a full barrier between levels, so a job
 *     never starts before everything it depends on has finished. Jobs
 *     within one level are independent by DAG structure - the caller is
 *     still responsible for not having two same-level jobs touch the same
 *     memory without declaring a dependency between them.
 *
 * Measured, not assumed (see bench/bench_fgjob-style micro-benchmark): the
 * original "spawn N pthreads per run() call, join at the end" design costs
 * a flat ~190us of pure pthread_create/join overhead on this machine for
 * ANY batch needing more than 1 worker, regardless of how much work is in
 * it - identical at n=10 and n=1000 jobs, only fading into the noise past
 * ~100k jobs. A Task_Graph with a handful of small levels pays that fixed
 * cost per level. So: threads are now spawned once in thread_pool_init and
 * parked on a condvar between batches (mirrors what Mojo's parallelize()
 * gets for free from its runtime-managed pool) - call thread_pool_destroy
 * when done with a pool. thread_pool_run/run_range keep the exact same
 * signature, so Task_Graph needed zero changes for this. */

#include "base.h"
#include "arena_alloc.h"

/* pthread.h lives here directly, not a shared platform-thread wrapper -
 * this is concurrency's own domain, already covered by arena/pool/
 * event_queue/coroutines, and fgjob.h is the sole real consumer of
 * pthreads in this codebase. pthread_t/mutex_t/cond_t are part of the
 * Thread_Pool struct below, so this needs to be unconditional, not
 * implementation-only. */
#include <pthread.h>

#ifndef FGJOB_MAX_WORKERS
#define FGJOB_MAX_WORKERS 256
#endif

#ifndef FGJOB_CHUNK
/* swept 8/16/32/64/128/256/512/1024/4096 on a 12-thread i7-8700 at
 * n_jobs=1000/100000/5000000 with a small (non-noop) per-job cost - not
 * monotonic (512 loses to 256 at every size tested), 256 was the best or
 * tied-best point at all three scales without degenerating into a
 * single-chunk (effectively serial) run the way 1024/4096 do at n=1000.
 * Old default was 64, measured 15-27% slower than 256 depending on n. */
#define FGJOB_CHUNK 256
#endif

typedef void (*Job_Dispatch_Fn)(i64 job_id, void *ctx);

typedef struct {
    i64 volatile next;  /* shared atomic work-stealing counter for the current batch */
    i64 n;
    i64 base;
    Job_Dispatch_Fn fn;
    void *ctx;
} Fgjob_Batch_;

typedef struct {
    pthread_t threads[FGJOB_MAX_WORKERS];
    i32 n_workers;

    pthread_mutex_t mu;
    pthread_cond_t work_cv;   /* workers park here, woken when a new batch is posted */
    pthread_cond_t done_cv;   /* thread_pool_run_range parks here until the batch finishes */

    Fgjob_Batch_ batch;
    u64 generation;      /* bumped per dispatch; how a woken worker tells "new batch" from "spurious" */
    i32 active_workers;  /* counts down to 0 as workers finish the current batch */
    i32 shutdown;
} Thread_Pool;

i32  thread_pool_default_worker_count(void);
void thread_pool_init(Thread_Pool *pool, i32 n_workers); /* n_workers <= 0 -> autodetect; spawns the persistent workers */
void thread_pool_destroy(Thread_Pool *pool);              /* signals shutdown, joins every worker - call when done with the pool */
void thread_pool_run(Thread_Pool *pool, i64 n_jobs, Job_Dispatch_Fn fn, void *ctx);
void thread_pool_run_range(Thread_Pool *pool, i64 start, i64 stop, Job_Dispatch_Fn fn, void *ctx);

/* growable i64 array, used for a job's dependency list and for the
 * flattened level order - local to fgjob, not a shared "array" abstraction
 * until something else actually needs one too */
typedef struct {
    i64 *items;
    size_t count;
    size_t capacity;
} I64_Array;

typedef struct {
    I64_Array *items; /* per-job dependency list, indexed by job id */
    size_t count;
    size_t capacity;
} Dep_Array;

typedef struct {
    Dep_Array deps;

    I64_Array levels_flat;
    I64_Array level_starts;
    I64_Array level_sizes;

    i32 sealed;
} Task_Graph;

void task_graph_init(Task_Graph *g);
i64  task_graph_add_job(Arena *a, Task_Graph *g);
void task_graph_add_dep(Arena *a, Task_Graph *g, i64 job, i64 depends_on);
i32  task_graph_seal(Arena *a, Task_Graph *g); /* 0 on success, -1 on a dependency cycle */
void task_graph_execute(Task_Graph *g, Thread_Pool *pool, Job_Dispatch_Fn fn, void *ctx);
void task_graph_execute_serial(Task_Graph *g, Job_Dispatch_Fn fn, void *ctx);

#endif /* HOLYC_FGJOB_H_ */

#ifdef FGJOB_IMPLEMENTATION
#ifndef FGJOB_IMPLEMENTATION_
#define FGJOB_IMPLEMENTATION_

#include <unistd.h>

static void fgjob_drain_batch_(i64 volatile *next, i64 n, i64 base, Job_Dispatch_Fn fn, void *ctx)
{
    i64 start, stop, i;
    for (;;) {
        start = __sync_fetch_and_add(next, FGJOB_CHUNK);
        if (start >= n) break;
        stop = start + FGJOB_CHUNK;
        if (stop > n) stop = n;
        for (i = start; i < stop; i++) fn(base + i, ctx);
    }
}

static void *fgjob_persistent_worker_main_(void *arg_)
{
    Thread_Pool *pool = (Thread_Pool *)arg_;
    u64 seen_generation = 0;

    for (;;) {
        i64 n, base;
        Job_Dispatch_Fn fn;
        void *ctx;

        pthread_mutex_lock(&pool->mu);
        while (pool->generation == seen_generation && !pool->shutdown) {
            pthread_cond_wait(&pool->work_cv, &pool->mu);
        }
        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mu);
            return NULL;
        }
        seen_generation = pool->generation;
        n = pool->batch.n;
        base = pool->batch.base;
        fn = pool->batch.fn;
        ctx = pool->batch.ctx;
        pthread_mutex_unlock(&pool->mu);

        fgjob_drain_batch_(&pool->batch.next, n, base, fn, ctx);

        pthread_mutex_lock(&pool->mu);
        pool->active_workers--;
        if (pool->active_workers == 0) {
            pthread_cond_signal(&pool->done_cv);
        }
        pthread_mutex_unlock(&pool->mu);
    }
}

i32 thread_pool_default_worker_count(void)
{
    i64 n = sysconf(_SC_NPROCESSORS_ONLN); /* sysconf's own return type is `long`; i64 is the same width on this platform (see rng.h's own comment on this equivalence) */
    i64 lo_mask, hi_mask;

    /* branchless clamp to [1, FGJOB_MAX_WORKERS] via select-by-mask
     * (comparison produces a 0/1 used as DATA to build an all-0s/all-1s
     * mask, not as a jump condition) - this is one-shot setup code (see
     * the persistent-worker-pool comment above: runs once per pool, not
     * per job), so it buys nothing performance-wise here; containers/CI
     * can report 0 or -1 for n, hence the lower clamp. */
    lo_mask = -(i64)(n < 1);
    n = (n & ~lo_mask) | (1 & lo_mask);

    hi_mask = -(i64)(n > FGJOB_MAX_WORKERS);
    n = (n & ~hi_mask) | ((i64)FGJOB_MAX_WORKERS & hi_mask);

    return (i32)n;
}

void thread_pool_init(Thread_Pool *pool, i32 n_workers)
{
    i32 i;
    i32 hi_mask;

    /* NOT made branchless: doing so would mean unconditionally calling
     * thread_pool_default_worker_count() (a sysconf() syscall) even when
     * the caller already passed a valid n_workers - a real behavior
     * change (extra syscall on every init), not a value-only clamp like
     * the one below. */
    if (n_workers <= 0) n_workers = thread_pool_default_worker_count();

    hi_mask = -(i32)(n_workers > FGJOB_MAX_WORKERS);
    n_workers = (n_workers & ~hi_mask) | (FGJOB_MAX_WORKERS & hi_mask);

    pool->n_workers = n_workers;
    pool->generation = 0;
    pool->active_workers = 0;
    pool->shutdown = 0;
    pool->batch.next = 0;
    pool->batch.n = 0;
    pool->batch.base = 0;
    pool->batch.fn = NULL;
    pool->batch.ctx = NULL;

    pthread_mutex_init(&pool->mu, NULL);
    pthread_cond_init(&pool->work_cv, NULL);
    pthread_cond_init(&pool->done_cv, NULL);

    for (i = 0; i < n_workers; i++) {
        pthread_create(&pool->threads[i], NULL, fgjob_persistent_worker_main_, pool);
    }
    /* Tried: pinning worker i to core i via pthread_setaffinity_np(),
     * reasoning "persistent threads that repeatedly park/wake shouldn't
     * lose cache/TLB warmth to migration". Measured with bench/bench_fgjob:
     * consistently SLOWER, not faster - n=1000 went from 20us to 28us,
     * n=10000 from 24us to 32us. Pinning removes the scheduler's freedom to
     * wake a thread on whichever core is currently idle; for this
     * bursty park/wake/park pattern (not sustained per-core work) that
     * flexibility matters more than avoiding migration. Reverted rather
     * than kept "because it should help in theory". */
}

void thread_pool_destroy(Thread_Pool *pool)
{
    i32 i;

    pthread_mutex_lock(&pool->mu);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->work_cv);
    pthread_mutex_unlock(&pool->mu);

    for (i = 0; i < pool->n_workers; i++) {
        pthread_join(pool->threads[i], NULL);
    }

    pthread_mutex_destroy(&pool->mu);
    pthread_cond_destroy(&pool->work_cv);
    pthread_cond_destroy(&pool->done_cv);
}

void thread_pool_run_range(Thread_Pool *pool, i64 start, i64 stop, Job_Dispatch_Fn fn, void *ctx)
{
    i64 n = stop - start;

    if (n <= 0) return;

    if (pool->n_workers <= 1 || n <= FGJOB_CHUNK) {
        /* n <= FGJOB_CHUNK: only one worker could ever claim anything (the
         * first fetch_add grabs the whole range), so every other worker
         * that wakes just pays a mutex/condvar round trip to discover
         * there is nothing left - measured at a flat ~19us regardless of
         * job count, so skip waking the pool at all and run inline */
        i64 volatile solo_next = 0;
        fgjob_drain_batch_(&solo_next, n, start, fn, ctx);
        return;
    }

    /* every persistent worker wakes on the broadcast below and participates -
     * do NOT cap how many actually run based on job count. A worker that
     * finds no work left just sees start >= n on its first fetch_add and
     * returns immediately, which is cheap and safe; capping active_workers
     * to fewer than n_workers while every thread still wakes would let
     * "extra" workers decrement a counter nothing was expecting them to
     * touch, and worse, race a straggler from THIS batch against the next
     * batch's freshly-reset counter once thread_pool_run_range returns. */
    pthread_mutex_lock(&pool->mu);
    pool->batch.next = 0;
    pool->batch.n = n;
    pool->batch.base = start;
    pool->batch.fn = fn;
    pool->batch.ctx = ctx;
    pool->active_workers = pool->n_workers;
    pool->generation++;
    pthread_cond_broadcast(&pool->work_cv);
    while (pool->active_workers > 0) {
        pthread_cond_wait(&pool->done_cv, &pool->mu);
    }
    pthread_mutex_unlock(&pool->mu);
}

void thread_pool_run(Thread_Pool *pool, i64 n_jobs, Job_Dispatch_Fn fn, void *ctx)
{
    thread_pool_run_range(pool, 0, n_jobs, fn, ctx);
}

/* ---- Task_Graph ---- */

void task_graph_init(Task_Graph *g)
{
    g->deps.items = NULL;    g->deps.count = 0;    g->deps.capacity = 0;
    g->levels_flat.items = NULL;  g->levels_flat.count = 0;  g->levels_flat.capacity = 0;
    g->level_starts.items = NULL; g->level_starts.count = 0; g->level_starts.capacity = 0;
    g->level_sizes.items = NULL;  g->level_sizes.count = 0;  g->level_sizes.capacity = 0;
    g->sealed = 0;
}

i64 task_graph_add_job(Arena *a, Task_Graph *g)
{
    I64_Array empty;
    i64 id = (i64)g->deps.count;

    ARENA_ASSERT(!g->sealed && "task_graph_add_job: graph already sealed");

    empty.items = NULL;
    empty.count = 0;
    empty.capacity = 0;
    arena_da_append(a, &g->deps, empty);
    return id;
}

void task_graph_add_dep(Arena *a, Task_Graph *g, i64 job, i64 depends_on)
{
    I64_Array *list;
    size_t i;

    ARENA_ASSERT(!g->sealed && "task_graph_add_dep: graph already sealed");
    ARENA_ASSERT(job >= 0 && (size_t)job < g->deps.count && "task_graph_add_dep: bad job id");
    ARENA_ASSERT(depends_on >= 0 && (size_t)depends_on < g->deps.count && "task_graph_add_dep: bad depends_on id");
    ARENA_ASSERT(job != depends_on && "task_graph_add_dep: a job cannot depend on itself");

    list = &g->deps.items[job];
    for (i = 0; i < list->count; i++) {
        if (list->items[i] == depends_on) return; /* already declared */
    }
    arena_da_append(a, list, depends_on);
}

i32 task_graph_seal(Arena *a, Task_Graph *g)
{
    size_t n = g->deps.count;
    Dep_Array dependents;
    I64_Array in_deg;
    I64_Array queue;
    I64_Array next_queue;
    I64_Array empty;
    size_t i, qi, processed;

    if (g->sealed) return 0;
    if (n == 0) { g->sealed = 1; return 0; }

    dependents.items = NULL; dependents.count = 0; dependents.capacity = 0;
    in_deg.items = NULL;     in_deg.count = 0;     in_deg.capacity = 0;

    for (i = 0; i < n; i++) {
        empty.items = NULL; empty.count = 0; empty.capacity = 0;
        arena_da_append(a, &dependents, empty);
        arena_da_append(a, &in_deg, (i64)g->deps.items[i].count);
    }

    for (i = 0; i < n; i++) {
        I64_Array *job_deps = &g->deps.items[i];
        size_t d;
        for (d = 0; d < job_deps->count; d++) {
            i64 dep = job_deps->items[d];
            arena_da_append(a, &dependents.items[dep], (i64)i);
        }
    }

    queue.items = NULL; queue.count = 0; queue.capacity = 0;
    for (i = 0; i < n; i++) {
        if (in_deg.items[i] == 0) arena_da_append(a, &queue, (i64)i);
    }

    processed = 0;
    while (queue.count > 0) {
        arena_da_append(a, &g->level_starts, (i64)g->levels_flat.count);
        arena_da_append(a, &g->level_sizes, (i64)queue.count);

        next_queue.items = NULL; next_queue.count = 0; next_queue.capacity = 0;
        for (qi = 0; qi < queue.count; qi++) {
            i64 job = queue.items[qi];
            size_t d;
            I64_Array *job_dependents = &dependents.items[job];

            arena_da_append(a, &g->levels_flat, job);
            processed++;

            for (d = 0; d < job_dependents->count; d++) {
                i64 downstream = job_dependents->items[d];
                in_deg.items[downstream] -= 1;
                if (in_deg.items[downstream] == 0) {
                    arena_da_append(a, &next_queue, downstream);
                }
            }
        }
        queue = next_queue;
    }

    g->sealed = 1;
    if (processed != n) return -1; /* dependency cycle: (n - processed) jobs unreachable */
    return 0;
}

typedef struct {
    Task_Graph *graph;
    Job_Dispatch_Fn fn;
    void *ctx;
    i64 level_start;
} Fgjob_Level_Ctx_;

static void task_graph_level_dispatch_(i64 i, void *ctx_)
{
    Fgjob_Level_Ctx_ *lc = (Fgjob_Level_Ctx_ *)ctx_;
    i64 job_id = lc->graph->levels_flat.items[lc->level_start + i];
    lc->fn(job_id, lc->ctx);
}

void task_graph_execute(Task_Graph *g, Thread_Pool *pool, Job_Dispatch_Fn fn, void *ctx)
{
    size_t li;

    ARENA_ASSERT(g->sealed && "task_graph_execute: call task_graph_seal(arena, g) first");

    for (li = 0; li < g->level_starts.count; li++) {
        Fgjob_Level_Ctx_ level_ctx;
        level_ctx.graph = g;
        level_ctx.fn = fn;
        level_ctx.ctx = ctx;
        level_ctx.level_start = g->level_starts.items[li];
        thread_pool_run(pool, g->level_sizes.items[li], task_graph_level_dispatch_, &level_ctx);
    }
}

void task_graph_execute_serial(Task_Graph *g, Job_Dispatch_Fn fn, void *ctx)
{
    size_t i;
    ARENA_ASSERT(g->sealed && "task_graph_execute_serial: call task_graph_seal(arena, g) first");
    for (i = 0; i < g->levels_flat.count; i++) {
        fn(g->levels_flat.items[i], ctx);
    }
}

#endif /* FGJOB_IMPLEMENTATION_ */
#endif /* FGJOB_IMPLEMENTATION */
