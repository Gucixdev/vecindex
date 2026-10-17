/* store.c - a RAG-shaped vector store on top of engine.c.
 *
 * The engine is an index KERNEL (row ids, no persistence, immutable).
 * A RAG corpus needs your own doc ids on results, incremental add,
 * delete, metadata filtering, concurrent readers, bad-input rejection,
 * and surviving a restart without rebuilding. This wraps all of that
 * around the engine without touching it.
 *
 *   store_add(_meta) / store_add_batch : append (vector, doc_id[, meta]),
 *       unit-normalised; a vector with a NaN / Inf component is rejected
 *       (return -1), never poisons the index.
 *   store_delete           : tombstone a doc_id.
 *   store_build            : compact + rebuild the committed index over
 *       every live vector; clears the pending tail. Its own arena, reset
 *       each time - rebuilds do not accumulate.
 *   store_search           : committed index + an exact scan of the not-
 *       yet-built pending tail, merged (same -||q-v||^2 scale); returns
 *       YOUR doc ids, skips tombstones. Safe to call from many threads.
 *   store_search_filter    : same, keeping only rows whose i64 meta tag
 *       matches (all_bits set) and (any_bits, if non-zero, intersect).
 *   store_save / store_load : the compacted vectors + doc ids + meta +
 *       build goal AND the built index itself to one .vxi file, so load
 *       is O(file), not O(rebuild).
 *
 * Concurrency: an rwlock - store_search(_filter) take the read lock and
 * touch no shared mutable state (own malloc scratch, engine_search_into
 * path); everything that mutates takes the write lock.
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -D_POSIX_C_SOURCE=200112L -O2 \
 *       -march=native -I. store.c -DSTORE_DEMO -lm -lpthread -o store && ./store
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE     /* pthread_rwlockattr_setkind_np */
#endif
#define ARENA_FLAT_RESERVE_BYTES ((size_t)16 << 30)
#define ENGINE_THREADS
#define ENGINE_SERIALIZE
#define PRINT_IMPLEMENTATION
#include "engine.c"
#include "print.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>
#include "hdict.c"
#include "tq.c"

/* From this many rows the TurboQuant scan is grouped into clusters (IVF) of about STORE_IVF_ROWS rows and a query
 * scans only the closest few. Below it a flat scan is fast enough (200k rows: ~1.3 ms) and always exact-ish. */
#ifndef STORE_IVF_MIN
#define STORE_IVF_MIN 600000
#endif
#ifndef STORE_IVF_ROWS
#define STORE_IVF_ROWS 2000
#endif      /* doc id -> row hash map: O(1) delete / has / get, O(1) live count */

#define STORE_MAGIC   0x34584956u   /* 'VIX4' - v4 = goal gains max_candidates */
#define STORE_VERSION 4
#define STORE_GROW    4096

typedef struct {
    FILE    *jf;            /* write-ahead journal of adds/deletes since the last full save (store_sync) */
    char    *jpath;
    i32      tq_probe;      /* IVF: clusters scanned per query (0 = the index default) */
    Tq      *tq;            /* fast mode: the TurboQuant-style flat index (rebuilt on load, not saved) */
    i32      fast;          /* 1: store_build skips the auto-tuner (engine_build_fast) */
    f32      fast_target;   /* recall@k the fast build grows nprobe to (default 0.9) */
    Arena   *arena;         /* the RagStore struct + idx_arena live here    */
    Arena    idx_arena;     /* committed index; arena_reset each build/load */
    Arena    data_arena[2]; /* vecs/ids/dead/meta - ping-pong on grow so a
                              long-lived store does not accumulate          */
    i32      data_which;
    i32      dim;

    f32     *vecs;          /* [cap*dim] unit-normalised                    */
    i64     *ids;           /* [cap] caller doc ids                         */
    i64     *meta;          /* [cap] caller i64 tag (0 = none)              */
    u8      *dead;          /* [cap] 1 = tombstoned                         */
    i32     *prev;          /* [cap] previous row with the same doc id, or -1 */
    H64      idmap;         /* doc id -> newest row carrying it             */
    i32      live;          /* rows that are not tombstoned                 */
    i32      count;
    i32      cap;
    i32      built_upto;    /* [0, built_upto) is in `idx`; the rest pends  */
    EngIndex *idx;
    EngGoal  goal;
    pthread_rwlock_t lock;
} RagStore;

/* ------------------------------------------------------------------ */

static int store_finite_(const f32 *v, i32 n)
{
    i32 d;
    for (d = 0; d < n; ++d) {
        union { f32 f; u32 u; } b;
        b.f = v[d];
        if ((b.u & 0x7f800000u) == 0x7f800000u) return 0;   /* Inf or NaN */
    }
    return 1;
}

static void store_normalise_(f32 *v, i32 dim)
{
    f32 nrm = (f32)eng_sqrt_((f64)vs_dot(v, v, dim));
    i32 d;
    if (nrm <= 0.0f) return;
    nrm = 1.0f / nrm;
    for (d = 0; d < dim; ++d) v[d] *= nrm;
}

static void store_reserve_(RagStore *s, i32 need)
{
    Arena *dst;
    i32 newcap;
    f32 *nv; i64 *ni; i64 *nm; u8 *nd; i32 *np;
    if (need <= s->cap) return;
    newcap = s->cap ? s->cap : STORE_GROW;
    while (newcap < need) newcap *= 2;

    dst = &s->data_arena[1 - s->data_which];
    arena_reset(dst);
    nv = (f32 *)arena_alloc(dst, (size_t)newcap * s->dim * sizeof(f32));
    ni = (i64 *)arena_alloc(dst, (size_t)newcap * sizeof(i64));
    nm = (i64 *)arena_alloc(dst, (size_t)newcap * sizeof(i64));
    nd = (u8  *)arena_alloc(dst, (size_t)newcap);
    np = (i32 *)arena_alloc(dst, (size_t)newcap * sizeof(i32));
    if (s->count) {
        memcpy(nv, s->vecs, (size_t)s->count * s->dim * sizeof(f32));
        memcpy(ni, s->ids,  (size_t)s->count * sizeof(i64));
        memcpy(nm, s->meta, (size_t)s->count * sizeof(i64));
        memcpy(nd, s->dead, (size_t)s->count);
        memcpy(np, s->prev, (size_t)s->count * sizeof(i32));
    }
    arena_reset(&s->data_arena[s->data_which]);
    s->data_which = 1 - s->data_which;
    s->vecs = nv; s->ids = ni; s->meta = nm; s->dead = nd; s->prev = np; s->cap = newcap;
}

RagStore *store_new(Arena *a, i32 dim)
{
    RagStore *s = (RagStore *)arena_alloc(a, sizeof(RagStore));
    memset(s, 0, sizeof(*s));
    s->arena = a;
    s->dim = dim;
    s->goal.mode = ENG_MAX_ACC;
    s->goal.k = 10;
    s->goal.max_code_bytes = 1 << 20;   /* unbounded -> most accurate config */
    s->goal.eval_queries = 0;           /* engine default (128) */
    s->goal.n_iters = 12;
    s->goal.seed = 0x5eedu;
    h64_init(&s->idmap, STORE_GROW);
    /* s->goal.max_candidates stays 0 (memset above) - no nprobe cap */
    {   /* writer-preferring: a stream of readers must not starve a
         * store_add / store_build waiting for the write lock.
         * (the *_NP name is an enum, not a macro - gate on __GLIBC__.) */
        pthread_rwlockattr_t at;
        pthread_rwlockattr_init(&at);
#ifdef __GLIBC__
        pthread_rwlockattr_setkind_np(
            &at, PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);
#endif
        pthread_rwlock_init(&s->lock, &at);
        pthread_rwlockattr_destroy(&at);
    }
    return s;
}

void store_destroy(RagStore *s)
{
    pthread_rwlock_destroy(&s->lock); h64_free(&s->idmap); tq_free(s->tq); s->tq = (Tq *)0;
    if (s->jf) fclose(s->jf);
    free(s->jpath);
    arena_free(&s->idx_arena); arena_free(&s->data_arena[0]); arena_free(&s->data_arena[1]);
}

/* IVF (large stores): how many clusters a query scans; 0 = the default (~16% of them). More = better recall, slower. */
void store_set_probe(RagStore *s, i32 nprobe) { s->tq_probe = nprobe; }

/* fast build: one k-means + q4 instead of the auto-tuner's candidate bake-off */
void store_set_fast(RagStore *s, f32 target_recall)
{
    pthread_rwlock_wrlock(&s->lock);
    s->fast = 1; s->fast_target = target_recall > 0.0f ? target_recall : 0.9f;
    if (!s->tq && s->count > 0 && s->built_upto == s->count) {   /* a loaded, already-built store */
        s->idx = (EngIndex *)0;
        s->tq = tq_build(s->vecs, s->count, s->dim, s->goal.seed, s->count >= STORE_IVF_MIN ? s->count / STORE_IVF_ROWS : 0);
    }
    pthread_rwlock_unlock(&s->lock);
}

void store_set_goal(RagStore *s, EngGoal g)
{
    pthread_rwlock_wrlock(&s->lock);
    s->goal = g;
    pthread_rwlock_unlock(&s->lock);
}

/* -1 if the vector has a non-finite component (not added), else 0. */
i32 store_add_meta(RagStore *s, const f32 *vec, i64 doc_id, i64 meta)
{
    if (!store_finite_(vec, s->dim)) return -1;
    pthread_rwlock_wrlock(&s->lock);
    store_reserve_(s, s->count + 1);
    memcpy(s->vecs + (size_t)s->count * s->dim, vec, (size_t)s->dim * sizeof(f32));
    store_normalise_(s->vecs + (size_t)s->count * s->dim, s->dim);
    s->ids[s->count] = doc_id;
    s->meta[s->count] = meta;
    s->dead[s->count] = 0;
    s->prev[s->count] = h64_get(&s->idmap, doc_id);   /* chain of rows sharing this id */
    h64_put(&s->idmap, doc_id, s->count);
    s->live += 1;
    s->count += 1;
    if (s->jf) {                                                  /* journal: 'A' id meta vec[dim] (the normalised row) */
        u8 op = 'A';
        fwrite(&op, 1, 1, s->jf); fwrite(&doc_id, 8, 1, s->jf); fwrite(&meta, 8, 1, s->jf);
        fwrite(s->vecs + (size_t)(s->count - 1) * s->dim, sizeof(f32), (size_t)s->dim, s->jf);
    }
    pthread_rwlock_unlock(&s->lock);
    return 0;
}

i32 store_add(RagStore *s, const f32 *vec, i64 doc_id)
{ return store_add_meta(s, vec, doc_id, 0); }

/* returns how many of the m rows were actually added (bad ones skipped) */
i32 store_add_batch(RagStore *s, const f32 *vecs, const i64 *doc_ids, i32 m)
{
    i32 i, ok = 0;
    for (i = 0; i < m; ++i)
        if (store_add_meta(s, vecs + (size_t)i * s->dim, doc_ids[i], 0) == 0)
            ++ok;
    return ok;
}

i32 store_delete(RagStore *s, i64 doc_id)
{
    i32 r, hit = 0;
    pthread_rwlock_wrlock(&s->lock);
    for (r = h64_get(&s->idmap, doc_id); r >= 0; r = s->prev[r])
        if (!s->dead[r]) { s->dead[r] = 1; ++hit; --s->live; }
    h64_del(&s->idmap, doc_id);
    if (s->jf) { u8 op = 'D'; fwrite(&op, 1, 1, s->jf); fwrite(&doc_id, 8, 1, s->jf); }
    pthread_rwlock_unlock(&s->lock);
    return hit;
}

i32 store_count(const RagStore *s) { return s->live; }
i32 store_pending(const RagStore *s) { return s->count - s->built_upto; }
/* 1 when a vector index exists and covers every row */
i32 store_built(const RagStore *s) { return (s->idx || s->tq) && s->built_upto == s->count; }

/* O(1) lookups through the id map. store_get_vec copies the unit-normalised vector of the newest
 * live row with that id into out[dim]; store_set_meta / store_get_meta touch its i64 tag. All return
 * 0 on success, -1 if the id is not live. */
static i32 store_live_row_(const RagStore *s, i64 doc_id)
{
    i32 r;
    for (r = h64_get(&s->idmap, doc_id); r >= 0 && s->dead[r]; r = s->prev[r]) {}
    return r;
}
/* borrowed pointer to the live row's vector (valid until the next add/build/compact; no lock -
 * for single-threaded callers such as vecdb), NULL if the id is not live */
const f32 *store_vec_ptr(const RagStore *s, i64 doc_id)
{
    i32 r = store_live_row_(s, doc_id);
    return r >= 0 ? s->vecs + (size_t)r * s->dim : (const f32 *)0;
}
i32 store_has(RagStore *s, i64 doc_id)
{
    i32 r;
    pthread_rwlock_rdlock(&s->lock);
    r = store_live_row_(s, doc_id);
    pthread_rwlock_unlock(&s->lock);
    return r >= 0;
}
i32 store_get_vec(RagStore *s, i64 doc_id, f32 *out)
{
    i32 r;
    pthread_rwlock_rdlock(&s->lock);
    r = store_live_row_(s, doc_id);
    if (r >= 0) memcpy(out, s->vecs + (size_t)r * s->dim, (size_t)s->dim * sizeof(f32));
    pthread_rwlock_unlock(&s->lock);
    return r >= 0 ? 0 : -1;
}
i32 store_set_meta(RagStore *s, i64 doc_id, i64 meta)
{
    i32 r;
    pthread_rwlock_wrlock(&s->lock);
    r = store_live_row_(s, doc_id);
    if (r >= 0) s->meta[r] = meta;
    if (r >= 0 && s->jf) { u8 op = 'M'; fwrite(&op, 1, 1, s->jf); fwrite(&doc_id, 8, 1, s->jf); fwrite(&meta, 8, 1, s->jf); }
    pthread_rwlock_unlock(&s->lock);
    return r >= 0 ? 0 : -1;
}
i32 store_get_meta(RagStore *s, i64 doc_id, i64 *meta)
{
    i32 r;
    pthread_rwlock_rdlock(&s->lock);
    r = store_live_row_(s, doc_id);
    if (r >= 0) *meta = s->meta[r];
    pthread_rwlock_unlock(&s->lock);
    return r >= 0 ? 0 : -1;
}

/* rebuild the id -> row map and the per-row chains from ids[] / dead[] */
static void store_reindex_(RagStore *s)
{
    i32 r, live = 0;
    h64_clear(&s->idmap);
    for (r = 0; r < s->count; ++r) {
        s->prev[r] = h64_get(&s->idmap, s->ids[r]);
        h64_put(&s->idmap, s->ids[r], r);
        if (!s->dead[r]) ++live;
    }
    s->live = live;
}

static i32 store_compact_(RagStore *s)
{
    i32 i, w = 0;
    for (i = 0; i < s->count; ++i) {
        if (s->dead[i]) continue;
        if (w != i) {
            memmove(s->vecs + (size_t)w * s->dim, s->vecs + (size_t)i * s->dim,
                    (size_t)s->dim * sizeof(f32));
            s->ids[w] = s->ids[i];
            s->meta[w] = s->meta[i];
        }
        s->dead[w] = 0;
        ++w;
    }
    s->count = w;
    if (s->built_upto > w) s->built_upto = w;
    store_reindex_(s);
    return w;
}

void store_build(RagStore *s)
{
    i32 w;
    pthread_rwlock_wrlock(&s->lock);
    w = store_compact_(s);
    arena_reset(&s->idx_arena);
    tq_free(s->tq); s->tq = (Tq *)0;
    /* Auto choice: with no memory budget (the default) the data-oblivious 4-bit scan + exact re-score wins
     * on accuracy and speed and needs no training. The q1-q5 / IVF / HNSW tuner (engine_build) only earns
     * its 10-70 s when the goal is a SMALL index (ENG_MIN_SIZE, or a code budget below f32 size). */
    if (s->fast || (s->goal.mode == ENG_MAX_ACC && s->goal.max_code_bytes >= s->dim * 4)) {
        s->idx = (EngIndex *)0;
        if (w > 0) s->tq = tq_build(s->vecs, w, s->dim, s->goal.seed, w >= STORE_IVF_MIN ? w / STORE_IVF_ROWS : 0);
    } else {
        s->idx = (w > 0) ? engine_build(&s->idx_arena, s->vecs, w, s->dim, s->goal) : (EngIndex *)0;
    }
    s->built_upto = w;
    pthread_rwlock_unlock(&s->lock);
}

/* ------------------------------------------------------------------ *
 * search - read-locked, own scratch, engine_search_into (arena-free)  *
 * ------------------------------------------------------------------ */

static i32 store_meta_ok_(i64 tag, i64 all_bits, i64 any_bits)
{
    if ((tag & all_bits) != all_bits) return 0;
    if (any_bits && !(tag & any_bits)) return 0;
    return 1;
}

static i32 store_search_impl_(RagStore *s, const f32 *query, i32 k,
                              i64 *out_doc_ids, f32 *out_scores,
                              i64 all_bits, i64 any_bits, int filtered)
{
    i32 dim = s->dim, over = filtered ? (k * 4 + 16) : k;
    f32 *qn  = (f32 *)malloc((size_t)dim * sizeof(f32));
    i64 *rid = (i64 *)malloc((size_t)over * sizeof(i64));
    f32 *rsc = (f32 *)malloc((size_t)over * sizeof(f32));
    EngScratch sc;
    f32 *slut = (f32 *)0;
    i32 have = 0, i, got, tail;

    memcpy(qn, query, (size_t)dim * sizeof(f32));
    store_normalise_(qn, dim);

    pthread_rwlock_rdlock(&s->lock);

    if (s->tq) {
        i32 R = filtered ? k * 40 + 256 : (k * 20 > 256 ? k * 20 : 256), n = 0, m, nth = 1;
        i32 *rows = (i32 *)malloc((size_t)R * sizeof(i32));
        if (s->tq->n >= 150000) { long c = sysconf(_SC_NPROCESSORS_ONLN); nth = c > 4 ? 4 : (c > 1 ? (i32)c : 1); }   /* the scan is memory-bound: 4 threads already saturate the bus */
        m = tq_search(s->tq, qn, R, rows, s->dead, s->meta, all_bits, any_bits, filtered, nth, s->tq_probe);
        for (i = 0; i < m; ++i) {                               /* exact re-score of the shortlist */
            i32 row = rows[i];
            if (row < 0 || row >= s->built_upto) continue;
            eng_topk_(out_doc_ids, out_scores, k, &have, s->ids[row], 2.0f * vs_dot(qn, s->vecs + (size_t)row * s->dim, s->dim) - 2.0f);
        }
        (void)n;
        free(rows);
    } else if (s->idx) {
        i32 sf = engine_scratch_floats(s->idx);
        slut = (f32 *)malloc((size_t)sf * sizeof(f32));
        sc.residual = slut;
        sc.lut      = slut + dim;
        sc.lids     = (i64 *)malloc((size_t)over * sizeof(i64));
        sc.lsc      = (f32 *)malloc((size_t)over * sizeof(f32));
        sc.probe    = (i32 *)malloc(64 * sizeof(i32));
        got = engine_search_into(s->idx, qn, over, rid, rsc, &sc);
        for (i = 0; i < got; ++i) {
            i32 row = (i32)rid[i];
            if (row < 0 || row >= s->built_upto || s->dead[row]) continue;
            if (filtered && !store_meta_ok_(s->meta[row], all_bits, any_bits)) continue;
            eng_topk_(out_doc_ids, out_scores, k, &have, s->ids[row], rsc[i]);
        }
        free(sc.lids); free(sc.lsc); free(sc.probe);
    }

    tail = s->count;
    for (i = s->built_upto; i < tail; ++i) {
        f32 nd;
        if (s->dead[i]) continue;
        if (filtered && !store_meta_ok_(s->meta[i], all_bits, any_bits)) continue;
        nd = 2.0f * vs_dot(qn, s->vecs + (size_t)i * s->dim, s->dim) - 2.0f;
        eng_topk_(out_doc_ids, out_scores, k, &have, s->ids[i], nd);
    }

    /* filter fallback: a very selective filter can exclude the whole
     * IVFPQ neighbourhood, so if the merged result is short of k, do an
     * exact filtered scan of every live row (the committed part too). */
    if (filtered && have < k) {
        i64 *seen = out_doc_ids;
        for (i = 0; i < s->count; ++i) {
            f32 nd; i32 j, dup = 0;
            if (s->dead[i] || i >= s->built_upto) continue;  /* tail done above */
            if (!store_meta_ok_(s->meta[i], all_bits, any_bits)) continue;
            for (j = 0; j < have; ++j) if (seen[j] == s->ids[i]) { dup = 1; break; }
            if (dup) continue;
            nd = 2.0f * vs_dot(qn, s->vecs + (size_t)i * s->dim, s->dim) - 2.0f;
            eng_topk_(out_doc_ids, out_scores, k, &have, s->ids[i], nd);
        }
    }

    pthread_rwlock_unlock(&s->lock);
    free(qn); free(rid); free(rsc);
    if (slut) free(slut);
    return have;
}

i32 store_search(RagStore *s, const f32 *query, i32 k,
                 i64 *out_doc_ids, f32 *out_scores)
{ return store_search_impl_(s, query, k, out_doc_ids, out_scores, 0, 0, 0); }

i32 store_search_filter(RagStore *s, const f32 *query, i32 k,
                        i64 *out_doc_ids, f32 *out_scores,
                        i64 all_bits, i64 any_bits)
{ return store_search_impl_(s, query, k, out_doc_ids, out_scores,
                            all_bits, any_bits, 1); }

/* ------------------------------------------------------------------ *
 * persistence - .vxi: metadata + compacted rows + the built index    *
 * ------------------------------------------------------------------ */

/* .vxi scalars go through engine.c's little-endian helpers (engs_wle_ /
 * engs_rle_, in scope because store.c includes engine.c with
 * ENGINE_SERIALIZE): 4-byte header words, then the EngGoal fields one by
 * one (never the raw struct - padding/enum width is not portable), then
 * the row arrays as LE f32 / i64. A big-endian host byte-swaps here. */
#define STORE_W32(f, p, n) engs_wle_((f), (p), 4, (n))
#define STORE_R32(f, p, n) engs_rle_((f), (p), 4, (n))
#define STORE_W64(f, p, n) engs_wle_((f), (p), 8, (n))
#define STORE_R64(f, p, n) engs_rle_((f), (p), 8, (n))

static int store_goal_w_(FILE *f, const EngGoal *g)
{
    i32 w[7]; union { f32 fv; i32 iv; } tr;
    w[0] = (i32)g->mode;    w[1] = g->k;         w[2] = g->max_code_bytes;
    w[3] = g->eval_queries; w[4] = g->n_iters;   w[5] = (i32)g->seed;
    w[6] = g->max_candidates;
    tr.fv = g->target_recall;
    return STORE_W32(f, w, 7) || STORE_W32(f, &tr.iv, 1) ? -1 : 0;
}
static int store_goal_r_(FILE *f, EngGoal *g)
{
    i32 w[7]; union { f32 fv; i32 iv; } tr;
    if (STORE_R32(f, w, 7) || STORE_R32(f, &tr.iv, 1)) return -1;
    g->mode = (EngMode)w[0]; g->k = w[1];  g->max_code_bytes = w[2];
    g->eval_queries = w[3];  g->n_iters = w[4]; g->seed = (u32)w[5];
    g->max_candidates = w[6];
    g->target_recall = tr.fv;
    return 0;
}

/* ---- journal: PATH.j = 'VXJ1' base_live(i32) base_bytes(i64) dim(i32), then records A/D/M ---- */
#define STORE_JMAGIC 0x314a5856u

/* flush a stdio file all the way to the disk */
static i32 store_fsync_(FILE *f)
{
    if (fflush(f) != 0) return -1;
    return fsync(fileno(f)) == 0 ? 0 : -1;
}

/* make a rename durable: fsync the directory that holds `path` */
static void store_fsync_dir_(const char *path)
{
    char *d = (char *)malloc(strlen(path) + 2), *sl;
    int fd;
    strcpy(d, path);
    sl = strrchr(d, '/');
    if (sl) { if (sl == d) sl[1] = 0; else *sl = 0; } else strcpy(d, ".");
    fd = open(d, O_RDONLY);
    if (fd >= 0) { fsync(fd); close(fd); }
    free(d);
}

static i64 store_file_size_(const char *path)
{
    FILE *f = fopen(path, "rb");
    long n;
    if (!f) return -1;
    fseek(f, 0, SEEK_END); n = ftell(f); fclose(f);
    return (i64)n;
}

static char *store_jname_(const char *path)
{
    char *j = (char *)malloc(strlen(path) + 3);
    strcpy(j, path); strcat(j, ".j");
    return j;
}

/* the base file at `path` was just written (or loaded): start a fresh journal against it */
static void store_journal_reset_(RagStore *s, const char *path)
{
    char *jn = store_jname_(path);
    u32 magic = STORE_JMAGIC;
    i32 live = s->count, dim = s->dim;
    i64 bytes = store_file_size_(path);
    if (s->jf) { fclose(s->jf); s->jf = (FILE *)0; }
    free(s->jpath); s->jpath = (char *)0;
    s->jf = fopen(jn, "wb");
    if (s->jf) {
        fwrite(&magic, 4, 1, s->jf); fwrite(&live, 4, 1, s->jf); fwrite(&bytes, 8, 1, s->jf); fwrite(&dim, 4, 1, s->jf);
        store_fsync_(s->jf);
        s->jpath = (char *)malloc(strlen(path) + 1); strcpy(s->jpath, path);
    }
    free(jn);
}

/* replay PATH.j onto a freshly loaded store; a torn last record is ignored; a journal that belongs to another
 * base is ignored. Afterwards the journal stays open for appending. */
static void store_journal_replay_(RagStore *s, const char *path)
{
    char *jn = store_jname_(path);
    FILE *f = fopen(jn, "rb");
    u32 magic; i32 live, dim; i64 bytes;
    f32 *vec = (f32 *)malloc((size_t)s->dim * sizeof(f32));
    if (f && fread(&magic, 4, 1, f) == 1 && fread(&live, 4, 1, f) == 1 && fread(&bytes, 8, 1, f) == 1 && fread(&dim, 4, 1, f) == 1
        && magic == STORE_JMAGIC && live == s->count && dim == s->dim && bytes == store_file_size_(path)) {
        long good = ftell(f);
        for (;;) {
            u8 op; i64 id, meta;
            if (fread(&op, 1, 1, f) != 1) break;
            if (op == 'A') {
                if (fread(&id, 8, 1, f) != 1 || fread(&meta, 8, 1, f) != 1 || fread(vec, sizeof(f32), (size_t)s->dim, f) != (size_t)s->dim) break;
                store_add_meta(s, vec, id, meta);
            } else if (op == 'D') {
                if (fread(&id, 8, 1, f) != 1) break;
                store_delete(s, id);
            } else if (op == 'M') {
                if (fread(&id, 8, 1, f) != 1 || fread(&meta, 8, 1, f) != 1) break;
                store_set_meta(s, id, meta);
            } else break;
            good = ftell(f);
        }
        fclose(f); f = (FILE *)0;
        /* drop a torn tail, then append after the good part. If the tail cannot be dropped, run
         * without a journal (as when fopen fails): records appended behind a torn one would never
         * be replayed. */
        if (truncate(jn, (off_t)good) == 0) {
            s->jf = fopen(jn, "ab");
            if (s->jf) { s->jpath = (char *)malloc(strlen(path) + 1); strcpy(s->jpath, path); }
        }
    } else {
        if (f) fclose(f);
        store_journal_reset_(s, path);
    }
    free(vec); free(jn);
}

i32 store_save(RagStore *s, const char *path)
{
    FILE *f;
    char *tmp;
    i32 hdr[4];
    i32 live;
    u8 has_idx;

    pthread_rwlock_wrlock(&s->lock);
    live = store_compact_(s);
    tmp = (char *)malloc(strlen(path) + 5); strcpy(tmp, path); strcat(tmp, ".tmp");
    f = fopen(tmp, "wb");                                        /* the old file stays intact until the new one is complete and on disk */
    if (!f) { free(tmp); pthread_rwlock_unlock(&s->lock); return -1; }

    hdr[0] = (i32)STORE_MAGIC; hdr[1] = STORE_VERSION;
    hdr[2] = s->dim; hdr[3] = live;
    has_idx = (u8)(s->idx != (EngIndex *)0 && s->built_upto == live);

    if (STORE_W32(f, hdr, 4)
        || store_goal_w_(f, &s->goal)
        || STORE_W32(f, s->vecs, (size_t)live * s->dim)
        || STORE_W64(f, s->ids,  (size_t)live)
        || STORE_W64(f, s->meta, (size_t)live)
        || fwrite(&has_idx, 1, 1, f) != 1
        || (has_idx && engine_save(s->idx, f))) {
        fclose(f); remove(tmp); free(tmp); pthread_rwlock_unlock(&s->lock); return -1;
    }
    if (store_fsync_(f) != 0 || fclose(f) != 0 || rename(tmp, path) != 0) { remove(tmp); free(tmp); pthread_rwlock_unlock(&s->lock); return -1; }
    free(tmp);
    store_fsync_dir_(path);
    store_journal_reset_(s, path);
    pthread_rwlock_unlock(&s->lock);
    return 0;
}

/* Cheap durable save: the first call (or one after the base changed) writes the full file; later calls only
 * flush the journal of what was added/deleted since, so the cost tracks the change, not the store size.
 * store_load replays the journal. Returns 0 on success. */
i32 store_sync(RagStore *s, const char *path)
{
    i32 r;
    if (s->jf && s->jpath && strcmp(s->jpath, path) == 0) {
        pthread_rwlock_wrlock(&s->lock);
        r = store_fsync_(s->jf);
        pthread_rwlock_unlock(&s->lock);
        return r;
    }
    return store_save(s, path);
}

RagStore *store_load(Arena *a, const char *path)
{
    FILE *f = fopen(path, "rb");
    i32 hdr[4];
    RagStore *s;
    i32 live;
    u8 has_idx;
    if (!f) return (RagStore *)0;
    if (STORE_R32(f, hdr, 4) || (u32)hdr[0] != STORE_MAGIC
        || hdr[1] != STORE_VERSION) { fclose(f); return (RagStore *)0; }

    s = store_new(a, hdr[2]);
    live = hdr[3];
    if (store_goal_r_(f, &s->goal)) { fclose(f); return (RagStore *)0; }
    store_reserve_(s, live > 0 ? live : 1);
    if (live && (STORE_R32(f, s->vecs, (size_t)live * s->dim)
              || STORE_R64(f, s->ids,  (size_t)live)
              || STORE_R64(f, s->meta, (size_t)live))) {
        fclose(f); return (RagStore *)0;
    }
    memset(s->dead, 0, (size_t)(live > 0 ? live : 1));
    s->count = live;
    store_reindex_(s);

    if (fread(&has_idx, 1, 1, f) != 1) { fclose(f); return (RagStore *)0; }
    if (has_idx) {
        s->idx = engine_load(&s->idx_arena, f);
        fclose(f);
        if (!s->idx) return (RagStore *)0;
        s->built_upto = live;
    } else {
        fclose(f);
        s->fast = 1; s->fast_target = 0.97f;                     /* fast stores are saved without an engine index */
        store_build(s);
    }
    store_journal_replay_(s, path);
    return s;
}

/* ================================================================== */
#ifdef STORE_DEMO

static u32 sd_rng_(u32 *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static f32 sd_f_(u32 *s) { return (f32)sd_rng_(s) / (f32)0xffffffffu - 0.5f; }

int main(void)
{
    static Arena arena;
    Arena *a = &arena;
    i32 dim = 128, n = 4000, per_topic, i, ok = 1, nprobed;
    u32 st = 20260901u;
    f32 *centers, *vv;
    i64 *dids;
    RagStore *s, *s2;
    i64 res[16]; f32 sc[16];
    i32 self_hit = 0, del_leak = 0, bad_rej;

    per_topic = n / 40;
    centers = (f32 *)arena_alloc(a, (size_t)40 * dim * sizeof(f32));
    vv = (f32 *)arena_alloc(a, (size_t)n * dim * sizeof(f32));
    dids = (i64 *)arena_alloc(a, (size_t)n * sizeof(i64));
    for (i = 0; i < 40 * dim; ++i) centers[i] = sd_f_(&st) * 6.0f;
    for (i = 0; i < n; ++i) {
        i32 t = (i / per_topic) % 40, d;
        for (d = 0; d < dim; ++d)
            vv[(size_t)i * dim + d] = centers[t * dim + d] + sd_f_(&st) * 0.3f;
        dids[i] = 1000 + i;
    }

    s = store_new(a, dim);
    /* add with a metadata tag = topic bit */
    for (i = 0; i < n - 200; ++i)
        store_add_meta(s, vv + (size_t)i * dim, dids[i],
                       (i64)1 << ((i / per_topic) % 40));
    store_build(s);
    for (i = n - 200; i < n; ++i)
        store_add_meta(s, vv + (size_t)i * dim, dids[i],
                       (i64)1 << ((i / per_topic) % 40));

    /* NaN rejection */
    { f32 *bad = (f32 *)arena_alloc(a, (size_t)dim * sizeof(f32));
      union { f32 f; u32 u; } q; q.u = 0x7fc00000u;   /* qNaN */
      for (i = 0; i < dim; ++i) bad[i] = 0.1f;
      bad[7] = q.f;
      bad_rej = store_add(s, bad, 999999); }
    print_cstr("store  dim="); print_i64(dim);
    print_cstr("  live="); print_i64(store_count(s));
    print_cstr("  pending="); print_i64(store_pending(s));
    print_cstr("  NaN-add rejected="); print_i64(bad_rej == -1); print_cstr("\n");
    if (bad_rej != -1) ok = 0;

    nprobed = 0;
    for (i = 0; i < n; i += 3) {
        i32 h = store_search(s, vv + (size_t)i * dim, 1, res, sc);
        ++nprobed;
        if (h > 0 && res[0] == dids[i]) ++self_hit;
    }
    print_cstr("  self-retrieval@1 = "); print_i64((i64)(100L * self_hit / nprobed));
    print_cstr("%\n");
    if (self_hit * 100 / nprobed < 90) ok = 0;

    /* metadata filter: query a topic-0 vector but restrict to topic-3.
     * Every hit must carry topic-3's bit (the fallback scan guarantees
     * we still get k of them even though topic-0's IVFPQ neighbourhood
     * has none). */
    {
        i64 fr[10]; f32 fs[10];
        i32 h = store_search_filter(s, vv + 0 * dim, 10, fr, fs,
                                    (i64)1 << 3, 0), j, wrong = 0;
        for (j = 0; j < h; ++j) {
            i32 row;
            for (row = 0; row < s->count; ++row)
                if (s->ids[row] == fr[j] && !s->dead[row]) {
                    if (!(s->meta[row] & ((i64)1 << 3))) ++wrong;
                    break;
                }
        }
        print_cstr("  filter(topic=3 from a topic-0 query): "); print_i64(h);
        print_cstr(" hits, wrong-tag="); print_i64(wrong); print_cstr("\n");
        if (wrong != 0 || h < 10) ok = 0;
    }

    for (i = 0; i < 300; ++i) store_delete(s, dids[i * 5]);
    for (i = 0; i < 300; ++i) {
        i32 h = store_search(s, vv + (size_t)(i * 5) * dim, 5, res, sc), j;
        for (j = 0; j < h; ++j) if (res[j] == dids[i * 5]) ++del_leak;
    }
    print_cstr("  after 300 deletes: live="); print_i64(store_count(s));
    print_cstr("  leaked="); print_i64(del_leak); print_cstr("\n");
    if (del_leak != 0) ok = 0;
    store_build(s);

    /* save / load - index is serialised, load must NOT rebuild */
    {
        f64 t0;
        RagStore *ld;
        i32 same = 0, tried = 0;
        if (store_save(s, "/tmp/store_demo.vxi") != 0) { print_cstr("  save FAIL\n"); ok = 0; }
        t0 = 0.0; (void)t0;
        s2 = store_load(a, "/tmp/store_demo.vxi");
        ld = s2;
        if (!ld) { print_cstr("  load FAIL\n"); ok = 0; }
        else {
            for (i = 0; i < s->count; i += 7) {
                i64 r1[3], r2[3]; f32 c1[3], c2[3];
                i32 h1 = store_search(s,  s->vecs + (size_t)i * dim, 3, r1, c1);
                i32 h2 = store_search(ld, s->vecs + (size_t)i * dim, 3, r2, c2);
                ++tried;
                if (h1 == h2 && h1 > 0 && r1[0] == r2[0]) ++same;
            }
            print_cstr("  save/load (index serialised): top-1 identical on ");
            print_i64((i64)(100L * same / tried)); print_cstr("%\n");
            if (same * 100 / tried < 98) ok = 0;
            store_destroy(ld);
        }
    }

    store_destroy(s);

    /* fast build (no auto-tuner): must retrieve and round-trip through .vxi too */
    {
        RagStore *f = store_new(a, dim), *fl;
        i32 hit = 0, tried = 0, same = 0;
        n = 6000;                                       /* above the flat-q5 shortcut */
        vv = (f32 *)arena_alloc(a, (size_t)n * dim * sizeof(f32));
        for (i = 0; i < n; ++i) {
            i32 t = (i / 150) % 40, d;
            for (d = 0; d < dim; ++d) vv[(size_t)i * dim + d] = centers[t * dim + d] + sd_f_(&st) * 0.3f;
        }
        store_set_fast(f, 0.9f);
        for (i = 0; i < n; ++i) store_add(f, vv + (size_t)i * dim, 5000 + i);
        store_build(f);
        for (i = 0; i < n; i += 5) {
            i32 h = store_search(f, vv + (size_t)i * dim, 1, res, sc);
            ++tried; if (h > 0 && res[0] == 5000 + i) ++hit;
        }
        print_cstr("  fast build: self-retrieval@1 = "); print_i64((i64)(100L * hit / tried)); print_cstr("%\n");
        if (hit * 100 / tried < 90) ok = 0;
        if (store_save(f, "/tmp/store_demo_fast.vxi") != 0) ok = 0;
        fl = store_load(a, "/tmp/store_demo_fast.vxi");
        if (!fl) ok = 0;
        else {
            tried = 0;
            for (i = 0; i < n; i += 11) {
                i64 r1[1], r2[1]; f32 c1[1], c2[1];
                i32 h1 = store_search(f, vv + (size_t)i * dim, 1, r1, c1), h2 = store_search(fl, vv + (size_t)i * dim, 1, r2, c2);
                ++tried; if (h1 == h2 && h1 > 0 && r1[0] == r2[0]) ++same;
            }
            print_cstr("  fast build: save/load top-1 identical on "); print_i64((i64)(100L * same / tried)); print_cstr("%\n");
            if (same * 100 / tried < 98) ok = 0;
            store_destroy(fl);
        }
        store_destroy(f);
    }

    /* IVF grouping of the TurboQuant scan (normally only for huge stores; exercised here on 6000 rows directly) */
    {
        Tq *flat, *ivf;
        i32 nq = 60, q, j, hit = 0, tot = 0, filt_ok = 1, all_ok = 1;
        u8 *dead = (u8 *)calloc((size_t)n, 1);
        i64 *meta = (i64 *)calloc((size_t)n, sizeof(i64));
        for (i = 0; i < n; ++i) { meta[i] = (i % 3 == 0) ? 1 : 2; dead[i] = (u8)(i % 11 == 0); }
        {   f32 *un = (f32 *)malloc((size_t)n * dim * sizeof(f32));                /* unit rows, as the store keeps them */
            for (i = 0; i < n; ++i) { f32 nr = 0; for (j = 0; j < dim; ++j) nr += vv[(size_t)i * dim + j] * vv[(size_t)i * dim + j]; nr = 1.0f / (f32)sqrt((f64)nr); for (j = 0; j < dim; ++j) un[(size_t)i * dim + j] = vv[(size_t)i * dim + j] * nr; }
            flat = tq_build(un, n, dim, 7u, 0); ivf = tq_build(un, n, dim, 7u, 60);
            for (q = 0; q < nq; ++q) {
                i32 rf[256], ri[256], rn[256], nf, ni, nn2;
                const f32 *qv = un + (size_t)((q * 97) % n) * dim;
                nf = tq_search(flat, qv, 256, rf, dead, meta, 0, 0, 0, 1, 0);
                ni = tq_search(ivf, qv, 256, ri, dead, meta, 0, 0, 0, 1, 0);           /* the default probe */
                nn2 = tq_search(ivf, qv, 256, rn, dead, meta, 0, 0, 0, 1, 60);         /* every cluster = the same rows as flat */
                for (j = 0; j < 10 && j < nf; ++j) { i32 k, f = 0; for (k = 0; k < ni; ++k) if (ri[k] == rf[j]) { f = 1; break; } hit += f; ++tot; }
                { i32 k, same = nf == nn2; for (k = 0; k < nf && k < 200 && same; ++k) { i32 z, f = 0; for (z = 0; z < nn2; ++z) if (rn[z] == rf[k]) { f = 1; break; } if (!f) same = 0; } if (!same) all_ok = 0; }
                for (j = 0; j < ni; ++j) if (dead[ri[j]]) all_ok = 0;                   /* tombstones never come back */
                ni = tq_search(ivf, qv, 64, ri, dead, meta, 1, 0, 1, 1, 60);            /* meta filter: only rows with bit 1 */
                for (j = 0; j < ni; ++j) if (!(meta[ri[j]] & 1)) filt_ok = 0;
            }
            print_cstr("  IVF scan: top-10 of the flat scan found "); print_i64((i64)(100L * hit / tot)); print_cstr("%; all clusters == flat: ");
            print_cstr(all_ok ? "yes" : "NO"); print_cstr("; meta filter respected: "); print_cstr(filt_ok ? "yes" : "NO"); print_cstr("\n");
            if (100L * hit / tot < 88 || !all_ok || !filt_ok) ok = 0;
            tq_free(flat); tq_free(ivf); free(un);
        }
        free(dead); free(meta);
    }

    /* journal: base save, then only the changes are synced; a reload replays them (torn tail tolerated) */
    {
        RagStore *j = store_new(a, dim), *jl;
        i32 jn = 500, hit = 0, gone = 0, want_live;
        FILE *g;
        store_set_fast(j, 0.9f);
        for (i = 0; i < jn; ++i) store_add(j, vv + (size_t)i * dim, 9000 + i);
        store_build(j);
        if (store_sync(j, "/tmp/store_demo_j.vxi") != 0) ok = 0;            /* first call = full save */
        for (i = jn; i < jn + 60; ++i) store_add(j, vv + (size_t)i * dim, 9000 + i);
        for (i = 0; i < 20; ++i) store_delete(j, 9000 + i * 7);
        store_set_meta(j, 9000 + 3, 42);
        if (store_sync(j, "/tmp/store_demo_j.vxi") != 0) ok = 0;            /* journal flush only */
        want_live = store_count(j);
        g = fopen("/tmp/store_demo_j.vxi.j", "ab"); if (g) { fwrite("\x41\x01\x02", 1, 3, g); fclose(g); }   /* torn record */
        jl = store_load(a, "/tmp/store_demo_j.vxi");
        if (!jl) ok = 0;
        else {
            for (i = jn; i < jn + 60; ++i) { i64 r1[1]; f32 c1[1]; if (store_search(jl, vv + (size_t)i * dim, 1, r1, c1) > 0 && r1[0] == 9000 + i) ++hit; }
            for (i = 0; i < 20; ++i) { i64 r1[3]; f32 c1[3]; i32 h = store_search(jl, vv + (size_t)(i * 7) * dim, 3, r1, c1), q2; for (q2 = 0; q2 < h; ++q2) if (r1[q2] == 9000 + i * 7) ++gone; }
            print_cstr("  journal: live "); print_i64(store_count(jl)); print_cstr(" of "); print_i64(want_live);
            print_cstr(", replayed adds found "); print_i64(hit); print_cstr("/60, deleted still visible "); print_i64(gone); print_cstr("\n");
            if (store_count(jl) != want_live || hit < 55 || gone != 0) ok = 0;
            store_destroy(jl);
        }
        store_destroy(j);
    }
    print_cstr(ok ? "  OK\n" : "  FAIL\n");
    print_flush();
    arena_free(a);
    return ok ? 0 : 1;
}
#endif /* STORE_DEMO */
