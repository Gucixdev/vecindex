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
 *       every live vector; clears the pending tail.
 *   store_search           : committed index + an exact scan of the not-
 *       yet-built pending tail, merged (same -||q-v||^2 scale); returns
 *       YOUR doc ids, skips tombstones. Safe to call from many threads.
 *   store_search_filter    : same, keeping only rows whose i64 meta tag
 *       matches (all_bits set) and (any_bits, if non-zero, intersect).
 *   store_save / store_load : the compacted vectors + doc ids + meta +
 *       build goal AND the built index itself to one .vxi file, so load
 *       is O(file), not O(rebuild).
 *   store_sync             : flush the journal of adds/deletes since the last save (PATH.j) to the disk;
 *       the write lock is held only while the stdio buffers go out, not through the fsync.
 *   store_batch_*          : several puts and deletes as one change, seen and journaled all or none (store_batch.c).
 *   store_snapshot / store_snapshot_poll : the same file, written in the background by a child process (fork:
 *       a copy-on-write image of the store at this moment) while searches and writes go on. Writes made
 *       meanwhile go to the old journal and to the new file's (PATH.jn), which takes over once the file is
 *       complete; whenever a crash comes, the file on disk and one of the journals belong together (the v7
 *       header and the journal carry the same 64-bit stamp).
 *
 * Memory, so a store can grow far past RAM:
 *   - rows live in column chunks of 65536 rows (vectors, doc ids, tags). A chunk never moves, so growing never
 *     copies anything (no realloc spike, no second copy left behind) and row pointers stay valid;
 *   - past STORE_SPILL_MIN bytes new chunks are pages of an unlinked file (store_set_spill; /var/tmp by default)
 *     instead of anonymous memory: the kernel writes them back and drops them under pressure instead of
 *     swapping or OOM-killing. A loaded store points its chunks straight into the .vxi mapping;
 *   - what a search scans (the 4-bit codes, the dead-row bitset) stays in RAM: d/2 bytes + 1 bit a row. The
 *     f32 rows are only read for the few hundred candidates re-scored per query; when those reads start missing
 *     the page cache, the kernel is asked for all of them at once (MADV_WILLNEED) instead of one fault at a time;
 *   - doc ids that arrive in increasing order (the usual case) need no id map at all: lookups binary-search the
 *     id column. Only rows added out of order go into a hash of row numbers (8 bytes a slot);
 *   - for the largest stores two knobs trade a little recall for a lot of room (store_set_codes,
 *     store_set_vec_bytes): 2-bit or 1-bit codes (d/4 or d/8 bytes a row in RAM instead of d/2) and int8 rows on
 *     disk (d + 4 bytes instead of 4d). 100M rows of 96 dimensions with 2-bit codes need 2.8 GB of RAM and 11.6 GB
 *     of disk;
 *   - store_tiers counts the two parts (hot: scanned by every query; cold: read for candidates and hits) and
 *     store_set_memory locks the hot part, or the rows too, in RAM with mlock so memory pressure cannot evict them.
 *
 * Self-tuning: store_set_target_recall(s, 0.95) and the store keeps that recall@10 on its own - every build
 * measures recall against exact search and picks the fewest IVF probes that reach it, on the queries it has
 * actually been answering (a reservoir of the last STORE_QLOG) or, before there are enough, on its own rows;
 * store_tune re-measures on demand, as the data or the traffic drift.
 *
 * Live mode: store_set_live(s, 1, grow) and nobody has to call store_build again. A thread of the store's own codes
 * each added row into its closest cluster (until then the pending tail's exact scan finds it); when the index has
 * aged - grown by `grow`, a window of new rows sitting far from every centroid (the data moved), a quarter of it
 * deleted - it rebuilds on a snapshot of the rows while searches go on with the old index, and swaps the new one in;
 * with a target recall set it re-tunes after each. The work runs on half the cores. store_maintain runs one such
 * step by hand. store_save writes the appended rows into the file's plain layout, and deletes no longer cost the
 * saved index: compaction renumbers the rows inside it instead of dropping it.
 *
 * Concurrency: an rwlock - store_search(_filter) take the read lock and
 * touch no shared mutable state (own malloc scratch, engine_search_into
 * path) except the query reservoir, behind its own mutex that a search only
 * try-locks (a busy reservoir skips the query, nobody waits); everything
 * that mutates takes the write lock.
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -D_POSIX_C_SOURCE=200112L -O2 \
 *       -march=native -Isrc tests/store_test.c -lm -lpthread -o store_test && ./store_test
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE     /* pthread_rwlockattr_setkind_np, sync_file_range, RUSAGE_THREAD */
#endif
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64    /* 64-bit file offsets on 32-bit CPUs too: the journal and the spill file can grow past 2 GB */
#endif
#define ARENA_FLAT_RESERVE_BYTES ((size_t)1 << (sizeof(size_t) > 4 ? 34 : 28))  /* 16 GB of address space; 256 MB on 32-bit */
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
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <sys/syscall.h>
#include <errno.h>
#include <sched.h>
#include "hdict.c"
#include "lex.c"
#include "tq.c"

#define STORE_HAS_SPILL 1

/* From this many rows the TurboQuant scan is grouped into clusters (IVF) and a query scans only the closest few.
 * Below it a flat scan is fast enough (200k rows: ~1.3 ms) and always exact-ish. The cluster count follows the usual
 * rule, 4 sqrt(rows) (1.2M rows: 4351 clusters of ~270; 100M: 40000 of 2500), but no cluster is planned smaller
 * than STORE_IVF_ROWS rows. Finer clusters than the old fixed 2000 rows: 30-40% more queries a second at the same
 * recall on GloVe-100, and a faster build. */
#ifndef STORE_IVF_MIN
#define STORE_IVF_MIN 600000
#endif
#ifndef STORE_IVF_ROWS
#define STORE_IVF_ROWS 256
#endif
static i32 store_ivf_min_ = STORE_IVF_MIN;   /* the self-test lowers it to reach the IVF paths on a small store */
static i32 store_nclusters_(i32 n)
{
    i32 a = (i32)(4.0 * eng_sqrt_((f64)n)), b = n / STORE_IVF_ROWS;
    return n < store_ivf_min_ ? 0 : (a < b ? a : b);
}
#ifndef STORE_QLOG
#define STORE_QLOG 256      /* queries kept for tuning (a reservoir sample of everything answered) */
#endif
#ifndef STORE_SPILL_MIN
#define STORE_SPILL_MIN ((i64)256 << 20)   /* column bytes kept in anonymous memory before chunks go to the spill file */
#endif

/* exact re-scoring touches rows at random: ask for the row a few candidates ahead while scoring this one */
#define STORE_PF_AHEAD 4
static void store_prefetch_row_(const f32 *row, i32 dim)
{
#if defined(__GNUC__)
    const char *p = (const char *)row, *end = (const char *)(row + dim);
    for (; p < end; p += 64) __builtin_prefetch(p, 0, 1);
#else
    (void)row; (void)dim;
#endif
}

#define STORE_MAGIC   0x34584956u   /* 'VIX4' - v4 = goal gains max_candidates */
#define STORE_VERSION 5             /* v5: 64-byte aligned sections, the scan index saved too, loaded by mmap */
#define STORE_VERSION_I8 6          /* v6: v5 + a 16-byte extension after the header (int8 rows) and their scales */
#define STORE_VERSION7 7            /* v7: the extension on every file - flags (bit 0: int8 rows), 0, and the 64-bit
                                     * stamp its journal carries too */
#define STORE_VERSION8 8            /* v8: v7 + texts (flags bit 1): after the meta column, the heap's length (u64),
                                     * each row's text offset | length << 40 (u64), then the heap; written only by a
                                     * store that holds texts */
#define STORE_VERSION9 9            /* v9: the v7/v8 layout (texts: flags bit 1), then block digests, Reed-Solomon
                                     * parity and two copies of the table of digests and its trailer (store_sum.c) */
#define STORE_LABELS 64             /* labels: names for the 64 bits of a row's tag (store_label.c) */
#define STORE_LABEL_NAME 64         /* bytes of a label's name, its 0 included */
#define STORE_SUM_K 32              /* data blocks a parity group, by default */
#define STORE_SUM_M 2               /* parity blocks a group, by default: any 2 damaged blocks of the 32 come back */

/* ---- column chunks ---- */
#define STORE_CSHIFT 16                           /* 65536 rows a chunk */
#define STORE_CROWS  ((i64)1 << STORE_CSHIFT)
enum { STORE_ANON = 0, STORE_SPILL = 1, STORE_MAPPED = 2 };

typedef struct {
    u8   **chunk;       /* [nchunk] chunk base pointers */
    u8    *how;         /* [nchunk] STORE_ANON / STORE_SPILL (offset in foff) / STORE_MAPPED (inside the loaded file) */
    i64   *foff;        /* [nchunk] spill-file offset of a STORE_SPILL chunk */
    i32    nchunk, ccap;
    size_t esz;         /* bytes a row */
    size_t elem;        /* bytes a scalar (4: f32 rows, 8: i64), for the little-endian file format */
} SCol;
#define SCOL_AT(c, r) ((c)->chunk[(size_t)(r) >> STORE_CSHIFT] + ((size_t)(r) & (size_t)(STORE_CROWS - 1)) * (c)->esz)

/* doc id -> row for rows added out of order: open addressing over row numbers (row + 1, 0 = empty) with a 32-bit
 * tag of the id's hash beside it, so a probe reads the id column only when the tags match */
typedef struct { u32 *row, *tag; u32 cap, n, bits; } SIdx;

/* the text heap: blocks of 1 MB, block i holding heap bytes [i << 20, (i + 1) << 20) */
typedef struct {
    u8     **b;             /* [n] block base pointers */
    u8      *how;           /* [n] STORE_ANON / STORE_SPILL / STORE_MAPPED */
    i32      n, cap;
    i64      used;          /* the next free byte */
} STx;

typedef struct {
    FILE    *jf;            /* write-ahead journal of adds/deletes since the last full save (store_sync) */
    char    *jpath;
    u64      jseq;          /* journal writes so far (under the write lock); a durable batch waits until jsynced >= its */
    u64      jsynced;       /* the jseq an fsync last covered (under jsync_mu) */
    pthread_mutex_t jsync_mu;   /* one fsync at a time: commits that arrive meanwhile find theirs covered */
    u64      stamp;         /* the stamp of the file the journal belongs to (v7); 0: an older file, or none */
    /* background snapshot (store_snapshot): one child process at a time writes the store's image to snap_path */
    pthread_mutex_t snap_mu;/* guards the fields below; store_save / store_build hold it through their compaction */
    i32      snap_pid;      /* the child writing a snapshot; 0: none */
    u64      snap_stamp;    /* the stamp it writes */
    char    *snap_path;
    FILE    *jf2;           /* the snapshot replaces the journal's own file: the new file's journal, fed beside jf */
    char    *jn2;           /* its name (PATH.jn) */
    i32      snap_run;      /* 1 while a child writes (read without snap_mu, atomically) */
    i32      ro;            /* opened read-only (store_load_ro): never writes its file or journal */
    i32      mem_tier;      /* store_set_memory: 0 the kernel decides, 1 the scanned index locked in RAM, 2 the rows too */
    i32      snap_last;     /* the last finished one: 0 ok, -1 failed, 2 none yet */
    i64      snap_t0_us;    /* when the running one started (CLOCK_MONOTONIC) */
    i64      snap_took_us;  /* how long the last finished one took */
    i64      snap_done;     /* snapshots finished so far, ok or not (read atomically) */
    i32      tq_probe;      /* IVF: clusters scanned per query (0 = the index default) */
    i32      tq_threads;    /* threads one query's scan may use (0 = auto: up to 4 on big stores) */
    f32      aggr;          /* how much of the machine to use, 0..1 (store_set_aggression) */
    i32      tq_rerank;     /* candidates re-scored exactly per query (0 = auto: max(20 k, 256)) */
    f32      tune_target;   /* recall@10 every build keeps by choosing the probes (0 = off) */
    f32      tune_recall;   /* what the last tuning measured (-1: never tuned) */
    f32     *qlog;          /* [STORE_QLOG x dim] a reservoir sample of the queries answered, unit length */
    i32      qlog_n;
    i64      qlog_seen;
    u32      qlog_rng;
    pthread_mutex_t qlog_mu;
    i32      tuned_n;       /* rows in the index when it was last tuned (0: not tuned since it was built) */
    i32      tuned_log;     /* that tuning ran on logged traffic (not on the store's own rows) */
    /* live mode (store_set_live): a thread of the store's own indexes new rows and rebuilds the index as it ages */
    i32      live_on;
    f32      live_grow;     /* rebuild once the rows added since the build reach this share of the rows built; 0: never */
    pthread_t mt;
    pthread_mutex_t mt_mu;  /* held through one maintenance step; store_build / store_save / store_tune take it too */
    pthread_mutex_t rb_mu;  /* held by a rebuild from its snapshot to its swap: store_build / store_save wait for it */
    pthread_t rb_th;        /* the rebuild running beside the live thread (rb_live: not joined yet) */
    i32      rb_live;
    pthread_mutex_t mt_cmu; /* with mt_cv: wakes the thread (rows pending, or stop) */
    pthread_cond_t  mt_cv;
    i32      mt_stop;
    i32      dead_built;    /* tombstones the last rebuild already left out of the index */
    i64      mods, rebuilt_mods;      /* adds + deletes so far, and when the last rebuild started */
    f64      win_sum;       /* the added rows' squared distances to their centroids, this window */
    i32      win_n, drifted;            /* rows in the window; windows that sat too far since the build */
    i64      n_indexed, n_rebuilds, rebuild_us;   /* what maintenance did (store_live_stats) */
    i64      rebuilding;    /* 1 while a live rebuild runs */
    Tq      *tq;            /* fast mode: the TurboQuant-style index */
    i32      fast;          /* 1: store_build skips the auto-tuner (engine_build_fast) */
    f32      fast_target;   /* recall@k the fast build grows nprobe to (default 0.9) */
    Arena   *arena;         /* the RagStore struct lives here */
    Arena    idx_arena;     /* committed engine index; arena_reset each build/load */
    i32      dim;

    SCol     vec, idc, metac;   /* the rows: unit vectors, caller doc ids, caller i64 tags (0 = none) */
    SCol     vsc;           /* int8 rows only: the f32 scale of each row */
    SCol     rck;           /* a 32-bit check of each row (its vector, id and tag), tested whenever a search uses it */
    SCol     nrm;           /* dot and L2 stores only: each row's length before it was normalised */
    i32      metric;        /* 0 cosine, 1 inner product, 2 L2 distance (store_set_metric; saved in the file) */
    char     lab_name[STORE_LABELS][STORE_LABEL_NAME];   /* labels: the name of each tag bit ("": free) */
    u64      lab_par[STORE_LABELS];   /* and the bits of its parents */
    i32      nlab;          /* labels in use */
    i32      vbytes;        /* 4: f32 rows; 1: int8 rows (each row times its scale) */
    i32      codebits;      /* 4, 2 or 1: the scan index's code width (store_set_codes) */
    i32      cascade;       /* 4-bit index with 1-bit codes beside: scan those, re-score a shortlist (store_set_cascade) */
    f32     *addbuf;        /* [dim] the row being added, normalised (under the write lock) */
    u64     *dead;          /* tombstones, one bit a row */
    i64      dead_words;
    i32      live;          /* rows that are not tombstoned */
    i32      count;
    i32      built_upto;    /* [0, built_upto) is in the index; the rest pends */
    EngIndex *idx;
    EngGoal  goal;
    pthread_rwlock_t lock;
    void    *map;           /* store_load's private copy-on-write mapping of the file (columns and tq point in) */
    size_t   map_len;
    i64      sum_blocks, sum_bad, sum_fixed;   /* what the load's checks found (store_integrity) */
    i32      sum_k, sum_m;  /* parity of the files this store writes (store_set_parity) */
    void    *msum;          /* the loaded file's layout and digests (an SSum), to rebuild blocks a row comes from */
    int      mfd;           /* that file, open for the rebuilds (-1: none) */
    pthread_mutex_t heal_mu; /* one rebuild at a time */
    i32      vreads;        /* test every row a search uses against its check (store_set_check_reads; default on) */
    i64      rot_rows, rot_healed;   /* rows that failed their check and stayed out, and those rebuilt from parity */
    i32      jdamaged;      /* the journal had a damaged record or header: its copy is PATH.j.damaged */

    i32      idmap_ok;      /* 0: nsorted / ids_h / dup are rebuilt on the next lookup (after a load or a compaction) */
    i32      nsorted;       /* rows [0, nsorted) carry strictly increasing ids: found by binary search, nothing stored */
    SIdx     ids_h;         /* the other rows: id -> newest row */
    H64      dup;           /* row -> the previous row with the same id (only rows that repeat an id) */

    char    *spill_dir;     /* NULL: never spill */
    i64      spill_min;     /* spill once the columns hold this many bytes */
    int      spill_fd;      /* -1 until the first spilled chunk */
    i64      spill_off;
    i64      col_bytes;     /* bytes of column chunks allocated so far */
    i32      pf_cold;       /* re-scoring missed the page cache lately: prefetch candidates with MADV_WILLNEED */
    u32      pf_tick;

    /* texts (store_put_text): a row's text sits in a byte heap of 1 MB blocks (block i holds heap bytes [i << 20,
     * (i + 1) << 20)); the txo column holds offset | length << 40, 0 = none. Replaced and deleted texts stay in the
     * heap until store_save rewrites it. */
    SCol     txo;
    i32      has_text;      /* the txo column exists (the first text added or loaded) */
    SCol     numc;          /* a caller i64 number a row (a time, a price) for range filters (0 = none set) */
    i32      has_num;       /* the numc column exists (the first number set or loaded) */
    STx      tx;            /* the heap */
    i64      txlive;        /* bytes of the live rows' texts */
    Lex     *lex;           /* BM25 over the live rows' texts: built by the first hybrid search, kept up to date after */
    pthread_mutex_t lex_mu; /* that first build */
} RagStore;

/* row accessors (also for tests and tools) */
#define STORE_VEC(s, r)  ((f32 *)(void *)SCOL_AT(&(s)->vec, r))      /* f32 stores only (vbytes == 4) */
#define STORE_I8(s, r)   ((i8 *)(void *)SCOL_AT(&(s)->vec, r))
#define STORE_SC(s, r)   (*(f32 *)(void *)SCOL_AT(&(s)->vsc, r))
#define STORE_ID(s, r)   (*(i64 *)(void *)SCOL_AT(&(s)->idc, r))
#define STORE_META(s, r) (*(i64 *)(void *)SCOL_AT(&(s)->metac, r))
#define STORE_RCK(s, r)  (*(u32 *)(void *)SCOL_AT(&(s)->rck, r))
#define STORE_NRM(s, r)  (*(f32 *)(void *)SCOL_AT(&(s)->nrm, r))
#define STORE_DEAD(s, r) ((i32)(((s)->dead[(size_t)(r) >> 6] >> ((u32)(r) & 63u)) & 1u))
#define STORE_KILL(s, r) ((s)->dead[(size_t)(r) >> 6] |= (u64)1 << ((u32)(r) & 63u))
#define STORE_TXO(s, r)  (*(u64 *)(void *)SCOL_AT(&(s)->txo, r))     /* text rows only (has_text) */
#define STORE_NUM(s, r)  (*(i64 *)(void *)SCOL_AT(&(s)->numc, r))    /* numbered stores only (has_num) */
#define STORE_TXSHIFT    20                                          /* 1 MB heap blocks */
#define STORE_TXB        ((i64)1 << STORE_TXSHIFT)
#define STORE_TXMAX      ((1 << 24) - 1)                             /* bytes a text, at most */
#define STORE_TXLEN(v)   ((i32)((v) >> 40))
#define STORE_TXOFF(v)   ((i64)((v) & (((u64)1 << 40) - 1u)))

static TqCol store_tqcol_(const SCol *c)
{
    TqCol t;
    t.chunk = c->chunk; t.shift = STORE_CSHIFT; t.esz = c->esz; t.i8 = 0; t.dim = 0; t.scale = (const TqCol *)0;
    return t;
}

/* the vector rows as the index builder reads them (int8 rows carry their scale column along) */
static TqCol store_vcol_(const RagStore *s, TqCol *sc)
{
    TqCol t = store_tqcol_(&s->vec);
    if (s->vbytes == 1) { *sc = store_tqcol_(&s->vsc); t.i8 = 1; t.dim = s->dim; t.scale = sc; }
    return t;
}

/* ---- chunk allocation: anonymous memory, or pages of the spill file once the store is big ---- */

static void store_spill_open_(RagStore *s)
{
    char *tmpl;
    int fd;
    if (!s->spill_dir) return;
    tmpl = (char *)malloc(strlen(s->spill_dir) + 32);
    sprintf(tmpl, "%s/vecindex-spill-XXXXXX", s->spill_dir);
    fd = mkstemp(tmpl);
    if (fd >= 0) { unlink(tmpl); fcntl(fd, F_SETFD, FD_CLOEXEC); s->spill_fd = fd; s->spill_off = 0; }
    else { free(s->spill_dir); s->spill_dir = (char *)0; }      /* no usable directory: stay in memory */
    free(tmpl);
}

static i32 store_pin_(const RagStore *s, const void *p, size_t len, i32 on);

static u8 *store_chunk_alloc_(RagStore *s, size_t bytes, u8 *how, i64 *foff)
{
    void *p;
    if (s->spill_dir && s->col_bytes + (i64)bytes > s->spill_min) {
        if (s->spill_fd < 0) store_spill_open_(s);
        if (s->spill_fd >= 0 && ftruncate(s->spill_fd, (off_t)(s->spill_off + (i64)bytes)) == 0) {
            p = mmap((void *)0, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, s->spill_fd, (off_t)s->spill_off);
            if (p != MAP_FAILED) {
                *how = STORE_SPILL; *foff = s->spill_off;
                s->spill_off += (i64)bytes; s->col_bytes += (i64)bytes;
                if (s->mem_tier >= 2) store_pin_(s, p, bytes, 1);   /* rows are to stay in RAM: these too */
                return (u8 *)p;
            }
        }
    }
    p = mmap((void *)0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) return (u8 *)0;
    *how = STORE_ANON; *foff = -1;
    s->col_bytes += (i64)bytes;
    return (u8 *)p;
}

static i32 scol_push_(RagStore *s, SCol *c)
{
    u8 *p, how;
    i64 off;
    if (c->nchunk == c->ccap) {
        i32 nc = c->ccap ? c->ccap * 2 : 16;
        c->chunk = (u8 **)realloc(c->chunk, (size_t)nc * sizeof(u8 *));
        c->how = (u8 *)realloc(c->how, (size_t)nc);
        c->foff = (i64 *)realloc(c->foff, (size_t)nc * sizeof(i64));
        c->ccap = nc;
    }
    p = store_chunk_alloc_(s, (size_t)STORE_CROWS * c->esz, &how, &off);
    if (!p) return -1;
    c->chunk[c->nchunk] = p; c->how[c->nchunk] = how; c->foff[c->nchunk] = off;
    ++c->nchunk;
    return 0;
}

/* a chunk that points into the loaded file is read-mostly: before rows are appended to it, copy it out */
static i32 scol_unshare_(RagStore *s, SCol *c, i32 ci, i64 rows_in_use)
{
    u8 *p, how;
    i64 off;
    if (ci >= c->nchunk || c->how[ci] != STORE_MAPPED) return 0;
    p = store_chunk_alloc_(s, (size_t)STORE_CROWS * c->esz, &how, &off);
    if (!p) return -1;
    memcpy(p, c->chunk[ci], (size_t)rows_in_use * c->esz);
    c->chunk[ci] = p; c->how[ci] = how; c->foff[ci] = off;
    return 0;
}

static void scol_free_(SCol *c)
{
    i32 i;
    for (i = 0; i < c->nchunk; ++i)
        if (c->how[i] != STORE_MAPPED) munmap(c->chunk[i], (size_t)STORE_CROWS * c->esz);
    free(c->chunk); free(c->how); free(c->foff);
    c->chunk = (u8 **)0; c->how = (u8 *)0; c->foff = (i64 *)0; c->nchunk = c->ccap = 0;
}

/* ---- the text heap ---- */

static i32 store_tx_push_(RagStore *s, STx *x)
{
    u8 *p, how;
    i64 off;
    if (x->n == x->cap) {
        i32 nc = x->cap ? x->cap * 2 : 16;
        u8 **nb = (u8 **)realloc(x->b, (size_t)nc * sizeof(u8 *));
        u8 *nh;
        if (!nb) return -1;
        x->b = nb;
        if (!(nh = (u8 *)realloc(x->how, (size_t)nc))) return -1;
        x->how = nh; x->cap = nc;
    }
    p = store_chunk_alloc_(s, (size_t)STORE_TXB, &how, &off);
    if (!p) return -1;
    x->b[x->n] = p; x->how[x->n] = how; ++x->n;
    return 0;
}

/* n bytes onto the end of heap x (blocks come from s's allocator): the row's txo value, or 0 (out of memory) */
static u64 store_tx_put_(RagStore *s, STx *x, const char *t, i32 n)
{
    i64 off = x->used, done = 0;
    if (n <= 0) return 0;
    while ((off + n + STORE_TXB - 1) >> STORE_TXSHIFT > (i64)x->n) if (store_tx_push_(s, x)) return 0;
    while (done < n) {
        i64 at = off + done, room = STORE_TXB - (at & (STORE_TXB - 1)), take = n - done < room ? n - done : room;
        memcpy(x->b[at >> STORE_TXSHIFT] + (at & (STORE_TXB - 1)), t + done, (size_t)take);
        done += take;
    }
    x->used += n;
    return (u64)off | ((u64)(u32)n << 40);
}

/* the first n bytes of text v into buf */
static void store_tx_copy_(const STx *x, u64 v, char *buf, i64 n)
{
    i64 at = STORE_TXOFF(v), done = 0;
    while (done < n) {
        i64 p = at + done, room = STORE_TXB - (p & (STORE_TXB - 1)), take = n - done < room ? n - done : room;
        memcpy(buf + done, x->b[p >> STORE_TXSHIFT] + (p & (STORE_TXB - 1)), (size_t)take);
        done += take;
    }
}

static void store_tx_release_(STx *x)
{
    i32 i;
    for (i = 0; i < x->n; ++i) if (x->how[i] != STORE_MAPPED) munmap(x->b[i], (size_t)STORE_TXB);
    free(x->b); free(x->how);
    memset(x, 0, sizeof(*x));
}

/* row r is being tombstoned: its text no longer counts, and leaves the word index */
static void store_tx_drop_(RagStore *s, i32 r)
{
    u64 v;
    if (!s->has_text || !(v = STORE_TXO(s, r))) return;
    s->txlive -= STORE_TXLEN(v);
    if (s->lex) lex_delete(s->lex, STORE_ID(s, r));
}

static void store_tx_free_(RagStore *s) { store_tx_release_(&s->tx); s->txlive = 0; }

/* Under the write lock with every row live (after store_compact_): when at least half the heap is garbage, copy
 * the live texts into fresh blocks, in row order, and let the old ones go. Skipped when memory runs out. */
static void store_tx_compact_(RagStore *s)
{
    STx nx;
    i32 r;
    u64 *nv;
    char *buf;
    if (!s->has_text || s->tx.used - s->txlive < s->txlive || s->tx.used - s->txlive < STORE_TXB) return;
    memset(&nx, 0, sizeof(nx));
    nv = (u64 *)malloc((size_t)(s->count > 0 ? s->count : 1) * sizeof(u64));
    buf = (char *)malloc((size_t)STORE_TXMAX + 1);
    for (r = 0; nv && buf && r < s->count; ++r) {
        u64 v = STORE_TXO(s, r);
        nv[r] = 0;
        if (!v) continue;
        store_tx_copy_(&s->tx, v, buf, STORE_TXLEN(v));
        if (!(nv[r] = store_tx_put_(s, &nx, buf, STORE_TXLEN(v)))) break;
    }
    if (!nv || !buf || r < s->count) { store_tx_release_(&nx); free(nv); free(buf); return; }
    for (r = 0; r < s->count; ++r) STORE_TXO(s, r) = nv[r];
    store_tx_release_(&s->tx);
    s->tx = nx;
    free(nv); free(buf);
}

/* rows [0, need) addressable in every column, the bitset long enough. 0 ok, -1 out of memory */
static i32 store_reserve_(RagStore *s, i64 need)
{
    i32 nch = (i32)((need + STORE_CROWS - 1) >> STORE_CSHIFT), ci;
    i64 words = (need + 63) >> 6;
    while (s->vec.nchunk < nch) if (scol_push_(s, &s->vec)) return -1;
    while (s->idc.nchunk < nch) if (scol_push_(s, &s->idc)) return -1;
    while (s->metac.nchunk < nch) if (scol_push_(s, &s->metac)) return -1;
    while (s->rck.nchunk < nch) if (scol_push_(s, &s->rck)) return -1;
    while (s->metric && s->nrm.nchunk < nch) if (scol_push_(s, &s->nrm)) return -1;
    while (s->vbytes == 1 && s->vsc.nchunk < nch) if (scol_push_(s, &s->vsc)) return -1;
    while (s->has_text && s->txo.nchunk < nch) if (scol_push_(s, &s->txo)) return -1;
    while (s->has_num && s->numc.nchunk < nch) if (scol_push_(s, &s->numc)) return -1;
    ci = (i32)((i64)s->count >> STORE_CSHIFT);                   /* the chunk the next row lands in */
    if (scol_unshare_(s, &s->vec, ci, (i64)s->count - ((i64)ci << STORE_CSHIFT))
        || scol_unshare_(s, &s->idc, ci, (i64)s->count - ((i64)ci << STORE_CSHIFT))
        || scol_unshare_(s, &s->metac, ci, (i64)s->count - ((i64)ci << STORE_CSHIFT))
        || scol_unshare_(s, &s->rck, ci, (i64)s->count - ((i64)ci << STORE_CSHIFT))
        || (s->metric && scol_unshare_(s, &s->nrm, ci, (i64)s->count - ((i64)ci << STORE_CSHIFT)))
        || (s->vbytes == 1 && scol_unshare_(s, &s->vsc, ci, (i64)s->count - ((i64)ci << STORE_CSHIFT)))
        || (s->has_text && scol_unshare_(s, &s->txo, ci, (i64)s->count - ((i64)ci << STORE_CSHIFT)))
        || (s->has_num && scol_unshare_(s, &s->numc, ci, (i64)s->count - ((i64)ci << STORE_CSHIFT)))) return -1;
    if (words > s->dead_words) {
        i64 nw = s->dead_words ? s->dead_words * 2 : 1024;
        u64 *d;
        while (nw < words) nw *= 2;
        d = (u64 *)realloc(s->dead, (size_t)nw * sizeof(u64));
        if (!d) return -1;
        memset(d + s->dead_words, 0, (size_t)(nw - s->dead_words) * sizeof(u64));
        s->dead = d; s->dead_words = nw;
    }
    return 0;
}

/* a chunk of the spill file just filled up: start writing it back now, so its pages are clean (cheap to drop)
 * by the time memory gets tight, instead of piling up dirty */
static void store_chunk_done_(RagStore *s, i32 ci)
{
#if defined(__linux__) && defined(SYNC_FILE_RANGE_WRITE)
    if (s->spill_fd >= 0) {
        if (ci < s->vec.nchunk && s->vec.how[ci] == STORE_SPILL)
            sync_file_range(s->spill_fd, (off_t)s->vec.foff[ci], (off_t)(STORE_CROWS * (i64)s->vec.esz), SYNC_FILE_RANGE_WRITE);
        if (ci < s->idc.nchunk && s->idc.how[ci] == STORE_SPILL)
            sync_file_range(s->spill_fd, (off_t)s->idc.foff[ci], (off_t)(STORE_CROWS * 8), SYNC_FILE_RANGE_WRITE);
        if (ci < s->metac.nchunk && s->metac.how[ci] == STORE_SPILL)
            sync_file_range(s->spill_fd, (off_t)s->metac.foff[ci], (off_t)(STORE_CROWS * 8), SYNC_FILE_RANGE_WRITE);
        if (ci < s->vsc.nchunk && s->vsc.how[ci] == STORE_SPILL)
            sync_file_range(s->spill_fd, (off_t)s->vsc.foff[ci], (off_t)(STORE_CROWS * 4), SYNC_FILE_RANGE_WRITE);
    }
#else
    (void)s; (void)ci;
#endif
}

/* [*a, *a + the result) = [p, p + len) widened to whole pages and kept inside the loaded file's mapping */
static size_t store_pages_(const RagStore *s, const void *p, size_t len, size_t *a)
{
    size_t pg = (size_t)sysconf(_SC_PAGESIZE), e = (size_t)p + len;
    *a = (size_t)p & ~(pg - 1);
    if (s->map && (const u8 *)p >= (const u8 *)s->map && (const u8 *)p < (const u8 *)s->map + s->map_len
        && e > (size_t)s->map + s->map_len) e = (size_t)s->map + s->map_len;
    return e > *a ? e - *a : 0;
}

/* madvise on [p, p + len), widened to whole pages and kept inside the loaded file's mapping */
static void store_madv_(const RagStore *s, const void *p, size_t len, int adv)
{
    size_t a, n = store_pages_(s, p, len, &a);
    if (n) madvise((void *)a, n, adv);
}

/* Keep [p, p + len) in RAM (on: mlock) or let the kernel drop it again (munlock), widened to whole pages. A refused
 * lock (the memlock limit: ulimit -l, systemd's LimitMEMLOCK=) falls back to reading the pages ahead and gives -1. */
static i32 store_pin_(const RagStore *s, const void *p, size_t len, i32 on)
{
    size_t a, n = p ? store_pages_(s, p, len, &a) : 0;
    if (!n) return 0;
    if (!on) { munlock((void *)a, n); return 0; }
    if (mlock((void *)a, n) == 0) return 0;
    madvise((void *)a, n, MADV_WILLNEED);
    return -1;
}

/* (un)pin the parts of an index that live in the file mapping (anonymous memory is in RAM anyway): hot, what every
 * search scans - the codes, or the 1-bit codes of a cascade - with the slot table and centroids; warm, a cascade's
 * 4-bit codes, read only for its shortlist */
static i32 store_tq_pin_(const RagStore *s, const Tq *t, i32 hot, i32 warm)
{
    i32 rc = 0;
    if (!t || !t->mapped) return 0;
    if (t->codes1) {
        if (!t->c1_own) rc |= store_pin_(s, t->codes1, (size_t)t->nblk * (size_t)t->vd1 * 16, hot);
        rc |= store_pin_(s, t->codes, (size_t)t->nblk * (size_t)t->vd * 16, warm);
    } else rc |= store_pin_(s, t->codes, (size_t)t->nblk * (size_t)t->vd * 16, hot);
    if (t->slot_row && !t->slot_own) rc |= store_pin_(s, t->slot_row, (size_t)t->nblk * 32 * 4, hot);
    if (t->nc > 1) {
        rc |= store_pin_(s, t->cent, (size_t)t->nc * (size_t)t->dim * 4, hot);
        rc |= store_pin_(s, t->cstart, ((size_t)t->nc + 1) * 4, hot);
    }
    rc |= store_pin_(s, t->s1, (size_t)t->dpad * 4, hot);
    rc |= store_pin_(s, t->s2, (size_t)t->dpad * 4, hot);
    rc |= store_pin_(s, t->perm, (size_t)t->dpad * 4, hot);
    return rc ? -1 : 0;
}

/* (un)pin the rows, their int8 scales, ids and tags that live in a file (the loaded .vxi, the spill file) */
static i32 store_rows_pin_(const RagStore *s, i32 on)
{
    const SCol *cols[7];
    i32 c, i, rc = 0;
    cols[0] = &s->vec; cols[1] = &s->vsc; cols[2] = &s->idc; cols[3] = &s->metac; cols[4] = &s->rck; cols[5] = &s->nrm; cols[6] = &s->numc;
    for (c = 0; c < 7; ++c)
        for (i = 0; i < cols[c]->nchunk; ++i) {
            i64 rows = (i64)s->count - ((i64)i << STORE_CSHIFT);
            if (rows > STORE_CROWS) rows = STORE_CROWS;
            if (rows > 0 && cols[c]->how[i] != STORE_ANON
                && store_pin_(s, cols[c]->chunk[i], (size_t)rows * cols[c]->esz, on) != 0) rc = -1;
        }
    return rc;
}

/* access pattern hint for the file-backed vector chunks: sequential for a build pass, random for searching */
static void store_advise_(RagStore *s, int advice)
{
    i32 i;
    for (i = 0; i < s->vec.nchunk; ++i) {
        i64 rows = (i64)s->count - ((i64)i << STORE_CSHIFT);
        if (rows > STORE_CROWS) rows = STORE_CROWS;
        if (rows > 0 && s->vec.how[i] != STORE_ANON) store_madv_(s, s->vec.chunk[i], (size_t)rows * s->vec.esz, advice);
    }
}

/* 1 when some vector rows live in a file (spilled or loaded): re-scoring may then miss the page cache */
static i32 store_vec_filed_(const RagStore *s)
{
    return s->vec.nchunk > 0 && (s->vec.how[0] != STORE_ANON || s->vec.how[s->vec.nchunk - 1] != STORE_ANON);
}

/* ---- id lookups ---- */

static u64 store_hid_(i64 id) { return h64_mix_((u64)id); }

static void sidx_free_(SIdx *h) { free(h->row); free(h->tag); memset(h, 0, sizeof(*h)); }

static u32 sidx_home_(const SIdx *h, u32 tag) { return (u32)(tag * 2654435761u) >> (32 - h->bits); }

static void sidx_grow_(SIdx *h)
{
    SIdx o = *h;
    u32 i, bits = o.bits ? o.bits + 1 : 10;
    h->bits = bits; h->cap = 1u << bits; h->n = 0;
    h->row = (u32 *)calloc(h->cap, sizeof(u32)); h->tag = (u32 *)malloc(h->cap * sizeof(u32));
    for (i = 0; i < o.cap; ++i)
        if (o.row[i]) {
            u32 j = sidx_home_(h, o.tag[i]);
            while (h->row[j]) j = (j + 1) & (h->cap - 1);
            h->row[j] = o.row[i]; h->tag[j] = o.tag[i]; ++h->n;
        }
    free(o.row); free(o.tag);
}

static i32 sidx_get_(const RagStore *s, i64 id)
{
    const SIdx *h = &s->ids_h;
    u32 tag, j;
    if (!h->n) return -1;
    tag = (u32)(store_hid_(id) >> 32);
    for (j = sidx_home_(h, tag); h->row[j]; j = (j + 1) & (h->cap - 1))
        if (h->tag[j] == tag && STORE_ID(s, h->row[j] - 1) == id) return (i32)(h->row[j] - 1);
    return -1;
}

/* id -> r (replaces an older row of the same id) */
static void sidx_put_(RagStore *s, i64 id, i32 r)
{
    SIdx *h = &s->ids_h;
    u32 tag = (u32)(store_hid_(id) >> 32), j;
    if ((h->n + 1) * 10 > h->cap * 7) sidx_grow_(h);
    for (j = sidx_home_(h, tag); h->row[j]; j = (j + 1) & (h->cap - 1))
        if (h->tag[j] == tag && STORE_ID(s, h->row[j] - 1) == id) { h->row[j] = (u32)r + 1; return; }
    h->row[j] = (u32)r + 1; h->tag[j] = tag; ++h->n;
}

/* newest row carrying doc_id (live or not), -1 if none */
static i32 store_find_(const RagStore *s, i64 id)
{
    i32 r = sidx_get_(s, id), lo = 0, hi = s->nsorted - 1;
    if (r >= 0) return r;
    while (lo <= hi) {
        i32 mid = lo + (hi - lo) / 2;
        i64 v = STORE_ID(s, mid);
        if (v == id) return mid;
        if (v < id) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}

static i32 store_prev_(const RagStore *s, i32 r) { return s->dup.n ? h64_get(&s->dup, r) : -1; }

/* book row r (already written) into the lookup structures */
static void store_note_row_(RagStore *s, i32 r)
{
    i64 id = STORE_ID(s, r);
    if (s->nsorted == r && (r == 0 || id > STORE_ID(s, r - 1))) { s->nsorted = r + 1; return; }   /* still in order */
    {
        i32 prev = store_find_(s, id);
        if (prev >= 0) h64_put(&s->dup, r, prev);
        sidx_put_(s, id, r);
    }
}

/* rebuild the lookups and the live count from the id column and the bitset */
static void store_reindex_(RagStore *s)
{
    i32 r, live = 0;
    sidx_free_(&s->ids_h);
    h64_clear(&s->dup);
    s->nsorted = 0;
    for (r = 0; r < s->count; ++r) {
        store_note_row_(s, r);
        if (!STORE_DEAD(s, r)) ++live;
    }
    s->live = live;
}

/* The lookups are only needed to add, delete or look up by id - never to search - so a loaded store builds them
 * on first use. Writers call this under the write lock; readers through store_need_idmap_. */
static i32 store_idmap_ready_(RagStore *s)
{
#if defined(__GNUC__)
    return __atomic_load_n(&s->idmap_ok, __ATOMIC_ACQUIRE);
#else
    return s->idmap_ok;
#endif
}
static void store_idmap_locked_(RagStore *s)
{
    if (s->idmap_ok) return;
    store_reindex_(s);
#if defined(__GNUC__)
    __atomic_store_n(&s->idmap_ok, 1, __ATOMIC_RELEASE);
#else
    s->idmap_ok = 1;
#endif
}
static void store_need_idmap_(RagStore *s)
{
    if (store_idmap_ready_(s)) return;
    pthread_rwlock_wrlock(&s->lock);
    store_idmap_locked_(s);
    pthread_rwlock_unlock(&s->lock);
}

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
    f64 ss = eng_ssq_(v, dim), inv;
    i32 d;
    if (ss <= 0.0) return;
    if (ss <= 3.4e38) {                                      /* the usual case, in f32 as it always was */
        f32 nrm = 1.0f / (f32)eng_sqrt_(ss);
        for (d = 0; d < dim; ++d) v[d] *= nrm;
        return;
    }
    inv = 1.0 / eng_sqrt_(ss);
    for (d = 0; d < dim; ++d) v[d] = (f32)((f64)v[d] * inv);
}

RagStore *store_new(Arena *a, i32 dim)
{
    RagStore *s = (RagStore *)arena_alloc(a, sizeof(RagStore));
    const char *sp = getenv("VECINDEX_SPILL");
    memset(s, 0, sizeof(*s));
    s->arena = a;
    s->dim = dim;
    s->idmap_ok = 1;
    s->goal.mode = ENG_MAX_ACC;
    s->goal.k = 10;
    s->goal.max_code_bytes = 1 << 20;   /* unbounded -> most accurate config */
    s->goal.eval_queries = 0;           /* engine default (128) */
    s->goal.n_iters = 12;
    s->goal.seed = 0x5eedu;
    s->vec.esz = (size_t)dim * sizeof(f32); s->vec.elem = sizeof(f32);
    s->vsc.esz = sizeof(f32); s->vsc.elem = sizeof(f32);
    s->vbytes = 4; s->codebits = 4;
    {   const char *ag = getenv("VECINDEX_AGGRESSION");             /* the default for every store of the process */
        f64 v = ag && *ag ? strtod(ag, (char **)0) : 1.0;
        s->aggr = v < 0.0 ? 0.0f : (v > 1.0 ? 1.0f : (f32)v); }
    s->addbuf = (f32 *)malloc((size_t)dim * sizeof(f32));
    s->idc.esz = sizeof(i64); s->idc.elem = sizeof(i64);
    s->metac.esz = sizeof(i64); s->metac.elem = sizeof(i64);
    s->rck.esz = sizeof(u32); s->rck.elem = sizeof(u32);
    s->nrm.esz = sizeof(f32); s->nrm.elem = sizeof(f32);
    s->txo.esz = sizeof(u64); s->txo.elem = sizeof(u64);
    s->numc.esz = sizeof(i64); s->numc.elem = sizeof(i64);
    pthread_mutex_init(&s->lex_mu, (const pthread_mutexattr_t *)0);
    h64_init(&s->dup, 8);
    s->tune_recall = -1.0f;
    s->qlog_rng = 0x9e3779b9u;
    pthread_mutex_init(&s->qlog_mu, (const pthread_mutexattr_t *)0);
    s->live_grow = 1.0f;
    pthread_mutex_init(&s->mt_mu, (const pthread_mutexattr_t *)0);
    pthread_mutex_init(&s->rb_mu, (const pthread_mutexattr_t *)0);
    pthread_mutex_init(&s->mt_cmu, (const pthread_mutexattr_t *)0);
    pthread_cond_init(&s->mt_cv, (const pthread_condattr_t *)0);
    pthread_mutex_init(&s->snap_mu, (const pthread_mutexattr_t *)0);
    pthread_mutex_init(&s->jsync_mu, (const pthread_mutexattr_t *)0);
    s->sum_k = STORE_SUM_K; s->sum_m = STORE_SUM_M;
    s->mfd = -1; s->vreads = 1;
    pthread_mutex_init(&s->heal_mu, (const pthread_mutexattr_t *)0);
    s->snap_last = 2;
    s->spill_fd = -1;
    s->spill_min = STORE_SPILL_MIN;
    if (!sp || strcmp(sp, "off") != 0) {                     /* VECINDEX_SPILL=off keeps everything in memory */
        const char *d = sp && *sp ? sp : "/var/tmp";
        s->spill_dir = (char *)malloc(strlen(d) + 1); strcpy(s->spill_dir, d);
    }
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

/* Where chunks go once the columns pass min_bytes: a directory on a disk (not a tmpfs - that is RAM again), or
 * NULL to keep everything in anonymous memory. The default is $VECINDEX_SPILL or /var/tmp, from 256 MB. The file
 * is unlinked as soon as it is created, so nothing is left behind, even after a crash. */
void store_set_spill(RagStore *s, const char *dir, i64 min_bytes)
{
    pthread_rwlock_wrlock(&s->lock);
    free(s->spill_dir); s->spill_dir = (char *)0;
    if (dir) { s->spill_dir = (char *)malloc(strlen(dir) + 1); strcpy(s->spill_dir, dir); }
    s->spill_min = min_bytes > 0 ? min_bytes : 0;
    pthread_rwlock_unlock(&s->lock);
}

static void store_live_stop_(RagStore *s);
static i32 store_snap_reap_(RagStore *s, i32 block);

static void store_msum_free_(void *p);                          /* store_sum.c */

void store_destroy(RagStore *s)
{
    store_live_stop_(s);
    pthread_mutex_lock(&s->snap_mu);
    store_snap_reap_(s, 1);                                      /* a snapshot still being written finishes first */
    pthread_mutex_unlock(&s->snap_mu);
    pthread_mutex_destroy(&s->snap_mu);
    pthread_mutex_destroy(&s->jsync_mu);
    free(s->snap_path); s->snap_path = (char *)0;
    pthread_mutex_destroy(&s->mt_mu); pthread_mutex_destroy(&s->rb_mu); pthread_mutex_destroy(&s->mt_cmu); pthread_cond_destroy(&s->mt_cv);
    pthread_rwlock_destroy(&s->lock); tq_free(s->tq); s->tq = (Tq *)0;
    pthread_mutex_destroy(&s->qlog_mu); free(s->qlog); s->qlog = (f32 *)0;
    sidx_free_(&s->ids_h); h64_free(&s->dup);
    if (s->jf) fclose(s->jf);
    free(s->jpath); s->jpath = (char *)0;
    arena_free(&s->idx_arena);
    scol_free_(&s->vec); scol_free_(&s->idc); scol_free_(&s->metac); scol_free_(&s->vsc); scol_free_(&s->txo); scol_free_(&s->rck); scol_free_(&s->nrm); scol_free_(&s->numc);
    if (s->msum) { store_msum_free_(s->msum); s->msum = (void *)0; }
    if (s->mfd >= 0) { close(s->mfd); s->mfd = -1; }
    store_tx_free_(s);
    if (s->lex) { lex_free(s->lex); free(s->lex); s->lex = (Lex *)0; }
    pthread_mutex_destroy(&s->lex_mu);
    free(s->addbuf); s->addbuf = (f32 *)0;
    free(s->dead); s->dead = (u64 *)0; s->dead_words = 0;
    if (s->map) { munmap(s->map, s->map_len); s->map = (void *)0; }
    if (s->spill_fd >= 0) { close(s->spill_fd); s->spill_fd = -1; }
    free(s->spill_dir); s->spill_dir = (char *)0;
}

/* Bytes the store holds: ram = anonymous memory (what really competes for RAM), file = pages backed by a file
 * (the spill file or the loaded .vxi): resident only while there is room, dropped first under pressure. */
void store_mem(RagStore *s, i64 *ram, i64 *file)
{
    i64 r = 0, f = 0;
    i32 c;
    SCol *cols[8];
    pthread_rwlock_rdlock(&s->lock);
    cols[0] = &s->vec; cols[1] = &s->idc; cols[2] = &s->metac; cols[3] = &s->vsc; cols[4] = &s->txo; cols[5] = &s->rck; cols[6] = &s->nrm; cols[7] = &s->numc;
    for (c = 0; c < 8; ++c) {
        i32 i;
        for (i = 0; i < cols[c]->nchunk; ++i) {
            i64 rows = (i64)s->count - ((i64)i << STORE_CSHIFT), b;
            if (rows > STORE_CROWS) rows = STORE_CROWS;
            if (rows < 0) rows = 0;
            b = rows * (i64)cols[c]->esz;
            if (cols[c]->how[i] == STORE_ANON) r += b; else f += b;
        }
    }
    r += s->dead_words * 8 + (i64)s->ids_h.cap * 8 + (i64)s->dup.cap * 13;
    for (c = 0; c < s->tx.n; ++c) {                                  /* the text heap (the last block may be partial) */
        i64 b = s->tx.used - ((i64)c << STORE_TXSHIFT);
        if (b > STORE_TXB) b = STORE_TXB;
        if (b < 0) b = 0;
        if (s->tx.how[c] == STORE_ANON) r += b; else f += b;
    }
    if (s->tq) {
        r += tq_ram_(s->tq);
        if (s->tq->mapped) f += (i64)s->tq->nblk * s->tq->vd * 16 + (s->tq->slot_row ? (i64)s->tq->nblk * 128 : 0);
    }
    pthread_rwlock_unlock(&s->lock);
    if (ram) *ram = r;
    if (file) *file = f;
}

/* Bytes by how searches read them. hot: scanned by every query - the index's codes (built and added), its slot table
 * and centroids, the dead-row bits; this is what has to stay in RAM for searches not to wait on the disk. cold: read
 * only for the few hundred candidates a query re-scores and the hits it returns - the rows, their int8 scales, ids,
 * tags and texts; this may stay on the disk. An index from the q1-q5 / HNSW tuner (ENG_MIN_SIZE) is not counted. */
void store_tiers(RagStore *s, i64 *hot, i64 *cold)
{
    i64 h = 0, c = 0;
    const SCol *cols[8];
    const Tq *t;
    i32 i;
    pthread_rwlock_rdlock(&s->lock);
    t = s->tq;
    if (t) {                                       /* a cascade scans its 1-bit codes; its 4-bit ones are read like rows */
        h += ((i64)t->nblk + t->xcap) * (t->codes1 ? t->vd1 : t->vd) * 16 + (i64)t->xcap * 128;
        if (t->codes1) c += ((i64)t->nblk + t->xcap) * t->vd * 16;
        if (t->slot_row) h += (i64)t->nblk * 128;
        if (t->nc > 1) h += (i64)t->nc * t->dim * 8 + ((i64)t->nc + 1) * 4;  /* the centroids, and transposed */
    }
    h += (((i64)s->count + 63) >> 6) * 8;                         /* the dead-row bits of the rows there are */
    cols[0] = &s->vec; cols[1] = &s->vsc; cols[2] = &s->idc; cols[3] = &s->metac; cols[4] = &s->txo; cols[5] = &s->rck; cols[6] = &s->nrm; cols[7] = &s->numc;
    for (i = 0; i < 8; ++i) if (cols[i]->nchunk) c += (i64)s->count * (i64)cols[i]->esz;
    c += s->tx.used;
    pthread_rwlock_unlock(&s->lock);
    if (hot) *hot = h;
    if (cold) *cold = c;
}

/* What stays in RAM whatever the memory pressure (mlock), so a search never waits on the disk for it:
 *   0  nothing is locked, the kernel decides (the default): a loaded index's codes are read ahead at load, pages that
 *      every search touches stay cached, and the rarely read rows are what the kernel drops first;
 *   1  the hot part (store_tiers): the codes, slot table and centroids of a loaded index - low-bit codes in RAM,
 *      full-precision rows on the disk, read only for the candidates a query re-scores;
 *   2  the rows, scales, ids and tags kept in a file as well (the loaded .vxi, the spill file, and rows spilled later
 *      as they come): nothing a search reads waits on the disk except texts.
 * Anonymous memory (a freshly built index, rows added since the load) is in RAM anyway. A rebuild hands the old
 * index's pages back. Returns 0, or -1 when the kernel refused part of the lock (the memlock limit: ulimit -l,
 * systemd's LimitMEMLOCK=); that part is then only read ahead and may still be dropped. */
i32 store_set_memory(RagStore *s, i32 tier)
{
    i32 rc;
    tier = tier < 0 ? 0 : (tier > 2 ? 2 : tier);
    pthread_rwlock_wrlock(&s->lock);
    s->mem_tier = tier;
    pthread_rwlock_unlock(&s->lock);
    pthread_rwlock_rdlock(&s->lock);                             /* reading gigabytes in may take a while: */
    rc = store_tq_pin_(s, s->tq, tier >= 1, tier >= 2) | store_rows_pin_(s, tier >= 2);   /* searches go on meanwhile */
    pthread_rwlock_unlock(&s->lock);
    return rc ? -1 : 0;
}

/* IVF (large stores): how many clusters a query scans; 0 = the default (min(16%, 4 sqrt(clusters))). More =
 * better recall, slower. */
void store_set_probe(RagStore *s, i32 nprobe) { s->tq_probe = nprobe; }

/* Threads a single query's scan may use; 0 = auto (up to 4 from 150k rows). 1 suits callers that already run
 * one query per thread. */
void store_set_threads(RagStore *s, i32 nthreads) { s->tq_threads = nthreads > 0 ? nthreads : 0; }

/* How many candidates of the 4-bit scan are re-scored exactly; 0 = auto (max(20 k, 256)). Fewer is faster, more
 * lifts recall when the scan's ranking is coarse. */
void store_set_rerank(RagStore *s, i32 r) { s->tq_rerank = r > 0 ? r : 0; }

/* CPUs this process may run on: its affinity mask (taskset, cpusets) cut to its cgroup's CPU quota (cpu.max of cgroup
 * v2, cpu.cfs_quota_us of v1: containers, systemd CPUQuota=). Measured once. */
static i32 store_cpu_quota_(void)
{
    char line[512], path[600];
    FILE *f;
    f64 q = 0.0, per = 0.0;
    path[0] = 0;
    if ((f = fopen("/proc/self/cgroup", "r"))) {                 /* v2: "0::/the/group" */
        while (fgets(line, sizeof(line), f))
            if (!strncmp(line, "0::", 3)) { line[strcspn(line, "\n")] = 0; sprintf(path, "/sys/fs/cgroup%s/cpu.max", line + 3); break; }
        fclose(f);
    }
    if ((path[0] && (f = fopen(path, "r"))) || (f = fopen("/sys/fs/cgroup/cpu.max", "r"))) {
        if (fgets(line, sizeof(line), f) && strncmp(line, "max", 3)) sscanf(line, "%lf %lf", &q, &per);
        fclose(f);
    } else if ((f = fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r"))) {   /* v1 */
        if (fscanf(f, "%lf", &q) != 1) q = 0.0;
        fclose(f);
        if ((f = fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r"))) { if (fscanf(f, "%lf", &per) != 1) per = 0.0; fclose(f); }
    }
    return q > 0.0 && per > 0.0 ? (i32)(q / per + 0.999) : 0;
}

static i32 store_cpus_avail_(void)
{
    static i32 cached;
    i32 n = 0, q;
    if (cached) return cached;
#if defined(__linux__) && defined(CPU_COUNT)
    {   cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof(set), &set) == 0) n = CPU_COUNT(&set); }
#endif
    if (n <= 0) { long c = sysconf(_SC_NPROCESSORS_ONLN); n = c > 0 ? (i32)c : 1; }
    q = store_cpu_quota_();
    if (q > 0 && q < n) n = q;
    cached = n < 1 ? 1 : n;
    return cached;
}

/* threads for a job the caller waits on (a build, a save's indexing, a batch of queries): the CPUs available,
 * times the aggression */
static i32 store_ncpu_(const RagStore *s)
{
    i32 n = (i32)(s->aggr * (f32)store_cpus_avail_() + 0.5f);
    return n < 1 ? 1 : n;
}

/* threads for work in the background (live indexing, rebuilds, tuning): half of that, so searches keep the rest */
static i32 store_bg_threads_(const RagStore *s) { i32 c = store_ncpu_(s) / 2; return c < 1 ? 1 : c; }

/* The calling thread (and the threads it starts) yields to everything else: nice 10 at aggression 1 up to 19 at 0,
 * and the disk at the lowest best-effort priority, or only when it is idle below aggression 0.5. For the live
 * thread, rebuilds and the snapshot child. */
static void store_bg_prio_(const RagStore *s)
{
#if defined(__linux__) && defined(SYS_gettid)
    i32 tid = (i32)syscall(SYS_gettid);
    setpriority(PRIO_PROCESS, (id_t)tid, 19 - (i32)(9.0f * s->aggr + 0.5f));   /* Linux keeps a nice value per thread */
#if defined(SYS_ioprio_set)
    syscall(SYS_ioprio_set, 1, tid, s->aggr > 0.5f ? (2 << 13) | 7 : 3 << 13);   /* IOPRIO_WHO_PROCESS; BE 7, or IDLE */
#endif
#else
    (void)s;
#endif
}

/* How much of the machine the store may take, 0..1 (default 1, or $VECINDEX_AGGRESSION): the threads of builds,
 * batches and a big store's single-query scan scale with it, and its background work runs at a lower priority the
 * lower it is. 1: every CPU the process may use (its affinity and cgroup quota); 0: one thread, background at nice
 * 19 and idle I/O. */
void store_set_aggression(RagStore *s, f32 a)
{
    s->aggr = a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a);
}

static Tq *store_tq_build_(RagStore *s, i32 n)
{
    TqCol sc, col = store_vcol_(s, &sc);
    Tq *t;
    store_advise_(s, MADV_SEQUENTIAL);                           /* the build reads every row in order, twice */
    t = tq_build(&col, n, s->dim, s->goal.seed, store_nclusters_(n), store_ncpu_(s), s->codebits);
    store_advise_(s, MADV_RANDOM);                               /* searching re-scores rows at random */
    if (t && s->cascade && tq_cascade_on(t, store_ncpu_(s)) != 0) s->cascade = 0;   /* no memory for it: plain 4-bit */
    return t;
}

/* The cascade (4-bit codes only): 1-bit codes - the signs, derived from the 4-bit codes, a quarter of their bytes -
 * beside them in the index. A search scans the 1-bit codes, re-scores the best 8 x R of them with their 4-bit codes
 * and the best R of those exactly from the rows, so the part every query reads shrinks four times while the rows the
 * exact re-score reads stay as few as with 4-bit codes alone. On from now: an index that exists gets its 1-bit codes
 * at once, later builds and rebuilds make them too; a store saved with them loads with the cascade on. 0, or -1 (not
 * 4-bit codes, or no memory). */
i32 store_set_cascade(RagStore *s, i32 on)
{
    i32 r = 0;
    pthread_mutex_lock(&s->mt_mu);                               /* not in the middle of a live step */
    pthread_rwlock_wrlock(&s->lock);
    if (on && s->codebits != 4) r = -1;
    else {
        s->cascade = on != 0;
        if (s->tq && on && tq_cascade_on(s->tq, store_ncpu_(s)) != 0) { s->cascade = 0; r = -1; }
        if (s->tq && !on) tq_cascade_off(s->tq);
    }
    pthread_rwlock_unlock(&s->lock);
    pthread_mutex_unlock(&s->mt_mu);
    return r;
}

/* Code width of the scan index from the next build on: 4 (default), 2 or 1. 2-bit codes take d/4 bytes a row in RAM
 * instead of d/2 and scan twice as fast, 1-bit codes d/8 (the signs: 32x smaller than f32 rows) and four times as
 * fast, but both rank candidates more coarsely: re-score more of them (store_set_rerank; 1-bit re-scores 4x as
 * many by default) to keep the recall. */
void store_set_codes(RagStore *s, i32 bits)
{
    pthread_rwlock_wrlock(&s->lock);
    s->codebits = bits == 2 || bits == 1 ? bits : 4;
    pthread_rwlock_unlock(&s->lock);
}

/* Bytes a coordinate in the stored rows: 4 (f32, default) or 1 (int8, with an f32 scale a row: d + 4 bytes instead
 * of 4d, re-scoring error around 0.3% of a score). Only on an empty store; returns 0, or -1 if it has rows.
 * int8 rows have no f32 copy to lend out, so store_vec_ptr returns NULL; store_get_vec dequantises. */
i32 store_set_vec_bytes(RagStore *s, i32 bytes)
{
    i32 r = 0;
    pthread_rwlock_wrlock(&s->lock);
    if (s->count > 0 || s->vec.nchunk > 0) r = -1;
    else if (bytes == 1) { s->vbytes = 1; s->vec.esz = (size_t)s->dim; s->vec.elem = 1; }
    else { s->vbytes = 4; s->vec.esz = (size_t)s->dim * sizeof(f32); s->vec.elem = sizeof(f32); }
    pthread_rwlock_unlock(&s->lock);
    return r;
}

/* row r as f32: a pointer into the column, or dequantised into buf[dim] */
static const f32 *store_row_(const RagStore *s, i32 r, f32 *buf)
{
    if (s->vbytes != 1) return STORE_VEC(s, r);
    vs_i8_scale(buf, STORE_I8(s, r), s->dim, STORE_SC(s, r));
    return buf;
}

/* q . row r (q unit length): what every exact re-score uses */
static f32 store_dot_(const RagStore *s, const f32 *q, i32 r)
{
    if (s->vbytes != 1) return vs_dot(q, STORE_VEC(s, r), s->dim);
    return vs_dot_i8(q, STORE_I8(s, r), s->dim) * STORE_SC(s, r);
}

#define STORE_U64(hi, lo) (((u64)(hi) << 32) | (u64)(lo))
/* The check of row r: its vector (the f32 values, or the int8 bytes and the scale), id and tag mixed in four lanes of
 * multiply-xor, folded to 32 bits. Built from values, not memory bytes: the same on any byte order. */
static u32 store_rck_row_(const RagStore *s, i32 r)
{
    const u64 K = STORE_U64(0xff51afd7u, 0xed558ccdu);
    u64 h0 = STORE_U64(0x9e3779b9u, 0x7f4a7c15u), h1 = h0 ^ K, h2 = h0 + K, h3 = ~h0, w;
    i32 i, d = s->dim;
    if (s->vbytes != 1) {
        const f32 *v = STORE_VEC(s, r);
        u32 a[8];
        for (i = 0; i + 8 <= d; i += 8) {
            memcpy(a, v + i, sizeof(a));
            h0 = (h0 ^ ((u64)a[0] | (u64)a[1] << 32)) * K; h1 = (h1 ^ ((u64)a[2] | (u64)a[3] << 32)) * K;
            h2 = (h2 ^ ((u64)a[4] | (u64)a[5] << 32)) * K; h3 = (h3 ^ ((u64)a[6] | (u64)a[7] << 32)) * K;
        }
        for (; i < d; ++i) { memcpy(a, v + i, 4); h0 = (h0 ^ (u64)a[0]) * K; }
    } else {
        const u8 *x = (const u8 *)STORE_I8(s, r);
        u32 sc;
        for (i = 0; i < d; i += 8) {
            i32 j;
            for (w = 0, j = 0; j < 8 && i + j < d; ++j) w |= (u64)x[i + j] << (8 * j);
            h1 = (h1 ^ w) * K;
        }
        memcpy(&sc, &STORE_SC(s, r), 4);
        h2 = (h2 ^ (u64)sc) * K;
    }
    h0 = (h0 ^ (u64)STORE_ID(s, r)) * K;
    h3 = (h3 ^ (u64)STORE_META(s, r)) * K;
    if (s->metric) { u32 nb; memcpy(&nb, &STORE_NRM(s, r), 4); h1 = (h1 ^ (u64)nb) * K; }
    if (s->has_num && STORE_NUM(s, r)) h2 = (h2 ^ (u64)STORE_NUM(s, r)) * K;   /* 0 mixes nothing in: the column can
                                                                  * appear under rows that already have their checks */
    w = h0 ^ (h1 << 17 | h1 >> 47) ^ (h2 << 31 | h2 >> 33) ^ (h3 << 47 | h3 >> 17) ^ (u64)(u32)d;
    w ^= w >> 33; w *= K; w ^= w >> 29;
    return (u32)(w ^ (w >> 32));
}

static i32 store_row_heal_(RagStore *s, i32 r);                 /* store_sum.c */

/* 1 when row r fails its check - after rebuilding the file blocks it was read from, when it was - and must not be
 * used: a search leaves it out, a lookup does not find it. Counted either way. */
static i32 store_row_bad_(RagStore *s, i32 r)
{
    i32 healed;
    if (!s->vreads || STORE_RCK(s, r) == store_rck_row_(s, r)) return 0;
    healed = store_row_heal_(s, r);
#if defined(__GNUC__)
    __atomic_add_fetch(healed ? &s->rot_healed : &s->rot_rows, 1, __ATOMIC_RELAXED);
#else
    if (healed) ++s->rot_healed; else ++s->rot_rows;
#endif
    return !healed;
}

/* The score a search ranks by, bigger is better: cosine stores 2 cos - 2 (-|q - v|^2 of the unit vectors), dot
 * stores the inner product of the vectors as they came, L2 stores minus their squared distance. qn: the unit query,
 * qlen its length before. */
static f32 store_score_(const RagStore *s, const f32 *qn, f32 qlen, i32 r)
{
    f32 c = store_dot_(s, qn, r), vl;
    if (!s->metric) return 2.0f * c - 2.0f;
    vl = STORE_NRM(s, r);
    if (s->metric == 1) return qlen * vl * c;
    return -(qlen * qlen + vl * vl - 2.0f * qlen * vl * c);
}

/* that score as callers see it: the cosine (1 same direction, -1 opposite), the inner product, or the L2 distance */
f32 store_score_out(const RagStore *s, f32 x)
{
    if (!s->metric) return 1.0f + 0.5f * x;
    if (s->metric == 1) return x;
    return x < 0.0f ? (f32)eng_sqrt_((f64)-x) : 0.0f;
}

/* the metric of an empty store: 0 cosine, 1 inner product, 2 L2. 0, or -1 when it has rows or m is not one of these */
i32 store_set_metric(RagStore *s, i32 m)
{
    i32 r = 0;
    pthread_rwlock_wrlock(&s->lock);
    if (m < 0 || m > 2 || s->count > 0 || s->vec.nchunk > 0) r = -1; else s->metric = m;
    pthread_rwlock_unlock(&s->lock);
    return r;
}

/* store a normalised f32 row into row r (int8: symmetric, scale = max |v| / 127) */
static void store_put_row_(RagStore *s, i32 r, const f32 *v)
{
    i32 d;
    if (s->vbytes != 1) { memcpy(STORE_VEC(s, r), v, (size_t)s->dim * sizeof(f32)); return; }
    {   f32 mx = 0.0f, inv;
        i8 *x = STORE_I8(s, r);
        for (d = 0; d < s->dim; ++d) { f32 a = v[d] < 0.0f ? -v[d] : v[d]; if (a > mx) mx = a; }
        inv = mx > 0.0f ? 127.0f / mx : 0.0f;
        for (d = 0; d < s->dim; ++d) { f32 y = v[d] * inv; x[d] = (i8)(y < 0.0f ? y - 0.5f : y + 0.5f); }
        STORE_SC(s, r) = mx / 127.0f; }
}

/* fast build: one k-means + q4 instead of the auto-tuner's candidate bake-off */
void store_set_fast(RagStore *s, f32 target_recall)
{
    pthread_mutex_lock(&s->mt_mu);
    pthread_rwlock_wrlock(&s->lock);
    s->fast = 1; s->fast_target = target_recall > 0.0f ? target_recall : 0.9f;
    if (!s->tq && s->count > 0 && s->built_upto == s->count) {   /* a loaded, already-built store */
        s->idx = (EngIndex *)0;
        s->tq = store_tq_build_(s, s->count);
    }
    pthread_rwlock_unlock(&s->lock);
    pthread_mutex_unlock(&s->mt_mu);
}

void store_set_goal(RagStore *s, EngGoal g)
{
    pthread_rwlock_wrlock(&s->lock);
    s->goal = g;
    pthread_rwlock_unlock(&s->lock);
}

static void store_jframe_(RagStore *s, const void *const *part, const size_t *plen, i32 np);   /* store_batch.c */

/* One journal record, under the write lock: 'A' (add) and 'P' (put: replace) id meta vec[dim] (the normalised row),
 * 'D' id, 'M' id meta, 'R' lo hi (a range of ids deleted), in a frame with its CRC-32C. It goes to the journal and,
 * while a snapshot is being written over the journal's own file, to that file's journal too. */
static void store_jrec_(RagStore *s, u8 op, i64 id, i64 meta, const f32 *vec)
{
    const void *pt[4];
    size_t pl[4];
    i32 np = 0;
    pt[np] = &op; pl[np++] = 1; pt[np] = &id; pl[np++] = 8;
    if (op != 'D') { pt[np] = &meta; pl[np++] = 8; }
    if (op == 'A' || op == 'P' || op == 'U') { pt[np] = vec; pl[np++] = sizeof(f32) * (size_t)s->dim; }
    store_jframe_(s, pt, pl, np);
}

/* one more row at the end, under the write lock (vec is finite; store_reserve_ already made room): normalised into
 * s->addbuf, which keeps it for the journal record */
static void store_append_locked_(RagStore *s, const f32 *vec, i64 doc_id, i64 meta)
{
    i32 r = s->count;
    f32 *v = s->addbuf;
    f32 len = 0.0f;
    i32 d;
    memcpy(v, vec, (size_t)s->dim * sizeof(f32));
    if (s->metric) len = (f32)eng_sqrt_(eng_ssq_(v, s->dim));          /* dot and L2 rank by the length too */
    store_normalise_(v, s->dim);
    store_put_row_(s, r, v);
    STORE_ID(s, r) = doc_id;
    STORE_META(s, r) = meta;
    if (s->metric) STORE_NRM(s, r) = len;
    STORE_RCK(s, r) = store_rck_row_(s, r);
    if (s->metric) for (d = 0; d < s->dim; ++d) v[d] *= len;     /* the journal gets the row as it came */
    if (s->has_text) STORE_TXO(s, r) = 0;
    if (s->has_num) STORE_NUM(s, r) = 0;
    s->dead[r >> 6] &= ~((u64)1 << (r & 63));
    store_note_row_(s, r);
    s->live += 1;
    s->count += 1;
    ++s->mods;
    if (((i64)s->count & (STORE_CROWS - 1)) == 0) store_chunk_done_(s, (i32)(((i64)s->count >> STORE_CSHIFT) - 1));
    if (s->live_on && ((s->count - s->built_upto) & 255) == 1) pthread_cond_signal(&s->mt_cv);   /* the live thread */
}

/* tombstones every live row with this id, under the write lock; returns how many */
static i32 store_kill_id_locked_(RagStore *s, i64 doc_id)
{
    i32 r, hit = 0;
    for (r = store_find_(s, doc_id); r >= 0; r = store_prev_(s, r))
        if (!STORE_DEAD(s, r)) { STORE_KILL(s, r); ++hit; --s->live; ++s->mods; store_tx_drop_(s, r); }
    return hit;
}

/* -1 if the vector has a non-finite component (not added) or memory ran out, else 0. An id already in the store gets
 * a second row (store_put replaces instead). */
i32 store_add_meta(RagStore *s, const f32 *vec, i64 doc_id, i64 meta)
{
    if (!store_finite_(vec, s->dim)) return -1;
    pthread_rwlock_wrlock(&s->lock);
    store_idmap_locked_(s);
    if (store_reserve_(s, (i64)s->count + 1) != 0) { pthread_rwlock_unlock(&s->lock); return -1; }
    store_append_locked_(s, vec, doc_id, meta);
    store_jrec_(s, 'A', doc_id, meta, s->addbuf);
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

/* Replace (upsert): every row with doc_id goes and the new one comes in, as one change - one journal record, so a
 * crash never leaves the id with neither. Returns as store_add_meta; on -1 nothing changed. */
i32 store_put(RagStore *s, const f32 *vec, i64 doc_id, i64 meta)
{
    if (!store_finite_(vec, s->dim)) return -1;
    pthread_rwlock_wrlock(&s->lock);
    store_idmap_locked_(s);
    if (store_reserve_(s, (i64)s->count + 1) != 0) { pthread_rwlock_unlock(&s->lock); return -1; }   /* before the kill */
    store_kill_id_locked_(s, doc_id);
    store_append_locked_(s, vec, doc_id, meta);
    store_jrec_(s, 'P', doc_id, meta, s->addbuf);
    pthread_rwlock_unlock(&s->lock);
    return 0;
}

/* A new vector for doc_id, the rest of the row kept: the newest live row's tag (labels with it), number and text go
 * to the new row, every row with doc_id goes. The text is not copied - the new row points at the same bytes, and the
 * word index, keyed by the id, stays as it is. Under the write lock, vec finite; 0, or -1: no live row with doc_id,
 * a damaged one, or no memory (nothing changed). The caller journals it. */
static i32 store_live_row_(const RagStore *s, i64 doc_id);
static i32 store_update_locked_(RagStore *s, const f32 *vec, i64 doc_id)
{
    i32 r = store_live_row_(s, doc_id), q, n;
    i64 meta, num;
    u64 txo;
    if (r < 0 || store_row_bad_(s, r)) return -1;
    meta = STORE_META(s, r);
    num = s->has_num ? STORE_NUM(s, r) : 0;
    txo = s->has_text ? STORE_TXO(s, r) : 0;
    if (store_reserve_(s, (i64)s->count + 1) != 0) return -1;   /* before the kill */
    for (q = store_find_(s, doc_id); q >= 0; q = store_prev_(s, q))
        if (!STORE_DEAD(s, q)) { STORE_KILL(s, q); --s->live; ++s->mods; if (q != r) store_tx_drop_(s, q); }
    store_append_locked_(s, vec, doc_id, meta);
    n = s->count - 1;
    if (txo) STORE_TXO(s, n) = txo;                              /* txlive and the word index: unchanged */
    if (num) { STORE_NUM(s, n) = num; STORE_RCK(s, n) = store_rck_row_(s, n); }
    return 0;
}
i32 store_update(RagStore *s, const f32 *vec, i64 doc_id)
{
    i32 rc;
    if (!store_finite_(vec, s->dim)) return -1;
    pthread_rwlock_wrlock(&s->lock);
    store_idmap_locked_(s);
    rc = store_update_locked_(s, vec, doc_id);
    if (rc == 0) store_jrec_(s, 'U', doc_id, 0, s->addbuf);
    pthread_rwlock_unlock(&s->lock);
    return rc;
}

i32 store_delete(RagStore *s, i64 doc_id)
{
    i32 hit;
    pthread_rwlock_wrlock(&s->lock);
    store_idmap_locked_(s);
    hit = store_kill_id_locked_(s, doc_id);
    store_jrec_(s, 'D', doc_id, 0, (const f32 *)0);
    pthread_rwlock_unlock(&s->lock);
    return hit;
}

/* tombstones every row whose id is in [lo, hi] (a whole document when ids are doc << bits | chunk); one pass over
 * the id column; returns how many */
i32 store_delete_range(RagStore *s, i64 lo, i64 hi)
{
    i32 r, hit = 0;
    pthread_rwlock_wrlock(&s->lock);
    for (r = 0; r < s->count; ++r)
        if (!STORE_DEAD(s, r) && STORE_ID(s, r) >= lo && STORE_ID(s, r) <= hi) {
            STORE_KILL(s, r); ++hit; --s->live; ++s->mods; store_tx_drop_(s, r);
        }
    store_jrec_(s, 'R', lo, hi, (const f32 *)0);
    pthread_rwlock_unlock(&s->lock);
    return hit;
}

/* counters under the read lock: the live thread moves built_upto, adders move count and live */
i32 store_count(const RagStore *s)
{
    i32 r;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&s->lock); r = s->live; pthread_rwlock_unlock((pthread_rwlock_t *)&s->lock);
    return r;
}
i32 store_pending(const RagStore *s)
{
    i32 r;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&s->lock); r = s->count - s->built_upto; pthread_rwlock_unlock((pthread_rwlock_t *)&s->lock);
    return r;
}
/* 1 when a vector index exists and covers every row */
i32 store_built(const RagStore *s)
{
    i32 r;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&s->lock);
    r = (s->idx || s->tq) && s->built_upto == s->count;
    pthread_rwlock_unlock((pthread_rwlock_t *)&s->lock);
    return r;
}

/* O(1)/O(log n) lookups. store_get_vec copies the unit-normalised vector of the newest live row with that id into
 * out[dim]; store_set_meta / store_get_meta touch its i64 tag. All return 0 on success, -1 if the id is not live. */
static i32 store_live_row_(const RagStore *s, i64 doc_id)
{
    i32 r;
    for (r = store_find_(s, doc_id); r >= 0 && STORE_DEAD(s, r); r = store_prev_(s, r)) {}
    return r;
}
/* borrowed pointer to the live row's vector (valid until the next build/compact - adds never move rows; no
 * lock - for callers that hold their own lock), NULL if the id is not live */
const f32 *store_vec_ptr(const RagStore *s, i64 doc_id)
{
    i32 r;
    store_need_idmap_((RagStore *)s);
    if (s->vbytes == 1) return (const f32 *)0;                   /* int8 rows: no f32 copy to lend (store_get_vec) */
    r = store_live_row_(s, doc_id);
    return r >= 0 ? STORE_VEC(s, r) : (const f32 *)0;
}
i32 store_has(RagStore *s, i64 doc_id)
{
    i32 r;
    store_need_idmap_(s);
    pthread_rwlock_rdlock(&s->lock);
    r = store_live_row_(s, doc_id);
    pthread_rwlock_unlock(&s->lock);
    return r >= 0;
}
i32 store_get_vec(RagStore *s, i64 doc_id, f32 *out)
{
    i32 r;
    store_need_idmap_(s);
    pthread_rwlock_rdlock(&s->lock);
    r = store_live_row_(s, doc_id);
    if (r >= 0 && store_row_bad_(s, r)) r = -1;                  /* damaged: not served */
    if (r >= 0) { const f32 *v = store_row_(s, r, out); if (v != out) memcpy(out, v, (size_t)s->dim * sizeof(f32)); }
    if (r >= 0 && s->metric) { i32 d; f32 l = STORE_NRM(s, r); for (d = 0; d < s->dim; ++d) out[d] *= l; }   /* as it came */
    pthread_rwlock_unlock(&s->lock);
    return r >= 0 ? 0 : -1;
}
i32 store_set_meta(RagStore *s, i64 doc_id, i64 meta)
{
    i32 r;
    pthread_rwlock_wrlock(&s->lock);
    store_idmap_locked_(s);
    r = store_live_row_(s, doc_id);
    if (r >= 0 && !store_row_bad_(s, r)) { STORE_META(s, r) = meta; STORE_RCK(s, r) = store_rck_row_(s, r); }
    else r = -1;
    if (r >= 0) store_jrec_(s, 'M', doc_id, meta, (const f32 *)0);
    pthread_rwlock_unlock(&s->lock);
    return r >= 0 ? 0 : -1;
}
i32 store_get_meta(RagStore *s, i64 doc_id, i64 *meta)
{
    i32 r;
    store_need_idmap_(s);
    pthread_rwlock_rdlock(&s->lock);
    r = store_live_row_(s, doc_id);
    if (r >= 0 && store_row_bad_(s, r)) r = -1;                  /* damaged: not served */
    if (r >= 0) *meta = STORE_META(s, r);
    pthread_rwlock_unlock(&s->lock);
    return r >= 0 ? 0 : -1;
}

/* A row's number, for range filters: the first one set gives every row a slot (0 until set). Under the write lock;
 * the row (or -1: no live row with this id, or a damaged one). The caller journals it. */
static i32 store_set_num_locked_(RagStore *s, i64 doc_id, i64 num)
{
    i32 r;
    if (!s->has_num) {
        s->has_num = 1;
        if (store_reserve_(s, (i64)s->count) != 0) { s->has_num = 0; return -1; }
    }
    r = store_live_row_(s, doc_id);
    if (r < 0 || store_row_bad_(s, r)) return -1;
    {   /* a chunk still in the loaded file's mapping is copied out before the write */
        i32 ci = (i32)((i64)r >> STORE_CSHIFT);
        i64 used = (i64)s->count - ((i64)ci << STORE_CSHIFT);
        if (scol_unshare_(s, &s->numc, ci, used > STORE_CROWS ? STORE_CROWS : used) != 0) return -1;
    }
    STORE_NUM(s, r) = num;
    STORE_RCK(s, r) = store_rck_row_(s, r);
    ++s->mods;
    return r;
}
i32 store_set_num(RagStore *s, i64 doc_id, i64 num)
{
    i32 r;
    pthread_rwlock_wrlock(&s->lock);
    store_idmap_locked_(s);
    r = store_set_num_locked_(s, doc_id, num);
    if (r >= 0) store_jrec_(s, 'N', doc_id, num, (const f32 *)0);
    pthread_rwlock_unlock(&s->lock);
    return r >= 0 ? 0 : -1;
}
i32 store_get_num(RagStore *s, i64 doc_id, i64 *num)
{
    i32 r;
    store_need_idmap_(s);
    pthread_rwlock_rdlock(&s->lock);
    r = store_live_row_(s, doc_id);
    if (r >= 0 && store_row_bad_(s, r)) r = -1;
    if (r >= 0) *num = s->has_num ? STORE_NUM(s, r) : 0;
    pthread_rwlock_unlock(&s->lock);
    return r >= 0 ? 0 : -1;
}

/* tombstones among rows [0, r), with pre[w] = tombstones in words [0, w) */
static i32 store_dead_before_(const RagStore *s, const i32 *pre, i32 r)
{
    return pre[r >> 6] + ((r & 63) ? tq_pop64_(s->dead[r >> 6] & (((u64)1 << (r & 63)) - 1u)) : 0);
}

/* drop tombstoned rows (moving the survivors down, order kept); a store without tombstones is left untouched. The
 * scan index follows its rows (tq_remap); an engine index cannot, so it is dropped (its rows pend until a build). */
static i32 store_compact_(RagStore *s)
{
    i32 i, w = 0;
    if (s->live == s->count) { if (s->built_upto > s->count) s->built_upto = s->count; return s->count; }
    {
        i64 words = ((i64)s->count + 63) >> 6, x;
        i32 *pre = (i32 *)malloc((size_t)(words + 1) * sizeof(i32)), acc = 0, bu;
        for (x = 0; x <= words; ++x) { pre[x] = acc; if (x < words) acc += tq_pop64_(s->dead[x]); }
        bu = s->built_upto - store_dead_before_(s, pre, s->built_upto);
        if (s->tq) tq_remap(s->tq, s->dead, pre, bu);
        if (s->idx) { s->idx = (EngIndex *)0; arena_reset(&s->idx_arena); bu = 0; }
        s->built_upto = bu;
        s->dead_built = 0;
        free(pre);
    }
    for (i = 0; i < s->count; ++i) {
        if (STORE_DEAD(s, i)) continue;
        if (w != i) {
            memcpy(SCOL_AT(&s->vec, w), SCOL_AT(&s->vec, i), s->vec.esz);
            if (s->vbytes == 1) STORE_SC(s, w) = STORE_SC(s, i);
            STORE_ID(s, w) = STORE_ID(s, i);
            STORE_META(s, w) = STORE_META(s, i);
            STORE_RCK(s, w) = STORE_RCK(s, i);
            if (s->metric) STORE_NRM(s, w) = STORE_NRM(s, i);
            if (s->has_text) STORE_TXO(s, w) = STORE_TXO(s, i);
            if (s->has_num) STORE_NUM(s, w) = STORE_NUM(s, i);
        }
        ++w;
    }
    memset(s->dead, 0, (size_t)s->dead_words * sizeof(u64));
    s->count = w;
    s->live = w;
#if defined(__GNUC__)
    __atomic_store_n(&s->idmap_ok, 0, __ATOMIC_RELEASE);        /* rows moved: the lookups are rebuilt on next use */
#else
    s->idmap_ok = 0;
#endif
    return w;
}

static f32 store_tune_run_(RagStore *s, i32 nth);

static i32 store_uses_tq_(const RagStore *s) { return s->fast || (s->goal.mode == ENG_MAX_ACC && s->goal.max_code_bytes >= s->dim * 4); }

/* the engine index (the q1-q5 / IVF / HNSW tuner) over rows [0, w), under the write lock */
static void store_engine_build_(RagStore *s, i32 w)
{
    f32 *flat = (f32 *)malloc((size_t)w * (size_t)s->dim * sizeof(f32));   /* the tuner wants one contiguous array */
    i32 i;
    for (i = 0; i < w; ++i) { f32 *dst = flat + (size_t)i * s->dim; const f32 *v = store_row_(s, i, dst); if (v != dst) memcpy(dst, v, (size_t)s->dim * sizeof(f32)); }
    s->idx = engine_build(&s->idx_arena, flat, w, s->dim, s->goal);
    free(flat);
}

void store_build(RagStore *s)
{
    i32 w;
    pthread_mutex_lock(&s->snap_mu);                             /* a snapshot child may still read rows that the */
    store_snap_reap_(s, 1);                                      /* compaction below moves: it finishes first */
    pthread_mutex_lock(&s->rb_mu);                               /* not in the middle of a live rebuild */
    pthread_mutex_lock(&s->mt_mu);
    pthread_rwlock_wrlock(&s->lock);
    if (s->mem_tier >= 1) store_tq_pin_(s, s->tq, 0, 0);         /* its pages in the file go back to the kernel */
    tq_free(s->tq); s->tq = (Tq *)0;                             /* rebuilt below: no use remapping it */
    w = store_compact_(s);
    arena_reset(&s->idx_arena);
    s->tuned_n = 0; s->dead_built = 0; s->win_sum = 0.0; s->win_n = 0; s->drifted = 0;
    /* Auto choice: with no memory budget (the default) the data-oblivious 4-bit scan + exact re-score wins
     * on accuracy and speed and needs no training. The q1-q5 / IVF / HNSW tuner (engine_build) only earns
     * its 10-70 s when the goal is a SMALL index (ENG_MIN_SIZE, or a code budget below f32 size). */
    s->idx = (EngIndex *)0;
    if (w > 0 && store_uses_tq_(s)) s->tq = store_tq_build_(s, w);
    else if (w > 0) store_engine_build_(s, w);
    s->built_upto = w;
    pthread_rwlock_unlock(&s->lock);
    store_tune_run_(s, store_ncpu_(s));                           /* keep the recall asked for, if one is (searches go on) */
    pthread_mutex_unlock(&s->mt_mu);
    pthread_mutex_unlock(&s->rb_mu);
    pthread_mutex_unlock(&s->snap_mu);
}

/* conditions of a search: every mask of any[0..n) shared by the row's tag, none of none; with num, the row's number
 * in [lo, hi] (a store without numbers: every row's is 0) */
typedef struct { i32 n; u64 any[16]; u64 none; i32 num; i64 lo, hi; } StoreCnf;
static i32 store_cnf_ok_(const RagStore *s, const StoreCnf *c, i32 row)
{
    u64 m = (u64)STORE_META(s, row);
    i32 i;
    if (m & c->none) return 0;
    for (i = 0; i < c->n; ++i) if (!(m & c->any[i])) return 0;
    if (c->num) { i64 v = s->has_num ? STORE_NUM(s, row) : 0; if (v < c->lo || v > c->hi) return 0; }
    return 1;
}

#include "store_tune.c"

#include "store_live.c"

#include "store_search.c"

#include "store_text.c"

#include "store_label.c"

#if defined(__SSE4_2__) && defined(__x86_64__)
#include <nmmintrin.h>      /* the CRC-32C instruction, for the batches' journal records */
#endif
#include "store_batch.c"

#include "store_sum.c"

#include "store_disk.c"
