/* vecindex.h - the C API of vecindex.
 *
 * One opaque handle per store, plain C types, a status or a count from every call. Nothing else is exported from
 * the library, so any language with a C foreign-function interface can bind it without a compiler: Python ctypes,
 * Go cgo, Rust extern "C", Node koffi or N-API, Java FFM (Panama), C# P/Invoke, Zig @cImport, Lua FFI.
 *
 *   ./run lib                      .build/libvecindex.so and .build/libvecindex.a, then tests/api_test.c against both
 *   cc -Iinclude app.c -L.build -lvecindex -lm -lpthread
 *
 * Vectors are float[dim], any length: compared by cosine unless the store was made with VI_DOT or VI_L2 (below). A
 * search's scores are then the cosine, bigger is better: 1 the same direction, 0 unrelated, -1 the opposite. Ids are
 * yours, any 64-bit value.
 *
 * Errors: a call that fails returns -1 (NULL for the ones that return a handle) and vi_error() says why, in words -
 * "the vector has a NaN or an infinity at index 3", "no row with id 42", "docs.vxi holds vectors of 384 dimensions,
 * not 768". A NULL handle is an error like any other, never a crash.
 *
 * Threads: every function may be called from any thread on the same handle at once. Searches run side by side;
 * writes (add, delete, set_tag, build, save) are serialised against them by one readers-writer lock.
 *
 * Durability: a store opened from a file keeps a journal of every add and delete next to it (PATH.j). vi_sync flushes
 * the journal to the disk; vi_save rewrites the file and starts a fresh journal; vi_snapshot does the same from a
 * child process while this one goes on serving. Opening the file again replays the journal. */
#ifndef VECINDEX_H
#define VECINDEX_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define VI_API __declspec(dllexport)
#elif defined(__GNUC__)
#define VI_API __attribute__((visibility("default")))
#else
#define VI_API
#endif

#if defined(__INT64_TYPE__) && defined(__GNUC__)
__extension__ typedef __INT64_TYPE__ vi_i64;   /* long long on 32-bit targets, fine under -std=c89 -pedantic */
#elif defined(_MSC_VER)
typedef __int64 vi_i64;
#else
#include <stdint.h>
typedef int64_t vi_i64;
#endif

typedef struct vi_db vi_db;

#define VI_VERSION "0.3.0"

/* flags for vi_open. By default rows are indexed as they come and the index is rebuilt as it ages, by a thread of the
 * store's own: there is no vi_build to forget. VI_MANUAL turns that off (then vi_build indexes what was added; until
 * then the new rows are found by an exact scan). VI_LIVE, the default now, is still accepted. */
#define VI_LIVE   1
#define VI_MANUAL 2
/* The metric of a new store (an existing one keeps its own; asking for another is an error). Default: cosine, scores
 * 1 to -1. VI_DOT: the inner product of the vectors as they came, scores bigger is better. VI_L2: the Euclidean
 * distance, scores are the distance, smaller is better - still best first. Rows keep their length beside the direction;
 * the index finds candidates by direction and re-scores four times as many of them exactly in the metric. vi_get
 * returns a dot or L2 store's vectors as they came, a cosine store's normalised. */
#define VI_DOT    4
#define VI_L2     8

/* The library's version, VI_VERSION of the build. */
VI_API const char *vi_version(void);

/* Why the calling thread's last failed call failed, in words ("" before any failure). Each thread has its own. */
VI_API const char *vi_error(void);

/* Opens the store in the file at path, or creates it with dim dimensions when there is no such file (dim 0: it must
 * exist; a dim that differs from the file's is an error). A file that exists but cannot be read is an error, never
 * silently replaced. path NULL: a store in memory only. Returns NULL on failure, and vi_error() says why. */
VI_API vi_db *vi_open(const char *path, int dim, int flags);

/* Flushes the journal to the disk (as vi_sync), waits for a running snapshot, and frees the handle. It does not
 * rewrite the file: the next vi_open replays the journal. */
VI_API void vi_close(vi_db *db);

/* Tenants without a server: a directory with a store per tenant, path/NAME.vxi. vi_dir_get opens a tenant's store
 * on its first use (vi_open: created with dim dimensions and flags when it does not exist yet) and returns the same
 * handle after; NULL for a bad name or a store that cannot be opened. Names are 1 to 64 of A-Z a-z 0-9 _ -, so a
 * name never leaves the directory. Safe from many threads. vi_dir_close closes every store it opened, and their
 * handles with them. vi_dir_open returns NULL when path is not a directory. On a 32-bit CPU each open store reserves
 * 256 MB of address space, so about ten fit at once there. */
typedef struct vi_dir vi_dir;
VI_API vi_dir *vi_dir_open(const char *path, int dim, int flags);
VI_API vi_db *vi_dir_get(vi_dir *d, const char *name);
VI_API void vi_dir_close(vi_dir *d);

/* Shards: every store in the directory (each NAME.vxi, opened as vi_dir_get opens it) searched, and the k best hits of
 * all of them, best first. A store holds at most 2^31 - 1 rows; past that, or past one disk, rows go into several
 * stores - by id range, by time, by source: the caller picks the store a row goes to - and one search covers them all.
 * Ids are the caller's, so they should not repeat across stores. Returns how many hits, or -1: a store that cannot be
 * opened, or one of another dimension or metric than the others (vi_error() names it). */
VI_API int vi_dir_search(vi_dir *d, const float *query, int k, vi_i64 *ids, float *scores);

/* Dimensions; live rows (added and not deleted). */
VI_API int vi_dim(const vi_db *db);
VI_API vi_i64 vi_count(vi_db *db);

/* Adds a row, or replaces the row with this id (one change: a crash leaves the old row or the new one, never
 * neither). 0 on success, -1 when the vector has a NaN or an infinity or is all zeros (it is refused) or memory ran
 * out. */
VI_API int vi_add(vi_db *db, vi_i64 id, const float *vec);

/* vi_add for n rows (vecs: n x dim, row after row); returns how many went in. When some were refused, vi_error()
 * names the first: "row 17 (id 9042): the vector has a NaN ...". */
VI_API int vi_add_batch(vi_db *db, int n, const vi_i64 *ids, const float *vecs);

/* Removes the row with this id; returns 1, or 0 when there is none. */
VI_API int vi_delete(vi_db *db, vi_i64 id);

/* Removes every row whose id is in [lo, hi] - all the chunks of a document when ids are doc << 20 | chunk;
 * returns how many. */
VI_API int vi_delete_range(vi_db *db, vi_i64 lo, vi_i64 hi);

/* The stored vector of id into out[dim] (a cosine store's unit-normalised, a dot or L2 store's as it came): 0, or -1
 * when there is no such row. */
VI_API int vi_get(vi_db *db, vi_i64 id, float *out);

/* vi_add with a text kept beside the row (a chunk of a document, a note: any bytes, len of them, at most 16 MB - 1;
 * len -1: a 0-terminated string). The text comes back with vi_get_text and feeds the words side of
 * vi_search_hybrid. 0, or -1 as vi_add (or a bad length). */
VI_API int vi_add_text(vi_db *db, vi_i64 id, const float *vec, const char *text, int len);

/* The text of id: its length (0: the row has none), the first min(length, cap) bytes copied into buf, no 0 added;
 * -1 when there is no such row. A length above cap means buf was too small: call again with a bigger one. */
VI_API int vi_get_text(vi_db *db, vi_i64 id, char *buf, int cap);

/* The ids in [lo, hi] that have a row, ascending: the smallest max of them into ids; returns how many (-1: out of
 * memory). With ids made as doc << 20 | chunk, vi_range(db, doc << 20, (doc << 20) | 0xfffff, ...) lists a
 * document's chunks: search the small chunks, then hand the model the chunks around a hit, or the whole document.
 * For more than max, call again from the last id + 1. */
VI_API int vi_range(vi_db *db, vi_i64 lo, vi_i64 hi, vi_i64 *ids, int max);

/* A write batch: adds or replacements (with or without a text), deletes and range deletes that searches see all or
 * none of, and that a crash leaves all or none of - say a document's old chunks out and its new ones in, as one
 * change. Vectors and texts are copied as they are added. An add that returns -1 (a NaN or an infinity, a bad
 * length, no memory) makes the commit refuse the whole batch. */
typedef struct vi_batch vi_batch;
VI_API vi_batch *vi_batch_begin(vi_db *db);
VI_API int vi_batch_add(vi_batch *b, vi_i64 id, const float *vec);
VI_API int vi_batch_add_text(vi_batch *b, vi_i64 id, const float *vec, const char *text, int len);
VI_API int vi_batch_delete(vi_batch *b, vi_i64 id);
VI_API int vi_batch_delete_range(vi_batch *b, vi_i64 lo, vi_i64 hi);
VI_API int vi_batch_set_num(vi_batch *b, vi_i64 id, vi_i64 num);   /* on the row the batch added or the store has */

/* Applies the batch as one change and frees it. sync 1: also on the disk when this returns (a store opened from a
 * file; commits arriving during an fsync share the next one). Returns how many changes it held; -1 refused, nothing
 * applied; -2 applied, but the fsync failed. vi_batch_abort frees it unapplied. */
VI_API int vi_batch_commit(vi_batch *b, int sync);
VI_API void vi_batch_abort(vi_batch *b);

/* A 64-bit id for a string - a file name, a URL, a document's key - the same on every machine and in every version
 * (the first 8 bytes of its BLAKE3, the top bit cleared, so never negative; -1 only for a NULL string). len -1:
 * 0-terminated. Two different strings share an id with a chance of about n^2 / 2^64 among n of them: 1 in 18 million
 * for a million keys, about 5% for a billion. To get the string back from a search's ids, keep it as the row's text
 * (vi_add_text). */
VI_API vi_i64 vi_key(const char *s, int len);

/* Labels: names for the 64 bits of a row's tag, in a hierarchy. vi_label makes a label (1 to 63 of A-Z a-z 0-9 _ - .
 * / :) and, with a parent, puts it under that one (made too when new; several parents allowed, cycles refused). A
 * label stands for itself and everything below it, worked out when a search runs: a new hierarchy is a few new links,
 * no row changes. vi_label_add / vi_label_remove put labels on a row or take them off (a list: "red,big"; labels
 * new to the store are made). vi_search_labeled is vi_search over the rows that have every label of `all`, at least
 * one of `any` and none of `none` (each a list, NULL or "": no condition; at most 15 in all). The table is saved in
 * the file and journaled. Labels and vi_set_tag share the same 64 bits: use one or the other. 0, or -1 (vi_error()). */
VI_API int vi_label(vi_db *db, const char *name, const char *parent);
VI_API int vi_label_add(vi_db *db, vi_i64 id, const char *labels);
VI_API int vi_label_remove(vi_db *db, vi_i64 id, const char *labels);
VI_API int vi_search_labeled(vi_db *db, const float *query, int k, const char *all, const char *any, const char *none,
                             vi_i64 *ids, float *scores);

/* A number on a row - a time, a price in cents, a version - for range filters; every row's is 0 until set, and a
 * row that vi_add replaces starts again at 0. vi_set_num / vi_get_num: 0, or -1 when there is no such row.
 * vi_search_range is vi_search over the rows whose number is in [lo, hi]; the range is tested inside the scan, so a
 * narrow one costs no more than a wide one. The numbers are saved in the file, journaled and under the row checks. */
VI_API int vi_set_num(vi_db *db, vi_i64 id, vi_i64 num);
VI_API int vi_get_num(vi_db *db, vi_i64 id, vi_i64 *num);
VI_API int vi_search_range(vi_db *db, const float *query, int k, vi_i64 lo, vi_i64 hi, vi_i64 *ids, float *scores);

/* A 64-bit tag on a row, for filtered search: 0, or -1 when there is no such row. */
VI_API int vi_set_tag(vi_db *db, vi_i64 id, vi_i64 tag);
VI_API int vi_get_tag(vi_db *db, vi_i64 id, vi_i64 *tag);

/* The k nearest rows to query: their ids and cosines into ids[k] and scores[k], best first; returns how many, or -1
 * (k below 1, a NULL buffer, a NaN in the query: vi_error() says which). Rows just added are found too. */
VI_API int vi_search(vi_db *db, const float *query, int k, vi_i64 *ids, float *scores);

/* The same, over rows whose tag has every bit of all set and, when any is not 0, at least one bit of any. */
VI_API int vi_search_tagged(vi_db *db, const float *query, int k, vi_i64 all, vi_i64 any, vi_i64 *ids, float *scores);

/* Hybrid search: the vector query (NULL: words only) and the words of text (tlen bytes, -1: 0-terminated; NULL: the
 * vector only) against the rows' texts (BM25), fused by rank - so an exact name, a number or a rare word is found
 * even where the embedding blurs it. ids and scores get up to k hits; returns how many. The scores rank the hits of
 * this one search (bigger is better) - a fused rank, not a cosine, so compare them only within one result. The first
 * such search builds the word index over every text; later writes keep it current. */
VI_API int vi_search_hybrid(vi_db *db, const float *query, const char *text, int tlen, int k, vi_i64 *ids, float *scores);

/* vi_search with at most one hit a group, the group of an id being id >> shift (unsigned): with ids made as
 * doc << 20 | chunk, shift 20 gives the best chunk of each of the k nearest documents instead of k chunks of one.
 * shift 0: every id its own group, as vi_search. Hits best first, scores as vi_search; returns how many groups were
 * found (fewer than k when the store has fewer, or when the 65536 nearest rows hold fewer), or -1 (as vi_search, or
 * a shift outside 0-63). */
VI_API int vi_search_grouped(vi_db *db, const float *query, int k, int shift, vi_i64 *ids, float *scores);

/* nq queries (queries: nq x dim) on up to threads threads (0: every core); ids and scores (cosines) get k slots a
 * query, a missing hit has id -1 and score -1. 0, or -1 when an argument or a query is wrong. */
VI_API int vi_search_batch(vi_db *db, const float *queries, int nq, int k, vi_i64 *ids, float *scores, int threads);

/* Builds the index over every row now (with VI_MANUAL; by default the store's own thread does it as rows come). */
VI_API void vi_build(vi_db *db);

/* Keeps recall@10 at target (e.g. 0.95): every build measures recall against an exact search and picks the fewest
 * probes that reach it. 0 turns it off. 0, or -1 for a target outside [0, 1). */
VI_API int vi_set_recall(vi_db *db, float target);

/* Memory per row before the first row is added: code_bits 4 (default), 2 or 1 for what a search scans (dim / 2,
 * dim / 4 or dim / 8 bytes a row; fewer bits scan faster and rank more coarsely, so more candidates are re-scored);
 * row_bytes 4 (default, f32 rows, the most precise re-scoring) or 1 (int8 rows, a quarter of the room). Returns 0, or
 * -1 when the store already has rows or a value is not one of these. */
VI_API int vi_set_layout(vi_db *db, int code_bits, int row_bytes);

/* The cascade, for 4-bit codes: 1-bit codes (the signs, derived from the 4-bit ones: dim / 8 more bytes a row) beside
 * them. A search scans the 1-bit codes, re-scores the best 8 x R of them with their 4-bit codes and the best R of
 * those exactly from the rows: every query reads a quarter of the code bytes, the exact re-score as few rows as
 * before. An existing index gets its 1-bit codes at once, builds and rebuilds keep making them, and the file keeps
 * them (a store saved with them opens with the cascade on). on 0 drops them. 0, or -1 (not 4-bit codes, no memory). */
VI_API int vi_set_cascade(vi_db *db, int on);

/* How much of the machine the store may use, 0..1 (default 1, or the environment variable VECINDEX_AGGRESSION):
 * the threads of builds, batches and a big store's single-query scan scale with it, out of the CPUs the process may
 * run on (its affinity mask and cgroup quota); background work (live indexing, rebuilds, snapshots) runs at nice 10
 * (aggression 1) to 19 (0), on the disk's lowest best-effort priority, or only when the disk is idle below 0.5. */
VI_API void vi_set_aggression(vi_db *db, float a);

/* What stays in RAM whatever the memory pressure (mlock). A search scans the hot part every time - the index's codes
 * (code_bits per dimension), its slot table and centroids - and reads the cold part only for the candidates it
 * re-scores and the hits it returns: the rows (f32 or int8), ids, tags, their checks and texts. tier 0 (default): nothing locked,
 * the kernel keeps what searches touch; 1: the hot part locked - low-bit codes in RAM, full-precision rows on the
 * disk; 2: the rows, ids, tags and checks locked too. Returns 0, or -1 when the memlock limit refused part of it (ulimit -l,
 * systemd's LimitMEMLOCK=); that part is then only read ahead. hot and cold (either may be NULL) get the bytes of
 * each part. */
VI_API int vi_set_memory(vi_db *db, int tier);
VI_API void vi_memory(vi_db *db, vi_i64 *hot, vi_i64 *cold);

/* Writes the whole store to path (NULL: the file it was opened from) and starts a fresh journal: 0 or -1. */
VI_API int vi_save(vi_db *db, const char *path);

/* Flushes the journal of the file the store was opened from to the disk (the first call on a store that has no file
 * yet writes it whole): 0 or -1. */
VI_API int vi_sync(vi_db *db);

/* Writes the store to path (NULL: its own file) from a child process, while this one goes on answering searches and
 * taking writes: 0 started, 1 one is already running, -1 failed. vi_snapshot_wait tells how it ended. */
VI_API int vi_snapshot(vi_db *db, const char *path);

/* The last snapshot: 0 done, 1 still running, -1 failed, 2 none yet. block 1 waits for a running one. */
VI_API int vi_snapshot_wait(vi_db *db, int block);

/* Integrity. Every file is written in 64 KiB blocks, each with a BLAKE3 digest, and with Reed-Solomon parity: for
 * every group of `data` blocks, `parity` blocks from which any `parity` damaged blocks of the group are rebuilt (by
 * default 32 and 2; the groups interleave, so a damaged run of neighbouring blocks spreads over many groups). The
 * table of digests and its trailer are kept twice. vi_set_parity sets it for the files this store writes from now
 * on (data 1-128, parity 0-64; 0: digests only): 0, or -1 out of range. */
VI_API int vi_set_parity(vi_db *db, int data, int parity);

/* What opening checks, for every vi_open after the call (process-wide): 2 every block, the default; 1 only the
 * blocks that hold the header; 0 nothing (to rescue a file whose checksums are gone). $VECINDEX_VERIFY sets the
 * default. Damaged blocks are rebuilt from the parity and written back; when they cannot be, a damaged index is
 * built again from the rows and damaged rows make vi_open fail - vi_error says why. */
VI_API void vi_set_verify(int level);
VI_API const char *vi_open_error(void);   /* vi_error, under its older name */

/* Every row also carries a 32-bit check of its vector, id and tag, tested whenever a search re-scores it or vi_get
 * reads it: a row that fails is first rebuilt from the parity of the file it was read from (the page went bad on the
 * disk after the open: the blocks are rebuilt and written back, as ZFS heals a read), else left out of every result -
 * damaged data is never returned. vi_set_check_reads(db, 0) turns the test off. */
VI_API void vi_set_check_reads(vi_db *db, int on);

/* What the checks found. */
typedef struct {
    vi_i64 blocks;              /* the file's blocks, data and parity (0: a file written before checksums) */
    vi_i64 bad, fixed;          /* blocks the open found damaged, and how many of them it rebuilt */
    vi_i64 rows_bad;            /* rows that failed their check when read since, and stayed out of every result */
    vi_i64 rows_healed;         /* rows that failed it and were rebuilt from the parity on the spot */
    int journal_damaged;        /* 1: a damaged journal record stopped the replay (a copy is kept as PATH.j.damaged) */
} vi_health;
VI_API int vi_integrity(vi_db *db, vi_health *h);

/* Checks every block of the store file at path against its digest, the parity blocks and both copies of the table,
 * reading from the disk where the page cache lets it; with repair, rebuilds what it can in place. Safe beside a
 * store that has the file open. Returns the damaged blocks found (0: all intact), -1 cannot read it, -2 a file
 * without checksums, -3 no readable table of digests; *fixed (may be NULL) how many were rebuilt. */
VI_API vi_i64 vi_scrub(const char *path, int repair, vi_i64 *fixed);

#ifdef __cplusplus
}
#endif

#endif /* VECINDEX_H */
