/* graph.h - links between rows, and walks over them: the graph beside a vecindex store.
 *
 * A vg_graph holds directed, typed, weighted links between ids - the ids of a store's rows (chunks, documents,
 * memories), though it does not need the store: it keeps only the ids. Types are 0 to 31 (the caller names them:
 * 0 related_to, 1 references, 2 belongs_to, ...), so a walk takes a mask of them. One link a (from, to, type): a
 * second vg_link of the same three sets its weight.
 *
 *   vg_graph *g = vg_open("docs.vxg");                 a file of its own; NULL path: in memory only
 *   vg_link(g, chunk, doc, 2, 1.0f);                   chunk -belongs_to-> doc
 *   n = vi_search(db, q, 8, ids, sc);                  the store's hits...
 *   m = vg_expand(g, ids, (const float *)0, n, &o, out, osc, odep, 64);   ...and what they link to
 *
 * The file is a log: 32-byte records, each with its CRC-32C, applied in order at open. A torn or damaged tail (a
 * crash in the middle of a write) is cut at the last whole record; nothing before it is lost. Writes reach the disk
 * on vg_sync; vg_compact rewrites the file with only the live links (a temporary file, fsync, rename), the way a
 * store's save does. Memory comes from the OS's pages, no malloc; walks take their working memory per call and
 * never recurse. Thread-safe: walks share the graph, writes take it alone.
 *
 * Built into libvecindex (vecindex.c includes graph.c); errors in words from vg_error(). */
#ifndef VECINDEX_GRAPH_H
#define VECINDEX_GRAPH_H

#include "vecindex.h"

typedef struct vg_graph vg_graph;

#define VG_OUT  1                   /* along links: from -> to */
#define VG_IN   2                   /* against them: to -> from */
#define VG_BOTH 3

/* Opens the graph in path (made when it is not there), or one in memory (path NULL). NULL on failure. */
VI_API vg_graph *vg_open(const char *path);
/* Flushes it to the disk (vg_sync) and frees it. */
VI_API void vg_close(vg_graph *g);
/* Why the calling thread's last failed vg_ call failed, in words. */
VI_API const char *vg_error(void);

/* from -> to of type 0 to 31, weight > 0 and finite (a walk multiplies along a path: 0 to 1 keeps scores falling);
 * an existing link of the same three gets the new weight. A link to itself is refused. 0, -1. */
VI_API int vg_link(vg_graph *g, vi_i64 from, vi_i64 to, int type, float weight);
/* Removes from -> to of that type (-1: of every type). Returns how many went (0: there was none), -1. */
VI_API int vg_unlink(vg_graph *g, vi_i64 from, vi_i64 to, int type);
/* Removes every link from or to id (the row was deleted). Returns how many went, -1. */
VI_API int vg_drop(vg_graph *g, vi_i64 id);

/* The links of id: dir VG_OUT (what it links to), VG_IN (what links to it) or VG_BOTH; types a mask (bit t: type t; 0:
 * all). Up to max of them into ids (the other ends), types and weights (either may be NULL). Returns how many there
 * are in all (more than max: only max written), -1. */
VI_API int vg_links(vg_graph *g, vi_i64 id, int dir, unsigned types, vi_i64 *ids, int *ltypes, float *weights, int max);
/* ids that have a link, and the live links */
VI_API void vg_count(vg_graph *g, vi_i64 *nodes, vi_i64 *links);

/* Every write so far on the disk: 0, -1. A graph in memory: 0. */
VI_API int vg_sync(vg_graph *g);
/* The file rewritten with only the live links, and the memory of removed ones given back: 0, -1 (the old file
 * stays whole). */
VI_API int vg_compact(vg_graph *g);

/* A walk out from seeds, level by level, to max_depth links away: a node's score is the best over the paths found
 * of seed weight x the links' weights x decay a step; a link is taken when its type is in types (0: all), its
 * weight >= min_weight, and it goes the way dir says. It stops at max_nodes nodes, seeds included. */
typedef struct {
    int max_depth;          /* default 2 */
    int max_nodes;          /* default 64 (at most 1 << 20) */
    int dir;                /* default VG_BOTH */
    unsigned types;         /* default 0: every type */
    float min_weight;       /* default 0 */
    float decay;            /* default 0.5 */
} vg_expand_opts;
VI_API void vg_expand_defaults(vg_expand_opts *o);

/* seeds[0, ns) with their weights (seed_w NULL: 1 each; higher is more - turn an L2 or hyperbolic distance into a
 * weight first); o NULL for the defaults. Up to max nodes, best score first, into ids, scores and depths (0 for a seed;
 * scores and depths may be NULL): the seeds themselves come back too, so it is one list. Returns how many, -1. */
VI_API int vg_expand(vg_graph *g, const vi_i64 *seeds, const float *seed_w, int ns, const vg_expand_opts *o,
                     vi_i64 *ids, float *scores, int *depths, int max);

/* Personalized PageRank from the seeds, approximated by local push (Andersen, Chung and Lang): the mass a random
 * walk that goes back to a seed with probability alpha at every step spends on each node, without touching the rest
 * of the graph. A step follows a link with a probability in proportion to its weight, among the links that dir and
 * types allow; a node with none keeps what reaches it. eps: a node pushes only when what waits on it is at least
 * eps x its links (smaller is closer to the true values, and reaches farther). It stops at max_nodes nodes: what would
 * reach more is dropped, so the scores then sum to less than 1. */
typedef struct {
    float alpha;            /* default 0.15 */
    float eps;              /* default 1e-4 */
    int max_nodes;          /* default 4096 (at most 1 << 20) */
    int dir;                /* default VG_BOTH */
    unsigned types;         /* default 0: every type */
} vg_ppr_opts;
VI_API void vg_ppr_defaults(vg_ppr_opts *o);

/* seeds and their weights as vg_expand (normalised to sum 1); up to max nodes, most mass first, into ids and scores.
 * Returns how many, -1. */
VI_API int vg_ppr(vg_graph *g, const vi_i64 *seeds, const float *seed_w, int ns, const vg_ppr_opts *o,
                  vi_i64 *ids, float *scores, int max);

#endif
