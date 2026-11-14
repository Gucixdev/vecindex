/* vecdb.c - a note database on top of vecindex.
 *
 *   vector search   store.c (quantised IVF/HNSW index, doc ids, save/load)
 *   hash dictionaries  hdict.c: title -> node, tag -> node; lex.c: term ->
 *                   postings, so BM25 finds exact words the embedding blurs
 *   node graph      notes, tags and "ghosts" (links to notes that do not
 *                   exist yet) joined by wikilink and hashtag edges; the
 *                   reverse edges are the backlinks
 *   wikilinks       [[Title]], [[Title|alias]], [[Title#heading]], ![[Title]]
 *   hashtags        #tag, #project/alpha (also registers #project), tags:
 *                   in a --- front matter block; never inside code, never
 *                   in a URL, never a "# heading" or a bare number
 *   MCTS            vecdb_explore: Monte Carlo tree search over the graph.
 *                   Seeds are the hybrid (vector + BM25) hits; the tree
 *                   expands along links, backlinks and tags, a random
 *                   rollout scores what lies a few hops beyond, UCB1 decides
 *                   where to spend the next iteration. It surfaces notes
 *                   that are weak matches themselves but sit between strong
 *                   ones - what a plain nearest-neighbour list cannot.
 *
 * Vectors: pass your own (vecdb_add_note_vec) or let vecdb_embed hash the
 * words into `dim` buckets (signed feature hashing + bigrams, no model, no
 * network, deterministic).
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -Werror -D_POSIX_C_SOURCE=200112L \
 *       -O2 -march=native -Isrc tests/vecdb_test.c -lm -lpthread -o vecdb_t && ./vecdb_t
 *   ... src/vecdb_cli.c is the command line tool
 */
#include "store.c"
#include "lex.c"

#include <dirent.h>
#include <sys/wait.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/* small containers                                                    */
/* ------------------------------------------------------------------ */

typedef struct { i32 *v; i32 n, cap; } VIVec;

/* Small allocations of the graph (link lists, node names) come from big chunks instead of malloc: no per-block
 * header, no rounding to 32 bytes. Link lists use power-of-two classes with a free list each (a list that grows
 * hands its old block to the next one); lists past VSLAB_MAXC ints and the chunks themselves are malloc'd. */
#define VSLAB_MAXC 1024
#define VSLAB_CHUNK ((size_t)1 << 20)
typedef struct { i32 *free_[11]; char **chunk; i32 nchunk, ccap; char *cur; size_t left; } VSlab;

static void *vslab_raw_(VSlab *s, size_t bytes)
{
    void *p;
    bytes = (bytes + 7u) & ~(size_t)7u;
    if (bytes > s->left) {
        size_t sz = bytes > VSLAB_CHUNK ? bytes : VSLAB_CHUNK;
        if (s->nchunk == s->ccap) { s->ccap = s->ccap ? s->ccap * 2 : 64; s->chunk = (char **)realloc(s->chunk, (size_t)s->ccap * sizeof(char *)); }
        s->chunk[s->nchunk++] = s->cur = (char *)malloc(sz);
        s->left = sz;
    }
    p = s->cur; s->cur += bytes; s->left -= bytes;
    return p;
}

static i32 vslab_cls_(i32 cap) { i32 c = 0; while ((2 << c) < cap) ++c; return c; }   /* cap = 2 << class */

static i32 *vslab_get_(VSlab *s, i32 cap)
{
    i32 c = vslab_cls_(cap), *p = s->free_[c];
    if (p) { memcpy(&s->free_[c], p, sizeof(i32 *)); return p; }
    return (i32 *)vslab_raw_(s, (size_t)cap * sizeof(i32));
}

static void vslab_put_(VSlab *s, i32 *p, i32 cap)
{
    i32 c = vslab_cls_(cap);
    memcpy(p, &s->free_[c], sizeof(i32 *));
    s->free_[c] = p;
}

static char *vslab_str_(VSlab *s, const char *t, i32 n)
{
    char *p = (char *)vslab_raw_(s, (size_t)n + 1);
    memcpy(p, t, (size_t)n); p[n] = 0;
    return p;
}

static void vslab_free_(VSlab *s)
{
    i32 i;
    for (i = 0; i < s->nchunk; ++i) free(s->chunk[i]);
    free(s->chunk);
    memset(s, 0, sizeof(*s));
}

static void vivec_add_(VSlab *s, VIVec *a, i32 x)
{
    if (a->n == a->cap) {
        i32 nc = a->cap ? a->cap * 2 : 2;
        if (nc > VSLAB_MAXC) {
            if (a->cap > VSLAB_MAXC) a->v = (i32 *)realloc(a->v, (size_t)nc * sizeof(i32));
            else { i32 *nv = (i32 *)malloc((size_t)nc * sizeof(i32)); memcpy(nv, a->v, (size_t)a->n * sizeof(i32)); vslab_put_(s, a->v, a->cap); a->v = nv; }
        } else {
            i32 *nv = vslab_get_(s, nc);
            if (a->cap) { memcpy(nv, a->v, (size_t)a->n * sizeof(i32)); vslab_put_(s, a->v, a->cap); }
            a->v = nv;
        }
        a->cap = nc;
    }
    a->v[a->n++] = x;
}

static i32 vivec_has_(const VIVec *a, i32 x)
{
    i32 i;
    for (i = 0; i < a->n; ++i) if (a->v[i] == x) return 1;
    return 0;
}

static void vivec_remove_(VIVec *a, i32 x)
{
    i32 i;
    for (i = 0; i < a->n; ++i)
        if (a->v[i] == x) { a->v[i] = a->v[--a->n]; return; }
}

/* a list of short strings in one buffer */
typedef struct { char *buf; i32 len, cap; i32 *off, *n; i32 count, ocap; } VStrs;

static void vstrs_free_(VStrs *s) { free(s->buf); free(s->off); free(s->n); memset(s, 0, sizeof(*s)); }

static i32 vstrs_add_(VStrs *s, const char *p, i32 n)
{
    i32 i;
    for (i = 0; i < s->count; ++i)                       /* keep the list distinct */
        if (s->n[i] == n && memcmp(s->buf + s->off[i], p, (size_t)n) == 0) return i;
    if (s->count == s->ocap) {
        s->ocap = s->ocap ? s->ocap * 2 : 16;
        s->off = (i32 *)realloc(s->off, (size_t)s->ocap * sizeof(i32));
        s->n   = (i32 *)realloc(s->n,   (size_t)s->ocap * sizeof(i32));
    }
    while (s->len + n + 1 > s->cap) {
        s->cap = s->cap ? s->cap * 2 : 1024;
        s->buf = (char *)realloc(s->buf, (size_t)s->cap);
    }
    memcpy(s->buf + s->len, p, (size_t)n);
    s->buf[s->len + n] = 0;
    s->off[s->count] = s->len; s->n[s->count] = n;
    s->len += n + 1;
    return s->count++;
}

/* ------------------------------------------------------------------ */
/* names                                                               */
/* ------------------------------------------------------------------ */

/* Normalise a title / tag into out (>= n+1 bytes): ASCII lowercase, runs of
 * blanks collapsed, trimmed, a trailing ".md" and any directory part of a
 * title dropped when is_title. Returns the length. */
static i32 vdb_norm_(const char *s, i32 n, char *out, i32 is_title)
{
    i32 i, w = 0, start = 0, sp = 0;
    if (is_title) {
        for (i = 0; i < n; ++i) if (s[i] == '/' || s[i] == '\\') start = i + 1;
        if (n - start > 3 && s[n - 3] == '.' && (s[n - 2] | 32) == 'm' && (s[n - 1] | 32) == 'd') n -= 3;
    }
    for (i = start; i < n; ++i) {
        u8 c = (u8)s[i];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') { sp = w > 0; continue; }
        if (sp) { out[w++] = ' '; sp = 0; }
        out[w++] = (c >= 'A' && c <= 'Z') ? (char)(c + 32) : (char)c;
    }
    out[w] = 0;
    return w;
}

/* ------------------------------------------------------------------ */
/* parsing: wikilinks and hashtags                                     */
/* ------------------------------------------------------------------ */

static i32 vdb_tagchar_(u8 c)
{
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c == '-' || c == '/' || c >= 0x80;
}

/* register a tag and all its ancestors: a/b/c -> a, a/b, a/b/c */
static void vdb_add_tag_(VStrs *tags, const char *t, i32 n)
{
    char *norm = (char *)malloc((size_t)n + 1);
    i32 m = vdb_norm_(t, n, norm, 0), i;
    for (i = 0; i < m; ++i)
        if (norm[i] == '/' && i > 0) vstrs_add_(tags, norm, i);
    if (m > 0) vstrs_add_(tags, norm, m);
    free(norm);
}

/* the front matter tags: line, "tags: a, b", "tags: [a, b]" or a "- a" list below it */
static void vdb_frontmatter_(const char *t, i32 len, VStrs *tags, i32 *body_off)
{
    const char *p = t, *end = t + len;
    i32 in_tags = 0;
    *body_off = 0;
    if (len < 4 || memcmp(t, "---", 3) != 0 || (t[3] != '\n' && t[3] != '\r')) return;
    p = t + 3;
    while (p < end && (*p == '\n' || *p == '\r')) ++p;
    while (p < end) {
        const char *e = p;
        while (e < end && *e != '\n') ++e;
        if (e - p >= 3 && memcmp(p, "---", 3) == 0) { *body_off = (i32)((e < end ? e + 1 : e) - t); return; }
        if (e - p >= 5 && memcmp(p, "tags:", 5) == 0) {
            const char *q = p + 5;
            in_tags = 1;
            while (q < e) {
                const char *s;
                while (q < e && (*q == ' ' || *q == ',' || *q == '[' || *q == ']' || *q == '\r' || *q == '"' || *q == '\'')) ++q;
                s = q;
                while (q < e && vdb_tagchar_((u8)*q)) ++q;
                if (q > s) { const char *c = s; if (*c == '#') ++c; vdb_add_tag_(tags, c, (i32)(q - c)); }
                else if (q < e) ++q;
            }
        } else if (in_tags && p < e && (*p == '-' || *p == ' ')) {
            const char *q = p;
            while (q < e && (*q == ' ' || *q == '-')) ++q;
            if (q < e && p[0] == '-' ) {
                const char *s = q;
                while (q < e && vdb_tagchar_((u8)*q)) ++q;
                if (q > s) vdb_add_tag_(tags, s, (i32)(q - s));
            }
        } else if (p < e && *p != ' ' && *p != '-') {
            in_tags = 0;
        }
        p = e < end ? e + 1 : e;
    }
}

/* Find the [[wikilinks]] and #tags of a note. links get the normalised link target
 * (no alias, no #heading, no folder, no .md); tags come with their ancestors. */
static void vdb_parse_(const char *text, i32 len, VStrs *links, VStrs *tags)
{
    const char *p, *end = text + len;
    i32 body, infence = 0;
    char fence = 0;

    vdb_frontmatter_(text, len, tags, &body);
    p = text + body;
    while (p < end) {
        const char *ls = p, *le = p, *c;
        i32 lead = 0;
        while (le < end && *le != '\n') ++le;
        c = ls;
        while (c < le && lead < 4 && *c == ' ') { ++c; ++lead; }
        if (le - c >= 3 && ((c[0] == '`' && c[1] == '`' && c[2] == '`') || (c[0] == '~' && c[1] == '~' && c[2] == '~'))) {
            if (!infence) { infence = 1; fence = c[0]; }
            else if (c[0] == fence) infence = 0;
            p = le < end ? le + 1 : le;
            continue;
        }
        if (!infence) {
            const char *q = ls;
            while (q < le) {
                if (*q == '`') {                                         /* inline code */
                    const char *e = q + 1;
                    while (e < le && *e != '`') ++e;
                    if (e < le) { q = e + 1; continue; }
                }
                if (*q == ':' && q + 2 < le && q[1] == '/' && q[2] == '/') {   /* a URL: skip it whole */
                    while (q < le && *q != ' ' && *q != '\t' && *q != ')' && *q != '>') ++q;
                    continue;
                }
                if (q[0] == '[' && q + 1 < le && q[1] == '[') {
                    const char *e = q + 2;
                    while (e + 1 < le && !(e[0] == ']' && e[1] == ']')) ++e;
                    if (e + 1 < le) {
                        const char *t0 = q + 2, *t1 = t0;
                        while (t1 < e && *t1 != '|' && *t1 != '#') ++t1;
                        if (t1 > t0) {
                            char *norm = (char *)malloc((size_t)(t1 - t0) + 1);
                            i32 m = vdb_norm_(t0, (i32)(t1 - t0), norm, 1);
                            if (m > 0) vstrs_add_(links, norm, m);
                            free(norm);
                        }
                        q = e + 2;
                        continue;
                    }
                }
                if (q[0] == '!' && q + 1 < le && q[1] == '[') {          /* ![alt](picture.png): a link to that file's note */
                    const char *e = q + 2;
                    while (e < le && *e != ']') ++e;
                    if (e + 1 < le && e[1] == '(') {
                        const char *t0 = e + 2, *t1 = t0;
                        while (t1 < le && *t1 != ')' && *t1 != ' ') ++t1;
                        if (t1 < le && t1 > t0) {
                            const char *sl = t0, *z;
                            i32 web = 0;
                            for (z = t0; z + 2 < t1; ++z) if (z[0] == ':' && z[1] == '/' && z[2] == '/') web = 1;
                            for (z = t0; z < t1; ++z) if (*z == '/') sl = z + 1;
                            if (!web && t1 > sl) {
                                char *norm = (char *)malloc((size_t)(t1 - sl) + 1);
                                i32 m = vdb_norm_(sl, (i32)(t1 - sl), norm, 1);
                                if (m > 0) vstrs_add_(links, norm, m);
                                free(norm);
                            }
                            q = t1; continue;
                        }
                    }
                }
                if (*q == '#') {
                    i32 okprev = (q == ls) || q[-1] == ' ' || q[-1] == '\t' || q[-1] == '(' || q[-1] == '[' || q[-1] == '{' ||
                                 q[-1] == ',' || q[-1] == ';' || q[-1] == '"' || q[-1] == '\'' || q[-1] == '>';
                    const char *t0 = q + 1, *t1 = t0;
                    i32 hasalpha = 0;
                    while (t1 < le && vdb_tagchar_((u8)*t1)) { if (!(*t1 >= '0' && *t1 <= '9')) hasalpha = 1; ++t1; }
                    while (t1 > t0 && (t1[-1] == '/' || t1[-1] == '-')) --t1;
                    if (okprev && t1 > t0 && hasalpha) { vdb_add_tag_(tags, t0, (i32)(t1 - t0)); q = t1; continue; }
                }
                ++q;
            }
        }
        p = le < end ? le + 1 : le;
    }
}

/* ------------------------------------------------------------------ */
/* embedding: signed feature hashing + bigrams                         */
/* ------------------------------------------------------------------ */

static u64 vdb_hash_(const char *s, i32 n, u64 seed)
{
    u64 h = seed ^ HD_C64(0xcbf29ce4, 0x84222325);
    i32 i;
    for (i = 0; i < n; ++i) { h ^= (u8)s[i]; h *= HD_C64(0x00000100, 0x000001b3); }
    h ^= h >> 29; h *= HD_C64(0xbf58476d, 0x1ce4e5b9); h ^= h >> 32;
    return h;
}

/* deterministic text embedding into out[dim] (unit length; zero text -> all zero) */
void vecdb_embed(const char *text, i32 len, f32 *out, i32 dim)
{
    const char *p = text, *end = text + len;
    char tok[LEX_MAXTOK + 1], prev[LEX_MAXTOK + 1];
    i32 n, pn = 0, d;
    f64 nrm = 0.0;
    f32 *acc = out;

    for (d = 0; d < dim; ++d) acc[d] = 0.0f;
    while ((n = lex_next_(&p, end, tok)) > 0) {
        u64 h = vdb_hash_(tok, n, 1);
        acc[h % (u64)dim] += ((h >> 40) & 1) ? -1.0f : 1.0f;
        if (pn > 0) {                                         /* the bigram: a phrase, half weight */
            u64 g = vdb_hash_(prev, pn, 2) * 31u + vdb_hash_(tok, n, 3);
            g ^= g >> 29;
            acc[g % (u64)dim] += ((g >> 41) & 1) ? -0.5f : 0.5f;
        }
        memcpy(prev, tok, (size_t)n); pn = n;
    }
    for (d = 0; d < dim; ++d) {                               /* sublinear: a word repeated 9x counts 3x */
        f32 v = acc[d];
        acc[d] = v < 0 ? -(f32)sqrt((f64)-v) : (f32)sqrt((f64)v);
        nrm += (f64)acc[d] * (f64)acc[d];
    }
    if (nrm > 0.0) { f32 inv = (f32)(1.0 / sqrt(nrm)); for (d = 0; d < dim; ++d) acc[d] *= inv; }
}

/* ------------------------------------------------------------------ */
/* the database                                                        */
/* ------------------------------------------------------------------ */

enum { VN_NOTE = 0, VN_TAG = 1, VN_GHOST = 2, VN_DEAD = 3 };

typedef struct {
    char *name;         /* note title / tag as written the first time (in the db's slab) */
    i64   toff;         /* note body: offset and length in the text file (not held in RAM) */
    VIVec out;          /* wikilinks (note -> note/ghost) and tag edges (note -> tag) */
    VIVec in;           /* backlinks; for a tag: the notes carrying it */
    i32   tlen;         /* -1: no body */
    signed char tagbit; /* tags: bit in the store's i64 meta (0..61), -1 = none */
    u8    space;        /* 0 text vectors, 1 image vectors */
    u8    kind;
} VNode;                /* 56 bytes */

typedef struct {
    RagStore *store;
    i32   dim;
    HStr  names;        /* normalised title -> node (notes and ghosts) */
    HStr  tags;         /* normalised tag -> node */
    VNode *nodes;
    i32   nn, ncap;                                              /* NOTE: node ids are NOT a dense 0..count-1 range of
                                                                  * notes - tags and ghosts (see below) share the same
                                                                  * id space, so a note's id can jump ahead of its
                                                                  * insertion order. Never assume id == the Nth note
                                                                  * added; always use the id vecdb_add_note* returns
                                                                  * (or vecdb_find). */
    Lex   lex;
    i32   ntagbits;
    u32   rng;
    Arena *arena;
    RagStore *store2;   /* image vectors (a different model, so a different space); NULL until vecdb_enable_images */
    i32   dim2;
    void *proc2;        /* the image model's server: images -> vectors, text queries -> the same space */
    FILE  *tf;          /* note bodies, append-only; an anonymous temp file until saved */
    char  *tpath;
    i64    tend;
    /* text -> unit vector; default vecdb_embed. Set it to your model before adding notes
     * AND before querying (the saved files store vectors, not the embedder).
     * kind 0 = a document being indexed, 1 = a search query (asymmetric models need the difference) */
    void (*embed)(void *ctx, const char *text, i32 len, i32 kind, f32 *out, i32 dim);
    void *embed_ctx;
    /* optional: many documents at once (kind as above); out is n x dim. Without it, texts go one by one */
    void (*embed_batch)(void *ctx, i32 n, const char **texts, const i32 *lens, i32 kind, f32 *out, i32 dim);
    /* the last few query vectors, so a repeated query does not go through the model again */
    char  *qc_text[32]; i32 qc_len[32]; f32 *qc_vec[32]; i32 qc_next;
    pthread_mutex_t qc_mu;
    pthread_rwlock_t lk;  /* searches share it; add/remove/build/save take it alone */
    pthread_mutex_t tf_mu, ec_mu;   /* the body file position; the embedding cache file */
    char  *dbpath;        /* the outer path (as passed to vecdb_save/sync) this instance is attached to */
    u64    bodygen;       /* 0: PATH.vdt/.vxi (bare names). Nonzero: PATH.vdt.g<hex>/.vxi.g<hex> - used only when a
                          * full rewrite would otherwise have to replace an already-existing file of that name
                          * (re-indexing an existing path from scratch); see vdbl_save. */
    struct VECache *ec;   /* content-hash cache of document vectors (vecdb_embed_cache_open) */
    VSlab  slab;          /* node names and link lists */
} VecDB;

typedef struct {
    i32 node;
    f32 sim;            /* cosine to the query (0 if unknown) */
    f32 lex;            /* BM25, scaled 0..1 against the best hit of the query */
    f32 score;          /* what the list is sorted by */
    i32 via;            /* explore: the node this one was reached from, or -1 */
    i32 depth;          /* explore: hops from a seed */
} VHit;

typedef struct {
    i32 seeds;          /* hybrid hits the search starts from (default 6) */
    i32 iters;          /* MCTS iterations (default 600) */
    i32 maxdepth;       /* longest path of the tree (default 4) */
    i32 rollout;        /* random steps after a new node (default 3) */
    f32 c;              /* UCB1 exploration constant (default 1.0) */
    f32 gamma;          /* discount per rollout step (default 0.8) */
    u32 seed;           /* rng seed (default 1) */
} VExplore;

VecDB *vecdb_new(Arena *a, i32 dim)
{
    VecDB *db = (VecDB *)calloc(1, sizeof(VecDB));
    db->store = store_new(a, dim);
    store_set_fast(db->store, 0.97f);
    db->dim = dim;
    hs_init(&db->names, 256);
    hs_init(&db->tags, 64);
    lex_init(&db->lex);
    db->rng = 1u;
    db->arena = a;
    db->tf = tmpfile();
    pthread_mutex_init(&db->qc_mu, (const pthread_mutexattr_t *)0);
    pthread_mutex_init(&db->tf_mu, (const pthread_mutexattr_t *)0);
    pthread_mutex_init(&db->ec_mu, (const pthread_mutexattr_t *)0);
    { pthread_rwlockattr_t at;                                   /* writers are not starved by a stream of searches */
      pthread_rwlockattr_init(&at);
#ifdef __GLIBC__
      pthread_rwlockattr_setkind_np(&at, PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);
#endif
      pthread_rwlock_init(&db->lk, &at);
      pthread_rwlockattr_destroy(&at); }
    return db;
}

/* ---- embedding cache: hash(model tag + text) -> vector, in a file next to the database ---- */
typedef struct VECache { FILE *f; H64 map; i32 n, dim; u64 seed; pthread_mutex_t mu; } VECache;

static u64 vdb_hash64_(const char *s, i32 n, u64 h)
{
    i32 i;
    for (i = 0; i < n; ++i) { h ^= (u8)s[i]; h *= HD_C64(0x00000100, 0x000001b3); }
    h ^= h >> 29; h *= HD_C64(0xbf58476d, 0x1ce4e5b9); h ^= h >> 32;
    return h;
}

/* Keep every document vector the model produced in PATH, keyed by the text and `tag` (name the model and its
 * settings in it: another tag = another model = separate entries). Re-indexing an unchanged vault then never
 * touches the model. Call before adding notes. Returns 0 on success. */
i32 vecdb_embed_cache_open(VecDB *db, const char *path, const char *tag)
{
    VECache *c = (VECache *)calloc(1, sizeof(*c));
    u64 seed = vdb_hash64_(tag, (i32)strlen(tag), HD_C64(0xcbf29ce4, 0x84222325));
    u32 magic = 0x31434556u;                                     /* 'VEC1' */
    i32 dim = 0;
    u64 sd = 0;
    size_t rec = 8 + (size_t)db->dim * sizeof(f32);
    c->dim = db->dim; c->seed = seed;
    pthread_mutex_init(&c->mu, (const pthread_mutexattr_t *)0);
    h64_init(&c->map, 1024);
    c->f = fopen(path, "r+b");
    if (c->f && fread(&magic, 4, 1, c->f) == 1 && fread(&dim, 4, 1, c->f) == 1 && fread(&sd, 8, 1, c->f) == 1 && magic == 0x31434556u && dim == db->dim && sd == seed) {
        char *buf = (char *)malloc(rec);
        for (;;) {                                               /* index the records; a torn last one is dropped */
            u64 h;
            if (fread(buf, 1, rec, c->f) != rec) break;
            memcpy(&h, buf, 8);
            h64_put(&c->map, (i64)h, c->n++);
        }
        free(buf);
        fseeko(c->f, (off_t)(16 + (size_t)c->n * rec), SEEK_SET);
    } else {                                                     /* new file, or another model: start empty */
        if (c->f) fclose(c->f);
        c->f = fopen(path, "w+b");
        if (!c->f) { h64_free(&c->map); free(c); return -1; }
        magic = 0x31434556u;
        fwrite(&magic, 4, 1, c->f); fwrite(&db->dim, 4, 1, c->f); fwrite(&seed, 8, 1, c->f);
        fflush(c->f);
    }
    db->ec = c;
    return 0;
}

static i32 vdb_ec_get_(VECache *c, u64 h, f32 *out)
{
    i32 row;
    size_t rec = 8 + (size_t)c->dim * sizeof(f32);
    i32 ok;
    pthread_mutex_lock(&c->mu);
    row = h64_get(&c->map, (i64)h);
    if (row < 0) { pthread_mutex_unlock(&c->mu); return 0; }
    fseeko(c->f, (off_t)(16 + (size_t)row * rec + 8), SEEK_SET);
    ok = fread(out, sizeof(f32), (size_t)c->dim, c->f) == (size_t)c->dim;
    pthread_mutex_unlock(&c->mu);
    return ok;
}

static i32 vdb_zero_(const f32 *v, i32 n);

static void vdb_ec_put_(VECache *c, u64 h, const f32 *vec)
{
    if (vdb_zero_(vec, c->dim)) return;
    pthread_mutex_lock(&c->mu);
    if (h64_get(&c->map, (i64)h) < 0) {
        fseeko(c->f, 0, SEEK_END);
        fwrite(&h, 8, 1, c->f); fwrite(vec, sizeof(f32), (size_t)c->dim, c->f);
        h64_put(&c->map, (i64)h, c->n++);
    }
    pthread_mutex_unlock(&c->mu);
}

i32 vecdb_embed_cache_count(const VecDB *db) { return db->ec ? db->ec->n : 0; }

/* a failed embedding is an all-zero vector; nothing built on it is stored or cached */
static i32 vdb_zero_(const f32 *v, i32 n)
{
    i32 i;
    for (i = 0; i < n; ++i) if (v[i] != 0.0f) return 0;
    return 1;
}

static void vdb_embed_(const VecDB *db, const char *t, i32 n, i32 kind, f32 *out)
{
    VecDB *m = (VecDB *)db;
    i32 i, hit = -1;
    if (kind == 1 && db->embed) {                                  /* query cache: only worth it for a slow model */
        pthread_mutex_lock(&m->qc_mu);
        for (i = 0; i < 32; ++i) if (m->qc_text[i] && m->qc_len[i] == n && memcmp(m->qc_text[i], t, (size_t)n) == 0) { hit = i; break; }
        if (hit >= 0) memcpy(out, m->qc_vec[hit], (size_t)db->dim * sizeof(f32));
        pthread_mutex_unlock(&m->qc_mu);
        if (hit >= 0) return;
    }
    if (db->embed) db->embed(db->embed_ctx, t, n, kind, out, db->dim); else vecdb_embed(t, n, out, db->dim);
    if (kind == 1 && db->embed) {
        pthread_mutex_lock(&m->qc_mu);
        i = m->qc_next; m->qc_next = (m->qc_next + 1) % 32;
        free(m->qc_text[i]); free(m->qc_vec[i]);
        m->qc_text[i] = (char *)malloc((size_t)n + 1); memcpy(m->qc_text[i], t, (size_t)n); m->qc_len[i] = n;
        m->qc_vec[i] = (f32 *)malloc((size_t)db->dim * sizeof(f32)); memcpy(m->qc_vec[i], out, (size_t)db->dim * sizeof(f32));
        pthread_mutex_unlock(&m->qc_mu);
    }
}

/* Release everything the database owns (its stores, files, dictionaries, caches). Model servers you opened
 * yourself (vecdb_embpool_open) stay yours to close. */
void vecdb_free(VecDB *db)
{
    i32 i;
    if (!db) return;
    for (i = 0; i < db->nn; ++i) {                              /* only lists past the slab classes were malloc'd */
        if (db->nodes[i].out.cap > VSLAB_MAXC) free(db->nodes[i].out.v);
        if (db->nodes[i].in.cap > VSLAB_MAXC) free(db->nodes[i].in.v);
    }
    free(db->nodes);
    vslab_free_(&db->slab);
    hs_free(&db->names); hs_free(&db->tags);
    lex_free(&db->lex);
    if (db->store) store_destroy(db->store);
    if (db->store2) store_destroy(db->store2);
    if (db->tf) fclose(db->tf);
    free(db->tpath); free(db->dbpath);
    if (db->ec) { if (db->ec->f) fclose(db->ec->f); h64_free(&db->ec->map); free(db->ec); }
    for (i = 0; i < 32; ++i) { free(db->qc_text[i]); free(db->qc_vec[i]); }
    pthread_mutex_destroy(&db->qc_mu); pthread_mutex_destroy(&db->tf_mu); pthread_mutex_destroy(&db->ec_mu);
    pthread_rwlock_destroy(&db->lk);
    free(db);
}

static i32 vdb_newnode_(VecDB *db, u8 kind, const char *name, i32 n)
{
    VNode *v;
    if (db->nn == db->ncap) {
        db->ncap = db->ncap ? db->ncap + db->ncap / 2 : 256;     /* 1.5x: less slack at millions of nodes */
        db->nodes = (VNode *)realloc(db->nodes, (size_t)db->ncap * sizeof(VNode));
    }
    v = &db->nodes[db->nn];
    memset(v, 0, sizeof(*v));
    v->kind = kind; v->name = vslab_str_(&db->slab, name, n); v->tagbit = -1; v->tlen = -1;
    return db->nn++;
}

/* node of a normalised title; created as a ghost when it does not exist */
static i32 vdb_title_node_(VecDB *db, const char *key, i32 n, i32 create)
{
    i32 id = hs_get(&db->names, key, n);
    if (id < 0 && create) {
        id = vdb_newnode_(db, VN_GHOST, key, n);
        hs_put(&db->names, key, n, id);
    }
    return id;
}

static i32 vdb_tag_node_(VecDB *db, const char *key, i32 n)
{
    i32 id = hs_get(&db->tags, key, n);
    if (id < 0) {
        id = vdb_newnode_(db, VN_TAG, key, n);
        if (db->ntagbits < 62) db->nodes[id].tagbit = (signed char)db->ntagbits++;
        hs_put(&db->tags, key, n, id);
    }
    return id;
}

static void vdb_edge_(VecDB *db, i32 from, i32 to)
{
    if (from == to || vivec_has_(&db->nodes[from].out, to)) return;
    vivec_add_(&db->slab, &db->nodes[from].out, to);
    vivec_add_(&db->slab, &db->nodes[to].in, from);
}

/* remove every edge that starts at id (its links and tags) */
static void vdb_unlink_(VecDB *db, i32 id)
{
    VNode *v = &db->nodes[id];
    i32 i;
    for (i = 0; i < v->out.n; ++i) vivec_remove_(&db->nodes[v->out.v[i]].in, id);
    v->out.n = 0;
}

/* append a body to the text file; the old bytes of a replaced note are dropped at the next save */
static void vdb_text_put_(VecDB *db, VNode *v, const char *t, i32 n)
{
    if (!db->tf) { v->tlen = -1; return; }
    pthread_mutex_lock(&db->tf_mu);
    fseeko(db->tf, 0, SEEK_END);
    v->toff = db->tend;
    if (n > 0 && fwrite(t, 1, (size_t)n, db->tf) != (size_t)n) { v->tlen = -1; pthread_mutex_unlock(&db->tf_mu); return; }
    v->tlen = n; db->tend += n;
    pthread_mutex_unlock(&db->tf_mu);
}

/* the body of a note as a new NUL-terminated string (free it), NULL if it has none */
static char *vdbl_text(const VecDB *db, i32 id)
{
    const VNode *v;
    char *b;
    if (id < 0 || id >= db->nn || !db->tf) return (char *)0;
    v = &db->nodes[id];
    if (v->tlen < 0) return (char *)0;
    b = (char *)malloc((size_t)v->tlen + 1);
    pthread_mutex_lock(&((VecDB *)db)->tf_mu);
    fseeko(db->tf, (off_t)v->toff, SEEK_SET);
    if (v->tlen > 0 && fread(b, 1, (size_t)v->tlen, db->tf) != (size_t)v->tlen) { pthread_mutex_unlock(&((VecDB *)db)->tf_mu); free(b); return (char *)0; }
    pthread_mutex_unlock(&((VecDB *)db)->tf_mu);
    b[v->tlen] = 0;
    return b;
}

/* the note's unit vector, borrowed from the store (one copy of the data, not two) */
static const f32 *vdb_vec_(const VecDB *db, i32 id) { return db->nodes[id].space ? (const f32 *)0 : store_vec_ptr(db->store, (i64)id); }

static i32 vdb_add_(VecDB *db, const char *title, const char *text, const f32 *vec, i32 to_store, i32 persist_text, i32 space);
static void vdb_embed_doc_(const VecDB *db, const char *title, const char *text, f32 *vec);

i32 vecdb_add_note_vec(VecDB *db, const char *title, const char *text, const f32 *vec)
{
    i32 id;
    pthread_rwlock_wrlock(&db->lk);
    id = vdb_add_(db, title, text, vec, 1, 1, 0);
    pthread_rwlock_unlock(&db->lk);
    return id;
}

/* Add (or replace) a note with your own vector. Returns the node id. */
static i32 vdb_add_(VecDB *db, const char *title, const char *text, const f32 *vec, i32 to_store, i32 persist_text, i32 space)
{
    i32 dm = space ? db->dim2 : db->dim;
    char *key = (char *)malloc(strlen(title) + 1);
    i32 klen = vdb_norm_(title, (i32)strlen(title), key, 1), id, i, tlen = (i32)strlen(text);
    VStrs links, tags;
    i64 meta = 0;
    f32 nrm = 0.0f, *nv;
    VNode *v;

    if (klen == 0) { free(key); return -1; }
    id = hs_get(&db->names, key, klen);
    if (id >= 0 && db->nodes[id].kind == VN_NOTE) {              /* replace */
        vdb_unlink_(db, id);
        store_delete(db->nodes[id].space ? db->store2 : db->store, (i64)id);
        lex_delete(&db->lex, (i64)id);
    } else if (id < 0) {
        id = vdb_newnode_(db, VN_NOTE, title, (i32)strlen(title));
        hs_put(&db->names, key, klen, id);
    }                                                            /* else: a ghost becomes a note, keeping its backlinks */
    v = &db->nodes[id];
    if (v->kind != VN_NOTE) v->name = vslab_str_(&db->slab, title, (i32)strlen(title));   /* the ghost's spelling stays in the slab */
    v->kind = VN_NOTE; v->space = (u8)space;
    if (persist_text) vdb_text_put_(db, v, text, tlen);
    nv = (f32 *)malloc((size_t)dm * sizeof(f32));
    for (i = 0; i < dm; ++i) { nv[i] = vec ? vec[i] : 0.0f; nrm += nv[i] * nv[i]; }
    if (nrm > 0.0f) { nrm = 1.0f / (f32)sqrt((f64)nrm); for (i = 0; i < dm; ++i) nv[i] *= nrm; }

    memset(&links, 0, sizeof(links)); memset(&tags, 0, sizeof(tags));
    vdb_parse_(text, tlen, &links, &tags);
    for (i = 0; i < links.count; ++i) {
        i32 t = vdb_title_node_(db, links.buf + links.off[i], links.n[i], 1);
        vdb_edge_(db, id, t);
    }
    for (i = 0; i < tags.count; ++i) {
        i32 t = vdb_tag_node_(db, tags.buf + tags.off[i], tags.n[i]);
        vdb_edge_(db, id, t);
        if (db->nodes[t].tagbit >= 0) meta |= (i64)1 << db->nodes[t].tagbit;
    }
    vstrs_free_(&links); vstrs_free_(&tags);

    if (to_store) store_add_meta(space ? db->store2 : db->store, nv, (i64)id, meta);
    free(nv);
    { i32 tl = (i32)strlen(title);
      char *both = (char *)malloc((size_t)(3 * (tl + 1) + tlen) + 1);
      i32 r;
      for (r = 0; r < 3; ++r) { memcpy(both + r * (tl + 1), title, (size_t)tl); both[r * (tl + 1) + tl] = '\n'; }   /* the title counts 3x in BM25 */
      memcpy(both + 3 * (tl + 1), text, (size_t)tlen);
      lex_add(&db->lex, (i64)id, both, 3 * (tl + 1) + tlen);
      free(both); }
    free(key);
    return id;
}

/* title + body -> vector (a real model sees the title once, the hash embedder twice) */
static void vdb_embed_doc_(const VecDB *db, const char *title, const char *text, f32 *vec)
{
    i32 tl = (i32)strlen(title), xl = (i32)strlen(text), rep = db->embed ? 1 : 2, r, w = 0;
    char *both = (char *)malloc((size_t)(rep * (tl + 1) + xl) + 1);
    for (r = 0; r < rep; ++r) { memcpy(both + w, title, (size_t)tl); w += tl; both[w++] = ' '; }
    memcpy(both + w, text, (size_t)xl); w += xl;
    if (db->ec && db->embed) {
        u64 h = vdb_hash64_(both, w, db->ec->seed);
        if (!vdb_ec_get_(db->ec, h, vec)) { vdb_embed_(db, both, w, 0, vec); vdb_ec_put_(db->ec, h, vec); }
    } else vdb_embed_(db, both, w, 0, vec);
    free(both);
}

/* Add (or replace) a note, embedding title + text with the database's embedder. Returns the node id. */
i32 vecdb_add_note(VecDB *db, const char *title, const char *text)
{
    f32 *vec = (f32 *)malloc((size_t)db->dim * sizeof(f32));
    i32 id;
    vdb_embed_doc_(db, title, text, vec);
    if (db->embed && vdb_zero_(vec, db->dim)) { free(vec); fprintf(stderr, "vecdb: embedding failed, note not added: %s\n", title); return -1; }
    id = vecdb_add_note_vec(db, title, text, vec);               /* takes the lock; the model call above did not */
    free(vec);
    return id;
}

typedef struct { const char **t; const i32 *l; f32 *out; i32 a, b, dim; } VEmbJob;

static void *vdb_embjob_(void *arg)
{
    VEmbJob *j = (VEmbJob *)arg;
    i32 i;
    for (i = j->a; i < j->b; ++i) vecdb_embed(j->t[i], j->l[i], j->out + (size_t)i * j->dim, j->dim);
    return (void *)0;
}

/* Add many notes at once. With a model server the texts go over in batches of 32 (one round trip each, the
 * model pads little and uses all cores); with the built-in embedder the embedding runs on all cores.
 * Same result as calling vecdb_add_note in a loop. Returns how many were added. */
i32 vecdb_add_notes(VecDB *db, i32 n, const char **titles, const char **texts)
{
    i32 i, rep = db->embed ? 1 : 2, added = 0;
    const char **docs = (const char **)malloc((size_t)(n ? n : 1) * sizeof(char *));
    i32 *lens = (i32 *)malloc((size_t)(n ? n : 1) * sizeof(i32));
    f32 *vec = (f32 *)malloc((size_t)(n ? n : 1) * (size_t)db->dim * sizeof(f32));
    for (i = 0; i < n; ++i) {
        i32 tl = (i32)strlen(titles[i]), xl = (i32)strlen(texts[i]), r, w = 0;
        char *d = (char *)malloc((size_t)(rep * (tl + 1) + xl) + 1);
        for (r = 0; r < rep; ++r) { memcpy(d + w, titles[i], (size_t)tl); w += tl; d[w++] = ' '; }
        memcpy(d + w, texts[i], (size_t)xl); w += xl; d[w] = 0;
        docs[i] = d; lens[i] = w;
    }
    if (db->ec && db->embed) {                                     /* cached documents cost nothing; only the misses go to the model */
        i32 *miss = (i32 *)malloc((size_t)(n ? n : 1) * sizeof(i32)), nm = 0;
        u64 *hs = (u64 *)malloc((size_t)(n ? n : 1) * sizeof(u64));
        for (i = 0; i < n; ++i) {
            hs[i] = vdb_hash64_(docs[i], lens[i], db->ec->seed);
            if (!vdb_ec_get_(db->ec, hs[i], vec + (size_t)i * db->dim)) miss[nm++] = i;
        }
        if (nm) {
            const char **mt = (const char **)malloc((size_t)nm * sizeof(char *));
            i32 *ml = (i32 *)malloc((size_t)nm * sizeof(i32));
            f32 *mv = (f32 *)malloc((size_t)nm * (size_t)db->dim * sizeof(f32));
            i32 z;
            for (z = 0; z < nm; ++z) { mt[z] = docs[miss[z]]; ml[z] = lens[miss[z]]; }
            if (db->embed_batch) db->embed_batch(db->embed_ctx, nm, mt, ml, 0, mv, db->dim);
            else for (z = 0; z < nm; ++z) db->embed(db->embed_ctx, mt[z], ml[z], 0, mv + (size_t)z * db->dim, db->dim);
            for (z = 0; z < nm; ++z) { memcpy(vec + (size_t)miss[z] * db->dim, mv + (size_t)z * db->dim, (size_t)db->dim * sizeof(f32)); vdb_ec_put_(db->ec, hs[miss[z]], mv + (size_t)z * db->dim); }
            free(mt); free(ml); free(mv);
        }
        fflush(db->ec->f);
        free(miss); free(hs);
    }
    else if (db->embed_batch) db->embed_batch(db->embed_ctx, n, docs, lens, 0, vec, db->dim);
    else if (db->embed) { for (i = 0; i < n; ++i) db->embed(db->embed_ctx, docs[i], lens[i], 0, vec + (size_t)i * db->dim, db->dim); }
    else {
        i32 nt = (i32)sysconf(_SC_NPROCESSORS_ONLN), t;
        VEmbJob jobs[16]; pthread_t th[16];
        if (nt < 1) nt = 1;
        if (nt > 16) nt = 16;
        if (n < 64) nt = 1;
        for (t = 0; t < nt; ++t) { jobs[t].t = docs; jobs[t].l = lens; jobs[t].out = vec; jobs[t].dim = db->dim; jobs[t].a = (i32)((i64)n * t / nt); jobs[t].b = (i32)((i64)n * (t + 1) / nt); }
        for (t = 1; t < nt; ++t) pthread_create(&th[t], (const pthread_attr_t *)0, vdb_embjob_, &jobs[t]);
        vdb_embjob_(&jobs[0]);
        for (t = 1; t < nt; ++t) pthread_join(th[t], (void **)0);
    }
    pthread_rwlock_wrlock(&db->lk);
    for (i = 0; i < n; ++i) {
        if (db->embed && vdb_zero_(vec + (size_t)i * db->dim, db->dim)) fprintf(stderr, "vecdb: embedding failed, note not added: %s\n", titles[i]);
        else if (vdb_add_(db, titles[i], texts[i], vec + (size_t)i * db->dim, 1, 1, 0) >= 0) ++added;
        free((void *)docs[i]);
    }
    pthread_rwlock_unlock(&db->lk);
    free(docs); free(lens); free(vec);
    return added;
}

/* Remove a note. If other notes still link to it, it stays behind as a ghost. */
static i32 vdbl_remove_note(VecDB *db, i32 id)
{
    VNode *v;
    if (id < 0 || id >= db->nn || db->nodes[id].kind != VN_NOTE) return 0;
    v = &db->nodes[id];
    vdb_unlink_(db, id);
    store_delete(v->space ? db->store2 : db->store, (i64)id);
    lex_delete(&db->lex, (i64)id);
    v->tlen = -1;
    if (v->in.n > 0) v->kind = VN_GHOST;
    else {
        char *key = (char *)malloc(strlen(v->name) + 1);
        i32 kl = vdb_norm_(v->name, (i32)strlen(v->name), key, 1);
        hs_del(&db->names, key, kl);
        free(key);
        v->kind = VN_DEAD;
    }
    return 1;
}

/* (re)build the vector index; a no-op when nothing was added since the last build/load */
static void vdbl_build(VecDB *db)
{
    if (!store_built(db->store)) store_build(db->store);
    if (db->store2 && !store_built(db->store2)) store_build(db->store2);
}

/* node id of a note (or ghost) by title, -1 if unknown */
static i32 vdbl_find(const VecDB *db, const char *title)
{
    char *key = (char *)malloc(strlen(title) + 1);
    i32 kl = vdb_norm_(title, (i32)strlen(title), key, 1), id = hs_get(&db->names, key, kl);
    free(key);
    return id;
}

static i32 vdbl_find_tag(const VecDB *db, const char *tag)
{
    char *key = (char *)malloc(strlen(tag) + 1);
    i32 kl, id;
    if (*tag == '#') ++tag;
    kl = vdb_norm_(tag, (i32)strlen(tag), key, 0);
    id = hs_get(&db->tags, key, kl);
    free(key);
    return id;
}

static i32 vdb_copy_kind_(const VecDB *db, const VIVec *l, u8 kind, i32 *out, i32 cap)
{
    i32 i, n = 0;
    for (i = 0; i < l->n && n < cap; ++i)
        if (db->nodes[l->v[i]].kind == kind) out[n++] = l->v[i];
    return n;
}

/* notes and ghosts a note links to */
static i32 vdbl_links(const VecDB *db, i32 id, i32 *out, i32 cap)
{
    const VIVec *l = &db->nodes[id].out;
    i32 i, n = 0;
    for (i = 0; i < l->n && n < cap; ++i) {
        u8 k = db->nodes[l->v[i]].kind;
        if (k == VN_NOTE || k == VN_GHOST) out[n++] = l->v[i];
    }
    return n;
}
/* notes that link to this one (the backlinks) */
static i32 vdbl_backlinks(const VecDB *db, i32 id, i32 *out, i32 cap) { return vdb_copy_kind_(db, &db->nodes[id].in, VN_NOTE, out, cap); }
/* tag nodes of a note */
static i32 vdbl_tags_of(const VecDB *db, i32 id, i32 *out, i32 cap) { return vdb_copy_kind_(db, &db->nodes[id].out, VN_TAG, out, cap); }
/* notes carrying a tag (a parent tag also holds its children's notes) */
static i32 vdbl_notes_with_tag(const VecDB *db, i32 tag, i32 *out, i32 cap) { return vdb_copy_kind_(db, &db->nodes[tag].in, VN_NOTE, out, cap); }

static i32 vdbl_count(const VecDB *db, u8 kind)
{
    i32 i, n = 0;
    for (i = 0; i < db->nn; ++i) if (db->nodes[i].kind == kind) ++n;
    return n;
}

static i32 vdb_nb_(const VecDB *db, i32 g, i32 idx);

/* Graph-aware embeddings: v' = normalise(v + alpha * mean(v of linked/backlinked notes)),
 * so a note also "sounds like" its neighbourhood. Rebuilds the vector store (call once
 * after ingest, then vecdb_build). Notes without neighbours are unchanged. */
static void vdbl_smooth(VecDB *db, f32 alpha)
{
    f32 *nv = (f32 *)malloc((size_t)db->nn * (size_t)db->dim * sizeof(f32)), *acc = (f32 *)malloc((size_t)db->dim * sizeof(f32));
    RagStore *fresh;
    i32 i, j, d;
    for (i = 0; i < db->nn; ++i) {
        VNode *v = &db->nodes[i];
        i32 cnt = 0;
        f32 nrm = 0.0f;
        if (v->kind != VN_NOTE || v->space) continue;
        for (d = 0; d < db->dim; ++d) acc[d] = 0.0f;
        for (j = 0; j < v->out.n + v->in.n; ++j) {
            i32 wi = vdb_nb_(db, i, j);
            const f32 *wv = db->nodes[wi].kind == VN_NOTE ? vdb_vec_(db, wi) : (const f32 *)0;
            if (!wv) continue;
            for (d = 0; d < db->dim; ++d) acc[d] += wv[d];
            ++cnt;
        }
        for (d = 0; d < db->dim; ++d) {
            f32 x = vdb_vec_(db, i)[d] + (cnt ? alpha * acc[d] / (f32)cnt : 0.0f);
            nv[(size_t)i * (size_t)db->dim + (size_t)d] = x; nrm += x * x;
        }
        if (nrm > 0.0f) { nrm = 1.0f / (f32)sqrt((f64)nrm); for (d = 0; d < db->dim; ++d) nv[(size_t)i * (size_t)db->dim + (size_t)d] *= nrm; }
    }
    fresh = store_new(db->arena, db->dim);
    store_set_fast(fresh, 0.97f);
    for (i = 0; i < db->nn; ++i) {
        VNode *v = &db->nodes[i];
        i64 meta = 0;
        if (v->kind != VN_NOTE || v->space) continue;
        for (j = 0; j < v->out.n; ++j) { i32 t = v->out.v[j]; if (db->nodes[t].kind == VN_TAG && db->nodes[t].tagbit >= 0) meta |= (i64)1 << db->nodes[t].tagbit; }
        store_add_meta(fresh, nv + (size_t)i * (size_t)db->dim, (i64)i, meta);
    }
    store_destroy(db->store);
    db->store = fresh;
    free(nv); free(acc);
}

#include "vecdb_embed.c"

#include "vecdb_search.c"

#include "vecdb_graph.c"

#include "vecdb_disk.c"

/* ---- public API: the locks ----
 * One readers-writer lock per database: searches and lookups share it, anything that changes the database
 * (add, remove, smooth, build, save/sync) is exclusive. Embedding a text (the slow part, a model call) happens
 * OUTSIDE the lock. The body file has its own mutex because reading it moves the file position. */
i32 vecdb_search(VecDB *db, const char *query, i32 k, i32 mode, i32 tag, VHit *out)
{
    i32 r;
    pthread_rwlock_rdlock(&((VecDB *)db)->lk);
    r = vdbl_search(db, query, k, mode, tag, out);
    pthread_rwlock_unlock(&((VecDB *)db)->lk);
    return r;
}

i32 vecdb_explore(VecDB *db, const char *query, i32 k, const VExplore *opt, VHit *out)
{
    i32 r;
    pthread_rwlock_rdlock(&((VecDB *)db)->lk);
    r = vdbl_explore(db, query, k, opt, out);
    pthread_rwlock_unlock(&((VecDB *)db)->lk);
    return r;
}

i32 vecdb_query(VecDB *db, const char *query, i32 k, i32 mcts, VHit *out)
{
    i32 r;
    pthread_rwlock_rdlock(&((VecDB *)db)->lk);
    r = vdbl_query(db, query, k, mcts, out);
    pthread_rwlock_unlock(&((VecDB *)db)->lk);
    return r;
}

i32 vecdb_search_images(VecDB *db, const char *query, i32 k, VHit *out)
{
    i32 r;
    pthread_rwlock_rdlock(&((VecDB *)db)->lk);
    r = vdbl_search_images(db, query, k, out);
    pthread_rwlock_unlock(&((VecDB *)db)->lk);
    return r;
}

i32 vecdb_similar_images(VecDB *db, i32 id, i32 k, VHit *out)
{
    i32 r;
    pthread_rwlock_rdlock(&((VecDB *)db)->lk);
    r = vdbl_similar_images(db, id, k, out);
    pthread_rwlock_unlock(&((VecDB *)db)->lk);
    return r;
}

i32 vecdb_links(const VecDB *db, i32 id, i32 *out, i32 cap)
{
    i32 r;
    pthread_rwlock_rdlock(&((VecDB *)db)->lk);
    r = vdbl_links(db, id, out, cap);
    pthread_rwlock_unlock(&((VecDB *)db)->lk);
    return r;
}

i32 vecdb_backlinks(const VecDB *db, i32 id, i32 *out, i32 cap)
{
    i32 r;
    pthread_rwlock_rdlock(&((VecDB *)db)->lk);
    r = vdbl_backlinks(db, id, out, cap);
    pthread_rwlock_unlock(&((VecDB *)db)->lk);
    return r;
}

i32 vecdb_tags_of(const VecDB *db, i32 id, i32 *out, i32 cap)
{
    i32 r;
    pthread_rwlock_rdlock(&((VecDB *)db)->lk);
    r = vdbl_tags_of(db, id, out, cap);
    pthread_rwlock_unlock(&((VecDB *)db)->lk);
    return r;
}

i32 vecdb_notes_with_tag(const VecDB *db, i32 tag, i32 *out, i32 cap)
{
    i32 r;
    pthread_rwlock_rdlock(&((VecDB *)db)->lk);
    r = vdbl_notes_with_tag(db, tag, out, cap);
    pthread_rwlock_unlock(&((VecDB *)db)->lk);
    return r;
}

i32 vecdb_find(const VecDB *db, const char *title)
{
    i32 r;
    pthread_rwlock_rdlock(&((VecDB *)db)->lk);
    r = vdbl_find(db, title);
    pthread_rwlock_unlock(&((VecDB *)db)->lk);
    return r;
}

i32 vecdb_find_tag(const VecDB *db, const char *tag)
{
    i32 r;
    pthread_rwlock_rdlock(&((VecDB *)db)->lk);
    r = vdbl_find_tag(db, tag);
    pthread_rwlock_unlock(&((VecDB *)db)->lk);
    return r;
}

i32 vecdb_count(const VecDB *db, u8 kind)
{
    i32 r;
    pthread_rwlock_rdlock(&((VecDB *)db)->lk);
    r = vdbl_count(db, kind);
    pthread_rwlock_unlock(&((VecDB *)db)->lk);
    return r;
}

char *vecdb_text(const VecDB *db, i32 id)
{
    char * r;
    pthread_rwlock_rdlock(&((VecDB *)db)->lk);
    r = vdbl_text(db, id);
    pthread_rwlock_unlock(&((VecDB *)db)->lk);
    return r;
}

i32 vecdb_remove_note(VecDB *db, i32 id)
{
    i32 r;
    pthread_rwlock_wrlock(&db->lk);
    r = vdbl_remove_note(db, id);
    pthread_rwlock_unlock(&db->lk);
    return r;
}

void vecdb_smooth(VecDB *db, f32 alpha)
{
    pthread_rwlock_wrlock(&db->lk);
    vdbl_smooth(db, alpha);
    pthread_rwlock_unlock(&db->lk);
}

void vecdb_build(VecDB *db)
{
    pthread_rwlock_wrlock(&db->lk);
    vdbl_build(db);
    pthread_rwlock_unlock(&db->lk);
}

i32 vecdb_save(VecDB *db, const char *path)
{
    i32 r;
    pthread_rwlock_wrlock(&db->lk);
    r = vdbl_save(db, path);
    pthread_rwlock_unlock(&db->lk);
    return r;
}

i32 vecdb_sync(VecDB *db, const char *path)
{
    i32 r;
    pthread_rwlock_wrlock(&db->lk);
    r = vdbl_sync(db, path);
    pthread_rwlock_unlock(&db->lk);
    return r;
}
