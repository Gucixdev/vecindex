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
 *       -O2 -march=native -I. vecdb.c -DVECDB_DEMO -lm -lpthread -o vecdb_t && ./vecdb_t
 *   ... -DVECDB_CLI  for the command line tool (see vecdb_cli below)
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

static void vivec_add_(VIVec *a, i32 x)
{
    if (a->n == a->cap) {
        a->cap = a->cap ? a->cap * 2 : 4;
        a->v = (i32 *)realloc(a->v, (size_t)a->cap * sizeof(i32));
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
    char *name;         /* note title / tag as written the first time */
    i64   toff;         /* note body: offset and length in the text file (not held in RAM) */
    i32   tlen;         /* -1: no body */
    u8    space;        /* 0 text vectors, 1 image vectors */
    VIVec out;          /* wikilinks (note -> note/ghost) and tag edges (note -> tag) */
    VIVec in;           /* backlinks; for a tag: the notes carrying it */
    i32   tagbit;       /* tags: bit in the store's i64 meta (0..62), -1 = none */
    u8    kind;
} VNode;

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

static char *vdb_dup_(const char *s, i32 n)
{
    char *p = (char *)malloc((size_t)n + 1);
    memcpy(p, s, (size_t)n); p[n] = 0;
    return p;
}

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
        fseek(c->f, (long)(16 + (size_t)c->n * rec), SEEK_SET);
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
    fseek(c->f, (long)(16 + (size_t)row * rec + 8), SEEK_SET);
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
        fseek(c->f, 0, SEEK_END);
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
    for (i = 0; i < db->nn; ++i) { free(db->nodes[i].name); free(db->nodes[i].out.v); free(db->nodes[i].in.v); }
    free(db->nodes);
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
        db->ncap = db->ncap ? db->ncap * 2 : 256;
        db->nodes = (VNode *)realloc(db->nodes, (size_t)db->ncap * sizeof(VNode));
    }
    v = &db->nodes[db->nn];
    memset(v, 0, sizeof(*v));
    v->kind = kind; v->name = vdb_dup_(name, n); v->tagbit = -1; v->tlen = -1;
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
        if (db->ntagbits < 62) db->nodes[id].tagbit = db->ntagbits++;
        hs_put(&db->tags, key, n, id);
    }
    return id;
}

static void vdb_edge_(VecDB *db, i32 from, i32 to)
{
    if (from == to || vivec_has_(&db->nodes[from].out, to)) return;
    vivec_add_(&db->nodes[from].out, to);
    vivec_add_(&db->nodes[to].in, from);
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
    fseek(db->tf, 0, SEEK_END);
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
    fseek(db->tf, (long)v->toff, SEEK_SET);
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
    if (v->kind != VN_NOTE) { free(v->name); v->name = vdb_dup_(title, (i32)strlen(title)); }
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

/* ------------------------------------------------------------------ */
/* search                                                              */
/* ------------------------------------------------------------------ */

#define VDB_RRF_K 60.0f

static f32 vdb_dot_(const f32 *a, const f32 *b, i32 n)
{
    f32 s = 0.0f; i32 i;
    for (i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

/* mode: 0 hybrid (vector + BM25, reciprocal rank fusion), 1 vector only, 2 lexical only.
 * tag >= 0 keeps only notes carrying that tag node. Returns the number of hits. */
/* ------------------------------------------------------------------ */
/* embedding server: a child process that keeps a model loaded.        */
/* Protocol (native endian): on start the child writes u32 dim; then   */
/* per request the parent writes  u8 kind ('D' document, 'Q' query,    */
/* 'I' image file path)  u32 len  bytes  and reads dim f32 (unit).     */
/* See embedsrv.py for the ready-made one (multilingual-e5, CLIP).     */
/* ------------------------------------------------------------------ */

typedef struct { pid_t pid; int to, from; i32 dim; } VEmbedProc;

VEmbedProc *vecdb_embproc_open(const char *cmd, const char *arg)
{
    int a[2], b[2];
    VEmbedProc *p;
    u32 dim = 0;
    if (pipe(a) != 0) return (VEmbedProc *)0;
    if (pipe(b) != 0) { close(a[0]); close(a[1]); return (VEmbedProc *)0; }
    p = (VEmbedProc *)calloc(1, sizeof(*p));
    p->pid = fork();
    if (p->pid < 0) { free(p); return (VEmbedProc *)0; }
    if (p->pid == 0) {
        dup2(a[0], 0); dup2(b[1], 1);
        close(a[0]); close(a[1]); close(b[0]); close(b[1]);
        if (arg) execl("/bin/sh", "sh", "-c", cmd, "sh", arg, (char *)0);
        else execl("/bin/sh", "sh", "-c", cmd, (char *)0);
        _exit(127);
    }
    close(a[0]); close(b[1]);
    p->to = a[1]; p->from = b[0];
    if (read(p->from, &dim, 4) != 4 || dim == 0 || dim > 8192) { close(p->to); close(p->from); kill(p->pid, SIGTERM); waitpid(p->pid, (int *)0, 0); free(p); return (VEmbedProc *)0; }
    p->dim = (i32)dim;
    return p;
}

static i32 vdb_rd_(int fd, void *buf, size_t n)
{
    size_t got = 0;
    while (got < n) { ssize_t r = read(fd, (char *)buf + got, n - got); if (r <= 0) return -1; got += (size_t)r; }
    return 0;
}

/* status byte, then (status 0) n vectors; a failed request returns -1 and leaves out untouched */
static i32 vdb_reply_(VEmbedProc *p, f32 *out, i32 n)
{
    u8 st;
    if (vdb_rd_(p->from, &st, 1) != 0) return -1;
    if (st != 0) return -1;
    return vdb_rd_(p->from, out, (size_t)n * (size_t)p->dim * sizeof(f32));
}

/* out[dim]; returns 0, or -1 if the server failed or is gone */
i32 vecdb_embproc_ask(VEmbedProc *p, char kind, const char *bytes, i32 n, f32 *out)
{
    u32 len = (u32)n;
    size_t off = 0;
    char hdr[5];
    hdr[0] = kind; memcpy(hdr + 1, &len, 4);
    if (write(p->to, hdr, 5) != 5) return -1;
    while (off < (size_t)n) { ssize_t w = write(p->to, bytes + off, (size_t)n - off); if (w <= 0) return -1; off += (size_t)w; }
    return vdb_reply_(p, out, 1);
}

/* n texts in one round trip (kind 'B' documents / 'C' queries); out is n x dim; 0 on success */
i32 vecdb_embproc_ask_batch(VEmbedProc *p, char kind, i32 n, const char **texts, const i32 *lens, f32 *out)
{
    size_t total = 4, off = 0;
    i32 i;
    char *buf, hdr[5];
    u32 len, cnt = (u32)n;
    for (i = 0; i < n; ++i) total += 4 + (size_t)lens[i];
    buf = (char *)malloc(total);
    memcpy(buf, &cnt, 4); off = 4;
    for (i = 0; i < n; ++i) { u32 l = (u32)lens[i]; memcpy(buf + off, &l, 4); off += 4; memcpy(buf + off, texts[i], (size_t)lens[i]); off += (size_t)lens[i]; }
    len = (u32)total; hdr[0] = kind; memcpy(hdr + 1, &len, 4);
    if (write(p->to, hdr, 5) != 5) { free(buf); return -1; }
    for (off = 0; off < total; ) { ssize_t w = write(p->to, buf + off, total - off); if (w <= 0) { free(buf); return -1; } off += (size_t)w; }
    free(buf);
    return vdb_reply_(p, out, n);
}

void vecdb_embproc_close(VEmbedProc *p)
{
    if (!p) return;
    close(p->to); close(p->from);
    waitpid(p->pid, (int *)0, 0);
    free(p);
}

/* Several servers at once (e.g. one on the GPU, one on the CPU): a batch is cut into groups of 32 and every
 * server pulls the next group as soon as it is free, so a fast server naturally does more of the work.
 * Single texts (queries) go to the first server. Spec: "cmd1|cmd2|..." */
typedef struct { VEmbedProc *p[8]; i32 n, dim; } VEmbedPool;

VEmbedPool *vecdb_embpool_open(const char *spec)
{
    VEmbedPool *pl = (VEmbedPool *)calloc(1, sizeof(*pl));
    char *s = (char *)malloc(strlen(spec) + 1), *t, *save;
    strcpy(s, spec);
    for (t = s; t; t = save) {
        VEmbedProc *p;
        save = strchr(t, '|');
        if (save) *save++ = 0;
        if (!*t || pl->n >= 8) continue;
        p = vecdb_embproc_open(t, (const char *)0);
        if (!p) { fprintf(stderr, "vecdb: embedding server failed to start: %s\n", t); continue; }
        if (pl->n && p->dim != pl->dim) { fprintf(stderr, "vecdb: server dimension mismatch, skipped: %s\n", t); vecdb_embproc_close(p); continue; }
        pl->p[pl->n++] = p; pl->dim = p->dim;
    }
    free(s);
    if (pl->n == 0) { free(pl); return (VEmbedPool *)0; }
    return pl;
}

void vecdb_embpool_close(VEmbedPool *pl)
{
    i32 i;
    if (!pl) return;
    for (i = 0; i < pl->n; ++i) vecdb_embproc_close(pl->p[i]);
    free(pl);
}

typedef struct {
    VEmbedProc *p; i32 kind, n, dim; const char **texts; const i32 *lens; f32 *out;
    i32 *next; pthread_mutex_t *mu; i32 done_groups;
} VPoolJob;

static void *vdb_pool_worker_(void *arg)
{
    VPoolJob *j = (VPoolJob *)arg;
    for (;;) {
        i32 s, m;
        pthread_mutex_lock(j->mu);
        s = *j->next; *j->next += 32;
        pthread_mutex_unlock(j->mu);
        if (s >= j->n) break;
        m = j->n - s < 32 ? j->n - s : 32;
        if (vecdb_embproc_ask_batch(j->p, j->kind ? 'C' : 'B', m, j->texts + s, j->lens + s, j->out + (size_t)s * j->dim) != 0) {
            i32 i;                                                   /* the batch failed: try the texts one by one, a text that still fails becomes a zero vector */
            for (i = 0; i < m; ++i) {
                f32 *o = j->out + (size_t)(s + i) * j->dim;
                if (vecdb_embproc_ask(j->p, j->kind ? 'Q' : 'D', j->texts[s + i], j->lens[s + i], o) != 0) { i32 z; for (z = 0; z < j->dim; ++z) o[z] = 0.0f; }
            }
        }
        ++j->done_groups;
    }
    return (void *)0;
}

static void vdb_pool_embed_(void *ctx, const char *t, i32 n, i32 kind, f32 *out, i32 dim)
{
    VEmbedPool *pl = (VEmbedPool *)ctx;
    if (pl->dim != dim || vecdb_embproc_ask(pl->p[0], kind ? 'Q' : 'D', t, n, out) != 0) { i32 i; for (i = 0; i < dim; ++i) out[i] = 0.0f; }
}

static void vdb_pool_batch_(void *ctx, i32 n, const char **texts, const i32 *lens, i32 kind, f32 *out, i32 dim)
{
    VEmbedPool *pl = (VEmbedPool *)ctx;
    VPoolJob jobs[8];
    pthread_t th[8];
    pthread_mutex_t mu;
    i32 next = 0, i;
    if (pl->dim != dim) { for (i = 0; i < n * dim; ++i) out[i] = 0.0f; return; }
    pthread_mutex_init(&mu, (const pthread_mutexattr_t *)0);
    for (i = 0; i < pl->n; ++i) {
        jobs[i].p = pl->p[i]; jobs[i].kind = kind; jobs[i].n = n; jobs[i].dim = dim; jobs[i].texts = texts; jobs[i].lens = lens;
        jobs[i].out = out; jobs[i].next = &next; jobs[i].mu = &mu; jobs[i].done_groups = 0;
    }
    for (i = 1; i < pl->n; ++i) pthread_create(&th[i], (const pthread_attr_t *)0, vdb_pool_worker_, &jobs[i]);
    vdb_pool_worker_(&jobs[0]);
    for (i = 1; i < pl->n; ++i) pthread_join(th[i], (void **)0);
    pthread_mutex_destroy(&mu);
    if (getenv("VECDB_POOL_STATS")) { for (i = 0; i < pl->n; ++i) fprintf(stderr, "  server %d did %d groups\n", i, jobs[i].done_groups); }
}

/* use the server(s) for every embedding of this database (documents and queries) */
void vecdb_use_embpool(VecDB *db, VEmbedPool *pl) { db->embed = vdb_pool_embed_; db->embed_ctx = pl; db->embed_batch = vdb_pool_batch_; }

/* candidate list of one query: at most a few dozen ids, so a linear find beats any dense O(nodes) array */
typedef struct { i32 id[256]; f32 fused[256], sim[256], lex[256]; i32 n; } VCand;

static i32 vdb_cand_(VCand *c, i32 id)
{
    i32 i;
    for (i = 0; i < c->n; ++i) if (c->id[i] == id) return i;
    if (c->n >= 256) return -1;
    c->id[c->n] = id; c->fused[c->n] = 0.0f; c->sim[c->n] = 0.0f; c->lex[c->n] = 0.0f;
    return c->n++;
}

/* search that also hands back the lexical hits (explore reuses them as rewards) */
static i32 vdb_search_(VecDB *db, const char *query, i32 k, i32 mode, i32 tag, VHit *out, i32 lexcap, i64 *lid_out, f32 *lsc_out, i32 *nl_out, f32 *qout)
{
    i32 over = k * 3 + 10, nv = 0, nl = 0, i, n, exact = -1;
    i64 vid[256], lid[256];
    f32 vsc[256], lsc[256], *q = (f32 *)malloc((size_t)db->dim * sizeof(f32));
    VCand *c = (VCand *)malloc(sizeof(VCand));
    i32 qlen = (i32)strlen(query);

    if (over < lexcap) over = lexcap;
    if (over > 250) over = 250;
    c->n = 0;
    vdb_embed_(db, query, qlen, 1, q);
    if (mode != 2 && !(db->embed && vdb_zero_(q, db->dim))) {          /* a failed query embedding falls back to words only */
        i32 bit = tag >= 0 ? db->nodes[tag].tagbit : -1;
        nv = bit >= 0 ? store_search_filter(db->store, q, over, vid, vsc, (i64)1 << bit, 0) : store_search(db->store, q, over, vid, vsc);
    }
    if (mode != 1) nl = lex_search(&db->lex, query, qlen, over, lid, lsc);
    /* Rank fusion (RRF), weighted. Base weights favour the words (1.5 vs 0.6): a query taken from deep inside a
     * note is invisible to a vector that only read the first tokens, while exact words are never wrong in that
     * way. A side whose best hit clearly stands out from its runner-up (a rare word, a near-duplicate) gets up
     * to +2, so a confident vector still wins. Plain RRF ties the sides and loses exact hits. */
    { f32 wv = 0.6f, wl = 1.5f;      /* base weights: measured on Wikipedia with two embedders, words beat a blind vector */
      if (nv > 1 && vsc[0] < 0.0f && vsc[1] < 0.0f) { f32 m = 1.0f - vsc[0] / vsc[1]; wv += 2.0f * (m > 0 ? (m < 1 ? m : 1.0f) : 0.0f); }
      if (nl == 1) wl = 3.0f;
      else if (nl > 1 && lsc[0] > 0.0f) wl += 2.0f * (1.0f - lsc[1] / lsc[0]);
      for (i = 0; i < nv; ++i) {
          i32 ci = vdb_cand_(c, (i32)vid[i]);
          if (ci < 0) continue;
          c->fused[ci] += wv / (VDB_RRF_K + (f32)i + 1.0f);
      }
      for (i = 0; i < nl; ++i) {
          i32 ci = vdb_cand_(c, (i32)lid[i]);
          if (ci < 0) continue;
          c->lex[ci] = lsc[0] > 0 ? lsc[i] / lsc[0] : 0.0f;
          c->fused[ci] += wl / (VDB_RRF_K + (f32)i + 1.0f);
      } }
    /* the hash dictionary: a query that IS a title goes straight to that note, first */
    if (mode != 1) {
        char *key = (char *)malloc((size_t)qlen + 1);
        i32 kl = vdb_norm_(query, qlen, key, 1), tid = kl > 0 ? hs_get(&db->names, key, kl) : -1;
        free(key);
        if (tid >= 0 && db->nodes[tid].kind == VN_NOTE) { i32 ci = vdb_cand_(c, tid); if (ci >= 0) { c->fused[ci] += 10.0f; exact = tid; } }
    }
    /* the vector side only knows the cosine of its own hits; fill in the rest */
    for (i = 0; i < c->n; ++i)
        if (c->sim[i] == 0.0f && vdb_vec_(db, c->id[i]) && db->nodes[c->id[i]].kind == VN_NOTE) c->sim[i] = vdb_dot_(q, vdb_vec_(db, c->id[i]), db->dim);
    if (tag >= 0) {                                             /* keep notes that carry the tag */
        i32 w = 0;
        for (i = 0; i < c->n; ++i) if (vivec_has_(&db->nodes[c->id[i]].out, tag)) {
            c->id[w] = c->id[i]; c->fused[w] = c->fused[i]; c->sim[w] = c->sim[i]; c->lex[w] = c->lex[i]; ++w;
        }
        c->n = w;
    }
    (void)exact;
    /* selection sort of the top k */
    n = 0;
    for (i = 0; i < c->n && n < k; ++i) {
        i32 j, best = i;
        f32 tf;
        for (j = i + 1; j < c->n; ++j) if (c->fused[j] > c->fused[best]) best = j;
        if (best != i) {
            i32 ti = c->id[i]; c->id[i] = c->id[best]; c->id[best] = ti;
            tf = c->fused[i]; c->fused[i] = c->fused[best]; c->fused[best] = tf;
            tf = c->sim[i]; c->sim[i] = c->sim[best]; c->sim[best] = tf;
            tf = c->lex[i]; c->lex[i] = c->lex[best]; c->lex[best] = tf;
        }
        out[n].node = c->id[i]; out[n].sim = c->sim[i]; out[n].lex = c->lex[i];
        out[n].score = c->fused[i]; out[n].via = -1; out[n].depth = 0; ++n;
    }
    if (lid_out) { memcpy(lid_out, lid, (size_t)nl * sizeof(i64)); memcpy(lsc_out, lsc, (size_t)nl * sizeof(f32)); *nl_out = nl; }
    if (qout) memcpy(qout, q, (size_t)db->dim * sizeof(f32));
    free(q); free(c);
    return n;
}

static i32 vdbl_search(VecDB *db, const char *query, i32 k, i32 mode, i32 tag, VHit *out)
{ return vdb_search_(db, query, k, mode, tag, out, 0, (i64 *)0, (f32 *)0, (i32 *)0, (f32 *)0); }


/* ---- images: a second vector space, filled by an image model (CLIP) that also embeds text queries into it ---- */

/* start the image space; p is a server started with embedsrv.py --clip */
void vecdb_enable_images(VecDB *db, VEmbedProc *p)
{
    db->proc2 = p; db->dim2 = p->dim;
    if (!db->store2) { db->store2 = store_new(db->arena, p->dim); store_set_fast(db->store2, 0.97f); }
}

/* after vecdb_load: attach the model so text queries can search a loaded image space */
void vecdb_set_image_model(VecDB *db, VEmbedProc *p) { if (db->store2 && p && p->dim == db->dim2) db->proc2 = p; }

/* Index a picture. `title` is how notes link to it ([[title]], ![[title]], ![](dir/title)); `caption`
 * (may be NULL) is searchable text. Returns the node id, or -1 if the model could not read the file. */
i32 vecdb_add_image(VecDB *db, const char *title, const char *path, const char *caption)
{
    VEmbedProc *p = (VEmbedProc *)db->proc2;
    f32 *vec;
    char *body;
    i32 id;
    if (!p || !db->store2) return -1;
    vec = (f32 *)malloc((size_t)db->dim2 * sizeof(f32));
    if (vecdb_embproc_ask(p, 'I', path, (i32)strlen(path), vec) != 0) { free(vec); return -1; }
    { i32 i; f32 s = 0.0f; for (i = 0; i < db->dim2; ++i) s += vec[i] * vec[i]; if (s < 1e-6f) { free(vec); return -1; } }
    body = (char *)malloc(strlen(caption ? caption : "") + strlen(path) + 32);
    sprintf(body, "%s\nfile: %s", caption ? caption : "", path);
    pthread_rwlock_wrlock(&db->lk);
    id = vdb_add_(db, title, body, vec, 1, 1, 1);
    pthread_rwlock_unlock(&db->lk);
    free(vec); free(body);
    return id;
}

/* text -> pictures: the query goes through the image model's text tower; BM25 over captions and file
 * names is fused in, so a word in a file name still wins. Only image notes are returned. */
static i32 vdbl_search_images(VecDB *db, const char *query, i32 k, VHit *out)
{
    VEmbedProc *p = (VEmbedProc *)db->proc2;
    i64 vid[128], lid[128]; f32 vsc[128], lsc[128], *q;
    VCand *c;
    i32 nv, nl, i, n = 0, over = k * 4 + 10;
    if (!p || !db->store2) return 0;
    if (over > 120) over = 120;
    q = (f32 *)malloc((size_t)db->dim2 * sizeof(f32));
    if (vecdb_embproc_ask(p, 'Q', query, (i32)strlen(query), q) != 0) { free(q); return 0; }
    c = (VCand *)malloc(sizeof(VCand)); c->n = 0;
    nv = store_search(db->store2, q, over, vid, vsc);
    nl = lex_search(&db->lex, query, (i32)strlen(query), over, lid, lsc);
    for (i = 0; i < nv; ++i) { i32 ci = vdb_cand_(c, (i32)vid[i]); if (ci >= 0) { c->fused[ci] += 1.0f / (VDB_RRF_K + (f32)i + 1.0f); c->sim[ci] = vdb_dot_(q, store_vec_ptr(db->store2, vid[i]), db->dim2); } }
    for (i = 0; i < nl; ++i) if (db->nodes[(i32)lid[i]].space) { i32 ci = vdb_cand_(c, (i32)lid[i]); if (ci >= 0) { c->fused[ci] += 1.0f / (VDB_RRF_K + (f32)i + 1.0f); c->lex[ci] = lsc[0] > 0 ? lsc[i] / lsc[0] : 0.0f; } }
    for (i = 0; i < c->n && n < k; ++i) {
        i32 j, best = i;
        for (j = i + 1; j < c->n; ++j) if (c->fused[j] > c->fused[best]) best = j;
        if (best != i) {
            i32 ti = c->id[i]; f32 tf;
            c->id[i] = c->id[best]; c->id[best] = ti;
            tf = c->fused[i]; c->fused[i] = c->fused[best]; c->fused[best] = tf;
            tf = c->sim[i]; c->sim[i] = c->sim[best]; c->sim[best] = tf;
            tf = c->lex[i]; c->lex[i] = c->lex[best]; c->lex[best] = tf;
        }
        out[n].node = c->id[i]; out[n].sim = c->sim[i]; out[n].lex = c->lex[i]; out[n].score = c->fused[i]; out[n].via = -1; out[n].depth = 0; ++n;
    }
    free(q); free(c);
    return n;
}

/* picture -> similar pictures (no model needed: uses the stored vector) */
static i32 vdbl_similar_images(VecDB *db, i32 id, i32 k, VHit *out)
{
    const f32 *q;
    i64 vid[64]; f32 vsc[64];
    i32 nv, i, n = 0;
    if (!db->store2 || id < 0 || id >= db->nn || !db->nodes[id].space) return 0;
    q = store_vec_ptr(db->store2, (i64)id);
    if (!q) return 0;
    nv = store_search(db->store2, q, k + 1 > 64 ? 64 : k + 1, vid, vsc);
    for (i = 0; i < nv && n < k; ++i) {
        if ((i32)vid[i] == id) continue;
        out[n].node = (i32)vid[i]; out[n].sim = vdb_dot_(q, store_vec_ptr(db->store2, vid[i]), db->dim2); out[n].lex = 0.0f;
        out[n].score = out[n].sim; out[n].via = -1; out[n].depth = 0; ++n;
    }
    for (i = 1; i < n; ++i) {                                    /* the index returns candidates, not an exact order */
        VHit x = out[i]; i32 j;
        for (j = i; j > 0 && out[j - 1].score < x.score; --j) out[j] = out[j - 1];
        out[j] = x;
    }
    return n;
}

/* ------------------------------------------------------------------ */
/* MCTS over the graph                                                 */
/* ------------------------------------------------------------------ */

typedef struct {
    i32 g;              /* graph node */
    i32 parent;         /* tree parent, -1 for the seeds' virtual root */
    i32 depth;
    i32 first_child, next_sib, nchild;
    i32 nact, tried, off;   /* neighbours of g: how many, how many were tried, random start offset */
    i32 visits;
    f64 wsum;
} VTree;

static u32 vdb_rng_(u32 *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }

static i32 vdb_nb_(const VecDB *db, i32 g, i32 idx)
{
    const VNode *v = &db->nodes[g];
    return idx < v->out.n ? v->out.v[idx] : v->in.v[idx - v->out.n];
}

static i32 vdb_on_path_(const VTree *t, i32 node, i32 g)
{
    for (; node >= 0; node = t[node].parent) if (t[node].g == g) return 1;
    return 0;
}

/* per-query state of the nodes MCTS touched; sparse (hash id -> slot), so a query costs
 * O(work done), never O(nodes in the database) */
typedef struct {
    const VecDB *db; const f32 *q;
    const i64 *lid; const f32 *lsc; i32 nl;
    H64 map;
    i32 *node, *vis, *via, *dep; f32 *rw; f64 *val;
    i32 n, cap;
} VSlots;

static i32 vs_slot_(VSlots *s, i32 g)
{
    i32 i = h64_get(&s->map, (i64)g);
    if (i >= 0) return i;
    if (s->n >= s->cap) return s->cap - 1;                    /* never happens: cap is a proven bound */
    i = s->n++;
    { const VNode *v = &s->db->nodes[g];
      f32 r = 0.0f;
      i32 j;
      for (j = 0; j < s->nl; ++j) if ((i32)s->lid[j] == g) { r = s->lsc[0] > 0 ? 0.4f * s->lsc[j] / s->lsc[0] : 0.0f; break; }
      if (v->kind == VN_NOTE && vdb_vec_(s->db, g)) { f32 c = vdb_dot_(s->q, vdb_vec_(s->db, g), s->db->dim); if (c > 0) r += 0.6f * c; }
      s->node[i] = g; s->rw[i] = r; s->vis[i] = 0; s->via[i] = -1; s->dep[i] = 0; s->val[i] = 0.0;
      h64_put(&s->map, (i64)g, i); }
    return i;
}
#define RW(g) (S.rw[vs_slot_(&S, (g))])

void vecdb_explore_defaults(VExplore *o)
{
    o->seeds = 6; o->iters = 600; o->maxdepth = 4; o->rollout = 3;
    o->c = 1.0f; o->gamma = 0.8f; o->seed = 1u;
}

/* Monte Carlo tree search from the query's hybrid hits. Returns up to k notes,
 * best first; hit.via / hit.depth say how each was reached. */
static i32 vdbl_explore(VecDB *db, const char *query, i32 k, const VExplore *opt, VHit *out)
{
    VExplore o;
    i32 nseed, i, it, ntree = 0, cap, n = 0, nlx = 0, scap;
    f32 *q = (f32 *)malloc((size_t)db->dim * sizeof(f32));
    i64 lid[256]; f32 lsc[256];
    VSlots S;
    VHit *seeds;
    VTree *t;
    i32 first_root = -1, last_root = -1, nroots = 0, *ord;
    u32 rng;
    f64 lg;

    if (opt) o = *opt; else vecdb_explore_defaults(&o);
    rng = o.seed ? o.seed : 1u;

    seeds = (VHit *)malloc((size_t)(o.seeds + 1) * sizeof(VHit));
    nseed = vdb_search_(db, query, o.seeds, 0, -1, seeds, 64, lid, lsc, &nlx, q);   /* hybrid seeds; its BM25 hits become rewards */
    scap = o.seeds + o.iters * (1 + o.rollout) + 16;
    S.db = db; S.q = q; S.lid = lid; S.lsc = lsc; S.nl = nlx; S.n = 0; S.cap = scap;
    h64_init(&S.map, scap);
    S.node = (i32 *)malloc((size_t)scap * sizeof(i32)); S.vis = (i32 *)malloc((size_t)scap * sizeof(i32));
    S.via = (i32 *)malloc((size_t)scap * sizeof(i32)); S.dep = (i32 *)malloc((size_t)scap * sizeof(i32));
    S.rw = (f32 *)malloc((size_t)scap * sizeof(f32)); S.val = (f64 *)malloc((size_t)scap * sizeof(f64));
    cap = nseed + o.iters + 8;
    t = (VTree *)calloc((size_t)cap, sizeof(VTree));
    for (i = 0; i < nseed; ++i) {                               /* the seeds are the root's children */
        VTree *r = &t[ntree];
        r->g = seeds[i].node; r->parent = -1; r->depth = 0; r->first_child = r->next_sib = -1;
        r->nact = db->nodes[r->g].out.n + db->nodes[r->g].in.n;
        r->off = r->nact ? (i32)(vdb_rng_(&rng) % (u32)r->nact) : 0;
        { i32 sl = vs_slot_(&S, r->g); S.vis[sl] += 1; S.val[sl] += S.rw[sl]; }
        if (first_root < 0) first_root = ntree; else t[last_root].next_sib = ntree;
        last_root = ntree; ++nroots; ++ntree;
    }
    for (it = 1; it <= o.iters && nroots > 0; ++it) {
        i32 cur = -1, node = -1, leafg, step, prevg, depth;
        f64 rew, best;
        lg = log((f64)it + 1.0);
        /* selection: walk down while the node is fully expanded */
        { i32 c0 = first_root;
          for (;;) {
              i32 c, bestc = -1;
              f64 bestv = -1.0;
              for (c = c0; c >= 0; c = t[c].next_sib) {           /* children of cur (or the roots) */
                  f64 v = t[c].visits == 0 ? 1e18 : t[c].wsum / t[c].visits + (f64)o.c * sqrt(lg / t[c].visits);
                  if (v > bestv) { bestv = v; bestc = c; }
              }
              if (bestc < 0) break;
              cur = bestc;
              if (t[cur].visits == 0) { node = cur; break; }      /* an unvisited seed or child: simulate it */
              if (t[cur].tried < t[cur].nact && t[cur].depth < o.maxdepth) {
                  /* expansion: the next neighbour not yet on the path */
                  i32 add = -1;
                  while (t[cur].tried < t[cur].nact) {
                      i32 g = vdb_nb_(db, t[cur].g, (t[cur].off + t[cur].tried++) % t[cur].nact);
                      if (db->nodes[g].kind != VN_DEAD && !vdb_on_path_(t, cur, g)) { add = g; break; }
                  }
                  if (add >= 0 && ntree < cap) {
                      VTree *nw = &t[ntree];
                      nw->g = add; nw->parent = cur; nw->depth = t[cur].depth + 1; nw->first_child = nw->next_sib = -1;
                      nw->nact = db->nodes[add].out.n + db->nodes[add].in.n;
                      nw->off = nw->nact ? (i32)(vdb_rng_(&rng) % (u32)nw->nact) : 0;
                      nw->next_sib = t[cur].first_child; t[cur].first_child = ntree; ++t[cur].nchild;
                      node = ntree++;
                      break;
                  }
              }
              if (t[cur].first_child < 0) { node = cur; break; }  /* terminal */
              c0 = t[cur].first_child;
          } }
        if (node < 0) continue;
        leafg = t[node].g;
        /* simulation: a short random walk; the value is the best (discounted) reward met on it */
        rew = RW(leafg); best = rew; prevg = -1; depth = t[node].depth;
        { i32 g = leafg;
          f64 disc = 1.0;
          for (step = 0; step < o.rollout; ++step) {
              i32 tot = db->nodes[g].out.n + db->nodes[g].in.n, tries, ng = -1;
              if (tot == 0) break;
              for (tries = 0; tries < 4; ++tries) {
                  i32 cand = vdb_nb_(db, g, (i32)(vdb_rng_(&rng) % (u32)tot));
                  if (cand != prevg && db->nodes[cand].kind != VN_DEAD) { ng = cand; break; }
              }
              if (ng < 0) break;
              prevg = g; g = ng; disc *= (f64)o.gamma;
              { f64 rg = disc * RW(g); if (rg > best) best = rg; }
          } }
        rew = best;
        { i32 sl = vs_slot_(&S, leafg);
          if (t[node].visits == 0 && S.vis[sl] == 0) { S.via[sl] = t[node].parent >= 0 ? t[t[node].parent].g : -1; S.dep[sl] = depth; }
          S.vis[sl] += 1; S.val[sl] += rew; }
        for (cur = node; cur >= 0; cur = t[cur].parent) { t[cur].visits += 1; t[cur].wsum += rew; }
    }
    /* rank the notes that were reached: half its own answer, half what the tree found through it */
    ord = (i32 *)malloc((size_t)(S.n + 1) * sizeof(i32));
    { f32 *sc = (f32 *)malloc((size_t)(S.n + 1) * sizeof(f32));
      for (i = 0; i < S.n; ++i) {
          sc[i] = 0.0f;
          if (db->nodes[S.node[i]].kind == VN_NOTE && S.vis[i] > 0) {
              sc[i] = 0.5f * S.rw[i] + 0.5f * (f32)(S.val[i] / S.vis[i]);
              ord[n++] = i;
          }
      }
      for (i = 1; i < n; ++i) {                                /* ord holds slots */
          i32 x = ord[i], j;
          for (j = i; j > 0 && sc[ord[j - 1]] < sc[x]; --j) ord[j] = ord[j - 1];
          ord[j] = x;
      }
      for (it = 0; it < n && it < k; ++it) {
          i32 sl = ord[it];
          out[it].node = S.node[sl]; out[it].sim = vdb_vec_(db, S.node[sl]) ? vdb_dot_(q, vdb_vec_(db, S.node[sl]), db->dim) : 0.0f;
          out[it].lex = S.rw[sl]; out[it].score = sc[sl]; out[it].via = S.via[sl]; out[it].depth = S.dep[sl];
      }
      if (n > k) n = k;
      if (nseed > 0 && seeds[0].score >= 5.0f && n > 0) {         /* the query is a title: that note stays first */
          VHit ex = seeds[0]; i32 j;
          for (j = 0; j < n && out[j].node != ex.node; ++j) {}
          if (j == n && n < k) ++n;
          if (j == n) j = n - 1;
          for (; j > 0; --j) out[j] = out[j - 1];
          ex.score = out[1 < n ? 1 : 0].score + 1.0f;
          out[0] = ex;
      }
      free(sc); }
    free(ord); free(t); free(seeds); free(q); h64_free(&S.map);
    free(S.node); free(S.vis); free(S.via); free(S.dep); free(S.rw); free(S.val);
    return n;
}

/* ---- one query string that mixes words with graph and tag conditions ----
 *   quantum entanglement                 words: hybrid search (vector + BM25)
 *   #physics  -#draft  tag:physics/x     only notes with / without that tag (nested: #a also matches #a/b)
 *   [[Bell tests]]                       notes linked to or from that note
 *   to:[[Bell tests]]                    notes that link to it (its backlinks)
 *   from:[[Bell tests]]                  notes it links to
 * Conditions combine with AND. Words are optional: "#physics to:[[X]]" lists matching notes, most linked first.
 * mcts != 0 runs the MCTS explorer for the words instead of plain hybrid search. */
typedef struct { i32 node, mode, neg; } VCond;   /* mode: 0 neighbour, 1 to, 2 from, 3 tag */

static i32 vdb_cond_ok_(const VecDB *db, i32 c, const VCond *f)
{
    const VNode *v = &db->nodes[c];
    i32 has = 0;
    /* mode 3 (#tag) and mode 1 (to:[[X]]) really do share this exact check, not a copy-paste
     * slip: a tag is stored as an ordinary out-edge to a VN_TAG-kind node, same list a wikilink
     * out-edge to a VN_NOTE-kind node lives in - what distinguishes them is f->node's kind at
     * the call site (vdb_query_'s parser resolves "#tag" via vecdb_find_tag, "to:[[X]]" via
     * vecdb_find), not a different edge list here. Kept as two arms (not merged with ||) so that
     * stays visible instead of looking like one of the two modes was dropped by mistake. */
    if (f->mode == 3) has = vivec_has_(&v->out, f->node);
    else if (f->mode == 1) has = vivec_has_(&v->out, f->node);
    else if (f->mode == 2) has = vivec_has_(&db->nodes[f->node].out, c);
    else has = vivec_has_(&v->out, f->node) || vivec_has_(&db->nodes[f->node].out, c);
    return f->neg ? !has : has;
}

static i32 vdbl_query(VecDB *db, const char *query, i32 k, i32 mcts, VHit *out)
{
    i32 ql = (i32)strlen(query), i = 0, nc = 0, nw = 0, n = 0, j, seedtag = -1, gi = -1;
    VCond *cond = (VCond *)malloc(((size_t)ql / 2 + 2) * sizeof(VCond));
    char *words = (char *)malloc((size_t)ql + 2), *key = (char *)malloc((size_t)ql + 1);   /* a word may end in the separator we add */
    VHit *hits;
    i32 want, ok = 1;

    while (i < ql) {
        char c = query[i];
        i32 neg = 0, mode = -1, st;
        if (c == ' ' || c == '\t') { ++i; continue; }
        st = i;
        if (c == '-' && i + 1 < ql && (query[i + 1] == '#' || query[i + 1] == '[')) { neg = 1; ++i; c = query[i]; }
        if (!strncmp(query + i, "to:", 3) && query[i + 3] == '[') { mode = 1; i += 3; c = query[i]; }
        else if (!strncmp(query + i, "from:", 5) && query[i + 5] == '[') { mode = 2; i += 5; c = query[i]; }
        else if (!strncmp(query + i, "tag:", 4)) { mode = 3; i += 4; c = query[i]; }
        if (c == '[' && i + 1 < ql && query[i + 1] == '[') {
            const char *e = strstr(query + i + 2, "]]");
            if (e) {
                i32 tl = (i32)(e - (query + i + 2)), m = vdb_norm_(query + i + 2, tl, key, 1), id = m > 0 ? hs_get(&db->names, key, m) : -1;
                if (id < 0) ok = 0;                                  /* an unknown note matches nothing */
                else { cond[nc].node = id; cond[nc].mode = mode < 0 ? 0 : mode; cond[nc].neg = neg; ++nc; }
                i = (i32)(e - query) + 2; continue;
            }
        }
        if (c == '#' || mode == 3) {
            i32 t0 = i + (c == '#' ? 1 : 0), t1 = t0, m, id;
            while (t1 < ql && vdb_tagchar_((u8)query[t1])) ++t1;
            m = vdb_norm_(query + t0, t1 - t0, key, 0);
            id = m > 0 ? hs_get(&db->tags, key, m) : -1;
            if (m > 0) { if (id < 0 && !neg) ok = 0; else if (id >= 0) { cond[nc].node = id; cond[nc].mode = 3; cond[nc].neg = neg; ++nc; } }
            i = t1 > t0 ? t1 : i + 1; continue;
        }
        i = st;                                                        /* an ordinary word */
        while (i < ql && query[i] != ' ' && query[i] != '\t') words[nw++] = query[i++];
        words[nw++] = ' ';
    }
    words[nw] = 0;
    free(key);
    if (!ok) { free(cond); free(words); return 0; }
    for (j = 0; j < nc; ++j) if (!cond[j].neg) { if (cond[j].mode == 3 && seedtag < 0) seedtag = cond[j].node; if (cond[j].mode != 3 && gi < 0) gi = j; }

    want = nc ? k * 12 + 40 : k;
    if (want > 250) want = 250;
    hits = (VHit *)malloc((size_t)(want + 1) * sizeof(VHit));
    if (nw > 0) {
        i32 got = mcts ? vdbl_explore(db, words, want, (const VExplore *)0, hits) : vdb_search_(db, words, want, 0, seedtag, hits, 0, (i64 *)0, (f32 *)0, (i32 *)0, (f32 *)0);
        for (j = 0; j < got && n < k; ++j) {
            i32 g, pass = 1;
            for (g = 0; g < nc && pass; ++g) pass = vdb_cond_ok_(db, hits[j].node, &cond[g]);
            if (pass && db->nodes[hits[j].node].kind == VN_NOTE) out[n++] = hits[j];
        }
    }
    if (gi >= 0 && n < k) {                                            /* the graph condition names the candidates: top up from them */
        const VCond *g0 = &cond[gi];
        const VNode *x = &db->nodes[g0->node];
        i32 total = (g0->mode == 1) ? x->in.n : (g0->mode == 2 ? x->out.n : x->in.n + x->out.n), t;
        f32 *qv = (nw > 0) ? (f32 *)malloc((size_t)db->dim * sizeof(f32)) : (f32 *)0;
        if (qv) vdb_embed_(db, words, nw, 1, qv);
        for (t = 0; t < total; ++t) {
            i32 c = g0->mode == 1 ? x->in.v[t] : (g0->mode == 2 ? x->out.v[t] : vdb_nb_(db, g0->node, t)), g, pass = 1, dup = 0;
            const f32 *cv;
            VHit h;
            if (db->nodes[c].kind != VN_NOTE) continue;
            for (g = 0; g < nc && pass; ++g) pass = vdb_cond_ok_(db, c, &cond[g]);
            if (!pass) continue;
            for (g = 0; g < n; ++g) if (out[g].node == c) { dup = 1; break; }
            if (dup) continue;
            cv = vdb_vec_(db, c);
            h.node = c; h.via = -1; h.depth = 0; h.lex = 0.0f;
            h.sim = (qv && cv) ? vdb_dot_(qv, cv, db->dim) : 0.0f;
            h.score = nw > 0 ? h.sim : (f32)db->nodes[c].in.n;        /* no words: the most linked first */
            if (n < k) out[n++] = h;                                    /* keep the k best of the candidates */
            else { i32 w = 0; for (g = 1; g < n; ++g) if (out[g].score < out[w].score) w = g; if (h.score > out[w].score) out[w] = h; }
        }
        free(qv);
        for (j = 1; j < n; ++j) { VHit x2 = out[j]; i32 z; for (z = j; z > 0 && out[z - 1].score < x2.score; --z) out[z] = out[z - 1]; out[z] = x2; }
    }
    if (nw == 0 && gi < 0 && nc > 0 && seedtag >= 0) {                 /* tags only: the notes carrying them, most linked first */
        const VNode *tn = &db->nodes[seedtag];
        for (j = 0; j < tn->in.n; ++j) {
            i32 c = tn->in.v[j], g, pass = 1;
            VHit h;
            if (db->nodes[c].kind != VN_NOTE) continue;
            for (g = 0; g < nc && pass; ++g) pass = vdb_cond_ok_(db, c, &cond[g]);
            if (!pass) continue;
            h.node = c; h.sim = 0.0f; h.lex = 0.0f; h.score = (f32)db->nodes[c].in.n; h.via = -1; h.depth = 0;
            if (n < k) out[n++] = h;
            else { i32 w = 0, z; for (z = 1; z < n; ++z) if (out[z].score < out[w].score) w = z; if (h.score > out[w].score) out[w] = h; }
        }
        for (j = 1; j < n; ++j) { VHit x2 = out[j]; i32 z; for (z = j; z > 0 && out[z - 1].score < x2.score; --z) out[z] = out[z - 1]; out[z] = x2; }
    }
    free(hits); free(cond); free(words);
    return n;
}

/* ------------------------------------------------------------------ */
/* save / load: PATH.vdb  nodes (kind, title, where the body sits)      */
/*              PATH.vdt  the note bodies (this becomes the live text   */
/*                        file: bodies stay on disk, never held in RAM) */
/*              PATH.vxi  the vector index (rows, meta and the graph)   */
/* Links, tags and the lexicon are rebuilt from the bodies on load       */
/* (~8 us per note); if the .vxi is missing the vectors are recomputed   */
/* with the embedder you pass to vecdb_load_with.                        */
/* ------------------------------------------------------------------ */

#define VDB_MAGIC_V3 0x33424456u  /* 'VDB3': the first release, header = magic dim nn dim2 */
#define VDB_MAGIC 0x58424456u     /* 'VDBX': header = magic version dim nn dim2 flags */
#define VDB_VERSION 5u            /* what this build writes; it reads 3, 4 (legacy) and 5. A higher number is a file from a newer build. */

static char vdb_err_[400] = "";
/* why the last vecdb_load failed ("" if it did not) */
const char *vecdb_error(void) { return vdb_err_; }

static i32 vdb_wi_(FILE *f, i32 x) { return fwrite(&x, 4, 1, f) == 1 ? 0 : -1; }
static i32 vdb_ri_(FILE *f, i32 *x) { return fread(x, 4, 1, f) == 1 ? 0 : -1; }

static char *vdb_ext_(const char *path, const char *ext)
{
    char *p = (char *)malloc(strlen(path) + strlen(ext) + 1);
    strcpy(p, path); strcat(p, ext);
    return p;
}

/* PATH+ext, or PATH+ext+".gXXXXXXXXXXXXXXXX" when gen != 0 - see VecDB.bodygen's comment */
static char *vdb_ext_gen_(const char *path, const char *ext, u64 gen)
{
    char *p;                                                      /* C89 has no 64-bit printf length modifier: format as two u32 halves */
    if (gen == 0) return vdb_ext_(path, ext);
    p = (char *)malloc(strlen(path) + strlen(ext) + 20);
    sprintf(p, "%s%s.g%08lx%08lx", path, ext, (unsigned long)(gen >> 32), (unsigned long)(gen & 0xffffffffu));
    return p;
}

static i32 vdb_write_nodes_(const VecDB *db, const char *path)
{
    char *tmp = vdb_ext_(path, ".tmp");
    FILE *f = fopen(tmp, "wb");                                  /* write beside, fsync, rename: the old table is valid until the new one is complete */
    i32 i;
    if (!f) { free(tmp); return -1; }
    vdb_wi_(f, (i32)VDB_MAGIC); vdb_wi_(f, (i32)VDB_VERSION); vdb_wi_(f, db->dim); vdb_wi_(f, db->nn); vdb_wi_(f, db->store2 ? db->dim2 : 0); vdb_wi_(f, 0);
    fwrite(&db->bodygen, sizeof(u64), 1, f);
    for (i = 0; i < db->nn; ++i) {
        const VNode *v = &db->nodes[i];
        i32 nl = (i32)strlen(v->name);
        vdb_wi_(f, v->kind); vdb_wi_(f, v->space); vdb_wi_(f, nl); fwrite(v->name, 1, (size_t)nl, f);
        fwrite(&v->toff, sizeof(i64), 1, f); vdb_wi_(f, v->tlen);
    }
    if (fflush(f) != 0 || fsync(fileno(f)) != 0 || fclose(f) != 0 || rename(tmp, path) != 0) { remove(tmp); free(tmp); return -1; }
    free(tmp);
    store_fsync_dir_(path);
    return 0;
}

/* True cross-file atomicity: PATH.vdt and PATH.vxi are tightly coupled (byte offsets, row<->id order) and a
 * full rewrite of one without the other mid-crash would let an old node table silently misread a new body file
 * (or vice versa) - two independently-atomic renames are not atomic AS A PAIR. So a full rewrite that would have
 * to replace an ALREADY-EXISTING PATH.vdt/.vxi (re-indexing an existing database from scratch: vecdb_new + save,
 * not vecdb_load) never touches those files at all: it writes the new set under a fresh generation suffix
 * (PATH.vdt.g<hex>, PATH.vxi.g<hex>, in lockstep - db->bodygen), and only the LAST step, the node table (the
 * real commit record, itself atomic), starts pointing at that generation. A crash before it leaves the old
 * generation's files completely untouched (loads as the old database); a crash after leaves the new generation's
 * files complete and fsynced (loads as the new one). Either way, never a mismatched pairing. The one-time cost:
 * an old, now-orphaned generation's files, cleaned up best-effort right after the commit succeeds.
 * A brand-new path (nothing there yet) skips all of this and just uses the bare names - cheap common case. */
static i32 vdbl_save(VecDB *db, const char *path)
{
    i32 i, r, need_full = !db->dbpath || strcmp(db->dbpath, path) != 0;
    char *vp, *old_vdt = (char *)0, *old_vxi = (char *)0, *old_ivxi = (char *)0, *old_vxij = (char *)0;

    if (need_full) {                                              /* attaching to `path` for the first time in this object's life */
        char *barename = vdb_ext_(path, ".vdt");
        struct stat sb;
        u64 gen = 0;
        char *newname, *tmp;
        FILE *nf;
        i64 pos = 0;

        if (stat(barename, &sb) == 0) {                           /* something already lives under the canonical name */
            u64 h = vdb_hash64_((const char *)&sb, sizeof(sb), (u64)time((time_t *)0) ^ (u64)(size_t)db ^ (u64)getpid());
            gen = h ? h : 1;                                       /* 0 is reserved to mean "bare name" */
            old_vdt = barename;                                    /* remember it: delete it once the new generation is safely committed */
            old_vxi = vdb_ext_(path, ".vxi");
            old_ivxi = vdb_ext_(path, ".img.vxi");
            old_vxij = vdb_ext_(path, ".vxi.j");
        } else free(barename);

        newname = vdb_ext_gen_(path, ".vdt", gen);
        tmp = vdb_ext_(newname, ".tmp");
        nf = fopen(tmp, "w+b");
        if (!nf) { free(tmp); free(newname); free(old_vdt); free(old_vxi); free(old_ivxi); free(old_vxij); return -1; }
        for (i = 0; i < db->nn; ++i) {
            char *t;
            if (db->nodes[i].kind != VN_NOTE) continue;
            t = vdbl_text(db, i);
            if (!t) { db->nodes[i].tlen = -1; continue; }
            if (db->nodes[i].tlen > 0) fwrite(t, 1, (size_t)db->nodes[i].tlen, nf);
            db->nodes[i].toff = pos; pos += db->nodes[i].tlen;
            free(t);
        }
        if (store_fsync_(nf) != 0 || rename(tmp, newname) != 0) {
            fclose(nf); remove(tmp); free(tmp); free(newname); free(old_vdt); free(old_vxi); free(old_ivxi); free(old_vxij);
            return -1;
        }
        store_fsync_dir_(newname);
        free(tmp);
        if (db->tf) fclose(db->tf);                                /* nf keeps working after the rename: same fd, same inode, new name */
        db->tf = nf; free(db->tpath); db->tpath = newname; db->tend = pos;
        db->bodygen = gen;
        free(db->dbpath); db->dbpath = (char *)malloc(strlen(path) + 1); strcpy(db->dbpath, path);
    }
    if (db->tf) store_fsync_(db->tf);

    /* commit order: bodies (already fsynced above), then the vector files (own atomic tmp+rename), then the
     * node table LAST - it is the commit record. A crash before it leaves the previous table valid; load then
     * sees the counts disagree and re-embeds (the embedding cache makes that cheap). */
    vdbl_build(db);
    vp = vdb_ext_gen_(path, ".vxi", db->bodygen);
    r = store_save(db->store, vp);
    if (r == 0 && db->store2) { char *ip = vdb_ext_gen_(path, ".img.vxi", db->bodygen); r = store_save(db->store2, ip); free(ip); }
    free(vp);
    if (r == 0) r = vdb_write_nodes_(db, path);
    if (r == 0 && old_vdt) { remove(old_vdt); remove(old_vxi); remove(old_ivxi); remove(old_vxij); }   /* best-effort: an orphan left behind is harmless */
    free(old_vdt); free(old_vxi); free(old_ivxi); free(old_vxij);
    return r == 0 ? 0 : -1;
}

/* Durable save that costs what changed, not what exists: bodies are already appended in place, the small node
 * table is rewritten, and the vector stores only flush their journals (see store_sync). The first call for a
 * path (or a path this object was not already attached to) does a full vecdb_save. */
static i32 vdbl_sync(VecDB *db, const char *path)
{
    char *vp;
    i32 r, warm = db->dbpath && strcmp(db->dbpath, path) == 0;
    if (warm) { vp = vdb_ext_gen_(path, ".vxi", db->bodygen); warm = db->store->jpath && strcmp(db->store->jpath, vp) == 0; if (!warm) free(vp); }
    if (!warm) return vdbl_save(db, path);
    r = db->tf ? store_fsync_(db->tf) : 0;                       /* bodies, then journals, then the node table (the commit record) */
    if (r == 0) r = store_sync(db->store, vp);
    if (r == 0 && db->store2) { char *ip = vdb_ext_gen_(path, ".img.vxi", db->bodygen); r = store_sync(db->store2, ip); free(ip); }
    free(vp);
    if (r != 0) return -1;
    return vdb_write_nodes_(db, path) == 0 ? 0 : -1;
}

VecDB *vecdb_load_with(Arena *a, const char *path, void (*embed)(void *, const char *, i32, i32, f32 *, i32), void *ctx)
{
    FILE *f = fopen(path, "rb");
    VecDB *db = (VecDB *)0;
    i32 magic, dim = 0, nn = 0, i, nnotes = 0, nimg = 0, use, use2 = 1, dim2 = 0;
    char **names = (char **)0, *tp, *vp;
    u8 *kinds = (u8 *)0, *spaces = (u8 *)0;
    i64 *toff = (i64 *)0; i32 *tlen = (i32 *)0;
    RagStore *st;
    long tsize;
    u64 bodygen = 0;

    vdb_err_[0] = 0;
    if (!f) { sprintf(vdb_err_, "cannot open %.100s", path); return db; }
    if (vdb_ri_(f, &magic)) { sprintf(vdb_err_, "%.100s is empty or truncated", path); fclose(f); return db; }
    if ((u32)magic == VDB_MAGIC) {
        i32 ver, flags;
        if (vdb_ri_(f, &ver) || vdb_ri_(f, &dim) || vdb_ri_(f, &nn) || vdb_ri_(f, &dim2) || vdb_ri_(f, &flags)) { sprintf(vdb_err_, "%.100s: header truncated", path); fclose(f); return db; }
        if ((u32)ver > VDB_VERSION) { sprintf(vdb_err_, "%.100s was written by a newer vecdb (format %d, this build reads up to %u)", path, ver, VDB_VERSION); fclose(f); return db; }
        if (ver >= 5 && fread(&bodygen, sizeof(u64), 1, f) != 1) { sprintf(vdb_err_, "%.100s: header truncated", path); fclose(f); return db; }
    } else if ((u32)magic == VDB_MAGIC_V3) {                     /* legacy: same node table, no version/flags/bodygen in the header */
        if (vdb_ri_(f, &dim) || vdb_ri_(f, &nn) || vdb_ri_(f, &dim2)) { sprintf(vdb_err_, "%.100s: header truncated", path); fclose(f); return db; }
    } else { sprintf(vdb_err_, "%.100s is not a vecdb file", path); fclose(f); return db; }
    if (dim <= 0 || nn < 0 || dim2 < 0) { sprintf(vdb_err_, "%.100s: corrupt header", path); fclose(f); return db; }
    names = (char **)calloc((size_t)nn + 1, sizeof(char *)); kinds = (u8 *)calloc((size_t)nn + 1, 1); spaces = (u8 *)calloc((size_t)nn + 1, 1);
    toff = (i64 *)calloc((size_t)nn + 1, sizeof(i64)); tlen = (i32 *)calloc((size_t)nn + 1, sizeof(i32));
    for (i = 0; i < nn; ++i) {
        i32 kind, space, nl;
        if (vdb_ri_(f, &kind) || vdb_ri_(f, &space) || vdb_ri_(f, &nl) || nl < 0) goto bad;
        names[i] = (char *)malloc((size_t)nl + 1);
        if (fread(names[i], 1, (size_t)nl, f) != (size_t)nl) goto bad;
        names[i][nl] = 0; kinds[i] = (u8)kind; spaces[i] = (u8)space;
        if (fread(&toff[i], sizeof(i64), 1, f) != 1 || vdb_ri_(f, &tlen[i])) goto bad;
    }
    fclose(f); f = (FILE *)0;

    db = vecdb_new(a, dim);
    db->embed = embed; db->embed_ctx = ctx;
    db->bodygen = bodygen;
    db->dbpath = (char *)malloc(strlen(path) + 1); strcpy(db->dbpath, path);
    tp = vdb_ext_gen_(path, ".vdt", bodygen); vp = vdb_ext_gen_(path, ".vxi", bodygen);
    if (db->tf) fclose(db->tf);
    db->tf = fopen(tp, "r+b"); db->tpath = tp;
    if (db->tf) { fseek(db->tf, 0, SEEK_END); tsize = ftell(db->tf); db->tend = tsize; }

    /* recreate the nodes in their original order so ids (= store doc ids) stay valid */
    for (i = 0; i < nn; ++i) {
        char *key = (char *)malloc(strlen(names[i]) + 1);
        i32 kl;
        if (kinds[i] == VN_TAG) { kl = vdb_norm_(names[i], (i32)strlen(names[i]), key, 0); vdb_tag_node_(db, key, kl); }
        else if (kinds[i] == VN_DEAD) { vdb_newnode_(db, VN_DEAD, names[i], (i32)strlen(names[i])); }
        else { kl = vdb_norm_(names[i], (i32)strlen(names[i]), key, 1); vdb_title_node_(db, key, kl, 1); }
        free(key);
        if (kinds[i] == VN_NOTE) { db->nodes[i].toff = toff[i]; db->nodes[i].tlen = tlen[i]; db->nodes[i].space = spaces[i]; if (spaces[i]) ++nimg; else ++nnotes; }
    }
    st = store_load(a, vp);
    use = st && store_count(st) == nnotes;
    if (use) { store_destroy(db->store); db->store = st; store_set_fast(st, 0.97f); }
    else if (st) store_destroy(st);
    if (dim2 > 0) {                                              /* the image space (needs its model to be searched, not to be loaded) */
        char *ip = vdb_ext_gen_(path, ".img.vxi", bodygen);
        RagStore *s2 = store_load(a, ip);
        db->dim2 = dim2;
        use2 = s2 && store_count(s2) == nimg;
        if (use2) { db->store2 = s2; store_set_fast(s2, 0.97f); }
        else { if (s2) store_destroy(s2); db->store2 = store_new(a, dim2); store_set_fast(db->store2, 0.97f); }
        free(ip);
    }
    for (i = 0; i < nn; ++i) {
        char *t;
        if (kinds[i] != VN_NOTE) continue;
        t = vdbl_text(db, i);
        if (!t) continue;
        if (spaces[i]) vdb_add_(db, names[i], t, (const f32 *)0, 0, 0, 1);   /* image vectors cannot be recomputed from text: an image index without its .img.vxi is lost */
        else if (use) vdb_add_(db, names[i], t, (const f32 *)0, 0, 0, 0);
        else { f32 *vec = (f32 *)malloc((size_t)dim * sizeof(f32)); vdb_embed_doc_(db, names[i], t, vec); vdb_add_(db, names[i], t, vec, 1, 0, 0); free(vec); }
        free(t);
    }
    (void)use2;
    free(vp);
    for (i = 0; i < nn; ++i) free(names[i]);
    free(names); free(kinds); free(spaces); free(toff); free(tlen);
    return db;
bad:
    if (f) fclose(f);
    for (i = 0; i < nn; ++i) free(names[i]);
    free(names); free(kinds); free(spaces); free(toff); free(tlen);
    return (VecDB *)0;
}

VecDB *vecdb_load(Arena *a, const char *path) { return vecdb_load_with(a, path, (void (*)(void *, const char *, i32, i32, f32 *, i32))0, (void *)0); }

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

/* ------------------------------------------------------------------ */
/* self-test                                                           */
/* ------------------------------------------------------------------ */

#ifdef VECDB_DEMO
#include <stdio.h>

static i32 vd_check_(const char *what, i32 cond) { printf("  %-58s %s\n", what, cond ? "ok" : "FAIL"); return cond; }

static i32 vd_has_(const i32 *a, i32 n, i32 x) { i32 i; for (i = 0; i < n; ++i) if (a[i] == x) return 1; return 0; }

static i32 vd_model_calls_ = 0;
/* a stand-in "model": counts how often it is asked, otherwise the hashed embedder */
static void vd_fake_model_(void *ctx, const char *t, i32 n, i32 kind, f32 *out, i32 dim) { (void)ctx; (void)kind; ++vd_model_calls_; vecdb_embed(t, n, out, dim); }
static void vd_fake_batch_(void *ctx, i32 n, const char **t, const i32 *l, i32 kind, f32 *out, i32 dim) { i32 i; (void)ctx; (void)kind; for (i = 0; i < n; ++i) { ++vd_model_calls_; vecdb_embed(t[i], l[i], out + (size_t)i * dim, dim); } }

int main(void)
{
    static Arena arena;
    VecDB *db;
    i32 ok = 1, i, n, buf[64];
    i32 a, b, c, ghost, bridge, quantum1;
    VHit hits[32];

    /* ---- parsing ---- */
    { VStrs l, t;
      const char *src = "---\ntags: [alpha, Beta/Gamma]\n---\n# Heading not a tag\n"
                        "See [[Note One]] and [[note two|alias]], [[Folder/Deep Note#part]] and ![[Embedded.md]] and ![shot](img/Pic.png) and ![web](http://x.org/a.png).\n"
                        "Tags: #idea #project/alpha, not #123, not a#nottag, `#incode`, and http://x.org/#frag.\n"
                        "```\n[[in fence]] #infence\n```\nend #last-tag\n";
      memset(&l, 0, sizeof l); memset(&t, 0, sizeof t);
      vdb_parse_(src, (i32)strlen(src), &l, &t);
      printf("parsing\n");
      ok &= vd_check_("5 links (alias, heading, folder, embed, markdown picture; web url skipped)", l.count == 5);
      ok &= vd_check_("link targets normalised", l.count == 5 && !strcmp(l.buf + l.off[4], "pic.png") && !strcmp(l.buf + l.off[0], "note one") && !strcmp(l.buf + l.off[1], "note two")
                      && !strcmp(l.buf + l.off[2], "deep note") && !strcmp(l.buf + l.off[3], "embedded"));
      ok &= vd_check_("front matter + inline tags, nested tag adds its parent", t.count == 7);
      { i32 has_last = 0, has_num = 0, has_frag = 0, has_code = 0, has_infence = 0, has_parent = 0;
        for (i = 0; i < t.count; ++i) {
            const char *s = t.buf + t.off[i];
            if (!strcmp(s, "last-tag")) has_last = 1;
            if (!strcmp(s, "123")) has_num = 1;
            if (!strcmp(s, "frag")) has_frag = 1;
            if (!strcmp(s, "incode")) has_code = 1;
            if (!strcmp(s, "infence")) has_infence = 1;
            if (!strcmp(s, "project")) has_parent = 1;
        }
        ok &= vd_check_("#last-tag found; #123, #frag, #incode, #infence not", has_last && !has_num && !has_frag && !has_code && !has_infence && has_parent); }
      vstrs_free_(&l); vstrs_free_(&t); }

    /* ---- graph ---- */
    db = vecdb_new(&arena, 128);
    a = vecdb_add_note(db, "Alpha", "Alpha links to [[Beta]] and [[Missing Note]]. #topic/x #shared");
    b = vecdb_add_note(db, "Beta", "Beta links back to [[alpha]]. #shared");
    c = vecdb_add_note(db, "Gamma", "Gamma is alone. #topic/y");
    printf("graph\n");
    ghost = vecdb_find(db, "Missing Note");
    ok &= vd_check_("ghost created for an unresolved link", ghost >= 0 && db->nodes[ghost].kind == VN_GHOST);
    n = vecdb_backlinks(db, a, buf, 64);
    ok &= vd_check_("backlinks of Alpha = {Beta}", n == 1 && buf[0] == b);
    n = vecdb_links(db, a, buf, 64);
    ok &= vd_check_("links of Alpha = {Beta, ghost}", n == 2 && vd_has_(buf, n, b) && vd_has_(buf, n, ghost));
    { i32 tg = vecdb_find_tag(db, "#topic"), tsh = vecdb_find_tag(db, "shared");
      n = vecdb_notes_with_tag(db, tg, buf, 64);
      ok &= vd_check_("#topic holds notes tagged #topic/x and #topic/y", n == 2 && vd_has_(buf, n, a) && vd_has_(buf, n, c));
      n = vecdb_notes_with_tag(db, tsh, buf, 64);
      ok &= vd_check_("#shared holds Alpha and Beta", n == 2 && vd_has_(buf, n, a) && vd_has_(buf, n, b)); }
    vecdb_add_note(db, "Missing Note", "Now it exists.");
    ok &= vd_check_("the ghost becomes a note, backlinks kept", db->nodes[ghost].kind == VN_NOTE && vecdb_backlinks(db, ghost, buf, 64) == 1);
    vecdb_add_note(db, "Alpha", "Rewritten: only [[Gamma]] now.");
    n = vecdb_links(db, a, buf, 64);
    ok &= vd_check_("re-adding replaces the edges", n == 1 && buf[0] == c && vecdb_backlinks(db, b, buf, 64) == 0 && vecdb_backlinks(db, c, buf, 64) == 1);
    vecdb_remove_note(db, c);
    ok &= vd_check_("removing a linked note leaves a ghost", db->nodes[c].kind == VN_GHOST);

    /* ---- MCTS finds what nearest-neighbour cannot ---- */
    { VecDB *d2 = vecdb_new(&arena, 256);
      char title[64], body[512];
      i32 nfill = 40, hit_knn = 0, hit_mcts = 0, rank = -1;
      printf("hybrid search and MCTS (planted bridge note)\n");
      for (i = 0; i < nfill; ++i) {              /* unrelated filler notes, each about its own words */
          sprintf(title, "Filler %d", i);
          sprintf(body, "kitchen recipe number %d uses flour%d sugar%d butter%d and eggs%d for baking cake%d", i, i, i, i, i, i);
          vecdb_add_note(d2, title, body);
      }
      bridge = vecdb_add_note(d2, "Bell inequality tests", "Loophole free experiments in Delft and Vienna. Cited by several of the notes here.");
      for (i = 0; i < 6; ++i) {                  /* six notes that are clearly about the query and link to the bridge */
          sprintf(title, "Quantum %d", i);
          sprintf(body, "quantum entanglement of photon pairs, entanglement swapping and quantum teleportation, part %d. See [[Bell inequality tests]]. #physics", i);
          quantum1 = vecdb_add_note(d2, title, body);
      }
      (void)quantum1;
      vecdb_build(d2);
      n = vecdb_search(d2, "quantum entanglement", 6, 1, -1, hits);
      for (i = 0; i < n; ++i) if (hits[i].node == bridge) hit_knn = 1;
      { VExplore o; vecdb_explore_defaults(&o); o.seeds = 6; o.iters = 800;
        n = vecdb_explore(d2, "quantum entanglement", 8, &o, hits);
        for (i = 0; i < n; ++i) if (hits[i].node == bridge) { hit_mcts = 1; rank = i + 1; } }
      ok &= vd_check_("vector kNN top-6 does not contain the bridge note", !hit_knn);
      printf("  (MCTS top-8 contains it at rank %d)\n", rank);
      ok &= vd_check_("MCTS top-8 does contain the bridge note", hit_mcts);
      n = vecdb_search(d2, "quantum entanglement", 8, 0, vecdb_find_tag(d2, "physics"), hits);
      ok &= vd_check_("tag filter #physics keeps only the six quantum notes", n == 6);
      n = vecdb_search(d2, "Delft", 3, 2, -1, hits);
      ok &= vd_check_("lexical: the rare word 'Delft' finds the bridge note first", n >= 1 && hits[0].node == bridge);

      n = vecdb_search(d2, "entangl*", 3, 2, -1, hits);
      ok &= vd_check_("lexical prefix 'entangl*' finds entanglement notes", n == 3);
      n = vecdb_search(d2, "teleportations photons", 3, 2, -1, hits);
      ok &= vd_check_("stemming: plural/other forms still match", n == 3);
      n = vecdb_search(d2, "flour", 5, 0, vecdb_find_tag(d2, "physics"), hits);
      ok &= vd_check_("tag filter is applied inside the vector search", n <= 6);

      { i32 in2 = 0;
        vecdb_smooth(d2, 1.0f); vecdb_build(d2);
        n = vecdb_search(d2, "quantum entanglement", 8, 1, -1, hits);
        for (i = 0; i < n; ++i) if (hits[i].node == bridge) in2 = 1;
        ok &= vd_check_("graph-smoothed embeddings: plain kNN now reaches the bridge note", in2); }

      { VecDB *da = vecdb_new(&arena, 256), *dbb = vecdb_new(&arena, 256);
        const char *tt[200], *xx[200]; char (*tb)[32] = (char (*)[32])malloc(200 * 32), (*xb)[64] = (char (*)[64])malloc(200 * 64);
        i32 same = 1, z;
        for (z = 0; z < 200; ++z) { sprintf(tb[z], "Batch %d", z); sprintf(xb[z], "note %d about topic%d and word%d", z, z % 7, z % 13); tt[z] = tb[z]; xx[z] = xb[z]; }
        for (z = 0; z < 200; ++z) vecdb_add_note(da, tt[z], xx[z]);
        ok &= vd_check_("vecdb_add_notes adds all 200 (threaded embedder)", vecdb_add_notes(dbb, 200, tt, xx) == 200);
        vecdb_build(da); vecdb_build(dbb);
        for (z = 0; z < 200; z += 9) { VHit h1[3], h2[3]; i32 a1 = vecdb_search(da, xx[z], 3, 1, -1, h1), a2 = vecdb_search(dbb, xx[z], 3, 1, -1, h2); if (a1 != a2 || (a1 > 0 && h1[0].node != h2[0].node)) same = 0; }
        ok &= vd_check_("same results as adding one by one", same);
        free(tb); free(xb); }
      printf("embedding cache\n");
      { VecDB *c1 = vecdb_new(&arena, 256), *c2 = vecdb_new(&arena, 256), *c3 = vecdb_new(&arena, 256);
        const char *tt[3] = { "Cache A", "Cache B", "Cache C" }, *xx[3] = { "first text", "second text", "third text" };
        i32 first, again;
        remove("/tmp/vecdb_test.emb");
        c1->embed = vd_fake_model_; c1->embed_batch = vd_fake_batch_;
        c2->embed = vd_fake_model_; c2->embed_batch = vd_fake_batch_;
        c3->embed = vd_fake_model_; c3->embed_batch = vd_fake_batch_;
        vecdb_embed_cache_open(c1, "/tmp/vecdb_test.emb", "modelA");
        vd_model_calls_ = 0; vecdb_add_notes(c1, 3, tt, xx); first = vd_model_calls_;
        vecdb_embed_cache_open(c2, "/tmp/vecdb_test.emb", "modelA");
        vd_model_calls_ = 0; vecdb_add_notes(c2, 3, tt, xx); vecdb_add_note(c2, "Cache A", "first text"); again = vd_model_calls_;
        ok &= vd_check_("first indexing asks the model 3 times, the cache holds 3", first == 3 && vecdb_embed_cache_count(c2) == 3);
        ok &= vd_check_("re-indexing the same texts (batch and single) asks it 0 times", again == 0);
        vecdb_embed_cache_open(c3, "/tmp/vecdb_test.emb", "modelB");
        vd_model_calls_ = 0; vecdb_add_notes(c3, 3, tt, xx);
        ok &= vd_check_("another model tag does not reuse the entries", vd_model_calls_ == 3);
        remove("/tmp/vecdb_test.emb"); }
      printf("query language\n");
      { i32 b6 = vecdb_find(d2, "Quantum 3");
        n = vecdb_query(d2, "to:[[Bell inequality tests]] #physics", 10, 0, hits);
        ok &= vd_check_("to:[[X]] #tag lists the six notes that link to X", n == 6);
        n = vecdb_query(d2, "entanglement from:[[Quantum 3]]", 10, 0, hits);
        ok &= vd_check_("from:[[Quantum 3]] + words returns its link target", n == 1 && hits[0].node == bridge);
        n = vecdb_query(d2, "[[Bell inequality tests]] -#physics", 10, 0, hits);
        ok &= vd_check_("[[X]] -#tag: neighbours without the tag -> none", n == 0);
        n = vecdb_query(d2, "quantum entanglement #physics", 3, 1, hits);
        ok &= vd_check_("words + #tag through MCTS", n == 3 && vivec_has_(&d2->nodes[hits[0].node].out, vecdb_find_tag(d2, "physics")));
        n = vecdb_query(d2, "#nosuchtag entanglement", 5, 0, hits);
        ok &= vd_check_("unknown tag matches nothing", n == 0);
        n = vecdb_query(d2, "#physics", 4, 0, hits);
        ok &= vd_check_("tag only: notes carrying it", n == 4);
        (void)b6; }

      /* ---- save / load ---- */
      vecdb_save(d2, "/tmp/vecdb_selftest.vdb");
      { VecDB *d3 = vecdb_load(&arena, "/tmp/vecdb_selftest.vdb");
        VHit h2[32]; i32 m, same = 1;
        ok &= vd_check_("save/load: file reopens", d3 != (VecDB *)0);
        if (d3) {
            vecdb_build(d3);
            n = vecdb_search(d2, "quantum entanglement", 8, 0, -1, hits);
            m = vecdb_search(d3, "quantum entanglement", 8, 0, -1, h2);
            for (i = 0; i < n && i < m; ++i) { i32 j, f = 0; for (j = 0; j < m; ++j) if (h2[j].node == hits[i].node) f = 1; if (!f) same = 0; }   /* near-duplicate notes tie, so compare as sets */
            ok &= vd_check_("save/load: same node ids and same hybrid result set", n == m && same);
            ok &= vd_check_("save/load: backlinks survived", vecdb_backlinks(d3, bridge, buf, 64) == 6);
        } }
      { VecDB *d4; i32 more;
        vecdb_add_note(d2, "Added after save", "Fresh note about photon entanglement. [[Bell inequality tests]]");
        vecdb_remove_note(d2, 0);
        ok &= vd_check_("sync after a save (journal, no full rewrite)", vecdb_sync(d2, "/tmp/vecdb_selftest.vdb") == 0);
        d4 = vecdb_load(&arena, "/tmp/vecdb_selftest.vdb");
        more = d4 ? vecdb_find(d4, "Added after save") : -1;
        ok &= vd_check_("sync: the new note and the removal survive a reload", d4 && more >= 0 && d4->nodes[more].kind == VN_NOTE && d4->nodes[0].kind != VN_NOTE);
        if (d4) { vecdb_build(d4); n = vecdb_search(d4, "photon entanglement", 3, 0, -1, hits); ok &= vd_check_("sync: it is searchable after reload", n > 0); }
      }
      { /* format versions: a legacy (v3) header still loads, a newer format and garbage are refused with a reason */
        FILE *f = fopen("/tmp/vecdb_selftest.vdb", "rb");
        long sz; char *img; VecDB *lg, *fut, *bad;
        vecdb_save(d2, "/tmp/vecdb_selftest.vdb");
        f = fopen("/tmp/vecdb_selftest.vdb", "rb"); fseek(f, 0, SEEK_END); sz = ftell(f); fseek(f, 0, SEEK_SET);
        img = (char *)malloc((size_t)sz); if (fread(img, 1, (size_t)sz, f) != (size_t)sz) ok = 0; fclose(f);
        f = fopen("/tmp/vecdb_selftest.v3", "wb");                      /* v3 = 'VDB3' dim nn dim2 + the same node table */
        { u32 m3 = VDB_MAGIC_V3; fwrite(&m3, 4, 1, f); fwrite(img + 8, 1, 12, f); fwrite(img + 32, 1, (size_t)sz - 32, f); }
        fclose(f);
        { char *cp; i32 nv = 99; f = fopen("/tmp/vecdb_selftest.v99", "wb"); fwrite(img, 1, (size_t)sz, f); fclose(f);
          f = fopen("/tmp/vecdb_selftest.v99", "r+b"); fseek(f, 4, SEEK_SET); fwrite(&nv, 4, 1, f); fclose(f); (void)cp; }
        f = fopen("/tmp/vecdb_selftest.bad", "wb"); fputs("this is not a database at all", f); fclose(f);
        { char cmd[200]; sprintf(cmd, "cp /tmp/vecdb_selftest.vdb.vdt /tmp/vecdb_selftest.v3.vdt; cp /tmp/vecdb_selftest.vdb.vxi /tmp/vecdb_selftest.v3.vxi"); if (system(cmd) != 0) ok = 0; }
        lg = vecdb_load(&arena, "/tmp/vecdb_selftest.v3");
        ok &= vd_check_("format: a legacy v3 file still loads", lg && vecdb_count(lg, VN_NOTE) == vecdb_count(d2, VN_NOTE));
        fut = vecdb_load(&arena, "/tmp/vecdb_selftest.v99");
        ok &= vd_check_("format: a file from a newer build is refused, with a reason", !fut && strstr(vecdb_error(), "newer"));
        bad = vecdb_load(&arena, "/tmp/vecdb_selftest.bad");
        ok &= vd_check_("format: garbage is refused, with a reason", !bad && strstr(vecdb_error(), "not a vecdb file"));
        ok &= vd_check_("format: a missing file is refused, with a reason", !vecdb_load(&arena, "/tmp/vecdb_nonexistent.vdb") && strstr(vecdb_error(), "cannot open"));
        free(img);
        remove("/tmp/vecdb_selftest.v3"); remove("/tmp/vecdb_selftest.v3.vdt"); remove("/tmp/vecdb_selftest.v3.vxi"); remove("/tmp/vecdb_selftest.v99"); remove("/tmp/vecdb_selftest.bad"); }
      remove("/tmp/vecdb_selftest.vdb"); remove("/tmp/vecdb_selftest.vdb.vdt"); remove("/tmp/vecdb_selftest.vdb.vxi"); remove("/tmp/vecdb_selftest.vdb.vxi.j"); }

    puts(ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
#endif

/* ------------------------------------------------------------------ */
/* command line:  vecdb DB.vdb index DIR | search Q | explore Q |      */
/*                links T | backlinks T | tag TAG | tags | ghosts     */
/* ------------------------------------------------------------------ */

#ifdef VECDB_CLI
#include <stdio.h>

static char *cli_slurp_(const char *path)
{
    FILE *f = fopen(path, "rb");
    long n; char *b;
    if (!f) return (char *)0;
    fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET);
    b = (char *)malloc((size_t)n + 1);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return (char *)0; }
    b[n] = 0; fclose(f);
    return b;
}

static const char *cli_extract_script_ = "extract.py";

/* text of a file: read as is for text formats, through extract.py for pdf/docx/... */
static char *cli_text_(const char *path, const char *name)
{
    size_t l = strlen(name);
    static const char *plain[] = { ".md", ".txt", ".org", ".rst", ".markdown", 0 };
    static const char *rich[] = { ".pdf", ".docx", ".odt", ".pptx", ".epub", ".html", ".htm", 0 };
    i32 i;
    for (i = 0; plain[i]; ++i) { size_t k = strlen(plain[i]); if (l > k && !strcmp(name + l - k, plain[i])) return cli_slurp_(path); }
    for (i = 0; rich[i]; ++i) {
        size_t k = strlen(rich[i]);
        if (l > k && !strcmp(name + l - k, rich[i])) {
            char cmd[8192], *b = (char *)malloc(1 << 16), *q = cmd;
            size_t n = 0, cap = 1 << 16, r;
            FILE *p;
            const char *s;
            q += sprintf(q, "python3 '%s' '", cli_extract_script_);
            for (s = path; *s && q < cmd + 8000; ++s) { if (*s == '\'') { memcpy(q, "'\\''", 4); q += 4; } else *q++ = *s; }
            strcpy(q, "'");
            p = popen(cmd, "r");
            if (!p) { free(b); return (char *)0; }
            while ((r = fread(b + n, 1, cap - n - 1, p)) > 0) { n += r; if (n + 1 >= cap) { cap *= 2; b = (char *)realloc(b, cap); } }
            pclose(p);
            b[n] = 0;
            return b;
        }
    }
    return (char *)0;
}

/* chunk size in characters: what the model actually reads (EMBED_MAXLEN tokens, ~3.5 chars each), so no chunk is
 * embedded from its first part only; VECDB_CHUNK overrides */
static i32 cli_chunk_ = 448;
#define CLI_CHUNK cli_chunk_

/* notes are queued and added 1024 at a time (so several model servers can share a batch), so the model server sees batches */
static char *cli_qt_[1024], *cli_qx_[1024];
static i32 cli_qn_ = 0;

static void cli_flush_(VecDB *db)
{
    i32 i;
    if (cli_qn_ == 0) return;
    vecdb_add_notes(db, cli_qn_, (const char **)cli_qt_, (const char **)cli_qx_);
    for (i = 0; i < cli_qn_; ++i) { free(cli_qt_[i]); free(cli_qx_[i]); }
    cli_qn_ = 0;
}

static void cli_queue_(VecDB *db, const char *title, const char *text)
{
    cli_qt_[cli_qn_] = (char *)malloc(strlen(title) + 1); strcpy(cli_qt_[cli_qn_], title);
    cli_qx_[cli_qn_] = (char *)malloc(strlen(text) + 1); strcpy(cli_qx_[cli_qn_], text);
    if (++cli_qn_ == 1024) cli_flush_(db);
}

/* a long text becomes a document note that links to its chunks; every chunk links back and to its
 * neighbours, so the graph (and MCTS) can walk through a document */
static void cli_add_doc_(VecDB *db, const char *name, const char *text, i32 *count)
{
    i32 len = (i32)strlen(text), nch = 0, cap = 64, i;
    i32 *st, *en;
    char title[4200], *body;
    const char *p = text;
    if (len <= CLI_CHUNK + CLI_CHUNK / 4) { cli_queue_(db, name, text); ++*count; return; }
    st = (i32 *)malloc((size_t)cap * sizeof(i32)); en = (i32 *)malloc((size_t)cap * sizeof(i32));
    while (p < text + len) {
        const char *e = p + CLI_CHUNK, *stop = text + len;
        if (e >= stop) e = stop;
        else {
            const char *q = e;                                   /* prefer a paragraph break, then a sentence, then a blank */
            while (q > p + CLI_CHUNK / 2 && !(q[0] == '\n' && q[-1] == '\n')) --q;
            if (q <= p + CLI_CHUNK / 2) { q = e; while (q > p + CLI_CHUNK / 2 && !(q[-1] == '.' && q[0] == ' ')) --q; }
            if (q <= p + CLI_CHUNK / 2) { q = e; while (q > p + CLI_CHUNK / 2 && q[0] != ' ') --q; }
            if (q > p + CLI_CHUNK / 2) e = q;
        }
        if (nch == cap) { cap *= 2; st = (i32 *)realloc(st, (size_t)cap * sizeof(i32)); en = (i32 *)realloc(en, (size_t)cap * sizeof(i32)); }
        st[nch] = (i32)(p - text); en[nch] = (i32)(e - text); ++nch;
        p = e; while (p < text + len && (*p == '\n' || *p == ' ')) ++p;
    }
    body = (char *)malloc((size_t)len + (size_t)nch * 8 + 256 + (size_t)nch * (strlen(name) + 24));
    for (i = 0; i < nch; ++i) {                                  /* chunks */
        char *w = body;
        sprintf(title, "%s \xc2\xa7%d", name, i + 1);
        memcpy(w, text + st[i], (size_t)(en[i] - st[i])); w += en[i] - st[i];
        w += sprintf(w, "\n\n[[%s]]", name);
        if (i > 0) w += sprintf(w, " [[%s \xc2\xa7%d]]", name, i);
        if (i + 1 < nch) w += sprintf(w, " [[%s \xc2\xa7%d]]", name, i + 2);
        cli_queue_(db, title, body);
    }
    { char *w = body;                                            /* the document: its opening + links to every chunk */
      i32 head = en[0] < 700 ? en[0] : 700;
      memcpy(w, text, (size_t)head); w += head;
      w += sprintf(w, "\n\n");
      for (i = 0; i < nch; ++i) w += sprintf(w, "[[%s \xc2\xa7%d]] ", name, i + 1);
      cli_queue_(db, name, body); }
    *count += nch + 1;
    free(body); free(st); free(en);
}

static i32 cli_is_image_(const char *name)
{
    static const char *ext[] = { ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", 0 };
    size_t l = strlen(name); i32 i;
    for (i = 0; ext[i]; ++i) { size_t k = strlen(ext[i]); if (l > k) { size_t j; i32 same = 1; for (j = 0; j < k; ++j) if ((name[l - k + j] | 32) != ext[i][j]) same = 0; if (same) return 1; } }
    return 0;
}

static void cli_index_(VecDB *db, const char *dir, i32 *count)
{
    DIR *d = opendir(dir);
    struct dirent *e;
    if (!d) return;
    while ((e = readdir(d)) != (struct dirent *)0) {
        char path[4096]; struct stat st;
        char *txt;
        if (e->d_name[0] == '.') continue;
        sprintf(path, "%s/%s", dir, e->d_name);
        if (stat(path, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) { cli_index_(db, path, count); continue; }
        if (db->store2 && cli_is_image_(e->d_name)) {
            char cap[512]; size_t i;
            strncpy(cap, e->d_name, 400); cap[400] = 0;
            for (i = 0; cap[i]; ++i) if (cap[i] == '_' || cap[i] == '-' || cap[i] == '.') cap[i] = ' ';
            if (vecdb_add_image(db, e->d_name, path, cap) >= 0) ++*count;
            continue;
        }
        txt = cli_text_(path, e->d_name);
        if (txt) { if (txt[0]) cli_add_doc_(db, e->d_name, txt, count); free(txt); }
    }
    closedir(d);
}

static void cli_hits_(const VecDB *db, const VHit *h, i32 n)
{
    i32 i;
    for (i = 0; i < n; ++i) {
        printf("%2d. %-36s score %.4f  cos %.2f  bm25 %.2f", i + 1, db->nodes[h[i].node].name, h[i].score, h[i].sim, h[i].lex);
        if (h[i].via >= 0) printf("  via %s (depth %d)", db->nodes[h[i].via].name, h[i].depth);
        putchar('\n');
    }
}

/* chunks of one document ("name §3") are shown once, under the document's name, at the rank of their best chunk */
static void cli_hits_grouped_(const VecDB *db, const VHit *h, i32 n, i32 want)
{
    i32 i, j, shown = 0;
    char base[512], other[512];
    for (i = 0; i < n && shown < want; ++i) {
        const char *nm = db->nodes[h[i].node].name;
        const char *p = strstr(nm, " \xc2\xa7");
        i32 bl = p ? (i32)(p - nm) : (i32)strlen(nm), dup = 0, more = 0;
        if (bl > 500) bl = 500;
        memcpy(base, nm, (size_t)bl); base[bl] = 0;
        for (j = 0; j < i; ++j) {                                /* an earlier hit of the same document already shown */
            const char *q = db->nodes[h[j].node].name, *pq = strstr(q, " \xc2\xa7");
            i32 ql = pq ? (i32)(pq - q) : (i32)strlen(q);
            if (ql == bl && memcmp(q, base, (size_t)bl) == 0) { dup = 1; break; }
        }
        if (dup) continue;
        for (j = i + 1; j < n; ++j) {
            const char *q = db->nodes[h[j].node].name, *pq = strstr(q, " \xc2\xa7");
            i32 ql = pq ? (i32)(pq - q) : (i32)strlen(q);
            if (ql == bl && memcmp(q, base, (size_t)bl) == 0) ++more;
        }
        (void)other;
        printf("%2d. %-36s score %.4f  cos %.2f  bm25 %.2f", ++shown, nm, h[i].score, h[i].sim, h[i].lex);
        if (more) printf("  (+%d more parts)", more);
        if (h[i].via >= 0) printf("  via %s (depth %d)", db->nodes[h[i].via].name, h[i].depth);
        putchar('\n');
    }
}

static void cli_names_(const VecDB *db, const i32 *ids, i32 n)
{
    i32 i;
    for (i = 0; i < n; ++i) printf("  %s%s\n", db->nodes[ids[i]].kind == VN_TAG ? "#" : "", db->nodes[ids[i]].name);
}

static const char *cli_path_ = "";
static VEmbedPool *cli_pool_ = (VEmbedPool *)0;

/* stop the model servers on the way out, so a pipe reading our output is not held open by them */
static void cli_close_(void) { vecdb_embpool_close(cli_pool_); cli_pool_ = (VEmbedPool *)0; }

static i32 cli_run_(VecDB *db, const char *cmd, const char *arg)
{
    VHit hits[64];
    i32 buf[4096], n, i;
    (void)n;
    if (!strcmp(cmd, "search")) cli_hits_grouped_(db, hits, vecdb_query(db, arg, 40, 0, hits), 10);
    else if (!strcmp(cmd, "vec")) cli_hits_grouped_(db, hits, vecdb_search(db, arg, 40, 1, -1, hits), 10);       /* vector only */
    else if (!strcmp(cmd, "lex")) cli_hits_grouped_(db, hits, vecdb_search(db, arg, 40, 2, -1, hits), 10);       /* words only */
    else if (!strcmp(cmd, "remove")) {
        i32 id = vecdb_find(db, arg);
        if (id < 0 || !vecdb_remove_note(db, id)) { fprintf(stderr, "no such note: %s\n", arg); return 1; }
        if (vecdb_sync(db, cli_path_) != 0) { fprintf(stderr, "vecdb: could not save\n"); return 1; }
        printf("removed %s\n", arg);
    }
    else if (!strcmp(cmd, "stats"))
        printf("%d notes, %d pictures, %d ghosts (linked, not yet written), %d tags, embedding cache %d\n", vecdb_count(db, VN_NOTE), db->store2 ? store_count(db->store2) : 0, vecdb_count(db, VN_GHOST), vecdb_count(db, VN_TAG), vecdb_embed_cache_count(db));
    else if (!strcmp(cmd, "images")) cli_hits_(db, hits, vecdb_search_images(db, arg, 10, hits));
    else if (!strcmp(cmd, "similar")) {
        i32 id = vecdb_find(db, arg);
        if (id < 0) { fprintf(stderr, "no such picture\n"); return 1; }
        cli_hits_(db, hits, vecdb_similar_images(db, id, 10, hits));
    }
    else if (!strcmp(cmd, "explore")) cli_hits_grouped_(db, hits, vecdb_query(db, arg, 40, 1, hits), 10);
    else if (!strcmp(cmd, "links") || !strcmp(cmd, "backlinks")) {
        i32 id = vecdb_find(db, arg);
        if (id < 0) { fprintf(stderr, "no such note\n"); return 1; }
        n = cmd[0] == 'l' ? vecdb_links(db, id, buf, 4096) : vecdb_backlinks(db, id, buf, 4096);
        cli_names_(db, buf, n);
    } else if (!strcmp(cmd, "tag")) {
        i32 id = vecdb_find_tag(db, arg);
        if (id < 0) { fprintf(stderr, "no such tag\n"); return 1; }
        cli_names_(db, buf, vecdb_notes_with_tag(db, id, buf, 4096));
    } else if (!strcmp(cmd, "tags")) {
        for (i = 0; i < db->nn; ++i) if (db->nodes[i].kind == VN_TAG) printf("  #%-24s %d\n", db->nodes[i].name, db->nodes[i].in.n);
    } else if (!strcmp(cmd, "ghosts")) {
        for (i = 0; i < db->nn; ++i) if (db->nodes[i].kind == VN_GHOST) printf("  %-30s linked from %d\n", db->nodes[i].name, db->nodes[i].in.n);
    } else return 2;
    return 0;
}

int main(int argc, char **argv)
{
    static Arena arena;
    VecDB *db;
    const char *cmd;

    if (argc < 3) { fprintf(stderr, "usage: vecdb DB.vdb index DIR | update DIR | remove TITLE | stats | search Q | images Q | similar PIC | explore Q | links T | backlinks T | tag TAG | tags | ghosts\n"); return 2; }
    cmd = argv[2];
    { const char *ch = getenv("VECDB_CHUNK"), *ml = getenv("EMBED_MAXLEN");
      if (ch && atoi(ch) > 100) cli_chunk_ = atoi(ch); else cli_chunk_ = (ml && atoi(ml) > 16 ? atoi(ml) : 128) * 7 / 2; }
    { static char sp[4096]; char *sl = strrchr(argv[0], '/'); if (getenv("VECDB_EXTRACT")) cli_extract_script_ = getenv("VECDB_EXTRACT"); else if (sl) { *sl = 0; sprintf(sp, "%s/../extract.py", argv[0]); *sl = '/'; cli_extract_script_ = sp; } }
    if (!strcmp(cmd, "index")) {
        i32 count = 0;
        struct stat sb;
        if (argc < 4 || stat(argv[3], &sb) != 0 || !S_ISDIR(sb.st_mode)) { fprintf(stderr, "vecdb: %s is not a directory\n", argc < 4 ? "(none)" : argv[3]); return 2; }
        { const char *sm = getenv("VECDB_SMOOTH"), *es = getenv("VECDB_EMBED_SERVER");
          VEmbedPool *ep = es ? vecdb_embpool_open(es) : (VEmbedPool *)0;
          if (es && !ep) { fprintf(stderr, "vecdb: embedding server failed to start: %s\n", es); return 1; }
          cli_pool_ = ep; atexit(cli_close_);
          db = vecdb_new(&arena, ep ? ep->dim : 256);
          if (ep) { const char *cp = getenv("VECDB_EMBED_CACHE"); char *cpath = (char *)malloc(strlen(argv[1]) + 6); vecdb_use_embpool(db, ep); if (cp) strcpy(cpath, cp); else sprintf(cpath, "%s.emb", argv[1]); vecdb_embed_cache_open(db, cpath, es); free(cpath); }
          { const char *cs = getenv("VECDB_CLIP_SERVER");
            if (cs) { VEmbedProc *cp = vecdb_embproc_open(cs, (const char *)0); if (cp) vecdb_enable_images(db, cp); else fprintf(stderr, "vecdb: image model failed to start, images skipped\n"); } }
          cli_index_(db, argv[3], &count);
          cli_flush_(db);
          if (sm) vecdb_smooth(db, (f32)atof(sm)); }
        vecdb_save(db, argv[1]);
        printf("%d notes (%d pictures), %d ghosts, %d tags -> %s\n", vecdb_count(db, VN_NOTE), db->store2 ? store_count(db->store2) : 0, vecdb_count(db, VN_GHOST), vecdb_count(db, VN_TAG), argv[1]);
        return 0;
    }
    { const char *es = getenv("VECDB_EMBED_SERVER");
      VEmbedPool *ep = es ? vecdb_embpool_open(es) : (VEmbedPool *)0;
      db = vecdb_load_with(&arena, argv[1], ep ? vdb_pool_embed_ : (void (*)(void *, const char *, i32, i32, f32 *, i32))0, ep);
      if (db && ep) vecdb_use_embpool(db, ep);
      cli_pool_ = ep; atexit(cli_close_); }
    if (!db) { fprintf(stderr, "vecdb: cannot open %s (run: vecdb %s index DIR)\n", argv[1], argv[1]); return 1; }
    { const char *cs = getenv("VECDB_CLIP_SERVER");
      if (cs && db->store2) { VEmbedProc *cp = vecdb_embproc_open(cs, (const char *)0); if (cp) vecdb_set_image_model(db, cp); } }
    vecdb_build(db);
    cli_path_ = argv[1];
    if (!strcmp(cmd, "update")) {                                /* re-read DIR: changed files re-embedded (unchanged ones come from the cache), vanished ones removed */
        i32 count = 0, i, gone = 0, *toff_len;
        i64 *toff;
        const char *es = getenv("VECDB_EMBED_SERVER");
        if (argc < 4) { fprintf(stderr, "usage: vecdb DB.vdb update DIR\n"); return 2; }
        if (es && db->embed) { const char *cp = getenv("VECDB_EMBED_CACHE"); char *cpath = (char *)malloc(strlen(argv[1]) + 6); if (cp) strcpy(cpath, cp); else sprintf(cpath, "%s.emb", argv[1]); vecdb_embed_cache_open(db, cpath, es); free(cpath); }
        { const char *cs = getenv("VECDB_CLIP_SERVER");
          if (cs && !db->store2) { VEmbedProc *cp = vecdb_embproc_open(cs, (const char *)0); if (cp) vecdb_enable_images(db, cp); } }
        toff = (i64 *)malloc((size_t)(db->nn + 1) * sizeof(i64)); toff_len = (i32 *)malloc((size_t)(db->nn + 1) * sizeof(i32));
        for (i = 0; i < db->nn; ++i) { toff[i] = db->nodes[i].toff; toff_len[i] = db->nodes[i].tlen; }
        { i32 before = db->nn;
          cli_index_(db, argv[3], &count);
          cli_flush_(db);
          for (i = 0; i < before; ++i)                            /* a note whose body was not rewritten was not in DIR any more */
              if (db->nodes[i].kind == VN_NOTE && db->nodes[i].toff == toff[i] && db->nodes[i].tlen == toff_len[i]) { vecdb_remove_note(db, i); ++gone; } }
        free(toff); free(toff_len);
        if (vecdb_sync(db, argv[1]) != 0) { fprintf(stderr, "vecdb: could not save\n"); return 1; }
        printf("%d files read, %d notes removed; now %d notes, %d ghosts, %d tags\n", count, gone, vecdb_count(db, VN_NOTE), vecdb_count(db, VN_GHOST), vecdb_count(db, VN_TAG));
        return 0;
    }
    if (argc < 4 && strcmp(cmd, "tags") && strcmp(cmd, "ghosts") && strcmp(cmd, "repl") && strcmp(cmd, "stats")) return 2;
    if (!strcmp(cmd, "repl")) {                                  /* keep the models loaded: one command per line */
        char line[2048];
        fprintf(stderr, "vecdb> ");
        while (fgets(line, sizeof line, stdin)) {
            char *arg = strchr(line, ' '), *nl = strchr(line, '\n');
            if (nl) *nl = 0;
            if (arg) *arg++ = 0;
            if (!strcmp(line, "quit") || !strcmp(line, "exit")) break;
            if (line[0]) cli_run_(db, line, arg ? arg : "");
            fflush(stdout); fprintf(stderr, "vecdb> ");
        }
        return 0;
    }
    return cli_run_(db, cmd, argc > 3 ? argv[3] : "");
}
#endif
