/* ingest.c - Markdown and JSON / JSON Lines into a store (include/ingest.h). Uses only the public C API of vecindex.h
 * (vi_chunk, VI_ID, the write batch, the fields) and mem.h's arena; vecindex.c includes it at its end.
 *
 * Markdown: an optional front matter block (--- ... --- with key: value lines) gives fields; an ATX heading (# to
 * ######, outside ``` or ~~~ fences) starts a section; a section is a page, cut by vi_chunk; the heading path of a
 * section goes in front of each chunk for the model, not into the stored text.
 * JSON: one parser, no recursion - its own stack of containers, 64 deep; strings decoded (\uXXXX and surrogate pairs
 * to UTF-8); every scalar of an element becomes a line "path: value", array indices left out of the path. An element
 * is a page: a top-level array's elements, a JSON Lines file's lines, or the single top-level value. */
#include "ingest.h"
#include "mem.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define ING_WANT 16                 /* fields an ingest can keep (a store has 16) */
#define ING_PATH 512                /* a JSON path's bytes */
#define ING_DEPTH 64                /* JSON nesting */
#define ING_DOCS 1024               /* document numbers a file owns */

typedef struct { i32 type; i64 iv; f64 fv; const char *sv; i32 sl; i32 set; } IngVal;

typedef struct {
    vi_db *db;
    vi_batch *b;
    vi_ingest_opts o;
    vi_ingest_stats st;
    Gen a;
    const char *name;
    i32 nlen, dim, cmax;
    i64 docbase;
    f32 *vec;
    int *offs, *lens;
    char want[ING_WANT][64];        /* the dotted paths asked for, and their field names (dots as _) */
    char fname[ING_WANT][32];
    i32 nwant;
    IngVal val[ING_WANT];
    char *buf;                      /* a growing buffer: a record's text, or a chunk with its headings */
    i32 bcap, blen;
} Ing;

static int ing_fail_(Ing *g, const char *a, const char *b, const char *c) { (void)g; return vi_fail_(-1, a, b, c); }

static i32 ing_room_(Ing *g, i32 need)                           /* buf holds blen + need bytes */
{
    char *nb;
    i32 cap;
    if (g->blen + need <= g->bcap) return 0;
    for (cap = g->bcap ? g->bcap : 4096; cap < g->blen + need; cap *= 2) if (cap > (1 << 29)) return -1;
    if (!(nb = (char *)gen_alloc_(&g->a, (size_t)cap))) return -1;  /* the old one stays in the arena until it ends */
    if (g->blen) memcpy(nb, g->buf, (size_t)g->blen);
    g->buf = nb; g->bcap = cap;
    return 0;
}
static i32 ing_put_(Ing *g, const char *s, i32 n)
{
    if (n <= 0) return 0;
    if (ing_room_(g, n)) return -1;
    memcpy(g->buf + g->blen, s, (size_t)n); g->blen += n;
    return 0;
}

/* the fields asked for: "lang, year, author.name" */
static i32 ing_want_(Ing *g, const char *f)
{
    g->nwant = 0;
    while (f && *f) {
        const char *e;
        i32 n, i;
        while (*f == ',' || *f == ' ') ++f;
        for (e = f; *e && *e != ','; ++e) {}
        n = (i32)(e - f);
        while (n > 0 && f[n - 1] == ' ') --n;
        if (n > 0) {
            if (g->nwant == ING_WANT || n > 31) return ing_fail_(g, "ingest: fields takes at most 16 keys of 31 bytes", (const char *)0, (const char *)0);
            memcpy(g->want[g->nwant], f, (size_t)n); g->want[g->nwant][n] = 0;
            for (i = 0; i < n; ++i) g->fname[g->nwant][i] = f[i] == '.' ? '_' : f[i];
            g->fname[g->nwant][n] = 0;
            ++g->nwant;
        }
        f = e;
    }
    return 0;
}

/* a scalar at path (dots, no indices) for the fields: the first value of a wanted path is kept */
static void ing_field_(Ing *g, const char *path, i32 pl, i32 type, const char *v, i32 vl)
{
    i32 w;
    IngVal *x;
    for (w = 0; w < g->nwant; ++w) if ((i32)strlen(g->want[w]) == pl && memcmp(g->want[w], path, (size_t)pl) == 0) break;
    if (w == g->nwant || g->val[w].set) return;
    x = &g->val[w];
    x->set = 1; x->type = type; x->sv = v; x->sl = vl;
    if (type != VI_STR) {
        char nb[64];
        i32 n = vl < 63 ? vl : 63, i, isint = 1;
        memcpy(nb, v, (size_t)n); nb[n] = 0;
        if (type == VI_INT && n > 0 && (nb[0] == 't' || nb[0] == 'f')) { x->iv = nb[0] == 't'; return; }
        for (i = 0; i < n; ++i) if (nb[i] == '.' || nb[i] == 'e' || nb[i] == 'E') isint = 0;
        if (isint) {
            u64 a = 0;
            i32 neg = nb[0] == '-', ok = n > neg;
            for (i = neg; i < n && ok; ++i) {
                if (nb[i] < '0' || nb[i] > '9' || a > ((((u64)1 << 63) - 1) - (u64)(nb[i] - '0')) / 10) ok = 0;
                else a = a * 10 + (u64)(nb[i] - '0');
            }
            if (ok) { x->type = VI_INT; x->iv = neg ? -(i64)a : (i64)a; return; }
        }
        x->type = VI_FLOAT; x->fv = strtod(nb, (char **)0);
        if (!(x->fv - x->fv == 0.0)) x->set = 0;                /* not a finite number: no field */
    }
}

/* body cut into chunks, each a row VI_ID(doc, page, c); prefix (may be empty) in front of each for the model */
static i32 ing_rows_(Ing *g, i64 doc, i32 page, const char *body, i32 bl, const char *prefix, i32 pl)
{
    i32 n, c, w;
    for (;;) {
        n = vi_chunk(body, bl, g->o.max_bytes, g->o.overlap, g->offs, g->lens, (int *)0, g->cmax);
        if (n < 0) return -1;
        if (n <= g->cmax) break;
        g->cmax = n;
        g->offs = (int *)gen_alloc_(&g->a, (size_t)n * sizeof(int));
        g->lens = (int *)gen_alloc_(&g->a, (size_t)n * sizeof(int));
        if (!g->offs || !g->lens) return ing_fail_(g, "ingest: out of memory", (const char *)0, (const char *)0);
    }
    if (n > ING_DOCS) return ing_fail_(g, "ingest: a section or element of more than 1024 chunks in ", g->name, " (VI_ID's limit)");
    for (c = 0; c < n; ++c) {
        vi_i64 id = VI_ID(doc, page, c);
        const char *t = body + g->offs[c];
        if (pl > 0) {                                            /* what the model reads: the headings, then the chunk */
            g->blen = 0;
            if (ing_put_(g, prefix, pl) || ing_put_(g, "\n\n", 2) || ing_put_(g, t, g->lens[c]))
                return ing_fail_(g, "ingest: out of memory", (const char *)0, (const char *)0);
            if (g->o.embed(g->o.embed_ctx, g->buf, g->blen, g->vec, g->dim)) return ing_fail_(g, "ingest: the embedding model failed on ", g->name, (const char *)0);
        } else if (g->o.embed(g->o.embed_ctx, t, g->lens[c], g->vec, g->dim)) return ing_fail_(g, "ingest: the embedding model failed on ", g->name, (const char *)0);
        if (vi_batch_add_text(g->b, id, g->vec, t, g->lens[c]) || vi_batch_set_str(g->b, id, "path", g->name, g->nlen)) return -1;
        for (w = 0; w < g->nwant; ++w) {
            const IngVal *x = &g->val[w];
            i32 r = 0;
            if (!x->set) continue;
            if (x->type == VI_INT) r = vi_batch_set_int(g->b, id, g->fname[w], (vi_i64)x->iv);
            else if (x->type == VI_FLOAT) r = vi_batch_set_float(g->b, id, g->fname[w], x->fv);
            else r = vi_batch_set_str(g->b, id, g->fname[w], x->sv, x->sl);
            if (r) return -1;
            ++g->st.fields;
        }
        ++g->st.chunks; g->st.bytes += g->lens[c];
    }
    return 0;
}

/* ---- Markdown ---- */

static i32 ing_ws_(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v'; }

static i32 ing_md_(Ing *g, const char *t, i32 L)
{
    const char *hs[7];
    i32 hl[7], lv, k, pos = 0, body = 0, fence = 0, page = 0;
    char pre[2048];
    for (lv = 0; lv < 7; ++lv) { hs[lv] = (const char *)0; hl[lv] = 0; }
    if (L >= 4 && memcmp(t, "---", 3) == 0 && (t[3] == '\n' || t[3] == '\r')) {   /* the front matter */
        i32 p = 4, end = -1;
        while (p < L) {
            i32 e = p, k, v, ve;
            while (e < L && t[e] != '\n') ++e;
            if ((e - p >= 3 && (memcmp(t + p, "---", 3) == 0 || memcmp(t + p, "...", 3) == 0))) { end = e < L ? e + 1 : e; break; }
            for (k = p; k < e && t[k] != ':'; ++k) {}
            if (k < e) {
                i32 ks = p, ke = k, type;
                while (ks < ke && ing_ws_(t[ks])) ++ks;
                while (ke > ks && ing_ws_(t[ke - 1])) --ke;
                v = k + 1; ve = e;
                while (v < ve && ing_ws_(t[v])) ++v;
                while (ve > v && ing_ws_(t[ve - 1])) --ve;
                if (ve - v >= 2 && (t[v] == '"' || t[v] == '\'') && t[ve - 1] == t[v]) { ++v; --ve; type = VI_STR; }
                else if ((ve - v == 4 && memcmp(t + v, "true", 4) == 0) || (ve - v == 5 && memcmp(t + v, "false", 5) == 0)) type = VI_INT;
                else {
                    i32 i, num = ve > v;
                    for (i = v; i < ve && num; ++i) num = (t[i] >= '0' && t[i] <= '9') || t[i] == '-' || t[i] == '+' || t[i] == '.' || t[i] == 'e' || t[i] == 'E';
                    type = num ? VI_FLOAT : VI_STR;
                }
                ing_field_(g, t + ks, ke - ks, type, t + v, ve - v);    /* a number: ing_field_ tells whole from not */
            }
            p = e + 1;
        }
        if (end > 0) pos = body = end;
    }
    while (pos <= L) {
        i32 e = pos, lvl = 0, is_h = 0;
        while (e < L && t[e] != '\n') ++e;
        if (pos < L && (memcmp(t + pos, "```", 3) == 0 || memcmp(t + pos, "~~~", 3) == 0) && e - pos >= 3) fence = !fence;
        else if (!fence && pos < L && t[pos] == '#') {
            while (pos + lvl < e && t[pos + lvl] == '#' && lvl < 7) ++lvl;
            is_h = lvl >= 1 && lvl <= 6 && (pos + lvl == e || t[pos + lvl] == ' ' || t[pos + lvl] == '\t');
        }
        if (is_h || pos == L) {                                  /* the section before ends here */
            i32 a = body, z = pos < L ? pos : L, pl = 0;
            while (a < z && ing_ws_(t[a])) ++a;
            if (a < z) {
                if (g->o.headings)
                    for (k = 1; k <= 6; ++k) {
                        if (!hs[k]) continue;
                        if (pl && pl + 3 < (i32)sizeof pre) { memcpy(pre + pl, " > ", 3); pl += 3; }
                        if (pl + k + 1 + hl[k] < (i32)sizeof pre) { memset(pre + pl, '#', (size_t)k); pl += k; pre[pl++] = ' '; memcpy(pre + pl, hs[k], (size_t)hl[k]); pl += hl[k]; }
                    }
                if (page >= ING_DOCS * 1024) return ing_fail_(g, "ingest: more than 2^20 sections in ", g->name, (const char *)0);
                if (ing_rows_(g, g->docbase + page / 1024, page % 1024, t + a, z - a, pre, pl)) return -1;
                ++page;
            }
            if (pos == L) break;
            {   i32 s = pos + lvl, f = e;                        /* the heading's title, without its # marks */
                while (s < f && ing_ws_(t[s])) ++s;
                while (f > s && (ing_ws_(t[f - 1]) || t[f - 1] == '#')) --f;
                hs[lvl] = t + s; hl[lvl] = f - s;
                for (k = lvl + 1; k <= 6; ++k) hs[k] = (const char *)0;
            }
            body = pos;                                          /* the heading line is the first of its section */
        }
        pos = e < L ? e + 1 : L;                                 /* at L, the loop's next turn flushes the last section */
    }
    g->st.docs = 1; g->st.sections = page;
    return 0;
}

/* ---- JSON ---- */

typedef struct { u8 kind; i32 plen; } IngFrame;   /* kind 'o' an object, 'a' an array; plen the path's length at its start */

static i32 ing_hex_(const char *s, u32 *v)
{
    i32 i;
    *v = 0;
    for (i = 0; i < 4; ++i) {
        char c = s[i];
        *v <<= 4;
        if (c >= '0' && c <= '9') *v |= (u32)(c - '0');
        else if (c >= 'a' && c <= 'f') *v |= (u32)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') *v |= (u32)(c - 'A' + 10);
        else return -1;
    }
    return 0;
}

/* the string at t[*p] (on its quote) decoded into the arena: its bytes and length; *p after the closing quote */
static const char *ing_str_(Ing *g, const char *t, i32 L, i32 *p, i32 *outl)
{
    i32 i = *p + 1, n = 0, start;
    char *o;
    for (start = i; i < L && t[i] != '"'; ++i) if (t[i] == '\\') ++i;   /* its end first: the size to take */
    if (i >= L) return (const char *)0;
    if (!(o = (char *)gen_alloc_(&g->a, (size_t)(i - start) + 1))) return (const char *)0;
    for (i = start; t[i] != '"'; ++i) {
        u32 u, lo;
        if ((u8)t[i] < 0x20) return (const char *)0;              /* a raw control character: not JSON */
        if (t[i] != '\\') { o[n++] = t[i]; continue; }
        ++i;
        switch (t[i]) {
        case '"': o[n++] = '"'; break;
        case '\\': o[n++] = '\\'; break;
        case '/': o[n++] = '/'; break;
        case 'b': o[n++] = '\b'; break;
        case 'f': o[n++] = '\f'; break;
        case 'n': o[n++] = '\n'; break;
        case 'r': o[n++] = '\r'; break;
        case 't': o[n++] = '\t'; break;
        case 'u':
            if (i + 4 >= L || ing_hex_(t + i + 1, &u)) return (const char *)0;
            i += 4;
            if (u >= 0xd800 && u <= 0xdbff && i + 6 < L && t[i + 1] == '\\' && t[i + 2] == 'u' && !ing_hex_(t + i + 3, &lo) && lo >= 0xdc00 && lo <= 0xdfff) {
                u = 0x10000 + ((u - 0xd800) << 10) + (lo - 0xdc00); i += 6;
            }
            if (u < 0x80) o[n++] = (char)u;                      /* UTF-8: never longer than the escape it came from */
            else if (u < 0x800) { o[n++] = (char)(0xc0 | (u >> 6)); o[n++] = (char)(0x80 | (u & 0x3f)); }
            else if (u < 0x10000) { o[n++] = (char)(0xe0 | (u >> 12)); o[n++] = (char)(0x80 | ((u >> 6) & 0x3f)); o[n++] = (char)(0x80 | (u & 0x3f)); }
            else { o[n++] = (char)(0xf0 | (u >> 18)); o[n++] = (char)(0x80 | ((u >> 12) & 0x3f)); o[n++] = (char)(0x80 | ((u >> 6) & 0x3f)); o[n++] = (char)(0x80 | (u & 0x3f)); }
            break;
        default: return (const char *)0;
        }
    }
    *p = i + 1; *outl = n;
    return o;
}

/* one scalar of the record: its text line and, when asked for, its field */
static i32 ing_scalar_(Ing *g, const char *path, i32 pl, i32 type, const char *v, i32 vl)
{
    if (pl == 0) { path = "value"; pl = 5; }
    if (ing_put_(g, path, pl) || ing_put_(g, ": ", 2) || ing_put_(g, v, vl) || ing_put_(g, "\n", 1)) return -1;
    if (type) ing_field_(g, path, pl, type, v, vl);
    return 0;
}

/* t[*p, L): one JSON value; every value at depth rec (0: the top value itself, 1: a top-level array's elements) is
 * an element - its scalars' text cut into rows, element number *elem. *p after it. 0, -1 (not JSON; vi_error says
 * where). */
static i32 ing_json_value_(Ing *g, const char *t, i32 L, i32 *p, i32 rec, i32 *elem)
{
    IngFrame st[ING_DEPTH];
    char path[ING_PATH];
    i32 sp = 0, pl = 0, in_rec = 0, want_key = 0, w;
    char where[48];
    for (;;) {
        i32 i = *p;
        while (i < L && ing_ws_(t[i])) ++i;
        *p = i;
        if (i >= L) goto bad;
        if (want_key) {                                          /* an object's key, or its end */
            const char *k;
            i32 kl;
            if (t[i] == '}' && want_key == 1) { ++*p; want_key = 0; --sp; pl = st[sp].plen; goto closed; }   /* {} */
            if (t[i] != '"' || !(k = ing_str_(g, t, L, p, &kl))) goto bad;
            pl = st[sp - 1].plen;
            if (pl + 1 + kl >= ING_PATH) goto bad;
            if (pl) path[pl++] = '.';
            memcpy(path + pl, k, (size_t)kl); pl += kl;
            i = *p;
            while (i < L && ing_ws_(t[i])) ++i;
            if (i >= L || t[i] != ':') goto bad;
            *p = i + 1;
            want_key = 0;
            continue;
        }
        if (sp == rec && !in_rec) {                              /* an element begins */
            in_rec = 1; g->blen = 0;
            for (w = 0; w < g->nwant; ++w) g->val[w].set = 0;
        }
        if (t[i] == '{' || t[i] == '[') {
            if (sp == ING_DEPTH) goto bad;
            st[sp].kind = t[i] == '{' ? 'o' : 'a'; st[sp].plen = pl; ++sp;
            *p = i + 1;
            if (t[i] == '{') { want_key = 1; continue; }
            while (*p < L && ing_ws_(t[*p])) ++*p;
            if (*p < L && t[*p] == ']') { ++*p; --sp; pl = st[sp].plen; goto closed; }   /* [] */
            continue;
        }
        if (t[i] == '"') {
            const char *s;
            i32 sl;
            if (!(s = ing_str_(g, t, L, p, &sl)) || ing_scalar_(g, path, pl, VI_STR, s, sl)) goto bad;
        } else {
            i32 e = i;
            while (e < L && !ing_ws_(t[e]) && t[e] != ',' && t[e] != '}' && t[e] != ']') ++e;
            if (e == i) goto bad;
            if ((e - i == 4 && memcmp(t + i, "true", 4) == 0) || (e - i == 5 && memcmp(t + i, "false", 5) == 0)) {
                if (ing_scalar_(g, path, pl, VI_INT, t + i, e - i)) goto bad;
            } else if (e - i == 4 && memcmp(t + i, "null", 4) == 0) {
                if (ing_scalar_(g, path, pl, 0, t + i, e - i)) goto bad;
            } else {
                i32 k, ok = 1;
                for (k = i; k < e && ok; ++k) ok = (t[k] >= '0' && t[k] <= '9') || t[k] == '-' || t[k] == '+' || t[k] == '.' || t[k] == 'e' || t[k] == 'E';
                if (!ok || ing_scalar_(g, path, pl, VI_FLOAT, t + i, e - i)) goto bad;
            }
            *p = e;
        }
    closed:                                                      /* a value is done: what comes after it */
        for (;;) {
            if (sp == rec && in_rec) {                           /* an element is done */
                if (*elem >= ING_DOCS * 1024) { vi_fail_(-1, "ingest: more than 2^20 elements in ", g->name, (const char *)0); return -1; }
                if (g->blen > 0 && ing_rows_(g, g->docbase + *elem / 1024, *elem % 1024, g->buf, g->blen, "", 0)) return -1;
                ++*elem; in_rec = 0;
            }
            if (sp == 0) return 0;
            i = *p;
            while (i < L && ing_ws_(t[i])) ++i;
            if (i >= L) goto bad;
            pl = st[sp - 1].plen;
            if (t[i] == ',') { *p = i + 1; if (st[sp - 1].kind == 'o') want_key = 2; break; }
            if ((t[i] == '}' && st[sp - 1].kind == 'o') || (t[i] == ']' && st[sp - 1].kind == 'a')) { *p = i + 1; --sp; pl = sp ? st[sp].plen : 0; continue; }
            goto bad;
        }
    }
bad:
    sprintf(where, " (at byte %d)", (int)*p);
    vi_fail_(-1, "ingest: not JSON in ", g->name, where);
    return -1;
}

static i32 ing_json_(Ing *g, const char *t, i32 L, i32 lines)
{
    i32 p = 0, elem = 0;
    if (lines) {                                                 /* JSON Lines: a value a line */
        while (p < L) {
            i32 e = p, q = p;
            while (e < L && t[e] != '\n') ++e;
            while (q < e && ing_ws_(t[q])) ++q;
            if (q < e && ing_json_value_(g, t, e, &q, 0, &elem)) return -1;
            while (q < e && ing_ws_(t[q])) ++q;
            if (q < e) { char w[48]; sprintf(w, " (at byte %d)", (int)q); return vi_fail_(-1, "ingest: more after a line's JSON value in ", g->name, w); }
            p = e + 1;
        }
    } else {
        while (p < L && ing_ws_(t[p])) ++p;
        if (p < L && ing_json_value_(g, t, L, &p, p < L && t[p] == '[' ? 1 : 0, &elem)) return -1;
        while (p < L && ing_ws_(t[p])) ++p;
        if (p < L) { char w[48]; sprintf(w, " (at byte %d)", (int)p); return vi_fail_(-1, "ingest: more after the JSON value in ", g->name, w); }
    }
    g->st.docs = elem; g->st.sections = elem;
    return 0;
}

/* ---- the stand-in model and the API ---- */

VI_API int vi_embed_hash(void *ctx, const char *text, int len, float *out, int dim)
{
    i32 i = 0, d;
    u64 prev = 0, h;
    f64 s = 0.0;
    (void)ctx;
    if (!text || !out || dim < 1) return -1;
    for (d = 0; d < dim; ++d) out[d] = 0.0f;
    while (i < len) {
        i32 j = i;                                               /* FNV-1a over a word, ASCII letters lowered */
        while (j < len && !(((u8)text[j] >= 0x80) || (text[j] >= '0' && text[j] <= '9') || ((text[j] | 32) >= 'a' && (text[j] | 32) <= 'z'))) ++j;
        if (j >= len) break;
        h = (u64)0xcbf29ce4u << 32 | 0x84222325u;
        for (i = j; i < len && (((u8)text[i] >= 0x80) || (text[i] >= '0' && text[i] <= '9') || ((text[i] | 32) >= 'a' && (text[i] | 32) <= 'z')); ++i)
            h = (h ^ (u64)(u8)((text[i] >= 'A' && text[i] <= 'Z') ? text[i] | 32 : text[i])) * ((u64)0x100u << 32 | 0x1b3u);
        out[(u32)(h >> 33) % (u32)dim] += (h & 1u) ? 1.0f : -1.0f;
        if (prev) { u64 b = (prev ^ (h * 0x9e3779b9u)) * ((u64)0x100u << 32 | 0x1b3u); out[(u32)(b >> 33) % (u32)dim] += (b & 1u) ? 0.5f : -0.5f; }
        prev = h;
    }
    for (d = 0; d < dim; ++d) s += (f64)out[d] * out[d];
    if (s > 0.0) { f32 r = (f32)(1.0 / sqrt(s)); for (d = 0; d < dim; ++d) out[d] *= r; }
    else out[0] = 1.0f;                                          /* no words: a vector all the same, never zero */
    return 0;
}

VI_API void vi_ingest_defaults(vi_ingest_opts *o)
{
    if (!o) return;
    o->embed = (vi_embed_fn)0; o->embed_ctx = (void *)0; o->max_bytes = 1200; o->overlap = 150; o->fields = (const char *)0; o->headings = 1;
}

VI_API void vi_ingest_range(const char *name, int len, vi_i64 *lo, vi_i64 *hi)
{
    vi_i64 base = (vi_key(name, len) >> 31) << 10;
    if (lo) *lo = VI_ID(base, 0, 0);
    if (hi) *hi = VI_ID(base + ING_DOCS - 1, 1023, 1023);
}

VI_API int vi_ingest_text(vi_db *db, const char *name, const char *text, int len, int kind, const vi_ingest_opts *o,
                          vi_ingest_stats *st)
{
    static Ing zero_;
    Ing gi, *g = &gi;
    vi_i64 lo, hi;
    i32 r, L;
    if (st) memset(st, 0, sizeof(*st));
    if (!db || !name || !*name || (!text && len != 0)) return vi_fail_(-1, "ingest: the store, a name and the text are needed", (const char *)0, (const char *)0);
    if (kind != VI_MD && kind != VI_JSON && kind != VI_JSONL) return vi_fail_(-1, "ingest: kind is VI_MD, VI_JSON or VI_JSONL", (const char *)0, (const char *)0);
    L = text ? vi_len_(text, len) : 0;
    gi = zero_;
    gen_init_(&g->a);
    g->db = db; g->name = name; g->nlen = (i32)strlen(name); g->dim = vi_dim(db);
    g->docbase = (vi_key(name, -1) >> 31) << 10;                 /* the file's 1024 document numbers, as vi_ingest_range */
    vi_ingest_defaults(&g->o);
    if (o) g->o = *o;
    if (!g->o.embed) g->o.embed = vi_embed_hash;
    r = ing_want_(g, g->o.fields);
    g->vec = (f32 *)gen_alloc_(&g->a, (size_t)g->dim * sizeof(f32));
    g->cmax = 64;
    g->offs = (int *)gen_alloc_(&g->a, (size_t)g->cmax * sizeof(int));
    g->lens = (int *)gen_alloc_(&g->a, (size_t)g->cmax * sizeof(int));
    if (!r && (!g->vec || !g->offs || !g->lens)) r = vi_fail_(-1, "ingest: out of memory", (const char *)0, (const char *)0);
    if (!r && !(g->b = vi_batch_begin(db))) r = -1;
    if (!r) {                                                    /* the file's old rows out, its new ones in: one batch */
        vi_ingest_range(name, -1, &lo, &hi);
        r = vi_batch_delete_range(g->b, lo, hi);
        if (!r) r = kind == VI_MD ? ing_md_(g, text, L) : ing_json_(g, text, L, kind == VI_JSONL);
        if (r) vi_batch_abort(g->b);
        else if (vi_batch_commit(g->b, 0) < 0) r = -1;
    }
    if (!r && st) *st = g->st;
    gen_free_(&g->a);
    return r ? -1 : 0;
}

VI_API int vi_ingest_file(vi_db *db, const char *path, const vi_ingest_opts *o, vi_ingest_stats *st)
{
    const char *dot;
    char ext[12];
    i32 kind = 0, i, fd, r;
    struct stat sb;
    void *m = (void *)0;
    if (st) memset(st, 0, sizeof(*st));
    if (!path) return vi_fail_(-1, "ingest: the path is NULL", (const char *)0, (const char *)0);
    dot = strrchr(path, '.');
    for (i = 0; dot && i < 11 && dot[i + 1]; ++i) ext[i] = (char)((dot[i + 1] >= 'A' && dot[i + 1] <= 'Z') ? dot[i + 1] | 32 : dot[i + 1]);
    ext[dot ? i : 0] = 0;
    if (!strcmp(ext, "md") || !strcmp(ext, "markdown")) kind = VI_MD;
    else if (!strcmp(ext, "json")) kind = VI_JSON;
    else if (!strcmp(ext, "jsonl") || !strcmp(ext, "ndjson")) kind = VI_JSONL;
    else return vi_fail_(-1, "ingest: .md, .markdown, .json, .jsonl or .ndjson in C (other formats: turn them into one of these): ", path, (const char *)0);
    if ((fd = open(path, O_RDONLY)) < 0) return vi_fail_(-1, "ingest: cannot open ", path, (const char *)0);
    if (fstat(fd, &sb) != 0 || sb.st_size > (off_t)0x7fffffff) { close(fd); return vi_fail_(-1, "ingest: cannot read, or over 2 GB: ", path, (const char *)0); }
    if (sb.st_size > 0 && (m = mmap((void *)0, (size_t)sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0)) == MAP_FAILED) { close(fd); return vi_fail_(-1, "ingest: cannot map ", path, (const char *)0); }
    r = vi_ingest_text(db, path, (const char *)m, (int)sb.st_size, kind, o, st);
    if (m) munmap(m, (size_t)sb.st_size);
    close(fd);
    return r;
}

VI_API vi_i64 vi_ingest_remove(vi_db *db, const char *name)
{
    vi_i64 lo, hi, before;
    if (!db || !name) return vi_fail_(-1, "ingest: the store and a name are needed", (const char *)0, (const char *)0);
    vi_ingest_range(name, -1, &lo, &hi);
    before = vi_count(db);
    if (vi_delete_range(db, lo, hi) < 0) return -1;
    return before - vi_count(db);
}
