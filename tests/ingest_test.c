/* ingest_test.c - include/ingest.h: Markdown (front matter, sections, fences, headings for the model only) and JSON /
 * JSON Lines (nesting, arrays, escapes) into a store, through the public API; a file replaced and removed whole; a
 * bad file leaving the store as it was; the embedding callback; the stand-in model. ./run builds it against
 * libvecindex.a. */
#include "vecindex.h"
#include "ingest.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define DIM 64

static int fails;

static void check(int cond, const char *what)
{
    printf(cond ? "  ok   %s\n" : "  FAIL %s\n", what);
    if (!cond) { ++fails; printf("       (%s)\n", vi_error()); }
}

static int calls;
static char seen[4096];
static int spy_(void *ctx, const char *t, int n, float *out, int dim)   /* the stand-in, and what it was given */
{
    (void)ctx;
    ++calls;
    if (calls == 3 && n < (int)sizeof seen) { memcpy(seen, t, (size_t)n); seen[n] = 0; }   /* the Install section */
    return vi_embed_hash((void *)0, t, n, out, dim);
}

static int text_(vi_db *db, vi_i64 id, char *b, int cap) { int n = vi_get_text(db, id, b, cap - 1); b[n > 0 ? n : 0] = 0; return n; }

int main(void)
{
    const char *md =
        "---\ntitle: Guide\nlang: c\nyear: 2024\ndraft: true\n---\n"
        "Intro text before any heading.\n\n"
        "# Guide\n\nThe guide's first words.\n\n"
        "## Install\n\nRun the build.\n\n```sh\n# not a heading, inside a fence\n./run\n```\n\n"
        "## Arena allocator\n\nThe arena hands out memory and frees it whole.\n";
    const char *js =
        "[ {\"title\": \"one\", \"year\": 2001, \"score\": 2.5, \"author\": {\"name\": \"Ann\", \"tags\": [\"a\", \"b\"]}},\n"
        "  {\"title\": \"caf\\u00e9 \\ud83d\\ude00\\nnew line\", \"year\": 2002, \"ok\": true, \"none\": null},\n"
        "  {}, {\"title\": \"three\", \"deep\": {\"x\": {\"y\": [1, [2, 3]]}}} ]";
    const char *jl = "{\"title\": \"line one\", \"year\": 1999}\n\n{\"title\": \"line two\", \"year\": 2000}\n";
    vi_db *db = vi_open((const char *)0, DIM, 0);
    vi_ingest_opts o;
    vi_ingest_stats st;
    vi_filter f;
    vi_i64 lo, hi, ids[8], v, before;
    float sc[8], q[DIM], q2[DIM];
    char b[512], path[128];
    int n;
    double dv;

    if (!db) return 1;
    vi_ingest_defaults(&o);
    o.fields = "lang, year, title, author.name, score, ok";
    o.embed = spy_;

    /* Markdown */
    check(vi_ingest_text(db, "notes/guide.md", md, -1, VI_MD, &o, &st) == 0, "vi_ingest_text: Markdown with a front matter block");
    check(st.docs == 1 && st.sections == 4 && st.chunks == 4 && calls == 4, "four sections (intro, Guide, Install, Arena allocator), a chunk each, each embedded");
    vi_ingest_range("notes/guide.md", -1, &lo, &hi);
    check(text_(db, lo + 0, b, sizeof b) > 0 && strncmp(b, "Intro text", 10) == 0, "page 0, chunk 0: the text before the first heading");
    check(text_(db, lo + (2 << 10), b, sizeof b) > 0 && strncmp(b, "## Install", 10) == 0 && strstr(b, "# not a heading") != (const char *)0,
          "page 2: the Install section, its heading line first, the fenced # not taken as a heading");
    check(strncmp(seen, "# Guide > ## Install\n\n## Install\n\nRun the build.", 47) == 0, "the model is given the heading path, then the chunk");
    check(text_(db, lo + (3 << 10), b, sizeof b) > 0 && strncmp(b, "## Arena allocator", 18) == 0, "the stored text has no heading path in front of it");
    check(vi_get_str_key(db, lo, "path", &v) == 0 && v == vi_key("notes/guide.md", -1) && vi_get_int(db, lo + (3 << 10), "year", &v) == 0 && v == 2024,
          "every row: the field path, and the front matter's year");
    check(vi_get_int(db, lo, "ok", &v) == -1, "a key the file does not have: no field");
    n = vi_search_hybrid(db, (const float *)0, "arena frees memory", -1, 1, ids, sc);
    check(n == 1 && ids[0] == lo + (3 << 10), "word search finds the arena section");
    check(vi_filter_parse(db, "lang = \"c\" and year >= 2020", &f) == 0 && vi_match(db, lo + (1 << 10), &f) == 1, "a filter on the front matter's fields");

    /* replaced whole; a bad file leaves everything as it was */
    check(vi_ingest_text(db, "notes/guide.md", "# Only\n\none section now\n", -1, VI_MD, &o, &st) == 0 && vi_count(db) == 1, "ingested again: the old rows gone, one new row");
    before = vi_count(db);
    check(vi_ingest_text(db, "data/bad.json", "[{\"a\": 1}, {\"b\": }]", -1, VI_JSON, &o, &st) == -1 && strstr(vi_error(), "at byte") != (const char *)0
          && vi_count(db) == before, "bad JSON: refused with the byte, nothing written (one batch)");

    /* JSON */
    check(vi_ingest_text(db, "data/books.json", js, -1, VI_JSON, &o, &st) == 0, "vi_ingest_text: a JSON array");
    check(st.docs == 4 && st.chunks == 3, "four elements, the empty one makes no row");
    vi_ingest_range("data/books.json", -1, &lo, &hi);
    check(text_(db, lo, b, sizeof b) > 0 && strstr(b, "author.name: Ann") && strstr(b, "author.tags: a\nauthor.tags: b"),
          "element 0: nested keys as dotted paths, an array's values each on a line");
    check(vi_get_str_key(db, lo, "author_name", &v) == 0 && v == vi_key("Ann", -1) && vi_get_float(db, lo, "score", &dv) == 0 && dv == 2.5
          && vi_get_int(db, lo, "year", &v) == 0 && v == 2001, "its fields: author.name as author_name, a float, an int");
    check(text_(db, lo + (1 << 10), b, sizeof b) > 0 && strstr(b, "title: caf\xc3\xa9 \xf0\x9f\x98\x80\nnew line") && strstr(b, "none: null")
          && vi_get_int(db, lo + (1 << 10), "ok", &v) == 0 && v == 1, "element 1: \\u00e9, a surrogate pair and \\n decoded; true as 1; null kept in the text");
    check(text_(db, lo + (3 << 10), b, sizeof b) > 0 && strstr(b, "deep.x.y: 1\ndeep.x.y: 2\ndeep.x.y: 3"), "element 3: arrays in arrays, the path without indices");

    /* JSON Lines, a file, removal */
    check(vi_ingest_text(db, "data/log.jsonl", jl, -1, VI_JSONL, &o, &st) == 0 && st.docs == 2 && st.chunks == 2, "JSON Lines: a value a line, the blank one skipped");
    sprintf(path, "/tmp/vecindex_ingest_%d.md", (int)getpid());
    {   FILE *fp = fopen(path, "w"); fputs("# File\n\nfrom the disk\n", fp); fclose(fp); }
    check(vi_ingest_file(db, path, &o, &st) == 0 && st.chunks == 1, "vi_ingest_file: by its extension");
    check(vi_ingest_file(db, "/tmp/x.pdf", &o, &st) == -1 && strstr(vi_error(), "turn them into") != (const char *)0, "another extension: refused, with what to do");
    n = (int)vi_ingest_remove(db, "data/books.json");
    check(n == 3 && vi_ingest_remove(db, path) == 1, "vi_ingest_remove: every row of a file, counted");
    remove(path);

    /* mutated input: refused or taken, never a crash or a read out of bounds (run under ASan in ./run's sanitizer pass) */
    {   static char m[2048];
        unsigned r = 12345u;
        int i, k, len, okc = 0, bad = 0;
        o.embed = (vi_embed_fn)0;
        for (i = 0; i < 3000; ++i) {
            const char *src = i % 2 ? js : md;
            len = (int)strlen(src);
            memcpy(m, src, (size_t)len);
            for (k = 0; k < 1 + i % 8; ++k) { r = r * 1103515245u + 12345u; m[(r >> 8) % (unsigned)len] = (char)(r >> 20); }
            r = r * 1103515245u + 12345u;
            len = (int)((r >> 8) % (unsigned)len) + 1;                   /* cut at a random length too */
            if (vi_ingest_text(db, "fuzz", m, len, i % 2 ? VI_JSON : VI_MD, &o, &st) == 0) ++okc; else ++bad;
        }
        sprintf(b, "3000 mutated inputs: %d taken, %d refused, no crash", okc, bad);
        check(okc + bad == 3000 && bad > 0, b); }

    /* the stand-in model */
    vi_embed_hash((void *)0, "the arena allocator frees memory", -1 + 33, q, DIM);
    vi_embed_hash((void *)0, "the arena allocator frees memory", 32, q2, DIM);
    {   int d, same = 1; double c = 0.0;
        for (d = 0; d < DIM; ++d) { same &= q[d] == q2[d]; c += (double)q[d] * q[d]; }
        check(same && c > 0.99 && c < 1.01, "vi_embed_hash: the same text the same unit vector"); }
    vi_close(db);
    printf("ingest: %s\n", fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}
