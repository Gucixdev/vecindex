/* chunk_test.c - vi_chunk and VI_ID through the C API: pages by form feed, the boundaries in their order, the size
 * cap, UTF-8 left whole, the overlap, and a search by pages and by documents over the chunks. ./run builds it against
 * libvecindex.a. */
#include "vecindex.h"
#include <stdio.h>
#include <string.h>

static int fails;

static void check(int cond, const char *what)
{
    printf(cond ? "  ok   %s\n" : "  FAIL %s\n", what);
    if (!cond) { ++fails; printf("       (%s)\n", vi_error()); }
}

static int is_(const char *t, const int *o, const int *l, int i, const char *want)
{
    return l[i] == (int)strlen(want) && memcmp(t + o[i], want, (size_t)l[i]) == 0;
}

int main(void)
{
    static char big[4096];
    int o[64], l[64], p[64], n, i, ok;
    const char *pages = "  first page.\n\n\fsecond page\f\f  fourth  \f";
    const char *para = "Alpha alpha alpha alpha alpha alpha alpha alpha.\n\nBeta beta beta beta beta beta beta beta beta beta.";

    check(vi_chunk("", 0, 64, 0, o, l, p, 64) == 0, "empty text: no chunks");
    check(vi_chunk("short", -1, 64, 0, o, l, p, 64) == 1 && is_("short", o, l, 0, "short"), "short text: one chunk, all of it");
    n = vi_chunk(pages, -1, 64, 0, o, l, p, 64);
    check(n == 3 && is_(pages, o, l, 0, "first page.") && p[0] == 0 && is_(pages, o, l, 1, "second page") && p[1] == 1
          && is_(pages, o, l, 2, "fourth") && p[2] == 3, "form feeds end pages; an empty page counts; white space trimmed");
    n = vi_chunk(para, -1, 64, 0, o, l, p, 64);
    check(n == 2 && is_(para, o, l, 0, "Alpha alpha alpha alpha alpha alpha alpha alpha."), "cut at the blank line, not mid-sentence");
    memset(big, 'x', 250); big[250] = 0;
    n = vi_chunk(big, -1, 100, 0, o, l, p, 64);
    check(n == 3 && l[0] == 100 && l[1] == 100 && l[2] == 50, "no boundary at all: cut at max_bytes");
    for (i = 0; i < 300; ++i) { big[2 * i] = (char)0xc5; big[2 * i + 1] = (char)0xbc; }   /* z with a dot, 2 bytes */
    big[600] = 0;
    n = vi_chunk(big, -1, 101, 0, o, l, p, 64);
    for (i = 0, ok = n > 1; i < n && ok; ++i) ok = ((unsigned char)big[o[i]] & 0xc0u) != 0x80u && l[i] % 2 == 0 && l[i] <= 101;
    check(ok, "UTF-8: never cut inside a character");
    for (i = 0; i < 40; ++i) memcpy(big + 10 * i, "word word ", 10);
    big[400] = 0;
    n = vi_chunk(big, -1, 100, 30, o, l, p, 64);
    for (i = 1, ok = n > 4; i < n && ok; ++i) ok = o[i] < o[i - 1] + l[i - 1] && o[i] >= o[i - 1] + l[i - 1] - 30;
    check(ok && o[n - 1] + l[n - 1] == 399, "overlap: each chunk starts at most 30 bytes back, the last ends at the text's end");
    check(vi_chunk(big, -1, 100, 0, o, l, p, 2) > 2, "more chunks than max: the count of all");
    check(vi_chunk(big, -1, 8, 0, o, l, p, 64) == -1 && vi_chunk(big, -1, 100, 60, o, l, p, 64) == -1, "bad arguments: -1");
    check(VI_ID(3, 2, 1) == ((3 << 20) | (2 << 10) | 1), "VI_ID: doc << 20 | page << 10 | chunk");

    {   /* chunks of two documents of two pages each, one dimension per (document, page): search by page and document */
        vi_db *db = vi_open((const char *)0, 4, 0);
        float v[4], q[4] = { 1.0f, 0.2f, 0.0f, 0.0f }, sc[8];
        vi_i64 ids[8];
        int d, pg, c;
        for (d = 0; d < 2; ++d)
            for (pg = 0; pg < 2; ++pg)
                for (c = 0; c < 3; ++c) {
                    memset(v, 0, sizeof v); v[2 * d + pg] = 1.0f; v[(2 * d + pg + 1) % 4] = 0.05f * (float)c;
                    vi_add(db, VI_ID(d, pg, c), v);
                }
        n = vi_search_grouped(db, q, 3, 10, ids, sc);
        check(n == 3 && (ids[0] >> 10) == ((0 << 10) | 0) && (ids[1] >> 10) == ((0 << 10) | 1), "shift 10: the best pages, one chunk each");
        n = vi_search_grouped(db, q, 2, 20, ids, sc);
        check(n == 2 && (ids[0] >> 20) == 0 && (ids[1] >> 20) == 1, "shift 20: the best documents");
        vi_close(db);
    }
    printf("chunks: %s\n", fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}
