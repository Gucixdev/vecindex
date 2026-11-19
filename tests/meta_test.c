/* meta_test.c - payload fields and filters through the C API (store_meta.c): the encodings and their order, the
 * errors, the filter language's logic, a filtered search through the built index against a brute-force answer,
 * the journal, a save, a compaction and a reopen. ./run builds it against libvecindex.a. */
#include "vecindex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#define DIM 16
#define N 4000

static int fails;

static void check(int cond, const char *what)
{
    printf(cond ? "  ok   %s\n" : "  FAIL %s\n", what);
    if (!cond) { ++fails; printf("       (%s)\n", vi_error()); }
}

static unsigned rng_ = 12345u;
static float rnd_(void) { rng_ = rng_ * 1103515245u + 12345u; return (float)((rng_ >> 8) & 0xffff) / 65536.0f - 0.5f; }

static const char *lang_[3] = { "c", "rust", "lean" };

/* row i: lang by i % 3, year 2000 + i % 30, score (i % 21 - 10) / 4.0, draft every 7th; row i % 11 == 0 has no lang */
static void fields_(vi_db *db, int i)
{
    if (i % 11) vi_set_str(db, i, "lang", lang_[i % 3], -1);
    vi_set_int(db, i, "year", 2000 + i % 30);
    vi_set_float(db, i, "score", (double)(i % 21 - 10) / 4.0);
    if (i % 7 == 0) vi_set_int(db, i, "draft", 1);
}

/* the expression of filter 0 by hand */
static int want0_(int i) { return i % 11 && i % 3 == 0 && 2000 + i % 30 >= 2015 && i % 7 != 0; }

static int brute_(const float *vs, const float *q, int (*ok)(int), vi_i64 *ids, int k)
{
    float best[16];
    int n = 0, i, j, d;
    for (i = 0; i < N; ++i) {
        float s = 0.0f, l = 0.0f;
        if (!ok(i)) continue;
        for (d = 0; d < DIM; ++d) { s += vs[i * DIM + d] * q[d]; l += vs[i * DIM + d] * vs[i * DIM + d]; }
        s /= (float)sqrt((double)l);
        if (n == k && s <= best[k - 1]) continue;
        j = n < k ? n++ : k - 1;
        while (j > 0 && best[j - 1] < s) { best[j] = best[j - 1]; ids[j] = ids[j - 1]; --j; }
        best[j] = s; ids[j] = i;
    }
    return n;
}

int main(void)
{
    char path[128], line[256], nm[32];
    float *vs = (float *)malloc((size_t)N * DIM * sizeof(float)), q[DIM], sc[16];
    vi_i64 ids[16], bid[16], v;
    double dv;
    vi_filter f, g;
    vi_db *db;
    int i, j, n, nb, ok, ty, hit;

    sprintf(path, "/tmp/vecindex_meta_%d.vxi", (int)getpid());
    remove(path);
    db = vi_open(path, DIM, 0);
    check(db != (vi_db *)0, "a store on the disk");
    if (!db || !vs) return 1;
    for (i = 0; i < N * DIM; ++i) vs[i] = rnd_();
    for (i = 0; i < N; ++i) vi_add(db, i, vs + (size_t)i * DIM);
    for (i = 0; i < N; ++i) fields_(db, i);

    /* values back, and the errors */
    check(vi_get_int(db, 45, "year", &v) == 0 && v == 2015, "vi_get_int: the year set");
    check(vi_get_float(db, 3, "score", &dv) == 0 && dv == -1.75, "vi_get_float: a negative float back exact");
    check(vi_get_str_key(db, 4, "lang", &v) == 0 && v == vi_key("rust", -1), "vi_get_str_key: the string's vi_key");
    check(vi_get_str_key(db, 22, "lang", &v) == 1, "a row that never set lang: 1, not set");
    check(vi_get_int(db, 1, "draft", &v) == 1, "draft set on other rows only: 1 here");
    check(vi_get_int(db, 0, "lang", &v) == -1, "vi_get_int on a string field: -1");
    check(vi_set_int(db, 1, "lang", 5) == -1, "vi_set_int on a string field: refused");
    check(vi_set_int(db, 1, "year", -(vi_i64)0x7fffffff * (vi_i64)0x100000000 - (vi_i64)0x100000000) == -1, "INT64_MIN: refused");
    {   volatile double z = 0.0;
        check(vi_set_float(db, 1, "score", z / z) == -1, "NaN: refused"); }
    check(vi_set_int(db, 1, "and", 1) == -1 && vi_set_int(db, 1, "9x", 1) == -1, "a filter word or a digit first: not a name");
    check(vi_set_int(db, 99999, "year", 1) == -1, "a row that is not there: -1");
    check(vi_set_float(db, 5, "score", -0.0) == 0 && vi_get_float(db, 5, "score", &dv) == 0 && dv == 0.0, "-0 stored as 0");
    vi_set_float(db, 5, "score", (double)(5 % 21 - 10) / 4.0);
    for (i = 0, ok = 0; i < 20; ++i) { sprintf(nm, "f%d", i); ok += vi_field(db, nm, VI_INT) == 0; }
    check(ok == 12, "16 fields a store: 4 used, 12 more, then refused");
    check(vi_field_info(db, 0, nm, &ty) == 0 && strcmp(nm, "year") == 0 && ty == VI_INT && vi_field_info(db, 16, nm, &ty) == -1,
          "vi_field_info: the first declared is year, an int; 16 past the last");
    check(vi_unset(db, 7, "draft") == 0 && vi_get_int(db, 7, "draft", &v) == 1, "vi_unset: the row has it no more");
    vi_set_int(db, 7, "draft", 1);

    /* the language */
    check(vi_filter_parse(db, "lang = \"c\" and year >= 2015 and not draft", &f) == 0, "parse: lang = \"c\" and year >= 2015 and not draft");
    for (i = 0, ok = 1; i < N; ++i) if (vi_match(db, i, &f) != want0_(i)) { ok = 0; break; }
    check(ok, "vi_match on every row: as the expression by hand");
    check(vi_filter_parse(db, "year < 2001 or year > 2028 and draft", &g) == 0, "parse: and binds tighter than or");
    for (i = 0, ok = 1; i < N; ++i) if (vi_match(db, i, &g) != (2000 + i % 30 < 2001 || (2000 + i % 30 > 2028 && i % 7 == 0))) { ok = 0; break; }
    check(ok, "a or b and c is a or (b and c)");
    vi_filter_parse(db, "(year < 2001 || year > 2028) && !draft", &g);
    for (i = 0, ok = 1; i < N; ++i) if (vi_match(db, i, &g) != ((2000 + i % 30 < 2001 || 2000 + i % 30 > 2028) && i % 7 != 0)) { ok = 0; break; }
    check(ok, "( ) && || ! as well");
    vi_filter_parse(db, "lang != \"c\"", &g);
    check(vi_match(db, 22, &g) == 0 && vi_match(db, 1, &g) == 1, "!= on a row without the field: false");
    vi_filter_parse(db, "not lang = \"c\"", &g);
    check(vi_match(db, 22, &g) == 1 && vi_match(db, 3, &g) == 0, "not ... = on a row without the field: true");
    vi_filter_parse(db, "lang in (\"lean\", \"rust\") and year not in (2000, 2001, 2002)", &g);
    for (i = 0, ok = 1; i < N; ++i) if (vi_match(db, i, &g) != (i % 11 && i % 3 != 0 && i % 30 > 2)) { ok = 0; break; }
    check(ok, "in and not in");
    vi_filter_parse(db, "score > -1.5 and score <= 0", &g);
    for (i = 0, ok = 1; i < N; ++i) { double x = (double)(i % 21 - 10) / 4.0; if (vi_match(db, i, &g) != (x > -1.5 && x <= 0.0)) { ok = 0; break; } }
    check(ok, "float ranges across 0 and the negatives, in order");
    vi_filter_parse(db, "has lang and not has draft", &g);
    check(vi_match(db, 1, &g) == 1 && vi_match(db, 22, &g) == 0 && vi_match(db, 7, &g) == 0, "has");
    check(vi_filter_parse(db, "lang < \"c\"", &g) == -1, "a string field with <: refused");
    check(vi_filter_parse(db, "nope = 1", &g) == -1, "a field that is not there: refused");
    check(vi_filter_parse(db, "(year = 1", &g) == -1 && vi_filter_parse(db, "year = 1)", &g) == -1, "unbalanced ( ): refused");
    check(vi_filter_parse(db, "year = 1.5", &g) == -1 && vi_filter_parse(db, "year = 1 and", &g) == -1, "a fraction for an int, a dangling and: refused");
    check(vi_filter_parse(db, "lang", &g) == -1 && strstr(vi_error(), "byte") != (const char *)0, "a string field alone: refused, the error says where");

    /* the search: through the scan index, against brute force */
    vi_build(db);
    for (j = 0, ok = 1, hit = 0; j < 20; ++j) {
        int a, b;
        for (i = 0; i < DIM; ++i) q[i] = rnd_();
        n = vi_search_filter(db, q, 10, &f, ids, sc);
        nb = brute_(vs, q, want0_, bid, 10);
        for (a = 0; a < n; ++a) if (!want0_((int)ids[a])) ok = 0;
        for (a = 0; a < nb && a < 10; ++a) for (b = 0; b < n; ++b) if (bid[a] == ids[b]) { ++hit; break; }
        if (n != 10) ok = 0;
    }
    check(ok, "vi_search_filter after vi_build: 10 hits a query, every one passing the filter");
    sprintf(line, "vi_search_filter recall@10 against brute force over the passing rows: %d/200", hit);
    check(hit >= 180, line);

    /* durability: the journal, then a save, a compaction, a reopen */
    vi_close(db);
    db = vi_open(path, DIM, 0);
    check(db && vi_get_int(db, 45, "year", &v) == 0 && v == 2015 && vi_get_str_key(db, 4, "lang", &v) == 0 && v == vi_key("rust", -1),
          "reopened: the fields from the journal");
    check(db && vi_filter_parse(db, "lang = \"c\" and year >= 2015 and not draft", &f) == 0, "a filter compiled for the reopened store");
    for (i = 0, ok = 1; db && i < N; ++i) if (vi_match(db, i, &f) != want0_(i)) { ok = 0; break; }
    check(ok, "every row matches as before");
    for (i = 0; db && i < N; i += 2) vi_delete(db, i);           /* half the rows go: the save compacts */
    check(db && vi_save(db, (const char *)0) == 0, "vi_save over the deleted rows (a compaction)");
    vi_close(db);
    db = vi_open(path, DIM, 0);
    for (i = 1, ok = db != (vi_db *)0; ok && i < N; i += 2) ok = vi_get_int(db, i, "year", &v) == 0 && v == 2000 + i % 30;
    check(ok, "reopened from the file: every kept row's year where it belongs");
    vi_filter_parse(db, "lang = \"c\" and year >= 2015 and not draft", &f);
    for (i = 1, ok = 1; db && i < N; i += 2) if (vi_match(db, i, &f) != want0_(i)) { ok = 0; break; }
    check(ok, "and the filter on them");
    check(db && vi_update(db, 3, vs) == 0 && vi_get_int(db, 3, "year", &v) == 0 && v == 2003, "vi_update keeps the fields");
    check(db && vi_add(db, 3, vs) == 0 && vi_get_int(db, 3, "year", &v) == 1, "vi_add over the row starts it over: no fields");
    {   vi_db *o = vi_open((const char *)0, DIM, 0);
        check(o && vi_match(o, 1, &f) == -1, "a filter of another store: refused");
        vi_close(o); }
    vi_close(db);
    remove(path);
    sprintf(line, "%s.j", path); remove(line);
    free(vs);
    printf("fields: %s\n", fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}
