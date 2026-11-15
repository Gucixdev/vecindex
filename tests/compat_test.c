/* compat_test.c - the files of every release open in this one, with every value as it was written.
 *
 * tests/compat/NAME.vxi.gz (and NAME.vxi.j.gz) are stores written by released builds and by this one (tests/compat/make.c,
 * values from tests/compat/compat.h): rows saved in the file, more changes in the journal only. Each is unpacked to a
 * temporary directory and opened; checked: the count, every row's vector (to 1e-5, or to the int8 rows' precision),
 * every text, every tag and label, the numbers and namespaces of files that have them, the nearest row of a row's own
 * vector, the deleted rows gone, and a clean scrub. Then the other way: a file of a newer build - a feature flag this
 * build does not know, a file version past its own - is refused with a message that says so, never misread.
 *
 *   gcc -std=gnu99 -O2 -Iinclude -Itests/compat tests/compat_test.c src/vecindex.c -lm -lpthread -o compat_test
 *   ./compat_test tests/compat
 *
 * A release that writes a new file version adds its files here (make.c built against it); none is ever removed. */
#define _XOPEN_SOURCE 700                                         /* mkdtemp */
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "vecindex.h"
#include "compat.h"

static int fails;
static void check(int ok, const char *what, const char *file)
{
    printf("  %s %s: %s\n", ok ? "ok  " : "FAIL", file, what);
    if (!ok) ++fails;
}

static int deleted_(vi_i64 id) { return id == 12 || (id <= CSAVED && id % 50 == 7); }

static float cos_(const float *a, const float *b)
{
    double ab = 0, aa = 0, bb = 0;
    int i;
    for (i = 0; i < CDIM; ++i) { ab += (double)a[i] * b[i]; aa += (double)a[i] * a[i]; bb += (double)b[i] * b[i]; }
    return (float)(ab / sqrt(aa * bb));
}

static void one_(const char *dir, const char *name)
{
    char tmp[64], cmd[8192], path[600], t[200], want[96], nm[128];
    float v[CDIM], g[CDIM], sc[3];
    vi_i64 id, ids[3], tag, x, rows[64];
    float rs[64];
    vi_db *db;
    int i8 = strstr(name, "i8") != (char *)0, texts = strstr(name, "plain") == (char *)0, now = !strncmp(name, "v10", 3);
    int okv = 1, okt = 1, okg = 1, okd = 1, oks = 1, okn = 1, okl = 1, n, j;
    float tol = i8 ? 0.995f : 0.999999f;
    strcpy(tmp, "/tmp/vecindex_compat_XXXXXX");
    if (!mkdtemp(tmp)) { check(0, "a temporary directory", name); return; }
    if (strlen(dir) > 1000 || strlen(name) > 100) { check(0, "a short path", name); return; }
    strcpy(nm, name);
    sprintf(cmd, "gzip -dc '%s/%s.vxi.gz' > '%s/%s.vxi' && { ! test -f '%s/%s.vxi.j.gz' || gzip -dc '%s/%s.vxi.j.gz' > '%s/%s.vxi.j'; }",
            dir, nm, tmp, nm, dir, nm, dir, nm, tmp, nm);
    if (system(cmd) != 0) { check(0, "unpacked", name); return; }
    sprintf(path, "%s/%s.vxi", tmp, nm);
    db = vi_open(path, 0, 0);
    check(db != (vi_db *)0, db ? "opens" : vi_error(), name);
    if (!db) return;
    check(vi_count(db) == CALL - 6 - 1, "the count: 340 written, 7 deleted (6 before the save, 1 in the journal)", name);
    for (id = 1; id <= CALL; ++id) {
        int got = vi_get(db, id, g) == 0;
        if (deleted_(id)) { if (got) okd = 0; continue; }
        if (!got) { okv = 0; continue; }
        vec_(id == 13 ? 1000 : id, v);
        if (cos_(v, g) < tol) okv = 0;
        if (vi_get_tag(db, id, &tag) != 0 || (tag & ~(vi_i64)7) != (id == 13 ? 0 : tag_(id))) okg = 0;   /* 13: replaced, a new row */
        if (id <= CSAVED && id != 13 && id % 5 == 0 && !(tag & 7)) okg = 0;   /* a label's bit */
        if (texts && id % 3 == 0 && id != 13) {
            text_(id, want);
            n = vi_get_text(db, id, t, (int)sizeof(t));
            if (n != (int)strlen(want) || memcmp(t, want, (size_t)n)) okt = 0;
        }
        if (now) {
            if (vi_get_num(db, id, &x) != 0 || x != (id == 13 ? 0 : num_(id))) okn = 0;
            if (vi_get_ns(db, id, &x) != 0 || x != ns_(id)) okn = 0;
        }
        if (id % 17 == 0 || id == 13 || id > CSAVED) {
            n = vi_search(db, v, 3, ids, sc);
            if (n < 1 || ids[0] != id) oks = 0;
        }
    }
    check(okv, i8 ? "every row's vector (int8 rows: cosine >= 0.995)" : "every row's vector (cosine >= 0.999999)", name);
    check(okd, "the deleted rows gone (before the save and in the journal)", name);
    check(okg, "every row's tag, and the labels' bits", name);
    if (texts) check(okt, "every text", name);
    check(oks, "a row's own vector finds it first (the saved rows, the replaced one, the journal's)", name);
    n = vi_search_labeled(db, v, 64, "pet", (const char *)0, (const char *)0, rows, rs);
    for (j = 0; j < n; ++j) if (rows[j] > CSAVED || rows[j] % 5) okl = 0;
    if (n < 10) okl = 0;
    n = vi_search_labeled(db, v, 64, "dog", (const char *)0, (const char *)0, rows, rs);
    for (j = 0; j < n; ++j) if (rows[j] % 10) okl = 0;
    check(okl, "labels: pet finds its rows and dog (under pet) only its own", name);
    if (now) {
        check(okn, "every row's number and namespace", name);
        n = vi_search_ns(db, 11, v, 64, rows, rs);
        for (j = 0, okn = n > 10; j < n; ++j) if (rows[j] % 4 != 1) okn = 0;
        check(okn, "a namespace's search sees only its rows", name);
    }
    vi_close(db);
    {   vi_i64 fixed = 0;
        check(vi_scrub(path, 0, &fixed) == 0, "the file's blocks all check out (scrub)", name); }
    sprintf(cmd, "rm -rf '%s'", tmp);
    if (system(cmd) != 0) {}
}

/* a newer build's file: a flag, then a version this build does not know */
static void newer_(const char *dir)
{
    char tmp[64], cmd[1024], path[200];
    FILE *f;
    int ext0, ver = 11;
    vi_db *db;
    strcpy(tmp, "/tmp/vecindex_compat_XXXXXX");
    if (!mkdtemp(tmp)) return;
    sprintf(path, "%s/n.vxi", tmp);
    sprintf(cmd, "gzip -dc '%s/v10-0.4.0-cos.vxi.gz' > '%s'", dir, path);
    if (system(cmd) != 0) { check(0, "unpacked", "newer"); return; }
    vi_set_verify(0);                            /* the edit below is not under the checksums: unchecked, the flag is read */
    f = fopen(path, "r+b");
    fseek(f, 16, SEEK_SET); if (fread(&ext0, 4, 1, f) != 1) {}
    ext0 |= 1 << 12;
    fseek(f, 16, SEEK_SET); fwrite(&ext0, 4, 1, f); fclose(f);
    db = vi_open(path, 0, 0);
    check(!db && strstr(vi_error(), "newer vecindex") && strstr(vi_error(), "0x1000"), db ? "opened!" : vi_error(), "a feature flag of a newer build");
    if (db) vi_close(db);
    f = fopen(path, "r+b");
    fseek(f, 4, SEEK_SET); fwrite(&ver, 4, 1, f); fclose(f);
    db = vi_open(path, 0, 0);
    check(!db && strstr(vi_error(), "file version 11"), db ? "opened!" : vi_error(), "a file version past this build's");
    if (db) vi_close(db);
    vi_set_verify(4);
    {   /* a journal with a change of a kind this build does not know, in a frame whose CRC vouches for it: refused,
         * and the journal left exactly as it was (a newer build's changes are not cut off as a torn tail) */
        char jp[220];
        unsigned char fr[9 + 17];
        unsigned int crc = 0xffffffffu, len = 17;
        long before, after;
        int i, k;
        sprintf(path, "%s/j.vxi", tmp); sprintf(jp, "%s.j", path);
        sprintf(cmd, "gzip -dc '%s/v9-0.4.0-plain.vxi.gz' > '%s' && gzip -dc '%s/v9-0.4.0-plain.vxi.j.gz' > '%s'", dir, path, dir, jp);
        if (system(cmd) != 0) { check(0, "unpacked", "newer journal"); return; }
        memset(fr, 0, sizeof(fr));
        fr[9] = 'Z';                                             /* op Z, an id and a value of zeros */
        for (i = 9; i < 9 + 17; ++i) { crc ^= fr[i]; for (k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0x82f63b78u & (0u - (crc & 1u))); }
        crc ^= 0xffffffffu;
        fr[0] = 'G'; memcpy(fr + 1, &len, 4); memcpy(fr + 5, &crc, 4);
        f = fopen(jp, "ab"); fwrite(fr, 1, sizeof(fr), f); fclose(f);
        f = fopen(jp, "rb"); fseek(f, 0, SEEK_END); before = ftell(f); fclose(f);
        db = vi_open(path, 0, 0);
        f = fopen(jp, "rb"); fseek(f, 0, SEEK_END); after = ftell(f); fclose(f);
        check(!db && strstr(vi_error(), "journal was written by a newer vecindex") && strstr(vi_error(), "'Z'") && before == after,
              db ? "opened!" : vi_error(), "a journal change of a newer build");
        if (db) vi_close(db);
    }
    sprintf(cmd, "rm -rf '%s'", tmp);
    if (system(cmd) != 0) {}
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : "tests/compat";
    DIR *d = opendir(dir);
    struct dirent *e;
    char names[32][128];
    int nn = 0, i;
    printf("file compatibility (%s)\n", dir);
    if (!d) { printf("  FAIL no %s\n", dir); return 1; }
    while ((e = readdir(d)) != (struct dirent *)0 && nn < 32) {
        size_t l = strlen(e->d_name);
        if (l > 7 && !strcmp(e->d_name + l - 7, ".vxi.gz") && l - 7 < 128) { memcpy(names[nn], e->d_name, l - 7); names[nn++][l - 7] = 0; }
    }
    closedir(d);
    qsort(names, (size_t)nn, sizeof(names[0]), (int (*)(const void *, const void *))strcmp);
    for (i = 0; i < nn; ++i) one_(dir, names[i]);
    newer_(dir);
    printf(fails ? "compat: FAILED (%d)\n" : "compat: OK (%d stores)\n", fails ? fails : nn);
    return fails != 0;
}
