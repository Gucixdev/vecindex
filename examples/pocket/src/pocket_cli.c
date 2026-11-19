/* pocket on the desktop: the cleanup over a folder of photos, for trying it before the phone app.
 *
 *   pocket add DB PHOTO [PNM]     describe PHOTO (its pixels from PNM when given: a .ppm/.pgm made from a JPEG by
 *                                 bin/pocket-scan) and store it; a photo already in DB is skipped
 *   pocket dups DB [same|similar|COSINE] [-t]
 *                                 the groups, the one to keep and what deleting the rest frees; -t: TSV
 *                                 (group, keep|delete, bytes, path)
 *
 * Images come in as binary PNM (P5 grey, P6 colour, 8 bits): the format is a header and the pixels, so no decoder is
 * needed here; on Android the platform decodes. Memory: mmap only (the file, and one block per command). */
#define _DEFAULT_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "pocket.h"

static void *pk_map_(size_t n)
{
    void *p = mmap(NULL, n ? n : 1, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}

/* A PNM header: P5 or P6, width, height, maxval 255, comments allowed; the pixels' offset, or -1. */
static long pk_pnm_(const unsigned char *s, long n, int *w, int *h, int *ch)
{
    long i = 2;
    int f, v[3];
    if (n < 3 || s[0] != 'P' || (s[1] != '5' && s[1] != '6')) return -1;
    *ch = s[1] == '6' ? 3 : 1;
    for (f = 0; f < 3; ++f) {
        long x = 0;
        for (;;) {
            while (i < n && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
            if (i < n && s[i] == '#') { while (i < n && s[i] != '\n') ++i; continue; }
            break;
        }
        if (i >= n || s[i] < '0' || s[i] > '9') return -1;
        while (i < n && s[i] >= '0' && s[i] <= '9' && x < 1000000L) x = x * 10 + (s[i++] - '0');
        v[f] = (int)x;
    }
    if (i >= n || v[2] != 255 || v[0] < 1 || v[1] < 1 || v[0] > 65535 || v[1] > 65535) return -1;
    ++i;                                            /* the one whitespace before the pixels */
    *w = v[0];
    *h = v[1];
    if (n - i < (long)*w * *h * *ch) return -1;
    return i;
}

static int pk_add_(vi_db *db, const char *photo, const char *pnm)
{
    struct stat st, sp;
    int fd, w, h, ch, k;
    long off, i, np;
    unsigned char *m, *rgba;
    pk_info in;
    vi_i64 id = 0, ids[4], have[256], from = 0;
    float sc[4];
    char got[4096];
    if (stat(photo, &sp) != 0) { fprintf(stderr, "pocket: %s: no such file\n", photo); return 1; }
    fd = open(pnm, O_RDONLY);
    if (fd < 0 || fstat(fd, &st) != 0 || st.st_size < 8) {
        fprintf(stderr, "pocket: %s: cannot read\n", pnm);
        if (fd >= 0) close(fd);
        return 1;
    }
    m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) return 1;
    off = pk_pnm_(m, (long)st.st_size, &w, &h, &ch);
    if (off < 0) { fprintf(stderr, "pocket: %s: not a binary 8-bit PNM (P5/P6)\n", pnm); munmap(m, (size_t)st.st_size); return 1; }
    np = (long)w * h;
    rgba = pk_map_((size_t)np * 4);
    if (!rgba) { munmap(m, (size_t)st.st_size); return 1; }
    for (i = 0; i < np; ++i) {
        const unsigned char *p = m + off + i * ch;
        rgba[4 * i] = p[0];
        rgba[4 * i + 1] = p[ch == 3];
        rgba[4 * i + 2] = p[ch == 3 ? 2 : 0];
        rgba[4 * i + 3] = 255;
    }
    munmap(m, (size_t)st.st_size);
    k = pk_photo(rgba, w, h, 4 * w, &in);
    munmap(rgba, (size_t)np * 4);
    if (k != 0) { fprintf(stderr, "pocket: %s: too small\n", pnm); return 1; }
    /* already there: its own descriptor is its nearest, with its path */
    k = vi_count(db) ? vi_search(db, in.vec, 4, ids, sc) : 0;
    for (i = 0; i < k; ++i) {
        int n;
        if (sc[i] < 0.9999f) break;
        n = vi_get_text(db, ids[i], got, sizeof got - 1);
        if (n >= 0 && n < (int)sizeof got && (got[n] = 0, strcmp(got, photo) == 0)) return 0;
    }
    for (;;) {                                      /* the next id: the highest + 1 */
        int g = vi_range(db, from, (vi_i64)0x7ffffffe, have, 256);
        if (g <= 0) break;
        id = have[g - 1] + 1;
        from = id;
        if (g < 256) break;
    }
    {   /* one batch: the row and its fields together, or nothing */
        vi_batch *b = vi_batch_begin(db);
        if (!b || vi_batch_add_text(b, id, in.vec, photo, (int)strlen(photo)) != 0
            || vi_batch_set_int(b, id, "w", in.w) != 0 || vi_batch_set_int(b, id, "h", in.h) != 0
            || vi_batch_set_float(b, id, "sharp", in.sharp) != 0
            || vi_batch_set_int(b, id, "bytes", (vi_i64)sp.st_size) != 0
            || vi_batch_set_int(b, id, "time", (vi_i64)sp.st_mtime) != 0) {
            fprintf(stderr, "pocket: %s: %s\n", photo, vi_error());
            if (b) vi_batch_abort(b);
            return 1;
        }
        if (vi_batch_commit(b, 0) < 0) {            /* it frees the batch either way */
            fprintf(stderr, "pocket: %s: %s\n", photo, vi_error());
            return 1;
        }
    }
    return 0;
}

static int pk_dups_(vi_db *db, float thr, int tsv)
{
    vi_i64 n = 0, from = 0, have[256], freed = 0, total = 0;
    int *parent, *cnt, *order, *w, *h, g, i, groups = 0;
    float *sh, *vv;
    vi_i64 *by;
    size_t bytes;
    char path[4096];
    for (;;) {
        int got = vi_range(db, from, (vi_i64)0x7ffffffe, have, 256);
        if (got <= 0) break;
        n = have[got - 1] + 1;
        from = n;
        if (got < 256) break;
    }
    bytes = (size_t)n * (sizeof(vi_i64) + (PK_DIM + 1) * sizeof(float) + 5 * sizeof(int)) + 64;
    by = pk_map_(bytes);
    if (!by) return 1;
    vv = (float *)(by + n);
    sh = vv + (size_t)n * PK_DIM;
    parent = (int *)(sh + n);
    cnt = parent + n;
    order = cnt + n;
    w = order + n;
    h = w + n;
    if (pk_groups(db, thr, 16, n, parent) != 0) { fprintf(stderr, "pocket: %s\n", vi_error()); return 1; }
    /* the members of each group together, in id order: a counting sort by root */
    for (i = 0; i < n; ++i) parent[i] = pk_root(parent, i);     /* every one straight to its root */
    for (i = 0; i < n; ++i) cnt[parent[i]]++;
    {
        int s = 0, c;
        for (i = 0; i < n; ++i) { c = cnt[i]; cnt[i] = s; s += c; }
        for (i = 0; i < n; ++i) order[cnt[parent[i]]++] = i;
    }
    for (i = 0; i < n; i = g) {
        int r = parent[order[i]], m, keep, j;
        vi_i64 v;
        double d;
        for (g = i; g < n && parent[order[g]] == r; ++g) {
            w[g - i] = h[g - i] = 0;
            sh[g - i] = 0;
            if (vi_get_int(db, order[g], "w", &v) == 0) w[g - i] = (int)v;
            if (vi_get_int(db, order[g], "h", &v) == 0) h[g - i] = (int)v;
            if (vi_get_float(db, order[g], "sharp", &d) == 0) sh[g - i] = (float)d;
            by[g - i] = 0;
            vi_get_int(db, order[g], "bytes", &by[g - i]);
            if (vi_get(db, order[g], vv + (size_t)(g - i) * PK_DIM) != 0) w[g - i] = 0;
        }
        m = g - i;
        if (m < 2 || w[0] == 0) continue;           /* alone, or an id without a row */
        keep = pk_keep(m, w, h, sh, by, vv);
        ++groups;
        if (!tsv) printf("group %d (%d photos)\n", groups, m);
        for (j = 0; j < m; ++j) {
            int id = order[i + j], k;
            k = vi_get_text(db, id, path, sizeof path - 1);
            path[k < 0 ? 0 : k < (int)sizeof path ? k : (int)sizeof path - 1] = 0;
            if (j != keep) freed += by[j];
            total += by[j];
            if (tsv) printf("%d\t%s\t%ld\t%s\n", groups, j == keep ? "keep" : "delete", (long)by[j], path);
            else printf("  %-6s  %s  (%dx%d, sharpness %.3f, %.0f KB)\n", j == keep ? "keep" : "delete", path,
                        w[j], h[j], sh[j], by[j] / 1024.0);
        }
    }
    if (!tsv) printf("%d groups; deleting the rest frees %.1f MB of %.1f MB in them\n", groups, freed / 1048576.0,
                     total / 1048576.0);
    munmap(by, bytes);
    return 0;
}

int main(int argc, char **argv)
{
    vi_db *db;
    int r = 2;
    if (argc >= 4 && strcmp(argv[1], "add") == 0) {
        db = vi_open(argv[2], PK_DIM, 0);
        if (!db) { fprintf(stderr, "pocket: %s: %s\n", argv[2], vi_error()); return 1; }
        r = pk_add_(db, argv[3], argc >= 5 ? argv[4] : argv[3]);
        vi_close(db);
    } else if (argc >= 3 && strcmp(argv[1], "dups") == 0) {
        float thr = PK_SAME;
        int tsv = 0, i;
        for (i = 3; i < argc; ++i) {
            if (strcmp(argv[i], "-t") == 0) tsv = 1;
            else if (strcmp(argv[i], "same") == 0) thr = PK_SAME;
            else if (strcmp(argv[i], "similar") == 0) thr = PK_SIMILAR;
            else thr = (float)atof(argv[i]);
        }
        if (thr <= 0 || thr > 1) { fprintf(stderr, "pocket: the cosine is in (0, 1]\n"); return 2; }
        db = vi_open(argv[2], 0, 0);
        if (!db) { fprintf(stderr, "pocket: %s: %s\n", argv[2], vi_error()); return 1; }
        r = pk_dups_(db, thr, tsv);
        vi_close(db);
    } else {
        fprintf(stderr, "usage: pocket add DB PHOTO [PNM]\n       pocket dups DB [same|similar|COSINE] [-t]\n");
    }
    return r;
}
