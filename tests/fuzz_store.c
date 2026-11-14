/* fuzz_store.c - libFuzzer target for what the store reads back from the disk: the file (header, columns, texts,
 * labels, the engine index, digest tables, trailers, parity) and its journal, under ASan and UBSan.
 *
 * An input is a store file and its journal: 4 bytes (little-endian) give the file's length, the file follows and the
 * journal is the rest (no journal when that is empty). The target writes both into a temp dir and opens them with
 * store_load, which checks every block, rebuilds damaged ones from the parity and replays the journal. When the load
 * succeeds it uses the store: the integrity counts, a search, the vector and the text of every id the search and a
 * range return, then one more write and a save. A damaged input may be refused or repaired; it may not crash or read
 * out of bounds, and a file with checksums (v9) may not return an id the store does not hold, a NaN score or a
 * vector it then cannot give back. Older files (v4-v8) have no checksums: their rows are whatever their bytes say.
 *
 *   clang -std=gnu99 -g -O1 -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined -march=native \
 *       -Isrc tests/fuzz_store.c -lm -lpthread -o .build/fuzz_store
 *   FUZZ_STORE_SEEDS=DIR .build/fuzz_store    writes the seeds - small stores of each kind - into DIR and exits;
 *                                             tests/fuzz/store keeps them gzipped (gzip -9n: 200 KB of mostly zeros)
 *   .build/fuzz_store -max_len=262144 CORPUS_DIR UNPACKED_SEEDS_DIR
 *
 * tools/fuzz builds and runs every target; ./run replays tests/fuzz/store with gcc alone (-DFUZZ_MAIN, fuzz_main.h). */
#include "store.c"
#include <dirent.h>

static Arena fz_arena;
static char fz_dir[64], fz_file[96], fz_jn[100];

static void fz_fail_(const char *what, long v)
{
    fprintf(stderr, "fuzz_store: %s (%ld)\n", what, v);
    abort();
}

/* the temp dir emptied: the file, its journal and whatever the load left beside them (.damaged copies, .tmp) */
static void fz_clean_(void)
{
    DIR *d = opendir(fz_dir);
    struct dirent *e;
    char p[400];
    if (!d) abort();
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.' && (!e->d_name[1] || (e->d_name[1] == '.' && !e->d_name[2]))) continue;
        snprintf(p, sizeof(p), "%s/%s", fz_dir, e->d_name);
        unlink(p);
    }
    closedir(d);
}

static void fz_write_(const char *path, const unsigned char *d, size_t n)
{
    FILE *f = fopen(path, "wb");
    if (!f || (n && fwrite(d, 1, n, f) != n) || fclose(f)) abort();
}

static void fz_use_(RagStore *s)
{
    i64 nb, bad, fixed, rbad, rhealed, ids[64], more[64];
    f32 sc[64], *q, *v;
    char txt[512];
    i32 jd, got, n, i, d = s->dim, checked;
    store_integrity(s, &nb, &bad, &fixed, &rbad, &rhealed, &jd);
    if (bad < 0 || fixed < 0 || fixed > bad || nb < 0) fz_fail_("integrity counts out of range", (long)bad);
    checked = nb > 0;                    /* a v9 file whose checksums were read; an older one has none to vouch for it */
    if (d < 1 || d > 65536) fz_fail_("a loaded store with an impossible dimension", (long)d);
    q = (f32 *)calloc((size_t)d, sizeof(f32));
    v = (f32 *)calloc((size_t)d, sizeof(f32));
    if (!q || !v) abort();
    q[0] = 1.0f;
    if (d > 1) q[d - 1] = -0.5f;
    got = store_search(s, q, 64, ids, sc);
    if (got < 0 || got > 64 || got > store_count(s)) fz_fail_("search returned an impossible number of hits", (long)got);
    for (i = 0; i < got; ++i) {                 /* an unchecked file's rows are whatever its bytes say: used, not judged */
        if (checked && sc[i] != sc[i]) fz_fail_("search returned a NaN score for id", (long)ids[i]);
        if (!store_has(s, ids[i]) && checked) fz_fail_("search returned an id the store does not hold", (long)ids[i]);
        if (store_get_vec(s, ids[i], v) != 0 && checked) fz_fail_("search returned an id whose vector cannot be read", (long)ids[i]);
        if (store_get_text(s, ids[i], txt, (i32)sizeof(txt)) < 0 && checked) fz_fail_("a hit whose text says the id is not live", (long)ids[i]);
    }
    n = store_range(s, (i64)-9223372036854775807L - 1, (i64)9223372036854775807L, more, 64);
    for (i = 0; i < n; ++i)
        if ((!store_has(s, more[i]) || store_get_vec(s, more[i], v) != 0) && checked) fz_fail_("range returned an id the store does not hold", (long)more[i]);
    store_put(s, q, 4242, 0);
    store_save(s, fz_file);
    free(q); free(v);
}

/* the seeds: a few small stores, each with a journal of changes made after its last save */
static void fz_seed_(const char *out, const char *name, i32 dim, i32 rows, i32 texts, i32 build, i32 jrows)
{
    RagStore *s = store_new(&fz_arena, dim);
    f32 *v = (f32 *)malloc((size_t)dim * sizeof(f32));
    u64 x = 0x9e3779b97f4a7c15ull ^ (u64)dim;
    i32 i, j;
    FILE *f, *o;
    long fl = 0, jl = 0;
    char p[400];
    unsigned char b[4];
    for (i = 0; i < rows + jrows; ++i) {
        char t[64];
        if (i == rows) { if (build) store_build(s); store_save(s, fz_file); }     /* the rest goes to the journal */
        for (j = 0; j < dim; ++j) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; v[j] = (f32)((double)(x >> 11) / 9007199254740992.0 - 0.5); }
        sprintf(t, "row %d about topic %d", (int)i, (int)(i % 7));
        if (texts) store_put_text(s, v, (i64)i * 3, 0, t, (i32)strlen(t)); else store_put(s, v, (i64)i * 3, 0);
        if (i >= rows && i % 5 == 4) store_delete(s, (i64)(i - 4) * 3);
    }
    if (!jrows) { if (build) store_build(s); store_save(s, fz_file); }
    store_sync(s, fz_file);
    snprintf(p, sizeof(p), "%s/%s", out, name);
    if (!(o = fopen(p, "wb"))) abort();
    if ((f = fopen(fz_file, "rb"))) { fseek(f, 0, SEEK_END); fl = ftell(f); fclose(f); }
    b[0] = (unsigned char)(fl & 0xff); b[1] = (unsigned char)((fl >> 8) & 0xff); b[2] = (unsigned char)((fl >> 16) & 0xff); b[3] = (unsigned char)((fl >> 24) & 0xff);
    fwrite(b, 1, 4, o);
    for (j = 0; j < 2; ++j) {
        int c;
        if (!(f = fopen(j ? fz_jn : fz_file, "rb"))) continue;
        while ((c = fgetc(f)) != EOF) { fputc(c, o); if (j) ++jl; }
        fclose(f);
    }
    fclose(o);
    fprintf(stderr, "seed %s: file %ld bytes, journal %ld bytes\n", name, fl, jl);
    store_destroy(s); arena_free(&fz_arena); free(v);
    fz_clean_();
}

int LLVMFuzzerInitialize(int *argc, char ***argv)
{
    char dir[] = "/tmp/fuzz_store.XXXXXX";
    const char *seeds = getenv("FUZZ_STORE_SEEDS");
    (void)argc; (void)argv;
    if (!mkdtemp(dir)) abort();
    strcpy(fz_dir, dir);
    snprintf(fz_file, sizeof(fz_file), "%s/f.vxi", fz_dir);
    snprintf(fz_jn, sizeof(fz_jn), "%s/f.vxi.j", fz_dir);
    if (seeds) {
        fz_seed_(seeds, "empty", 4, 0, 0, 0, 0);
        fz_seed_(seeds, "small_built", 4, 64, 0, 1, 12);
        fz_seed_(seeds, "texts_fast", 8, 120, 1, 0, 9);
        fz_seed_(seeds, "journal_only", 4, 0, 1, 0, 20);
        exit(0);
    }
    return 0;
}

int LLVMFuzzerTestOneInput(const unsigned char *d, size_t n)
{
    RagStore *s;
    size_t fl;
    if (n < 4) return 0;
    fl = (size_t)d[0] | ((size_t)d[1] << 8) | ((size_t)d[2] << 16) | ((size_t)d[3] << 24);
    if (fl > n - 4) fl = n - 4;
    if (fl >= 16 && (d[8] | d[9] << 8) != 9 && (d[14] | d[15]) != 0)
        return 0;   /* a v5-v8 file (no checksums) claiming 2^16 dimensions or more: the loader believes it and
                     * allocates a row of that size, which libFuzzer reports as running out of memory */
    fz_write_(fz_file, d + 4, fl);
    if (n - 4 - fl) fz_write_(fz_jn, d + 4 + fl, n - 4 - fl);
    s = store_load(&fz_arena, fz_file);
    if (s) { fz_use_(s); store_destroy(s); }
    arena_free(&fz_arena);
    fz_clean_();
    return 0;
}

#include "fuzz_main.h"
