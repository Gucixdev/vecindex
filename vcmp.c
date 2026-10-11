/* vcmp.c - vecindex store vs another index on the SAME vectors (e.g. turbovec, see vcmp_turbovec.py).
 *   vcmp dump wiki.rec OUT_DIR     embed the articles with vecdb_embed -> OUT_DIR/{data,queries,truth}.bin
 *   vcmp run  OUT_DIR              build the store on data.bin, report build time, recall@10, ms/query
 * data.bin: N x DIM f32 (unit); queries.bin: NQ x DIM (perturbed corpus rows); truth.bin: NQ x 10 i32 exact top-10. */
#include "store.c"
#include <stdio.h>
#include <time.h>
#include <math.h>
static int DIM = 256;
#define NQ 300
static double now_(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec; }


static void embed_(const char *t, i32 n, f32 *out)
{
    /* same recipe as vecdb.c: signed feature hashing of words + bigrams */
    i32 i = 0, d; f32 nrm = 0.0f; u64 prev = 0; i32 have = 0;
    for (d = 0; d < DIM; ++d) out[d] = 0.0f;
    while (i < n) {
        u64 h = 1469598103934665603ULL; i32 w = 0;
        while (i < n && !((t[i] >= '0' && t[i] <= '9') || (t[i] >= 'a' && t[i] <= 'z') || (t[i] >= 'A' && t[i] <= 'Z') || (u8)t[i] >= 0x80)) ++i;
        while (i < n && ((t[i] >= '0' && t[i] <= '9') || (t[i] >= 'a' && t[i] <= 'z') || (t[i] >= 'A' && t[i] <= 'Z') || (u8)t[i] >= 0x80)) { u8 c = (u8)t[i]; if (c >= 'A' && c <= 'Z') c += 32; h ^= c; h *= 1099511628211ULL; ++i; w = 1; }
        if (!w) break;
        h ^= h >> 29; h *= 0xbf58476d1ce4e5b9ULL; h ^= h >> 32;
        out[h % DIM] += ((h >> 40) & 1) ? -1.0f : 1.0f;
        if (have) { u64 g = (prev * 31u + h); g ^= g >> 29; out[g % DIM] += ((g >> 41) & 1) ? -0.5f : 0.5f; }
        prev = h; have = 1;
    }
    for (d = 0; d < DIM; ++d) { f32 v = out[d]; out[d] = v < 0 ? -sqrtf(-v) : sqrtf(v); nrm += out[d] * out[d]; }
    if (nrm > 0) { nrm = 1.0f / sqrtf(nrm); for (d = 0; d < DIM; ++d) out[d] *= nrm; }
}

int main(int argc, char **argv)
{
    static Arena arena;
    char path[1024];
    if (argc >= 4 && !strcmp(argv[1], "dump")) {
        FILE *f = fopen(argv[2], "rb"), *o;
        char *title = (char *)malloc(4096), *buf = (char *)malloc(1 << 20), line[64];
        i32 n = 0, cap = 1 << 16, i, j, k;
        f32 *data = (f32 *)malloc((size_t)cap * DIM * 4), *q = (f32 *)malloc((size_t)NQ * DIM * 4);
        i32 *truth = (i32 *)malloc((size_t)NQ * 10 * 4);
        u32 r = 7;
        if (!f) return 2;
        while (fgets(title, 4096, f) && fgets(line, sizeof line, f)) {
            i32 len = atoi(line);
            if (len <= 0 || len >= (1 << 20) || fread(buf, 1, (size_t)len, f) != (size_t)len) break;
            fgetc(f);
            if (n == cap) { cap *= 2; data = (f32 *)realloc(data, (size_t)cap * DIM * 4); }
            embed_(buf, len, data + (size_t)n * DIM); ++n;
        }
        for (i = 0; i < NQ; ++i) {                        /* queries: a corpus row plus noise, renormalised */
            const f32 *src = data + (size_t)((r = r * 1664525u + 1013904223u) % (u32)n) * DIM; f32 nrm = 0;
            for (j = 0; j < DIM; ++j) { r = r * 1664525u + 1013904223u; q[(size_t)i * DIM + j] = src[j] + ((f32)(r >> 8) / 16777216.0f - 0.5f) * 0.06f; nrm += q[(size_t)i * DIM + j] * q[(size_t)i * DIM + j]; }
            nrm = 1.0f / sqrtf(nrm); for (j = 0; j < DIM; ++j) q[(size_t)i * DIM + j] *= nrm;
        }
        for (i = 0; i < NQ; ++i) {                        /* exact top-10 by inner product */
            f32 bs[10]; i32 bi[10]; i32 m = 0;
            for (j = 0; j < n; ++j) {
                f32 s = 0; const f32 *a = q + (size_t)i * DIM, *b = data + (size_t)j * DIM;
                for (k = 0; k < DIM; ++k) s += a[k] * b[k];
                if (m == 10 && s <= bs[9]) continue;
                k = m < 10 ? m++ : 9;
                while (k > 0 && bs[k - 1] < s) { bs[k] = bs[k - 1]; bi[k] = bi[k - 1]; --k; }
                bs[k] = s; bi[k] = j;
            }
            for (k = 0; k < 10; ++k) truth[i * 10 + k] = bi[k];
        }
        sprintf(path, "%s/data.bin", argv[3]); o = fopen(path, "wb"); fwrite(data, 4, (size_t)n * DIM, o); fclose(o);
        sprintf(path, "%s/queries.bin", argv[3]); o = fopen(path, "wb"); fwrite(q, 4, (size_t)NQ * DIM, o); fclose(o);
        sprintf(path, "%s/truth.bin", argv[3]); o = fopen(path, "wb"); fwrite(truth, 4, (size_t)NQ * 10, o); fclose(o);
        printf("dumped %d vectors x %d, %d queries\n", n, DIM, NQ);
        return 0;
    }
    if (argc >= 5 && !strcmp(argv[1], "synth")) {               /* N clustered unit vectors (topics + noise), dimension argv[3] */
        i32 n = atoi(argv[2]), i, j, k, nt = 3000; f32 *cent, *data, *q; i32 *truth; FILE *o; u32 r = 5;
        DIM = atoi(argv[3]);
        cent = (f32 *)malloc((size_t)nt * DIM * 4); data = (f32 *)malloc((size_t)n * DIM * 4); q = (f32 *)malloc((size_t)NQ * DIM * 4); truth = (i32 *)malloc((size_t)NQ * 40);
        for (i = 0; i < nt * DIM; ++i) { r = r * 1664525u + 1013904223u; cent[i] = (f32)(r >> 8) / 16777216.0f - 0.5f; }
        for (i = 0; i < n; ++i) {
            f32 nrm = 0; const f32 *c;
            r = r * 1664525u + 1013904223u; c = cent + (size_t)((r >> 8) % (u32)nt) * DIM;
            for (j = 0; j < DIM; ++j) { r = r * 1664525u + 1013904223u; data[(size_t)i * DIM + j] = c[j] + ((f32)(r >> 8) / 16777216.0f - 0.5f) * 0.35f; nrm += data[(size_t)i * DIM + j] * data[(size_t)i * DIM + j]; }
            nrm = 1.0f / sqrtf(nrm); for (j = 0; j < DIM; ++j) data[(size_t)i * DIM + j] *= nrm;
        }
        for (i = 0; i < NQ; ++i) {
            const f32 *src = data + (size_t)((r = r * 1664525u + 1013904223u) % (u32)n) * DIM; f32 nrm = 0;
            for (j = 0; j < DIM; ++j) { r = r * 1664525u + 1013904223u; q[(size_t)i * DIM + j] = src[j] + ((f32)(r >> 8) / 16777216.0f - 0.5f) * 0.06f; nrm += q[(size_t)i * DIM + j] * q[(size_t)i * DIM + j]; }
            nrm = 1.0f / sqrtf(nrm); for (j = 0; j < DIM; ++j) q[(size_t)i * DIM + j] *= nrm;
        }
        for (i = 0; i < NQ; ++i) {
            f32 bs[10]; i32 bi[10]; i32 m = 0;
            for (j = 0; j < n; ++j) {
                f32 s = 0; const f32 *a = q + (size_t)i * DIM, *b = data + (size_t)j * DIM;
                for (k = 0; k < DIM; ++k) s += a[k] * b[k];
                if (m == 10 && s <= bs[9]) continue;
                k = m < 10 ? m++ : 9;
                while (k > 0 && bs[k - 1] < s) { bs[k] = bs[k - 1]; bi[k] = bi[k - 1]; --k; }
                bs[k] = s; bi[k] = j;
            }
            for (k = 0; k < 10; ++k) truth[i * 10 + k] = bi[k];
        }
        sprintf(path, "%s/data.bin", argv[4]); o = fopen(path, "wb"); fwrite(data, 4, (size_t)n * DIM, o); fclose(o);
        sprintf(path, "%s/queries.bin", argv[4]); o = fopen(path, "wb"); fwrite(q, 4, (size_t)NQ * DIM, o); fclose(o);
        sprintf(path, "%s/truth.bin", argv[4]); o = fopen(path, "wb"); fwrite(truth, 4, (size_t)NQ * 10, o); fclose(o);
        sprintf(path, "%s/dim.txt", argv[4]); o = fopen(path, "w"); fprintf(o, "%d\n", DIM); fclose(o);
        printf("synth %d x %d\n", n, DIM);
        return 0;
    }
    if (argc >= 4 && !strcmp(argv[1], "dumpstore")) {           /* vectors of a saved store (e.g. real e5 embeddings) */
        RagStore *st = store_load(&arena, argv[2]);
        FILE *o; i32 n, i, j, k; f32 *data, *q; i32 *truth; u32 r = 7;
        if (!st) { fprintf(stderr, "cannot load %s\n", argv[2]); return 2; }
        n = st->count; DIM = st->dim;
        data = st->vecs; q = (f32 *)malloc((size_t)NQ * DIM * 4); truth = (i32 *)malloc((size_t)NQ * 10 * 4);
        for (i = 0; i < NQ; ++i) {
            const f32 *src = data + (size_t)((r = r * 1664525u + 1013904223u) % (u32)n) * DIM; f32 nrm = 0;
            for (j = 0; j < DIM; ++j) { r = r * 1664525u + 1013904223u; q[(size_t)i * DIM + j] = src[j] + ((f32)(r >> 8) / 16777216.0f - 0.5f) * 0.06f; nrm += q[(size_t)i * DIM + j] * q[(size_t)i * DIM + j]; }
            nrm = 1.0f / sqrtf(nrm); for (j = 0; j < DIM; ++j) q[(size_t)i * DIM + j] *= nrm;
        }
        for (i = 0; i < NQ; ++i) {
            f32 bs[10]; i32 bi[10]; i32 m = 0;
            for (j = 0; j < n; ++j) {
                f32 s = 0; const f32 *a = q + (size_t)i * DIM, *b = data + (size_t)j * DIM;
                for (k = 0; k < DIM; ++k) s += a[k] * b[k];
                if (m == 10 && s <= bs[9]) continue;
                k = m < 10 ? m++ : 9;
                while (k > 0 && bs[k - 1] < s) { bs[k] = bs[k - 1]; bi[k] = bi[k - 1]; --k; }
                bs[k] = s; bi[k] = j;
            }
            for (k = 0; k < 10; ++k) truth[i * 10 + k] = bi[k];
        }
        sprintf(path, "%s/data.bin", argv[3]); o = fopen(path, "wb"); fwrite(data, 4, (size_t)n * DIM, o); fclose(o);
        sprintf(path, "%s/queries.bin", argv[3]); o = fopen(path, "wb"); fwrite(q, 4, (size_t)NQ * DIM, o); fclose(o);
        sprintf(path, "%s/truth.bin", argv[3]); o = fopen(path, "wb"); fwrite(truth, 4, (size_t)NQ * 10, o); fclose(o);
        sprintf(path, "%s/dim.txt", argv[3]); o = fopen(path, "w"); fprintf(o, "%d\n", DIM); fclose(o);
        printf("dumped %d vectors x %d\n", n, DIM);
        return 0;
    }
    if (argc >= 3 && !strcmp(argv[1], "run")) {
        FILE *f; long sz; i32 n, i, j, hit = 0, tot = 0;
        f32 *data, *q; i32 *truth; RagStore *s; double t0, ts = 0;
        sprintf(path, "%s/dim.txt", argv[2]); f = fopen(path, "r"); if (f) { if (fscanf(f, "%d", &DIM) != 1) DIM = 256; fclose(f); }
        sprintf(path, "%s/data.bin", argv[2]); f = fopen(path, "rb"); fseek(f, 0, SEEK_END); sz = ftell(f); fseek(f, 0, SEEK_SET);
        n = (i32)(sz / (DIM * 4)); data = (f32 *)malloc((size_t)sz); if (fread(data, 1, (size_t)sz, f) != (size_t)sz) return 2; fclose(f);
        q = (f32 *)malloc((size_t)NQ * DIM * 4); truth = (i32 *)malloc((size_t)NQ * 40);
        sprintf(path, "%s/queries.bin", argv[2]); f = fopen(path, "rb"); if (fread(q, 4, (size_t)NQ * DIM, f) != (size_t)NQ * DIM) return 2; fclose(f);
        sprintf(path, "%s/truth.bin", argv[2]); f = fopen(path, "rb"); if (fread(truth, 4, (size_t)NQ * 10, f) != (size_t)NQ * 10) return 2; fclose(f);
        s = store_new(&arena, DIM);
        if (argc > 3 && !strcmp(argv[3], "fast")) store_set_fast(s, 0.97f);
        if (getenv("VCMP_PROBE")) store_set_probe(s, atoi(getenv("VCMP_PROBE")));
        t0 = now_(); for (i = 0; i < n; ++i) store_add(s, data + (size_t)i * DIM, i); printf("add %d: %.2f s\n", n, now_() - t0);
        t0 = now_(); store_build(s); printf("build: %.2f s\n", now_() - t0);
        for (i = 0; i < NQ; ++i) {
            i64 ids[10]; f32 sc[10]; i32 h;
            t0 = now_(); h = store_search(s, q + (size_t)i * DIM, 10, ids, sc); ts += now_() - t0;
            for (j = 0; j < h; ++j) { i32 k; for (k = 0; k < 10; ++k) if (truth[i * 10 + k] == (i32)ids[j]) { ++hit; break; } }
            tot += 10;
        }
        printf("vecindex store (%s): recall@10 %.1f%%, %.3f ms/query\n", argc > 3 ? argv[3] : "tuned", 100.0 * hit / tot, ts * 1e3 / NQ);
        return 0;
    }
    fprintf(stderr, "usage: vcmp dump wiki.rec OUT | vcmp run OUT [fast]\n");
    return 2;
}
