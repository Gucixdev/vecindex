/* store_sum.c - checksums and Reed-Solomon parity for the store file. Every 64 KiB block has a BLAKE3 digest; the
 * blocks are dealt into groups - block b into group b mod G, so a run of neighbouring damaged blocks lands in
 * different groups - and each group of at most k data blocks has m parity blocks, from which any m damaged blocks of
 * the group are rebuilt. The digests say which blocks are damaged and confirm every rebuild. The table of digests and
 * the trailer that vouches for it are written twice. Part of store.c, which includes it after store_batch.c (CRC-32C);
 * not compiled on its own.
 *
 * A v9 file is the v7/v8 layout up to byte D, then: zeros up to S = D rounded up to a block (N = S / B data blocks);
 * P = G m parity blocks, group g's m blocks one after another (block b of the file is at b B, parity included);
 * table A, the digests of the N + P blocks; trailer A at the next 4 KiB boundary; table B at the one after it;
 * trailer B, the file's last 128 bytes. A trailer, little-endian: 'VXS1', 1, log2 B, k, m, 0, D, N, G, P, the offsets
 * of table A, trailer A and table B, the BLAKE3 of the table, the CRC-32C of the 112 bytes before it, zeros.
 *
 * Parity block p of a group is the sum over its data blocks i of c(p, i) times block i in GF(2^8) (polynomial 0x11d),
 * with the Cauchy coefficients c(p, i) = 1 / ((k + p) xor i): every square part of that matrix is invertible, so the
 * damaged data blocks of a group come back from as many intact parity blocks by one small inverse. */
#include "blake3.c"

#define SSUM_BLOG  16               /* 64 KiB blocks */
#define SSUM_TRL   128
#define SSUM_PG    4096u
#define SSUM_AL(x) (((x) + (u64)SSUM_PG - 1u) & ~(u64)(SSUM_PG - 1u))

typedef struct {
    u32 blog, k, m;
    u64 D, N, G, P, ta, tra, tb, fsize;
    u8  root[32];
    u8 *tab;                        /* the (N + P) digests, malloc'd */
} SSum;

/* ---- GF(2^8) ---- */

static u8 gf_exp_[512], gf_log_[256];
static pthread_once_t gf_once_ = PTHREAD_ONCE_INIT;
static void gf_init_(void)
{
    u32 i, x = 1;
    for (i = 0; i < 255; ++i) { gf_exp_[i] = (u8)x; gf_log_[x] = (u8)i; x <<= 1; if (x & 0x100u) x ^= 0x11du; }
    for (i = 255; i < 512; ++i) gf_exp_[i] = gf_exp_[i - 255];
}
static u8 gf_mul_(u32 a, u32 b) { return (u8)(a && b ? gf_exp_[gf_log_[a] + gf_log_[b]] : 0); }
static u8 gf_inv_(u32 a) { return gf_exp_[255 - gf_log_[a]]; }
static u8 gf_coef_(u32 k, u32 p, u32 i) { return gf_inv_((k + p) ^ i); }

/* dst ^= c src over n bytes */
static void gf_madd_(u8 *dst, const u8 *src, u32 c, size_t n)
{
    u8 t[256];
    size_t i;
    u32 x;
    if (!c) return;
    if (c == 1) { for (i = 0; i < n; ++i) dst[i] ^= src[i]; return; }
    for (x = 0; x < 256; ++x) t[x] = gf_mul_(x, c);
    for (i = 0; i < n; ++i) dst[i] ^= t[src[i]];
}

/* the inverse of the n x n matrix a (row-major, destroyed) into inv: 0, or -1 when it has none */
static i32 gf_invert_(u8 *a, u8 *inv, i32 n)
{
    i32 i, j, r, c;
    for (i = 0; i < n; ++i) for (j = 0; j < n; ++j) inv[i * n + j] = (u8)(i == j);
    for (c = 0; c < n; ++c) {
        u8 f;
        for (r = c; r < n && !a[r * n + c]; ++r) {}
        if (r == n) return -1;
        if (r != c) for (j = 0; j < n; ++j) {
            u8 x = a[r * n + j]; a[r * n + j] = a[c * n + j]; a[c * n + j] = x;
            x = inv[r * n + j]; inv[r * n + j] = inv[c * n + j]; inv[c * n + j] = x;
        }
        f = gf_inv_(a[c * n + c]);
        for (j = 0; j < n; ++j) { a[c * n + j] = gf_mul_(a[c * n + j], f); inv[c * n + j] = gf_mul_(inv[c * n + j], f); }
        for (r = 0; r < n; ++r) {
            if (r == c || !(f = a[r * n + c])) continue;
            for (j = 0; j < n; ++j) { a[r * n + j] ^= gf_mul_(f, a[c * n + j]); inv[r * n + j] ^= gf_mul_(f, inv[c * n + j]); }
        }
    }
    return 0;
}

/* ---- layout, trailer ---- */

static void ssum_put_(u8 *p, u64 v, i32 n) { i32 i; for (i = 0; i < n; ++i) p[i] = (u8)(v >> (8 * i)); }
static u64 ssum_get_(const u8 *p, i32 n) { u64 v = 0; i32 i; for (i = n - 1; i >= 0; --i) v = v << 8 | p[i]; return v; }

/* the layout that D, the block size and k, m make: 0, or -1 when they are out of range */
static i32 ssum_layout_(SSum *z, u64 D, u32 blog, u32 k, u32 m)
{
    u64 B, T;
    if (blog < 12 || blog > 24 || k < 1 || k > 128 || m > 64 || D == 0) return -1;
    B = (u64)1 << blog;
    z->blog = blog; z->k = k; z->m = m; z->D = D;
    z->N = (D + B - 1) >> blog;
    z->G = (z->N + k - 1) / k;
    z->P = z->G * m;
    z->ta = (z->N + z->P) << blog;
    T = (z->N + z->P) * 32u;
    z->tra = SSUM_AL(z->ta + T);
    z->tb = SSUM_AL(z->tra + SSUM_TRL);
    z->fsize = z->tb + T + SSUM_TRL;
    return 0;
}

static void ssum_trailer_(const SSum *z, u8 t[SSUM_TRL])
{
    memset(t, 0, SSUM_TRL);
    ssum_put_(t, SSUM_MAGIC, 4); ssum_put_(t + 4, 1, 4); ssum_put_(t + 8, z->blog, 4); ssum_put_(t + 12, z->k, 4);
    ssum_put_(t + 16, z->m, 4);
    ssum_put_(t + 24, z->D, 8); ssum_put_(t + 32, z->N, 8); ssum_put_(t + 40, z->G, 8); ssum_put_(t + 48, z->P, 8);
    ssum_put_(t + 56, z->ta, 8); ssum_put_(t + 64, z->tra, 8); ssum_put_(t + 72, z->tb, 8);
    memcpy(t + 80, z->root, 32);
    ssum_put_(t + 112, store_crc32c_(t, 112), 4);
}

/* a whole trailer of a file of fsize bytes into z: 0, or -1 */
static i32 ssum_parse_(SSum *z, const u8 *t, u64 fsize)
{
    SSum w;
    if (ssum_get_(t, 4) != SSUM_MAGIC || ssum_get_(t + 4, 4) != 1 || (u32)ssum_get_(t + 112, 4) != store_crc32c_(t, 112)
        || ssum_layout_(&w, ssum_get_(t + 24, 8), (u32)ssum_get_(t + 8, 4), (u32)ssum_get_(t + 12, 4), (u32)ssum_get_(t + 16, 4))
        || w.N != ssum_get_(t + 32, 8) || w.G != ssum_get_(t + 40, 8) || w.P != ssum_get_(t + 48, 8)
        || w.ta != ssum_get_(t + 56, 8) || w.tra != ssum_get_(t + 64, 8) || w.tb != ssum_get_(t + 72, 8) || w.fsize != fsize) return -1;
    memcpy(w.root, t + 80, 32);
    w.tab = (u8 *)0;
    *z = w;
    return 0;
}

static i32 ssum_read_(int fd, u8 *buf, size_t n, u64 off)
{
    size_t got = 0;
    if (lseek(fd, (off_t)off, SEEK_SET) == (off_t)-1) return -1;
    while (got < n) { ssize_t r = read(fd, buf + got, n - got); if (r <= 0) return -1; got += (size_t)r; }
    return 0;
}
static i32 ssum_write_(int fd, const u8 *buf, size_t n, u64 off)
{
    size_t put = 0;
    if (lseek(fd, (off_t)off, SEEK_SET) == (off_t)-1) return -1;
    while (put < n) { ssize_t r = write(fd, buf + put, n - put); if (r <= 0) return -1; put += (size_t)r; }
    return 0;
}

/* The trailers and tables of the file open as fd (fsize bytes): 0 with z filled and z->tab malloc'd, or -1 when no
 * trailer with its table is readable. *dmg gets bit 0 trailer B, 1 trailer A, 2 table A, 3 table B: copies that are
 * damaged (the intact one stood in). walk: look for trailer A when B is damaged (only for a file whose header says
 * v9 - an older file has neither, and the walk reads back through up to 1/64 of the file). */
static i32 ssum_open_(SSum *z, int fd, u64 fsize, i32 *dmg, i32 walk)
{
    u8 t[SSUM_TRL], h[32];
    u64 off, lim, T;
    SSum w;
    i32 ok = 0, d = 0, good = 0;
    memset(z, 0, sizeof(*z));
    *dmg = 0;
    if (fsize < SSUM_TRL + 64) return -1;
    if (ssum_read_(fd, t, SSUM_TRL, fsize - SSUM_TRL) == 0 && ssum_parse_(z, t, fsize) == 0) {
        ok = 1;
        if (ssum_read_(fd, t, SSUM_TRL, z->tra) != 0 || ssum_parse_(&w, t, fsize) != 0) d |= 2;
    } else {                                    /* trailer B is damaged: look for A on the 4 KiB boundaries before it */
        d |= 1;
        if (!walk) return -1;
        lim = fsize / 64 + 2 * SSUM_PG;         /* the tables take at most 1/64 of a file */
        for (off = (fsize - SSUM_TRL) & ~(u64)(SSUM_PG - 1u); ; off -= SSUM_PG) {
            if (ssum_read_(fd, t, SSUM_TRL, off) == 0 && ssum_parse_(z, t, fsize) == 0 && z->tra == off) { ok = 1; break; }
            if (off < SSUM_PG || fsize - off > lim) break;
        }
        if (!ok) return -1;
    }
    T = (z->N + z->P) * 32u;
    if (!(z->tab = (u8 *)malloc((size_t)T))) return -1;
    if (ssum_read_(fd, z->tab, (size_t)T, z->ta) == 0 && (b3_hash(z->tab, (size_t)T, h), memcmp(h, z->root, 32) == 0)) good = 1;
    else d |= 4;
    {   u8 *tb = (u8 *)malloc((size_t)T);       /* table B too: a load trusts A, a scrub mends B */
        if (!tb || ssum_read_(fd, tb, (size_t)T, z->tb) != 0 || (b3_hash(tb, (size_t)T, h), memcmp(h, z->root, 32) != 0)) d |= 8;
        else if (!good) { memcpy(z->tab, tb, (size_t)T); good = 1; }
        free(tb);
    }
    *dmg = d;
    if (!good) { free(z->tab); z->tab = (u8 *)0; return -1; }
    return 0;
}

/* ---- checking and rebuilding groups ---- */

typedef struct {
    const SSum *z;
    const u8 *map;                  /* the file mapped for reading, or NULL: blocks are read through a descriptor */
    const char *path;               /* opened by each worker for its own descriptors */
    i32 heal, parity, quick;        /* write rebuilt blocks back; check parity blocks too; only group 0 */
    u64 g0, gstep;                  /* this worker's groups: g0, g0 + gstep, ... */
    i64 bad, fixed;
    u64 ubad;                       /* offset of the first data block that stays damaged, ~0: none */
    i32 err;
} SJob;

static const u8 *ssum_blk_(const SJob *j, int rfd, u64 b, u8 *buf)
{
    size_t B = (size_t)1 << j->z->blog;
    if (j->map) return j->map + (b << j->z->blog);
    return ssum_read_(rfd, buf, B, b << j->z->blog) == 0 ? buf : (const u8 *)0;
}
static i32 ssum_ok_(const SJob *j, u64 b, const u8 *p)
{
    u8 h[32];
    if (!p) return 0;
    b3_hash(p, (size_t)1 << j->z->blog, h);
    return memcmp(h, j->z->tab + b * 32u, 32) == 0;
}

/* group g: its data blocks (and with j->parity its parity blocks) against their digests; the damaged ones rebuilt
 * when no more than m of them are, confirmed by their digests, written back with j->heal. mem: (1 + 3m) blocks. */
static void ssum_group_(SJob *j, u64 g, int rfd, int wfd, u8 *mem)
{
    const SSum *z = j->z;
    u32 m = z->m, k = z->k, ng = (u32)((z->N - g + z->G - 1) / z->G), i, p, q, nmd = 0, nmp = 0;
    u32 md[128], mp[64], use[64];
    size_t B = (size_t)1 << z->blog;
    u8 *tmp = mem, *r = tmp + B, *x = r + (size_t)m * B, *pp = x + (size_t)m * B, h[32], a[64 * 64], ai[64 * 64];
    const u8 *src;
    for (i = 0; i < ng; ++i) if (!ssum_ok_(j, g + (u64)i * z->G, ssum_blk_(j, rfd, g + (u64)i * z->G, tmp))) md[nmd++] = i;
    if (j->parity || nmd)
        for (p = 0; p < m; ++p) if (!ssum_ok_(j, z->N + g * m + p, ssum_blk_(j, rfd, z->N + g * m + p, tmp))) mp[nmp++] = p;
    if (!nmd && !nmp) return;
    j->bad += (i64)(nmd + nmp);
    if (nmd + nmp > m) goto fail;
    if (nmd) {                                  /* r_q = parity use[q] + the intact blocks' share of it: the damaged ones' */
        for (p = 0, q = 0; p < m && q < nmd; ++p) {
            for (i = 0; i < nmp && mp[i] != p; ++i) {}
            if (i == nmp) use[q++] = p;
        }
        for (q = 0; q < nmd; ++q) {
            if (!(src = ssum_blk_(j, rfd, z->N + g * m + use[q], tmp))) goto fail;
            memcpy(r + (size_t)q * B, src, B);
        }
        for (i = 0; i < ng; ++i) {
            for (q = 0; q < nmd && md[q] != i; ++q) {}
            if (q < nmd) continue;
            if (!(src = ssum_blk_(j, rfd, g + (u64)i * z->G, tmp))) goto fail;
            for (q = 0; q < nmd; ++q) gf_madd_(r + (size_t)q * B, src, gf_coef_(k, use[q], i), B);
        }
        for (q = 0; q < nmd; ++q) for (p = 0; p < nmd; ++p) a[q * nmd + p] = gf_coef_(k, use[q], md[p]);
        if (gf_invert_(a, ai, (i32)nmd)) goto fail;
        memset(x, 0, (size_t)nmd * B);
        for (p = 0; p < nmd; ++p) for (q = 0; q < nmd; ++q) gf_madd_(x + (size_t)p * B, r + (size_t)q * B, ai[p * nmd + q], B);
        for (p = 0; p < nmd; ++p) {
            b3_hash(x + (size_t)p * B, B, h);
            if (memcmp(h, z->tab + (g + (u64)md[p] * z->G) * 32u, 32) != 0) goto fail;
        }
    }
    if (nmp) {                                  /* damaged parity blocks again from the data, rebuilt blocks included */
        memset(pp, 0, (size_t)nmp * B);
        for (i = 0; i < ng; ++i) {
            for (q = 0; q < nmd && md[q] != i; ++q) {}
            if (q < nmd) src = x + (size_t)q * B;
            else if (!(src = ssum_blk_(j, rfd, g + (u64)i * z->G, tmp))) goto fail;
            for (q = 0; q < nmp; ++q) gf_madd_(pp + (size_t)q * B, src, gf_coef_(k, mp[q], i), B);
        }
        for (q = 0; q < nmp; ++q) {
            b3_hash(pp + (size_t)q * B, B, h);
            if (memcmp(h, z->tab + (z->N + g * m + mp[q]) * 32u, 32) != 0) goto fail;
        }
    }
    if (wfd < 0) goto fail;                     /* rebuilt, but nothing may be written: found, not fixed */
    for (q = 0; q < nmd; ++q) if (ssum_write_(wfd, x + (size_t)q * B, B, (g + (u64)md[q] * z->G) << z->blog)) goto fail;
    for (q = 0; q < nmp; ++q) if (ssum_write_(wfd, pp + (size_t)q * B, B, (z->N + g * m + mp[q]) << z->blog)) goto fail;
    j->fixed += (i64)(nmd + nmp);
    return;
fail:
    if (nmd && ((g + (u64)md[0] * z->G) << z->blog) < j->ubad) j->ubad = (g + (u64)md[0] * z->G) << z->blog;
}

static void *ssum_job_(void *arg)
{
    SJob *j = (SJob *)arg;
    int rfd = -1, wfd = -1;
    u8 *mem = (u8 *)malloc(((size_t)1 + 3u * j->z->m) << j->z->blog);
    u64 g;
    if (!mem || (!j->map && (rfd = open(j->path, O_RDONLY)) < 0) || (j->heal && (wfd = open(j->path, O_WRONLY)) < 0)) {
        j->err = 1;
        j->heal = 0;
        if (!mem || (!j->map && rfd < 0)) { free(mem); if (wfd >= 0) close(wfd); return (void *)0; }
    }
    for (g = j->g0; g < (j->quick ? 1u : j->z->G); g += j->gstep) ssum_group_(j, g, rfd, j->heal ? wfd : -1, mem);
    if (wfd >= 0) { if (j->fixed && fdatasync(wfd) != 0) j->err = 1; close(wfd); }
    if (rfd >= 0) close(rfd);
    free(mem);
    return (void *)0;
}

/* threads for work over `bytes` of a file in G groups: one for small files and inside a snapshot child */
static i32 ssum_nth_(u64 G, u64 bytes, i32 cap)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    i32 t = n > 0 ? (i32)n : 1;
    if (t > cap) t = cap;
    if ((u64)t > G) t = (i32)G;
    return bytes < ((u64)32 << 20) || t < 1 ? 1 : t;
}

/* every group of the file at path (map: the file mapped, or NULL) checked by nth workers; returns the damaged
 * blocks found, *fixed the rebuilt ones, *ubad the first data block that stays damaged (~0: none) */
static i64 ssum_check_(const SSum *z, const char *path, const u8 *map, i32 heal, i32 parity, i32 quick, i32 nth,
                       i64 *fixed, u64 *ubad, i32 *err)
{
    SJob jobs[16];
    pthread_t th[16];
    i32 t, started[16];
    i64 bad = 0;
    pthread_once(&gf_once_, gf_init_);
    if (nth < 1 || quick) nth = 1;
    if (nth > 16) nth = 16;
    *fixed = 0; *ubad = ~(u64)0; *err = 0;
    for (t = 0; t < nth; ++t) {
        memset(&jobs[t], 0, sizeof(SJob));
        jobs[t].z = z; jobs[t].map = map; jobs[t].path = path; jobs[t].heal = heal; jobs[t].parity = parity;
        jobs[t].quick = quick; jobs[t].g0 = (u64)t; jobs[t].gstep = (u64)nth; jobs[t].ubad = ~(u64)0;
        started[t] = nth > 1 && pthread_create(&th[t], (const pthread_attr_t *)0, ssum_job_, &jobs[t]) == 0;
        if (!started[t]) ssum_job_(&jobs[t]);
    }
    for (t = 0; t < nth; ++t) {
        if (started[t]) pthread_join(th[t], (void **)0);
        bad += jobs[t].bad; *fixed += jobs[t].fixed;
        if (jobs[t].ubad < *ubad) *ubad = jobs[t].ubad;
        if (jobs[t].err) *err = 1;
    }
    return bad;
}

/* ---- writing ---- */

typedef struct { const SSum *z; const u8 *map; const char *path; u8 *tab; u64 g0, gstep; i32 err; } SEnc;

static void *ssum_enc_job_(void *arg)
{
    SEnc *e = (SEnc *)arg;
    const SSum *z = e->z;
    size_t B = (size_t)1 << z->blog;
    u8 *mem = (u8 *)malloc(B * (1u + z->m));
    int rfd = -1, wfd = open(e->path, O_WRONLY);
    u64 g;
    u32 i, p;
    if (!mem || wfd < 0 || (!e->map && (rfd = open(e->path, O_RDONLY)) < 0)) e->err = 1;
    for (g = e->g0; !e->err && g < z->G; g += e->gstep) {
        u32 ng = (u32)((z->N - g + z->G - 1) / z->G);
        u8 *par = mem + B;
        memset(par, 0, B * z->m);
        for (i = 0; i < ng && !e->err; ++i) {
            u64 b = g + (u64)i * z->G;
            const u8 *src = e->map ? e->map + (b << z->blog) : (ssum_read_(rfd, mem, B, b << z->blog) == 0 ? mem : (const u8 *)0);
            if (!src) { e->err = 1; break; }
            b3_hash(src, B, e->tab + b * 32u);
            for (p = 0; p < z->m; ++p) gf_madd_(par + p * B, src, gf_coef_(z->k, p, i), B);
        }
        for (p = 0; p < z->m && !e->err; ++p) {
            b3_hash(par + p * B, B, e->tab + (z->N + g * z->m + p) * 32u);
            if (ssum_write_(wfd, par + p * B, B, (z->N + g * z->m + p) << z->blog)) e->err = 1;
        }
    }
    if (wfd >= 0) close(wfd);
    if (rfd >= 0) close(rfd);
    free(mem);
    return (void *)0;
}

static i32 ssum_zeros_(FILE *f, u64 n)
{
    static const u8 z[4096];
    while (n) { size_t c = n > sizeof(z) ? sizeof(z) : (size_t)n; if (fwrite(z, 1, c, f) != c) return -1; n -= c; }
    return 0;
}

/* After D bytes of data written to f (the file at path, opened for reading too): the padding, the parity blocks,
 * both tables and both trailers, by up to cap workers. 0 or -1. */
static i32 ssum_write_file_(FILE *f, const char *path, u64 D, u32 k, u32 m, i32 cap)
{
    SSum z;
    SEnc jobs[16];
    pthread_t th[16];
    i32 t, started[16], bad = 0, nth;
    u64 S, T;
    u8 *map = (u8 *)0, trl[SSUM_TRL];
    void *mp;
    if (ssum_layout_(&z, D, SSUM_BLOG, k, m)) return -1;
    nth = ssum_nth_(z.G, D, cap);
    pthread_once(&gf_once_, gf_init_);
    S = z.N << z.blog; T = (z.N + z.P) * 32u;
    if (ssum_zeros_(f, S - D) || fflush(f) != 0) return -1;
    if (!(z.tab = (u8 *)malloc((size_t)T))) return -1;
    mp = mmap((void *)0, (size_t)S, PROT_READ, MAP_SHARED, fileno(f), 0);
    if (mp != MAP_FAILED) map = (u8 *)mp;
    if (nth > 16) nth = 16;
    for (t = 0; t < nth; ++t) {
        jobs[t].z = &z; jobs[t].map = map; jobs[t].path = path; jobs[t].tab = z.tab;
        jobs[t].g0 = (u64)t; jobs[t].gstep = (u64)nth; jobs[t].err = 0;
        started[t] = nth > 1 && pthread_create(&th[t], (const pthread_attr_t *)0, ssum_enc_job_, &jobs[t]) == 0;
        if (!started[t]) ssum_enc_job_(&jobs[t]);
    }
    for (t = 0; t < nth; ++t) { if (started[t]) pthread_join(th[t], (void **)0); if (jobs[t].err) bad = 1; }
    if (map) munmap(map, (size_t)S);
    if (!bad) {
        b3_hash(z.tab, (size_t)T, z.root);
        ssum_trailer_(&z, trl);
        bad = fseeko(f, (off_t)z.ta, SEEK_SET) != 0 || fwrite(z.tab, 1, (size_t)T, f) != (size_t)T
           || ssum_zeros_(f, z.tra - (z.ta + T)) || fwrite(trl, 1, SSUM_TRL, f) != SSUM_TRL
           || ssum_zeros_(f, z.tb - (z.tra + SSUM_TRL)) || fwrite(z.tab, 1, (size_t)T, f) != (size_t)T
           || fwrite(trl, 1, SSUM_TRL, f) != SSUM_TRL;
    }
    free(z.tab);
    return bad ? -1 : 0;
}

/* ---- loading and scrubbing ---- */

static i32 store_verify_ = -1;
static char store_lerr_[256];

/* What a load checks: 2 every data block, and the parity of a group with a damaged one (store_scrub reads all the
 * parity); 3 the lens - the header's group, the small index arrays and the label table at the load, the rest when it
 * is needed and by a thread walking the file one block at a time; 1 the group of the first block (the header); 0
 * nothing (to rescue a file whose checksums are gone). The default: 2 for a file up to half of RAM, 3 above.
 * For every load after the call, process-wide; $VECINDEX_VERIFY (0, 1, 2 or lens) sets the default. */
#define STORE_VERIFY_AUTO 4
void store_set_verify(i32 level) { store_verify_ = level < 0 ? 0 : (level > 3 ? 3 : level); }
static i32 store_verify_level_(void)
{
    if (store_verify_ < 0) {
        const char *e = getenv("VECINDEX_VERIFY");
        store_verify_ = e && *e >= '0' && *e <= '2' && !e[1] ? *e - '0' : (e && !strcmp(e, "lens") ? 3 : STORE_VERIFY_AUTO);
    }
    return store_verify_;
}

/* why the last store_load or store_load_ro returned NULL, "" when it did not say (not thread-safe) */
const char *store_load_error(void) { return store_lerr_; }

/* Before a load reads the file at path: its blocks against their digests (all of them at level 2, the first group
 * at level 1), damaged ones rebuilt and written back unless ro. 1 when the file has checksums (*z: its layout, z->tab
 * kept for the caller; *bad, *fixed, *ubad as ssum_check_), 0 when it has none that can be read, -1 when they could
 * not all be checked. */
/* 1 when the file open as fd says it is a v9 (or v10) store in its header */
static i32 ssum_v9_(int fd)
{
    u8 hd[8];
    return ssum_read_(fd, hd, 8, 0) == 0 && ssum_get_(hd, 4) == STORE_MAGIC && STORE_V9UP(ssum_get_(hd + 4, 4));
}

static i32 store_sum_load_(const char *path, i32 ro, i32 lvl, SSum *z, i64 *bad, i64 *fixed, u64 *ubad)
{
    int fd = open(path, O_RDONLY);
    struct stat st;
    i32 dmg, err, r = 0;
    *bad = 0; *fixed = 0; *ubad = ~(u64)0;
    if (fd < 0) return 0;
    if (fstat(fd, &st) == 0 && ssum_open_(z, fd, (u64)st.st_size, &dmg, ssum_v9_(fd)) == 0) {
        r = 1;
        if (lvl > 0) {
            void *mp = mmap((void *)0, (size_t)(z->N + z->P) << z->blog, PROT_READ, MAP_SHARED, fd, 0);
            const u8 *map = mp == MAP_FAILED ? (const u8 *)0 : (const u8 *)mp;
            *bad = ssum_check_(z, path, map, !ro, 0, lvl == 1, ssum_nth_(z->G, z->D, 8), fixed, ubad, &err);
            if (map) munmap(mp, (size_t)(z->N + z->P) << z->blog);
            if (err && *bad > *fixed) r = -1;          /* a block went unread: unknown */
        }
        if (r < 0) { free(z->tab); z->tab = (u8 *)0; }
    }
    close(fd);
    return r;
}

/* Checks every block of the store file at path against its digest - data and parity, and both copies of the table and
 * the trailer - reading from the disk where the page cache lets it. With repair, rebuilds what it can and writes it back
 * in place. Returns the damaged blocks and copies found (0: all intact), -1 when the file cannot be read, -2 when it
 * has no checksums (older than v9), -3 when neither copy of its table can be read. *fixed (may be NULL) gets how many
 * of them were rebuilt. A store that has the file loaded may go on meanwhile; a repair rewrites the bytes that were
 * there before the damage. */
i64 store_scrub(const char *path, i32 repair, i64 *fixed)
{
    int fd = open(path, repair ? O_RDWR : O_RDONLY);
    struct stat st;
    SSum z;
    i32 dmg, err, c, v9;
    i64 bad, fx = 0;
    u64 ubad;
    u8 trl[SSUM_TRL];
    void *mp;
    if (fixed) *fixed = 0;
    if (fd < 0 || fstat(fd, &st) != 0) { if (fd >= 0) close(fd); return -1; }
    v9 = ssum_v9_(fd);
    if (ssum_open_(&z, fd, (u64)st.st_size, &dmg, v9) != 0) { close(fd); return v9 ? -3 : -2; }
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);                 /* read what is on the disk, not an old cached copy */
    mp = mmap((void *)0, (size_t)(z.N + z.P) << z.blog, PROT_READ, MAP_SHARED, fd, 0);
    bad = ssum_check_(&z, path, mp == MAP_FAILED ? (const u8 *)0 : (const u8 *)mp, repair, 1, 0,
                      ssum_nth_(z.G, z.D, 8), &fx, &ubad, &err);
    if (mp != MAP_FAILED) munmap(mp, (size_t)(z.N + z.P) << z.blog);
    for (c = 0; c < 4; ++c) if (dmg >> c & 1) ++bad;             /* the copies of the trailer and the table */
    if (dmg && repair) {
        u64 T = (z.N + z.P) * 32u;
        ssum_trailer_(&z, trl);
        if ((dmg & 1) && ssum_write_(fd, trl, SSUM_TRL, (u64)st.st_size - SSUM_TRL) == 0) ++fx;
        if ((dmg & 2) && ssum_write_(fd, trl, SSUM_TRL, z.tra) == 0) ++fx;
        if ((dmg & 4) && ssum_write_(fd, z.tab, (size_t)T, z.ta) == 0) ++fx;
        if ((dmg & 8) && ssum_write_(fd, z.tab, (size_t)T, z.tb) == 0) ++fx;
        if (fdatasync(fd) != 0) err = 1;
    }
    free(z.tab);
    close(fd);
    if (fixed) *fixed = fx;
    return err && bad > fx ? -1 : bad;
}

/* Parity for the files this store writes from now on: m parity blocks for every group of k data blocks (1-128 and
 * 0-64; m 0: digests only). 0, or -1 when out of range. A loaded store keeps its file's. */
i32 store_set_parity(RagStore *s, i32 k, i32 m)
{
    SSum z;
    if (ssum_layout_(&z, 1, SSUM_BLOG, (u32)k, (u32)m) || k < 1 || m < 0) return -1;
    pthread_rwlock_wrlock(&s->lock);
    s->sum_k = k; s->sum_m = m;
    pthread_rwlock_unlock(&s->lock);
    return 0;
}

/* What the load found: the file's blocks (data and parity, 0 for a file without checksums), the damaged ones and how
 * many of those were rebuilt; since then, the rows that failed their check when a search or lookup read them and
 * stayed out (rows_bad), and those whose blocks were rebuilt from the parity on the spot (rows_healed); jdamaged 1
 * when the journal had a damaged record or header (its copy kept as PATH.j.damaged). Any pointer may be NULL. */
void store_integrity(RagStore *s, i64 *blocks, i64 *bad, i64 *fixed, i64 *rows_bad, i64 *rows_healed, i32 *jdamaged)
{
    pthread_mutex_lock(&s->heal_mu);                             /* the lens and row rebuilds count under it */
    if (blocks) *blocks = s->sum_blocks;
    if (bad) *bad = s->sum_bad;
    if (fixed) *fixed = s->sum_fixed;
    pthread_mutex_unlock(&s->heal_mu);
#if defined(__GNUC__)
    if (rows_bad) *rows_bad = __atomic_load_n(&s->rot_rows, __ATOMIC_RELAXED);
    if (rows_healed) *rows_healed = __atomic_load_n(&s->rot_healed, __ATOMIC_RELAXED);
#else
    if (rows_bad) *rows_bad = s->rot_rows;
    if (rows_healed) *rows_healed = s->rot_healed;
#endif
    if (jdamaged) *jdamaged = s->jdamaged;
}

/* Whether searches and lookups test every row they read against its check (on by default; off saves the hashing of
 * the re-scored rows, a few percent of a search). */
void store_set_check_reads(RagStore *s, i32 on) { s->vreads = on != 0; }

static void store_msum_free_(void *p) { SSum *z = (SSum *)p; if (z) { free(z->tab); free(z); } }

/* Row r failed its check. Its bytes that lie in the mapped file (the store never writes to mapped chunks) are rebuilt:
 * the blocks under them from the parity, confirmed by their digests and written back - the mapping shows the page
 * cache, so the row reads true again, as ZFS heals a read. 1 when the row passes its check afterwards. */
static i32 store_row_heal_(RagStore *s, i32 r)
{
    SSum *z = (SSum *)s->msum;
    const u8 *part[8], *m0 = (const u8 *)s->map;
    size_t n[8];
    u64 done[16], off, b, e, g;
    i32 np = 0, nd = 0, i, k;
    u8 *mem;
    SJob j;
    if (!z || !z->tab || s->mfd < 0 || !m0) return 0;
    part[np] = SCOL_AT(&s->vec, r); n[np++] = s->vec.esz;
    if (s->vbytes == 1) { part[np] = SCOL_AT(&s->vsc, r); n[np++] = 4; }
    part[np] = SCOL_AT(&s->idc, r); n[np++] = 8;
    part[np] = SCOL_AT(&s->metac, r); n[np++] = 8;
    part[np] = SCOL_AT(&s->rck, r); n[np++] = 4;
    if (s->metric) { part[np] = SCOL_AT(&s->nrm, r); n[np++] = 4; }
    if (s->has_num) { part[np] = SCOL_AT(&s->numc, r); n[np++] = 8; }
    if (s->has_ns) { part[np] = SCOL_AT(&s->nsc, r); n[np++] = 8; }
    if (!(mem = (u8 *)malloc(((size_t)1 + 3u * z->m) << z->blog))) return 0;
    memset(&j, 0, sizeof(j));
    j.z = z; j.heal = 1; j.gstep = 1; j.ubad = ~(u64)0;
    pthread_mutex_lock(&s->heal_mu);                             /* one descriptor, rare: one rebuild at a time */
    for (i = 0; i < np; ++i) {
        if (part[i] < m0 || part[i] + n[i] > m0 + s->map_len) continue;   /* not from the file: nothing to rebuild */
        off = (u64)(part[i] - m0);
        for (b = off >> z->blog, e = (off + n[i] - 1) >> z->blog; b <= e && b < z->N; ++b) {
            g = b % z->G;
            for (k = 0; k < nd && done[k] != g; ++k) {}
            if (k < nd) continue;
            if (nd < 16) done[nd++] = g;
            ssum_group_(&j, g, s->mfd, s->mfd, mem);
        }
    }
    if (j.fixed && fdatasync(s->mfd) != 0) j.err = 1;
    s->sum_bad += j.bad; s->sum_fixed += j.fixed;
    pthread_mutex_unlock(&s->heal_mu);
    free(mem);
    return STORE_RCK(s, r) == store_rck_row_(s, r);
}

/* The blocks of the loaded file under [p, p + n) rebuilt from the parity and written back (bytes outside the mapping:
 * nothing to rebuild from). Under heal_mu. How many blocks were rebuilt. */
static i64 store_bytes_heal_(RagStore *s, const void *p, size_t n)
{
    SSum *z = (SSum *)s->msum;
    const u8 *m0 = (const u8 *)s->map, *q = (const u8 *)p;
    u64 b, e;
    u8 *mem;
    SJob j;
    if (!z || !z->tab || s->mfd < 0 || !m0 || !n || q < m0 || q + n > m0 + s->map_len) return 0;
    if (!(mem = (u8 *)malloc(((size_t)1 + 3u * z->m) << z->blog))) return 0;
    memset(&j, 0, sizeof(j));
    j.z = z; j.heal = !s->ro; j.gstep = 1; j.ubad = ~(u64)0;
    for (b = (u64)(q - m0) >> z->blog, e = (u64)(q - m0 + n - 1) >> z->blog; b <= e && b < z->N; ++b)
        ssum_group_(&j, b % z->G, s->mfd, s->ro ? -1 : s->mfd, mem);
    if (j.fixed && fdatasync(s->mfd) != 0) j.err = 1;
    s->sum_bad += j.bad; s->sum_fixed += j.fixed;
    free(mem);
    return j.fixed;
}

/* the CRC-32C of text v as it lies in the heap's blocks */
static u32 store_tx_crc_(const RagStore *s, u64 v)
{
    i64 at = STORE_TXOFF(v), n = STORE_TXLEN(v), done = 0;
    u32 c = 0xffffffffu;
    while (done < n) {
        i64 p = at + done, room = STORE_TXB - (p & (STORE_TXB - 1)), take = n - done < room ? n - done : room;
        c = store_crc32c_up_(c, s->tx.b[p >> STORE_TXSHIFT] + (p & (STORE_TXB - 1)), (size_t)take);
        done += take;
    }
    c = ~c;
    return c ? c : 1u;
}

/* Row r's text, checked against its CRC on every read (a text from an older file, without one: through the lens, when
 * the store has it); failing, its bytes rebuilt from the parity and checked again. 1 sound (or no text), 0 damaged. */
static i32 store_tx_ok_(RagStore *s, i32 r)
{
    u64 v = s->has_text ? STORE_TXO(s, r) : 0;
    u32 want = v ? STORE_TCK(s, r) : 0;
    i64 at, n, done = 0;
    i32 ok;
    if (!v) return 1;
    if (!want) return store_tx_vouch_(s, v) == 0;
    if (store_tx_crc_(s, v) == want) return 1;
    pthread_mutex_lock(&s->heal_mu);                             /* one rebuild at a time, as a row's */
    for (at = STORE_TXOFF(v), n = STORE_TXLEN(v); done < n; ) {
        i64 p = at + done, room = STORE_TXB - (p & (STORE_TXB - 1)), take = n - done < room ? n - done : room;
        store_bytes_heal_(s, s->tx.b[p >> STORE_TXSHIFT] + (p & (STORE_TXB - 1)), (size_t)take);
        done += take;
    }
    ok = store_tx_crc_(s, v) == want;
    pthread_mutex_unlock(&s->heal_mu);
    return ok;
}

/* ---- the lens: a file checked block by block when its blocks are needed, and by a thread in order ---- */

static i32 store_lens_ms_ = 4;      /* the thread's pause between two blocks: 64 KiB every 4 ms, 16 MB/s (tests: 0) */

/* block b of the file through rfd against its digest; damaged, its group rebuilt from the parity and written back
 * through wfd (-1: found, not fixed). mem: (1 + 3m) blocks. 0 intact (or rebuilt), -1 damaged beyond repair. */
static i32 ssum_vouch_blk_(const SSum *z, int rfd, int wfd, u64 b, u8 *mem, i64 *bad, i64 *fixed)
{
    size_t B = (size_t)1 << z->blog;
    u8 h[32];
    SJob j;
    if (ssum_read_(rfd, mem, B, b << z->blog) == 0) { b3_hash(mem, B, h); if (!memcmp(h, z->tab + b * 32u, 32)) return 0; }
    memset(&j, 0, sizeof(j));
    j.z = z; j.heal = wfd >= 0; j.parity = b >= z->N; j.gstep = 1; j.ubad = ~(u64)0;
    ssum_group_(&j, b < z->N ? b % z->G : (b - z->N) / z->m, rfd, wfd, mem);
    *bad += j.bad; *fixed += j.fixed;
    if (j.fixed && fdatasync(wfd) != 0) return -1;
    if (ssum_read_(rfd, mem, B, b << z->blog) != 0) return -1;
    b3_hash(mem, B, h);
    return memcmp(h, z->tab + b * 32u, 32) == 0 ? 0 : -1;
}

/* the blocks under file bytes [off, off + n), at a load (no store yet): 0 every one intact or rebuilt, else -1 */
static i32 ssum_vouch_(const SSum *z, const char *path, i32 ro, u64 off, u64 n)
{
    i64 bad = 0, fixed = 0;
    int rfd, wfd;
    u64 b, e;
    u8 *mem;
    i32 r = 0;
    if (!z || !z->tab || !n) return 0;
    if (off + n > z->N << z->blog) return -1;                    /* past the data blocks: not what the table covers */
    if (!(mem = (u8 *)malloc(((size_t)1 + 3u * z->m) << z->blog))) return -1;
    rfd = open(path, O_RDONLY);
    wfd = ro ? -1 : open(path, O_RDWR);
    if (rfd < 0) r = -1;
    for (b = off >> z->blog, e = (off + n - 1) >> z->blog; !r && b <= e; ++b) r = ssum_vouch_blk_(z, rfd, wfd, b, mem, &bad, &fixed);
    if (rfd >= 0) close(rfd);
    if (wfd >= 0) close(wfd);
    free(mem);
    return r;
}

/* block b of the loaded file, once: under heal_mu, its bit set when it is intact (or was rebuilt); 0, or -1 damaged */
static i32 store_vouch_b_(RagStore *s, i64 b)
{
    const SSum *z = (const SSum *)s->msum;
    i32 r = 0;
    u64 w = (u64)1 << (b & 63);
    if (!s->vbits || b < 0 || b >= s->vblocks) return 0;
#if defined(__GNUC__)
    if (__atomic_load_n(&s->vbits[b >> 6], __ATOMIC_ACQUIRE) & w) return 0;
#endif
    pthread_mutex_lock(&s->heal_mu);
    if (!(s->vbits[b >> 6] & w)) {
        r = ssum_vouch_blk_(z, s->mfd, s->ro ? -1 : s->mfd, (u64)b, s->vmem, &s->sum_bad, &s->sum_fixed);
#if defined(__GNUC__)
        if (!r) __atomic_fetch_or(&s->vbits[b >> 6], w, __ATOMIC_RELEASE);
#else
        if (!r) s->vbits[b >> 6] |= w;
#endif
    }
    pthread_mutex_unlock(&s->heal_mu);
    return r;
}

/* the blocks under [p, p + n) of the loaded file's mapping (bytes elsewhere: nothing to check); 0, or -1 damaged */
static i32 store_vouch_(RagStore *s, const void *p, size_t n)
{
    const SSum *z = (const SSum *)s->msum;
    const u8 *m0 = (const u8 *)s->map, *q = (const u8 *)p;
    i64 b, e;
    i32 r = 0;
    if (!s->vbits || !n || q < m0 || q + n > m0 + s->map_len) return 0;
    for (b = (i64)((size_t)(q - m0) >> z->blog), e = (i64)((size_t)(q - m0 + n - 1) >> z->blog); b <= e; ++b) r |= store_vouch_b_(s, b);
    return r;
}

/* the bytes of text v (store_tx_copy_'s walk over the heap's blocks) */
static i32 store_tx_vouch_(RagStore *s, u64 v)
{
    i64 at = STORE_TXOFF(v), n = STORE_TXLEN(v), done = 0;
    i32 r = 0;
    if (!s->vbits) return 0;
    while (done < n) {
        i64 p = at + done, room = STORE_TXB - (p & (STORE_TXB - 1)), take = n - done < room ? n - done : room;
        r |= store_vouch_(s, s->tx.b[p >> STORE_TXSHIFT] + (p & (STORE_TXB - 1)), (size_t)take);
        done += take;
    }
    return r;
}

/* every block not checked yet (before a save or a snapshot copies the file's bytes into a new one under fresh digests):
 * how many stayed damaged */
static i64 store_lens_finish_(RagStore *s)
{
    i64 b, bad = 0;
    for (b = 0; s->vbits && b < s->vblocks; ++b) bad += store_vouch_b_(s, b) != 0;
    return bad;
}

static void *store_lens_(void *arg)
{
    RagStore *s = (RagStore *)arg;
    i64 b;
    pthread_mutex_lock(&s->lens_mu);
    for (b = 0; b < s->vblocks && !s->lens_stop; ++b) {
        pthread_mutex_unlock(&s->lens_mu);
        store_vouch_b_(s, b);
        pthread_mutex_lock(&s->lens_mu);
        s->lens_at = b + 1;
        if (store_lens_ms_ > 0 && !s->lens_stop) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += (long)store_lens_ms_ * 1000000L;
            if (ts.tv_nsec >= 1000000000L) { ts.tv_sec += ts.tv_nsec / 1000000000L; ts.tv_nsec %= 1000000000L; }
            pthread_cond_timedwait(&s->lens_cv, &s->lens_mu, &ts);
        }
    }
    pthread_mutex_unlock(&s->lens_mu);
    return (void *)0;
}

/* after a load with the lens: the bitmap, the scratch, and the thread. Without memory or a thread: every block now */
static void store_lens_start_(RagStore *s)
{
    const SSum *z = (const SSum *)s->msum;
    if (!z || !z->tab || s->mfd < 0 || !s->map) return;
    s->vblocks = (i64)(z->N + z->P);
    s->vbits = (u64 *)calloc((size_t)((s->vblocks + 63) >> 6), sizeof(u64));
    s->vmem = (u8 *)malloc(((size_t)1 + 3u * z->m) << z->blog);
    if (!s->vbits || !s->vmem) { free(s->vbits); s->vbits = (u64 *)0; free(s->vmem); s->vmem = (u8 *)0; return; }
    s->lens_run = pthread_create(&s->lens_th, (const pthread_attr_t *)0, store_lens_, s) == 0;
    if (!s->lens_run) store_lens_finish_(s);
}

static void store_lens_stop_(RagStore *s)
{
    if (!s->lens_run) return;
    pthread_mutex_lock(&s->lens_mu);
    s->lens_stop = 1;
    pthread_cond_signal(&s->lens_cv);
    pthread_mutex_unlock(&s->lens_mu);
    pthread_join(s->lens_th, (void **)0);
    s->lens_run = 0;
}

/* the lens's progress: blocks checked in order so far, and of how many (0 and 0: the load checked every block) */
void store_lens_info(RagStore *s, i64 *done, i64 *total)
{
    pthread_mutex_lock(&s->lens_mu);
    *done = s->vbits ? (s->lens_run ? s->lens_at : s->vblocks) : 0;
    *total = s->vbits ? s->vblocks : 0;
    pthread_mutex_unlock(&s->lens_mu);
}

