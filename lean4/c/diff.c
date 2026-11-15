/* diff.c - vecindex's C kernels on the inputs of the Lean oracle, for lean4/run. It includes the whole store, so its
 * static functions are in reach, and the kernels TrustLean printed (gen/trustlean.c).
 *
 *   diff gf | vd | layout | chunks     the lines `vecindex-oracle CMD` prints, from the C code
 *   diff crc < MSGS | block < CASES | topk < CASES
 *   diff cases-crc N SEED | cases-block N SEED | cases-topk N SEED   random inputs for both sides
 *   diff props N SEED                            what is checked in C alone; "ok ..." lines, exit 1 on a failure
 *
 * Built by lean4/run three times (-march=native, haswell, x86-64), so the AVX-512, AVX2 and scalar kernels and both
 * CRC paths all meet the same oracle. */
#include "store.c"
#include "trustlean.c"

static u64 rs_ = 88172645463325252u;
static u64 rnd_(void) { rs_ ^= rs_ << 13; rs_ ^= rs_ >> 7; rs_ ^= rs_ << 17; return rs_; }
static u32 rnd32_(void) { return (u32)(rnd_() >> 16); }
static i32 fails_;

static void put_u64_(u64 v)
{
    char b[24];
    i32 i = 23;
    b[i] = 0;
    do { b[--i] = (char)('0' + (i32)(v % 10u)); v /= 10u; } while (v);
    fputs(b + i, stdout);
}
static void put_i64_(i64 v) { if (v < 0) { putchar('-'); put_u64_((u64)0 - (u64)v); } else put_u64_((u64)v); }
static void sp_u64_(u64 v) { putchar(' '); put_u64_(v); }

static void check_(i32 ok, const char *what)
{
    printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) fails_ = 1;
}

/* ---- the oracle's tables ---- */

static void gf_out_(void)
{
    u32 a, b, k, p, i;
    u64 h = 14695981039346656037u;
    gf_init_();
    fputs("exp", stdout); for (i = 0; i < 512; ++i) sp_u64_(gf_exp_[i]); putchar('\n');
    fputs("log", stdout); for (i = 0; i < 256; ++i) sp_u64_(gf_log_[i]); putchar('\n');
    for (a = 0; a < 256; ++a) {
        fputs("mul ", stdout); put_u64_(a);
        for (b = 0; b < 256; ++b) sp_u64_(gf_mul_(a, b));
        putchar('\n');
    }
    fputs("inv", stdout); for (i = 0; i < 256; ++i) sp_u64_(gf_inv_(i)); putchar('\n');
    for (k = 1; k <= 128; ++k) for (p = 0; p < 64; ++p) for (i = 0; i < k; ++i)
        h = (h ^ (u64)gf_coef_(k, p, i)) * 1099511628211u;
    fputs("coef ", stdout); put_u64_(h); putchar('\n');
}

static void vd_out_(void)
{
    const i32 bits[3] = { 1, 2, 4 };
    i32 b, i;
    for (b = 0; b < 3; ++b) {
        printf("vd %d", (int)bits[b]);
        for (i = 0; i < 2048; ++i) sp_u64_((u64)tq_vd_(2 * i + 2, bits[b]));
        putchar('\n');
    }
}

static void layout_out_(void)
{
    static const u64 Ds[10] = { 1u, 4095u, 4096u, 65535u, 65536u, 65537u, 1000000u, 123456789u, 4294967296u,
                                1099511627777u };
    static const u32 blogs[3] = { 12, 16, 24 }, ks[7] = { 1, 2, 3, 7, 32, 100, 128 }, ms[5] = { 0, 1, 2, 5, 64 };
    i32 a, b, c, d;
    SSum z;
    for (a = 0; a < 10; ++a) for (b = 0; b < 3; ++b) for (c = 0; c < 7; ++c) for (d = 0; d < 5; ++d) {
        if (ssum_layout_(&z, Ds[a], blogs[b], ks[c], ms[d])) { printf("layout refused\n"); continue; }
        fputs("layout", stdout);
        sp_u64_(Ds[a]); sp_u64_(blogs[b]); sp_u64_(ks[c]); sp_u64_(ms[d]);
        sp_u64_(z.N); sp_u64_(z.G); sp_u64_(z.P); sp_u64_(z.ta); sp_u64_(z.tra); sp_u64_(z.tb); sp_u64_(z.fsize);
        putchar('\n');
    }
}

/* the chunks of tq_wht_, read off its output: a one-hot input spreads over exactly its own chunk */
static void chunks_out_(void)
{
    static f32 x[1024];
    i32 dpad, i, j, lo, hi;
    for (dpad = 1; dpad <= 1024; ++dpad) {
        printf("chunks %d", (int)dpad);
        for (i = 0; i < dpad; i = hi + 1) {
            for (j = 0; j < dpad; ++j) x[j] = 0.0f;
            x[i] = 1.0f;
            tq_wht_(x, dpad);
            for (lo = 0; lo < dpad && x[lo] == 0.0f; ++lo) {}
            for (hi = dpad - 1; hi > lo && x[hi] == 0.0f; --hi) {}
            printf(" %d:%d", (int)lo, (int)(hi - lo + 1));
        }
        putchar('\n');
    }
}

/* ---- random cases, read back ---- */

static i32 hexv_(i32 c) { return c <= '9' ? c - '0' : (c | 32) - 'a' + 10; }

static void crc_in_(void)
{
    static u8 buf[1 << 16];
    char line[(1 << 17) + 8];
    while (fgets(line, (int)sizeof(line), stdin)) {
        size_t n = 0, i = 0;
        while (line[i] && line[i] != '\n' && line[i] != '-' && line[i + 1]) {
            buf[n++] = (u8)(hexv_(line[i]) * 16 + hexv_(line[i + 1]));
            i += 2;
        }
        printf("crc %08lx\n", (unsigned long)store_crc32c_(buf, n));
    }
}

static void cases_crc_(i32 n)
{
    i32 c, i, len;
    printf("-\n");
    printf("313233343536373839\n");
    for (c = 0; c < n; ++c) {
        len = c % 4 == 0 ? (i32)(rnd32_() % 4000u) : (i32)(rnd32_() % 40u);
        if (!len) { printf("-\n"); continue; }
        for (i = 0; i < len; ++i) printf("%02x", (unsigned)(rnd32_() & 255u));
        putchar('\n');
    }
}

static i32 read_i_(i64 *v)
{
    long x;
    if (scanf("%ld", &x) != 1) return 0;
    *v = (i64)x;
    return 1;
}

static void block_in_(void)
{
    i64 vd, x;
    i32 i;
    u16 out[32];
    while (read_i_(&vd)) {
        u8 *blk = (u8 *)calloc((size_t)vd * 16 + 64, 1), *lut = (u8 *)calloc((size_t)vd * 16 + 64, 1);
        for (i = 0; i < vd * 16; ++i) { if (!read_i_(&x)) return; blk[i] = (u8)x; }
        for (i = 0; i < vd * 16; ++i) { if (!read_i_(&x)) return; lut[i] = (u8)x; }
        tq_block_(blk, lut, (i32)vd, 0, out);
        fputs("block", stdout);
        for (i = 0; i < 32; ++i) sp_u64_(out[i]);
        putchar('\n');
        free(blk); free(lut);
    }
}

static void cases_block_(i32 n)
{
    i32 c, i, vd;
    for (c = 0; c < n; ++c) {
        vd = 2 * (1 + (i32)(rnd32_() % 64u));                     /* even, as tq_vd_ makes it */
        printf("%d\n", (int)vd);
        for (i = 0; i < vd * 16; ++i) printf("%u ", (unsigned)(rnd32_() & 255u));
        putchar('\n');
        for (i = 0; i < vd * 16; ++i) printf("%u ", (unsigned)(c % 3 ? rnd32_() & 255u : rnd32_() % 7u));
        putchar('\n');
    }
}

static void topk_in_(void)
{
    i64 k, n, id, sc;
    i32 i, have;
    while (read_i_(&k) && read_i_(&n)) {
        i64 *ids = (i64 *)malloc(sizeof(i64) * (size_t)(k + 1));
        f32 *scores = (f32 *)malloc(sizeof(f32) * (size_t)(k + 1));
        have = 0;
        for (i = 0; i < n; ++i) {
            if (!read_i_(&id) || !read_i_(&sc)) return;
            eng_topk_(ids, scores, (i32)k, &have, id, (f32)sc);       /* the store's own insertion (core.c) */
        }
        fputs("topk", stdout);
        for (i = 0; i < have; ++i) { putchar(' '); put_i64_(ids[i]); putchar(':'); put_i64_((i64)scores[i]); }
        putchar('\n');
        free(ids); free(scores);
    }
}

static void cases_topk_(i32 n)
{
    i32 c, i, m, k;
    for (c = 0; c < n; ++c) {
        k = 1 + (i32)(rnd32_() % 12u);
        m = (i32)(rnd32_() % 60u);
        printf("%d %d", (int)k, (int)m);
        for (i = 0; i < m; ++i) printf(" %d %d", (int)i, (int)(rnd32_() % 21u) - 10);
        putchar('\n');
    }
}

/* the namespace set of a search over several (tq_in_set_): cases "n v a1 .. an", the set sorted without repeats */
static void nsset_in_(void)
{
    i64 n, v, a[64];
    i32 i;
    while (read_i_(&n) && read_i_(&v)) {
        for (i = 0; i < n; ++i) if (!read_i_(&a[i])) return;
        printf("nsset %d\n", (int)tq_in_set_(a, (i32)n, v));
    }
}

static void cases_nsset_(i32 n)
{
    i32 c, i, m;
    i64 a[64], v;
    for (c = 0; c < n * 10; ++c) {
        m = 1 + (i32)(rnd32_() % 40u);
        a[0] = (c % 7 == 0) ? -(i64)(((u64)1 << 62) - 1) * 2 - 2 : (i64)(rnd32_() % 41u) - 20;   /* i64's minimum, at times */
        for (i = 1; i < m; ++i) a[i] = a[i - 1] + 1 + (i64)(rnd32_() % 5u);
        if (c % 11 == 0) a[m - 1] = (i64)(((u64)1 << 63) - 1);                                   /* ... and maximum */
        switch (rnd32_() % 4u) {
        case 0: v = a[rnd32_() % (u32)m]; break;                                                  /* a member */
        case 1: v = a[0] > -(i64)(((u64)1 << 62)) ? a[0] - 1 - (i64)(rnd32_() % 3u) : a[0]; break;   /* below, or the least */
        case 2: v = a[m - 1] == (i64)(((u64)1 << 63) - 1) ? a[m - 1] : a[m - 1] + 1; break;       /* above, or the most */
        default: v = a[rnd32_() % (u32)m] + 1; break;                                             /* next to one */
        }
        printf("%d ", (int)m); put_i64_(v);
        for (i = 0; i < m; ++i) { putchar(' '); put_i64_(a[i]); }
        putchar('\n');
    }
}

/* ---- checked in C alone ---- */

static i32 count_below_(f32 v) { i32 i, c = 0; for (i = 0; i < 15; ++i) c += v > TQ_B[i]; return c; }

static void props_codes_(i32 n)
{
    static const f32 special[] = { 0.0f, -0.0f, 1e30f, -1e30f, 3.4e38f, -3.4e38f, 1e-40f, -1e-40f };
    i32 i, ok = 1, ok3 = 1;
    f32 v, nan;
    for (i = 0; i < 16; ++i) {                                  /* the boundaries themselves and both neighbours */
        f32 b = TQ_B[i < 15 ? i : 14];
        f32 e[3];
        i32 j;
        e[0] = b; e[1] = b * 1.0001f + 1e-6f; e[2] = b * 0.9999f - 1e-6f;
        for (j = 0; j < 3; ++j) { if (tq_code_(e[j]) != count_below_(e[j])) ok = 0; if (((tq_code_(e[j]) & 8u) != 0) != (e[j] > 0.0f)) ok3 = 0; }
    }
    for (i = 0; i < (i32)(sizeof(special) / sizeof(special[0])); ++i) {
        if (tq_code_(special[i]) != count_below_(special[i])) ok = 0;
        if (((tq_code_(special[i]) & 8u) != 0) != (special[i] > 0.0f)) ok3 = 0;
    }
    for (i = 0; i < n; ++i) {
        v = (f32)((i64)(rnd_() % 2000001u) - 1000000) / 250000.0f;
        if (tq_code_(v) != count_below_(v)) ok = 0;
        if (((tq_code_(v) & 8u) != 0) != (v > 0.0f)) ok3 = 0;
    }
    nan = 0.0f; nan = nan / nan;
    if (tq_code_(nan) != 0) ok = 0;
    check_(ok, "tq_code_ is the count of TQ_B boundaries below the value (random, boundaries, specials, NaN)");
    check_(ok3, "bit 3 of tq_code_ is v > 0, the 1-bit code (cascade)");
}

static void props_trustlean_(i32 n)
{
    u32 x = 1, i, b, c, k, cd;
    i32 ok = 1, j;
    u8 blk[16], cdv[1], msg[64];
    gf_init_();
    for (i = 0; i < 255; ++i) { if (gf_exp_[i] != x) ok = 0; x = tl_gf_xtime(x); }
    check_(ok && x == 1, "TrustLean xtime walks gf_exp_ exactly and returns to 1 after 255 steps");
    ok = 1;
    for (j = 0; j < n; ++j) {
        c = rnd32_();
        for (b = 0; b < 256; ++b) { u8 by = (u8)b; if (store_crc32c_up_(c, &by, 1) != tl_crc32c_byte(c, b)) ok = 0; }
    }
    for (j = 0; j < n; ++j) {
        u32 len = rnd32_() % 64u, r = 0xffffffffu;
        for (i = 0; i < len; ++i) msg[i] = (u8)rnd32_();
        for (i = 0; i < len; ++i) r = tl_crc32c_byte(r, msg[i]);
        if ((~r) != store_crc32c_(msg, len)) ok = 0;
    }
    check_(ok, "TrustLean CRC-32C byte kernel = store_crc32c_up_ on every byte, and on whole messages");
    ok = 1;
    for (b = 0; b < 256; ++b) for (k = 0; k < 32; ++k) for (cd = 0; cd < 16; ++cd) {
        for (j = 0; j < 16; ++j) blk[j] = (u8)(b ^ (u32)j * 37u);
        blk[k & 15] = (u8)b;
        cdv[0] = (u8)cd;
        tq_put_lane_(blk, 1, (i32)k, cdv);
        if (blk[k & 15] != (u8)tl_put_nib(b, k, cd)) ok = 0;
        for (j = 0; j < 16; ++j) if (j != (i32)(k & 15) && blk[j] != (u8)((u32)b ^ (u32)j * 37u)) ok = 0;
    }
    check_(ok, "TrustLean lane kernel = tq_put_lane_ on every byte, row and code; other bytes untouched");
}


static void props_rotation_(i32 n)
{
    static f32 x[1024], y[1024];
    i32 c, i, dpad, ok = 1;
    for (c = 0; c < n; ++c) {
        f64 before = 0.0, after = 0.0, nx = 0.0, ny = 0.0;
        dpad = 1 + (i32)(rnd32_() % 1023u);
        for (i = 0; i < dpad; ++i) { x[i] = (f32)((i32)(rnd32_() % 2001u) - 1000) / 1000.0f; y[i] = (f32)((i32)(rnd32_() % 2001u) - 1000) / 1000.0f; }
        for (i = 0; i < dpad; ++i) { before += (f64)x[i] * y[i]; nx += (f64)x[i] * x[i]; ny += (f64)y[i] * y[i]; }
        tq_wht_(x, dpad); tq_wht_(y, dpad);
        for (i = 0; i < dpad; ++i) after += (f64)x[i] * y[i];
        if (fabs(after - before) > 1e-5 * sqrt(nx * ny) + 1e-9) ok = 0;
    }
    check_(ok, "tq_wht_ keeps inner products (to f32 rounding), random dpad up to 1023");
}

/* the rebuild end to end: a file with its parity, damage spread over its groups, then the scrub */
static void props_rs_(i32 n)
{
    static const u32 ks[6] = { 1, 2, 3, 7, 32, 128 }, ms[5] = { 0, 1, 2, 3, 5 };
    char path[64];
    i32 c, ok = 1, okb = 1, fd;
    for (c = 0; c < n; ++c) {
        u32 k = ks[rnd32_() % 6u], m = ms[rnd32_() % 5u], over = c % 4 == 3;
        u64 D = (u64)(1 + rnd32_() % 40u) * 65536u - (rnd32_() % 65536u), off, i, g, nb, bad = 0, ubad;
        size_t fsz;
        u8 *orig, *now;
        FILE *f;
        SSum z, zz;
        i32 dmg, err, groups_over = 0;
        i64 fixed, found;
        struct stat st;
        sprintf(path, "/tmp/vecindex-rs-%d-%d", (int)getpid(), (int)c);
        f = fopen(path, "w+b");
        if (!f) { ok = 0; break; }
        for (off = 0; off < D; ++off) fputc((i32)(rnd32_() & 255u), f);
        fflush(f);
        if (ssum_write_file_(f, path, D, k, m, 1)) { ok = 0; fclose(f); break; }
        fclose(f);
        stat(path, &st); fsz = (size_t)st.st_size;
        orig = (u8 *)malloc(fsz); now = (u8 *)malloc(fsz);
        f = fopen(path, "rb"); if (fread(orig, 1, fsz, f) != fsz) ok = 0; fclose(f);
        ssum_layout_(&z, D, SSUM_BLOG, k, m);
        f = fopen(path, "r+b");
        for (g = 0; g < z.G; ++g) {                               /* up to m damaged blocks a group, sometimes m + 1 */
            u64 ng = (z.N - g + z.G - 1) / z.G, want = rnd32_() % (m + 1u), used[70];
            u64 t, cand;
            if (over && g == 0) { want = m + 1; ++groups_over; }
            if (want > ng + m) want = ng + m;
            for (t = 0; t < want; ++t) {
                u64 pick, b, q;
                do {
                    pick = rnd_() % (ng + m);
                    for (q = 0; q < t && used[q] != pick; ++q) {}
                } while (q < t);
                used[t] = pick;
                b = pick < ng ? g + pick * z.G : z.N + g * m + (pick - ng);
                cand = (b << SSUM_BLOG) + rnd32_() % 65536u;
                fseeko(f, (off_t)cand, SEEK_SET);
                fputc((i32)(orig[cand] ^ (u8)(1u + rnd32_() % 255u)), f);
                ++bad;
            }
        }
        fclose(f);
        fd = open(path, O_RDONLY);
        if (ssum_open_(&zz, fd, (u64)fsz, &dmg, 0)) { ok = 0; close(fd); break; }
        close(fd);
        found = ssum_check_(&zz, path, (const u8 *)0, 1, 1, 0, 1, &fixed, &ubad, &err);
        free(zz.tab);
        f = fopen(path, "rb"); if (fread(now, 1, fsz, f) != fsz) ok = 0; fclose(f);
        if ((u64)found != bad) ok = 0;
        if (!groups_over) {
            if ((u64)fixed != bad || ubad != ~(u64)0 || memcmp(orig, now, fsz) != 0) ok = 0;
        } else {
            if ((u64)fixed >= bad || ubad == ~(u64)0) okb = 0;
        }
        nb = 0;
        for (i = 0; i < fsz; ++i) nb += orig[i] != now[i];
        (void)nb;
        free(orig); free(now);
        remove(path);
    }
    check_(ok, "store_sum: every file with at most m damaged blocks a group (data or parity) comes back byte for byte");
    check_(okb, "store_sum: a group with m + 1 damaged blocks is reported, not passed off as rebuilt");
}

int main(int argc, char **argv)
{
    const char *cmd = argc > 1 ? argv[1] : "";
    i32 n = argc > 2 ? atoi(argv[2]) : 100;
    if (argc > 3) rs_ = (u64)strtoul(argv[3], (char **)0, 10) * 2654435761u + 1u;
    if (!strcmp(cmd, "gf")) gf_out_();
    else if (!strcmp(cmd, "vd")) vd_out_();
    else if (!strcmp(cmd, "layout")) layout_out_();
    else if (!strcmp(cmd, "chunks")) chunks_out_();
    else if (!strcmp(cmd, "crc")) crc_in_();
    else if (!strcmp(cmd, "block")) block_in_();
    else if (!strcmp(cmd, "topk")) topk_in_();
    else if (!strcmp(cmd, "nsset")) nsset_in_();
    else if (!strcmp(cmd, "cases-crc")) cases_crc_(n);
    else if (!strcmp(cmd, "cases-block")) cases_block_(n);
    else if (!strcmp(cmd, "cases-topk")) cases_topk_(n);
    else if (!strcmp(cmd, "cases-nsset")) cases_nsset_(n);
    else if (!strcmp(cmd, "props")) {
        props_codes_(n * 1000);
        props_trustlean_(n);
        props_rotation_(n);
        props_rs_(n / 4 > 4 ? n / 4 : 4);
        return fails_;
    } else { fputs("usage: diff gf|vd|layout|chunks|crc|block|topk|cases-crc|cases-block|cases-topk|props [N SEED]\n", stderr); return 2; }
    return 0;
}
