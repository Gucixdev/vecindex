/* vsdiff.c - the server's kernels on the inputs of the Lean oracle, for lean4/run: the tenant id mapping and the key
 * file's tenant number (Tenant.lean), the replica's journal walk (Journal.lean) and the token bucket (Rate.lean). It
 * includes the whole server (vserv.c, its main renamed, built without TLS), so the static functions are in reach.
 *
 *   vsdiff tenant < CASES | journal < CASES | rate < CASES    the lines `vecindex-oracle CMD` prints, from the C code
 *   vsdiff cases-tenant N SEED | cases-journal N SEED | cases-rate N SEED   inputs for both sides: grids of the
 *                                                     edges, then N random cases from SEED */
#define main vserv_main_
#include "vserv.c"
#undef main

static u64 rs_ = 88172645463325252u;
static u64 rnd_(void) { rs_ ^= rs_ << 13; rs_ ^= rs_ >> 7; rs_ ^= rs_ << 17; return rs_; }
static char vs_tmp_[64];

static void put_u64_(u64 v)
{
    char b[24];
    i32 i = 23;
    b[i] = 0;
    do { b[--i] = (char)('0' + (i32)(v % 10u)); v /= 10u; } while (v);
    fputs(b + i, stdout);
}
static void put_i64_(i64 v) { if (v < 0) { putchar('-'); put_u64_((u64)0 - (u64)v); } else put_u64_((u64)v); }
static void sp_i64_(i64 v) { putchar(' '); put_i64_(v); }

static i64 get_i64_(const char **p)
{
    const char *s = *p;
    u64 v = 0;
    i32 neg = 0;
    while (*s == ' ') ++s;
    if (*s == '-') { neg = 1; ++s; }
    while (*s >= '0' && *s <= '9') v = v * 10u + (u64)(*s++ - '0');
    *p = s;
    return neg ? (i64)((u64)0 - v) : (i64)v;
}

static i32 hexv_(char c) { return c >= '0' && c <= '9' ? c - '0' : (c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1); }

/* ---- tenants ---- */

static i64 tn_edge_[] = { 0, 1, 2, 3, 524286, 524287 };

static void tenant_case_(const char *op, i64 tn, i64 a, i64 b)
{
    fputs(op, stdout); sp_i64_(tn); sp_i64_(a);
    if (!strcmp(op, "rin")) sp_i64_(b);
    putchar('\n');
}

static void cases_tenant_(i32 n)
{
    const i64 M = VS_TMASK, B = (i64)1 << 44;
    i64 xs[11];
    i32 i, j, k;
    const char *pv[] = { "1", "7", "524287", "524288", "0", "00001", "0524287", "5242870", "99999999999999999999",
                         "12a", "a", "-1", "+1", "1-", "000000000000000000000000000000001" };
    xs[0] = -(i64)((u64)1 << 62) * 2; xs[1] = -1; xs[2] = 0; xs[3] = 1; xs[4] = M - 1; xs[5] = M; xs[6] = M + 1;
    xs[7] = (i64)1 << 62; xs[8] = (i64)(((u64)1 << 63) - 1); xs[9] = 12345; xs[10] = -M;
    for (i = 0; i < 6; ++i) {
        i64 tn = tn_edge_[i];
        for (j = 0; j < 11; ++j) {
            tenant_case_("in", tn, xs[j], 0);
            for (k = 0; k < 11; ++k) tenant_case_("rin", tn, xs[j], xs[k]);
        }
        if (tn) {
            i64 ys[8];
            ys[0] = tn * B; ys[1] = tn * B + M; ys[2] = (tn + 1) * B; ys[3] = tn * B - 1; ys[4] = -1; ys[5] = 0;
            ys[6] = -B; ys[7] = (i64)(((u64)1 << 63) - 1);
            for (j = 0; j < 8; ++j) tenant_case_("own", tn, ys[j], 0);
            for (j = 0; j < 4; ++j) if (ys[j] >= 0) tenant_case_("out", tn, ys[j], 0);
        }
    }
    for (i = 0; i < n; ++i) {
        i64 tn = (i64)(rnd_() % 524288u), x = (i64)(rnd_() % ((u64)M + 3u)) - 1, y = (i64)(rnd_() % ((u64)M + 1u));
        i64 lo = (i64)(rnd_() % ((u64)M + 9u)) - 4, hi = (i64)(rnd_() % ((u64)M + 9u)) - 4;
        tenant_case_("in", tn, x, 0);
        tenant_case_("rin", tn, lo, hi);
        if (tn) { tenant_case_("own", tn, tn * B + y, 0); tenant_case_("own", tn, (i64)(rnd_() >> 1), 0); tenant_case_("out", tn, tn * B + y, 0); }
    }
    for (i = 0; i < (i32)(sizeof(pv) / sizeof(pv[0])); ++i) {
        const char *s = pv[i];
        fputs("parse ", stdout);
        while (*s) printf("%02x", (unsigned)(unsigned char)*s++);
        putchar('\n');
    }
    for (i = 0; i < n; ++i) {
        static const char al[] = "0123456789a-+";
        i32 l = 1 + (i32)(rnd_() % 9u);
        fputs("parse ", stdout);
        for (j = 0; j < l; ++j) printf("%02x", (unsigned)(unsigned char)(rnd_() % 4u ? al[rnd_() % 10u] : al[rnd_() % 13u]));
        putchar('\n');
    }
}

/* the key file's tenant number through vs_keys_load_ itself */
static void parse_c_(const char *tok)
{
    char path[96];
    FILE *f;
    i32 r, i;
    sprintf(path, "%s/k.tsv", vs_tmp_);
    f = fopen(path, "w");
    fprintf(f, "k_parse_0123456789abcd %s\n", tok);
    fclose(f);
    if (vs_kst) for (i = 0; i < vs_nkeys; ++i) pthread_mutex_destroy(&vs_kst[i].mu);   /* after a refused file: none */
    free(vs_keys); free(vs_kst); vs_keys = (VsKey *)0; vs_kst = (VsKeySt *)0; vs_nkeys = 0;
    r = vs_keys_load_(path);
    if (r == 0) put_i64_(vs_keys[0].tn); else fputs("none", stdout);
}

static void tenant_in_(void)
{
    char line[512];
    VsConn c;
    while (fgets(line, sizeof(line), stdin)) {
        const char *p = line + 3;                               /* "in " / "out" / "own" / "rin", then numbers */
        i64 tn, a, b;
        memset(&c, 0, sizeof(c));
        if (!strncmp(line, "parse ", 6)) {
            char tok[128];
            i32 n = 0;
            const char *h = line + 6;
            while (hexv_(h[0]) >= 0 && hexv_(h[1]) >= 0 && n < 126) { tok[n++] = (char)(hexv_(h[0]) * 16 + hexv_(h[1])); h += 2; }
            tok[n] = 0;
            line[strcspn(line, "\n")] = 0;
            printf("%s ", line);
            parse_c_(tok);
            putchar('\n');
            continue;
        }
        if (strlen(line) < 4) continue;
        tn = get_i64_(&p); a = get_i64_(&p);
        c.tn = tn;
        line[3] = 0;
        if (!strcmp(line, "in ")) {
            i64 x = a;
            fputs("in", stdout); sp_i64_(tn); sp_i64_(a); putchar(' ');
            if (vs_in_(&c, &x) == 0) put_i64_(x); else fputs("none", stdout);
        } else if (!strcmp(line, "out")) {
            i64 y = a;
            vs_out_(&c, &y, 1);
            fputs("out", stdout); sp_i64_(tn); sp_i64_(a); sp_i64_(y);
        } else if (!strcmp(line, "own")) {
            i64 y = a;
            fputs("own", stdout); sp_i64_(tn); sp_i64_(a); putchar(' ');
            if (vs_own_(&c, &y)) put_i64_(y); else fputs("none", stdout);
        } else if (!strcmp(line, "rin")) {
            i64 lo = a, hi;
            b = get_i64_(&p); hi = b;
            fputs("rin", stdout); sp_i64_(tn); sp_i64_(a); sp_i64_(b); putchar(' ');
            if (vs_rin_(&c, &lo, &hi) == 0) { put_i64_(lo); sp_i64_(hi); } else fputs("none", stdout);
        } else continue;
        putchar('\n');
    }
}

/* ---- the replica's journal walk ---- */

static void put_le32_(u8 *p, u32 v) { p[0] = (u8)v; p[1] = (u8)(v >> 8); p[2] = (u8)(v >> 16); p[3] = (u8)(v >> 24); }

static void cases_journal_(i32 n)
{
    u8 b[4096];
    i32 i;
    for (i = 0; i < n; ++i) {
        u64 st = rnd_() | 1u, fst = st;
        i32 len = 0, k, nf = (i32)(rnd_() % 5u), mode = (i32)(rnd_() % 8u), fr[8];
        u32 crc, dim = (u32)(rnd_() % 1000u);
        put_le32_(b, 0x334a5856u); memcpy(b + 4, &dim, 4); memcpy(b + 8, &st, 8);
        crc = store_crc32c_(b + 4, 12); put_le32_(b + 16, crc);
        len = 20;
        for (k = 0; k < nf; ++k) {
            u32 pl = (u32)(rnd_() % 48u), j;
            fr[k] = len;
            b[len] = 'G'; put_le32_(b + len + 1, pl);
            for (j = 0; j < pl; ++j) b[len + 9 + j] = (u8)rnd_();
            put_le32_(b + len + 5, store_crc32c_(b + len + 9, pl));
            len += 9 + (i32)pl;
        }
        switch (mode) {
        case 1: len = (i32)(rnd_() % (u64)(len + 1)); break;                           /* a torn copy: any prefix */
        case 2: b[rnd_() % (u64)len] ^= (u8)(1u << (rnd_() % 8u)); break;              /* one bit flipped */
        case 3: fst ^= 1u << (rnd_() % 64u); break;                                    /* the file beside is another */
        case 4: if (nf) put_le32_(b + fr[rnd_() % (u64)nf] + 1, (u32)STORE_BATCH_MAX + 1u); break;   /* past the cap */
        case 5: { i32 z = (i32)(rnd_() % 30u); memset(b + len, 0, (size_t)z); len += z; } break;    /* zeros after */
        case 6: if (nf) b[fr[rnd_() % (u64)nf]] = 'H'; break;                         /* not a frame */
        case 7: fst = 0; break;                                                         /* a file with no stamp */
        default: break;
        }
        fputs("journal ", stdout); put_u64_(fst); putchar(' ');
        if (!len) putchar('-');
        for (k = 0; k < len; ++k) printf("%02x", (unsigned)b[k]);
        putchar('\n');
    }
}

static void journal_in_(void)
{
    static char line[16384];
    char path[96], jn[100];
    while (fgets(line, sizeof(line), stdin)) {
        const char *p = line + 8, *h;
        u64 fst;
        u8 hdr[32];
        FILE *f;
        if (strncmp(line, "journal ", 8)) continue;
        fst = (u64)get_i64_(&p);
        h = p + 1;
        sprintf(path, "%s/r.vxi", vs_tmp_); sprintf(jn, "%s/r.vxi.j", vs_tmp_);
        memset(hdr, 0, sizeof(hdr));
        put_le32_(hdr, STORE_MAGIC); put_le32_(hdr + 4, (u32)STORE_VERSION9);
        put_le32_(hdr + 24, (u32)(fst & 0xffffffffu)); put_le32_(hdr + 28, (u32)(fst >> 32));
        f = fopen(path, "wb"); fwrite(hdr, 1, 32, f); fclose(f);
        f = fopen(jn, "wb");
        if (*h != '-') while (hexv_(h[0]) >= 0 && hexv_(h[1]) >= 0) { fputc(hexv_(h[0]) * 16 + hexv_(h[1]), f); h += 2; }
        fclose(f);
        fputs("journal ", stdout); put_u64_(vs_rjtrim_(path, jn)); putchar('\n');
    }
}

/* ---- the token bucket ---- */

static void cases_rate_(i32 n)
{
    static const u32 qs[] = { 1, 2, 3, 5, 10, 100, 1000 };
    i32 i, k;
    for (i = 0; i < n; ++i) {
        u32 q = qs[rnd_() % 7u], t = 0, spread = 1u + (u32)(rnd_() % 4096u);
        printf("rate %u", (unsigned)q);
        for (k = 0; k < 200; ++k) { t += (u32)(rnd_() % spread); printf(" %u", (unsigned)t); }
        putchar('\n');
    }
}

static void rate_in_(void)
{
    static char line[8192];
    while (fgets(line, sizeof(line), stdin)) {
        const char *p = line + 5;
        f64 q, tok, last = 0.0;
        if (strncmp(line, "rate ", 5)) continue;
        q = (f64)get_i64_(&p);
        tok = q;
        fputs("rate ", stdout);
        while (*p == ' ') {
            f64 t = (f64)get_i64_(&p) / 1024.0;
            putchar(vs_bucket_(&tok, &last, t, q) ? '1' : '0');
        }
        putchar('\n');
    }
}

int main(int argc, char **argv)
{
    const char *cmd = argc > 1 ? argv[1] : "";
    i32 n = argc > 2 ? atoi(argv[2]) : 100;
    if (argc > 3) rs_ = (u64)strtoul(argv[3], (char **)0, 10) * 2654435761u + 1u;
    strcpy(vs_tmp_, "/tmp/vsdiff-XXXXXX");
    if (!strcmp(cmd, "tenant") || !strcmp(cmd, "journal")) {
        if (!mkdtemp(vs_tmp_)) return 2;
        if (!freopen("/dev/null", "w", stderr)) {}                  /* the key loader's refusals */
    }
    if (!strcmp(cmd, "tenant")) tenant_in_();
    else if (!strcmp(cmd, "journal")) journal_in_();
    else if (!strcmp(cmd, "rate")) rate_in_();
    else if (!strcmp(cmd, "cases-tenant")) cases_tenant_(n);
    else if (!strcmp(cmd, "cases-journal")) cases_journal_(n);
    else if (!strcmp(cmd, "cases-rate")) cases_rate_(n);
    else { fputs("usage: vsdiff tenant|journal|rate|cases-tenant|cases-journal|cases-rate [N SEED]\n", stderr); return 2; }
    if (vs_tmp_[13] != 'X') {
        char c2[96];
        sprintf(c2, "%s/k.tsv", vs_tmp_); remove(c2);
        sprintf(c2, "%s/r.vxi", vs_tmp_); remove(c2);
        sprintf(c2, "%s/r.vxi.j", vs_tmp_); remove(c2);
        rmdir(vs_tmp_);
    }
    return 0;
}
