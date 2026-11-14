/* battle_damage.c - the store's integrity claim against damage on the disk: a saved store of 30000 rows, then copies of
 * its file with blocks destroyed at random, in runs, exactly at the limit the proofs give, and in the digest tables
 * and trailers - each copy opened with store_load, as a server opens its file, and every row read back.
 *
 * What should happen comes from lean4: block b of the file is in group b mod G, parity block N + g m + j in group g
 * (layout_cover, layout_slot); a group whose damaged data and parity blocks number at most m gets its data back byte
 * for byte (recover_gf); a run of at most m G neighbouring blocks puts at most m into any group (burst_within_parity,
 * from layout_burst), and a run of m G + 1 puts m + 1 into the first block's group (burst_beyond_parity), so the
 * first is rebuilt and the second is not. A group with more damage than
 * that has lost information. The store may then refuse the file or rebuild its index from intact rows, and every
 * other outcome is a failure: a damaged block it does not count, a refusal of a file the parity can rebuild, a
 * search that differs from the undamaged store's, and above all a row that comes back different from the one saved.
 *
 *   tests/battle_damage [TRIALS] [SEED]   random trials per parity setting (default 40); ./run builds and runs it
 */
#include "store.c"

#define BD_DIM  64
#define BD_ROWS 30000
#define BD_NQ   8

static Arena bd_arena;
static char bd_dir[64], bd_orig[96], bd_copy[96];
static f32 bd_pv[(size_t)BD_ROWS * BD_DIM];          /* every row of the undamaged store, as it gives them back */
static f32 bd_q[BD_NQ * BD_DIM], bd_ps[BD_NQ * 10];
static i64 bd_pid[BD_NQ * 10];
static i32 bd_pn[BD_NQ];
static i32 bd_fails;
static u32 bd_rs = 0x2545f491u;

static u32 bd_rng_(void) { bd_rs ^= bd_rs << 13; bd_rs ^= bd_rs >> 17; bd_rs ^= bd_rs << 5; return bd_rs; }
static u32 bd_below_(u32 n) { return n ? bd_rng_() % n : 0; }

static void bd_check_(i32 ok, const char *what)
{
    printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++bd_fails;
}

static u64 bd_get_(const u8 *p, i32 n) { u64 v = 0; i32 i; for (i = n - 1; i >= 0; --i) v = (v << 8) | p[i]; return v; }

static void bd_clean_(void)
{
    static const char *const sfx[] = { "", ".j", ".jn", ".tmp", ".snap", ".j.damaged" };
    char p[128];
    u32 i;
    for (i = 0; i < sizeof(sfx) / sizeof(sfx[0]); ++i) { sprintf(p, "%s%s", bd_copy, sfx[i]); remove(p); }
}

static u8 *bd_read_(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    u8 *b;
    long len;
    if (!f) return (u8 *)0;
    fseek(f, 0, SEEK_END); len = ftell(f); fseek(f, 0, SEEK_SET);
    b = (u8 *)malloc((size_t)len);
    if (!b || fread(b, 1, (size_t)len, f) != (size_t)len) { fclose(f); free(b); return (u8 *)0; }
    fclose(f);
    *n = (size_t)len;
    return b;
}

/* one damaged copy: what the parity layout says, what the load did */
typedef struct {
    u64 B, N, G, P, ta, tra, tb, T, fsize;
    u32 m;
    const u8 *orig;
    u8 *buf;
    u8 *hit;                                            /* per block: damaged in this copy */
} BdFile;

/* destroy block b of the copy, three ways (a few flipped bytes, zeros, noise), so that it really differs */
static void bd_hit_(BdFile *f, u64 b)
{
    u8 *p = f->buf + b * f->B;
    u32 how = bd_below_(3), i, n;
    if (how == 0) { n = 1 + bd_below_(16); for (i = 0; i < n; ++i) p[bd_below_((u32)f->B)] ^= (u8)(1 + bd_below_(255)); }
    else if (how == 1) memset(p, 0, (size_t)f->B);
    else for (i = 0; i < (u32)f->B; ++i) p[i] = (u8)bd_rng_();
    if (!memcmp(p, f->orig + b * f->B, (size_t)f->B)) p[bd_below_((u32)f->B)] ^= 0x5a;
    f->hit[b] = 1;
}

/* overwrite [off, off + n) of the copy with noise (tables and trailers) */
static void bd_smear_(BdFile *f, u64 off, u64 n)
{
    u64 i, s = off + bd_below_((u32)(n > 64 ? n - 64 : 1));
    for (i = s; i < s + 64 && i < off + n; ++i) f->buf[i] ^= (u8)(1 + bd_below_(255));
}

/* 1 when every group can give back its data: no damaged data block, or at most m damaged blocks in all. *seen: the
 * damaged blocks a load finds - every data block is checked, a group's parity only when one of its data blocks is
 * damaged (api.md: the open checks every data block; the scrub reads the parity too) */
static i32 bd_recoverable_(const BdFile *f, i64 *seen)
{
    u64 g, b;
    i32 ok = 1;
    *seen = 0;
    for (g = 0; g < f->G; ++g) {
        u32 dd = 0, dp = 0, j;
        for (b = g; b < f->N; b += f->G) dd += f->hit[b];
        for (j = 0; j < f->m; ++j) dp += f->hit[f->N + g * f->m + j];
        if (dd && dd + dp > f->m) ok = 0;
        *seen += dd + (dd ? dp : 0);
    }
    return ok;
}

/* one more damaged block in group g: a data block if one is left, else a parity block */
static void bd_more_(BdFile *f, u64 g)
{
    u64 b;
    u32 j;
    for (b = g; b < f->N; b += f->G) if (!f->hit[b]) { bd_hit_(f, b); return; }
    for (j = 0; j < f->m; ++j) if (!f->hit[f->N + g * f->m + j]) { bd_hit_(f, f->N + g * f->m + j); return; }
}

typedef struct { i32 trials, rebuilt, refused, reindexed, withheld, wrong, restored; } BdTally;

/* a scrub with repair after the load: a copy whose damage the parity covers must be the saved file again, every byte */
static void bd_scrub_(BdFile *f, i32 rec, const char *what, BdTally *t)
{
    i64 fx = 0, r = store_scrub(bd_copy, 1, &fx);
    size_t n = 0;
    u8 *back = bd_read_(bd_copy, &n);
    i32 same = back && n == (size_t)f->fsize && !memcmp(back, f->orig, n);
    char msg[1024];
    if (rec && !same) {
        sprintf(msg, "%s: the scrub (found %ld, rebuilt %ld) did not restore the file byte for byte", what, (long)r, (long)fx);
        bd_check_(0, msg);
    }
    if (same) ++t->restored;
    free(back);
}

/* load the damaged copy, compare it with the undamaged store; expect: 1 recoverable, 0 not, -1 the checksums gone */
static void bd_trial_(BdFile *f, i32 expect, const char *what, BdTally *t)
{
    RagStore *s;
    FILE *o;
    i64 nb, bad, fixed, ndam, id;
    i32 i, q, missing = 0, wrong = 0, same = 1, rec;
    f32 v[BD_DIM], sc[10];
    i64 ids[10];
    char msg[1024];
    rec = bd_recoverable_(f, &ndam);
    if (expect == 1 && !rec) { sprintf(msg, "%s: the pattern itself is beyond the parity (a bug in this test)", what); bd_check_(0, msg); }
    if (expect == -1) rec = 0;
    if (!(o = fopen(bd_copy, "wb")) || fwrite(f->buf, 1, (size_t)f->fsize, o) != (size_t)f->fsize || fclose(o)) { bd_check_(0, "write the copy"); return; }
    ++t->trials;
    s = store_load(&bd_arena, bd_copy);
    if (!s) {
        if (rec) {
            sprintf(msg, "%s: %ld damaged blocks the parity can rebuild, but the load refused the file: %s", what, (long)ndam, store_load_error());
            bd_check_(0, msg);
        }
        ++t->refused;
        arena_free(&bd_arena);
        bd_scrub_(f, rec, what, t);
        bd_clean_();
        return;
    }
    store_integrity(s, &nb, &bad, &fixed, (i64 *)0, (i64 *)0, (i32 *)0);
    for (id = 0; id < BD_ROWS; ++id) {
        if (store_get_vec(s, id, v) != 0) { ++missing; continue; }
        if (memcmp(v, bd_pv + (size_t)id * BD_DIM, sizeof(v))) ++wrong;
    }
    for (q = 0; q < BD_NQ; ++q) {
        i32 n = store_search(s, bd_q + q * BD_DIM, 10, ids, sc);
        if (n != bd_pn[q]) same = 0;
        for (i = 0; i < n && i < bd_pn[q]; ++i) if (ids[i] != bd_pid[q * 10 + i] || sc[i] != bd_ps[q * 10 + i]) same = 0;
    }
    if (wrong) {
        sprintf(msg, "%s: %d rows came back different from the ones saved", what, (int)wrong);
        bd_check_(0, msg); t->wrong += wrong;
    }
    if (expect == -1) {
        sprintf(msg, "%s: loaded although both copies of the checksums were damaged", what);
        bd_check_(0, msg);
    } else if (nb != (i64)(f->N + f->P) || bad != ndam || fixed > bad) {
        u64 b;
        u8 h[32];
        sprintf(msg, "%s: integrity says %ld blocks, %ld damaged, %ld rebuilt; the copy has %lu blocks, %ld damaged", what,
                (long)nb, (long)bad, (long)fixed, (unsigned long)(f->N + f->P), (long)ndam);
        bd_check_(0, msg);
        for (b = 0; b < f->N + f->P; ++b) {
            if (!f->hit[b]) continue;
            b3_hash(f->buf + b * f->B, (size_t)f->B, h);
            printf("       block %lu (%s, group %lu): digest %s\n", (unsigned long)b, b < f->N ? "data" : "parity",
                   (unsigned long)(b < f->N ? b % f->G : (b - f->N) / f->m), memcmp(h, f->orig + f->ta + b * 32u, 32) ? "differs" : "SAME as saved");
        }
    }
    if (rec && (missing || !same)) {
        sprintf(msg, "%s: rebuildable damage, yet %d rows missing%s", what, (int)missing, same ? "" : " and searches differ");
        bd_check_(0, msg);
    }
    if (rec) ++t->rebuilt;
    else if (missing) ++t->withheld;
    else ++t->reindexed;
    store_destroy(s);
    arena_free(&bd_arena);
    bd_scrub_(f, rec, what, t);
    bd_clean_();
}

static void bd_reset_(BdFile *f)
{
    memcpy(f->buf, f->orig, (size_t)f->fsize);
    memset(f->hit, 0, (size_t)(f->N + f->P));
}

/* one parity setting: a fresh store saved with it, then the trials */
static void bd_setting_(i32 k, i32 m, i32 trials)
{
    RagStore *s;
    BdFile f;
    BdTally t;
    size_t fsize;
    u8 *orig, *tr;
    i32 i, q;
    u64 burst;
    char msg[1024];
    memset(&t, 0, sizeof(t));
    s = store_new(&bd_arena, BD_DIM);
    store_set_parity(s, k, m);
    {   u32 r = 0x9e3779b9u ^ (u32)(k * 131 + m);
        f32 *x = (f32 *)malloc((size_t)BD_ROWS * BD_DIM * sizeof(f32));
        i64 *id = (i64 *)malloc((size_t)BD_ROWS * sizeof(i64));
        for (i = 0; i < BD_ROWS * BD_DIM; ++i) { r ^= r << 13; r ^= r >> 17; r ^= r << 5; x[i] = (f32)r / (f32)0xffffffffu - 0.5f; }
        for (i = 0; i < BD_ROWS; ++i) id[i] = i;
        store_add_batch(s, x, id, BD_ROWS);
        free(x); free(id); }
    store_build(s);
    if (store_save(s, bd_orig) != 0) { bd_check_(0, "save the store"); store_destroy(s); arena_free(&bd_arena); return; }
    store_destroy(s); arena_free(&bd_arena);
    {   char j[128]; sprintf(j, "%s.j", bd_orig); remove(j); }    /* the copies are the file alone */

    s = store_load(&bd_arena, bd_orig);                          /* the undamaged store, as it reads back */
    if (!s) { bd_check_(0, "load the undamaged store"); return; }
    for (i = 0; i < BD_ROWS; ++i) store_get_vec(s, i, bd_pv + (size_t)i * BD_DIM);
    for (q = 0; q < BD_NQ; ++q) bd_pn[q] = store_search(s, bd_q + q * BD_DIM, 10, bd_pid + q * 10, bd_ps + q * 10);
    store_destroy(s); arena_free(&bd_arena);

    orig = bd_read_(bd_orig, &fsize);
    if (!orig || fsize < SSUM_TRL) { bd_check_(0, "read the file"); return; }
    tr = orig + fsize - SSUM_TRL;                                /* trailer B: the layout as the file states it */
    memset(&f, 0, sizeof(f));
    f.B = (u64)1 << bd_get_(tr + 8, 4); f.m = (u32)bd_get_(tr + 16, 4);
    f.N = bd_get_(tr + 32, 8); f.G = bd_get_(tr + 40, 8); f.P = bd_get_(tr + 48, 8);
    f.ta = bd_get_(tr + 56, 8); f.tra = bd_get_(tr + 64, 8); f.tb = bd_get_(tr + 72, 8);
    f.T = (f.N + f.P) * 32u; f.fsize = fsize; f.orig = orig;
    {   u64 D = bd_get_(tr + 24, 8);                             /* the format's own rules, not ssum_layout_ */
        sprintf(msg, "k=%d m=%d: the trailer's layout follows the format (N = ceil(D/B) = %lu, G = ceil(N/k) = %lu, P = G m = %lu)",
                (int)k, (int)m, (unsigned long)f.N, (unsigned long)f.G, (unsigned long)f.P);
        bd_check_(bd_get_(tr, 4) == SSUM_MAGIC && (u32)bd_get_(tr + 12, 4) == (u32)k && f.m == (u32)m && f.N == (D + f.B - 1) / f.B
                  && f.G == (f.N + (u64)k - 1) / (u64)k && f.P == f.G * f.m && f.ta == (f.N + f.P) * f.B
                  && f.tb + f.T + SSUM_TRL == fsize && f.tra % 4096 == 0 && f.tb % 4096 == 0, msg); }
    f.buf = (u8 *)malloc(fsize);
    f.hit = (u8 *)calloc((size_t)(f.N + f.P), 1);
    if (!f.buf || !f.hit) { bd_check_(0, "memory for the copies"); free(orig); return; }

    burst = (u64)m * f.G;                         /* burst_within_parity: the longest run the parity covers */
    if (m > 0 && burst <= f.N) {
        bd_reset_(&f);
        {   u64 b, s0 = bd_below_((u32)(f.N - burst + 1));
            for (b = s0; b < s0 + burst; ++b) bd_hit_(&f, b); }
        sprintf(msg, "a run of m G = %lu neighbouring data blocks (%lu KiB)", (unsigned long)burst, (unsigned long)(burst * f.B >> 10));
        bd_trial_(&f, 1, msg, &t);
        bd_check_(t.rebuilt == 1, msg);
    }
    if (burst + 1 <= f.N) {
        i32 before = t.rebuilt;
        bd_reset_(&f);
        {   u64 b, s0 = bd_below_((u32)(f.N - burst));
            for (b = s0; b < s0 + burst + 1; ++b) bd_hit_(&f, b); }
        sprintf(msg, "a run of m G + 1 = %lu blocks: one group has m + 1 and loses data", (unsigned long)(burst + 1));
        bd_trial_(&f, 0, msg, &t);
        bd_check_(t.rebuilt == before, msg);
    }
    if (m > 0) {                                                 /* exactly m in one group, data and parity mixed */
        u64 g = bd_below_((u32)f.G), b;
        u32 j, nd = 0;
        bd_reset_(&f);
        for (b = g; b < f.N && nd < (u32)(m + 1) / 2; b += f.G) { bd_hit_(&f, b); ++nd; }
        for (j = 0; nd + j < (u32)m; ++j) bd_hit_(&f, f.N + g * f.m + j);
        bd_trial_(&f, 1, "exactly m damaged blocks in one group, data and parity mixed", &t);
        bd_more_(&f, g);
        bd_trial_(&f, 0, "m + 1 damaged blocks in one group", &t);
    }
    for (i = 0; i < 4; ++i) {                                    /* one copy of the checksums damaged: harmless */
        static const char *const which[] = { "table A", "trailer A", "table B", "trailer B" };
        bd_reset_(&f);
        if (i == 0) bd_smear_(&f, f.ta, f.T);
        else if (i == 1) bd_smear_(&f, f.tra, SSUM_TRL);
        else if (i == 2) bd_smear_(&f, f.tb, f.T);
        else bd_smear_(&f, fsize - SSUM_TRL, SSUM_TRL);
        sprintf(msg, "%s damaged, the other copy intact", which[i]);
        bd_trial_(&f, 1, msg, &t);
    }
    bd_reset_(&f);                                               /* both copies: nothing left to check the blocks with */
    bd_smear_(&f, f.ta, f.T); bd_smear_(&f, f.tb, f.T);
    bd_trial_(&f, -1, "both tables of checksums damaged", &t);
    for (i = 0; i < trials; ++i) {                               /* random damage, from a block or two to many */
        u64 n, b, cap = f.N + f.P;
        bd_reset_(&f);
        if (bd_below_(2)) {
            n = 1 + bd_below_((u32)(f.G * (u64)(m + 1) + 1));
            while (n--) bd_hit_(&f, bd_below_((u32)cap));
        } else {
            u64 L = 1 + bd_below_((u32)(2 * burst + 2)), s0;
            if (L > f.N) L = f.N;
            s0 = bd_below_((u32)(f.N - L + 1));
            for (b = s0; b < s0 + L; ++b) bd_hit_(&f, b);
        }
        sprintf(msg, "random trial %d", (int)i);
        bd_trial_(&f, 0, msg, &t);
    }
    sprintf(msg, "k=%d m=%d (N %lu, G %lu, P %lu blocks of %lu KiB): %d copies - %d rebuilt byte for byte, %d refused, %d loaded with "
            "the index rebuilt from intact rows, %d with rows withheld; %d files restored by a scrub; wrong rows: %d", (int)k, (int)m,
            (unsigned long)f.N, (unsigned long)f.G, (unsigned long)f.P, (unsigned long)(f.B >> 10), (int)t.trials, (int)t.rebuilt,
            (int)t.refused, (int)t.reindexed, (int)t.withheld, (int)t.restored, (int)t.wrong);
    bd_check_(t.wrong == 0, msg);
    free(f.buf); free(f.hit); free(orig);
}

int main(int argc, char **argv)
{
    i32 trials = argc > 1 ? atoi(argv[1]) : 40, i;
    char dir[] = "/tmp/battle_damage.XXXXXX";
    if (argc > 2) bd_rs = (u32)strtoul(argv[2], (char **)0, 10) | 1u;
    if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }
    strcpy(bd_dir, dir);
    sprintf(bd_orig, "%s/orig.vxi", bd_dir);
    sprintf(bd_copy, "%s/copy.vxi", bd_dir);
    for (i = 0; i < BD_NQ * BD_DIM; ++i) bd_q[i] = (f32)bd_rng_() / (f32)0xffffffffu - 0.5f;
    printf("damage on the disk: %d random copies per parity setting, seed %lu\n", (int)trials, (unsigned long)bd_rs);
    bd_setting_(32, 2, trials);                                  /* the default */
    bd_setting_(8, 1, trials);
    bd_setting_(16, 4, trials);
    bd_setting_(32, 0, trials);                                  /* digests only: damage is found, nothing rebuilt */
    {   char p[128]; sprintf(p, "%s.j", bd_orig); remove(p); remove(bd_orig); rmdir(bd_dir); }
    printf("%s\n", bd_fails ? "battle_damage: FAILED" : "battle_damage: every damaged copy rebuilt exactly or refused");
    return bd_fails ? 1 : 0;
}
