/* fuzz_cfg.c - libFuzzer target for what vserv reads from files and peers besides the protocols: the key file of -K
 * (vs_keys_load_), a replica's journal at start (vs_rjtrim_) and a certificate's DER as a replica reads the primary's
 * (vs_spki_), under ASan and UBSan. The first byte picks which; the rest is the input.
 *
 * Checked beyond memory errors: a key file that loads holds only keys of 16 to 128 printable characters, each once,
 * with a tenant number 1 to 524287 (or the operator's *); the journal walk keeps at most the file, and walking what it
 * kept keeps all of it (Journal.lean: what the walk keeps is whole frames); a public key found in a certificate lies
 * inside it and is a SEQUENCE.
 *
 *   clang -std=gnu99 -g -O1 -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined -DVSERV_NO_TLS \
 *       -Isrc tests/fuzz_cfg.c -lm -lpthread -o .build/fuzz_cfg
 *   .build/fuzz_cfg -max_len=4096 CORPUS_DIR tests/fuzz/cfg
 *
 * tools/fuzz builds and runs every target; ./run replays tests/fuzz/cfg with gcc alone (-DFUZZ_MAIN, fuzz_main.h). */
#define VSERV_FUZZ 1
#define main vserv_main_
#include "vserv.c"
#undef main

static char fz_dir[64], fz_kf[96], fz_vx[96], fz_jn[100];

int LLVMFuzzerInitialize(int *argc, char ***argv)
{
    (void)argc; (void)argv;
    strcpy(fz_dir, "/tmp/fuzz_cfg-XXXXXX");
    if (!mkdtemp(fz_dir)) abort();
    sprintf(fz_kf, "%s/k.tsv", fz_dir); sprintf(fz_vx, "%s/r.vxi", fz_dir); sprintf(fz_jn, "%s/r.vxi.j", fz_dir);
    if (!getenv("FUZZ_CFG_STDERR") && !freopen("/dev/null", "w", stderr)) {}   /* the loader's refusals */
    return 0;
}

static void fz_write_(const char *path, const unsigned char *d, size_t n)
{
    FILE *f = fopen(path, "wb");
    if (!f) abort();
    if (n && fwrite(d, 1, n, f) != n) abort();
    fclose(f);
}

static void fz_keys_(const unsigned char *d, size_t n)
{
    i32 i, j;
    fz_write_(fz_kf, d, n);
    if (vs_kst) for (i = 0; i < vs_nkeys; ++i) pthread_mutex_destroy(&vs_kst[i].mu);   /* after a refused file: none */
    free(vs_keys); free(vs_kst); vs_keys = (VsKey *)0; vs_kst = (VsKeySt *)0; vs_nkeys = 0;
    if (vs_keys_load_(fz_kf) != 0) return;
    for (i = 0; i < vs_nkeys; ++i) {
        const VsKey *k = &vs_keys[i];
        if (k->klen < VS_KEYMIN || k->klen > VS_KEYMAX) abort();
        for (j = 0; j < k->klen; ++j) if ((unsigned char)k->key[j] < 0x21 || (unsigned char)k->key[j] > 0x7e) abort();
        if (!k->admin && (k->tn < 1 || k->tn > VS_TMAXN)) abort();
        if (k->rows < 0 || k->qps < 0) abort();
        for (j = 0; j < i; ++j) if (vs_keys[j].klen == k->klen && !memcmp(vs_keys[j].key, k->key, (size_t)k->klen)) abort();
    }
}

static void fz_journal_(const unsigned char *d, size_t n)
{
    u8 h[32];
    u64 st = 0, g, g2;
    struct stat sb;
    memset(h, 0, sizeof(h));
    if (n >= 16) memcpy(&st, d + 8, 8);                          /* the file beside it: the stamp the header names */
    if (n && (d[0] & 1)) st ^= 1;                                /* ... or another one */
    vs_put32_((char *)h, STORE_MAGIC); vs_put32_((char *)h + 4, (u32)STORE_VERSION9);
    vs_put32_((char *)h + 24, (u32)(st & 0xffffffffu)); vs_put32_((char *)h + 28, (u32)(st >> 32));
    fz_write_(fz_vx, h, 32);
    fz_write_(fz_jn, d, n);
    g = vs_rjtrim_(fz_vx, fz_jn);
    if (g > n) abort();
    if (g && (stat(fz_jn, &sb) != 0 || (u64)sb.st_size != g)) abort();   /* cut back to the good part */
    g2 = vs_rjtrim_(fz_vx, fz_jn);
    if (g && g2 != g) abort();                                   /* what it kept is whole */
}

static void fz_der_(const unsigned char *d, size_t n)
{
    const unsigned char *k;
    size_t kn;
    if (vs_spki_(d, n, &k, &kn) != 0) return;
    if (k < d || kn < 2 || (size_t)(k - d) + kn > n || k[0] != 0x30) abort();
}

int LLVMFuzzerTestOneInput(const unsigned char *d, size_t n)
{
    if (!n) return 0;
    switch (d[0] % 3) {
    case 0: fz_keys_(d + 1, n - 1); break;
    case 1: fz_journal_(d + 1, n - 1); break;
    default: fz_der_(d + 1, n - 1); break;
    }
    return 0;
}

#include "fuzz_main.h"
