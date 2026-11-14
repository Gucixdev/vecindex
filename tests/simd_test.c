/* simd_test.c - the SIMD kernels (src/vsimd.h and the ones inside src/tq.c) against plain references, whatever the
 * build picked: AVX-512, AVX2, NEON or scalar. ./run builds it for this CPU and, on x86-64, again for an AVX2-only
 * CPU and for no SIMD at all; tools/cross runs it on ARM (NEON), RISC-V and s390x. */
#define ARENA_ALLOC_IMPLEMENTATION
#include "tq.c"
#include <stdio.h>
#include <math.h>

static u32 st_ = 12345u;
static u32 sd_rng_(void) { st_ ^= st_ << 13; st_ ^= st_ >> 17; st_ ^= st_ << 5; return st_; }
static f32 sd_f_(void) { return (f32)sd_rng_() / (f32)0xffffffffu * 2.0f - 1.0f; }

static i32 fails_ = 0;
static void sd_check_(i32 ok, const char *what, i32 n)
{
    if (!ok && fails_++ < 20) printf("  FAIL %s (n=%d)\n", what, (int)n);
}

/* |got - want| within a few ulps of the magnitudes summed */
static i32 sd_close_(f64 got, f64 want, f64 mag) { return fabs(got - want) <= 1e-5 * mag + 1e-6; }

static const char *sd_isa_(void)
{
#if defined(__AVX512BW__)
    return "AVX-512";
#elif defined(__AVX2__) && defined(__FMA__)
    return "AVX2";
#elif defined(VS_NEON) && defined(__aarch64__)
    return "NEON (aarch64)";
#elif defined(VS_NEON)
    return "NEON (32-bit ARM)";
#else
    return "scalar";
#endif
}

static void sd_reductions_(void)
{
    static f32 a[300], b[300], o[300], lut[64 * 256];
    static i8 x[300];
    static u8 row[64];
    i32 n, i, rep;
    for (rep = 0; rep < 4; ++rep)
        for (n = 0; n <= 260; n += (n < 40 ? 1 : 37)) {
            f64 dot = 0, sq = 0, di8 = 0, mag = 0, magi = 0, magq = 0;
            for (i = 0; i < n; ++i) {
                a[i] = sd_f_(); b[i] = sd_f_(); x[i] = (i8)((i32)(sd_rng_() % 255u) - 127);
                dot += (f64)a[i] * b[i]; sq += ((f64)a[i] - b[i]) * ((f64)a[i] - b[i]); di8 += (f64)a[i] * x[i];
                mag += fabs((f64)a[i] * b[i]); magq += ((f64)a[i] - b[i]) * ((f64)a[i] - b[i]); magi += fabs((f64)a[i] * x[i]);
            }
            sd_check_(sd_close_(vs_dot(a, b, n), dot, mag), "vs_dot", n);
            sd_check_(sd_close_(vs_sqdist(a, b, n), sq, magq), "vs_sqdist", n);
            sd_check_(sd_close_(vs_sub_normsq(o, a, b, n), sq, magq), "vs_sub_normsq", n);
            for (i = 0; i < n; ++i) sd_check_(o[i] == a[i] - b[i], "vs_sub_normsq out", n);
            sd_check_(sd_close_(vs_dot_i8(a, x, n), di8, magi), "vs_dot_i8", n);
            vs_i8_scale(o, x, n, 0.37f);
            for (i = 0; i < n; ++i) sd_check_(o[i] == 0.37f * (f32)x[i], "vs_i8_scale", n);
        }
    for (n = 0; n <= 64; ++n) {
        f64 want = 0, mag = 0;
        for (i = 0; i < n * 256; ++i) lut[i] = sd_f_();
        for (i = 0; i < n; ++i) { row[i] = (u8)sd_rng_(); want += lut[i * 256 + row[i]]; mag += fabs(lut[i * 256 + row[i]]); }
        sd_check_(sd_close_(vs_lut_gather_dot(lut, row, n), want, mag), "vs_lut_gather_dot", n);
    }
}

/* the fast-scan block: 32 rows, dpad (even) code columns of 16 bytes; low half-bytes are rows 0-15, high 16-31 */
static void sd_block_(void)
{
    static const i32 dps[] = { 2, 4, 6, 8, 10, 50, 64, 100, 128, 256, 300, 770 };   /* 300 and 770 wrap the u16 sums */
    static u8 blk[770 * 16], lut[770 * 16];
    u16 out[32], ref[32];
    i32 k, rep, d, j;
    for (k = 0; k < (i32)(sizeof(dps) / sizeof(dps[0])); ++k)
        for (rep = 0; rep < 8; ++rep) {
            i32 dp = dps[k];
            u16 thr = rep == 0 ? 0 : rep == 1 ? 65535 : (u16)sd_rng_();
            u32 m, mref = 0;
            for (d = 0; d < dp * 16; ++d) { blk[d] = (u8)sd_rng_(); lut[d] = rep == 2 ? 255 : (u8)sd_rng_(); }
            for (j = 0; j < 32; ++j) ref[j] = 0;
            for (d = 0; d < dp; ++d)
                for (j = 0; j < 32; ++j) {
                    u8 byte = blk[d * 16 + (j & 15)];
                    ref[j] = (u16)(ref[j] + lut[d * 16 + (j < 16 ? (byte & 15) : (byte >> 4))]);
                }
            for (j = 0; j < 32; ++j) if (ref[j] > thr) mref |= 1u << j;
            m = tq_block_(blk, lut, dp, thr, out);
            for (j = 0; j < 32; ++j) sd_check_(out[j] == ref[j], "tq_block_ sums", dp);
            sd_check_(m == mref, "tq_block_ mask", dp);
        }
}

/* centroid scoring: 16 centroids a group, transposed; padding lanes score 0 */
static void sd_score_(void)
{
    static f32 cent[70 * 130], v[130], sc[80];
    i32 nc, dim, c, d, l;
    for (nc = 1; nc <= 70; nc += 23)
        for (dim = 1; dim <= 130; dim += 43) {
            f32 *ct;
            i32 ng = (nc + 15) / 16;
            for (c = 0; c < nc * dim; ++c) cent[c] = sd_f_();
            for (d = 0; d < dim; ++d) v[d] = sd_f_();
            ct = tq_transpose_(cent, nc, dim);
            tq_score_(v, ct, ng, dim, sc);
            for (l = 0; l < ng * 16; ++l) {
                f64 want = 0, mag = 0;
                if (l < nc) for (d = 0; d < dim; ++d) { want += (f64)v[d] * cent[l * dim + d]; mag += fabs((f64)v[d] * cent[l * dim + d]); }
                sd_check_(sd_close_(sc[l], want, mag), "tq_score_", nc * 1000 + dim);
            }
            free(ct);
        }
}

/* Walsh-Hadamard on each power-of-two chunk of x[dpad], largest first, orthonormal */
static void sd_wht_(void)
{
    static f32 x[400], y[400];
    static f64 ref[400];
    i32 dp, i, j, base, len;
    for (dp = 1; dp <= 400; dp += (dp < 70 ? 1 : 33)) {
        for (i = 0; i < dp; ++i) x[i] = y[i] = sd_f_();
        for (len = 1; len * 2 <= dp; len *= 2) {}
        for (base = 0; base < dp; base += len) {
            while (len > dp - base) len /= 2;
            for (i = 0; i < len; ++i) {
                f64 s = 0;
                for (j = 0; j < len; ++j) {
                    u32 p = (u32)(i & j), par = 0;
                    while (p) { par ^= p & 1u; p >>= 1; }
                    s += par ? -(f64)x[base + j] : (f64)x[base + j];
                }
                ref[base + i] = s / sqrt((f64)len);
            }
        }
        tq_wht_(y, dp);
        for (i = 0; i < dp; ++i) sd_check_(fabs(y[i] - ref[i]) <= 1e-5 * (1.0 + fabs(ref[i])), "tq_wht_", dp);
        tq_wht_(y, dp);                                                   /* orthonormal and symmetric: twice = identity */
        for (i = 0; i < dp; ++i) sd_check_(fabs(y[i] - x[i]) <= 1e-5, "tq_wht_ twice", dp);
    }
}

int main(void)
{
    (void)&tq_ram_; (void)&tq_prepare_;                     /* store.c's helpers, unused here */
    printf("simd: %s kernels\n", sd_isa_());
    sd_reductions_();
    sd_block_();
    sd_score_();
    sd_wht_();
    if (fails_) { printf("simd: %d checks FAILED\n", (int)fails_); return 1; }
    printf("simd: ok (dot, sqdist, sub_normsq, int8 dot and scale, lut gather, fast-scan block, centroid scores, WHT)\n");
    return 0;
}
