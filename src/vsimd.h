/* vsimd.h - the f32 and int8 reductions the vecindex hot paths need, in
 * AVX2+FMA (AVX-512 where it pays) when the compiler is told the target
 * has it (-march=native / -mavx2 -mfma), NEON on ARM (aarch64, and 32-bit
 * ARM built with -mfpu=neon...), scalar with independent partial sums
 * everywhere else. Header-only, all static.
 *
 * immintrin.h and arm_neon.h are compiler-vendored (ship with gcc/clang),
 * not glibc - same stance base.h takes for its own AVX_MATH block. */
#ifndef HOLYC_VECINDEX_VSIMD_H_
#define HOLYC_VECINDEX_VSIMD_H_

#include "base.h"

#ifdef __GNUC__
#define VS_UNUSED __attribute__((unused))
#else
#define VS_UNUSED
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>

static VS_UNUSED f32 vs_hsum256_(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

/* sum over d of (a[d] - b[d])^2 */
static VS_UNUSED f32 vs_sqdist(const f32 *a, const f32 *b, i32 n)
{
    __m256 acc = _mm256_setzero_ps();
    f32 r; i32 d = 0;
    for (; d + 8 <= n; d += 8) {
        __m256 t = _mm256_sub_ps(_mm256_loadu_ps(a + d), _mm256_loadu_ps(b + d));
        acc = _mm256_fmadd_ps(t, t, acc);
    }
    r = vs_hsum256_(acc);
    for (; d < n; ++d) { f32 t = a[d] - b[d]; r += t * t; }
    return r;
}

/* out[d] = a[d] - b[d]; returns sum over d of out[d]^2 */
static VS_UNUSED f32 vs_sub_normsq(f32 *out, const f32 *a, const f32 *b, i32 n)
{
    __m256 acc = _mm256_setzero_ps();
    f32 r; i32 d = 0;
    for (; d + 8 <= n; d += 8) {
        __m256 t = _mm256_sub_ps(_mm256_loadu_ps(a + d), _mm256_loadu_ps(b + d));
        _mm256_storeu_ps(out + d, t);
        acc = _mm256_fmadd_ps(t, t, acc);
    }
    r = vs_hsum256_(acc);
    for (; d < n; ++d) { f32 t = a[d] - b[d]; out[d] = t; r += t * t; }
    return r;
}

/* sum over d of a[d] * b[d] */
static VS_UNUSED f32 vs_dot(const f32 *a, const f32 *b, i32 n)
{
#if defined(__AVX512F__)
    /* two independent accumulators hide the FMA latency; the tail is one masked load */
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    i32 d = 0;
    for (; d + 32 <= n; d += 32) {
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + d), _mm512_loadu_ps(b + d), s0);
        s1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + d + 16), _mm512_loadu_ps(b + d + 16), s1);
    }
    for (; d + 16 <= n; d += 16)
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + d), _mm512_loadu_ps(b + d), s0);
    if (d < n) {
        __mmask16 m = (__mmask16)((1u << (n - d)) - 1u);
        s1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, a + d), _mm512_maskz_loadu_ps(m, b + d), s1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(s0, s1));
#else
    __m256 acc = _mm256_setzero_ps(), acc2 = _mm256_setzero_ps();
    f32 r; i32 d = 0;
    for (; d + 16 <= n; d += 16) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + d), _mm256_loadu_ps(b + d), acc);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + d + 8), _mm256_loadu_ps(b + d + 8), acc2);
    }
    for (; d + 8 <= n; d += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + d), _mm256_loadu_ps(b + d), acc);
    r = vs_hsum256_(_mm256_add_ps(acc, acc2));
    for (; d < n; ++d) r += a[d] * b[d];
    return r;
#endif
}

/* sum over g in [0, cb) of lut[g*256 + row[g]] - the per-code LUT
 * gather-scan the qN searches spend most of their time in. 8-wide
 * vector gather; base [0,256,..,1792] + the zero-extended code bytes. */
static VS_UNUSED f32 vs_lut_gather_dot(const f32 *lut, const u8 *row, i32 cb)
{
    __m256i base = _mm256_setr_epi32(0, 256, 512, 768, 1024, 1280, 1536, 1792);
    __m256 acc = _mm256_setzero_ps();
    f32 r; i32 g = 0;
    for (; g + 8 <= cb; g += 8) {
        __m128i b8 = _mm_loadl_epi64((const __m128i *)(const void *)(row + g));
        __m256i idx = _mm256_add_epi32(_mm256_cvtepu8_epi32(b8), base);
        acc = _mm256_add_ps(acc,
            _mm256_i32gather_ps(lut + (size_t)g * 256, idx, 4));
    }
    r = vs_hsum256_(acc);
    for (; g < cb; ++g) r += lut[(size_t)g * 256 + row[g]];
    return r;
}


/* sum over d of q[d] * x[d] for int8 x (the caller applies the row's scale) */
static VS_UNUSED f32 vs_dot_i8(const f32 *q, const i8 *x, i32 n)
{
    f32 r; i32 d = 0;
#if defined(__AVX512F__)
    __m512 a = _mm512_setzero_ps();
    for (; d + 16 <= n; d += 16)
        a = _mm512_fmadd_ps(_mm512_loadu_ps(q + d), _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(x + d)))), a);
    r = _mm512_reduce_add_ps(a);
#else
    __m256 a = _mm256_setzero_ps(), b = a;
    for (; d + 16 <= n; d += 16) {
        a = _mm256_fmadd_ps(_mm256_loadu_ps(q + d), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)(x + d)))), a);
        b = _mm256_fmadd_ps(_mm256_loadu_ps(q + d + 8), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)(x + d + 8)))), b);
    }
    for (; d + 8 <= n; d += 8)
        a = _mm256_fmadd_ps(_mm256_loadu_ps(q + d), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)(x + d)))), a);
    r = vs_hsum256_(_mm256_add_ps(a, b));
#endif
    for (; d < n; ++d) r += q[d] * (f32)x[d];
    return r;
}

/* out[d] = sc * x[d] */
static VS_UNUSED void vs_i8_scale(f32 *out, const i8 *x, i32 n, f32 sc)
{
    i32 d = 0;
#if defined(__AVX512F__)
    for (; d + 16 <= n; d += 16)
        _mm512_storeu_ps(out + d, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(x + d)))), _mm512_set1_ps(sc)));
#else
    for (; d + 8 <= n; d += 8)
        _mm256_storeu_ps(out + d, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)(x + d)))), _mm256_set1_ps(sc)));
#endif
    for (; d < n; ++d) out[d] = sc * (f32)x[d];
}

#elif (defined(__ARM_NEON) || defined(__ARM_NEON__)) && !defined(__ARM_BIG_ENDIAN)
#include <arm_neon.h>
#define VS_NEON 1

#if defined(__ARM_FEATURE_FMA)
#define VS_FMAQ(acc, a, b) vfmaq_f32(acc, a, b)
#else
#define VS_FMAQ(acc, a, b) vmlaq_f32(acc, a, b)
#endif

static VS_UNUSED f32 vs_hsumq_(float32x4_t v)
{
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(s, s), 0);
#endif
}

/* four int8 lanes of x (sign-extended) as floats: lanes 0-3, 4-7, 8-11, 12-15 of a 16-byte load */
static VS_UNUSED void vs_i8x16_(const i8 *x, float32x4_t *f)
{
    int8x16_t b = vld1q_s8(x);
    int16x8_t lo = vmovl_s8(vget_low_s8(b)), hi = vmovl_s8(vget_high_s8(b));
    f[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))); f[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
    f[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))); f[3] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
}

static VS_UNUSED f32 vs_sqdist(const f32 *a, const f32 *b, i32 n)
{
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0;
    f32 r; i32 d = 0;
    for (; d + 8 <= n; d += 8) {
        float32x4_t t0 = vsubq_f32(vld1q_f32(a + d), vld1q_f32(b + d));
        float32x4_t t1 = vsubq_f32(vld1q_f32(a + d + 4), vld1q_f32(b + d + 4));
        s0 = VS_FMAQ(s0, t0, t0); s1 = VS_FMAQ(s1, t1, t1);
    }
    for (; d + 4 <= n; d += 4) { float32x4_t t = vsubq_f32(vld1q_f32(a + d), vld1q_f32(b + d)); s0 = VS_FMAQ(s0, t, t); }
    r = vs_hsumq_(vaddq_f32(s0, s1));
    for (; d < n; ++d) { f32 t = a[d] - b[d]; r += t * t; }
    return r;
}

static VS_UNUSED f32 vs_sub_normsq(f32 *out, const f32 *a, const f32 *b, i32 n)
{
    float32x4_t s0 = vdupq_n_f32(0.0f);
    f32 r; i32 d = 0;
    for (; d + 4 <= n; d += 4) {
        float32x4_t t = vsubq_f32(vld1q_f32(a + d), vld1q_f32(b + d));
        vst1q_f32(out + d, t);
        s0 = VS_FMAQ(s0, t, t);
    }
    r = vs_hsumq_(s0);
    for (; d < n; ++d) { f32 t = a[d] - b[d]; out[d] = t; r += t * t; }
    return r;
}

static VS_UNUSED f32 vs_dot(const f32 *a, const f32 *b, i32 n)
{
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;     /* four chains hide the FMA latency */
    f32 r; i32 d = 0;
    for (; d + 16 <= n; d += 16) {
        s0 = VS_FMAQ(s0, vld1q_f32(a + d), vld1q_f32(b + d));
        s1 = VS_FMAQ(s1, vld1q_f32(a + d + 4), vld1q_f32(b + d + 4));
        s2 = VS_FMAQ(s2, vld1q_f32(a + d + 8), vld1q_f32(b + d + 8));
        s3 = VS_FMAQ(s3, vld1q_f32(a + d + 12), vld1q_f32(b + d + 12));
    }
    for (; d + 4 <= n; d += 4) s0 = VS_FMAQ(s0, vld1q_f32(a + d), vld1q_f32(b + d));
    r = vs_hsumq_(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; d < n; ++d) r += a[d] * b[d];
    return r;
}

static VS_UNUSED f32 vs_dot_i8(const f32 *q, const i8 *x, i32 n)
{
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0, f[4];
    f32 r; i32 d = 0;
    for (; d + 16 <= n; d += 16) {
        vs_i8x16_(x + d, f);
        s0 = VS_FMAQ(s0, vld1q_f32(q + d), f[0]);      s1 = VS_FMAQ(s1, vld1q_f32(q + d + 4), f[1]);
        s2 = VS_FMAQ(s2, vld1q_f32(q + d + 8), f[2]);  s3 = VS_FMAQ(s3, vld1q_f32(q + d + 12), f[3]);
    }
    r = vs_hsumq_(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; d < n; ++d) r += q[d] * (f32)x[d];
    return r;
}

static VS_UNUSED void vs_i8_scale(f32 *out, const i8 *x, i32 n, f32 sc)
{
    float32x4_t s = vdupq_n_f32(sc), f[4];
    i32 d = 0;
    for (; d + 16 <= n; d += 16) {
        vs_i8x16_(x + d, f);
        vst1q_f32(out + d, vmulq_f32(f[0], s));      vst1q_f32(out + d + 4, vmulq_f32(f[1], s));
        vst1q_f32(out + d + 8, vmulq_f32(f[2], s));  vst1q_f32(out + d + 12, vmulq_f32(f[3], s));
    }
    for (; d < n; ++d) out[d] = sc * (f32)x[d];
}

/* no gather on NEON: four independent sums */
static VS_UNUSED f32 vs_lut_gather_dot(const f32 *lut, const u8 *row, i32 cb)
{
    f32 r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f; i32 g = 0;
    for (; g + 4 <= cb; g += 4) {
        r0 += lut[(size_t)g * 256 + row[g]];           r1 += lut[(size_t)(g + 1) * 256 + row[g + 1]];
        r2 += lut[(size_t)(g + 2) * 256 + row[g + 2]]; r3 += lut[(size_t)(g + 3) * 256 + row[g + 3]];
    }
    for (; g < cb; ++g) r0 += lut[(size_t)g * 256 + row[g]];
    return (r0 + r1) + (r2 + r3);
}
#else /* scalar: independent partial sums, so the adds overlap (and a compiler may vectorise them) */

static VS_UNUSED f32 vs_sqdist(const f32 *a, const f32 *b, i32 n)
{
    f32 r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f; i32 d = 0;
    for (; d + 4 <= n; d += 4) {
        f32 t0 = a[d] - b[d], t1 = a[d + 1] - b[d + 1], t2 = a[d + 2] - b[d + 2], t3 = a[d + 3] - b[d + 3];
        r0 += t0 * t0; r1 += t1 * t1; r2 += t2 * t2; r3 += t3 * t3;
    }
    for (; d < n; ++d) { f32 t = a[d] - b[d]; r0 += t * t; }
    return (r0 + r1) + (r2 + r3);
}

static VS_UNUSED f32 vs_sub_normsq(f32 *out, const f32 *a, const f32 *b, i32 n)
{
    f32 r0 = 0.0f, r1 = 0.0f; i32 d = 0;
    for (; d + 2 <= n; d += 2) {
        f32 t0 = a[d] - b[d], t1 = a[d + 1] - b[d + 1];
        out[d] = t0; out[d + 1] = t1; r0 += t0 * t0; r1 += t1 * t1;
    }
    for (; d < n; ++d) { f32 t = a[d] - b[d]; out[d] = t; r0 += t * t; }
    return r0 + r1;
}

static VS_UNUSED f32 vs_dot(const f32 *a, const f32 *b, i32 n)
{
    f32 r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f; i32 d = 0;
    for (; d + 4 <= n; d += 4) { r0 += a[d] * b[d]; r1 += a[d + 1] * b[d + 1]; r2 += a[d + 2] * b[d + 2]; r3 += a[d + 3] * b[d + 3]; }
    for (; d < n; ++d) r0 += a[d] * b[d];
    return (r0 + r1) + (r2 + r3);
}

static VS_UNUSED f32 vs_dot_i8(const f32 *q, const i8 *x, i32 n)
{
    f32 r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f; i32 d = 0;
    for (; d + 4 <= n; d += 4) {
        r0 += q[d] * (f32)x[d]; r1 += q[d + 1] * (f32)x[d + 1]; r2 += q[d + 2] * (f32)x[d + 2]; r3 += q[d + 3] * (f32)x[d + 3];
    }
    for (; d < n; ++d) r0 += q[d] * (f32)x[d];
    return (r0 + r1) + (r2 + r3);
}

static VS_UNUSED void vs_i8_scale(f32 *out, const i8 *x, i32 n, f32 sc)
{
    i32 d;
    for (d = 0; d < n; ++d) out[d] = sc * (f32)x[d];
}

static VS_UNUSED f32 vs_lut_gather_dot(const f32 *lut, const u8 *row, i32 cb)
{
    f32 r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f; i32 g = 0;
    for (; g + 4 <= cb; g += 4) {
        r0 += lut[(size_t)g * 256 + row[g]];           r1 += lut[(size_t)(g + 1) * 256 + row[g + 1]];
        r2 += lut[(size_t)(g + 2) * 256 + row[g + 2]]; r3 += lut[(size_t)(g + 3) * 256 + row[g + 3]];
    }
    for (; g < cb; ++g) r0 += lut[(size_t)g * 256 + row[g]];
    return (r0 + r1) + (r2 + r3);
}

#endif

#endif /* HOLYC_VECINDEX_VSIMD_H_ */
