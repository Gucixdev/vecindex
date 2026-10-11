/* vsimd.h - the three f32 reductions the vecindex hot paths need, in
 * AVX2+FMA when the compiler is told the target has it (-march=native /
 * -mavx2 -mfma, i.e. __AVX2__ && __FMA__), scalar otherwise. Header-
 * only, all static; a genuine math primitive shared the same way
 * math/mathx.h is, not a cross-IMPLEMENTATION LUT copy.
 *
 * immintrin.h is compiler-vendored (ships with gcc/clang), not glibc -
 * same stance base.h takes for its own AVX_MATH block. */
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
    __m256 acc = _mm256_setzero_ps();
    f32 r; i32 d = 0;
    for (; d + 8 <= n; d += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + d), _mm256_loadu_ps(b + d), acc);
    r = vs_hsum256_(acc);
    for (; d < n; ++d) r += a[d] * b[d];
    return r;
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

#else /* scalar */

static VS_UNUSED f32 vs_sqdist(const f32 *a, const f32 *b, i32 n)
{
    f32 r = 0.0f; i32 d;
    for (d = 0; d < n; ++d) { f32 t = a[d] - b[d]; r += t * t; }
    return r;
}

static VS_UNUSED f32 vs_sub_normsq(f32 *out, const f32 *a, const f32 *b, i32 n)
{
    f32 r = 0.0f; i32 d;
    for (d = 0; d < n; ++d) { f32 t = a[d] - b[d]; out[d] = t; r += t * t; }
    return r;
}

static VS_UNUSED f32 vs_dot(const f32 *a, const f32 *b, i32 n)
{
    f32 r = 0.0f; i32 d;
    for (d = 0; d < n; ++d) r += a[d] * b[d];
    return r;
}

static VS_UNUSED f32 vs_lut_gather_dot(const f32 *lut, const u8 *row, i32 cb)
{
    f32 r = 0.0f; i32 g;
    for (g = 0; g < cb; ++g) r += lut[(size_t)g * 256 + row[g]];
    return r;
}

#endif

#endif /* HOLYC_VECINDEX_VSIMD_H_ */
