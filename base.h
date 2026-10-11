#ifndef HOLYC_BASE_H_
#define HOLYC_BASE_H_

/* Single-file bootstrap: base types (formerly base_type.h) + bit tricks
 * (formerly bit.h) + SIMD (formerly avx_math.h/sse_math.h), with their
 * REAL bodies pasted in directly below - not pulled in via #include.
 * The original split files are not deleted, just retired to
 * src/unused/ (kept around in case a future need for the smaller,
 * individually-swappable pieces comes back) - this file is now the one
 * every consumer actually includes for base types/bit/SIMD.
 *
 * vec2/vec3/vec4/Quat/Mat4 (`src/math/mathx.h`), print (`src/core/
 * print.h`), and log (`src/core/log.h`) used to live here too - all
 * three moved out to their own files, own scope, per explicit user
 * direction: base.h stays scoped to base types/bit/SIMD only, a
 * consumer needing math types/print/log includes those files
 * separately (they each #include "base.h" themselves for the types
 * they need, so a consumer never has to include base.h AND one of
 * those directly for the shared types to resolve correctly).
 *
 * Implementation guard macros keep their ORIGINAL per-component names
 * (BASE_TYPE_IMPLEMENTATION, BIT_IMPLEMENTATION, AVX_MATH_IMPLEMENTATION,
 * SSE_MATH_IMPLEMENTATION) - a consumer that only ever needed base types
 * still only pays for base types, not SIMD being compiled in too.
 *
 * ZERO glibc: no #include of any kind, system or project-internal.
 * size_t comes from the compiler builtin __SIZE_TYPE__ (same trick
 * base_type.h already used for i8/u8/etc via __INTn_TYPE__); NULL/
 * offsetof are the standard's own textbook definitions, not glibc
 * calls. `immintrin.h` further below is a COMPILER-vendored intrinsics
 * header (ships with gcc/clang itself), not glibc, so it doesn't count
 * against this and is kept as-is. */

/* ============================================================
 * base types (base_type.h)
 * o void   u unsigned   i int   f float   fp fixed-point   a array
 * sizes: 8 16 32 64
 * ============================================================ */

/* TODO: pelna implementacja xmacro tu w tym pliku koniecznie i operacje na
 * stringach tez tu (str_stream, arena_str, etc) - podstawowe operacje
 * stringowe. */

#if !defined(__INT8_TYPE__) || !defined(__INT64_TYPE__) || !defined(__SIZE_TYPE__)
#error "compiler does not expose __INTn_TYPE__/__SIZE_TYPE__ builtins, base.h needs a different backend on this toolchain"
#endif

typedef __SIZE_TYPE__ size_t;

/* NULL/offsetof are stddef.h's other two "not really libc, just
 * compiler-level tricks" facilities - same reasoning as size_t above,
 * defined here rather than pulled in via #include. offsetof's
 * ((type*)0)->member trick is the standard's own textbook
 * implementation, not a shortcut version of it. */
#ifndef NULL
#define NULL ((void *)0)
#endif
#define offsetof(type, member) ((size_t)&((type *)0)->member)

/* alignof(type) (C11 <stdalign.h>, no libc dependency needed): the
 * classic struct-offset trick built directly on offsetof above - a
 * struct with a single leading `char` gets padded before its second
 * member exactly up to that member's own natural alignment, so that
 * padding amount (== the second member's offset) IS alignof(type).
 * This exact technique already lived privately as arena_alloc.h's own
 * ARENA_ALIGNOF; this is the same trick promoted to a public, base.h-
 * level facility so every consumer gets it without needing arena_alloc.h. */
#define HOLYC_ALIGNOF(type) offsetof(struct { char c_; type d_; }, d_)

/* alignas(n) (C11 <stdalign.h>): no portable C89 way to request a
 * specific alignment - GCC/Clang's own `__attribute__((aligned))`
 * extension is the real, standard-in-practice way every C89 codebase
 * targeting these compilers actually does this (same "GCC extension
 * where the standard has nothing" stance mem/raii.h's cleanup
 * attribute and this project's own SIMD intrinsics below already
 * take). Place BEFORE the declaration, same position C11's real
 * `alignas` keyword uses: `HOLYC_ALIGNAS(16) u8 buf[64];` */
#define HOLYC_ALIGNAS(n) __attribute__((aligned(n)))

/* static_assert(cond, msg) (C11 <assert.h> / C23 keyword): the
 * classic C89 emulation - a typedef to an array of size [1] on
 * success or [-1] (a compile error, negative array size) on failure.
 * Needs a unique type name per use site (__LINE__-based, same
 * two-level token-paste trick help/async.h's own DEFER macro uses)
 * since two static_asserts on different lines would otherwise redeclare
 * the same typedef name. `msg` is NOT a string here (C89 has no way to
 * surface an arbitrary message text into a compile error without a
 * real _Static_assert) - it must be a bare identifier fragment
 * (e.g. `size_must_be_4`, not `"size must be 4"`), token-pasted
 * directly into the generated type's own name so it's still visible
 * in the compiler's error output (the "negative array size" message
 * will mention `holyc_static_assert_..._size_must_be_4_42`, line
 * number included) - a real, if cruder, substitute for a proper
 * message string. */
#define HOLYC_STATIC_ASSERT_CONCAT_(a, b) a##b
#define HOLYC_STATIC_ASSERT_CONCAT(a, b) HOLYC_STATIC_ASSERT_CONCAT_(a, b)
#define HOLYC_STATIC_ASSERT(cond, msg) \
    typedef char HOLYC_STATIC_ASSERT_CONCAT(HOLYC_STATIC_ASSERT_CONCAT(holyc_static_assert_##msg##_, __LINE__), _)[(cond) ? 1 : -1]

typedef void o;

typedef __INT8_TYPE__  i8;
typedef __INT16_TYPE__ i16;
typedef __INT32_TYPE__ i32;
typedef __INT64_TYPE__ i64;

typedef __UINT8_TYPE__  u8;
typedef __UINT16_TYPE__ u16;
typedef __UINT32_TYPE__ u32;
typedef __UINT64_TYPE__ u64;

typedef float  f32;
typedef double f64;

/* fp: Q-format fixed point. add/sub are just + and - on the underlying int,
 * no wrapper needed for those - that is the whole point of fixed point. */
typedef i16 fp16; /* Q8.8   */
typedef i32 fp32; /* Q16.16 */
typedef i64 fp64; /* Q32.32 */

#define FP16_FRAC_BITS 8
#define FP32_FRAC_BITS 16
#define FP64_FRAC_BITS 32

fp16 fp16_from_f32(f32 v);
f32  fp16_to_f32(fp16 v);
fp16 fp16_mul(fp16 a, fp16 b);
fp16 fp16_div(fp16 a, fp16 b);

fp32 fp32_from_f32(f32 v);
f32  fp32_to_f32(fp32 v);
fp32 fp32_mul(fp32 a, fp32 b);
fp32 fp32_div(fp32 a, fp32 b);

fp64 fp64_from_f64(f64 v);
f64  fp64_to_f64(fp64 v);
/* fp64_mul/fp64_div need a 64x64->128 bit widen that plain C89 does not have
 * (no __int128 without a compiler extension) - not implemented yet. */

/* a: bounds-carrying, non-owning view over existing memory (pointer + count). */
typedef struct { u8  *data; size_t count; } a8;
typedef struct { u16 *data; size_t count; } a16;
typedef struct { u32 *data; size_t count; } a32;
typedef struct { u64 *data; size_t count; } a64;

a8  a8_make(u8 *data, size_t count);
a16 a16_make(u16 *data, size_t count);
a32 a32_make(u32 *data, size_t count);
a64 a64_make(u64 *data, size_t count);

/* FNV-1a over a byte buffer of known length. */
u32 fnv1a_hash(const void *data, size_t len, u32 seed);

/* memset_explicit (C23 <string.h>): unlike a plain memset, a compiler
 * is explicitly FORBIDDEN from optimizing this write away even if it
 * can prove the memory is never read afterward - the real problem
 * plain memset has for zeroing secrets (a password, a key) right
 * before freeing/returning: dead-store elimination sees the write as
 * pointless and deletes it, leaving the secret sitting in memory. The
 * `volatile` pointer here is what defeats that optimization - the
 * standard, portable-in-practice technique (the same one glibc's own
 * explicit_bzero and OpenBSD's explicit_bzero use under the hood). */
void *memset_explicit(void *dst, i32 val, size_t n);

/* ============================================================
 * bit (bit.h)
 * ============================================================ */

#define BIT(n)              ((u64)1 << (n))
#define FLAG_SET(x, m)      ((x) |= (m))
#define FLAG_CLEAR(x, m)    ((x) &= ~(m))
#define FLAG_TOGGLE(x, m)   ((x) ^= (m))
#define FLAG_HAS(x, m)      (((x) & (m)) == (m))

u32 bit_extract32(u32 x, i32 pos, i32 width);
u32 bit_insert32(u32 x, u32 value, i32 pos, i32 width); /* replace */
u64 bit_extract64(u64 x, i32 pos, i32 width);
u64 bit_insert64(u64 x, u64 value, i32 pos, i32 width);

/* software PDEP/PEXT (BMI2's actual instructions, portable). */
u32 bit_pdep32(u32 src, u32 mask);
u32 bit_pext32(u32 src, u32 mask);

/* bitboard: lowest-set-bit tricks */
u64 bit_isolate_lsb64(u64 x);  /* the lowest set bit alone, 0 if x == 0 */
u64 bit_clear_lsb64(u64 x);    /* x with its lowest set bit cleared */
i32 bit_lsb_index64(u64 x);    /* index of the lowest set bit, -1 if x == 0 */

u8  bit_reverse8(u8 x);
u32 bit_reverse32(u32 x);

/* align must be a power of 2 - not checked. */
size_t align_up(size_t x, size_t align);
size_t align_down(size_t x, size_t align);
i32 is_pow2(size_t x); /* 0 counts as NOT a power of 2 */

/* branchless select/min/max/abs/clamp on plain integers */
i32 bit_select_i32(i32 cond, i32 a, i32 b); /* cond must be 0 or 1: a if 1, b if 0 */
i32 min_i32(i32 a, i32 b);
i32 max_i32(i32 a, i32 b);
i32 clamp_i32(i32 x, i32 lo, i32 hi);
i32 abs_i32(i32 x);

u16 byteswap16(u16 x);
u32 byteswap32(u32 x);
u64 byteswap64(u64 x);
i32 is_little_endian(void); /* runtime check - portable */

/* UB on zero input for clz/ctz, matching the underlying compiler builtins. */
i32 popcount32(u32 x);
i32 popcount64(u64 x);
i32 clz32(u32 x); /* undefined for x == 0 */
i32 ctz32(u32 x); /* undefined for x == 0 */

u32 next_pow2_32(u32 x); /* x itself if already a power of 2; 1 for x == 0 */

i32 sign_extend32(u32 x, i32 bits); /* bits in 1..32 */

/* Trap-based assert, replacing assert.h's - __builtin_trap() is a
 * compiler intrinsic (illegal-instruction trap), not glibc, so this
 * stays inside the "0 glibc" rule. No printed file/line/expression
 * message the way glibc's assert() gives you: this project's own
 * print.h needs PRINT_IMPLEMENTATION opted into separately, which
 * would make this macro's behavior depend on whether the includer
 * happened to also want print - simpler and more predictable to just
 * trap (a debugger or core dump still tells you exactly where).
 * Compiles out under NDEBUG, matching assert()'s own convention. The
 * various *_ASSERT macros across the codebase (ARENA_ASSERT,
 * POOL_ASSERT, REF_COUNT_ASSERT, etc.) default to this instead of
 * glibc's assert now. */
#ifdef NDEBUG
#define HOLYC_ASSERT(cond) ((void)0)
#else
#define HOLYC_ASSERT(cond) ((cond) ? (void)0 : __builtin_trap())
#endif

/* ============================================================
 * SIMD (avx_math.h / sse_math.h) - x86/x86_64 only, no-op elsewhere.
 * Call *_math_available() before using the rest - none of these fall
 * back to scalar internally, that's the caller's job.
 * ============================================================ */

#if defined(__x86_64__) || defined(__i386__)

i32 avx_math_available(void);
f32 avx_sum8(const f32 *data);
f32 avx_dot8(const f32 *a, const f32 *b);
f32 avx_min8(const f32 *data);
f32 avx_max8(const f32 *data);

i32 sse_math_available(void);
f32 sse_sum4(const f32 *data);
f32 sse_dot4(const f32 *a, const f32 *b);
f32 sse_min4(const f32 *data);
f32 sse_max4(const f32 *data);

#endif /* x86 */

#endif /* HOLYC_BASE_H_ */

/* Implementation guards below are kept as the SAME per-component macro
 * names the original split files used (BASE_TYPE_IMPLEMENTATION,
 * BIT_IMPLEMENTATION, AVX_MATH_IMPLEMENTATION, SSE_MATH_IMPLEMENTATION)
 * rather than one combined macro - a consumer that only needs base
 * types still only pays for base types, not SIMD being compiled in too. */

#ifdef BASE_TYPE_IMPLEMENTATION
#ifndef BASE_TYPE_IMPLEMENTATION_
#define BASE_TYPE_IMPLEMENTATION_

fp16 fp16_from_f32(f32 v) { return (fp16)(v * (f32)(1 << FP16_FRAC_BITS)); }
f32  fp16_to_f32(fp16 v)  { return (f32)v / (f32)(1 << FP16_FRAC_BITS); }
fp16 fp16_mul(fp16 a, fp16 b) { return (fp16)(((i32)a * (i32)b) >> FP16_FRAC_BITS); }
fp16 fp16_div(fp16 a, fp16 b) { return (fp16)(((i32)a << FP16_FRAC_BITS) / b); }

fp32 fp32_from_f32(f32 v) { return (fp32)(v * (f32)(1 << FP32_FRAC_BITS)); }
f32  fp32_to_f32(fp32 v)  { return (f32)v / (f32)(1 << FP32_FRAC_BITS); }
fp32 fp32_mul(fp32 a, fp32 b) { return (fp32)(((i64)a * (i64)b) >> FP32_FRAC_BITS); }
fp32 fp32_div(fp32 a, fp32 b) { return (fp32)(((i64)a << FP32_FRAC_BITS) / b); }

fp64 fp64_from_f64(f64 v) { return (fp64)(v * (f64)((i64)1 << FP64_FRAC_BITS)); }
f64  fp64_to_f64(fp64 v)  { return (f64)v / (f64)((i64)1 << FP64_FRAC_BITS); }

u32 fnv1a_hash(const void *data, size_t len, u32 seed)
{
    const u8 *p = (const u8 *)data;
    u32 h = 2166136261u ^ seed;
    size_t i;
    for (i = 0; i < len; i++) {
        h ^= p[i];
        h *= 16777619u;
    }
    return h;
}

void *memset_explicit(void *dst, i32 val, size_t n)
{
    volatile u8 *p = (volatile u8 *)dst;
    size_t i;
    for (i = 0; i < n; i++) p[i] = (u8)val;
    return dst;
}

a8  a8_make(u8 *data, size_t count)  { a8  s; s.data = data; s.count = count; return s; }
a16 a16_make(u16 *data, size_t count) { a16 s; s.data = data; s.count = count; return s; }
a32 a32_make(u32 *data, size_t count) { a32 s; s.data = data; s.count = count; return s; }
a64 a64_make(u64 *data, size_t count) { a64 s; s.data = data; s.count = count; return s; }

#endif /* BASE_TYPE_IMPLEMENTATION_ */
#endif /* BASE_TYPE_IMPLEMENTATION */

#ifdef BIT_IMPLEMENTATION
#ifndef BIT_IMPLEMENTATION_
#define BIT_IMPLEMENTATION_

u32 bit_extract32(u32 x, i32 pos, i32 width)
{
    u32 mask = (width >= 32) ? (u32)0xffffffffu : (((u32)1 << width) - 1);
    return (x >> pos) & mask;
}

u32 bit_insert32(u32 x, u32 value, i32 pos, i32 width)
{
    u32 mask = (width >= 32) ? (u32)0xffffffffu : (((u32)1 << width) - 1);
    return (x & ~(mask << pos)) | ((value & mask) << pos);
}

u64 bit_extract64(u64 x, i32 pos, i32 width)
{
    u64 mask = (width >= 64) ? ~(u64)0 : (((u64)1 << width) - 1);
    return (x >> pos) & mask;
}

u64 bit_insert64(u64 x, u64 value, i32 pos, i32 width)
{
    u64 mask = (width >= 64) ? ~(u64)0 : (((u64)1 << width) - 1);
    return (x & ~(mask << pos)) | ((value & mask) << pos);
}

u32 bit_pdep32(u32 src, u32 mask)
{
    u32 result = 0;
    u32 bit = 1;
    while (mask != 0) {
        u32 lsb = mask & (~mask + 1); /* isolate mask's lowest set bit */
        if (src & bit) result |= lsb;
        mask &= mask - 1; /* clear that bit from mask */
        bit <<= 1;
    }
    return result;
}

u32 bit_pext32(u32 src, u32 mask)
{
    u32 result = 0;
    u32 bit = 1;
    while (mask != 0) {
        u32 lsb = mask & (~mask + 1);
        if (src & lsb) result |= bit;
        mask &= mask - 1;
        bit <<= 1;
    }
    return result;
}

u64 bit_isolate_lsb64(u64 x)
{
    return x & (~x + 1);
}

u64 bit_clear_lsb64(u64 x)
{
    return x & (x - 1);
}

i32 bit_lsb_index64(u64 x)
{
    i32 not_zero = (x != 0);
    u64 safe_x = x | (u64)(!not_zero); /* avoid UB from ctzll(0) below */
    return (i32)(__builtin_ctzl((unsigned long)safe_x)) * not_zero + (-1) * (1 - not_zero);
}

u8 bit_reverse8(u8 x)
{
    x = (u8)(((x & 0xf0u) >> 4) | ((x & 0x0fu) << 4));
    x = (u8)(((x & 0xccu) >> 2) | ((x & 0x33u) << 2));
    x = (u8)(((x & 0xaau) >> 1) | ((x & 0x55u) << 1));
    return x;
}

u32 bit_reverse32(u32 x)
{
    x = ((x & 0xffff0000u) >> 16) | ((x & 0x0000ffffu) << 16);
    x = ((x & 0xff00ff00u) >> 8)  | ((x & 0x00ff00ffu) << 8);
    x = ((x & 0xf0f0f0f0u) >> 4)  | ((x & 0x0f0f0f0fu) << 4);
    x = ((x & 0xccccccccu) >> 2)  | ((x & 0x33333333u) << 2);
    x = ((x & 0xaaaaaaaau) >> 1)  | ((x & 0x55555555u) << 1);
    return x;
}

size_t align_up(size_t x, size_t align)
{
    return (x + align - 1) & ~(align - 1);
}

size_t align_down(size_t x, size_t align)
{
    return x & ~(align - 1);
}

i32 is_pow2(size_t x)
{
    return (x != 0) && ((x & (x - 1)) == 0);
}

i32 bit_select_i32(i32 cond, i32 a, i32 b)
{
    i32 mask = -cond; /* cond==1 -> all-1s, cond==0 -> 0 */
    return (a & mask) | (b & ~mask);
}

i32 min_i32(i32 a, i32 b)
{
    return b ^ ((a ^ b) & -(a < b));
}

i32 max_i32(i32 a, i32 b)
{
    return a ^ ((a ^ b) & -(a < b));
}

i32 clamp_i32(i32 x, i32 lo, i32 hi)
{
    return min_i32(max_i32(x, lo), hi);
}

i32 abs_i32(i32 x)
{
    i32 mask = x >> 31; /* all-1s if negative, 0 if non-negative (arithmetic shift) */
    return (x ^ mask) - mask;
}

u16 byteswap16(u16 x)
{
    return (u16)((x >> 8) | (x << 8));
}

u32 byteswap32(u32 x)
{
    return ((x >> 24) & 0x000000ffu) | ((x >> 8) & 0x0000ff00u) |
           ((x << 8) & 0x00ff0000u) | ((x << 24) & 0xff000000u);
}

u64 byteswap64(u64 x)
{
    u64 lo = byteswap32((u32)(x & 0xffffffffu));
    u64 hi = byteswap32((u32)(x >> 32));
    return (lo << 32) | hi;
}

i32 is_little_endian(void)
{
    const u32 one = 1;
    return *(const u8 *)&one == 1;
}

i32 popcount32(u32 x)
{
    return __builtin_popcount(x);
}

i32 popcount64(u64 x)
{
    return __builtin_popcountl((unsigned long)x);
}

i32 clz32(u32 x)
{
    return __builtin_clz(x);
}

i32 ctz32(u32 x)
{
    return __builtin_ctz(x);
}

u32 next_pow2_32(u32 x)
{
    i32 is_zero = (x == 0);
    u32 v = x - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return (v + 1) | (u32)(-is_zero & 1); /* x==0 -> 1, same as calling this on 1 */
}

i32 sign_extend32(u32 x, i32 bits)
{
    u32 m = (u32)1 << (bits - 1);
    return (i32)((x ^ m) - m);
}

#endif /* BIT_IMPLEMENTATION_ */
#endif /* BIT_IMPLEMENTATION */


#ifdef AVX_MATH_IMPLEMENTATION
#ifndef AVX_MATH_IMPLEMENTATION_
#define AVX_MATH_IMPLEMENTATION_

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>

i32 avx_math_available(void)
{
    static i32 cached = -1;
    if (cached < 0) cached = __builtin_cpu_supports("avx2") ? 1 : 0;
    return cached;
}

__attribute__((target("avx2")))
f32 avx_sum8(const f32 *data)
{
    __m256 v = _mm256_loadu_ps(data);
    f32 buf[8];
    _mm256_storeu_ps(buf, v);
    return buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
}

__attribute__((target("avx2")))
f32 avx_dot8(const f32 *a, const f32 *b)
{
    __m256 va = _mm256_loadu_ps(a);
    __m256 vb = _mm256_loadu_ps(b);
    __m256 prod = _mm256_mul_ps(va, vb);
    f32 buf[8];
    _mm256_storeu_ps(buf, prod);
    return buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
}

__attribute__((target("avx2")))
f32 avx_min8(const f32 *data)
{
    __m256 v = _mm256_loadu_ps(data);
    f32 buf[8];
    f32 m;
    i32 i;
    _mm256_storeu_ps(buf, v);
    m = buf[0];
    for (i = 1; i < 8; i++) if (buf[i] < m) m = buf[i];
    return m;
}

__attribute__((target("avx2")))
f32 avx_max8(const f32 *data)
{
    __m256 v = _mm256_loadu_ps(data);
    f32 buf[8];
    f32 m;
    i32 i;
    _mm256_storeu_ps(buf, v);
    m = buf[0];
    for (i = 1; i < 8; i++) if (buf[i] > m) m = buf[i];
    return m;
}

#endif /* x86 */

#endif /* AVX_MATH_IMPLEMENTATION_ */
#endif /* AVX_MATH_IMPLEMENTATION */

#ifdef SSE_MATH_IMPLEMENTATION
#ifndef SSE_MATH_IMPLEMENTATION_
#define SSE_MATH_IMPLEMENTATION_

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>

i32 sse_math_available(void)
{
    static i32 cached = -1;
    if (cached < 0) cached = __builtin_cpu_supports("sse2") ? 1 : 0;
    return cached;
}

__attribute__((target("sse2")))
f32 sse_sum4(const f32 *data)
{
    __m128 v = _mm_loadu_ps(data);
    f32 buf[4];
    _mm_storeu_ps(buf, v);
    return buf[0] + buf[1] + buf[2] + buf[3];
}

__attribute__((target("sse2")))
f32 sse_dot4(const f32 *a, const f32 *b)
{
    __m128 va = _mm_loadu_ps(a);
    __m128 vb = _mm_loadu_ps(b);
    __m128 prod = _mm_mul_ps(va, vb);
    f32 buf[4];
    _mm_storeu_ps(buf, prod);
    return buf[0] + buf[1] + buf[2] + buf[3];
}

__attribute__((target("sse2")))
f32 sse_min4(const f32 *data)
{
    __m128 v = _mm_loadu_ps(data);
    f32 buf[4];
    f32 m;
    i32 i;
    _mm_storeu_ps(buf, v);
    m = buf[0];
    for (i = 1; i < 4; i++) if (buf[i] < m) m = buf[i];
    return m;
}

__attribute__((target("sse2")))
f32 sse_max4(const f32 *data)
{
    __m128 v = _mm_loadu_ps(data);
    f32 buf[4];
    f32 m;
    i32 i;
    _mm_storeu_ps(buf, v);
    m = buf[0];
    for (i = 1; i < 4; i++) if (buf[i] > m) m = buf[i];
    return m;
}

#endif /* x86 */

#endif /* SSE_MATH_IMPLEMENTATION_ */
#endif /* SSE_MATH_IMPLEMENTATION */
