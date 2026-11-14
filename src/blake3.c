/* blake3.c - BLAKE3 (the specification: github.com/BLAKE3-team/BLAKE3-specs), its plain hash mode: the 32-byte
 * digest of a buffer, in one call. C89, no SIMD, the same digest on any byte order (words are read and written
 * little-endian byte by byte). store_sum.c checks every block of a store file with it; tests/store_test.c checks it
 * against the reference implementation's digests. */
#ifndef BLAKE3_C
#define BLAKE3_C
#include "base.h"
#include <string.h>

#define B3_CHUNK_START 1u
#define B3_CHUNK_END   2u
#define B3_PARENT      4u
#define B3_ROOT        8u

static const u32 b3_iv_[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                               0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };
static const u8 b3_perm_[16] = { 2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8 };

#define B3_ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define B3_G(a, b, c, d, x, y) do { \
    v[a] += v[b] + (x); v[d] = B3_ROTR(v[d] ^ v[a], 16); v[c] += v[d]; v[b] = B3_ROTR(v[b] ^ v[c], 12); \
    v[a] += v[b] + (y); v[d] = B3_ROTR(v[d] ^ v[a], 8);  v[c] += v[d]; v[b] = B3_ROTR(v[b] ^ v[c], 7); } while (0)

/* the compression function; out (may be cv itself) gets the first half of its output, a chaining value */
static void b3_compress_(const u32 cv[8], const u32 blk[16], u64 ctr, u32 len, u32 flags, u32 out[8])
{
    u32 v[16], m[16], t[16];
    i32 r, i;
    for (i = 0; i < 8; ++i) v[i] = cv[i];
    v[8] = b3_iv_[0]; v[9] = b3_iv_[1]; v[10] = b3_iv_[2]; v[11] = b3_iv_[3];
    v[12] = (u32)ctr; v[13] = (u32)(ctr >> 32); v[14] = len; v[15] = flags;
    for (i = 0; i < 16; ++i) m[i] = blk[i];
    for (r = 0; r < 7; ++r) {
        B3_G(0, 4, 8, 12, m[0], m[1]);   B3_G(1, 5, 9, 13, m[2], m[3]);
        B3_G(2, 6, 10, 14, m[4], m[5]);  B3_G(3, 7, 11, 15, m[6], m[7]);
        B3_G(0, 5, 10, 15, m[8], m[9]);  B3_G(1, 6, 11, 12, m[10], m[11]);
        B3_G(2, 7, 8, 13, m[12], m[13]); B3_G(3, 4, 9, 14, m[14], m[15]);
        if (r < 6) { for (i = 0; i < 16; ++i) t[i] = m[b3_perm_[i]]; for (i = 0; i < 16; ++i) m[i] = t[i]; }
    }
    for (i = 0; i < 8; ++i) out[i] = v[i] ^ v[i + 8];
}

/* n <= 64 bytes as 16 little-endian words, zero-padded */
static void b3_words_(const u8 *p, size_t n, u32 w[16])
{
    u8 b[64];
    i32 i;
    memset(b, 0, sizeof(b));
    if (n) memcpy(b, p, n);
    for (i = 0; i < 16; ++i) w[i] = (u32)b[4 * i] | (u32)b[4 * i + 1] << 8 | (u32)b[4 * i + 2] << 16 | (u32)b[4 * i + 3] << 24;
}

/* what is left to compress of a chunk (or a parent node): its last block, not yet compressed */
typedef struct { u32 cv[8], w[16], len, flags; u64 ctr; } B3Out;

/* a chunk of n <= 1024 bytes, number ctr: every block but the last compressed into the chaining value */
static void b3_chunk_(const u8 *p, size_t n, u64 ctr, B3Out *o)
{
    size_t off = 0;
    u32 w[16], start = B3_CHUNK_START;
    i32 i;
    for (i = 0; i < 8; ++i) o->cv[i] = b3_iv_[i];
    while (n - off > 64) {
        b3_words_(p + off, 64, w);
        b3_compress_(o->cv, w, ctr, 64, start, o->cv);
        start = 0;
        off += 64;
    }
    b3_words_(p ? p + off : p, n - off, o->w);
    o->len = (u32)(n - off); o->ctr = ctr; o->flags = start | B3_CHUNK_END;
}

/* the parent node of two chaining values */
static void b3_parent_(const u32 l[8], const u32 r[8], u32 out[8])
{
    u32 w[16];
    i32 i;
    for (i = 0; i < 8; ++i) { w[i] = l[i]; w[i + 8] = r[i]; }
    b3_compress_(b3_iv_, w, 0, 64, B3_PARENT, out);
}

/* the 32-byte BLAKE3 digest of n bytes */
static void b3_hash(const void *in, size_t n, u8 out[32])
{
    const u8 *p = (const u8 *)in;
    u32 stack[54][8], cv[8];
    i32 sp = 0, i;
    u64 ctr = 0, t;
    B3Out o;
    while (n > 1024) {                                           /* a whole chunk with more after it: not the root */
        b3_chunk_(p, 1024, ctr, &o);
        b3_compress_(o.cv, o.w, o.ctr, o.len, o.flags, cv);
        for (t = ++ctr; !(t & 1u); t >>= 1) b3_parent_(stack[--sp], cv, cv);   /* the subtrees it completes */
        memcpy(stack[sp++], cv, sizeof(cv));
        p += 1024; n -= 1024;
    }
    b3_chunk_(p, n, ctr, &o);                                    /* the last chunk, then up the right edge of the tree */
    while (sp > 0) {
        b3_compress_(o.cv, o.w, o.ctr, o.len, o.flags, cv);
        --sp;
        for (i = 0; i < 8; ++i) { o.w[i] = stack[sp][i]; o.w[i + 8] = cv[i]; o.cv[i] = b3_iv_[i]; }
        o.ctr = 0; o.len = 64; o.flags = B3_PARENT;
    }
    b3_compress_(o.cv, o.w, 0, o.len, o.flags | B3_ROOT, cv);
    for (i = 0; i < 8; ++i) {
        out[4 * i] = (u8)cv[i]; out[4 * i + 1] = (u8)(cv[i] >> 8);
        out[4 * i + 2] = (u8)(cv[i] >> 16); out[4 * i + 3] = (u8)(cv[i] >> 24);
    }
}
#endif
