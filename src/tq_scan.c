/* tq_scan.c - the scan: the fast-scan block kernel, the reservoir of candidates, probing the clusters, one query or a batch.
 * Part of tq.c, which includes it at this point; not compiled on its own. */

/* ---- scanning ---- */

/* Scores of the 32 vectors of one block into out[32] (u16, bigger = closer). Returns a bit mask of the rows
 * scoring above thr, so the caller only touches those (most rows of a scan never get near the top R).
 * AVX-512BW: 4 dimensions per 64-byte step; AVX2: 2 per step; NEON: 2 per step (two 16-byte loads); the lane sums
 * and the threshold test stay in registers. */
#if defined(VS_NEON)
/* the 16-entry byte table t indexed by the 16 bytes of i (each < 16) */
static uint8x16_t tq_tbl16_(uint8x16_t t, uint8x16_t i)
{
#if defined(__aarch64__)
    return vqtbl1q_u8(t, i);
#else
    uint8x8x2_t tt;
    tt.val[0] = vget_low_u8(t); tt.val[1] = vget_high_u8(t);
    return vcombine_u8(vtbl2_u8(tt, vget_low_u8(i)), vtbl2_u8(tt, vget_high_u8(i)));
#endif
}
#endif

static u32 tq_block_(const u8 *blk, const u8 *lut, i32 dpad, u16 thr, u16 *out)
{
#if defined(__AVX2__)
    __m256i a0, a1, b0, b1;
    const __m256i m0f = _mm256_set1_epi8(0x0f);
    __m128i x0, x1, y0, y1, r0q, r1q, r2q, r3q;
    i32 d = 0;
#if defined(__AVX512BW__)
    {
        __m512i A0 = _mm512_setzero_si512(), A1 = A0, B0 = A0, B1 = A0;
        const __m512i M0F = _mm512_set1_epi8(0x0f);
        for (; d + 4 <= dpad; d += 4) {
            __m512i codes = _mm512_loadu_si512((const void *)(blk + (size_t)d * 16));
            __m512i lv = _mm512_loadu_si512((const void *)(lut + (size_t)d * 16));
            __m512i lo = _mm512_and_si512(codes, M0F);
            __m512i hi = _mm512_and_si512(_mm512_srli_epi16(codes, 4), M0F);
            __m512i r0 = _mm512_shuffle_epi8(lv, lo), r1 = _mm512_shuffle_epi8(lv, hi);
            A0 = _mm512_add_epi16(A0, r0); A1 = _mm512_add_epi16(A1, _mm512_srli_epi16(r0, 8));
            B0 = _mm512_add_epi16(B0, r1); B1 = _mm512_add_epi16(B1, _mm512_srli_epi16(r1, 8));
        }
        a0 = _mm256_add_epi16(_mm512_castsi512_si256(A0), _mm512_extracti64x4_epi64(A0, 1));
        a1 = _mm256_add_epi16(_mm512_castsi512_si256(A1), _mm512_extracti64x4_epi64(A1, 1));
        b0 = _mm256_add_epi16(_mm512_castsi512_si256(B0), _mm512_extracti64x4_epi64(B0, 1));
        b1 = _mm256_add_epi16(_mm512_castsi512_si256(B1), _mm512_extracti64x4_epi64(B1, 1));
    }
#else
    a0 = _mm256_setzero_si256(); a1 = a0; b0 = a0; b1 = a0;
#endif
    for (; d < dpad; d += 2) {
        __m256i codes = _mm256_loadu_si256((const __m256i *)(blk + (size_t)d * 16));
        __m256i lv = _mm256_loadu_si256((const __m256i *)(lut + (size_t)d * 16));
        __m256i lo = _mm256_and_si256(codes, m0f);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(codes, 4), m0f);
        __m256i r0 = _mm256_shuffle_epi8(lv, lo), r1 = _mm256_shuffle_epi8(lv, hi);
        a0 = _mm256_add_epi16(a0, r0); a1 = _mm256_add_epi16(a1, _mm256_srli_epi16(r0, 8));
        b0 = _mm256_add_epi16(b0, r1); b1 = _mm256_add_epi16(b1, _mm256_srli_epi16(r1, 8));
    }
    /* fold the two dimension lanes; a1/b1 hold the odd rows of 0-15 / 16-31, and a0/b0 every 16-bit word whole (even
     * row + 256 x odd row, mod 2^16 - FAISS's trick: no mask per step), so the even rows are a0 - (a1 << 8) */
    x0 = _mm_add_epi16(_mm256_castsi256_si128(a0), _mm256_extracti128_si256(a0, 1));
    x1 = _mm_add_epi16(_mm256_castsi256_si128(a1), _mm256_extracti128_si256(a1, 1));
    y0 = _mm_add_epi16(_mm256_castsi256_si128(b0), _mm256_extracti128_si256(b0, 1));
    y1 = _mm_add_epi16(_mm256_castsi256_si128(b1), _mm256_extracti128_si256(b1, 1));
    x0 = _mm_sub_epi16(x0, _mm_slli_epi16(x1, 8));
    y0 = _mm_sub_epi16(y0, _mm_slli_epi16(y1, 8));
    r0q = _mm_unpacklo_epi16(x0, x1); r1q = _mm_unpackhi_epi16(x0, x1);     /* rows 0-7, 8-15 */
    r2q = _mm_unpacklo_epi16(y0, y1); r3q = _mm_unpackhi_epi16(y0, y1);     /* rows 16-23, 24-31 */
    _mm_storeu_si128((__m128i *)(void *)out, r0q); _mm_storeu_si128((__m128i *)(void *)(out + 8), r1q);
    _mm_storeu_si128((__m128i *)(void *)(out + 16), r2q); _mm_storeu_si128((__m128i *)(void *)(out + 24), r3q);
    {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
        __m128i t = _mm_set1_epi16((short)thr);
        return (u32)_mm_cmpgt_epu16_mask(r0q, t) | ((u32)_mm_cmpgt_epu16_mask(r1q, t) << 8)
             | ((u32)_mm_cmpgt_epu16_mask(r2q, t) << 16) | ((u32)_mm_cmpgt_epu16_mask(r3q, t) << 24);
#else
        /* unsigned v > thr: flip the sign bits, compare signed, pack the 0/-1 lanes to bytes */
        __m128i bias = _mm_set1_epi16((short)0x8000), t = _mm_set1_epi16((short)(thr ^ 0x8000));
        u32 m0 = (u32)_mm_movemask_epi8(_mm_packs_epi16(_mm_cmpgt_epi16(_mm_xor_si128(r0q, bias), t),
                                                        _mm_cmpgt_epi16(_mm_xor_si128(r1q, bias), t)));
        u32 m1 = (u32)_mm_movemask_epi8(_mm_packs_epi16(_mm_cmpgt_epi16(_mm_xor_si128(r2q, bias), t),
                                                        _mm_cmpgt_epi16(_mm_xor_si128(r3q, bias), t)));
        return m0 | (m1 << 16);
#endif
    }
#elif defined(VS_NEON)
    /* NEON: one 16-byte table lookup per half-byte row; two dimensions are summed in u16 before they meet the
     * accumulators, so each accumulator chain takes one add per two dimensions. Same u16 sums as the scalar code. */
    uint16x8_t a0 = vdupq_n_u16(0), a1 = a0, b0 = a0, b1 = a0;
    const uint8x16_t m0f = vdupq_n_u8(0x0f);
    u32 m = 0;
    i32 d = 0, j;
    for (; d + 2 <= dpad; d += 2) {
        uint8x16_t c0 = vld1q_u8(blk + (size_t)d * 16), c1 = vld1q_u8(blk + (size_t)d * 16 + 16);
        uint8x16_t l0 = vld1q_u8(lut + (size_t)d * 16), l1 = vld1q_u8(lut + (size_t)d * 16 + 16);
        uint8x16_t r0 = tq_tbl16_(l0, vandq_u8(c0, m0f)), r1 = tq_tbl16_(l0, vshrq_n_u8(c0, 4));
        uint8x16_t s0 = tq_tbl16_(l1, vandq_u8(c1, m0f)), s1 = tq_tbl16_(l1, vshrq_n_u8(c1, 4));
        a0 = vaddq_u16(a0, vaddl_u8(vget_low_u8(r0), vget_low_u8(s0)));
        a1 = vaddq_u16(a1, vaddl_u8(vget_high_u8(r0), vget_high_u8(s0)));
        b0 = vaddq_u16(b0, vaddl_u8(vget_low_u8(r1), vget_low_u8(s1)));
        b1 = vaddq_u16(b1, vaddl_u8(vget_high_u8(r1), vget_high_u8(s1)));
    }
    if (d < dpad) {
        uint8x16_t c0 = vld1q_u8(blk + (size_t)d * 16), l0 = vld1q_u8(lut + (size_t)d * 16);
        uint8x16_t r0 = tq_tbl16_(l0, vandq_u8(c0, m0f)), r1 = tq_tbl16_(l0, vshrq_n_u8(c0, 4));
        a0 = vaddw_u8(a0, vget_low_u8(r0)); a1 = vaddw_u8(a1, vget_high_u8(r0));
        b0 = vaddw_u8(b0, vget_low_u8(r1)); b1 = vaddw_u8(b1, vget_high_u8(r1));
    }
    vst1q_u16(out, a0); vst1q_u16(out + 8, a1); vst1q_u16(out + 16, b0); vst1q_u16(out + 24, b1);
#if defined(__aarch64__)
    {   /* rows above thr: compare, weight lane i by 1 << i, add the lanes */
        static const u16 w8[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };
        const uint16x8_t w = vld1q_u16(w8), t = vdupq_n_u16(thr);
        m = (u32)vaddvq_u16(vandq_u16(vcgtq_u16(a0, t), w)) | ((u32)vaddvq_u16(vandq_u16(vcgtq_u16(a1, t), w)) << 8)
          | ((u32)vaddvq_u16(vandq_u16(vcgtq_u16(b0, t), w)) << 16) | ((u32)vaddvq_u16(vandq_u16(vcgtq_u16(b1, t), w)) << 24);
        (void)j;
    }
#else
    for (j = 0; j < 32; ++j) if (out[j] > thr) m |= 1u << j;
#endif
    return m;
#else
    i32 d, j;
    u32 m = 0;
    for (j = 0; j < 32; ++j) out[j] = 0;
    for (d = 0; d < dpad; ++d)
        for (j = 0; j < 32; ++j) {
            u8 byte = blk[(size_t)d * 16 + (j & 15)];
            u8 c = j < 16 ? (u8)(byte & 15) : (u8)(byte >> 4);
            out[j] = (u16)(out[j] + lut[(size_t)d * 16 + c]);
        }
    for (j = 0; j < 32; ++j) if (out[j] > thr) m |= 1u << j;
    return m;
#endif
}

/* index of the lowest set bit of a non-zero mask */
static i32 tq_ctz_(u32 m)
{
#if defined(__GNUC__)
    return __builtin_ctz(m);
#else
    i32 k = 0;
    while (!(m & 1u)) { m >>= 1; ++k; }
    return k;
#endif
}

/* moves the k largest keys of a[0..n) to a[0..k), in no particular order (quickselect, median of three, O(n)) */
static void tq_sel64_(u64 *a, i32 n, i32 k)
{
    i32 lo = 0, hi = n - 1;
    if (k <= 0 || k >= n) return;
    while (lo < hi) {
        i32 i = lo, j = hi;
        u64 x = a[lo], y = a[lo + (hi - lo) / 2], z = a[hi], piv;
        piv = x < y ? (y < z ? y : (x < z ? z : x)) : (x < z ? x : (y < z ? z : y));
        while (i <= j) {
            while (a[i] > piv) ++i;
            while (a[j] < piv) --j;
            if (i <= j) { u64 w = a[i]; a[i] = a[j]; a[j] = w; ++i; --j; }
        }
        if (k - 1 <= j) hi = j;
        else if (k - 1 >= i) lo = i;
        else break;
    }
}

/* sorts a[0..n) by bytes b0 .. b0 + nb - 1 of the high half (key >> 32), biggest first; everything else rides along.
 * LSD radix (b0 = 0, nb = 2: u16 scores; b0 = 2, nb = 2: the top 16 bits of an f32 key, close enough to order the
 * probes); tmp: n entries. Stable, O(n), no comparisons. */
static void tq_rsort_(u64 *a, i32 n, u64 *tmp, i32 b0, i32 nb)
{
    i32 c[256], p, i, k, s;
    u64 *src = a, *dst = tmp, *x;
    if (n < 2) return;
    for (p = b0; p < b0 + nb; ++p) {
        i32 sh = 32 + 8 * p;
        memset(c, 0, sizeof c);
        for (i = 0; i < n; ++i) ++c[255 - (i32)((src[i] >> sh) & 255u)];
        for (k = 0, s = 0; k < 256; ++k) { i32 w = c[k]; c[k] = s; s += w; }
        for (i = 0; i < n; ++i) dst[c[255 - (i32)((src[i] >> sh) & 255u)]++] = src[i];
        x = src; src = dst; dst = x;
    }
    if (src != a) memcpy(a, src, (size_t)n * sizeof(u64));
}

/* an f32 as a u32 that sorts the same way */
static u32 tq_fkey_(f32 f)
{
    u32 u;
    memcpy(&u, &f, sizeof u);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/* The R best (score << 32 | row) keys seen so far. A key that beats thr is appended to a buffer of 2R; a full buffer
 * is cut back to its R best (quickselect), which raises thr to the R-th best key. An append is a store and a cut is
 * O(R) once per R appends - a heap paid a sift of log R steps on every row that got in. */
typedef struct { u64 *v; u64 thr; i32 n, R, cut; } TqRes;

static void tq_res_init_(TqRes *r, u64 *buf, i32 R) { r->v = buf; r->thr = 0; r->n = 0; r->R = R; r->cut = 0; }

static void tq_res_cut_(TqRes *r)
{
    i32 i;
    u64 m;
    if (r->n < r->R) return;
    if (r->n > r->R) { tq_sel64_(r->v, r->n, r->R); r->n = r->R; }
    for (m = r->v[0], i = 1; i < r->R; ++i) if (r->v[i] < m) m = r->v[i];
    r->thr = m; r->cut = 1;
}

static void tq_res_push_(TqRes *r, u64 key)
{
    if (r->cut && key <= r->thr) return;
    r->v[r->n++] = key;
    if (r->n == 2 * r->R) tq_res_cut_(r);
}

/* The kernel's threshold for the next block. A row that ties the R-th best score still gets the full key comparison,
 * so the R best keys come out the same whatever order the blocks are scanned in (one query or a batch, IVF or flat).
 * tq_res_all_: no threshold yet, every row counts. */
static u16 tq_res_kthr_(const TqRes *r) { u16 f = r->cut ? (u16)(r->thr >> 32) : (u16)0; return f ? (u16)(f - 1) : (u16)0; }
static i32 tq_res_all_(const TqRes *r) { return !r->cut || !(r->thr >> 32); }

/* the reservoir's rows, best key first, into rows[]; returns how many (at most R) */
static i32 tq_res_rows_(TqRes *r, i32 *rows, u64 *tmp)
{
    i32 i, j;
    u64 *a = r->v;
    if (r->n > r->R) { tq_sel64_(a, r->n, r->R); r->n = r->R; }
    tq_rsort_(a, r->n, tmp, 0, 2);
    for (i = 1; i < r->n; ++i) {                                 /* equal scores: bigger row first, as the whole key says */
        u64 v = a[i];
        for (j = i; j > 0 && a[j - 1] < v; --j) a[j] = a[j - 1];
        a[j] = v;
    }
    for (i = 0; i < r->n; ++i) rows[i] = (i32)(u32)(a[i] & 0xffffffffu);
    return r->n;
}

/* pulls a block of codes toward L1 (vd * 16 bytes) */
static void tq_prefetch_blk_(const u8 *p, i32 bytes)
{
#if defined(__GNUC__)
    i32 o;
    for (o = 0; o < bytes; o += 64) __builtin_prefetch((const void *)(p + o), 0, 3);
#else
    (void)p; (void)bytes;
#endif
}

typedef struct {
    const Tq *t; const u8 *lut;
    i32 b0, b1, R;
    const i32 *blks;                 /* IVF: the blocks to scan (b0..b1 index into this list); NULL = blocks b0..b1 */
    const u64 *dead; const TqFilt *f;   /* NULL: every live row */
    i32 slots;                       /* keep slot numbers (block * 32 + lane) instead of rows: the cascade's first stage */
    TqRes res;
} TqJob;

static void *tq_job_(void *arg)
{
    TqJob *j = (TqJob *)arg;
    const Tq *t = j->t;
    i32 b, k, bb = t->vd * 16;
    u16 sc[32];
    for (b = j->b0; b < j->b1; ++b) {
        i32 blk = j->blks ? j->blks[b] : b;
        u32 m;
        if (b + 2 < j->b1)                                       /* two blocks ahead: the hardware prefetcher stops at every
                                                                  * 4 KB page and every jump to another cluster */
            tq_prefetch_blk_(tq_bptr_(t, j->blks ? j->blks[b + 2] : b + 2), bb);
        m = tq_block_(tq_bptr_(t, blk), j->lut, t->vd, tq_res_kthr_(&j->res), sc);
        if (tq_res_all_(&j->res)) m = 0xffffffffu;               /* no threshold yet: every row counts */
        while (m) {
            i32 row;
            k = tq_ctz_(m); m &= m - 1u;
            row = tq_srow_(t, blk, k);
            if (row < 0 || row >= t->n) continue;                /* padding */
            if (j->dead && ((j->dead[(u32)row >> 6] >> (row & 63)) & 1u)) continue;
            if (j->f) {
                if (j->f->meta) {
                    i64 tag = *(const i64 *)(const void *)TQ_AT(j->f->meta, row);
                    if ((tag & j->f->all) != j->f->all) continue;
                    if (j->f->any && !(tag & j->f->any)) continue;
                }
                if (j->f->num) {
                    i64 v = *(const i64 *)(const void *)TQ_AT(j->f->num, row);
                    if (v < j->f->lo || v > j->f->hi) continue;
                }
                if (j->f->ns) {
                    i64 v = *(const i64 *)(const void *)TQ_AT(j->f->ns, row);
                    if (j->f->nns ? !tq_in_set_(j->f->nss, j->f->nns, v) : v != j->f->nsv) continue;
                }
            }
            tq_res_push_(&j->res, ((u64)sc[k] << 32) | (u64)(u32)(j->slots ? blk * 32 + k : row));
        }
    }
    return (void *)0;
}

/* the query's 16-entry tables, one per code column, quantised to u8 (lut: vd * 16 + 32 bytes; rot, tmp: dpad
 * floats; lf: vd * 16 floats). sc, base (may be NULL): what turns a row's u16 sum S back into its estimate - the rotated
 * query dotted with the row's levels is S / sc + base, within vd / (2 sc) (each table entry rounded to the nearest) */
static void tq_lut_ex_(const Tq *t, const f32 *q, f32 *rot, f32 *tmp, f32 *lf, u8 *lut, f32 *sc, f64 *base)
{
    f32 maxr = 0.0f, scale;
    f64 bs = 0.0;
    i32 d, c, i, lutmax;
    tq_rot_(t, q, rot, tmp);
    for (d = 0; d < t->vd; ++d) {
        f32 mn = 1e30f, mx = -1e30f;
        if (t->bits == 2) {                                      /* column d = coordinates 2d, 2d+1, 2 bits each */
            f32 a = 2 * d < t->dpad ? rot[2 * d] : 0.0f, b = 2 * d + 1 < t->dpad ? rot[2 * d + 1] : 0.0f;
            for (c = 0; c < 16; ++c) { f32 v = a * TQ_CENT2[c & 3] + b * TQ_CENT2[c >> 2]; lf[d * 16 + c] = v; if (v < mn) mn = v; if (v > mx) mx = v; }
        } else if (t->bits == 1) {                               /* column d = coordinates 4d..4d+3, a sign bit each */
            f32 x[4];
            for (i = 0; i < 4; ++i) x[i] = 4 * d + i < t->dpad ? rot[4 * d + i] * TQ_L1 : 0.0f;
            for (c = 0; c < 16; ++c) {
                f32 v = (c & 1 ? x[0] : -x[0]) + (c & 2 ? x[1] : -x[1]) + (c & 4 ? x[2] : -x[2]) + (c & 8 ? x[3] : -x[3]);
                lf[d * 16 + c] = v; if (v < mn) mn = v; if (v > mx) mx = v;
            }
        } else
            for (c = 0; c < 16; ++c) { f32 v = rot[d] * TQ_CENT[c]; lf[d * 16 + c] = v; if (v < mn) mn = v; if (v > mx) mx = v; }
        for (c = 0; c < 16; ++c) lf[d * 16 + c] -= mn;
        if (mx - mn > maxr) maxr = mx - mn;
        bs += mn;
    }
    lutmax = 65535 / t->vd; if (lutmax > 255) lutmax = 255;
    scale = maxr > 0.0f ? (f32)lutmax / maxr : 1.0f;
    for (i = 0; i < t->vd * 16; ++i) { i32 v = (i32)(lf[i] * scale + 0.5f); lut[i] = (u8)(v > lutmax ? lutmax : v); }
    if (sc) *sc = scale;
    if (base) *base = bs;
}

static void tq_lut_(const Tq *t, const f32 *q, f32 *rot, f32 *tmp, f32 *lf, u8 *lut)
{
    tq_lut_ex_(t, q, rot, tmp, lf, lut, (f32 *)0, (f64 *)0);
}

#if TQ_VNNI
/* the query for VNNI: q[d] * c8s[d] scaled to +-127, plus 128 (unsigned bytes), packed four dims to a word */
static void tq_q8_(const Tq *t, const f32 *q, u32 *qw)
{
    i32 d, n4 = t->c8d4 * 4;
    f32 mx = 0.0f, sc;
    for (d = 0; d < t->dim; ++d) { f32 v = (f32)fabs((f64)(q[d] * t->c8s[d])); if (v > mx) mx = v; }
    sc = mx > 0.0f ? 127.0f / mx : 0.0f;
    for (d = 0; d < t->c8d4; ++d) qw[d] = 0;
    for (d = 0; d < n4; ++d) {
        f32 x = d < t->dim ? q[d] * t->c8s[d] * sc : 0.0f;
        i32 v = (i32)(x >= 0.0f ? x + 0.5f : x - 0.5f);
        v = v > 127 ? 127 : (v < -127 ? -127 : v);
        qw[d / 4] |= (u32)(128 + v) << (8 * (d % 4));
    }
}

/* int scores of the query against every centroid, bigger = closer, out[16 * ngroups] (padding lanes: INT_MIN). One
 * VNNI instruction scores 16 centroids on 4 dimensions; 8 groups at a time keep the unit busy. 4x fewer bytes than
 * the f32 centroids (GloVe-100: 0.44 MB, which stays in L2) and 4x the work per instruction. */
static void tq_score8_(const Tq *t, const u32 *qw, i32 *out)
{
    i32 ng = (t->nc + 15) / 16, d4 = t->c8d4, g = 0, k;
    size_t gs = (size_t)d4 * 64;
    for (; g + 8 <= ng; g += 8) {
        const i8 *c = t->c8 + (size_t)g * gs;
        __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
        for (k = 0; k < d4; ++k) {
            __m512i x = _mm512_set1_epi32((int)qw[k]);
            const i8 *p = c + (size_t)k * 64;
            a0 = _mm512_dpbusd_epi32(a0, x, _mm512_loadu_si512((const void *)p));
            a1 = _mm512_dpbusd_epi32(a1, x, _mm512_loadu_si512((const void *)(p + gs)));
            a2 = _mm512_dpbusd_epi32(a2, x, _mm512_loadu_si512((const void *)(p + 2 * gs)));
            a3 = _mm512_dpbusd_epi32(a3, x, _mm512_loadu_si512((const void *)(p + 3 * gs)));
            a4 = _mm512_dpbusd_epi32(a4, x, _mm512_loadu_si512((const void *)(p + 4 * gs)));
            a5 = _mm512_dpbusd_epi32(a5, x, _mm512_loadu_si512((const void *)(p + 5 * gs)));
            a6 = _mm512_dpbusd_epi32(a6, x, _mm512_loadu_si512((const void *)(p + 6 * gs)));
            a7 = _mm512_dpbusd_epi32(a7, x, _mm512_loadu_si512((const void *)(p + 7 * gs)));
        }
        { i32 *o = out + (size_t)g * 16; const i32 *bz = t->c8b + (size_t)g * 16;
          _mm512_storeu_si512((void *)o, _mm512_sub_epi32(a0, _mm512_loadu_si512((const void *)bz)));
          _mm512_storeu_si512((void *)(o + 16), _mm512_sub_epi32(a1, _mm512_loadu_si512((const void *)(bz + 16))));
          _mm512_storeu_si512((void *)(o + 32), _mm512_sub_epi32(a2, _mm512_loadu_si512((const void *)(bz + 32))));
          _mm512_storeu_si512((void *)(o + 48), _mm512_sub_epi32(a3, _mm512_loadu_si512((const void *)(bz + 48))));
          _mm512_storeu_si512((void *)(o + 64), _mm512_sub_epi32(a4, _mm512_loadu_si512((const void *)(bz + 64))));
          _mm512_storeu_si512((void *)(o + 80), _mm512_sub_epi32(a5, _mm512_loadu_si512((const void *)(bz + 80))));
          _mm512_storeu_si512((void *)(o + 96), _mm512_sub_epi32(a6, _mm512_loadu_si512((const void *)(bz + 96))));
          _mm512_storeu_si512((void *)(o + 112), _mm512_sub_epi32(a7, _mm512_loadu_si512((const void *)(bz + 112)))); }
    }
    for (; g < ng; ++g) {
        const i8 *c = t->c8 + (size_t)g * gs;
        __m512i a0 = _mm512_setzero_si512(), a1 = a0;
        for (k = 0; k + 2 <= d4; k += 2) {
            a0 = _mm512_dpbusd_epi32(a0, _mm512_set1_epi32((int)qw[k]), _mm512_loadu_si512((const void *)(c + (size_t)k * 64)));
            a1 = _mm512_dpbusd_epi32(a1, _mm512_set1_epi32((int)qw[k + 1]), _mm512_loadu_si512((const void *)(c + (size_t)k * 64 + 64)));
        }
        if (k < d4) a0 = _mm512_dpbusd_epi32(a0, _mm512_set1_epi32((int)qw[k]), _mm512_loadu_si512((const void *)(c + (size_t)k * 64)));
        _mm512_storeu_si512((void *)(out + (size_t)g * 16),
                            _mm512_sub_epi32(_mm512_add_epi32(a0, a1), _mm512_loadu_si512((const void *)(t->c8b + (size_t)g * 16))));
    }
    for (k = t->nc; k < ng * 16; ++k) out[k] = -2147483647 - 1;
}

/* the same for 8 queries at once (qw[r] -> out[r]): every 64 bytes of centroids loaded feed 8 queries, so a batch
 * reads the int8 centroids once per 8 queries instead of once per query */
#define TQ_D8(r) { __m512i x = _mm512_set1_epi32((int)qw[r][k]); \
                   a##r##0 = _mm512_dpbusd_epi32(a##r##0, x, c0); a##r##1 = _mm512_dpbusd_epi32(a##r##1, x, c1); }
#define TQ_S8(r) { _mm512_storeu_si512((void *)(out[r] + (size_t)g * 16), _mm512_sub_epi32(a##r##0, b0)); \
                   _mm512_storeu_si512((void *)(out[r] + (size_t)g * 16 + 16), _mm512_sub_epi32(a##r##1, b1)); }
static void tq_score8x8_(const Tq *t, u32 *const *qw, i32 *const *out)
{
    i32 ng = (t->nc + 15) / 16, d4 = t->c8d4, g = 0, k, r;
    size_t gs = (size_t)d4 * 64;
    for (; g + 2 <= ng; g += 2) {
        const i8 *c = t->c8 + (size_t)g * gs;
        __m512i a00 = _mm512_setzero_si512(), a01 = a00, a10 = a00, a11 = a00, a20 = a00, a21 = a00, a30 = a00, a31 = a00;
        __m512i a40 = a00, a41 = a00, a50 = a00, a51 = a00, a60 = a00, a61 = a00, a70 = a00, a71 = a00, b0, b1;
        for (k = 0; k < d4; ++k) {
            __m512i c0 = _mm512_loadu_si512((const void *)(c + (size_t)k * 64)), c1 = _mm512_loadu_si512((const void *)(c + gs + (size_t)k * 64));
            TQ_D8(0) TQ_D8(1) TQ_D8(2) TQ_D8(3) TQ_D8(4) TQ_D8(5) TQ_D8(6) TQ_D8(7)
        }
        b0 = _mm512_loadu_si512((const void *)(t->c8b + (size_t)g * 16)); b1 = _mm512_loadu_si512((const void *)(t->c8b + (size_t)g * 16 + 16));
        TQ_S8(0) TQ_S8(1) TQ_S8(2) TQ_S8(3) TQ_S8(4) TQ_S8(5) TQ_S8(6) TQ_S8(7)
    }
    for (; g < ng; ++g) {                                        /* an odd last group */
        const i8 *c = t->c8 + (size_t)g * gs;
        for (r = 0; r < 8; ++r) {
            __m512i a = _mm512_setzero_si512();
            for (k = 0; k < d4; ++k) a = _mm512_dpbusd_epi32(a, _mm512_set1_epi32((int)qw[r][k]), _mm512_loadu_si512((const void *)(c + (size_t)k * 64)));
            _mm512_storeu_si512((void *)(out[r] + (size_t)g * 16), _mm512_sub_epi32(a, _mm512_loadu_si512((const void *)(t->c8b + (size_t)g * 16))));
        }
    }
    for (r = 0; r < 8; ++r) for (k = t->nc; k < ng * 16; ++k) out[r][k] = -2147483647 - 1;
}
#undef TQ_D8
#undef TQ_S8

/* how many of s[0..n16) are >= T */
static i32 tq_count_ge_(const i32 *s, i32 n16, i32 T)
{
    __m512i tv = _mm512_set1_epi32(T);
    i32 i, c = 0;
    for (i = 0; i < n16; i += 16) c += __builtin_popcount((u32)_mm512_cmpge_epi32_mask(_mm512_loadu_si512((const void *)(s + i)), tv));
    return c;
}

/* From the int8 scores sc of query q to candidates: the largest threshold that keeps at least K = np + np/4 + 16 of
 * them (a binary search on a vector count, stopping once at most 1.5 K pass - no sort), then exact f32 scores for
 * those few as keys. Returns how many keys. */
static i32 tq_cands_(const Tq *t, const f32 *q, i32 np, const i32 *sc, u64 *key, i32 *cand)
{
    i32 nc = t->nc, ng16 = (nc + 15) / 16 * 16, K = np + np / 4 + 16, K2, lo, hi, i, nk = 0;
    __m512i tv, idx = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    if (K > nc) K = nc;
    K2 = K + K / 2;
    lo = hi = sc[0];
    for (i = 1; i < nc; ++i) { if (sc[i] < lo) lo = sc[i]; if (sc[i] > hi) hi = sc[i]; }
    while (lo < hi) {                                            /* the largest T with at least K scores >= T */
        i32 mid = lo + (i32)(((i64)hi - (i64)lo + 1) / 2), c = tq_count_ge_(sc, ng16, mid);
        if (c >= K) { lo = mid; if (c <= K2) break; }
        else hi = mid - 1;
    }
    tv = _mm512_set1_epi32(lo);
    for (i = 0; i < ng16; i += 16) {
        __mmask16 m = _mm512_cmpge_epi32_mask(_mm512_loadu_si512((const void *)(sc + i)), tv);
        _mm512_mask_compressstoreu_epi32((void *)(cand + nk), m, idx);
        nk += __builtin_popcount((u32)m);
        idx = _mm512_add_epi32(idx, _mm512_set1_epi32(16));
    }
    for (i = 0; i < nk; ++i) {                                    /* exact scores for the few that are left */
        i32 c = cand[i] < nc ? cand[i] : 0;
        key[i] = ((u64)tq_fkey_(vs_dot(q, t->cent + (size_t)c * t->dim, t->dim)) << 32) | (u32)c;
    }
    return nk;
}
#endif

/* scratch bytes for tq_probe_ */
static size_t tq_probe_scr_(const Tq *t)
{
    size_t ng16 = (size_t)((t->nc + 15) / 16) * 16;
    return ng16 * 24 + (size_t)(t->dim + 3) / 4 * 4 + 64;
}

/* the np best of nk keys into out[np]: sorted closest first (by the top 16 bits of the score - enough to make the
 * threshold rise early), or (sorted = 0) only the closest moved first */
static void tq_pick_(u64 *key, i32 nk, i32 np, u64 *tmp, i32 *out, i32 sorted)
{
    i32 i;
    if (nk > np) tq_sel64_(key, nk, np);
    else np = nk;
    if (sorted) tq_rsort_(key, np, tmp, 2, 2);
    else { i32 b = 0; u64 w; for (i = 1; i < np; ++i) if (key[i] > key[b]) b = i; w = key[0]; key[0] = key[b]; key[b] = w; }
    for (i = 0; i < np; ++i) out[i] = (i32)(u32)key[i];
}

/* tq_probe_ through the second level (tq_top.c): the top cells closest to q, taken until they hold tq_top_want_(np)
 * centroids, and f32 scores of those centroids only. */
static void tq_probe2_(const Tq *t, const f32 *q, i32 np, u64 *key, u64 *tmp, i32 *out)
{
    f32 ts[TQ_TOP_KMAX];
    i32 want = tq_top_want_(np), nk = 0, g, i, taken = 0;
    for (g = 0; g < t->nk1; ++g) ts[g] = vs_dot(q, t->top + (size_t)g * t->dim, t->dim);
    while (nk < want && taken < t->nk1) {
        i32 b = 0;
        for (g = 1; g < t->nk1; ++g) if (ts[g] > ts[b]) b = g;
        ts[b] = -4.0f; ++taken;                                  /* (a dot of unit vectors is above -4) */
        for (i = t->tstart[b]; i < t->tstart[b + 1]; ++i) {
            i32 c = t->tlist[i];
            key[nk++] = ((u64)tq_fkey_(vs_dot(q, t->cent + (size_t)c * t->dim, t->dim)) << 32) | (u32)c;
        }
    }
    if (np > nk) np = nk;                                        /* never: want >= np, and every cluster is in a cell */
    tq_pick_(key, nk, np, tmp, out, 1);
}

/* The np clusters closest to q into out[np], closest first. VNNI: int8 scores of every centroid, a threshold that
 * keeps the best np + np/4 + 16 or so, exact f32 scores for only those - so the probes are the f32 ones. Otherwise:
 * f32 scores of every centroid. scr: tq_probe_scr_ bytes. */
static void tq_probe_(const Tq *t, const f32 *q, i32 np, u8 *scr, i32 *out)
{
    i32 nc = t->nc, ng16 = (nc + 15) / 16 * 16, i, nk;
    u64 *key = (u64 *)(void *)scr, *tmp = key + ng16;
    i32 *sc = (i32 *)(void *)(tmp + ng16);
    if (np > nc) np = nc;
    if (np <= 0) return;
    if (t->top && tq_top_want_(np) < nc) { tq_probe2_(t, q, np, key, tmp, out); return; }
#if TQ_VNNI
    if (t->c8) {
        i32 *cand = sc + ng16;
        u32 *qw = (u32 *)(void *)(cand + ng16);
        tq_q8_(t, q, qw);
        tq_score8_(t, qw, sc);
        nk = tq_cands_(t, q, np, sc, key, cand);
    } else
#endif
    {
        f32 *fs = (f32 *)(void *)sc;
        tq_score_(q, t->ct, ng16 / 16, t->dim, fs);              /* every centroid, 16 per instruction */
        for (i = 0; i < nc; ++i) key[i] = ((u64)tq_fkey_(fs[i]) << 32) | (u32)i;
        nk = nc;
    }
    tq_pick_(key, nk, np, tmp, out, 1);
}

/* Query-aware pruning of the probes (the SPANN rule): a cluster is scanned only when its centroid is at most
 * (1 + eps) times as far from the query as the closest centroid is; an easy query, close to one centroid, then reads
 * a few clusters and a hard one all np. The closest TQ_PRUNE_KEEP always stay. eps 0: off. pc[0] must be the closest.
 * compact: the kept probes moved to the front and counted; else a dropped probe becomes -1 and np is returned. */
#ifndef VI_PROBE_PRUNE
#define VI_PROBE_PRUNE 0.0f
#endif
#define TQ_PRUNE_KEEP 4
static f32 tq_prune_eps_ = VI_PROBE_PRUNE;
static i32 tq_prune_(const Tq *t, const f32 *q, i32 *pc, i32 np, i32 compact)
{
    f32 lim, e = 1.0f + tq_prune_eps_;
    i32 i, n = 0, kept = 0;
    if (tq_prune_eps_ <= 0.0f || np <= TQ_PRUNE_KEEP) return np;
    lim = e * e * (2.0f - 2.0f * vs_dot(q, t->cent + (size_t)pc[0] * t->dim, t->dim));   /* squared distances of unit vectors */
    for (i = 0; i < np; ++i) {
        i32 keep = kept < TQ_PRUNE_KEEP || 2.0f - 2.0f * vs_dot(q, t->cent + (size_t)pc[i] * t->dim, t->dim) <= lim;
        kept += keep;
        if (compact) { if (keep) pc[n++] = pc[i]; }
        else if (!keep) pc[i] = -1;
    }
    return compact ? n : np;
}

/* The R best rows (by the quantised inner product) that are alive and pass filt (NULL: no filter); returns how
 * many were found, rows[] best first. dead (a bitset) may be NULL. nthreads > 1 splits the blocks.
 * probe: IVF clusters to scan (0 = the index default, >= nc = all). slots: their slot numbers instead of rows. */
static i32 tq_search_(const Tq *t, const f32 *q, i32 R, i32 *rows, const u64 *dead, const TqFilt *filt, i32 nthreads, i32 probe, i32 slots)
{
    i32 *blks = (i32 *)0, nblk_scan, i, nj, per, tot, np = 0;
    f32 *rot, *tmp, *lf;
    u8 *lut;
    u64 *buf, *mt;
    TqJob jobs[16];
    TqRes all_r;
    pthread_t th[16];
    Scr sc;

    if (t->n == 0 || R <= 0) return 0;
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 16) nthreads = 16;
    if (t->nc > 1) { np = probe > 0 ? probe : t->nprobe; if (np > t->nc) np = t->nc; }
    /* everything the query needs, from one pooled buffer (scratch.h): no malloc on the way, and no memory is an
     * answer - no hits, counted by scr_oom_count_ - not a crash */
    if (scr_get_(&sc, scr_need_((size_t)t->dpad * sizeof(f32)) * 2 + scr_need_((size_t)t->vd * 16 * sizeof(f32)) + scr_need_((size_t)t->vd * 16 + 32)
                      + (t->nc > 1 ? scr_need_(tq_probe_scr_(t)) + scr_need_((size_t)np * sizeof(i32)) + scr_need_((size_t)(t->nblk + t->xn + 1) * sizeof(i32)) : 0)
                      + scr_need_((size_t)nthreads * 2 * (size_t)R * sizeof(u64)) + scr_need_((size_t)R * sizeof(u64))) != 0) return 0;
    rot = (f32 *)scr_take_(&sc, (size_t)t->dpad * sizeof(f32)); tmp = (f32 *)scr_take_(&sc, (size_t)t->dpad * sizeof(f32));
    lf = (f32 *)scr_take_(&sc, (size_t)t->vd * 16 * sizeof(f32)); lut = (u8 *)scr_take_(&sc, (size_t)t->vd * 16 + 32);
    tq_lut_(t, q, rot, tmp, lf, lut);
    if (t->nc > 1) {                                             /* IVF: the blocks of the nprobe clusters closest to the query */
        i32 nb = 0, *pc = (i32 *)scr_take_(&sc, (size_t)np * sizeof(i32));
        u8 *scr = (u8 *)scr_take_(&sc, tq_probe_scr_(t));
        blks = (i32 *)scr_take_(&sc, (size_t)(t->nblk + t->xn + 1) * sizeof(i32));   /* (room for every block: pages it never touches) */
        tq_probe_(t, q, np, scr, pc);
        np = tq_prune_(t, q, pc, np, 1);
        for (i = 0; i < np; ++i) nb += tq_cblocks_(t, pc[i], blks + nb);
        nblk_scan = nb;
    } else nblk_scan = t->nblk + t->xn;                          /* flat: added blocks hold rows in order too */
    if (nblk_scan < 64) nthreads = 1;
    nj = nthreads; per = (nblk_scan + nj - 1) / nj;
    buf = (u64 *)scr_take_(&sc, (size_t)nj * 2 * (size_t)R * sizeof(u64));
    mt = (u64 *)scr_take_(&sc, (size_t)R * sizeof(u64));
    for (i = 0; i < nj; ++i) {
        jobs[i].t = t; jobs[i].lut = lut; jobs[i].blks = blks; jobs[i].b0 = i * per; jobs[i].b1 = (i + 1) * per < nblk_scan ? (i + 1) * per : nblk_scan;
        if (jobs[i].b0 > jobs[i].b1) jobs[i].b0 = jobs[i].b1;
        jobs[i].R = R; jobs[i].dead = dead; jobs[i].f = filt && (filt->meta || filt->num || filt->ns) ? filt : (const TqFilt *)0;
        jobs[i].slots = slots;
        tq_res_init_(&jobs[i].res, buf + (size_t)i * 2 * (size_t)R, R);
    }
    for (i = 1; i < nj; ++i) pthread_create(&th[i], (const pthread_attr_t *)0, tq_job_, &jobs[i]);
    tq_job_(&jobs[0]);
    for (i = 1; i < nj; ++i) pthread_join(th[i], (void **)0);

    for (tot = 0, i = 0; i < nj; ++i) {                          /* every thread's keys side by side, then the R best */
        if (buf + tot != jobs[i].res.v) memmove(buf + tot, jobs[i].res.v, (size_t)jobs[i].res.n * sizeof(u64));
        tot += jobs[i].res.n;
    }
    tq_res_init_(&all_r, buf, R); all_r.n = tot;
    tot = tq_res_rows_(&all_r, rows, mt);
    scr_put_(&sc);
    return tot;
}

i32 tq_search(const Tq *t, const f32 *q, i32 R, i32 *rows, const u64 *dead, const TqFilt *filt, i32 nthreads, i32 probe)
{
    return tq_search_(t, q, R, rows, dead, filt, nthreads, probe, 0);
}

static int tq_key_desc_(const void *a, const void *b)
{
    u64 x = *(const u64 *)a, y = *(const u64 *)b;
    return x < y ? 1 : (x > y ? -1 : 0);
}

/* The cascade, on an index with 1-bit codes beside its 4-bit ones (tq_cascade_on): the S best slots by the 1-bit
 * codes (a quarter of the bytes of the 4-bit scan), those S re-scored with their 4-bit codes, and the R best of them
 * as rows, best first. Arguments and result as tq_search; an index without 1-bit codes is searched as by tq_search. */
i32 tq_search_cascade(const Tq *t, const f32 *q, i32 S, i32 R, i32 *rows, const u64 *dead, const TqFilt *filt, i32 nthreads, i32 probe)
{
    Tq v;
    i32 *sl, ns, i, d, n;
    f32 *rot, *tmp;
    u64 *key;
    if (!t->codes1) return tq_search_(t, q, R, rows, dead, filt, nthreads, probe, 0);
    if (t->n == 0 || R <= 0) return 0;
    if (S < R) S = R;
    v = *t;                                                      /* the 1-bit view: same slots, clusters and rotation */
    v.codes = t->codes1; v.xcodes = t->xcodes1; v.vd = t->vd1; v.bits = 1;
    sl = (i32 *)malloc((size_t)S * sizeof(i32));
    key = (u64 *)malloc((size_t)S * sizeof(u64));
    rot = (f32 *)malloc((size_t)t->dpad * sizeof(f32)); tmp = (f32 *)malloc((size_t)t->dpad * sizeof(f32));
    if (!sl || !key || !rot || !tmp) { free(sl); free(key); free(rot); free(tmp); return 0; }
    ns = tq_search_(&v, q, S, sl, dead, filt, nthreads, probe, 1);
    tq_rot_(t, q, rot, tmp);
    for (i = 0; i < ns; ++i) {                                   /* the inner product with each one's 4-bit codes */
        i32 b = sl[i] >> 5, k = sl[i] & 31;
        const u8 *blk = tq_bptr_(t, b);
        f32 sum = 0.0f;
        if (k < 16) for (d = 0; d < t->dpad; ++d) sum += rot[d] * TQ_CENT[blk[(size_t)d * 16 + (size_t)k] & 15u];
        else for (d = 0; d < t->dpad; ++d) sum += rot[d] * TQ_CENT[blk[(size_t)d * 16 + (size_t)(k - 16)] >> 4];
        key[i] = ((u64)tq_fkey_(sum) << 32) | (u64)(u32)i;
    }
    n = ns < R ? ns : R;
    if (ns > n) tq_sel64_(key, ns, n);
    qsort(key, (size_t)n, sizeof(u64), tq_key_desc_);
    for (i = 0; i < n; ++i) { i32 s = sl[(u32)(key[i] & 0xffffffffu)]; rows[i] = tq_srow_(t, s >> 5, s & 31); }
    free(sl); free(key); free(rot); free(tmp);
    return n;
}

/* the cascade for many queries: each one on its own, the queries split over the threads */
typedef struct { const Tq *t; const f32 *Q; i32 S, R, probe; i32 *rows, *cnt; const u64 *dead; } TqCasMany;
static void tq_cas_many_(void *p, i64 a, i64 b, i32 tid)
{
    TqCasMany *m = (TqCasMany *)p;
    i64 i;
    (void)tid;
    for (i = a; i < b; ++i)
        m->cnt[i] = tq_search_cascade(m->t, m->Q + (size_t)i * (size_t)m->t->dim, m->S, m->R, m->rows + (size_t)i * (size_t)m->R,
                                      m->dead, (const TqFilt *)0, 1, m->probe);
}
void tq_search_many_cascade(const Tq *t, const f32 *Q, i32 nq, i32 S, i32 R, i32 *rows, i32 *cnt, const u64 *dead, i32 nthreads, i32 probe)
{
    TqCasMany m;
    m.t = t; m.Q = Q; m.S = S; m.R = R; m.probe = probe; m.rows = rows; m.cnt = cnt; m.dead = dead;
    tq_par2_(tq_cas_many_, &m, nq, nthreads);
}

/* ---- many queries at once: the scan reads more code bytes than the cache holds, so turn it around - each thread
 * takes a share of the queries, and every block of codes is scored for all of its queries that probe it while the
 * block sits in L1: the codes come in from memory once per batch instead of once per query. No cross-thread merge:
 * a query's candidates live in one reservoir. ---- */
#define TQ_MANY_SUB 256              /* queries a thread batches at a time (bounds the reservoirs: 256 x 2R x 8 bytes) */

typedef struct {
    const Tq *t;
    const f32 *Q;                    /* the batch's queries, unit */
    i32 nq, R, np, lutb;
    u8 *luts;                        /* nq x lutb */
    i32 *probes;                     /* nq x np (IVF) */
    i32 *cq0, *cq;                   /* cluster c's queries: cq[cq0[c] .. cq0[c + 1]) (IVF) */
    const u64 *dead;
    TqRes *res;                      /* nq; each is cut right after the query's closest cluster, so the rest of the scan
                                      * starts with a threshold */
} TqMany;

static void tq_many_push_(const TqMany *m, TqRes *rs, const u8 *blk, i32 blkno, const u8 *lut)
{
    const Tq *t = m->t;
    u16 sc[32];
    i32 k;
    u32 msk = tq_block_(blk, lut, t->vd, tq_res_kthr_(rs), sc);
    if (tq_res_all_(rs)) msk = 0xffffffffu;
    while (msk) {
        i32 row;
        k = tq_ctz_(msk); msk &= msk - 1u;
        row = tq_srow_(t, blkno, k);
        if (row < 0 || row >= t->n) continue;
        if (m->dead && ((m->dead[(u32)row >> 6] >> (row & 63)) & 1u)) continue;
        tq_res_push_(rs, ((u64)sc[k] << 32) | (u64)(u32)row);
    }
}

#define TQ_MANY_QG 32                /* flat scan: queries a tile */
#define TQ_MANY_BG 64                /* flat scan: blocks a tile (about 100 KB of codes at d = 100) */

/* one thread, nq (<= TQ_MANY_SUB) queries: rows nq x R, cnt nq */
static void tq_many_one_(const Tq *t, const f32 *Q, i32 nq, i32 R, i32 np, i32 *rows, i32 *cnt, const u64 *dead)
{
    TqMany m;
    i32 i, c, q;
    size_t bb = (size_t)t->vd * 16;
    f32 *rot = (f32 *)malloc((size_t)t->dpad * sizeof(f32)), *tmp = (f32 *)malloc((size_t)t->dpad * sizeof(f32));
    f32 *lf = (f32 *)malloc((size_t)t->vd * 16 * sizeof(f32));
    u8 *scr = np ? (u8 *)malloc(tq_probe_scr_(t)) : (u8 *)0;
    u64 *buf = (u64 *)malloc((size_t)nq * 2 * (size_t)R * sizeof(u64)), *mt = (u64 *)malloc((size_t)R * sizeof(u64));
    memset(&m, 0, sizeof(m));
    m.t = t; m.Q = Q; m.nq = nq; m.R = R; m.np = np; m.dead = dead; m.lutb = t->vd * 16 + 32;
    m.luts = (u8 *)malloc((size_t)nq * (size_t)m.lutb);
    m.res = (TqRes *)malloc((size_t)nq * sizeof(TqRes));
    for (i = 0; i < nq; ++i) tq_res_init_(&m.res[i], buf + (size_t)i * 2 * (size_t)R, R);
    if (np) m.probes = (i32 *)malloc((size_t)nq * (size_t)np * sizeof(i32));
    for (q = 0; q < nq; ++q) tq_lut_(t, Q + (size_t)q * t->dim, rot, tmp, lf, m.luts + (size_t)q * m.lutb);
#if TQ_VNNI
    if (np && t->c8) {                                           /* probes, 8 queries per pass over the int8 centroids */
        i32 ng16 = (t->nc + 15) / 16 * 16, d4 = t->c8d4, r;
        u32 *qws = (u32 *)malloc((size_t)nq * (size_t)d4 * sizeof(u32)), *qp[8];
        i32 *s8 = (i32 *)malloc((size_t)8 * (size_t)ng16 * sizeof(i32)), *sp[8];
        u64 *key = (u64 *)(void *)scr, *ktmp = key + ng16;
        i32 *cand = (i32 *)(void *)(ktmp + ng16) + ng16;
        for (q = 0; q < nq; ++q) tq_q8_(t, Q + (size_t)q * t->dim, qws + (size_t)q * d4);
        for (r = 0; r < 8; ++r) sp[r] = s8 + (size_t)r * ng16;
        for (q = 0; q < nq; q += 8) {
            i32 g = nq - q < 8 ? nq - q : 8;
            if (g == 8) { for (r = 0; r < 8; ++r) qp[r] = qws + (size_t)(q + r) * d4; tq_score8x8_(t, qp, sp); }
            else for (r = 0; r < g; ++r) tq_score8_(t, qws + (size_t)(q + r) * d4, sp[r]);
            for (r = 0; r < g; ++r)                              /* only the closest has to come first: clusters are */
                tq_pick_(key, tq_cands_(t, Q + (size_t)(q + r) * t->dim, np, sp[r], key, cand), np, ktmp,   /* scanned in order */
                         m.probes + (size_t)(q + r) * np, 0);
        }
        free(qws); free(s8);
    } else
#endif
    for (q = 0; q < nq && np; ++q) tq_probe_(t, Q + (size_t)q * t->dim, np, scr, m.probes + (size_t)q * np);
    for (q = 0; q < nq && np; ++q) {                             /* each query's closest cluster first: a threshold early */
        i32 b2;
        tq_prune_(t, Q + (size_t)q * t->dim, m.probes + (size_t)q * np, np, 0);   /* (never drops the first) */
        c = m.probes[(size_t)q * np];
        for (b2 = t->cstart[c]; b2 < t->cstart[c + 1]; ++b2) tq_many_push_(&m, &m.res[q], t->codes + (size_t)b2 * bb, b2, m.luts + (size_t)q * m.lutb);
        for (b2 = t->xhead ? t->xhead[c] : -1; b2 >= 0; b2 = t->xnext[b2])
            tq_many_push_(&m, &m.res[q], t->xcodes + (size_t)b2 * bb, t->nblk + b2, m.luts + (size_t)q * m.lutb);
        tq_res_cut_(&m.res[q]);
    }
    if (np) {                                                    /* cluster -> the queries that still have to scan it */
        i32 nx;
        m.cq0 = (i32 *)calloc((size_t)t->nc + 2, sizeof(i32));
        m.cq = (i32 *)malloc((size_t)nq * (size_t)np * sizeof(i32));
        for (i = 0; i < nq * np; ++i) if (i % np && m.probes[i] >= 0) ++m.cq0[m.probes[i] + 2];
        for (c = 0; c < t->nc; ++c) m.cq0[c + 2] += m.cq0[c + 1];
        for (i = 0; i < nq * np; ++i) if (i % np && m.probes[i] >= 0) m.cq[m.cq0[m.probes[i] + 1]++] = i / np;
        for (c = 0; c < t->nc; c = nx) {                         /* cluster by cluster: its blocks x its queries */
            i32 b2, x;
            for (nx = c + 1; nx < t->nc && m.cq0[nx] == m.cq0[nx + 1]; ++nx) {}
            if (m.cq0[c] == m.cq0[c + 1]) continue;
            for (b2 = t->cstart[c]; b2 < t->cstart[c + 1]; ++b2) {
                const u8 *blk = t->codes + (size_t)b2 * bb;
                i32 pf = b2 + 2 - t->cstart[c + 1];                  /* two blocks ahead, into the next probed cluster */
                if (pf < 0) tq_prefetch_blk_(blk + 2 * bb, (i32)bb);
                else if (nx < t->nc && t->cstart[nx] + pf < t->cstart[nx + 1]) tq_prefetch_blk_(t->codes + (size_t)(t->cstart[nx] + pf) * bb, (i32)bb);
                for (x = m.cq0[c]; x < m.cq0[c + 1]; ++x) { q = m.cq[x]; tq_many_push_(&m, &m.res[q], blk, b2, m.luts + (size_t)q * m.lutb); }
            }
            for (b2 = t->xhead ? t->xhead[c] : -1; b2 >= 0; b2 = t->xnext[b2]) {   /* rows added since the build */
                const u8 *blk = t->xcodes + (size_t)b2 * bb;
                if (t->xnext[b2] >= 0) tq_prefetch_blk_(t->xcodes + (size_t)t->xnext[b2] * bb, (i32)bb);
                for (x = m.cq0[c]; x < m.cq0[c + 1]; ++x) { q = m.cq[x]; tq_many_push_(&m, &m.res[q], blk, t->nblk + b2, m.luts + (size_t)q * m.lutb); }
            }
        }
    } else {                                                     /* flat: tiles of blocks x groups of queries */
        i32 b0, q0, b2, nbt = t->nblk + t->xn;
        for (b0 = 0; b0 < nbt; b0 += TQ_MANY_BG)
            for (q0 = 0; q0 < nq; q0 += TQ_MANY_QG)
                for (b2 = b0; b2 < nbt && b2 < b0 + TQ_MANY_BG; ++b2)
                    for (q = q0; q < nq && q < q0 + TQ_MANY_QG; ++q)
                        tq_many_push_(&m, &m.res[q], tq_bptr_(t, b2), b2, m.luts + (size_t)q * m.lutb);
    }
    for (q = 0; q < nq; ++q) cnt[q] = tq_res_rows_(&m.res[q], rows + (size_t)q * R, mt);   /* best first */
    free(m.res); free(m.luts); free(m.probes); free(m.cq0); free(m.cq);
    free(buf); free(mt); free(rot); free(tmp); free(lf); free(scr);
}

typedef struct { const Tq *t; const f32 *Q; i32 R, np; i32 *rows, *cnt; const u64 *dead; } TqManyJob;

static void tq_many_range_(void *p, i64 a, i64 b, i32 tid)
{
    TqManyJob *j = (TqManyJob *)p;
    i64 q;
    (void)tid;
    for (q = a; q < b; q += TQ_MANY_SUB) {
        i32 m = b - q < TQ_MANY_SUB ? (i32)(b - q) : TQ_MANY_SUB;
        tq_many_one_(j->t, j->Q + (size_t)q * j->t->dim, m, j->R, j->np, j->rows + (size_t)q * j->R, j->cnt + q, j->dead);
    }
}

/* tq_search for nq queries (no meta filter): rows is nq x R, cnt[q] how many rows query q got; each thread batches
 * its share of the queries. The rows found are the same as tq_search's up to ties in the quantised score. */
void tq_search_many(const Tq *t, const f32 *Q, i32 nq, i32 R, i32 *rows, i32 *cnt, const u64 *dead, i32 nthreads, i32 probe)
{
    TqManyJob j;
    i32 i;
    if (t->n == 0 || nq <= 0 || R <= 0) { for (i = 0; i < nq; ++i) cnt[i] = 0; return; }
    j.t = t; j.Q = Q; j.R = R; j.rows = rows; j.cnt = cnt; j.dead = dead;
    j.np = t->nc > 1 ? (probe > 0 ? probe : t->nprobe) : 0;
    if (j.np > t->nc) j.np = t->nc;
    tq_par2_(tq_many_range_, &j, nq, nthreads);
}
