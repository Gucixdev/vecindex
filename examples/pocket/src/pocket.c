/* pocket: descriptors, groups and the photo to keep (pocket.h). C89, no allocation, no state: safe from any thread. */
#include <math.h>
#include <string.h>

#include "pocket.h"

#define PK_S 32            /* the luminance thumbnail: PK_S x PK_S, squashed to a square as perceptual hashes do */
#define PK_F 8             /* the DCT's lowest PK_F x PK_F frequencies, the DC left out */
#define PK_G 4             /* the chroma grid: PK_G x PK_G cells, two channels */
#define PK_PI 3.14159265358979f
#define PK_CW 0.5         /* the chroma part's length at most; the structure's is 1 */

/* The luminance of pixel (x, y), 0..255, the BT.601 weights in fixed point. */
static int pk_y_(const unsigned char *p)
{
    return (77 * p[0] + 150 * p[1] + 29 * p[2]) >> 8;
}

/* Box averages of R, G, B over a PK_S x PK_S grid: each cell the mean of the pixels that fall in it (at least one, so
 * an image smaller than the grid repeats pixels). */
static void pk_thumb_(const unsigned char *rgba, int w, int h, int stride, float *r, float *g, float *b)
{
    int cy, cx;
    for (cy = 0; cy < PK_S; ++cy) {
        int y0 = (int)((long)cy * h / PK_S), y1 = (int)((long)(cy + 1) * h / PK_S);
        if (y1 <= y0) y1 = y0 + 1;
        for (cx = 0; cx < PK_S; ++cx) {
            int x0 = (int)((long)cx * w / PK_S), x1 = (int)((long)(cx + 1) * w / PK_S), x, y;
            unsigned long sr = 0, sg = 0, sb = 0, n;
            if (x1 <= x0) x1 = x0 + 1;
            for (y = y0; y < y1; ++y) {
                const unsigned char *p = rgba + (long)y * stride + 4L * x0;
                for (x = x0; x < x1; ++x, p += 4) { sr += p[0]; sg += p[1]; sb += p[2]; }
            }
            n = (unsigned long)(y1 - y0) * (unsigned long)(x1 - x0);
            r[cy * PK_S + cx] = (float)sr / (float)n;
            g[cy * PK_S + cx] = (float)sg / (float)n;
            b[cy * PK_S + cx] = (float)sb / (float)n;
        }
    }
}

/* The luminance of the 2 x 2 block at (x, y), 0..1020: half resolution, where a lossy re-encode's block edges and the
 * sensor's noise have faded and a blur of a few pixels has not. */
static int pk_y2_(const unsigned char *p, int stride)
{
    return pk_y_(p) + pk_y_(p + 4) + pk_y_(p + stride) + pk_y_(p + stride + 4);
}

/* Variance of the 4-neighbour Laplacian of the half-resolution luminance over the variance of that luminance (every
 * second block and row of blocks past 4 megapixels). Blur lowers it; brightness and contrast cancel. */
static float pk_sharp_(const unsigned char *rgba, int w, int h, int stride)
{
    int step = (long)w * h > 4000000L ? 4 : 2, x, y;
    double sl = 0, sl2 = 0, sy = 0, sy2 = 0, n = 0, vl, vy;
    for (y = 2; y + 3 < h; y += step) {
        const unsigned char *row = rgba + (long)y * stride;
        for (x = 2; x + 3 < w; x += step) {
            const unsigned char *p = row + 4L * x;
            int c = pk_y2_(p, stride);
            int l = pk_y2_(p - 8, stride) + pk_y2_(p + 8, stride) + pk_y2_(p - 2L * stride, stride)
                  + pk_y2_(p + 2L * stride, stride) - 4 * c;
            sl += l; sl2 += (double)l * l; sy += c; sy2 += (double)c * c; n += 1;
        }
    }
    if (n < 1) return 0;
    vl = sl2 / n - (sl / n) * (sl / n);
    vy = sy2 / n - (sy / n) * (sy / n);
    return (float)(vl / (vy + 16.0));
}

int pk_photo(const unsigned char *rgba, int w, int h, int stride, pk_info *out)
{
    float r[PK_S * PK_S], g[PK_S * PK_S], b[PK_S * PK_S], lum[PK_S * PK_S];
    float c[PK_F][PK_S], tmp[PK_S][PK_F];
    double nl = 0, nall = 0;
    int i, u, v, x, y, o = 0;
    if (!rgba || !out || w < 2 || h < 2 || stride < 4 * w) return -1;
    memset(out, 0, sizeof *out);
    out->w = w;
    out->h = h;
    pk_thumb_(rgba, w, h, stride, r, g, b);
    {   /* the mean taken out first: a flat picture gives exact zeros, not the DCT's rounding */
        double m = 0;
        for (i = 0; i < PK_S * PK_S; ++i) { lum[i] = 0.299f * r[i] + 0.587f * g[i] + 0.114f * b[i]; m += lum[i]; }
        m /= PK_S * PK_S;
        for (i = 0; i < PK_S * PK_S; ++i) lum[i] -= (float)m;
    }
    /* the separable DCT-II, only the PK_F lowest frequencies each way */
    for (u = 0; u < PK_F; ++u)
        for (x = 0; x < PK_S; ++x) c[u][x] = (float)cos((2 * x + 1) * u * PK_PI / (2 * PK_S));
    for (y = 0; y < PK_S; ++y)
        for (u = 0; u < PK_F; ++u) {
            float s = 0;
            for (x = 0; x < PK_S; ++x) s += lum[y * PK_S + x] * c[u][x];
            tmp[y][u] = s;
        }
    for (v = 0; v < PK_F; ++v)
        for (u = 0; u < PK_F; ++u) {
            float s = 0;
            if (!u && !v) continue;                 /* DC: the brightness, which a copy may change */
            for (y = 0; y < PK_S; ++y) s += tmp[y][u] * c[v][y];
            out->vec[o++] = s;
        }
    for (i = 0; i < o; ++i) nl += (double)out->vec[i] * out->vec[i];
    nl = sqrt(nl);             /* under 1e-2: no structure (a grey step of 1 gives hundreds) */
    for (i = 0; i < o; ++i) out->vec[i] = nl > 1e-2 ? (float)(out->vec[i] / nl) : 0.0f;
    /* chroma: Cb and Cr means over a PK_G x PK_G grid, their length capped at PK_CW: a gray picture is all
     * structure, and two pictures of one shape in other colours still stay apart */
    {
        int c0 = o;
        double nc = 0;
        for (y = 0; y < PK_G; ++y)
            for (x = 0; x < PK_G; ++x) {
                double cb = 0, cr = 0;
                int yy, xx, q = PK_S / PK_G;
                for (yy = y * q; yy < (y + 1) * q; ++yy)
                    for (xx = x * q; xx < (x + 1) * q; ++xx) {
                        int k = yy * PK_S + xx;
                        {
                            float yk = 0.299f * r[k] + 0.587f * g[k] + 0.114f * b[k];
                            cb += b[k] - yk;
                            cr += r[k] - yk;
                        }
                    }
                out->vec[o++] = (float)(cb / (q * q) / 255.0);
                out->vec[o++] = (float)(cr / (q * q) / 255.0);
            }
        for (i = c0; i < o; ++i) nc += (double)out->vec[i] * out->vec[i];
        nc = sqrt(nc);
        if (nc > PK_CW)
            for (i = c0; i < o; ++i) out->vec[i] = (float)(out->vec[i] * PK_CW / nc);
    }
    for (i = 0; i < PK_DIM; ++i) nall += (double)out->vec[i] * out->vec[i];
    nall = sqrt(nall);
    if (nall < 1e-4) { memset(out->vec, 0, sizeof out->vec); out->vec[0] = 1.0f; }            /* a flat gray image: any unit vector, all of them alike */
    else for (i = 0; i < PK_DIM; ++i) out->vec[i] = (float)(out->vec[i] / nall);
    out->sharp = pk_sharp_(rgba, w, h, stride);
    return 0;
}

int pk_root(int *parent, int i)
{
    while (parent[i] != i) {
        parent[i] = parent[parent[i]];              /* path halving */
        i = parent[i];
    }
    return i;
}

static void pk_join_(int *parent, int a, int b)
{
    a = pk_root(parent, a);
    b = pk_root(parent, b);
    if (a < b) parent[b] = a;                       /* the smaller id is the root */
    else if (b < a) parent[a] = b;
}

#define PK_BATCH 64
#define PK_KMAX 64

int pk_groups(vi_db *db, float thr, int k, vi_i64 n, int *parent)
{
    float q[PK_BATCH * PK_DIM], sc[PK_BATCH * PK_KMAX];
    vi_i64 have[PK_BATCH], ids[PK_BATCH * PK_KMAX], from = 0, i;
    if (!db || !parent || n < 0 || n > 0x7fffffffL || k < 2 || k > PK_KMAX || vi_dim(db) != PK_DIM) return -1;
    for (i = 0; i < n; ++i) parent[i] = (int)i;
    for (;;) {
        int got = vi_range(db, from, n - 1, have, PK_BATCH), j, m;
        if (got < 0) return -1;
        if (got == 0) break;
        for (j = 0; j < got; ++j)
            if (vi_get(db, have[j], q + (long)j * PK_DIM) != 0) return -1;
        if (vi_search_batch(db, q, got, k, ids, sc, 0) != 0) return -1;
        for (j = 0; j < got; ++j)
            for (m = 0; m < k; ++m) {
                vi_i64 o = ids[j * k + m];
                if (o < 0 || sc[j * k + m] < thr) break;    /* best first: the rest are further */
                if (o != have[j] && o < n) pk_join_(parent, (int)have[j], (int)o);
            }
        from = have[got - 1] + 1;
        if (got < PK_BATCH || from >= n) break;
    }
    return 0;
}

/* Is i a better copy of the same picture than j: more pixels by a tenth, else the bigger file, else the sharper. */
static int pk_better_copy_(int i, int j, const int *w, const int *h, const float *sharp, const vi_i64 *bytes)
{
    double pi = (double)w[i] * h[i], pj = (double)w[j] * h[j];
    if (pi > 1.1 * pj) return 1;
    if (pj > 1.1 * pi) return 0;
    if (bytes && bytes[i] != bytes[j]) return bytes[i] > bytes[j];
    return sharp[i] > sharp[j] || (sharp[i] == sharp[j] && i < j);
}

int pk_keep(int m, const int *w, const int *h, const float *sharp, const vi_i64 *bytes, const float *vecs)
{
    double most = 0;
    int i, j, best = -1;
    for (i = 0; i < m; ++i)
        if ((double)w[i] * h[i] > most) most = (double)w[i] * h[i];
    for (i = 0; i < m; ++i) {
        int rep = 1;
        if ((double)w[i] * h[i] < 0.9 * most) continue;
        if (vecs)                                   /* the best copy of its own picture, or not a candidate */
            for (j = 0; j < m && rep; ++j) {
                double c = 0;
                int d;
                if (j == i) continue;
                for (d = 0; d < PK_DIM; ++d) c += (double)vecs[(long)i * PK_DIM + d] * vecs[(long)j * PK_DIM + d];
                if (c >= PK_COPY && pk_better_copy_(j, i, w, h, sharp, bytes)) rep = 0;
            }
        if (!rep) continue;
        if (best < 0 || sharp[i] > sharp[best]) best = i;
    }
    return best < 0 ? 0 : best;
}
