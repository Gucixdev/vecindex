/* pocket's self-test: synthetic photos (gradients, shapes, texture, noise) and their copies - half size, brighter,
 * posterized (as a lossy re-encode bands colours), blurred, cropped - must find each other and nothing else; the
 * photo kept is the full-size sharp one; the groups survive a reopen. Prints the margins it measured. */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "../src/pocket.h"

#define W 512
#define H 384
#define SCENES 100
#define VARS 6
static const char *VNAME[VARS] = {"original", "half", "brighter", "posterized", "blurred", "cropped"};

static int fails, checks;
static void ok(int c, const char *what)
{
    ++checks;
    if (!c) { ++fails; printf("  FAIL %s\n", what); }
    else printf("  ok   %s\n", what);
}

static unsigned long rs;
static unsigned long rnd(void)
{
    rs ^= rs << 13; rs &= 0xffffffffUL; rs ^= rs >> 17; rs ^= rs << 5; rs &= 0xffffffffUL;
    return rs;
}
static int ri(int n) { return (int)(rnd() % (unsigned long)n); }
static unsigned char clamp(double v) { return (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v + 0.5); }

static unsigned char base[H * W * 4], var[H * W * 4], tmp[H * W * 4], big[2000 * 3000 * 4];

/* A scene: a gradient between two colours, shapes, a striped patch, noise. */
static void scene(unsigned long seed, unsigned char *img, int w, int h)
{
    int c0[3], c1[3], i, x, y, n;
    double ang, ca, sa;
    rs = seed * 2654435761UL + 12345UL;
    if (!rs) rs = 1;
    for (i = 0; i < 3; ++i) { c0[i] = ri(256); c1[i] = ri(256); }
    ang = ri(628) / 100.0; ca = cos(ang); sa = sin(ang);
    for (y = 0; y < h; ++y)
        for (x = 0; x < w; ++x) {
            double t = ((x - w / 2) * ca + (y - h / 2) * sa) / (double)w + 0.5;
            unsigned char *p = img + 4L * ((long)y * w + x);
            if (t < 0) t = 0;
            if (t > 1) t = 1;
            for (i = 0; i < 3; ++i) p[i] = clamp(c0[i] + (c1[i] - c0[i]) * t);
            p[3] = 255;
        }
    n = 5 + ri(8);
    while (n--) {
        int cx = ri(w), cy = ri(h), rx = 10 + ri(w / 4), ry = 10 + ri(h / 4), ell = ri(2), col[3];
        for (i = 0; i < 3; ++i) col[i] = ri(256);
        for (y = cy - ry; y <= cy + ry; ++y)
            for (x = cx - rx; x <= cx + rx; ++x) {
                double dx = (x - cx) / (double)rx, dy = (y - cy) / (double)ry;
                unsigned char *p;
                if (x < 0 || y < 0 || x >= w || y >= h || (ell && dx * dx + dy * dy > 1)) continue;
                p = img + 4L * ((long)y * w + x);
                for (i = 0; i < 3; ++i) p[i] = (unsigned char)col[i];
            }
    }
    {   /* stripes: fine detail, what blur takes away */
        int px = ri(w / 2), py = ri(h / 2), pw = w / 4 + ri(w / 4), ph = h / 4 + ri(h / 4);
        double f = 0.5 + ri(100) / 100.0;
        for (y = py; y < py + ph && y < h; ++y)
            for (x = px; x < px + pw && x < w; ++x) {
                unsigned char *p = img + 4L * ((long)y * w + x);
                double s = 40 * sin((x + y * 0.7) * f);
                for (i = 0; i < 3; ++i) p[i] = clamp(p[i] + s);
            }
    }
    for (i = 0; i < w * h; ++i) {
        int d = ri(7) - 3, k;
        for (k = 0; k < 3; ++k) img[4 * i + k] = clamp(img[4 * i + k] + d);
    }
}

/* The variants of base (W x H) into var; their size in *w, *h. */
static void variant(int v, int *w, int *h)
{
    int x, y, i, k, pass;
    *w = W; *h = H;
    switch (v) {
    case 0: memcpy(var, base, sizeof base); break;
    case 1:
        *w = W / 2; *h = H / 2;
        for (y = 0; y < H / 2; ++y)
            for (x = 0; x < W / 2; ++x)
                for (k = 0; k < 4; ++k) {
                    const unsigned char *p = base + 4L * ((2L * y) * W + 2 * x) + k;
                    var[4L * ((long)y * (W / 2) + x) + k] = (unsigned char)((p[0] + p[4] + p[4 * W] + p[4 * W + 4] + 2) / 4);
                }
        break;
    case 2:
        for (i = 0; i < W * H * 4; ++i) var[i] = (i & 3) == 3 ? 255 : clamp(base[i] * 0.9 + 30);
        break;
    case 3:
        for (i = 0; i < W * H * 4; ++i) var[i] = (unsigned char)((i & 3) == 3 ? 255 : (base[i] & 0xf0) | 8);
        break;
    case 4:     /* two passes of a 5 x 5 box blur */
        memcpy(var, base, sizeof base);
        for (pass = 0; pass < 2; ++pass) {
            for (y = 0; y < H; ++y)
                for (x = 0; x < W; ++x)
                    for (k = 0; k < 3; ++k) {
                        int s = 0, n = 0, d;
                        for (d = -2; d <= 2; ++d)
                            if (x + d >= 0 && x + d < W) { s += var[4L * ((long)y * W + x + d) + k]; ++n; }
                        tmp[4L * ((long)y * W + x) + k] = (unsigned char)(s / n);
                    }
            for (y = 0; y < H; ++y)
                for (x = 0; x < W; ++x)
                    for (k = 0; k < 3; ++k) {
                        int s = 0, n = 0, d;
                        for (d = -2; d <= 2; ++d)
                            if (y + d >= 0 && y + d < H) { s += tmp[4L * ((long)(y + d) * W + x) + k]; ++n; }
                        var[4L * ((long)y * W + x) + k] = (unsigned char)(s / n);
                    }
        }
        break;
    case 5: {   /* 4% off every side */
        int mx = W / 25, my = H / 25;
        *w = W - 2 * mx; *h = H - 2 * my;
        for (y = 0; y < *h; ++y) memcpy(var + 4L * y * *w, base + 4L * ((long)(y + my) * W + mx), 4L * *w);
        break;
    }
    }
}

static pk_info info[SCENES * VARS];

static float cosv(const float *a, const float *b)
{
    double s = 0;
    int i;
    for (i = 0; i < PK_DIM; ++i) s += (double)a[i] * b[i];
    return (float)s;
}

static int parent[SCENES * VARS];

/* Each scene's photos with variants in mask in one group, and no group across two scenes. */
static int groups_right(int mask, int *ngroups)
{
    int s, v, good = 1, i;
    for (s = 0; s < SCENES; ++s) {
        int r = pk_root(parent, s * VARS);
        for (v = 1; v < VARS; ++v)
            if ((mask >> v & 1) && pk_root(parent, s * VARS + v) != r) good = 0;
    }
    for (i = 0; i < SCENES * VARS; ++i)
        if (pk_root(parent, i) / VARS != i / VARS) good = 0;    /* the root is the group's smallest id */
    *ngroups = 0;
    for (i = 0; i < SCENES * VARS; ++i) *ngroups += pk_root(parent, i) == i;
    return good;
}

int main(void)
{
    vi_db *db;
    int s, v, i, ng, n = SCENES * VARS;
    float lo[VARS], cross = -1;
    char msg[256];
    const char *path = "/tmp/pocket_test.vxi";
    clock_t t0;
    double secs;

    printf("pocket:\n");
    secs = 0;
    for (s = 0; s < SCENES; ++s) {
        scene((unsigned long)s + 1, base, W, H);
        for (v = 0; v < VARS; ++v) {
            int w, h;
            variant(v, &w, &h);
            t0 = clock();
            i = pk_photo(var, w, h, 4 * w, &info[s * VARS + v]);
            secs += (double)(clock() - t0) / CLOCKS_PER_SEC;
            if (i != 0) { printf("  FAIL pk_photo\n"); return 1; }
        }
    }
    for (v = 0; v < VARS; ++v) lo[v] = 2;
    for (s = 0; s < SCENES; ++s)
        for (v = 1; v < VARS; ++v) {
            float c = cosv(info[s * VARS].vec, info[s * VARS + v].vec);
            if (c < lo[v]) lo[v] = c;
        }
    for (s = 0; s < SCENES; ++s)
        for (i = s + 1; i < SCENES; ++i) {
            float c = cosv(info[s * VARS].vec, info[i * VARS].vec);
            if (c > cross) cross = c;
        }
    for (v = 1; v < VARS; ++v) {
        sprintf(msg, "%-10s: lowest cosine to its original %.4f (>= %.2f)", VNAME[v], lo[v],
                v <= 3 ? PK_SAME : PK_SIMILAR);
        ok(lo[v] >= (v <= 3 ? PK_SAME : PK_SIMILAR), msg);
    }
    sprintf(msg, "two different scenes: highest cosine %.4f (< %.2f)", cross, PK_SIMILAR);
    ok(cross < PK_SIMILAR, msg);

    remove(path);
    {
        char j[64];
        sprintf(j, "%s.j", path);
        remove(j);
    }
    db = vi_open(path, PK_DIM, 0);
    ok(db != NULL, "a store of PK_DIM dimensions");
    if (!db) return 1;
    for (i = 0; i < n; ++i) {
        vi_add(db, i, info[i].vec);
        vi_set_int(db, i, "w", info[i].w);
        vi_set_int(db, i, "h", info[i].h);
        vi_set_float(db, i, "sharp", info[i].sharp);
    }
    ok(pk_groups(db, PK_SIMILAR, 16, n, parent) == 0 && groups_right(0x3e, &ng) && ng == SCENES,
       "similar (0.90): every scene one group of 6, none across scenes, 100 groups");
    ok(pk_groups(db, PK_SAME, 16, n, parent) == 0 && groups_right(0x0e, &ng),
       "same (0.97): original, half, brighter, posterized together; none across scenes");
    sprintf(msg, "same (0.97): %d groups (blurred and cropped may stand alone)", ng);
    ok(ng >= SCENES && ng <= 3 * SCENES, msg);

    {   /* the photo kept: never the half size, the blurred or the cropped one */
        int good = 1;
        pk_groups(db, PK_SIMILAR, 16, n, parent);
        for (s = 0; s < SCENES; ++s) {
            int w[VARS], h[VARS], k;
            float sh[VARS];
            for (v = 0; v < VARS; ++v) { w[v] = info[s * VARS + v].w; h[v] = info[s * VARS + v].h; sh[v] = info[s * VARS + v].sharp; }
            k = pk_keep(VARS, w, h, sh, NULL, NULL);
            if (k == 1 || k == 4 || k == 5) good = 0;
        }
        ok(good, "keep: the full-size sharp photo (original or its brightened copy), never half, blurred, cropped");
        {
            int w[2] = {W, W}, h[2] = {H, H};
            float sh[2];
            sh[0] = info[4].sharp; sh[1] = info[0].sharp;
            sprintf(msg, "keep: original over blurred (sharpness %.3f vs %.3f)", info[0].sharp, info[4].sharp);
            ok(pk_keep(2, w, h, sh, NULL, NULL) == 1 && info[0].sharp > 2 * info[4].sharp, msg);
        }
        {   /* copies of one picture: the biggest file wins over a re-encode that looks sharper; another shot of
             * the scene competes by sharpness */
            static float vv[4 * PK_DIM];
            int w[4] = {W, W, W, W}, h[4] = {H, H, H, H}, d;
            float sh[4];
            vi_i64 by[4];
            for (d = 0; d < PK_DIM; ++d) {
                vv[d] = vv[PK_DIM + d] = info[0].vec[d];        /* 0 and 1: one picture */
                vv[2 * PK_DIM + d] = vv[3 * PK_DIM + d] = info[5].vec[d];   /* 2 and 3: another shot */
            }
            sh[0] = 1.0f; sh[1] = 4.0f; sh[2] = 2.0f; sh[3] = 0.5f;
            by[0] = 900000; by[1] = 300000; by[2] = 800000; by[3] = 500000;
            ok(pk_keep(4, w, h, sh, by, vv) == 2,
               "keep: of each picture the biggest file (not the re-encode that looks 4x sharper), then the sharper shot");
            sh[2] = 0.8f;
            ok(pk_keep(4, w, h, sh, by, vv) == 0, "keep: the other way round when the other shot is the blurrier");
        }
    }

    vi_delete(db, 2);
    ok(pk_groups(db, PK_SIMILAR, 16, n, parent) == 0 && pk_root(parent, 2) == 2 && pk_root(parent, 3) == 0,
       "a deleted photo stands alone, the rest of its group stays");
    vi_add(db, 2, info[2].vec);
    vi_close(db);
    db = vi_open(path, 0, 0);
    ok(db && pk_groups(db, PK_SIMILAR, 16, n, parent) == 0 && groups_right(0x3e, &ng) && ng == SCENES,
       "reopened from the file: the same 100 groups");
    if (db) {
        double sh = 0;
        vi_i64 w = 0;
        ok(vi_get_int(db, 5, "w", &w) == 0 && w == info[5].w && vi_get_float(db, 5, "sharp", &sh) == 0
           && fabs(sh - info[5].sharp) < 1e-4, "fields w and sharp kept");
        vi_close(db);
    }

    {   /* edges */
        pk_info e;
        unsigned char gray[16 * 16 * 4];
        memset(gray, 128, sizeof gray);
        ok(pk_photo(gray, 16, 16, 64, &e) == 0 && fabs(e.vec[0] - 1) < 1e-6 && e.sharp < 1e-3,
           "a flat gray picture: a unit vector, sharpness 0");
        ok(pk_photo(gray, 2, 2, 8, &e) == 0, "2 x 2 taken");
        ok(pk_photo(gray, 1, 16, 4, &e) == -1 && pk_photo(gray, 16, 16, 60, &e) == -1 && pk_photo(NULL, 16, 16, 64, &e) == -1,
           "1 wide, a short stride, NULL: refused");
    }
    sprintf(msg, "%d photos of %dx%d (and smaller) described in %.2f s: %.0f a second", n, W, H, secs, n / secs);
    ok(1, msg);
    {
        pk_info e;
        scene(7, big, 3000, 2000);
        t0 = clock();
        pk_photo(big, 3000, 2000, 4 * 3000, &e);
        sprintf(msg, "a 6-megapixel photo: %.1f ms", 1000.0 * (clock() - t0) / CLOCKS_PER_SEC);
        ok(1, msg);
    }
    remove(path);
    {
        char j[64];
        sprintf(j, "%s.j", path);
        remove(j);
    }
    printf(fails ? "pocket: %d of %d checks FAILED\n" : "pocket: OK (%d checks)\n", fails ? fails : checks, checks);
    return fails != 0;
}
