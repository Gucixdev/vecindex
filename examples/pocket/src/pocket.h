/* pocket: the phone's own files, sorted on the phone - nothing leaves it. Part one, the cleanup: duplicate and
 * similar photos found with vecindex, and the one to keep chosen. Next parts (README): a document scanner, offline
 * transcription. The decoding of images is the platform's (Android: BitmapFactory into RGBA, then JNI; the desktop
 * CLI reads PNM); everything here takes pixels and a vecindex store, and allocates nothing.
 *
 *   pk_photo     pixels -> the descriptor (the vector stored) and the sharpness
 *   pk_groups    the store -> which photos belong together (union-find over the near neighbours)
 *   pk_keep      a group -> the photo to keep (the most pixels, then the sharpest)
 *
 * A photo's id in the store is its number, 0 up: pk_groups takes n, the highest id + 1. */
#ifndef POCKET_H
#define POCKET_H

#include "vecindex.h"

#define PK_DIM 96          /* 63 luminance DCT coefficients, 32 chroma means, 1 zero: the store's dim */

/* How alike two photos must be (the cosine of their descriptors) to be one group. */
#define PK_SAME    0.97f   /* the same picture: resized, re-encoded, brightened, a copy sent back by a messenger */
#define PK_SIMILAR 0.90f   /* the same scene: a burst, a small crop, a slight blur */

typedef struct {
    float vec[PK_DIM];     /* the descriptor, unit length */
    float sharp;           /* variance of the Laplacian over variance of the image: bigger is sharper, contrast cancels */
    int w, h;
} pk_info;

/* rgba: h rows of stride bytes, 4 bytes a pixel (R, G, B, A; A unused). 0, or -1 for a size under 2 x 2. */
int pk_photo(const unsigned char *rgba, int w, int h, int stride, pk_info *out);

/* Fills parent[n] so that pk_root(parent, i) == pk_root(parent, j) when photos i and j are one group: every pair
 * with a cosine of thr or more among each photo's k nearest (k 2..64, 16 a good start) is joined, and groups chain
 * (a ~ b ~ c). An id without a row stays alone. The root of a group is its smallest id. 0, or -1 (vi_error). */
int pk_groups(vi_db *db, float thr, int k, vi_i64 n, int *parent);
int pk_root(int *parent, int i);

/* Of m photos (their sizes, sharpness, file bytes and descriptors, in one order), the index of the one to keep.
 * Copies of one picture (cosine PK_COPY or more: a re-encode, a resize, a brightened copy) are judged by pixels, then
 * bytes - the least compressed - because a lossy re-encode's block edges look sharp; the best copy of each picture
 * then competes with the other shots by sharpness, among those with at least 90% of the most pixels. bytes NULL:
 * unknown; vecs NULL: every photo its own shot. */
#define PK_COPY 0.99f
int pk_keep(int m, const int *w, const int *h, const float *sharp, const vi_i64 *bytes, const float *vecs);

#endif
