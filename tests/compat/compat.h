/* compat.h - the values of the golden stores (tests/compat): every one a function of the row's id */
#define CDIM 16
#define CSAVED 300                   /* rows 1..300 saved in the file (7, 57, 107 ... deleted before the save) */
#define CALL 340                     /* 301..340 in the journal only; there too: 12 deleted, 13 replaced by 1000's vector */

static void vec_(vi_i64 id, float *v)
{
    unsigned x = (unsigned)id * 2654435761u + 12345u;
    int i;
    for (i = 0; i < CDIM; ++i) { x = x * 1103515245u + 12345u; v[i] = (float)((int)(x >> 16 & 0x7fff) - 16384) / 4096.0f; }
}
static void text_(vi_i64 id, char *t) { sprintf(t, "the text of row %ld, kept since the format of its file", (long)id); }
static vi_i64 tag_(vi_i64 id) { return (vi_i64)(((unsigned)id * 40503u) & 0xffffu) << 8; }   /* bits 8-23: labels own 0-2 */
static vi_i64 num_(vi_i64 id) { return id * 1000 - 77; }
static vi_i64 ns_(vi_i64 id) { return id % 4 == 0 ? 0 : (id % 4 == 1 ? 11 : -5); }
