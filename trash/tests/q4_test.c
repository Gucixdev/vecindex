/* q4_test.c - the self-test of src/q4.c; ./run builds and runs it. */
#define ARENA_ALLOC_IMPLEMENTATION
#include "q4.c"
#define PRINT_IMPLEMENTATION
#include "print.h"

static u32 q4d_rng_(u32 *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static f32 q4d_f_(u32 *s) { return (f32)q4d_rng_(s) / (f32)0xffffffffu - 0.5f; }

static f32 *q4d_corpus_(Arena *a, i32 n, i32 dim, u32 seed)
{
    f32 *out = (f32 *)arena_alloc(a, (size_t)n * dim * sizeof(f32));
    u32 st = seed | 1u;
    i32 i;
    for (i = 0; i < n * dim; ++i) out[i] = q4d_f_(&st) * 2.0f;
    return out;
}

int main(void)
{
    static Arena arena;
    Arena *a = &arena;
    i32 dim = 64, n = 500;
    f32 *vecs;
    Q4Index *ix;
    i64 ids[16]; f32 sc[16];
    i32 i, hit1 = 0, overlap = 0, qn = 0, exq;

    vecs = q4d_corpus_(a, n, dim, 42u);
    ix = q4_build(a, vecs, n, dim);

    for (i = 0; i < n; ++i) {
        i32 h = q4_search(ix, vecs + (size_t)i * dim, 1, ids, sc);
        if (h > 0 && ids[0] == (i64)i) ++hit1;
    }
    print_cstr("q4  dim="); print_i64(dim);
    print_cstr(" n=");      print_i64(n);
    print_cstr("  code_bytes/vec="); print_i64(ix->code_bytes);
    print_cstr(" (+8 rnorm+qn2)  self-retrieval@1=");
    print_i64((i64)(100.0 * hit1 / n)); print_cstr("%\n");

    for (exq = 0; exq < n; exq += n / 20) {
        const f32 *q = vecs + (size_t)exq * dim;
        i64 approx[10]; f32 as[10];
        i32 na = q4_search(ix, q, 10, approx, as), r, j2;
        f32 best[10]; i64 bid[10]; i32 bh = 0;
        for (r = 0; r < n; ++r) {
            const f32 *row = vecs + (size_t)r * dim;
            f32 dot = 0.0f, qq = 0.0f, rr = 0.0f, cs; i32 d;
            for (d = 0; d < dim; ++d) { dot += q[d]*row[d]; qq += q[d]*q[d]; rr += row[d]*row[d]; }
            cs = (qq > 0.0f && rr > 0.0f) ? dot / (f32)(q4_sqrt_(qq) * q4_sqrt_(rr)) : 0.0f;
            { i32 hh = bh;
              if (hh < 10) { bid[hh]=r; best[hh]=cs; while(hh>0&&best[hh-1]<best[hh]){f32 t=best[hh-1];i64 ti=bid[hh-1];best[hh-1]=best[hh];bid[hh-1]=bid[hh];best[hh]=t;bid[hh]=ti;--hh;} ++bh; }
              else if (cs > best[9]) { hh=9; bid[hh]=r; best[hh]=cs; while(hh>0&&best[hh-1]<best[hh]){f32 t=best[hh-1];i64 ti=bid[hh-1];best[hh-1]=best[hh];bid[hh-1]=bid[hh];best[hh]=t;bid[hh]=ti;--hh;} } }
        }
        for (r = 0; r < na; ++r)
            for (j2 = 0; j2 < 10; ++j2)
                if (approx[r] == bid[j2]) { ++overlap; break; }
        ++qn;
    }
    print_cstr("     recall@10 vs exact f32 = ");
    print_i64((i64)(100.0 * overlap / (qn * 10))); print_cstr("%\n");
    print_flush();

    arena_free(a);
    return (hit1 * 100 / n >= 95) ? 0 : 1;
}
