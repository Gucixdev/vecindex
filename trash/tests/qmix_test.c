/* qmix_test.c - the self-test of src/qmix.c; ./run builds and runs it. */
#define ARENA_ALLOC_IMPLEMENTATION
#include "qmix.c"
#define PRINT_IMPLEMENTATION
#include "print.h"

static u32 qm_rng_(u32 *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static f32 qm_f_(u32 *s) { return (f32)qm_rng_(s) / (f32)0xffffffffu - 0.5f; }

/* corpus with a deliberately skewed per-dim variance: the first ~20% of
 * dims swing hard, the tail barely moves - the shape qmix is built for. */
static f32 *qm_corpus_(Arena *a, i32 n, i32 dim, u32 seed)
{
    f32 *o = (f32 *)arena_alloc(a, (size_t)n * dim * sizeof(f32));
    u32 st = seed | 1u; i32 i, d;
    for (i = 0; i < n; ++i)
        for (d = 0; d < dim; ++d) {
            f32 amp = (d < dim / 5) ? 3.0f : 0.15f;
            o[(size_t)i * dim + d] = qm_f_(&st) * amp;
        }
    return o;
}

static i32 qm_recall_(const QMixIndex *ix, const f32 *v, i32 n, i32 dim, i32 nq)
{
    i32 qi, hit = 0, tot = 0;
    for (qi = 0; qi < n; qi += (n / nq ? n / nq : 1)) {
        const f32 *q = v + (size_t)qi * dim;
        i64 ap[10], ex[10]; f32 as[10], bs[10]; i32 na, r, x, bh = 0;
        na = qmix_search(ix, q, 10, ap, as);
        for (r = 0; r < n; ++r) {
            const f32 *row = v + (size_t)r * dim;
            f32 dot = vs_dot(q, row, dim), qq = vs_dot(q, q, dim), rr = vs_dot(row, row, dim);
            f32 cs = (qq > 0.0f && rr > 0.0f)
                   ? dot / (f32)(qmix_sqrt_(qq) * qmix_sqrt_(rr)) : 0.0f;
            if (bh < 10) { i32 h = bh; ex[h] = r; bs[h] = cs;
                while (h > 0 && bs[h-1] < bs[h]) { f32 t = bs[h-1]; i64 ti = ex[h-1];
                    bs[h-1]=bs[h]; ex[h-1]=ex[h]; bs[h]=t; ex[h]=ti; --h; } ++bh; }
            else if (cs > bs[9]) { i32 h = 9; ex[h] = r; bs[h] = cs;
                while (h > 0 && bs[h-1] < bs[h]) { f32 t = bs[h-1]; i64 ti = ex[h-1];
                    bs[h-1]=bs[h]; ex[h-1]=ex[h]; bs[h]=t; ex[h]=ti; --h; } }
        }
        for (r = 0; r < na; ++r)
            for (x = 0; x < 10; ++x) if (ap[r] == ex[x]) { ++hit; break; }
        tot += 10;
    }
    return (i32)(100L * hit / (tot ? tot : 1));
}

int main(void)
{
    static Arena arena;
    Arena *a = &arena;
    i32 dim = 128, n = 2000, i, hit1 = 0;
    f32 *v = qm_corpus_(a, n, dim, 7u);
    QMixIndex *ix = qmix_build(a, v, n, dim);
    Q2Index *q2 = q2_build(a, v, n, dim);
    i64 id[4]; f32 sc[4];
    i32 rm, r2;

    for (i = 0; i < n; ++i) {
        i32 h = qmix_search(ix, v + (size_t)i * dim, 1, id, sc);
        if (h > 0 && id[0] == (i64)i) ++hit1;
    }
    print_cstr("qmix  dim="); print_i64(dim); print_cstr(" n="); print_i64(n);
    print_cstr("  split hi/mid/lo="); print_i64(ix->n_hi); print_cstr("/");
    print_i64(ix->n_mid); print_cstr("/"); print_i64(ix->n_lo);
    print_cstr("  code_bytes="); print_i64(ix->code_bytes);
    print_cstr(" (q2 uniform = "); print_i64(q2_code_bytes(dim)); print_cstr(")\n");
    print_cstr("  self-retrieval@1 = "); print_i64((i64)(100L * hit1 / n)); print_cstr("%\n");

    rm = qm_recall_(ix, v, n, dim, 60);
    {
        i32 qi, hit = 0, tot = 0;
        for (qi = 0; qi < n; qi += n / 60) {
            const f32 *q = v + (size_t)qi * dim;
            i64 ap[10], ex[10]; f32 as[10], bs[10]; i32 na, r, x, bh = 0;
            na = q2_search(q2, q, 10, ap, as);
            for (r = 0; r < n; ++r) {
                const f32 *row = v + (size_t)r * dim;
                f32 dot = vs_dot(q, row, dim), qq = vs_dot(q,q,dim), rr = vs_dot(row,row,dim);
                f32 cs = (qq>0.0f&&rr>0.0f)? dot/(f32)(qmix_sqrt_(qq)*qmix_sqrt_(rr)):0.0f;
                if (bh < 10) { i32 h=bh; ex[h]=r; bs[h]=cs;
                    while(h>0&&bs[h-1]<bs[h]){f32 t=bs[h-1];i64 ti=ex[h-1];bs[h-1]=bs[h];ex[h-1]=ex[h];bs[h]=t;ex[h]=ti;--h;} ++bh; }
                else if (cs>bs[9]) { i32 h=9; ex[h]=r; bs[h]=cs;
                    while(h>0&&bs[h-1]<bs[h]){f32 t=bs[h-1];i64 ti=ex[h-1];bs[h-1]=bs[h];ex[h-1]=ex[h];bs[h]=t;ex[h]=ti;--h;} }
            }
            for (r=0;r<na;++r) for(x=0;x<10;++x) if(ap[r]==ex[x]){++hit;break;}
            tot += 10;
        }
        r2 = (i32)(100L * hit / tot);
    }
    print_cstr("  recall@10 vs exact f32:  qmix = "); print_i64(rm);
    print_cstr("%   uniform q2 = "); print_i64(r2); print_cstr("%\n");
    print_flush();
    arena_free(a);
    return (hit1 * 100 / n >= 95 && rm + 2 >= r2) ? 0 : 1;
}
