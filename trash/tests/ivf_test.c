/* ivf_test.c - the self-test of src/ivf.c; ./run builds and runs it. */
#define ARENA_ALLOC_IMPLEMENTATION
#include "ivf.c"
#define PRINT_IMPLEMENTATION
#include "print.h"

static u32 ivfd_rng_(u32 *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static f32 ivfd_f_(u32 *s) { return (f32)ivfd_rng_(s) / (f32)0xffffffffu - 0.5f; }

static f32 *ivfd_clustered_(Arena *a, i32 per, i32 topics, i32 dim, f32 spread, u32 seed)
{
    i32 n = per * topics, c, i, d;
    f32 *centers = (f32 *)arena_alloc(a, (size_t)topics * dim * sizeof(f32));
    f32 *out = (f32 *)arena_alloc(a, (size_t)n * dim * sizeof(f32));
    u32 st = seed | 1u;
    for (i = 0; i < topics * dim; ++i) centers[i] = ivfd_f_(&st) * 10.0f;
    for (c = 0; c < topics; ++c)
        for (i = 0; i < per; ++i)
            for (d = 0; d < dim; ++d)
                out[((size_t)c * per + i) * dim + d] = centers[c * dim + d] + ivfd_f_(&st) * spread;
    return out;
}

int main(void)
{
    static Arena arena;
    Arena *a = &arena;
    i32 dim = 32, topics = 8, per = 50, n = topics * per;
    f32 *v = ivfd_clustered_(a, per, topics, dim, 0.3f, 7u);
    i32 nc = ivf_suggest_clusters(n);
    Ivf *ix = ivf_train(a, v, n, dim, nc, 12, 123u);
    i32 probe[8], i, found = 0, checked = 0;

    print_cstr("ivf  n="); print_i64(n); print_cstr(" dim="); print_i64(dim);
    print_cstr("  n_clusters="); print_i64(nc);
    print_cstr("  (suggest = round(sqrt(n)))\n");

    /* last CSR offset must be n */
    if (ix->list_off[nc] != (i64)n) { print_cstr("  BAD: csr total != n\n"); print_flush(); return 1; }

    /* every row's own vector must route into a probed cluster whose
     * member list contains it, at nprobe=1 */
    for (i = 0; i < n; i += 3) {
        i32 np = ivf_route(ix, v + (size_t)i * dim, 1, probe), mc, m, hit = 0;
        const i64 *mem = ivf_members(ix, probe[0], &mc);
        (void)np;
        for (m = 0; m < mc; ++m) if (mem[m] == (i64)i) { hit = 1; break; }
        found += hit; ++checked;
    }
    print_cstr("  self-vector routes into own cluster: ");
    print_i64((i64)(100 * found / checked)); print_cstr("%\n");
    print_flush();
    arena_free(a);
    return (found == checked) ? 0 : 1;
}
