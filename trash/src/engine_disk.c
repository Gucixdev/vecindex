/* engine_disk.c - saving and loading a built engine index (ENGINE_SERIALIZE): little-endian, magic and version guarded.
 * Part of engine.c, which includes it at this point; not compiled on its own. */

/* ================================================================== *
 * serialise / load a built index - so a consumer does not have to
 * rebuild it from raw vectors on every start. On-disk format is
 * LITTLE-ENDIAN, IEEE-754 f32/f64, magic+version guarded; scalars are
 * written as explicit i32/i64/f32, never raw structs. A big-endian host
 * byte-swaps on the way in/out (dead code on x86). Needs <stdio.h>.
 * ================================================================== */
#ifdef ENGINE_SERIALIZE
#include <stdio.h>
#include <sys/stat.h>

#define ENG_SER_MAGIC   0x35584756u   /* 'VGX5' - v5 adds the ENG_HNSW vector-graph topology */

static i32 eng_levels_(EngScheme s)
{
    switch (s) {
    case ENG_Q1: return 2; case ENG_Q3: return 3; case ENG_Q2: return 4;
    case ENG_Q4: return 16; case ENG_QMIX: case ENG_Q4FS: return 0; /* not a uniform scheme */
    default: return 256;
    }
}

static int eng_host_be_(void) { u32 x = 1u; return *(const u8 *)&x == 0; }

static void eng_bswap_(void *p, size_t esz, size_t n)
{
    u8 *b = (u8 *)p;
    size_t i, j;
    for (i = 0; i < n; ++i) {
        u8 *e = b + i * esz;
        for (j = 0; j < esz / 2; ++j) {
            u8 t = e[j]; e[j] = e[esz - 1 - j]; e[esz - 1 - j] = t;
        }
    }
}

/* write / read n elements of esz bytes as little-endian */
static int engs_wle_(FILE *f, const void *p, size_t esz, size_t n)
{
    if (esz == 1 || !eng_host_be_())
        return fwrite(p, esz, n, f) == n ? 0 : -1;
    {
        void *tmp = malloc(esz * (n ? n : 1));
        int rc;
        if (!tmp) return -1;
        memcpy(tmp, p, esz * n);
        eng_bswap_(tmp, esz, n);
        rc = fwrite(tmp, esz, n, f) == n ? 0 : -1;
        free(tmp);
        return rc;
    }
}
static int engs_rle_(FILE *f, void *p, size_t esz, size_t n)
{
    if (fread(p, esz, n, f) != n) return -1;
    if (esz > 1 && eng_host_be_()) eng_bswap_(p, esz, n);
    return 0;
}
/* Bytes left in f from where it stands. An index read from a file nothing vouched for - one older than v9, a load
 * that checked only the header's blocks (VECINDEX_VERIFY=1), VECINDEX_VERIFY=0 - may claim any count, so no array is
 * allocated that the rest of the file could not fill: tests/fuzz_store.c saw an IVF header ask the arena for more
 * than it could map, which the arena answers with a trap. */
static i64 eng_left_(FILE *f)
{
    struct stat st;
    off_t at = ftello(f);
    if (at < 0 || fstat(fileno(f), &st) != 0) return 0;
    return (i64)st.st_size - (i64)at;
}
/* n elements of esz bytes, n from the file: 1 when they fit in what is left of it */
static int eng_fits_(FILE *f, i64 n, i64 esz) { return n >= 0 && n <= eng_left_(f) / esz; }
static int eng_scheme_ok_(i32 s) { return s >= ENG_Q1 && s <= ENG_Q4FS && s != ENG_NS; }

#define ENG_W32(f, p, n)  engs_wle_((f), (p), 4, (n))
#define ENG_R32(f, p, n)  engs_rle_((f), (p), 4, (n))
#define ENG_W64(f, p, n)  engs_wle_((f), (p), 8, (n))
#define ENG_R64(f, p, n)  engs_rle_((f), (p), 8, (n))

static int eng_q_save_(FILE *f, EngScheme s, i32 dim, const void *qix)
{
    const EngQHead *h = (const EngQHead *)qix;
    i32 L = eng_levels_(s), hdr[3];
    hdr[0] = h->dim; hdr[1] = h->code_bytes; hdr[2] = h->count;
    return ENG_W32(f, hdr, 3)
        || ENG_W32(f, &h->l2_lambda, 1)
        || ENG_W32(f, h->centroid, (size_t)dim)
        || ENG_W32(f, h->codebook, (size_t)dim * L)
        || ENG_W32(f, h->bounds,   (size_t)dim * (L - 1))
        || engs_wle_(f, h->codes,  1, (size_t)h->count * h->code_bytes)
        || ENG_W32(f, h->rnorm,    (size_t)h->count)
        || ENG_W32(f, h->qn2,      (size_t)h->count);
}
static void *eng_q_load_(Arena *a, FILE *f, EngScheme s, i32 dim)
{
    EngQHead *h = (EngQHead *)arena_alloc(a, sizeof(EngQHead));
    i32 L = eng_levels_(s), hdr[3], cnt;
    if (ENG_R32(f, hdr, 3) || ENG_R32(f, &h->l2_lambda, 1)) return (void *)0;
    h->arena = a; h->dim = hdr[0]; h->code_bytes = hdr[1]; h->count = hdr[2];
    if (dim < 0 || hdr[0] != dim || L < 2 || hdr[1] < 0 || !eng_fits_(f, (i64)dim * L * 2, 4)
        || !eng_fits_(f, (i64)hdr[2], (i64)hdr[1] + 8)) return (void *)0;
    cnt = h->count > 0 ? h->count : 1;
    h->centroid = (f32 *)arena_alloc(a, (size_t)dim * sizeof(f32));
    h->codebook = (f32 *)arena_alloc(a, (size_t)dim * L * sizeof(f32));
    h->bounds   = (f32 *)arena_alloc(a, (size_t)dim * (L - 1) * sizeof(f32));
    h->codes    = (u8  *)arena_alloc(a, (size_t)cnt * h->code_bytes);
    h->rnorm    = (f32 *)arena_alloc(a, (size_t)cnt * sizeof(f32));
    h->qn2      = (f32 *)arena_alloc(a, (size_t)cnt * sizeof(f32));
    if (ENG_R32(f, h->centroid, (size_t)dim)
        || ENG_R32(f, h->codebook, (size_t)dim * L)
        || ENG_R32(f, h->bounds,   (size_t)dim * (L - 1))
        || engs_rle_(f, h->codes,  1, (size_t)h->count * h->code_bytes)
        || ENG_R32(f, h->rnorm,    (size_t)h->count)
        || ENG_R32(f, h->qn2,      (size_t)h->count))
        return (void *)0;
    return (void *)h;
}

/* qmix: perm + rn2 + qn2 + l2_lambda, then its three sub-indexes reuse
 * eng_q_save_/eng_q_load_ (they are EngQHead-layout Q4/Q2/Q1 indexes). */
static int eng_qmix_save_(FILE *f, const void *qix)
{
    const QMixIndex *ix = (const QMixIndex *)qix;
    i32 hdr[6];
    hdr[0] = ix->dim; hdr[1] = ix->code_bytes; hdr[2] = ix->count;
    hdr[3] = ix->n_hi; hdr[4] = ix->n_mid; hdr[5] = ix->n_lo;
    return ENG_W32(f, hdr, 6)
        || ENG_W32(f, &ix->l2_lambda, 1)
        || ENG_W32(f, ix->perm, (size_t)ix->dim)
        || ENG_W32(f, ix->rn2,  (size_t)ix->count)
        || ENG_W32(f, ix->qn2,  (size_t)ix->count)
        || eng_q_save_(f, ENG_Q4, ix->n_hi,  ix->hi)
        || eng_q_save_(f, ENG_Q2, ix->n_mid, ix->mid)
        || eng_q_save_(f, ENG_Q1, ix->n_lo,  ix->lo);
}
static void *eng_qmix_load_(Arena *a, FILE *f)
{
    QMixIndex *ix = (QMixIndex *)arena_alloc(a, sizeof(QMixIndex));
    i32 hdr[6], cnt;
    if (ENG_R32(f, hdr, 6) || ENG_R32(f, &ix->l2_lambda, 1)) return (void *)0;
    ix->arena = a; ix->dim = hdr[0]; ix->code_bytes = hdr[1]; ix->count = hdr[2];
    ix->n_hi = hdr[3]; ix->n_mid = hdr[4]; ix->n_lo = hdr[5];
    if (hdr[3] < 0 || hdr[4] < 0 || hdr[5] < 0 || (i64)hdr[3] + hdr[4] + hdr[5] != (i64)hdr[0]
        || !eng_fits_(f, (i64)hdr[0], 4) || !eng_fits_(f, (i64)hdr[2], 8)) return (void *)0;
    cnt = ix->count > 0 ? ix->count : 1;
    ix->perm = (i32 *)arena_alloc(a, (size_t)(ix->dim > 0 ? ix->dim : 1) * sizeof(i32));
    ix->rn2  = (f32 *)arena_alloc(a, (size_t)cnt * sizeof(f32));
    ix->qn2  = (f32 *)arena_alloc(a, (size_t)cnt * sizeof(f32));
    if (ENG_R32(f, ix->perm, (size_t)ix->dim)
        || ENG_R32(f, ix->rn2, (size_t)ix->count)
        || ENG_R32(f, ix->qn2, (size_t)ix->count))
        return (void *)0;
    ix->hi  = (Q4Index *)eng_q_load_(a, f, ENG_Q4, ix->n_hi);
    ix->mid = (Q2Index *)eng_q_load_(a, f, ENG_Q2, ix->n_mid);
    ix->lo  = (Q1Index *)eng_q_load_(a, f, ENG_Q1, ix->n_lo);
    if (!ix->hi || !ix->mid || !ix->lo) return (void *)0;
    return (void *)ix;
}
static int eng_ivf_save_(FILE *f, const Ivf *v)
{
    i32 hdr[3];
    hdr[0] = v->dim; hdr[1] = v->n_clusters; hdr[2] = v->n;
    return ENG_W32(f, hdr, 3)
        || ENG_W32(f, v->centroids, (size_t)v->n_clusters * v->dim)
        || ENG_W64(f, v->list_ids,  (size_t)v->n)
        || ENG_W64(f, v->list_off,  (size_t)(v->n_clusters + 1))
        || ENG_W64(f, v->cluster_of,(size_t)v->n);
}
static Ivf *eng_ivf_load_(Arena *a, FILE *f, i32 dim)
{
    Ivf *v = (Ivf *)arena_alloc(a, sizeof(Ivf));
    i32 hdr[3];
    if (ENG_R32(f, hdr, 3)) return (Ivf *)0;
    v->arena = a; v->dim = hdr[0]; v->n_clusters = hdr[1]; v->n = hdr[2];
    if (hdr[0] != dim || hdr[1] < 1 || !eng_fits_(f, (i64)hdr[1], (i64)dim * 4 + 8) || !eng_fits_(f, (i64)hdr[2], 16))
        return (Ivf *)0;
    v->centroids  = (f32 *)arena_alloc(a, (size_t)v->n_clusters * v->dim * sizeof(f32));
    v->list_ids   = (i64 *)arena_alloc(a, (size_t)(v->n > 0 ? v->n : 1) * sizeof(i64));
    v->list_off   = (i64 *)arena_alloc(a, (size_t)(v->n_clusters + 1) * sizeof(i64));
    v->cluster_of = (i64 *)arena_alloc(a, (size_t)(v->n > 0 ? v->n : 1) * sizeof(i64));
    if (ENG_R32(f, v->centroids, (size_t)v->n_clusters * v->dim)
        || ENG_R64(f, v->list_ids,  (size_t)v->n)
        || ENG_R64(f, v->list_off,  (size_t)(v->n_clusters + 1))
        || ENG_R64(f, v->cluster_of,(size_t)v->n))
        return (Ivf *)0;
    return v;
}

/* scheme-agnostic fine-index (de)serialise: qmix has its own shape;
 * q4fs stores its wrapped Q4Index and rebuilds the transpose on load;
 * the five uniform schemes share eng_q_save_/eng_q_load_. */
static int eng_qany_save_(FILE *f, EngScheme s, i32 dim, const void *ix)
{
    if (s == ENG_QMIX) return eng_qmix_save_(f, ix);
    if (s == ENG_Q4FS) return eng_q_save_(f, ENG_Q4, dim,
                                          ((const Q4FSIndex *)ix)->base);
    return eng_q_save_(f, s, dim, ix);
}
static void *eng_qany_load_(Arena *a, FILE *f, EngScheme s, i32 dim)
{
    if (s == ENG_QMIX) return eng_qmix_load_(a, f);
    if (s == ENG_Q4FS) {
        void *base = eng_q_load_(a, f, ENG_Q4, dim);
        return base ? (void *)q4fs_wrap(a, (const Q4Index *)base) : (void *)0;
    }
    return eng_q_load_(a, f, s, dim);
}

/* 0 on success; a self-contained index blob is appended to `f`. */
int engine_save(const EngIndex *e, FILE *f)
{
    i32 hdr[7];
    hdr[0] = (i32)ENG_SER_MAGIC; hdr[1] = (i32)e->topo; hdr[2] = (i32)e->scheme;
    hdr[3] = e->nprobe; hdr[4] = e->code_bytes; hdr[5] = e->dim;
    hdr[6] = (i32)e->casc_hi_s;
    if (ENG_W32(f, hdr, 7)
        || ENG_W32(f, &e->count, 1)
        || ENG_W32(f, &e->recall, 1))
        return -1;
    if (e->topo == ENG_FLAT)
        return eng_qany_save_(f, e->scheme, e->dim, e->flat_ix);
    if (e->topo == ENG_CASCADE)
        return eng_q_save_(f, e->scheme, e->dim, e->flat_ix)
            || eng_q_save_(f, e->casc_hi_s, e->dim, e->casc_hi);
    if (e->topo == ENG_HNSW) {
        /* the graph is rebuilt from the vectors + seed on load */
        return ENG_W32(f, &e->vhnsw_seed, 1)
            || ENG_W32(f, e->vhnsw_pts, (size_t)e->count * e->dim);
    }
    if (eng_ivf_save_(f, e->ivf)) return -1;
    {
        u8 has_router = (u8)(e->hrouter != (Hnsw *)0);
        if (fwrite(&has_router, 1, 1, f) != 1) return -1;
        if (ENG_W32(f, &e->router_seed, 1)) return -1;
    }
    {
        i32 c, nc = e->ivf->n_clusters;
        for (c = 0; c < nc; ++c) {
            u8 present = (u8)(e->cluster_ix[c] != (void *)0);
            if (fwrite(&present, 1, 1, f) != 1) return -1;
            if (present && eng_qany_save_(f, e->scheme, e->dim, e->cluster_ix[c]))
                return -1;
        }
    }
    return 0;
}

/* Reconstruct an EngIndex from `f` into arena `a`. NULL on any error. */
EngIndex *engine_load(Arena *a, FILE *f)
{
    EngIndex *e = (EngIndex *)arena_alloc(a, sizeof(EngIndex));
    i32 hdr[7];
    if (ENG_R32(f, hdr, 7) || (u32)hdr[0] != ENG_SER_MAGIC)
        return (EngIndex *)0;
    memset(e, 0, sizeof(*e));
    e->arena = a; e->topo = (EngTopo)hdr[1]; e->scheme = (EngScheme)hdr[2];
    e->nprobe = hdr[3]; e->code_bytes = hdr[4]; e->dim = hdr[5];
    e->casc_hi_s = (EngScheme)hdr[6];
    if (ENG_R32(f, &e->count, 1)
        || ENG_R32(f, &e->recall, 1))
        return (EngIndex *)0;
    if (e->dim < 1 || e->count < 0 || hdr[1] < ENG_FLAT || hdr[1] > ENG_HNSW || !eng_scheme_ok_(hdr[2])
        || (e->topo == ENG_CASCADE && !eng_scheme_ok_(hdr[6])))
        return (EngIndex *)0;
    if (e->topo == ENG_FLAT) {
        e->flat_ix = eng_qany_load_(a, f, e->scheme, e->dim);
        return e->flat_ix ? e : (EngIndex *)0;
    }
    if (e->topo == ENG_CASCADE) {
        e->flat_ix = eng_q_load_(a, f, e->scheme, e->dim);
        e->casc_hi = eng_q_load_(a, f, e->casc_hi_s, e->dim);
        return (e->flat_ix && e->casc_hi) ? e : (EngIndex *)0;
    }
    if (e->topo == ENG_HNSW) {
        size_t nv = (size_t)e->count * e->dim;
        HnswParams hp;
        if (!eng_fits_(f, (i64)nv, 4)) return (EngIndex *)0;
        e->vhnsw_pts = (f32 *)arena_alloc(a, nv * sizeof(f32));
        if (ENG_R32(f, &e->vhnsw_seed, 1) || ENG_R32(f, e->vhnsw_pts, nv))
            return (EngIndex *)0;
        hp.M = 24; hp.seed = e->vhnsw_seed;
        hp.ef_construction = (e->count <= 12000) ? 256 : 128;   /* match engine_build */
        e->vhnsw = hnsw_build(a, e->vhnsw_pts, e->count, e->dim, hp);
        return e->vhnsw ? e : (EngIndex *)0;
    }
    e->ivf = eng_ivf_load_(a, f, e->dim);
    if (!e->ivf) return (EngIndex *)0;
    {
        u8 has_router;
        if (fread(&has_router, 1, 1, f) != 1) return (EngIndex *)0;
        if (ENG_R32(f, &e->router_seed, 1)) return (EngIndex *)0;
        if (has_router) {
            HnswParams hp;
            hp.M = 16; hp.ef_construction = 96; hp.seed = e->router_seed;
            e->hrouter = hnsw_build(a, e->ivf->centroids, e->ivf->n_clusters,
                                    e->ivf->dim, hp);
        }
    }
    {
        i32 c, nc = e->ivf->n_clusters;
        e->cluster_ix = (void **)arena_alloc(a, (size_t)nc * sizeof(void *));
        for (c = 0; c < nc; ++c) {
            u8 present;
            if (fread(&present, 1, 1, f) != 1) return (EngIndex *)0;
            if (present) {
                e->cluster_ix[c] = eng_qany_load_(a, f, e->scheme, e->dim);
                if (!e->cluster_ix[c]) return (EngIndex *)0;
            } else {
                e->cluster_ix[c] = (void *)0;
            }
        }
    }
    return e;
}
#endif /* ENGINE_SERIALIZE */
