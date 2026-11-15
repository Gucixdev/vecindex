/* tq_split.c - clusters split in place as they grow (the SPFresh idea), instead of rebuilding the whole index.
 * Part of tq.c, which includes it at this point; not compiled on its own.
 *
 * A cluster past its limit is cut in two by a 2-means of its own rows: one half keeps its number, the other becomes a
 * new cluster. Then the rows of its nearest clusters that now sit closer to one of the two halves move there, and
 * rows of the two halves that sit closer to one of those neighbours move to it (SPFresh's reassignment). A row's code
 * does not depend on its cluster (the codes are of the rotated row itself), so a move copies its code lane into a
 * slot of the other cluster and frees the old slot: nothing is coded again, and the global clustering stays as it was
 * everywhere else. A freed slot stays in its block as a hole (scanned, never returned) until the next rebuild. */

/* one row of a split: where it sits and which row it is */
typedef struct { i32 b, k, row; } TqMem;

/* the live rows of cluster c into out (NULL: count them); dead: tombstones (may be NULL). -1: no memory */
static i32 tq_members_(const Tq *t, i32 c, const u64 *dead, TqMem *out)
{
    Gen g;
    i32 *bl, nb, b, k, m = 0;
    gen_init_(&g);
    if (!(bl = (i32 *)gen_alloc_(&g, (size_t)(t->nblk + t->xn + 1) * sizeof(i32)))) { gen_free_(&g); return -1; }
    nb = tq_cblocks_(t, c, bl);
    for (b = 0; b < nb; ++b)
        for (k = 0; k < 32; ++k) {
            i32 r = tq_srow_(t, bl[b], k);
            if (r < 0 || r >= t->n || TQ_SKIP(dead, r)) continue;
            if (out) { out[m].b = bl[b]; out[m].k = k; out[m].row = r; }
            ++m;
        }
    gen_free_(&g);
    return m;
}

/* a free slot at the end of cluster c (its last extra block, or the padding of its last built block), else a new
 * extra block: the block into *b, the lane into *k. 0, or -1 when memory runs out. */
static i32 tq_free_slot_(Tq *t, i32 c, i32 *b, i32 *k)
{
    *b = -1; *k = 32;
    if (t->xtail && t->xtail[c] >= 0) {
        i32 xb = t->xtail[c];
        for (*k = 0; *k < 32 && t->xslot[(size_t)xb * 32 + (size_t)*k] >= 0; ++*k) {}
        *b = t->nblk + xb;
    } else if (t->nc > 1 && !t->mapped) {                       /* (a mapped index's built codes stay as loaded) */
        i32 b0 = t->cstart[c], last = t->cstart[c + 1] - 1;
        if (last >= b0) { for (*k = 0; *k < 32 && t->slot_row[(size_t)last * 32 + (size_t)*k] >= 0; ++*k) {} *b = last; }
    }
    if (*b < 0 || *k >= 32) {
        if (tq_xblock_(t, c) != 0) return -1;
        *b = t->nblk + t->xtail[c]; *k = 0;
    }
    return 0;
}

/* the row in slot (b, k) to cluster c, its code with it; the old slot becomes a hole. 0, or -1 (no memory: nothing
 * moved). cd, c1: scratch of vd and vd1 bytes. */
static i32 tq_move_(Tq *t, i32 b, i32 k, i32 from, i32 c, u8 *cd, u8 *c1)
{
    i32 nb, nk, row = tq_srow_(t, b, k);
    if (tq_free_slot_(t, c, &nb, &nk) != 0) return -1;
    tq_get_lane_(tq_bptr_(t, b), t->vd, k, cd);
    tq_put_lane_((u8 *)tq_bptr_(t, nb), t->vd, nk, cd);
    if (t->codes1) { tq_get_lane_(tq_bptr1_(t, b), t->vd1, k, c1); tq_put_lane_(tq_bptr1_(t, nb), t->vd1, nk, c1); }
    if (nb >= t->nblk) t->xslot[(size_t)(nb - t->nblk) * 32 + (size_t)nk] = row;
    else t->slot_row[(size_t)nb * 32 + (size_t)nk] = row;
    if (b >= t->nblk) t->xslot[(size_t)(b - t->nblk) * 32 + (size_t)k] = -1;
    else { t->slot_row[(size_t)b * 32 + (size_t)k] = -1; ++t->sgen; }
    ++t->holes;
    if (t->csize) {
        t->csq += 2.0 * ((f64)t->csize[c] - (f64)t->csize[from]) + 2.0;
        --t->csize[from]; ++t->csize[c];
    }
    return 0;
}

/* room for one more cluster: cent, cstart, csize and the extra-block heads grown by one (copied out of a file
 * mapping first). 0, or -1 (no memory: t unchanged). */
static i32 tq_grow_nc_(Tq *t)
{
    i32 nc = t->nc, dim = t->dim;
    if (t->ccap < nc + 1 || t->reg[TQ_R_CENT].p != (void *)t->cent) {   /* room for half as many again: a split copies O(1) on average */
        i32 cap = nc + 1 + nc / 2 + 16;
        if (reg_fit_(&t->reg[TQ_R_CENT], (void **)&t->cent, (size_t)nc * dim * sizeof(f32), (size_t)cap * dim * sizeof(f32)) != 0
            || reg_fit_(&t->reg[TQ_R_CSTART], (void **)&t->cstart, (size_t)(nc + 1) * sizeof(i32), (size_t)(cap + 1) * sizeof(i32)) != 0
            || (t->csize && reg_fit_(&t->reg[TQ_R_CSIZE], (void **)&t->csize, (size_t)nc * sizeof(i32), (size_t)cap * sizeof(i32)) != 0)
            || (t->xhead && (reg_fit_(&t->reg[TQ_R_XHEAD], (void **)&t->xhead, (size_t)nc * sizeof(i32), (size_t)cap * sizeof(i32)) != 0
                             || reg_fit_(&t->reg[TQ_R_XTAIL], (void **)&t->xtail, (size_t)nc * sizeof(i32), (size_t)cap * sizeof(i32)) != 0))) return -1;
        t->ccap = cap;
        if (t->mapped) t->cent_own = 1;                           /* (off the file mapping: tq_ram_ counts them) */
    }
    if (t->mapped && !t->slot_own) {                             /* the slots will change too: off the mapping */
        if (reg_fit_(&t->reg[TQ_R_SLOT], (void **)&t->slot_row, (size_t)t->nblk * 32 * sizeof(i32), (size_t)(t->nblk ? t->nblk : 1) * 32 * sizeof(i32)) != 0) return -1;
        t->slot_own = 1;
    }
    t->cstart[nc + 1] = t->cstart[nc];                           /* no built blocks: only extra ones */
    if (t->csize) t->csize[nc] = 0;
    if (t->xhead) t->xhead[nc] = t->xtail[nc] = -1;
    t->nc = nc + 1;
    return 0;
}

/* the clusters closest to centroid v, but not a or b: up to n of them into out, closest first; returns how many */
static i32 tq_near_c_(const Tq *t, const f32 *v, i32 a, i32 b, i32 n, i32 *out)
{
    i32 c, m = 0, z;
    Gen g;
    f32 *sc;
    gen_init_(&g);
    sc = (f32 *)gen_alloc_(&g, (size_t)(n > 0 ? n : 1) * sizeof(f32));
    for (c = 0; sc && c < t->nc; ++c) {
        f32 d;
        if (c == a || c == b) continue;
        d = vs_dot(v, t->cent + (size_t)c * t->dim, t->dim);
        if (m == n && d <= sc[n - 1]) continue;
        for (z = m < n ? m++ : n - 1; z > 0 && sc[z - 1] < d; --z) { sc[z] = sc[z - 1]; out[z] = out[z - 1]; }
        sc[z] = d; out[z] = c;
    }
    gen_free_(&g);
    return m;
}

/* v[dim] / |v| (a zero sum stays zero) */
static void tq_unit_(f32 *v, i32 dim)
{
    f64 n = 0.0;
    i32 d;
    for (d = 0; d < dim; ++d) n += (f64)v[d] * v[d];
    if (n > 0.0) { f32 s = (f32)(1.0 / sqrt(n)); for (d = 0; d < dim; ++d) v[d] *= s; }
}

#define TQ_SPLIT_NEAR 8             /* neighbours whose rows may move into the two halves */
#define TQ_SPLIT_ROUNDS 8           /* rounds of the 2-means */

/* Cluster c cut in two: its rows' 2-means (spherical, started from a row and the row least like it), one half keeps c,
 * the other is cluster t->nc (new). Then the reassignment against the TQ_SPLIT_NEAR clusters nearest c. The probe
 * tables (ct, c8) are NOT redone here: tq_split_big does that once for all its splits. rows: the vectors (unit),
 * dead: tombstones. Returns the rows moved, or -1 (too few rows, one half empty, or no memory - then t is as it was
 * apart from rows already moved, which stay valid). Writes t: no search may run meanwhile. */
static i32 tq_split_(Tq *t, i32 c, const TqCol *rows, const u64 *dead)
{
    i32 dim = t->dim, m, i, it, d, nn, moved = 0, near[TQ_SPLIT_NEAR], nnear, j, *side;
    TqMem *mem;
    f32 *x, *ca, *cb, *rb;
    u8 *cd, *c1;
    Gen tmp;                                                     /* the split's working arrays */
    m = tq_members_(t, c, dead, (TqMem *)0);
    if (m < 64) return -1;
    gen_init_(&tmp);
    rb = (f32 *)gen_alloc_(&tmp, (size_t)dim * sizeof(f32));
    cd = (u8 *)gen_alloc_(&tmp, (size_t)t->vd + 16); c1 = (u8 *)gen_alloc_(&tmp, (size_t)(t->vd1 > 0 ? t->vd1 : 1) + 16);
    mem = (TqMem *)gen_alloc_(&tmp, (size_t)m * sizeof(TqMem));
    x = (f32 *)gen_alloc_(&tmp, (size_t)m * dim * sizeof(f32));
    side = (i32 *)gen_alloc_(&tmp, (size_t)m * sizeof(i32));
    ca = (f32 *)gen_alloc_(&tmp, (size_t)dim * 2 * sizeof(f32)); cb = ca ? ca + dim : (f32 *)0;
    if (!rb || !cd || !c1 || !mem || !x || !side || !ca) { gen_free_(&tmp); return -1; }
    tq_members_(t, c, dead, mem);
    for (i = 0; i < m; ++i) {
        const f32 *v = tq_row_(rows, mem[i].row, rb);
        memcpy(x + (size_t)i * dim, v, (size_t)dim * sizeof(f32));
    }
    {   /* the start: the first row, and the row least like it */
        i32 far = 0;
        f32 lo = 2.0f;
        for (i = 0; i < m; ++i) { f32 s = vs_dot(x, x + (size_t)i * dim, dim); if (s < lo) { lo = s; far = i; } }
        memcpy(ca, x, (size_t)dim * sizeof(f32)); memcpy(cb, x + (size_t)far * dim, (size_t)dim * sizeof(f32));
    }
    for (it = 0; it < TQ_SPLIT_ROUNDS; ++it) {
        i32 na = 0;
        for (i = 0; i < m; ++i) { side[i] = vs_dot(x + (size_t)i * dim, cb, dim) > vs_dot(x + (size_t)i * dim, ca, dim); na += !side[i]; }
        if (na == 0 || na == m) break;
        memset(ca, 0, (size_t)dim * 2 * sizeof(f32));
        for (i = 0; i < m; ++i) { f32 *to = side[i] ? cb : ca; const f32 *v = x + (size_t)i * dim; for (d = 0; d < dim; ++d) to[d] += v[d]; }
        tq_unit_(ca, dim); tq_unit_(cb, dim);
    }
    for (i = 0, j = 0; i < m; ++i) j += side[i];
    if (j == 0 || j == m || tq_grow_nc_(t) != 0) { gen_free_(&tmp); return -1; }
    nn = t->nc - 1;
    nnear = tq_near_c_(t, t->cent + (size_t)c * dim, c, nn, TQ_SPLIT_NEAR, near);   /* around the old centroid */
    memcpy(t->cent + (size_t)c * dim, ca, (size_t)dim * sizeof(f32));
    memcpy(t->cent + (size_t)nn * dim, cb, (size_t)dim * sizeof(f32));
    for (i = 0; i < m; ++i) {                                    /* the two halves, and their rows nearer a neighbour */
        const f32 *v = x + (size_t)i * dim;
        f32 best = vs_dot(v, side[i] ? cb : ca, dim);
        i32 to = side[i] ? nn : c;
        for (j = 0; j < nnear; ++j) { f32 s = vs_dot(v, t->cent + (size_t)near[j] * dim, dim); if (s > best) { best = s; to = near[j]; } }
        if (to != c) { if (tq_move_(t, mem[i].b, mem[i].k, c, to, cd, c1) != 0) break; ++moved; }
    }
    for (j = 0; j < nnear; ++j) {                                /* the neighbours' rows nearer one of the halves */
        i32 g = near[j], mg = tq_members_(t, g, dead, (TqMem *)0), q;
        TqMem *gm = mg >= 0 ? (TqMem *)gen_alloc_(&tmp, (size_t)(mg > 0 ? mg : 1) * sizeof(TqMem)) : (TqMem *)0;
        if (!gm) break;
        tq_members_(t, g, dead, gm);
        for (q = 0; q < mg; ++q) {
            const f32 *v = tq_row_(rows, gm[q].row, rb);
            f32 own = vs_dot(v, t->cent + (size_t)g * dim, dim), sa = vs_dot(v, ca, dim), sb = vs_dot(v, cb, dim);
            if (sa > own || sb > own) { if (tq_move_(t, gm[q].b, gm[q].k, g, sa >= sb ? c : nn, cd, c1) != 0) break; ++moved; }
        }
    }
    tq_top_add_(t, nn);                                          /* (c keeps its top cell: its centroid moved a little) */
    ++t->nsplit;
    gen_free_(&tmp);
    return moved;
}

/* Every cluster with more than limit live rows split (tq_split_), at most maxn of them, then the probe tables
 * redone and nprobe kept the same share of the clusters it had when last set. rows: the vectors (unit); dead:
 * tombstones. Returns the clusters split. Writes t: no search may run meanwhile. */
i32 tq_split_big(Tq *t, const TqCol *rows, const u64 *dead, i32 limit, i32 maxn)
{
    i32 c, done = 0, nc0 = t->nc;
    if (t->nc <= 1 || !t->slot_row || limit < 64) return 0;
    if (!t->csize) tq_count_(t);
    if (!t->pnc) { t->pnc = t->nc; t->pnp = t->nprobe; }
    for (c = 0; c < nc0 && done < maxn; ++c) {
        if (t->csize[c] <= limit) continue;
        if (tq_split_(t, c, rows, dead) >= 0) ++done;
    }
    if (done) {
        tq_tab_reset_(t);                                        /* the probe tables, from the new centroids */
        t->ct = tq_transpose_(&t->tab, t->cent, t->nc, t->dim);
        tq_c8_(t);
        {   i32 np = (i32)((f64)t->pnp * (f64)t->nc / (f64)t->pnc + 0.5);
            t->nprobe = np < 1 ? 1 : (np > t->nc ? t->nc : np); }
    }
    return done;
}
