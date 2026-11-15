/* tq_add.c - rows added after the build, the fit numbers that say when to rebuild, and renumbering after a compaction.
 * Part of tq.c, which includes it at this point; not compiled on its own. */

/* ---- rows added after the build ---- */

/* rows prepared for tq_add_put: the closest cluster of each (0 when flat), its squared distance to that centroid, and
 * its code bytes (vd a row) */
typedef struct { i32 m; i32 *cl; f32 *d2; u8 *cd; Gen g; i32 fail; } TqAdd;   /* g: the arena the three arrays are in */

typedef struct { const Tq *t; const TqCol *rows; i64 r0; TqAdd *a; } TqPrepJob;

static void tq_prep_rows_(void *p, i64 a, i64 b, i32 tid)
{
    TqPrepJob *j = (TqPrepJob *)p;
    const Tq *t = j->t;
    i32 dim = t->dim, r, *cl = j->a->cl;
    Gen g;
    f32 *x, *rot, *tmp;
    u8 *scr;
#if TQ_VNNI
    i32 ng16 = (t->nc + 15) / 16 * 16;
    u32 *qws, *qp[8];
    i32 *s8, *sp[8];
#endif
    i64 i;
    (void)tid;
    gen_init_(&g);
    x = (f32 *)gen_alloc_(&g, (size_t)8 * (size_t)dim * sizeof(f32));
    rot = (f32 *)gen_alloc_(&g, (size_t)t->dpad * sizeof(f32)); tmp = (f32 *)gen_alloc_(&g, (size_t)t->dpad * sizeof(f32));
    scr = (u8 *)gen_alloc_(&g, t->nc > 1 ? tq_probe_scr_(t) : 1);
#if TQ_VNNI
    qws = (u32 *)gen_alloc_(&g, (size_t)8 * (size_t)(t->c8d4 > 0 ? t->c8d4 : 1) * sizeof(u32));
    s8 = (i32 *)gen_alloc_(&g, (size_t)8 * (size_t)(ng16 > 0 ? ng16 : 16) * sizeof(i32));
    if (!qws || !s8) { gen_free_(&g); j->a->fail = 1; return; }
#endif
    if (!x || !rot || !tmp || !scr) { gen_free_(&g); j->a->fail = 1; return; }
    for (i = a; i < b; i += 8) {
        i32 g = b - i < 8 ? (i32)(b - i) : 8;
        for (r = 0; r < g; ++r) {
            f32 *dst = x + (size_t)r * dim;
            const f32 *src = tq_row_(j->rows, j->r0 + i + r, dst);
            if (src != dst) memcpy(dst, src, (size_t)dim * sizeof(f32));
        }
        if (t->nc > 1) {
#if TQ_VNNI
            if (t->c8 && g == 8) {                               /* 8 rows per pass over the int8 centroids, like a batch */
                u64 *key = (u64 *)(void *)scr, *ktmp = key + ng16;
                i32 *cand = (i32 *)(void *)(ktmp + ng16) + ng16;
                for (r = 0; r < 8; ++r) { qp[r] = qws + (size_t)r * t->c8d4; sp[r] = s8 + (size_t)r * ng16; tq_q8_(t, x + (size_t)r * dim, qp[r]); }
                tq_score8x8_(t, qp, sp);
                for (r = 0; r < 8; ++r) tq_pick_(key, tq_cands_(t, x + (size_t)r * dim, 1, sp[r], key, cand), 1, ktmp, cl + i + r, 0);
            } else
#endif
            for (r = 0; r < g; ++r) tq_probe_(t, x + (size_t)r * dim, 1, scr, cl + i + r);
            for (r = 0; r < g; ++r) j->a->d2[i + r] = 2.0f - 2.0f * vs_dot(x + (size_t)r * dim, t->cent + (size_t)cl[i + r] * dim, dim);
        } else for (r = 0; r < g; ++r) { cl[i + r] = 0; j->a->d2[i + r] = 0.0f; }
        for (r = 0; r < g; ++r) { tq_rot_(t, x + (size_t)r * dim, rot, tmp); tq_encode_row_(t, rot, j->a->cd + (size_t)(i + r) * (size_t)t->vd); }
    }
    gen_free_(&g);
}

void tq_add_free(TqAdd *a) { gen_free_(&a->g); a->cl = (i32 *)0; a->d2 = (f32 *)0; a->cd = (u8 *)0; a->m = 0; }

/* Rows r0 .. r0 + m - 1 of `rows` (unit length) prepared for tq_add_put, on nth threads. Only reads t, so it can run
 * while searches do, as long as nothing adds to t or replaces it meanwhile. 0, or -1 when memory runs out. */
i32 tq_add_prep(const Tq *t, const TqCol *rows, i64 r0, i32 m, i32 nth, TqAdd *a)
{
    TqPrepJob j;
    size_t n1 = (size_t)(m > 0 ? m : 1);
    a->m = m > 0 ? m : 0;
    gen_init_(&a->g); a->fail = 0;
    a->cl = (i32 *)gen_alloc_(&a->g, n1 * sizeof(i32)); a->d2 = (f32 *)gen_alloc_(&a->g, n1 * sizeof(f32)); a->cd = (u8 *)gen_alloc_(&a->g, n1 * (size_t)t->vd);
    if (!a->cl || !a->d2 || !a->cd) { tq_add_free(a); return -1; }
    j.t = t; j.rows = rows; j.r0 = r0; j.a = a;
    if (a->m) tq_par_(tq_prep_rows_, &j, a->m, nth);
    if (a->fail) { tq_add_free(a); return -1; }                  /* a thread found no memory */
    return 0;
}

/* a new extra block at the end of cluster c's chain: 0, or -1 when memory runs out */
static i32 tq_xblock_(Tq *t, i32 c)
{
    i32 b, k, nh = t->nc > 1 ? t->nc : 1;
    if (!t->xhead) {
        i32 cc = t->ccap > nh ? t->ccap : nh;                     /* (tq_grow_nc_ may have set room for more clusters) */
        if (reg_fit_(&t->reg[TQ_R_XHEAD], (void **)&t->xhead, 0, (size_t)cc * sizeof(i32)) != 0
            || reg_fit_(&t->reg[TQ_R_XTAIL], (void **)&t->xtail, 0, (size_t)cc * sizeof(i32)) != 0) {
            reg_free_(&t->reg[TQ_R_XHEAD]); reg_free_(&t->reg[TQ_R_XTAIL]); t->xhead = t->xtail = (i32 *)0; return -1;
        }
        for (k = 0; k < nh; ++k) t->xhead[k] = t->xtail[k] = -1;
    }
    if (t->xn == t->xcap) {                                      /* grow by half: an add copies O(1) bytes on average */
        i32 cap = t->xcap + t->xcap / 2 + 64;
        size_t have = (size_t)t->xn;
        if (reg_fit_(&t->reg[TQ_R_XCODES], (void **)&t->xcodes, have * (size_t)t->vd * 16, (size_t)cap * (size_t)t->vd * 16) != 0
            || reg_fit_(&t->reg[TQ_R_XSLOT], (void **)&t->xslot, have * 32 * sizeof(i32), (size_t)cap * 32 * sizeof(i32)) != 0
            || reg_fit_(&t->reg[TQ_R_XNEXT], (void **)&t->xnext, have * sizeof(i32), (size_t)cap * sizeof(i32)) != 0) return -1;
        /* the cascade's 1-bit codes grow beside them */
        if (t->codes1 && reg_fit_(&t->reg[TQ_R_XCODES1], (void **)&t->xcodes1, have * (size_t)t->vd1 * 16, (size_t)cap * (size_t)t->vd1 * 16) != 0) return -1;
        t->xcap = cap;
    }
    b = t->xn++;
    memset(t->xcodes + (size_t)b * (size_t)t->vd * 16, 0, (size_t)t->vd * 16);
    if (t->codes1) memset(t->xcodes1 + (size_t)b * (size_t)t->vd1 * 16, 0, (size_t)t->vd1 * 16);
    for (k = 0; k < 32; ++k) t->xslot[(size_t)b * 32 + (size_t)k] = -1;
    t->xnext[b] = -1;
    if (t->xtail[c] >= 0) t->xnext[t->xtail[c]] = b; else t->xhead[c] = b;
    t->xtail[c] = b;
    return 0;
}

/* Rows r0 .. r0 + m - 1, prepared by tq_add_prep against this same t, into the index: r0 must be t->n (rows go in in
 * order). A row takes a free slot of its cluster's last block, or opens an extra block. Writes t: no search may run
 * meanwhile. 0, or -1 (r0 out of order, or memory ran out - the rows placed until then stay). */
i32 tq_add_put(Tq *t, const TqAdd *a, i64 r0)
{
    i32 i, d, vd = t->vd;
    u8 *c1 = (u8 *)0;
    Gen g;
    if (r0 != (i64)t->n) return -1;
    gen_init_(&g);
    if (t->codes1 && !(c1 = (u8 *)gen_alloc_(&g, (size_t)t->vd1 + 16))) { gen_free_(&g); return -1; }   /* a row's 1-bit columns, for the cascade */
    for (i = 0; i < a->m; ++i) {
        i32 row = (i32)(r0 + i), c = t->nc > 1 ? a->cl[i] : 0, b = -1, k = 32;
        const u8 *cd = a->cd + (size_t)i * (size_t)vd;
        u8 *blk;
        if (t->xtail && t->xtail[c] >= 0) {                      /* the cluster already spilled into extra blocks */
            i32 xb = t->xtail[c];
            for (k = 0; k < 32 && t->xslot[(size_t)xb * 32 + (size_t)k] >= 0; ++k) {}
            b = t->nblk + xb;
        } else {                                                 /* else the padding of its last built block */
            i32 b0 = t->nc > 1 ? t->cstart[c] : 0, last = t->nc > 1 ? t->cstart[c + 1] - 1 : t->nblk - 1;
            if (last >= b0) {
                if (t->slot_row) for (k = 0; k < 32 && t->slot_row[(size_t)last * 32 + (size_t)k] >= 0; ++k) {}
                else k = row - last * 32;                        /* flat, slot = row */
                b = last;
            }
        }
        if (b < 0 || k < 0 || k >= 32) {                         /* full, or no block yet: open an extra one */
            if (tq_xblock_(t, c) != 0) { gen_free_(&g); return -1; }
            b = t->nblk + t->xtail[c]; k = 0;
        }
        blk = (u8 *)tq_bptr_(t, b);
        if (k < 16) for (d = 0; d < vd; ++d) { u8 *q = blk + (size_t)d * 16 + (size_t)k; *q = (u8)((*q & 0xf0) | cd[d]); }
        else for (d = 0; d < vd; ++d) { u8 *q = blk + (size_t)d * 16 + (size_t)(k - 16); *q = (u8)((*q & 0x0f) | (cd[d] << 4)); }
        if (c1) { tq_c1_row_(cd, t->dpad, t->vd1, c1); tq_put_lane_(tq_bptr1_(t, b), t->vd1, k, c1); }
        if (b >= t->nblk) t->xslot[(size_t)(b - t->nblk) * 32 + (size_t)k] = row;
        else if (t->slot_row) t->slot_row[(size_t)b * 32 + (size_t)k] = row;
        t->n = row + 1;
        t->fit_add += (f64)a->d2[i];
        ++t->nadd;
        if (t->csize && t->nc > 1) { t->csq += 2.0 * t->csize[c] + 1.0; ++t->csize[c]; t->csum += 1.0; }
    }
    gen_free_(&g);
    return 0;
}

/* what an index that came from a file lacks (neither is saved): fit0 - the mean squared distance of up to 4096 of its
 * rows to their centroids, read through rows - and the cluster sizes, with skew0 taken from them */
void tq_fit0(Tq *t, const TqCol *rows)
{
    i64 ns = (i64)t->nblk * 32, sl, step = ns > 4096 ? ns / 4096 : 1;
    f64 sum = 0.0;
    i32 got = 0, c = 0;
    f32 *rb;
    Gen g;
    if (t->nc <= 1 || !t->slot_row || ns == 0) return;
    gen_init_(&g);
    rb = (f32 *)gen_alloc_(&g, (size_t)t->dim * sizeof(f32));
    for (sl = 0; rb && sl < ns; sl += step) {
        i32 r = t->slot_row[sl], b = (i32)(sl / 32);
        if (r < 0 || r >= t->n) continue;
        while (c < t->nc - 1 && t->cstart[c + 1] <= b) ++c;    /* slots in order: the cluster only moves forward */
        sum += 2.0 - 2.0 * (f64)vs_dot(tq_row_(rows, r, rb), t->cent + (size_t)c * t->dim, t->dim);
        ++got;
    }
    t->fit0 = got ? (f32)(sum / got) : 0.0f;
    gen_free_(&g);
    if (!t->csize) { tq_count_(t); t->skew0 = tq_skew(t); }
}

static i32 tq_pop64_(u64 x)
{
#if defined(__GNUC__)
    return __builtin_popcountll(x);                     /* ll: long is 32-bit on 32-bit CPUs */
#else
    i32 c = 0;
    for (; x; x &= x - 1u) ++c;
    return c;
#endif
}

/* The caller dropped its dead rows and moved the rest down, order kept. Every slot follows its row: row r becomes
 * r minus the dead rows before it, a dead row's slot goes free, and the index covers nlive rows. dead: the bitset
 * from before (rows [0, t->n)); pre[w]: dead rows in its words [0, w). A flat index gets an explicit slot_row. */
void tq_remap(Tq *t, const u64 *dead, const i32 *pre, i32 nlive)
{
    i64 s, ns = (i64)t->nblk * 32;
    if (!t->slot_row) {
        t->slot_row = (i32 *)gen_alloc_(&t->mem, (size_t)(ns ? ns : 1) * sizeof(i32));
        if (!t->slot_row) return;                                /* no memory: the index stays flat and stale; the caller rebuilds */
        for (s = 0; s < ns; ++s) t->slot_row[s] = s < (i64)t->n ? (i32)s : -1;
        t->slot_own = t->mapped;
    }
    for (s = 0; s < ns + (i64)t->xn * 32; ++s) {
        i32 *p = s < ns ? &t->slot_row[s] : &t->xslot[s - ns], r = *p;
        if (r < 0 || r >= t->n || TQ_SKIP(dead, r)) { *p = -1; continue; }
        *p = r - pre[r >> 6] - tq_pop64_(dead[r >> 6] & (((u64)1 << (r & 63)) - 1u));
    }
    t->n = nlive;
    if (t->csize) tq_count_(t);                                  /* the dead rows left their clusters */
}
