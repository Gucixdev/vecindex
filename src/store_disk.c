/* store_disk.c - saving and loading the .vxi file, and the journal of changes since the last save.
 * Part of store.c, which includes it at this point; not compiled on its own. */

/* ------------------------------------------------------------------ *
 * persistence - .vxi: metadata + compacted rows + the built index    *
 * ------------------------------------------------------------------ */

/* .vxi scalars go through engine.c's little-endian helpers (engs_wle_ /
 * engs_rle_, in scope because store.c includes engine.c with
 * ENGINE_SERIALIZE): 4-byte header words, then the EngGoal fields one by
 * one (never the raw struct - padding/enum width is not portable), then
 * the row arrays as LE f32 / i64. A big-endian host byte-swaps here. */
#define STORE_W32(f, p, n) engs_wle_((f), (p), 4, (n))
#define STORE_R32(f, p, n) engs_rle_((f), (p), 4, (n))
#define STORE_W64(f, p, n) engs_wle_((f), (p), 8, (n))
#define STORE_R64(f, p, n) engs_rle_((f), (p), 8, (n))

static int store_goal_w_(FILE *f, const EngGoal *g)
{
    i32 w[7]; union { f32 fv; i32 iv; } tr;
    w[0] = (i32)g->mode;    w[1] = g->k;         w[2] = g->max_code_bytes;
    w[3] = g->eval_queries; w[4] = g->n_iters;   w[5] = (i32)g->seed;
    w[6] = g->max_candidates;
    tr.fv = g->target_recall;
    return STORE_W32(f, w, 7) || STORE_W32(f, &tr.iv, 1) ? -1 : 0;
}
static int store_goal_r_(FILE *f, EngGoal *g)
{
    i32 w[7]; union { f32 fv; i32 iv; } tr;
    if (STORE_R32(f, w, 7) || STORE_R32(f, &tr.iv, 1)) return -1;
    g->mode = (EngMode)w[0]; g->k = w[1];  g->max_code_bytes = w[2];
    g->eval_queries = w[3];  g->n_iters = w[4]; g->seed = (u32)w[5];
    g->max_candidates = w[6];
    g->target_recall = tr.fv;
    return 0;
}

/* ---- the journal: PATH.j = 'VXJ3' dim(i32) stamp(u64) and their CRC-32C, then frames 'G' len(u32) crc(u32)
 * records[len]: one record (A/P/D/M/R from store_jrec_, Q from store_jrec_text_) or a whole batch (store_batch.c), in
 * the host's byte order. The stamp is the one in the header of the file it belongs to (v7+), so a journal is replayed
 * only onto that file. A frame whose CRC fails with more journal after it is damage, not a torn tail: replay stops
 * there as for a tail, and a copy of the journal is kept as PATH.j.damaged. 'VXJ2' (no header CRC, records not
 * framed) still replays. A file from before v7 has no stamp: its journal is 'VXJ1' base_live(i32) base_bytes(i64)
 * dim(i32), matched on rows and file size. ---- */
#define STORE_JMAGIC  0x314a5856u
#define STORE_JMAGIC2 0x324a5856u
#define STORE_JMAGIC3 0x334a5856u

/* flush a stdio file all the way to the disk */
static i32 store_fsync_(FILE *f)
{
    if (fflush(f) != 0) return -1;
    return fsync(fileno(f)) == 0 ? 0 : -1;
}

/* make a rename durable: fsync the directory that holds `path` */
static void store_fsync_dir_(const char *path)
{
    char *d = (char *)malloc(strlen(path) + 2), *sl;
    int fd;
    strcpy(d, path);
    sl = strrchr(d, '/');
    if (sl) { if (sl == d) sl[1] = 0; else *sl = 0; } else strcpy(d, ".");
    fd = open(d, O_RDONLY);
    if (fd >= 0) { fsync(fd); close(fd); }
    free(d);
}

static i64 store_file_size_(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0 ? (i64)st.st_size : -1;
}

/* path + suffix, malloc'd */
static char *store_jname_(const char *path, const char *suffix)
{
    char *j = (char *)malloc(strlen(path) + strlen(suffix) + 1);
    strcpy(j, path); strcat(j, suffix);
    return j;
}

/* a stamp for a new file: 8 bytes of /dev/urandom, or the clock, the pid and a counter mixed when there is none;
 * never 0 (0 means "no stamp") */
static u64 store_stamp_new_(void)
{
    static u32 ctr;
    u64 v = 0;
    FILE *r = fopen("/dev/urandom", "rb");
    if (r) { if (fread(&v, 8, 1, r) != 1) v = 0; fclose(r); }
    if (!v) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        v = h64_mix_(((u64)ts.tv_sec << 32) ^ (u64)ts.tv_nsec ^ ((u64)getpid() << 20) ^ (u64)++ctr);
    }
    return v ? v : 1;
}

/* a journal's header: VXJ3 for a stamped file, VXJ1 for an older one; 0 or -1 */
static i32 store_jhead_(FILE *f, i32 dim, u64 stamp, i32 live, i64 bytes)
{
    u32 m = stamp ? STORE_JMAGIC3 : STORE_JMAGIC;
    if (fwrite(&m, 4, 1, f) != 1) return -1;
    if (stamp) {
        u8 b[12];
        u32 crc;
        memcpy(b, &dim, 4); memcpy(b + 4, &stamp, 8);
        crc = store_crc32c_(b, 12);
        return fwrite(b, 1, 12, f) == 12 && fwrite(&crc, 4, 1, f) == 1 ? 0 : -1;
    }
    return fwrite(&live, 4, 1, f) == 1 && fwrite(&bytes, 8, 1, f) == 1 && fwrite(&dim, 4, 1, f) == 1 ? 0 : -1;
}

/* the file at `path` was just written (or loaded): start a fresh journal against it */
static void store_journal_reset_(RagStore *s, const char *path)
{
    char *jn = store_jname_(path, ".j");
    if (s->jf) { fclose(s->jf); s->jf = (FILE *)0; }
    free(s->jpath); s->jpath = (char *)0;
    s->jf = fopen(jn, "wb");
    ++s->jgen; s->jdur = 0;
    if (s->jf) {
        if (store_jhead_(s->jf, s->dim, s->stamp, s->count, store_file_size_(path)) == 0 && store_fsync_(s->jf) == 0)
            s->jdur = (i64)ftello(s->jf);
        s->jpath = store_jname_(path, "");
    }
    free(jn);
}

/* One journal record (its op already read from f) applied to the store being loaded: 0, or -1 when it is torn or
 * not a record. vec is [dim] scratch, *txt a buffer of *txcap bytes grown as texts need. */
static i32 store_jrec_play_(RagStore *s, FILE *f, u8 op, f32 *vec, char **txt, i32 *txcap)
{
    i64 id, meta;
    if (op == 'Q') {                                             /* an upsert with its text */
        u32 n;
        if (fread(&id, 8, 1, f) != 1 || fread(&meta, 8, 1, f) != 1 || fread(vec, sizeof(f32), (size_t)s->dim, f) != (size_t)s->dim
            || fread(&n, 4, 1, f) != 1 || n > (u32)STORE_TXMAX) return -1;
        if (n > (u32)*txcap) {
            char *t = (char *)realloc(*txt, (size_t)n);
            if (!t) return -1;
            *txt = t; *txcap = (i32)n;
        }
        if (n && fread(*txt, 1, (size_t)n, f) != (size_t)n) return -1;
        store_put_text(s, vec, id, meta, *txt, (i32)n);
    } else if (op == 'A' || op == 'P' || op == 'U') {
        if (fread(&id, 8, 1, f) != 1 || fread(&meta, 8, 1, f) != 1 || fread(vec, sizeof(f32), (size_t)s->dim, f) != (size_t)s->dim) return -1;
        if (op == 'A') store_add_meta(s, vec, id, meta); else if (op == 'U') store_update(s, vec, id); else store_put(s, vec, id, meta);
    } else if (op == 'D') {
        if (fread(&id, 8, 1, f) != 1) return -1;
        store_delete(s, id);
    } else if (op == 'M' || op == 'R' || op == 'N' || op == 'S' || op == 'Y') {
        if (fread(&id, 8, 1, f) != 1 || fread(&meta, 8, 1, f) != 1) return -1;
        if (op == 'M') store_set_meta(s, id, meta);
        else if (op == 'N') store_set_num(s, id, meta);
        else if (op == 'S') store_set_ns(s, id, meta);
        else if (op == 'Y') store_delete_ns(s, id);
        else store_delete_range(s, id, meta);
    } else if (op == 'L') {                                      /* a label: its bit, name and parents */
        u8 b;
        char nm[STORE_LABEL_NAME];
        u64 par;
        if (fread(&b, 1, 1, f) != 1 || fread(nm, 1, STORE_LABEL_NAME, f) != STORE_LABEL_NAME || fread(&par, 8, 1, f) != 1) return -1;
        nm[STORE_LABEL_NAME - 1] = 0;
        store_label_set_(s, (i32)b, nm, par);
    } else { s->jnewer = op; return -3; }                        /* the frame's CRC vouched for it: a newer build's */
    return 0;
}

/* A frame, 'G' len crc records[len] (its op already read): applied only when all of it is there and its CRC-32C
 * matches - nothing of a torn or damaged one. 0; -1 torn (the journal ends inside it); -2 damaged (all of it there,
 * the CRC or a record wrong). */
static i32 store_jgroup_play_(RagStore *s, FILE *f, f32 *vec, char **txt, i32 *txcap)
{
    u32 n, crc;
    u8 *g, op;
    FILE *m;
    i32 r = 0;
    if (fread(&n, 4, 1, f) != 1 || fread(&crc, 4, 1, f) != 1) return -1;
    if ((size_t)n > STORE_BATCH_MAX) return -2;
    if (n == 0) return 0;
    if (!(g = (u8 *)malloc((size_t)n))) return -1;
    if (fread(g, 1, (size_t)n, f) != (size_t)n) { free(g); return -1; }
    if (store_crc32c_(g, (size_t)n) != crc) { free(g); return -2; }
    if (!(m = fmemopen(g, (size_t)n, "rb"))) { free(g); return -1; }
    while (fread(&op, 1, 1, m) == 1) {                          /* the CRC vouched for the bytes: these are whole */
        i32 q = store_jrec_play_(s, m, op, vec, txt, txcap);
        if (q != 0) { r = q == -3 ? -3 : -2; break; }
    }
    fclose(m);
    free(g);
    return r;
}

/* 1 when nothing but zeros follows in f: a tail the file system extended without the bytes landing, not damage */
static i32 store_jzeros_(FILE *f)
{
    int c;
    while ((c = fgetc(f)) != EOF) if (c) return 0;
    return 1;
}

/* Replays the journal file jn onto a freshly loaded store when it belongs to the file at path. Returns the length of
 * its good part (a torn last record or batch is not in it), -1 when it belongs to another file or cannot be read, -2
 * when its header is damaged, -3 when a whole record is of a kind this build does not know (s->jnewer: a newer build
 * wrote it). *dmg gets 1 when it stopped at a damaged frame with more journal after it. */
static i64 store_journal_play_(RagStore *s, const char *path, const char *jn, i32 *dmg)
{
    FILE *f = fopen(jn, "rb");
    u32 magic;
    i32 match = 0, txcap = 0, framed = 0;
    i64 good;
    f32 *vec;
    char *txt = (char *)0;
    *dmg = 0;
    if (!f) return -1;
    if (fread(&magic, 4, 1, f) == 1) {
        if (magic == STORE_JMAGIC3) {
            u8 b[12];
            u32 crc;
            i32 dim; u64 stamp;
            if (fread(b, 1, 12, f) == 12 && fread(&crc, 4, 1, f) == 1) {
                if (store_crc32c_(b, 12) != crc) { fclose(f); return -2; }
                memcpy(&dim, b, 4); memcpy(&stamp, b + 4, 8);
                match = s->stamp && dim == s->dim && stamp == s->stamp;
                framed = 1;
            }
        } else if (magic == STORE_JMAGIC2) {
            i32 dim; u64 stamp;
            match = s->stamp && fread(&dim, 4, 1, f) == 1 && fread(&stamp, 8, 1, f) == 1 && dim == s->dim && stamp == s->stamp;
        } else if (magic == STORE_JMAGIC) {
            i32 live, dim; i64 bytes;
            match = !s->stamp && fread(&live, 4, 1, f) == 1 && fread(&bytes, 8, 1, f) == 1 && fread(&dim, 4, 1, f) == 1
                 && live == s->count && dim == s->dim && bytes == store_file_size_(path);
        }
    }
    if (!match) { fclose(f); return -1; }
    vec = (f32 *)malloc((size_t)s->dim * sizeof(f32));
    good = (i64)ftello(f);
    for (;;) {
        u8 op;
        i32 r;
        if (fread(&op, 1, 1, f) != 1) break;
        if (op == 'G') r = store_jgroup_play_(s, f, vec, &txt, &txcap);
        else { r = framed ? -2 : store_jrec_play_(s, f, op, vec, &txt, &txcap); if (r == -3) r = -1; }   /* unframed: torn */
        if (r == -3) { fclose(f); free(vec); free(txt); return -3; }   /* a newer build's record: touch nothing */
        if (r != 0) { if (r == -2 && !store_jzeros_(f)) *dmg = 1; break; }
        good = (i64)ftello(f);
    }
    fclose(f);
    free(vec); free(txt);
    return good;
}

/* Replays PATH.j onto a freshly loaded store, or PATH.jn when a snapshot replaced the file and its journal had not
 * moved over yet (then it moves now). Afterwards the journal stays open for appending; when none belongs to this
 * file, a fresh one starts. */
/* keeps a damaged journal as NAME.damaged (or .damaged.1 to .9, the first free one): moved away, or copied when it stays
 * in use (cut back to its good part right after) */
static void store_jkeep_(const char *jn, i32 move)
{
    char *d = (char *)malloc(strlen(jn) + 16), buf[8192];
    struct stat st;
    FILE *in, *out;
    size_t n;
    i32 i;
    if (!d) return;
    for (i = 0; i < 10; ++i) {
        if (i) sprintf(d, "%s.damaged.%d", jn, (int)i); else sprintf(d, "%s.damaged", jn);
        if (stat(d, &st) != 0) break;
    }
    if (i < 10) {
        if (move) { if (rename(jn, d) == 0) store_fsync_dir_(jn); }
        else if ((in = fopen(jn, "rb")) != (FILE *)0) {
            if ((out = fopen(d, "wb")) != (FILE *)0) {
                while ((n = fread(buf, 1, sizeof(buf), in)) > 0) if (fwrite(buf, 1, n, out) != n) break;
                store_fsync_(out);
                fclose(out);
            }
            fclose(in);
        }
    }
    free(d);
}

static void store_journal_replay_(RagStore *s, const char *path)
{
    char *jn = store_jname_(path, ".j"), *jn2 = store_jname_(path, ".jn");
    const char *use = jn;
    i64 good, g2;
    i32 dmg = 0;
    if (s->ro) {                                                 /* read the journals as they are; move, cut, keep nothing */
        good = store_journal_play_(s, path, jn, &dmg);
        if (good == -3) { free(jn); free(jn2); return; }
        if (good == -2 || dmg) s->jdamaged = 1;
        if (good < 0 && s->stamp && (store_journal_play_(s, path, jn2, &dmg) == -2 || dmg)) s->jdamaged = 1;
        free(jn); free(jn2);
        return;
    }
    good = store_journal_play_(s, path, jn, &dmg);
    if (good == -3) { free(jn); free(jn2); return; }             /* a newer build's journal: refused, nothing cut or moved */
    if (good == -2) { store_jkeep_(jn, 1); s->jdamaged = 1; good = -1; }   /* its header: moved aside, never overwritten */
    if (good < 0 && s->stamp) {
        if ((g2 = store_journal_play_(s, path, jn2, &dmg)) == -3) { free(jn); free(jn2); return; }
        if (g2 == -2) { store_jkeep_(jn2, 1); s->jdamaged = 1; }
        else if (g2 >= 0) {
            good = g2;
            if (rename(jn2, jn) == 0) store_fsync_dir_(path);
            else use = jn2;                                      /* append where it is: the next load finds it there too */
        }
    } else if (good >= 0)
        remove(jn2);                                             /* left by a snapshot that never finished: it matches nothing */
    if (good >= 0) {
        if (dmg) { store_jkeep_(use, 0); s->jdamaged = 1; }    /* all of it, before the cut below drops the damaged rest */
        /* drop a torn tail, then append after the good part. If the tail cannot be dropped, run without a journal (as
         * when fopen fails): records appended behind a torn one would never be replayed. */
        if (truncate(use, (off_t)good) == 0) {
            s->jf = fopen(use, "ab");
            ++s->jgen; s->jdur = good;                           /* read back from the disk: on it */
            if (s->jf) s->jpath = store_jname_(path, "");
        }
    } else
        store_journal_reset_(s, path);
    free(jn); free(jn2);
}

/* a store whose journal holds a newer build's record: refused, with the reason; its files as they were */
static RagStore *store_jnewer_(RagStore *s)
{
    if (!s->jnewer) return s;
    sprintf(store_lerr_, "its journal was written by a newer vecindex (a change of kind '%c' this build does not know): "
                         "open it with that version or later", (s->jnewer >= 33 && s->jnewer < 127) ? s->jnewer : '?');
    store_destroy(s);
    return (RagStore *)0;
}

/* v5+ layout: every array starts on a 64-byte boundary, so store_load can map the file and point straight into it
 * (no copy, pages come in when first touched). The scan index (tq) is saved as well: a load is O(header), not a
 * rebuild. */
#define STORE_AL(x) (((size_t)(x) + 63u) & ~(size_t)63u)
static int store_pad_(FILE *f)
{
    static const u8 z[64];
    i64 pos = (i64)ftello(f);
    size_t pad;
    if (pos < 0) return -1;
    pad = STORE_AL(pos) - (size_t)pos;
    return pad && fwrite(z, 1, pad, f) != pad ? -1 : 0;
}

/* rows [0, n) of a column, chunk by chunk */
static int store_col_w_(FILE *f, const SCol *c, i64 n)
{
    i32 i;
    size_t per = c->elem;
    for (i = 0; (i64)i << STORE_CSHIFT < n; ++i) {
        i64 rows = n - ((i64)i << STORE_CSHIFT);
        if (rows > STORE_CROWS) rows = STORE_CROWS;
        if (engs_wle_(f, c->chunk[i], per, (size_t)rows * (c->esz / per))) return -1;
    }
    return 0;
}

/* rows [0, n) of a column except the tombstoned ones (dead NULL: none), in runs of consecutive live rows */
static int store_col_w_live_(FILE *f, const SCol *c, i64 n, const u64 *dead)
{
    i64 r = 0, e, lim;
    size_t per = c->elem, k = c->esz / per;
    if (!dead) return store_col_w_(f, c, n);
    while (r < n) {
        if ((dead[r >> 6] >> (r & 63)) & 1u) { ++r; continue; }
        lim = ((r >> STORE_CSHIFT) + 1) << STORE_CSHIFT;         /* a run stays inside its chunk */
        if (lim > n) lim = n;
        for (e = r + 1; e < lim && !((dead[e >> 6] >> (e & 63)) & 1u); ++e) {}
        if (engs_wle_(f, SCOL_AT(c, r), per, (size_t)(e - r) * k)) return -1;
        r = e;
    }
    return 0;
}

/* The blocks go out cluster by cluster - the built ones, then those added since - so the file holds a plain index
 * again (a flat index without slot_row adds rows in order, so its blocks stay in row order) */
static int store_tq_w_(FILE *f, const Tq *t)
{
    i32 h[8], c, nb, i, nc1 = t->nc > 1 ? t->nc : 1, *bl, *cs, bad = 0;
    size_t bb = (size_t)t->vd * 16;
    h[0] = t->dim; h[1] = t->dpad; h[2] = t->n; h[3] = t->nblk + t->xn; h[4] = t->nc; h[5] = t->nprobe;
    h[6] = t->slot_row != (i32 *)0;
    h[7] = t->codes1 ? 5 : (t->bits == 4 ? 0 : t->bits);        /* 0 = 4-bit (older files); 5 = 4-bit, then the 1-bit codes */
    if (store_pad_(f) || STORE_W32(f, h, 8)
        || store_pad_(f) || STORE_W32(f, t->s1, (size_t)t->dpad)
        || store_pad_(f) || STORE_W32(f, t->s2, (size_t)t->dpad)
        || store_pad_(f) || STORE_W32(f, t->perm, (size_t)t->dpad)) return -1;
    bl = (i32 *)malloc((size_t)(h[3] > 0 ? h[3] : 1) * sizeof(i32));
    cs = (i32 *)malloc((size_t)(nc1 + 1) * sizeof(i32));
    for (nb = 0, c = 0; c < nc1; ++c) { cs[c] = nb; nb += tq_cblocks_(t, c, bl + nb); }
    cs[nc1] = nb;
    if (t->nc > 1 && (store_pad_(f) || STORE_W32(f, t->cent, (size_t)t->nc * (size_t)t->dim)
                      || store_pad_(f) || STORE_W32(f, cs, (size_t)t->nc + 1))) bad = 1;
    if (!bad && h[6]) {
        bad = store_pad_(f) != 0;
        for (i = 0; i < nb && !bad; ++i)
            bad = STORE_W32(f, bl[i] < t->nblk ? t->slot_row + (size_t)bl[i] * 32 : t->xslot + (size_t)(bl[i] - t->nblk) * 32, 32) != 0;
    }
    if (!bad) bad = store_pad_(f) != 0;
    for (i = 0; i < nb && !bad; ++i) bad = fwrite(tq_bptr_(t, bl[i]), 1, bb, f) != bb;
    if (t->codes1) {                                             /* the cascade's 1-bit codes, blocks in the same order */
        size_t b1 = (size_t)t->vd1 * 16;
        if (!bad) bad = store_pad_(f) != 0;
        for (i = 0; i < nb && !bad; ++i) bad = fwrite(tq_bptr1_(t, bl[i]), 1, b1, f) != b1;
    }
    free(bl); free(cs);
    return bad ? -1 : 0;
}

/* the texts of rows [0, n) except the tombstoned ones: the heap's length (u64), every written row's offset | length
 * << 40 with the offsets of the new, gap-free heap (u64, little-endian), then the heap */
static int store_text_w_(FILE *f, const RagStore *s, i64 n, const u64 *dead)
{
    u64 buf[512], len = 0, at = 0;
    i32 nb = 0;
    i64 r;
    char *t;
    for (r = 0; r < n; ++r) if (!(dead && ((dead[r >> 6] >> (r & 63)) & 1u))) len += (u64)STORE_TXLEN(STORE_TXO(s, r));
    if (store_pad_(f) || engs_wle_(f, &len, 8, 1) || store_pad_(f)) return -1;
    for (r = 0; r < n; ++r) {
        u64 v;
        if (dead && ((dead[r >> 6] >> (r & 63)) & 1u)) continue;
        v = STORE_TXO(s, r);
        buf[nb++] = v ? at | ((u64)(u32)STORE_TXLEN(v) << 40) : 0;
        at += (u64)STORE_TXLEN(v);
        if (nb == 512) { if (engs_wle_(f, buf, 8, 512)) return -1; nb = 0; }
    }
    if ((nb && engs_wle_(f, buf, 8, (size_t)nb)) || store_pad_(f)) return -1;
    if (!(t = (char *)malloc((size_t)STORE_TXMAX + 1))) return -1;
    for (r = 0; r < n; ++r) {
        u64 v;
        if (dead && ((dead[r >> 6] >> (r & 63)) & 1u)) continue;
        if (!(v = STORE_TXO(s, r))) continue;
        store_tx_copy_(&s->tx, v, t, STORE_TXLEN(v));
        if (fwrite(t, 1, (size_t)STORE_TXLEN(v), f) != (size_t)STORE_TXLEN(v)) { free(t); return -1; }
    }
    free(t);
    return 0;
}

/* the label table: for each of the 64 bits its name (64 bytes, 0-padded; empty: free) and its parents' bits (u64) */
static int store_labels_w_(FILE *f, const RagStore *s)
{
    i32 b;
    if (store_pad_(f)) return -1;
    for (b = 0; b < STORE_LABELS; ++b)
        if (fwrite(s->lab_name[b], 1, STORE_LABEL_NAME, f) != STORE_LABEL_NAME || STORE_W64(f, &s->lab_par[b], 1)) return -1;
    return 0;
}

static i32 store_labels_r_(RagStore *s, FILE *f, size_t off, size_t len)
{
    i32 b;
    if (off + (size_t)STORE_LABELS * (STORE_LABEL_NAME + 8) > len || fseeko(f, (off_t)off, SEEK_SET) != 0) return -1;
    for (b = 0; b < STORE_LABELS; ++b) {
        if (fread(s->lab_name[b], 1, STORE_LABEL_NAME, f) != STORE_LABEL_NAME || STORE_R64(f, &s->lab_par[b], 1)) return -1;
        s->lab_name[b][STORE_LABEL_NAME - 1] = 0;
        if (s->lab_name[b][0]) ++s->nlab;
    }
    return 0;
}

/* Writes the store to tmp - the data, then its block digests and parity (store_sum.c) - flushes it to the disk,
 * renames it over path and makes the rename durable: 0 or -1. Takes
 * no lock - the caller holds the write lock (store_save, after its compaction), or is the snapshot child, alone with
 * a private image of the store. Tombstoned rows are left out; only the child (child = 1) may then renumber the scan
 * index to match, since that edits the index in place (its own copy). */
static i32 store_write_file_(RagStore *s, const char *path, const char *tmp, u64 stamp, i32 child)
{
    FILE *f;
    i32 hdr[4], ext[4], live = s->count, all_in, bad;
    i64 w, nw = ((i64)s->count + 63) >> 6;
    const u64 *dead = (const u64 *)0;
    u8 kind;
    for (w = 0; w < nw; ++w) {                                   /* tombstones among rows [0, count) */
        u64 x = s->dead[w];
        if (w == nw - 1 && (s->count & 63)) x &= ((u64)1 << (s->count & 63)) - 1u;
        live -= tq_pop64_(x);
    }
    all_in = s->built_upto == s->count;                          /* every row is in the index */
    if (live != s->count) {
        if (!child) return -1;                                   /* store_save compacts first */
        dead = s->dead;
        if (s->tq && all_in && s->tq->n == s->count) {           /* renumber the scan index as a compaction would */
            i32 *pre = (i32 *)malloc((size_t)(nw + 1) * sizeof(i32)), acc = 0;
            if (!pre) return -1;
            for (w = 0; w <= nw; ++w) { pre[w] = acc; if (w < nw) acc += tq_pop64_(s->dead[w]); }
            tq_remap(s->tq, s->dead, pre, live);
            free(pre);
        }
    }
    /* 0: no index (a load rebuilds it), 1: the engine index, 2: the scan index */
    kind = (u8)(s->idx && all_in && !dead ? 1 : (s->tq && all_in && s->tq->n == live ? 2 : 0));
    f = fopen(tmp, "w+b");                                       /* the old file stays intact until the new one is complete */
    if (!f) return -1;
    hdr[0] = (i32)STORE_MAGIC; hdr[2] = s->dim; hdr[3] = live;
    ext[0] = (s->vbytes == 1 ? 1 : 0) | (s->has_text ? 2 : 0) | (s->nlab ? 4 : 0) | (s->has_num ? 8 : 0) | (s->has_ns ? 16 : 0) | (s->has_text ? 32 : 0); ext[1] = s->metric;   /* v9: 0 cosine, 1 dot, 2 L2 */
    hdr[1] = (ext[0] & ~STORE_FLAGS_V9) ? STORE_VERSION10 : STORE_VERSION9;   /* the oldest version that holds it */
    ext[2] = (i32)(u32)(stamp & 0xffffffffu); ext[3] = (i32)(u32)(stamp >> 32);
    bad = STORE_W32(f, hdr, 4) || STORE_W32(f, ext, 4) || store_goal_w_(f, &s->goal)
       || store_pad_(f) || store_col_w_live_(f, &s->vec, s->count, dead)
       || (s->vbytes == 1 && (store_pad_(f) || store_col_w_live_(f, &s->vsc, s->count, dead)))
       || store_pad_(f) || store_col_w_live_(f, &s->idc, s->count, dead)
       || store_pad_(f) || store_col_w_live_(f, &s->metac, s->count, dead)
       || store_pad_(f) || store_col_w_live_(f, &s->rck, s->count, dead)
       || (s->metric && (store_pad_(f) || store_col_w_live_(f, &s->nrm, s->count, dead)))
       || (s->has_text && store_text_w_(f, s, s->count, dead))
       || (s->nlab && store_labels_w_(f, s))
       || (s->has_num && (store_pad_(f) || store_col_w_live_(f, &s->numc, s->count, dead)))
       || (s->has_ns && (store_pad_(f) || store_col_w_live_(f, &s->nsc, s->count, dead)))
       || (s->has_text && (store_pad_(f) || store_col_w_live_(f, &s->tck, s->count, dead)))
       || fwrite(&kind, 1, 1, f) != 1
       || (kind == 1 && engine_save(s->idx, f))
       || (kind == 2 && store_tq_w_(f, s->tq));
    if (!bad) {                                                  /* the digests and parity of all of it */
        i64 D = (i64)ftello(f);
        bad = D <= 0 || ssum_write_file_(f, tmp, (u64)D, (u32)s->sum_k, (u32)s->sum_m, child ? 1 : 8) != 0;
    }
    if (bad || store_fsync_(f) != 0) { fclose(f); remove(tmp); return -1; }
    if (fclose(f) != 0 || rename(tmp, path) != 0) { remove(tmp); return -1; }
    store_fsync_dir_(path);
    return 0;
}

i32 store_save(RagStore *s, const char *path)
{
    char *tmp;
    i32 live, had_idx, r;
    u64 stamp;
    if (s->ro) return -1;
    stamp = store_stamp_new_();
    store_lens_finish_(s);                                       /* the lens: every block checked before it is copied */

    pthread_mutex_lock(&s->snap_mu);                             /* never beside a snapshot child: it may be reading rows */
    store_snap_reap_(s, 1);                                      /* the compaction below moves, and writing the same file */
    pthread_mutex_lock(&s->rb_mu);                               /* a compaction must not run under a live rebuild */
    pthread_mutex_lock(&s->mt_mu);
    store_index_pending_(s, store_ncpu_(s));                      /* the pending tail into the index, searches going on */
    pthread_rwlock_wrlock(&s->lock);
    store_index_locked_(s);                                      /* and what came in meanwhile: the file gets the index */
    had_idx = s->idx != (EngIndex *)0;
    live = store_compact_(s);                                    /* the scan index follows its rows */
    store_tx_compact_(s);                                        /* and the texts of the rows that went */
    if (had_idx && !s->idx && live > 0) { store_engine_build_(s, live); s->built_upto = live; }   /* an engine index cannot */
    tmp = store_jname_(path, ".tmp");
    store_advise_(s, MADV_SEQUENTIAL);                           /* the save streams every row: read ahead */
    r = store_write_file_(s, path, tmp, stamp, 0);
    store_advise_(s, MADV_RANDOM);
    free(tmp);
    if (r == 0) { s->stamp = stamp; store_journal_reset_(s, path); }
    pthread_rwlock_unlock(&s->lock);
    pthread_mutex_unlock(&s->mt_mu);
    pthread_mutex_unlock(&s->rb_mu);
    pthread_mutex_unlock(&s->snap_mu);
    return r;
}

/* Cheap durable save: the first call (or one after the file changed) writes the full file; later calls only flush the
 * journal of what was added/deleted since, so the cost tracks the change, not the store size. The write lock is held
 * while the stdio buffers go to the kernel, not through the fsync. store_load replays the journal. 0 on success. */
i32 store_sync(RagStore *s, const char *path)
{
    if (s->ro) return -1;
    if (s->jf && s->jpath && strcmp(s->jpath, path) == 0) {
        int fd[2], i;
        i32 r = 0;
        u64 gen;
        i64 sz[2];
        fd[0] = fd[1] = -1;
        pthread_rwlock_wrlock(&s->lock);                         /* adders write these buffers under the same lock */
        if (fflush(s->jf) != 0) r = -1; else fd[0] = dup(fileno(s->jf));
        if (s->jf2) { if (fflush(s->jf2) != 0) r = -1; else fd[1] = dup(fileno(s->jf2)); }
        gen = store_jmark_(s, sz);
        pthread_rwlock_unlock(&s->lock);
        for (i = 0; i < 2; ++i)
            if (fd[i] >= 0) { if (fsync(fd[i]) != 0) r = -1; close(fd[i]); }
        if (r == 0) store_jdur_(s, gen, sz);
        return r;
    }
    return store_save(s, path);
}

/* ---- background snapshots ---- */

/* the stamp in the header of the file at path (0: no stamp, or not a v7 file) */
static u64 store_file_stamp_(const char *path)
{
    FILE *f = fopen(path, "rb");
    i32 h[8];
    u64 st = 0;
    if (!f) return 0;
    if (!STORE_R32(f, h, 8) && (u32)h[0] == STORE_MAGIC && (h[1] == STORE_VERSION7 || h[1] == STORE_VERSION8 || STORE_V9UP(h[1])))
        st = (u64)(u32)h[6] | ((u64)(u32)h[7] << 32);
    fclose(f);
    return st;
}

static i64 store_now_us_(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (i64)ts.tv_sec * 1000000 + (i64)ts.tv_nsec / 1000;
}

/* the child keeps none of the parent's descriptors (sockets, the journal): a client connection or a listening port
 * never outlives the server because a snapshot is still being written */
static void store_child_fds_(void)
{
    long fd, mx = sysconf(_SC_OPEN_MAX);
#if defined(__linux__) && defined(SYS_close_range)
    if (syscall(SYS_close_range, 3u, ~0u, 0u) == 0) return;
#endif
    if (mx < 0 || mx > 65536) mx = 65536;
    for (fd = 3; fd < mx; ++fd) close((int)fd);
}

/* With snap_mu held: reaps the snapshot child when it is done (block: waits for it). When it replaced the journal's
 * own file, the new file's journal takes over. Returns 1 while it runs, else the outcome of the last one (0 ok, -1
 * failed, 2 none yet). */
static i32 store_snap_reap_(RagStore *s, i32 block)
{
    int st = 0;
    pid_t r;
    i32 ok;
    if (s->snap_pid <= 0) return s->snap_last;
    do r = waitpid((pid_t)s->snap_pid, &st, block ? 0 : WNOHANG); while (r < 0 && errno == EINTR);
    if (r == 0) return 1;
    if (r < 0) ok = store_file_stamp_(s->snap_path) == s->snap_stamp;   /* reaped elsewhere (SIGCHLD ignored): the file tells */
    else ok = WIFEXITED(st) && WEXITSTATUS(st) == 0;
    pthread_rwlock_wrlock(&s->lock);
    if (s->jf2) {
        if (ok) {                                                /* the new file is on disk: its journal takes over */
            char *jn = store_jname_(s->snap_path, ".j");
            fflush(s->jf2);
            rename(s->jn2, jn);                                  /* if this fails, the next load finds PATH.jn */
            free(jn);
            if (s->jf) fclose(s->jf);
            s->jf = s->jf2;
            s->jdur = s->jdur2; ++s->jgen;
            free(s->jpath); s->jpath = store_jname_(s->snap_path, "");
            s->stamp = s->snap_stamp;
        } else {
            fclose(s->jf2);
            remove(s->jn2);
        }
        s->jf2 = (FILE *)0;
    }
    pthread_rwlock_unlock(&s->lock);
    if (ok && s->jn2) store_fsync_dir_(s->snap_path);
    free(s->jn2); s->jn2 = (char *)0;
    s->snap_pid = 0;
    s->snap_last = ok ? 0 : -1;
    s->snap_took_us = store_now_us_() - s->snap_t0_us;
#if defined(__GNUC__)
    __atomic_store_n(&s->snap_run, 0, __ATOMIC_RELEASE);
    __atomic_add_fetch(&s->snap_done, 1, __ATOMIC_RELEASE);
#else
    s->snap_run = 0;
    ++s->snap_done;
#endif
    return s->snap_last;
}

/* Background snapshot: the store as it is now, written to path by a child process (fork: the child sees a
 * copy-on-write image of this moment) while this process goes on answering searches and taking writes. When path is
 * the file the journal belongs to (or there is no journal yet), writes made meanwhile go to the old journal and to the
 * new file's (PATH.jn), which takes over once the file is complete (store_snapshot_poll). Pages the parent writes
 * while the child runs are copied, so memory can grow by what changes meanwhile. Returns 0 started, 1 one is running
 * (or the store is busy with a build or a save), -1 failed (fork, files). */
i32 store_snapshot(RagStore *s, const char *path)
{
    char *tmp, *jn2 = (char *)0;
    FILE *j2 = (FILE *)0;
    u64 stamp = store_stamp_new_();
    i32 home;
    pid_t pid;
    if (s->ro) return -1;
    store_lens_finish_(s);                                       /* the lens: every block checked before it is copied */
    if (pthread_mutex_trylock(&s->snap_mu) != 0) return 1;     /* a build or a save holds it: try later */
    if (store_snap_reap_(s, 0) == 1) { pthread_mutex_unlock(&s->snap_mu); return 1; }
    home = !s->jpath || strcmp(s->jpath, path) == 0;
    if (home) {                                                  /* the new file's journal, ready before the fork */
        jn2 = store_jname_(path, ".jn");
        j2 = fopen(jn2, "wb");
        if (!j2 || store_jhead_(j2, s->dim, stamp, 0, 0) || store_fsync_(j2)) {
            if (j2) { fclose(j2); remove(jn2); }
            free(jn2); pthread_mutex_unlock(&s->snap_mu);
            return -1;
        }
    }
    tmp = store_jname_(path, ".snap");
    pthread_mutex_lock(&s->mt_mu);                               /* not beside a maintenance step */
    store_index_pending_(s, store_ncpu_(s));                      /* the pending tail into the index, searches going on */
    pthread_rwlock_wrlock(&s->lock);                             /* nobody is in the middle of a change: a clean image */
    store_index_locked_(s);
    pid = fork();
    if (pid == 0) {
        store_child_fds_();
        store_bg_prio_(s);                                       /* the copy yields to the serving process */
        _exit(store_write_file_(s, path, tmp, stamp, 1) == 0 ? 0 : 1);
    }
    if (pid > 0) {
        s->jf2 = j2; s->jn2 = jn2;                               /* from here on every record goes to both journals */
        s->jdur2 = j2 ? (i64)ftello(j2) : 0; ++s->jgen;
        s->snap_pid = (i32)pid; s->snap_stamp = stamp;
        free(s->snap_path); s->snap_path = store_jname_(path, "");
        s->snap_t0_us = store_now_us_();
#if defined(__GNUC__)
        __atomic_store_n(&s->snap_run, 1, __ATOMIC_RELEASE);
#else
        s->snap_run = 1;
#endif
    }
    pthread_rwlock_unlock(&s->lock);
    pthread_mutex_unlock(&s->mt_mu);
    if (pid < 0 && j2) { fclose(j2); remove(jn2); free(jn2); }
    free(tmp);
    pthread_mutex_unlock(&s->snap_mu);
    return pid > 0 ? 0 : -1;
}

/* Reaps a finished snapshot (block: waits for a running one): 1 while it runs, 0 when the last one was written, -1
 * when it failed, 2 when there has been none. A non-blocking call never waits for a build or a save either. */
i32 store_snapshot_poll(RagStore *s, i32 block)
{
    i32 r;
    if (block) pthread_mutex_lock(&s->snap_mu);
    else if (pthread_mutex_trylock(&s->snap_mu) != 0) return 1;
    r = store_snap_reap_(s, block);
    pthread_mutex_unlock(&s->snap_mu);
    return r;
}

/* Bytes in the journal of the file at path (what a snapshot would fold into the file), or -1 when the store keeps
 * no journal for that file */
i64 store_journal_bytes(RagStore *s, const char *path)
{
    i64 n = -1;
    pthread_rwlock_rdlock(&s->lock);                             /* a save or a snapshot swaps the journal under the write lock */
    if (s->jf && s->jpath && strcmp(s->jpath, path) == 0) n = (i64)ftello(s->jf);
    pthread_rwlock_unlock(&s->lock);
    return n;
}

/* Changes made so far (adds, replaces, deletes): a snapshot is due only once this has moved */
i64 store_changes(RagStore *s)
{
    i64 m;
    pthread_rwlock_rdlock(&s->lock);
    m = s->mods;
    pthread_rwlock_unlock(&s->lock);
    return m;
}

/* Snapshots finished so far, ok or not: one more means store_snapshot_info has a new outcome to tell */
i64 store_snapshots_done(RagStore *s)
{
#if defined(__GNUC__)
    return __atomic_load_n(&s->snap_done, __ATOMIC_ACQUIRE);
#else
    return s->snap_done;
#endif
}

/* For stats, without waiting: whether a snapshot runs, how the last one ended (0 ok, -1 failed, 2 none yet) and how
 * many microseconds it took (or, while one runs, how long it has been running). */
void store_snapshot_info(RagStore *s, i32 *running, i32 *last, i64 *us)
{
    i32 run;
#if defined(__GNUC__)
    run = __atomic_load_n(&s->snap_run, __ATOMIC_ACQUIRE);
#else
    run = s->snap_run;
#endif
    if (running) *running = run;
    if (pthread_mutex_trylock(&s->snap_mu) == 0) {
        if (last) *last = s->snap_last;
        if (us) *us = run ? store_now_us_() - s->snap_t0_us : s->snap_took_us;
        pthread_mutex_unlock(&s->snap_mu);
    } else {
        if (last) *last = 1;
        if (us) *us = 0;
    }
}

/* n elements of esz bytes at byte offset off: a pointer into the mapping when there is one (no copy), else
 * read (and byte-swapped on a big-endian host) into malloc'd memory. NULL when the range is not inside the file. */
static void *store_take_(u8 *map, size_t len, FILE *f, size_t off, size_t esz, size_t n)
{
    void *p;
    if (off > len || esz * n > len - off) return (void *)0;
    if (map) return (void *)(map + off);
    p = malloc(esz * (n ? n : 1));
    if (!p) return (void *)0;
    if (fseeko(f, (off_t)off, SEEK_SET) != 0 || (n && engs_rle_(f, p, esz, n))) { free(p); return (void *)0; }
    return p;
}

/* a column of `rows` rows at byte offset off: chunks point into the mapping, or are read into fresh chunks */
/* a column the file does not have (written before it existed): zeros for its rows */
static i32 store_col_zero_(RagStore *s, SCol *c, i32 rows)
{
    i32 nch = (i32)(((i64)rows + STORE_CROWS - 1) >> STORE_CSHIFT);
    while (c->nchunk < nch) if (scol_push_(s, c)) return -1;
    return 0;
}

static i32 store_col_r_(RagStore *s, SCol *c, u8 *map, size_t len, FILE *f, size_t off, i32 rows)
{
    i32 nch = (i32)(((i64)rows + STORE_CROWS - 1) >> STORE_CSHIFT), i;
    size_t per = c->elem;
    if (off > len || (size_t)rows * c->esz > len - off) return -1;
    for (i = 0; i < nch; ++i) {
        i64 n = (i64)rows - ((i64)i << STORE_CSHIFT);
        if (n > STORE_CROWS) n = STORE_CROWS;
        if (map) {
            if (c->nchunk == c->ccap) {
                i32 cc = c->ccap ? c->ccap * 2 : 16;
                c->chunk = (u8 **)realloc(c->chunk, (size_t)cc * sizeof(u8 *)); c->how = (u8 *)realloc(c->how, (size_t)cc);
                c->foff = (i64 *)realloc(c->foff, (size_t)cc * sizeof(i64)); c->ccap = cc;
            }
            c->chunk[c->nchunk] = map + off + (size_t)i * (size_t)STORE_CROWS * c->esz;
            c->how[c->nchunk] = STORE_MAPPED; c->foff[c->nchunk] = -1;
            ++c->nchunk;
        } else {
            if (scol_push_(s, c)) return -1;
            if (fseeko(f, (off_t)(off + (size_t)i * (size_t)STORE_CROWS * c->esz), SEEK_SET) != 0
                || engs_rle_(f, c->chunk[i], per, (size_t)n * (c->esz / per))) return -1;
        }
    }
    return 0;
}

/* vz (the lens): the blocks under the index's header and small arrays checked before anything is derived from them -
 * a damaged cluster table would send a scan outside the codes */
static Tq *store_tq_r_(u8 *map, size_t len, FILE *f, size_t off, i32 rows, const SSum *vz, const char *path, i32 ro)
{
    i32 h[8];
    Tq *t;
    size_t o = STORE_AL(off);
    i32 *hp;
    if (vz && ssum_vouch_(vz, path, ro, (u64)o, 32)) return (Tq *)0;
    hp = (i32 *)store_take_(map, len, f, o, 4, 8);
    if (!hp) return (Tq *)0;
    memcpy(h, hp, sizeof(h));
    if (!map) free(hp);
    if (h[0] <= 0 || h[1] < h[0] || h[2] != rows || h[3] < 0 || h[4] < 0) return (Tq *)0;
    t = (Tq *)calloc(1, sizeof(Tq));
    t->dim = h[0]; t->dpad = h[1]; t->n = h[2]; t->nblk = h[3]; t->nc = h[4]; t->nprobe = h[5];
    t->nbuilt = t->n;                                            /* growth counts from the load (fit0: on first add) */
    t->bits = h[7] == 2 || h[7] == 1 ? h[7] : 4; t->vd = tq_vd_(t->dpad, t->bits);
    t->mapped = map != (u8 *)0;
    o = STORE_AL(o + 32);
    t->s1 = (f32 *)store_take_(map, len, f, o, 4, (size_t)t->dpad); o = STORE_AL(o + (size_t)t->dpad * 4);
    t->s2 = (f32 *)store_take_(map, len, f, o, 4, (size_t)t->dpad); o = STORE_AL(o + (size_t)t->dpad * 4);
    t->perm = (i32 *)store_take_(map, len, f, o, 4, (size_t)t->dpad); o = STORE_AL(o + (size_t)t->dpad * 4);
    if (t->nc > 1) {
        t->cent = (f32 *)store_take_(map, len, f, o, 4, (size_t)t->nc * (size_t)t->dim);
        o = STORE_AL(o + (size_t)t->nc * (size_t)t->dim * 4);
        t->cstart = (i32 *)store_take_(map, len, f, o, 4, (size_t)t->nc + 1);
        o = STORE_AL(o + ((size_t)t->nc + 1) * 4);
    }
    if (vz && ssum_vouch_(vz, path, ro, (u64)STORE_AL(off), (u64)(o - STORE_AL(off)))) { tq_free(t); return (Tq *)0; }
    if (h[6]) { t->slot_row = (i32 *)store_take_(map, len, f, o, 4, (size_t)t->nblk * 32); o = STORE_AL(o + (size_t)t->nblk * 128); }
    t->codes = (u8 *)store_take_(map, len, f, o, 1, (size_t)t->nblk * (size_t)t->vd * 16);
    if (h[7] == 5 && t->codes) {                                 /* the cascade's 1-bit codes follow */
        o = STORE_AL(o + (size_t)t->nblk * (size_t)t->vd * 16);
        t->vd1 = tq_vd_(t->dpad, 1);
        t->codes1 = (u8 *)store_take_(map, len, f, o, 1, (size_t)t->nblk * (size_t)t->vd1 * 16);
        t->c1_own = !map;
    }
    if (!t->s1 || !t->s2 || !t->perm || (t->nc > 1 && (!t->cent || !t->cstart)) || (h[6] && !t->slot_row) || !t->codes
        || (h[7] == 5 && !t->codes1)) {
        tq_free(t);
        return (Tq *)0;
    }
    tq_prepare_(t);
    if (map) {                    /* what every query scans (the 1-bit codes of a cascade, else the codes): read it now -
                                   * unless it is more than RAM is sure to hold: then a search reads its clusters */
        const u8 *sc = t->codes1 ? t->codes1 : t->codes;
        size_t pg = (size_t)sysconf(_SC_PAGESIZE), a = (size_t)sc & ~(pg - 1);
        i64 hot = (i64)t->nblk * (t->codes1 ? t->vd1 : t->vd) * 16 + (t->slot_row ? (i64)t->nblk * 128 : 0);
        if (t->nc > 1 && hot > (store_cold_min_ >= 0 ? store_cold_min_ : store_ram_half_())) t->cold = 1;
        else madvise((void *)a, (size_t)sc + (size_t)t->nblk * (size_t)(t->codes1 ? t->vd1 : t->vd) * 16 - a, MADV_WILLNEED);
    }
    return t;
}

/* the text heap of len bytes at byte offset off: blocks that point into the mapping, or read into fresh blocks.
 * New texts start on a fresh block either way. 0 or -1. */
static i32 store_heap_r_(RagStore *s, u8 *map, size_t flen, FILE *f, size_t off, u64 len)
{
    i64 nb = (i64)((len + (u64)STORE_TXB - 1) >> STORE_TXSHIFT), i;
    if (off > flen || len > (u64)(flen - off)) return -1;
    for (i = 0; i < nb; ++i) {
        size_t at = off + (size_t)(i << STORE_TXSHIFT);
        if (map) {
            if (s->tx.n == s->tx.cap) {
                i32 nc = s->tx.cap ? s->tx.cap * 2 : 16;
                u8 **b = (u8 **)realloc(s->tx.b, (size_t)nc * sizeof(u8 *));
                u8 *h;
                if (!b) return -1;
                s->tx.b = b;
                if (!(h = (u8 *)realloc(s->tx.how, (size_t)nc))) return -1;
                s->tx.how = h; s->tx.cap = nc;
            }
            s->tx.b[s->tx.n] = map + at; s->tx.how[s->tx.n] = STORE_MAPPED; ++s->tx.n;
        } else {
            size_t n = (size_t)(len - ((u64)i << STORE_TXSHIFT));
            if (n > (size_t)STORE_TXB) n = (size_t)STORE_TXB;
            if (store_tx_push_(s, &s->tx) || fseeko(f, (off_t)at, SEEK_SET) != 0 || fread(s->tx.b[i], 1, n, f) != n) return -1;
        }
    }
    s->tx.used = nb << STORE_TXSHIFT;
    return 0;
}

/* the checks of rows [0, count) computed, for a file older than v9 that does not carry them: 0, or -1 */
static i32 store_rck_fill_(RagStore *s)
{
    i32 r;
    i64 nch = ((i64)s->count + STORE_CROWS - 1) >> STORE_CSHIFT;
    while (s->rck.nchunk < nch) if (scol_push_(s, &s->rck)) return -1;
    for (r = 0; r < s->count; ++r) STORE_RCK(s, r) = store_rck_row_(s, r);
    return 0;
}

/* v5 to v9: map the file privately (copy-on-write: the store may still edit rows in place, the file is never
 * written through the mapping) and point the columns and the scan index into it. Nothing is read up front;
 * pages arrive on first touch, the id lookups are built on first use. dend: where a v9 file's data ends (0: the file's
 * end); ubad: the first byte of a block its checks could not rebuild (~0: none) - before the index the load is refused,
 * inside it the index is built again from the rows. */
static RagStore *store_load_v5_(Arena *a, const char *path, FILE *f, const i32 *hdr, i32 ro, u64 dend, u64 ubad, i32 lvl, const SSum *vz)
{
    RagStore *s;
    i32 live = hdr[3], dim = hdr[2], ext[4], v7 = hdr[1] == STORE_VERSION7 || hdr[1] == STORE_VERSION8 || STORE_V9UP(hdr[1]),
        i8rows = hdr[1] == STORE_VERSION_I8, texts = 0, drop_idx = 0;
    struct stat st;
    size_t len, o_vecs, o_vsc = 0, o_ids, o_meta, o_kind, o_txo = 0, o_heap = 0, o_rck = 0, o_nrm = 0, o_num = 0, o_ns = 0, o_tck = 0;
    u8 *map = (u8 *)0, kind = 0;
    u64 stamp = 0, hlen = 0;

    if (live < 0 || dim <= 0 || fstat(fileno(f), &st) != 0) { fclose(f); return (RagStore *)0; }
    if (ubad == 0) {                     /* the header's own block could not be rebuilt: its dimension and row count are
                                          * noise, and tests/fuzz_store.c saw one ask store_new for 2.4 GB before the
                                          * refusal further down */
        strcpy(store_lerr_, "damaged beyond repair at byte 0, in the header (a scrub lists the blocks)");
        fclose(f); return (RagStore *)0;
    }
    len = (size_t)st.st_size;
    if (dend && dend < (u64)len) len = (size_t)dend;               /* the digests and parity after it are not data */
    if ((v7 || i8rows) && STORE_R32(f, ext, 4)) { fclose(f); return (RagStore *)0; }
    if (v7) { i8rows = ext[0] & 1; texts = (hdr[1] == STORE_VERSION8 || STORE_V9UP(hdr[1])) && (ext[0] & 2); stamp = (u64)(u32)ext[2] | ((u64)(u32)ext[3] << 32); }
    else if (i8rows && ext[0] != 1) { fclose(f); return (RagStore *)0; }
    if (v7 && (ext[0] & ~STORE_FLAGS_KNOWN)) {                   /* a feature of a newer build: refused, never misread */
        sprintf(store_lerr_, "was written by a newer vecindex (features 0x%x this build does not know): open it with that version or later",
                (unsigned)(ext[0] & ~STORE_FLAGS_KNOWN));
        fclose(f); return (RagStore *)0;
    }
    s = store_new(a, dim);
    if (!s->addbuf) {                                            /* not even one row of that dimension fits in memory */
        sprintf(store_lerr_, "no memory for one row of %d dimensions", (int)dim);
        fclose(f); store_destroy(s); return (RagStore *)0;
    }
    s->ro = ro;
    if (i8rows) store_set_vec_bytes(s, 1);
    if (store_goal_r_(f, &s->goal)) { fclose(f); store_destroy(s); return (RagStore *)0; }
    o_vecs = STORE_AL((v7 || i8rows ? 32 : 16) + 32);
    if (i8rows) { o_vsc = STORE_AL(o_vecs + (size_t)live * (size_t)dim); o_ids = STORE_AL(o_vsc + (size_t)live * 4); }
    else o_ids = STORE_AL(o_vecs + (size_t)live * (size_t)dim * 4);
    o_meta = STORE_AL(o_ids + (size_t)live * 8);
    o_kind = o_meta + (size_t)live * 8;
    if (STORE_V9UP(hdr[1])) { o_rck = STORE_AL(o_kind); o_kind = o_rck + (size_t)live * 4; }   /* the rows' checks */
    if (STORE_V9UP(hdr[1]) && ext[1]) {                    /* a dot or L2 store: the rows' lengths */
        if (ext[1] < 0 || ext[1] > 2) { fclose(f); store_destroy(s); return (RagStore *)0; }
        s->metric = ext[1];
        o_nrm = STORE_AL(o_kind); o_kind = o_nrm + (size_t)live * 4;
    }
    if (texts) {                                                 /* v8: the heap's length, the offsets, the heap */
        size_t o_len = STORE_AL(o_kind);
        if (o_len + 8 > len || fseeko(f, (off_t)o_len, SEEK_SET) != 0 || STORE_R64(f, (i64 *)(void *)&hlen, 1)) {
            fclose(f); store_destroy(s); return (RagStore *)0;
        }
        o_txo = STORE_AL(o_len + 8);
        o_heap = STORE_AL(o_txo + (size_t)live * 8);
        if (o_heap > len || hlen > (u64)(len - o_heap)) { fclose(f); store_destroy(s); return (RagStore *)0; }
        o_kind = o_heap + (size_t)hlen;
        s->has_text = 1;
    }
    if (STORE_V9UP(hdr[1]) && (ext[0] & 4)) {             /* the label table */
        if ((vz && ssum_vouch_(vz, path, ro, (u64)STORE_AL(o_kind), (u64)STORE_LABELS * (STORE_LABEL_NAME + 8)))
            || store_labels_r_(s, f, STORE_AL(o_kind), len) != 0) { fclose(f); store_destroy(s); return (RagStore *)0; }
        o_kind = STORE_AL(o_kind) + (size_t)STORE_LABELS * (STORE_LABEL_NAME + 8);
    }
    if (STORE_V9UP(hdr[1]) && (ext[0] & 8)) {             /* the rows' numbers */
        o_num = STORE_AL(o_kind); o_kind = o_num + (size_t)live * 8;
        s->has_num = 1;
    }
    if (STORE_V9UP(hdr[1]) && (ext[0] & 16)) {            /* the rows' namespaces */
        o_ns = STORE_AL(o_kind); o_kind = o_ns + (size_t)live * 8;
        s->has_ns = 1;
    }
    if (STORE_V9UP(hdr[1]) && (ext[0] & 32)) {            /* the texts' CRCs */
        o_tck = STORE_AL(o_kind); o_kind = o_tck + (size_t)live * 4;
    }
    if (o_kind >= len) { fclose(f); store_destroy(s); return (RagStore *)0; }
    if (ubad != ~(u64)0) {                                       /* a block the checks could not rebuild */
        if (ubad <= (u64)o_kind) {
            sprintf(store_lerr_, "damaged beyond repair at byte %.0f, in the rows, ids or texts (a scrub lists the blocks)",
                    (double)ubad);
            fclose(f); store_destroy(s); return (RagStore *)0;
        }
        drop_idx = 1;                                            /* only the saved index: built again from the rows */
    }
    if (!eng_host_be_() && live > 0) {
        void *m = mmap((void *)0, len, PROT_READ | PROT_WRITE, MAP_PRIVATE, fileno(f), 0);
        if (m != MAP_FAILED) { map = (u8 *)m; s->map = m; s->map_len = len; }
    }
    if (live > 0) {
        if (store_col_r_(s, &s->vec, map, len, f, o_vecs, live) || store_col_r_(s, &s->idc, map, len, f, o_ids, live)
            || store_col_r_(s, &s->metac, map, len, f, o_meta, live)
            || (i8rows && store_col_r_(s, &s->vsc, map, len, f, o_vsc, live))
            || (texts && store_col_r_(s, &s->txo, map, len, f, o_txo, live))
            || (o_rck && store_col_r_(s, &s->rck, map, len, f, o_rck, live))
            || (o_nrm && store_col_r_(s, &s->nrm, map, len, f, o_nrm, live))
            || (o_num && store_col_r_(s, &s->numc, map, len, f, o_num, live))
            || (o_ns && store_col_r_(s, &s->nsc, map, len, f, o_ns, live))
            || (texts && (o_tck ? store_col_r_(s, &s->tck, map, len, f, o_tck, live) : store_col_zero_(s, &s->tck, live)))) { fclose(f); store_destroy(s); return (RagStore *)0; }
        s->dead_words = ((i64)live + 63) >> 6;
        s->dead = (u64 *)calloc((size_t)s->dead_words, sizeof(u64));
        if (map) {                                               /* re-scoring reads rows at random: no read-ahead */
            size_t pg = (size_t)sysconf(_SC_PAGESIZE), b = o_vecs - o_vecs % pg;
            madvise(map + b, o_ids - b, MADV_RANDOM);
        }
    }
    if (texts && store_heap_r_(s, map, len, f, o_heap, hlen)) { fclose(f); store_destroy(s); return (RagStore *)0; }
    if (texts && (!dend || lvl != 2)) {        /* nothing vouched for the offsets: each inside the heap */
        i32 r;                                                   /* (tests/fuzz_store.c: a v8 file's pointed past it, and
                                                                  * the first save read text from a block that is not there) */
        for (r = 0; r < live; ++r) {
            u64 v = STORE_TXO(s, r);
            if (v && (u64)STORE_TXOFF(v) + (u64)STORE_TXLEN(v) > hlen) {
                sprintf(store_lerr_, "the text of row %d lies past the text heap: the file is damaged", (int)r);
                fclose(f); store_destroy(s); return (RagStore *)0;
            }
        }
    }
    s->txlive = (i64)hlen;
    s->count = live;
    s->live = live;
    if (!o_rck && live > 0 && store_rck_fill_(s)) { fclose(f); store_destroy(s); return (RagStore *)0; }   /* older file */
    s->idmap_ok = live == 0;
    if (vz && ssum_vouch_(vz, path, ro, (u64)o_kind, 1)) { fclose(f); store_destroy(s); return (RagStore *)0; }
    if (map) kind = map[o_kind];
    else if (fseeko(f, (off_t)o_kind, SEEK_SET) != 0 || fread(&kind, 1, 1, f) != 1) { fclose(f); store_destroy(s); return (RagStore *)0; }
    if (drop_idx) kind = 0;

    if (kind == 1) {
        if (fseeko(f, (off_t)(o_kind + 1), SEEK_SET) != 0) { fclose(f); store_destroy(s); return (RagStore *)0; }
        s->idx = engine_load(&s->idx_arena, f);
        fclose(f);
        if (!s->idx || s->idx->dim != s->dim) { store_destroy(s); return (RagStore *)0; }   /* another dimension: not ours */
        s->built_upto = live;
    } else {
        if (kind == 2) s->tq = store_tq_r_(map, len, f, o_kind + 1, live, map ? vz : (const SSum *)0, path, ro);
        fclose(f);
        s->fast = 1; s->fast_target = 0.97f;                     /* fast stores are saved without an engine index */
        if (s->tq) { s->built_upto = live; s->cascade = s->tq->codes1 != (u8 *)0; }   /* saved with the cascade: on */
        else store_build(s);                                      /* no saved scan index (or an unreadable one): build it */
    }
    s->stamp = stamp;
    store_journal_replay_(s, path);
    return store_jnewer_(s);
}

static RagStore *store_load_impl_(Arena *a, const char *path, i32 ro)
{
    FILE *f;
    i32 hdr[4];
    RagStore *s;
    i32 live, i, lvl = store_verify_level_(), sums;
    struct stat pst;
    u8 has_idx;
    SSum z;
    i64 bad, fixed;
    u64 ubad;
    strcpy(store_lerr_, "cannot be read, or not a store file of a version this build knows");
    /* the checks come first, before even the header is trusted: they rebuild a damaged one too */
    memset(&z, 0, sizeof(z));
    if (lvl == STORE_VERIFY_AUTO)                                /* every block when the file fits RAM twice, else the lens */
        lvl = stat(path, &pst) == 0 && (i64)pst.st_size > store_ram_half_() ? 3 : 2;
    if (lvl == 3 && eng_host_be_()) lvl = 2;                     /* the lens checks the mapping: little-endian hosts only */
    sums = store_sum_load_(path, ro, lvl == 3 ? 1 : lvl, &z, &bad, &fixed, &ubad);
    if (sums < 0) { strcpy(store_lerr_, "some blocks could not be read to check them"); return (RagStore *)0; }
    if (!(f = fopen(path, "rb"))) { free(z.tab); return (RagStore *)0; }
    if (STORE_R32(f, hdr, 4) || (u32)hdr[0] != STORE_MAGIC) { fclose(f); free(z.tab); return (RagStore *)0; }
    if (hdr[1] > STORE_VERSION10) {                              /* a newer format: refused, never misread */
        sprintf(store_lerr_, "was written by a newer vecindex (file version %d; this build reads 4 to %d)", (int)hdr[1], STORE_VERSION10);
        fclose(f); free(z.tab); return (RagStore *)0;
    }
    if (hdr[1] != 4 && hdr[1] != STORE_VERSION && hdr[1] != STORE_VERSION_I8 && hdr[1] != STORE_VERSION7
        && hdr[1] != STORE_VERSION8 && !STORE_V9UP(hdr[1])) { fclose(f); free(z.tab); return (RagStore *)0; }
    if (STORE_V9UP(hdr[1]) && !sums && lvl > 0) {
        strcpy(store_lerr_, "neither copy of its checksums can be read (VECINDEX_VERIFY=0 loads it unchecked)");
        fclose(f);
        return (RagStore *)0;
    }
    if (sums && hdr[1] == 4) { free(z.tab); z.tab = (u8 *)0; }
    if (hdr[1] != 4) {
        s = store_load_v5_(a, path, f, hdr, ro, sums ? z.D : 0, lvl > 0 ? ubad : ~(u64)0, lvl, lvl == 3 && sums && z.tab ? &z : (const SSum *)0);
        if (s) {
            store_lerr_[0] = 0;
            if (sums) { s->sum_blocks = (i64)(z.N + z.P); s->sum_bad = bad; s->sum_fixed = fixed; s->sum_k = (i32)z.k; s->sum_m = (i32)z.m; }
            if (sums && z.tab && s->map && (s->msum = malloc(sizeof(SSum))) != (void *)0) {   /* rows read later can heal */
                memcpy(s->msum, &z, sizeof(SSum));
                z.tab = (u8 *)0;
                s->mfd = open(path, ro ? O_RDONLY : O_RDWR);
                if (lvl == 3) store_lens_start_(s);              /* the rest of the file: checked as it is needed */
            }
        }
        if (sums) free(z.tab);
        return s;
    }

    /* v4: packed sections, no saved scan index - read everything, rebuild the scan */
    {   struct stat st4;                /* no checksums: the header is believed only as far as the file can hold it -
                                         * tests/fuzz_store.c saw a damaged row count reserve 16 TB of spill file */
        if (hdr[2] <= 0 || hdr[3] < 0 || fstat(fileno(f), &st4) != 0
            || (u64)hdr[3] > (u64)st4.st_size / ((u64)hdr[2] * 4u + 16u)) { fclose(f); return (RagStore *)0; }
    }
    s = store_new(a, hdr[2]);
    if (!s->addbuf) { fclose(f); store_destroy(s); return (RagStore *)0; }
    s->ro = ro;
    live = hdr[3];
    if (live < 0 || store_goal_r_(f, &s->goal) || store_reserve_(s, live > 0 ? live : 1)) { fclose(f); store_destroy(s); return (RagStore *)0; }
    for (i = 0; i < live; ++i) if (STORE_R32(f, STORE_VEC(s, i), (size_t)s->dim)) { fclose(f); store_destroy(s); return (RagStore *)0; }
    for (i = 0; i < live; ++i) if (STORE_R64(f, &STORE_ID(s, i), 1)) { fclose(f); store_destroy(s); return (RagStore *)0; }
    for (i = 0; i < live; ++i) if (STORE_R64(f, &STORE_META(s, i), 1)) { fclose(f); store_destroy(s); return (RagStore *)0; }
    s->count = live;
    if (store_rck_fill_(s)) { fclose(f); store_destroy(s); return (RagStore *)0; }
    store_reindex_(s);

    if (fread(&has_idx, 1, 1, f) != 1) { fclose(f); store_destroy(s); return (RagStore *)0; }
    if (has_idx) {
        s->idx = engine_load(&s->idx_arena, f);
        fclose(f);
        if (!s->idx || s->idx->dim != s->dim) { store_destroy(s); return (RagStore *)0; }
        s->built_upto = live;
    } else {
        fclose(f);
        s->fast = 1; s->fast_target = 0.97f;                     /* fast stores are saved without an engine index */
        store_build(s);
    }
    store_journal_replay_(s, path);
    store_lerr_[0] = 0;
    return store_jnewer_(s);
}

RagStore *store_load(Arena *a, const char *path) { return store_load_impl_(a, path, 0); }

/* The store in path and its journal as they are at this moment, for reading only: nothing is written, moved or cut
 * (a torn journal tail is skipped, not dropped), so a server may go on writing to them meanwhile; what it writes
 * after the open is not seen. Changes to the returned store stay in memory (store_save, store_sync and
 * store_snapshot refuse it). */
RagStore *store_load_ro(Arena *a, const char *path) { return store_load_impl_(a, path, 1); }
