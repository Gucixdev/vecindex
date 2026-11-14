/* store_batch.c - write batches: replacements, texts, deletes and range deletes that a search sees all or none of,
 * and that the journal keeps all or none of. The batch is encoded as journal records while it is built; commit takes
 * the write lock once, first allocates what can run out (rows, text heap), then applies every change in order, and
 * journals the batch as one 'G' record: its length, its CRC-32C, then its records. Replay applies a 'G' record only
 * when all of it is there and its CRC matches; a torn or damaged one ends the journal there, as a torn record does.
 * A durable commit (sync) returns once an fsync covers it; commits that arrive during an fsync share the next one.
 * Part of store.c, which includes it at this point; not compiled on its own.
 *
 *   StoreWb b;
 *   store_batch_init(&b, s);
 *   store_batch_delete_range(&b, doc << 20, (doc << 20) | 0xfffff);       a document's old chunks go
 *   store_batch_put_text(&b, vec, doc << 20, 0, text, len);              its new ones come, as one change
 *   store_batch_commit(&b, 1);                                           applied and on the disk
 *   store_batch_free(&b); */

#define STORE_BATCH_MAX ((size_t)1 << 30)                        /* bytes of records a batch may carry */

typedef struct {
    RagStore *s;
    u8 *buf;                /* the records, as in the journal: 'P' id meta row, 'Q' id meta row len text, 'D' id, 'R' lo hi */
    size_t len, cap;
    f32 *tmp;               /* [dim] scratch: a row normalised, or copied out of buf aligned */
    i32 n, nput, ntext;     /* changes, rows they add, texts among them */
    i32 bad;                /* a refused change (non-finite vector, bad text, no memory): commit refuses the batch */
} StoreWb;

/* ---- CRC-32C (Castagnoli), the SSE4.2 instruction when there is one ---- */

#if !(defined(__SSE4_2__) && defined(__x86_64__))
static u32 store_crc_tab_[256];
static pthread_once_t store_crc_once_ = PTHREAD_ONCE_INIT;
static void store_crc_init_(void)
{
    u32 i, j, c;
    for (i = 0; i < 256; ++i) {
        for (c = i, j = 0; j < 8; ++j) c = c & 1u ? (c >> 1) ^ 0x82f63b78u : c >> 1;
        store_crc_tab_[i] = c;
    }
}
#endif

/* the CRC-32C register after n more bytes (start at 0xffffffff, finish with ~) */
static u32 store_crc32c_up_(u32 crc, const u8 *p, size_t n)
{
#if defined(__SSE4_2__) && defined(__x86_64__)
    while (n >= 8) { u64 v; memcpy(&v, p, 8); crc = (u32)_mm_crc32_u64(crc, v); p += 8; n -= 8; }
    while (n--) crc = _mm_crc32_u8(crc, *p++);
#else
    pthread_once(&store_crc_once_, store_crc_init_);
    while (n--) crc = store_crc_tab_[(crc ^ *p++) & 0xffu] ^ (crc >> 8);
#endif
    return crc;
}
static u32 store_crc32c_(const u8 *p, size_t n) { return ~store_crc32c_up_(0xffffffffu, p, n); }

/* ---- building ---- */

void store_batch_init(StoreWb *b, RagStore *s)
{
    memset(b, 0, sizeof(*b));
    b->s = s;
    b->tmp = (f32 *)malloc((size_t)s->dim * sizeof(f32));
    if (!b->tmp) b->bad = 1;
}

void store_batch_free(StoreWb *b)
{
    free(b->buf); free(b->tmp);
    memset(b, 0, sizeof(*b));
}

/* room for more bytes of records; -1 (and the batch refused) when there is none */
static i32 store_batch_room_(StoreWb *b, size_t more)
{
    if (b->bad) return -1;
    if (b->len + more > STORE_BATCH_MAX) { b->bad = 1; return -1; }
    if (b->len + more > b->cap) {
        size_t c = b->cap ? b->cap : 4096;
        u8 *nb;
        while (c < b->len + more) c *= 2;
        if (!(nb = (u8 *)realloc(b->buf, c))) { b->bad = 1; return -1; }
        b->buf = nb; b->cap = c;
    }
    return 0;
}

static void store_batch_w_(StoreWb *b, const void *p, size_t n) { memcpy(b->buf + b->len, p, n); b->len += n; }

/* 'P' or 'Q': the row normalised now, so the record is final before commit (replay normalises it once more, as
 * commit does: the store after a restart is the store before it) */
static i32 store_batch_row_(StoreWb *b, u8 op, const f32 *vec, i64 id, i64 meta, size_t extra)
{
    size_t vb = (size_t)b->s->dim * sizeof(f32);
    if (b->bad) return -1;
    if (!store_finite_(vec, b->s->dim)) { b->bad = 1; return -1; }
    if (store_batch_room_(b, 17 + vb + extra) != 0) return -1;
    memcpy(b->tmp, vec, vb);
    if (!b->s->metric) store_normalise_(b->tmp, b->s->dim);     /* dot and L2 stores keep the length: commit takes it */
    store_batch_w_(b, &op, 1); store_batch_w_(b, &id, 8); store_batch_w_(b, &meta, 8); store_batch_w_(b, b->tmp, vb);
    ++b->n; ++b->nput;
    return 0;
}

/* Replace (or add): every row with id goes, this one comes. 0, or -1: the batch is refused at commit (a non-finite
 * vector, no memory). The vector is copied: the caller may reuse it at once. */
i32 store_batch_put(StoreWb *b, const f32 *vec, i64 id, i64 meta)
{
    return store_batch_row_(b, 'P', vec, id, meta, 0);
}

/* The same with a text (len 0: none; at most STORE_TXMAX bytes) */
i32 store_batch_put_text(StoreWb *b, const f32 *vec, i64 id, i64 meta, const char *text, i32 len)
{
    u32 n = (u32)len;
    if (len == 0) return store_batch_put(b, vec, id, meta);
    if (len < 0 || len > STORE_TXMAX || !text) { b->bad = 1; return -1; }
    if (store_batch_row_(b, 'Q', vec, id, meta, 4 + (size_t)len) != 0) return -1;
    store_batch_w_(b, &n, 4); store_batch_w_(b, text, (size_t)len);
    ++b->ntext;
    return 0;
}

i32 store_batch_delete(StoreWb *b, i64 id)
{
    u8 op = 'D';
    if (store_batch_room_(b, 9) != 0) return -1;
    store_batch_w_(b, &op, 1); store_batch_w_(b, &id, 8);
    ++b->n;
    return 0;
}

/* every row whose id is in [lo, hi] */
i32 store_batch_delete_range(StoreWb *b, i64 lo, i64 hi)
{
    u8 op = 'R';
    if (store_batch_room_(b, 17) != 0) return -1;
    store_batch_w_(b, &op, 1); store_batch_w_(b, &lo, 8); store_batch_w_(b, &hi, 8);
    ++b->n;
    return 0;
}

/* bytes of the record at r */
static size_t store_batch_reclen_(const RagStore *s, const u8 *r)
{
    size_t vb = (size_t)s->dim * sizeof(f32);
    u32 n;
    if (r[0] == 'D') return 9;
    if (r[0] == 'R') return 17;
    if (r[0] == 'P') return 17 + vb;
    memcpy(&n, r + 17 + vb, 4);                                  /* 'Q' */
    return 21 + vb + n;
}

/* ---- committing ---- */

/* 'G' len(u32) crc(u32) records[len], to the journal and, during a snapshot, to the new file's journal too */
static void store_jgroup_(RagStore *s, const u8 *rec, u32 len, u32 crc)
{
    FILE *fs[2];
    u8 op = 'G';
    i32 i;
    ++s->jseq;
    fs[0] = s->jf; fs[1] = s->jf2;
    for (i = 0; i < 2; ++i) {
        if (!fs[i]) continue;
        fwrite(&op, 1, 1, fs[i]); fwrite(&len, 4, 1, fs[i]); fwrite(&crc, 4, 1, fs[i]); fwrite(rec, 1, (size_t)len, fs[i]);
    }
}

/* one record as a 'G' frame of its own - the parts' length and CRC-32C, then the parts - so that replay finds a
 * damaged record instead of applying it */
static void store_jframe_(RagStore *s, const void *const *part, const size_t *plen, i32 np)
{
    FILE *fs[2];
    u8 op = 'G';
    u32 len = 0, crc = 0xffffffffu;
    i32 i, j;
    for (j = 0; j < np; ++j) { len += (u32)plen[j]; crc = store_crc32c_up_(crc, (const u8 *)part[j], plen[j]); }
    crc = ~crc;
    ++s->jseq;
    fs[0] = s->jf; fs[1] = s->jf2;
    for (i = 0; i < 2; ++i) {
        if (!fs[i]) continue;
        fwrite(&op, 1, 1, fs[i]); fwrite(&len, 4, 1, fs[i]); fwrite(&crc, 4, 1, fs[i]);
        for (j = 0; j < np; ++j) if (plen[j]) fwrite(part[j], 1, plen[j], fs[i]);
    }
}

/* Durable up to journal write seq: one fsync at a time under jsync_mu, the write lock held only while the stdio
 * buffers go to the kernel; a commit whose write an fsync started after it already covered returns at once. */
static i32 store_jsync_(RagStore *s, u64 seq)
{
    i32 r = 0, i;
    pthread_mutex_lock(&s->jsync_mu);
    if (s->jsynced < seq) {
        int fd[2];
        u64 at;
        fd[0] = fd[1] = -1;
        pthread_rwlock_wrlock(&s->lock);
        at = s->jseq;
        if (s->jf) { if (fflush(s->jf) != 0) r = -1; else fd[0] = dup(fileno(s->jf)); }
        if (s->jf2) { if (fflush(s->jf2) != 0) r = -1; else fd[1] = dup(fileno(s->jf2)); }
        pthread_rwlock_unlock(&s->lock);
        for (i = 0; i < 2; ++i)
            if (fd[i] >= 0) { if (fsync(fd[i]) != 0) r = -1; close(fd[i]); }
        if (r == 0) s->jsynced = at;
    }
    pthread_mutex_unlock(&s->jsync_mu);
    return r;
}

/* Applies the batch as one change: searches see none of it before, all of it after, and the journal keeps it whole or
 * not at all. With sync, it is on the disk when this returns (a store opened from a file; without a journal there
 * is nothing to sync). Returns the changes applied (0 for an empty batch); -1 refused, nothing applied (a refused
 * change in it, no memory, a read-only store); -2 applied, but the fsync failed. The batch can be reused after
 * store_batch_free and store_batch_init. */
i32 store_batch_commit(StoreWb *b, i32 sync)
{
    RagStore *s = b->s;
    size_t vb = (size_t)s->dim * sizeof(f32), at, rl;
    u64 *tv = (u64 *)0, seq;
    u32 crc;
    i32 k, i;
    if (b->bad || s->ro) return -1;
    if (b->n == 0) return 0;
    if (b->ntext && !(tv = (u64 *)malloc((size_t)b->ntext * sizeof(u64)))) return -1;
    crc = store_crc32c_(b->buf, b->len);                        /* the records are final: no lock needed for this */
    pthread_rwlock_wrlock(&s->lock);
    store_idmap_locked_(s);
    if (b->ntext) s->has_text = 1;                               /* from now on the column has a slot for every row */
    if (store_reserve_(s, (i64)s->count + b->nput) != 0) { pthread_rwlock_unlock(&s->lock); free(tv); return -1; }
    /* the texts first, the other thing that can run out; what a refused batch left in the heap is garbage that the
     * next save drops */
    for (at = 0, k = 0; at < b->len; at += store_batch_reclen_(s, b->buf + at))
        if (b->buf[at] == 'Q') {
            u32 n;
            memcpy(&n, b->buf + at + 17 + vb, 4);
            if (!(tv[k++] = store_tx_put_(s, &s->tx, (const char *)b->buf + at + 21 + vb, (i32)n))) {
                pthread_rwlock_unlock(&s->lock);
                free(tv);
                return -1;
            }
        }
    /* then every change, in order; none of these can fail */
    for (at = 0, k = 0; at < b->len; at += rl) {
        const u8 *r = b->buf + at;
        i64 id, meta;
        rl = store_batch_reclen_(s, r);
        memcpy(&id, r + 1, 8);
        if (r[0] == 'D') { store_kill_id_locked_(s, id); continue; }
        memcpy(&meta, r + 9, 8);
        if (r[0] == 'R') {
            for (i = 0; i < s->count; ++i)
                if (!STORE_DEAD(s, i) && STORE_ID(s, i) >= id && STORE_ID(s, i) <= meta) {
                    STORE_KILL(s, i); --s->live; ++s->mods; store_tx_drop_(s, i);
                }
            continue;
        }
        memcpy(b->tmp, r + 17, vb);                              /* aligned for the row code */
        store_kill_id_locked_(s, id);
        store_append_locked_(s, b->tmp, id, meta);
        if (r[0] == 'Q') {
            u32 n;
            memcpy(&n, r + 17 + vb, 4);
            STORE_TXO(s, s->count - 1) = tv[k++];
            s->txlive += (i64)n;
            if (s->lex) lex_add(s->lex, id, (const char *)r + 21 + vb, (i32)n);
        }
    }
    store_jgroup_(s, b->buf, (u32)b->len, crc);
    seq = s->jseq;
    pthread_rwlock_unlock(&s->lock);
    free(tv);
    if (sync && store_jsync_(s, seq) != 0) return -2;
    return b->n;
}
