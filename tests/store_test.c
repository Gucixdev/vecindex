/* store_test.c - the self-test of src/store.c; ./run builds and runs it. */
#include "store.c"

static u32 sd_rng_(u32 *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static f32 sd_f_(u32 *s) { return (f32)sd_rng_(s) / (f32)0xffffffffu - 0.5f; }

/* rows [from, to) of x into s from a thread of its own; `done` tells the searching thread how far it got */
typedef struct { RagStore *s; const f32 *x; i32 dim, from, to; i64 id0; i32 done; } SdAdder;
static void *sd_adder_(void *p)
{
    SdAdder *a = (SdAdder *)p;
    i32 i;
    for (i = a->from; i < a->to; ++i) {
        store_add(a->s, a->x + (size_t)i * a->dim, a->id0 + i);
        __atomic_store_n(&a->done, i + 1, __ATOMIC_RELEASE);
        if (i % 1000 == 999) { struct timespec ts; ts.tv_sec = 0; ts.tv_nsec = 1000000L; nanosleep(&ts, (struct timespec *)0); }
    }
    return (void *)0;
}

/* recall@10 of store s on queries Q against the exact top 10 of rows [0, n) of x (ids id0 + row; gone[r]: deleted) */
static f32 sd_recall_(RagStore *s, const f32 *x, i32 n, i32 dim, const f32 *Q, i32 nq, i64 id0, const u8 *gone)
{
    i32 q, r, j, hits = 0;
    for (q = 0; q < nq; ++q) {
        const f32 *qv = Q + (size_t)q * dim;
        f32 bs[10]; i32 br[10];
        i64 got[10]; f32 gs[10];
        i32 h;
        for (j = 0; j < 10; ++j) { bs[j] = -1e30f; br[j] = -1; }
        for (r = 0; r < n; ++r) if (!gone || !gone[r]) store_topk_put_(bs, br, 10, vs_dot(qv, x + (size_t)r * dim, dim), r);
        h = store_search(s, qv, 10, got, gs);
        for (j = 0; j < h; ++j) { i32 z; for (z = 0; z < 10; ++z) if (br[z] >= 0 && id0 + br[z] == got[j]) { ++hits; break; } }
    }
    return (f32)hits / (f32)(nq * 10);
}

/* ---- texts: put / get / upsert / delete, id ranges, hybrid search, save + load (v8), the journal, a snapshot, the
 * heap's compaction ---- */
#define SDT_DIM 16
static i32 sdt_text_(i64 id, char *buf)                  /* the text of row id: words, its own rare word, some bytes */
{
    i32 n = sprintf(buf, "document %d chunk %d about tea and cats zq%dx ", (int)(id >> 20), (int)(id & 0xfffff), (int)((id >> 20) * 100 + (id & 0xfffff)));
    buf[n++] = (char)0; buf[n++] = (char)0xff; buf[n++] = (char)(id & 0x7f);
    return n;
}
static i32 sdt_same_(RagStore *s, i64 id, const char *want, i32 wn)
{
    static char got[128];
    i32 n = store_get_text(s, id, got, (i32)sizeof(got));
    return n == wn && wn <= 128 && memcmp(got, want, (size_t)wn) == 0;
}
static i32 sd_texts_(Arena *a)
{
    const char *path = "/tmp/store_demo_text.vxi", *snap = "/tmp/store_demo_text_snap.vxi";
    RagStore *s = store_new(a, SDT_DIM), *l, *m;
    static f32 v[SDT_DIM], far_[SDT_DIM];
    static char buf[128];
    char *big = (char *)malloc((size_t)3 << 20), *bb = (char *)malloc((size_t)3 << 20);
    i64 ids[64], doc5[16];
    f32 sc[64];
    i32 ok = 1, d, c, i, n, bad = 0, hyb = 0, hn;
    u32 st = 99u;
    i64 used0;
    remove(path); remove(snap);
    for (i = 0; i < (3 << 20); ++i) big[i] = (char)(sd_rng_(&st) & 0xff);
    for (d = 0; d < 30; ++d)                             /* 30 documents of 10 chunks, ids doc << 20 | chunk, in order */
        for (c = 0; c < 10; ++c) {
            i64 id = ((i64)d << 20) | c;
            for (i = 0; i < SDT_DIM; ++i) v[i] = sd_f_(&st);
            n = sdt_text_(id, buf);
            if (store_put_text(s, v, id, 0, buf, n)) bad = 1;
        }
    for (i = 0; i < SDT_DIM; ++i) v[i] = sd_f_(&st);
    if (store_put_text(s, v, (i64)40 << 20, 0, big, (3 << 20) - 7)) bad = 1;      /* crosses heap blocks */
    for (d = 0; d < 30 && !bad; ++d)
        for (c = 0; c < 10; ++c) { i64 id = ((i64)d << 20) | c; n = sdt_text_(id, buf); if (!sdt_same_(s, id, buf, n)) bad = 1; }
    n = store_get_text(s, (i64)40 << 20, bb, 3 << 20);
    if (n != (3 << 20) - 7 || memcmp(bb, big, (size_t)n) != 0) bad = 1;
    if (store_get_text(s, (i64)40 << 20, bb, 10) != (3 << 20) - 7 || memcmp(bb, big, 10) != 0) bad = 1;   /* cap < length */
    /* upsert with a new text, upsert without one, delete, delete a range */
    if (store_put_text(s, v, (i64)3 << 20, 0, "replaced", 8) || !sdt_same_(s, (i64)3 << 20, "replaced", 8)) bad = 1;
    if (store_put(s, v, ((i64)3 << 20) | 1, 0) || store_get_text(s, ((i64)3 << 20) | 1, buf, 8) != 0) bad = 1;
    if (store_delete(s, ((i64)3 << 20) | 2) != 1 || store_get_text(s, ((i64)3 << 20) | 2, buf, 8) != -1) bad = 1;
    if (store_delete_range(s, (i64)7 << 20, ((i64)7 << 20) | 0xfffff) != 10) bad = 1;
    if (store_count(s) != 300 + 1 - 1 - 10) bad = 1;
    /* ranges: a document's chunks; then rows out of order, merged; paging */
    n = store_range(s, (i64)5 << 20, ((i64)5 << 20) | 0xfffff, doc5, 16);
    if (n != 10) bad = 1;
    for (i = 0; i < n; ++i) if (doc5[i] != (((i64)5 << 20) | i)) bad = 1;
    if (store_range(s, (i64)7 << 20, ((i64)7 << 20) | 0xfffff, ids, 16) != 0) bad = 1;
    if (store_range(s, (i64)3 << 20, ((i64)3 << 20) | 0xfffff, ids, 16) != 9 || ids[2] != (((i64)3 << 20) | 3)) bad = 1;
    store_put_text(s, v, ((i64)5 << 20) | 30, 0, "late a", 6);                /* behind the sorted rows */
    store_put_text(s, v, ((i64)5 << 20) | 20, 0, "late b", 6);
    store_put_text(s, v, ((i64)50 << 20), 0, "far doc", 7);
    n = store_range(s, (i64)5 << 20, ((i64)5 << 20) | 0xfffff, ids, 16);
    if (n != 12 || ids[9] != (((i64)5 << 20) | 9) || ids[10] != (((i64)5 << 20) | 20) || ids[11] != (((i64)5 << 20) | 30)) bad = 1;
    if (store_range(s, ((i64)5 << 20) | 19, ((i64)5 << 20) | 21, ids, 16) != 1 || ids[0] != (((i64)5 << 20) | 20)) bad = 1;   /* short */
    n = store_range(s, ((i64)5 << 20) | 8, (i64)1 << 40, ids, 3);              /* wide range, paged by 3 */
    if (n != 3 || ids[0] != (((i64)5 << 20) | 8) || ids[1] != (((i64)5 << 20) | 9) || ids[2] != (((i64)5 << 20) | 20)) bad = 1;
    if (bad) { print_cstr("  texts: put/get/upsert/delete/range FAILED\n"); ok = 0; bad = 0; }
    /* hybrid: a rare word finds its row whatever the vector says; words only; vector only */
    for (i = 0; i < SDT_DIM; ++i) far_[i] = sd_f_(&st);
    for (d = 0; d < 30; d += 3) {
        i64 id = ((i64)d << 20) | 4;
        sprintf(buf, "zq%dx", d * 100 + 4);
        hn = store_search_hybrid(s, far_, buf, (i32)strlen(buf), 5, ids, sc);
        if (hn > 0 && ids[0] == id) ++hyb;
        if (store_search_hybrid(s, (const f32 *)0, buf, (i32)strlen(buf), 5, ids, sc) < 1 || ids[0] != id) bad = 1;
    }
    if (store_search_hybrid(s, far_, (const char *)0, 0, 5, ids, sc) != 5) bad = 1;
    if (hyb != 10 || bad) { print_cstr("  texts: hybrid search FAILED\n"); ok = 0; bad = 0; }
    /* save (v8) and load: every text, the ranges, the word index rebuilt */
    if (store_save(s, path) != 0 || !(l = store_load(a, path))) { print_cstr("  texts: save/load FAILED\n"); store_destroy(s); free(big); free(bb); return 0; }
    for (d = 0; d < 30; ++d)
        for (c = 0; c < 10; ++c) {
            i64 id = ((i64)d << 20) | c;
            i32 want = d == 7 || (d == 3 && c == 2) ? -1 : d == 3 && c == 1 ? 0 : -2;
            if (want == -2) { if (d == 3 && c == 0) { if (!sdt_same_(l, id, "replaced", 8)) bad = 1; } else { n = sdt_text_(id, buf); if (!sdt_same_(l, id, buf, n)) bad = 1; } }
            else if (store_get_text(l, id, buf, 8) != want) bad = 1;
        }
    n = store_get_text(l, (i64)40 << 20, bb, 3 << 20);
    if (n != (3 << 20) - 7 || memcmp(bb, big, (size_t)n) != 0) bad = 1;
    if (store_range(l, (i64)5 << 20, ((i64)5 << 20) | 0xfffff, ids, 16) != 12) bad = 1;
    sprintf(buf, "zq%dx", 2700 + 4);
    if (store_search_hybrid(l, (const f32 *)0, buf, (i32)strlen(buf), 3, ids, sc) < 1 || ids[0] != (((i64)27 << 20) | 4)) bad = 1;
    /* the journal: texts added after the load come back on the next one */
    for (i = 0; i < 50; ++i) { sprintf(buf, "journal row %d qj%dz", i, i); if (store_put_text(l, v, ((i64)60 << 20) | i, 0, buf, (i32)strlen(buf))) bad = 1; }
    store_put_text(l, v, ((i64)5 << 20) | 3, 0, "upserted after load", 19);
    if (store_sync(l, path) != 0) bad = 1;
    store_destroy(l);
    if (!(l = store_load(a, path))) { print_cstr("  texts: reload FAILED\n"); store_destroy(s); free(big); free(bb); return 0; }
    for (i = 0; i < 50; ++i) { sprintf(buf, "journal row %d qj%dz", i, i); if (!sdt_same_(l, ((i64)60 << 20) | i, buf, (i32)strlen(buf))) bad = 1; }
    if (!sdt_same_(l, ((i64)5 << 20) | 3, "upserted after load", 19)) bad = 1;
    if (store_search_hybrid(l, (const f32 *)0, "qj17z", 5, 3, ids, sc) < 1 || ids[0] != (((i64)60 << 20) | 17)) bad = 1;
    if (bad) { print_cstr("  texts: save/load/journal FAILED\n"); ok = 0; bad = 0; }
    /* a snapshot with deleted rows: the child leaves them and their texts out */
    store_delete_range(l, (i64)60 << 20, ((i64)60 << 20) | 24);
    if (store_snapshot(l, snap) != 0 || store_snapshot_poll(l, 1) != 0 || !(m = store_load(a, snap))) bad = 1;
    else {
        for (i = 0; i < 50; ++i) {
            sprintf(buf, "journal row %d qj%dz", i, i);
            if (i < 25 ? store_get_text(m, ((i64)60 << 20) | i, buf, 8) != -1 : !sdt_same_(m, ((i64)60 << 20) | i, buf, (i32)strlen(buf))) bad = 1;
        }
        n = store_get_text(m, (i64)40 << 20, bb, 3 << 20);
        if (n != (3 << 20) - 7 || memcmp(bb, big, (size_t)n) != 0) bad = 1;
        store_destroy(m);
    }
    if (bad) { print_cstr("  texts: snapshot FAILED\n"); ok = 0; bad = 0; }
    /* garbage: replacing the big text again and again leaves old copies in the heap; a save reclaims them */
    for (i = 0; i < 4; ++i) store_put_text(l, v, (i64)40 << 20, 0, big, (3 << 20) - 7 - i);
    used0 = l->tx.used;
    if (store_save(l, path) != 0) bad = 1;
    if (!(l->tx.used < used0 / 2 && l->tx.used <= l->txlive + 2 * STORE_TXB)) bad = 1;
    n = store_get_text(l, (i64)40 << 20, bb, 3 << 20);
    if (n != (3 << 20) - 10 || memcmp(bb, big, (size_t)n) != 0) bad = 1;
    print_cstr("  texts: 301 rows put with texts (one of 3 MB across heap blocks), upserts, deletes, a document's chunks by id range,");
    print_cstr(" hybrid search found the row of a rare word "); print_i64(hyb); print_cstr("/10 with a vector pointing elsewhere;");
    print_cstr(" save + load (v8), the journal and a snapshot keep every text; a save shrank the heap ");
    print_i64(used0 >> 20); print_cstr(" -> "); print_i64(l->tx.used >> 20); print_cstr(" MB\n");
    if (bad) { print_cstr("  texts: heap compaction FAILED\n"); ok = 0; }
    store_destroy(l); store_destroy(s);
    remove(path); remove(snap);
    { char *j = store_jname_(path, ".j"); remove(j); free(j); j = store_jname_(snap, ".j"); remove(j); free(j); }
    free(big); free(bb);
    return ok;
}

/* ---- aggression: the threads follow it, background work yields ---- */
typedef struct { RagStore *s; i32 nice; } SdPrio;
static void *sd_prio_(void *p)
{
    SdPrio *x = (SdPrio *)p;
    store_bg_prio_(x->s);
    errno = 0;
    x->nice = getpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid));
    return (void *)0;
}
static i32 sd_aggression_(Arena *a)
{
    RagStore *g = store_new(a, 8);
    i32 avail = store_cpus_avail_(), ok = 1, n1, nh, n0, b0;
    SdPrio p1, p0;
    pthread_t th;
    store_set_aggression(g, 1.0f); n1 = store_ncpu_(g);
    p1.s = g; pthread_create(&th, (const pthread_attr_t *)0, sd_prio_, &p1); pthread_join(th, (void **)0);
    store_set_aggression(g, 0.5f); nh = store_ncpu_(g);
    store_set_aggression(g, 0.0f); n0 = store_ncpu_(g); b0 = store_bg_threads_(g);
    p0.s = g; pthread_create(&th, (const pthread_attr_t *)0, sd_prio_, &p0); pthread_join(th, (void **)0);
    store_set_aggression(g, 7.0f);
    if (n1 != avail || nh != (avail + 1) / 2 || n0 != 1 || b0 != 1 || g->aggr != 1.0f) ok = 0;
#if defined(__linux__)
    if (p1.nice != 10 || p0.nice != 19) ok = 0;
#endif
    print_cstr("  aggression: "); print_i64(avail); print_cstr(" CPUs this process may use (affinity, cgroup quota); threads at 1: ");
    print_i64(n1); print_cstr(", at 0.5: "); print_i64(nh); print_cstr(", at 0: "); print_i64(n0); print_cstr("; background work at nice ");
    print_i64(p1.nice); print_cstr(" (1) and "); print_i64(p0.nice); print_cstr(" (0)\n");
    store_destroy(g);
    return ok;
}

/* ---- memory tiers: the bytes a search scans (hot) and reads only for candidates (cold); locking either in RAM ---- */
static i64 sd_vmlck_(void)                                /* kB of locked memory (VmLck in /proc/self/status), -1: unknown */
{
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    i64 kb = -1;
    if (!f) return -1;
    while (fgets(line, (int)sizeof(line), f)) if (!strncmp(line, "VmLck:", 6)) { kb = (i64)atol(line + 6); break; }
    fclose(f);
    return kb;
}
#define SDM_N   3000
#define SDM_DIM 32
static i32 sdm_finds_(RagStore *s, const f32 *x)          /* rows 17, 1700 and 2999 find themselves first */
{
    static const i32 rows[3] = { 17, 1700, 2999 };
    i64 ids[4]; f32 sc[4];
    i32 i;
    for (i = 0; i < 3; ++i)
        if (store_search(s, x + (size_t)rows[i] * SDM_DIM, 4, ids, sc) < 1 || ids[0] != 7000 + rows[i]) return 0;
    return 1;
}
static i32 sd_tiers_(Arena *a)
{
    const char *path = "/tmp/store_demo_tiers.vxi";
    char *jn = store_jname_(path, ".j");
    static f32 x[SDM_N * SDM_DIM];
    RagStore *s = store_new(a, SDM_DIM), *l;
    i64 hot, cold, k0, k1, k2, k3, k4;
    i32 i, ok = 1, r1, r2, r0, r3;
    u32 st = 4242u;
    remove(path); remove(jn);
    for (i = 0; i < SDM_N * SDM_DIM; ++i) x[i] = sd_f_(&st);
    for (i = 0; i < SDM_N; ++i) store_add(s, x + (size_t)i * SDM_DIM, 7000 + i);
    store_build(s);
    store_tiers(s, &hot, &cold);
    /* cold: every row's f32 vector, id, tag and check, exactly; hot: the 4-bit codes at least (dim / 2 bytes a row) */
    if (!s->tq || cold != (i64)SDM_N * (SDM_DIM * 4 + 8 + 8 + 4) || hot < (i64)SDM_N * (SDM_DIM / 2) || hot >= cold) ok = 0;
    print_cstr("  tiers: hot "); print_i64(hot); print_cstr(" B, cold "); print_i64(cold); print_cstr(" B for ");
    print_i64(SDM_N); print_cstr(" rows of "); print_i64(SDM_DIM); print_cstr(" dims\n");
    if (!ok) print_cstr("  tiers: sizes FAILED\n");
    if (store_save(s, path) != 0 || !(l = store_load(a, path))) {
        print_cstr("  tiers: save/load FAILED\n"); store_destroy(s); free(jn); return 0;
    }
    store_destroy(s);
    k0 = sd_vmlck_();
    r1 = store_set_memory(l, 1); k1 = sd_vmlck_();                /* the mapped codes, slot table, centroids */
    if (l->mem_tier != 1 || !sdm_finds_(l, x)) { print_cstr("  tiers: tier 1 FAILED\n"); ok = 0; }
    r2 = store_set_memory(l, 2); k2 = sd_vmlck_();                /* the rows, ids and tags too */
    if (l->mem_tier != 2 || !sdm_finds_(l, x)) { print_cstr("  tiers: tier 2 FAILED\n"); ok = 0; }
    r0 = store_set_memory(l, 0); k3 = sd_vmlck_();                /* everything back to the kernel */
    if (l->mem_tier != 0 || !sdm_finds_(l, x)) { print_cstr("  tiers: tier 0 FAILED\n"); ok = 0; }
    r3 = store_set_memory(l, 1);
    store_build(l);                                               /* the mapped index is replaced: its lock goes */
    k4 = sd_vmlck_();
    if (!sdm_finds_(l, x)) { print_cstr("  tiers: search after the rebuild FAILED\n"); ok = 0; }
    print_cstr("  tiers: locked kB "); print_i64(k0); print_cstr(" -> tier 1: "); print_i64(k1); print_cstr(" -> tier 2: ");
    print_i64(k2); print_cstr(" -> tier 0: "); print_i64(k3); print_cstr(" -> tier 1 and a rebuild: "); print_i64(k4); print_cstr("\n");
    if (k0 < 0) print_cstr("  tiers: no VmLck here - what got locked is not checked\n");
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    else if (1) print_cstr("  tiers: built with a sanitizer, which turns mlock into a no-op - what got locked is not checked\n");
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
    else if (1) print_cstr("  tiers: built with a sanitizer, which turns mlock into a no-op - what got locked is not checked\n");
#endif
#endif
    else if (!l->map) print_cstr("  tiers: the store is not mapped here (a big-endian CPU) - nothing to lock, not checked\n");
    else if (r1 != 0 || r2 != 0 || r0 != 0 || r3 != 0)
        print_cstr("  tiers: the memlock limit refused the lock (ulimit -l) - what got locked is not checked\n");
    else if (k1 < k0 + (i64)SDM_N * (SDM_DIM / 2) / 1024 || k2 < k1 + (i64)SDM_N * SDM_DIM * 4 / 1024 || k3 > k0 + 8 || k4 > k0 + 8) {
        print_cstr("  tiers: locking FAILED\n"); ok = 0;
    }
    store_destroy(l);
    remove(path); remove(jn); free(jn);
    return ok;
}

/* ---- write batches: all or none for searches and after a crash; a torn or damaged batch in the journal; a reader
 * that must never see half of one; durable commits from several threads ---- */
#define SDB_DIM 8
static void sdb_vec_(i64 id, f32 *v)
{
    u32 s = (u32)((u64)id * 2654435761u) | 1u;
    i32 d;
    for (d = 0; d < SDB_DIM; ++d) v[d] = sd_f_(&s);
}
static i32 sdb_doc_(RagStore *s, i64 doc)                 /* rows of document doc (ids doc << 20 | chunk) */
{
    static i64 ids[64];
    return store_range(s, doc << 20, (doc << 20) | 0xfffff, ids, 64);
}
static i32 sdb_text_is_(RagStore *s, i64 id, const char *want)
{
    char got[64];
    i32 n = store_get_text(s, id, got, (i32)sizeof(got));
    return n == (i32)strlen(want) && memcmp(got, want, (size_t)n) == 0;
}
static i64 sdb_fsize_(const char *p)
{
    FILE *f = fopen(p, "rb");
    long n;
    if (!f) return -1;
    fseek(f, 0, SEEK_END); n = ftell(f); fclose(f);
    return (i64)n;
}
static i32 sdb_write_(const char *p, const u8 *b, i64 n)  /* replace the file's bytes */
{
    FILE *f = fopen(p, "wb");
    i32 r;
    if (!f) return -1;
    r = fwrite(b, 1, (size_t)n, f) == (size_t)n ? 0 : -1;
    return fclose(f) == 0 ? r : -1;
}
typedef struct { RagStore *s; i32 stop, bad, reads; } SdbReader;
static void *sdb_reader_(void *p)                          /* document 7 has 5 chunks or 3, never anything else */
{
    SdbReader *x = (SdbReader *)p;
    while (!__atomic_load_n(&x->stop, __ATOMIC_ACQUIRE)) {
        i32 n = sdb_doc_(x->s, 7);
        if (n != 5 && n != 3) x->bad = n + 1000;
        ++x->reads;
    }
    return (void *)0;
}
typedef struct { RagStore *s; i32 t, bad; } SdbCommitter;
static void *sdb_committer_(void *p)                       /* 50 durable batches of 2 rows each */
{
    SdbCommitter *c = (SdbCommitter *)p;
    StoreWb b;
    f32 v[SDB_DIM];
    i32 i, j;
    for (i = 0; i < 50; ++i) {
        store_batch_init(&b, c->s);
        for (j = 0; j < 2; ++j) { i64 id = ((i64)(c->t + 1) << 32) | (i * 2 + j); sdb_vec_(id, v); store_batch_put(&b, v, id, 0); }
        if (store_batch_commit(&b, 1) != 2) c->bad = 1;
        store_batch_free(&b);
    }
    return (void *)0;
}
static i32 sd_batches_(Arena *a)
{
    const char *path = "/tmp/store_demo_batch.vxi", *path2 = "/tmp/store_demo_batch2.vxi";
    char *jn = store_jname_(path, ".j"), *jn2 = store_jname_(path2, ".j");
    RagStore *s = store_new(a, SDB_DIM), *l, *g;
    StoreWb b;
    SdbReader rd;
    SdbCommitter cm[4];
    pthread_t th[4];
    static f32 v[SDB_DIM];
    static char buf[64];
    union { f32 f; u32 u; } nan_;
    u8 *jb = (u8 *)0;
    i64 jlen;
    i32 ok = 1, d, c, r, n, i, flips;
    remove(path); remove(jn); remove(path2); remove(jn2);
    for (d = 0; d < 3; ++d)                                   /* 3 documents of 5 chunks, each with a text */
        for (c = 0; c < 5; ++c) {
            i64 id = ((i64)d << 20) | c;
            sdb_vec_(id, v); n = sprintf(buf, "doc %d chunk %d", d, c);
            store_put_text(s, v, id, 0, buf, n);
        }
    if (store_save(s, path) != 0) { print_cstr("  batches: save FAILED\n"); store_destroy(s); free(jn); free(jn2); return 0; }

    /* one batch: document 1 replaced (5 chunks out, 3 new ones in, with texts), one chunk of document 2 out, a row in */
    store_batch_init(&b, s);
    store_batch_delete_range(&b, (i64)1 << 20, ((i64)1 << 20) | 0xfffff);
    for (c = 0; c < 3; ++c) {
        i64 id = ((i64)1 << 20) | (100 + c);
        sdb_vec_(id, v); n = sprintf(buf, "new doc 1 chunk %d", c);
        store_batch_put_text(&b, v, id, 0, buf, n);
    }
    store_batch_delete(&b, (i64)2 << 20);
    sdb_vec_((i64)9 << 20, v); store_batch_put(&b, v, (i64)9 << 20, 0);
    if (sdb_doc_(s, 1) != 5 || store_count(s) != 15) { print_cstr("  batches: a change showed before the commit\n"); ok = 0; }
    r = store_batch_commit(&b, 1);
    store_batch_free(&b);
    if (r != 6 || sdb_doc_(s, 1) != 3 || sdb_doc_(s, 2) != 4 || sdb_doc_(s, 9) != 1 || store_count(s) != 13
        || !sdb_text_is_(s, ((i64)1 << 20) | 101, "new doc 1 chunk 1") || !sdb_text_is_(s, ((i64)2 << 20) | 1, "doc 2 chunk 1")) {
        print_cstr("  batches: commit FAILED\n"); ok = 0;
    }
    /* refused: a NaN in it, so the range delete in it does not happen either */
    store_batch_init(&b, s);
    store_batch_delete_range(&b, 0, 0xfffff);
    nan_.u = 0x7fc00000u;
    sdb_vec_(77, v); v[3] = nan_.f;
    if (store_batch_put(&b, v, 77, 0) != -1) ok = 0;
    r = store_batch_commit(&b, 0);
    store_batch_free(&b);
    if (r != -1 || sdb_doc_(s, 0) != 5 || store_count(s) != 13) { print_cstr("  batches: a refused batch changed something\n"); ok = 0; }
    /* dropped without a commit, and an empty one */
    store_batch_init(&b, s); store_batch_delete_range(&b, 0, 0xfffff); store_batch_free(&b);
    store_batch_init(&b, s); r = store_batch_commit(&b, 1); store_batch_free(&b);
    if (r != 0 || sdb_doc_(s, 0) != 5 || store_count(s) != 13) { print_cstr("  batches: an aborted or empty batch changed something\n"); ok = 0; }
    /* the journal replays the batch (read-only load: the journal is left as it is) */
    if (!(l = store_load_ro(a, path))) { print_cstr("  batches: reload FAILED\n"); ok = 0; }
    else {
        if (sdb_doc_(l, 1) != 3 || sdb_doc_(l, 2) != 4 || sdb_doc_(l, 9) != 1 || store_count(l) != 13
            || !sdb_text_is_(l, ((i64)1 << 20) | 102, "new doc 1 chunk 2")) { print_cstr("  batches: replay FAILED\n"); ok = 0; }
        store_destroy(l);
    }
    /* another batch, then its journal record cut short, then one byte of it changed: a load gets none of it */
    store_batch_init(&b, s);
    store_batch_delete_range(&b, 0, 0xfffff);
    for (c = 0; c < 2; ++c) { i64 id = ((i64)5 << 20) | c; sdb_vec_(id, v); store_batch_put(&b, v, id, 0); }
    r = store_batch_commit(&b, 1);
    store_batch_free(&b);
    if (r != 3 || sdb_doc_(s, 0) != 0 || sdb_doc_(s, 5) != 2 || store_count(s) != 10) { print_cstr("  batches: the second commit FAILED\n"); ok = 0; }
    jlen = sdb_fsize_(jn);
    if (jlen > 40 && (jb = (u8 *)malloc((size_t)jlen)) != (u8 *)0) {
        FILE *f = fopen(jn, "rb");
        i32 got = f && fread(jb, 1, (size_t)jlen, f) == (size_t)jlen;
        if (f) fclose(f);
        flips = 0;
        for (i = 0; i < 3 && got; ++i) {                          /* cut short, one byte changed, as it was */
            i32 dropped = i < 2;
            if (i == 0) sdb_write_(jn, jb, jlen - 5);
            else if (i == 1) { jb[jlen - 20] ^= 0x5a; sdb_write_(jn, jb, jlen); jb[jlen - 20] ^= 0x5a; }
            else sdb_write_(jn, jb, jlen);
            if (!(l = store_load_ro(a, path))) { ok = 0; continue; }
            if (dropped ? (sdb_doc_(l, 0) == 5 && sdb_doc_(l, 5) == 0 && store_count(l) == 13)
                        : (sdb_doc_(l, 0) == 0 && sdb_doc_(l, 5) == 2 && store_count(l) == 10)) ++flips;
            store_destroy(l);
        }
        if (!got || flips != 3) { print_cstr("  batches: a torn or damaged batch in the journal FAILED\n"); ok = 0; }
        free(jb);
    } else { print_cstr("  batches: no journal to damage\n"); ok = 0; }
    store_destroy(s);

    /* a reader never sees half a batch: document 7 swaps between 5 chunks and 3, 300 times */
    g = store_new(a, SDB_DIM);
    for (c = 0; c < 5; ++c) { i64 id = ((i64)7 << 20) | c; sdb_vec_(id, v); store_put(g, v, id, 0); }
    memset(&rd, 0, sizeof(rd)); rd.s = g;
    pthread_create(&th[0], (const pthread_attr_t *)0, sdb_reader_, &rd);
    for (i = 0; i < 300; ++i) {
        store_batch_init(&b, g);
        store_batch_delete_range(&b, (i64)7 << 20, ((i64)7 << 20) | 0xfffff);
        for (c = 0; c < (i % 2 ? 5 : 3); ++c) { i64 id = ((i64)7 << 20) | (i * 8 + c); sdb_vec_(id, v); store_batch_put(&b, v, id, 0); }
        if (store_batch_commit(&b, 0) < 0) ok = 0;
        store_batch_free(&b);
    }
    __atomic_store_n(&rd.stop, 1, __ATOMIC_RELEASE);
    pthread_join(th[0], (void **)0);
    print_cstr("  batches: a reader looked "); print_i64(rd.reads); print_cstr(" times while 300 batches swapped a document");
    print_cstr(rd.bad ? ": saw half of one\n" : ", never saw half of one\n");
    if (rd.bad || rd.reads == 0) ok = 0;
    store_destroy(g);

    /* durable commits from 4 threads at once share fsyncs; every row is on the disk after */
    g = store_new(a, SDB_DIM);
    if (store_save(g, path2) != 0) { ok = 0; print_cstr("  batches: save FAILED\n"); }
    for (i = 0; i < 4; ++i) { cm[i].s = g; cm[i].t = i; cm[i].bad = 0; pthread_create(&th[i], (const pthread_attr_t *)0, sdb_committer_, &cm[i]); }
    for (i = 0; i < 4; ++i) { pthread_join(th[i], (void **)0); if (cm[i].bad) ok = 0; }
    if (store_count(g) != 400) { print_cstr("  batches: durable commits from 4 threads FAILED\n"); ok = 0; }
    if (!(l = store_load_ro(a, path2)) || store_count(l) != 400) { print_cstr("  batches: the synced batches are not all on the disk\n"); ok = 0; }
    if (l) store_destroy(l);
    print_cstr("  batches: 4 threads x 50 durable commits -> "); print_i64(store_count(g)); print_cstr(" rows, on the disk too\n");
    store_destroy(g);
    remove(path); remove(jn); remove(path2); remove(jn2);
    free(jn); free(jn2);
    return ok;
}

/* ---- the cascade: the 1-bit codes derived from the 4-bit ones are a 1-bit build's codes exactly, rows added later
 * get theirs, it finds what the 4-bit scan finds (IVF), its hot part is smaller, the file keeps it ---- */
#define SDC_DIM 32
#define SDC_N   20000
static void sdc_unit_(f32 *v, u32 *st)
{
    f32 nn = 0.0f;
    i32 d;
    for (d = 0; d < SDC_DIM; ++d) { v[d] = sd_f_(st); nn += v[d] * v[d]; }
    nn = 1.0f / (f32)sqrt((f64)nn);
    for (d = 0; d < SDC_DIM; ++d) v[d] *= nn;
}
static i32 sd_cascade_(Arena *a)
{
    const char *path = "/tmp/store_demo_cascade.vxi";
    char *jn = store_jname_(path, ".j");
    f32 *x = (f32 *)malloc((size_t)SDC_N * SDC_DIM * sizeof(f32)), *Q = (f32 *)malloc((size_t)50 * SDC_DIM * sizeof(f32));
    u8 *hold = (u8 *)0;
    TqCol col;
    TqAdd ad;
    Tq *t4, *t1;
    RagStore *s, *l;
    i64 h4, c4, h1, c1;
    f32 r4, rc, rl, ro;
    i32 i, b, ok = 1, same = 1, keep = store_ivf_min_;
    u32 st = 31337u;
    for (i = 0; i < SDC_N; ++i) sdc_unit_(x + (size_t)i * SDC_DIM, &st);
    for (i = 0; i < 50; ++i) sdc_unit_(Q + (size_t)i * SDC_DIM, &st);

    /* tq: 2048 rows (64 full blocks) with the cascade, 512 more added; a flat 1-bit build of all 2560 has the same
     * rotation and slots, so its codes must be the cascade's 1-bit codes, byte for byte */
    col = tq_col_flat(x, SDC_DIM * sizeof(f32), &hold);
    t4 = tq_build(&col, 2048, SDC_DIM, 9u, 0, 1, 4);
    t1 = tq_build(&col, 2560, SDC_DIM, 9u, 0, 1, 1);
    memset(&ad, 0, sizeof(ad));
    if (!t4 || !t1 || tq_cascade_on(t4, 1) != 0 || t4->vd1 != t1->vd || tq_cascade_on(t1, 1) != -1) ok = 0;
    else if (tq_add_prep(t4, &col, 2048, 512, 1, &ad) != 0 || tq_add_put(t4, &ad, 2048) != 0) ok = 0;
    tq_add_free(&ad);
    for (b = 0; ok && b < 2560 / 32; ++b) if (memcmp(tq_bptr1_(t4, b), tq_bptr_(t1, b), (size_t)t1->vd * 16) != 0) same = 0;
    if (!ok || !same) { print_cstr("  cascade: the derived 1-bit codes FAILED\n"); ok = 0; }
    tq_free(t4); tq_free(t1);

    /* a store with IVF: the cascade's recall against the 4-bit scan's, its hot part, saved and loaded, switched off */
    store_ivf_min_ = 1000;
    s = store_new(a, SDC_DIM);
    for (i = 0; i < SDC_N; ++i) store_add(s, x + (size_t)i * SDC_DIM, 50000 + i);
    store_build(s);
    r4 = sd_recall_(s, x, SDC_N, SDC_DIM, Q, 50, 50000, (const u8 *)0);
    store_tiers(s, &h4, &c4);
    if (store_set_cascade(s, 1) != 0 || !s->tq || !s->tq->codes1 || s->tq->nc < 2) { print_cstr("  cascade: switching it on FAILED\n"); ok = 0; }
    rc = sd_recall_(s, x, SDC_N, SDC_DIM, Q, 50, 50000, (const u8 *)0);
    store_tiers(s, &h1, &c1);
    print_cstr("  cascade: recall@10 "); print_i64((i64)(r4 * 1000.0f)); print_cstr("/1000 with 4-bit codes, ");
    print_i64((i64)(rc * 1000.0f)); print_cstr("/1000 through the cascade; hot "); print_i64(h4); print_cstr(" B -> ");
    print_i64(h1); print_cstr(" B\n");
    if (rc < r4 - 0.05f || h1 >= h4 || c1 <= c4) { print_cstr("  cascade: recall or the hot part FAILED\n"); ok = 0; }
    remove(path); remove(jn);
    if (store_save(s, path) != 0 || !(l = store_load(a, path))) { print_cstr("  cascade: save/load FAILED\n"); ok = 0; l = (RagStore *)0; }
    if (l) {
        rl = sd_recall_(l, x, SDC_N, SDC_DIM, Q, 50, 50000, (const u8 *)0);
        if (!l->cascade || !l->tq || !l->tq->codes1 || rl < rc - 0.01f || rl > rc + 0.01f) { print_cstr("  cascade: the file did not keep it\n"); ok = 0; }
        if (store_set_cascade(l, 0) != 0 || l->tq->codes1) ok = 0;
        ro = sd_recall_(l, x, SDC_N, SDC_DIM, Q, 50, 50000, (const u8 *)0);
        if (ro < r4 - 0.01f || ro > r4 + 0.01f) { print_cstr("  cascade: switched off, the 4-bit scan came back different\n"); ok = 0; }
        store_destroy(l);
    }
    store_destroy(s);
    store_ivf_min_ = keep;
    remove(path); remove(jn); free(jn);
    free(x); free(Q);
    return ok;
}

/* ---- integrity: BLAKE3, the block digests and Reed-Solomon parity of the file, the journal's frames ---- */
/* numbers on rows: a range filter tested inside the IVF scan finds the few rows of a narrow range far from the
 * query (a filter applied only after the scan would see none of them among its candidates), a wide range keeps the
 * recall of an exact search over the rows in it, and the cascade does the same */
static i32 sdn_exact_(const f32 *x, const f32 *q, i64 lo, i64 hi, i32 k, i32 *top)   /* the k best rows with i in [lo, hi] */
{
    f32 best[16];
    i32 i, j, d, n = 0;
    for (i = (i32)lo; i <= (i32)hi && i < SDC_N; ++i) {
        f32 dp = 0.0f;
        for (d = 0; d < SDC_DIM; ++d) dp += x[(size_t)i * SDC_DIM + d] * q[d];
        if (n == k && dp <= best[k - 1]) continue;
        for (j = n < k ? n++ : k - 1; j > 0 && best[j - 1] < dp; --j) { best[j] = best[j - 1]; top[j] = top[j - 1]; }
        best[j] = dp; top[j] = i;
    }
    return n;
}
static i32 sd_nums_(Arena *a)
{
    f32 *x = (f32 *)malloc((size_t)SDC_N * SDC_DIM * sizeof(f32));
    RagStore *s;
    StoreCnf c;
    i64 ids[16];
    f32 sc[16];
    i32 i, q, j, n, ok = 1, pass, hit = 0, tot = 0, uhit = 0, utot = 0, keep = store_ivf_min_, top[16];
    u32 st = 4242u;
    for (i = 0; i < SDC_N; ++i) sdc_unit_(x + (size_t)i * SDC_DIM, &st);
    store_ivf_min_ = 1000;
    s = store_new(a, SDC_DIM);
    for (i = 0; i < SDC_N; ++i) store_add(s, x + (size_t)i * SDC_DIM, i);
    store_build(s);
    for (i = 0; i < SDC_N; ++i) if (store_set_num(s, i, i) != 0) ok = 0;
    if (!s->tq || s->tq->nc < 2 || !s->has_num) { print_cstr("  numbers: no IVF scan index to test FAILED\n"); ok = 0; }
    for (pass = 0; ok && pass < 2; ++pass) {
        if (pass == 1 && store_set_cascade(s, 1) != 0) { ok = 0; break; }
        memset(&c, 0, sizeof(c)); c.num = 1;
        for (q = 0; q < 20; ++q) {                               /* five rows, anywhere: all five, and only they */
            c.lo = 977 * q + 3; c.hi = c.lo + 4;
            n = store_search_cnf(s, x + (size_t)(q * 13) * SDC_DIM, 10, ids, sc, &c);
            if (n != 5) ok = 0;
            for (j = 0; j < n; ++j) if (ids[j] < c.lo || ids[j] > c.hi) ok = 0;
        }
        if (pass == 0) for (i = 0; i < SDC_N; i += 2) store_set_meta(s, i, 1);   /* a tag bit on the even rows */
        c.n = 1; c.any[0] = 1;
        for (q = 0; q < 20; ++q) {                               /* ten rows, the tag on five: those five */
            c.lo = 991 * q + 1; c.hi = c.lo + 9;
            n = store_search_cnf(s, x + (size_t)(q * 17) * SDC_DIM, 10, ids, sc, &c);
            if (n != 5) ok = 0;
            for (j = 0; j < n; ++j) if (ids[j] < c.lo || ids[j] > c.hi || (ids[j] & 1)) ok = 0;
        }
        c.n = 0; c.any[0] = 0;
        for (q = 0; q < 50; ++q) {                               /* half the rows: recall@10 against the exact answer */
            const f32 *qv = x + (size_t)(q * 211 + 5) * SDC_DIM;
            i32 m;
            c.lo = 10000; c.hi = SDC_N - 1;
            n = store_search_cnf(s, qv, 10, ids, sc, &c);
            m = sdn_exact_(x, qv, c.lo, c.hi, 10, top);
            for (j = 0; j < n; ++j) { i32 z; if (ids[j] < c.lo) ok = 0; for (z = 0; z < m; ++z) if (top[z] == ids[j]) { ++hit; break; } }
            tot += m;
            n = store_search(s, qv, 10, ids, sc);                /* the same query unfiltered, for the baseline */
            m = sdn_exact_(x, qv, 0, SDC_N - 1, 10, top);
            for (j = 0; j < n; ++j) { i32 z; for (z = 0; z < m; ++z) if (top[z] == ids[j]) { ++uhit; break; } }
            utot += m;
        }
    }
    print_cstr("  numbers: narrow ranges found whole, wide-range recall@10 "); print_i64((i64)(1000 * hit / (tot ? tot : 1)));
    print_cstr("/1000 against "); print_i64((i64)(1000 * uhit / (utot ? utot : 1)));
    print_cstr("/1000 unfiltered (4-bit scan and cascade)\n");
    if (!ok || hit * 1000 < (i64)tot * (1000 * uhit / (utot ? utot : 1) - 30)) { print_cstr("  numbers FAILED\n"); ok = 0; }
    {   /* a namespace too big for its list: searched through the IVF scan with the namespace tested inside it - 50 rows
         * spread over 20000, so only the test inside the scan finds 10 of them; none of another namespace; a namespace
         * and a number range together find their one row */
        i32 keepx = store_ns_exact_max_, iso = 1, okr = 1, nn;
        StoreCnf cn;
        store_ns_exact_max_ = 10;
        store_set_cascade(s, 0);
        for (i = 0; i < SDC_N; i += 400) if (store_set_ns(s, i, 9) != 0) iso = 0;
        for (q = 0; q < 30; ++q) {
            nn = store_search_ns(s, 9, x + (size_t)(q * 101 + 2) * SDC_DIM, 10, ids, sc);
            if (nn != 10) iso = 0;
            for (j = 0; j < nn; ++j) if (ids[j] % 400) iso = 0;
            memset(&cn, 0, sizeof(cn)); cn.ns_on = 1; cn.ns = 9; cn.num = 1; cn.lo = 400 * q; cn.hi = cn.lo + 399;
            nn = store_search_cnf(s, x + (size_t)(q * 13) * SDC_DIM, 10, ids, sc, &cn);
            if (nn != 1 || ids[0] != 400 * q) okr = 0;
        }
        print_cstr(iso && okr ? "  namespaces: a big one through the IVF scan - 10 of its 50 rows in 20000, none of another; with a range its one row\n" : "  namespaces in the scan FAILED\n");
        if (!iso || !okr) ok = 0;
        {   /* several at once (store_search_nss): through the scan, only rows of the set and of both namespaces; exact
             * (the lists small enough together), the merge of the namespaces' own exact searches, whatever the order,
             * repeats or unknown namespaces in the set */
            i64 set1[2], set2[5], ia[10], ib[10], m2[20];
            f32 sa[10], sb[10], ms[20];
            i32 inset = 1, both7 = 0, both9 = 0, same = 1, na, nb, z, h;
            set1[0] = 9; set1[1] = 7;
            set2[0] = 7; set2[1] = 12345; set2[2] = 9; set2[3] = 7; set2[4] = -3;
            for (i = 200; i < SDC_N; i += 400) if (store_set_ns(s, i, 7) != 0) inset = 0;
            for (q = 0; q < 30; ++q) {
                nn = store_search_nss(s, set1, 2, x + (size_t)(q * 97 + 3) * SDC_DIM, 10, ids, sc);
                if (nn != 10) inset = 0;
                for (j = 0; j < nn; ++j) { if (ids[j] % 400 == 0) ++both9; else if (ids[j] % 400 == 200) ++both7; else inset = 0; }
            }
            store_ns_exact_max_ = keepx;
            for (q = 0; q < 30; ++q) {
                const f32 *qv = x + (size_t)(q * 89 + 11) * SDC_DIM;
                nn = store_search_nss(s, set2, 5, qv, 10, ids, sc);
                na = store_search_ns(s, 9, qv, 10, ia, sa);
                nb = store_search_ns(s, 7, qv, 10, ib, sb);
                for (z = h = 0; z < na; ++z) { m2[h] = ia[z]; ms[h++] = sa[z]; }
                for (z = 0; z < nb; ++z) { m2[h] = ib[z]; ms[h++] = sb[z]; }
                for (z = 0; z < h; ++z) {                        /* best first: a selection sort of at most 20 */
                    i32 b2 = z, y;
                    for (y = z + 1; y < h; ++y) if (ms[y] > ms[b2]) b2 = y;
                    { i64 ti = m2[z]; f32 ts = ms[z]; m2[z] = m2[b2]; ms[z] = ms[b2]; m2[b2] = ti; ms[b2] = ts; }
                }
                if (nn != (h < 10 ? h : 10)) same = 0;
                for (j = 0; same && j < nn; ++j) if (ids[j] != m2[j]) same = 0;
            }
            print_cstr(inset && both7 && both9 && same ? "  several namespaces: through the scan only rows of the set, of both; exact, the merge of each one's exact search\n"
                                                       : "  several namespaces FAILED\n");
            if (!inset || !both7 || !both9 || !same) ok = 0;
            for (i = 200; i < SDC_N; i += 400) store_set_ns(s, i, 0);
        }
        store_ns_exact_max_ = keepx;
        store_set_cascade(s, 1);
    }
    {   /* saved and loaded cold - the codes taken as larger than RAM, so each search asks for its clusters - and warm:
         * the same hits, for plain, range and tag searches, with the cascade on (pass 1 left it on) */
        const char *path = "/tmp/store_demo_cold.vxi";
        RagStore *wm = (RagStore *)0, *cd = (RagStore *)0;
        i64 i2[10], i3[10];
        f32 s2[10], s3[10];
        i32 same = 1, n2, n3;
        remove(path);
        if (store_save(s, path) != 0 || !(wm = store_load(a, path))) same = 0;
        store_cold_min_ = 0;
        if (same && !(cd = store_load(a, path))) same = 0;
        store_cold_min_ = -1;
        if (same && (!cd->tq || !cd->tq->cold || !cd->tq->codes1 || !wm->tq || wm->tq->cold)) same = 0;
        for (q = 0; same && q < 30; ++q) {
            const f32 *qv = x + (size_t)(q * 331 + 7) * SDC_DIM;
            memset(&c, 0, sizeof(c));
            if (q % 3 == 1) { c.num = 1; c.lo = 1000; c.hi = 9000; }
            if (q % 3 == 2) { c.n = 1; c.any[0] = 1; }
            n2 = q % 3 ? store_search_cnf(wm, qv, 10, i2, s2, &c) : store_search(wm, qv, 10, i2, s2);
            n3 = q % 3 ? store_search_cnf(cd, qv, 10, i3, s3, &c) : store_search(cd, qv, 10, i3, s3);
            if (n2 != n3 || n2 != 10) same = 0;
            for (j = 0; same && j < n2; ++j) if (i2[j] != i3[j]) same = 0;
        }
        print_cstr(same ? "  cold index: a store loaded as larger than RAM gives the warm one's hits (plain, range, tag; cascade)\n"
                        : "  cold index FAILED\n");
        if (!same) ok = 0;
        if (wm) store_destroy(wm);
        if (cd) store_destroy(cd);
        remove(path);
        {   char jn[64]; sprintf(jn, "%s.j", path); remove(jn); }
    }
    store_destroy(s);
    store_ivf_min_ = keep;
    free(x);
    return ok;
}

static void sdx_flip_(const char *path, u64 off)
{
    FILE *f = fopen(path, "r+b");
    int c;
    if (!f) return;
    if (fseeko(f, (off_t)off, SEEK_SET) == 0 && (c = fgetc(f)) != EOF && fseeko(f, (off_t)off, SEEK_SET) == 0) fputc(c ^ 0x5a, f);
    fclose(f);
}
static u8 *sdx_slurp_(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    u8 *b = (u8 *)0;
    long len;
    *n = 0;
    if (!f) return b;
    if (fseek(f, 0, SEEK_END) == 0 && (len = ftell(f)) > 0 && fseek(f, 0, SEEK_SET) == 0 && (b = (u8 *)malloc((size_t)len)) != 0
        && fread(b, 1, (size_t)len, f) == (size_t)len) *n = (size_t)len;
    fclose(f);
    return b;
}
static void sdx_spit_(const char *path, const u8 *b, size_t n)
{
    FILE *f = fopen(path, "wb");
    if (!f) return;
    if (fwrite(b, 1, n, f) != n) print_cstr("  (write failed)\n");
    fclose(f);
}
static i32 sdx_trailer_(const char *path, SSum *z)
{
    int fd = open(path, O_RDONLY);
    struct stat st;
    u8 t[SSUM_TRL];
    i32 r = -1;
    if (fd < 0) return -1;
    if (fstat(fd, &st) == 0 && ssum_read_(fd, t, SSUM_TRL, (u64)st.st_size - SSUM_TRL) == 0) r = ssum_parse_(z, t, (u64)st.st_size);
    close(fd);
    return r;
}
static i32 sdx_same_(RagStore *l, const f32 *ref, i32 nref)          /* rows 0, 700, 1400, ... read back unchanged */
{
    f32 v[24];
    i32 i;
    for (i = 0; i < nref; ++i) if (store_get_vec(l, 5000 + 700 * i, v) != 0 || memcmp(v, ref + i * 24, sizeof(v)) != 0) return 0;
    return 1;
}

/* the offset of needle in the file's bytes, or -1 */
static i64 sdl_find_(const u8 *b, size_t n, const char *needle)
{
    size_t k = strlen(needle), i;
    for (i = 0; i + k <= n; ++i) if (b[i] == (u8)needle[0] && !memcmp(b + i, needle, k)) return (i64)i;
    return -1;
}
/* the lens has walked every block (bounded wait) */
static i32 sdl_wait_(RagStore *l)
{
    i64 d = 0, t = 1;
    i32 i;
    for (i = 0; i < 2000; ++i) {
        store_lens_info(l, &d, &t);
        if (t > 0 && d >= t) return 1;
        { struct timespec ts; ts.tv_sec = 0; ts.tv_nsec = 5000000L; nanosleep(&ts, (struct timespec *)0); }
    }
    return 0;
}

#define SDX_N 3000
#define SDX_DIM 24
static i32 sd_sums_(Arena *a)
{
    static const char *hx[] = {        /* length, digest: the reference implementation's, input byte i = i mod 251 */
        "0", "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262",
        "1", "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213",
        "63", "e9bc37a594daad83be9470df7f7b3798297c3d834ce80ba85d6e207627b7db7b",
        "64", "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98",
        "65", "de1e5fa0be70df6d2be8fffd0e99ceaa8eb6e8c93a63f2d8d1c30ecb6b263dee",
        "1023", "10108970eeda3eb932baac1428c7a2163b0e924c9a9e25b35bba72b28f70bd11",
        "1024", "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7",
        "1025", "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444",
        "2048", "e776b6028c7cd22a4d0ba182a8bf62205d2ef576467e838ed6f2529b85fba24a",
        "2049", "5f4d72f40d7a5f82b15ca2b2e44b1de3c2ef86c426c95c1af0b6879522563030",
        "3072", "b98cb0ff3623be03326b373de6b9095218513e64f1ee2edd2525c7ad1e5cffd2",
        "4097", "9b4052b38f1c5fc8b1f9ff7ac7b27cd242487b3d890d15c96a1c25b8aa0fb995",
        "8193", "bab6c09cb8ce8cf459261398d2e7aef35700bf488116ceb94a36d0f5f1b7bc3b",
        "31744", "62b6960e1a44bcc1eb1a611a8d6235b6b4b78f32e7abc4fb4c6cdcce94895c47",
        "102400", "bc3e3d41a1146b069abffad3c0d44860cf664390afce4d9661f7902e7943e085",
        (const char *)0 };
    const char *path = "/tmp/store_demo_sums.vxi", *bare = "/tmp/store_demo_sums_bare.vxi";
    char *jn = store_jname_(path, ".j"), *jd = store_jname_(path, ".j.damaged"), hex[65];
    f32 *x = (f32 *)malloc((size_t)SDX_N * SDX_DIM * sizeof(f32)), ref[5 * SDX_DIM];
    u8 *buf = (u8 *)malloc(102400), *clean, h[32];
    size_t clen;
    RagStore *s, *l;
    SSum z;
    i64 nb, bad, fx, r1, r2, f1, rb = 0, rh = 0;
    i32 i, e, ok = 1, jdm;
    u32 st = 4242u;
    struct stat sst;

    for (i = 0; i < 102400; ++i) buf[i] = (u8)(i % 251);
    for (i = 0; hx[i]; i += 2) {
        b3_hash(buf, (size_t)atoi(hx[i]), h);
        for (e = 0; e < 32; ++e) sprintf(hex + 2 * e, "%02x", (unsigned)h[e]);
        if (strcmp(hex, hx[i + 1]) != 0) { print_cstr("  blake3: the digest of "); print_cstr(hx[i]); print_cstr(" bytes FAILED\n"); ok = 0; }
    }
    b3_hash("abc", 3, h);
    for (e = 0; e < 32; ++e) sprintf(hex + 2 * e, "%02x", (unsigned)h[e]);
    if (strcmp(hex, "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85") != 0) { print_cstr("  blake3: abc FAILED\n"); ok = 0; }

    /* a store over several blocks, parity 1 for every 2 blocks: block b and b + G share a group */
    remove(path); remove(jn); remove(jd); remove(bare);
    for (i = 0; i < SDX_N * SDX_DIM; ++i) x[i] = sd_f_(&st);
    s = store_new(a, SDX_DIM);
    if (store_set_parity(s, 2, 1) != 0 || store_set_parity(s, 0, 1) != -1 || store_set_parity(s, 2, 65) != -1) ok = 0;
    for (i = 0; i < SDX_N; ++i) store_add(s, x + (size_t)i * SDX_DIM, 5000 + i);
    store_build(s);
    if (store_save(s, path) != 0 || sdx_trailer_(path, &z) != 0 || z.k != 2 || z.m != 1 || z.N < 5) {
        print_cstr("  sums: the trailer of a saved store FAILED\n"); store_destroy(s); free(x); free(buf); free(jn); free(jd); return 0;
    }
    store_destroy(s);
    clean = sdx_slurp_(path, &clen);
    l = store_load(a, path);
    if (!l) ok = 0;
    else {
        store_integrity(l, &nb, &bad, &fx, &rb, &rh, &jdm);
        if (nb != (i64)(z.N + z.P) || bad || fx || jdm) { print_cstr("  sums: a clean load FAILED\n"); ok = 0; }
        for (i = 0; i < 5; ++i) store_get_vec(l, 5000 + 700 * i, ref + i * SDX_DIM);
        store_destroy(l);
    }
    if (store_scrub(path, 0, &fx) != 0) { print_cstr("  sums: a clean scrub FAILED\n"); ok = 0; }

    /* one damaged block of rows: rebuilt at load, written back, the rows as they were */
    sdx_flip_(path, ((u64)1 << 16) + 1000);
    l = store_load(a, path);
    if (!l) { print_cstr("  sums: a load with one damaged block FAILED\n"); ok = 0; }
    else {
        store_integrity(l, &nb, &bad, &fx, &rb, &rh, &jdm);
        if (bad != 1 || fx != 1 || !sdx_same_(l, ref, 5)) { print_cstr("  sums: rebuilding a damaged block FAILED\n"); ok = 0; }
        store_destroy(l);
    }
    if (store_scrub(path, 0, &fx) != 0) { print_cstr("  sums: the rebuilt block was not written back\n"); ok = 0; }

    /* the disk goes bad under a loaded store: a row read later fails its check, its block is rebuilt from the parity
     * and written back, the row is served as it was (ZFS heals a read the same way); a row damaged in memory, where
     * there is nothing to rebuild from, is left out */
    l = store_load(a, path);
    if (!l) ok = 0;
    else {
        f32 v[SDX_DIM];
        i64 hits[4]; f32 hsc[4];
        if (l->map) {   /* a store that is not mapped (a big-endian CPU) holds its rows in memory and never reads them back from the file */
            sdx_flip_(path, 64 + (u64)700 * SDX_DIM * 4 + 10);   /* row 700 (id 5700): v9 rows start at byte 64 */
            if (store_get_vec(l, 5700, v) != 0 || memcmp(v, ref + SDX_DIM, sizeof(v)) != 0) { print_cstr("  sums: healing a row read after the load FAILED\n"); ok = 0; }
            store_integrity(l, &nb, &bad, &fx, &rb, &rh, &jdm);
            if (rh != 1 || rb != 0) { print_cstr("  sums: the healed row was not counted\n"); ok = 0; }
        } else print_cstr("  sums: the store is not mapped here (a big-endian CPU) - healing a row on read is not checked\n");
        store_add(l, x + (size_t)5 * SDX_DIM, 7777);                /* a row only in memory, then a bit of it flips */
        STORE_VEC(l, l->count - 1)[3] += 0.25f;
        if (store_get_vec(l, 7777, v) == 0) { print_cstr("  sums: a row damaged in memory was served\n"); ok = 0; }
        if (store_search(l, x + (size_t)5 * SDX_DIM, 4, hits, hsc) > 0 && hits[0] == 7777) { print_cstr("  sums: a damaged row came out of a search\n"); ok = 0; }
        store_integrity(l, &nb, &bad, &fx, &rb, &rh, &jdm);
        if (rb < 1) { print_cstr("  sums: the damaged row was not counted\n"); ok = 0; }
        store_destroy(l);
    }
    if (store_scrub(path, 0, &fx) != 0) { print_cstr("  sums: the block healed on read was not written back\n"); ok = 0; }

    /* two damaged blocks of one group, parity 1: the rows are lost - the load says so instead of serving them */
    sdx_spit_(path, clean, clen);
    sdx_flip_(path, ((u64)1 << 16) + 1000);
    sdx_flip_(path, ((u64)(1 + z.G) << 16) + 2000);
    l = store_load(a, path);
    if (l || !*store_load_error()) { print_cstr("  sums: refusing damage beyond repair FAILED\n"); ok = 0; }
    if (l) store_destroy(l);
    if (store_scrub(path, 1, &fx) != 2 || fx != 0) { print_cstr("  sums: a scrub of damage beyond repair FAILED\n"); ok = 0; }

    /* a damaged copy of the trailer, of the table, a damaged parity block: the load uses the other copy, a scrub mends */
    sdx_spit_(path, clean, clen);
    sdx_flip_(path, (u64)clen - 60);
    sdx_flip_(path, z.ta + 5);
    sdx_flip_(path, (z.N << 16) + 7);
    l = store_load(a, path);
    if (!l || !sdx_same_(l, ref, 5)) { print_cstr("  sums: loading past damaged copies FAILED\n"); ok = 0; }
    if (l) store_destroy(l);
    r1 = store_scrub(path, 1, &f1);
    r2 = store_scrub(path, 0, &fx);
    if (r1 != 3 || f1 != 3 || r2 != 0) { print_cstr("  sums: a scrub mending the copies and the parity FAILED\n"); ok = 0; }

    /* the data without its checksums: refused, loaded only unchecked; a scrub finds no table */
    sdx_spit_(bare, clean, (size_t)z.D);
    l = store_load(a, bare);
    if (l) { print_cstr("  sums: a v9 file without checksums was loaded\n"); ok = 0; store_destroy(l); }
    store_set_verify(0);
    l = store_load_ro(a, bare);
    store_set_verify(2);
    if (!l || store_count(l) != SDX_N) { print_cstr("  sums: loading it unchecked FAILED\n"); ok = 0; }
    if (l) store_destroy(l);
    if (store_scrub(bare, 0, &fx) != -3) { print_cstr("  sums: a scrub without a table FAILED\n"); ok = 0; }

    /* the journal: every record in a frame with its CRC. A damaged frame in the middle: replay stops before it and a
     * copy is kept; a torn tail: cut quietly, no copy */
    sdx_spit_(path, clean, clen);
    remove(jn);
    l = store_load(a, path);
    if (l) {
        for (i = 0; i < 10; ++i) store_add(l, x + (size_t)i * SDX_DIM, 9000 + i);
        store_sync(l, path);
        store_destroy(l);
    }
    sdx_flip_(jn, 20 + 5 * (9 + 17 + 4 * SDX_DIM) + 30);           /* VXJ3 header 20 bytes; 'A' frames of 9 + 17 + 4 dim */
    l = store_load(a, path);
    if (!l) ok = 0;
    else {
        store_integrity(l, &nb, &bad, &fx, &rb, &rh, &jdm);
        if (!jdm || !store_has(l, 9004) || store_has(l, 9005) || stat(jd, &sst) != 0) { print_cstr("  journal: a damaged frame FAILED\n"); ok = 0; }
        store_destroy(l);
    }
    remove(jd); remove(jn);
    sdx_spit_(path, clean, clen);
    l = store_load(a, path);
    if (l) {
        for (i = 0; i < 3; ++i) store_add(l, x + (size_t)i * SDX_DIM, 9100 + i);
        store_sync(l, path);
        store_destroy(l);
    }
    if (truncate(jn, (off_t)(20 + 3 * (9 + 17 + 4 * SDX_DIM) - 10)) != 0) ok = 0;
    l = store_load(a, path);
    if (!l) ok = 0;
    else {
        store_integrity(l, &nb, &bad, &fx, &rb, &rh, &jdm);
        if (jdm || !store_has(l, 9101) || store_has(l, 9102) || stat(jd, &sst) == 0) { print_cstr("  journal: a torn tail FAILED\n"); ok = 0; }
        store_destroy(l);
    }
    if (ok) print_cstr("  integrity: BLAKE3 matches the reference; damaged blocks rebuilt from Reed-Solomon parity and written\n"
                       "             back, at load and on a read after it; damage beyond repair refused or left out; copies of the trailer and table mended by a scrub;\n"
                       "             a damaged journal frame stops replay with a copy kept, a torn tail does not\n");
    remove(path); remove(jn); remove(jd); remove(bare);
    free(clean); free(x); free(buf); free(jn); free(jd);
    return ok;
}

/* the lens (VECINDEX_VERIFY=lens, the default above half of RAM): the load checks only the header's group, the index's
 * small arrays and the label table - a damaged label table is rebuilt there; a text is checked before it is read -
 * rebuilt, or not served when it cannot be; a thread walks the rest one block at a time and rebuilds what it finds;
 * a store being closed stops it at once */
static i32 sd_lens_(Arena *a)
{
    const char *path = "/tmp/store_demo_lens.vxi";
    char jn[64], tb[64];
    f32 *x = (f32 *)malloc((size_t)SDX_N * SDX_DIM * sizeof(f32));
    u8 *clean;
    size_t clen;
    u32 st = 99u;
    RagStore *s, *l;
    SSum z;
    StoreCnf c;
    const char *bad = (const char *)0;
    i64 nb, bd, fx, rb, rh, d, t, off;
    i32 i, ok = 1, jdm, keep = store_lens_ms_;
    sprintf(jn, "%s.j", path);
    remove(path); remove(jn);
    for (i = 0; i < SDX_N * SDX_DIM; ++i) x[i] = sd_f_(&st);
    s = store_new(a, SDX_DIM);
    store_set_parity(s, 2, 1);
    for (i = 0; i < SDX_N; ++i) {
        char txt[48];
        sprintf(txt, "LENSTEXT-%d some words about row %d", i, i);
        if (i == 7) {                                            /* 400 kB: blocks that hold nothing but this text */
            char *big = (char *)malloc(400000);
            memset(big, 'w', 400000); memcpy(big, "LENSTEXT-7 ", 11); memcpy(big + 200000, "MIDMARK", 7);
            store_put_text(s, x + (size_t)i * SDX_DIM, 5000 + i, 0, big, 400000);
            free(big);
        } else if (i < 10) store_put_text(s, x + (size_t)i * SDX_DIM, 5000 + i, 0, txt, (i32)strlen(txt));
        else store_add(s, x + (size_t)i * SDX_DIM, 5000 + i);
    }
    if (store_label_row(s, 5005, "lenslabel", 1, &bad) != 0) ok = 0;
    store_build(s);
    if (store_save(s, path) != 0 || sdx_trailer_(path, &z) != 0) { store_destroy(s); free(x); print_cstr("  lens: save FAILED\n"); return 0; }
    store_destroy(s);
    clean = sdx_slurp_(path, &clen);
    store_lens_ms_ = 0;
    store_set_verify(3);

    l = store_load(a, path);                                     /* clean: every block walked, nothing found */
    if (!l || !sdl_wait_(l)) ok = 0;
    else {
        store_lens_info(l, &d, &t);
        store_integrity(l, &nb, &bd, &fx, &rb, &rh, &jdm);
        if (t != (i64)(z.N + z.P) || bd || fx) { print_cstr("  lens: a clean walk FAILED\n"); ok = 0; }
    }
    if (l) store_destroy(l);

    off = sdl_find_(clean, clen, "LENSTEXT-3 ");                 /* a text's block: rebuilt before the text is read */
    sdx_flip_(path, (u64)off + 2);
    l = store_load(a, path);
    if (!l || store_get_text(l, 5003, tb, sizeof(tb)) != 33 || memcmp(tb, "LENSTEXT-3 some words about row 3", 33)) {
        print_cstr("  lens: a damaged text rebuilt before it is read FAILED\n"); ok = 0;
    }
    if (l) { sdl_wait_(l); store_integrity(l, &nb, &bd, &fx, &rb, &rh, &jdm); if (fx < 1) ok = 0; store_destroy(l); }
    if (store_scrub(path, 0, &fx) != 0) { print_cstr("  lens: the rebuilt text block was not written back\n"); ok = 0; }

    sdx_spit_(path, clean, clen);                                /* the label table: checked, and rebuilt, at the load */
    off = sdl_find_(clean, clen, "lenslabel");
    sdx_flip_(path, (u64)off + 1);
    l = store_load(a, path);
    if (!l || store_label_cnf(l, "lenslabel", (const char *)0, (const char *)0, &c, &bad) != 0) { print_cstr("  lens: a damaged label table FAILED\n"); ok = 0; }
    if (l) store_destroy(l);
    store_lens_ms_ = 1000;                                       /* the thread asleep: the load alone wrote it back */
    l = store_load(a, path);
    if (l) store_destroy(l);
    store_lens_ms_ = 0;

    sdx_spit_(path, clean, clen);                                /* a text's block and its group's other one, in the rows: lost */
    off = sdl_find_(clean, clen, "MIDMARK");
    {   u64 b = (u64)off >> 16, p2 = b % z.G, rows_end = 64 + (u64)SDX_N * SDX_DIM * 4;
        while (p2 == 0 || p2 == b || ((p2 + 1) << 16) > rows_end) { p2 += z.G; if (p2 >= z.N) break; }
        if (p2 >= z.N || ((p2 + 1) << 16) > rows_end) { print_cstr("  lens: no rows block shares a group with the text's (layout changed?) FAILED\n"); ok = 0; }
        sdx_flip_(path, (u64)off + 1);
        sdx_flip_(path, (p2 << 16) + 100); }
    l = store_load(a, path);
    if (!l) { print_cstr("  lens: a load with damage beyond repair in a text FAILED\n"); ok = 0; }
    else {
        if (store_get_text(l, 5007, (char *)0, 0) != -1 || store_get_text(l, 5003, tb, sizeof(tb)) != 33) { print_cstr("  lens: a text beyond repair was served, or a sound one was not\n"); ok = 0; }
        if (!sdl_wait_(l)) ok = 0;
        store_integrity(l, &nb, &bd, &fx, &rb, &rh, &jdm);
        if (bd < 2) { print_cstr("  lens: the walk did not count the damage\n"); ok = 0; }
        store_destroy(l);
    }

    sdx_spit_(path, clean, clen);                                /* every block checked at the load, then the disk */
    store_set_verify(2);                                         /* goes bad under it: the text's CRC catches it on */
    l = store_load(a, path);                                     /* the read, and the parity rebuilds it */
    if (l && l->map) {
        off = sdl_find_(clean, clen, "LENSTEXT-4 ");
        sdx_flip_(path, (u64)off + 3);
        if (store_get_text(l, 5004, tb, sizeof(tb)) != 33 || memcmp(tb, "LENSTEXT-4 some words about row 4", 33)) {
            print_cstr("  texts: a text damaged on the disk after the load FAILED\n"); ok = 0;
        }
        {   /* a text damaged in memory (the mapping is private: a write stays in this process) - the file is sound, so
             * there is nothing to rebuild it from */
            i32 r2 = -1, q2;
            for (q2 = 0; q2 < l->count; ++q2) if (STORE_ID(l, q2) == 5002) r2 = q2;
            if (r2 < 0 || !STORE_TXO(l, r2)) ok = 0;
            else {
                u64 v = STORE_TXO(l, r2);
                u8 *t = l->tx.b[STORE_TXOFF(v) >> STORE_TXSHIFT] + (STORE_TXOFF(v) & (STORE_TXB - 1));
                t[1] ^= 0x20;
                if (store_get_text(l, 5002, tb, sizeof(tb)) != -1) { print_cstr("  texts: a text damaged in memory was served\n"); ok = 0; }
            }
        }
    }
    if (l) store_destroy(l);
    store_set_verify(3);
    sdx_spit_(path, clean, clen);                                /* a slow lens, closed at once: no wait for it */
    store_lens_ms_ = 1000;
    l = store_load(a, path);
    if (!l) ok = 0; else store_destroy(l);
    store_lens_ms_ = keep;
    store_set_verify(2);
    print_cstr(ok ? "  lens: header, index arrays and labels checked at the load; a text checked before it is read; the walk finds and rebuilds the rest\n"
                  : "  lens FAILED\n");
    remove(path); remove(jn);
    free(clean); free(x);
    return ok;
}

int main(void)
{
    static Arena arena;
    Arena *a = &arena;
    i32 dim = 128, n = 4000, per_topic, i, ok = 1, nprobed;
    u32 st = 20260901u;
    f32 *centers, *vv;
    i64 *dids;
    RagStore *s, *s2;
    i64 res[16]; f32 sc[16];
    i32 self_hit = 0, del_leak = 0, bad_rej;

    per_topic = n / 40;
    centers = (f32 *)arena_alloc(a, (size_t)40 * dim * sizeof(f32));
    vv = (f32 *)arena_alloc(a, (size_t)n * dim * sizeof(f32));
    dids = (i64 *)arena_alloc(a, (size_t)n * sizeof(i64));
    for (i = 0; i < 40 * dim; ++i) centers[i] = sd_f_(&st) * 6.0f;
    for (i = 0; i < n; ++i) {
        i32 t = (i / per_topic) % 40, d;
        for (d = 0; d < dim; ++d)
            vv[(size_t)i * dim + d] = centers[t * dim + d] + sd_f_(&st) * 0.3f;
        dids[i] = 1000 + i;
    }

    s = store_new(a, dim);
    /* add with a metadata tag = topic bit */
    for (i = 0; i < n - 200; ++i)
        store_add_meta(s, vv + (size_t)i * dim, dids[i],
                       (i64)1 << ((i / per_topic) % 40));
    store_build(s);
    for (i = n - 200; i < n; ++i)
        store_add_meta(s, vv + (size_t)i * dim, dids[i],
                       (i64)1 << ((i / per_topic) % 40));

    /* NaN rejection */
    { f32 *bad = (f32 *)arena_alloc(a, (size_t)dim * sizeof(f32));
      union { f32 f; u32 u; } q; q.u = 0x7fc00000u;   /* qNaN */
      for (i = 0; i < dim; ++i) bad[i] = 0.1f;
      bad[7] = q.f;
      bad_rej = store_add(s, bad, 999999); }
    print_cstr("store  dim="); print_i64(dim);
    print_cstr("  live="); print_i64(store_count(s));
    print_cstr("  pending="); print_i64(store_pending(s));
    print_cstr("  NaN-add rejected="); print_i64(bad_rej == -1); print_cstr("\n");
    if (bad_rej != -1) ok = 0;

    nprobed = 0;
    for (i = 0; i < n; i += 3) {
        i32 h = store_search(s, vv + (size_t)i * dim, 1, res, sc);
        ++nprobed;
        if (h > 0 && res[0] == dids[i]) ++self_hit;
    }
    print_cstr("  self-retrieval@1 = "); print_i64((i64)(100L * self_hit / nprobed));
    print_cstr("%\n");
    if (self_hit * 100 / nprobed < 90) ok = 0;

    /* metadata filter: query a topic-0 vector but restrict to topic-3.
     * Every hit must carry topic-3's bit (the fallback scan guarantees
     * we still get k of them even though topic-0's IVFPQ neighbourhood
     * has none). */
    {
        i64 fr[10]; f32 fs[10];
        i32 h = store_search_filter(s, vv + 0 * dim, 10, fr, fs,
                                    (i64)1 << 3, 0), j, wrong = 0;
        for (j = 0; j < h; ++j) {
            i32 row;
            for (row = 0; row < s->count; ++row)
                if (STORE_ID(s, row) == fr[j] && !STORE_DEAD(s, row)) {
                    if (!(STORE_META(s, row) & ((i64)1 << 3))) ++wrong;
                    break;
                }
        }
        print_cstr("  filter(topic=3 from a topic-0 query): "); print_i64(h);
        print_cstr(" hits, wrong-tag="); print_i64(wrong); print_cstr("\n");
        if (wrong != 0 || h < 10) ok = 0;
    }

    for (i = 0; i < 300; ++i) store_delete(s, dids[i * 5]);
    for (i = 0; i < 300; ++i) {
        i32 h = store_search(s, vv + (size_t)(i * 5) * dim, 5, res, sc), j;
        for (j = 0; j < h; ++j) if (res[j] == dids[i * 5]) ++del_leak;
    }
    print_cstr("  after 300 deletes: live="); print_i64(store_count(s));
    print_cstr("  leaked="); print_i64(del_leak); print_cstr("\n");
    if (del_leak != 0) ok = 0;
    store_build(s);

    /* save / load - index is serialised, load must NOT rebuild */
    {
        f64 t0;
        RagStore *ld;
        i32 same = 0, tried = 0;
        if (store_save(s, "/tmp/store_demo.vxi") != 0) { print_cstr("  save FAIL\n"); ok = 0; }
        t0 = 0.0; (void)t0;
        s2 = store_load(a, "/tmp/store_demo.vxi");
        ld = s2;
        if (!ld) { print_cstr("  load FAIL\n"); ok = 0; }
        else {
            for (i = 0; i < s->count; i += 7) {
                i64 r1[3], r2[3]; f32 c1[3], c2[3];
                i32 h1 = store_search(s,  STORE_VEC(s, i), 3, r1, c1);
                i32 h2 = store_search(ld, STORE_VEC(s, i), 3, r2, c2);
                ++tried;
                if (h1 == h2 && h1 > 0 && r1[0] == r2[0]) ++same;
            }
            print_cstr("  save/load (index serialised): top-1 identical on ");
            print_i64((i64)(100L * same / tried)); print_cstr("%\n");
            if (same * 100 / tried < 98) ok = 0;
            store_destroy(ld);
        }
    }

    store_destroy(s);

    /* fast build (no auto-tuner): must retrieve and round-trip through .vxi too */
    {
        RagStore *f = store_new(a, dim), *fl;
        i32 hit = 0, tried = 0, same = 0;
        n = 6000;                                       /* above the flat-q5 shortcut */
        vv = (f32 *)arena_alloc(a, (size_t)n * dim * sizeof(f32));
        for (i = 0; i < n; ++i) {
            i32 t = (i / 150) % 40, d;
            for (d = 0; d < dim; ++d) vv[(size_t)i * dim + d] = centers[t * dim + d] + sd_f_(&st) * 0.3f;
        }
        store_set_fast(f, 0.9f);
        for (i = 0; i < n; ++i) store_add(f, vv + (size_t)i * dim, 5000 + i);
        store_build(f);
        for (i = 0; i < n; i += 5) {
            i32 h = store_search(f, vv + (size_t)i * dim, 1, res, sc);
            ++tried; if (h > 0 && res[0] == 5000 + i) ++hit;
        }
        print_cstr("  fast build: self-retrieval@1 = "); print_i64((i64)(100L * hit / tried)); print_cstr("%\n");
        if (hit * 100 / tried < 90) ok = 0;
        if (store_save(f, "/tmp/store_demo_fast.vxi") != 0) ok = 0;
        fl = store_load(a, "/tmp/store_demo_fast.vxi");
        if (!fl) ok = 0;
        else {
            tried = 0;
            for (i = 0; i < n; i += 11) {
                i64 r1[1], r2[1]; f32 c1[1], c2[1];
                i32 h1 = store_search(f, vv + (size_t)i * dim, 1, r1, c1), h2 = store_search(fl, vv + (size_t)i * dim, 1, r2, c2);
                ++tried; if (h1 == h2 && h1 > 0 && r1[0] == r2[0]) ++same;
            }
            print_cstr("  fast build: save/load top-1 identical on "); print_i64((i64)(100L * same / tried)); print_cstr("%\n");
            if (same * 100 / tried < 98) ok = 0;
            store_destroy(fl);
        }
        store_destroy(f);
    }

    /* IVF grouping of the TurboQuant scan (normally only for huge stores; exercised here on 6000 rows directly) */
    {
        Tq *flat, *ivf;
        i32 nq = 60, q, j, hit = 0, tot = 0, filt_ok = 1, all_ok = 1;
        u64 *dead = (u64 *)calloc((size_t)(n + 63) / 64, sizeof(u64));
        i64 *meta = (i64 *)calloc((size_t)n, sizeof(i64));
        u8 *hv, *hm;
        TqCol vcol, mcol;
        for (i = 0; i < n; ++i) { meta[i] = (i % 3 == 0) ? 1 : 2; if (i % 11 == 0) dead[i >> 6] |= (u64)1 << (i & 63); }
        mcol = tq_col_flat(meta, sizeof(i64), &hm);
        {   f32 *un = (f32 *)malloc((size_t)n * dim * sizeof(f32));                /* unit rows, as the store keeps them */
            for (i = 0; i < n; ++i) { f32 nr = 0; for (j = 0; j < dim; ++j) nr += vv[(size_t)i * dim + j] * vv[(size_t)i * dim + j]; nr = 1.0f / (f32)sqrt((f64)nr); for (j = 0; j < dim; ++j) un[(size_t)i * dim + j] = vv[(size_t)i * dim + j] * nr; }
            vcol = tq_col_flat(un, (size_t)dim * sizeof(f32), &hv);
            flat = tq_build(&vcol, n, dim, 7u, 0, 2, 4); ivf = tq_build(&vcol, n, dim, 7u, 60, 2, 4);
            for (q = 0; q < nq; ++q) {
                i32 rf[256], ri[256], rn[256], nf, ni, nn2;
                const f32 *qv = un + (size_t)((q * 97) % n) * dim;
                nf = tq_search(flat, qv, 256, rf, dead, (const TqFilt *)0, 1, 0);
                ni = tq_search(ivf, qv, 256, ri, dead, (const TqFilt *)0, 1, 0);           /* the default probe */
                nn2 = tq_search(ivf, qv, 256, rn, dead, (const TqFilt *)0, 1, 60);         /* every cluster = the same rows as flat */
                for (j = 0; j < 10 && j < nf; ++j) { i32 k, f = 0; for (k = 0; k < ni; ++k) if (ri[k] == rf[j]) { f = 1; break; } hit += f; ++tot; }
                { i32 k, same = nf == nn2; for (k = 0; k < nf && k < 200 && same; ++k) { i32 z, f = 0; for (z = 0; z < nn2; ++z) if (rn[z] == rf[k]) { f = 1; break; } if (!f) same = 0; } if (!same) all_ok = 0; }
                for (j = 0; j < ni; ++j) if ((dead[ri[j] >> 6] >> (ri[j] & 63)) & 1u) all_ok = 0;   /* tombstones never come back */
                { TqFilt tf; memset(&tf, 0, sizeof(tf)); tf.meta = &mcol; tf.all = 1; ni = tq_search(ivf, qv, 64, ri, dead, &tf, 1, 60); }   /* meta filter: only rows with bit 1 */
                for (j = 0; j < ni; ++j) if (!(meta[ri[j]] & 1)) filt_ok = 0;
            }
            print_cstr("  IVF scan: top-10 of the flat scan found "); print_i64((i64)(100L * hit / tot)); print_cstr("%; all clusters == flat: ");
            print_cstr(all_ok ? "yes" : "NO"); print_cstr("; meta filter respected: "); print_cstr(filt_ok ? "yes" : "NO"); print_cstr("\n");
            if (100L * hit / tot < 88 || !all_ok || !filt_ok) ok = 0;
            tq_free(flat); tq_free(ivf); free(un);
        }
        free(dead); free(meta);
    }

    /* columns spilled to a file from the first chunk: same answers as in memory, ids looked up out of order */
    {
        RagStore *m = store_new(a, dim), *d = store_new(a, dim);
        i32 same = 0, tried = 0, found = 0;
        i64 ram = 0, file = 0;
        store_set_spill(m, (const char *)0, 0);
        store_set_spill(d, "/tmp", 0);
        store_set_fast(m, 0.9f); store_set_fast(d, 0.9f);
        for (i = 0; i < n; ++i) {
            i64 id = (i64)((i * 7919) % n) * 3 + 1;              /* ids out of order: the hash path */
            store_add(m, vv + (size_t)i * dim, id); store_add(d, vv + (size_t)i * dim, id);
        }
        store_build(m); store_build(d);
        for (i = 0; i < n; i += 13) {
            i64 r1[5], r2[5]; f32 c1[5], c2[5];
            i32 h1 = store_search(m, vv + (size_t)i * dim, 5, r1, c1), h2 = store_search(d, vv + (size_t)i * dim, 5, r2, c2), x, eq = h1 == h2;
            for (x = 0; x < h1 && eq; ++x) if (r1[x] != r2[x]) eq = 0;
            ++tried; same += eq;
            if (store_has(d, (i64)((i * 7919) % n) * 3 + 1)) ++found;
        }
        store_mem(d, &ram, &file);
        print_cstr("  spilled columns: results identical on "); print_i64((i64)(100L * same / tried));
        print_cstr("%, ids found "); print_i64(found); print_cstr("/"); print_i64(tried);
        print_cstr(", file-backed bytes "); print_i64(file); print_cstr("\n");
        if (same != tried || found != tried || file < (i64)n * dim * 4) ok = 0;
        store_destroy(m); store_destroy(d);
    }

    /* the compact modes: int8 rows + 2-bit codes. Same neighbours as the f32 / 4-bit store, a v6 file that loads back
     * to identical answers, and the 2-bit IVF scan of every cluster equal to the 2-bit flat scan */
    {
        RagStore *ref = store_new(a, dim), *c8 = store_new(a, dim), *cl;
        i32 tried = 0, agree = 0, same = 0, self = 0, x;
        f32 back[128], cosm = 1.0f;
        store_set_fast(ref, 0.9f); store_set_fast(c8, 0.9f);
        if (store_set_vec_bytes(c8, 1) != 0) ok = 0;
        store_set_codes(c8, 2);
        store_set_rerank(c8, 1000);                                  /* 2-bit ranks coarsely: re-score more */
        for (i = 0; i < n; ++i) { store_add(ref, vv + (size_t)i * dim, 7000 + i); store_add(c8, vv + (size_t)i * dim, 7000 + i); }
        if (store_set_vec_bytes(c8, 4) != -1) ok = 0;               /* not once it has rows */
        store_build(ref); store_build(c8);
        for (i = 0; i < n; i += 7) {   /* a hit counts when its exact score is within the int8 error of the exact 10th best: rows of a topic are near
                                       * ties (cosine ~0.998), so the exact ids are a coin toss; real-data recall is in bench/ */
            i64 r1[10], r2[10]; f32 c1[10], c2[10];
            i32 h1 = store_search(ref, vv + (size_t)i * dim, 10, r1, c1), h2 = store_search(c8, vv + (size_t)i * dim, 10, r2, c2);
            for (x = 0; x < h2 && h1 == 10; ++x) {
                f32 ex[128];
                if (store_get_vec(ref, r2[x], ex) == 0) {
                    f32 d2 = 0.0f, qq = 0.0f; i32 z;
                    for (z = 0; z < dim; ++z) { d2 += ex[z] * vv[(size_t)i * dim + z]; qq += vv[(size_t)i * dim + z] * vv[(size_t)i * dim + z]; }
                    if (2.0f * d2 / (f32)sqrt((f64)qq) - 2.0f >= c1[9] - 0.01f) ++agree;   /* within the int8 error */
                }
            }
            tried += h1;
            if (h2 > 0 && r2[0] == 7000 + i) ++self;
        }
        for (i = 0; i < n; i += 97) {
            f32 dot = 0.0f, nn = 0.0f;
            if (store_get_vec(c8, 7000 + i, back) != 0) { ok = 0; continue; }
            for (x = 0; x < dim; ++x) { dot += back[x] * vv[(size_t)i * dim + x]; nn += vv[(size_t)i * dim + x] * vv[(size_t)i * dim + x]; }
            dot /= (f32)sqrt((f64)nn);
            if (dot < cosm) cosm = dot;
        }
        if (store_vec_ptr(c8, 7000) != (const f32 *)0) ok = 0;      /* no f32 copy to lend */
        if (store_save(c8, "/tmp/store_demo_i8.vxi") != 0) ok = 0;
        cl = store_load(a, "/tmp/store_demo_i8.vxi");
        if (!cl || cl->vbytes != 1 || !cl->tq || cl->tq->bits != 2) ok = 0;
        else for (i = 0; i < n; i += 11) {
            i64 r1[5], r2[5]; f32 c1[5], c2[5];
            i32 h1 = store_search(c8, vv + (size_t)i * dim, 5, r1, c1), h2 = store_search(cl, vv + (size_t)i * dim, 5, r2, c2), eq = h1 == h2;
            for (x = 0; x < h1 && eq; ++x) if (r1[x] != r2[x]) eq = 0;
            same += eq;
        }
        print_cstr("  int8 rows + 2-bit codes: top-10 as close as the exact top-10 on "); print_i64((i64)(100L * agree / (tried ? tried : 1)));
        print_cstr("%, self@1 "); print_i64((i64)(100L * self / ((n + 6) / 7))); print_cstr("%, min cosine of a dequantised row ");
        print_i64((i64)(cosm * 10000.0f)); print_cstr("e-4, v6 save/load identical on "); print_i64((i64)(100L * same / ((n + 10) / 11)));
        print_cstr("%\n");
        if (agree * 100 < tried * 95 || self * 100 < ((n + 6) / 7) * 95 || cosm < 0.998f || same != (n + 10) / 11) ok = 0;
        if (cl) store_destroy(cl);
        store_destroy(ref); store_destroy(c8);
        remove("/tmp/store_demo_i8.vxi"); remove("/tmp/store_demo_i8.vxi.j");
    }
    /* 1-bit codes: the signs of the rotated coordinates, dim / 8 bytes a row, through the same 16-entry tables and
     * kernel; with f32 rows the default re-scores four times as many candidates */
    {
        RagStore *ref = store_new(a, dim), *b1 = store_new(a, dim), *bl;
        i32 tried = 0, agree = 0, self = 0, same = 0, x, vd1 = ((dim + 3) / 4 + 1) & ~1;
        store_set_fast(ref, 0.9f); store_set_fast(b1, 0.9f);
        store_set_codes(b1, 1);
        for (i = 0; i < n; ++i) { store_add(ref, vv + (size_t)i * dim, 9000 + i); store_add(b1, vv + (size_t)i * dim, 9000 + i); }
        store_build(ref); store_build(b1);
        if (!b1->tq || b1->tq->bits != 1 || b1->tq->vd != vd1) ok = 0;
        for (i = 0; i < n; i += 7) {          /* a hit counts when its exact score reaches the exact 10th best (rows of a topic tie) */
            i64 r1[10], r2[10]; f32 c1[10], c2[10];
            i32 h1 = store_search(ref, vv + (size_t)i * dim, 10, r1, c1), h2 = store_search(b1, vv + (size_t)i * dim, 10, r2, c2);
            for (x = 0; x < h2 && h1 == 10; ++x) if (c2[x] >= c1[9] - 1e-4f) ++agree;
            tried += h1;
            if (h2 > 0 && r2[0] == 9000 + i) ++self;
        }
        if (store_save(b1, "/tmp/store_demo_b1.vxi") != 0 || !(bl = store_load(a, "/tmp/store_demo_b1.vxi")) || !bl->tq || bl->tq->bits != 1) { ok = 0; bl = (RagStore *)0; }
        for (i = 0; bl && i < n; i += 11) {
            i64 r1[5], r2[5]; f32 c1[5], c2[5];
            i32 h1 = store_search(b1, vv + (size_t)i * dim, 5, r1, c1), h2 = store_search(bl, vv + (size_t)i * dim, 5, r2, c2), eq = h1 == h2;
            for (x = 0; x < h1 && eq; ++x) if (r1[x] != r2[x]) eq = 0;
            same += eq;
        }
        print_cstr("  1-bit codes ("); print_i64(vd1 / 2); print_cstr(" bytes a row for "); print_i64(dim); print_cstr(" dimensions) + f32 rows: top-10 as close as the exact top-10 on ");
        print_i64((i64)(100L * agree / (tried ? tried : 1))); print_cstr("%, self@1 "); print_i64((i64)(100L * self / ((n + 6) / 7)));
        print_cstr("%, save/load identical on "); print_i64((i64)(100L * same / ((n + 10) / 11))); print_cstr("%\n");
        if (agree * 100 < tried * 90 || self * 100 < ((n + 6) / 7) * 95 || same != (n + 10) / 11) ok = 0;
        if (bl) store_destroy(bl);
        store_destroy(ref); store_destroy(b1);
        remove("/tmp/store_demo_b1.vxi"); remove("/tmp/store_demo_b1.vxi.j");
    }
    {   /* 2-bit IVF over every cluster == 2-bit flat */
        f32 *un = (f32 *)malloc((size_t)n * dim * sizeof(f32));
        u8 *hv;
        TqCol vcol;
        Tq *flat, *ivf;
        i32 q, all_ok = 1, j;
        for (i = 0; i < n; ++i) { f32 nr = 0; for (j = 0; j < dim; ++j) nr += vv[(size_t)i * dim + j] * vv[(size_t)i * dim + j]; nr = 1.0f / (f32)sqrt((f64)nr); for (j = 0; j < dim; ++j) un[(size_t)i * dim + j] = vv[(size_t)i * dim + j] * nr; }
        vcol = tq_col_flat(un, (size_t)dim * sizeof(f32), &hv);
        flat = tq_build(&vcol, n, dim, 7u, 0, 2, 2); ivf = tq_build(&vcol, n, dim, 7u, 60, 2, 2);
        for (q = 0; q < 40; ++q) {
            i32 rf[200], rn[200], nf, nn2, k;
            const f32 *qv = un + (size_t)((q * 131) % n) * dim;
            nf = tq_search(flat, qv, 200, rf, (const u64 *)0, (const TqFilt *)0, 1, 0);
            nn2 = tq_search(ivf, qv, 200, rn, (const u64 *)0, (const TqFilt *)0, 1, 60);
            if (nf != nn2) all_ok = 0;
            for (k = 0; k < nf && k < 150 && all_ok; ++k) { i32 z, f = 0; for (z = 0; z < nn2; ++z) if (rn[z] == rf[k]) { f = 1; break; } if (!f) all_ok = 0; }
        }
        {   /* tq_search_many (cluster-major batch) == tq_search, IVF and flat */
            i32 nq2 = 64, R2 = 100, *mr = (i32 *)malloc((size_t)nq2 * R2 * sizeof(i32)), mc[64], z, same2 = 0, pass;
            for (pass = 0; pass < 2; ++pass) {
                Tq *tt = pass ? flat : ivf;
                tq_search_many(tt, un, nq2, R2, mr, mc, (const u64 *)0, 2, 12);
                for (q = 0; q < nq2; ++q) {
                    i32 one[100], no = tq_search(tt, un + (size_t)q * dim, R2, one, (const u64 *)0, (const TqFilt *)0, 1, 12), eq = no == mc[q];
                    for (z = 0; z < no && eq; ++z) if (one[z] != mr[(size_t)q * R2 + z]) eq = 0;
                    same2 += eq;
                }
            }
            print_cstr("  tq_search_many == tq_search on "); print_i64(same2); print_cstr("/128 (IVF and flat, 2-bit)\n");
            if (same2 != 2 * nq2) all_ok = 0;
            free(mr);
        }
        print_cstr("  2-bit IVF, every cluster == 2-bit flat: "); print_cstr(all_ok ? "yes" : "NO"); print_cstr("\n");
        if (!all_ok || flat->vd != dim / 2 || ivf->bits != 2) ok = 0;
        tq_free(flat); tq_free(ivf); free(un);
    }

    {   /* two-level clustering (more than TQ_FLAT_MAX clusters): rows land in their truly closest cluster, every
         * cluster == flat, batch == one at a time */
        i32 n2 = 24000, d2 = 32, nc2 = 1500, j, q, inbest = 0, same = 0, all_ok = 1, *mr, mc[32];
        f32 *x2 = (f32 *)malloc((size_t)n2 * d2 * sizeof(f32)), *sc2;
        u8 *hv;
        u32 s2 = 77u;
        TqCol vcol;
        Tq *flat, *ivf;
        for (i = 0; i < n2; ++i) {
            f32 nr = 0;
            for (j = 0; j < d2; ++j) { f32 v = sd_f_(&s2); x2[(size_t)i * d2 + j] = v; nr += v * v; }
            nr = 1.0f / (f32)sqrt((f64)nr);
            for (j = 0; j < d2; ++j) x2[(size_t)i * d2 + j] *= nr;
        }
        vcol = tq_col_flat(x2, (size_t)d2 * sizeof(f32), &hv);
        flat = tq_build(&vcol, n2, d2, 7u, 0, 2, 4); ivf = tq_build(&vcol, n2, d2, 7u, nc2, 2, 4);
        sc2 = (f32 *)malloc((size_t)(ivf->nc + 15) / 16 * 16 * sizeof(f32));
        for (j = 0; j < ivf->nc; ++j) {                         /* the cluster each row sits in vs its closest centroid */
            i32 sl;
            for (sl = ivf->cstart[j] * 32; sl < ivf->cstart[j + 1] * 32; ++sl) {
                i32 row = ivf->slot_row[sl];
                if (row < 0) continue;
                tq_score_(x2 + (size_t)row * d2, ivf->ct, (ivf->nc + 15) / 16, d2, sc2);
                inbest += tq_argmax_(sc2, ivf->nc) == j;
            }
        }
        for (q = 0; q < 30; ++q) {
            i32 rf[100], rn[100], nf, nn2, k;
            const f32 *qv = x2 + (size_t)(q * 797) * d2;
            nf = tq_search(flat, qv, 100, rf, (const u64 *)0, (const TqFilt *)0, 1, 0);
            nn2 = tq_search(ivf, qv, 100, rn, (const u64 *)0, (const TqFilt *)0, 1, ivf->nc);
            if (nf != nn2) all_ok = 0;
            for (k = 0; k < nf && all_ok; ++k) if (rf[k] != rn[k]) all_ok = 0;
        }
        mr = (i32 *)malloc((size_t)32 * 100 * sizeof(i32));
        tq_search_many(ivf, x2, 32, 100, mr, mc, (const u64 *)0, 2, 40);
        for (q = 0; q < 32; ++q) {
            i32 one[100], no = tq_search(ivf, x2 + (size_t)q * d2, 100, one, (const u64 *)0, (const TqFilt *)0, 1, 40), eq = no == mc[q], z;
            for (z = 0; z < no && eq; ++z) if (one[z] != mr[(size_t)q * 100 + z]) eq = 0;
            same += eq;
        }
        print_cstr("  two-level IVF ("); print_i64(ivf->nc); print_cstr(" clusters): rows in their closest cluster ");
        print_i64((i64)(1000L * inbest / n2)); print_cstr("/1000, every cluster == flat: "); print_cstr(all_ok ? "yes" : "NO");
        print_cstr(", batch == one at a time "); print_i64(same); print_cstr("/32\n");
        if (ivf->nc <= TQ_FLAT_MAX || 1000L * inbest / n2 < 900 || !all_ok || same != 32) ok = 0;
        tq_free(flat); tq_free(ivf); free(x2); free(sc2); free(mr);
    }

    {   /* self-tuning: ask for a recall, check it on queries the tuning never saw */
        i32 n3 = 30000, d3 = 32, j, q, nq3 = 200, hits = 0, p95, p80, *tr3;
        f32 *x3 = (f32 *)malloc((size_t)n3 * d3 * sizeof(f32)), *q3 = (f32 *)malloc((size_t)nq3 * d3 * sizeof(f32)), r95, r80;
        u32 s3 = 991u;
        RagStore *t3 = store_new(a, d3);
        for (i = 0; i < n3; ++i) {                              /* 300 topics, rows scattered around them */
            u32 tp = sd_rng_(&s3) % 300u, s4 = 5000u + tp;
            for (j = 0; j < d3; ++j) x3[(size_t)i * d3 + j] = sd_f_(&s4) * 1.0f + sd_f_(&s3) * 1.5f;
            store_add(t3, x3 + (size_t)i * d3, 70000 + i);
        }
        for (q = 0; q < nq3; ++q) {                             /* held-out queries: near a topic, not a row */
            u32 tp = sd_rng_(&s3) % 300u, s4 = 5000u + tp;
            for (j = 0; j < d3; ++j) q3[(size_t)q * d3 + j] = sd_f_(&s4) * 1.0f + sd_f_(&s3) * 1.5f;
        }
        store_ivf_min_ = 2000;
        store_set_target_recall(t3, 0.95f);
        store_build(t3);
        p95 = t3->tq ? t3->tq->nprobe : -1; r95 = store_tuned_recall(t3);
        tr3 = (i32 *)malloc((size_t)nq3 * 10 * sizeof(i32));
        for (q = 0; q < nq3; ++q) store_normalise_(q3 + (size_t)q * d3, d3);
        { TqCol sc3, c3 = store_vcol_(t3, &sc3); store_truth_(&c3, t3->dead, t3->built_upto, d3, q3, (const i32 *)0, nq3, 10, tr3, 2); }
        for (q = 0; q < nq3; ++q) {
            i64 r10[10]; f32 c10[10];
            i32 got = store_search(t3, q3 + (size_t)q * d3, 10, r10, c10), x, z;
            for (x = 0; x < got; ++x) for (z = 0; z < 10; ++z) if (tr3[(size_t)q * 10 + z] + 70000 == r10[x]) { ++hits; break; }
        }
        store_set_target_recall(t3, 0.80f);
        r80 = store_tune(t3);                                   /* now on the 200 queries it has just answered */
        p80 = t3->tq ? t3->tq->nprobe : -1;
        print_cstr("  self-tuning ("); print_i64(t3->tq ? t3->tq->nc : 0); print_cstr(" clusters): target 0.95 -> ");
        print_i64(p95); print_cstr(" probes (measured "); print_i64((i64)(1000.0f * r95)); print_cstr("/1000), held-out recall@10 ");
        print_i64((i64)(1000L * hits / (nq3 * 10))); print_cstr("/1000; target 0.80 on its own traffic -> "); print_i64(p80);
        print_cstr(" probes ("); print_i64((i64)(1000.0f * r80)); print_cstr("/1000)\n");
        if (!t3->tq || t3->tq->nc <= 1 || r95 < 0.95f || 1000L * hits / (nq3 * 10) < 920 || p80 > p95 || r80 < 0.80f || p95 >= t3->tq->nc) ok = 0;
        store_ivf_min_ = STORE_IVF_MIN;
        store_destroy(t3); free(x3); free(q3); free(tr3);
    }

    /* batch search: the same answers as one query at a time (flat scan here; IVF + batch in the 2-bit test above) */
    {
        RagStore *b = store_new(a, dim);
        i32 nq = 300, same = 0, x;
        i64 *bi = (i64 *)malloc((size_t)nq * 10 * sizeof(i64)), r1[10];
        f32 *bs = (f32 *)malloc((size_t)nq * 10 * sizeof(f32)), c1[10];
        store_set_fast(b, 0.9f);
        for (i = 0; i < n; ++i) store_add(b, vv + (size_t)i * dim, 300000 + i);
        store_build(b);
        for (i = 0; i < 40; ++i) store_add(b, vv + (size_t)(i * 13) * dim, 900000 + i);   /* a pending tail too */
        store_delete(b, 300000 + 5);
        store_search_batch(b, vv, nq, 10, bi, bs, 2);
        for (i = 0; i < nq; ++i) {
            i32 h = store_search(b, vv + (size_t)i * dim, 10, r1, c1), eq = 1;
            for (x = 0; x < 10; ++x) if ((x < h ? r1[x] : -1) != bi[(size_t)i * 10 + x]) eq = 0;
            same += eq;
        }
        print_cstr("  batch search == one at a time on "); print_i64(same); print_cstr("/"); print_i64(nq); print_cstr(" queries\n");
        if (same != nq) ok = 0;
        free(bi); free(bs);
        store_destroy(b);
    }

    {   /* Live ingest. (1) rows added to a built index are rows like the built ones: probing every cluster gives the
         * flat scan of all the rows, the batch path agrees, a flat index takes added rows the same way, and an added row
         * lands in its closest cluster. (2) a live store: rows streamed in by another thread while this one searches -
         * indexed by the store's own thread, rebuilt as it grows, tuned to the recall asked for; deletes + save keep
         * the index (nothing rebuilt at load); rows from new topics are noticed as a drift and trigger a rebuild. */
        i32 n4 = 24000, n40 = 16000, d4 = 32, nd = 3000, nq4 = 100, j, q, bits, same = 0, tot = 0, inbest = 0, sameb = 0;
        i32 miss = 0, looks = 0, sameld = 0, mc[32], did = 0, reasons = 0, nsnap = 0, snap_at = 0;
        i32 *mr = (i32 *)malloc((size_t)32 * 100 * sizeof(i32));
        f32 *x4 = (f32 *)malloc((size_t)(n4 + nd) * d4 * sizeof(f32)), *q4 = (f32 *)malloc((size_t)(2 * nq4) * d4 * sizeof(f32));
        f32 *qd = q4 + (size_t)nq4 * d4, sc4[64], r_live, r_saved, r_d0, r_d1, rt;
        u8 *hv, *gone = (u8 *)calloc((size_t)(n4 + nd), 1);
        u32 s5 = 4242u;
        TqCol vcol;
        RagStore *lv, *ld;
        SdAdder ad4;
        pthread_t th;
        i64 nidx = 0, nreb = 0, nreb0 = 0;
        f64 rbs = 0.0;
        for (i = 0; i < n4 + nd + 2 * nq4; ++i) {               /* 200 topics; the drift rows and queries: 30 new ones */
            i32 newt = (i >= n4 && i < n4 + nd) || i >= n4 + nd + nq4;
            u32 tp = newt ? 1000u + sd_rng_(&s5) % 30u : sd_rng_(&s5) % 200u, s6 = 300u + tp;
            f32 *v = i < n4 + nd ? x4 + (size_t)i * d4 : q4 + (size_t)(i - n4 - nd) * d4, nr = 0.0f;
            for (j = 0; j < d4; ++j) { v[j] = sd_f_(&s6) + sd_f_(&s5) * 0.8f; nr += v[j] * v[j]; }
            nr = 1.0f / (f32)sqrt((f64)nr);
            for (j = 0; j < d4; ++j) v[j] *= nr;
        }
        vcol = tq_col_flat(x4, (size_t)d4 * sizeof(f32), &hv);
        for (bits = 4; bits >= 2; bits -= 2) {
            Tq *all = tq_build(&vcol, n4, d4, 7u, 0, 2, bits), *ivf = tq_build(&vcol, n40, d4, 7u, 60, 2, bits);
            Tq *fl = tq_build(&vcol, n40 + 7, d4, 7u, 0, 2, bits);
            TqAdd ta;
            if (tq_add_prep(ivf, &vcol, n40, n4 - n40, 2, &ta) != 0 || tq_add_put(ivf, &ta, n40) != 0) ok = 0;
            if (bits == 4) for (j = 0; j < ta.m; ++j) {          /* the closest centroid, by exact f32 scores */
                tq_score_(x4 + (size_t)(n40 + j) * d4, ivf->ct, (ivf->nc + 15) / 16, d4, sc4);
                inbest += tq_argmax_(sc4, ivf->nc) == ta.cl[j];
            }
            tq_add_free(&ta);
            if (tq_add_prep(fl, &vcol, n40 + 7, n4 - n40 - 7, 2, &ta) != 0 || tq_add_put(fl, &ta, n40 + 7) != 0) ok = 0;
            tq_add_free(&ta);
            for (q = 0; q < 30; ++q) {
                i32 ra[100], ri[100], rf[100], na, ni, nf, k, eq;
                const f32 *qv = x4 + (size_t)(q * 797) * d4;
                na = tq_search(all, qv, 100, ra, (const u64 *)0, (const TqFilt *)0, 1, 0);
                ni = tq_search(ivf, qv, 100, ri, (const u64 *)0, (const TqFilt *)0, 1, ivf->nc);
                nf = tq_search(fl, qv, 100, rf, (const u64 *)0, (const TqFilt *)0, 2, 0);
                eq = na == ni && na == nf;
                for (k = 0; k < na && eq; ++k) if (ra[k] != ri[k] || ra[k] != rf[k]) eq = 0;
                same += eq; ++tot;
            }
            tq_search_many(ivf, x4 + (size_t)n40 * d4, 32, 100, mr, mc, (const u64 *)0, 2, 12);   /* queries: added rows */
            for (q = 0; q < 32; ++q) {
                i32 one[100], no = tq_search(ivf, x4 + (size_t)(n40 + q) * d4, 100, one, (const u64 *)0, (const TqFilt *)0, 1, 12), eq = no == mc[q], z;
                for (z = 0; z < no && eq; ++z) if (one[z] != mr[(size_t)q * 100 + z]) eq = 0;
                sameb += eq;
            }
            if (ivf->xn == 0 || fl->xn == 0 || ivf->n != n4 || fl->n != n4) ok = 0;
            tq_free(all); tq_free(ivf); tq_free(fl);
        }
        print_cstr("  live: added rows == built rows (all clusters of built + added == flat of all, flat + added == flat of all) ");
        print_i64(same); print_cstr("/"); print_i64(tot); print_cstr(", batch == one at a time "); print_i64(sameb);
        print_cstr("/64, added rows in their closest cluster "); print_i64((i64)(1000L * inbest / (n4 - n40))); print_cstr("/1000\n");
        if (same != tot || sameb != 64 || 1000L * inbest / (n4 - n40) < 990) ok = 0;

        store_ivf_min_ = 2000;                                   /* IVF from 2000 rows, so a small store has clusters */
        lv = store_new(a, d4);
        store_set_target_recall(lv, 0.95f);
        store_set_live(lv, 1, 1.0f);
        ad4.s = lv; ad4.x = x4; ad4.dim = d4; ad4.from = 0; ad4.to = n4; ad4.id0 = 600000; ad4.done = 0;
        pthread_create(&th, (const pthread_attr_t *)0, sd_adder_, &ad4);
        while ((j = __atomic_load_n(&ad4.done, __ATOMIC_ACQUIRE)) < n4) {   /* every row added so far is found */
            i64 r1[1]; f32 c1[1];
            i32 row;
            if (j == 0) continue;
            row = (i32)(sd_rng_(&s5) % (u32)j);
            if (store_search(lv, x4 + (size_t)row * d4, 1, r1, c1) < 1 || r1[0] != 600000 + row) ++miss;
            ++looks;
            if (j >= snap_at) { if (store_snapshot(lv, "/tmp/store_demo_live_snap.vxi") == 0) ++nsnap; snap_at += n4 / 6; }   /* forks beside the
                                                                  * live thread and the adder */
        }
        pthread_join(th, (void **)0);
        while (looks < 100) {                    /* a slow machine (an emulator) can add every row before 100 lookups */
            i64 r1[1]; f32 c1[1];
            i32 row = (i32)(sd_rng_(&s5) % (u32)n4);
            if (store_search(lv, x4 + (size_t)row * d4, 1, r1, c1) < 1 || r1[0] != 600000 + row) ++miss;
            ++looks;
        }
        if (store_snapshot_poll(lv, 1) != 0) ok = 0;
        {   RagStore *ls = store_load(a, "/tmp/store_demo_live_snap.vxi");
            i32 have = ls ? store_count(ls) : 0, fnd = 0, tr = 0;
            for (j = 0; ls && j < have; j += 97) {
                i64 r1[1]; f32 c1[1];
                ++tr;
                if (store_search(ls, x4 + (size_t)j * d4, 1, r1, c1) == 1 && r1[0] == 600000 + j) ++fnd;
            }
            print_cstr("  live store: "); print_i64(nsnap); print_cstr(" snapshots taken while rows streamed in, the last one loads with ");
            print_i64(have); print_cstr(" rows, found "); print_i64(fnd); print_cstr("/"); print_i64(tr); print_cstr("\n");
            if (!ls || nsnap < 1 || have < 1 || have > n4 || fnd * 100 < tr * 98) ok = 0;
            if (ls) store_destroy(ls);
            remove("/tmp/store_demo_live_snap.vxi"); remove("/tmp/store_demo_live_snap.vxi.j"); remove("/tmp/store_demo_live_snap.vxi.jn");
        }
        for (j = 0; j < 20000 && store_pending(lv) > 0; ++j) { struct timespec ts; ts.tv_sec = 0; ts.tv_nsec = 1000000L; nanosleep(&ts, (struct timespec *)0); }
        store_set_live(lv, 0, 1.0f);                             /* stop the thread; settle by hand */
        for (j = 0; j < 20 && store_maintain(lv); ++j) {}
        store_live_stats(lv, &nidx, &nreb, &rbs);
        rt = store_tuned_recall(lv);
        r_live = sd_recall_(lv, x4, n4, d4, q4, nq4, 600000, (const u8 *)0);
        print_cstr("  live store: "); print_i64(n4); print_cstr(" rows streamed in while searching - added rows found "); print_i64(looks - miss);
        print_cstr("/"); print_i64(looks); print_cstr(", "); print_i64(nidx); print_cstr(" indexed as they came, "); print_i64(nreb);
        print_cstr(" rebuilds, "); print_i64(lv->tq ? lv->tq->nc : 0); print_cstr(" clusters, tuned to "); print_i64((i64)(1000.0f * rt));
        print_cstr("/1000, held-out recall@10 "); print_i64((i64)(1000.0f * r_live)); print_cstr("/1000\n");
        if (miss || looks < 100 || store_pending(lv) != 0 || !lv->tq || lv->tq->nc <= 1 || nreb < 1 || rt < 0.95f || r_live < 0.92f) ok = 0;

        for (i = 0; i < n4; i += 12) { store_delete(lv, 600000 + i); gone[i] = 1; }
        r_live = sd_recall_(lv, x4, n4, d4, q4, nq4, 600000, gone);
        if (store_save(lv, "/tmp/store_demo_live.vxi") != 0) ok = 0;
        r_saved = sd_recall_(lv, x4, n4, d4, q4, nq4, 600000, gone);   /* compacted: the index followed its rows */
        ld = store_load(a, "/tmp/store_demo_live.vxi");
        if (!ld || !ld->tq || ld->tq->xn != 0 || ld->tq->n != store_count(ld) || store_pending(ld) != 0 || ld->tq->nprobe != lv->tq->nprobe) ok = 0;
        else for (q = 0; q < nq4; ++q) {
            i64 r1[10], r2[10]; f32 c1[10], c2[10];
            i32 h1 = store_search(lv, q4 + (size_t)q * d4, 10, r1, c1), h2 = store_search(ld, q4 + (size_t)q * d4, 10, r2, c2), eq = h1 == h2, z;
            for (z = 0; z < h1 && eq; ++z) if (r1[z] != r2[z]) eq = 0;
            for (z = 0; z < h2; ++z) if (r2[z] >= 600000 && r2[z] < 600000 + n4 && gone[r2[z] - 600000]) eq = 0;
            sameld += eq;
        }
        print_cstr("  deletes + save: recall@10 "); print_i64((i64)(1000.0f * r_live)); print_cstr(" -> "); print_i64((i64)(1000.0f * r_saved));
        print_cstr(" after the compaction, loaded without a rebuild, same top-10 on "); print_i64(sameld); print_cstr("/"); print_i64(nq4); print_cstr("\n");
        if (r_saved != r_live || sameld != nq4) ok = 0;

        if (ld) {
            store_live_stats(ld, (i64 *)0, &nreb0, (f64 *)0);
            store_set_live(ld, 0, 0.0f);                         /* no rebuilds: just index the new rows */
            for (i = n4; i < n4 + nd; ++i) store_add(ld, x4 + (size_t)i * d4, 600000 + i);
            store_maintain(ld);
            store_set_probe(ld, 8);
            r_d0 = sd_recall_(ld, x4, n4 + nd, d4, qd, nq4, 600000, gone);
            reasons = ld->drifted;
            store_set_live(ld, 0, 1.0f);                         /* rebuilds allowed again */
            did = store_maintain(ld);
            store_live_stats(ld, (i64 *)0, &nreb, (f64 *)0);
            r_d1 = sd_recall_(ld, x4, n4 + nd, d4, qd, nq4, 600000, gone);
            print_cstr("  drift: "); print_i64(nd); print_cstr(" rows from 30 new topics -> "); print_i64(reasons);
            print_cstr(" far windows, rebuilt "); print_i64(nreb - nreb0); print_cstr("x; recall@10 of their queries at 8 probes ");
            print_i64((i64)(1000.0f * r_d0)); print_cstr(" -> "); print_i64((i64)(1000.0f * r_d1)); print_cstr("/1000\n");
            if (reasons < 1 || !(did & 2) || nreb != nreb0 + 1 || r_d1 < r_d0 + 0.05f) ok = 0;
            store_destroy(ld);
        }
        store_ivf_min_ = STORE_IVF_MIN;
        store_destroy(lv);
        remove("/tmp/store_demo_live.vxi"); remove("/tmp/store_demo_live.vxi.j");
        free(mr); free(x4); free(q4); free(gone);
    }

    /* journal: base save, then only the changes are synced; a reload replays them (torn tail tolerated) */
    {
        RagStore *j = store_new(a, dim), *jl;
        i32 jn = 500, hit = 0, gone = 0, want_live;
        FILE *g;
        store_set_fast(j, 0.9f);
        for (i = 0; i < jn; ++i) store_add(j, vv + (size_t)i * dim, 9000 + i);
        store_build(j);
        if (store_sync(j, "/tmp/store_demo_j.vxi") != 0) ok = 0;            /* first call = full save */
        for (i = jn; i < jn + 60; ++i) store_add(j, vv + (size_t)i * dim, 9000 + i);
        for (i = 0; i < 20; ++i) store_delete(j, 9000 + i * 7);
        store_set_meta(j, 9000 + 3, 42);
        if (store_sync(j, "/tmp/store_demo_j.vxi") != 0) ok = 0;            /* journal flush only */
        want_live = store_count(j);
        g = fopen("/tmp/store_demo_j.vxi.j", "ab"); if (g) { fwrite("\x47\x01\x02", 1, 3, g); fclose(g); }   /* a torn frame */
        jl = store_load(a, "/tmp/store_demo_j.vxi");
        if (!jl) ok = 0;
        else {
            for (i = jn; i < jn + 60; ++i) { i64 r1[1]; f32 c1[1]; if (store_search(jl, vv + (size_t)i * dim, 1, r1, c1) > 0 && r1[0] == 9000 + i) ++hit; }
            for (i = 0; i < 20; ++i) { i64 r1[3]; f32 c1[3]; i32 h = store_search(jl, vv + (size_t)(i * 7) * dim, 3, r1, c1), q2; for (q2 = 0; q2 < h; ++q2) if (r1[q2] == 9000 + i * 7) ++gone; }
            print_cstr("  journal: live "); print_i64(store_count(jl)); print_cstr(" of "); print_i64(want_live);
            print_cstr(", replayed adds found "); print_i64(hit); print_cstr("/60, deleted still visible "); print_i64(gone); print_cstr("\n");
            if (store_count(jl) != want_live || hit < 55 || gone != 0) ok = 0;
            { i32 jd = 1; store_integrity(jl, (i64 *)0, (i64 *)0, (i64 *)0, (i64 *)0, (i64 *)0, &jd); if (jd) { print_cstr("  journal: a torn tail taken for damage\n"); ok = 0; } }
            store_destroy(jl);
        }
        store_destroy(j);
    }

    /* background snapshots: a child writes a copy-on-write image of this moment while rows keep coming. A copy to
     * another file holds that moment only; a snapshot of the store's own file takes the journal over, so the file and
     * its journal hold everything, including what came in while the child wrote. */
    {
        RagStore *b = store_new(a, dim), *lc, *lo;
        const char *P = "/tmp/store_demo_snap.vxi", *C = "/tmp/store_demo_snap_copy.vxi";
        i32 nb = 3000, at_fork, r1, r2, r3, w1, w2, in_copy = 0, late_in_copy = 0, found = 0, gone = 0, self = 0;
        i64 us = 0;
        for (i = 0; i < nb; ++i) store_add(b, vv + (size_t)i * dim, 20000 + i);
        store_build(b);
        if (store_save(b, P) != 0) ok = 0;
        for (i = 0; i < 100; ++i) store_delete(b, 20000 + i * 3);    /* tombstones the child leaves out */
        at_fork = store_count(b);
        r1 = store_snapshot(b, C);
        for (i = nb; i < nb + 400; ++i) store_add(b, vv + (size_t)i * dim, 20000 + i);
        w1 = store_snapshot_poll(b, 1);
        r2 = store_snapshot(b, P);
        for (i = nb + 400; i < nb + 600; ++i) store_add(b, vv + (size_t)i * dim, 20000 + i);
        store_delete(b, 20000 + 1);
        w2 = store_snapshot_poll(b, 1);
        store_snapshot_info(b, (i32 *)0, &r3, &us);
        for (i = nb + 600; i < nb + 700; ++i) store_add(b, vv + (size_t)i * dim, 20000 + i);
        if (store_sync(b, P) != 0) ok = 0;
        lc = store_load(a, C);
        lo = store_load(a, P);
        if (!lc || !lo) ok = 0;
        else {
            for (i = 0; i < nb + 700; i += 7) {
                i64 r1h[3]; f32 c1[3];
                i32 h = store_search(lc, vv + (size_t)i * dim, 3, r1h, c1), q, hit = 0;
                for (q = 0; q < h; ++q) if (r1h[q] == 20000 + i) hit = 1;
                if (i < nb && hit) ++in_copy;
                if (i >= nb && hit) ++late_in_copy;
                h = store_search(lo, vv + (size_t)i * dim, 3, r1h, c1);
                for (q = 0, hit = 0; q < h; ++q) if (r1h[q] == 20000 + i) hit = 1;
                if ((i < 300 && i % 3 == 0) || i == 1) { if (hit) ++gone; }
                else { if (hit) ++found; ++self; }
            }
            print_cstr("  snapshot: a copy of "); print_i64(store_count(lc)); print_cstr(" rows (the moment of the fork: ");
            print_i64(at_fork); print_cstr(", none of the 700 added later); its own file + journal "); print_i64(store_count(lo));
            print_cstr(" of "); print_i64(store_count(b)); print_cstr(", rows found "); print_i64(found); print_cstr("/"); print_i64(self);
            print_cstr(", deleted still visible "); print_i64(gone); print_cstr(", last one took "); print_i64(us / 1000); print_cstr(" ms\n");
            if (r1 != 0 || r2 != 0 || w1 != 0 || w2 != 0 || r3 != 0 || store_count(lc) != at_fork || late_in_copy != 0
                || in_copy * 100 < ((nb + 6) / 7 - 15) * 95 || store_count(lo) != store_count(b) || found != self || gone != 0 || lo->jf2) ok = 0;
            store_destroy(lc); store_destroy(lo);
        }
        store_destroy(b);
        remove(P); remove("/tmp/store_demo_snap.vxi.j"); remove("/tmp/store_demo_snap.vxi.jn");
        remove(C); remove("/tmp/store_demo_snap_copy.vxi.j");
    }
    if (!sd_texts_(a)) ok = 0;
    if (!sd_aggression_(a)) ok = 0;
    if (!sd_tiers_(a)) ok = 0;
    if (!sd_batches_(a)) ok = 0;
    if (!sd_cascade_(a)) ok = 0;
    if (!sd_nums_(a)) ok = 0;
    if (!sd_sums_(a)) ok = 0;
    if (!sd_lens_(a)) ok = 0;
    print_cstr(ok ? "  OK\n" : "  FAIL\n");
    print_flush();
    arena_free(a);
    return ok ? 0 : 1;
}
