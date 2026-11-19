/* graph.c - links between rows and walks over them (include/graph.h): a module of its own beside the store. It knows
 * only ids; vecindex.c includes it so it is part of libvecindex, but nothing of the store is used here.
 *
 * Memory: structure of arrays, each a region of its own from the OS's pages (mem.h's Reg), doubled as it fills -
 * nodes (id, first link out, first link in), links (from, to, type, weight, live, next out, next in), and a map
 * id -> node by linear probing. A removed link stays in its lists, marked dead, until vg_compact lays the live ones
 * out again. A walk's working memory is an arena of its own (mem.h's Gen), freed whole when it returns.
 *
 * The file: a 16-byte header ('VXG1', version 1, 8 bytes 0), then 32-byte records, little-endian:
 *     op u8 ('L' link, 'U' unlink, 'X' drop), type u8 (255: every type), 0 u16, weight f32, a i64, b i64, CRC-32C of
 *     the 24 bytes before it u32, 0 u32
 * applied in order at open; the first short record or one whose CRC fails ends the log, and the file is cut there
 * so the next write follows a whole record. A record of an op this build does not know, with a good CRC, is a newer
 * build's: the open is refused, never misread. */
#include "graph.h"
#include "base.h"
#include "mem.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#define VG_MAGIC 0x31475856u        /* 'VXG1' */
#define VG_VERSION 1
#define VG_HDR 16
#define VG_REC 32
#define VG_NMAX (1 << 20)           /* the most nodes one walk keeps */
#define VG_PATH 1024
#define VG_EBUF 256

/* ---- errors: a buffer a thread, from the OS's pages ---- */

static pthread_key_t vg_ek_;
static pthread_once_t vg_eonce_ = PTHREAD_ONCE_INIT;
static void vg_efree_(void *p) { mem_unpages_(p, VG_EBUF); }
static void vg_ekey_(void) { pthread_key_create(&vg_ek_, vg_efree_); }
static char *vg_ebuf_(void)
{
    char *e;
    pthread_once(&vg_eonce_, vg_ekey_);
    e = (char *)pthread_getspecific(vg_ek_);
    if (!e && (e = (char *)mem_pages_(VG_EBUF)) != (char *)0) pthread_setspecific(vg_ek_, e);
    return e;
}
static int vg_fail_(const char *a, const char *b)
{
    char *e = vg_ebuf_();
    size_t n;
    if (!e) return -1;
    n = strlen(a); if (n > VG_EBUF / 2) n = VG_EBUF / 2;
    memcpy(e, a, n); e[n] = 0;
    if (b) strncat(e, b, (size_t)VG_EBUF - 1 - n);
    return -1;
}
VI_API const char *vg_error(void) { char *e = vg_ebuf_(); return e ? e : "out of memory"; }

/* ---- bytes ---- */

static u32 vg_crc_(const u8 *p, size_t n)                        /* CRC-32C, bit by bit: 24 bytes a record */
{
    u32 c = 0xffffffffu;
    size_t i;
    i32 k;
    for (i = 0; i < n; ++i) { c ^= p[i]; for (k = 0; k < 8; ++k) c = (c >> 1) ^ (0x82f63b78u & (0u - (c & 1u))); }
    return ~c;
}
static void vg_p32_(u8 *b, u32 v) { b[0] = (u8)v; b[1] = (u8)(v >> 8); b[2] = (u8)(v >> 16); b[3] = (u8)(v >> 24); }
static void vg_p64_(u8 *b, u64 v) { vg_p32_(b, (u32)v); vg_p32_(b + 4, (u32)(v >> 32)); }
static u32 vg_g32_(const u8 *b) { return (u32)b[0] | (u32)b[1] << 8 | (u32)b[2] << 16 | (u32)b[3] << 24; }
static u64 vg_g64_(const u8 *b) { return (u64)vg_g32_(b) | (u64)vg_g32_(b + 4) << 32; }

static u64 vg_mix_(u64 x)                                        /* splitmix64's finaliser */
{
    x += (u64)0x9e3779b9u << 32 | 0x7f4a7c15u;
    x = (x ^ (x >> 30)) * ((u64)0xbf58476du << 32 | 0x1ce4e5b9u);
    x = (x ^ (x >> 27)) * ((u64)0x94d049bbu << 32 | 0x133111ebu);
    return x ^ (x >> 31);
}

static i32 vg_okf_(f32 w) { return w > 0.0f && w - w == 0.0f; }   /* > 0 and finite */

/* ---- the graph ---- */

struct vg_graph {
    pthread_rwlock_t lock;
    i64 *nid;                       /* nodes: the id */
    i32 *head, *rhead;              /* first link out, first link in (-1: none) */
    i32 nn, ncap;
    i64 *hk;                        /* id -> node + 1 (0: empty) */
    i32 *hv, hcap;
    i32 *esrc, *edst, *enext, *ernext;   /* links: from and to (nodes), next out of from, next into to */
    f32 *ew;
    u8 *ety, *elive;
    i32 ne, ecap, nlive;
    Reg rn[3], rh[2], re[7];
    FILE *f;
    i32 broken;                     /* the file could not be opened again after a compaction: writes refused */
    char path[VG_PATH];
};

#define VG_FIT(reg, ptr, type, have, want) \
    (vp_ = (void *)(ptr), reg_fit_(&(reg), &vp_, (size_t)(have) * sizeof(type), (size_t)(want) * sizeof(type)) ? -1 : ((ptr) = (type *)vp_, 0))

static i32 vg_ngrow_(vg_graph *g)
{
    void *vp_;
    i32 nc = g->ncap ? g->ncap * 2 : 256;
    if (VG_FIT(g->rn[0], g->nid, i64, g->nn, nc) || VG_FIT(g->rn[1], g->head, i32, g->nn, nc)
        || VG_FIT(g->rn[2], g->rhead, i32, g->nn, nc)) return -1;
    g->ncap = nc;
    return 0;
}

static i32 vg_egrow_(vg_graph *g)
{
    void *vp_;
    i32 nc = g->ecap ? g->ecap * 2 : 1024;
    if (VG_FIT(g->re[0], g->esrc, i32, g->ne, nc) || VG_FIT(g->re[1], g->edst, i32, g->ne, nc)
        || VG_FIT(g->re[2], g->enext, i32, g->ne, nc) || VG_FIT(g->re[3], g->ernext, i32, g->ne, nc)
        || VG_FIT(g->re[4], g->ew, f32, g->ne, nc) || VG_FIT(g->re[5], g->ety, u8, g->ne, nc)
        || VG_FIT(g->re[6], g->elive, u8, g->ne, nc)) return -1;
    g->ecap = nc;
    return 0;
}

static i32 vg_find_(const vg_graph *g, i64 id)                   /* the node of id, -1 none */
{
    u32 m, i;
    if (!g->hcap) return -1;
    m = (u32)g->hcap - 1u;
    for (i = (u32)vg_mix_((u64)id) & m; g->hv[i]; i = (i + 1u) & m) if (g->hk[i] == id) return g->hv[i] - 1;
    return -1;
}

static i32 vg_rehash_(vg_graph *g, i32 cap)                      /* cap: a power of two above 2 x the nodes */
{
    Reg rk, rv;
    void *pk = (void *)0, *pv = (void *)0;
    i64 *k;
    i32 *v, n;
    u32 m, i;
    memset(&rk, 0, sizeof(rk)); memset(&rv, 0, sizeof(rv));
    if (reg_fit_(&rk, &pk, 0, (size_t)cap * sizeof(i64)) || reg_fit_(&rv, &pv, 0, (size_t)cap * sizeof(i32))) {
        reg_free_(&rk); reg_free_(&rv);
        return -1;
    }
    k = (i64 *)pk; v = (i32 *)pv;                                /* the OS's pages: zeroed */
    m = (u32)cap - 1u;
    for (n = 0; n < g->nn; ++n) {
        for (i = (u32)vg_mix_((u64)g->nid[n]) & m; v[i]; i = (i + 1u) & m) {}
        k[i] = g->nid[n]; v[i] = n + 1;
    }
    reg_free_(&g->rh[0]); reg_free_(&g->rh[1]);
    g->rh[0] = rk; g->rh[1] = rv; g->hk = k; g->hv = v; g->hcap = cap;
    return 0;
}

static i32 vg_node_(vg_graph *g, i64 id)                         /* the node of id, made when it is not there; -1 */
{
    i32 n = vg_find_(g, id);
    u32 m, i;
    if (n >= 0) return n;
    if (g->nn >= 0x3fffffff) return -1;
    if ((g->nn + 1) * 2 > g->hcap && vg_rehash_(g, g->hcap ? g->hcap * 2 : 512)) return -1;
    if (g->nn == g->ncap && vg_ngrow_(g)) return -1;
    n = g->nn++;
    g->nid[n] = id; g->head[n] = -1; g->rhead[n] = -1;
    m = (u32)g->hcap - 1u;
    for (i = (u32)vg_mix_((u64)id) & m; g->hv[i]; i = (i + 1u) & m) {}
    g->hk[i] = id; g->hv[i] = n + 1;
    return n;
}

static void vg_kill_(vg_graph *g, i32 e) { g->elive[e] = 0; --g->nlive; }

/* the changes themselves, under the write lock, not logged */
static i32 vg_link_(vg_graph *g, i64 a, i64 b, i32 t, f32 w)
{
    i32 na = vg_node_(g, a), nb, e;
    if (na < 0 || (nb = vg_node_(g, b)) < 0) return -1;
    for (e = g->head[na]; e >= 0; e = g->enext[e])
        if (g->elive[e] && g->edst[e] == nb && g->ety[e] == (u8)t) { g->ew[e] = w; return 0; }
    if (g->ne == g->ecap && vg_egrow_(g)) return -1;
    e = g->ne++;
    g->esrc[e] = na; g->edst[e] = nb; g->ety[e] = (u8)t; g->ew[e] = w; g->elive[e] = 1;
    g->enext[e] = g->head[na]; g->head[na] = e;
    g->ernext[e] = g->rhead[nb]; g->rhead[nb] = e;
    ++g->nlive;
    return 0;
}
static i32 vg_unlink_(vg_graph *g, i64 a, i64 b, i32 t)
{
    i32 na = vg_find_(g, a), nb = vg_find_(g, b), e, c = 0;
    if (na < 0 || nb < 0) return 0;
    for (e = g->head[na]; e >= 0; e = g->enext[e])
        if (g->elive[e] && g->edst[e] == nb && (t < 0 || g->ety[e] == (u8)t)) { vg_kill_(g, e); ++c; }
    return c;
}
static i32 vg_drop_(vg_graph *g, i64 a)
{
    i32 n = vg_find_(g, a), e, c = 0;
    if (n < 0) return 0;
    for (e = g->head[n]; e >= 0; e = g->enext[e]) if (g->elive[e]) { vg_kill_(g, e); ++c; }
    for (e = g->rhead[n]; e >= 0; e = g->ernext[e]) if (g->elive[e]) { vg_kill_(g, e); ++c; }
    return c;
}

/* ---- the file ---- */

static void vg_rec_(u8 *r, u8 op, i32 t, f32 w, i64 a, i64 b)
{
    u32 wb;
    memset(r, 0, VG_REC);
    r[0] = op; r[1] = (u8)(t < 0 ? 255 : t);
    memcpy(&wb, &w, 4); vg_p32_(r + 4, wb);
    vg_p64_(r + 8, (u64)a); vg_p64_(r + 16, (u64)b);
    vg_p32_(r + 24, vg_crc_(r, 24));
}

static i32 vg_log_(vg_graph *g, u8 op, i32 t, f32 w, i64 a, i64 b)
{
    u8 r[VG_REC];
    if (g->broken) return vg_fail_("the graph's file was lost in a compaction: reopen it ", g->path);
    if (!g->f) return 0;
    vg_rec_(r, op, t, w, a, b);
    return fwrite(r, 1, VG_REC, g->f) == VG_REC ? 0 : vg_fail_("cannot write the graph's file ", g->path);
}

static void vg_fsync_dir_(const char *path)
{
    char d[VG_PATH];
    const char *s = strrchr(path, '/');
    int fd;
    if (!s) { d[0] = '.'; d[1] = 0; }
    else if (s == path) { d[0] = '/'; d[1] = 0; }
    else { memcpy(d, path, (size_t)(s - path)); d[s - path] = 0; }
    if ((fd = open(d, O_RDONLY)) >= 0) { fsync(fd); close(fd); }
}

static i32 vg_head_w_(FILE *f)
{
    u8 h[VG_HDR];
    memset(h, 0, sizeof(h));
    vg_p32_(h, VG_MAGIC); vg_p32_(h + 4, VG_VERSION);
    return fwrite(h, 1, VG_HDR, f) == VG_HDR ? 0 : -1;
}

/* the log of f applied to g: 0, or -1 (not a graph's file, a newer build's, out of memory) */
static i32 vg_replay_(vg_graph *g, FILE *f)
{
    u8 h[VG_HDR], r[VG_REC];
    off_t good = VG_HDR;
    size_t got = fread(h, 1, VG_HDR, f);
    if (got == 0 && feof(f)) {                                   /* empty: a crash between making it and its header */
        if (fseeko(f, 0, SEEK_SET) != 0 || vg_head_w_(f) || fflush(f) || fsync(fileno(f))) return vg_fail_("cannot write ", g->path);
        return 0;
    }
    if (got != VG_HDR || vg_g32_(h) != VG_MAGIC) return vg_fail_("not a graph's file: ", g->path);
    if (vg_g32_(h + 4) != VG_VERSION) return vg_fail_("a graph's file of a newer version: ", g->path);
    while (fread(r, 1, VG_REC, f) == VG_REC && vg_g32_(r + 24) == vg_crc_(r, 24)) {
        i64 a = (i64)vg_g64_(r + 8), b = (i64)vg_g64_(r + 16);
        i32 t = r[1] == 255 ? -1 : (i32)r[1];
        u32 wb = vg_g32_(r + 4);
        f32 w;
        memcpy(&w, &wb, 4);
        if (r[0] == 'L') { if (t < 0 || t > 31 || !vg_okf_(w) || a == b) return vg_fail_("a damaged link in ", g->path); if (vg_link_(g, a, b, t, w)) return vg_fail_("out of memory reading ", g->path); }
        else if (r[0] == 'U') vg_unlink_(g, a, b, t);
        else if (r[0] == 'X') vg_drop_(g, a);
        else return vg_fail_("written by a newer build (a record it does not know): ", g->path);
        good += VG_REC;
    }
    fflush(f);
    if (ftruncate(fileno(f), good) != 0) return vg_fail_("cannot cut the torn end of ", g->path);   /* a torn or damaged tail */
    return fseeko(f, good, SEEK_SET) == 0 ? 0 : vg_fail_("cannot seek in ", g->path);
}

VI_API vg_graph *vg_open(const char *path)
{
    vg_graph *g = (vg_graph *)mem_pages_(sizeof(vg_graph));
    FILE *f;
    if (!g) { vg_fail_("out of memory", (const char *)0); return (vg_graph *)0; }
    pthread_rwlock_init(&g->lock, (const pthread_rwlockattr_t *)0);
    if (!path) return g;
    if (strlen(path) >= VG_PATH) { vg_fail_("the path is too long", (const char *)0); vg_close(g); return (vg_graph *)0; }
    strcpy(g->path, path);
    if ((f = fopen(path, "r+b")) != (FILE *)0) {
        if (vg_replay_(g, f)) { fclose(f); vg_close(g); return (vg_graph *)0; }
    } else {
        if (!(f = fopen(path, "w+b")) || vg_head_w_(f) || fflush(f) || fsync(fileno(f))) {
            vg_fail_("cannot make the graph's file ", path);
            if (f) fclose(f);
            vg_close(g);
            return (vg_graph *)0;
        }
        vg_fsync_dir_(path);
    }
    g->f = f;
    return g;
}

VI_API void vg_close(vg_graph *g)
{
    i32 i;
    if (!g) return;
    if (g->f) { fflush(g->f); fsync(fileno(g->f)); fclose(g->f); }
    for (i = 0; i < 3; ++i) reg_free_(&g->rn[i]);
    for (i = 0; i < 2; ++i) reg_free_(&g->rh[i]);
    for (i = 0; i < 7; ++i) reg_free_(&g->re[i]);
    pthread_rwlock_destroy(&g->lock);
    mem_unpages_(g, sizeof(vg_graph));
}

VI_API int vg_sync(vg_graph *g)
{
    i32 r = 0;
    if (!g) return vg_fail_("the graph is NULL", (const char *)0);
    pthread_rwlock_wrlock(&g->lock);
    if (g->f && (fflush(g->f) != 0 || fsync(fileno(g->f)) != 0)) r = vg_fail_("cannot flush ", g->path);
    pthread_rwlock_unlock(&g->lock);
    return r;
}

/* ---- the writes ---- */

VI_API int vg_link(vg_graph *g, vi_i64 from, vi_i64 to, int type, float weight)
{
    i32 r;
    if (!g) return vg_fail_("the graph is NULL", (const char *)0);
    if (type < 0 || type > 31) return vg_fail_("a link's type is 0 to 31", (const char *)0);
    if (!vg_okf_(weight)) return vg_fail_("a link's weight is above 0 and finite", (const char *)0);
    if (from == to) return vg_fail_("a link from an id to itself", (const char *)0);
    pthread_rwlock_wrlock(&g->lock);
    r = vg_log_(g, 'L', type, weight, (i64)from, (i64)to);       /* the log first: the file is what an open believes */
    if (!r && vg_link_(g, (i64)from, (i64)to, type, weight)) r = vg_fail_("out of memory", (const char *)0);
    pthread_rwlock_unlock(&g->lock);
    return r;
}

VI_API int vg_unlink(vg_graph *g, vi_i64 from, vi_i64 to, int type)
{
    i32 r;
    if (!g) return vg_fail_("the graph is NULL", (const char *)0);
    if (type < -1 || type > 31) return vg_fail_("a link's type is 0 to 31, or -1 for every type", (const char *)0);
    pthread_rwlock_wrlock(&g->lock);
    r = vg_log_(g, 'U', type, 1.0f, (i64)from, (i64)to);
    if (!r) r = vg_unlink_(g, (i64)from, (i64)to, type);
    pthread_rwlock_unlock(&g->lock);
    return r;
}

VI_API int vg_drop(vg_graph *g, vi_i64 id)
{
    i32 r;
    if (!g) return vg_fail_("the graph is NULL", (const char *)0);
    pthread_rwlock_wrlock(&g->lock);
    r = vg_log_(g, 'X', -1, 1.0f, (i64)id, 0);
    if (!r) r = vg_drop_(g, (i64)id);
    pthread_rwlock_unlock(&g->lock);
    return r;
}

/* the file and the memory laid out again with only the live links, and only the ids that still have one */
VI_API int vg_compact(vg_graph *g)
{
    vg_graph ng;
    i32 e, i, r = 0;
    char tmp[VG_PATH + 8];
    FILE *f = (FILE *)0;
    if (!g) return vg_fail_("the graph is NULL", (const char *)0);
    pthread_rwlock_wrlock(&g->lock);
    if (g->f) {                                                  /* the file first: if it fails, nothing has changed */
        u8 rec[VG_REC];
        if (strlen(g->path) + 5 > sizeof(tmp)) r = vg_fail_("the path is too long", (const char *)0);
        else { strcpy(tmp, g->path); strcat(tmp, ".tmp"); }
        if (!r && (!(f = fopen(tmp, "wb")) || vg_head_w_(f))) r = vg_fail_("cannot write ", tmp);
        for (e = 0; !r && e < g->ne; ++e) {
            if (!g->elive[e]) continue;
            vg_rec_(rec, 'L', g->ety[e], g->ew[e], g->nid[g->esrc[e]], g->nid[g->edst[e]]);
            if (fwrite(rec, 1, VG_REC, f) != VG_REC) r = vg_fail_("cannot write ", tmp);
        }
        if (!r && (fflush(f) != 0 || fsync(fileno(f)) != 0)) r = vg_fail_("cannot flush ", tmp);
        if (f) fclose(f);
        f = (FILE *)0;
        if (!r && rename(tmp, g->path) != 0) r = vg_fail_("cannot rename it over ", g->path);
        if (r) { remove(tmp); pthread_rwlock_unlock(&g->lock); return -1; }
        vg_fsync_dir_(g->path);
        fclose(g->f);
        if (!(g->f = fopen(g->path, "r+b")) || fseek(g->f, 0, SEEK_END) != 0) {
            if (g->f) fclose(g->f);
            g->f = (FILE *)0; g->broken = 1;                     /* the file is whole on the disk; this handle can write no more */
            r = vg_fail_("cannot open the new ", g->path);
        }
    }
    memset(&ng, 0, sizeof(ng));                                  /* the memory: a new graph, the live links into it */
    for (e = 0; !r && e < g->ne; ++e)
        if (g->elive[e] && vg_link_(&ng, g->nid[g->esrc[e]], g->nid[g->edst[e]], g->ety[e], g->ew[e])) r = vg_fail_("out of memory", (const char *)0);
    if (r) {
        for (i = 0; i < 3; ++i) reg_free_(&ng.rn[i]);
        for (i = 0; i < 2; ++i) reg_free_(&ng.rh[i]);
        for (i = 0; i < 7; ++i) reg_free_(&ng.re[i]);
    } else {
        for (i = 0; i < 3; ++i) { reg_free_(&g->rn[i]); g->rn[i] = ng.rn[i]; }
        for (i = 0; i < 2; ++i) { reg_free_(&g->rh[i]); g->rh[i] = ng.rh[i]; }
        for (i = 0; i < 7; ++i) { reg_free_(&g->re[i]); g->re[i] = ng.re[i]; }
        g->nid = ng.nid; g->head = ng.head; g->rhead = ng.rhead; g->nn = ng.nn; g->ncap = ng.ncap;
        g->hk = ng.hk; g->hv = ng.hv; g->hcap = ng.hcap;
        g->esrc = ng.esrc; g->edst = ng.edst; g->enext = ng.enext; g->ernext = ng.ernext;
        g->ew = ng.ew; g->ety = ng.ety; g->elive = ng.elive; g->ne = ng.ne; g->ecap = ng.ecap; g->nlive = ng.nlive;
    }
    pthread_rwlock_unlock(&g->lock);
    return r;
}

/* ---- the reads ---- */

static i32 vg_take_(const vg_graph *g, i32 e, unsigned types, f32 minw)
{
    return g->elive[e] && (!types || ((types >> g->ety[e]) & 1u)) && g->ew[e] >= minw;
}

VI_API int vg_links(vg_graph *g, vi_i64 id, int dir, unsigned types, vi_i64 *ids, int *ltypes, float *weights, int max)
{
    i32 n, e, c = 0;
    if (!g) return vg_fail_("the graph is NULL", (const char *)0);
    if (dir < VG_OUT || dir > VG_BOTH) return vg_fail_("dir is VG_OUT, VG_IN or VG_BOTH", (const char *)0);
    if (max > 0 && !ids) return vg_fail_("ids is NULL", (const char *)0);
    pthread_rwlock_rdlock(&g->lock);
    if ((n = vg_find_(g, (i64)id)) >= 0) {
        if (dir & VG_OUT)
            for (e = g->head[n]; e >= 0; e = g->enext[e]) if (vg_take_(g, e, types, 0.0f)) {
                if (c < max) { ids[c] = (vi_i64)g->nid[g->edst[e]]; if (ltypes) ltypes[c] = g->ety[e]; if (weights) weights[c] = g->ew[e]; }
                ++c;
            }
        if (dir & VG_IN)
            for (e = g->rhead[n]; e >= 0; e = g->ernext[e]) if (vg_take_(g, e, types, 0.0f)) {
                if (c < max) { ids[c] = (vi_i64)g->nid[g->esrc[e]]; if (ltypes) ltypes[c] = g->ety[e]; if (weights) weights[c] = g->ew[e]; }
                ++c;
            }
    }
    pthread_rwlock_unlock(&g->lock);
    return c;
}

VI_API void vg_count(vg_graph *g, vi_i64 *nodes, vi_i64 *links)
{
    i32 n, e;
    i64 c = 0;
    if (!g) return;
    pthread_rwlock_rdlock(&g->lock);
    for (n = 0; n < g->nn; ++n) {
        i32 has = 0;
        for (e = g->head[n]; e >= 0 && !has; e = g->enext[e]) has = g->elive[e];
        for (e = g->rhead[n]; e >= 0 && !has; e = g->ernext[e]) has = g->elive[e];
        c += has;
    }
    if (nodes) *nodes = (vi_i64)c;
    if (links) *links = (vi_i64)g->nlive;
    pthread_rwlock_unlock(&g->lock);
}

/* ---- walks ---- */

/* a walk's map id -> its slot (+ 1; 0 empty), over a power of two at least twice its slots */
typedef struct { i64 *k; i32 *v; u32 m; } VgMap;

static i32 vg_map_init_(Gen *a, VgMap *mp, i32 slots)
{
    u32 cap = 16;
    while (cap < (u32)slots * 2u) cap <<= 1;
    mp->k = (i64 *)gen_alloc_(a, (size_t)cap * sizeof(i64));
    mp->v = (i32 *)gen_alloc_(a, (size_t)cap * sizeof(i32));      /* zeroed */
    mp->m = cap - 1u;
    return mp->k && mp->v ? 0 : -1;
}
static u32 vg_map_at_(const VgMap *mp, i64 id)                   /* the cell of id, or the empty one it would take */
{
    u32 i;
    for (i = (u32)vg_mix_((u64)id) & mp->m; mp->v[i] && mp->k[i] != id; i = (i + 1u) & mp->m) {}
    return i;
}

typedef struct { i64 id; i32 node, dep; f32 sc; } VgHit;

static int vg_hit_cmp_(const void *a, const void *b)
{
    const VgHit *x = (const VgHit *)a, *y = (const VgHit *)b;
    if (x->sc != y->sc) return x->sc > y->sc ? -1 : 1;
    if (x->dep != y->dep) return x->dep < y->dep ? -1 : 1;
    return x->id < y->id ? -1 : x->id > y->id;
}

VI_API void vg_expand_defaults(vg_expand_opts *o)
{
    if (!o) return;
    o->max_depth = 2; o->max_nodes = 64; o->dir = VG_BOTH; o->types = 0; o->min_weight = 0.0f; o->decay = 0.5f;
}

VI_API int vg_expand(vg_graph *g, const vi_i64 *seeds, const float *seed_w, int ns, const vg_expand_opts *o,
                     vi_i64 *ids, float *scores, int *depths, int max)
{
    vg_expand_opts d;
    Gen a;
    VgMap mp;
    VgHit *h;
    i32 n = 0, i, lo, hi, dep, e;
    if (!g) return vg_fail_("the graph is NULL", (const char *)0);
    vg_expand_defaults(&d);
    if (o) d = *o;
    if (ns < 0 || (ns > 0 && !seeds)) return vg_fail_("seeds is NULL", (const char *)0);
    if (max < 0 || (max > 0 && !ids)) return vg_fail_("ids is NULL", (const char *)0);
    if (d.max_nodes < 1 || d.max_nodes > VG_NMAX) return vg_fail_("max_nodes is 1 to 1 << 20", (const char *)0);
    if (d.max_depth < 0 || d.dir < VG_OUT || d.dir > VG_BOTH || !(d.decay > 0.0f && d.decay <= 1.0f))
        return vg_fail_("max_depth >= 0, dir VG_OUT, VG_IN or VG_BOTH, decay in (0, 1]", (const char *)0);
    gen_init_(&a);
    h = (VgHit *)gen_alloc_(&a, (size_t)d.max_nodes * sizeof(VgHit));
    if (!h || vg_map_init_(&a, &mp, d.max_nodes)) { gen_free_(&a); return vg_fail_("out of memory", (const char *)0); }
    pthread_rwlock_rdlock(&g->lock);
    for (i = 0; i < ns; ++i) {                                   /* depth 0: the seeds */
        f32 w = seed_w ? seed_w[i] : 1.0f;
        u32 c;
        if (!vg_okf_(w)) continue;
        c = vg_map_at_(&mp, (i64)seeds[i]);
        if (mp.v[c]) { if (w > h[mp.v[c] - 1].sc) h[mp.v[c] - 1].sc = w; continue; }
        if (n == d.max_nodes) continue;
        h[n].id = (i64)seeds[i]; h[n].node = vg_find_(g, (i64)seeds[i]); h[n].dep = 0; h[n].sc = w;
        mp.k[c] = h[n].id; mp.v[c] = ++n;
    }
    for (lo = 0, hi = n, dep = 1; dep <= d.max_depth && lo < hi; lo = hi, hi = n, ++dep)   /* one level at a time */
        for (i = lo; i < hi; ++i) {
            i32 u = h[i].node, side;
            if (u < 0) continue;
            for (side = VG_OUT; side <= VG_IN; ++side) {
                if (!(d.dir & side)) continue;
                for (e = side == VG_OUT ? g->head[u] : g->rhead[u]; e >= 0; e = side == VG_OUT ? g->enext[e] : g->ernext[e]) {
                    i32 v;
                    u32 c;
                    f32 sc;
                    if (!vg_take_(g, e, d.types, d.min_weight)) continue;
                    v = side == VG_OUT ? g->edst[e] : g->esrc[e];
                    sc = h[i].sc * g->ew[e] * d.decay;
                    c = vg_map_at_(&mp, g->nid[v]);
                    if (mp.v[c]) { if (sc > h[mp.v[c] - 1].sc) h[mp.v[c] - 1].sc = sc; continue; }   /* the best path found */
                    if (n == d.max_nodes) continue;
                    h[n].id = g->nid[v]; h[n].node = v; h[n].dep = dep; h[n].sc = sc;
                    mp.k[c] = h[n].id; mp.v[c] = ++n;
                }
            }
        }
    pthread_rwlock_unlock(&g->lock);
    qsort(h, (size_t)n, sizeof(VgHit), vg_hit_cmp_);
    if (n > max) n = max;
    for (i = 0; i < n; ++i) { ids[i] = (vi_i64)h[i].id; if (scores) scores[i] = h[i].sc; if (depths) depths[i] = h[i].dep; }
    gen_free_(&a);
    return n;
}

VI_API void vg_ppr_defaults(vg_ppr_opts *o)
{
    if (!o) return;
    o->alpha = 0.15f; o->eps = 1e-4f; o->max_nodes = 4096; o->dir = VG_BOTH; o->types = 0;
}

typedef struct { i64 id; i32 node; u8 inq, force; f64 p, r; } VgPr;

static int vg_pr_cmp_(const void *a, const void *b)
{
    const VgPr *x = (const VgPr *)a, *y = (const VgPr *)b;
    if (x->p != y->p) return x->p > y->p ? -1 : 1;
    return x->id < y->id ? -1 : x->id > y->id;
}

/* the weight and the count of the links a walk may take out of node u */
static void vg_out_w_(const vg_graph *g, i32 u, i32 dir, unsigned types, f64 *w, i32 *cnt)
{
    i32 e;
    *w = 0.0; *cnt = 0;
    if (dir & VG_OUT) for (e = g->head[u]; e >= 0; e = g->enext[e]) if (vg_take_(g, e, types, 0.0f)) { *w += g->ew[e]; ++*cnt; }
    if (dir & VG_IN) for (e = g->rhead[u]; e >= 0; e = g->ernext[e]) if (vg_take_(g, e, types, 0.0f)) { *w += g->ew[e]; ++*cnt; }
}

VI_API int vg_ppr(vg_graph *g, const vi_i64 *seeds, const float *seed_w, int ns, const vg_ppr_opts *o,
                  vi_i64 *ids, float *scores, int max)
{
    vg_ppr_opts d;
    Gen a;
    VgMap mp;
    VgPr *h;
    i32 *q, n = 0, qh = 0, qn = 0, i, e, out;
    f64 tot = 0.0;
    i64 pushes = 0, cap;
    if (!g) return vg_fail_("the graph is NULL", (const char *)0);
    vg_ppr_defaults(&d);
    if (o) d = *o;
    if (ns < 1 || !seeds) return vg_fail_("at least one seed", (const char *)0);
    if (max < 0 || (max > 0 && !ids)) return vg_fail_("ids is NULL", (const char *)0);
    if (d.max_nodes < 1 || d.max_nodes > VG_NMAX) return vg_fail_("max_nodes is 1 to 1 << 20", (const char *)0);
    if (!(d.alpha > 0.0f && d.alpha < 1.0f) || !(d.eps > 0.0f) || d.dir < VG_OUT || d.dir > VG_BOTH)
        return vg_fail_("alpha in (0, 1), eps above 0, dir VG_OUT, VG_IN or VG_BOTH", (const char *)0);
    for (i = 0; i < ns; ++i) { f32 w = seed_w ? seed_w[i] : 1.0f; if (vg_okf_(w)) tot += w; }
    if (!(tot > 0.0)) return vg_fail_("the seeds' weights must be above 0 (one at least)", (const char *)0);
    gen_init_(&a);
    h = (VgPr *)gen_alloc_(&a, (size_t)d.max_nodes * sizeof(VgPr));
    q = (i32 *)gen_alloc_(&a, (size_t)d.max_nodes * sizeof(i32));    /* a ring: a node is in it at most once */
    if (!h || !q || vg_map_init_(&a, &mp, d.max_nodes)) { gen_free_(&a); return vg_fail_("out of memory", (const char *)0); }
    pthread_rwlock_rdlock(&g->lock);
    for (i = 0; i < ns; ++i) {
        f32 w = seed_w ? seed_w[i] : 1.0f;
        u32 c;
        if (!vg_okf_(w)) continue;
        c = vg_map_at_(&mp, (i64)seeds[i]);
        if (!mp.v[c]) {
            if (n == d.max_nodes) continue;
            memset(&h[n], 0, sizeof(VgPr));
            h[n].id = (i64)seeds[i]; h[n].node = vg_find_(g, (i64)seeds[i]);
            mp.k[c] = h[n].id; mp.v[c] = ++n;
        }
        h[mp.v[c] - 1].r += (f64)w / tot;
        h[mp.v[c] - 1].force = 1;                                /* a seed pushes once however little it holds */
        if (!h[mp.v[c] - 1].inq) { h[mp.v[c] - 1].inq = 1; q[(qh + qn++) % d.max_nodes] = mp.v[c] - 1; }
    }
    cap = (i64)64 * d.max_nodes + (i64)(64.0 / ((f64)d.eps * d.alpha) < 1e9 ? 64.0 / ((f64)d.eps * d.alpha) : 1e9);
    while (qn && pushes < cap) {                                 /* local push */
        i32 s = q[qh], u, cnt, side;
        f64 W, R, give;
        qh = (qh + 1) % d.max_nodes; --qn;
        h[s].inq = 0;
        u = h[s].node; R = h[s].r;
        if (u < 0) { h[s].p += R; h[s].r = 0.0; continue; }      /* an id with no links: it keeps its mass */
        vg_out_w_(g, u, d.dir, d.types, &W, &cnt);
        if (!cnt || !(W > 0.0)) { h[s].p += R; h[s].r = 0.0; continue; }
        if (!h[s].force && R < (f64)d.eps * cnt) continue;
        h[s].force = 0;
        ++pushes;
        h[s].p += (f64)d.alpha * R;
        h[s].r = 0.0;
        give = (1.0 - (f64)d.alpha) * R / W;
        for (side = VG_OUT; side <= VG_IN; ++side) {
            if (!(d.dir & side)) continue;
            for (e = side == VG_OUT ? g->head[u] : g->rhead[u]; e >= 0; e = side == VG_OUT ? g->enext[e] : g->ernext[e]) {
                i32 v, t;
                u32 c;
                if (!vg_take_(g, e, d.types, 0.0f)) continue;
                v = side == VG_OUT ? g->edst[e] : g->esrc[e];
                c = vg_map_at_(&mp, g->nid[v]);
                if (!mp.v[c]) {
                    if (n == d.max_nodes) continue;              /* past max_nodes: this mass is dropped */
                    memset(&h[n], 0, sizeof(VgPr));
                    h[n].id = g->nid[v]; h[n].node = v;
                    mp.k[c] = h[n].id; mp.v[c] = ++n;
                }
                t = mp.v[c] - 1;
                h[t].r += give * g->ew[e];
                if (!h[t].inq && h[t].r >= (f64)d.eps) { h[t].inq = 1; q[(qh + qn++) % d.max_nodes] = t; }
            }
        }
    }
    pthread_rwlock_unlock(&g->lock);
    qsort(h, (size_t)n, sizeof(VgPr), vg_pr_cmp_);
    for (i = out = 0; i < n && out < max; ++i) {
        if (!(h[i].p > 0.0)) break;
        ids[out] = (vi_i64)h[i].id; if (scores) scores[out] = (float)h[i].p; ++out;
    }
    gen_free_(&a);
    return out;
}

/* ---- a context through the graph: the store's public API only ---- */

#define VG_CTX_MAX 256              /* as VI_CTX_HITS_MAX */

VI_API void vg_ctx_defaults(vg_ctx_opts *o)
{
    if (!o) return;
    o->walk = VG_WALK_EXPAND; o->hits = 16;
    vg_expand_defaults(&o->expand);
    vg_ppr_defaults(&o->ppr);
}

typedef struct { i64 id; f32 sc; } VgFz;
static int vg_fz_cmp_(const void *a, const void *b)
{
    const VgFz *x = (const VgFz *)a, *y = (const VgFz *)b;
    if (x->sc != y->sc) return x->sc > y->sc ? -1 : 1;
    return x->id < y->id ? -1 : x->id > y->id;
}

VI_API int vg_context(vg_graph *g, vi_db *db, const float *query, const char *text, int tlen, const vi_ctx_opts *co,
                      const vg_ctx_opts *go, vi_ctx_span *spans, int max_spans, char *buf, int cap, int *used)
{
    vi_ctx_opts c;
    vg_ctx_opts o;
    vi_i64 sid[VG_CTX_MAX], wid[VG_CTX_MAX], pid[VG_CTX_MAX];
    float ssc[VG_CTX_MAX], wsc[VG_CTX_MAX], psc[VG_CTX_MAX];
    VgFz fz[2 * VG_CTX_MAX];
    i32 ns, nw = 0, np = 0, i, j, nf, r;
    if (used) *used = 0;
    if (!g || !db) return vg_fail_("the graph or the store is NULL", (const char *)0);
    vi_ctx_defaults(&c); if (co) c = *co;
    vg_ctx_defaults(&o); if (go) o = *go;
    if (c.k < 1 || c.k > VG_CTX_MAX) return vg_fail_("co's k (the seeds) is 1 to 256", (const char *)0);
    if (o.hits < 1 || o.hits > VG_CTX_MAX) return vg_fail_("hits is 1 to 256", (const char *)0);
    if (o.walk < VG_WALK_EXPAND || o.walk > VG_WALK_BOTH) return vg_fail_("walk is VG_WALK_EXPAND, VG_WALK_PPR or VG_WALK_BOTH", (const char *)0);
    if ((ns = vi_search_mmr(db, query, text, tlen, c.filter, c.k, c.lambda, c.dup, sid, ssc)) < 0) return vg_fail_("the search: ", vi_error());
    if (ns == 0) return 0;
    for (i = 0; i < ns; ++i) ssc[i] = 1.0f / (float)(i + 1);     /* any metric's scores as weights: by rank */
    if (o.walk & VG_WALK_EXPAND) {
        if (o.expand.max_nodes < o.hits) o.expand.max_nodes = o.hits;
        if ((nw = vg_expand(g, sid, ssc, ns, &o.expand, wid, wsc, (int *)0, VG_CTX_MAX)) < 0) return -1;
    }
    if (o.walk & VG_WALK_PPR) {
        if (o.ppr.max_nodes < o.hits) o.ppr.max_nodes = o.hits;
        if ((np = vg_ppr(g, sid, ssc, ns, &o.ppr, pid, psc, VG_CTX_MAX)) < 0) return -1;
    }
    if (o.walk == VG_WALK_BOTH) {                                /* reciprocal rank fusion of the two lists */
        for (i = nf = 0; i < nw; ++i) { fz[nf].id = (i64)wid[i]; fz[nf].sc = 1.0f / (60.0f + (float)i); ++nf; }
        for (i = 0; i < np; ++i) {
            for (j = 0; j < nf && fz[j].id != (i64)pid[i]; ++j) {}
            if (j == nf) { fz[nf].id = (i64)pid[i]; fz[nf].sc = 0.0f; ++nf; }
            fz[j].sc += 1.0f / (60.0f + (float)i);
        }
        qsort(fz, (size_t)nf, sizeof(VgFz), vg_fz_cmp_);
        for (i = 0; i < nf && i < VG_CTX_MAX; ++i) { wid[i] = (vi_i64)fz[i].id; wsc[i] = fz[i].sc; }
        nw = nf < VG_CTX_MAX ? nf : VG_CTX_MAX;
    } else if (o.walk == VG_WALK_PPR) {
        for (i = 0; i < np; ++i) { wid[i] = pid[i]; wsc[i] = psc[i]; }
        nw = np;
    }
    if (nw > o.hits) nw = o.hits;
    r = vi_context_ids(db, wid, wsc, nw, &c, spans, max_spans, buf, cap, used);
    return r < 0 ? vg_fail_("the context: ", vi_error()) : r;
}
