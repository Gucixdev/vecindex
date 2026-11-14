/* vecdb_embed.c - the embedding server: a child process that keeps a model loaded, and a pool of them.
 * Part of vecdb.c, which includes it at this point; not compiled on its own. */

/* ------------------------------------------------------------------ */
/* embedding server: a child process that keeps a model loaded.        */
/* Protocol (native endian): on start the child writes u32 dim; then   */
/* per request the parent writes  u8 kind ('D' document, 'Q' query,    */
/* 'I' image file path)  u32 len  bytes  and reads dim f32 (unit).     */
/* See py/embedsrv.py for the ready-made one (multilingual-e5, CLIP).  */
/* ------------------------------------------------------------------ */

typedef struct { pid_t pid; int to, from; i32 dim; } VEmbedProc;

VEmbedProc *vecdb_embproc_open(const char *cmd, const char *arg)
{
    int a[2], b[2];
    VEmbedProc *p;
    u32 dim = 0;
    if (pipe(a) != 0) return (VEmbedProc *)0;
    if (pipe(b) != 0) { close(a[0]); close(a[1]); return (VEmbedProc *)0; }
    p = (VEmbedProc *)calloc(1, sizeof(*p));
    p->pid = fork();
    if (p->pid < 0) { free(p); return (VEmbedProc *)0; }
    if (p->pid == 0) {
        dup2(a[0], 0); dup2(b[1], 1);
        close(a[0]); close(a[1]); close(b[0]); close(b[1]);
        if (arg) execl("/bin/sh", "sh", "-c", cmd, "sh", arg, (char *)0);
        else execl("/bin/sh", "sh", "-c", cmd, (char *)0);
        _exit(127);
    }
    close(a[0]); close(b[1]);
    p->to = a[1]; p->from = b[0];
    if (read(p->from, &dim, 4) != 4 || dim == 0 || dim > 8192) { close(p->to); close(p->from); kill(p->pid, SIGTERM); waitpid(p->pid, (int *)0, 0); free(p); return (VEmbedProc *)0; }
    p->dim = (i32)dim;
    return p;
}

static i32 vdb_rd_(int fd, void *buf, size_t n)
{
    size_t got = 0;
    while (got < n) { ssize_t r = read(fd, (char *)buf + got, n - got); if (r <= 0) return -1; got += (size_t)r; }
    return 0;
}

/* status byte, then (status 0) n vectors; a failed request returns -1 and leaves out untouched */
static i32 vdb_reply_(VEmbedProc *p, f32 *out, i32 n)
{
    u8 st;
    if (vdb_rd_(p->from, &st, 1) != 0) return -1;
    if (st != 0) return -1;
    return vdb_rd_(p->from, out, (size_t)n * (size_t)p->dim * sizeof(f32));
}

/* out[dim]; returns 0, or -1 if the server failed or is gone */
i32 vecdb_embproc_ask(VEmbedProc *p, char kind, const char *bytes, i32 n, f32 *out)
{
    u32 len = (u32)n;
    size_t off = 0;
    char hdr[5];
    hdr[0] = kind; memcpy(hdr + 1, &len, 4);
    if (write(p->to, hdr, 5) != 5) return -1;
    while (off < (size_t)n) { ssize_t w = write(p->to, bytes + off, (size_t)n - off); if (w <= 0) return -1; off += (size_t)w; }
    return vdb_reply_(p, out, 1);
}

/* n texts in one round trip (kind 'B' documents / 'C' queries); out is n x dim; 0 on success */
i32 vecdb_embproc_ask_batch(VEmbedProc *p, char kind, i32 n, const char **texts, const i32 *lens, f32 *out)
{
    size_t total = 4, off = 0;
    i32 i;
    char *buf, hdr[5];
    u32 len, cnt = (u32)n;
    for (i = 0; i < n; ++i) total += 4 + (size_t)lens[i];
    buf = (char *)malloc(total);
    memcpy(buf, &cnt, 4); off = 4;
    for (i = 0; i < n; ++i) { u32 l = (u32)lens[i]; memcpy(buf + off, &l, 4); off += 4; memcpy(buf + off, texts[i], (size_t)lens[i]); off += (size_t)lens[i]; }
    len = (u32)total; hdr[0] = kind; memcpy(hdr + 1, &len, 4);
    if (write(p->to, hdr, 5) != 5) { free(buf); return -1; }
    for (off = 0; off < total; ) { ssize_t w = write(p->to, buf + off, total - off); if (w <= 0) { free(buf); return -1; } off += (size_t)w; }
    free(buf);
    return vdb_reply_(p, out, n);
}

void vecdb_embproc_close(VEmbedProc *p)
{
    if (!p) return;
    close(p->to); close(p->from);
    waitpid(p->pid, (int *)0, 0);
    free(p);
}

/* Several servers at once (e.g. one on the GPU, one on the CPU): a batch is cut into groups of 32 and every
 * server pulls the next group as soon as it is free, so a fast server naturally does more of the work.
 * Single texts (queries) go to the first server. Spec: "cmd1|cmd2|..." */
typedef struct { VEmbedProc *p[8]; i32 n, dim; } VEmbedPool;

VEmbedPool *vecdb_embpool_open(const char *spec)
{
    VEmbedPool *pl = (VEmbedPool *)calloc(1, sizeof(*pl));
    char *s = (char *)malloc(strlen(spec) + 1), *t, *save;
    strcpy(s, spec);
    for (t = s; t; t = save) {
        VEmbedProc *p;
        save = strchr(t, '|');
        if (save) *save++ = 0;
        if (!*t || pl->n >= 8) continue;
        p = vecdb_embproc_open(t, (const char *)0);
        if (!p) { fprintf(stderr, "vecdb: embedding server failed to start: %s\n", t); continue; }
        if (pl->n && p->dim != pl->dim) { fprintf(stderr, "vecdb: server dimension mismatch, skipped: %s\n", t); vecdb_embproc_close(p); continue; }
        pl->p[pl->n++] = p; pl->dim = p->dim;
    }
    free(s);
    if (pl->n == 0) { free(pl); return (VEmbedPool *)0; }
    return pl;
}

void vecdb_embpool_close(VEmbedPool *pl)
{
    i32 i;
    if (!pl) return;
    for (i = 0; i < pl->n; ++i) vecdb_embproc_close(pl->p[i]);
    free(pl);
}

typedef struct {
    VEmbedProc *p; i32 kind, n, dim; const char **texts; const i32 *lens; f32 *out;
    i32 *next; pthread_mutex_t *mu; i32 done_groups;
} VPoolJob;

static void *vdb_pool_worker_(void *arg)
{
    VPoolJob *j = (VPoolJob *)arg;
    for (;;) {
        i32 s, m;
        pthread_mutex_lock(j->mu);
        s = *j->next; *j->next += 32;
        pthread_mutex_unlock(j->mu);
        if (s >= j->n) break;
        m = j->n - s < 32 ? j->n - s : 32;
        if (vecdb_embproc_ask_batch(j->p, j->kind ? 'C' : 'B', m, j->texts + s, j->lens + s, j->out + (size_t)s * j->dim) != 0) {
            i32 i;                                                   /* the batch failed: try the texts one by one, a text that still fails becomes a zero vector */
            for (i = 0; i < m; ++i) {
                f32 *o = j->out + (size_t)(s + i) * j->dim;
                if (vecdb_embproc_ask(j->p, j->kind ? 'Q' : 'D', j->texts[s + i], j->lens[s + i], o) != 0) { i32 z; for (z = 0; z < j->dim; ++z) o[z] = 0.0f; }
            }
        }
        ++j->done_groups;
    }
    return (void *)0;
}

static void vdb_pool_embed_(void *ctx, const char *t, i32 n, i32 kind, f32 *out, i32 dim)
{
    VEmbedPool *pl = (VEmbedPool *)ctx;
    if (pl->dim != dim || vecdb_embproc_ask(pl->p[0], kind ? 'Q' : 'D', t, n, out) != 0) { i32 i; for (i = 0; i < dim; ++i) out[i] = 0.0f; }
}

static void vdb_pool_batch_(void *ctx, i32 n, const char **texts, const i32 *lens, i32 kind, f32 *out, i32 dim)
{
    VEmbedPool *pl = (VEmbedPool *)ctx;
    VPoolJob jobs[8];
    pthread_t th[8];
    pthread_mutex_t mu;
    i32 next = 0, i;
    if (pl->dim != dim) { for (i = 0; i < n * dim; ++i) out[i] = 0.0f; return; }
    pthread_mutex_init(&mu, (const pthread_mutexattr_t *)0);
    for (i = 0; i < pl->n; ++i) {
        jobs[i].p = pl->p[i]; jobs[i].kind = kind; jobs[i].n = n; jobs[i].dim = dim; jobs[i].texts = texts; jobs[i].lens = lens;
        jobs[i].out = out; jobs[i].next = &next; jobs[i].mu = &mu; jobs[i].done_groups = 0;
    }
    for (i = 1; i < pl->n; ++i) pthread_create(&th[i], (const pthread_attr_t *)0, vdb_pool_worker_, &jobs[i]);
    vdb_pool_worker_(&jobs[0]);
    for (i = 1; i < pl->n; ++i) pthread_join(th[i], (void **)0);
    pthread_mutex_destroy(&mu);
    if (getenv("VECDB_POOL_STATS")) { for (i = 0; i < pl->n; ++i) fprintf(stderr, "  server %d did %d groups\n", i, jobs[i].done_groups); }
}

/* use the server(s) for every embedding of this database (documents and queries) */
void vecdb_use_embpool(VecDB *db, VEmbedPool *pl) { db->embed = vdb_pool_embed_; db->embed_ctx = pl; db->embed_batch = vdb_pool_batch_; }

