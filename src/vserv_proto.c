/* vserv_proto.c - the three request protocols, told apart by the first byte of a request: binary frames (0xB5), a
 * line protocol (a lowercase letter) and HTTP/1.1 (anything else).
 * Part of vserv.c, which includes it at this point; not compiled on its own. */

/* ---- what the protocols share: the store operations, and the log lines for writes ---- */

/* The replace-or-add every protocol's add means (one row per id). While the connection has a batch open (the line
 * protocol's begin), adds and deletes from any protocol wait in it: logged when the batch commits, not one by one. */
static i32 vs_put_(VsConn *c, const f32 *v, i64 id)
{
    i32 r;
    if (c->wb) return store_batch_put(c->wb, v, id, 0);
    r = store_put(c->t->s, v, id, 0);
    if (vs_logging_()) vs_log_(c, r != 0, "add id=%ld %s", (long)id, r == 0 ? "ok" : "refused (NaN or Inf)");
    return r;
}

/* rows removed; 0 when the delete waits in a batch */
static i32 vs_del_(VsConn *c, i64 lo, i64 hi)
{
    i32 n;
    if (c->wb) { if (lo == hi) store_batch_delete(c->wb, lo); else store_batch_delete_range(c->wb, lo, hi); return 0; }
    n = lo == hi ? store_delete(c->t->s, lo) : store_delete_range(c->t->s, lo, hi);
    if (vs_logging_()) {
        if (lo == hi) vs_log_(c, 0, "del id=%ld removed=%d", (long)lo, n);
        else vs_log_(c, 0, "del lo=%ld hi=%ld removed=%d", (long)lo, (long)hi, n);
    }
    return n;
}

/* the line protocol's begin: a batch for this connection's later writes */
static const char *vs_begin_(VsConn *c)
{
    if (c->wb) return "a batch is open already: commit or abort it";
    if (!(c->wb = (StoreWb *)malloc(sizeof(StoreWb)))) return "out of memory";
    store_batch_init(c->wb, c->t->s);
    return (const char *)0;
}

/* commit (sync: on the disk before the reply) or, with apply 0, abort the open batch: as store_batch_commit, 0 for
 * an abort */
static i32 vs_end_(VsConn *c, i32 apply, i32 sync)
{
    i32 n = c->wb->n, r = 0;
    double t0 = vs_now_();
    if (apply) {
        r = store_batch_commit(c->wb, sync);
        vs_log_(c, r < 0, "commit changes=%d%s %s ms=%.0f", (int)n, sync ? " sync" : "",
                r >= 0 ? "ok" : (r == -1 ? "refused, nothing applied" : "applied, the fsync failed"), (vs_now_() - t0) * 1e3);
    } else vs_log_(c, 0, "abort changes=%d", (int)n);
    store_batch_free(c->wb); free(c->wb); c->wb = (StoreWb *)0;
    return r;
}

static void vs_build_(VsConn *c)
{
    double t0 = vs_now_();
    store_build(c->t->s);
    vs_log_(c, 0, "build ok ms=%.0f", (vs_now_() - t0) * 1e3);
}

static i32 vs_save_(VsConn *c)
{
    double t0 = vs_now_();
    i32 r = store_save(c->t->s, c->t->path);
    vs_log_(c, r != 0, "save %s ms=%.0f", r == 0 ? "ok" : "failed", (vs_now_() - t0) * 1e3);
    return r;
}

/* 0 started, 1 one runs already (or a build or save holds the store), -1 failed */
static i32 vs_snapshot_(VsConn *c)
{
    i32 r = store_snapshot(c->t->s, c->t->path);
    vs_log_(c, r < 0, "snapshot %s", r == 0 ? "started" : (r == 1 ? "busy" : "failed"));
    return r;
}

static void vs_logsearch_(VsConn *c, i32 k, i32 got, double t0)
{
    if (vs_log_all) vs_log_(c, 0, "search k=%d hits=%d us=%.0f", k, got, (vs_now_() - t0) * 1e6);
}

#include "vserv_text.c"

/* the text of id into the worker's buffer: its length (0: none), -1 no such row, -2 out of memory */
static i32 vs_get_text_(VsWorker *w, VsConn *c, i64 id, char **t)
{
    i32 n = store_get_text(c->t->s, id, (char *)0, 0);
    if (n <= 0) { *t = w->txt; return n; }
    if (!(*t = vs_txt_(w, n))) return -2;
    return store_get_text(c->t->s, id, *t, n) == n ? n : -1;   /* replaced in between: -1, ask again */
}

/* hybrid search into w->ids / w->sc: the vector when hv, the words t[0, tn) when tn > 0 */
static i32 vs_hybrid_(VsWorker *w, VsConn *c, i32 hv, const char *t, i32 tn, i32 k)
{
    return store_search_hybrid(c->t->s, hv ? w->vec : (const f32 *)0, t, tn, k, w->ids, w->sc);
}

/* grouped search (one hit an id >> shift group, shift 1-63) or, with rng, the rows whose number is in [lo, hi], of
 * the vector in w->vec, into w->ids / w->sc scored as vs_search_ */
static i32 vs_search_by_(VsWorker *w, VsConn *c, i32 k, i32 shift, i32 rng, i64 lo, i64 hi)
{
    i32 n = rng ? store_search_num(c->t->s, w->vec, k, lo, hi, w->ids, w->sc)
                : store_search_grouped(c->t->s, w->vec, k, shift, w->ids, w->sc), i;
    if (n < 0) n = 0;
    for (i = 0; i < n; ++i) w->sc[i] = store_score_out(c->t->s, w->sc[i]);
    return n;
}

/* a new vector for id, its tag, number and text kept (store_update); queued in an open batch. 0, or -1: no such row */
static i32 vs_update_(VsConn *c, const f32 *v, i64 id)
{
    i32 r;
    if (c->wb) return store_batch_update(c->wb, v, id);
    r = store_update(c->t->s, v, id);
    if (vs_logging_()) vs_log_(c, r != 0, "update id=%ld %s", (long)id, r == 0 ? "ok" : "refused (no such id, NaN or Inf)");
    return r;
}

/* id's number for range searches (store_set_num); queued in an open batch. 0, or -1: no such row */
static i32 vs_setnum_(VsConn *c, i64 id, i64 num)
{
    i32 r;
    if (c->wb) return store_batch_set_num(c->wb, id, num);
    r = store_set_num(c->t->s, id, num);
    if (vs_logging_()) vs_log_(c, r != 0, "num id=%ld num=%ld %s", (long)id, (long)num, r == 0 ? "ok" : "refused (no such id)");
    return r;
}

/* ---- binary protocol: little-endian frames, nothing to parse ----
 *   request  u8 0xB5 | u8 op | u16 k | u32 nbytes | payload[nbytes]
 *   response u8 0xB5 | u8 status | u16 count | u32 nbytes | payload[nbytes]
 *   status 0 ok, 1 refused or failed, 2 bad request, 3 unknown op, 4 busy, 5 no tenant chosen (a tenant server)
 *   op 1 search        payload: dim x f32            reply: count x { i64 id, f32 score }
 *   op 2 add           payload: i64 id, dim x f32    (replaces a row with the same id)
 *   op 3 delete        payload: i64 id               reply count: rows removed
 *   op 4 build   op 5 save   op 6 stats              stats reply: i64 live, dim, pending, built
 *   op 7 snapshot      (in the background; status 4 while one runs)
 *   op 8 get           payload: i64 id               reply: dim x f32, the stored unit vector (status 1: no row)
 *   op 9 delete range  payload: i64 lo, i64 hi       reply: i64 rows removed (every id in [lo, hi])
 *   op 10 add with a text  payload: i64 id, dim x f32, then the text's bytes (the rest of the payload)
 *   op 11 get the text     payload: i64 id           reply: the text's bytes (status 1: no row)
 *   op 12 range            payload: i64 lo, i64 hi   reply: count x i64, the ids in [lo, hi] ascending (k at most)
 *   op 13 hybrid search    payload: u32 1 (a vector follows) or 0, [dim x f32], then the words' bytes
 *                          reply: count x { i64 id, f32 fused score }
 *   op 14 use a tenant     payload: its name (a tenant server; opened or created) - the connection's later requests
 *                          go to its store. status 1 with the reason as the payload when refused
 *   op 15 grouped search   payload: u32 shift (1-63), dim x f32   reply: count x { i64 id, f32 score }, one hit an
 *                          id >> shift group (shift 20 and ids doc << 20 | chunk: the best chunk of k documents)
 *   op 16 number range     payload: i64 lo, i64 hi, dim x f32     reply: count x { i64 id, f32 score }, the rows
 *                          whose number is in [lo, hi]
 *   op 17 update           payload: i64 id, dim x f32   a new vector, the row's tag, number and text kept (status 1:
 *                          no row)
 *   op 18 set a number     payload: i64 id, i64 number  (status 1: no row)
 *   op 19 get a number     payload: i64 id              reply: i64 number (status 1: no row) */

/* a reply whose payload sits elsewhere (a text): the header, then the payload as it is */
static void vs_bin_reply_at_(VsWorker *w, VsConn *c, u8 status, u16 count, const char *payload, u32 nbytes)
{
    char h[8];
    h[0] = (char)VS_MAGIC; h[1] = (char)status; h[2] = (char)(count & 0xff); h[3] = (char)(count >> 8);
    h[4] = (char)(nbytes & 0xff); h[5] = (char)((nbytes >> 8) & 0xff); h[6] = (char)((nbytes >> 16) & 0xff); h[7] = (char)(nbytes >> 24);
    vs_send_(w, c, h, 8);
    if (nbytes) vs_send_(w, c, payload, (i32)nbytes);
}

static void vs_bin_reply_(VsWorker *w, VsConn *c, u8 status, u16 count, const char *payload, u32 nbytes)
{
    char *o = w->out;
    o[0] = (char)VS_MAGIC; o[1] = (char)status;
    o[2] = (char)(count & 0xff); o[3] = (char)(count >> 8);
    o[4] = (char)(nbytes & 0xff); o[5] = (char)((nbytes >> 8) & 0xff); o[6] = (char)((nbytes >> 16) & 0xff); o[7] = (char)(nbytes >> 24);
    if (nbytes && payload != o + 8) memmove(o + 8, payload, nbytes);
    vs_send_(w, c, o, (i32)(8 + nbytes));
}

static u32 vs_le32_(const u8 *p) { return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24); }
static i64 vs_le64_(const u8 *p) { return (i64)((u64)vs_le32_(p) | ((u64)vs_le32_(p + 4) << 32)); }
static void vs_put32_(char *o, u32 v) { o[0] = (char)(v & 0xff); o[1] = (char)((v >> 8) & 0xff); o[2] = (char)((v >> 16) & 0xff); o[3] = (char)(v >> 24); }
static void vs_put64_(char *o, i64 v) { vs_put32_(o, (u32)((u64)v & 0xffffffffu)); vs_put32_(o + 4, (u32)((u64)v >> 32)); }
static u32 vs_fbits_(f32 f) { u32 u; memcpy(&u, &f, 4); return u; }

/* n little-endian f32 from p into v (a copy, whatever the alignment and the host's byte order) */
static void vs_getf_(f32 *v, const u8 *p, i32 n)
{
    i32 i;
    if (!eng_host_be_()) { memcpy(v, p, (size_t)n * 4); return; }
    for (i = 0; i < n; ++i) { u32 u = vs_le32_(p + 4 * i); memcpy(&v[i], &u, 4); }
}

/* returns bytes consumed, 0 if the frame is incomplete, -1 to close */
static i32 vs_binary_(VsWorker *w, VsConn *c)
{
    const u8 *b = (const u8 *)c->buf;
    u32 nb;
    i32 op, k, i, got;
    if (c->len < 8) return 0;
    op = b[1]; k = (i32)(b[2] | (b[3] << 8)); nb = vs_le32_(b + 4);
    if (nb > (u32)vs_reqmax - 8) { vs_log_(c, 1, "binary frame of %lu bytes refused", (unsigned long)nb); return -1; }
    if ((u32)c->len < 8 + nb) return 0;
    b += 8;
    if (op == 14) {                                              /* use a tenant: the payload is its name */
        const char *why = nb <= VS_TNAME ? vs_use_(c, (const char *)b, (i32)nb) : "a tenant's name is 1 to 64 of A-Z a-z 0-9 _ -";
        if (why) vs_bin_reply_(w, c, 1, 0, why, (u32)strlen(why));
        else vs_bin_reply_(w, c, 0, 0, (const char *)0, 0);
        return (i32)(8 + nb);
    }
    if (!c->t) { vs_bin_reply_(w, c, 5, 0, (const char *)0, 0); return (i32)(8 + nb); }   /* a tenant server: none chosen */
    switch (op) {
    case 1: {
        double t0 = vs_now_();
        if (nb != (u32)vs_dim * 4) { vs_bin_reply_(w, c, 2, 0, (const char *)0, 0); break; }
        vs_getf_(w->vec, b, vs_dim);
        if (k < 1) k = 10;
        if (k > VS_MAXK) k = VS_MAXK;
        got = vs_search_(w, c, w->vec, k);
        {   char *o = w->out + 8;
            for (i = 0; i < got; ++i) { vs_put64_(o, w->ids[i]); vs_put32_(o + 8, vs_fbits_(w->sc[i])); o += 12; }
            vs_bin_reply_(w, c, 0, (u16)got, w->out + 8, (u32)got * 12u); }
        vs_logsearch_(c, k, got, t0);
        break; }
    case 2:
        if (nb != 8 + (u32)vs_dim * 4) { vs_bin_reply_(w, c, 2, 0, (const char *)0, 0); break; }
        vs_getf_(w->vec, b + 8, vs_dim);
        vs_bin_reply_(w, c, vs_put_(c, w->vec, vs_le64_(b)) == 0 ? 0 : 1, 0, (const char *)0, 0);
        break;
    case 3:
        if (nb != 8) { vs_bin_reply_(w, c, 2, 0, (const char *)0, 0); break; }
        {   i64 id = vs_le64_(b);
            vs_bin_reply_(w, c, 0, (u16)vs_del_(c, id, id), (const char *)0, 0); }
        break;
    case 4: vs_build_(c); vs_bin_reply_(w, c, 0, 0, (const char *)0, 0); break;
    case 5: vs_bin_reply_(w, c, (u8)(vs_save_(c) == 0 ? 0 : 1), 0, (const char *)0, 0); break;
    case 6: {
        i64 v[4]; char p[32];
        vs_stats_(c, v);
        for (i = 0; i < 4; ++i) vs_put64_(p + 8 * i, v[i]);
        vs_bin_reply_(w, c, 0, 4, p, sizeof(p));
        break; }
    case 7: { i32 r = vs_snapshot_(c); vs_bin_reply_(w, c, (u8)(r == 0 ? 0 : (r == 1 ? 4 : 1)), 0, (const char *)0, 0); } break;
    case 8:
        if (nb != 8) { vs_bin_reply_(w, c, 2, 0, (const char *)0, 0); break; }
        if (store_get_vec(c->t->s, vs_le64_(b), w->vec) != 0) { vs_bin_reply_(w, c, 1, 0, (const char *)0, 0); break; }
        {   char *o = w->out + 8;
            for (i = 0; i < vs_dim; ++i) vs_put32_(o + 4 * i, vs_fbits_(w->vec[i]));
            vs_bin_reply_(w, c, 0, 0, o, (u32)vs_dim * 4u); }
        break;
    case 9:
        if (nb != 16) { vs_bin_reply_(w, c, 2, 0, (const char *)0, 0); break; }
        {   char p[8];
            vs_put64_(p, (i64)vs_del_(c, vs_le64_(b), vs_le64_(b + 8)));
            vs_bin_reply_(w, c, 0, 0, p, 8); }
        break;
    case 10:
        if (nb < 8 + (u32)vs_dim * 4) { vs_bin_reply_(w, c, 2, 0, (const char *)0, 0); break; }
        vs_getf_(w->vec, b + 8, vs_dim);
        vs_bin_reply_(w, c, vs_put_text_(c, w->vec, vs_le64_(b), (const char *)b + 8 + vs_dim * 4, (i32)(nb - 8 - (u32)vs_dim * 4)) == 0 ? 0 : 1,
                      0, (const char *)0, 0);
        break;
    case 11:
        if (nb != 8) { vs_bin_reply_(w, c, 2, 0, (const char *)0, 0); break; }
        {   char *t;
            i32 n = vs_get_text_(w, c, vs_le64_(b), &t);
            if (n < 0) vs_bin_reply_(w, c, 1, 0, (const char *)0, 0);
            else vs_bin_reply_at_(w, c, 0, 0, t, (u32)n); }
        break;
    case 12:
        if (nb != 16) { vs_bin_reply_(w, c, 2, 0, (const char *)0, 0); break; }
        if (k < 1) k = 100;
        if (k > VS_MAXK) k = VS_MAXK;
        got = store_range(c->t->s, vs_le64_(b), vs_le64_(b + 8), w->ids, k);
        if (got < 0) { vs_bin_reply_(w, c, 1, 0, (const char *)0, 0); break; }
        {   char *o = w->out + 8;
            for (i = 0; i < got; ++i) vs_put64_(o + 8 * i, w->ids[i]);
            vs_bin_reply_(w, c, 0, (u16)got, o, (u32)got * 8u); }
        break;
    case 13: {
        u32 hv = nb >= 4 ? vs_le32_(b) : 2u, at = 4u + (hv == 1 ? (u32)vs_dim * 4u : 0u);
        double t0 = vs_now_();
        if (hv > 1 || nb < at) { vs_bin_reply_(w, c, 2, 0, (const char *)0, 0); break; }
        if (hv) vs_getf_(w->vec, b + 4, vs_dim);
        if (k < 1) k = 10;
        if (k > VS_MAXK) k = VS_MAXK;
        got = vs_hybrid_(w, c, (i32)hv, (const char *)b + at, (i32)(nb - at), k);
        {   char *o = w->out + 8;
            for (i = 0; i < got; ++i) { vs_put64_(o, w->ids[i]); vs_put32_(o + 8, vs_fbits_(w->sc[i])); o += 12; }
            vs_bin_reply_(w, c, 0, (u16)got, w->out + 8, (u32)got * 12u); }
        vs_logsearch_(c, k, got, t0);
        break; }
    case 15: case 16: {
        u32 at = op == 15 ? 4u : 16u;
        double t0 = vs_now_();
        if (nb != at + (u32)vs_dim * 4 || (op == 15 && (vs_le32_(b) < 1 || vs_le32_(b) > 63))) { vs_bin_reply_(w, c, 2, 0, (const char *)0, 0); break; }
        vs_getf_(w->vec, b + at, vs_dim);
        if (k < 1) k = 10;
        if (k > VS_MAXK) k = VS_MAXK;
        got = op == 15 ? vs_search_by_(w, c, k, (i32)vs_le32_(b), 0, 0, 0) : vs_search_by_(w, c, k, 0, 1, vs_le64_(b), vs_le64_(b + 8));
        {   char *o = w->out + 8;
            for (i = 0; i < got; ++i) { vs_put64_(o, w->ids[i]); vs_put32_(o + 8, vs_fbits_(w->sc[i])); o += 12; }
            vs_bin_reply_(w, c, 0, (u16)got, w->out + 8, (u32)got * 12u); }
        vs_logsearch_(c, k, got, t0);
        break; }
    case 17:
        if (nb != 8 + (u32)vs_dim * 4) { vs_bin_reply_(w, c, 2, 0, (const char *)0, 0); break; }
        vs_getf_(w->vec, b + 8, vs_dim);
        vs_bin_reply_(w, c, vs_update_(c, w->vec, vs_le64_(b)) == 0 ? 0 : 1, 0, (const char *)0, 0);
        break;
    case 18:
        if (nb != 16) { vs_bin_reply_(w, c, 2, 0, (const char *)0, 0); break; }
        vs_bin_reply_(w, c, vs_setnum_(c, vs_le64_(b), vs_le64_(b + 8)) == 0 ? 0 : 1, 0, (const char *)0, 0);
        break;
    case 19:
        if (nb != 8) { vs_bin_reply_(w, c, 2, 0, (const char *)0, 0); break; }
        {   i64 v = 0;
            char p[8];
            if (store_get_num(c->t->s, vs_le64_(b), &v) != 0) { vs_bin_reply_(w, c, 1, 0, (const char *)0, 0); break; }
            vs_put64_(p, v);
            vs_bin_reply_(w, c, 0, 0, p, 8); }
        break;
    default: vs_bin_reply_(w, c, 3, 0, (const char *)0, 0); break;
    }
    return (i32)(8 + nb);
}

/* ---- the line protocol: one line in, one line out ----
 * A request is words separated by spaces or tabs, ending in \n (or \r\n); the reply is one line, "ok ..." or
 * "err ...", so any language - or nc, socat, a shell - can speak it with a readline:
 *   ping                      ok pong
 *   use NAME                  ok                           (a tenant server: later lines go to NAME's store, opened or
 *                                                           created; until then only ping, help and quit answer)
 *   stats                     ok count=N dim=D pending=P built=0|1 snapshot=idle|running last=ok|failed|none
 *                                hot=BYTES cold=BYTES memory=0|1|2   (store_tiers; -m sets the memory tier)
 *   search K X1 .. XD         ok N ID SCORE ID SCORE ...   (N <= K hits, best first; SCORE = -|q - v|^2)
 *   add ID X1 .. XD           ok                           (replaces the row with the same id)
 *   get ID                    ok X1 .. XD                  (the stored unit vector)
 *   del ID                    ok REMOVED
 *   del LO HI                 ok REMOVED                   (every id in [LO, HI])
 *   begin                     ok                           (later adds and deletes, in any protocol on this
 *                                                           connection, wait: "ok queued"; searches do not see them)
 *   commit [sync]             ok CHANGES                   (all of them at once, journaled as one record; sync: on
 *                                                           the disk before the reply)
 *   abort                     ok 0                         (none of them; closing the connection does the same)
 *   build | save | snapshot   ok, or err busy (a snapshot already runs)
 *   addt ID X1 .. XD TEXT     ok                           (add with a text: the rest of the line, \n \r \t \0 \\
 *                                                          escaped; replaces the row with the same id)
 *   text ID                   ok LEN TEXT                  (LEN bytes, escaped the same way; LEN 0: no text)
 *   range LO HI [MAX]         ok N ID ...                  (the ids in [LO, HI], ascending, at most MAX = 100)
 *   hybrid K X1 .. XD WORDS   ok N ID SCORE ...            (the vector and the words against the texts, fused)
 *   words K WORDS             ok N ID SCORE ...            (the words alone)
 *   group K SHIFT X1 .. XD    ok N ID SCORE ...            (one hit an id >> SHIFT group, SHIFT 1-63: with ids
 *                                                          doc << 20 | chunk, group K 20 is the best chunk of K docs)
 *   between K LO HI X1 .. XD  ok N ID SCORE ...            (the rows whose number is in [LO, HI])
 *   num ID [N]                ok N | ok                    (a row's number for between: read it, or set it to N)
 *   update ID X1 .. XD        ok                           (a new vector, the row's tag, number and text kept)
 *   help                      ok COMMANDS...
 *   quit                      ok bye, and the server closes the connection */

static const char *vs_word_(const char *p) { while (*p == ' ' || *p == '\t') ++p; return p; }

/* an integer word; NULL when there is none */
static const char *vs_wint_(const char *p, i64 *v)
{
    char *e;
    i64 x;
    p = vs_word_(p);
    errno = 0;
    x = (i64)strtoll(p, &e, 10);                                 /* 64-bit ids on 32-bit CPUs too (long is 32 bits there) */
    if (e == p || errno || (*e && *e != ' ' && *e != '\t')) return (const char *)0;
    *v = x;
    return e;
}

/* vs_dim numbers into w->vec; just past the last one, or NULL */
static const char *vs_wnums_(VsWorker *w, const char *p)
{
    i32 d;
    for (d = 0; d < vs_dim; ++d) {
        char *e;
        p = vs_word_(p);
        w->vec[d] = (f32)strtod(p, &e);
        if (e == p || (*e && *e != ' ' && *e != '\t')) return (const char *)0;
        p = e;
    }
    return p;
}

/* exactly vs_dim numbers into w->vec, and nothing after them; 0 or -1 */
static i32 vs_wvec_(VsWorker *w, const char *p)
{
    return (p = vs_wnums_(w, p)) && !*vs_word_(p) ? 0 : -1;
}

/* a line's hits as "ok N ID SCORE ..." */
static void vs_line_hits_(VsWorker *w, VsConn *c, i32 got)
{
    char *o = w->out, *p = vs_str_(o, "ok ");
    i32 i;
    p = vs_i64_(p, got);
    for (i = 0; i < got; ++i) { *p++ = ' '; p = vs_i64_(p, w->ids[i]); *p++ = ' '; p = vs_f32_(p, w->sc[i]); }
    *p++ = '\n';
    vs_send_(w, c, o, (i32)(p - o));
}

static void vs_line_reply_(VsWorker *w, VsConn *c, char *o, char *p)
{
    *p++ = '\n';
    vs_send_(w, c, o, (i32)(p - o));
}

#define VS_LINE_ERR(msg) do { char *p_ = vs_str_(o, "err "); p_ = vs_str_(p_, msg); vs_line_reply_(w, c, o, p_); } while (0)

/* returns bytes consumed, 0 if the line is not complete yet, -1 to close */
static i32 vs_line_(VsWorker *w, VsConn *c)
{
    char *b = c->buf, *nl = (char *)memchr(b, '\n', (size_t)c->len), *o = w->out, *p;
    const char *a;
    i32 used, i, got;
    i64 x, y;
    if (!nl) {
        if (c->len >= vs_reqmax) { VS_LINE_ERR("line too long"); vs_log_(c, 1, "line too long, closed"); return -1; }
        return 0;
    }
    used = (i32)(nl - b) + 1;
    *nl = 0;
    if (nl > b && nl[-1] == '\r') nl[-1] = 0;
#define VS_CMD(word) (!strncmp(b, word, sizeof(word) - 1) && (b[sizeof(word) - 1] == 0 || b[sizeof(word) - 1] == ' ' || b[sizeof(word) - 1] == '\t'))
    a = b;
    while (*a && *a != ' ' && *a != '\t') ++a;                   /* a: just past the command */
    if (VS_CMD("use")) {                                         /* a tenant server: the store later lines go to */
        const char *e, *f, *why;
        while (*a == ' ' || *a == '\t') ++a;
        for (e = a; *e && *e != ' ' && *e != '\t'; ++e) {}
        for (f = e; *f == ' ' || *f == '\t'; ++f) {}
        why = e == a || *f ? "usage: use NAME" : vs_use_(c, a, (i32)(e - a));
        if (why) VS_LINE_ERR(why);
        else vs_line_reply_(w, c, o, vs_str_(o, "ok"));
    } else if (!c->t && !VS_CMD("ping") && !VS_CMD("help") && !VS_CMD("quit")) VS_LINE_ERR("no tenant: use NAME first");
    else if (VS_CMD("search")) {
        double t0 = vs_now_();
        if (!(a = vs_wint_(a, &x)) || vs_wvec_(w, a) != 0) VS_LINE_ERR("usage: search K X1 .. XD (D numbers)");
        else {
            i32 k = x < 1 ? 1 : (x > VS_MAXK ? VS_MAXK : (i32)x);
            got = vs_search_(w, c, w->vec, k);
            p = vs_str_(o, "ok "); p = vs_i64_(p, got);
            for (i = 0; i < got; ++i) { *p++ = ' '; p = vs_i64_(p, w->ids[i]); *p++ = ' '; p = vs_f32_(p, w->sc[i]); }
            vs_line_reply_(w, c, o, p);
            vs_logsearch_(c, k, got, t0);
        }
    } else if (VS_CMD("add")) {
        if (!(a = vs_wint_(a, &x)) || vs_wvec_(w, a) != 0) VS_LINE_ERR("usage: add ID X1 .. XD (D numbers)");
        else if (vs_put_(c, w->vec, x) != 0) VS_LINE_ERR(c->wb ? "vector has NaN or Inf: the batch will be refused, abort it" : "vector has NaN or Inf");
        else vs_line_reply_(w, c, o, vs_str_(o, c->wb ? "ok queued" : "ok"));
    } else if (VS_CMD("get")) {
        if (!(a = vs_wint_(a, &x)) || *vs_word_(a)) VS_LINE_ERR("usage: get ID");
        else if (store_get_vec(c->t->s, x, w->vec) != 0) VS_LINE_ERR("no such id");
        else {
            p = vs_str_(o, "ok");
            for (i = 0; i < vs_dim; ++i) { *p++ = ' '; p = vs_f32_(p, w->vec[i]); }
            vs_line_reply_(w, c, o, p);
        }
    } else if (VS_CMD("del") || VS_CMD("delete")) {
        if (!(a = vs_wint_(a, &x))) VS_LINE_ERR("usage: del ID | del LO HI");
        else {
            y = x;
            if (*vs_word_(a) && (!(a = vs_wint_(a, &y)) || *vs_word_(a) || y < x)) VS_LINE_ERR("usage: del ID | del LO HI");
            else if (c->wb) { vs_del_(c, x, y); vs_line_reply_(w, c, o, vs_str_(o, "ok queued")); }
            else { p = vs_str_(o, "ok "); p = vs_i64_(p, vs_del_(c, x, y)); vs_line_reply_(w, c, o, p); }
        }
    } else if (VS_CMD("stats")) {
        i64 v[4], us;
        i32 run, last;
        vs_stats_(c, v);
        store_snapshot_info(c->t->s, &run, &last, &us);
        p = vs_str_(o, "ok count="); p = vs_i64_(p, v[0]); p = vs_str_(p, " dim="); p = vs_i64_(p, v[1]);
        p = vs_str_(p, " pending="); p = vs_i64_(p, v[2]); p = vs_str_(p, " built="); p = vs_i64_(p, v[3]);
        p = vs_str_(p, run ? " snapshot=running" : " snapshot=idle");
        p = vs_str_(p, last == 0 ? " last=ok" : (last == -1 ? " last=failed" : " last=none"));
        {   i64 hot, cold;
            store_tiers(c->t->s, &hot, &cold);
            p = vs_str_(p, " hot="); p = vs_i64_(p, hot); p = vs_str_(p, " cold="); p = vs_i64_(p, cold);
            p = vs_str_(p, " memory="); p = vs_i64_(p, (i64)c->t->s->mem_tier); }
        {   i64 nb, bad, fx, rb, rh;
            i32 jd;
            store_integrity(c->t->s, &nb, &bad, &fx, &rb, &rh, &jd);
            p = vs_str_(p, " damaged="); p = vs_i64_(p, bad); p = vs_str_(p, " rebuilt="); p = vs_i64_(p, fx);
            p = vs_str_(p, " rows_bad="); p = vs_i64_(p, rb); p = vs_str_(p, " rows_healed="); p = vs_i64_(p, rh);
            p = vs_str_(p, jd ? " journal=damaged" : " journal=ok"); }
        {   i64 ld, lt;
            store_lens_info(c->t->s, &ld, &lt);
            if (lt) { p = vs_str_(p, " lens="); p = vs_i64_(p, ld); *p++ = '/'; p = vs_i64_(p, lt); } }
        vs_line_reply_(w, c, o, p);
    } else if (VS_CMD("build")) { vs_build_(c); vs_line_reply_(w, c, o, vs_str_(o, "ok")); }
    else if (VS_CMD("save")) { if (vs_save_(c) == 0) vs_line_reply_(w, c, o, vs_str_(o, "ok")); else VS_LINE_ERR("save failed"); }
    else if (VS_CMD("snapshot")) {
        i32 r = vs_snapshot_(c);
        if (r == 0) vs_line_reply_(w, c, o, vs_str_(o, "ok started"));
        else VS_LINE_ERR(r == 1 ? "busy" : "snapshot failed");
    } else if (VS_CMD("scrub")) {                             /* every block of the file checked, damage rebuilt */
        i64 fx = 0, r;
        double t0 = vs_now_();
        r = store_scrub(c->t->path, 1, &fx);
        vs_log_(c, r != 0, "scrub file=%s damaged=%.0f rebuilt=%.0f ms=%.0f", c->t->path, (double)r, (double)fx, (vs_now_() - t0) * 1e3);
        if (r < 0) VS_LINE_ERR(r == -2 ? "this file has no checksums (written before v9)" : "scrub failed");
        else { p = vs_str_(o, "ok damaged="); p = vs_i64_(p, r); p = vs_str_(p, " rebuilt="); p = vs_i64_(p, fx); vs_line_reply_(w, c, o, p); }
    } else if (VS_CMD("begin")) {
        const char *why = *vs_word_(a) ? "usage: begin" : vs_begin_(c);
        if (why) VS_LINE_ERR(why); else vs_line_reply_(w, c, o, vs_str_(o, "ok"));
    } else if (VS_CMD("commit") || VS_CMD("abort")) {
        const char *s = vs_word_(a);
        i32 sync = !strncmp(s, "sync", 4) && (s[4] == 0 || s[4] == ' ' || s[4] == '\t'), r;
        if (!c->wb) VS_LINE_ERR("no batch: begin first");
        else if (VS_CMD("abort") ? *s != 0 : (*s && !sync)) VS_LINE_ERR("usage: commit [sync] | abort");
        else if ((r = vs_end_(c, VS_CMD("commit"), sync)) >= 0) { p = vs_str_(o, "ok "); p = vs_i64_(p, r); vs_line_reply_(w, c, o, p); }
        else VS_LINE_ERR(r == -1 ? "refused, nothing applied (a NaN or Inf, a bad text or no memory in the batch)" : "applied, but the fsync failed");
    } else if (VS_CMD("addt")) {
        const char *t;
        char *buf;
        i32 n;
        if (!(a = vs_wint_(a, &x)) || !(t = vs_wnums_(w, a)) || (*t != ' ' && *t != '\t')) VS_LINE_ERR("usage: addt ID X1 .. XD TEXT");
        else if (!(buf = vs_txt_(w, (i32)(nl - t))) || (n = vs_unesc_(t + 1, nl, buf)) < 0) VS_LINE_ERR("bad escape in the text (\\n \\r \\t \\0 \\\\)");
        else if (vs_put_text_(c, w->vec, x, buf, n) != 0) VS_LINE_ERR(c->wb ? "vector has NaN or Inf, or the text is too long: the batch will be refused, abort it" : "vector has NaN or Inf, or the text is too long");
        else vs_line_reply_(w, c, o, vs_str_(o, c->wb ? "ok queued" : "ok"));
    } else if (VS_CMD("text")) {
        char *t, *r;
        i32 n;
        if (!(a = vs_wint_(a, &x)) || *vs_word_(a)) VS_LINE_ERR("usage: text ID");
        else if ((n = vs_get_text_(w, c, x, &t)) < 0) VS_LINE_ERR(n == -1 ? "no such id" : "out of memory");
        else if (!(r = (char *)malloc((size_t)n * 2 + 32))) VS_LINE_ERR("out of memory");
        else {
            p = vs_str_(r, "ok "); p = vs_i64_(p, n); *p++ = ' ';
            p = vs_esc_(p, t, n); *p++ = '\n';
            vs_send_(w, c, r, (i32)(p - r));
            free(r);
        }
    } else if (VS_CMD("range")) {
        i64 mx = 100;
        if (!(a = vs_wint_(a, &x)) || !(a = vs_wint_(a, &y)) || y < x || (*vs_word_(a) && (!(a = vs_wint_(a, &mx)) || *vs_word_(a) || mx < 1)))
            VS_LINE_ERR("usage: range LO HI [MAX]");
        else if ((got = store_range(c->t->s, x, y, w->ids, (i32)(mx > VS_MAXK ? VS_MAXK : mx))) < 0) VS_LINE_ERR("out of memory");
        else {
            p = vs_str_(o, "ok "); p = vs_i64_(p, got);
            for (i = 0; i < got; ++i) { *p++ = ' '; p = vs_i64_(p, w->ids[i]); }
            vs_line_reply_(w, c, o, p);
        }
    } else if (VS_CMD("hybrid") || VS_CMD("words")) {
        i32 hv = b[0] == 'h', k;
        const char *t = (const char *)0;
        double t0 = vs_now_();
        if (!(a = vs_wint_(a, &x)) || (hv && !(a = vs_wnums_(w, a)))) VS_LINE_ERR(hv ? "usage: hybrid K X1 .. XD WORDS" : "usage: words K WORDS");
        else {
            t = vs_word_(a);
            k = x < 1 ? 1 : (x > VS_MAXK ? VS_MAXK : (i32)x);
            got = vs_hybrid_(w, c, hv, t, (i32)(nl - t), k);
            vs_line_hits_(w, c, got);
            vs_logsearch_(c, k, got, t0);
        }
    } else if (VS_CMD("group") || VS_CMD("between")) {
        i32 gr = b[0] == 'g', kk;
        i64 z = 0;
        double t0 = vs_now_();
        if (!(a = vs_wint_(a, &x)) || !(a = vs_wint_(a, &y)) || (!gr && !(a = vs_wint_(a, &z))) || vs_wvec_(w, a) != 0
            || (gr && (y < 1 || y > 63)) || (!gr && z < y))
            VS_LINE_ERR(gr ? "usage: group K SHIFT X1 .. XD (SHIFT 1 to 63)" : "usage: between K LO HI X1 .. XD (LO <= HI)");
        else {
            kk = x < 1 ? 1 : (x > VS_MAXK ? VS_MAXK : (i32)x);
            got = gr ? vs_search_by_(w, c, kk, (i32)y, 0, 0, 0) : vs_search_by_(w, c, kk, 0, 1, y, z);
            vs_line_hits_(w, c, got);
            vs_logsearch_(c, kk, got, t0);
        }
    } else if (VS_CMD("num")) {
        if (!(a = vs_wint_(a, &x))) VS_LINE_ERR("usage: num ID [N]");
        else if (!*vs_word_(a)) {
            if (store_get_num(c->t->s, x, &y) != 0) VS_LINE_ERR("no such id");
            else { p = vs_str_(o, "ok "); p = vs_i64_(p, y); vs_line_reply_(w, c, o, p); }
        } else if (!(a = vs_wint_(a, &y)) || *vs_word_(a)) VS_LINE_ERR("usage: num ID [N]");
        else if (vs_setnum_(c, x, y) != 0) VS_LINE_ERR(c->wb ? "out of memory, or the batch is over 1 GB" : "no such id");
        else vs_line_reply_(w, c, o, vs_str_(o, c->wb ? "ok queued" : "ok"));
    } else if (VS_CMD("update")) {
        if (!(a = vs_wint_(a, &x)) || vs_wvec_(w, a) != 0) VS_LINE_ERR("usage: update ID X1 .. XD (D numbers)");
        else if (vs_update_(c, w->vec, x) != 0) VS_LINE_ERR(c->wb ? "vector has NaN or Inf: the batch will be refused, abort it" : "no such id, or the vector has NaN or Inf");
        else vs_line_reply_(w, c, o, vs_str_(o, c->wb ? "ok queued" : "ok"));
    } else if (VS_CMD("ping")) vs_line_reply_(w, c, o, vs_str_(o, "ok pong"));
    else if (VS_CMD("help")) vs_line_reply_(w, c, o, vs_str_(o, "ok ping use stats search add addt get text range hybrid words group between num update del begin commit abort build save snapshot help quit"));
    else if (VS_CMD("quit")) { vs_line_reply_(w, c, o, vs_str_(o, "ok bye")); c->closing = 1; }
    else { VS_LINE_ERR("unknown command (help lists them)"); vs_log_(c, 1, "unknown line command"); }
#undef VS_CMD
    return used;
}
#undef VS_LINE_ERR

/* ---- HTTP/1.1 ---- */

/* offset just past "\r\n\r\n" in p[0..n), or -1. SSE2: 16 bytes per step looking for '\r' (the
 * picohttpparser idea, reduced to what this server needs); scalar check of each hit. */
static i32 vs_head_end_(const char *p, i32 n)
{
    i32 i = 0;
#if defined(__SSE2__)
    const __m128i cr = _mm_set1_epi8('\r');
    for (; i + 16 <= n; i += 16) {
        i32 m = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)(const void *)(p + i)), cr));
        while (m) {
            i32 j = i + tq_ctz_((u32)m);
            m &= m - 1;
            if (j + 3 < n && p[j + 1] == '\n' && p[j + 2] == '\r' && p[j + 3] == '\n') return j + 4;
        }
    }
    if (i >= 3) i -= 3;                          /* a terminator may straddle the last full block */
#endif
    for (; i + 3 < n; ++i)
        if (p[i] == '\r' && p[i + 1] == '\n' && p[i + 2] == '\r' && p[i + 3] == '\n') return i + 4;
    return -1;
}

static i32 vs_ieq_(const char *a, const char *b, i32 n)
{
    i32 i;
    for (i = 0; i < n; ++i) {
        char x = a[i], y = b[i];
        if (x >= 'A' && x <= 'Z') x = (char)(x + 32);
        if (x != y) return 0;
    }
    return 1;
}

/* value of query parameter name= in the request target [t, t+tn), or def when it is not there. *bad becomes 1 when it
 * is there but not a decimal 64-bit integer - empty, a sign alone, anything but & or # after the digits, or beyond 64
 * bits (tests/fuzz_proto.c found 20 digits wrapping around into another row's id: /delete?id= removed that row) */
static i64 vs_qparam_(const char *t, i32 tn, const char *name, i64 def, i32 *bad)
{
    i32 nl = (i32)strlen(name), i;
    for (i = 0; i < tn; ++i) {
        if ((t[i] == '?' || t[i] == '&') && i + 1 + nl < tn && !memcmp(t + i + 1, name, (size_t)nl) && t[i + 1 + nl] == '=') {
            u64 v = 0, lim;
            i32 j = i + 2 + nl, neg = 0, nd = 0;
            if (j < tn && t[j] == '-') { neg = 1; ++j; }
            lim = neg ? (u64)1 << 63 : ((u64)1 << 63) - 1;
            for (; j < tn && t[j] >= '0' && t[j] <= '9'; ++j, ++nd) {
                u64 d = (u64)(t[j] - '0');
                if (v > (lim - d) / 10) { *bad = 1; return def; }
                v = v * 10 + d;
            }
            if (!nd || (j < tn && t[j] != '&' && t[j] != '#')) { *bad = 1; return def; }
            if (!neg || v == 0) return (i64)v;
            return -(i64)(v - 1) - 1;                            /* -2^63 too, without overflow */
        }
    }
    return def;
}

/* JSON array of exactly vs_dim numbers -> w->vec; 0 ok, -1 bad */
static i32 vs_json_vec_(VsWorker *w, const char *p, i32 n)
{
    const char *e = p + n;
    char num[64];
    i32 d = 0;
    while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
    if (p >= e || *p != '[') return -1;
    ++p;
    for (;;) {
        i32 k = 0;
        while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
        if (p < e && *p == ']' && d == 0) break;
        while (p < e && k < 63 && ((*p >= '0' && *p <= '9') || *p == '-' || *p == '+' || *p == '.' || *p == 'e' || *p == 'E')) num[k++] = *p++;
        if (k == 0 || d >= vs_dim) return -1;
        num[k] = 0;
        w->vec[d++] = (f32)strtod(num, (char **)0);
        while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
        if (p < e && *p == ',') { ++p; continue; }
        if (p < e && *p == ']') break;
        return -1;
    }
    return d == vs_dim ? 0 : -1;
}

static void vs_http_reply_(VsWorker *w, VsConn *c, const char *status, const char *body, i32 blen, i32 keep);
#define VS_HTTP_LIT(w, c, status, lit, keep) vs_http_reply_(w, c, status, lit, (i32)sizeof(lit) - 1, keep)   /* a literal body */

static void vs_http_reply_(VsWorker *w, VsConn *c, const char *status, const char *body, i32 blen, i32 keep)
{
    char head[256], *p = head;
    i32 hn;
    p = vs_str_(p, "HTTP/1.1 "); p = vs_str_(p, status);
    p = vs_str_(p, "\r\nContent-Type: application/json\r\nContent-Length: ");
    p = vs_i64_(p, blen);
    p = vs_str_(p, keep ? "\r\n\r\n" : "\r\nConnection: close\r\n\r\n");
    hn = (i32)(p - head);
    if (body == w->out + 256) {                  /* body already sits after a 256-byte gap: prepend in place */
        memcpy(w->out + 256 - hn, head, (size_t)hn);
        vs_send_(w, c, w->out + 256 - hn, hn + blen);
    } else {
        vs_send_(w, c, head, hn);
        if (blen) vs_send_(w, c, body, blen);
    }
    if (!keep) c->closing = 1;
}

/* returns bytes consumed, 0 if incomplete, -1 to close */
static i32 vs_http_(VsWorker *w, VsConn *c)
{
    const char *b = c->buf, *line, *t, *body;
    i32 he = vs_head_end_(b, c->len), i, mlen, tn, clen = 0, keep = 1, raw = 0, got, k, qbad = 0;
    char *o;
    if (he < 0) {
        if (c->len >= vs_reqcap) { VS_HTTP_LIT(w, c, "431 Request Header Fields Too Large", "{\"error\":\"headers too large\"}", 0); return -1; }
        return 0;
    }
    /* request line: METHOD SP TARGET SP HTTP/1.x */
    for (mlen = 0; mlen < he && b[mlen] != ' '; ++mlen) {}
    t = b + mlen + 1;
    for (tn = 0; mlen + 1 + tn < he && t[tn] != ' ' && t[tn] != '\r'; ++tn) {}
    if (mlen >= he || tn == 0) { VS_HTTP_LIT(w, c, "400 Bad Request", "{\"error\":\"bad request line\"}", 0); return -1; }
    if (mlen + 1 + tn + 9 <= he && !memcmp(t + tn, " HTTP/1.0", 9)) keep = 0;
    /* headers */
    for (line = b; ; ) {
        const char *nl = (const char *)memchr(line, '\n', (size_t)(b + he - line));
        if (!nl) break;
        line = nl + 1;
        if (line >= b + he - 2) break;
        if (b + he - line > 15 && vs_ieq_(line, "content-length:", 15)) {
            const char *v = line + 15;
            while (*v == ' ') ++v;
            for (clen = 0; *v >= '0' && *v <= '9'; ++v) { clen = clen * 10 + (*v - '0'); if (clen > vs_reqmax) break; }
        } else if (b + he - line > 13 && vs_ieq_(line, "content-type:", 13)) {
            const char *v = line + 13;
            while (*v == ' ') ++v;
            if (b + he - v > 24 && vs_ieq_(v, "application/octet-stream", 24)) raw = 1;
        } else if (b + he - line > 11 && vs_ieq_(line, "connection:", 11)) {
            const char *v = line + 11;
            while (*v == ' ') ++v;
            if (b + he - v >= 5 && vs_ieq_(v, "close", 5)) keep = 0;
            if (b + he - v >= 10 && vs_ieq_(v, "keep-alive", 10)) keep = 1;
        }
    }
    if (clen > vs_reqmax - he) {
        VS_HTTP_LIT(w, c, "413 Content Too Large", "{\"error\":\"body too large\"}", 0);
        vs_log_(c, 1, "http body of %d bytes refused", clen);
        return -1;
    }
    if (c->len < he + clen) return 0;
    body = b + he;
    o = w->out + 256;                             /* bodies are written here; vs_http_reply_ prepends headers */

#define VS_IS(m, path) (mlen == (i32)strlen(m) && !memcmp(b, m, (size_t)mlen) && tn >= (i32)strlen(path) \
                        && !memcmp(t, path, strlen(path)) && (tn == (i32)strlen(path) || t[strlen(path)] == '?'))
#define VS_BADQ() VS_HTTP_LIT(w, c, "400 Bad Request", "{\"error\":\"a query parameter is not a 64-bit integer\"}", keep)
    if (vs_dir && !VS_IS("GET", "/health")) {                    /* a tenant server: /t/NAME/search?k=10 and so on */
        const char *why = "no tenant: the path starts /t/NAME/";
        i32 e = 3;
        if (tn > 3 && !memcmp(t, "/t/", 3)) {
            while (e < tn && t[e] != '/' && t[e] != '?') ++e;
            why = vs_use_(c, t + 3, e - 3);
        }
        if (why) {
            char *p = vs_str_(o, "{\"error\":\""); p = vs_str_(p, why); p = vs_str_(p, "\"}");
            vs_http_reply_(w, c, e == 3 ? "400 Bad Request" : "403 Forbidden", o, (i32)(p - o), keep);
            return he + clen;
        }
        t += e; tn -= e;                                         /* the rest routes as on a server of one store */
    }
    if (VS_IS("GET", "/health")) {
        VS_HTTP_LIT(w, c, "200 OK", "{\"ok\":true}", keep);
    } else if (VS_IS("GET", "/stats")) {
        i64 v[4], us, hot, cold; i32 run, last; char *p = o;
        vs_stats_(c, v);
        store_snapshot_info(c->t->s, &run, &last, &us);
        store_tiers(c->t->s, &hot, &cold);
        p = vs_str_(p, "{\"count\":"); p = vs_i64_(p, v[0]); p = vs_str_(p, ",\"dim\":"); p = vs_i64_(p, v[1]);
        p = vs_str_(p, ",\"pending\":"); p = vs_i64_(p, v[2]); p = vs_str_(p, ",\"built\":"); p = vs_str_(p, v[3] ? "true" : "false");
        p = vs_str_(p, ",\"snapshot\":"); p = vs_str_(p, run ? "\"running\"" : "\"idle\"");
        p = vs_str_(p, ",\"last_snapshot\":"); p = vs_str_(p, last == 0 ? "\"ok\"" : (last == -1 ? "\"failed\"" : "\"none\""));
        p = vs_str_(p, ",\"last_snapshot_ms\":"); p = vs_i64_(p, us / 1000);
        p = vs_str_(p, ",\"hot\":"); p = vs_i64_(p, hot); p = vs_str_(p, ",\"cold\":"); p = vs_i64_(p, cold);
        p = vs_str_(p, ",\"memory\":"); p = vs_i64_(p, (i64)c->t->s->mem_tier); *p++ = '}';
        vs_http_reply_(w, c, "200 OK", o, (i32)(p - o), keep);
    } else if (VS_IS("POST", "/search")) {
        char *p = o;
        double t0 = vs_now_();
        i64 kq = vs_qparam_(t, tn, "k", 10, &qbad), gs = vs_qparam_(t, tn, "group", 0, &qbad);
        i64 lo = vs_qparam_(t, tn, "lo", (i64)((u64)1 << 63), &qbad), hi = vs_qparam_(t, tn, "hi", (i64)(((u64)1 << 63) - 1), &qbad);
        i32 rng = vs_qparam_(t, tn, "lo", 0, &qbad) == vs_qparam_(t, tn, "lo", 1, &qbad)        /* lo= or hi= there */
               || vs_qparam_(t, tn, "hi", 0, &qbad) == vs_qparam_(t, tn, "hi", 1, &qbad);
        if (qbad) VS_BADQ();
        else if (gs < 0 || gs > 63 || (gs && rng) || lo > hi)
            VS_HTTP_LIT(w, c, "400 Bad Request", "{\"error\":\"group is 1 to 63, lo <= hi, and not group with lo/hi\"}", keep);
        else if (raw ? clen != vs_dim * 4 : vs_json_vec_(w, body, clen) != 0) {
            VS_HTTP_LIT(w, c, "400 Bad Request", "{\"error\":\"expected a vector of the store's dimension\"}", keep);
        } else {
            if (raw) vs_getf_(w->vec, (const u8 *)body, vs_dim);
            k = kq < 1 ? 1 : (kq > VS_MAXK ? VS_MAXK : (i32)kq);
            got = gs || rng ? vs_search_by_(w, c, k, (i32)gs, rng, lo, hi) : vs_search_(w, c, w->vec, k);
            p = vs_str_(p, "{\"ids\":[");
            for (i = 0; i < got; ++i) { if (i) *p++ = ','; p = vs_i64_(p, w->ids[i]); }
            p = vs_str_(p, "],\"scores\":[");
            for (i = 0; i < got; ++i) { if (i) *p++ = ','; p = vs_f32_(p, w->sc[i]); }
            p = vs_str_(p, "]}");
            vs_http_reply_(w, c, "200 OK", o, (i32)(p - o), keep);
            vs_logsearch_(c, k, got, t0);
        }
    } else if (VS_IS("POST", "/add")) {
        i64 id = vs_qparam_(t, tn, "id", -1, &qbad);
        i32 hv = 1, txl = -1;
        const char *tx = w->txt;
        if (raw) { if (clen > vs_dim * 4) { tx = body + vs_dim * 4; txl = clen - vs_dim * 4; } }   /* f32[dim], then a text */
        else if (vs_json_vec_(w, body, clen) != 0 && (vs_json_obj_(w, body, clen, &hv, &txl) != 0 || !hv)) hv = 0;
        tx = raw ? tx : w->txt;
        if (qbad) VS_BADQ();
        else if (id < 0 || !hv || (raw && clen < vs_dim * 4))
            VS_HTTP_LIT(w, c, "400 Bad Request", "{\"error\":\"need ?id=N and a vector of the store's dimension\"}", keep);
        else {
            if (raw) vs_getf_(w->vec, (const u8 *)body, vs_dim);
            if ((txl > 0 ? vs_put_text_(c, w->vec, id, tx, txl) : vs_put_(c, w->vec, id)) == 0) VS_HTTP_LIT(w, c, "200 OK", "{\"ok\":true}", keep);
            else VS_HTTP_LIT(w, c, "400 Bad Request", "{\"error\":\"vector has NaN or Inf, or the text is too long\"}", keep);
        }
    } else if (VS_IS("GET", "/text")) {
        char *tx, *r, *p;
        i64 id = vs_qparam_(t, tn, "id", -1, &qbad);
        i32 n = qbad ? -1 : vs_get_text_(w, c, id, &tx);
        if (qbad) VS_BADQ();
        else if (n == -1) VS_HTTP_LIT(w, c, "404 Not Found", "{\"error\":\"no such id\"}", keep);
        else if (n < 0 || !(r = (char *)malloc((size_t)n * 6 + 16))) VS_HTTP_LIT(w, c, "500 Internal Server Error", "{\"error\":\"out of memory\"}", keep);
        else {
            p = vs_str_(r, "{\"text\":"); p = vs_jesc_(p, tx, n); *p++ = '}';
            vs_http_reply_(w, c, "200 OK", r, (i32)(p - r), keep);
            free(r);
        }
    } else if (VS_IS("GET", "/range")) {
        char *p = o;
        i64 mx = vs_qparam_(t, tn, "max", 100, &qbad), lo = vs_qparam_(t, tn, "lo", 0, &qbad), hi = vs_qparam_(t, tn, "hi", -1, &qbad);
        got = qbad ? 0 : store_range(c->t->s, lo, hi, w->ids, (i32)(mx < 1 ? 1 : (mx > VS_MAXK ? VS_MAXK : mx)));
        if (qbad) VS_BADQ();
        else if (got < 0) VS_HTTP_LIT(w, c, "500 Internal Server Error", "{\"error\":\"out of memory\"}", keep);
        else {
            p = vs_str_(p, "{\"ids\":[");
            for (i = 0; i < got; ++i) { if (i) *p++ = ','; p = vs_i64_(p, w->ids[i]); }
            p = vs_str_(p, "]}");
            vs_http_reply_(w, c, "200 OK", o, (i32)(p - o), keep);
        }
    } else if (VS_IS("POST", "/hybrid")) {
        char *p = o;
        i32 hv, txl;
        double t0 = vs_now_();
        i64 kq = vs_qparam_(t, tn, "k", 10, &qbad);
        if (qbad) VS_BADQ();
        else if (vs_json_obj_(w, body, clen, &hv, &txl) != 0 || (!hv && txl <= 0))
            VS_HTTP_LIT(w, c, "400 Bad Request", "{\"error\":\"expected {\\\"vector\\\": [...], \\\"text\\\": \\\"...\\\"}\"}", keep);
        else {
            k = kq < 1 ? 1 : (kq > VS_MAXK ? VS_MAXK : (i32)kq);
            got = vs_hybrid_(w, c, hv, w->txt, txl > 0 ? txl : 0, k);
            p = vs_str_(p, "{\"ids\":[");
            for (i = 0; i < got; ++i) { if (i) *p++ = ','; p = vs_i64_(p, w->ids[i]); }
            p = vs_str_(p, "],\"scores\":[");
            for (i = 0; i < got; ++i) { if (i) *p++ = ','; p = vs_f32_(p, w->sc[i]); }
            p = vs_str_(p, "]}");
            vs_http_reply_(w, c, "200 OK", o, (i32)(p - o), keep);
            vs_logsearch_(c, k, got, t0);
        }
    } else if (VS_IS("GET", "/get")) {
        char *p = o;
        i64 id = vs_qparam_(t, tn, "id", -1, &qbad);
        if (qbad) VS_BADQ();
        else if (store_get_vec(c->t->s, id, w->vec) != 0)
            VS_HTTP_LIT(w, c, "404 Not Found", "{\"error\":\"no such id\"}", keep);
        else {
            p = vs_str_(p, "{\"vector\":[");
            for (i = 0; i < vs_dim; ++i) { if (i) *p++ = ','; p = vs_f32_(p, w->vec[i]); }
            p = vs_str_(p, "]}");
            vs_http_reply_(w, c, "200 OK", o, (i32)(p - o), keep);
        }
    } else if (VS_IS("POST", "/update")) {
        i64 id = vs_qparam_(t, tn, "id", -1, &qbad);
        if (qbad) VS_BADQ();
        else if (raw ? clen != vs_dim * 4 : vs_json_vec_(w, body, clen) != 0)
            VS_HTTP_LIT(w, c, "400 Bad Request", "{\"error\":\"need ?id=N and a vector of the store's dimension\"}", keep);
        else {
            if (raw) vs_getf_(w->vec, (const u8 *)body, vs_dim);
            if (vs_update_(c, w->vec, id) == 0) VS_HTTP_LIT(w, c, "200 OK", "{\"ok\":true}", keep);
            else VS_HTTP_LIT(w, c, "404 Not Found", "{\"error\":\"no such id, or the vector has NaN or Inf\"}", keep);
        }
    } else if (VS_IS("GET", "/num") || VS_IS("POST", "/num")) {
        char *p = o;
        i64 id = vs_qparam_(t, tn, "id", -1, &qbad), nv = vs_qparam_(t, tn, "num", 0, &qbad), v = 0;
        i32 has = vs_qparam_(t, tn, "num", 0, &qbad) == vs_qparam_(t, tn, "num", 1, &qbad);
        if (qbad) VS_BADQ();
        else if (b[0] == 'P' && !has) VS_HTTP_LIT(w, c, "400 Bad Request", "{\"error\":\"need ?id=N&num=V\"}", keep);
        else if (b[0] == 'P') {
            if (vs_setnum_(c, id, nv) == 0) VS_HTTP_LIT(w, c, "200 OK", "{\"ok\":true}", keep);
            else VS_HTTP_LIT(w, c, "404 Not Found", "{\"error\":\"no such id\"}", keep);
        } else if (store_get_num(c->t->s, id, &v) != 0) VS_HTTP_LIT(w, c, "404 Not Found", "{\"error\":\"no such id\"}", keep);
        else { p = vs_str_(p, "{\"num\":"); p = vs_i64_(p, v); *p++ = '}'; vs_http_reply_(w, c, "200 OK", o, (i32)(p - o), keep); }
    } else if (VS_IS("POST", "/delete")) {
        char *p = o;
        i64 lo = vs_qparam_(t, tn, "lo", -1, &qbad), hi = vs_qparam_(t, tn, "hi", -1, &qbad), id = vs_qparam_(t, tn, "id", -1, &qbad);
        if (qbad) VS_BADQ();
        else {
            p = vs_str_(p, "{\"removed\":");
            p = vs_i64_(p, lo >= 0 && hi >= lo ? vs_del_(c, lo, hi) : vs_del_(c, id, id));
            *p++ = '}';
            vs_http_reply_(w, c, "200 OK", o, (i32)(p - o), keep);
        }
    } else if (VS_IS("POST", "/build")) {
        vs_build_(c);
        VS_HTTP_LIT(w, c, "200 OK", "{\"ok\":true}", keep);
    } else if (VS_IS("POST", "/save")) {
        if (vs_save_(c) == 0) VS_HTTP_LIT(w, c, "200 OK", "{\"ok\":true}", keep);
        else VS_HTTP_LIT(w, c, "500 Internal Server Error", "{\"error\":\"save failed\"}", keep);
    } else if (VS_IS("POST", "/snapshot")) {
        i32 r = vs_snapshot_(c);
        if (r == 0) VS_HTTP_LIT(w, c, "202 Accepted", "{\"ok\":true,\"snapshot\":\"started\"}", keep);
        else if (r == 1) VS_HTTP_LIT(w, c, "409 Conflict", "{\"error\":\"busy\"}", keep);
        else VS_HTTP_LIT(w, c, "500 Internal Server Error", "{\"error\":\"snapshot failed\"}", keep);
    } else {
        VS_HTTP_LIT(w, c, "404 Not Found", "{\"error\":\"not found\"}", keep);
    }
#undef VS_BADQ
#undef VS_IS
    return he + clen;
}
