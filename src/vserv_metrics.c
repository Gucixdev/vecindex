/* vserv_metrics.c - the server's numbers in two shapes, two names: GET /metrics answers JSON (for a script, a
 * dashboard, curl | jq), GET /prometheus the Prometheus text format (for a Prometheus scrape). The same numbers in
 * both: requests, refusals, search latency in buckets, and for every open store its rows, what is not indexed yet,
 * memory, damage found and rebuilt, snapshots and the lens. Only for the operator (no -K, or the * key, or the Unix
 * socket): a tenant's key gets 403 - the totals are other tenants' business.
 * Part of vserv.c (through vserv_proto.c, which includes it at this point); not compiled on its own. */

static const f64 vs_m_le[] = { 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 1.0 };   /* seconds */
#define VS_M_NB 10
static u64 vs_m_search, vs_m_write, vs_m_refused, vs_m_us, vs_m_bucket[VS_M_NB + 1];
static f64 vs_m_t0;                                  /* when the server started serving */

#if defined(__GNUC__)
#define VS_M_ADD(x, v) __atomic_fetch_add(&(x), (u64)(v), __ATOMIC_RELAXED)
#define VS_M_GET(x) __atomic_load_n(&(x), __ATOMIC_RELAXED)
#else
#define VS_M_ADD(x, v) ((x) += (u64)(v))
#define VS_M_GET(x) (x)
#endif

static void vs_m_search_(f64 seconds)
{
    i32 b;
    for (b = 0; b < VS_M_NB && seconds > vs_m_le[b]; ++b) {}
    VS_M_ADD(vs_m_search, 1);
    VS_M_ADD(vs_m_us, (u64)(seconds * 1e6));
    VS_M_ADD(vs_m_bucket[b], 1);
}

/* a growing text, for a reply of any length */
typedef struct { char *p; size_t n, cap; i32 bad; } VsSb;
static void vs_sb_(VsSb *b, const char *s)
{
    size_t k = strlen(s);
    if (b->bad) return;
    if (b->n + k + 1 > b->cap) {
        size_t c = b->cap ? b->cap * 2 : 4096;
        char *np;
        while (c < b->n + k + 1) c *= 2;
        if (!(np = (char *)realloc(b->p, c))) { b->bad = 1; return; }
        b->p = np; b->cap = c;
    }
    memcpy(b->p + b->n, s, k + 1);
    b->n += k;
}
static void vs_sbi_(VsSb *b, i64 v) { char t[24]; *vs_i64_(t, v) = 0; vs_sb_(b, t); }
static void vs_sbf_(VsSb *b, f64 v) { char t[40]; sprintf(t, "%.6g", v); vs_sb_(b, t); }

static const char *vs_m_nm[] = { "rows", "rows_pending", "index_built", "hot_bytes", "cold_bytes", "blocks", "blocks_damaged",
                                 "blocks_rebuilt", "rows_bad", "rows_healed", "journal_damaged", "snapshot_running",
                                 "snapshots_done", "lens_checked", "lens_blocks" };
static const char *vs_m_help[] = { "Rows.", "Rows not in the index yet.", "1: the index covers every row.",
                                   "Bytes a search scans (RAM).", "Bytes read only for candidates (disk).",
                                   "Checked blocks of the store file.", "Blocks found damaged.", "Blocks rebuilt from the parity.",
                                   "Rows that failed their check when read.", "Rows rebuilt on the spot.",
                                   "1: a damaged journal record stopped the replay.", "1: a snapshot is being written.",
                                   "Snapshots finished.", "Blocks the lens has checked.", "Blocks the lens walks." };
#define VS_M_NS 15

/* one store's numbers, in vs_m_nm's order */
static void vs_m_vals_(const VsTen *t, i64 *v)
{
    i64 hot, cold, us, ld, lt;
    i32 run, last, jd;
    v[0] = store_count(t->s); v[1] = store_pending(t->s); v[2] = store_built(t->s);
    store_tiers(t->s, &hot, &cold); v[3] = hot; v[4] = cold;
    store_integrity(t->s, &v[5], &v[6], &v[7], &v[8], &v[9], &jd); v[10] = jd;
    store_snapshot_info(t->s, &run, &last, &us); v[11] = run; v[12] = store_snapshots_done(t->s);
    store_lens_info(t->s, &ld, &lt); v[13] = ld; v[14] = lt;
}

/* the open stores: one, or the tenants opened so far (under vs_ten_mu) */
static const VsTen *vs_m_first_(void) { return vs_dir ? vs_tens : &vs_one; }
static const VsTen *vs_m_next_(const VsTen *t) { return vs_dir ? t->next : (const VsTen *)0; }
static const char *vs_m_name_(const VsTen *t) { return vs_dir ? t->name : "default"; }

/* the whole reply into b: prom 1 the Prometheus text format, 0 JSON */
static void vs_metrics_(VsSb *b, i32 prom)
{
    u64 n = VS_M_GET(vs_m_search), cum = 0;
    const VsTen *t;
    i32 i, first = 1;
    f64 up = vs_now_() - vs_m_t0;
    if (prom) {
        vs_sb_(b, "# HELP vserv_uptime_seconds Seconds since the server started serving.\n# TYPE vserv_uptime_seconds gauge\nvserv_uptime_seconds ");
        vs_sbf_(b, up);
        vs_sb_(b, "\n# HELP vserv_writes_total Adds, updates, deletes and numbers set.\n# TYPE vserv_writes_total counter\nvserv_writes_total ");
        vs_sbi_(b, (i64)VS_M_GET(vs_m_write));
        vs_sb_(b, "\n# HELP vserv_refused_total Requests refused for a key: unknown, or over its limit.\n# TYPE vserv_refused_total counter\nvserv_refused_total ");
        vs_sbi_(b, (i64)VS_M_GET(vs_m_refused));
        vs_sb_(b, "\n# HELP vserv_search_seconds Time a search took in the server.\n# TYPE vserv_search_seconds histogram\n");
        for (i = 0; i <= VS_M_NB; ++i) {
            cum += VS_M_GET(vs_m_bucket[i]);
            vs_sb_(b, "vserv_search_seconds_bucket{le=\"");
            if (i < VS_M_NB) vs_sbf_(b, vs_m_le[i]); else vs_sb_(b, "+Inf");
            vs_sb_(b, "\"} "); vs_sbi_(b, (i64)cum); vs_sb_(b, "\n");
        }
        vs_sb_(b, "vserv_search_seconds_sum "); vs_sbf_(b, (f64)VS_M_GET(vs_m_us) / 1e6);
        vs_sb_(b, "\nvserv_search_seconds_count "); vs_sbi_(b, (i64)n); vs_sb_(b, "\n");
    } else {
        vs_sb_(b, "{\"uptime_seconds\":"); vs_sbf_(b, up);
        vs_sb_(b, ",\"searches\":"); vs_sbi_(b, (i64)n);
        vs_sb_(b, ",\"writes\":"); vs_sbi_(b, (i64)VS_M_GET(vs_m_write));
        vs_sb_(b, ",\"refused\":"); vs_sbi_(b, (i64)VS_M_GET(vs_m_refused));
        vs_sb_(b, ",\"search_seconds\":{\"sum\":"); vs_sbf_(b, (f64)VS_M_GET(vs_m_us) / 1e6); vs_sb_(b, ",\"le\":{");
        for (i = 0; i <= VS_M_NB; ++i) {
            cum += VS_M_GET(vs_m_bucket[i]);
            if (i) vs_sb_(b, ",");
            vs_sb_(b, "\"");
            if (i < VS_M_NB) vs_sbf_(b, vs_m_le[i]); else vs_sb_(b, "+Inf");
            vs_sb_(b, "\":"); vs_sbi_(b, (i64)cum);
        }
        vs_sb_(b, "}},\"stores\":[");
    }
    pthread_mutex_lock(&vs_ten_mu);
    if (prom) {                                      /* a family's samples together: one name, every store */
        i64 v[VS_M_NS];
        i32 m;
        for (m = 0; m < VS_M_NS; ++m) {
            vs_sb_(b, "# HELP vserv_"); vs_sb_(b, vs_m_nm[m]); vs_sb_(b, " "); vs_sb_(b, vs_m_help[m]);
            vs_sb_(b, "\n# TYPE vserv_"); vs_sb_(b, vs_m_nm[m]); vs_sb_(b, " gauge\n");
            for (t = vs_m_first_(); t; t = vs_m_next_(t)) {
                vs_m_vals_(t, v);
                vs_sb_(b, "vserv_"); vs_sb_(b, vs_m_nm[m]); vs_sb_(b, "{store=\""); vs_sb_(b, vs_m_name_(t)); vs_sb_(b, "\"} ");
                vs_sbi_(b, v[m]); vs_sb_(b, "\n");
            }
        }
    } else {
        i64 v[VS_M_NS];
        i32 m;
        for (t = vs_m_first_(); t; t = vs_m_next_(t)) {
            vs_m_vals_(t, v);
            if (!first) vs_sb_(b, ",");
            first = 0;
            vs_sb_(b, "{\"store\":\""); vs_sb_(b, vs_m_name_(t)); vs_sb_(b, "\"");
            for (m = 0; m < VS_M_NS; ++m) { vs_sb_(b, ",\""); vs_sb_(b, vs_m_nm[m]); vs_sb_(b, "\":"); vs_sbi_(b, v[m]); }
            vs_sb_(b, "}");
        }
        vs_sb_(b, "]}");
    }
    pthread_mutex_unlock(&vs_ten_mu);
}
