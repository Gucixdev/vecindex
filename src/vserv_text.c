/* vserv_text.c - texts on the wire: the escapes of the line protocol, JSON strings and the request object
 * {"vector": [...], "text": "..."}, and the store calls with texts every protocol shares.
 * Part of vserv.c (through vserv_proto.c, which includes it at this point); not compiled on its own. */

static i32 vs_json_vec_(VsWorker *w, const char *p, i32 n);

/* the worker's text buffer, at least need bytes (texts are at most STORE_TXMAX); NULL when memory runs out */
static char *vs_txt_(VsWorker *w, i32 need)
{
    if (need > w->txtcap) {
        i32 nc = need < 65536 ? 65536 : need;
        char *t = (char *)realloc(w->txt, (size_t)nc);
        if (!t) return (char *)0;
        w->txt = t; w->txtcap = nc;
    }
    return w->txt;
}

/* upsert with a text (every protocol's add with a text) */
static i32 vs_put_ns_(VsConn *c, const f32 *v, i64 id, const char *t, i32 n);
static i32 vs_in_(const VsConn *c, i64 *id);
static i32 vs_key_room_(const VsConn *c, i64 id);
static i32 vs_put_text_(VsConn *c, const f32 *v, i64 id, const char *t, i32 n)
{
    i32 r;
    if (vs_in_(c, &id)) return -2;                               /* a tenant's id: 0 to 2^44 - 1 */
    if (!vs_key_room_(c, id)) return -3;                         /* the key's row limit */
    if (c->wb) {                                                 /* waits in the open batch, its namespace with it */
        r = store_batch_put_text(c->wb, v, id, 0, t, n);
        return r || !c->tn ? r : store_batch_set_ns(c->wb, id, c->tn);
    }
    r = c->tn ? vs_put_ns_(c, v, id, t, n) : store_put_text(c->t->s, v, id, 0, t, n);
    if (vs_logging_()) vs_log_(c, r != 0, "add id=%ld text=%d %s", (long)id, n, r == 0 ? "ok" : "refused (NaN, Inf or too long)");
    return r;
}

/* ---- the line protocol's escapes: \n \r \t \0 \\ (a text is the rest of its line) ---- */

/* decodes [p, e) into out (room for e - p bytes is enough); the length, or -1 at a bad escape */
static i32 vs_unesc_(const char *p, const char *e, char *out)
{
    char *o = out;
    while (p < e) {
        char ch = *p++;
        if (ch != '\\') { *o++ = ch; continue; }
        if (p >= e) return -1;
        ch = *p++;
        if (ch == 'n') *o++ = '\n';
        else if (ch == 'r') *o++ = '\r';
        else if (ch == 't') *o++ = '\t';
        else if (ch == '0') *o++ = 0;
        else if (ch == '\\') *o++ = '\\';
        else return -1;
    }
    return (i32)(o - out);
}

/* n bytes of t escaped onto o (needs 2 n bytes); returns the end */
static char *vs_esc_(char *o, const char *t, i32 n)
{
    i32 i;
    for (i = 0; i < n; ++i) {
        char ch = t[i];
        if (ch == '\n') { *o++ = '\\'; *o++ = 'n'; }
        else if (ch == '\r') { *o++ = '\\'; *o++ = 'r'; }
        else if (ch == '\t') { *o++ = '\\'; *o++ = 't'; }
        else if (ch == 0) { *o++ = '\\'; *o++ = '0'; }
        else if (ch == '\\') { *o++ = '\\'; *o++ = '\\'; }
        else *o++ = ch;
    }
    return o;
}

/* ---- JSON ---- */

static const char *vs_jws_(const char *p, const char *e) { while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p; return p; }

static i32 vs_hex4_(const char *p, u32 *v)
{
    i32 i;
    *v = 0;
    for (i = 0; i < 4; ++i) {
        char ch = p[i];
        u32 d = ch >= '0' && ch <= '9' ? (u32)(ch - '0') : ch >= 'a' && ch <= 'f' ? (u32)(ch - 'a' + 10) : ch >= 'A' && ch <= 'F' ? (u32)(ch - 'A' + 10) : 99u;
        if (d > 15) return -1;
        *v = *v << 4 | d;
    }
    return 0;
}

/* the JSON string starting at p (its opening quote) into out (room for e - p bytes is enough: no escape grows);
 * *end: just past the closing quote. The length, or -1 when it is not a well-formed string. */
static i32 vs_jstr_(const char *p, const char *e, char *out, const char **end)
{
    char *o = out;
    if (p >= e || *p != '"') return -1;
    for (++p; p < e && *p != '"'; ) {
        u32 u, lo;
        char ch = *p++;
        if ((u8)ch < 0x20) return -1;
        if (ch != '\\') { *o++ = ch; continue; }
        if (p >= e) return -1;
        ch = *p++;
        switch (ch) {
        case '"': *o++ = '"'; continue;
        case '\\': *o++ = '\\'; continue;
        case '/': *o++ = '/'; continue;
        case 'b': *o++ = '\b'; continue;
        case 'f': *o++ = '\f'; continue;
        case 'n': *o++ = '\n'; continue;
        case 'r': *o++ = '\r'; continue;
        case 't': *o++ = '\t'; continue;
        case 'u': break;
        default: return -1;
        }
        if (e - p < 4 || vs_hex4_(p, &u)) return -1;
        p += 4;
        if (u >= 0xd800 && u < 0xdc00) {                         /* a surrogate pair */
            if (e - p < 6 || p[0] != '\\' || p[1] != 'u' || vs_hex4_(p + 2, &lo) || lo < 0xdc00 || lo > 0xdfff) return -1;
            p += 6;
            u = 0x10000 + ((u - 0xd800) << 10) + (lo - 0xdc00);
        }
        if (u < 0x80) *o++ = (char)u;                            /* UTF-8: never longer than the escape it came from */
        else if (u < 0x800) { *o++ = (char)(0xc0 | (u >> 6)); *o++ = (char)(0x80 | (u & 0x3f)); }
        else if (u < 0x10000) { *o++ = (char)(0xe0 | (u >> 12)); *o++ = (char)(0x80 | ((u >> 6) & 0x3f)); *o++ = (char)(0x80 | (u & 0x3f)); }
        else { *o++ = (char)(0xf0 | (u >> 18)); *o++ = (char)(0x80 | ((u >> 12) & 0x3f)); *o++ = (char)(0x80 | ((u >> 6) & 0x3f)); *o++ = (char)(0x80 | (u & 0x3f)); }
    }
    if (p >= e) return -1;
    *end = p + 1;
    return (i32)(o - out);
}

/* n bytes of t as a JSON string (with its quotes) onto o (needs 6 n + 2 bytes); bytes from 0x80 go as they are */
static char *vs_jesc_(char *o, const char *t, i32 n)
{
    static const char hex[] = "0123456789abcdef";
    i32 i;
    *o++ = '"';
    for (i = 0; i < n; ++i) {
        u8 ch = (u8)t[i];
        if (ch == '"' || ch == '\\') { *o++ = '\\'; *o++ = (char)ch; }
        else if (ch == '\n') { *o++ = '\\'; *o++ = 'n'; }
        else if (ch == '\r') { *o++ = '\\'; *o++ = 'r'; }
        else if (ch == '\t') { *o++ = '\\'; *o++ = 't'; }
        else if (ch < 0x20) { *o++ = '\\'; *o++ = 'u'; *o++ = '0'; *o++ = '0'; *o++ = hex[ch >> 4]; *o++ = hex[ch & 15]; }
        else *o++ = (char)ch;
    }
    *o++ = '"';
    return o;
}

/* skips one JSON value of any kind at p; just past it, or NULL */
static const char *vs_jskip_(const char *p, const char *e)
{
    i32 depth = 0;
    p = vs_jws_(p, e);
    if (p < e && *p == '"') {
        for (++p; p < e && *p != '"'; ++p) if (*p == '\\') ++p;
        return p < e ? p + 1 : (const char *)0;
    }
    for (; p < e; ++p) {
        if (*p == '"') { for (++p; p < e && *p != '"'; ++p) if (*p == '\\') ++p; if (p >= e) return (const char *)0; continue; }
        if (*p == '[' || *p == '{') ++depth;
        else if (*p == ']' || *p == '}') { if (depth == 0) return p; if (--depth == 0) return p + 1; }
        else if (*p == ',' && depth == 0) return p;
    }
    return depth ? (const char *)0 : p;
}

/* {"vector": [...], "text": "..."} with either key missing and any other key skipped: *hv 1 when the vector came
 * (into w->vec, exactly vs_dim numbers), the text into the worker's buffer (*tn bytes; -1 when absent). 0 or -1. */
static i32 vs_json_obj_(VsWorker *w, const char *p, i32 n, i32 *hv, i32 *tn)
{
    const char *e = p + n;
    *hv = 0; *tn = -1;
    p = vs_jws_(p, e);
    if (p >= e || *p != '{') return -1;
    p = vs_jws_(p + 1, e);
    if (p < e && *p == '}') return 0;
    for (;;) {
        char key[16];
        const char *after;
        i32 kl;
        if (p >= e || *p != '"') return -1;
        for (kl = 0, ++p; p < e && *p != '"'; ++p, ++kl) {     /* keys of interest have no escapes */
            if (*p == '\\') return -1;
            if (kl < 15) key[kl] = *p;
        }
        if (p >= e) return -1;
        key[kl < 15 ? kl : 0] = 0;                               /* a long key matches nothing: it is skipped */
        p = vs_jws_(p + 1, e);
        if (p >= e || *p != ':') return -1;
        p = vs_jws_(p + 1, e);
        if (!strcmp(key, "vector")) {
            const char *q = p;
            while (q < e && *q != ']') ++q;
            if (q >= e || vs_json_vec_(w, p, (i32)(q - p) + 1) != 0) return -1;
            *hv = 1;
            after = q + 1;
        } else if (!strcmp(key, "text")) {
            char *t = vs_txt_(w, (i32)(e - p));
            if (!t || (*tn = vs_jstr_(p, e, t, &after)) < 0) return -1;
        } else if (!(after = vs_jskip_(p, e))) return -1;
        p = vs_jws_(after, e);
        if (p < e && *p == ',') { p = vs_jws_(p + 1, e); continue; }
        if (p < e && *p == '}') return 0;
        return -1;
    }
}
