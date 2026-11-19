/* store_meta.c - payload: named, typed fields a row (an int, a float or a string each), and filters over them -
 * comparisons, sets, presence, and / or / not - compiled once to a small postfix program and tested inside the scan,
 * candidate by candidate, like labels and numbers. No recursion (the parser is shunting-yard, the program runs on a
 * 64-bit stack of truth values) and no malloc (a compiled filter is a fixed-size value the caller holds).
 *
 * A field is a column of u64 a row, encoded so that 0 is "not set" and the unsigned order of the codes is the order
 * of the values: an int with its sign bit flipped (INT64_MIN is the one int that cannot be stored), a float by the
 * usual order-preserving map of its bits (NaN refused, -0 as 0), a string by its key (vi_key's, 0 taken as 1) - so a
 * string field takes = != in, not < or >, and gives back its key, not its bytes (keep the bytes in the row's text).
 * A field once declared stays, at the same index (what compiled filters refer to); at most STORE_FIELDS a store.
 * Part of store.c, included before the conditions of a search (StoreCnf); not compiled on its own. */

#define STORE_ENC_SIGN ((u64)1 << 63)

static void b3_hash(const void *in, size_t n, u8 out[32]);      /* blake3.c */

static u64 store_enc_int_(i64 v) { return (u64)v ^ STORE_ENC_SIGN; }
static i64 store_dec_int_(u64 c) { return (i64)(c ^ STORE_ENC_SIGN); }
static u64 store_enc_f64_(f64 v)
{
    u64 b;
    if (v == 0.0) v = 0.0;                                       /* -0 is 0 */
    memcpy(&b, &v, 8);
    return (b & STORE_ENC_SIGN) ? ~b : b | STORE_ENC_SIGN;
}
static f64 store_dec_f64_(u64 c)
{
    u64 b = (c & STORE_ENC_SIGN) ? c & ~STORE_ENC_SIGN : ~c;
    f64 v;
    memcpy(&v, &b, 8);
    return v;
}
static u64 store_enc_str_(const char *t, i32 n)
{
    u8 h[32];
    u64 v = 0;
    i32 i;
    b3_hash(t, (size_t)n, h);
    for (i = 7; i >= 0; --i) v = v << 8 | (u64)h[i];
    v &= ~STORE_ENC_SIGN;                                        /* vi_key's */
    return v ? v : 1;
}

static i32 store_isal_(i32 c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
static i32 store_isdig_(i32 c) { return c >= '0' && c <= '9'; }

/* the words a filter keeps for itself: never a field's name */
static i32 store_fword_(const char *t, i32 n, const char *w)
{
    i32 i;
    for (i = 0; i < n; ++i) if (!w[i] || (t[i] | 32) != w[i]) return 0;
    return w[n] == 0;
}
static i32 store_freserved_(const char *t, i32 n)
{
    return store_fword_(t, n, "and") || store_fword_(t, n, "or") || store_fword_(t, n, "not") || store_fword_(t, n, "in")
        || store_fword_(t, n, "has") || store_fword_(t, n, "true") || store_fword_(t, n, "false");
}

/* a field's name: 1 to 31 of A-Z a-z 0-9 _, not starting with a digit, not a word of the filter language */
static i32 store_fname_ok_(const char *nm, i32 n)
{
    i32 i;
    if (n < 1 || n >= STORE_FIELD_NAME || !store_isal_(nm[0])) return 0;
    for (i = 1; i < n; ++i) if (!store_isal_(nm[i]) && !store_isdig_(nm[i])) return 0;
    return !store_freserved_(nm, n);
}

static i32 store_field_find_(const RagStore *s, const char *nm, i32 n)
{
    i32 f;
    if (n < 1 || n >= STORE_FIELD_NAME) return -1;
    for (f = 0; f < s->nfld; ++f) if (memcmp(s->fld_name[f], nm, (size_t)n) == 0 && s->fld_name[f][n] == 0) return f;
    return -1;
}

/* Declares a field under the write lock: its index (an existing one of the same type too); -1 out of memory, -2 it
 * exists with another type, -3 the store has STORE_FIELDS already, -4 not a name. journal 1: a 'K' record. */
static i32 store_field_decl_locked_(RagStore *s, const char *nm, i32 n, i32 type, i32 journal)
{
    i32 f = store_field_find_(s, nm, n);
    if (f >= 0) return s->fld_type[f] == type ? f : -2;
    if (!store_fname_ok_(nm, n)) return -4;
    if (type < STORE_FT_INT || type > STORE_FT_STR) return -4;
    if (s->nfld >= STORE_FIELDS) return -3;
    f = s->nfld;
    memset(s->fld_name[f], 0, STORE_FIELD_NAME);
    memcpy(s->fld_name[f], nm, (size_t)n);
    s->fld_type[f] = (u8)type;
    s->nfld = f + 1;
    if (store_reserve_(s, (i64)s->count) != 0) { s->nfld = f; memset(s->fld_name[f], 0, STORE_FIELD_NAME); return -1; }
    if (journal) {
        u8 op = 'K', b[2];
        const void *pt[3];
        size_t pl[3];
        b[0] = (u8)f; b[1] = (u8)type;
        pt[0] = &op; pl[0] = 1; pt[1] = b; pl[1] = 2; pt[2] = s->fld_name[f]; pl[2] = STORE_FIELD_NAME;
        store_jframe_(s, pt, pl, 3);
    }
    ++s->mods;
    return f;
}

/* field f of the live row with this id set to code (0: unset), under the write lock: the row, or -1 (no live row, a
 * damaged one, out of memory). journal 1: a 'V' record. */
static i32 store_field_set_locked_(RagStore *s, i64 doc_id, i32 f, u64 code, i32 journal)
{
    i32 r = store_live_row_(s, doc_id);
    if (r < 0 || store_row_bad_(s, r) || f < 0 || f >= s->nfld) return -1;
    {   i32 ci = (i32)((i64)r >> STORE_CSHIFT);
        i64 used = (i64)s->count - ((i64)ci << STORE_CSHIFT);
        if (scol_unshare_(s, &s->fldc[f], ci, used > STORE_CROWS ? STORE_CROWS : used) != 0) return -1;
    }
    STORE_FLD(s, f, r) = code;
    STORE_RCK(s, r) = store_rck_row_(s, r);
    ++s->mods;
    if (journal) {
        u8 op = 'V', fb = (u8)f;
        const void *pt[4];
        size_t pl[4];
        pt[0] = &op; pl[0] = 1; pt[1] = &doc_id; pl[1] = 8; pt[2] = &fb; pl[2] = 1; pt[3] = &code; pl[3] = 8;
        store_jframe_(s, pt, pl, 4);
    }
    return r;
}

/* Declares a field (journaled): its index, or -1..-4 as store_field_decl_locked_ */
i32 store_field_decl(RagStore *s, const char *nm, i32 n, i32 type)
{
    i32 f;
    if (s->ro) return -1;
    pthread_rwlock_wrlock(&s->lock);
    f = store_field_decl_locked_(s, nm, n, type, 1);
    pthread_rwlock_unlock(&s->lock);
    return f;
}

/* Sets field nm (declared with type on its first use) of the row with this id to code, 0 unsets it (journaled): 0, -1
 * no live row with this id, -2 the field has another type, -3 no room for another field, -4 not a name, -5 out of
 * memory */
i32 store_field_set(RagStore *s, i64 doc_id, const char *nm, i32 n, i32 type, u64 code)
{
    i32 f, r;
    if (s->ro) return -5;
    pthread_rwlock_wrlock(&s->lock);
    store_idmap_locked_(s);
    f = store_field_find_(s, nm, n);
    if (f >= 0 && s->fld_type[f] != type) { pthread_rwlock_unlock(&s->lock); return -2; }
    if (f < 0 && code == 0) { pthread_rwlock_unlock(&s->lock); return store_live_row_(s, doc_id) >= 0 ? 0 : -1; }   /* unset a field no row has */
    if (f < 0 && store_live_row_(s, doc_id) < 0) { pthread_rwlock_unlock(&s->lock); return -1; }   /* nothing declared for a missing row */
    if (f < 0 && (f = store_field_decl_locked_(s, nm, n, type, 1)) < 0) { pthread_rwlock_unlock(&s->lock); return f == -1 ? -5 : f; }
    r = store_field_set_locked_(s, doc_id, f, code, 1);
    pthread_rwlock_unlock(&s->lock);
    return r >= 0 ? 0 : -1;
}

/* field nm of the row with this id: 0 set (*type, *code), 1 not set, -1 no live row, -2 no such field */
i32 store_field_get(RagStore *s, i64 doc_id, const char *nm, i32 n, i32 *type, u64 *code)
{
    i32 r, f, ret;
    store_need_idmap_(s);
    pthread_rwlock_rdlock(&s->lock);
    f = store_field_find_(s, nm, n);
    r = store_live_row_(s, doc_id);
    if (r >= 0 && store_row_bad_(s, r)) r = -1;
    if (r < 0) ret = -1;
    else if (f < 0) ret = -2;
    else { *type = s->fld_type[f]; *code = STORE_FLD(s, f, r); ret = *code ? 0 : 1; }
    pthread_rwlock_unlock(&s->lock);
    return ret;
}

/* field f's name (into out[STORE_FIELD_NAME]) and type: 0, -1 past the last */
i32 store_field_info(RagStore *s, i32 f, char *out, i32 *type)
{
    i32 r = -1;
    pthread_rwlock_rdlock(&s->lock);
    if (f >= 0 && f < s->nfld) { memcpy(out, s->fld_name[f], STORE_FIELD_NAME); *type = s->fld_type[f]; r = 0; }
    pthread_rwlock_unlock(&s->lock);
    return r;
}

/* ---- filters ---- */

enum { SF_EQ = 1, SF_NE, SF_LT, SF_LE, SF_GT, SF_GE, SF_IN, SF_NIN, SF_HAS, SF_AND, SF_OR, SF_NOT };
#define STORE_FLT_OPS 96            /* instructions of one filter */
#define STORE_FLT_VALS 256          /* literals of one filter, every in (...) list's together */
#define STORE_FLT_STR 1024          /* the longest string literal, in bytes */

typedef struct { u8 op, fld; u16 a, n; } StoreFop;              /* field fld against val[a] (val[a, a + n) for in) */
typedef struct { u64 sid; i32 nop, nval; StoreFop op[STORE_FLT_OPS]; u64 val[STORE_FLT_VALS]; } StoreFlt;   /* sid: its store */

static i32 store_flt_in_(const u64 *v, i32 n, u64 x)              /* v sorted: log2 n steps */
{
    i32 lo = 0, hi = n;
    while (lo < hi) { i32 m = lo + (hi - lo) / 2; if (v[m] < x) lo = m + 1; else hi = m; }
    return lo < n && v[lo] == x;
}

/* the filter on row r: 1 it passes, 0 not. A comparison of a field the row has not set is false, whatever the
 * operator (!= and not in too): "not lang = c" is what also takes the rows without a lang. Under the read lock. */
static i32 store_flt_eval_(const RagStore *s, const StoreFlt *f, i32 row)
{
    u64 st = 0, v, x, b;
    i32 i;
    for (i = 0; i < f->nop; ++i) {
        const StoreFop *o = &f->op[i];
        if (o->op == SF_AND) { x = st & 1u; st >>= 1; st = (st & ~(u64)1) | (st & x); continue; }
        if (o->op == SF_OR) { x = st & 1u; st >>= 1; st |= x; continue; }
        if (o->op == SF_NOT) { st ^= 1u; continue; }
        v = STORE_FLD(s, o->fld, row);
        x = f->val[o->a];                                        /* a literal's code: never 0 */
        switch (o->op) {
        case SF_EQ: b = v == x; break;
        case SF_NE: b = v != 0 && v != x; break;
        case SF_LT: b = v != 0 && v < x; break;
        case SF_LE: b = v != 0 && v <= x; break;
        case SF_GT: b = v > x; break;
        case SF_GE: b = v >= x; break;
        case SF_IN: b = v != 0 && store_flt_in_(f->val + o->a, o->n, v); break;
        case SF_NIN: b = v != 0 && !store_flt_in_(f->val + o->a, o->n, v); break;
        default: b = v != 0; break;                              /* SF_HAS */
        }
        st = st << 1 | b;
    }
    return (i32)(st & 1u);
}

/* what the scan calls for a candidate (tq_scan.c, TqFilt.ok) */
typedef struct { const RagStore *s; const StoreFlt *f; } StoreFltX;
static i32 store_flt_cb_(const void *p, i32 row)
{
    const StoreFltX *x = (const StoreFltX *)p;
    return store_flt_eval_(x->s, x->f, row);
}

/* ---- the parser: shunting-yard over tokens, atoms compiled as they come ---- */

enum { STK_END = 0, STK_ID, STK_NUM, STK_STR, STK_LP, STK_RP, STK_COMMA, STK_CMP, STK_AND, STK_OR, STK_NOT, STK_BAD };

typedef struct { const char *e, *p, *t; i32 n, kind, cmp; } StoreLex;   /* t, n: the token's text; cmp: an SF_ op */

static void store_lex_next_(StoreLex *L)
{
    const char *p = L->p;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
    L->t = p; L->n = 1; L->cmp = 0;
    if (!*p) { L->kind = STK_END; L->n = 0; L->p = p; return; }
    if (*p == '(') L->kind = STK_LP;
    else if (*p == ')') L->kind = STK_RP;
    else if (*p == ',') L->kind = STK_COMMA;
    else if (*p == '&' && p[1] == '&') { L->kind = STK_AND; L->n = 2; }
    else if (*p == '|' && p[1] == '|') { L->kind = STK_OR; L->n = 2; }
    else if (*p == '!' && p[1] == '=') { L->kind = STK_CMP; L->cmp = SF_NE; L->n = 2; }
    else if (*p == '!') L->kind = STK_NOT;
    else if (*p == '=') { L->kind = STK_CMP; L->cmp = SF_EQ; L->n = p[1] == '=' ? 2 : 1; }
    else if (*p == '<') { L->kind = STK_CMP; L->cmp = p[1] == '=' ? SF_LE : p[1] == '>' ? SF_NE : SF_LT; L->n = p[1] == '=' || p[1] == '>' ? 2 : 1; }
    else if (*p == '>') { L->kind = STK_CMP; L->cmp = p[1] == '=' ? SF_GE : SF_GT; L->n = p[1] == '=' ? 2 : 1; }
    else if (*p == '"') {                                        /* to the closing quote, \" and \\ inside */
        const char *q = p + 1;
        while (*q && *q != '"') q += (*q == '\\' && q[1]) ? 2 : 1;
        L->kind = *q == '"' ? STK_STR : STK_BAD; L->n = (i32)(q - p) + (*q == '"');
    } else if (store_isdig_(*p) || ((*p == '-' || *p == '+' || *p == '.') && (store_isdig_(p[1]) || (p[1] == '.' && store_isdig_(p[2]))))) {
        const char *q = p + 1;
        while (store_isdig_(*q) || *q == '.' || *q == 'e' || *q == 'E' || ((*q == '-' || *q == '+') && (q[-1] == 'e' || q[-1] == 'E'))) ++q;
        L->kind = STK_NUM; L->n = (i32)(q - p);
    } else if (store_isal_(*p)) {
        const char *q = p + 1;
        while (store_isal_(*q) || store_isdig_(*q)) ++q;
        L->n = (i32)(q - p);
        L->kind = store_fword_(p, L->n, "and") ? STK_AND : store_fword_(p, L->n, "or") ? STK_OR
                : store_fword_(p, L->n, "not") ? STK_NOT : STK_ID;
    } else L->kind = STK_BAD;
    L->p = p + L->n;
}

/* a literal (the current token) as field type ty's code into *c: 0, or -1 with *why */
static i32 store_lit_(const StoreLex *L, i32 ty, u64 *c, const char **why)
{
    if (ty == STORE_FT_STR) {
        char b[STORE_FLT_STR];
        i32 i, n = 0;
        if (L->kind != STK_STR) { *why = "a string field takes a \"string\""; return -1; }
        for (i = 1; i < L->n - 1; ++i) {
            if (n >= STORE_FLT_STR) { *why = "a string longer than 1024 bytes"; return -1; }
            b[n++] = L->t[i] == '\\' && i + 1 < L->n - 1 ? L->t[++i] : L->t[i];
        }
        *c = store_enc_str_(b, n);
        return 0;
    }
    if (L->kind == STK_ID && (store_fword_(L->t, L->n, "true") || store_fword_(L->t, L->n, "false"))) {
        if (ty != STORE_FT_INT) { *why = "true and false are ints (1 and 0)"; return -1; }
        *c = store_enc_int_(store_fword_(L->t, L->n, "true") ? 1 : 0);
        return 0;
    }
    if (L->kind != STK_NUM) { *why = ty == STORE_FT_INT ? "an int field takes a whole number" : "a float field takes a number"; return -1; }
    if (ty == STORE_FT_INT) {
        u64 v = 0;
        i32 i = 0, neg = 0;
        if (L->t[0] == '-' || L->t[0] == '+') { neg = L->t[0] == '-'; i = 1; }
        if (i >= L->n) { *why = "not a number"; return -1; }
        for (; i < L->n; ++i) {
            if (!store_isdig_(L->t[i])) { *why = "an int field takes a whole number (no . or e)"; return -1; }
            if (v > (~(u64)0 - 9) / 10) { *why = "a number outside the 64-bit ints"; return -1; }
            v = v * 10 + (u64)(L->t[i] - '0');
        }
        if (v >= ((u64)1 << 63)) { *why = "a number outside the stored ints (-2^63 + 1 to 2^63 - 1)"; return -1; }
        *c = store_enc_int_(neg ? -(i64)v : (i64)v);
        return 0;
    } else {
        char b[64];
        char *end;
        f64 d;
        if (L->n >= 64) { *why = "a number too long"; return -1; }
        memcpy(b, L->t, (size_t)L->n); b[L->n] = 0;
        d = strtod(b, &end);
        if (end != b + L->n || d != d || d - d != 0.0) { *why = "not a finite number"; return -1; }
        *c = store_enc_f64_(d);
        return 0;
    }
}

static i32 store_flt_emit_(StoreFlt *f, i32 op, i32 fld, i32 a, i32 n, i32 *depth, const char **why)
{
    if (f->nop >= STORE_FLT_OPS) { *why = "more than 96 steps"; return -1; }
    f->op[f->nop].op = (u8)op; f->op[f->nop].fld = (u8)fld; f->op[f->nop].a = (u16)a; f->op[f->nop].n = (u16)n;
    ++f->nop;
    if (op == SF_AND || op == SF_OR) --*depth;
    else if (op != SF_NOT && ++*depth > 63) { *why = "nested deeper than 63"; return -1; }
    return 0;
}

/* one atom at the current token (a name): field op literal | field [not] in (literals) | has field | field (an int:
 * set and not 0). Leaves L at the token after it. */
static i32 store_flt_atom_(const RagStore *s, StoreLex *L, StoreFlt *f, i32 *depth, const char **why)
{
    i32 fld, ty, has = store_fword_(L->t, L->n, "has"), neg = 0;
    if (has) { store_lex_next_(L); if (L->kind != STK_ID) { *why = "has takes a field"; return -1; } }
    if ((fld = store_field_find_(s, L->t, L->n)) < 0) { *why = "no field of that name (one is made by setting it on a row)"; return -1; }
    ty = s->fld_type[fld];
    store_lex_next_(L);
    if (has) return store_flt_emit_(f, SF_HAS, fld, 0, 0, depth, why);
    if (L->kind == STK_NOT) {                                    /* not in */
        const char *keep = L->p;
        store_lex_next_(L);
        if (!(L->kind == STK_ID && store_fword_(L->t, L->n, "in"))) { L->p = keep; *why = "a field then not takes in (...)"; return -1; }
        neg = 1;
    }
    if (L->kind == STK_ID && store_fword_(L->t, L->n, "in")) {
        i32 a = f->nval, n = 0, i, j;
        store_lex_next_(L);
        if (L->kind != STK_LP) { *why = "in takes (a, b, ...)"; return -1; }
        for (;;) {
            u64 c;
            store_lex_next_(L);
            if (f->nval >= STORE_FLT_VALS) { *why = "more than 256 values"; return -1; }
            if (store_lit_(L, ty, &c, why)) return -1;
            for (i = a; i < a + n && f->val[i] < c; ++i) {}       /* sorted, no repeats */
            if (i == a + n || f->val[i] != c) {
                for (j = a + n; j > i; --j) f->val[j] = f->val[j - 1];
                f->val[i] = c; ++n; ++f->nval;
            }
            store_lex_next_(L);
            if (L->kind == STK_RP) break;
            if (L->kind != STK_COMMA) { *why = "in (...) takes values with commas between them"; return -1; }
        }
        store_lex_next_(L);
        return store_flt_emit_(f, neg ? SF_NIN : SF_IN, fld, a, n, depth, why);
    }
    if (L->kind == STK_CMP) {
        i32 op = L->cmp;
        u64 c;
        if (ty == STORE_FT_STR && op != SF_EQ && op != SF_NE) { *why = "a string field takes = != in and not in, never < or >"; return -1; }
        store_lex_next_(L);
        if (f->nval >= STORE_FLT_VALS) { *why = "more than 256 values"; return -1; }
        if (store_lit_(L, ty, &c, why)) return -1;
        f->val[f->nval] = c;
        store_lex_next_(L);
        return store_flt_emit_(f, op, fld, f->nval++, 1, depth, why);
    }
    if (ty != STORE_FT_INT) { *why = "a field alone must be an int (true when set and not 0); else compare it, or has"; return -1; }
    if (f->nval >= STORE_FLT_VALS) { *why = "more than 256 values"; return -1; }
    f->val[f->nval] = store_enc_int_(0);
    return store_flt_emit_(f, SF_NE, fld, f->nval++, 1, depth, why);
}

/* Compiles expression e into *f for store s: 0, or -1 with *why and *at (the byte of e where it went wrong). */
i32 store_flt_parse(RagStore *s, const char *e, StoreFlt *f, const char **why, i32 *at)
{
    StoreLex L;
    char ops[64];                                                /* the operator stack: '(' 'a' 'o' 'n' */
    i32 nops = 0, want = 1, depth = 0, r = 0;                    /* want 1: an operand next, 0: an operator */
    memset(f, 0, sizeof(*f));
    f->sid = (u64)(size_t)s;
    *why = (const char *)0; *at = 0;
    L.e = L.p = e;
    pthread_rwlock_rdlock(&s->lock);
    store_lex_next_(&L);
    while (!r) {
        *at = (i32)(L.t - e);
        if (want) {
            if (L.kind == STK_NOT) { if (nops >= 64) { *why = "nested deeper than 63"; r = -1; break; } ops[nops++] = 'n'; store_lex_next_(&L); }
            else if (L.kind == STK_LP) { if (nops >= 64) { *why = "nested deeper than 63"; r = -1; break; } ops[nops++] = '('; store_lex_next_(&L); }
            else if (L.kind == STK_ID) { if (store_flt_atom_(s, &L, f, &depth, why)) { r = -1; break; } want = 0; }
            else { *why = L.kind == STK_END ? "the filter ends where a condition should be" : "a condition expected (a field, not, or ()"; r = -1; }
        } else if (L.kind == STK_AND || L.kind == STK_OR) {
            i32 pn = L.kind == STK_AND ? 2 : 1;
            while (nops && ops[nops - 1] != '(' && (ops[nops - 1] == 'n' ? 3 : ops[nops - 1] == 'a' ? 2 : 1) >= pn) {
                char o = ops[--nops];
                if (store_flt_emit_(f, o == 'n' ? SF_NOT : o == 'a' ? SF_AND : SF_OR, 0, 0, 0, &depth, why)) { r = -1; break; }
            }
            if (r) break;
            ops[nops++] = L.kind == STK_AND ? 'a' : 'o';
            want = 1;
            store_lex_next_(&L);
        } else if (L.kind == STK_RP || L.kind == STK_END) {
            while (nops && ops[nops - 1] != '(') {
                char o = ops[--nops];
                if (store_flt_emit_(f, o == 'n' ? SF_NOT : o == 'a' ? SF_AND : SF_OR, 0, 0, 0, &depth, why)) { r = -1; break; }
            }
            if (r) break;
            if (L.kind == STK_END) { if (nops) { *why = "a ( without its )"; r = -1; } break; }
            if (!nops) { *why = "a ) without its ("; r = -1; break; }
            --nops;
            store_lex_next_(&L);
        } else { *why = "and, or or ) expected"; r = -1; }
    }
    pthread_rwlock_unlock(&s->lock);
    if (!r && depth != 1) { *why = "not one condition"; r = -1; }  /* never: the grammar leaves one */
    return r;
}

/* the filter on the live row with this id: 1, 0, -1 no such row */
i32 store_flt_match(RagStore *s, i64 doc_id, const StoreFlt *f)
{
    i32 r;
    store_need_idmap_(s);
    pthread_rwlock_rdlock(&s->lock);
    r = store_live_row_(s, doc_id);
    if (r >= 0 && store_row_bad_(s, r)) r = -1;
    if (r >= 0) r = store_flt_eval_(s, f, r);
    pthread_rwlock_unlock(&s->lock);
    return r;
}
