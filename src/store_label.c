/* store_label.c - labels: names for the 64 bits of a row's tag, in a hierarchy. A label may sit under other labels (a
 * graph without cycles, several parents allowed); asking for a label means it or any label below it, worked out when
 * the search runs - so a new hierarchy is new links between labels, and no row is touched. Searches take conditions
 * on labels: every one of these, at least one of those, none of the rest. The table is saved in the file and every
 * change to it is journaled. Part of store.c, which includes it after store_text.c; not compiled on its own. */

/* a label's name: 1 to 63 of A-Z a-z 0-9 _ - . / : (commas and spaces separate names in a list) */
static i32 store_label_name_ok_(const char *n, size_t len)
{
    size_t i;
    if (len < 1 || len >= STORE_LABEL_NAME) return 0;
    for (i = 0; i < len; ++i) {
        char c = n[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.'
              || c == '/' || c == ':')) return 0;
    }
    return 1;
}

/* the bit of the label called n (len bytes), or -1; under a lock */
static i32 store_label_find_(const RagStore *s, const char *n, size_t len)
{
    i32 b;
    for (b = 0; b < STORE_LABELS; ++b)
        if (s->lab_name[b][0] && strlen(s->lab_name[b]) == len && memcmp(s->lab_name[b], n, len) == 0) return b;
    return -1;
}

/* bit b and every label below it, as a mask; under a lock */
static u64 store_label_tree_(const RagStore *s, i32 b)
{
    u64 m = (u64)1 << b, was;
    i32 j;
    do {
        was = m;
        for (j = 0; j < STORE_LABELS; ++j) if (s->lab_name[j][0] && (s->lab_par[j] & m)) m |= (u64)1 << j;
    } while (m != was);
    return m;
}

/* label b's whole state (its name and parents) to the journal: replay sets it as it is now */
static void store_label_jrec_(RagStore *s, i32 b)
{
    const void *pt[4];
    size_t pl[4];
    u8 op = 'L', bb = (u8)b;
    pt[0] = &op; pl[0] = 1; pt[1] = &bb; pl[1] = 1; pt[2] = s->lab_name[b]; pl[2] = STORE_LABEL_NAME;
    pt[3] = &s->lab_par[b]; pl[3] = 8;
    store_jframe_(s, pt, pl, 4);
}

/* the bit of label n, made when there is none yet (and journaled); -1 when n is not a name or all 64 are in use.
 * Under the write lock. */
static i32 store_label_get_(RagStore *s, const char *n, size_t len)
{
    i32 b = store_label_find_(s, n, len);
    if (b >= 0 || !store_label_name_ok_(n, len)) return b;
    for (b = 0; b < STORE_LABELS && s->lab_name[b][0]; ++b) {}
    if (b == STORE_LABELS) return -1;
    memcpy(s->lab_name[b], n, len); s->lab_name[b][len] = 0;
    s->lab_par[b] = 0;
    ++s->nlab;
    store_label_jrec_(s, b);
    return b;
}

/* Label name (0-terminated), made when new, and placed under parent (NULL: no new link; made when new too). Returns its
 * bit, -1 for a bad name or when all 64 are in use, -2 when the link would close a cycle. */
i32 store_label(RagStore *s, const char *name, const char *parent)
{
    i32 b, p;
    pthread_rwlock_wrlock(&s->lock);
    b = name ? store_label_get_(s, name, strlen(name)) : -1;
    if (b >= 0 && parent) {
        p = store_label_get_(s, parent, strlen(parent));
        if (p < 0) b = -1;
        else if (store_label_tree_(s, b) & ((u64)1 << p)) b = -2;  /* the parent is below the label already */
        else if (!(s->lab_par[b] & ((u64)1 << p))) { s->lab_par[b] |= (u64)1 << p; store_label_jrec_(s, b); }
    }
    pthread_rwlock_unlock(&s->lock);
    return b;
}

/* Label b set as the journal says (replay): its name ("" frees it) and parents */
static void store_label_set_(RagStore *s, i32 b, const char *name, u64 par)
{
    if (b < 0 || b >= STORE_LABELS) return;
    if (s->lab_name[b][0] && !name[0]) --s->nlab;
    if (!s->lab_name[b][0] && name[0]) ++s->nlab;
    memcpy(s->lab_name[b], name, STORE_LABEL_NAME);
    s->lab_name[b][STORE_LABEL_NAME - 1] = 0;
    s->lab_par[b] = par;
}

/* The bits of a list of label names (commas or spaces between them) each with everything below it, one mask a name
 * into masks[max]: how many, -1 when a name is unknown or bad (*bad: where it starts), -2 for more than max. make 1:
 * unknown names are made (under the write lock: for labelling rows). */
static i32 store_label_masks_(RagStore *s, const char *list, u64 *masks, i32 max, i32 make, const char **bad)
{
    i32 n = 0, b;
    const char *p = list;
    if (!list) return 0;
    for (;;) {
        size_t len;
        while (*p == ',' || *p == ' ') ++p;
        if (!*p) break;
        for (len = 0; p[len] && p[len] != ',' && p[len] != ' '; ++len) {}
        b = make ? store_label_get_(s, p, len) : store_label_find_(s, p, len);
        if (b < 0) { if (bad) *bad = p; return -1; }
        if (n == max) return -2;
        masks[n++] = make ? (u64)1 << b : store_label_tree_(s, b);
        p += len;
    }
    return n;
}

/* Puts the labels of a list on row id (on 1) or takes them off (on 0); labels new to the store are made. 0, -1 no such
 * row (or a damaged one), -2 a bad name or all 64 labels in use (*bad: where). Journaled as a tag change. */
i32 store_label_row(RagStore *s, i64 id, const char *list, i32 on, const char **bad)
{
    u64 m[STORE_LABELS];
    i64 tag = 0;
    i32 n, i;
    pthread_rwlock_wrlock(&s->lock);
    n = store_label_masks_(s, list, m, STORE_LABELS, 1, bad);
    pthread_rwlock_unlock(&s->lock);
    if (n < 0) return -2;
    if (store_get_meta(s, id, &tag) != 0) return -1;
    for (i = 0; i < n; ++i) tag = on ? (i64)((u64)tag | m[i]) : (i64)((u64)tag & ~m[i]);
    return store_set_meta(s, id, tag) == 0 ? 0 : -1;
}

/* The conditions of a search from three lists: every label of `all` (it or one below it), one at least of `any`, none of
 * `none`. 0, -1 an unknown label (*bad: where), -2 more than 15 in `all`. */
i32 store_label_cnf(RagStore *s, const char *all, const char *any, const char *none, StoreCnf *c, const char **bad)
{
    u64 m[STORE_LABELS];
    i32 n, i, r = 0;
    memset(c, 0, sizeof(*c));
    pthread_rwlock_rdlock(&s->lock);
    n = store_label_masks_(s, all, c->any, 15, 0, bad);
    if (n < 0) r = n;
    else {
        c->n = n;
        if ((n = store_label_masks_(s, any, m, STORE_LABELS, 0, bad)) < 0) r = n;
        else if (n > 0) { c->any[c->n] = 0; for (i = 0; i < n; ++i) c->any[c->n] |= m[i]; ++c->n; }
        if (!r && (n = store_label_masks_(s, none, m, STORE_LABELS, 0, bad)) < 0) r = n;
        else if (!r) for (i = 0; i < n; ++i) c->none |= m[i];
    }
    pthread_rwlock_unlock(&s->lock);
    return r;
}

/* the name of label b into out[64] ("" when it is free) and its parents' bits; -1 when b is out of range */
i32 store_label_info(RagStore *s, i32 b, char *out, u64 *parents)
{
    if (b < 0 || b >= STORE_LABELS) return -1;
    pthread_rwlock_rdlock(&s->lock);
    if (out) memcpy(out, s->lab_name[b], STORE_LABEL_NAME);
    if (parents) *parents = s->lab_par[b];
    pthread_rwlock_unlock(&s->lock);
    return 0;
}

/* the k nearest rows whose labels meet c */
i32 store_search_cnf(RagStore *s, const f32 *query, i32 k, i64 *out_doc_ids, f32 *out_scores, const StoreCnf *c)
{
    if (!store_finite_(query, s->dim)) return 0;                 /* as store_search */
    return store_search_impl_(s, query, k, out_doc_ids, out_scores, 0, c->n ? (i64)c->any[0] : 0, 1, c);
}
