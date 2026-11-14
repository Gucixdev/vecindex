/* mem_rules.c - the rule engine: a small Lisp/Datalog-flavoured s-expression reader and evaluator over the graph.
 * Part of mem.c, which includes it at this point; not compiled on its own. */

/* ------------------------------------------------------------------ */
/* the rule engine: a small Lisp/Datalog-flavoured s-expression reader  */
/* ------------------------------------------------------------------ */

typedef struct SExpr {
    enum { SX_SYM, SX_NUM, SX_LIST } t;
    char *sym;
    f64 num;
    struct SExpr **items; i32 nitems, cap;
} SExpr;

static SExpr *sx_new_(int t) { SExpr *e = (SExpr *)calloc(1, sizeof(SExpr)); e->t = t; return e; }
static void sx_add_(SExpr *l, SExpr *item) { if (l->nitems == l->cap) { l->cap = l->cap ? l->cap * 2 : 4; l->items = (SExpr **)realloc(l->items, (size_t)l->cap * sizeof(SExpr *)); } l->items[l->nitems++] = item; }
static void sx_free_(SExpr *e) { i32 i; if (!e) return; if (e->t == SX_LIST) { for (i = 0; i < e->nitems; ++i) sx_free_(e->items[i]); free(e->items); } free(e->sym); free(e); }

/* rule files are meant to be hand-edited (see mem_rules_parse_'s own comment on why malformed
 * top-level forms are just skipped, not fatal) - but nothing upstream of this reader validates
 * that before it runs, and mrfuzz.c found that a rule string is not necessarily trusted input
 * even when it usually is operator-authored (a config file that gets appended to, say). A run of
 * more than this many nested '(' used to blow the C stack (sx_read_ recursed once per nesting
 * level with no bound - confirmed crashing well under 200,000 levels, which is a trivially small
 * malicious/malformed file, not an exotic one). Real rules nest at most a handful of levels
 * (rule -> and -> similar/kind/comparison), so this is generous, not a real limit in practice. */
#define SX_MAX_DEPTH 200

/* one s-expression starting at *p (whitespace and ';' comments before it are skipped); *p is
 * left just past it. Returns NULL at end of input, on a stray ')', or when `depth` (nesting
 * level, 0 at the top) hits SX_MAX_DEPTH - the over-deep list is skipped as a whole (balanced-
 * paren scan, so *p always still advances) rather than recursed into further. */
static SExpr *sx_read_(const char **p, const char *end, i32 depth)
{
    const char *s;
    for (;;) {
        while (*p < end && (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r')) ++*p;
        if (*p < end && **p == ';') { while (*p < end && **p != '\n') ++*p; continue; }
        break;
    }
    if (*p >= end) return (SExpr *)0;
    if (**p == ')') return (SExpr *)0;
    if (**p == '(') {
        SExpr *l;
        if (depth >= SX_MAX_DEPTH) {
            i32 nest = 1; ++*p;
            while (*p < end && nest > 0) { if (**p == '(') ++nest; else if (**p == ')') --nest; ++*p; }
            return (SExpr *)0;
        }
        l = sx_new_(SX_LIST);
        ++*p;
        for (;;) {
            SExpr *item;
            while (*p < end && (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r')) ++*p;
            if (*p < end && **p == ';') { while (*p < end && **p != '\n') ++*p; continue; }
            if (*p >= end || **p == ')') { if (*p < end) ++*p; break; }
            item = sx_read_(p, end, depth + 1);
            if (!item) break;
            sx_add_(l, item);
        }
        return l;
    }
    s = *p;
    while (*p < end && **p != ' ' && **p != '\t' && **p != '\n' && **p != '\r' && **p != '(' && **p != ')') ++*p;
    { i32 n = (i32)(*p - s), i; int isnum = n > 0;
      for (i = 0; i < n; ++i) if (!((s[i] >= '0' && s[i] <= '9') || s[i] == '.' || s[i] == '-' || s[i] == '+')) { isnum = 0; break; }
      if (isnum && n > 0 && !(n == 1 && (s[0] == '-' || s[0] == '+'))) { SExpr *e = sx_new_(SX_NUM); e->num = strtod(s, (char **)0); return e; }
      { SExpr *e = sx_new_(SX_SYM); e->sym = (char *)malloc((size_t)n + 1); memcpy(e->sym, s, (size_t)n); e->sym[n] = 0; return e; } }
}

typedef struct {
    char name[64];
    SExpr *cond;
    SExpr **actions; i32 nactions;
} MemRule;

/* parse a rule file's text into rules (caller frees with mem_rules_free); malformed top-level
 * forms are skipped (a rule file is meant to be hand-edited - this stays permissive rather than
 * aborting the whole set over one typo, matching the "organism keeps functioning" spirit). */
static MemRule *mem_rules_parse_(const char *src, i32 len, i32 *out_n)
{
    const char *p = src, *end = src + len;
    MemRule *rules = (MemRule *)0;
    i32 n = 0, cap = 0;
    SExpr *form;
    while ((form = sx_read_(&p, end, 0)) != (SExpr *)0) {
        if (form->t == SX_LIST && form->nitems >= 3 && form->items[0]->t == SX_SYM && !strcmp(form->items[0]->sym, "rule") && form->items[1]->t == SX_SYM) {
            MemRule *r;
            if (n == cap) { cap = cap ? cap * 2 : 8; rules = (MemRule *)realloc(rules, (size_t)cap * sizeof(MemRule)); }
            r = &rules[n++];
            memset(r, 0, sizeof(*r));
            strncpy(r->name, form->items[1]->sym, sizeof(r->name) - 1);
            r->cond = form->items[2]; form->items[2] = (SExpr *)0;                 /* ownership moves to the rule */
            { i32 i; for (i = 3; i < form->nitems; ++i) { r->actions = (SExpr **)realloc(r->actions, (size_t)(r->nactions + 1) * sizeof(SExpr *)); r->actions[r->nactions++] = form->items[i]; form->items[i] = (SExpr *)0; } }
        }
        sx_free_(form);
    }
    *out_n = n;
    return rules;
}

static void mem_rules_free_(MemRule *rules, i32 n)
{
    i32 i;
    for (i = 0; i < n; ++i) { sx_free_(rules[i].cond); { i32 j; for (j = 0; j < rules[i].nactions; ++j) sx_free_(rules[i].actions[j]); } free(rules[i].actions); }
    free(rules);
}

typedef struct { MemKind kind; f32 relevance; i32 corroboration, contradiction, access_count; f64 age, idle; Mem *m; i32 vid; } MemCtx;

static i32 mem_kind_of_name_(const char *s)
{
    if (!strcmp(s, "event")) return MK_EVENT;
    if (!strcmp(s, "fact")) return MK_FACT;
    if (!strcmp(s, "concept")) return MK_CONCEPT;
    return -1;
}

/* is there a live SAME-KIND node other than ctx->vid with cosine >= thresh against it? a real
 * vector search, not a stub - used by both the (similar T) condition and the merge action.
 * out_other (may be NULL) gets the best such neighbour's vid. */
static i32 mem_similar_(const MemCtx *ctx, f32 thresh, i32 *out_other)
{
    char *text = vecdb_text(ctx->m->db, ctx->vid);
    VHit hits[8];
    i32 nh, k, found = 0;
    if (!text) return 0;
    nh = vecdb_search(ctx->m->db, text, 8, 1, -1, hits);
    for (k = 0; k < nh; ++k) {
        MemMeta *o;
        if (hits[k].node == ctx->vid || hits[k].sim < thresh) continue;
        o = mem_get_(ctx->m, hits[k].node);
        if (o && !o->archived && o->kind == ctx->kind) { found = 1; if (out_other) *out_other = hits[k].node; break; }
    }
    free(text);
    return found;
}

static f64 mem_field_(const MemCtx *c, const char *name)
{
    if (!strcmp(name, "relevance")) return (f64)c->relevance;
    if (!strcmp(name, "corroboration")) return (f64)c->corroboration;
    if (!strcmp(name, "contradiction")) return (f64)c->contradiction;
    if (!strcmp(name, "access_count")) return (f64)c->access_count;
    if (!strcmp(name, "age")) return c->age;
    if (!strcmp(name, "idle")) return c->idle;
    return 0.0;
}

static i32 sx_eval_bool_(const SExpr *e, MemCtx *ctx);

static i32 sx_eval_bool_(const SExpr *e, MemCtx *ctx)
{
    const char *op;
    if (!e || e->t != SX_LIST || e->nitems == 0 || e->items[0]->t != SX_SYM) return 0;
    op = e->items[0]->sym;
    if (!strcmp(op, "and")) { i32 i; for (i = 1; i < e->nitems; ++i) if (!sx_eval_bool_(e->items[i], ctx)) return 0; return 1; }
    if (!strcmp(op, "or")) { i32 i; for (i = 1; i < e->nitems; ++i) if (sx_eval_bool_(e->items[i], ctx)) return 1; return 0; }
    if (!strcmp(op, "not")) return e->nitems >= 2 ? !sx_eval_bool_(e->items[1], ctx) : 0;
    if (!strcmp(op, "kind")) return e->nitems >= 2 && e->items[1]->t == SX_SYM && mem_kind_of_name_(e->items[1]->sym) == (i32)ctx->kind;
    if (!strcmp(op, "similar")) return e->nitems >= 2 && e->items[1]->t == SX_NUM ? mem_similar_(ctx, (f32)e->items[1]->num, (i32 *)0) : 0;
    if ((!strcmp(op, "<") || !strcmp(op, "<=") || !strcmp(op, ">") || !strcmp(op, ">=") || !strcmp(op, "=") || !strcmp(op, "!="))
        && e->nitems >= 3 && e->items[1]->t == SX_SYM && e->items[2]->t == SX_NUM) {
        f64 lhs = mem_field_(ctx, e->items[1]->sym), rhs = e->items[2]->num;
        if (!strcmp(op, "<")) return lhs < rhs;
        if (!strcmp(op, "<=")) return lhs <= rhs;
        if (!strcmp(op, ">")) return lhs > rhs;
        if (!strcmp(op, ">=")) return lhs >= rhs;
        if (!strcmp(op, "=")) return lhs == rhs;
        return lhs != rhs;
    }
    return 0;
}

/* The shared per-node rule-application loop behind mem_consolidate/mem_consolidate_ids: `ids`
 * lists the vids to evaluate (order = evaluation order), already independent of m->meta's own
 * layout (mem_consolidate builds a throwaway snapshot for this; mem_consolidate_ids's caller
 * supplies its own list directly, e.g. the hot set - see mem_consolidate_hot). Locks m->lk once
 * for the whole pass. Same "acts immediately, never twice on what a prior action already
 * retired" discipline either way - see mem_consolidate's own doc comment for the full grammar
 * and per-action semantics, unchanged here. */
static i32 mem_consolidate_run_(Mem *m, MemRule *rules, i32 nrules, const i32 *ids, i32 nids)
{
    i32 i, fired = 0;
    i64 now = mem_now_();
    pthread_rwlock_wrlock(&m->lk);
    for (i = 0; i < nids; ++i) {
        MemMeta *e = mem_get_(m, ids[i]);
        MemCtx ctx; i32 r;
        if (!e || e->archived) continue;
        ctx.kind = e->kind; ctx.relevance = mem_relevance_(m, e); ctx.corroboration = e->corroboration;
        ctx.contradiction = e->contradiction; ctx.access_count = e->access_count;
        ctx.age = (f64)(now - e->born_at); ctx.idle = (f64)(now - e->last_touch);
        ctx.m = m; ctx.vid = e->vid;
        for (r = 0; r < nrules; ++r) {
            i32 a;
            e = mem_get_(m, ids[i]);
            if (!e || e->archived) break;                          /* an earlier action already retired this node */
            if (!sx_eval_bool_(rules[r].cond, &ctx)) continue;
            for (a = 0; a < rules[r].nactions; ++a) {
                SExpr *act = rules[r].actions[a];
                if (act->t != SX_LIST || act->nitems == 0 || act->items[0]->t != SX_SYM) continue;
                e = mem_get_(m, ids[i]);
                if (!e || e->archived) break;
                if (!strcmp(act->items[0]->sym, "prune")) {
                    if (e->corroboration == 0 && mem_relevance_(m, e) < 1.0f) {
                        i32 vid = e->vid, j;
                        for (j = 0; j < m->n; ++j) if (m->meta[j].vid != vid) mem_prov_remove_(&m->meta[j], vid);
                        e->archived = 1; h64_del(&m->byvid, (i64)vid); mem_hot_drop_(m, vid);
                        pthread_rwlock_unlock(&m->lk); vecdb_remove_note(m->db, vid); pthread_rwlock_wrlock(&m->lk);
                        ++fired;
                    }
                } else if (!strcmp(act->items[0]->sym, "archive")) {
                    if (m->cf) { i32 vid = e->vid; pthread_rwlock_unlock(&m->lk); mem_archive(m, vid); pthread_rwlock_wrlock(&m->lk); ++fired; }
                } else if (!strcmp(act->items[0]->sym, "merge")) {
                    /* closest same-kind neighbour above a bar (default 0.85; (merge THRESH) to override -
                     * the hashed built-in embedder puts genuine paraphrases around 0.6-0.7, so 0.92 is only
                     * right for a strong embedder finding near-duplicates; calibrate per embedder/corpus) */
                    char *text = vecdb_text(m->db, e->vid);
                    f32 bar = (act->nitems >= 2 && act->items[1]->t == SX_NUM) ? (f32)act->items[1]->num : 0.85f;
                    if (text) {
                        VHit hits[8]; i32 nh, k, other = -1; f32 bs = 0.0f;
                        pthread_rwlock_unlock(&m->lk);
                        nh = vecdb_search(m->db, text, 8, 1, -1, hits);
                        pthread_rwlock_wrlock(&m->lk);
                        for (k = 0; k < nh; ++k) { MemMeta *o; if (hits[k].node == e->vid) continue; o = mem_get_(m, hits[k].node); if (o && !o->archived && o->kind == e->kind && hits[k].sim > bs) { bs = hits[k].sim; other = hits[k].node; } }
                        free(text);
                        if (other >= 0 && bs >= bar) { i32 vid = e->vid; pthread_rwlock_unlock(&m->lk); mem_merge(m, vid, other); pthread_rwlock_wrlock(&m->lk); ++fired; }
                    }
                } else if (!strcmp(act->items[0]->sym, "promote") && act->nitems >= 2 && act->items[1]->t == SX_SYM && !e->promoted) {
                    i32 target = mem_kind_of_name_(act->items[1]->sym);
                    f32 bar = (act->nitems >= 3 && act->items[2]->t == SX_NUM) ? (f32)act->items[2]->num : 0.6f;
                    if (target == (i32)e->kind + 1) {                /* only ever one level up: event->fact->concept */
                        char *text = vecdb_text(m->db, e->vid);
                        const char *title = m->db->nodes[e->vid].name;
                        i32 vid = e->vid;
                        if (text) {
                            VHit hits[8]; i32 nh, k, other = -1; f32 bs = 0.0f;
                            pthread_rwlock_unlock(&m->lk);
                            nh = vecdb_search(m->db, text, 8, 1, -1, hits);
                            pthread_rwlock_wrlock(&m->lk);
                            for (k = 0; k < nh; ++k) { MemMeta *o = mem_get_(m, hits[k].node); if (o && !o->archived && (i32)o->kind == target && hits[k].sim > bs) { bs = hits[k].sim; other = hits[k].node; } }
                            if (other >= 0 && bs >= bar) { mem_prov_add_(mem_get_(m, other), vid); mem_get_(m, other)->corroboration++; mem_get_(m, other)->last_touch = now; mem_hot_touch_(m, other); ++fired; }
                            else {
                                i32 nvid; char nt[300];
                                /* NEVER reuse the source's bare title: vecdb_add_note upserts by title, so an
                                 * identical title would silently REPLACE the source note in place instead of
                                 * creating a sibling - a real bug this test suite caught (see mem.c's history). */
                                sprintf(nt, "%s: %.270s", act->items[1]->sym, title);
                                pthread_rwlock_unlock(&m->lk);
                                nvid = vecdb_add_note(m->db, nt, text);
                                pthread_rwlock_wrlock(&m->lk);
                                if (nvid >= 0) { MemMeta *nm = mem_new_meta_(m, nvid, (MemKind)target); mem_prov_add_(nm, vid); nm->corroboration = 1; mem_hot_touch_(m, nvid); ++fired; }
                            }
                            free(text);
                            { MemMeta *e2 = mem_get_(m, vid); if (e2) e2->promoted = 1; }   /* re-fetched: mem_new_meta_ above may have moved m->meta */
                        }
                    }
                }
            }
        }
    }
    pthread_rwlock_unlock(&m->lk);
    return fired;
}

/* Run one consolidation pass ("hippocampus": replay every memory item, reinforce, generalise,
 * let the rest keep fading) over `rules_src` (see the grammar below). Every live event/fact/
 * concept is evaluated against every rule in file order; a matching rule's actions run in order.
 * A node mutated by one rule (e.g. promoted, so it is no longer live under its old vid) is
 * skipped by the rules that follow it in THIS pass - a promotion/merge/prune takes effect
 * immediately, not after the whole pass, so a rule never acts twice on what is already gone.
 * Returns how many actions actually fired.
 *
 * Grammar:
 *   (rule NAME COND ACTION...)
 *   COND    := (and COND...) | (or COND...) | (not COND)
 *            | (kind event|fact|concept) | (similar THRESH)
 *            | (OP FIELD NUM)   OP in < <= > >= = != ; FIELD in relevance corroboration
 *              contradiction access_count age idle
 *   ACTION  := (promote fact|concept [THRESH]) | (merge [THRESH]) | (prune) | (archive)
 * promote: reinforces the best existing same-target-kind neighbour above THRESH (default 0.6 -
 *   calibrate to the embedder in use; the built-in hashed one puts real paraphrases around
 *   0.6-0.7, a strong model would want this higher) if one exists, else creates a new one
 *   (extractive: the source's own title/text) with this node as its sole provenance so far.
 * merge: merges with the closest same-kind neighbour, if any is above THRESH (default 0.85).
 * prune / archive: mem_prune's/mem_archive's own single-node logic (never destroys a supported
 *   node even if a rule naively asks).
 *
 * Cost: O(live nodes) vecdb_search calls per pass (see docs/mem.md and bench/mbench.c - that per-call cost
 * itself grows with corpus size, so a full pass gets expensive well before vecdb's own
 * million-note ceiling). mem_consolidate_ids/mem_consolidate_hot below let a caller scope a
 * pass to a working set instead, for exactly that reason. */
i32 mem_consolidate(Mem *m, const char *rules_src)
{
    MemRule *rules;
    i32 nrules, i, nn, fired, *snapshot;
    rules = mem_rules_parse_(rules_src, (i32)strlen(rules_src), &nrules);
    pthread_rwlock_wrlock(&m->lk);
    nn = m->n;
    snapshot = (i32 *)malloc((size_t)(nn ? nn : 1) * sizeof(i32));
    for (i = 0; i < nn; ++i) snapshot[i] = m->meta[i].vid;
    pthread_rwlock_unlock(&m->lk);
    fired = mem_consolidate_run_(m, rules, nrules, snapshot, nn);
    free(snapshot);
    mem_rules_free_(rules, nrules);
    return fired;
}

/* Same as mem_consolidate, but evaluated only over `ids` (in the order given) rather than every
 * live node - the caller picks the working set. A natural choice is the hot list
 * (mem_consolidate_hot, below, is exactly that); another is "ids touched since I last
 * consolidated", tracked by the caller. Ids that are no longer live are silently skipped, same
 * as everywhere else in this file - a stale id here is harmless, not a bug. Cost: O(nids)
 * vecdb_search calls per pass instead of O(all live nodes), which is the whole point (see
 * mem_consolidate's doc comment on why the full sweep gets expensive at scale). Run a full
 * mem_consolidate() occasionally (e.g. off the hot path, on a timer) so nodes that fall out of
 * the working set without ever matching a rule still eventually get evaluated. */
i32 mem_consolidate_ids(Mem *m, const char *rules_src, const i32 *ids, i32 nids)
{
    MemRule *rules;
    i32 nrules, fired;
    rules = mem_rules_parse_(rules_src, (i32)strlen(rules_src), &nrules);
    fired = mem_consolidate_run_(m, rules, nrules, ids, nids);
    mem_rules_free_(rules, nrules);
    return fired;
}

/* Convenience: consolidate just the current hot set (mem_hot_ids) - the cheap, frequent pass a
 * caller can run on every write or every few, since its cost is bounded by hot_cap rather than
 * corpus size. Pair with an occasional full mem_consolidate() sweep (nodes that never get
 * touched enough to stay hot still need a pass eventually, e.g. old events that should decay
 * into a prune rather than linger unconsolidated forever). */
i32 mem_consolidate_hot(Mem *m, const char *rules_src)
{
    i32 n, fired, *ids;
    pthread_rwlock_rdlock(&m->lk);
    n = m->hot_n;
    ids = (i32 *)malloc((size_t)(n ? n : 1) * sizeof(i32));
    memcpy(ids, m->hot, (size_t)n * sizeof(i32));
    pthread_rwlock_unlock(&m->lk);
    fired = mem_consolidate_ids(m, rules_src, ids, n);
    free(ids);
    return fired;
}
