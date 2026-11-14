/* mem.c - a memory layer on top of vecdb: lifetime/decay, metamemory, event->fact->concept
 * consolidation, and a small rule engine to drive it. C89, includes vecdb.c.
 *
 * SINGLE SOURCE OF TRUTH: the Event. Every fact and concept is a REBUILDABLE, LOSSY
 * materialization over some set of events - never itself an independent source. This is the
 * same discipline vecdb.c already applies one level down (the graph, BM25 index and vectors
 * are all rebuilt from note text on load, never trusted as primary) - mem.c just extends it
 * one level up: events are vecdb notes with kind=MK_EVENT; a fact or concept is a vecdb note
 * whose PROVENANCE (MemMeta.prov[]) names the events (or facts, for a concept) it was derived
 * from. Delete every fact and concept and re-run mem_consolidate() over the events and you get
 * back (up to the engine's own determinism) the same facts and concepts - nothing is lost that
 * was not already implied by the events. A fact/concept's own text is EXTRACTIVE (the text of
 * its best-supporting source, not a synthesised summary - this engine has no language model to
 * summarise with; say so plainly rather than pretend otherwise).
 *
 * THE THREE LEVELS (event > fact > concept), same graph, same vector space, same BM25 index as
 * plain vecdb notes - MemKind is metadata bolted on the side (MemMeta, keyed by vecdb node id),
 * not a new VNode.kind, so vecdb.c itself is untouched:
 *   event   - a raw observation: one note as added, unreviewed, decays fastest.
 *   fact    - promoted from one or more corroborating events; decays slower; holds them as
 *             provenance (backlinks in the vecdb graph too, since promotion links the fact's
 *             text with `[[the event's title]]`, so vecdb_backlinks already answers "why do we
 *             believe this").
 *   concept - promoted from one or more corroborating facts; decays slowest; the thing closest
 *             to "settled knowledge" this engine has.
 *
 * LIFETIME / DECAY: relevance(vid, now) = exp(-lambda_kind * idle_seconds) * (1 + 0.15 *
 * corroboration) - 0.1 * contradiction, clamped to [0,1]; lambda_kind = ln(2) / half_life_kind
 * (mem_set_lifetime configures the half-life per kind; events default to 7 days, facts to 90,
 * concepts to 720). idle_seconds is time since the last access OR corroboration, i.e. every
 * touch resets the clock - "used and reinforced knowledge does not fade", the rest does.
 *
 * METAMEMORY: confidence(vid) = relevance * (corroboration+1) / (corroboration+contradiction+1)
 * - low with no support, falls as contradictions accumulate relative to support, never exceeds
 * relevance. mem_known() answers "have I got this, and how sure am I" INCLUDING THE HONEST
 * ANSWER "no" (MEM_UNKNOWN) when nothing in the index is even a candidate - the one metamemory
 * signal this engine can give with no self-deception, since it is grounded in the same hybrid
 * search already tested end to end. mem_gaps() lists established facts/concepts held with thin
 * evidence - "what I have an opinion on but shouldn't be trusted on", not "what I have never
 * heard of" (that would need a taxonomy of what SHOULD be known, which this engine does not
 * have and does not pretend to).
 *
 * HOT / WARM / COLD:
 *   hot  - a small fixed-capacity LRU of node ids (MemHot), a fast "what has this session
 *          actually been touching" signal, no vectors/text of its own (they already live in
 *          vecdb's own RAM-resident store - this hot tier is bookkeeping, not a second copy).
 *   warm - the vecdb+store default: RAM-resident vectors, on-disk journal-synced, exactly what
 *          the rest of this codebase already is. Most nodes live here.
 *   cold - mem_archive() appends a node's full record (title, text, MemMeta) to an append-only,
 *          fsynced PATH.mem.cold log and removes it from the active vecdb graph; mem_recall()
 *          reads a record back and re-adds it as a fresh event. A real, tested round trip, not
 *          a stub - and it is the one place data can still be lost on purpose (mem_forget on an
 *          unsupported node deletes outright rather than archiving; see its own comment).
 *
 * OPERATIONS: split, merge, prune, forget, fix - see each function's own comment; consolidate
 * (the "hippocampus" pass - what a sleep cycle does to hippocampal memory: replay, reinforce,
 * generalise, let the rest fade) drives split(never automatic - a human/caller decision) aside,
 * all of merge/prune/promote/archive through the rule engine below.
 *
 * RULE ENGINE: a small Lisp/Datalog-flavoured forward-chaining engine over s-expressions -
 * conditions read a node's kind/relevance/corroboration/contradiction/access_count/age/idle
 * (plus `(similar T)`, a real same-kind nearest-neighbour search), actions are `(promote fact)`,
 * `(promote concept)`, `(merge)`, `(prune)`, `(archive)`. See mem_consolidate's own comment for
 * the exact grammar and an example rule file. Decay itself is NOT a rule action - relevance is
 * continuous (computed from elapsed time whenever asked), so a rule reacts to the decayed value
 * rather than causing decay.
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -Werror -D_POSIX_C_SOURCE=200112L -O2 -march=native \
 *       -Isrc tests/mem_test.c -lm -lpthread -o mem_t && ./mem_t
 */
#ifndef MEM_C
#define MEM_C

#include "vecdb.c"
#include <time.h>
#include <math.h>

typedef enum { MK_EVENT = 0, MK_FACT = 1, MK_CONCEPT = 2 } MemKind;

typedef struct {
    i32     vid;            /* the vecdb node id this metadata is about; -1 = a free/dead slot */
    MemKind kind;
    i64     born_at;        /* unix seconds */
    i64     last_touch;     /* unix seconds: last access or corroboration - the decay clock */
    i32     access_count;
    i32     corroboration;  /* independent supporting events/facts (see mem_corroborate) */
    i32     contradiction;  /* explicit contradictions (see mem_contradict) */
    i32    *prov;           /* vecdb node ids this was derived from / supported by */
    i32     nprov, provcap;
    u8      archived;       /* 1: this vid no longer exists in the live vecdb graph (cold) */
    u8      promoted;       /* 1: a `promote` action has already fired for this node (see mem_consolidate) -
                              * without this, an event that already created or reinforced a fact would just
                              * keep re-reinforcing the same fact (or a moved-on one) on every later pass */
} MemMeta;

typedef struct {
    VecDB   *db;
    H64      byvid;          /* vecdb node id -> index into meta[] */
    MemMeta *meta;
    i32      n, cap;
    f32      half_life[3];   /* seconds, by MemKind - mem_set_lifetime */
    i32     *hot;             /* MRU-ordered ring of vids, hot[0] most recent */
    i32      hot_n, hot_cap;
    FILE    *cf;              /* PATH.mem.cold - append-only, opened by mem_open/mem_save */
    char    *cpath;
    pthread_rwlock_t lk;
} Mem;

/* ------------------------------------------------------------------ */
/* construction, the side-table, hot tier                              */
/* ------------------------------------------------------------------ */

static i64 mem_now_(void) { return (i64)time((time_t *)0); }

Mem *mem_open(VecDB *db)
{
    Mem *m = (Mem *)calloc(1, sizeof(Mem));
    m->db = db;
    h64_init(&m->byvid, 256);
    m->half_life[MK_EVENT] = 7.0f * 86400.0f;
    m->half_life[MK_FACT] = 90.0f * 86400.0f;
    m->half_life[MK_CONCEPT] = 720.0f * 86400.0f;
    m->hot_cap = 256;
    m->hot = (i32 *)malloc((size_t)m->hot_cap * sizeof(i32));
    pthread_rwlock_init(&m->lk, (const pthread_rwlockattr_t *)0);
    return m;
}

void mem_free(Mem *m)
{
    i32 i;
    if (!m) return;
    for (i = 0; i < m->n; ++i) free(m->meta[i].prov);
    free(m->meta); h64_free(&m->byvid); free(m->hot);
    if (m->cf) fclose(m->cf);
    free(m->cpath);
    pthread_rwlock_destroy(&m->lk);
    free(m);
}

/* half_life_seconds <= 0 leaves that kind's setting unchanged */
void mem_set_lifetime(Mem *m, MemKind kind, f32 half_life_seconds)
{
    if (half_life_seconds <= 0.0f) return;
    pthread_rwlock_wrlock(&m->lk);                                /* mem_relevance_ reads half_life[] under this same lock */
    m->half_life[(i32)kind] = half_life_seconds;
    pthread_rwlock_unlock(&m->lk);
}

/* Every "is this vid live" check in this file goes through here, so it is the one place that
 * enforces it: an archived entry is never returned, even if some future removal path forgets to
 * call h64_del (as mem_archive itself did until mstress.c caught it - a single missing h64_del
 * call there was silently defeating every liveness check in the file, since they all bottom out
 * in this function). Belt and suspenders: the byvid table SHOULD be kept in sync too, but this
 * makes that a performance detail, not a correctness requirement. */
static MemMeta *mem_get_(Mem *m, i32 vid)
{
    i32 i = h64_get(&m->byvid, (i64)vid);
    if (i < 0) return (MemMeta *)0;
    return m->meta[i].archived ? (MemMeta *)0 : &m->meta[i];
}

/* vecdb_add_note upserts by title - re-adding a title that already exists REUSES that vid rather
 * than making a new one (documented at the "promote" action's own call site, the first place this
 * bit). A caller that derives a title from something that can repeat (mem_split re-splitting the
 * same parent, two sources producing the same chunk title, ...) can therefore end up asking for a
 * meta entry on a vid that already has one live. Silently pushing a second slot for the same vid
 * would inflate the live count past vecdb's own and leave the old slot as permanent orphaned
 * bookkeeping - mstress.c caught exactly this. So: if vid already has a live entry, reset it in
 * place (fresh clock, empty provenance, this kind) instead of appending a duplicate - correct
 * either way, since vecdb having reused the vid means whatever was there before is gone too. */
static MemMeta *mem_new_meta_(Mem *m, i32 vid, MemKind kind)
{
    MemMeta *e = mem_get_(m, vid);
    i32 i;
    if (e && !e->archived) {
        free(e->prov);
        memset(e, 0, sizeof(*e));
        e->vid = vid; e->kind = kind; e->born_at = e->last_touch = mem_now_();
        return e;
    }
    if (m->n == m->cap) { m->cap = m->cap ? m->cap * 2 : 64; m->meta = (MemMeta *)realloc(m->meta, (size_t)m->cap * sizeof(MemMeta)); }
    i = m->n++;
    e = &m->meta[i];
    memset(e, 0, sizeof(*e));
    e->vid = vid; e->kind = kind; e->born_at = e->last_touch = mem_now_();
    h64_put(&m->byvid, (i64)vid, i);
    return e;
}

static void mem_prov_add_(MemMeta *e, i32 vid)
{
    i32 i;
    for (i = 0; i < e->nprov; ++i) if (e->prov[i] == vid) return;
    if (e->nprov == e->provcap) { e->provcap = e->provcap ? e->provcap * 2 : 4; e->prov = (i32 *)realloc(e->prov, (size_t)e->provcap * sizeof(i32)); }
    e->prov[e->nprov++] = vid;
}

static void mem_prov_remove_(MemMeta *e, i32 vid)
{
    i32 i;
    for (i = 0; i < e->nprov; ++i) if (e->prov[i] == vid) { e->prov[i] = e->prov[--e->nprov]; return; }
}

/* does vid resolve to something live right now - either a live meta entry, or (for a vid that was
 * never mem-tracked at all) a live vecdb note. The exact condition mem_fix enforces everywhere
 * else; used here to stop a new dangling provenance edge from being recorded in the first place
 * rather than relying on mem_fix to clean it up after the fact - a caller handing back a vid that
 * has since been pruned/merged/forgotten elsewhere (easy to do: ids get stale) must not silently
 * poison another entry's provenance list with it. */
static i32 mem_is_live_(Mem *m, i32 vid)
{
    if (mem_get_(m, vid)) return 1;
    return vid >= 0 && vid < m->db->nn && m->db->nodes[vid].kind == VN_NOTE;
}

static void mem_hot_touch_(Mem *m, i32 vid)
{
    i32 i;
    for (i = 0; i < m->hot_n; ++i) if (m->hot[i] == vid) { memmove(m->hot + 1, m->hot, (size_t)i * sizeof(i32)); m->hot[0] = vid; return; }
    if (m->hot_n < m->hot_cap) memmove(m->hot + 1, m->hot, (size_t)m->hot_n++ * sizeof(i32));
    else memmove(m->hot + 1, m->hot, (size_t)(m->hot_cap - 1) * sizeof(i32));
    m->hot[0] = vid;
}

static void mem_hot_drop_(Mem *m, i32 vid)
{
    i32 i;
    for (i = 0; i < m->hot_n; ++i) if (m->hot[i] == vid) { memmove(m->hot + i, m->hot + i + 1, (size_t)(m->hot_n - i - 1) * sizeof(i32)); --m->hot_n; return; }
}

/* the current hot set, most-recently-touched first; returns how many were copied */
i32 mem_hot_ids(const Mem *m, i32 *out, i32 cap)
{
    i32 n;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&m->lk);
    n = m->hot_n < cap ? m->hot_n : cap;
    memcpy(out, m->hot, (size_t)n * sizeof(i32));
    pthread_rwlock_unlock((pthread_rwlock_t *)&m->lk);
    return n;
}

/* Add a note as a memory item of the given kind (default entry point is MK_EVENT - see
 * mem_note_event). Returns the vecdb node id, or -1 if the embedding failed (nothing recorded). */
i32 mem_note(Mem *m, MemKind kind, const char *title, const char *text)
{
    i32 vid;
    pthread_rwlock_wrlock(&m->lk);
    vid = vecdb_add_note(m->db, title, text);
    if (vid >= 0) { mem_new_meta_(m, vid, kind); mem_hot_touch_(m, vid); }
    pthread_rwlock_unlock(&m->lk);
    return vid;
}
i32 mem_note_event(Mem *m, const char *title, const char *text) { return mem_note(m, MK_EVENT, title, text); }

/* An event (or fact) corroborates an existing node: it counts as independent support, resets
 * the decay clock, and is recorded as provenance (so vecdb_backlinks/mem_provenance can answer
 * "why do we believe this"). Symmetric mem_contradict records evidence against instead. A stale
 * from_vid (already pruned/merged/forgotten/archived elsewhere - easy for a caller to hand back
 * an id that used to be live) is a no-op rather than poisoning vid's provenance list with a
 * dangling reference: a source that no longer exists cannot count as fresh corroboration/
 * contradiction. Returns 1 if it took effect, 0 if vid or from_vid is not live. */
i32 mem_corroborate(Mem *m, i32 vid, i32 from_vid)
{
    MemMeta *e;
    i32 ok;
    pthread_rwlock_wrlock(&m->lk);
    e = mem_get_(m, vid);
    ok = e && mem_is_live_(m, from_vid);
    if (ok) { ++e->corroboration; e->last_touch = mem_now_(); mem_prov_add_(e, from_vid); mem_hot_touch_(m, vid); }
    pthread_rwlock_unlock(&m->lk);
    return ok;
}
i32 mem_contradict(Mem *m, i32 vid, i32 from_vid)
{
    MemMeta *e;
    i32 ok;
    pthread_rwlock_wrlock(&m->lk);
    e = mem_get_(m, vid);
    ok = e && mem_is_live_(m, from_vid);
    if (ok) { ++e->contradiction; e->last_touch = mem_now_(); mem_prov_add_(e, from_vid); mem_hot_touch_(m, vid); }
    pthread_rwlock_unlock(&m->lk);
    return ok;
}

/* mere access (a search hit): resets the decay clock and counts, but is not corroboration -
 * looking a fact up does not, on its own, make it more likely to be true. */
void mem_touch(Mem *m, i32 vid)
{
    MemMeta *e;
    pthread_rwlock_wrlock(&m->lk);
    e = mem_get_(m, vid);
    if (e) { ++e->access_count; e->last_touch = mem_now_(); mem_hot_touch_(m, vid); }
    pthread_rwlock_unlock(&m->lk);
}

/* vecdb_search/explore/query wrapped to touch every hit - the normal way to read memory */
i32 mem_search(Mem *m, const char *query, i32 k, VHit *out)
{
    i32 n, i;
    n = vecdb_search(m->db, query, k, 0, -1, out);
    for (i = 0; i < n; ++i) mem_touch(m, out[i].node);
    return n;
}

/* provenance of a memory item (the event/fact ids it rests on); returns how many were copied */
i32 mem_provenance(const Mem *m, i32 vid, i32 *out, i32 cap)
{
    i32 n = 0;
    MemMeta *e;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&m->lk);
    e = mem_get_((Mem *)m, vid);
    if (e) { n = e->nprov < cap ? e->nprov : cap; memcpy(out, e->prov, (size_t)n * sizeof(i32)); }
    pthread_rwlock_unlock((pthread_rwlock_t *)&m->lk);
    return n;
}

MemKind mem_kind(const Mem *m, i32 vid)
{
    MemMeta *e;
    MemKind k;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&m->lk);
    e = mem_get_((Mem *)m, vid);
    k = e ? e->kind : MK_EVENT;
    pthread_rwlock_unlock((pthread_rwlock_t *)&m->lk);
    return k;
}

/* ------------------------------------------------------------------ */
/* lifetime / decay / metamemory                                       */
/* ------------------------------------------------------------------ */

/* core formula, no lock - the caller must already hold m->lk (any mode) and have a live e */
static f32 mem_relevance_(const Mem *m, const MemMeta *e)
{
    f64 idle, lambda, base, r;
    idle = (f64)(mem_now_() - e->last_touch); if (idle < 0.0) idle = 0.0;
    lambda = log(2.0) / (f64)m->half_life[(i32)e->kind];
    base = exp(-lambda * idle);
    r = base * (1.0 + 0.15 * e->corroboration) - 0.1 * e->contradiction;
    if (r < 0.0) r = 0.0;
    if (r > 1.0) r = 1.0;
    return (f32)r;
}

/* the current, continuously-decayed relevance of a memory item, or 0 if it has none (never
 * noted, or archived/forgotten). See the file header for the formula. */
f32 mem_relevance(const Mem *m, i32 vid)
{
    MemMeta *e;
    f32 r;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&m->lk);
    e = mem_get_((Mem *)m, vid);
    r = e ? mem_relevance_(m, e) : 0.0f;
    pthread_rwlock_unlock((pthread_rwlock_t *)&m->lk);
    return r;
}

/* core formula, no lock - see mem_confidence's own comment for what it means */
static f32 mem_confidence_(const Mem *m, const MemMeta *e)
{
    f32 rel = mem_relevance_(m, e);
    if (e->kind == MK_EVENT) return rel / (1.0f + (f32)e->contradiction);
    return rel * (f32)e->corroboration / (f32)(e->corroboration + e->contradiction + 1);
}

/* metamemory: how sure this engine is that a held item is actually true.
 *   event   - direct evidence (this was observed, not inferred): trusted by default, discounted
 *             only by contradiction and by decay. corroboration does not apply - an event does
 *             not need OTHER events to back it up to be believed as an observation.
 *   fact/concept - always derived, never directly observed: trust must be EARNED by corroboration,
 *             relative to a small prior (+1) so a single unconfirmed assertion starts near 0, not
 *             near 1 just because nothing yet contradicts it.
 * Either way this never exceeds relevance - a decayed, unused memory cannot be "confident". */
f32 mem_confidence(const Mem *m, i32 vid)
{
    MemMeta *e;
    f32 c;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&m->lk);
    e = mem_get_((Mem *)m, vid);
    c = e ? mem_confidence_(m, e) : 0.0f;
    pthread_rwlock_unlock((pthread_rwlock_t *)&m->lk);
    return c;
}

typedef enum { MEM_UNKNOWN = 0, MEM_UNCERTAIN = 1, MEM_KNOWN = 2 } MemAnswer;

/* how much a hit actually pertains to the query, independent of confidence in its content -
 * vecdb_search always returns up to k hits ranked among whatever exists, even when nothing is
 * genuinely related (a raw cosine near 0 and no lexical overlap at all); this is the floor below
 * which a hit does not count as a real candidate. */
#define MEM_CANDIDATE_FLOOR 0.15f

/* The one honest metamemory question this engine can answer: given a query, is there anything in
 * the index that is even a genuine candidate for it (MEM_UNKNOWN if not - a real "I don't know",
 * requiring actual vector or lexical overlap, not just "the search returned its top-k regardless")
 * and, among candidates, how confident is the best one in its own content (>= known_at, default
 * 0.5, -> MEM_KNOWN; otherwise MEM_UNCERTAIN: "I have something on this, but do not trust it
 * much"). *out_conf, *out_vid are optional. This is NOT "what topics exist that I have never
 * heard of" - this engine has no model of the space of things it could know, only of what it has
 * actually been told. */
MemAnswer mem_known(Mem *m, const char *query, f32 known_at, f32 *out_conf, i32 *out_vid)
{
    VHit hits[8];
    i32 n = vecdb_search(m->db, query, 8, 0, -1, hits), i, best = -1;   /* outside m->lk: its own lock (db->lk), independent */
    f32 bc = -1.0f;
    if (known_at <= 0.0f) known_at = 0.5f;
    pthread_rwlock_rdlock(&m->lk);
    for (i = 0; i < n; ++i) {
        f32 c; MemMeta *e;
        if (hits[i].sim < MEM_CANDIDATE_FLOOR && hits[i].lex <= 0.0f) continue;   /* not a real candidate, just whatever ranked highest */
        e = mem_get_(m, hits[i].node);
        c = e ? mem_confidence_(m, e) : 0.0f;
        if (c > bc) { bc = c; best = hits[i].node; }
    }
    pthread_rwlock_unlock(&m->lk);
    if (out_conf) *out_conf = best >= 0 ? bc : 0.0f;
    if (out_vid) *out_vid = best;
    if (best < 0) return MEM_UNKNOWN;
    return bc >= known_at ? MEM_KNOWN : MEM_UNCERTAIN;
}

/* established facts/concepts held on thin evidence: confidence < max_conf. Returns how many
 * were copied (most-thinly-evidenced first). A real, bounded metamemory scan, not a stub. */
i32 mem_gaps(const Mem *m, f32 max_conf, i32 *out, i32 cap)
{
    i32 i, n = 0;
    f32 *sc;
    i32 *ids;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&m->lk);             /* m->n itself must be read under the lock too - a concurrent
                                                                      * grow between sizing these buffers and the loop below would
                                                                      * otherwise be a real overflow, not just a benign race */
    sc = (f32 *)malloc((size_t)(m->n ? m->n : 1) * sizeof(f32));
    ids = (i32 *)malloc((size_t)(m->n ? m->n : 1) * sizeof(i32));
    for (i = 0; i < m->n; ++i) {
        if (m->meta[i].archived || m->meta[i].kind == MK_EVENT) continue;    /* events are raw, not "held knowledge" */
        { f32 c = mem_confidence_(m, &m->meta[i]); if (c < max_conf) { ids[n] = m->meta[i].vid; sc[n] = c; ++n; } }
    }
    pthread_rwlock_unlock((pthread_rwlock_t *)&m->lk);
    { i32 j; for (i = 1; i < n; ++i) { i32 x = ids[i]; f32 xs = sc[i]; for (j = i; j > 0 && sc[j - 1] > xs; --j) { ids[j] = ids[j - 1]; sc[j] = sc[j - 1]; } ids[j] = x; sc[j] = xs; } }
    if (n > cap) n = cap;
    memcpy(out, ids, (size_t)n * sizeof(i32));
    free(sc); free(ids);
    return n;
}

/* ------------------------------------------------------------------ */
/* cold tier: an append-only archive                                   */
/* ------------------------------------------------------------------ */

/* open (or create) PATH.mem.cold for archive/recall; call once, alongside vecdb_load/vecdb_save.
 * Safe to call while other threads are using this Mem (m->cf is shared with mem_archive/
 * mem_recall/mem_cold_count under the same lock) - but a thread with a `mem_recall`d text
 * pointer or similar mid-flight will simply see the new cold file from here on, not the old
 * one; do not re-point cold storage while another thread might still be recalling from it. */
i32 mem_cold_open(Mem *m, const char *path)
{
    char *cp = (char *)malloc(strlen(path) + 10);
    i32 r;
    sprintf(cp, "%s.mem.cold", path);
    pthread_rwlock_wrlock(&m->lk);
    if (m->cf) fclose(m->cf);
    m->cf = fopen(cp, "a+b");
    free(m->cpath); m->cpath = cp;
    r = m->cf ? 0 : -1;
    pthread_rwlock_unlock(&m->lk);
    return r;
}

/* core, no lock - the caller must already hold m->lk (mem_archive does; the public wrapper below
 * takes it for a standalone call). Walks the exact layout mem_archive writes - 24-byte header,
 * title, text, nprov * 4 bytes of provenance - and stops at a corrupt or torn record. */
static i32 mem_cold_count_(Mem *m)
{
    i32 n = 0;
    i64 size, pos = 0;
    if (!m->cf) return 0;
    if (fseeko(m->cf, 0, SEEK_END) != 0 || (size = (i64)ftello(m->cf)) < 0) return 0;
    for (;;) {
        i32 hdr[6];
        i64 body;
        if (fseeko(m->cf, (off_t)pos, SEEK_SET) != 0 || fread(hdr, 4, 6, m->cf) != 6) break;   /* kind,nl,tl,nprov,corrob,contra */
        if (hdr[1] < 0 || hdr[2] < 0 || hdr[3] < 0) break;
        body = (i64)hdr[1] + (i64)hdr[2] + (i64)hdr[3] * 4;
        if (body > size - pos - 24) break;                                              /* header without its full body */
        pos += 24 + body;
        ++n;
    }
    return n;
}

/* how many records mem.cold holds (their 0-based index is the id mem_recall takes) */
i32 mem_cold_count(Mem *m)
{
    i32 n;
    pthread_rwlock_wrlock(&m->lk);                                /* moves the shared file position - not a read-only op */
    n = mem_cold_count_(m);
    pthread_rwlock_unlock(&m->lk);
    return n;
}

/* Archive a live memory item: appends its full record (title, text, kind, provenance,
 * corroboration/contradiction) to PATH.mem.cold (fsynced - this is the one place besides the
 * main vecdb files that is held to the same durability bar) and removes it from the active
 * vecdb graph (vecdb_remove_note - a ghost is left behind if anything still links to it, exactly
 * as plain vecdb already does). Every OTHER live entry's provenance list that named this vid has
 * that reference removed too (same as mem_prune/mem_forget/mem_merge already do for a node they
 * remove - mstress.c caught this one missing it: mem_forget's own "archive because something
 * depends on it" path was creating the exact dangling references mem_fix exists to clean up).
 * Returns the cold id (for mem_recall), or -1 on failure. */
i32 mem_archive(Mem *m, i32 vid)
{
    MemMeta *e;
    char *text; const char *title;
    i32 tl, nl, cid, i;
    pthread_rwlock_wrlock(&m->lk);
    if (!m->cf) { pthread_rwlock_unlock(&m->lk); return -1; }       /* read under the lock: mem_cold_open can change it concurrently */
    e = mem_get_(m, vid);
    if (!e || e->archived) { pthread_rwlock_unlock(&m->lk); return -1; }
    title = m->db->nodes[vid].name; nl = (i32)strlen(title);
    text = vecdb_text(m->db, vid); tl = text ? (i32)strlen(text) : 0;
    cid = mem_cold_count_(m);
    fseeko(m->cf, 0, SEEK_END);
    { i32 hdr[6]; hdr[0] = (i32)e->kind; hdr[1] = nl; hdr[2] = tl; hdr[3] = e->nprov; hdr[4] = e->corroboration; hdr[5] = e->contradiction;
      fwrite(hdr, 4, 6, m->cf); }
    fwrite(title, 1, (size_t)nl, m->cf);
    if (tl > 0) fwrite(text, 1, (size_t)tl, m->cf);
    for (i = 0; i < e->nprov; ++i) fwrite(&e->prov[i], 4, 1, m->cf);
    store_fsync_(m->cf);
    free(text);
    vecdb_remove_note(m->db, vid);
    for (i = 0; i < m->n; ++i) if (m->meta[i].vid != vid) mem_prov_remove_(&m->meta[i], vid);
    e->archived = 1;
    h64_del(&m->byvid, (i64)vid);                                    /* was missing - see mem_get_'s own comment on what this broke */
    mem_hot_drop_(m, vid);
    pthread_rwlock_unlock(&m->lk);
    return cid;
}

/* Bring a cold record back as a fresh event (a new vid - vecdb does not support reinserting
 * under an old, now-reused id). Its kind and corroboration/contradiction are restored; the decay
 * clock restarts (a recalled memory is, correctly, freshly touched). Provenance ids that were
 * live at archive time but are not live NOW (pruned/merged/forgotten/re-archived while this
 * record sat in cold storage - nothing could have kept them in sync during that time, unlike
 * every other removal path, which cleans up as it happens) are silently dropped rather than
 * reviving a dangling reference - mstress.c is what found that this needs doing at all.
 * Returns the new vid, or -1 if cold_id is out of range or embedding failed. */
i32 mem_recall(Mem *m, i32 cold_id)
{
    i32 i, vid = -1;
    pthread_rwlock_wrlock(&m->lk);                                /* the whole scan: it shares m->cf's file position with mem_archive,
                                                                    * and m->cf itself is read under the lock too (mem_cold_open) */
    if (!m->cf) { pthread_rwlock_unlock(&m->lk); return -1; }
    fseeko(m->cf, 0, SEEK_SET);
    for (i = 0; ; ++i) {
        i32 hdr[6]; char *title, *text; i32 *prov;
        if (fread(hdr, 4, 6, m->cf) != 6) { pthread_rwlock_unlock(&m->lk); return -1; }
        title = (char *)malloc((size_t)hdr[1] + 1); text = (char *)malloc((size_t)hdr[2] + 1); prov = (i32 *)malloc((size_t)(hdr[3] ? hdr[3] : 1) * 4);
        if (hdr[1] > 0 && fread(title, 1, (size_t)hdr[1], m->cf) != (size_t)hdr[1]) { free(title); free(text); free(prov); pthread_rwlock_unlock(&m->lk); return -1; }
        title[hdr[1]] = 0;
        if (hdr[2] > 0 && fread(text, 1, (size_t)hdr[2], m->cf) != (size_t)hdr[2]) { free(title); free(text); free(prov); pthread_rwlock_unlock(&m->lk); return -1; }
        text[hdr[2]] = 0;
        if (hdr[3] > 0 && fread(prov, 4, (size_t)hdr[3], m->cf) != (size_t)hdr[3]) { free(title); free(text); free(prov); pthread_rwlock_unlock(&m->lk); return -1; }
        if (i == cold_id) {
            i32 j;
            vid = vecdb_add_note(m->db, title, text);
            if (vid >= 0) {
                MemMeta *e = mem_new_meta_(m, vid, (MemKind)hdr[0]);
                e->corroboration = hdr[4]; e->contradiction = hdr[5];
                for (j = 0; j < hdr[3]; ++j) {
                    i32 pv = prov[j];
                    i32 okmeta = mem_get_(m, pv) != (MemMeta *)0;
                    i32 okvid = pv >= 0 && pv < m->db->nn && m->db->nodes[pv].kind == VN_NOTE;
                    if (okmeta || okvid) mem_prov_add_(e, pv);
                }
                mem_hot_touch_(m, vid);
            }
            free(title); free(text); free(prov);
            pthread_rwlock_unlock(&m->lk);
            return vid;
        }
        free(title); free(text); free(prov);
    }
}

/* ------------------------------------------------------------------ */
/* operations: split, merge, prune, forget, fix                        */
/* ------------------------------------------------------------------ */

/* An event/fact outgrows itself: split its text into up to `chunk_chars`-sized pieces (paragraph
 * boundaries preferred) as new sibling events, each linked from the parent (which is kept, not
 * replaced - "budding off" children, not destruction). Returns how many children were created;
 * their ids are written to out (may be NULL if the caller does not need them), up to cap. */
i32 mem_split(Mem *m, i32 vid, i32 chunk_chars, i32 *out, i32 cap)
{
    char *text, *body;
    i32 tlen, n = 0, pos = 0;
    const char *title;
    MemKind srckind;
    if (chunk_chars < 200) chunk_chars = 200;
    pthread_rwlock_wrlock(&m->lk);
    text = vecdb_text(m->db, vid);
    if (!text) { pthread_rwlock_unlock(&m->lk); return 0; }
    { MemMeta *se = mem_get_(m, vid); srckind = se ? se->kind : MK_EVENT; }
    tlen = (i32)strlen(text);
    title = m->db->nodes[vid].name;
    body = (char *)malloc((size_t)chunk_chars + (size_t)strlen(title) + 40);
    while (pos < tlen && (out == (i32 *)0 || n < cap) ) {
        i32 end = pos + chunk_chars < tlen ? pos + chunk_chars : tlen, cut = end, cid;
        char ct[300];
        if (end < tlen) { i32 q = end; while (q > pos + chunk_chars / 2 && !(text[q] == '\n' && text[q - 1] == '\n')) --q; if (q > pos + chunk_chars / 2) cut = q; }
        sprintf(ct, "%.200s \xc2\xa7%d", title, n + 1);
        sprintf(body, "[[%.200s]]\n\n", title);
        { i32 bl = (i32)strlen(body), take = cut - pos; if (take > chunk_chars) take = chunk_chars; memcpy(body + bl, text + pos, (size_t)take); body[bl + take] = 0; }
        cid = vecdb_add_note(m->db, ct, body);
        if (cid >= 0) { mem_new_meta_(m, cid, srckind); mem_hot_touch_(m, cid); if (out) out[n] = cid; ++n; }
        pos = cut;
    }
    free(body); free(text);
    pthread_rwlock_unlock(&m->lk);
    return n;
}

/* Merge two same-purpose nodes into one: the survivor is the higher-relevance of the two (ties
 * favour the lower vid); the loser's provenance is unioned in, corroboration/contradiction
 * summed, and every OTHER node's provenance list that named the loser is repointed at the
 * survivor (no dangling provenance after a merge - mem_fix would otherwise have to clean this
 * up later). The loser is removed via vecdb_remove_note (a ghost if still linked). Returns the
 * survivor's vid, or -1 if either id is not a known memory item. */
i32 mem_merge(Mem *m, i32 a, i32 b)
{
    MemMeta *ea, *eb, *win, *lose;
    i32 loser_vid, i, survivor;
    pthread_rwlock_wrlock(&m->lk);
    if (a == b) { i32 r = mem_is_live_(m, a) ? a : -1; pthread_rwlock_unlock(&m->lk); return r; }   /* merging something with itself:
        * a no-op, not "archive and delete the only copy" - mem_get_(a) and mem_get_(b) would be the SAME pointer below, so
        * win==lose and the rest of this function would archive/delete the node it was asked to keep (mstress.c found this) */
    ea = mem_get_(m, a); eb = mem_get_(m, b);
    if (!ea || !eb) { pthread_rwlock_unlock(&m->lk); return -1; }
    { f32 ra = mem_relevance_(m, ea), rb = mem_relevance_(m, eb);
      if (rb > ra || (rb == ra && b < a)) { win = eb; lose = ea; survivor = b; loser_vid = a; }
      else { win = ea; lose = eb; survivor = a; loser_vid = b; } }
    for (i = 0; i < lose->nprov; ++i) mem_prov_add_(win, lose->prov[i]);
    win->corroboration += lose->corroboration; win->contradiction += lose->contradiction;
    if (lose->last_touch > win->last_touch) win->last_touch = lose->last_touch;
    for (i = 0; i < m->n; ++i) if (m->meta[i].vid != loser_vid) mem_prov_remove_(&m->meta[i], loser_vid);
    mem_prov_remove_(win, loser_vid);
    lose->archived = 1;                                           /* drop the meta slot's usefulness; vid no longer resolves via byvid after removal below */
    h64_del(&m->byvid, (i64)loser_vid);
    mem_hot_drop_(m, loser_vid);
    vecdb_remove_note(m->db, loser_vid);
    mem_hot_touch_(m, survivor);
    pthread_rwlock_unlock(&m->lk);
    return survivor;
}

/* Natural decay's garbage collector: removes any UNSUPPORTED (corroboration == 0) memory item
 * whose relevance has fallen below min_relevance. Supported items are never pruned this way,
 * however low their relevance - something else depends on them; use mem_forget for that case
 * explicitly. Returns how many were removed. */
i32 mem_prune(Mem *m, f32 min_relevance)
{
    i32 i, removed = 0;
    i32 *victims;
    i32 nv = 0;
    pthread_rwlock_wrlock(&m->lk);                                /* m->n read under the lock: see mem_gaps' comment on the same pattern */
    victims = (i32 *)malloc((size_t)(m->n ? m->n : 1) * sizeof(i32));
    for (i = 0; i < m->n; ++i) {
        MemMeta *e = &m->meta[i];
        if (e->archived || e->corroboration > 0) continue;
        if (mem_relevance_(m, e) < min_relevance) victims[nv++] = e->vid;
    }
    for (i = 0; i < nv; ++i) {
        i32 vid = victims[i], j;
        MemMeta *e = mem_get_(m, vid);
        if (!e) continue;
        for (j = 0; j < m->n; ++j) if (m->meta[j].vid != vid) mem_prov_remove_(&m->meta[j], vid);
        e->archived = 1; h64_del(&m->byvid, (i64)vid); mem_hot_drop_(m, vid);
        vecdb_remove_note(m->db, vid);
        ++removed;
    }
    free(victims);
    pthread_rwlock_unlock(&m->lk);
    return removed;
}

/* Explicit, immediate forgetting (as opposed to prune's natural decay): if anything still
 * depends on this item (corroboration > 0, or another item's provenance names it) it is
 * archived - kept as an auditable cold record, "we chose to let this go" rather than silently
 * losing what rested on it; otherwise it is deleted outright (mem_archive with no cold file
 * open falls back to a hard delete too - forgetting must not silently no-op). Returns 1 if
 * archived, 0 if deleted, -1 if vid is not a known memory item. */
i32 mem_forget(Mem *m, i32 vid)
{
    MemMeta *e;
    i32 supported = 0, i;
    pthread_rwlock_wrlock(&m->lk);
    e = mem_get_(m, vid);
    if (!e) { pthread_rwlock_unlock(&m->lk); return -1; }
    supported = e->corroboration > 0;
    for (i = 0; i < m->n && !supported; ++i) { i32 j; for (j = 0; j < m->meta[i].nprov; ++j) if (m->meta[i].prov[j] == vid) { supported = 1; break; } }
    pthread_rwlock_unlock(&m->lk);
    /* m->cf is not read here - mem_archive checks it itself, under its own lock (mem_cold_open
     * can change it concurrently). If archiving fails (no cold storage configured, or vid was
     * removed by another thread between the checks above and here), fall through to the hard
     * delete below rather than silently doing nothing - forgetting must always have an effect. */
    if (supported && mem_archive(m, vid) >= 0) return 1;
    pthread_rwlock_wrlock(&m->lk);
    for (i = 0; i < m->n; ++i) if (m->meta[i].vid != vid) mem_prov_remove_(&m->meta[i], vid);
    e = mem_get_(m, vid);
    if (e) { e->archived = 1; h64_del(&m->byvid, (i64)vid); mem_hot_drop_(m, vid); }
    vecdb_remove_note(m->db, vid);
    pthread_rwlock_unlock(&m->lk);
    return 0;
}

/* Consistency repair: (a) meta entries whose vid is no longer a live vecdb note (removed some
 * other way, bypassing mem.c) are dropped; (b) provenance ids that name neither a live meta
 * entry nor a live vecdb node are dropped (a dangling reference); (c) corroboration/contradiction
 * are clamped to >= 0; (d) the hot list is purged of any id that is no longer live. Returns how
 * many individual issues were fixed. */
i32 mem_fix(Mem *m)
{
    i32 i, fixed = 0;
    pthread_rwlock_wrlock(&m->lk);
    for (i = 0; i < m->n; ++i) {
        MemMeta *e = &m->meta[i];
        i32 live = !e->archived && e->vid >= 0 && e->vid < m->db->nn && m->db->nodes[e->vid].kind == VN_NOTE;
        if (!live && !e->archived) { e->archived = 1; h64_del(&m->byvid, (i64)e->vid); mem_hot_drop_(m, e->vid); ++fixed; }
    }
    for (i = 0; i < m->n; ++i) {
        MemMeta *e = &m->meta[i]; i32 j;
        if (e->archived) continue;
        for (j = 0; j < e->nprov; ) {
            i32 pv = e->prov[j];
            i32 okmeta = mem_get_(m, pv) != (MemMeta *)0;
            i32 okvid = pv >= 0 && pv < m->db->nn && m->db->nodes[pv].kind == VN_NOTE;
            if (!okmeta && !okvid) { e->prov[j] = e->prov[--e->nprov]; ++fixed; } else ++j;
        }
        if (e->corroboration < 0) { e->corroboration = 0; ++fixed; }
        if (e->contradiction < 0) { e->contradiction = 0; ++fixed; }
    }
    { i32 w = 0;
      for (i = 0; i < m->hot_n; ++i) { MemMeta *e = mem_get_(m, m->hot[i]); if (e && !e->archived) m->hot[w++] = m->hot[i]; else ++fixed; }
      m->hot_n = w; }
    pthread_rwlock_unlock(&m->lk);
    return fixed;
}

#include "mem_rules.c"

#include "mem_disk.c"

#endif /* MEM_C */
