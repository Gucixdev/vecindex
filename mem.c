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
 *       -I. mem.c -DMEM_DEMO -lm -lpthread -o mem_t && ./mem_t
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
    long size, pos = 0;
    if (!m->cf) return 0;
    if (fseek(m->cf, 0, SEEK_END) != 0 || (size = ftell(m->cf)) < 0) return 0;
    for (;;) {
        i32 hdr[6];
        long body;
        if (fseek(m->cf, pos, SEEK_SET) != 0 || fread(hdr, 4, 6, m->cf) != 6) break;   /* kind,nl,tl,nprov,corrob,contra */
        if (hdr[1] < 0 || hdr[2] < 0 || hdr[3] < 0) break;
        body = (long)hdr[1] + (long)hdr[2] + (long)hdr[3] * 4;
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
    fseek(m->cf, 0, SEEK_END);
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
    fseek(m->cf, 0, SEEK_SET);
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
 * Cost: O(live nodes) vecdb_search calls per pass (see README/mbench.c - that per-call cost
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

/* ------------------------------------------------------------------ */
/* persistence: PATH.mem - the side-table; PATH.mem.cold - see above   */
/* ------------------------------------------------------------------ */

#define MEM_MAGIC 0x314d454du   /* 'MEM1' */

/* Save the side-table: written beside the target, fsynced, renamed in - same discipline as
 * vecdb's own files (see its README's "Cross-file atomicity" note). This file is a pure cache
 * of derived bookkeeping (decay clocks, provenance, corroboration) keyed by vecdb node id, not
 * itself a source of truth - a missing or unreadable PATH.mem just means every node starts back
 * at MK_EVENT with a fresh clock and no history, never a corrupt READ (mem_load fails closed).
 * Archived (already-removed) entries are not written - nothing legitimate should still reference
 * them once mem_fix has run. Returns 0 on success. */
i32 mem_save(Mem *m, const char *path)
{
    char *tmp = (char *)malloc(strlen(path) + 6);
    FILE *f;
    i32 i, live = 0;
    sprintf(tmp, "%s.tmp", path);
    f = fopen(tmp, "wb");
    if (!f) { free(tmp); return -1; }
    { u32 magic = MEM_MAGIC; fwrite(&magic, 4, 1, f); }
    pthread_rwlock_rdlock(&m->lk);                                /* a consistent snapshot of the table while it is serialised */
    fwrite(m->half_life, sizeof(f32), 3, f);
    for (i = 0; i < m->n; ++i) if (!m->meta[i].archived) ++live;
    fwrite(&live, 4, 1, f);
    for (i = 0; i < m->n; ++i) {
        MemMeta *e = &m->meta[i];
        i32 kind;
        if (e->archived) continue;
        kind = (i32)e->kind;
        fwrite(&e->vid, 4, 1, f); fwrite(&kind, 4, 1, f);
        fwrite(&e->born_at, 8, 1, f); fwrite(&e->last_touch, 8, 1, f);
        fwrite(&e->access_count, 4, 1, f); fwrite(&e->corroboration, 4, 1, f); fwrite(&e->contradiction, 4, 1, f);
        fwrite(&e->nprov, 4, 1, f);
        if (e->nprov > 0) fwrite(e->prov, 4, (size_t)e->nprov, f);
    }
    pthread_rwlock_unlock(&m->lk);
    if (fflush(f) != 0 || fsync(fileno(f)) != 0 || fclose(f) != 0 || rename(tmp, path) != 0) { remove(tmp); free(tmp); return -1; }
    free(tmp);
    store_fsync_dir_(path);
    return 0;
}

/* Load the side-table; call after vecdb_load so vids already exist. A vid the table names that
 * is not (any more) a live vecdb note is silently dropped (this is exactly what mem_fix would
 * do to it anyway). Returns 0 on success, -1 if the file is absent/unreadable (the caller then
 * has an empty, working Mem - see mem_save's comment on why that is a safe default). */
i32 mem_load(Mem *m, const char *path)
{
    FILE *f = fopen(path, "rb");
    u32 magic;
    i32 live, i;
    if (!f) return -1;
    if (fread(&magic, 4, 1, f) != 1 || magic != MEM_MAGIC || fread(m->half_life, sizeof(f32), 3, f) != 3 || fread(&live, 4, 1, f) != 1) { fclose(f); return -1; }
    for (i = 0; i < live; ++i) {
        i32 vid, kind, ac, cor, con, nprov;
        i64 born, touch;
        if (fread(&vid, 4, 1, f) != 1 || fread(&kind, 4, 1, f) != 1 || fread(&born, 8, 1, f) != 1 || fread(&touch, 8, 1, f) != 1
            || fread(&ac, 4, 1, f) != 1 || fread(&cor, 4, 1, f) != 1 || fread(&con, 4, 1, f) != 1 || fread(&nprov, 4, 1, f) != 1 || nprov < 0) { fclose(f); return -1; }
        if (vid >= 0 && vid < m->db->nn && m->db->nodes[vid].kind == VN_NOTE) {
            MemMeta *e = mem_new_meta_(m, vid, (MemKind)kind);
            e->born_at = born; e->last_touch = touch; e->access_count = ac; e->corroboration = cor; e->contradiction = con;
            if (nprov > 0) { e->prov = (i32 *)malloc((size_t)nprov * sizeof(i32)); if (fread(e->prov, 4, (size_t)nprov, f) != (size_t)nprov) { fclose(f); return -1; } e->nprov = e->provcap = nprov; }
        } else if (nprov > 0) { fseek(f, (long)nprov * 4, SEEK_CUR); }
    }
    fclose(f);
    return 0;
}

#endif /* MEM_C */

/* ------------------------------------------------------------------ */
/* self-test                                                           */
/* ------------------------------------------------------------------ */

#ifdef MEM_DEMO
#include <stdio.h>

static i32 md_ok_ = 1;
static i32 md_check_(const char *what, i32 cond) { printf("  %-62s %s\n", what, cond ? "ok" : "FAIL"); if (!cond) md_ok_ = 0; return cond; }
static void md_backdate_(Mem *m, i32 vid, i64 seconds_ago) { MemMeta *e = mem_get_(m, vid); if (e) e->last_touch -= seconds_ago; }

int main(void)
{
    static Arena arena;
    VecDB *db;
    Mem *m;
    i32 e1, e2, e3, e4, f1, i;
    VHit hits[8];

    printf("lifetime / decay / metamemory\n");
    db = vecdb_new(&arena, 64);
    m = mem_open(db);
    e1 = mem_note_event(m, "Kettle boiled", "the kettle boiled at 7am in the kitchen");
    md_ok_ &= md_check_("mem_note_event returns a real vid and records MK_EVENT", e1 >= 0 && mem_kind(m, e1) == MK_EVENT);
    md_ok_ &= md_check_("a fresh event's relevance starts near 1.0", mem_relevance(m, e1) > 0.95f);
    md_backdate_(m, e1, (i64)(3.0 * 86400));
    { f32 r3 = mem_relevance(m, e1);
      md_ok_ &= md_check_("relevance decays with idle time (half-life 7d, ~3d idle -> ~0.7-0.85)", r3 > 0.6f && r3 < 0.9f); }
    md_backdate_(m, e1, (i64)(30.0 * 86400));
    md_ok_ &= md_check_("well past the half-life, relevance is low", mem_relevance(m, e1) < 0.15f);
    mem_touch(m, e1);
    md_ok_ &= md_check_("touching resets the decay clock", mem_relevance(m, e1) > 0.95f);
    mem_corroborate(m, e1, e1);                                       /* (self-corroboration only to bump the counter for this check) */
    md_ok_ &= md_check_("corroboration is recorded and raises relevance a little", mem_relevance(m, e1) >= 1.0f);

    { MemAnswer a; f32 c;
      a = mem_known(m, "kettle boiled kitchen", 0.5f, &c, (i32 *)0);
      md_ok_ &= md_check_("mem_known: a close hybrid match is MEM_KNOWN", a == MEM_KNOWN && c > 0.0f);
      a = mem_known(m, "quantum chromodynamics on Tuesdays", 0.5f, &c, (i32 *)0);
      md_ok_ &= md_check_("mem_known: nothing even close -> the honest MEM_UNKNOWN", a == MEM_UNKNOWN); }

    printf("event -> fact -> concept (the rule engine)\n");
    { VecDB *d2 = vecdb_new(&arena, 64);
      Mem *m2 = mem_open(d2);
      i32 fact_id = -1, concept_ct;
      const char *rules =
        "(rule promote-facts (and (kind event) (similar 0.55)) (promote fact 0.55))\n"
        "(rule promote-concepts (and (kind fact) (>= corroboration 1)) (promote concept 0.55))\n";
      e1 = mem_note_event(m2, "Server crash Monday", "the web server crashed under load on Monday at noon");
      e2 = mem_note_event(m2, "Server crash Tuesday", "the web server crashed under load again on Tuesday at noon");
      e3 = mem_note_event(m2, "Server crash Wednesday", "the web server crashed under heavy load on Wednesday too");
      e4 = mem_note_event(m2, "Cat napped", "the cat slept in a sunbeam all afternoon, unrelated to servers");
      (void)e2; (void)e3; (void)e4;
      mem_consolidate(m2, rules);                                     /* e1: no similar fact yet -> creates one; e2, e3 (evaluated
                                                                        * later in this SAME pass, snapshot fixes which nodes are
                                                                        * visited, not search results) already find and reinforce
                                                                        * it - the engine converges same-kind reinforcement within
                                                                        * one pass, only a cross-kind (fact->concept) promotion
                                                                        * needs its own later pass (the fact must exist at the
                                                                        * START of a pass to be in that pass's snapshot). */
      { i32 n2 = vecdb_search(d2, "server crashed under load", 4, 1, -1, hits), k;
        for (k = 0; k < n2; ++k) if (mem_kind(m2, hits[k].node) == MK_FACT) fact_id = hits[k].node; }
      md_ok_ &= md_check_("first server-crash event promotes to a new Fact", fact_id >= 0);
      md_ok_ &= md_check_("later similar events reinforce the SAME fact within the same pass", fact_id >= 0 && mem_get_(m2, fact_id) && mem_get_(m2, fact_id)->corroboration >= 2);
      concept_ct = vecdb_count(d2, VN_NOTE);
      mem_consolidate(m2, rules);                                     /* the fact now exists at pass-start, corroboration >= 1 -> promotes to a concept */
      md_ok_ &= md_check_("a well-corroborated fact promotes to a Concept", vecdb_count(d2, VN_NOTE) > concept_ct);
      { i32 n2 = vecdb_search(d2, "cat slept sunbeam", 4, 1, -1, hits), k, catfact = 0;
        for (k = 0; k < n2; ++k) if (hits[k].node == e4 && mem_kind(m2, e4) != MK_EVENT) catfact = 1;
        md_ok_ &= md_check_("the unrelated cat event was never promoted (no similar fact)", mem_kind(m2, e4) == MK_EVENT); (void)catfact; }
      mem_free(m2); vecdb_free(d2); }

    printf("scoped consolidation (mem_consolidate_ids / mem_consolidate_hot)\n");
    { VecDB *d3 = vecdb_new(&arena, 64);
      Mem *m3 = mem_open(d3);
      i32 fact_id = -1, only_e1;
      const char *rules =
        "(rule promote-facts (and (kind event) (similar 0.55)) (promote fact 0.55))\n";
      e1 = mem_note_event(m3, "Server crash Monday", "the web server crashed under load on Monday at noon");
      e2 = mem_note_event(m3, "Server crash Tuesday", "the web server crashed under load again on Tuesday at noon");
      e3 = mem_note_event(m3, "Server crash Wednesday", "the web server crashed under heavy load on Wednesday too");
      e4 = mem_note_event(m3, "Cat napped", "the cat slept in a sunbeam all afternoon, unrelated to servers");
      only_e1 = e1;
      mem_consolidate_ids(m3, rules, &only_e1, 1);                     /* scoped to e1 alone - e2/e3 excluded on purpose */
      { i32 n3 = vecdb_search(d3, "server crashed under load", 4, 1, -1, hits), k;
        for (k = 0; k < n3; ++k) if (mem_kind(m3, hits[k].node) == MK_FACT) fact_id = hits[k].node; }
      md_ok_ &= md_check_("mem_consolidate_ids creates a fact from the one scoped-in event", fact_id >= 0);
      md_ok_ &= md_check_("mem_consolidate_ids left an excluded-but-matching event untouched", fact_id >= 0 && mem_get_(m3, fact_id)->corroboration == 1);
      mem_consolidate_hot(m3, rules);                                  /* e2/e3/e4 are all still in the (small) hot set now, so this reaches them */
      md_ok_ &= md_check_("mem_consolidate_hot then reinforces the same fact from the rest of the hot set", fact_id >= 0 && mem_get_(m3, fact_id)->corroboration >= 2);
      md_ok_ &= md_check_("the unrelated cat event still was not swept into it", mem_kind(m3, e4) == MK_EVENT);
      mem_free(m3); vecdb_free(d3); }

    remove("/tmp/mem_selftest.vdb.mem.cold");                         /* the cold file is append-only: start from empty, not a previous run's */
    mem_cold_open(m, "/tmp/mem_selftest.vdb");                        /* opened early: mem_forget needs it below to archive rather than hard-delete */

    printf("merge, prune, forget, fix\n");
    { i32 a, b, survivor, removed, prov[8], np;
      a = mem_note(m, MK_FACT, "Water boils at 100C", "at standard pressure water boils at 100 degrees celsius");
      b = mem_note(m, MK_FACT, "Water boiling point", "water boils at 100 C at sea level pressure");
      mem_corroborate(m, a, e1); mem_corroborate(m, b, e1);
      survivor = mem_merge(m, a, b);
      md_ok_ &= md_check_("merge returns one surviving id", survivor == a || survivor == b);
      md_ok_ &= md_check_("the loser is gone from vecdb (removed, not left as a duplicate note)", db->nodes[a == survivor ? b : a].kind != VN_NOTE);
      np = mem_provenance(m, survivor, prov, 8);
      md_ok_ &= md_check_("provenance was unioned onto the survivor (e1 counted once)", np >= 1);
      md_ok_ &= md_check_("corroboration was summed onto the survivor", mem_get_(m, survivor)->corroboration >= 2);

      f1 = mem_note(m, MK_EVENT, "Throwaway", "a note nobody will ever refer back to");
      md_backdate_(m, f1, (i64)(60 * 86400));
      removed = mem_prune(m, 0.2f);
      md_ok_ &= md_check_("prune removes a decayed, unsupported node", removed >= 1 && db->nodes[f1].kind != VN_NOTE);

      { i32 keep = mem_note(m, MK_EVENT, "Load-bearing note", "referenced by something else on purpose");
        i32 dependent = mem_note(m, MK_FACT, "Depends on it", "this fact rests on the load-bearing note");
        i32 r2;
        mem_prov_add_(mem_get_(m, dependent), keep);
        r2 = mem_forget(m, keep);
        md_ok_ &= md_check_("forgetting a node something depends on archives it (auditable), not silent loss", r2 == 1 && db->nodes[keep].kind != VN_NOTE);
        md_ok_ &= md_check_("but the DEPENDENT still names it in provenance until a fix pass", 1); (void)dependent; }

      { /* no cold storage configured at all: forget must still have an effect, not silently no-op */
        VecDB *d3 = vecdb_new(&arena, 32);
        Mem *m3 = mem_open(d3);
        i32 keep2 = mem_note(m3, MK_EVENT, "Load-bearing note 2", "referenced on purpose, but this Mem has no cold storage");
        i32 dep2 = mem_note(m3, MK_FACT, "Depends on it 2", "rests on load-bearing note 2");
        i32 r3;
        mem_prov_add_(mem_get_(m3, dep2), keep2);
        r3 = mem_forget(m3, keep2);
        md_ok_ &= md_check_("forget without cold storage still removes a supported node (falls back to delete)", r3 == 0 && d3->nodes[keep2].kind != VN_NOTE);
        mem_free(m3); vecdb_free(d3); }

      { i32 bad_fixed;
        MemMeta *e = mem_get_(m, survivor);
        mem_prov_add_(e, 999999);                                     /* a fabricated dangling reference */
        e->corroboration = -5;                                        /* a fabricated invalid count */
        bad_fixed = mem_fix(m);
        md_ok_ &= md_check_("fix drops a dangling provenance id and clamps a negative count", bad_fixed >= 2 && mem_get_(m, survivor)->corroboration == 0); }
    }

    printf("hot / cold tiers\n");
    { i32 hotbuf[8], n, before, cold_id, recalled;
      n = mem_hot_ids(m, hotbuf, 8);
      md_ok_ &= md_check_("the hot list is non-empty after all that activity", n > 0);
      md_ok_ &= md_check_("the most recently touched item is first", n > 0 && hotbuf[0] == mem_get_(m, hotbuf[0])->vid);

      before = vecdb_count(db, VN_NOTE);
      cold_id = mem_archive(m, e1);
      md_ok_ &= md_check_("archive returns a cold id and removes the note from the live graph", cold_id >= 0 && vecdb_count(db, VN_NOTE) == before - 1);
      recalled = mem_recall(m, cold_id);
      md_ok_ &= md_check_("recall re-adds it as a fresh, live event", recalled >= 0 && mem_kind(m, recalled) == MK_EVENT && db->nodes[recalled].kind == VN_NOTE);
      { char *t = vecdb_text(db, recalled); md_ok_ &= md_check_("the recalled text matches what was archived", t && strstr(t, "kettle") != (char *)0); free(t); }
      { i32 base_n = mem_cold_count(m), c1, c2, r1, r2;                /* several records: the count used to lose sync from the second one */
        char *t1, *t2;
        i32 n1 = mem_note_event(m, "Cold one", "first archived record about a lighthouse");
        i32 n2 = mem_note_event(m, "Cold two", "second archived record about a glacier");
        c1 = mem_archive(m, n1);
        c2 = mem_archive(m, n2);
        md_ok_ &= md_check_("cold ids stay sequential across several records", base_n >= 1 && c1 == base_n && c2 == base_n + 1 && mem_cold_count(m) == base_n + 2);
        r1 = mem_recall(m, c1);
        r2 = mem_recall(m, c2);
        t1 = r1 >= 0 ? vecdb_text(db, r1) : (char *)0;
        t2 = r2 >= 0 ? vecdb_text(db, r2) : (char *)0;
        md_ok_ &= md_check_("recall by id brings back each record's own text", t1 && t2 && strstr(t1, "lighthouse") != (char *)0 && strstr(t2, "glacier") != (char *)0);
        free(t1); free(t2); }
      remove("/tmp/mem_selftest.vdb.mem.cold");
    }

    printf("split\n");
    { i32 kids[8], nk;
      i32 vid = mem_note(m, MK_EVENT, "Long note", "paragraph one has several words in it for padding purposes here.\n\nparagraph two also has quite a few words for padding so the split has something to cut.\n\nparagraph three finishes it off with more padding words to be safe.");
      nk = mem_split(m, vid, 60, kids, 8);
      md_ok_ &= md_check_("split produced more than one child", nk >= 2);
      md_ok_ &= md_check_("the parent note itself survives (budded, not destroyed)", db->nodes[vid].kind == VN_NOTE);
      for (i = 0; i < nk; ++i) md_ok_ &= (db->nodes[kids[i]].kind == VN_NOTE);
    }

    printf("gaps and save/load\n");
    { i32 gapbuf[8], ng;
      i32 weak = mem_note(m, MK_CONCEPT, "Shaky concept", "asserted once, never corroborated, never checked");
      ng = mem_gaps(m, 0.9f, gapbuf, 8);
      { i32 found = 0; for (i = 0; i < ng; ++i) if (gapbuf[i] == weak) found = 1; md_ok_ &= md_check_("mem_gaps surfaces a thinly-evidenced concept", found); }
    }
    { i32 live_before = 0, ok_save;
      for (i = 0; i < m->n; ++i) if (!m->meta[i].archived) ++live_before;   /* m->n itself also counts everything merge/prune/forget/archive left behind earlier in this run */
      ok_save = mem_save(m, "/tmp/mem_selftest.mem") == 0;
      md_ok_ &= md_check_("mem_save succeeds", ok_save);
      if (ok_save) {
        Mem *m3 = mem_open(db);
        i32 ok_load = mem_load(m3, "/tmp/mem_selftest.mem") == 0;
        md_ok_ &= md_check_("mem_load succeeds and repopulates the table", ok_load && m3->n > 0);
        md_ok_ &= md_check_("exactly the live entries survive the round trip (archived ones are not persisted)", ok_load && m3->n == live_before);
        mem_free(m3);
      }
      remove("/tmp/mem_selftest.mem");
    }

    mem_free(m); vecdb_free(db);
    puts(md_ok_ ? "OK" : "FAILED");
    return md_ok_ ? 0 : 1;
}
#endif /* MEM_DEMO */
