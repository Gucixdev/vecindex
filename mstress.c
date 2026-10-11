/* mstress.c - hammer mem.c with a long random churn of every operation and check real invariants
 * hold throughout, not just after a repair pass. Mirrors stress.c's role for store.c/engine.c one
 * layer up: not a demo (mem.c's own -DMEM_DEMO covers that), not a timing benchmark (mbench.c
 * does), a correctness stress test with an oracle-grade check: mem_fix must find ZERO issues at
 * every checkpoint, because if the ops (merge/prune/forget/archive/consolidate) are implemented
 * correctly they should never let an invariant slip in the first place - mem_fix finding
 * something here would mean one of them has a real bug, not that the "repair" step is doing its
 * job as intended.
 *   gcc -std=c89 -pedantic -Wall -Wextra -Werror -D_POSIX_C_SOURCE=200112L -O2 -march=native \
 *       -I. mstress.c -DMSTRESS -lm -lpthread -o mstress && ./mstress
 */
#include "mem.c"
#include <stdio.h>
#include <time.h>

static i32 g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  FAIL: %s\n", msg); ++g_fail; } } while (0)

static double now_(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec; }
static u32 xs_(u32 *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }

/* every invariant mem.c's own comments promise, checked directly against the live structures -
 * not just "mem_fix returns 0" (checked separately, at the call site, since running mem_fix
 * itself mutates state and we want to know it found nothing BEFORE it would have fixed it) */
static void check_invariants_(Mem *m, const char *phase)
{
    i32 i, live = 0, hot_dup_or_dead = 0;
    i32 wlcap = m->db->nn > 16 ? m->db->nn : 16;   /* upper bound on any one tag's note list - never truncated */
    i32 *wl = (i32 *)malloc((size_t)wlcap * sizeof(i32));
    for (i = 0; i < m->n; ++i) {
        MemMeta *e = &m->meta[i];
        i32 j;
        if (e->archived) continue;
        ++live;
        CHECK(e->vid >= 0 && e->vid < m->db->nn && m->db->nodes[e->vid].kind == VN_NOTE, "live meta entry's vid is not a live vecdb note");
        CHECK(e->corroboration >= 0 && e->contradiction >= 0, "negative corroboration/contradiction on a live entry");
        { f32 r = mem_relevance(m, e->vid), c = mem_confidence(m, e->vid);
          CHECK(r >= 0.0f && r <= 1.0f, "relevance out of [0,1]");
          CHECK(c >= 0.0f && c <= 1.0f && c <= r + 1e-4f, "confidence out of [0,1] or exceeds relevance"); }
        for (j = 0; j < e->nprov; ++j) {
            i32 pv = e->prov[j];
            i32 okmeta = mem_get_(m, pv) != (MemMeta *)0;
            i32 okvid = pv >= 0 && pv < m->db->nn && m->db->nodes[pv].kind == VN_NOTE;
            CHECK(okmeta || okvid, "a live entry's provenance names neither a live meta entry nor a live vecdb node");
        }
        /* wikilinks/tags: vecdb's own graph must stay symmetric under all this mem.c churn -
         * A -[[link]]-> B implies B lists A in its backlinks, and A #tag implies A shows up in
         * that tag's note list, from both directions (vecdb_links/_tags_of are the out edges,
         * vecdb_backlinks/_notes_with_tag the in edges vdb_link_ is supposed to keep matched) */
        { i32 lk[16], nlk = vecdb_links(m->db, e->vid, lk, 16), q;
          for (q = 0; q < nlk; ++q) {
              i32 tgt = lk[q];
              if (m->db->nodes[tgt].kind != VN_NOTE) continue;   /* ghosts have no backlink list to check */
              { i32 nbl = vecdb_backlinks(m->db, tgt, wl, wlcap), z, found = 0;   /* wl reused: sized to never truncate */
                for (z = 0; z < nbl; ++z) if (wl[z] == e->vid) found = 1;
                CHECK(found, "A links to note B but B's backlinks do not list A"); }
          }
        }
        { i32 tg[16], ntg = vecdb_tags_of(m->db, e->vid, tg, 16), q;
          for (q = 0; q < ntg; ++q) {
              i32 nwl = vecdb_notes_with_tag(m->db, tg[q], wl, wlcap), z, found = 0;
              for (z = 0; z < nwl; ++z) if (wl[z] == e->vid) found = 1;
              CHECK(found, "A has tag T but T's note list does not list A");
          }
        }
    }
    free(wl);
    CHECK(live == vecdb_count(m->db, VN_NOTE), "live meta-entry count != vecdb's own live note count (mem.c was the only writer here)");
    { i32 seen[300], ns = 0;
      CHECK(m->hot_n <= m->hot_cap, "hot list over capacity");
      for (i = 0; i < m->hot_n; ++i) {
          MemMeta *e = mem_get_(m, m->hot[i]);
          i32 j, dup = 0;
          if (!e || e->archived) hot_dup_or_dead = 1;
          for (j = 0; j < ns; ++j) if (seen[j] == m->hot[i]) dup = 1;
          if (dup) hot_dup_or_dead = 1; else if (ns < 300) seen[ns++] = m->hot[i];
      }
      CHECK(!hot_dup_or_dead, "hot list has a dead or duplicate id"); }
    printf("  [%s] live=%d meta_total=%d hot=%d\n", phase, live, m->n, m->hot_n);
}

int main(void)
{
    static Arena arena;
    VecDB *db = vecdb_new(&arena, 48);
    Mem *m = mem_open(db);
    u32 st = 0xC0FFEEu;
    i32 round, live_ids_cap = 20000, *live_ids = (i32 *)malloc((size_t)live_ids_cap * sizeof(i32)), nlive = 0;
    double t0 = now_();
    i32 rounds = 60;

    mem_cold_open(m, "/tmp/mstress.vdb");
    printf("mstress: %d churn rounds\n", rounds);

    for (round = 0; round < rounds; ++round) {
        i32 ops = 80 + (i32)(xs_(&st) % 120u), k;
        for (k = 0; k < ops; ++k) {
            u32 pick = xs_(&st) % 100u;
            if (pick < 45 || nlive < 3) {                                     /* 45%: add an event, maybe linking a live one, always tagged */
                char t[48], x[192]; i32 vid, tag = (i32)(xs_(&st) % 12u);
                sprintf(t, "n%d_%d", round, k);
                if (nlive > 0 && (xs_(&st) & 3) == 0) {
                    i32 tgt = live_ids[xs_(&st) % (u32)nlive];
                    sprintf(x, "body %d links [[%s]] word%d #topic%d", k, db->nodes[tgt].name, (i32)(xs_(&st) % 30), tag);
                } else sprintf(x, "body %d standalone word%d word%d #topic%d", k, (i32)(xs_(&st) % 30), (i32)(xs_(&st) % 30), tag);
                vid = mem_note_event(m, t, x);
                if (vid >= 0 && nlive < live_ids_cap) live_ids[nlive++] = vid;
            } else if (pick < 55 && nlive > 1) {                              /* corroborate */
                i32 a = live_ids[xs_(&st) % (u32)nlive], b = live_ids[xs_(&st) % (u32)nlive];
                mem_corroborate(m, a, b);
            } else if (pick < 60 && nlive > 1) {                              /* contradict */
                i32 a = live_ids[xs_(&st) % (u32)nlive], b = live_ids[xs_(&st) % (u32)nlive];
                mem_contradict(m, a, b);
            } else if (pick < 66 && nlive > 1) {                              /* merge - remove the loser from live_ids by rebuilding is expensive; just leave stale ids, mem_get_ handles it */
                i32 a = live_ids[xs_(&st) % (u32)nlive], b = live_ids[xs_(&st) % (u32)nlive];
                mem_merge(m, a, b);
            } else if (pick < 72) {                                          /* prune (natural decay) */
                mem_prune(m, 0.001f + (f32)(xs_(&st) % 100) / 1000.0f);
            } else if (pick < 78 && nlive > 0) {                             /* forget (explicit) */
                mem_forget(m, live_ids[xs_(&st) % (u32)nlive]);
            } else if (pick < 84 && nlive > 0) {                             /* archive + sometimes recall right back */
                i32 vid = live_ids[xs_(&st) % (u32)nlive], cid = mem_archive(m, vid);
                if (cid >= 0 && (xs_(&st) & 1)) { i32 back = mem_recall(m, cid); if (back >= 0 && nlive < live_ids_cap) live_ids[nlive++] = back; }
            } else if (pick < 90) {                                          /* split a random live note */
                if (nlive > 0) { i32 kids[4]; mem_split(m, live_ids[xs_(&st) % (u32)nlive], 220, kids, 4); }
            } else if (pick < 92) {                                          /* consolidation scoped to the hot set - cheap, frequent */
                i32 fired = mem_consolidate_hot(m, "(rule promote-facts (and (kind event) (similar 0.5)) (promote fact 0.5))\n"
                                                    "(rule promote-concepts (and (kind fact) (>= corroboration 1)) (promote concept 0.5))\n"
                                                    "(rule prune-decayed (and (< relevance 0.02) (= corroboration 0)) (prune))\n");
                (void)fired;
            } else if (pick < 94 && nlive > 0) {                            /* consolidation scoped to an explicit (possibly-stale) working set */
                i32 ids[8], nids = 0, q;
                for (q = 0; q < 8 && q < nlive; ++q) ids[nids++] = live_ids[xs_(&st) % (u32)nlive];
                { i32 fired = mem_consolidate_ids(m, "(rule promote-facts (and (kind event) (similar 0.5)) (promote fact 0.5))\n"
                                                      "(rule promote-concepts (and (kind fact) (>= corroboration 1)) (promote concept 0.5))\n"
                                                      "(rule prune-decayed (and (< relevance 0.02) (= corroboration 0)) (prune))\n",
                                                   ids, nids);
                  (void)fired; }
            } else if (pick < 96) {                                          /* a full consolidation pass, every so often */
                i32 fired = mem_consolidate(m, "(rule promote-facts (and (kind event) (similar 0.5)) (promote fact 0.5))\n"
                                                "(rule promote-concepts (and (kind fact) (>= corroboration 1)) (promote concept 0.5))\n"
                                                "(rule prune-decayed (and (< relevance 0.02) (= corroboration 0)) (prune))\n");
                (void)fired;
            } else {                                                        /* touch / search */
                char q[32]; sprintf(q, "word%d body", (i32)(xs_(&st) % 30)); { VHit h[5]; mem_search(m, q, 5, h); }
            }
        }
        if (round % 150000 == 0) vecdb_build(db);                            /* keep the vector index reasonably fresh without dominating the loop cost */
        check_invariants_(m, "mid-churn");
        { i32 fixed = mem_fix(m); CHECK(fixed == 0, "mem_fix found something to fix after normal churn - one of the ops left an invariant broken"); }
    }
    printf("churn done in %.2f s\n\n", now_() - t0);

    /* -------- save/load at the end, then the same invariants must hold on the reloaded copy -------- */
    {
        i32 live_before = 0, i;
        for (i = 0; i < m->n; ++i) if (!m->meta[i].archived) ++live_before;
        CHECK(vecdb_save(db, "/tmp/mstress.vdb") == 0, "vecdb_save failed");
        CHECK(mem_save(m, "/tmp/mstress.mem") == 0, "mem_save failed");
        {
            static Arena a2;
            VecDB *db2 = vecdb_load(&a2, "/tmp/mstress.vdb");
            Mem *m2;
            CHECK(db2 != (VecDB *)0, "vecdb_load failed after a full churn run");
            if (db2) {
                m2 = mem_open(db2);
                CHECK(mem_load(m2, "/tmp/mstress.mem") == 0, "mem_load failed");
                CHECK(m2->n == live_before, "reloaded live entry count != what was saved");
                vecdb_build(db2);
                check_invariants_(m2, "after save/load");
                { i32 fixed = mem_fix(m2); CHECK(fixed == 0, "mem_fix found something on the reloaded copy"); }
                mem_free(m2); vecdb_free(db2);
            }
        }
    }

    printf("\nmstress: %d rounds, %d final live notes, %s\n", rounds, vecdb_count(db, VN_NOTE), g_fail ? "FAILURES" : "all invariants held");
    remove("/tmp/mstress.vdb"); remove("/tmp/mstress.vdb.vdt"); remove("/tmp/mstress.vdb.vxi"); remove("/tmp/mstress.vdb.vxi.j");
    remove("/tmp/mstress.vdb.mem.cold"); remove("/tmp/mstress.mem");
    free(live_ids);
    mem_free(m); vecdb_free(db);
    return g_fail ? 1 : 0;
}
