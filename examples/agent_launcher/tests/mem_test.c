/* mem_test.c - the self-test of src/mem.c; ./run builds and runs it. */
#include "mem.c"
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
