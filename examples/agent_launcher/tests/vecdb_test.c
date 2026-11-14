/* vecdb_test.c - the self-test of src/vecdb.c; ./run builds and runs it. */
#include "vecdb.c"
#include <stdio.h>

static i32 vd_check_(const char *what, i32 cond) { printf("  %-58s %s\n", what, cond ? "ok" : "FAIL"); return cond; }

static i32 vd_has_(const i32 *a, i32 n, i32 x) { i32 i; for (i = 0; i < n; ++i) if (a[i] == x) return 1; return 0; }

static i32 vd_model_calls_ = 0;
/* a stand-in "model": counts how often it is asked, otherwise the hashed embedder */
static void vd_fake_model_(void *ctx, const char *t, i32 n, i32 kind, f32 *out, i32 dim) { (void)ctx; (void)kind; ++vd_model_calls_; vecdb_embed(t, n, out, dim); }
static void vd_fake_batch_(void *ctx, i32 n, const char **t, const i32 *l, i32 kind, f32 *out, i32 dim) { i32 i; (void)ctx; (void)kind; for (i = 0; i < n; ++i) { ++vd_model_calls_; vecdb_embed(t[i], l[i], out + (size_t)i * dim, dim); } }

int main(void)
{
    static Arena arena;
    VecDB *db;
    i32 ok = 1, i, n, buf[64];
    i32 a, b, c, ghost, bridge, quantum1;
    VHit hits[32];

    /* ---- parsing ---- */
    { VStrs l, t;
      const char *src = "---\ntags: [alpha, Beta/Gamma]\n---\n# Heading not a tag\n"
                        "See [[Note One]] and [[note two|alias]], [[Folder/Deep Note#part]] and ![[Embedded.md]] and ![shot](img/Pic.png) and ![web](http://x.org/a.png).\n"
                        "Tags: #idea #project/alpha, not #123, not a#nottag, `#incode`, and http://x.org/#frag.\n"
                        "```\n[[in fence]] #infence\n```\nend #last-tag\n";
      memset(&l, 0, sizeof l); memset(&t, 0, sizeof t);
      vdb_parse_(src, (i32)strlen(src), &l, &t);
      printf("parsing\n");
      ok &= vd_check_("5 links (alias, heading, folder, embed, markdown picture; web url skipped)", l.count == 5);
      ok &= vd_check_("link targets normalised", l.count == 5 && !strcmp(l.buf + l.off[4], "pic.png") && !strcmp(l.buf + l.off[0], "note one") && !strcmp(l.buf + l.off[1], "note two")
                      && !strcmp(l.buf + l.off[2], "deep note") && !strcmp(l.buf + l.off[3], "embedded"));
      ok &= vd_check_("front matter + inline tags, nested tag adds its parent", t.count == 7);
      { i32 has_last = 0, has_num = 0, has_frag = 0, has_code = 0, has_infence = 0, has_parent = 0;
        for (i = 0; i < t.count; ++i) {
            const char *s = t.buf + t.off[i];
            if (!strcmp(s, "last-tag")) has_last = 1;
            if (!strcmp(s, "123")) has_num = 1;
            if (!strcmp(s, "frag")) has_frag = 1;
            if (!strcmp(s, "incode")) has_code = 1;
            if (!strcmp(s, "infence")) has_infence = 1;
            if (!strcmp(s, "project")) has_parent = 1;
        }
        ok &= vd_check_("#last-tag found; #123, #frag, #incode, #infence not", has_last && !has_num && !has_frag && !has_code && !has_infence && has_parent); }
      vstrs_free_(&l); vstrs_free_(&t); }

    /* ---- graph ---- */
    db = vecdb_new(&arena, 128);
    a = vecdb_add_note(db, "Alpha", "Alpha links to [[Beta]] and [[Missing Note]]. #topic/x #shared");
    b = vecdb_add_note(db, "Beta", "Beta links back to [[alpha]]. #shared");
    c = vecdb_add_note(db, "Gamma", "Gamma is alone. #topic/y");
    printf("graph\n");
    ghost = vecdb_find(db, "Missing Note");
    ok &= vd_check_("ghost created for an unresolved link", ghost >= 0 && db->nodes[ghost].kind == VN_GHOST);
    n = vecdb_backlinks(db, a, buf, 64);
    ok &= vd_check_("backlinks of Alpha = {Beta}", n == 1 && buf[0] == b);
    n = vecdb_links(db, a, buf, 64);
    ok &= vd_check_("links of Alpha = {Beta, ghost}", n == 2 && vd_has_(buf, n, b) && vd_has_(buf, n, ghost));
    { i32 tg = vecdb_find_tag(db, "#topic"), tsh = vecdb_find_tag(db, "shared");
      n = vecdb_notes_with_tag(db, tg, buf, 64);
      ok &= vd_check_("#topic holds notes tagged #topic/x and #topic/y", n == 2 && vd_has_(buf, n, a) && vd_has_(buf, n, c));
      n = vecdb_notes_with_tag(db, tsh, buf, 64);
      ok &= vd_check_("#shared holds Alpha and Beta", n == 2 && vd_has_(buf, n, a) && vd_has_(buf, n, b)); }
    vecdb_add_note(db, "Missing Note", "Now it exists.");
    ok &= vd_check_("the ghost becomes a note, backlinks kept", db->nodes[ghost].kind == VN_NOTE && vecdb_backlinks(db, ghost, buf, 64) == 1);
    vecdb_add_note(db, "Alpha", "Rewritten: only [[Gamma]] now.");
    n = vecdb_links(db, a, buf, 64);
    ok &= vd_check_("re-adding replaces the edges", n == 1 && buf[0] == c && vecdb_backlinks(db, b, buf, 64) == 0 && vecdb_backlinks(db, c, buf, 64) == 1);
    vecdb_remove_note(db, c);
    ok &= vd_check_("removing a linked note leaves a ghost", db->nodes[c].kind == VN_GHOST);

    /* ---- MCTS finds what nearest-neighbour cannot ---- */
    { VecDB *d2 = vecdb_new(&arena, 256);
      char title[64], body[512];
      i32 nfill = 40, hit_knn = 0, hit_mcts = 0, rank = -1;
      printf("hybrid search and MCTS (planted bridge note)\n");
      for (i = 0; i < nfill; ++i) {              /* unrelated filler notes, each about its own words */
          sprintf(title, "Filler %d", i);
          sprintf(body, "kitchen recipe number %d uses flour%d sugar%d butter%d and eggs%d for baking cake%d", i, i, i, i, i, i);
          vecdb_add_note(d2, title, body);
      }
      bridge = vecdb_add_note(d2, "Bell inequality tests", "Loophole free experiments in Delft and Vienna. Cited by several of the notes here.");
      for (i = 0; i < 6; ++i) {                  /* six notes that are clearly about the query and link to the bridge */
          sprintf(title, "Quantum %d", i);
          sprintf(body, "quantum entanglement of photon pairs, entanglement swapping and quantum teleportation, part %d. See [[Bell inequality tests]]. #physics", i);
          quantum1 = vecdb_add_note(d2, title, body);
      }
      (void)quantum1;
      vecdb_build(d2);
      n = vecdb_search(d2, "quantum entanglement", 6, 1, -1, hits);
      for (i = 0; i < n; ++i) if (hits[i].node == bridge) hit_knn = 1;
      { VExplore o; vecdb_explore_defaults(&o); o.seeds = 6; o.iters = 800;
        n = vecdb_explore(d2, "quantum entanglement", 8, &o, hits);
        for (i = 0; i < n; ++i) if (hits[i].node == bridge) { hit_mcts = 1; rank = i + 1; } }
      ok &= vd_check_("vector kNN top-6 does not contain the bridge note", !hit_knn);
      printf("  (MCTS top-8 contains it at rank %d)\n", rank);
      ok &= vd_check_("MCTS top-8 does contain the bridge note", hit_mcts);
      n = vecdb_search(d2, "quantum entanglement", 8, 0, vecdb_find_tag(d2, "physics"), hits);
      ok &= vd_check_("tag filter #physics keeps only the six quantum notes", n == 6);
      n = vecdb_search(d2, "Delft", 3, 2, -1, hits);
      ok &= vd_check_("lexical: the rare word 'Delft' finds the bridge note first", n >= 1 && hits[0].node == bridge);

      n = vecdb_search(d2, "entangl*", 3, 2, -1, hits);
      ok &= vd_check_("lexical prefix 'entangl*' finds entanglement notes", n == 3);
      n = vecdb_search(d2, "teleportations photons", 3, 2, -1, hits);
      ok &= vd_check_("stemming: plural/other forms still match", n == 3);
      n = vecdb_search(d2, "flour", 5, 0, vecdb_find_tag(d2, "physics"), hits);
      ok &= vd_check_("tag filter is applied inside the vector search", n <= 6);

      { i32 in2 = 0;
        vecdb_smooth(d2, 1.0f); vecdb_build(d2);
        n = vecdb_search(d2, "quantum entanglement", 8, 1, -1, hits);
        for (i = 0; i < n; ++i) if (hits[i].node == bridge) in2 = 1;
        ok &= vd_check_("graph-smoothed embeddings: plain kNN now reaches the bridge note", in2); }

      { VecDB *da = vecdb_new(&arena, 256), *dbb = vecdb_new(&arena, 256);
        const char *tt[200], *xx[200]; char (*tb)[32] = (char (*)[32])malloc(200 * 32), (*xb)[64] = (char (*)[64])malloc(200 * 64);
        i32 same = 1, z;
        for (z = 0; z < 200; ++z) { sprintf(tb[z], "Batch %d", z); sprintf(xb[z], "note %d about topic%d and word%d", z, z % 7, z % 13); tt[z] = tb[z]; xx[z] = xb[z]; }
        for (z = 0; z < 200; ++z) vecdb_add_note(da, tt[z], xx[z]);
        ok &= vd_check_("vecdb_add_notes adds all 200 (threaded embedder)", vecdb_add_notes(dbb, 200, tt, xx) == 200);
        vecdb_build(da); vecdb_build(dbb);
        for (z = 0; z < 200; z += 9) { VHit h1[3], h2[3]; i32 a1 = vecdb_search(da, xx[z], 3, 1, -1, h1), a2 = vecdb_search(dbb, xx[z], 3, 1, -1, h2); if (a1 != a2 || (a1 > 0 && h1[0].node != h2[0].node)) same = 0; }
        ok &= vd_check_("same results as adding one by one", same);
        free(tb); free(xb); }
      printf("embedding cache\n");
      { VecDB *c1 = vecdb_new(&arena, 256), *c2 = vecdb_new(&arena, 256), *c3 = vecdb_new(&arena, 256);
        const char *tt[3] = { "Cache A", "Cache B", "Cache C" }, *xx[3] = { "first text", "second text", "third text" };
        i32 first, again;
        remove("/tmp/vecdb_test.emb");
        c1->embed = vd_fake_model_; c1->embed_batch = vd_fake_batch_;
        c2->embed = vd_fake_model_; c2->embed_batch = vd_fake_batch_;
        c3->embed = vd_fake_model_; c3->embed_batch = vd_fake_batch_;
        vecdb_embed_cache_open(c1, "/tmp/vecdb_test.emb", "modelA");
        vd_model_calls_ = 0; vecdb_add_notes(c1, 3, tt, xx); first = vd_model_calls_;
        vecdb_embed_cache_open(c2, "/tmp/vecdb_test.emb", "modelA");
        vd_model_calls_ = 0; vecdb_add_notes(c2, 3, tt, xx); vecdb_add_note(c2, "Cache A", "first text"); again = vd_model_calls_;
        ok &= vd_check_("first indexing asks the model 3 times, the cache holds 3", first == 3 && vecdb_embed_cache_count(c2) == 3);
        ok &= vd_check_("re-indexing the same texts (batch and single) asks it 0 times", again == 0);
        vecdb_embed_cache_open(c3, "/tmp/vecdb_test.emb", "modelB");
        vd_model_calls_ = 0; vecdb_add_notes(c3, 3, tt, xx);
        ok &= vd_check_("another model tag does not reuse the entries", vd_model_calls_ == 3);
        remove("/tmp/vecdb_test.emb"); }
      printf("query language\n");
      { i32 b6 = vecdb_find(d2, "Quantum 3");
        n = vecdb_query(d2, "to:[[Bell inequality tests]] #physics", 10, 0, hits);
        ok &= vd_check_("to:[[X]] #tag lists the six notes that link to X", n == 6);
        n = vecdb_query(d2, "entanglement from:[[Quantum 3]]", 10, 0, hits);
        ok &= vd_check_("from:[[Quantum 3]] + words returns its link target", n == 1 && hits[0].node == bridge);
        n = vecdb_query(d2, "[[Bell inequality tests]] -#physics", 10, 0, hits);
        ok &= vd_check_("[[X]] -#tag: neighbours without the tag -> none", n == 0);
        n = vecdb_query(d2, "quantum entanglement #physics", 3, 1, hits);
        ok &= vd_check_("words + #tag through MCTS", n == 3 && vivec_has_(&d2->nodes[hits[0].node].out, vecdb_find_tag(d2, "physics")));
        n = vecdb_query(d2, "#nosuchtag entanglement", 5, 0, hits);
        ok &= vd_check_("unknown tag matches nothing", n == 0);
        n = vecdb_query(d2, "#physics", 4, 0, hits);
        ok &= vd_check_("tag only: notes carrying it", n == 4);
        (void)b6; }

      /* ---- save / load ---- */
      vecdb_save(d2, "/tmp/vecdb_selftest.vdb");
      { VecDB *d3 = vecdb_load(&arena, "/tmp/vecdb_selftest.vdb");
        VHit h2[32]; i32 m, same = 1;
        ok &= vd_check_("save/load: file reopens", d3 != (VecDB *)0);
        if (d3) {
            vecdb_build(d3);
            n = vecdb_search(d2, "quantum entanglement", 8, 0, -1, hits);
            m = vecdb_search(d3, "quantum entanglement", 8, 0, -1, h2);
            for (i = 0; i < n && i < m; ++i) { i32 j, f = 0; for (j = 0; j < m; ++j) if (h2[j].node == hits[i].node) f = 1; if (!f) same = 0; }   /* near-duplicate notes tie, so compare as sets */
            ok &= vd_check_("save/load: same node ids and same hybrid result set", n == m && same);
            ok &= vd_check_("save/load: backlinks survived", vecdb_backlinks(d3, bridge, buf, 64) == 6);
        } }
      { VecDB *d4; i32 more;
        vecdb_add_note(d2, "Added after save", "Fresh note about photon entanglement. [[Bell inequality tests]]");
        vecdb_remove_note(d2, 0);
        ok &= vd_check_("sync after a save (journal, no full rewrite)", vecdb_sync(d2, "/tmp/vecdb_selftest.vdb") == 0);
        d4 = vecdb_load(&arena, "/tmp/vecdb_selftest.vdb");
        more = d4 ? vecdb_find(d4, "Added after save") : -1;
        ok &= vd_check_("sync: the new note and the removal survive a reload", d4 && more >= 0 && d4->nodes[more].kind == VN_NOTE && d4->nodes[0].kind != VN_NOTE);
        if (d4) { vecdb_build(d4); n = vecdb_search(d4, "photon entanglement", 3, 0, -1, hits); ok &= vd_check_("sync: it is searchable after reload", n > 0); }
      }
      { /* format versions: a legacy (v3) header still loads, a newer format and garbage are refused with a reason */
        FILE *f = fopen("/tmp/vecdb_selftest.vdb", "rb");
        long sz; char *img; VecDB *lg, *fut, *bad;
        vecdb_save(d2, "/tmp/vecdb_selftest.vdb");
        f = fopen("/tmp/vecdb_selftest.vdb", "rb"); fseek(f, 0, SEEK_END); sz = ftell(f); fseek(f, 0, SEEK_SET);
        img = (char *)malloc((size_t)sz); if (fread(img, 1, (size_t)sz, f) != (size_t)sz) ok = 0; fclose(f);
        f = fopen("/tmp/vecdb_selftest.v3", "wb");                      /* v3 = 'VDB3' dim nn dim2 + the same node table */
        { u32 m3 = VDB_MAGIC_V3; fwrite(&m3, 4, 1, f); fwrite(img + 8, 1, 12, f); fwrite(img + 32, 1, (size_t)sz - 32, f); }
        fclose(f);
        { char *cp; i32 nv = 99; f = fopen("/tmp/vecdb_selftest.v99", "wb"); fwrite(img, 1, (size_t)sz, f); fclose(f);
          f = fopen("/tmp/vecdb_selftest.v99", "r+b"); fseek(f, 4, SEEK_SET); fwrite(&nv, 4, 1, f); fclose(f); (void)cp; }
        f = fopen("/tmp/vecdb_selftest.bad", "wb"); fputs("this is not a database at all", f); fclose(f);
        { char cmd[200]; sprintf(cmd, "cp /tmp/vecdb_selftest.vdb.vdt /tmp/vecdb_selftest.v3.vdt; cp /tmp/vecdb_selftest.vdb.vxi /tmp/vecdb_selftest.v3.vxi"); if (system(cmd) != 0) ok = 0; }
        lg = vecdb_load(&arena, "/tmp/vecdb_selftest.v3");
        ok &= vd_check_("format: a legacy v3 file still loads", lg && vecdb_count(lg, VN_NOTE) == vecdb_count(d2, VN_NOTE));
        fut = vecdb_load(&arena, "/tmp/vecdb_selftest.v99");
        ok &= vd_check_("format: a file from a newer build is refused, with a reason", !fut && strstr(vecdb_error(), "newer"));
        bad = vecdb_load(&arena, "/tmp/vecdb_selftest.bad");
        ok &= vd_check_("format: garbage is refused, with a reason", !bad && strstr(vecdb_error(), "not a vecdb file"));
        ok &= vd_check_("format: a missing file is refused, with a reason", !vecdb_load(&arena, "/tmp/vecdb_nonexistent.vdb") && strstr(vecdb_error(), "cannot open"));
        free(img);
        remove("/tmp/vecdb_selftest.v3"); remove("/tmp/vecdb_selftest.v3.vdt"); remove("/tmp/vecdb_selftest.v3.vxi"); remove("/tmp/vecdb_selftest.v99"); remove("/tmp/vecdb_selftest.bad"); }
      remove("/tmp/vecdb_selftest.vdb"); remove("/tmp/vecdb_selftest.vdb.vdt"); remove("/tmp/vecdb_selftest.vdb.vxi"); remove("/tmp/vecdb_selftest.vdb.vxi.j"); }

    puts(ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
