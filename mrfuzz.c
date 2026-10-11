/* mrfuzz.c - random input against mem.c's rule parser/evaluator (sx_read_, mem_rules_parse_,
 * sx_eval_bool_, mem_consolidate's action dispatch), the one piece of mem.c's own attack surface
 * that took real text as input and had never been fuzzed (see README's "Honest gaps in mem.c").
 * Mirrors vfuzz.c's role for vecdb.c's own parser one layer down: junk tokens (a mix of real
 * rule-DSL vocabulary and garbage, so most rounds parse into SOMETHING rather than nothing) fed
 * through the reader and then through a live mem_consolidate against a small real Mem, under
 * ASan/UBSan. A prior version of sx_read_ recursed once per '(' nesting level with no bound and
 * crashed (stack overflow) on a few hundred thousand nested parens - trivial to construct, not
 * exotic - found by hand while writing this file, fixed with SX_MAX_DEPTH before this fuzzer
 * existed to keep re-finding it; it stays as the deep-nesting round below so a regression here
 * would be caught immediately.
 *   gcc -std=gnu99 -g -O1 -fsanitize=address,undefined -I. mrfuzz.c -lm -lpthread -o .build/mrfuzz && .build/mrfuzz [rounds]
 */
#include "mem.c"
#include <stdio.h>

static u32 r_ = 424242u;
static u32 rnd_(void) { return vdb_rng_(&r_); }

/* a mix of real rule-DSL vocabulary (so parses often produce well-formed-looking rules that then
 * exercise sx_eval_bool_/the action dispatch, not just the reader) and pure junk (mismatched
 * parens, bad numbers, unknown symbols, wrong arities, wrong types in field position) */
static void junk_rule_(char *b, i32 cap)
{
    static const char *tok[] = {
        "(", ")", "rule", "and", "or", "not", "kind", "event", "fact", "concept", "similar",
        "promote", "merge", "prune", "archive", "<", "<=", ">", ">=", "=", "!=",
        "relevance", "corroboration", "contradiction", "access_count", "age", "idle",
        "0.55", "0.85", "-1", "1e300", "99999999999999999999", "nan", "inf", "--5", "1.2.3",
        "+", "-", ".", "r1", "r2", "x", "\xc3\xa9\xc3\xb1", "\xff\xfe", ";comment\n", "\n", " ",
        "\t", "()", "((((", "))))", "\"quoted\"", "foo bar baz", "0", "1"
    };
    i32 w = 0;
    while (w < cap - 32) {
        const char *t = tok[rnd_() % (sizeof tok / sizeof *tok)];
        i32 l = (i32)strlen(t);
        if (w + l >= cap - 1) break;
        memcpy(b + w, t, (size_t)l); w += l; b[w++] = ' ';
    }
    b[w] = 0;
}

int main(int argc, char **argv)
{
    static Arena arena;
    VecDB *db = vecdb_new(&arena, 48);
    Mem *m = mem_open(db);
    i32 rounds = argc > 1 ? atoi(argv[1]) : 5000, i;
    char rulebuf[2048];

    /* a handful of real, live nodes so (similar T)/promote/merge/prune/archive actions that a
     * garbage-but-well-formed-enough rule happens to trigger have something real to act on,
     * instead of every round being a no-op on an empty graph */
    mem_note_event(m, "Server crash Monday", "the web server crashed under load on Monday at noon");
    mem_note_event(m, "Server crash Tuesday", "the web server crashed under load again on Tuesday");
    mem_note_event(m, "Cat napped", "the cat slept in a sunbeam all afternoon");
    vecdb_build(db);
    mem_cold_open(m, "/tmp/mrfuzz.vdb");

    for (i = 0; i < rounds; ++i) {
        junk_rule_(rulebuf, (i32)sizeof rulebuf);
        /* both the standalone parse (exercises sx_read_/mem_rules_parse_/mem_rules_free_ alone,
         * cheap, most of the rounds) and, less often, a real mem_consolidate pass (also drives
         * sx_eval_bool_ and every action's dispatch with whatever this round's rules turned out
         * to mean - real coverage of the evaluator, not just the reader) */
        { i32 nrules; MemRule *rules = mem_rules_parse_(rulebuf, (i32)strlen(rulebuf), &nrules); mem_rules_free_(rules, nrules); }
        if (i % 4 == 0) mem_consolidate(m, rulebuf);
        if (i % 500 == 0) { vecdb_build(db); mem_note_event(m, "filler", "keeps the graph from ever going fully empty"); }
    }

    /* deep-nesting round: this is exactly what crashed the pre-fix sx_read_ (unbounded recursion
     * on '(' nesting) - kept here, not just in the one-off repro that found it, so a regression
     * in SX_MAX_DEPTH (or its removal) fails this test immediately instead of waiting to be
     * hand-found again */
    { i32 n = 500000, j, w = 0;
      char *deep = (char *)malloc((size_t)n * 2 + 32);
      for (j = 0; j < n; ++j) deep[w++] = '(';
      for (j = 0; j < n; ++j) deep[w++] = ')';
      deep[w] = 0;
      { i32 nrules; MemRule *rules = mem_rules_parse_(deep, w, &nrules); mem_rules_free_(rules, nrules); }
      mem_consolidate(m, deep);
      free(deep); }

    /* unbalanced variants (opens with no closes, closes with no opens, deeply nested with no
     * closes at all) - different code paths than the balanced deep-nesting case above */
    { i32 n = 500000, j, w = 0;
      char *un = (char *)malloc((size_t)n + 32);
      for (j = 0; j < n; ++j) un[w++] = '(';
      un[w] = 0;
      { i32 nrules; MemRule *rules = mem_rules_parse_(un, w, &nrules); mem_rules_free_(rules, nrules); }
      w = 0; for (j = 0; j < n; ++j) un[w++] = ')'; un[w] = 0;
      { i32 nrules; MemRule *rules = mem_rules_parse_(un, w, &nrules); mem_rules_free_(rules, nrules); }
      free(un); }

    /* the empty file, whitespace-only, comment-only - the "nothing here" edge, both ends */
    { i32 nrules; MemRule *rules;
      rules = mem_rules_parse_("", 0, &nrules); mem_rules_free_(rules, nrules);
      rules = mem_rules_parse_("   \n\t  \n", 8, &nrules); mem_rules_free_(rules, nrules);
      rules = mem_rules_parse_(";just a comment\n;another\n", 26, &nrules); mem_rules_free_(rules, nrules);
      { i32 j, w = 0; char *cm = (char *)malloc(200002); for (j = 0; j < 100000; ++j) { cm[w++] = ';'; cm[w++] = '\n'; } cm[w] = 0;
        rules = mem_rules_parse_(cm, w, &nrules); mem_rules_free_(rules, nrules); free(cm); } }   /* many comment lines: used to be recursive too (fixed alongside the depth bound) */

    { i32 fixed = mem_fix(m); printf("mrfuzz: %d rounds, mem_fix found %d issue(s) to repair, %d live notes\n", rounds, fixed, vecdb_count(db, VN_NOTE)); }
    remove("/tmp/mrfuzz.vdb"); remove("/tmp/mrfuzz.vdb.vdt"); remove("/tmp/mrfuzz.vdb.vxi"); remove("/tmp/mrfuzz.vdb.vxi.j");
    remove("/tmp/mrfuzz.vdb.mem.cold");
    mem_free(m); vecdb_free(db);
    printf("mrfuzz ok: survived %d rounds + deep-nesting/unbalanced/empty edge cases under whatever sanitizers this build has\n", rounds);
    return 0;
}
