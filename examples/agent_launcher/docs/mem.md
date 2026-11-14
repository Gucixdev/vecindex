# mem.c: lifetime, metamemory, event->fact->concept, a rule engine

A memory layer on top of vecdb - `#include "mem.c"` (it includes vecdb.c). Design decisions and
exact formulas are in the file's own header comment (single source of truth, the three levels,
decay/confidence formulas, hot/warm/cold, the rule grammar) - not duplicated here. Summary:

- **Single source of truth: the event.** Facts and concepts are lossy, rebuildable, EXTRACTIVE
  (no LLM here - a promoted node's text is literally its best source's text, not a summary)
  materializations over events, tracked via `MemMeta.prov[]` (provenance), same discipline vecdb
  itself already applies one level down (graph/BM25/vectors rebuilt from note text, never trusted
  as primary).
- **Lifetime/decay**: `mem_relevance` - continuous, `exp(-ln2/half_life * idle_seconds)`, half-life
  configurable per kind (`mem_set_lifetime`, defaults 7d/90d/720d for event/fact/concept),
  reinforced by touches (`mem_touch`, and `mem_search` which touches every hit) and corroboration.
- **Metamemory**: `mem_confidence` (events trust direct observation, discounted only by
  contradiction; facts/concepts must earn trust via corroboration), `mem_known` (a real "I don't
  know" - MEM_UNKNOWN when nothing clears a real relevance floor, not just "search returned its
  top-k anyway"), `mem_gaps` (thinly-evidenced held facts/concepts).
- **Hot/warm/cold**: `mem_hot_ids` (bounded MRU of recently touched ids), warm = plain vecdb
  (unchanged), `mem_archive`/`mem_recall` (append-only fsynced PATH.mem.cold, a real tested round
  trip, not a stub).
- **Operations**: `mem_split` (an event buds off smaller children, parent kept), `mem_merge`
  (survivor by relevance, provenance unioned, every OTHER node's dangling reference to the loser
  repointed), `mem_prune` (decay's garbage collector - never touches a supported node),
  `mem_forget` (explicit - archives if anything depends on it, else deletes; never silently no-ops),
  `mem_fix` (consistency repair: dead vids, dangling provenance, invalid counts, stale hot entries).
- **The rule engine** (`mem_consolidate` - the "hippocampus" pass): a small forward-chaining
  s-expression interpreter (parser + evaluator, no library) - conditions read
  kind/relevance/corroboration/contradiction/access_count/age/idle plus a real `(similar T)`
  vector search; actions `(promote fact|concept [THRESH])`, `(merge [THRESH])`, `(prune)`,
  `(archive)`. Same-kind reinforcement converges within one pass (a newly created fact is visible
  to later-evaluated events in that pass); cross-kind promotion (fact->concept) needs its own
  later pass (the fact must exist at pass-start to be in that pass's snapshot) - both behaviors
  are exercised in the self-test. Calibrate thresholds to the embedder: the built-in hashed one
  puts real paraphrases around 0.55-0.65, not 0.85+. `mem_consolidate_ids`/`mem_consolidate_hot`
  run the same engine scoped to a caller-given/hot working set instead of every live node - see
  "Timing" below for why, and the numbers.
- **Real bugs the self-test caught while writing this** (worth naming, since they show what
  "tested" actually bought here): a promoted fact reusing its source event's exact title made
  `vecdb_add_note`'s upsert-by-title silently REPLACE the source event instead of creating a
  sibling; an unguarded rule re-fired on every pass, double-counting corroboration forever
  (fixed with `MemMeta.promoted`); a naive "no rule fired => nothing relevant" assumption in
  `mem_known` was wrong because `vecdb_search` always returns its top-k regardless of quality
  (fixed with an explicit candidate floor); `mem_confidence` originally gave a freshly-asserted,
  zero-corroboration concept ~1.0 confidence (fixed by requiring corroboration to earn trust for
  derived kinds, while events keep direct-observation trust).
- Tests: 36 checks in `-DMEM_DEMO` (`./run`), clean under ASan+UBSan+LeakSanitizer (zero leaks -
  better than vecdb.c's own self-test baseline, which does not free everything at exit); plus
  `tests/mthreads.c`/`tests/mcrash.c`/`tests/mrfuzz.c` (`./run`), `tests/mstress.c` (`./run stress`) and `bench/mbench.c`
  (`./run bench`) below.
- **Concurrency** (`tests/mthreads.c`, `./run`): 4 readers (search/touch, relevance, confidence, known,
  gaps, hot list, provenance) + a writer (note, corroborate/contradict, merge, prune, forget,
  consolidate, save) on one `Mem`, under ThreadSanitizer. This is what actually found the real bugs
  worth naming: `mem_relevance`/`mem_confidence`/`mem_gaps`/`mem_hot_ids`/`mem_provenance`/
  `mem_kind` never took the lock at all (73 TSan warnings on the first run); `mem_merge`/
  `mem_prune`/`mem_consolidate`/`mem_split` called those (now-locked) functions while *already*
  holding `m->lk` - a real self-deadlock risk, not just a race, fixed by splitting each into an
  unlocked `_()` core (used internally, caller already holds the lock) and a locked public wrapper
  (the same pattern `src/vecdb.c` already uses for its own `vdbl_*`/`vecdb_*` split); `mem_gaps` and
  `mem_prune` sized their scratch buffers from `m->n` *before* taking the lock - under concurrent
  growth that is a real buffer overflow, not a benign read race; `mem_recall`'s cold-file scan
  wasn't lock-protected at all, unlike `mem_archive`'s. Now clean under TSan and ASan+UBSan+
  LeakSanitizer (0 leaks), rerun several times to rule out timing luck.
- **Crash safety** (`tests/mcrash.c`, `./run`, same discipline as `tests/vcrash.c`/`tests/vcrash2.c`): a child adds
  events, corroborates/contradicts, archives some to cold storage, and syncs `PATH.mem` +
  `PATH.mem.cold` alongside vecdb's own files; the parent SIGKILLs it at a random point and checks
  every meta entry's vid is a real live vecdb note and `mem_fix` finds nothing left to repair.
  20-90 rounds, always consistent.
- **A second pass, this time systematic, not reactive**: TSan can only report a race it actually
  schedules into; it does not prove the rest of the file is clean. So every `mem_*` function was
  read by hand afterwards for "does every read/write of `m->n`/`m->meta`/`m->byvid`/`m->hot`/`m->cf`/
  `m->half_life` happen under `m->lk`", which found four more real bugs TSan's own run had not
  happened to hit: `mem_set_lifetime` wrote `half_life[]` with no lock at all; `mem_cold_open`
  swapped `m->cf` (closing the old `FILE*`) with no lock, while `mem_archive`/`mem_recall` read it
  under lock; `mem_archive`/`mem_recall` each checked `!m->cf` *before* acquiring the lock, so that
  one check itself raced `mem_cold_open`; and `mem_forget` read `m->cf` directly to decide whether
  to archive, unlocked, and (a real logic bug independent of threading) its own doc comment
  promised a hard-delete fallback when there is no cold storage that the code never actually did -
  fixed by having `mem_forget` simply call `mem_archive` and fall through to delete on failure,
  removing the direct `m->cf` read entirely. `tests/mthreads.c` was extended to exercise
  `mem_set_lifetime`/`mem_cold_open`/`mem_archive`/`mem_recall`/`mem_cold_count` concurrently too
  (kept infrequent - they do real fsync'd I/O under the lock, and TSan's per-access overhead makes
  that add up fast) and a new self-test case covers `mem_forget` with no cold storage configured.
  Clean under TSan (reran 3x) and ASan+UBSan+LeakSanitizer after this pass too.
- **A third pass, this time an adversarial correctness stress test** (`tests/mstress.c`, `./run
  stress`): 60 rounds of weighted-random churn (80-199 ops/round) across every operation
  (note/corroborate/contradict/merge/prune/forget/archive+recall/split/consolidate/touch/search),
  driven from an id pool that is *deliberately* not kept in sync with real liveness (merge losers
  are left in the pool as stale ids on purpose - "`mem_get_` handles it" was the working
  assumption being tested). After every round: real invariant checks against the live structures
  (every live vid resolves to a live vecdb note; relevance/confidence in [0,1] with confidence
  never exceeding relevance; every provenance id resolves to something live; live meta count
  equals vecdb's own live note count; hot list within capacity with no dead/duplicate entries;
  wikilink and tag symmetry against vecdb's own backlink/tag-membership lists, since "does mem.c
  respect vecdb's graph" was flagged as untested) - then `mem_fix` is asserted to find *zero*
  issues, since a correctly-implemented op should never let an invariant slip in the first place.
  This found five distinct, real bugs no amount of code review had caught, through actual failing
  runs, not static analysis:
  - **`mem_merge(m, a, b)` with `a == b`** (a self-merge, picked independently from the same
    stale-tolerant id pool): both sides resolved to the same entry, so the function proceeded to
    archive/delete the very node it was about to treat as the survivor, then re-added the now-dead
    id to the hot list. Fixed with an early `a == b` guard that returns the id unchanged if it's
    still live.
  - **`mem_new_meta_` could create a duplicate/orphaned entry**: `vecdb_add_note`'s upsert-by-title
    can silently *reuse* an existing vid (e.g. `mem_split` re-splitting the same parent produces
    identical child titles the second time), and `mem_new_meta_` never checked whether that vid
    already had a live entry before appending a new slot - inflating the live meta count past
    vecdb's own live note count. Fixed by resetting the existing entry in place instead of
    appending when the vid is already live.
  - **`mem_corroborate`/`mem_contradict` never validated `from_vid`'s liveness** before recording
    it as provenance, so a caller could poison a live entry's provenance list with a dangling
    reference to an already-dead id. Fixed by validating `from_vid` first; both functions'
    return type changed `void` -> `i32` (1 = took effect, 0 = no-op) so callers can tell.
  - **`mem_recall` blindly restored historical provenance from cold storage** without checking
    whether those ids were still live at recall time - nothing could have kept them in sync while
    the record sat archived, unlike every other removal path, which cleans up as it happens.
    Fixed by filtering restored provenance to ids that currently resolve to something live.
  - **The root cause, found last and explaining nearly everything else this pass touched:
    `mem_archive` never called `h64_del(&m->byvid, vid)`** - a plain oversight since the function
    was first written, never caught by the second-pass locking audit above because that audit
    checked *locking*, not this specific missing state-consistency call. Since `mem_get_` (the
    sole liveness check used everywhere in the file) only consulted the `byvid` table and never
    checked the `archived` flag itself, this single omission meant archived nodes stayed fully
    "live" forever as far as every downstream check was concerned - silently defeating every
    invariant, corroboration check, and dangling-reference guard in the file. Took an extensive
    multi-round trace-instrumented debugging session to isolate (temporary `TRACE`/`watched_`
    scaffolding, since removed) down to one missing line. Fixed two ways: (1) added the missing
    `h64_del` call, right where every other removal path already has it; (2) hardened `mem_get_`
    itself to also check `!archived` as defense-in-depth, so no future single-line omission of
    this kind can have this magnitude of silent, cascading effect again.
  Clean (zero failures across all 60 rounds + a save/load round-trip check) under plain, TSan, and
  ASan+UBSan+LeakSanitizer builds after all five fixes.
- **Timing** (`bench/mbench.c`, `./run bench`): ingest ~7.5 us/note, `mem_touch`/`mem_relevance`/
  `mem_confidence` all sub-microsecond, `mem_search` ~0.3 ms/query, `mem_archive` ~0.08 ms/call
  (fsynced), `mem_save`/`mem_load` a few ms at N=20000. The number that matters -
  `mem_consolidate` cost vs corpus size, sweeping N via the same rule set:

  | N      | time/pass | per-node   |
  |--------|-----------|------------|
  | 1,000  | 0.35 s    | 350 us     |
  | 4,000  | 1.9 s     | 476 us     |
  | 16,000 | 11.6 s    | 725 us     |
  | 64,000 | ~97 s     | 1,527 us   |

  This is **super-linear**, not the flat O(nodes) the per-call cost alone would suggest - per-node
  cost more than quadruples from N=1,000 to N=64,000. The likely cause is the same one [vecdb.md](vecdb.md)
  calls out for large stores generally (IVF/HNSW routing overhead growing with corpus size
  even though each individual `vecdb_search` call stays sub-linear); `mem_consolidate` amplifies it
  by issuing one such search per node, every pass.
- **The fix: scoped consolidation.** `mem_consolidate_ids(m, rules, ids, nids)` runs the exact
  same engine over a caller-given id list instead of every live node, and `mem_consolidate_hot(m,
  rules)` is a thin wrapper scoping it to the current hot set (`mem_hot_ids`, bounded at
  `hot_cap`) - cost O(working set) instead of O(corpus). Measured on the same N=64,000 corpus as
  the worst row above: **106.6 s for a full `mem_consolidate` vs 0.29 s for `mem_consolidate_hot`**
  (bounded at 256 nodes) - a ~365x difference, because the cost is now capped by `hot_cap`, not by
  N. `tests/mstress.c`'s own churn loop dropped from ~45 s to ~14 s once most of its consolidation calls
  switched from the full sweep to the scoped variants, which is the same effect in practice, not
  just in a microbenchmark. Practical guidance: run `mem_consolidate_hot` on the hot/write path
  (or `mem_consolidate_ids` over whatever working set the caller already tracks, e.g. "touched
  since I last consolidated") and reserve a full `mem_consolidate` for an occasional off-path
  sweep (a timer, idle time, etc.) so nodes that never stay hot long enough still eventually get
  evaluated - a purely hot-scoped strategy would let those decay unconsolidated forever.
- **Rule parser fuzzing** (`tests/mrfuzz.c`, `./run`): random junk (a mix of real rule-DSL vocabulary
  and garbage - mismatched parens, bad numbers, unknown symbols, wrong arities/types) fed through
  the reader alone and through a full `mem_consolidate` pass against a live `Mem`, under
  ASan+UBSan. Found one real bug on the first write of this file, not a contrived one:
  **`sx_read_` recursed once per `(` nesting level with no depth limit at all - a few hundred
  thousand nested parens (a trivial, not exotic, malformed input) blew the C stack and crashed the
  process**, confirmed with a standalone repro before the fuzzer even ran. Fixed with
  `SX_MAX_DEPTH` (200 - generous for real rules, which nest at most a handful of levels): past it,
  `sx_read_` skips the over-deep list with a bounded balanced-paren scan instead of recursing
  further, so `*p` still always advances and the caller sees a clean "nothing parsed here" rather
  than a crash. The `;`-comment skip was also converted from a recursive tail call to a loop while
  in there, since a long run of comment lines was the same unbounded-recursion shape (compilers
  often optimise a tail call like that into a loop, but nothing guaranteed it). `tests/mrfuzz.c` keeps
  the exact deep-nesting shape that found this as a standing round (500k nested parens, balanced
  and unbalanced both directions), not just the reader on its own - so a regression here fails
  immediately instead of waiting to be hand-found again. 5,000-30,000 rounds, clean under
  ASan+UBSan+LeakSanitizer after the fix.

## Honest gaps in mem.c

- `mem_consolidate`'s `(similar T)`/promote/merge searches are O(nodes) vecdb_search calls per
  pass, and that per-call cost itself grows with corpus size - see the `bench/mbench.c` table above.
  `mem_consolidate_ids`/`mem_consolidate_hot` (above) are the fix for the hot/write path; a full
  sweep is still O(nodes) and still needed periodically to catch what falls outside the working
  set, and gets expensive (~100s) at 64k nodes - not yet measured near vecdb's own million-note
  ceiling.
- Extractive only, by design - no summarisation. A promoted fact/concept's text is its single
  best source's text verbatim, not a synthesis of everything that supports it.
- Even the systematic locking pass and the adversarial stress pass above were done by the same
  person who wrote the code, not an independent review - the honest bar is "no known gaps left",
  not "provably none remain".

