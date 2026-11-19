# Roadmap

The owner's list and where each item stands (2026-11-14). "Not run" means written and compiled, tests and benchmarks
still to run on his machine.

Scope: vecindex is a database for RAG and LLMs, with no dependencies, built as one project, in a repository of its
own (github.com/Gucixdev/vecindex) again. What a RAG pipeline does with the hits before a model reads them is part of
it (2026-11-19: `store_rag.c`, below); the note database, agent memory (mem.c and its rule engine) and the
cross-encoder re-rank stay in `examples/agent_launcher/`, a separate program that uses vecindex as a library.
Embedding models, tokenizers and answer generation stay outside.

Done and tested: quantization with an exact re-rank (4, 2 and 1-bit codes), SIMD (AVX-512, AVX2, NEON, a scalar path
elsewhere), hybrid BM25 search, concurrent updates (one rwlock, live indexing), background snapshots, a line protocol beside binary and HTTP, system logs instead of users and roles, any language through
the C ABI and the protocols, i64 ids, one CPU knob, texts with chunk ranges, vfind.

Done and tested since 2026-11-06 (`./run`, every step passing on a 2-vCPU Xeon): memory tiers (hot codes in RAM,
cold rows on disk, mlock), tenants (`vserv DIR`, `vi_dir_*`), write
batches (all or none, CRC-32C journal records, fsync on commit), integrity (BLAKE3 digests of every 64 KiB block, Reed-Solomon parity - 2 for every
32 blocks, interleaved - the table of digests twice, checks at load that rebuild and write back, a scrub, a CRC-32C on
every journal record, a check on every row tested on every read with the blocks rebuilt on the spot, `vi_scrub`,
`vserv` `scrub`), the cascade (1-bit scan, 4-bit shortlist, exact
re-rank: `vi_set_cascade`, `vserv -C`, bench methods `vecindex-cascade` and `-i8`; `tests/store_test.c` checks the
derived 1-bit codes against a 1-bit build byte for byte, added rows, recall against the 4-bit scan, the file), the
store self-test under ASan and UBSan in `./run`. The integrity math, CRC-32C, the fast-scan arithmetic, the codes,
the rotation and the top-k insertion are also proved in Lean 4 and the C code is tested against the proved model:
[`lean4`](../lean4).

Done and tested since 2026-11-10 (`./run`, 28 steps passing on a 2-vCPU Xeon): grouped search (`vi_search_grouped`,
the best hit of each `id >> shift` group: one chunk a document), a 64-bit number a row with range filters tested in
the scan (`vi_set_num`, `vi_search_range`, `vi_batch_set_num`; in the file, the journal and the row checks;
`tests/store_test.c` finds 5-row ranges anywhere in an IVF store, and a mutation of the scan's range test fails it),
shards (`vi_dir_search`, the hits of one store with every row, from three), `vi_update` (a new vector, the
tag, number and text kept, through the journal, a batch and a save), `vi_search_where` (labels and a number range
in one search), all of them in vserv too (line `group`, `between`, `num`, `update`; binary ops 15-19; HTTP), a cold index (codes past half of RAM stay on the disk at load, a search
pages in its probed clusters: 0.7 ms against 0.4 ms warm on 2M rows, `bench/cold.c`), the lens (a file past half of RAM opens without reading every block:
the small parts at once, texts when read, everything before a save, the rest walked one block at a time), namespaces (tenants in one store: small ones searched over their own rows, big
ones through the scan with the namespace inside it; `vi_add_ns`, `vi_search_ns`, `vi_delete_ns`), a CRC-32C a text checked on every read (rebuilt from the parity, or not served), API keys in vserv (`-K`: a key per tenant, a namespace and ids of its own in one
store, or its file in a tenant directory; housekeeping stays with the server), limits a key (rows, requests a second), `/metrics` (JSON)
and `/prometheus` (the Prometheus text format) for the operator, a replica on a second machine (`vserv -F`: the file and
its journal, never ahead of the primary's last fsync, a new file fetched and checked block by block after a snapshot,
failover by starting `vserv` on it), encryption at rest left to LUKS, ZFS or fscrypt (docs/server.md).

Done since 2026-11-13 (branch `vecindex-features`; `./run` 27 steps and `lean4/run` 34 steps passing on a 2-vCPU
Xeon unless a line says otherwise):
- search with a proof (`vi_search_exact`, `vi_search_exact_batch`, vserv `exact=1` and op 24, Python, Rust, JS): the
  exact top k with every other row ruled out by a bound proved in `lean4/Vecindex/Exact.lean`. GloVe-100, 1.18M
  rows: recall@10 1.0000 at 3.7-4.0 ms median against 50-56 ms for every row
- e-means clustering as the default (`lean4/Vecindex/Emeans.lean`)
- recall 0.90 as the default target of a store opened without `VI_MANUAL` (speed first, for RAG)
- clusters split in place as they grow instead of a whole rebuild (`src/tq_split.c`, `tests/split_test.c`).
  Synthetic data only: 67 splits, no rebuild, recall 0.996 against 0.772 without. Not measured on real data
- a second level over the centroids from 16384 clusters (`src/tq_top.c`), saved in the file (flag 64).
  Tested for equal probes and recall on 400k synthetic rows; not measured at the scale it is for (16M rows up)
- query-aware probe pruning (`VI_PROBE_PRUNE`, off by default): at eps 0.2 recall 0.977 against 0.996 on the
  synthetic test; how many clusters it saves is not measured
- scan threads only for long scans (`VI_SCAN_THREAD_BLOCKS`): measured on 2 vCPUs only
- `src/config.h` (every compile-time setting, the RAM limit among them), `src/format.h` (the file's layout)
- the rules behind the configurations proved (`lean4/Vecindex/Config.lean`): the cluster count grows as sqrt(n),
  re-scoring needs finer steps than the scan's codes, more probes never lose a candidate
- the first engine retired to `trash/` (q1-q5, qmix, q4fs, hnsw, ivf, engine: 4,842 lines nothing reached)
- no malloc on any search path (`src/scratch.h`) and none in the scan index (`src/mem.h`: arenas over the OS's
  pages; `tq.c`, `tq_add.c`, `tq_split.c`, `tq_top.c`)

Done since 2026-11-19 (built and its own test run here, `tests/rag_test.c` under ASan and UBSan too; the rest of
`./run` not run):
- RAG in the store (`src/store_rag.c`): `vi_search_mmr` (diverse hits by maximal marginal relevance, near-duplicates
  dropped by cosine), `vi_neighbors` (the chunks around a hit, never past its document), `vi_context` (MMR hits, their
  neighbours, one span a run of a document's chunks, cut to a byte budget with the hits' own chunks first). No
  malloc: the scratch pool. Not in vserv or the bindings yet
- the three branches (main, merged, needmerge) one: main
- payload fields and filters (`src/store_meta.c`): up to 16 typed fields a row (int, float, string by its key),
  journaled (`K`, `V`), saved (file flag 128), mixed into the row's check and parity; a filter language (= != < <=
  > >= in, not in, has, and, or, not, parentheses) compiled to a postfix program and tested inside the scan through
  `TqFilt.ok`; `vi_search_filter`, `vi_match`, and the filter of `vi_search_mmr` and `vi_context`.
  `tests/meta_test.c` (42 checks, run here, ASan and UBSan too)
- documents into pages and chunks (`vi_chunk`, `VI_ID`; asked in his braindump): form feeds end pages, boundaries
  blank line > line end > sentence end > space, UTF-8 whole, overlap; search by page = `vi_search_grouped` shift 10.
  `tests/chunk_test.c` 12 checks. Fields in write batches (`vi_batch_set_int/float/str`, the journal's `V` record
  inside the batch's frame; meta_test +6, the replay of a batch-only field checked)
- the graph (`include/graph.h`, `src/graph.c`, a module of its own): typed, weighted links between ids in a log
  file with CRCs (torn tail cut, compaction by rename), `vg_links`, `vg_expand` (levels, types, direction, weights,
  budget), `vg_ppr` (personalized PageRank by local push). `tests/graph_test.c` (29 checks; PPR within 5e-8 of
  power iteration), ASan and UBSan too. Then `vg_context` (the store's hits seed a walk, its best nodes go to the
  new `vi_context_ids` with the same budget) and `bench/graph_ablation.c`: A-D at one budget on synthetic two-hop
  data - A 0% complete, B, C, D 100%; expansion costs nothing over the search (p50 91 us), PPR ~400 us more at
  eps 1e-4. Next: the ablation on real documents with real links (citations, a code base's includes), the graph in
  vserv and the bindings, PPR's eps tuned for latency
- the hyperbolic metric (`VI_LORENTZ`, metric 3): rows as tangent vectors at the origin, the exact score a
  cancellation-free line in the cosine (so the certified search holds), the distance out. `tests/lorentz_test.c`:
  recall@10 500/500 after a build on 20,000 random points (33 dims); real hyperbolic embeddings not measured yet. (A
  first try - an inner-product store with x0 negated - gave 47/500 after a build: the index ranks by direction, and
  Lorentz products are negative and fall with the norm)

Open, in the order they are planned:
1. **The rest of the code off malloc** (158 call sites): the store (`store.c`, `store_disk.c`, `store_exact.c`,
   `store_tune.c`, `store_sum.c`), then the text index and the id table (`lex.c`, `hdict.c`), then the server. A
   build that runs out of memory should fail with an error everywhere (now: the main arrays are checked, the
   working ones not all)
2. **Sparse vectors** (the caller gives dimension-weight pairs, e.g. from SPLADE), indexed on the posting lists of
   `lex.c` and fused with the dense hits: hybrid search beyond BM25
3. **Metadata as JSON with indexed fields, and a planner for filters**: a selective filter lists its rows first
   (pre-filter), a wide one is tested inside the scan (now: tags, 64 labels, one number, namespaces; the pre-filter
   only for small namespaces)
4. **Effort chosen per query, no knob**: the scan stops when further clusters stop changing the top k; needs the
   scan loop rebuilt (its block list is made up front today). Then `VI_PROBE_PRUNE` and the recall target can go
5. **The tuner**: it searches probe counts assuming recall only rises with them; `Config.lean` has the
   counterexample (a re-rank depth below the candidate count). Measure each count instead
6. **Splits, finished**: a moved row should take a free slot of its new cluster (24,000 added rows left 28,500
   holes, and a quarter of the slots empty forces the rebuild splits were to avoid); merge clusters that shrank
7. **SPANN, the rest, filtered** (asked 2026-11-19): rows near a cluster boundary kept in several lists. A row in more
   than one slot touches search with a proof, compaction and splits - to be done with tests running, not blind.
   Filtered: (a) a summary a cluster, beside its centroid - the OR of its rows' tags, min and max of the number and of
   each field's code, a small Bloom filter of the string keys - so a filter skips a cluster that cannot match before
   any of its codes is read (from RAM or the disk); widened on every write to a row of the cluster, never narrowed
   until a rebuild; (b) filter-aware probing: past a skipped cluster the next nearest one, until the probes hold
   enough matching rows or the query's read budget is spent; (c) a filter the summaries count as very selective
   (under ~0.5% of the rows) goes the way a small namespace does: an index of the field's rows, scored exactly; (d)
   the same summaries a store in a directory, so `vi_dir_search` and a router skip whole shards
8. **One measurement at scale**: 100M to 1B rows from disk with the top cells, splits and pruning on; the factor in
   `4 sqrt(n)` swept (`Config.lean`: the best one is sqrt(b p / a) for the measured costs, 4 is a rule of thumb)
9. **Profile the query path, then SIMD where it shows**: the per-query tables (rotation, 16 entries a column), the
   top cells and centroids scored one dot product at a time
10. **A guarantee from calibration** (conformal): "at most a share alpha of queries below recall r", valid only for
    queries like the logged ones
11. **Past one machine**: shards behind a router in vserv (plain and exact search fanned out, top k merged, the
    merge proved in Lean); then failover by promoting a replica, and read/write/admin roles on keys
12. **Sparse and multi-vector rows** beyond item 2 (late interaction)
13. **An optional GPU scan**: a Vulkan compute shader, `libvulkan` opened with dlopen (nothing linked; without it
    the CPU path), `VI_VRAM_LIMIT_MB` its limit. Tested here on a software Vulkan only; real numbers need his card
14. **The live visualisation from the real engine** (WASM)
15. **Ternary codes**, measured before any code: as a first sieve for row-to-centroid comparisons at build and
    split time (`trash/src/q3.c` has the packing). Not for the scan's codes or the rows: Config.lean shows why
16. **Fields, next**: in vserv and the bindings; an index on a field (sorted
    row lists) so a very selective filter starts from its rows instead of widening the scan; the filter in
    `vi_search_exact` (the certified search) and `vi_search_hybrid`; strings kept whole on demand, beside the key
17. **Compressed texts** (asked 2026-11-19, next; the design: [compression.md](compression.md)): the documents stay
    apart from the vectors, as now, and their heap is compressed in the file by an LZ4-style coder of our own with a
    dictionary trained on the store's own texts - and nothing may fall apart: compressed bytes go through the block
    digests and Reed-Solomon parity like every other byte, each text is its own frame (damage beyond the parity loses
    one text, never its neighbours), the dictionary is stored twice with its own CRCs, the CRC stays on the raw text,
    the decoder is bounds-checked, fuzzed and its bounds proved
18. **RAG, next**: `vi_context` and `vi_search_mmr` in vserv (line, binary, HTTP) and the Python/Rust/JS bindings;
    MMR over the 1-bit or 4-bit codes instead of the f32 rows when the rows are cold (k x 4 x dim floats read a
    query today); a dedup of exact text copies by their check without the vectors
19. Housekeeping: `Codes.lean` still proves the digit tables of the retired q1-q5; `src/fine_grained.h` (a thread
    pool) has no user left in `src/`; `src/arena_alloc.h` holds only the store's own struct (`store_new`) and
    should give way to `mem.h`

Open from before, not yet placed in that order:
- labels beyond 64 a store: sets of rows as bitmaps, unlimited, with the same hierarchy (now: 64 named labels in a
  hierarchy on the tag's bits, `vi_label*`); relations between rows
- the file format: zones (parts of a store split by id range, time or tag) and namespaces inside a tenant, one
  format readable on every machine (the journal is written in the host's byte order today)
- scale at both ends: the smallest boards (fewest bytes a vector, RAM to spare) and past one machine's RAM. Shards as
  files are in (`vi_dir_search`: one search over every store of a directory); a store holds at most 2^31 - 1 rows
  (i32 row numbers). The plan for 10^9 rows on one machine, at ~25 bytes a row hot (2-bit codes at d=96, or the
  1-bit cascade) = 25 GB: the codes read by cluster from the disk instead of mapped whole (the probed clusters only,
  read ahead in one request), the build streamed in passes over the rows instead of holding them, the shards searched
  on threads. 10^12 rows is 25 TB: a few disks, still shards as files on one machine - no object store, no cluster.
  10^15 (25 PB) is out of scope.
  Mixed machines (2026-11-19: a trillion is the target, across Raspberry Pis and bigger boxes): each node sizes its
  own shard and clusters from its `VI_RAM_LIMIT_MB` (centroids and summaries in RAM, codes and rows on its disk),
  publishes a coarse sketch of its centroids and summaries, and a router asks only the nodes whose sketch is near
  the query and can match its filter; a slow node gets a time budget, and a result that misses a node says so
  (recall reported, never assumed). Honest sizes at d=384: 4-bit codes 192 B, int8 row 388 B, id, tag and check
  20 B - about 600 B a row, 1.6 billion rows a TB; 10^12 rows is about 600 TB, so hundreds of Pi-class nodes with a
  TB each, or fewer bigger ones; 1-bit codes with no stored rows (no exact re-score) is ~70 B a row, ~70 TB
- p99 under a steady write load: a burst faster than one core indexes waits in the exact-scan tail, and a rebuild
  holds two indexes; index bursts on more cores and bound the tail
- the cascade's benchmark, and its shortlist tuned rather than fixed at 8 x R
- head-to-head runs: Qdrant (HNSW) and LEANN are in the harness (`py/bench.py`, `./run compare`) but have not
  finished a run; Qdrant's TurboQuant mode, the RaBitQ engines (Milvus, LanceDB) and USearch are not in it yet;
  a graph index at very high recall
- streaming chunking; lazy loading beyond what mmap and tenants opened on first use already do
- integrity, next: the scan's codes checked on a read after the load too (now: at the load, by a scrub, or kept in RAM);
  texts under the row checks; SIMD for BLAKE3 and the GF(2^8) parity; parity encoding on threads in a snapshot child
- versioning: older snapshots kept and readable
- compression of rows and texts
- building the index from the quantized vectors, without a full-precision pass, to cut serving cost
- SSD and SATA disks as the index's home, not only RAM (cold rows are on disk already; codes read by block next)
- a fuzzer for the store's file loader and the server's protocols (the fuzzers left with the agent layer)
- packages over the C ABI for Python, Rust and Go (the bindings in `docs/api.md` are run by hand)
- the constraint over all of it: nothing that needs an IT department to run
