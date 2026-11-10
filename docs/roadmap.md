# Roadmap

The owner's list and where each item stands (2026-11-01). "Not run" means written and compiled, tests and benchmarks
still to run on his machine.

Scope: vecindex is the vector database alone, with no dependencies, built as one project. The note database, agent
memory (mem.c and its rule engine) and the cross-encoder re-rank moved to `examples/agent_launcher/` in this repo,
a separate project that uses vecindex as a library; vecindex itself is back in holyc as `vecindex/`.

Done and tested: quantization with an exact re-rank (4, 2 and 1-bit codes), SIMD (AVX-512, AVX2, NEON, a scalar path
elsewhere), hybrid BM25 search, concurrent updates (one rwlock, live indexing), background snapshots instead of
replication, a line protocol beside binary and HTTP, system logs instead of users and roles, any language through
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
in one search), all of them in vserv too (line `group`, `between`, `num`, `update`; binary ops 15-19; HTTP).

Open:
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
  10^15 (25 PB) is out of scope
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
