# The store: scan, clusters, tuning, live ingest, scale

`src/store.c` and its parts hold the rows, the ids, the texts and the index; `src/tq.c` is the scan. This page is
how they work and what was measured; the API is in [api.md](api.md), the numbers against other libraries in
[benchmarks.md](benchmarks.md).

## TurboQuant lesson (fast vector index)

`bench/vcmp.c` + `py/vcmp_turbovec.py` on 201k Simple English Wikipedia articles (`py/wikiprep.py`), embedded
by a hashed 256-d embedder; the same vectors go to both sides. Recall@10 against exact search:


| index | build | recall@10 | ms/query |
|---|---|---|---|
| vecindex auto-tuned (IVF+q4) | 70 s | 76% | 7.9 |
| vecindex IVF fast build | 8.6 s | 65% | 2.1 |
| vecindex flat q4 | 2.9 s | 56% | 24.8 |
| turbovec 4-bit (TurboQuant) | 0.35 s | 86% | 1.5 |
| turbovec 2-bit | 0.23 s | 59% | 0.5 |
| **`src/tq.c` (rotation + 4-bit fast-scan + exact re-score)** | **1.1 s** | **100%** | **1.2-1.4** |

`src/tq.c` copies what makes TurboQuant work: a fixed random rotation (signs + Walsh-Hadamard) turns every
coordinate into ~N(0,1/d) for any data, so one fixed Lloyd-Max 4-bit codebook is near-optimal with no
training; codes sit in a fast-scan layout scored with PSHUFB (AVX2 or AVX-512, threaded above 150k rows). The scan only
proposes 256+ candidates; the f32 rows the store keeps re-score them exactly, which is what lifts 86% to
100%. `store_build` uses it by default (Index choice, below).

## Index choice

`store_build` picks automatically: with no memory budget (default) the TurboQuant-style scan (`src/tq.c`);
with `ENG_MIN_SIZE` or a `max_code_bytes` below the f32 size, the q1-q5 / IVF / HNSW tuner (`engine_build`),
which is what small-footprint goals need.

## Stemming: Polish and English together

`src/lex.c` folds Polish letters to ASCII (`ą ć ę ł ń ó ś ź ż` -> `a c e l n o s z`, so a query typed without diacritics
still matches) and strips one longest matching suffix from every token, then a final vowel. There is no language
detection: one rule set covers both (Polish case endings, `-owanie / -uje / -ować`, `-ami -ach -ów`; English
`-s -es -ed -ing`). Tokens with digits are left alone. Tests keep families together (kot/kota/kotem/kotami/kotów,
komputer/komputerze, programować/programuje/programowanie, zapytanie/zapytań, notes/note) and look-alikes apart
(kot/kos). English retrieval on Wikipedia did not move (hybrid 98.7% on lead words, 96.4% on deep-body words).
Known limits: stem-final alternations (dokumentacja/dokumentacji, kot/kocie, pies/psa) are not handled, and the
English side is a light suffix stripper, not Porter, so `data`/`date` meet.

## IVF for the TurboQuant scan (large stores)

From `STORE_IVF_MIN` = 600 000 rows the store groups the rows by cluster (IVF) and a query scans only the blocks of
the clusters closest to it. The cluster count follows the usual rule, 4 sqrt(rows), with no cluster planned below
`STORE_IVF_ROWS` = 256 rows: 1.2M rows get 4350 clusters of ~270, 100M rows 40000 of 2500. Training is spherical
k-means on 256 rows per centroid (the sample is capped at 512 MB); above 1024 clusters it is two-level: ~sqrt(nc) top
cells, trained on a subsample, each split into sub-clusters in proportion to its share of the sample. A row goes to the
closest sub-cluster among those of its 4 closest top cells (2 past 128 cells), and one Lloyd round over the whole
sample then lets neighbouring cells settle their shared borders. With only the closest cell, a quarter of GloVe's rows
sat in the wrong cluster - rows a query then looks for in vain; now 4.5% do, and the true top 10 lies inside the 96
closest clusters for 90.0% of queries instead of 87.3%, with 10% fewer rows to scan. Centroids are scored 16 at a time
(transposed, AVX-512 FMA, 4 rows x 4 groups in registers); encoding runs on every core, each thread owning a range of
blocks so no byte is shared.

A query picks its clusters from an int8 copy of the centroids: VNNI scores 16 centroids on 4 dimensions per
instruction, a quarter of the bytes of f32, so the copy stays in L2 (0.44 MB for GloVe). A binary search on a vector
count keeps about probe + probe/4 + 16 of them, and only those get exact f32 scores, so the clusters probed are the
ones f32 would pick. In a batch, 8 queries share each pass over the centroids. The scan keeps its candidates in a
reservoir (append while above the threshold, cut to the best R with a quickselect when full) and prefetches two
blocks ahead. The default probe is min(16%, 4 sqrt(clusters)) of the clusters (GloVe-100: 264 of 4350, recall@10
0.95); `store_set_probe` trades recall for speed, `store_set_rerank` sets how many candidates are re-scored exactly.
Below 600k rows the flat scan is used.

`./run` exercises the IVF scan directly (recall, tombstones, meta filter, all-clusters == flat, 2-bit too, the
two-level clustering on 1500 clusters, batch == one query at a time), and the store demo and stress test also pass
when forced onto IVF at 1500 rows (`-DSTORE_IVF_MIN=1500 -DSTORE_IVF_ROWS=100`).

## Self-tuning: ask for a recall, the store keeps it

`store_set_target_recall(s, 0.95)`: from then on every `store_build` measures recall@10 against exact search and
picks the fewest IVF probes that reach the target, aiming 1.5 standard errors above it so fresh queries meet it too.
It measures on the queries the store has actually been answering (a reservoir sample of the last 256, kept while a
target is set) or, before 64 have come in, on a spread of its own rows (each one's own row left out of the truth).
`store_tune(s)` measures again on demand - after the traffic or the data have drifted - and the probe count it
settles on is saved with the index. GloVe-100, `bench/tune.c`, on the Cascade Lake host of [benchmarks.md](benchmarks.md):

| target | tuned at build, on its own rows | held-out recall@10 (2000 queries) | re-tuned on those 2000 | recall@10 on the other 8000 |
|---|---|---|---|---|
| 0.90 | 118 probes | 0.9134 | 108 probes | 0.9053 |
| 0.95 | 264 probes | 0.9528 | 290 probes | 0.9584 |

A tuning costs about 2 s on GloVe, most of it the exact search for the truth; on bigger stores it uses fewer
queries (at most ~3e10 multiply-adds). The self-test checks it on held-out queries too.

## Live ingest: rows indexed as they come, the index rebuilt as it ages

`store_set_live(s, 1, 1.0f)` (or `vserv -L`) and nobody has to call `store_build` again:

- A thread of the store's own codes every added row into its closest cluster, 4096 rows at a time: coded off the
  lock, placed under the write lock. The rotation and the 4-bit codebook need no training, so an added row gets the
  same codes a build would give it (the self-test checks it: built + added rows probed over every cluster == a flat
  scan of all of them). Until a row is indexed, the exact scan of the pending tail finds it, as before.
- It rebuilds the index when it has aged: grown by `grow` since the last build (1.0 = doubled), a window of new rows
  sitting 1.25x farther from their centroids than the built rows did (the data moved), the cluster sizes 1.2x more
  uneven than at the build (new rows piling into a few clusters), or a quarter of it deleted. The rebuild runs beside
  the indexing, catches up on the rows that came in meanwhile and is swapped in under the write lock; searches use
  the old index until then. With a target recall set (`store_set_target_recall`) it re-tunes after each rebuild.
- Background work takes half the cores. `store_maintain` runs one step by hand, `store_live_stats` reports.
- Deletes no longer cost the saved index: the compaction renumbers the rows inside it, `store_save` writes added rows
  in the plain layout, and a load is still an mmap.

GloVe-100, `bench/live.c`, on the Cascade Lake host of [benchmarks.md](benchmarks.md). QPS: one batch of the 10k queries on 2
threads at the fastest probe count that reaches the recall (the same index measured twice differs by up to 10%):

| what the index holds | recall@10 >= 0.85 | >= 0.90 | >= 0.95 | >= 0.97 |
|---|---:|---:|---:|---:|
| one build over all 1,183,514 rows | 13,800 | 10,430 | 6,229 | 4,292 |
| 60% built, the other 40% added live | 13,707 (0.99x) | 8,858 (0.85x) | 5,745 (0.92x) | 3,806 (0.89x) |
| 74% built, 26% from regions the build never saw added live | 12,803 (0.93x) | 8,723 (0.84x) | 5,694 (0.91x) | 3,712 (0.86x) |
| 60% built, the 40% leaning one direction added live | 10,105 (0.73x) | 6,820 (0.65x) | 4,943 (0.79x) | 3,175 (0.74x) |
| the same after the rebuild live mode chose on its own (skew 1.51, 1.13 at the build) | 14,928 (1.08x) | 10,151 (0.97x) | 6,305 (1.01x) | 3,900 (0.91x) |

Before live mode the same 40% waited in the exact-scan tail: 96 QPS at recall@10 0.9589, against 5,393 at 0.9608
with them indexed (56x). Indexing them took 4.5 s on one core (105k rows/s).

Searching while rows stream in: one query at a time on one thread, 473k rows added by another thread at 50k rows/s,
live mode with rebuilds at +50%:

| phase | queries | p50 | p99 | max |
|---|---:|---:|---:|---:|
| idle, before | 3,000 | 0.42 ms | 0.71 ms | 4.7 ms |
| rows streaming in, indexed as they come | 10,517 | 0.61 ms | 2.63 ms | 15.6 ms |
| the rebuild live mode started at +50% (8.7 s) | 9,976 | 0.73 ms | 4.40 ms | 11.1 ms |
| after | 1,692 | 0.58 ms | 0.83 ms | 1.7 ms |

Before the rebuild ran beside the indexing (rows arriving meanwhile waited in the exact tail): p50 3.57 ms, p99 9.22
ms during it and p50 1.23 ms right after. A burst at full speed - 473k rows in 1.07 s, four times faster than one core
indexes - still fills the tail: p50 2.60 ms, p99 17.3 ms for the ~5 s until it is indexed.

Which signal fires where, measured: new rows drawn like the old ones leave the cluster-size skew where it was (1.142
-> 1.152); rows from 8 held-out Voronoi regions (26% of GloVe) raise it from 1.147 to 1.229 and sit only 3.6% farther
from their centroids - in 100 dimensions the remaining centroids still cover them, no rebuild is due, and the table
above shows they cost about what random rows do; rows leaning one direction raise it to 1.509, and live mode rebuilds.
On the self-test's 32-dimensional topics, rows from 30 new topics sit far from every centroid: two far windows, one
rebuild, recall@10 of their queries at 8 probes 0.763 -> 0.953.

Costs and limits: a rebuild holds the old and the new index at once (GloVe, 4-bit: +59 MB of codes; 100M rows, 2-bit:
+2.4 GB, too much for the 5.84 GiB box of the 100M run - keep `grow` at 0 there); rebuilding 100M rows takes minutes
(the build alone was 225 s); a burst faster than one core indexes waits in the exact tail; the growth and skew
baselines start over when a store is loaded.

## Scale: rows in chunks, past RAM

What capped the store before (measured, [benchmarks.md](benchmarks.md)): every f32 row sat in anonymous memory, and the arrays grew by
doubling into a second arena whose old copy was never released (RSS 1.5-3x the data), so the bash cgroup's 5.84 GiB
OOM-killed it between 7M and 8M rows of 128 dimensions - during ingest, before any index existed.

Now the rows live in column chunks of 65536 rows (vectors, doc ids, tags). A chunk never moves, so growth copies
nothing. Past 256 MB new chunks are pages of an unlinked spill file (`store_set_spill`, `$VECINDEX_SPILL`,
`/var/tmp` by default - use a disk, not a tmpfs), started on their way to disk as each fills (`sync_file_range`), so
the kernel drops them under pressure instead of swapping or OOM-killing. A loaded store points its chunks into the
.vxi mapping. What a search scans stays in RAM: the codes (d/2 bytes a row, d/4 in 2-bit mode), slot map and a
dead-row bitset. Re-scoring reads only the few hundred candidate rows a query needs; when those reads start missing
the page cache (major faults), the store asks for all of a query's rows at once (`MADV_WILLNEED`) instead of one
blocking fault after another - ~190k IOPS batched against ~21k one by one on this disk. Ids that arrive in increasing
order (the usual case) need no id map at all (binary search over the id column); only rows added out of order go
into an 8-byte-a-slot hash.

A loaded index whose scanned codes are more than half of RAM (the machine's, or a cgroup's cap) is cold: the load no
longer reads all of them in, and a search pages in only the clusters it probes. Measured with `bench/cold.c` on this
VM (2M rows, d=96, 97 MB of codes, 301 of 5647 clusters probed, page cache dropped first): 0.7 ms median a query and
12 major faults, against 0.4 ms warm. Asking for each probed cluster with `MADV_WILLNEED` before the scan was tried
and dropped: fewer faults, slower here. One catch: the default load checks every block's digest
(`VECINDEX_VERIFY=2`), which reads the whole file; a store larger than RAM opens with `VECINDEX_VERIFY=1` (the header's
blocks) and a scrub in the background.

For stores past what RAM and disk allow at f32, two knobs: `store_set_codes(s, 2)` (2-bit codes: half the RAM,
twice the scan speed, re-score more candidates) and `store_set_vec_bytes(s, 1)` (int8 rows with an f32 scale: d + 4
bytes on disk instead of 4d, about one recall point on GloVe - the cost of 8 bits, a rotation first does not change
it). Measured trade-offs are in [benchmarks.md](benchmarks.md).

## A real retrieval benchmark, not a hand-labeled one (real_val.c, SQuAD + AG-News)

Hand-written questions with one's own labels are a sanity check, not a number to set next to a leaderboard.
`bench/real_val.c` (`./run real_val`) is the closer-to-standard version:
**48,891 real passages** (18,891 unique SQuAD/Wikipedia contexts + 30,000 AG-News snippets as distractors,
`py/real_gen.py`) embedded with `sentence-transformers/all-MiniLM-L6-v2` (384-dim), scored against **1,000
real SQuAD questions** with their actual gold passage - a genuine asymmetric question -> passage retrieval
task (not synthetic near-duplicates), the same shape as the retrieval half of a BEIR-style eval even though
it is one dataset, not the full BEIR suite. Two numbers per config: `recall@10` (against exact-cosine ANN
ground truth - "did the index find what exhaustive search would have") and `gold@10` (against the real
answer - "was the actual right passage in the top 10", the metric that matters for RAG):

| config | recall@10 | gold@10 | latency |
|---|---|---|---|
| HNSW-vec (max accuracy) | 99.0% | 85.6% | 427 us/query |
| IVF q4fs (>=85% floor) | 92.2% | 83.1% | 2.45 ms/query |
| flat (max compression, 96 B/vec) | 80.2% | 83.6% | 4.43 ms/query |
| `store` (auto-tuned, what the C API and `vserv` use) | 97.1% | 83.5% | 6.11 ms/query |

The ceiling to compare against: exact cosine search with this SAME embedder only finds the gold passage in
its own top-10 **87.1%** of the time (measured directly from `py/real_gen.py`'s own output) - the embedder, not
the index, is what caps retrieval quality here. Every config above lands within 1.5-4 points of that ceiling,
and the best (HNSW) is within 1.5 points of it while beating it slightly on some runs (index construction and
the exact-cosine baseline use different tie-breaking, so being *at* the ceiling is the right way to read a
±1-2 point gap, not literally below it). Read honestly: this says the ANN index itself is close to lossless
relative to what a mid-tier 118M-param embedder can retrieve - it does **not** say retrieval quality is
state-of-the-art, since a stronger embedder would raise the 87.1% ceiling itself, which this benchmark cannot
show on its own. A cross-encoder second pass over the top hits closes part of that gap without a new embedder
(measured on this data in [`examples/agent_launcher`](../examples/agent_launcher): gold@1 62.1% -> 81.5% where it has room to help); see
[comparison.md](comparison.md) for the full picture.
