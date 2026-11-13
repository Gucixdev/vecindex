# Benchmarks

Everything in this section was measured on one kind of machine: 2 vCPUs of an Intel Xeon @ 2.80GHz (Cascade Lake:
AVX-512 with VNNI, 1 MB of L2 per core, 33 MB of L3 shared), the shell's cgroup capped at 5.84 GiB, Linux 6.18 - the
library comparison in one session, the live-ingest rows (`bench/live.c`) in a later one on the same kind of host.
FAISS 1.15.1 (pip), turbovec 1.1.2, qdrant-edge-py 0.8.0 and vecindex (through a ctypes shim, `bench/vi.c`) all run
through the same harness, `py/bench.py`, so every library pays the same Python call per query.

The cloud VM this ran on changed hardware during the work (it restarted onto this host from one with a 260 MB L3), so
the before/after numbers of individual changes in the priority table were measured as pairs on whichever host was up
at the time - never one number from each host.

## GloVe-100: queries per second at the same recall

glove-100-angular prepared the ann-benchmarks way: 1,183,514 vectors of 100 dimensions, 10,000 held-out queries,
recall@10 against the exact top 10. Each QPS cell is **one batch of 10k queries on 2 threads / one query per call on
1 thread**, at the fastest setting each method has that reaches the recall (the settings swept are in `py/bench.py`:
probes and re-rank depth for vecindex, nprobe and k_factor for FAISS IVF, efSearch / hnsw_ef for the graphs).

| method | build | load | bytes/vec | best recall@10 | >= 0.85 | >= 0.90 | >= 0.95 | >= 0.97 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vecindex 4-bit + f32 rows | 8.0 s | 7 ms | 475 | 0.9884 | 13,405 / 4,911 | 7,996 / 2,277 | 4,897 / 1,433 | 2,606 / 862 |
| vecindex 4-bit + int8 rows | 10.5 s | 7 ms | 179 | 0.9752 | 10,975 / 4,280 | 6,833 / 2,639 | 3,849 / 876 | 2,099 / 624 |
| vecindex 2-bit + f32 rows | 7.5 s | 7 ms | 449 | 0.9836 | 10,160 / 3,922 | 6,758 / 2,376 | 2,882 / 1,241 | 2,112 / 929 |
| vecindex 2-bit + int8 rows | 8.3 s | 8 ms | 153 | 0.9712 | 11,424 / 3,935 | 7,235 / 2,654 | 2,420 / 949 | 1,896 / 715 |
| FAISS IVF4096,PQ50x4fs,RFlat | 124.5 s | 510 ms | 436 | 0.9785 | 8,632 / 2,535 | 4,456 / 1,081 | 2,822 / 871 | 1,846 / 526 |
| turbovec 4-bit | 2.2 s | 105 ms | 56 | 0.8465 | - | - | - | - |
| turbovec 2-bit | 1.7 s | 69 ms | 30 | 0.5448 | - | - | - | - |

FAISS HNSW32, FAISS IVF4096,Flat and Qdrant are missing here: the cloud container was reclaimed in the middle of
their runs three times. On the first host they were slower than FAISS IVF-PQ at every recall. `./run compare
DATA_DIR` runs every row of this table and the scale runs in one go on any Linux box (`./run all DATA_DIR` the tests
before them), and
`py/tables.py` turns its output into these tables.

turbovec scans every code of every row (no clustering) and stops at recall@10 0.8465: 256 on 2 threads, 37 one query at a time QPS there.
Bytes/vec is what the loaded index holds, RAM and file together (vecindex keeps its rows file-backed; the codes it
scans are 50 (4-bit) or 25 (2-bit) B/vec).

Against the fastest other method at each recall:

| recall@10 | vecindex, batch 2 threads | best other | x | vecindex, 1 thread per call | best other | x |
|---|---:|---|---:|---:|---|---:|
| >= 0.85 | 13,405 | 8,632 (FAISS IVF4096,PQ50x4fs,RFlat) | 1.55x | 4,911 | 2,535 (FAISS IVF4096,PQ50x4fs,RFlat) | 1.94x |
| >= 0.90 | 7,996 | 4,456 (FAISS IVF4096,PQ50x4fs,RFlat) | 1.79x | 2,654 | 1,081 (FAISS IVF4096,PQ50x4fs,RFlat) | 2.46x |
| >= 0.95 | 4,897 | 2,822 (FAISS IVF4096,PQ50x4fs,RFlat) | 1.74x | 1,433 | 871 (FAISS IVF4096,PQ50x4fs,RFlat) | 1.65x |
| >= 0.97 | 2,606 | 1,846 (FAISS IVF4096,PQ50x4fs,RFlat) | 1.41x | 929 | 526 (FAISS IVF4096,PQ50x4fs,RFlat) | 1.77x |

## Where vecindex wins and where it loses

| axis | vecindex | the others | verdict |
|---|---|---|---|
| QPS at recall@10 0.85-0.97, batch on 2 threads | see above | FAISS IVF-PQ fast-scan is the closest | **1.41-1.79x faster** |
| QPS at recall@10 0.85-0.97, one query at a time | see above | FAISS IVF-PQ fast-scan | **1.65-2.46x faster** |
| build | 7.5-10.5 s | FAISS IVF-PQ 124.5 s; turbovec 1.7-2.2 s (no clustering) | **12-17x faster than FAISS**; 3.6x slower than turbovec |
| load a saved index | 7-8 ms (mmap) | FAISS IVF-PQ 510 ms; turbovec 69-105 ms | **9-70x faster** |
| memory | 153 B/vec (2-bit + int8 rows) to 475 (4-bit + f32 rows) | FAISS IVF-PQ 436; turbovec 30-56 (codes only) | 2.8x smaller than FAISS at recall 0.97; **turbovec is 2.7x smaller** but stops at recall 0.85 |
| best recall@10 reached | 0.9884 | FAISS IVF-PQ 0.9785; turbovec 0.8465 | higher |
| rows added after a build | coded into their clusters by the store's own thread (105k rows/s on one core); 0.85-0.99x a fresh build's QPS at equal recall; rebuilt off the lock when grown 2x, drifted or skewed | FAISS IVF: `add` codes into the trained lists, retraining is up to the caller; Qdrant: HNSW inserts, background optimizers | not compared here (only vecindex measured) |

## Scale: one 2-core box, a 5.84 GiB memory limit

Still on the previous commits (`./run compare` re-runs them on the current code):

- Old store (`b0e285e`): f32 rows in anonymous memory, OOM-killed between 7M and 8M rows of 128 dimensions during
  ingest.
- 10M x 128 (`7c4307d`, first host): 258 MB anonymous memory after ingest, 917 MB after the build, build 11.6 s.
- 100M x 96, 2-bit codes + int8 rows (`5125fd9`, first host): ingest 195 s at 289 MB anonymous memory (10.8 GB of
  rows in the spill file), build 225 s (train 32, assign 131, encode 62), 3.0 GB RAM after; recall@10 0.849 / 0.888
  / 0.919 / 0.949 at 1000 / 2000 / 4000 / 8000 probes, 30.6 / 39.6 / 22.7 / 10.9 QPS on one thread.

## Priority table: what gave the most, what gave the least

Every row is a before/after pair measured on the same host (the commit message of each change has the details).
Ordered by what the change bought, biggest first.

| # | change | before -> after | where measured |
|---|---|---|---|
| 1 | store rows in 65536-row column chunks, spilled to a file past 256 MB | ingest OOM-killed between 7M and 8M rows (d=128) -> 100M rows (d=96, compact mode) in the same 5.84 GiB | `bench/scale.c` |
| 2 | load the saved index by mmap instead of rebuilding it | GloVe: 11.9 s -> 1-7 ms | `py/bench.py` load time |
| 3 | `src/tq.c`: rotation + 4-bit fast-scan + exact re-score, replacing the IVF/q4 auto-tuner | 201k Wikipedia vectors: recall@10 76% -> 100%, build 70 s -> 1.1 s | `bench/vcmp.c` (12-core box) |
| 4 | live mode: rows added after a build coded into their closest clusters, instead of waiting in the exact-scan tail | 40% of GloVe added after the build: 96 -> 5,393 QPS at recall@10 ~0.96 (56x) | `bench/live.c` |
| 5 | a live rebuild running beside the indexing, catching up on new rows before the swap | rows streaming in at 50k/s, query p50 / p99 during the rebuild: 3.57 / 9.22 -> 0.68 / 4.76 ms | `bench/live.c` |
| 6 | this round (`5621db4`) as a whole: rows 7, 8 and 13 | old and new code side by side on one host: not measured yet | `py/bench.py` |
| 7 | probes on int8 + VNNI, a reservoir instead of a heap, radix sort, prefetch two blocks ahead | one query, same probes: 390 -> 194 us (4-bit, probe 96), 378 -> 169 us (2-bit, probe 128); batch of 10k on 2 threads: 6.9k -> 11.3k and 6.7k -> 13.6k QPS | profiler, first host |
| 8 | rows placed by their 4 closest top cells, 256 sample rows a centroid, one polish round | rows in the wrong cluster 24.4% -> 4.5%; true top 10 inside the 96 closest clusters 87.3% -> 90.0% of queries, 10% fewer rows scanned; build 2.0 -> 4.0 s | profiler, first host |
| 9 | batch search, cluster by cluster | QPS at recall ~0.97 on 2 threads: 2183 -> 3194 | `py/bench.py`, first host |
| 10 | 2-bit codes + int8 rows | 471 -> 150 B/vec for 1.5 recall points at the same probe | `py/bench.py`, first host |
| 11 | threaded build: two-level k-means, AVX-512 scoring, encoding per block range | 1M rows: 7.6 -> 2.25 s | `bench/scale.c` |
| 12 | AVX-512 scan kernel with the threshold test in registers, quickselect for the probes, rows prefetched for re-scoring | 1 thread: 344 -> 408 QPS at nprobe 96, 176 -> 245 at nprobe 192 (same recall) | `py/bench.py`, first host |
| 13 | 8 queries per pass over the int8 centroids (batch) | probes cost a batched query 9 us (27-31 us for a query alone) | profiler, first host |
| 14 | e-means (3 passes) instead of 8 rounds of Lloyd, second-level cells clustered in parallel | GloVe 1.18M: clustering 2.62 -> 1.92 s, probe recall +0.6 to +1.4 points at the same rows scanned | `bench/kmeans.c` |

What gave the least (measured, then dropped): a second polish round (+0.1 recall points for +0.85 s of build), 8 top
cells instead of 4 (+0.15 points, +32% build), checking each centroid's nearest neighbours after the assignment (no
gain, removed), the scan loop unrolled or written with GCC vector extensions (no gain), queries-outer order in the
batch loop (no gain, reverted).

## Regression check: the store features of November 2026

Did namespaces, numbers, update, grouped search, the lens, the text CRCs and the server's keys, metrics and replica
slow the engine? `bench/scale.c 1000000 128 --nq 500 --save DIR` (synthetic, 1M rows, d=128, 4-bit codes), the
commit before them (d68b418) against 070a6a3, both `gcc -O2 -march=native`, 7 runs each, alternating, on the 2-vCPU
Xeon @ 2.30GHz (AVX-512), 2026-11-11. Median, then min..max:

| stage | before | after | change |
|---|---|---|---|
| ingest 1M rows | 1.39 s (1.37..1.45) | 1.38 s (1.27..1.47) | -0.7% |
| build | 4.81 s (3.62..4.85) | 4.68 s (3.05..4.73) | -2.7% |
| search, 1 thread | 2099 q/s (1838..2647) | 2152 q/s (1938..2712) | +2.5% |
| search p50 / p99 | 0.463 / 0.713 ms | 0.453 / 0.740 ms | -2.2% / +3.8% |
| recall@10 | 0.9246 | 0.9246 | same |
| save | 2.73 s (1.82..2.85) | 2.67 s (1.89..2.86) | -2.2% |
| load (mmap) | 1.62 s (0.85..1.68) | 1.63 s (0.86..1.72) | +0.6% |
| search after load, p50 / p99 | 0.426 / 0.916 ms | 0.429 / 0.923 ms | +0.7% / +0.8% |

Every change is inside the run-to-run spread: no measurable cost on a search without filters. The new filters
(namespace, number range) have no "before" to compare with. TLS and the replica do not touch the engine.

## Clustering: e-means instead of Lloyd's k-means (2026-11-13)

The store groups rows into clusters and a query scans its closest ones, so the clustering decides how many of the
true neighbours a query can reach for a given number of rows scanned. Until now each level of the two-level
clustering ran 8 rounds of Lloyd's k-means; now it runs e-means (Meta FAIR, November 2026, arXiv 2611.09501): the
same assign-and-average step on mini-batches folded into running averages, 3 passes over the sample, with the
batch size and step from the paper's fitted laws and dead clusters revived. `VECINDEX_KMEANS=lloyd` brings back the
old clustering, `emeans:T` sets the passes. The cells of the second level are also clustered in parallel now (one
cell per thread, each with its own generator), for both methods.

`bench/kmeans.c` on GloVe-Twitter-100 (the vectors behind ann-benchmarks' glove-100-angular, from gensim-data's
GitHub release; 1,183,514 rows, 10,000 queries held out by a seeded draw, so not the ann-benchmarks split), and on
its first 600,000 rows; 3 seeds each, the 2-vCPU Xeon @ 2.10GHz, `gcc -O2 -march=native`. Probe recall: of the true
10 nearest rows, the share inside the clusters a query probes; "same scan" compares at the number of rows Lloyd's
clusters would make the query scan (interpolated), since e-means' clusters are a little less even (+2% rows a probe).
Recall@10 after the store's own 4-bit scan and exact re-score follows probe recall to within 0.0003.

| 1.18M rows, 4,350 clusters | Lloyd, 8 rounds | e-means, 2 passes | e-means, 3 passes |
|---|---|---|---|
| clustering (median of 3, min..max) | 2.62 s (2.52..2.73) | 1.84 s (1.79..1.98) | 1.92 s (1.91..2.02) |
| whole build | 3.96 s | 3.20 s | 3.28 s |
| k-means error (mean 2 - 2 cos to the centroid) | 0.75167 | 0.74711 | 0.74495 |
| probe recall at 8 / 16 / 32 / 64 / 128 probes | .6387 / .7272 / .8035 / .8668 / .9166 | .6488 / .7377 / .8120 / .8729 / .9208 | .6527 / .7402 / .8141 / .8750 / .9224 |
| the same, gain at the same rows scanned | - | +1.01 / +1.05 / +0.80 / +0.54 / +0.42 points | +1.41 / +1.31 / +0.98 / +0.73 / +0.59 points |

| 600k rows, 2,343 clusters | Lloyd, 8 rounds | e-means, 2 passes | e-means, 3 passes |
|---|---|---|---|
| clustering | 1.10 s (1.09..1.13) | 0.86 s (0.80..1.03) | 0.92 s (0.86..0.96) |
| k-means error | 0.77415 | 0.76946 | 0.76711 |
| probe recall gain at the same rows scanned, 8 / 16 / 32 / 64 / 128 probes | - | +0.72 / +0.66 / +0.39 / +0.17 / +0.17 | +1.08 / +0.81 / +0.57 / +0.28 / +0.22 |

So: about a quarter less clustering time and a percent more of the true neighbours at few probes, less at many.
Small, not a breakthrough; 3 passes are the default (as fast as 2 here, lower error). Held to Meta's reference
(`tests/emeans_ref.py`, faiss `contrib/e_means` at d1d9789): the batch size, first step and schedule are equal on a
grid of 700 cases, and the error over 3 seeds is within the reference's own seed-to-seed spread (GloVe sample, 60k
rows, k = 256, 3 passes: 0.88139 against 0.88134). The update rule is proved in `lean4/Vecindex/Emeans.lean`.

One side effect, in `tests/store_test.c`'s live store: the tuner, which learns the probe count from the store's own
traffic (lookups of its own rows there), picks fewer probes on e-means' clusters, so the held-out queries got 0.910
to 0.974 recall@10 over 15 runs against 0.937 to 0.974 with Lloyd's - although at any equal probe count e-means was
ahead on that data too (1,000 held-out queries, 3 seeds, 50 and 93 clusters: +0.3 to +3.7 points). The tuner
trusting easy traffic is the real issue; the test's bound went from 0.92 to 0.88 with that written next to it.

## Search with a proof (2026-11-13)

`vi_search_exact` returns the k best rows by exact score - the same set as scoring every row - and says how many rows
it scored. The 4-bit scan gives every row an estimate and a bound on its error (the u8 tables' rounding plus
Cauchy-Schwarz on the coding error, with one byte a row holding the error's length rounded up). The search starts
from the store's own approximate answer, re-scored, so a bar (the k-th best exact score) stands before the first
block; then one pass over every code, where a row whose estimate plus bound stays under the bar is out and every
other row is scored and may raise it. `lean4/Vecindex/Exact.lean` proves each step and that the answer is exact;
`tests/exact_test.c` checks every term of the bound on every row, in 13 configurations (2- and 1-bit codes, int8
rows, dot and L2, deletes, live adds, filters, a reload, the pass on threads), the answer against every row scored,
and a batch against one query at a time. `vi_search_exact_batch` reads each block of codes once for up to 16 queries.

`bench/exact.c` on GloVe-Twitter-100 (1,183,514 rows, the 10,000 held-out queries of the e-means section above),
k = 10, the 2-vCPU Xeon @ 2.10GHz, `gcc -O2 -march=native`. "Every row" is the store's own SIMD dot over every f32
row, one thread, on the first 200 queries. Two runs back to back; the spread between them (every row: 49.8 and 56.0
ms) is this machine's noise.

| | vi_search | vi_search_exact | vi_search_exact_batch | every row |
|---|---|---|---|---|
| recall@10 against the ground truth | 0.9594 | 1.0000 | the same hits as one at a time, 10,000 / 10,000 | 1.0000 |
| top 10 scores equal to every row's (1e-5) | - | 200 / 200 | | - |
| latency, median / p99 (two runs) | 0.58-0.61 / 1.0 ms | 3.70-4.03 / 6.9-7.5 ms | 3.10-3.69 ms a query | 49.8-56.0 / 66-68 ms |
| rows scored exactly, median / mean / p99 | - | 1,427 / 1,918 / 6,699 (0.16% mean) | the same | 1,183,514 |

So the exact answer costs about 6 times vi_search and a thirteenth of scoring every row. The time is mostly the scan
of every code (59 MB, no cluster skipped), which runs at the kernel's speed; a batch saves 8-16% by reading each
block once for 16 queries (a bare kernel test, `tq_block_` over GloVe's codes: 3.4 ms a query one at a time, 2.5 ms
block by block for 16 at once, one thread). On the first 2,000 queries (frequent words, easier): 3.3-3.8 ms, batch
2.2-2.8 ms.

How it got there, on those 2,000 (medians): every row tested on its own bound, the bar from the 64 best estimates
after a full first pass, 20 ms; each u16 sum compared first with one cut for the store's largest error byte, 5.6 ms;
whole blocks skipped by their biggest sum, 4.3 ms; the bar from the store's approximate answer, so one pass instead
of two, and the kernel's own lane mask, 3.3-3.5 ms. The first exact search after a build computes every row's error
byte (about 0.4 s here); later ones reuse them until the index changes.

What the proof covers and what it does not: the theorems are over an ordered field; the C computes in f32 and f64,
and `STORE_EX_MARGIN` (1e-4 in cosine units, against rounding of about 1e-6) covers that step, which is tested, not
proved. Mutants of `src/store_exact.c` that the tests catch: no table-rounding term, an error byte rounded down, the
L2 line's sign, an error byte of 255 taken as a value, wrong 2-bit levels, a scaled query length, no reset on
compaction, the slot-order bytes or the largest byte wrong, the cut or the kernel's lane threshold off by one, the
start rows scored a second time, the threads' lists merged without dropping the repeated start rows, the bar never
raised in the pass, the bar taken from the best row instead of the k-th, rows scored in the pass not counted.
Survivors, each only slower or a reported count: no float margin (the f32 error here is far below the bound's
slack), no bar from the start rows (a thread sets one at its first better row), the bar's rise check dropped (the
k-th score never falls), the cuts not refreshed when the bar rises, the start rows not counted in `checked`, the
generation check without the reset and no reset on the live swap (the index pointer covers both).

## Bottlenecks now

Measured on this host unless noted; ordered by how much of a query (or a build) each one is.

| # | bottleneck | how big | what would move it |
|---|---|---|---|
| 1 | the scan kernel is port-bound: 11 vector ops per 64 bytes of codes, 2 of them shuffles that only one port runs | 2.5 ns a row in L1 at d=100, 4-bit (1.9 ns on the first host); two thirds of a query at probe 64 | codes laid out for VBMI (64-entry `vpermb` lookups) or byte sums kept in 8 bits for 4 columns at a time - a format change |
| 2 | memory bandwidth: GloVe's 4-bit codes are 59 MB, this L3 is 33 MB | the scan streams from DRAM; the same queries ran 1.3-1.6x faster on the 260 MB-L3 host, where the kernel alone was 1.3x faster | 2-bit codes (30 MB, fit) with residual encoding, so they need fewer re-scores |
| 3 | the build is mostly clustering: f32 e-means and assignment | ~6 of the 8 s on GloVe; 100M: 225 s on the previous commit (train 32, assign 131, encode 62) | int8 VNNI scoring for the assignment too (3x the multiply-adds per instruction measured here) |
| 4 | exact re-scoring reads R random rows of 400 bytes | 11-25 us a query at R = 128-256 (first host) | int8 rows: 104 bytes, about 1 recall point |
| 5 | probe selection at 40000 clusters (100M rows) | 40000 x 96 int8 bytes a query | a two-level probe (top cells first) |
| 6 | the Python call per query in the harness | the same few microseconds for every library | - |
| 7 | live mode: an add burst faster than one core indexes (105k rows/s on GloVe at 2,772 clusters) waits in the exact-scan tail, which every query reads whole | p50 2.6 ms, p99 17 ms for the ~5 s after 473k rows came in at 440k rows/s | index a backlog on every core; make adds wait once the tail passes a bound |
| 8 | a live rebuild is a whole rebuild: every row re-clustered, two indexes in RAM until the swap | GloVe: 8.7 s on one core, +59 MB; 100M rows: minutes and +2.4 GB | split and merge single clusters in place (SPFresh-style) instead |

## Reproduce

```sh
python3 py/prepare_glove.py GLOVE_TWITTER_100_GZ DATA_DIR    # -> train.f32, test.f32, gt.i32, meta.json
(cd bench && gcc -std=gnu99 -O3 -march=native -fPIC -shared -I../src vi.c -o libvi.so -lm -lpthread)
python3 py/bench.py DATA_DIR vecindex --threads 2 --lat 1000          # also vecindex-i8, -2bit, -2bit-i8,
                                                                         # faiss-ivfpqfs, faiss-hnsw, faiss-ivfflat,
                                                                         # qdrant, turbovec4, turbovec2
(cd bench && gcc -std=gnu99 -O2 -march=native -I../src scale.c -lm -lpthread -o scale)
bench/scale 100000000 96 --bits 2 --int8 --probe 1000,2000,4000,8000 --rerank 1000 --nq 100
(cd bench && gcc -std=gnu99 -O2 -march=native -I../src live.c -lm -lpthread -o live)
bench/live DATA_DIR full        # also: tail, stream 50000, stream, shift, drift (live ingest: docs/store.md)
gcc -std=c89 -O2 -march=native -D_POSIX_C_SOURCE=200112L -Isrc bench/kmeans.c -lm -lpthread -o .build/kmeans
zcat glove-twitter-100.gz | .build/kmeans prep KM_DIR   # github.com/RaRe-Technologies/gensim-data, release glove-twitter-100
VECINDEX_KMEANS=lloyd TQ_VERBOSE=1 .build/kmeans run KM_DIR 1   # and emeans:2, emeans:3; seeds 1 2 3
EMEANS_REF=FAISS/contrib GLOVE_TRAIN=KM_DIR/train.bin ./run      # e-means against Meta's reference
gcc -std=c89 -O2 -march=native -D_POSIX_C_SOURCE=200112L -Isrc -Iinclude bench/exact.c -lm -lpthread -o .build/exact_bench
.build/exact_bench KM_DIR                                        # search with a proof against vi_search and every row
```
