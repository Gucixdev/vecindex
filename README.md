# vecindex

An embedded vector index and note database in C89: a TurboQuant-style rotated 4-bit (or 2-bit) fast-scan with
exact re-scoring, IVF clustering, bit-packed 1-5 bit quantizers, BM25 hybrid search and an optional
agent-memory layer - plus `vserv.c`, a small epoll server for it (Unix and TCP sockets, a binary protocol and
HTTP/1.1, TLS 1.2 through the vendored BearSSL). Nothing beyond libc, libm and pthreads (and BearSSL for TLS).

```sh
git clone https://github.com/Gucixdev/vecindex && cd vecindex
./run            # self-tests, fuzzers, thread and crash-safety tests, the server end to end (needs bash and gcc)
./run stress     # plus the churn / oracle stress tests
```

It used to live in [holyc](https://github.com/Gucixdev/holyc) as `ragdb/`; its history came along.

## Results

Everything in this section was measured on one machine, in one session: 2 vCPUs of an Intel Xeon @ 2.80GHz (Cascade
Lake: AVX-512 with VNNI, 1 MB of L2 per core, 33 MB of L3 shared), the shell's cgroup capped at 5.84 GiB, Linux 6.18.
FAISS 1.15.1 (pip), turbovec 1.1.2, qdrant-edge-py 0.8.0 and vecindex (through a ctypes shim, `bench/vi.c`) all run
through the same harness, `bench/bench.py`, so every library pays the same Python call per query.

The cloud VM this ran on changed hardware during the work (it restarted onto this host from one with a 260 MB L3), so
the before/after numbers of individual changes in the priority table were measured as pairs on whichever host was up
at the time - never one number from each host.

### GloVe-100: queries per second at the same recall

glove-100-angular prepared the ann-benchmarks way: 1,183,514 vectors of 100 dimensions, 10,000 held-out queries,
recall@10 against the exact top 10. Each QPS cell is **one batch of 10k queries on 2 threads / one query per call on
1 thread**, at the fastest setting each method has that reaches the recall (the settings swept are in `bench.py`:
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
their runs three times. On the first host they were slower than FAISS IVF-PQ at every recall. `bench/run_all.sh
DATA_DIR` runs every row of this table, the scale runs and vecdb's 8M notes in one go on any Linux box, and
`bench/tables.py` turns its output into these tables.

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

### Where vecindex wins and where it loses

| axis | vecindex | the others | verdict |
|---|---|---|---|
| QPS at recall@10 0.85-0.97, batch on 2 threads | see above | FAISS IVF-PQ fast-scan is the closest | **1.41-1.79x faster** |
| QPS at recall@10 0.85-0.97, one query at a time | see above | FAISS IVF-PQ fast-scan | **1.65-2.46x faster** |
| build | 7.5-10.5 s | FAISS IVF-PQ 124.5 s; turbovec 1.7-2.2 s (no clustering) | **12-17x faster than FAISS**; 3.6x slower than turbovec |
| load a saved index | 7-8 ms (mmap) | FAISS IVF-PQ 510 ms; turbovec 69-105 ms | **9-70x faster** |
| memory | 153 B/vec (2-bit + int8 rows) to 475 (4-bit + f32 rows) | FAISS IVF-PQ 436; turbovec 30-56 (codes only) | 2.8x smaller than FAISS at recall 0.97; **turbovec is 2.7x smaller** but stops at recall 0.85 |
| best recall@10 reached | 0.9884 | FAISS IVF-PQ 0.9785; turbovec 0.8465 | higher |

### Scale: one 2-core box, a 5.84 GiB memory limit

Still on the previous commits (`bench/run_all.sh` re-runs them on the current code):

- Old store (`b0e285e`): f32 rows in anonymous memory, OOM-killed between 7M and 8M rows of 128 dimensions during
  ingest.
- 10M x 128 (`7c4307d`, first host): 258 MB anonymous memory after ingest, 917 MB after the build, build 11.6 s.
- 100M x 96, 2-bit codes + int8 rows (`5125fd9`, first host): ingest 195 s at 289 MB anonymous memory (10.8 GB of
  rows in the spill file), build 225 s (train 32, assign 131, encode 62), 3.0 GB RAM after; recall@10 0.849 / 0.888
  / 0.919 / 0.949 at 1000 / 2000 / 4000 / 8000 probes, 30.6 / 39.6 / 22.7 / 10.9 QPS on one thread.
- vecdb, 8M notes (`10bb6d8`, first host): ingest 146 s, build 19 s, 3.6 GB anonymous memory, hybrid search 7.7
  ms/query, MCTS explore 53 ms/query. The old code died at ~4M.

### Priority table: what gave the most, what gave the least

Every row is a before/after pair measured on the same host (the commit message of each change has the details).
Ordered by what the change bought, biggest first.

| # | change | before -> after | where measured |
|---|---|---|---|
| 1 | store rows in 65536-row column chunks, spilled to a file past 256 MB | ingest OOM-killed between 7M and 8M rows (d=128) -> 100M rows (d=96, compact mode) in the same 5.84 GiB | `bench/scale.c` |
| 2 | load the saved index by mmap instead of rebuilding it | GloVe: 11.9 s -> 1-7 ms | `bench.py` load time |
| 3 | `tq.c`: rotation + 4-bit fast-scan + exact re-score, replacing the IVF/q4 auto-tuner | 201k Wikipedia vectors: recall@10 76% -> 100%, build 70 s -> 1.1 s | `vcmp.c` (12-core box) |
| 4 | this round (`5621db4`) as a whole: rows 5, 6 and 12 | old and new code side by side on one host: not measured yet | `bench.py` |
| 5 | probes on int8 + VNNI, a reservoir instead of a heap, radix sort, prefetch two blocks ahead | one query, same probes: 390 -> 194 us (4-bit, probe 96), 378 -> 169 us (2-bit, probe 128); batch of 10k on 2 threads: 6.9k -> 11.3k and 6.7k -> 13.6k QPS | profiler, first host |
| 6 | rows placed by their 4 closest top cells, 256 sample rows a centroid, one polish round | rows in the wrong cluster 24.4% -> 4.5%; true top 10 inside the 96 closest clusters 87.3% -> 90.0% of queries, 10% fewer rows scanned; build 2.0 -> 4.0 s | profiler, first host |
| 7 | batch search, cluster by cluster | QPS at recall ~0.97 on 2 threads: 2183 -> 3194 | `bench.py`, first host |
| 8 | 2-bit codes + int8 rows | 471 -> 150 B/vec for 1.5 recall points at the same probe | `bench.py`, first host |
| 9 | vecdb: varint postings, 12-byte hash slots, a slab | 669 -> 392 B a note; 4M -> 8M notes in the cgroup | `vbench1m.c` |
| 10 | threaded build: two-level k-means, AVX-512 scoring, encoding per block range | 1M rows: 7.6 -> 2.25 s | `bench/scale.c` |
| 11 | AVX-512 scan kernel with the threshold test in registers, quickselect for the probes, rows prefetched for re-scoring | 1 thread: 344 -> 408 QPS at nprobe 96, 176 -> 245 at nprobe 192 (same recall) | `bench.py`, first host |
| 12 | 8 queries per pass over the int8 centroids (batch) | probes cost a batched query 9 us (27-31 us for a query alone) | profiler, first host |

What gave the least (measured, then dropped): a second polish round (+0.1 recall points for +0.85 s of build), 8 top
cells instead of 4 (+0.15 points, +32% build), checking each centroid's nearest neighbours after the assignment (no
gain, removed), the scan loop unrolled or written with GCC vector extensions (no gain), queries-outer order in the
batch loop (no gain, reverted).

### Bottlenecks now

Measured on this host unless noted; ordered by how much of a query (or a build) each one is.

| # | bottleneck | how big | what would move it |
|---|---|---|---|
| 1 | the scan kernel is port-bound: 11 vector ops per 64 bytes of codes, 2 of them shuffles that only one port runs | 2.5 ns a row in L1 at d=100, 4-bit (1.9 ns on the first host); two thirds of a query at probe 64 | codes laid out for VBMI (64-entry `vpermb` lookups) or byte sums kept in 8 bits for 4 columns at a time - a format change |
| 2 | memory bandwidth: GloVe's 4-bit codes are 59 MB, this L3 is 33 MB | the scan streams from DRAM; the same queries ran 1.3-1.6x faster on the 260 MB-L3 host, where the kernel alone was 1.3x faster | 2-bit codes (30 MB, fit) with residual encoding, so they need fewer re-scores |
| 3 | the build is mostly clustering: f32 k-means and assignment | ~6 of the 8 s on GloVe; 100M: 225 s on the previous commit (train 32, assign 131, encode 62) | int8 VNNI scoring for the assignment too (3x the multiply-adds per instruction measured here) |
| 4 | exact re-scoring reads R random rows of 400 bytes | 11-25 us a query at R = 128-256 (first host) | int8 rows: 104 bytes, about 1 recall point |
| 5 | probe selection at 40000 clusters (100M rows) | 40000 x 96 int8 bytes a query | a two-level probe (top cells first) |
| 6 | the Python call per query in the harness | the same few microseconds for every library | - |

### Reproduce

```sh
python3 bench/prepare_glove.py GLOVE_TWITTER_100_GZ DATA_DIR    # -> train.f32, test.f32, gt.i32, meta.json
(cd bench && gcc -std=gnu99 -O3 -march=native -fPIC -shared -I.. vi.c -o libvi.so -lm -lpthread)
python3 bench/bench.py DATA_DIR vecindex --threads 2 --lat 1000          # also vecindex-i8, -2bit, -2bit-i8,
                                                                         # faiss-ivfpqfs, faiss-hnsw, faiss-ivfflat,
                                                                         # qdrant, turbovec4, turbovec2
(cd bench && gcc -std=gnu99 -O2 -march=native -I.. scale.c -lm -lpthread -o scale)
bench/scale 100000000 96 --bits 2 --int8 --probe 1000,2000,4000,8000 --rerank 1000 --nq 100
gcc -std=gnu99 -O2 -march=native -I. vbench1m.c -lm -lpthread -o .build/vbench1m && .build/vbench1m 8000000
```

## Overview

A note/vector database (`vecdb.c`, on `vecindex` + `hdict.c` + `lex.c` + `store.c`) plus an
optional agent-memory layer on top of it (`mem.c`) - embedded, C89, no server required. This
section is the executive summary; every claim in it is backed by a test in `./run` and a detailed
section further down.

## Production readiness

**Verdict: production-ready as an embedded, single-node library.** Not equivalent to a hosted,
multi-tenant vector-DB service (Pinecone/Qdrant/Milvus) - that would be a different project, not a
gap this repo is missing by oversight. Every number below is measured, not estimated: the older rows on a
GTX 1050 Ti box (7.8 GB RAM, 12 cores), the scale, server and benchmark rows on the 2-core box of the Results
section; run `./run` / `./run stress` / `./run bench` / `./run real_val` yourself to reproduce any of it.

| area | status | evidence |
|---|---|---|
| Crash safety | atomic per-file + cross-file (generation-suffix) writes | `vcrash.c`/`vcrash2.c`, 25-90 rounds, always consistent |
| Concurrency | one rwlock per layer, both TSan-clean | `vthreads.c`/`mthreads.c` in `./run` |
| Fuzzing | 3 fuzzers (parser, rule engine, malformed input) | `vfuzz.c`/`mrfuzz.c` - the rule-parser one found and fixed a real stack-overflow crash on its first run |
| Adversarial stress | real invariant checks, not just "doesn't crash" | `stress.c`/`mstress.c` - found and fixed 5 real bugs this way |
| Static analysis | clang analyzer + clang-tidy across the core, findings triaged not just counted | 2 real (currently-unreachable) issues fixed, the rest confirmed false-positive/non-actionable by hand-tracing, not dismissed on sight |
| Scale | measured to the actual breaking point, not extrapolated | store: 100M vectors (d=96, 2-bit + int8) ingested, built and searched in a 5.84 GiB cgroup on 2 cores; the old store was OOM-killed between 7M and 8M. vecdb: 8M notes end to end where the old code died at ~4M (see Results) |
| Server | `vserv.c`: epoll, Unix + TCP, HTTP/1.1 + binary, TLS 1.2 (BearSSL), mutual TLS | `bench/vserv_test.py` (26 checks) and `bench/vserv_tls_test.py` (22 checks) in `./run` |
| Retrieval quality | a real SQuAD-based benchmark, not hand-labeled questions | `real_val.c`: gold@10 up to 85.6%, vs an 87.1% ceiling for this embedder (measured directly) - the ANN index is near-lossless relative to what the embedder allows |
| Reranking | a real cross-encoder second pass, with a measured effect | `rerank.c`: MRR 0.751->0.887, gold@1 62.1%->81.5% on the subset where it has room to help |
| Independent review | **not done** | every test here was written and run by the same person who wrote the code - real bugs were still found this way, but this is not the same claim as external review |
| Multi-tenancy / auth / replication / HA | **not present, out of scope** | this is a library, not a service - would be a different project, not a fix |

Full comparison to mainstream RAG/vector-DB stacks (what's genuinely ahead, what's genuinely
behind, no hedging either way): see "How this compares" near the end of this file.

## vserv: the store over a socket

`vserv.c` serves one store file: Linux epoll, one worker thread per core, a fixed request buffer per connection
(allocated on its first byte, reused after), replies built in a per-thread buffer - a request allocates nothing.
The store is opened with `store_load` (mmap), so a 100M-row file is ready in milliseconds.

```sh
./run                                    # builds .build/vserv (BearSSL into .build/libbearssl.a first)
.build/vserv -s /run/vx.sock -p 8077 -t 2 -x docs.vxi            # Unix socket (0600) + TCP on 127.0.0.1
.build/vserv -p 8443 -c cert.pem -k key.pem [-a clients-ca.pem] [-T] docs.vxi   # TLS, optional mutual TLS
curl -s localhost:8077/stats
curl -s -X POST 'localhost:8077/search?k=5' -d '[0.1, 0.2, ...]'
```

Every socket speaks two protocols, told apart by the first byte: HTTP/1.1 (`GET /health`, `GET /stats`,
`POST /search?k=`, `/add?id=`, `/delete?id=`, `/build`, `/save`; vectors as a JSON array or raw little-endian
f32, keep-alive, pipelining) and a binary protocol (`0xB5`, op, k, length; search/add/delete/build/save/stats)
with nothing to parse. The HTTP header scan looks for the blank line 16 bytes at a time with SSE2 (the
picohttpparser idea, cut down to what this server needs).

TLS (`-c`/`-k`): TLS 1.2 through the vendored [BearSSL](https://bearssl.org) 0.6 (MIT, unmodified, no crypto
code of our own): ECDHE only (forward secrecy), AEAD suites only (AES-GCM with AES-NI, ChaCha20-Poly1305), EC or
RSA keys, sessions resumed from one cache shared by every worker, keys wiped when a connection closes. `-a CA`
requires client certificates (mutual TLS). TLS is sniffed per connection (`0x16`): the TCP port takes only TLS
(`-P` lets plaintext in too); the Unix socket takes TLS or, for local tools, plaintext (`-T`: TLS only).

Measured: 2.4 MB idle RSS; 4.2 KB per open plaintext connection, 20.7 KB per TLS connection (300 open). From
Python clients on 2 cores: ~15-19k searches/s plaintext, ~10.7k over TLS on the first 2-core host; 8.9-10.6k and
5.7-7.5k on the Cascade Lake host of the Results section (2k rows; the Python clients are the bottleneck).
`bench/vserv_test.py` (26 checks: both protocols, both sockets, pipelining, a request trickled byte by byte,
oversized and malformed input, 8 concurrent clients, SIGTERM saves, restart) and `bench/vserv_tls_test.py` (22
checks: resumption on every reconnect, multi-record replies, plaintext refused, untrusted CA, mTLS accept/reject,
RSA key, curl, TLS on the Unix socket) run in `./run`.

## vecdb: notes on top of vecindex

`hdict.c` (open-addressing hash dictionaries, i64 and string keys), `lex.c` (BM25
term dictionary + postings) and `vecdb.c` sit on `store.c`.

* **Hash dictionary inside the index** - `store.c` keeps an id -> row map, so
  `store_has/get_vec/set_meta/get_meta`, delete and replace are O(1) instead of a scan.
* **Precision** - `vecdb_search` fuses vector kNN and BM25 with reciprocal rank
  fusion, so rare exact words ("Delft") win even when the embedding blurs them.
* **Graph** - `[[wikilinks]]` (alias, `#heading`, `![[embeds]]`), backlinks,
  `#hashtags` (nested `#a/b` also tag `#a`, ignored in code/URLs/headings),
  ghost nodes for links to notes that do not exist yet.
* **MCTS** - `vecdb_explore` runs UCB1 tree search over links, backlinks and tags
  from the hybrid hits and surfaces bridge notes that plain kNN misses.
* CLI: `gcc ... vecdb.c -DVECDB_CLI -lm -lpthread -o vecdb_cli`, then
  `vecdb_cli DB.vdb index DIR | search Q | explore Q | links T | backlinks T | tag T | tags | ghosts`.
* Tests: `./run` (includes hdict, lex, vecdb), `./run stress`.

### vecdb benchmarks

`vbench.c` (synthetic, any N), `vbench_wiki.c` (real Simple English Wikipedia:
`bzcat simplewiki-latest-pages-articles.xml.bz2 | python3 wikiprep.py > wiki.rec`, then
`.build/vbench_wiki wiki.rec`) and `vcmp.c` + `vcmp_turbovec.py` (vecindex vs turbovec on identical vectors).

201k articles, 12 cores, hit rates over 300 queries (hashed 256-d embedder):

| | vecindex (vector only) | vecdb hybrid | vecdb MCTS |
|---|---|---|---|
| body's first 12 words -> article in top 10 | 72% | 97% | 96% |
| title -> exact article first | 23% | 100% | 100% |
| title -> its real link neighbours (recall@10) | 5% | 21% | 32% |
| ms / query | 1.6 | 2.3 | 3.7 |

### TurboQuant lesson (fast vector index)

`vcmp` on the same 201k x 256 vectors, recall@10 against exact search:

| index | build | recall@10 | ms/query |
|---|---|---|---|
| vecindex auto-tuned (IVF+q4) | 70 s | 76% | 7.9 |
| vecindex IVF fast build | 8.6 s | 65% | 2.1 |
| vecindex flat q4 | 2.9 s | 56% | 24.8 |
| turbovec 4-bit (TurboQuant) | 0.35 s | 86% | 1.5 |
| turbovec 2-bit | 0.23 s | 59% | 0.5 |
| **`tq.c` (rotation + 4-bit fast-scan + exact re-score)** | **1.1 s** | **100%** | **1.2-1.4** |

`tq.c` copies what makes TurboQuant work: a fixed random rotation (signs + Walsh-Hadamard) turns every
coordinate into ~N(0,1/d) for any data, so one fixed Lloyd-Max 4-bit codebook is near-optimal with no
training; codes sit in a fast-scan layout scored with PSHUFB (AVX2 or AVX-512, threaded above 150k rows). The scan only
proposes 256+ candidates; the f32 rows the store keeps re-score them exactly, which is what lifts 86% to
100%. `store_set_fast` selects it (vecdb does); the IVF auto-tuner remains for the plain store API.

### Persistence

`vecdb_save` writes PATH.vdb (node table), PATH.vdt (bodies, stay on disk), PATH.vxi (vector rows) and
PATH.img.vxi. `vecdb_sync` afterwards only appends to journals (`store_sync`, PATH.vxi.j, a torn tail is
ignored on load), so a save costs the change, not the store. Loading rebuilds links, tags and BM25 from the
bodies (~10 us per note); the vector rows and the TurboQuant index are mapped from PATH.vxi as they are (no
rebuild, pages come in when first touched).

### Real embeddings, documents, pictures

`db->embed` / `vecdb_use_embproc` plug in any model through a child process; `embedsrv.py` serves
multilingual-e5-small (Polish ok) and `--clip` serves CLIP for pictures. The CLI (`VECDB_EMBED_SERVER`,
`VECDB_CLIP_SERVER` env vars, `repl` keeps the models loaded) indexes .md/.txt, and through `extract.py`
pdf/docx/odt/pptx/epub/html: long documents become a document note linked to overlapping-free chunks
(each chunk links back and to its neighbours), pictures go to a second vector space and are linked by
`![[pic.png]]` / `![](pic.png)`. Search: `search`, `images` (text -> pictures), `similar` (picture -> pictures).

### Cross-encoder reranking (rerank.c) - a real second-stage pass, with a real measured effect

Hybrid (vector+BM25 RRF) search is a real substitute for a lot of what reranking buys, but not the same as a
cross-encoder that actually reads query and candidate together - a stage most production RAG stacks add on
top of ANN retrieval that this repo didn't have. `rerank.c` (`#include "vecdb.c"`, same layering as mem.c -
vecdb.c itself untouched) adds it: `vecdb_rerank_open` spawns `rerank_srv.py` (any cross-encoder model, same
child-process/pipe pattern as `vecdb_embproc_open`, just a different protocol since a reranker scores
(query, candidate) PAIRS, not single texts), `vecdb_rerank_hits` re-scores and re-sorts an existing
`VHit[]` (from `vecdb_search`/`vecdb_query`) in place, falling back to the original hybrid order untouched if
the rerank server is unavailable or fails - always an optional refinement, never the only ranking path.
Self-test: `./run` (`RERANK_DEMO`), degrades gracefully (prints a skip message, does not fail the run) when
no torch-capable Python is reachable.

**Measured effect**, not just "we added an API": against `real_val.c`'s own SQuAD benchmark data (see below),
on the 871/1000 questions where the gold passage is already somewhere in the raw top-10 cosine neighbours (a
reranker can only reorder what retrieval already found - the other 129 are retrieval misses, unaffected by
any reranking and reported separately), `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking those 10 candidates:

| | MRR | gold@1 |
|---|---|---|
| before (raw cosine order) | 0.751 | 62.1% |
| after (cross-encoder rerank) | 0.887 | 81.5% |
| delta | **+0.136** | **+19.4 points** |

(`emb/eval_rerank.py`, reusing `real_val.c`'s own ground truth - `emb/real_texts.py` reconstructs the
corpus/query text alongside it, since `real_gen.py` itself only ever saved vectors.) A +19.4 point jump in
"is the right answer literally first" is a substantial, real quality gain for the ~330/871 questions where
the gold passage was in the top-10 but not already first - exactly the case a reranker exists for.

### Threads

Searches on a store take a read lock and run concurrently (stress test: 8 readers + a writer);
the TurboQuant scan itself is multi-threaded on big stores, and `store_search_batch` spreads a batch of queries
over threads. The index build is threaded (k-means, assignment, encoding). Lexical (BM25) searches on one index
take turns behind a mutex. Embedding and ingest are single-threaded.

### Query language

`vecdb_query(db, "words #tag -#draft [[Note]] to:[[Note]] from:[[Note]]", k, mcts, out)` (CLI `search` and
`explore` use it): `#tag` / `-#tag` / `tag:x` (nested: `#a` also matches `#a/b`), `[[X]]` notes linked to or from X,
`to:[[X]]` X's backlinks, `from:[[X]]` X's links. Conditions are ANDed with the words; without words the
matches are listed most-linked first; an unknown note or tag matches nothing.

### Embedding: CPU only, batched, threaded

`embedsrv.py` never touches the GPU (`CUDA_VISIBLE_DEVICES=""`). `vecdb_add_notes` sends 32 texts per round trip
(`B`/`C` batch kinds) or, with the built-in embedder, embeds on all cores. Measured on 1024 Wikipedia leads
(e5-small, 12 threads): batching alone changes little (the model is compute-bound: ~200 GFLOP/s already),
what pays is fewer tokens and int8: `EMBED_INT8=1` ~ -20% time, `EMBED_MAXLEN=128` ~ -50%
(25 ms/article instead of 57), with no loss in hybrid retrieval (vector-only -1..2 points). Recommended for
ingest: `EMBED_INT8=1 EMBED_MAXLEN=128`. Repeated query strings are served from a 32-entry cache
(a model call is ~12 ms, the search itself <1 ms).

### Index choice

`store_build` picks automatically: with no memory budget (default) the TurboQuant-style scan (`tq.c`);
with `ENG_MIN_SIZE` or a `max_code_bytes` below the f32 size, the q1-q5 / IVF / HNSW tuner (`engine_build`),
which is what small-footprint goals need.

### GPU / CPU mix

`VECDB_EMBED_SERVER="cmd1|cmd2"` starts several servers; batches (1024 texts per flush in the CLI) are cut into
groups of 32 and each server pulls the next group when free; single queries go to the first one.
`EMBED_DEVICE=cuda` puts a server on the GPU (fp32: Pascal's fp16 is throttled and slower), `EMBED_THREADS`
limits its torch threads. Measured on 4094 Wikipedia leads (e5-small, 128 tokens): GPU alone 5.5 ms/article,
CPU alone 36, GPU + CPU int8 22.1 s vs 22.6 s GPU alone (no gain: the CPU server steals cores the GPU
server needs for tokenising), GPU + CPU fp32 31.7 s (worse). For bulk ingest use the GPU alone; a CPU server
is for queries (12 ms) once the card is needed elsewhere.

### Embedding cache

`vecdb_embed_cache_open(db, path, tag)` keeps every document vector the model made in a file, keyed by
hash(tag + text); `tag` names the model and its settings (another tag = separate entries, nothing is mixed).
Re-indexing an unchanged vault asks the model 0 times, a changed vault only for the changed notes (batches
send just the misses). The CLI turns it on whenever `VECDB_EMBED_SERVER` is set: `DB.emb` next to the database,
or `VECDB_EMBED_CACHE=path`. Only documents are cached (queries have their own 32-entry in-memory cache).

`EMBED_MAXLEN` now defaults to 128 tokens (25 ms/article on CPU int8, 5.5 ms on GPU fp32, no loss measured
on Wikipedia leads; raise it for long passages).

### Idea, not done: fixed-point (int8) embedding on the GPU

The GTX 1050 Ti (Pascal, sm_61) has the `dp4a` int8 dot instruction, so a hand-written int8 e5-small could in
theory run 2-3x faster than fp32. Not worth it now: PyTorch has no int8 GEMM path for this card (`_int_mm`
needs newer hardware), so it means custom CUDA kernels (or llama.cpp / onnxruntime int8), the card already
sustains ~1 TFLOPS of its ~2, and the whole 201k-article Wikipedia costs 18 minutes once (the embedding cache
and `vecdb_sync` mean only new or changed notes are ever embedded again). The CPU already uses dynamic int8
(`EMBED_INT8=1`, -20% time, no AVX-VNNI on this machine to gain more). Revisit if a corpus of tens of millions
of chunks shows up.

### Production notes

- **Failure handling.** The model server answers every request with a status byte; a batch that fails is retried
  text by text, a text that still fails yields no vector, and such a note is *not added* (message on stderr) and
  never cached. A query whose embedding failed falls back to words only. The server halves its batch on GPU
  out-of-memory (a card shared with the desktop has 0.3-3.5 GB free at any time) and only gives up on a single
  text. Before this a failed batch silently stored zero vectors.
- **`vecdb_free`** releases a database; `vfuzz.c` (random markup, queries, replace/remove, save/load under
  ASan/UBSan, run by `./run`) found and fixed one buffer overflow in the query parser.
- **CLI**: `index DIR`, `update DIR` (changed files re-embedded through the cache, vanished ones removed, saved
  through the journal), `remove TITLE`, `stats`, `search`, `explore`, `images`, `similar`, `links`, `backlinks`,
  `tag`, `tags`, `ghosts`, `repl`. Chunks are sized to what the model reads (`EMBED_MAXLEN` x 3.5 characters,
  `VECDB_CHUNK` overrides). `./run` also drives the CLI end to end.
- **Choosing `EMBED_MAXLEN`** (3000 Wikipedia articles, e5-small, GPU): for queries taken from deep inside an
  article (beyond the first tokens) 128 tokens gives vector-only 47%, 512 gives 78% (hybrid 92% -> 97%), at
  4.4x the ingest time on a shared card (46 vs 10.6 ms/article). Leads and titles are the same at both. So: keep
  128 and let the CLI chunk long documents (each chunk then fits what is embedded), use 512 only for whole
  long notes that must be one vector.
- **Fusion weights** (words 1.5, vector 0.6 before the confidence bonus) were tuned on Wikipedia with both the
  hashed and the e5 embedder: deep-body hybrid 91.8% -> 95%, linked-neighbour recall 38% -> 46%, leads/titles
  unchanged. They are constants in `vdb_search_`.

### Durability, formats, threads

- **Atomic writes.** Every file that is replaced is written beside the old one, `fsync`ed, then `rename`d (and the
  directory `fsync`ed). Commit order for an incremental save: bodies (append-only, fsynced) -> vector files -> the
  node table *last*, which is the commit record. `vecdb_sync` does the same with the journals. If a crash lands
  between the vector files and the node table, the counts disagree on load and the vectors are recomputed (the
  embedding cache makes that cheap). `vcrash.c` SIGKILLs a writer at random moments, 25 rounds per `./run`
  (hundreds of rounds tested): the database always reloads with everything that was synced. That proves process
  crashes; power loss depends on the disk honouring `fsync` and is not something I can test here.
- **Cross-file atomicity (the part per-file atomicity does not give you).** PATH.vdt and PATH.vxi are tightly
  coupled - a node table's byte offsets only mean anything against the *exact* bodies file they were written for.
  Two independent atomic renames are not atomic *as a pair*: a crash between them can leave an old node table
  paired with new bodies (or vice versa) - silently wrong reads, not a crash. This only happens on a **full
  rewrite of an already-existing path** (`vecdb_new` + `save`/`sync`, not `vecdb_load` first - what the CLI's
  `index` does when re-indexing on top of a database that already exists there); the normal incremental
  append-then-sync path never replaces anything, so it was never exposed to this. Fix: a full rewrite that would
  have to replace an existing PATH.vdt/.vxi never touches them - it writes the whole new file set under a fresh
  random generation suffix (PATH.vdt.g\<hex\>, PATH.vxi.g\<hex\>, `VecDB.bodygen`), and only the *last* step (the
  node table, itself atomic) starts pointing at that generation. A crash before it leaves the old generation's
  files untouched (loads as the old database, unaffected); a crash after leaves the new generation's files
  complete and fsynced (loads as the new one). Never a mismatched pairing, at the cost of a harmless orphaned
  file on an interrupted attempt (cleaned up opportunistically on the next successful save to that path). A brand
  new path (nothing there yet) skips the generation dance and uses the bare names - the common case stays cheap.
  `vcrash2.c` (`./run`) forks a child that re-indexes on top of an existing database and kills it at a random
  point spanning the real completion boundary: over 30 rounds, roughly a even mix of "landed on the old copy"
  and "landed on the complete new copy", zero mixed/corrupted, zero load failures.
- **Format versions.** `.vdb` starts with magic `VDBX` and a version (now 5, which added `bodygen`). Version 3
  and 4 files still load; a file from a newer build, garbage, a missing or truncated file are refused and
  `vecdb_error()` says why. When the format changes, bump `VDB_VERSION` and add a branch to the loader.
- **Threads.** One readers-writer lock per database (`vecdb_*` are safe to call from many threads): searches,
  queries, MCTS and lookups share it; add / remove / smooth / build / save / sync are exclusive. Model calls happen
  outside the lock. The body file and the embedding cache have their own mutexes. `vthreads.c` (4 readers + a
  writer doing adds, replaces, removals, rebuilds and syncs) runs under ThreadSanitizer and ASan in `./run`.

### Real notes check (77 markdown files: memory notes, project docs; Polish and English mixed)

Indexed with e5-small on the GPU (128 tokens): 2075 notes/chunks in 47 s. 20 hand-written questions in Polish and
English (vague on purpose, each with one expected file), hit@1 / hit@3: hybrid 12 / 14, vector only 10 / 11, words
only 11 / 14. So the mix beats each half, but modestly, and it is a 20-question sample with my own labels.
What the misses show: an index note (`MEMORY.md`, one line about every project) matches almost any query
lexically; documents cut into many chunks get many chances to rank; a Polish question about an English note
depends on e5-small's cross-lingual ability (`git` vs `gita`: the light stemmer leaves 4-letter words alone).
The CLI prints one line per document (best chunk, "+N more parts").

### Stemming: Polish and English together

`lex.c` folds Polish letters to ASCII (`ą ć ę ł ń ó ś ź ż` -> `a c e l n o s z`, so a query typed without diacritics
still matches) and strips one longest matching suffix from every token, then a final vowel. There is no language
detection: one rule set covers both (Polish case endings, `-owanie / -uje / -ować`, `-ami -ach -ów`; English
`-s -es -ed -ing`). Tokens with digits are left alone. Tests keep families together (kot/kota/kotem/kotami/kotów,
komputer/komputerze, programować/programuje/programowanie, zapytanie/zapytań, notes/note) and look-alikes apart
(kot/kos). English retrieval on Wikipedia did not move (hybrid 98.7% on lead words, 96.4% on deep-body words).
Known limits: stem-final alternations (dokumentacja/dokumentacji, kot/kocie, pies/psa) are not handled, and the
English side is a light suffix stripper, not Porter, so `data`/`date` meet.

### Concurrency model

The vector scan is memory-bound: on 201k x 256 (26 MB of 4-bit codes) one thread needs 2.3 ms, four threads 1.3 ms,
eight or twelve no better, a persistent thread pool no better (tried and removed). So the design stays: one
readers-writer lock for the API, a few data-parallel threads for the scan (4, enough to saturate the bus),
self-scheduling groups for the model servers (a shared counter, which is why a fast GPU takes more of the work),
no coroutines (nothing in-process waits on I/O that a thread would not wait on) and no fixed master/slave
partitions (uneven work makes the static split wait for its slowest half, which is what the CPU+GPU mix showed).
Less memory traffic (2-bit codes with a longer shortlist, IVF pruning) is what would make the scan faster.

### IVF for the TurboQuant scan (large stores)

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

### Self-tuning: ask for a recall, the store keeps it

`store_set_target_recall(s, 0.95)`: from then on every `store_build` measures recall@10 against exact search and
picks the fewest IVF probes that reach the target, aiming 1.5 standard errors above it so fresh queries meet it too.
It measures on the queries the store has actually been answering (a reservoir sample of the last 256, kept while a
target is set) or, before 64 have come in, on a spread of its own rows (each one's own row left out of the truth).
`store_tune(s)` measures again on demand - after the traffic or the data have drifted - and the probe count it
settles on is saved with the index. GloVe-100, `bench/tune.c`, on the Cascade Lake host of the Results section:

| target | tuned at build, on its own rows | held-out recall@10 (2000 queries) | re-tuned on those 2000 | recall@10 on the other 8000 |
|---|---|---|---|---|
| 0.90 | 118 probes | 0.9134 | 108 probes | 0.9053 |
| 0.95 | 264 probes | 0.9528 | 290 probes | 0.9584 |

A tuning costs about 2 s on GloVe, most of it the exact search for the truth; on bigger stores it uses fewer
queries (at most ~3e10 multiply-adds). The self-test checks it on held-out queries too.

### Scale: rows in chunks, past RAM

What capped the store before (measured, see Results): every f32 row sat in anonymous memory, and the arrays grew by
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

For stores past what RAM and disk allow at f32, two knobs: `store_set_codes(s, 2)` (2-bit codes: half the RAM,
twice the scan speed, re-score more candidates) and `store_set_vec_bytes(s, 1)` (int8 rows with an f32 scale: d + 4
bytes on disk instead of 4d, about one recall point on GloVe - the cost of 8 bits, a rotation first does not change
it). Measured trade-offs are in the Results tables.

### vecdb at scale (hashed embedder - no model server)

`vbench1m.c` (`./.build/vbench1m N [SAVEDIR]`) adds N notes with links, tags and a body through the public API,
builds, queries, saves, and reloads. Per note vecdb used ~670 bytes of C structures on top of its vector; now 392
(500k notes, measured per structure): BM25 postings are varint-packed and kept inline while they fit in 8 bytes
(271 -> 54 B), string hashes have 12-byte slots at load 0.7 (terms 105 -> 84 B, titles 61 -> 42 B), node names
and link lists come from a slab with free lists (74 -> 57 B), and a node is 56 bytes instead of 64.

In the 5.84 GiB cgroup on 2 cores the old code died at about 4M notes; now 8M notes run end to end: ingest 146 s
(18 us/note, linear), index build 19 s, 3.6 GB anonymous memory after the build (the vectors stay file-backed),
hybrid search 7.7 ms/query with 300/300 exact-id hits, MCTS explore 53 ms/query.

On the old 12-core box (history, before the changes above): 1M notes in 1.48 GB RSS, and a swap-thrashing cliff
between 3.4M and 3.5M notes with ~3 GB of the machine taken by another process.

Real notes with real (larger) embeddings and a model server will cost more per note than this hashed-embedder
test - see the Wikipedia numbers above for that side of it.

Building this benchmark surfaced a footgun, now called out where `VecDB` is declared: a note's id is **not**
its insertion index. Tags and ghost nodes share the same id space, so `vecdb_add_note`'s return value can jump
ahead of a simple counter; always use the id it returns (or `vecdb_find`), never `id == the Nth call`.

### A real retrieval benchmark, not a hand-labeled one (real_val.c, SQuAD + AG-News)

The "Real notes check" above (20 hand-written questions, own labels) is informal - fine as a sanity check, not
a number to set next to a leaderboard. `real_val.c` (`./run real_val`) is the closer-to-standard version:
**48,891 real passages** (18,891 unique SQuAD/Wikipedia contexts + 30,000 AG-News snippets as distractors,
`emb/real_gen.py`) embedded with `sentence-transformers/all-MiniLM-L6-v2` (384-dim), scored against **1,000
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
| `store` (auto-tuned wrapper, what the CLI actually uses) | 97.1% | 83.5% | 6.11 ms/query |

The ceiling to compare against: exact cosine search with this SAME embedder only finds the gold passage in
its own top-10 **87.1%** of the time (measured directly from `real_gen.py`'s own output) - the embedder, not
the index, is what caps retrieval quality here. Every config above lands within 1.5-4 points of that ceiling,
and the best (HNSW) is within 1.5 points of it while beating it slightly on some runs (index construction and
the exact-cosine baseline use different tie-breaking, so being *at* the ceiling is the right way to read a
±1-2 point gap, not literally below it). Read honestly: this says the ANN index itself is close to lossless
relative to what a mid-tier 118M-param embedder can retrieve - it does **not** say retrieval quality is
state-of-the-art, since a stronger embedder would raise the 87.1% ceiling itself, which this benchmark cannot
show on its own - reranking (above) is a real, measured way to close *some* of that gap without swapping
embedders; see "How this compares to mainstream RAG/vector-DB stacks" for the honest full picture.

## mem.c: lifetime, metamemory, event->fact->concept, a rule engine

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
  `mthreads.c`/`mcrash.c`/`mrfuzz.c` (`./run`), `mstress.c` (`./run stress`) and `mbench.c`
  (`./run bench`) below.
- **Concurrency** (`mthreads.c`, `./run`): 4 readers (search/touch, relevance, confidence, known,
  gaps, hot list, provenance) + a writer (note, corroborate/contradict, merge, prune, forget,
  consolidate, save) on one `Mem`, under ThreadSanitizer. This is what actually found the real bugs
  worth naming: `mem_relevance`/`mem_confidence`/`mem_gaps`/`mem_hot_ids`/`mem_provenance`/
  `mem_kind` never took the lock at all (73 TSan warnings on the first run); `mem_merge`/
  `mem_prune`/`mem_consolidate`/`mem_split` called those (now-locked) functions while *already*
  holding `m->lk` - a real self-deadlock risk, not just a race, fixed by splitting each into an
  unlocked `_()` core (used internally, caller already holds the lock) and a locked public wrapper
  (the same pattern `vecdb.c` already uses for its own `vdbl_*`/`vecdb_*` split); `mem_gaps` and
  `mem_prune` sized their scratch buffers from `m->n` *before* taking the lock - under concurrent
  growth that is a real buffer overflow, not a benign read race; `mem_recall`'s cold-file scan
  wasn't lock-protected at all, unlike `mem_archive`'s. Now clean under TSan and ASan+UBSan+
  LeakSanitizer (0 leaks), rerun several times to rule out timing luck.
- **Crash safety** (`mcrash.c`, `./run`, same discipline as `vcrash.c`/`vcrash2.c`): a child adds
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
  removing the direct `m->cf` read entirely. `mthreads.c` was extended to exercise
  `mem_set_lifetime`/`mem_cold_open`/`mem_archive`/`mem_recall`/`mem_cold_count` concurrently too
  (kept infrequent - they do real fsync'd I/O under the lock, and TSan's per-access overhead makes
  that add up fast) and a new self-test case covers `mem_forget` with no cold storage configured.
  Clean under TSan (reran 3x) and ASan+UBSan+LeakSanitizer after this pass too.
- **A third pass, this time an adversarial correctness stress test** (`mstress.c`, `./run
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
- **Timing** (`mbench.c`, `./run bench`): ingest ~7.5 us/note, `mem_touch`/`mem_relevance`/
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
  cost more than quadruples from N=1,000 to N=64,000. The likely cause is the same one vecdb's own
  README calls out for large stores generally (IVF/HNSW routing overhead growing with corpus size
  even though each individual `vecdb_search` call stays sub-linear); `mem_consolidate` amplifies it
  by issuing one such search per node, every pass.
- **The fix: scoped consolidation.** `mem_consolidate_ids(m, rules, ids, nids)` runs the exact
  same engine over a caller-given id list instead of every live node, and `mem_consolidate_hot(m,
  rules)` is a thin wrapper scoping it to the current hot set (`mem_hot_ids`, bounded at
  `hot_cap`) - cost O(working set) instead of O(corpus). Measured on the same N=64,000 corpus as
  the worst row above: **106.6 s for a full `mem_consolidate` vs 0.29 s for `mem_consolidate_hot`**
  (bounded at 256 nodes) - a ~365x difference, because the cost is now capped by `hot_cap`, not by
  N. `mstress.c`'s own churn loop dropped from ~45 s to ~14 s once most of its consolidation calls
  switched from the full sweep to the scoped variants, which is the same effect in practice, not
  just in a microbenchmark. Practical guidance: run `mem_consolidate_hot` on the hot/write path
  (or `mem_consolidate_ids` over whatever working set the caller already tracks, e.g. "touched
  since I last consolidated") and reserve a full `mem_consolidate` for an occasional off-path
  sweep (a timer, idle time, etc.) so nodes that never stay hot long enough still eventually get
  evaluated - a purely hot-scoped strategy would let those decay unconsolidated forever.
- **Rule parser fuzzing** (`mrfuzz.c`, `./run`): random junk (a mix of real rule-DSL vocabulary
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
  often optimise a tail call like that into a loop, but nothing guaranteed it). `mrfuzz.c` keeps
  the exact deep-nesting shape that found this as a standing round (500k nested parens, balanced
  and unbalanced both directions), not just the reader on its own - so a regression here fails
  immediately instead of waiting to be hand-found again. 5,000-30,000 rounds, clean under
  ASan+UBSan+LeakSanitizer after the fix.

### Honest gaps in mem.c

- `mem_consolidate`'s `(similar T)`/promote/merge searches are O(nodes) vecdb_search calls per
  pass, and that per-call cost itself grows with corpus size - see the `mbench.c` table above.
  `mem_consolidate_ids`/`mem_consolidate_hot` (above) are the fix for the hot/write path; a full
  sweep is still O(nodes) and still needed periodically to catch what falls outside the working
  set, and gets expensive (~100s) at 64k nodes - not yet measured near vecdb's own million-note
  ceiling.
- Extractive only, by design - no summarisation. A promoted fact/concept's text is its single
  best source's text verbatim, not a synthesis of everything that supports it.
- Even the systematic locking pass and the adversarial stress pass above were done by the same
  person who wrote the code, not an independent review - the honest bar is "no known gaps left",
  not "provably none remain".

## How this compares to mainstream RAG/vector-DB stacks

Asked directly, honestly, not as a sales pitch:

**Ahead of what most RAG stacks offer:**
- **mem.c's memory architecture is not something Pinecone/Weaviate/Qdrant/Milvus have at all** -
  they are ANN indexes plus a metadata filter. Decay/half-life, metamemory (a real "I don't
  know"), event->fact->concept promotion via a deterministic rule engine is usually hand-rolled
  today with an LLM call doing the summarising (Mem0, Letta/MemGPT-style agent memory). Doing it
  with zero LLM calls (a small forward-chaining s-expression interpreter over real vector
  similarity) is a genuine, unusual design choice - cheaper, deterministic, and testable in a way
  an LLM-summarised memory store is not.
- **The graph (wikilinks/backlinks/tags/ghost nodes, MCTS bridge-note exploration) is native**,
  not bolted on as metadata filters - most vector DBs have no equivalent.
- **Engineering rigor unusual for a solo/embedded project**: atomic cross-file writes (the
  generation-suffix fix), TSan-clean concurrency at two layers, adversarial stress tests with real
  invariants that found real bugs (not just "does it crash" fuzzing), a rule-parser fuzzer that
  found a real stack-overflow crash on its first run. This is closer to how a production index
  engine gets built than a typical tutorial RAG project.
- **Embedded, zero dependencies, single C89 file per layer** - and an optional server (`vserv.c`,
  2.4 MB idle, TLS through vendored BearSSL) when it has to sit behind a socket. A real advantage
  where self-hosted/offline/small-footprint matters, which a hosted vector DB cannot give you.
- **Speed at the same recall, build and load.** On GloVe-100 (Results) it answers 1.4-1.8x more
  queries per second than FAISS's IVF-PQ fast-scan in a batch on 2 threads, and 1.65-2.5x one query
  at a time, at every recall@10 from 0.85 to 0.97. The index builds in 8.0 s where FAISS IVF-PQ takes 124.5 s, and a saved index is ready in 7 ms
  (mmap) where FAISS needs half a second.

**Behind, honestly, no hedging:**
- **Scale.** Single-node only - no sharding/replication. Measured, not guessed: 100M vectors on one
  2-core box with a 5.84 GiB memory limit (compact mode), 8M notes through the whole vecdb layer;
  past one machine's disk there is nothing. Milvus/Qdrant/Pinecone scale across machines, to
  billions of vectors.
- **Memory and build time against a codes-only index.** turbovec keeps nothing but 4-bit codes (56
  B/vec on GloVe) and builds in 2.2 s with no clustering; vecindex keeps the rows to re-score, so its
  smallest mode is 153 B/vec (2-bit codes + int8 rows) and its build is 8 s. turbovec tops out at
  recall@10 0.85 on GloVe and scans every code of every row per query. No graph index here either.
- **Embedding quality.** The default is `multilingual-e5-small` (118M params) - mid-tier. Larger
  open models (e5-mistral-7b, NV-Embed) and commercial APIs (OpenAI text-embedding-3-large,
  Voyage, Cohere) score meaningfully higher on standard retrieval benchmarks. `real_val.c` (above)
  gives a real number for THIS repo on a real SQuAD-based task (gold@10 up to 85.6%, near the
  87.1% ceiling this specific embedder allows) - not a claim of beating stronger embedders, a
  measurement of how close to lossless the ANN index itself is relative to whatever embedder
  it's given.
- **No reranker, until this round** - now real (`rerank.c`/`rerank_srv.py`, measured +19.4 points
  gold@1 on the SQuAD subset where it has room to help - see above). Still only one reranker
  model wired up and tested (`cross-encoder/ms-marco-MiniLM-L-6-v2`, English-only); a multilingual
  swap (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` or similar) is not itself tested here yet.
- **No multi-tenancy/auth/replication/HA** - this is a library, not a hosted service. Out of
  scope for what it is, not a bug, but a real category difference from a managed vector-DB
  product.
- **No independent review, no production traffic.** Every test in this repo was written and run
  by the same person who wrote the code being tested. Real bugs were found this way (the whole
  point of writing real tests instead of just reading the code), but "no known gaps" is not the
  same claim as "reviewed/battle-tested by others."

**Net:** as a *vector search engine*, this is a solid, well-tested single-node library that, on one
machine, builds and loads far faster than FAISS and Qdrant and searches faster at high recall - it does
not compete with Pinecone/Qdrant on multi-machine scale or with top embedding APIs on raw retrieval
quality, and says so with real numbers rather than assuming either way. As a *memory layer for an agent* (event
lifetime, decay, metamemory, deterministic consolidation), it does something none of the
mainstream RAG/vector-DB products offer natively - which is the more interesting half of this
project, not "does it beat Qdrant."
