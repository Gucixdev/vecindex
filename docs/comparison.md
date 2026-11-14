# How this compares to mainstream RAG/vector-DB stacks

The field first - what each project ships, read from its own sources - then where vecindex is ahead and where it is
behind.

## The field

Read on 2026-10-29 and 2026-10-30 from each project's repository, documentation or release notes (links under
Sources). Licences come from the LICENSE file in each repository. A cell marked † is from earlier knowledge and was not
re-checked; `?` means not checked; `-` means the project does not offer it.

"Smallest code" is the fewest bits per dimension a vector can be stored in. "BM25 hybrid" means keyword (BM25) and
vector scores ranked together in one query inside the engine.

### In-process: libraries and embedded engines

| project | written in, used from | index | smallest code | BM25 hybrid | keeps texts | durability | licence |
|---|---|---|---|---|---|---|---|
| **vecindex** | C89, no dependencies; C ABI, line, binary and HTTP server | IVF over rotated 4/2/1-bit codes, exact re-rank from f32 or int8 rows | 1 bit, plus a row per vector for the re-rank | yes: BM25 + weighted RRF | yes: chunks per document, range reads | journal + background snapshots | MIT |
| FAISS | C++; Python | flat, IVF, PQ, HNSW†; RaBitQ | 1 bit (binary indexes†, RaBitQ) | - | - | index files† | MIT |
| hnswlib | C++ header; Python | HNSW | f32† | - | - | index file† | Apache-2.0 |
| USearch | C++11 header; 10+ languages | HNSW, user-defined metrics | 1 bit (`b1`); also f16, bf16, 8-bit floats, i8 | - | - | index file, served memory-mapped | Apache-2.0 |
| DiskANN | Rust (C++ on an unmaintained branch) | Vamana graph; index on SSD, PQ codes in RAM† | PQ | - | - | index files | MIT |
| KIOXIA AiSAQ | C++ (a DiskANN fork) | Vamana graph with its PQ codes on the SSD too, not in RAM; 4.8 billion vectors shown on one server | PQ | - | - | index files on SSD | MIT |
| RaBitQ-Library | C++17; Python | IVF, HNSW, SymphonyQG over RaBitQ codes | 1 bit | - | - | index files | Apache-2.0 |
| turbovec | Rust; Python | scan of every TurboQuant code, no clustering or graph; id allowlist filter | 2 bit | - | - | one file, written with fsync + rename | MIT |
| sqlite-vec | C, SQLite extension; runs where SQLite runs, WASM too | brute force (`vec0`); IVF and DiskANN in the 0.1.10 alphas | 1 bit (binary) | FTS5, fused in SQL | SQLite tables | SQLite | MIT or Apache-2.0 |
| SQLite vec1 | C (one file), SQLite extension | IVF with OPQ codes (IVFADC), needs a training pass | 1 bit (`bq`) | FTS5, fused in SQL | SQLite tables | SQLite | ? |
| DuckDB VSS | C++, DuckDB extension | HNSW (USearch) | f32† | DuckDB `fts`, fused in SQL† | DuckDB tables | experimental; no incremental index writes | MIT |
| LanceDB | Rust; Python, TypeScript, REST | IVF_PQ, IVF_SQ, IVF_RQ (RaBitQ), IVF_HNSW | 1 bit (RaBitQ) | yes: BM25 + RRF | yes: columnar Lance tables | versioned Lance files | Apache-2.0 |
| Chroma | Rust core; Python, JS; also a server | HNSW†; SPANN with RaBitQ codes | ? | yes: sparse BM25 or SPLADE + RRF | yes | local files or server† | Apache-2.0 |
| zvec (Alibaba) | C++ (Proxima); Python, Node.js, Go | HNSW, IVF, DiskANN, RaBitQ | 1 bit (RaBitQ) | yes: vectors, full text and filters in one query | yes: text fields | WAL | Apache-2.0 |
| Milvus Lite | Python over FAISS | FAISS indexes, LSM storage | ? | yes: BM25 + hybrid | yes | WAL | Apache-2.0 |
| Qdrant Edge | Rust; Python | Qdrant's engine in-process† | ? | ? | payloads† | ? | Apache-2.0 |
| LEANN | Python over a FAISS fork | graph that recomputes embeddings at query time instead of storing them | none stored | ? | yes: it re-embeds them | index files | MIT |
| VecGrid | Python | hnswlib partitions spread over embedded nodes | f32† | - | ? | WAL + snapshots, a synchronous backup copy on another node | GPL-3.0 |

### Servers and hosted services

| project | written in | runs as | index | smallest code | BM25 hybrid | across machines | licence |
|---|---|---|---|---|---|---|---|
| Qdrant | Rust | server, cloud; Edge in-process | HNSW; scalar, binary, PQ and TurboQuant codes (1/1.5/2/4-bit since 1.18, 4-bit as the only stored copy since 1.19, 8- and 16-bit types in 1.20) | 1 bit | yes: BM25 on full-text indexes (1.20), RRF fusion | sharding, replication | Apache-2.0 |
| Milvus | Go, C++† | server, cloud; Lite in-process | HNSW, IVF, DiskANN, RaBitQ (2.6) | 1 bit (RaBitQ) | yes: BM25 full text | distributed, billions of vectors | Apache-2.0 |
| Weaviate | Go | server, cloud | HNSW, flat† | 1 bit (BQ, rotational RQ) | yes: BM25F | sharding, replication | BSD-3-Clause; some features under the Weaviate License, behind a key |
| pgvector | C, PostgreSQL extension | inside PostgreSQL | HNSW, IVFFlat, iterative scans | 1 bit (`bit` type) | full text (`tsvector`) + RRF in SQL | PostgreSQL replication | PostgreSQL |
| Elasticsearch | Java | server, cloud | HNSW, DiskBBQ | 1 bit (BBQ, the default above 384 dimensions) | yes: BM25 + vectors, RRF | sharding, replicas | AGPL-3.0, SSPL or Elastic License 2.0 |
| OpenSearch | Java | server, cloud† | HNSW, IVF (Lucene and FAISS engines)† | ? | yes: hybrid query† | sharding, replicas† | Apache-2.0 |
| Redis 8 | C | server, cloud† | FLAT, HNSW, SVS-VAMANA; vector sets | ? | yes: Query Engine full text | cluster, replicas† | RSALv2, SSPLv1 or AGPL-3.0 |
| Vespa | Java, C++† | server, cloud | HNSW or exact; binary vectors with hamming distance | 1 bit (packed in int8) | yes: BM25 rank feature and vectors in one multi-phase ranking | content clusters† | Apache-2.0 |
| Pinecone | closed source | cloud only | proprietary | ? | yes: dense + sparse; BM25 full text in public preview since May 2026 | serverless; dedicated read nodes (GA April 2026) | proprietary |
| MongoDB Vector Search | C++, Java† | Atlas cloud; Community and Enterprise self-managed | HNSW† | 1 bit: binary quantization with a re-score from the full vectors, or int8 | yes: with Atlas Search full text | sharding† | ? |
| VAST DataBase | closed source | VAST's own storage platform | hierarchical clustering ("Hyperscale Vector Index"), only the needed parts of the quantized vectors loaded per query | ? | ? - SQL filters and vectors in one query path | shared-everything nodes, no sharding; trillions claimed | proprietary |
| UBASE (ByteDance) | closed, internal | ByteDance's search infrastructure (7,000+ clusters, 300 PB) | vectors, lexical matching and filters in one engine | ? | yes | trillion-vector scale (paper, 2026) | not available |

Not covered: ScaNN, Annoy, NMSLIB, Azure AI Search, turbopuffer, Typesense, Marqo.

### What the field already has

- **Rotated low-bit codes with an exact re-rank** - the technique vecindex's codes use - are common now. Qdrant ships
  TurboQuant since 1.18 and turbovec is built on it; RaBitQ (rotate, then keep one or a few bits per dimension with a
  correction factor) is in Milvus 2.6, LanceDB, FAISS, zvec and its own library; Weaviate has rotational
  quantization, Elasticsearch BBQ. The codes are expected, not a difference.
- **BM25 hybrid inside the engine** is standard on the servers (Qdrant, Milvus, Weaviate, Elasticsearch, OpenSearch,
  Redis, Vespa, Pinecone) and present in-process in LanceDB, Chroma, zvec and Milvus Lite.
- **Past one machine's RAM** the field goes two ways: the index on SSD (DiskANN; KIOXIA AiSAQ keeps even the PQ
  codes there and showed 4.8 billion vectors on one server) or a hierarchy of clusters loading only what a query
  needs (VAST and ByteDance's UBASE claim trillions). vecindex keeps rows on disk and its codes in RAM: one level of
  clusters, which is the next thing to grow ([roadmap.md](roadmap.md)).
- **Graph indexes** (HNSW, Vamana) are the default almost everywhere; IVF-only (vecindex, vec1) and a full scan of
  codes (turbovec) are the exceptions.

## Where vecindex is ahead

- **Portability and footprint.** C89, no dependencies, one unity build; SIMD where the CPU has it (AVX2, NEON) and a
  scalar path everywhere else. The tests pass under qemu on aarch64, armv7, riscv64 and big-endian s390x
  (`tools/cross`); speed on real ARM hardware is not measured. Of the in-process engines above only sqlite-vec and
  vec1 are plain C; RaBitQ-Library's x86-64 builds require AVX2 and FMA.
- **Texts and hybrid search in one embedded C library.** Vectors, a BM25 word index and the texts (chunks with
  document ids, range reads for small-to-big context) share one file and one journal. In-process, LanceDB (Rust),
  Chroma (Rust), zvec (C++) and Milvus Lite (Python) do the same; on SQLite it takes a vector extension, FTS5 and SQL
  to fuse the two.
- **Speed at the same recall, build and load.** On GloVe-100 (README, Results: 2 vCPUs of a Cascade Lake Xeon) it
  answers 1.4-1.8x more queries per second than FAISS's IVF-PQ fast-scan in a batch on 2 threads, and 1.65-2.5x one
  query at a time, at every recall@10 from 0.85 to 0.97. The index builds in 8.0 s where FAISS IVF-PQ takes 124.5 s,
  and a saved index is ready in 7 ms (mmap) where FAISS needs half a second. On a second host (2 vCPUs of a Xeon @
  2.10GHz, 2026-10-29, same harness) the same pair measured 1.6-1.9x in a batch and 1.6-2.2x one query at a time,
  4.1 s against 209 s to build and 5 ms against 445 ms to load. VecGrid 0.1.3 on that host, at its one tested
  setting: recall@10 0.971 at 38 queries per second, 349 s to build, 2,516 bytes per vector.
- **One knob for CPU use.** An aggression setting from 0 to 1 sizes the worker and background threads from the CPUs
  the process may actually use (affinity and cgroup quota) and runs background work at a lower priority. Durability
  is a journal plus background snapshots rather than replication; the server speaks a line protocol next to binary and
  HTTP.
- **Engineering rigor unusual for a solo/embedded project**: kill -9 tests against the journal and the background
  snapshots, TSan-clean concurrency including the live indexing thread, adversarial stress tests with real invariants
  (not just "does it crash"), every SIMD kernel checked against a scalar reference and cross-built for four other CPU
  families under qemu. This is closer to how a production index engine gets built than a typical tutorial RAG project.
- **Embedded, zero dependencies, single C89 file per layer** - and an optional server (`src/vserv.c`,
  2.4 MB idle, TLS through vendored BearSSL) when it has to sit behind a socket. A real advantage
  where self-hosted/offline/small-footprint matters, which a hosted vector DB cannot give you.

## Where it is behind

- **Scale.** Single-node only - no sharding/replication. Measured, not guessed: 100M vectors on one
  2-core box with a 5.84 GiB memory limit (compact mode);
  past one machine's disk there is nothing. Milvus/Qdrant/Pinecone scale across machines, to
  billions of vectors.
- **No graph index.** IVF only, where most of the field uses HNSW or Vamana. FAISS HNSW and Qdrant never finished on
  the host of the README table (on the first host they were slower than FAISS IVF-PQ at every recall), so the
  comparison against a graph at high recall is open.
- **No codes-only storage.** vecindex keeps a row per vector for the exact re-rank, so its smallest mode on GloVe-100
  is 140 bytes per vector (1-bit codes + int8 rows), and 2-bit codes at 153 bytes are the better trade there: 10,687
  against 2,527 queries per second at recall@10 >= 0.90 (batch, 2 threads, 2026-10-30). The 32x of 1-bit codes holds
  for the codes alone. turbovec keeps nothing but codes (56 bytes per vector at 4-bit; with 1.1.2 its recall@10 on
  GloVe tops out at 0.85), Qdrant can keep only 4-bit TurboQuant codes since 1.19, and LEANN stores no vectors at
  all.
- **Not measured against the newest rivals.** Qdrant's TurboQuant, Milvus's and LanceDB's RaBitQ, zvec, USearch,
  sqlite-vec, vec1 and LEANN have not run through the same harness; "faster" is shown only against FAISS IVF-PQ,
  turbovec and VecGrid.
- **Bindings and filters.** A C ABI and three wire protocols, no packages for other languages (LanceDB, zvec and
  USearch ship several). Filtering is one 64-bit tag per row (`vi_search_tagged`), not arbitrary metadata or SQL.
- **No models.** vecindex stores the vectors it is given and re-scores with them; embedding and cross-encoder
  re-ranking are the caller's, where Weaviate, Chroma, Milvus and others can call a model for you. `bench/real_val.c`
  ([store.md](store.md)) measures the index on a SQuAD-based task with `all-MiniLM-L6-v2`: gold@10 up to 85.6%, near
  the 87.1% ceiling that embedder allows - how close to lossless the index is, not a claim about embedders.
- **No users and roles, replication or HA.** Access is the operating system's and TLS's business and accountability is
  the log; a standby is a copy of the file and its journal. Tenants are a store file each, in the library
  (`vi_dir_*`) or the server, and a batch of writes goes in all or none - both tested in `./run`. A
  real category difference from a managed vector-DB product, on purpose.
- **No independent review, no production traffic.** Every test in this repo was written and run
  by the same person who wrote the code being tested. Real bugs were found this way (the whole
  point of writing real tests instead of just reading the code), but "no known gaps" is not the
  same claim as "reviewed/battle-tested by others."

**Net:** as a *vector search engine*, this is a well-tested single-node library that, on one machine, builds and
loads far faster than FAISS IVF-PQ and searches faster at equal recall on two hosts, runs on CPUs most of the field
does not target, and keeps texts and BM25 next to the vectors in plain C. It does not compete with Milvus, Qdrant or
Pinecone across machines, has no graph index, and has not yet been measured against Qdrant's TurboQuant or the RaBitQ
engines.

## Sources

Repositories, read on 2026-10-30; the date is the last commit on the default branch that day:
[FAISS](https://github.com/facebookresearch/faiss) 2026-10-28,
[hnswlib](https://github.com/nmslib/hnswlib) 2026-09-14,
[USearch](https://github.com/unum-cloud/usearch) 2026-10-25,
[DiskANN](https://github.com/microsoft/DiskANN) 2026-10-29,
[RaBitQ-Library](https://github.com/VectorDB-NTU/RaBitQ-Library) 2026-10-29,
[turbovec](https://github.com/RyanCodrai/turbovec) 2026-10-06,
[sqlite-vec](https://github.com/asg017/sqlite-vec) 2026-05-17,
[DuckDB VSS](https://github.com/duckdb/duckdb-vss) 2026-10-14,
[LanceDB](https://github.com/lancedb/lancedb) 2026-10-30,
[Chroma](https://github.com/chroma-core/chroma) 2026-10-21,
[zvec](https://github.com/alibaba/zvec) 2026-10-30,
[Milvus](https://github.com/milvus-io/milvus) 2026-10-30,
[Milvus Lite](https://github.com/milvus-io/milvus-lite) 2026-10-20,
[Qdrant](https://github.com/qdrant/qdrant) 2026-10-21,
[Weaviate](https://github.com/weaviate/weaviate) 2026-10-30,
[pgvector](https://github.com/pgvector/pgvector) 2026-10-28,
[Elasticsearch](https://github.com/elastic/elasticsearch) 2026-10-30,
[OpenSearch](https://github.com/opensearch-project/OpenSearch) 2026-10-29,
[Redis](https://github.com/redis/redis) 2026-10-30,
[Vespa](https://github.com/vespa-engine/vespa) 2026-10-30,
[LEANN](https://github.com/yichuan-w/LEANN) 2026-10-28,
[VecGrid](https://github.com/Arveti/VecGrid) 2026-03-28.

Pages:
- Qdrant: [1.20 release post](https://qdrant.tech/blog/qdrant-1.20.x/), [1.18 release post](https://qdrant.tech/blog/qdrant-1.18.x/index.md),
  [releases](https://github.com/qdrant/qdrant/releases); Qdrant Edge: the `qdrant-edge-py` 0.9.0 package metadata.
- RaBitQ-Library and where RaBitQ is used: [rabitqlib on PyPI](https://pypi.org/project/rabitqlib/0.7.1/).
- SQLite vec1: [overview](https://sqlite.org/vec1/doc/trunk/doc/vec1.md), [reference](https://sqlite.org/vec1/doc/trunk/doc/vec1ref.md),
  [forum post, no release yet as of March 2026](https://sqlite.org/forum/forumpost/9fea7e0fbb6ddd2f),
  [moved into the SQLite source tree on 2026-09-30](https://book.st-hakky.com/news/sqlite-vec1-vector-search-extension).
- Vespa: [hybrid search tutorial](https://docs.vespa.ai/en/tutorials/hybrid-search.html),
  [binary vectors](https://pyvespa.readthedocs.io/en/stable/examples/mixedbread-binary-embeddings-with-sentence-transformers-cloud.html).
- KIOXIA AiSAQ: [repository](https://github.com/kioxia-jp/aisaq-diskann) (MIT, last commit 2026-10-05),
  [4.8 billion vectors on one server](https://www.storagenewsletter.com/2026/03/19/nvidia-gtc-2026-kioxia-achieves-4-8-billion-high-dimensional-vector-search-database-on-a-single-server-with-7-8x-index-build-time-acceleration-via-gpus/).
- VAST: [AI OS 5.5, the Hyperscale Vector Index](https://www.vastdata.com/blog/vast-ai-os-5-5-is-generally-available),
  [no sharding, trillions](https://www.vastdata.com/features/ai-ready).
- MongoDB: [Vector Search changelog](https://www.mongodb.com/docs/atlas/search-changelog.md),
  [binary quantization with rescoring](https://mongodb.com/blog/post/binary-quantization-rescoring-96-less-memory-faster-search).
- UBASE: [the paper's abstract](https://www.emergentmind.com/papers/2608.30607) (arXiv 2608.30607).
- Pinecone: [2026 release notes](https://docs.pinecone.io/release-notes/2026.md),
  [dedicated read nodes](https://docs.pinecone.io/guides/index-data/dedicated-read-nodes/overview).
- Measurements: `py/bench.py` on GloVe-100, results kept with commit, machine and date; the README table and
  [benchmarks.md](benchmarks.md) for the first host.
