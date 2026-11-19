# Layout

| path | what is there |
|---|---|
| `include/vecindex.h` | the C API: one opaque handle, plain C types; `libvecindex.so` / `.a` export nothing else but it, `graph.h` and `ingest.h` |
| `include/graph.h` | the graph beside a store: typed, weighted links between ids, `vg_expand` and personalized PageRank (`src/graph.c`, built into the same library) |
| `include/ingest.h` | documents into a store: `.md` and `.json`/`.jsonl` parsed in C, the embedding model a callback (`src/ingest.c`, built into the same library) |
| `src/` | the library and the programs: headers, quantizers, indexes, the store, the C API, `vserv.c`, `vfind.c` |
| `tests/` | one self-test per module (`NAME_test.c`), plus the crash and stress tests, the server's tests, the damage battletest (`battle_damage.c`) and the fuzz targets (`fuzz_*.c`, their seeds and every input that once failed in `tests/fuzz/`) |
| `bench/` | benchmark programs in C; `./run compare` runs every measurement in [docs/benchmarks.md](docs/benchmarks.md) |
| `py/` | all the Python: the harness that times the other libraries, dataset preparation, the server's end-to-end tests |
| `docs/` | the long-form documentation (list below) |
| `trash/` | what was retired, kept whole: the first engine (bit-packed quantizers q1-q5, qmix, q4fs, the IVF and HNSW routers, `engine.c`), its tests, generator and benches. The scan index (`tq`) replaced it; nothing in `src/` uses it |
| `tools/fuzz`, `tools/cov`, `tools/cross` | coverage-guided fuzzing (clang's libFuzzer); line coverage of `./run`; the core tests on other CPUs under qemu |
| `lean4/` | Lean 4 proofs about the kernels, and the test that holds the C code to them ([lean4/README.md](lean4/README.md)) |
| `third_party/bearssl/` | BearSSL 0.6, MIT, unmodified |
| `emb/` | generated retrieval benchmark data, not committed |

Everything is one translation unit: a program includes the `.c` file of the layer it needs (`store.c`) and compiles once with `-Isrc`. `./run` builds each test into `.build/`. A file named `LAYER_part.c` is a
part of `LAYER.c`, which includes it; it is not compiled on its own.

| `src/` files | what they hold |
|---|---|
| `base.h`, `arena_alloc.h`, `contracts.h`, `print.h`, `fine_grained.h`, `vsimd.h` | fixed-size types, the arena allocator, contracts, output without stdio, the job system, SIMD helpers |
| `core.c`, `config.h`, `format.h`, `mem.h`, `scratch.h` | the helpers every layer shares (top-k insertion, square root, little-endian file I/O); every compile-time setting; the file format's layout, versions and flags; arenas over the OS's pages (an index's memory, a build's, a call's - none of it from malloc); the pool of working buffers a search takes |
| `tq.c`, `tq_scan.c`, `tq_add.c` | the rotated 4-bit / 2-bit fast-scan index: build, scan, rows added after the build |
| `hdict.c`, `lex.c` | hash dictionaries; the term dictionary, postings and BM25 |
| `blake3.c`, `store_sum.c` | BLAKE3; the file's block digests, Reed-Solomon parity, the checks at load and the scrub |
| `store.c`, `store_tune.c`, `store_live.c`, `store_search.c`, `store_meta.c`, `store_text.c`, `store_rag.c`, `store_disk.c` | the vector store: rows, ids, build; self-tuning; live mode; search; payload fields and their filters; texts, id ranges and hybrid search; RAG (MMR, near-duplicates, neighbour chunks, a context to a byte budget); the file and its journal |
| `vecindex.c` | the C API of `include/vecindex.h` over the store |
| `graph.c` | the graph of `include/graph.h`: its own memory, its own file (a log of CRC'd records), no part of the store |
| `ingest.c` | the ingest of `include/ingest.h`: front matter and ATX headings into sections (pages), an iterative JSON parser (its own 64-level stack), one write batch a file, the hash embedder; only the public API underneath |
| `vfind.c` | search a store's texts as you type, like fzf; Enter opens the whole document |
| `vserv.c`, `vserv_proto.c`, `vserv_text.c`, `vserv_log.c`, `vserv_pem.c` | the server: sockets, workers, TLS; the line, binary and HTTP protocols; texts and JSON; the log; certificates and keys |
