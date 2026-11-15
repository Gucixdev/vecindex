# Layout

| path | what is there |
|---|---|
| `include/vecindex.h` | the C API: one opaque handle, plain C types; `libvecindex.so` / `.a` export nothing else |
| `src/` | the library and the programs: headers, quantizers, indexes, the store, the C API, `vserv.c`, `vfind.c` |
| `tests/` | one self-test per module (`NAME_test.c`), plus the crash and stress tests, the server's tests, the damage battletest (`battle_damage.c`) and the fuzz targets (`fuzz_*.c`, their seeds and every input that once failed in `tests/fuzz/`) |
| `bench/` | benchmark programs in C; `./run compare` runs every measurement in [docs/benchmarks.md](docs/benchmarks.md) |
| `py/` | all the Python: the harness that times the other libraries, dataset preparation, the server's end-to-end tests |
| `docs/` | the long-form documentation (list below) |
| `tools/gen` | regenerates `src/q2.c`, `q4.c`, `q5.c` and their tests from the q1 templates |
| `tools/fuzz`, `tools/cov`, `tools/cross` | coverage-guided fuzzing (clang's libFuzzer); line coverage of `./run`; the core tests on other CPUs under qemu |
| `lean4/` | Lean 4 proofs about the kernels, and the test that holds the C code to them ([lean4/README.md](lean4/README.md)) |
| `third_party/bearssl/` | BearSSL 0.6, MIT, unmodified |
| `emb/` | generated retrieval benchmark data, not committed |

Everything is one translation unit: a program includes the `.c` file of the layer it needs (`store.c`, or
`engine.c` for the quantizers alone) and compiles once with `-Isrc`. `./run` builds each test into `.build/`. A file named `LAYER_part.c` is a
part of `LAYER.c`, which includes it; it is not compiled on its own.

| `src/` files | what they hold |
|---|---|
| `base.h`, `arena_alloc.h`, `contracts.h`, `print.h`, `fine_grained.h`, `vsimd.h` | fixed-size types, the arena allocator, contracts, output without stdio, the job system, SIMD helpers |
| `q1.c` - `q5.c`, `qmix.c`, `q4fs.c` | bit-packed 1-8 bit scalar quantizers, mixed depth per dimension, the 4-bit fast-scan |
| `ivf.c`, `hnsw.c`, `engine.c`, `engine_disk.c` | the k-means coarse router, the HNSW graph router, the adaptive engine and its file format |
| `tq.c`, `tq_scan.c`, `tq_add.c` | the rotated 4-bit / 2-bit fast-scan index: build, scan, rows added after the build |
| `hdict.c`, `lex.c` | hash dictionaries; the term dictionary, postings and BM25 |
| `blake3.c`, `store_sum.c` | BLAKE3; the file's block digests, Reed-Solomon parity, the checks at load and the scrub |
| `store.c`, `store_tune.c`, `store_live.c`, `store_search.c`, `store_text.c`, `store_disk.c` | the vector store: rows, ids, build; self-tuning; live mode; search; texts, id ranges and hybrid search; the file and its journal |
| `vecindex.c` | the C API of `include/vecindex.h` over the store |
| `vfind.c` | search a store's texts as you type, like fzf; Enter opens the whole document |
| `vserv.c`, `vserv_proto.c`, `vserv_text.c`, `vserv_log.c`, `vserv_pem.c` | the server: sockets, workers, TLS; the line, binary and HTTP protocols; texts and JSON; the log; certificates and keys |
