# vecindex

A vector database in C89 for approximate nearest-neighbour search: a TurboQuant-style rotated 4-bit (or 2-bit, or
1-bit in front of the 4-bit) fast-scan with exact re-scoring, IVF clustering, bit-packed 1-8 bit quantizers and BM25
hybrid search; cosine, inner product or Euclidean distance; string keys and labels in a hierarchy. It runs inside a program in any language through its C API (`include/vecindex.h`, one shared or
static library), or as `vserv`, a small epoll server (Unix and TCP sockets, a binary protocol and HTTP/1.1, TLS 1.2
through the vendored BearSSL). Nothing beyond libc, libm and pthreads (and BearSSL for TLS).

It is part of [holyc](https://github.com/Gucixdev/holyc): it lived there as `ragdb/`, was its own repository
(github.com/Gucixdev/vecindex) from 2026-10-17 to 2026-11-01 and came back as `vecindex/`, history included. The note
database and agent memory that grew on top of it are [`examples/agent_launcher`](examples/agent_launcher), which uses vecindex as a
library.

## Layout

`include/vecindex.h` is the C API (one opaque handle; `libvecindex.so` / `.a` export only `vi_*`); `src/` the library and the programs (`vserv.c`, `vfind.c`);
`tests/` one self-test per module plus the crash, stress, damage and fuzz tests; `bench/` and `py/` the benchmarks and the Python harness; `tools/` fuzzing,
coverage and cross-CPU runs; `lean4/` the proofs ([lean4/README.md](lean4/README.md)); `third_party/bearssl/` BearSSL 0.6, unmodified. Everything is one translation
unit: a program includes the `.c` file of the layer it needs and compiles once with `-Isrc`. The full table of paths and of every `src/` file: [docs/layout.md](docs/layout.md).

## Build and test

```sh
git clone https://github.com/Gucixdev/holyc && cd holyc/vecindex
./run            # self-tests, sanitizer, thread and crash-safety tests, the damage battletest, the fuzzers' regression
                 # inputs, the server end to end (needs bash and gcc)
./run stress     # plus the churn / oracle stress tests
./run all DATA   # everything in one go: every test, stress, bench, real_val, the cross-built tests, then every
                 # benchmark of docs/benchmarks.md on DATA (py/prepare_glove.py); without DATA, all but those
tools/cross      # the core tests for aarch64, armv7, riscv64 and s390x under qemu (needs their gcc and qemu-user)
./run lean       # the Lean proofs and the C kernels against them (needs elan; lean4/README.md)
tools/fuzz       # 60 s of coverage-guided fuzzing per target, or `tools/fuzz proto 600` (needs clang and compiler-rt)
tools/cov        # line coverage of ./run, per file of src/, least covered first (needs gcov and python3)
```

## Use it

`./run` builds `.build/libvecindex.so` and `.build/libvecindex.a`, whose only exports are the `vi_*` functions of `include/vecindex.h`.
Open a store, `vi_add` rows, `vi_search` (a complete C program: [docs/api.md](docs/api.md)); the same from Python (ctypes), Go (cgo) and Rust (`extern "C"`).
For an LLM each row can carry its text: `vi_add_text`, `vi_search_hybrid` (vector and BM25 fused by rank), `vi_range` for the chunks around a hit.
`.build/vfind docs.vxi` searches those texts as you type, like fzf. Over a socket, `vserv` serves one store file in one-line commands
(`search 10 0.1 0.2 ...` -> `ok 3 42 0.99 ...`), binary frames or HTTP ([docs/server.md](docs/server.md)).

## Results

GloVe-100 (1,183,514 vectors, 10,000 held-out queries, recall@10 against the exact top 10), 2 vCPUs of an Intel
Xeon (Cascade Lake). Queries per second at the fastest setting that reaches each recall, as one batch on 2 threads /
one query per call on 1 thread:

| method | build | bytes/vec | best recall@10 | >= 0.90 | >= 0.95 | >= 0.97 |
|---|---:|---:|---:|---:|---:|---:|
| vecindex 4-bit + f32 rows | 8.0 s | 475 | 0.9884 | 7,996 / 2,277 | 4,897 / 1,433 | 2,606 / 862 |
| vecindex 2-bit + int8 rows | 8.3 s | 153 | 0.9712 | 7,235 / 2,654 | 2,420 / 949 | 1,896 / 715 |
| FAISS IVF4096,PQ50x4fs,RFlat | 124.5 s | 436 | 0.9785 | 4,456 / 1,081 | 2,822 / 871 | 1,846 / 526 |
| turbovec 4-bit | 2.2 s | 56 | 0.8465 | - | - | - |

The method, every row, the scale runs (100M vectors in 5.84 GiB), what each optimisation bought and the bottlenecks
left are in [docs/benchmarks.md](docs/benchmarks.md).

## Production readiness

Embedded first, one machine: the library runs inside the application like SQLite, and `vserv` puts the same store on a socket. A tenant is a store file, a batch of
writes lands whole or not at all, every block of the file has a BLAKE3 digest and Reed-Solomon parity, and the integrity math is proved in Lean 4. Measured, not
estimated: crash safety, CPUs (x86-64, ARM, RISC-V, s390x), concurrency under TSan, ASan/UBSan, fuzzing, damage on disk, scale to 100M vectors, 84.9% line coverage.
Not present, on purpose: users and roles, replication. Independent review: not done. Every row with its evidence: [docs/production-readiness.md](docs/production-readiness.md).

## Documentation

- [docs/layout.md](docs/layout.md) - every path and every file of `src/`.
- [docs/production-readiness.md](docs/production-readiness.md) - the table: area, status, evidence.
- [docs/api.md](docs/api.md) - the C API, bindings from Python, Go and Rust, durability (journal and snapshots).
- [docs/benchmarks.md](docs/benchmarks.md) - GloVe-100 against FAISS and turbovec (the Qdrant runs never finished); scale; the priority table
  of optimisations; bottlenecks; how to reproduce.
- [docs/server.md](docs/server.md) - `vserv`: sockets, protocols, TLS, live mode.
- [docs/store.md](docs/store.md) - the store: the TurboQuant-style scan, IVF, self-tuning, live ingest, rows past
  RAM, stemming, the SQuAD retrieval benchmark.
- [docs/roadmap.md](docs/roadmap.md) - what is done, what is written but not run, what is open.
- [docs/comparison.md](docs/comparison.md) - how this compares to mainstream RAG and vector-DB stacks.
- [docs/quantizers.txt](docs/quantizers.txt) - the bit-packed quantizer files q1-q5, qmix, q4fs, ivf, hnsw, engine.

## License

MIT, see `LICENSE`. BearSSL is MIT too (`third_party/bearssl/LICENSE.txt`).
