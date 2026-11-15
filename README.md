# vecindex

A vector database in C89 for approximate nearest-neighbour search: a TurboQuant-style rotated 4-bit (or 2-bit, or
1-bit in front of the 4-bit) fast-scan with exact re-scoring, IVF clustering, bit-packed 1-8 bit quantizers and BM25
hybrid search; cosine, inner product or Euclidean distance; string keys, labels in a hierarchy, a number a row for
range filters and grouped search (one hit a document); and search with a proof (`vi_search_exact`: the exact k best,
every other row ruled out by a bound proved in Lean). It runs inside a program in any language through its C API (`include/vecindex.h`, one shared or
static library), or as `vserv`, a small epoll server (Unix and TCP sockets, a binary protocol and HTTP/1.1, TLS 1.2
through the vendored BearSSL). Nothing beyond libc, libm and pthreads (and BearSSL for TLS).

It is part of [holyc](https://github.com/Gucixdev/holyc): it lived there as `ragdb/`, was its own repository
(github.com/Gucixdev/vecindex) from 2026-10-17 to 2026-11-01 and came back as `vecindex/`, history included. The note
database and agent memory that grew on top of it are [`examples/agent_launcher`](examples/agent_launcher), which uses vecindex as a
library.

## Layout

See [docs/layout.md](docs/layout.md): the repository layout and the source files of each layer.

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

From any language: `./run` builds `.build/libvecindex.so` and `.build/libvecindex.a`, whose only exports are the
`vi_*` functions of `include/vecindex.h`. In C:

```c
#include "vecindex.h"
#include <stdio.h>

int main(void)
{
    vi_db *db = vi_open("demo.vxi", 4, 0);
    float a[4] = { 1, 0, 0, 0 }, b[4] = { 0, 1, 0, 0 }, q[4] = { 0.9f, 0.1f, 0, 0 }, scores[2];
    vi_i64 ids[2];
    int n, i;
    vi_add(db, 1, a);
    vi_add(db, 2, b);
    n = vi_search(db, q, 2, ids, scores);
    for (i = 0; i < n; ++i) printf("%d %.3f\n", (int)ids[i], scores[i]);
    vi_close(db);
    return 0;
}
```

```sh
cc -Iinclude demo.c -L.build -lvecindex -lm -lpthread -o demo && LD_LIBRARY_PATH=.build ./demo
1 0.994
2 0.110
```

For an LLM, each row can carry its text: `vi_add_text`, then `vi_search_hybrid` (the vector and BM25 over the texts,
fused by rank) and `vi_range` over a document's ids to hand the model the chunks around a hit, or the whole document.
`.build/vfind docs.vxi` searches those texts as you type, like fzf: a few lines of each hit, Enter opens the whole
document, Enter again prints the choice. The same from Python (ctypes), Go (cgo) and Rust (`extern "C"`), every
function, texts, and how durability works: [docs/api.md](docs/api.md). Inside a C program the library can also be
one more file of the build (`src/vecindex.c`, or `store.c` itself for the internal API). Over a socket, `vserv` serves one store file to anything that can open a
socket, in one-line commands (`search 10 0.1 0.2 ...` -> `ok 3 42 0.99 ...`), binary frames or HTTP
([docs/server.md](docs/server.md)).

## Documentation

- [docs/api.md](docs/api.md) - the C API, bindings from Python, Go and Rust, durability (journal and snapshots).
- [docs/benchmarks.md](docs/benchmarks.md) - GloVe-100 against FAISS and turbovec (the Qdrant runs never finished); scale; the priority table
  of optimisations; bottlenecks; how to reproduce.
- [docs/server.md](docs/server.md) - `vserv`: sockets, protocols, TLS, live mode.
- [docs/format.md](docs/format.md) - the file format and what stays compatible: every release's files open in every
  later one, a newer build's are refused rather than misread, the C API only grows (all tested by `./run`)
- [docs/store.md](docs/store.md) - the store: the TurboQuant-style scan, IVF, self-tuning, live ingest, rows past
  RAM, stemming, the SQuAD retrieval benchmark.
- [docs/roadmap.md](docs/roadmap.md) - what is done, what is written but not run, what is open.
- [docs/layout.md](docs/layout.md), [docs/production-readiness.md](docs/production-readiness.md), [docs/bindings.md](docs/bindings.md), [docs/results.md](docs/results.md) - repository layout, production readiness, the Python/Rust/JavaScript packages, the GloVe-100 table.
- [docs/comparison.md](docs/comparison.md) - how this compares to mainstream RAG and vector-DB stacks.

## License

MIT, see `LICENSE`. BearSSL is MIT too (`third_party/bearssl/LICENSE.txt`).
