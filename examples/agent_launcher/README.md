# agent_launcher

Notes and memory for LLM agents, built on the store of [vecindex](https://github.com/Gucixdev/vecindex):

- `src/vecdb.c` and its parts: a note database. Notes with `[[wikilinks]]`, backlinks, `#tags` and ghost notes;
  hybrid search (vector kNN and BM25 fused by rank); MCTS over the link graph; embedding servers in child processes;
  a command line (`vecdb_cli`). Its own files, journals, a readers-writer lock.
- `src/mem.c`, `mem_rules.c`, `mem_disk.c`: agent memory on top of vecdb. Lifetime and tiers, event -> fact ->
  concept, a rule engine, its own files.
- `src/rerank.c`: a cross-encoder second pass over vecdb's hybrid search, served by `py/rerank_srv.py`.

These files lived in vecindex until commit `bde1f23`; their history came into holyc with it (`git log bde1f23 --
src/vecdb.c`). vecindex is the vector database alone, and this layer uses it as a library: every program here
includes `../../src/store.c`.

## Build and test

```sh
cd holyc/vecindex/examples/agent_launcher     # builds against ../..; another copy: VECINDEX=/path/to/vecindex
./run            # vecdb and mem self-tests, the CLI end to end, fuzzers (ASan/UBSan), threads (TSan), kill -9 tests, rerank
./run stress     # also tests/mstress.c
./run bench      # also bench/mbench.c
```

One program by hand: `gcc -std=c89 -O2 -Isrc -I../../src tests/vecdb_test.c -lm -lpthread`.

| path | what is there |
|---|---|
| `src/` | vecdb (notes, embedding, search, graph, files, CLI), mem (memory, rules, files), rerank |
| `tests/` | self-tests, fuzzers, thread and crash tests for vecdb and mem; the rerank demo |
| `bench/` | `vbench.c` (synthetic), `vbench1m.c` (millions of notes), `vbench_wiki.c` (Simple English Wikipedia, `wiki.rec` from `../../py/wikiprep.py`), `mbench.c` (mem timings) |
| `py/` | `embedsrv.py` (multilingual-e5, CLIP), `rerank_srv.py`, `eval_rerank.py` (on `../..`'s `real_val` data), `extract.py` (text from PDF, DOCX and others for the CLI) |
| `docs/` | `vecdb.md`, `mem.md`: design and the numbers measured |

## Status

Moved as is. Every program compiles with gcc and clang (`-std=c89 -pedantic -Werror` for the self-tests and the CLI)
against `../..`; the tests have not been run since the move. The numbers in `docs/` were measured inside vecindex
before it. The rule engine (`mem_rules.c`) adds little and is the first candidate to simplify or drop.

## License

MIT, see [`../../LICENSE`](../../LICENSE).
