# vecdb: notes on top of vecindex

`src/vecdb.c` sits on vecindex's `src/store.c`, `src/hdict.c` (open-addressing hash dictionaries, i64 and string
keys) and `src/lex.c` (BM25 term dictionary + postings); `../vecindex/src` is on the include path.

* **Hash dictionary inside the index** - `src/store.c` keeps an id -> row map, so
  `store_has/get_vec/set_meta/get_meta`, delete and replace are O(1) instead of a scan.
* **Precision** - `vecdb_search` fuses vector kNN and BM25 with reciprocal rank
  fusion, so rare exact words ("Delft") win even when the embedding blurs them.
* **Graph** - `[[wikilinks]]` (alias, `#heading`, `![[embeds]]`), backlinks,
  `#hashtags` (nested `#a/b` also tag `#a`, ignored in code/URLs/headings),
  ghost nodes for links to notes that do not exist yet.
* **MCTS** - `vecdb_explore` runs UCB1 tree search over links, backlinks and tags
  from the hybrid hits and surfaces bridge notes that plain kNN misses.
* CLI: `gcc ... src/vecdb_cli.c -lm -lpthread -o vecdb_cli`, then
  `vecdb_cli DB.vdb index DIR | search Q | explore Q | links T | backlinks T | tag T | tags | ghosts`.
* Tests: `./run` here (vecdb, mem, the CLI, fuzzers, threads, crash safety, rerank), `./run stress`; hdict, lex and
  the store are tested by vecindex's own `./run`.

## vecdb benchmarks

`bench/vbench.c` (synthetic, any N), `bench/vbench_wiki.c` (real Simple English Wikipedia:
`bzcat simplewiki-latest-pages-articles.xml.bz2 | python3 ../vecindex/py/wikiprep.py > wiki.rec`, then
`.build/vbench_wiki wiki.rec`); vecindex's `bench/vcmp.c` + `py/vcmp_turbovec.py` compare its store with turbovec on
identical vectors.

201k articles, 12 cores, hit rates over 300 queries (hashed 256-d embedder):

| | vecindex (vector only) | vecdb hybrid | vecdb MCTS |
|---|---|---|---|
| body's first 12 words -> article in top 10 | 72% | 97% | 96% |
| title -> exact article first | 23% | 100% | 100% |
| title -> its real link neighbours (recall@10) | 5% | 21% | 32% |
| ms / query | 1.6 | 2.3 | 3.7 |

The store underneath (the TurboQuant-style scan, IVF, self-tuning, live ingest, rows past RAM, stemming, the
SQuAD retrieval benchmark) is documented in [vecindex/docs/store.md](../../vecindex/docs/store.md).

## Persistence

`vecdb_save` writes PATH.vdb (node table), PATH.vdt (bodies, stay on disk), PATH.vxi (vector rows) and
PATH.img.vxi. `vecdb_sync` afterwards only appends to journals (`store_sync`, PATH.vxi.j, a torn tail is
ignored on load), so a save costs the change, not the store. Loading rebuilds links, tags and BM25 from the
bodies (~10 us per note); the vector rows and the TurboQuant index are mapped from PATH.vxi as they are (no
rebuild, pages come in when first touched).

## Real embeddings, documents, pictures

`db->embed` / `vecdb_use_embproc` plug in any model through a child process; `py/embedsrv.py` serves
multilingual-e5-small (Polish ok) and `--clip` serves CLIP for pictures. The CLI (`VECDB_EMBED_SERVER`,
`VECDB_CLIP_SERVER` env vars, `repl` keeps the models loaded) indexes .md/.txt, and through `py/extract.py`
pdf/docx/odt/pptx/epub/html: long documents become a document note linked to overlapping-free chunks
(each chunk links back and to its neighbours), pictures go to a second vector space and are linked by
`![[pic.png]]` / `![](pic.png)`. Search: `search`, `images` (text -> pictures), `similar` (picture -> pictures).

## Cross-encoder reranking (rerank.c) - a real second-stage pass, with a real measured effect

Hybrid (vector+BM25 RRF) search is a real substitute for a lot of what reranking buys, but not the same as a
cross-encoder that actually reads query and candidate together - a stage most production RAG stacks add on
top of ANN retrieval that this repo didn't have. `src/rerank.c` (`#include "vecdb.c"`, same layering as mem.c -
vecdb.c itself untouched) adds it: `vecdb_rerank_open` spawns `py/rerank_srv.py` (any cross-encoder model, same
child-process/pipe pattern as `vecdb_embproc_open`, just a different protocol since a reranker scores
(query, candidate) PAIRS, not single texts), `vecdb_rerank_hits` re-scores and re-sorts an existing
`VHit[]` (from `vecdb_search`/`vecdb_query`) in place, falling back to the original hybrid order untouched if
the rerank server is unavailable or fails - always an optional refinement, never the only ranking path.
Self-test: `./run` (`RERANK_DEMO`), degrades gracefully (prints a skip message, does not fail the run) when
no torch-capable Python is reachable.

**Measured effect**, not just "we added an API": against vecindex's `bench/real_val.c` SQuAD benchmark data ([store.md](../../vecindex/docs/store.md)),
on the 871/1000 questions where the gold passage is already somewhere in the raw top-10 cosine neighbours (a
reranker can only reorder what retrieval already found - the other 129 are retrieval misses, unaffected by
any reranking and reported separately), `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking those 10 candidates:

| | MRR | gold@1 |
|---|---|---|
| before (raw cosine order) | 0.751 | 62.1% |
| after (cross-encoder rerank) | 0.887 | 81.5% |
| delta | **+0.136** | **+19.4 points** |

(`py/eval_rerank.py EMB_DIR`, reusing vecindex's `bench/real_val.c` ground truth - its `py/real_texts.py` rebuilds the
corpus/query text alongside it, since its `py/real_gen.py` only ever saved vectors.) A +19.4 point jump in
"is the right answer literally first" is a substantial, real quality gain for the ~330/871 questions where
the gold passage was in the top-10 but not already first - exactly the case a reranker exists for.

## Threads

Searches on a store take a read lock and run concurrently (stress test: 8 readers + a writer);
the TurboQuant scan itself is multi-threaded on big stores, and `store_search_batch` spreads a batch of queries
over threads. The index build is threaded (k-means, assignment, encoding). Lexical (BM25) searches on one index
take turns behind a mutex. Embedding and ingest are single-threaded.

## Query language

`vecdb_query(db, "words #tag -#draft [[Note]] to:[[Note]] from:[[Note]]", k, mcts, out)` (CLI `search` and
`explore` use it): `#tag` / `-#tag` / `tag:x` (nested: `#a` also matches `#a/b`), `[[X]]` notes linked to or from X,
`to:[[X]]` X's backlinks, `from:[[X]]` X's links. Conditions are ANDed with the words; without words the
matches are listed most-linked first; an unknown note or tag matches nothing.

## Embedding: CPU only, batched, threaded

`py/embedsrv.py` never touches the GPU (`CUDA_VISIBLE_DEVICES=""`). `vecdb_add_notes` sends 32 texts per round trip
(`B`/`C` batch kinds) or, with the built-in embedder, embeds on all cores. Measured on 1024 Wikipedia leads
(e5-small, 12 threads): batching alone changes little (the model is compute-bound: ~200 GFLOP/s already),
what pays is fewer tokens and int8: `EMBED_INT8=1` ~ -20% time, `EMBED_MAXLEN=128` ~ -50%
(25 ms/article instead of 57), with no loss in hybrid retrieval (vector-only -1..2 points). Recommended for
ingest: `EMBED_INT8=1 EMBED_MAXLEN=128`. Repeated query strings are served from a 32-entry cache
(a model call is ~12 ms, the search itself <1 ms).

## GPU / CPU mix

`VECDB_EMBED_SERVER="cmd1|cmd2"` starts several servers; batches (1024 texts per flush in the CLI) are cut into
groups of 32 and each server pulls the next group when free; single queries go to the first one.
`EMBED_DEVICE=cuda` puts a server on the GPU (fp32: Pascal's fp16 is throttled and slower), `EMBED_THREADS`
limits its torch threads. Measured on 4094 Wikipedia leads (e5-small, 128 tokens): GPU alone 5.5 ms/article,
CPU alone 36, GPU + CPU int8 22.1 s vs 22.6 s GPU alone (no gain: the CPU server steals cores the GPU
server needs for tokenising), GPU + CPU fp32 31.7 s (worse). For bulk ingest use the GPU alone; a CPU server
is for queries (12 ms) once the card is needed elsewhere.

## Embedding cache

`vecdb_embed_cache_open(db, path, tag)` keeps every document vector the model made in a file, keyed by
hash(tag + text); `tag` names the model and its settings (another tag = separate entries, nothing is mixed).
Re-indexing an unchanged vault asks the model 0 times, a changed vault only for the changed notes (batches
send just the misses). The CLI turns it on whenever `VECDB_EMBED_SERVER` is set: `DB.emb` next to the database,
or `VECDB_EMBED_CACHE=path`. Only documents are cached (queries have their own 32-entry in-memory cache).

`EMBED_MAXLEN` now defaults to 128 tokens (25 ms/article on CPU int8, 5.5 ms on GPU fp32, no loss measured
on Wikipedia leads; raise it for long passages).

## Idea, not done: fixed-point (int8) embedding on the GPU

The GTX 1050 Ti (Pascal, sm_61) has the `dp4a` int8 dot instruction, so a hand-written int8 e5-small could in
theory run 2-3x faster than fp32. Not worth it now: PyTorch has no int8 GEMM path for this card (`_int_mm`
needs newer hardware), so it means custom CUDA kernels (or llama.cpp / onnxruntime int8), the card already
sustains ~1 TFLOPS of its ~2, and the whole 201k-article Wikipedia costs 18 minutes once (the embedding cache
and `vecdb_sync` mean only new or changed notes are ever embedded again). The CPU already uses dynamic int8
(`EMBED_INT8=1`, -20% time, no AVX-VNNI on this machine to gain more). Revisit if a corpus of tens of millions
of chunks shows up.

## Production notes

- **Failure handling.** The model server answers every request with a status byte; a batch that fails is retried
  text by text, a text that still fails yields no vector, and such a note is *not added* (message on stderr) and
  never cached. A query whose embedding failed falls back to words only. The server halves its batch on GPU
  out-of-memory (a card shared with the desktop has 0.3-3.5 GB free at any time) and only gives up on a single
  text. Before this a failed batch silently stored zero vectors.
- **`vecdb_free`** releases a database; `tests/vfuzz.c` (random markup, queries, replace/remove, save/load under
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

## Durability, formats, threads

- **Atomic writes.** Every file that is replaced is written beside the old one, `fsync`ed, then `rename`d (and the
  directory `fsync`ed). Commit order for an incremental save: bodies (append-only, fsynced) -> vector files -> the
  node table *last*, which is the commit record. `vecdb_sync` does the same with the journals. If a crash lands
  between the vector files and the node table, the counts disagree on load and the vectors are recomputed (the
  embedding cache makes that cheap). `tests/vcrash.c` SIGKILLs a writer at random moments, 25 rounds per `./run`
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
  `tests/vcrash2.c` (`./run`) forks a child that re-indexes on top of an existing database and kills it at a random
  point spanning the real completion boundary: over 30 rounds, roughly a even mix of "landed on the old copy"
  and "landed on the complete new copy", zero mixed/corrupted, zero load failures.
- **Format versions.** `.vdb` starts with magic `VDBX` and a version (now 5, which added `bodygen`). Version 3
  and 4 files still load; a file from a newer build, garbage, a missing or truncated file are refused and
  `vecdb_error()` says why. When the format changes, bump `VDB_VERSION` and add a branch to the loader.
- **Threads.** One readers-writer lock per database (`vecdb_*` are safe to call from many threads): searches,
  queries, MCTS and lookups share it; add / remove / smooth / build / save / sync are exclusive. Model calls happen
  outside the lock. The body file and the embedding cache have their own mutexes. `tests/vthreads.c` (4 readers + a
  writer doing adds, replaces, removals, rebuilds and syncs) runs under ThreadSanitizer and ASan in `./run`.

## Real notes check (77 markdown files: memory notes, project docs; Polish and English mixed)

Indexed with e5-small on the GPU (128 tokens): 2075 notes/chunks in 47 s. 20 hand-written questions in Polish and
English (vague on purpose, each with one expected file), hit@1 / hit@3: hybrid 12 / 14, vector only 10 / 11, words
only 11 / 14. So the mix beats each half, but modestly, and it is a 20-question sample with my own labels.
What the misses show: an index note (`MEMORY.md`, one line about every project) matches almost any query
lexically; documents cut into many chunks get many chances to rank; a Polish question about an English note
depends on e5-small's cross-lingual ability (`git` vs `gita`: the light stemmer leaves 4-letter words alone).
The CLI prints one line per document (best chunk, "+N more parts").

## Concurrency model

The vector scan is memory-bound: on 201k x 256 (26 MB of 4-bit codes) one thread needs 2.3 ms, four threads 1.3 ms,
eight or twelve no better, a persistent thread pool no better (tried and removed). So the design stays: one
readers-writer lock for the API, a few data-parallel threads for the scan (4, enough to saturate the bus),
self-scheduling groups for the model servers (a shared counter, which is why a fast GPU takes more of the work),
no coroutines (nothing in-process waits on I/O that a thread would not wait on) and no fixed master/slave
partitions (uneven work makes the static split wait for its slowest half, which is what the CPU+GPU mix showed).
Less memory traffic (2-bit codes with a longer shortlist, IVF pruning) is what would make the scan faster.

## vecdb at scale (hashed embedder - no model server)

`bench/vbench1m.c` (`./.build/vbench1m N [SAVEDIR]`) adds N notes with links, tags and a body through the public API,
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
