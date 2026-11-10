# The C API

`include/vecindex.h` is the whole interface: one opaque handle per store, plain C types, a status or a count from
every call. `./run` builds `.build/libvecindex.so` and `.build/libvecindex.a` from `src/vecindex.c`; the shared
library exports the `vi_*` functions and nothing else (it is built with `-fvisibility=hidden`, and `./run` checks
the exported symbols). Any language with a C foreign-function interface binds it without a compiler step of its
own. The examples below were run against the library; each prints

```
1 0.994
2 0.110
```

Scores are the cosine, bigger is better: 1 the same direction, 0 unrelated, -1 the opposite. Every call may be
made from any thread on the same handle at once. A call that fails returns -1 (NULL for `vi_open`, `vi_dir_*` and
`vi_batch_begin`) and `vi_error()` says why in words, per thread: "the vector has a NaN or an infinity at index 3",
"no row with id 42", "docs.vxi holds vectors of 384 dimensions, not 768". A NULL handle is such an error, not a
crash. Rows are indexed as they come by a thread of the store's own (`VI_MANUAL` at `vi_open` turns it off, then
`vi_build` indexes).

Ids are 64-bit numbers; `vi_key("docs/readme.md", -1)` turns a string into one, the same on every machine (the
first 8 bytes of its BLAKE3; two strings share an id with a chance of about n^2 / 2^64 among n of them). Keep the
string as the row's text to get it back from a search.

Labels name the 64 bits of a row's tag and sit in a hierarchy: `vi_label(db, "dog", "animal")`, then
`vi_label_add(db, id, "dog,big")` and `vi_search_labeled(db, q, 10, "animal", NULL, "big", ids, scores)` - every
row labelled dog or cat (or anything else under animal) that is not big. A label stands for everything below it,
worked out when the search runs, so putting a new label over existing ones (`vi_label(db, "dog", "pet")`) changes no
row. 64 labels a store; the table is saved in the file and every change is journaled.

A row also carries one 64-bit number for range filters - a time, a price in cents, a version: `vi_set_num(db, id,
t)`, then `vi_search_range(db, q, 10, from, to, ids, scores)` is the nearest rows with a number in `[from, to]`. The
range is tested inside the scan, before the candidates are picked, so a range that holds a handful of rows far from
the query still finds them all. Every row's number is 0 until set; `vi_add` over a row starts it again at 0, and
`vi_batch_set_num` sets it in the same batch as the row, so a crash never leaves the row without it.

A store compares by cosine unless it is made with `VI_DOT` (the inner product of the vectors as they came: scores
bigger is better) or `VI_L2` (the Euclidean distance: scores are the distance, smaller is better, still best first).
Such a store keeps each row's length beside its direction (4 bytes a row); the index finds candidates by direction
and four times as many of them are re-scored exactly in the metric, so with lengths that vary a lot a true neighbour
far off in direction can be missed - raise the re-scored count when recall matters there. `vi_get` returns a dot or
L2 store's vectors as they came. A file keeps the metric it was made with; opening it with another flag is an error.

## Functions

| function | what it does |
|---|---|
| `vi_open(path, dim, flags)` | opens the store in `path`, or creates it with `dim` dimensions; `path` NULL: in memory; `flags` `VI_LIVE` indexes added rows as they come |
| `vi_close(db)` | flushes the journal to the disk and frees the handle |
| `vi_add(db, id, vec)`, `vi_add_batch(db, n, ids, vecs)` | adds rows, or replaces the row with that id (one change: a crash leaves the old row or the new one); a vector with a NaN or an infinity is refused |
| `vi_delete(db, id)` | removes the row with this id |
| `vi_delete_range(db, lo, hi)` | removes every row with an id in `[lo, hi]`: all the chunks of a document when ids are `doc << 20 \| chunk` |
| `vi_get(db, id, out)` | the stored, unit-normalised vector |
| `vi_add_text(db, id, vec, text, len)` | `vi_add` with a text kept beside the row (any bytes, up to 16 MB - 1; `len` -1: a C string) |
| `vi_get_text(db, id, buf, cap)` | the row's text: its length, the first `cap` bytes copied into `buf` |
| `vi_range(db, lo, hi, ids, max)` | the ids in `[lo, hi]` that have a row, ascending: the chunks of a document |
| `vi_dir_open(path, dim, flags)`, `vi_dir_get(dir, name)`, `vi_dir_close(dir)` | tenants without a server: a directory with a store per tenant (`path/NAME.vxi`), each opened - or created - on its first `vi_dir_get` and handed back after; names are 1 to 64 of `A-Z a-z 0-9 _ -` |
| `vi_batch_begin(db)`, `vi_batch_add(b, id, vec)`, `vi_batch_add_text(b, id, vec, text, len)`, `vi_batch_delete(b, id)`, `vi_batch_delete_range(b, lo, hi)`, `vi_batch_set_num(b, id, num)` | a write batch: its changes are copied in as they are added and wait |
| `vi_batch_commit(b, sync)`, `vi_batch_abort(b)` | applies the batch as one change - searches see none of it before and all of it after, and a crash leaves all or none of it (one journal record with a CRC-32C) - and frees it; `sync` 1: on the disk before it returns. The number of changes, -1 refused with nothing applied (a NaN, a bad text, no memory), -2 applied but the fsync failed |
| `vi_search_hybrid(db, q, text, tlen, k, ids, scores)` | the vector and the words (BM25 over the texts), fused by rank; either side may be left out |
| `vi_set_tag`, `vi_get_tag` | a 64-bit tag per row |
| `vi_set_num(db, id, num)`, `vi_get_num(db, id, &num)`, `vi_search_range(db, q, k, lo, hi, ids, scores)` | a 64-bit number per row (0 until set), and a search over the rows whose number is in `[lo, hi]` |
| `vi_error()` | why the calling thread's last failed call failed, in words |
| `vi_key(s, len)` | a 64-bit id for a string (a path, a URL), the same everywhere |
| `vi_label(db, name, parent)`, `vi_label_add(db, id, labels)`, `vi_label_remove(db, id, labels)`, `vi_search_labeled(db, q, k, all, any, none, ids, scores)` | labels in a hierarchy on rows, and a search over every label of `all`, one of `any`, none of `none` |
| `vi_search(db, q, k, ids, scores)` | the k nearest rows, best first |
| `vi_search_tagged(db, q, k, all, any, ids, scores)` | the same over rows whose tag has every bit of `all` and one of `any` |
| `vi_search_grouped(db, q, k, shift, ids, scores)` | at most one hit a group (`id >> shift`): with `shift` 20 and ids `doc << 20 \| chunk`, the best chunk of each of the k nearest documents |
| `vi_search_batch(db, qs, nq, k, ids, scores, threads)` | many queries on several threads |
| `vi_build(db)` | builds the index (searches before it scan the new rows exactly) |
| `vi_set_recall(db, target)` | keeps recall@10 at `target`: every build picks the fewest probes that reach it |
| `vi_set_layout(db, code_bits, row_bytes)` | 4-, 2- or 1-bit scan codes (dim/2, dim/4 or dim/8 bytes a row), f32 or int8 rows, before the first row |
| `vi_set_aggression(db, a)` | how much of the machine to take, 0..1 (default 1, or `VECINDEX_AGGRESSION`): threads out of the CPUs the process may use (affinity, cgroup quota); background work at nice 10 to 19 |
| `vi_set_memory(db, tier)` | what stays in RAM whatever the memory pressure (mlock): 0 nothing, the kernel decides (default); 1 what every search scans (codes, slot table, centroids), rows stay on the disk; 2 rows, ids and tags too. -1 when the memlock limit refused part of it |
| `vi_set_parity(db, data, parity)`, `vi_set_check_reads(db, on)`, `vi_set_verify(level)`, `vi_open_error()`, `vi_integrity(db, &health)`, `vi_scrub(path, repair, &fixed)` | integrity: Reed-Solomon parity of the files written (default 32 + 2), row checks on reads (default on), what an open checks (default every block), why an open failed, what the checks found, a scrub of a file (see Integrity below) |
| `vi_memory(db, &hot, &cold)` | bytes scanned by every search (hot) and bytes read only for candidates and hits (cold) |
| `vi_save(db, path)` | rewrites the file and starts a fresh journal |
| `vi_sync(db)` | flushes the journal to the disk |
| `vi_snapshot(db, path)`, `vi_snapshot_wait(db, block)` | writes the file from a child process while this one keeps serving |

## C

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
```

Linking `.build/libvecindex.a` instead needs no `LD_LIBRARY_PATH`; compiling `src/vecindex.c` into the program
(`-Isrc -Iinclude`) needs no library at all.

## Python (ctypes, standard library)

```python
import ctypes

vi = ctypes.CDLL("libvecindex.so")
F = ctypes.POINTER(ctypes.c_float)
vi.vi_open.restype = ctypes.c_void_p
vi.vi_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
vi.vi_add.argtypes = [ctypes.c_void_p, ctypes.c_int64, F]
vi.vi_search.argtypes = [ctypes.c_void_p, F, ctypes.c_int, ctypes.POINTER(ctypes.c_int64), F]
vi.vi_close.argtypes = [ctypes.c_void_p]

Vec = ctypes.c_float * 4
db = vi.vi_open(b"demo.vxi", 4, 0)
vi.vi_add(db, 1, Vec(1, 0, 0, 0))
vi.vi_add(db, 2, Vec(0, 1, 0, 0))
ids, scores = (ctypes.c_int64 * 2)(), (ctypes.c_float * 2)()
n = vi.vi_search(db, Vec(0.9, 0.1, 0, 0), 2, ids, scores)
for i in range(n):
    print(ids[i], round(scores[i], 3))
vi.vi_close(db)
```

```sh
LD_LIBRARY_PATH=.build python3 demo.py
```

## Go (cgo)

```go
package main

// #cgo LDFLAGS: -lvecindex -lm -lpthread
// #include <stdlib.h>
// #include "vecindex.h"
import "C"

import (
	"fmt"
	"unsafe"
)

func main() {
	path := C.CString("demo.vxi")
	defer C.free(unsafe.Pointer(path))
	db := C.vi_open(path, 4, 0)
	defer C.vi_close(db)
	C.vi_add(db, 1, &[]C.float{1, 0, 0, 0}[0])
	C.vi_add(db, 2, &[]C.float{0, 1, 0, 0}[0])
	ids, scores := make([]C.vi_i64, 2), make([]C.float, 2)
	n := C.vi_search(db, &[]C.float{0.9, 0.1, 0, 0}[0], 2, &ids[0], &scores[0])
	for i := 0; i < int(n); i++ {
		fmt.Printf("%d %.3f\n", ids[i], scores[i])
	}
}
```

```sh
CGO_CFLAGS=-I/path/to/vecindex/include CGO_LDFLAGS=-L/path/to/vecindex/.build go build -o demo .
LD_LIBRARY_PATH=/path/to/vecindex/.build ./demo
```

## Rust (extern "C", no crates)

```rust
use std::ffi::CString;
use std::os::raw::{c_char, c_float, c_int, c_void};

#[link(name = "vecindex")]
extern "C" {
    fn vi_open(path: *const c_char, dim: c_int, flags: c_int) -> *mut c_void;
    fn vi_add(db: *mut c_void, id: i64, vec: *const c_float) -> c_int;
    fn vi_search(db: *mut c_void, q: *const c_float, k: c_int, ids: *mut i64, scores: *mut c_float) -> c_int;
    fn vi_close(db: *mut c_void);
}

fn main() {
    let path = CString::new("demo.vxi").unwrap();
    unsafe {
        let db = vi_open(path.as_ptr(), 4, 0);
        vi_add(db, 1, [1.0f32, 0.0, 0.0, 0.0].as_ptr());
        vi_add(db, 2, [0.0f32, 1.0, 0.0, 0.0].as_ptr());
        let (mut ids, mut scores) = ([0i64; 2], [0f32; 2]);
        let n = vi_search(db, [0.9f32, 0.1, 0.0, 0.0].as_ptr(), 2, ids.as_mut_ptr(), scores.as_mut_ptr());
        for i in 0..n as usize {
            println!("{} {:.3}", ids[i], scores[i]);
        }
        vi_close(db);
    }
}
```

```sh
rustc -O demo.rs -L .build -o demo && LD_LIBRARY_PATH=.build ./demo
```

Other languages follow the same pattern through their own FFI (Node with koffi or N-API, Java with the FFM API, C#
with P/Invoke, Zig with `@cImport`, Lua with LuaJIT's FFI); those were not run here.

## Texts: from a small chunk to the whole document

A model answers better from a whole passage than from the sentence that matched, and an embedding matches a short
chunk better than a long one. So keep both: one row per small chunk, its text beside it, and an id that names the
document - `doc << 20 | chunk` - so the chunks around a hit, or all of them, come back with one `vi_range`. When one
document holds many close chunks, `vi_search_grouped(db, q, k, 20, ids, scores)` returns k documents, each by its
best chunk, instead of k chunks of the same one.

```c
#include "vecindex.h"
#include <stdio.h>

int main(void)
{
    vi_db *db = vi_open("rag.vxi", 4, 0);
    const char *chunk[3] = { "Tea came to Europe in the 1600s.", "Green and black tea come from one plant.",
                             "Oolong is partly oxidised." };
    float v[3][4] = { { 1, 0, 0, 0 }, { 0.9f, 0.4f, 0, 0 }, { 0.8f, 0, 0.6f, 0 } }, q[4] = { 0.95f, 0.3f, 0, 0 }, sc[3];
    vi_i64 doc = 42, ids[8], hit;
    char text[256];
    int i, n;
    for (i = 0; i < 3; ++i) vi_add_text(db, doc << 20 | i, v[i], chunk[i], -1);   /* id = document << 20 | chunk */
    n = vi_search_hybrid(db, q, "oxidised", -1, 1, ids, sc);                      /* the best small chunk... */
    hit = ids[0];
    printf("hit: doc %d chunk %d\n", (int)(hit >> 20), (int)(hit & 0xfffff));
    n = vi_range(db, hit >> 20 << 20, (hit >> 20 << 20) | 0xfffff, ids, 8);       /* ...then its whole document */
    for (i = 0; i < n; ++i) {
        int len = vi_get_text(db, ids[i], text, (int)sizeof(text));
        printf("%d: %.*s\n", (int)(ids[i] & 0xfffff), len, text);
    }
    vi_close(db);
    return 0;
}
```

```
hit: doc 42 chunk 2
0: Tea came to Europe in the 1600s.
1: Green and black tea come from one plant.
2: Oolong is partly oxidised.
```

The query vector is closest to chunk 1; the word "oxidised" picks chunk 2. `vi_search_hybrid` runs both sides and
fuses them by rank (reciprocal rank fusion with weights measured on Wikipedia: words 1.5, vector 0.6, plus up to 2 for a
side whose best hit stands well clear of its runner-up), so an exact name, a number or a rare word is found where the
embedding blurs it. Pass `NULL` for the words to search by the vector alone, or for the vector to search by words.

Texts are journaled with their row (one record, so a crash keeps both or neither), saved in the file and mapped from
it on open, so nothing is read until it is asked for; texts added since the last save live in memory (spilled to a
file with the columns past 256 MB). Replaced and deleted texts stay until the next `vi_save`, which writes only the
live ones and, when at least half the heap is garbage, compacts the copy in memory too. The word index is built by the
first hybrid search after an open (one pass over the texts) and kept current by every write after it.

To look through a store's texts by hand, `.build/vfind store.vxi` searches them as you type, like fzf: each hit with
its document, chunk and a few lines around the words, Enter opens the whole document (its chunks by `vi_range`), Enter
there prints `ID<TAB>DOCUMENT<TAB>CHUNK` on stdout. It opens the store read-only, so a running `vserv` may keep
writing to it.

## Durability

A store opened from a file keeps a journal of every add and delete next to it (`PATH.j`). `vi_sync` flushes it to
the disk; `vi_open` replays it. `vi_save` rewrites the whole file and starts a fresh journal. `vi_snapshot` does the
same from a child process (`fork`: it writes a copy-on-write image of the store at that moment) while the parent keeps
answering searches and taking writes; the writes made meanwhile go to the old journal and to the new file's
(`PATH.jn`), which takes over once the file is complete. At any moment of a crash, the file on disk and one of the
two journals belong together: both carry the same 64-bit stamp. `tests/scrash.c` kills a writer with snapshots running
at random moments, in every other round together with its snapshot child, and checks that every synced row is there
exactly once.

Every single write is atomic: one journal record, and a replacement removes the old row and adds the new one in that
same record, so a crash leaves one or the other. Several writes are atomic together in a batch: replacing a document
is deleting its range of ids and adding its new chunks, and

```c
vi_batch *b = vi_batch_begin(db);
vi_batch_delete_range(b, doc << 20, (doc << 20) | 0xfffff);
for (i = 0; i < nchunks; ++i) vi_batch_add_text(b, (doc << 20) | i, vec[i], text[i], -1);
if (vi_batch_commit(b, 1) < 0) { /* -1: nothing changed; -2: changed, the fsync failed */ }
```

makes searches see the old document or the new one, never a mix, and leaves the same after a crash: the batch is one
journal record (its length, a CRC-32C, the records), and replaying the journal applies it only when all of it is there
and the CRC matches. Without `sync` the batch is on the disk at the next `vi_sync` (the server flushes every second);
with it, `vi_batch_commit` returns once an fsync covers it, and commits that arrive while an fsync runs share the
next one. A journal with batches in it needs this version of vecindex or a later one to be replayed.

## Integrity: checksums, Reed-Solomon parity, scrub

Every file is written in 64 KiB blocks, each with a BLAKE3 digest. The blocks are dealt into groups of at most 32 -
block b into group b mod G, so a damaged run of neighbouring blocks spreads over many groups - and every group gets 2
Reed-Solomon parity blocks (GF(2^8), a Cauchy matrix): any 2 damaged blocks of a group are rebuilt, about 6% more
disk. The table of digests and the trailer that vouches for it (its BLAKE3 and a CRC-32C) are written twice, 4 KiB
apart at least. `vi_set_parity(db, data, parity)` changes the group for the files the store writes from then on.

`vi_open` checks every data block before it reads anything, the header included. A damaged block is rebuilt from its
group, confirmed by its digest and written back to the file (unless the store was opened read-only). When a group has
more damage than parity: inside the saved index, the index is built again from the rows; anywhere else (rows, ids,
tags, texts) `vi_open` fails and `vi_open_error()` says where, rather than serving rows that are not the ones written.
`vi_integrity` reports what the open found. Checking every block reads the whole file once; `vi_set_verify(1)` (or
`VECINDEX_VERIFY=1`) checks only the header's blocks, for stores too large to read at every start - then run
`vi_scrub`, which reads every block, the parity and both copies of the table from the disk, and with `repair` rebuilds
and rewrites what it can, beside a store that has the file open. `VECINDEX_VERIFY=0` opens a file whose checksums are
gone.

The journal frames every record with its length and CRC-32C. A frame that fails its CRC with more journal after it is
damage, not a crash: replay stops there as for a torn tail, a copy of the whole journal is kept as `PATH.j.damaged`,
and `vi_integrity` says so. A zero-filled tail (a file system that grew the file without the bytes landing) counts as
torn.

Reads after the open are checked as well. Every row carries a 32-bit check of its vector, id and tag; a search tests
each row it re-scores, `vi_get` the row it returns. A row that fails was read back from a disk page that went bad after
the open: the blocks under it are rebuilt from the parity, confirmed by their digests and written back, and the row is
used as it was - ZFS heals a read the same way. When there is nothing to rebuild from (the damage is in memory, or
beyond the parity) the row is left out of every result: damaged data is never returned. `vi_integrity` counts both.
The test costs a hash of each re-scored row; `vi_set_check_reads(db, 0)` turns it off. The scan's codes are an index
built from the rows - damage there costs recall, never a wrong row - and are checked at the open and by a scrub, or
never leave RAM with `vi_set_memory(db, 1)`.

Damage beyond the parity of a group is found, not repaired. Files written by an older version (v8 and before) load
as before, without block checks and with the row checks computed at the open, until the next save writes v9.

A snapshot costs memory for the pages the parent changes while the child writes, up to a second copy of whatever is
rewritten meanwhile. Under strict overcommit (`vm.overcommit_memory=2`) `fork` can fail for a store larger than the
free commit limit; `vi_snapshot` then returns -1 and nothing changes.
