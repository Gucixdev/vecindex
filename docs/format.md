# The file format and what stays compatible

A store is a file, `NAME.vxi`, and its journal, `NAME.vxi.j`. An application ships with its data in them, so they
have to outlive the build that wrote them. From 0.4.0 these are promises, and `./run` checks each one on every change:

1. **Every file a release wrote opens in every later release, with every value.** `tests/compat` keeps stores
   written by each release (by `tests/compat/make.c` built against it): rows saved in the file, more changes only in
   the journal, texts, labels and tags, int8 rows, the Euclidean metric, numbers and namespaces where the release had
   them. `tests/compat_test.c` opens each with this build and checks every row's vector, text, tag, label, number and
   namespace, the deleted rows gone, a search, and a clean scrub. A release that writes something new adds its files
   there; none is ever taken out.
2. **A file a build cannot read whole is refused, never misread.** A newer file version, a feature flag it does not
   know, or a journal record of a kind it does not know (in a frame whose CRC vouches for it, so it is not damage):
   the open fails with a message that names the newer build as the reason, and nothing on the disk is touched - the
   journal is not cut as if it were a torn tail. Tested the same way.
3. **A build writes the oldest version that holds the data.** A store that uses nothing past v9 is written as v9, so
   0.3.0 still opens it; one with numbers, namespaces or texts (whose CRCs are new) is written as v10, which 0.3.0
   refuses (it knows versions up to 9).
4. **The C API only grows.** Every declaration of a release's `vecindex.h` stays, word for word
   (`tests/compat/api-VERSION.txt`, made by `tests/compat/api_decls.sh`); `./run` fails on one that changed or went.
   `vi_version()` says which build is running.

Not promised: that an older build opens a newer file (it refuses, cleanly), that the bytes of a file stay the same
across releases, or anything about the server's protocols yet.

One case the promise cannot reach back to: 0.3.0 was written before rule 2. A v9 file whose journal already holds a
change it does not know (a number or a namespace set after the last save, by 0.4.0) makes 0.3.0 stop its replay at
that change, report a damaged journal (`vi_integrity`: `journal_damaged`) and keep the whole journal as
`NAME.vxi.j.damaged` - not silent, but the later changes are missing in 0.3.0 until a newer build opens the store.

## Versions

| file version | written by | what came with it | tested in `tests/compat` |
|---|---|---|---|
| 4 | early builds | the oldest layout the reader still knows | no file of it survives: the reader is kept, untested |
| 5 | | 64-byte aligned sections, the scan index saved, loaded by mmap | as 4 |
| 6 | | int8 rows and their scales | as 4 |
| 7 | | an extension on every file: flags, the stamp the journal is bound to | as 4 |
| 8 | | texts | as 4 |
| 9 | 0.3.0, and 0.4.0 for stores without what v10 adds | block digests, Reed-Solomon parity, the rows' checks, the metric, labels | yes: 0.3.0 (cosine; int8 rows with L2), 0.4.0 |
| 10 | 0.4.0 | numbers, namespaces, CRCs of the texts | yes: 0.4.0 (cosine; int8 rows with L2) |

Journals: `VXJ1` (bound to the file by its size), `VXJ2` (by the stamp), `VXJ3` (a header with its CRC, then frames
each with a CRC-32C). A build reads all three; 0.3.0 and 0.4.0 write `VXJ3`. The kinds of change in a frame: `A`, `P` and
`Q` (rows), `D` and `R` (deletes), `M` (a tag), `L` (a label), and from 0.4.0 `U` (a row's vector updated), `N` (a
number), `S` (a namespace) and `Y` (a namespace deleted).

## Adding a version

A change to what a file holds gets a new flag or a new version, never a new meaning for an old one. Then: the reader
learns it (`STORE_FLAGS_KNOWN`, the version check in `store_load_impl_`), the writer uses it only when the data needs
it (`store_save`: the oldest version that holds the data), `make.c` gets the feature behind the release's `-D`, its
files go into `tests/compat`, and this table gets a row.
