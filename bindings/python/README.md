# vecindex for Python

The vecindex vector database as a Python package: one C library built into it, loaded with ctypes, nothing else to
install. One wheel serves every Python 3 from 3.8 on (`py3-none-PLATFORM`); tested on 3.8, 3.9, 3.10, 3.11, 3.12,
3.13 and 3.14.

```sh
pip install ./bindings/python          # from the repository; builds the C (gcc or clang)
VECINDEX_PORTABLE=1 pip wheel ./bindings/python   # a wheel for other machines (no -march=native)
```

```python
import vecindex

with vecindex.open("docs.vxi", dim=384) as db:          # made when it is not there; dim 0 opens an existing one
    db.add(1, embedding, text="the chunk the embedding came from")
    db.set_ns(1, "alice")                                # a namespace by name, or any int
    for id, score in db.search(query, k=5):              # [(id, score)], best first
        print(id, score, db.text(id))
    db.search(query, k=5, ns=["alice", "shared"])        # several namespaces at once
    db.search(query, k=5, text="invoice 4417")           # words too: BM25 fused with the vector
    db.search(query, k=5, num=(1700000000, 1800000000))  # a range of the rows' numbers (a date, a price)
    db.search(query, k=5, group=20)                      # one hit a document, with ids doc << 20 | chunk
    hits, checked = db.search_exact(query, k=5)          # the exact 5 best, with a proof; checked: rows scored exactly
```

Vectors are any sequence of floats; `array.array("f")` and float32 numpy arrays go in without a per-element copy.
Every change is in the journal at once (`db.sync()` makes it durable), `db.save()` rewrites the file. A refused call
raises `vecindex.VecindexError` with the library's reason.

The whole API: `open`, `key`, `version`; on a store `add`, `add_many`, `update`, `get`, `text`, `delete`,
`delete_range`, `set_tag`/`tag`, `set_num`/`num`, `set_ns`/`ns`, `count_ns`, `delete_ns`, `label`, `add_labels`,
`remove_labels`, `search`, `build`, `save`, `sync`, `close`, `len(db)`, `id in db`. The C API underneath and what each
call promises: `include/vecindex.h` and `docs/api.md` of vecindex.

Tests: `python -m unittest discover -s tests` with the package installed or built in place
(`python setup.py build_ext --inplace`).
