# Bindings: Python, Rust and JavaScript

Thin packages over the same C, each in its own folder under `bindings/`, each built and tested by `./run`:

| | install | what it is | runs on |
|---|---|---|---|
| Python | `pip install ./bindings/python` | the C built into the package, loaded with ctypes: no Python C API, no dependency | one wheel for every Python 3 from 3.8 (tested 3.8 to 3.14) |
| Rust | `vecindex = { path = "bindings/rust" }` | `build.rs` compiles the C with the system's compiler, no crate dependencies; `Db` is `Send + Sync` | `rust-version` 1.56 |
| JavaScript | `npm install ./bindings/js` | `vecindex.wasm` (300 kB) with its own WASI layer: files in a directory (Node) or in memory (browsers) | Node 16 to 22, Bun, Deno (tested); browsers |

```python
import vecindex
with vecindex.open("docs.vxi", dim=384) as db:
    db.add(1, embedding, text="the chunk")
    print(db.search(query, k=5, ns=["alice", "shared"]))
```

A store written by one opens in the others: it is the same file. `bindings/vendor.sh` copies the C into each package
for an sdist, `cargo package` or `npm pack` away from the repository.
