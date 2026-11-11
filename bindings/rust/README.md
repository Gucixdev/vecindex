# vecindex for Rust

The vecindex vector database as a crate: `build.rs` compiles its C (one file, `src/vecindex.c`) with the system's C
compiler into a static library - no Rust dependencies, no build dependencies (not even `cc`). `Db` is `Send + Sync`:
the C library takes its own locks, so one store serves many threads.

```toml
[dependencies]
vecindex = { path = "bindings/rust" }      # from the repository
```

```rust
use vecindex::{key, Db, Filter};

let db = Db::open("docs.vxi", 384)?;                       // made when it is not there; 0 opens an existing one
db.add_text(1, &embedding, "the chunk the embedding came from")?;
db.set_ns(1, key("alice"))?;
for (id, score) in db.search(&query, 5)? {
    println!("{} {} {:?}", id, score, db.text(id));
}
db.search_where(Some(&query), 5, Some(Filter::AnyNs(&[key("alice"), key("shared")])))?;
db.search_where(None, 5, Some(Filter::Text("invoice 4417")))?;   // words alone; with a vector, fused
db.search_where(Some(&query), 5, Some(Filter::Group(20)))?;      // one hit a document (ids doc << 20 | chunk)
```

Errors are `vecindex::Error` with the library's reason. `vecindex::ffi` is the raw C API, for what the safe one does
not cover. `CC` and `AR` pick the tools; `-march=native` is used unless cross-compiling or `VECINDEX_PORTABLE=1`.

`rust-version` is 1.56 (edition 2018); clippy, which checks the standard library calls against it, is clean. Built
and tested here with rustc 1.97; an older toolchain itself was not available to try.

Tests: `cargo test` (a store on disk and reopened, errors, every kind of filter, four threads at once).
