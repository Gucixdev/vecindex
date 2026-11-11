# vecindex for JavaScript

The vecindex vector database compiled to WebAssembly: one `vecindex.wasm` (300 kB) that runs the same in Node, Deno,
Bun and browsers. Nothing is compiled at install and there are no dependencies: the module imports only the few WASI
calls `wasi.mjs` implements itself, so it does not depend on the runtime's own WASI support either. Tested on Node 16,
18, 20 and 22, Bun 1.4 and Deno 2.9; browsers run the same code with files in memory (not run in a browser here).

```js
import { open } from 'vecindex';

const db = await open('docs.vxi', { dim: 384, dir: './data' });  // Node: the files in ./data; elsewhere: memory
db.add(1, embedding, 'the chunk the embedding came from');
db.setNs(1, 'alice');
for (const [id, score] of db.search(query, 5)) console.log(id, score, db.text(id));
db.search(query, 5, { ns: ['alice', 'shared'] });   // several namespaces at once
db.search(null, 5, { text: 'invoice 4417' });       // words alone (with a vector: fused with it)
db.search(query, 5, { group: 20 });                 // one hit a document, ids doc << 20 | chunk
db.close();
```

In a browser the store lives in a `MemoryFiles`; keep it across visits by exporting its files (IndexedDB, a download)
and importing them next time:

```js
import { Library, MemoryFiles } from 'vecindex';
const files = new MemoryFiles();
const db = (await Library.load({ files })).open('docs.vxi', { dim: 384 });
// ... db.save(), then: files.exportFile('docs.vxi') -> Uint8Array; later files.importFile('docs.vxi', bytes)
```

The files are the C library's own: a store written here opens in the native build, and back (`test/test.mjs` checks
both, with the stores in `tests/compat`). In WebAssembly there is one thread, so the store's background indexing and
snapshots do not run: rows not yet indexed are found by an exact scan, `db.build()` indexes them, `db.save()` writes
the file.

Build the module: `WASI_SDK=/path/to/wasi-sdk WASM_OPT=wasm-opt ./build.sh` (the WASI SDK from
github.com/WebAssembly/wasi-sdk/releases; wasm-opt from binaryen re-encodes it for engines before Node 18). Tests:
`node test/test.mjs`.
