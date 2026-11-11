// Tests of the npm package: node test/test.mjs (plain asserts and no test runner, so every Node it supports runs them).
import assert from 'node:assert';
import * as fs from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { open, version, Library, MemoryFiles, NodeFiles } from '../index.mjs';

const DIM = 8;
const MAGIC = 0x34584956;                                  // "VIX4", STORE_MAGIC of src/store.c
const vec = i => Array.from({ length: DIM }, (_, d) => Math.sin(i * 1.3 + d * 0.7) + (d === i % DIM ? 0.5 : 0));
let n = 0;
const t = async (name, f) => { await f(); n++; console.log('  ok   ' + name); };

const dir = fs.mkdtempSync(join(tmpdir(), 'vecindex_js_'));
try {
  await t('the version is the package\'s', async () => {
    assert.strictEqual(await version(), JSON.parse(fs.readFileSync(new URL('../package.json', import.meta.url))).version);
  });

  await t('add, search and texts in a file; reopened, all of it back', async () => {
    let db = await open('t.vxi', { dim: DIM, dir });
    for (let i = 1; i <= 200; i++) db.add(i, vec(i), i % 5 === 0 ? `chunk ${i}` : null);
    assert.strictEqual(db.length, 200);
    const h = db.search(vec(42), 3);
    assert.strictEqual(h[0][0], 42);
    assert.ok(h[0][1] > 0.99);
    assert.strictEqual(db.text(15), 'chunk 15');
    assert.strictEqual(db.text(16), null);
    db.sync();
    db.close();
    assert.ok(fs.existsSync(join(dir, 't.vxi')) || fs.existsSync(join(dir, 't.vxi.j')));
    db = await open('t.vxi', { dir });
    assert.strictEqual(db.dim, DIM);
    assert.strictEqual(db.length, 200);
    assert.strictEqual(db.search(vec(7), 1)[0][0], 7);
    assert.strictEqual(db.text(20), 'chunk 20');
    assert.ok(db.get(9) instanceof Float32Array && db.get(999) === null);
    db.save();
    db.close();
  });

  await t('the file is the C library\'s own: its magic, file version 9 or 10, 200 rows, 8 dimensions', async () => {
    const b = fs.readFileSync(join(dir, 't.vxi'));
    assert.strictEqual(b.readUInt32LE(0), MAGIC);
    assert.ok([9, 10].includes(b.readInt32LE(4)));
    assert.strictEqual(b.readInt32LE(8), DIM);
    assert.strictEqual(b.readInt32LE(12), 200);
  });

  await t('errors: a wrong length, a NaN, a file that is not a store', async () => {
    const db = await open(null, { dim: DIM });
    assert.throws(() => db.add(1, [1, 2]), /floats was expected/);
    assert.throws(() => db.add(1, new Array(DIM).fill(NaN)));
    assert.strictEqual(db.delete(99), false);
    assert.throws(() => db.search(vec(1), 3, { text: 'x', ns: 'a' }), /one condition/);
    db.close();
    fs.writeFileSync(join(dir, 'bad.vxi'), 'not a store at all, but some bytes');
    await assert.rejects(open('bad.vxi', { dir }));
  });

  await t('namespaces, numbers, labels, words, an update, a namespace deleted', async () => {
    const db = await open(null, { dim: DIM });
    for (let i = 1; i <= 300; i++) {
      db.add(i, vec(i), i === 77 ? 'the rare word zyxwv' : `plain text ${i}`);
      db.setNum(i, i * 10);
      db.setNs(i, i % 2 ? 'odd' : 'even');
      if (i % 3 === 0) db.addLabels(i, i % 6 === 0 ? 'dog' : 'cat');
    }
    db.label('dog', 'pet'); db.label('cat', 'pet');
    const q = vec(150);
    assert.ok(db.search(q, 20, { ns: 'even' }).every(([i]) => i % 2 === 0));
    assert.strictEqual(db.countNs('odd'), 150);
    assert.strictEqual(db.search(q, 20, { ns: ['odd', 'even'] })[0][0], 150);
    assert.ok(db.search(q, 20, { num: [1000, 1500] }).every(([i]) => i >= 100 && i <= 150));
    const pets = db.search(q, 50, { labels: { all: 'pet' } });
    assert.ok(pets.length && pets.every(([i]) => i % 3 === 0));
    assert.ok(db.search(q, 50, { labels: { all: ['dog'] } }).every(([i]) => i % 6 === 0));
    assert.strictEqual(db.search(null, 1, { text: 'zyxwv' })[0][0], 77);
    db.update(150, vec(151));
    assert.strictEqual(db.num(150), 1500);
    assert.strictEqual(db.ns(5), (await Library.load({ files: new MemoryFiles() })).key('odd'));
    assert.strictEqual(db.deleteNs('odd'), 150);
    assert.strictEqual(db.length, 150);
    db.close();
  });

  await t('one hit a document (ids doc << 20 | chunk); ids past 2^53 as BigInt', async () => {
    const db = await open(null, { dim: DIM });
    for (let doc = 1; doc <= 5; doc++) for (let c = 0; c < 4; c++) db.add(doc * 2 ** 20 + c, vec(doc * 10 + c));
    const docs = db.search(vec(30), 3, { group: 20 }).map(([i]) => Math.floor(i / 2 ** 20));
    assert.strictEqual(docs[0], 3);
    assert.strictEqual(new Set(docs).size, docs.length);
    db.add(2n ** 60n + 5n, vec(500));
    assert.strictEqual(db.search(vec(500), 1)[0][0], 2n ** 60n + 5n);
    db.close();
  });

  await t('memory files: a store exported as bytes and imported into another instance (a browser\'s IndexedDB)', async () => {
    const files = new MemoryFiles();
    const lib = await Library.load({ files });
    const db = lib.open('m.vxi', { dim: DIM });
    for (let i = 1; i <= 50; i++) db.add(i, vec(i), `m ${i}`);
    db.save();
    db.close();
    const bytes = files.exportFile('m.vxi');
    assert.ok(bytes && bytes.length > 0);
    const f2 = new MemoryFiles();
    f2.importFile('m.vxi', bytes);
    const db2 = (await Library.load({ files: f2 })).open('m.vxi');
    assert.strictEqual(db2.length, 50);
    assert.strictEqual(db2.text(33), 'm 33');
    assert.strictEqual(db2.search(vec(33), 1)[0][0], 33);
    db2.close();
  });

  await t('a store written by the WebAssembly build opens in the native one (same file format)', async () => {
    const native = process.env.VECINDEX_NATIVE_PROBE;     // ./run sets it: a C program that prints count and a text
    if (!native) { console.log('       (no VECINDEX_NATIVE_PROBE: skipped)'); return; }
    const { execFileSync } = await import('node:child_process');
    const out = execFileSync(native, [join(dir, 't.vxi')]).toString().trim();
    assert.strictEqual(out, '200 chunk 20');
  });

  await t('stores the native builds wrote (tests/compat: 0.3.0 and 0.4.0) open here with their rows and texts', async () => {
    const compat = new URL('../../../tests/compat/', import.meta.url);
    if (!fs.existsSync(compat)) { console.log('       (not inside the repository: skipped)'); return; }
    const { gunzipSync } = await import('node:zlib');
    for (const name of ['v9-0.3.0-cos', 'v10-0.4.0-cos', 'v9-0.3.0-l2i8']) {
      const files = new MemoryFiles();
      files.importFile(name + '.vxi', gunzipSync(fs.readFileSync(new URL(name + '.vxi.gz', compat))));
      files.importFile(name + '.vxi.j', gunzipSync(fs.readFileSync(new URL(name + '.vxi.j.gz', compat))));
      const db = (await Library.load({ files })).open(name + '.vxi');
      assert.strictEqual(db.length, 333, name);                // tests/compat/compat.h: 340 written, 7 deleted
      assert.strictEqual(db.text(3), 'the text of row 3, kept since the format of its file', name);
      assert.strictEqual(db.search(db.get(42), 1)[0][0], 42, name);
      db.close();
    }
  });

  const rt = globalThis.Deno ? 'deno ' + Deno.version.deno : process.versions.bun ? 'bun ' + process.versions.bun : 'node ' + process.versions.node;
  console.log(`js: OK (${n} tests, ${rt})`);
} finally {
  fs.rmSync(dir, { recursive: true, force: true });
}
