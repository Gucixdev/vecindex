// Tests of the npm package: node test/test.mjs (plain asserts and no test runner, so every Node it supports runs them).
import assert from 'node:assert';
import * as fs from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { open, version, Library, MemoryFiles, NodeFiles } from '../index.mjs';

const DIM = 8;
const MAGIC = 0x34584956;                                  // "VIX4", STORE_MAGIC of src/store.c
const vec = i => Array.from({ length: DIM }, (_, d) => Math.sin(i * 1.3 + d * 0.7) + (d === i % DIM ? 0.5 : 0));
let n = 0, skipped = 0;
// a test that returns 'skip' did not run: it is counted apart, never as passed
const t = async (name, f) => { if ((await f()) === 'skip') { skipped++; console.log('  skip ' + name); } else { n++; console.log('  ok   ' + name); } };

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

  await t('scores: each the cosine of its row with the query, best first', async () => {
    const db = await open(null, { dim: DIM });
    for (let i = 1; i <= 100; i++) db.add(i, vec(i));
    const q = vec(1000), hits = db.search(q, 10);
    const cos = (a, b) => { let ab = 0, aa = 0, bb = 0; for (let d = 0; d < DIM; d++) { ab += a[d] * b[d]; aa += a[d] * a[d]; bb += b[d] * b[d]; } return ab / Math.sqrt(aa * bb); };
    assert.strictEqual(hits.length, 10);
    for (let j = 0; j < hits.length; j++) {
      assert.ok(Math.abs(hits[j][1] - cos(q, vec(hits[j][0]))) < 1e-4, `hit ${j}: ${hits[j][1]} vs ${cos(q, vec(hits[j][0]))}`);
      if (j) assert.ok(hits[j][1] <= hits[j - 1][1]);
    }
    db.close();
  });

  await t('a store opened three times, written each time: the journal appended to, not overwritten', async () => {
    for (let round = 0; round < 3; round++) {
      const db = await open('three.vxi', { dim: DIM, dir });
      assert.strictEqual(db.length, round * 10);
      for (let i = 0; i < 10; i++) db.add(round * 10 + i, vec(round * 10 + i), `r${round}`);
      db.sync();
      db.close();
    }
    const db = await open('three.vxi', { dir });
    assert.strictEqual(db.length, 30);
    assert.strictEqual(db.text(25), 'r2');
    db.close();
  });

  await t('the file layer: create, truncate on open, append, read at an offset, rename, unlink, list', async () => {
    for (const files of [new MemoryFiles(), new NodeFiles(fs, fs.mkdtempSync(join(dir, 'fl-')))]) {
      const kind = files.constructor.name;
      let r = files.open('a', true, false, false);
      files.write(r.h, new Uint8Array([1, 2, 3, 4, 5]), 0);
      files.close(r.h);
      r = files.open('a', true, false, true);                        // O_TRUNC
      assert.strictEqual(files.size(r.h), 0, kind + ' truncate');
      files.write(r.h, new Uint8Array([9, 8]), 0);
      files.write(r.h, new Uint8Array([7]), files.size(r.h));
      const b = new Uint8Array(2);
      assert.strictEqual(files.read(r.h, b, 1), 2, kind + ' pread');
      assert.deepStrictEqual(Array.from(b), [8, 7], kind + ' pread');
      files.close(r.h);
      assert.strictEqual(files.open('a', true, true, false).err, 20, kind + ' O_EXCL on an existing file');
      assert.strictEqual(files.open('nope', false, false, false).err, 44, kind + ' ENOENT');
      assert.strictEqual(files.rename('a', 'b'), 0);
      assert.strictEqual(files.stat('a'), null);
      assert.strictEqual(files.stat('b').size, 3, kind + ' rename');
      assert.deepStrictEqual(files.list('').map(e => e.name), ['b'], kind + ' list');
      assert.strictEqual(files.unlink('b'), 0);
      assert.strictEqual(files.stat('b'), null, kind + ' unlink');
    }
    // and through the WASI calls the C library makes: a file opened for appending writes at its end
    const files = new MemoryFiles(), lib = await Library.load({ files });
    const db = lib.open('ap.vxi', { dim: DIM });
    for (let i = 0; i < 5; i++) db.add(i, vec(i));
    db.save(); db.close();
    const j0 = files.exportFile('ap.vxi.j').length;
    const db2 = lib.open('ap.vxi');
    db2.add(99, vec(99)); db2.sync(); db2.close();
    const j1 = files.exportFile('ap.vxi.j');
    assert.ok(j1.length > j0, 'the journal grew');
    assert.strictEqual(j1[0], 0x56, 'its header kept (VXJ3)');
    assert.strictEqual(lib.open('ap.vxi').length, 6);
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

  await t('one hit a document (ids doc << 20 | chunk), where a plain search gives one document\'s chunks; ids past 2^53', async () => {
    const db = await open(null, { dim: DIM });
    for (let doc = 1; doc <= 5; doc++) for (let c = 0; c < 4; c++) db.add(doc * 2 ** 20 + c, vec(doc * 10).map((x, d) => x + (d === c ? 0.001 : 0)));
    const plain = db.search(vec(30), 3).map(([i]) => Math.floor(i / 2 ** 20));
    assert.deepStrictEqual(plain, [3, 3, 3]);                   // the three nearest are chunks of one document
    const docs = db.search(vec(30), 3, { group: 20 }).map(([i]) => Math.floor(i / 2 ** 20));
    assert.strictEqual(docs[0], 3);
    assert.strictEqual(new Set(docs).size, 3);
    db.add(2n ** 60n + 5n, vec(500));
    assert.strictEqual(db.search(vec(500), 1)[0][0], 2n ** 60n + 5n);
    db.close();
  });

  await t('metrics: dot scores are the inner product, L2 scores the distance; an unknown metric refused', async () => {
    const dot = (a, b) => a.reduce((s, x, d) => s + x * b[d], 0);
    const dist = (a, b) => Math.sqrt(a.reduce((s, x, d) => s + (x - b[d]) ** 2, 0));
    const big = i => vec(i).map(x => x * (1 + i % 7));             // lengths that differ: cosine would rank otherwise
    for (const [metric, f, better] of [['dot', dot, (a, b) => a >= b], ['l2', dist, (a, b) => a <= b]]) {
      const db = await open(null, { dim: DIM, metric });
      for (let i = 1; i <= 50; i++) db.add(i, big(i));
      const q = big(1001), h = db.search(q, 5);
      h.forEach(([i, s], j) => {
        assert.ok(Math.abs(s - f(q, big(i))) < 1e-3 * Math.max(1, Math.abs(s)), `${metric} ${i}: ${s} vs ${f(q, big(i))}`);
        if (j) assert.ok(better(h[j - 1][1], s), metric + ' best first');
      });
      db.close();
    }
    await assert.rejects(open(null, { dim: DIM, metric: 'manhattan' }), /metric/);
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
    if (!native) return 'skip';                              // no native build to read it with
    const { execFileSync } = await import('node:child_process');
    const out = execFileSync(native, [join(dir, 't.vxi')]).toString().trim();
    assert.strictEqual(out, '200 chunk 20');
  });

  await t('stores the native builds wrote (tests/compat: 0.3.0 and 0.4.0) open here with their rows and texts', async () => {
    const compat = new URL('../../../tests/compat/', import.meta.url);
    if (!fs.existsSync(compat)) return 'skip';             // not inside the repository
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
  console.log(`js: OK (${n} tests${skipped ? ', ' + skipped + ' skipped' : ''}, ${rt})`);
} finally {
  fs.rmSync(dir, { recursive: true, force: true });
}
