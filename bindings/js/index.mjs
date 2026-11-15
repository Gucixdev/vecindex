// vecindex - a vector database in one C file, here as WebAssembly: the same vecindex.wasm in Node, Deno, Bun and
// browsers, no build at install, no dependencies.
//
//   import { open } from 'vecindex';
//   const db = await open('docs.vxi', { dim: 384, dir: './data' });   // Node: files in ./data; browser: memory
//   db.add(1, embedding, 'the chunk the embedding came from');
//   for (const [id, score] of db.search(query, 5)) console.log(id, score, db.text(id));
//   db.close();
//
// Ids are numbers (or BigInt past 2^53); namespaces are names or BigInt keys. Vectors are arrays or Float32Arrays.
// The C API underneath, and what each call promises: include/vecindex.h of vecindex.
import { MemoryFiles, NodeFiles, wasiImports } from './wasi.mjs';

export { MemoryFiles, NodeFiles };

const isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

async function wasmBytes(src) {
  if (src instanceof ArrayBuffer || ArrayBuffer.isView(src)) return src;
  const url = src || new URL('./vecindex.wasm', import.meta.url);
  if (isNode && (typeof url === 'string' ? !/^https?:/.test(url) : url.protocol === 'file:')) {
    const { readFileSync } = await import('node:fs');
    return readFileSync(url);
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error('vecindex: cannot fetch ' + url + ': ' + r.status);
  return r.arrayBuffer();
}

/** One instance of the library over one set of files (a directory in Node, memory elsewhere). */
export class Library {
  /** files: a MemoryFiles or NodeFiles; wasm: the module's bytes or URL (default: vecindex.wasm beside this file). */
  static async load({ files, wasm, env } = {}) {
    if (!files) {
      if (isNode) { const fs = await import('node:fs'); files = new NodeFiles(fs, process.cwd()); }
      else files = new MemoryFiles();
    }
    const w = wasiImports(files, env || {});
    const { instance } = await WebAssembly.instantiate(await wasmBytes(wasm), w.imports);
    w.bind(instance.exports.memory);
    if (instance.exports._initialize) instance.exports._initialize();
    return new Library(instance.exports, files);
  }

  constructor(x, files) { this.x = x; this.files = files; this.enc = new TextEncoder(); this.dec = new TextDecoder(); }
  get u8() { return new Uint8Array(this.x.memory.buffer); }
  cstr(s) {
    const b = this.enc.encode(s + '\0'), p = this.x.malloc(b.length);
    this.u8.set(b, p);
    return p;
  }
  bytes(b) { const p = this.x.malloc(b.length || 1); this.u8.set(b, p); return p; }
  read(p) { const m = this.u8; let e = p; while (m[e]) e++; return this.dec.decode(m.subarray(p, e)); }
  error(what) { return new Error(what + ': ' + (this.read(this.x.vi_error()) || 'failed')); }
  version() { return this.read(this.x.vi_version()); }
  /** A 64-bit key for a name (BigInt), the same in every language. */
  key(name) {
    const b = this.enc.encode(name), p = this.bytes(b);
    try { return this.x.vi_key(p, b.length); } finally { this.x.free(p); }
  }
  /** The store at path (relative to the files' root), made with dim dimensions if it is not there. */
  open(path, { dim = 0, metric = 'cos', manual = false } = {}) {
    const flags = ({ cos: 0, cosine: 0, dot: 4, l2: 8 })[metric];
    if (flags === undefined) throw new Error('metric is one of cos, dot, l2');
    const p = path == null ? 0 : this.cstr(path);
    const h = this.x.vi_open(p, dim, flags | (manual ? 2 : 0));
    if (p) this.x.free(p);
    if (!h) throw this.error('open ' + path);
    return new DB(this, h);
  }
}

const big = v => (typeof v === 'bigint' ? v : BigInt(v));
const id64 = v => (v >= -9007199254740991n && v <= 9007199254740991n ? Number(v) : v);

export class DB {
  constructor(lib, h) {
    this.lib = lib; this.x = lib.x; this.h = h;
    this.dim = this.x.vi_dim(h);
    this.q = this.x.malloc(this.dim * 4);          // scratch for one vector
    this.k = 0; this.ids = 0; this.sc = 0;
  }
  _h() { if (!this.h) throw new Error('the store is closed'); return this.h; }
  _ns(ns) { return typeof ns === 'string' ? this.lib.key(ns) : big(ns); }
  _vec(v, to = this.q) {
    if (v.length !== this.dim) throw new Error(`a vector of ${this.dim} floats was expected, not ${v.length}`);
    new Float32Array(this.x.memory.buffer, to, this.dim).set(v);
    return to;
  }
  _room(k) {
    if (k > this.k) {
      if (this.ids) { this.x.free(this.ids); this.x.free(this.sc); }
      this.ids = this.x.malloc(k * 8); this.sc = this.x.malloc(k * 4); this.k = k;
    }
  }
  _hits(n) {
    if (n < 0) throw this.lib.error('search');
    const ids = new BigInt64Array(this.x.memory.buffer, this.ids, n), sc = new Float32Array(this.x.memory.buffer, this.sc, n);
    const out = [];
    for (let i = 0; i < n; i++) out.push([id64(ids[i]), sc[i]]);
    return out;
  }
  _ok(r, what) { if (r !== 0) throw this.lib.error(what); }

  get length() { return Number(this.x.vi_count(this._h())); }

  /** Adds the row id (or replaces it), with an optional text kept beside it (string or bytes). */
  add(id, vec, text) {
    const q = this._vec(vec);
    if (text == null) return this._ok(this.x.vi_add(this._h(), big(id), q), 'add ' + id);
    const b = typeof text === 'string' ? this.lib.enc.encode(text) : text, p = this.lib.bytes(b);
    try { this._ok(this.x.vi_add_text(this._h(), big(id), q, p, b.length), 'add ' + id); } finally { this.x.free(p); }
  }
  /** Many rows: ids, and vecs as an array of vectors or one flat Float32Array. */
  addMany(ids, vecs) {
    const n = ids.length, flat = vecs instanceof Float32Array ? vecs : Float32Array.from(vecs.flatMap(v => Array.from(v)));
    if (flat.length !== n * this.dim) throw new Error(`${n} ids need ${n * this.dim} floats, not ${flat.length}`);
    const pi = this.x.malloc(n * 8), pv = this.x.malloc(flat.length * 4);
    try {
      new BigInt64Array(this.x.memory.buffer, pi, n).set(ids.map(big));
      new Float32Array(this.x.memory.buffer, pv, flat.length).set(flat);
      const r = this.x.vi_add_batch(this._h(), n, pi, pv);
      if (r !== n) throw this.lib.error(`addMany: ${Math.max(r, 0)} of ${n} went in`);
    } finally { this.x.free(pi); this.x.free(pv); }
  }
  /** A new vector for the row id; its text, tag, number and namespace stay. */
  update(id, vec) { this._ok(this.x.vi_update(this._h(), big(id), this._vec(vec)), 'update ' + id); }
  /** The row's vector (a Float32Array), or null. */
  get(id) {
    if (this.x.vi_get(this._h(), big(id), this.q) !== 0) return null;
    return new Float32Array(this.x.memory.buffer, this.q, this.dim).slice();
  }
  /** The row's text (a string, or bytes with raw), null when it has none. */
  text(id, raw = false) {
    let cap = 4096, p = this.x.malloc(cap);
    try {
      let n = this.x.vi_get_text(this._h(), big(id), p, cap);
      if (n > cap) { this.x.free(p); cap = n; p = this.x.malloc(cap); n = this.x.vi_get_text(this._h(), big(id), p, cap); }
      if (n <= 0) return null;
      const b = new Uint8Array(this.x.memory.buffer, p, n).slice();
      return raw ? b : this.lib.dec.decode(b);
    } finally { this.x.free(p); }
  }
  has(id) { return this.x.vi_get(this._h(), big(id), this.q) === 0; }
  /** Deletes the row id; true when there was one. */
  delete(id) { return this.x.vi_delete(this._h(), big(id)) === 1; }
  /** Deletes every row with an id in [lo, hi]; how many. */
  deleteRange(lo, hi) {
    const n = this.x.vi_delete_range(this._h(), big(lo), big(hi));
    if (n < 0) throw this.lib.error('deleteRange');
    return n;
  }

  _get64(fn, id) {
    const p = this.x.malloc(8);
    try { return fn(this._h(), big(id), p) === 0 ? new BigInt64Array(this.x.memory.buffer, p, 1)[0] : null; } finally { this.x.free(p); }
  }
  setTag(id, tag) { this._ok(this.x.vi_set_tag(this._h(), big(id), big(tag)), 'setTag ' + id); }
  tag(id) { return this._get64(this.x.vi_get_tag, id); }
  setNum(id, num) { this._ok(this.x.vi_set_num(this._h(), big(id), big(num)), 'setNum ' + id); }
  num(id) { const v = this._get64(this.x.vi_get_num, id); return v === null ? null : id64(v); }
  /** Puts the row in a namespace: a name, or a BigInt key. */
  setNs(id, ns) { this._ok(this.x.vi_set_ns(this._h(), big(id), this._ns(ns)), 'setNs ' + id); }
  ns(id) { return this._get64(this.x.vi_get_ns, id); }
  countNs(ns) { return Number(this.x.vi_count_ns(this._h(), this._ns(ns))); }
  deleteNs(ns) { return Number(this.x.vi_delete_ns(this._h(), this._ns(ns))); }

  _str(s) { if (s == null) return 0; return this.lib.cstr(Array.isArray(s) ? s.join(',') : s); }
  label(name, parent) {
    const a = this._str(name), b = this._str(parent);
    try { this._ok(this.x.vi_label(this._h(), a, b), 'label ' + name); } finally { this.x.free(a); if (b) this.x.free(b); }
  }
  addLabels(id, labels) {
    const a = this._str(labels);
    try { this._ok(this.x.vi_label_add(this._h(), big(id), a), 'addLabels ' + id); } finally { this.x.free(a); }
  }
  removeLabels(id, labels) {
    const a = this._str(labels);
    try { this._ok(this.x.vi_label_remove(this._h(), big(id), a), 'removeLabels ' + id); } finally { this.x.free(a); }
  }

  /**
   * The k nearest rows as [[id, score]], best first. One condition at a time in opts:
   * text (words too; vec may be null for words alone), ns (a namespace or an array of them), num ([lo, hi]),
   * labels ({all, any, none}), group (a shift: one hit per id >> shift).
   */
  search(vec, k = 10, opts = {}) {
    const { text, ns, num, labels, group } = opts;
    if ([text, ns, num, labels, group].filter(v => v != null).length > 1) throw new Error('search takes one condition at a time');
    if (k <= 0) return [];
    if (vec == null && text == null) throw new Error('a query vector, or text');
    this._room(k);
    const db = this._h(), q = vec == null ? 0 : this._vec(vec), ids = this.ids, sc = this.sc;
    let n;
    if (text != null) {
      const b = typeof text === 'string' ? this.lib.enc.encode(text) : text, p = this.lib.bytes(b);
      try { n = this.x.vi_search_hybrid(db, q, p, b.length, k, ids, sc); } finally { this.x.free(p); }
    } else if (ns != null) {
      if (Array.isArray(ns)) {
        const p = this.x.malloc(ns.length * 8);
        try {
          new BigInt64Array(this.x.memory.buffer, p, ns.length).set(ns.map(v => this._ns(v)));
          n = this.x.vi_search_in(db, p, ns.length, q, k, ids, sc);
        } finally { this.x.free(p); }
      } else n = this.x.vi_search_ns(db, this._ns(ns), q, k, ids, sc);
    } else if (num != null) {
      n = this.x.vi_search_range(db, q, k, big(num[0]), big(num[1]), ids, sc);
    } else if (labels != null) {
      const a = this._str(labels.all), b = this._str(labels.any), c = this._str(labels.none);
      try { n = this.x.vi_search_labeled(db, q, k, a, b, c, ids, sc); } finally { [a, b, c].forEach(p => p && this.x.free(p)); }
    } else if (group != null) {
      n = this.x.vi_search_grouped(db, q, k, group, ids, sc);
    } else n = this.x.vi_search(db, q, k, ids, sc);
    return this._hits(n);
  }

  /**
   * The k rows with the highest exact score - the same as scoring every row - as {hits, checked}: the fast scan's
   * error bound (proved in lean4/Vecindex/Exact.lean) rules the other rows out; checked rows were scored exactly.
   */
  searchExact(vec, k = 10) {
    if (k <= 0) return { hits: [], checked: 0 };
    this._room(k);
    const db = this._h(), q = this._vec(vec), c = this.x.malloc(8);
    try {
      const n = this.x.vi_search_exact(db, q, k, this.ids, this.sc, c);
      const hits = this._hits(n);
      return { hits, checked: Number(new BigInt64Array(this.x.memory.buffer, c, 1)[0]) };
    } finally { this.x.free(c); }
  }

  /** Indexes what was added (rows are indexed as they are searched by default; this does it now). */
  build() { this.x.vi_build(this._h()); }
  /** Writes the whole store (to path, or its own file) and starts a fresh journal. */
  save(path) {
    const p = path == null ? 0 : this.lib.cstr(path);
    try { this._ok(this.x.vi_save(this._h(), p), 'save'); } finally { if (p) this.x.free(p); }
  }
  /** Flushes the journal: everything before it survives a crash (with Node files). */
  sync() { this._ok(this.x.vi_sync(this._h()), 'sync'); }
  /** Flushes the journal and frees the store. */
  close() {
    if (!this.h) return;
    this.x.vi_close(this.h);
    this.x.free(this.q);
    if (this.ids) { this.x.free(this.ids); this.x.free(this.sc); }
    this.h = 0;
  }
}

const libs = new Map();

/**
 * open(path, {dim, metric, dir, files, wasm}): a store. In Node the files are in dir (default: the working
 * directory); elsewhere in memory, or in the files given (a MemoryFiles to export them from).
 */
export async function open(path, opts = {}) {
  let lib;
  if (opts.files || opts.wasm || opts.env) lib = await Library.load(opts);
  else {
    const dir = opts.dir || (isNode ? process.cwd() : ':memory:');
    if (!libs.has(dir)) {
      const files = isNode && dir !== ':memory:' ? new NodeFiles(await import('node:fs'), dir) : new MemoryFiles();
      libs.set(dir, Library.load({ files }));
    }
    lib = await libs.get(dir);
  }
  return lib.open(path, opts);
}

/** The library's version. */
export async function version() { return (await Library.load({ files: new MemoryFiles() })).version(); }
