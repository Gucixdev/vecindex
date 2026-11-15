// wasi.mjs - the few WASI preview1 calls vecindex.wasm makes, implemented here so it runs the same in Node, Deno, Bun
// and browsers, whatever their own WASI support (or lack of it). Files live in a backend: memory (a Map; export and
// import them as bytes, e.g. into IndexedDB) or, in Node, a real directory through node:fs.
//
// Only what the C library uses: files and directories under one root, clocks, an empty or given environment. No
// sockets, no processes, no random.

const OK = 0, EBADF = 8, EEXIST = 20, EINVAL = 28, EIO = 29, EISDIR = 31, ENOENT = 44, ENOSYS = 52, ENOTDIR = 54,
  ENOTEMPTY = 55;
const FT_DIR = 3, FT_REG = 4;
const O_CREAT = 1, O_DIRECTORY = 2, O_EXCL = 4, O_TRUNC = 8;
const FD_APPEND = 1;

class Exit extends Error {
  constructor(code) { super('vecindex.wasm exited with ' + code); this.code = code; }
}

function norm(p) {
  const out = [];
  for (const part of p.split('/')) {
    if (part === '' || part === '.') continue;
    if (part === '..') { if (!out.length) return null; out.pop(); } else out.push(part);
  }
  return out.join('/');
}

/** Files in memory: path -> Uint8Array (grown as written), directories as a set. */
export class MemoryFiles {
  constructor() { this.files = new Map(); this.dirs = new Set(['']); }
  _parent(p) { const i = p.lastIndexOf('/'); return i < 0 ? '' : p.slice(0, i); }
  stat(p) {
    if (this.dirs.has(p)) return { type: FT_DIR, size: 0 };
    const f = this.files.get(p);
    return f ? { type: FT_REG, size: f.size } : null;
  }
  open(p, create, excl, trunc) {
    let f = this.files.get(p);
    if (this.dirs.has(p)) return { err: EISDIR };
    if (f && excl && create) return { err: EEXIST };
    if (!f) {
      if (!create) return { err: ENOENT };
      if (!this.dirs.has(this._parent(p))) return { err: ENOENT };
      f = { data: new Uint8Array(0), size: 0 };
      this.files.set(p, f);
    }
    if (trunc) f.size = 0;
    return { h: f };
  }
  size(h) { return h.size; }
  read(h, buf, pos) {
    const n = Math.max(0, Math.min(buf.length, h.size - pos));
    buf.set(h.data.subarray(pos, pos + n));
    return n;
  }
  write(h, buf, pos) {
    const end = pos + buf.length;
    if (end > h.data.length) {
      const g = new Uint8Array(Math.max(end, h.data.length * 2, 4096));
      g.set(h.data.subarray(0, h.size));
      h.data = g;
    }
    if (pos > h.size) h.data.fill(0, h.size, pos);
    h.data.set(buf, pos);
    if (end > h.size) h.size = end;
    return buf.length;
  }
  truncate(h, n) {
    if (n > h.data.length) { const g = new Uint8Array(n); g.set(h.data.subarray(0, h.size)); h.data = g; }
    if (n > h.size) h.data.fill(0, h.size, n);
    h.size = n;
  }
  sync() {}
  close() {}
  unlink(p) { if (!this.files.delete(p)) return this.dirs.has(p) ? EISDIR : ENOENT; return OK; }
  rmdir(p) {
    if (!this.dirs.has(p)) return ENOENT;
    for (const k of this.files.keys()) if (k.startsWith(p + '/')) return ENOTEMPTY;
    this.dirs.delete(p);
    return OK;
  }
  mkdir(p) { if (this.dirs.has(p) || this.files.has(p)) return EEXIST; this.dirs.add(p); return OK; }
  rename(a, b) {
    const f = this.files.get(a);
    if (!f) return ENOENT;
    this.files.delete(a); this.files.set(b, f);
    return OK;
  }
  list(p) {
    const pre = p ? p + '/' : '', names = new Set();
    for (const k of [...this.files.keys(), ...this.dirs]) {
      if (k && k.startsWith(pre)) { const rest = k.slice(pre.length); if (rest && !rest.includes('/')) names.add(rest); }
    }
    return [...names].map(n => ({ name: n, type: this.dirs.has(pre + n) ? FT_DIR : FT_REG }));
  }
  /** The bytes of a file (a copy), or null. */
  exportFile(p) { const f = this.files.get(norm(p)); return f ? f.data.slice(0, f.size) : null; }
  /** Puts bytes in as a file (a store and its journal, saved earlier with exportFile). */
  importFile(p, bytes) {
    const q = norm(p), d = new Uint8Array(bytes);
    let dir = this._parent(q);
    while (dir && !this.dirs.has(dir)) { this.dirs.add(dir); dir = this._parent(dir); }
    this.files.set(q, { data: d, size: d.length });
  }
}

/** Files in a real directory, through node:fs (pass the module: `import * as fs from 'node:fs'`). */
export class NodeFiles {
  constructor(fs, root) { this.fs = fs; this.root = root.replace(/\/+$/, '') || '/'; }
  _p(p) { return p ? this.root + '/' + p : this.root; }
  _err(e) {
    switch (e && e.code) {
      case 'ENOENT': return ENOENT; case 'EEXIST': return EEXIST; case 'EISDIR': return EISDIR;
      case 'ENOTDIR': return ENOTDIR; case 'ENOTEMPTY': return ENOTEMPTY; case 'EINVAL': return EINVAL;
      default: return EIO;
    }
  }
  stat(p) {
    try { const s = this.fs.statSync(this._p(p)); return { type: s.isDirectory() ? FT_DIR : FT_REG, size: s.size }; }
    catch (e) { return null; }
  }
  open(p, create, excl, trunc) {
    try {
      const s = this.stat(p);
      if (s && s.type === FT_DIR) return { err: EISDIR };
      const flags = (create ? (excl ? 'wx+' : (s ? 'r+' : 'w+')) : 'r+');
      let fd;
      try { fd = this.fs.openSync(this._p(p), flags); }
      catch (e) { if (!create && e.code === 'EACCES') fd = this.fs.openSync(this._p(p), 'r'); else throw e; }
      if (trunc) this.fs.ftruncateSync(fd, 0);
      return { h: fd };
    } catch (e) { return { err: this._err(e) }; }
  }
  size(h) { return this.fs.fstatSync(h).size; }
  read(h, buf, pos) { return this.fs.readSync(h, buf, 0, buf.length, pos); }
  write(h, buf, pos) { return this.fs.writeSync(h, buf, 0, buf.length, pos); }
  truncate(h, n) { this.fs.ftruncateSync(h, n); }
  sync(h) { this.fs.fsyncSync(h); }
  close(h) { this.fs.closeSync(h); }
  unlink(p) { try { this.fs.unlinkSync(this._p(p)); return OK; } catch (e) { return this._err(e); } }
  rmdir(p) { try { this.fs.rmdirSync(this._p(p)); return OK; } catch (e) { return this._err(e); } }
  mkdir(p) { try { this.fs.mkdirSync(this._p(p)); return OK; } catch (e) { return this._err(e); } }
  rename(a, b) { try { this.fs.renameSync(this._p(a), this._p(b)); return OK; } catch (e) { return this._err(e); } }
  list(p) {
    return this.fs.readdirSync(this._p(p), { withFileTypes: true })
      .map(d => ({ name: d.name, type: d.isDirectory() ? FT_DIR : FT_REG }));
  }
}

/** The import object for vecindex.wasm over a file backend; env: {NAME: value} the C library sees (VECINDEX_*). */
export function wasiImports(files, env = {}) {
  let mem = null;
  const fds = new Map();                         // fd -> {path, h, pos, append, dir}
  fds.set(3, { path: '', dir: true });           // the one preopened directory: the root
  let nextFd = 4;
  const enc = new TextEncoder(), dec = new TextDecoder();
  const dv = () => new DataView(mem.buffer);
  const u8 = () => new Uint8Array(mem.buffer);
  const str = (p, n) => dec.decode(u8().subarray(p, p + n));
  const envs = Object.entries(env).map(([k, v]) => enc.encode(k + '=' + v + '\0'));
  const at = (dirfd, p, n) => {
    const d = fds.get(dirfd);
    if (!d || !d.dir) return null;
    const q = norm((d.path ? d.path + '/' : '') + str(p, n));
    return q;
  };
  const iovs = (p, n) => {
    const v = dv(), out = [];
    for (let i = 0; i < n; i++) out.push([v.getUint32(p + i * 8, true), v.getUint32(p + i * 8 + 4, true)]);
    return out;
  };
  const filestat = (buf, st) => {
    const v = dv();
    for (let i = 0; i < 64; i += 8) v.setBigUint64(buf + i, 0n, true);
    v.setUint8(buf + 16, st.type);
    v.setBigUint64(buf + 24, 1n, true);
    v.setBigUint64(buf + 32, BigInt(st.size), true);
  };
  const guard = f => (...a) => {
    try { return f(...a); } catch (e) { if (e instanceof Exit) throw e; return EIO; }
  };
  const w = {
    environ_sizes_get: (cnt, size) => {
      dv().setUint32(cnt, envs.length, true);
      dv().setUint32(size, envs.reduce((s, e) => s + e.length, 0), true);
      return OK;
    },
    environ_get: (ptrs, buf) => {
      let o = buf;
      envs.forEach((e, i) => { dv().setUint32(ptrs + i * 4, o, true); u8().set(e, o); o += e.length; });
      return OK;
    },
    clock_time_get: (id, prec, out) => {
      const ns = id === 0 ? BigInt(Date.now()) * 1000000n
        : BigInt(Math.round((typeof performance !== 'undefined' ? performance.now() : Date.now()) * 1e6));
      dv().setBigUint64(out, ns, true);
      return OK;
    },
    fd_prestat_get: (fd, buf) => {
      if (fd !== 3) return EBADF;
      dv().setUint8(buf, 0); dv().setUint32(buf + 4, 1, true);
      return OK;
    },
    fd_prestat_dir_name: (fd, p, n) => { if (fd !== 3) return EBADF; u8()[p] = 47; return OK; },
    fd_fdstat_get: (fd, buf) => {
      const f = fds.get(fd);
      if (!f && fd > 2) return EBADF;
      const v = dv();
      v.setUint8(buf, f ? (f.dir ? FT_DIR : FT_REG) : 2);
      v.setUint16(buf + 2, f && f.append ? FD_APPEND : 0, true);
      v.setBigUint64(buf + 8, 0xffffffffffffffffn, true);
      v.setBigUint64(buf + 16, 0xffffffffffffffffn, true);
      return OK;
    },
    fd_fdstat_set_flags: (fd, flags) => { const f = fds.get(fd); if (!f) return EBADF; f.append = !!(flags & FD_APPEND); return OK; },
    path_open: guard((dirfd, dirflags, p, n, oflags, rb, ri, fdflags, out) => {
      const q = at(dirfd, p, n);
      if (q === null) return EBADF;
      const st = files.stat(q);
      if (oflags & O_DIRECTORY || (st && st.type === FT_DIR)) {
        if (!st) return ENOENT;
        if (st.type !== FT_DIR) return ENOTDIR;
        fds.set(nextFd, { path: q, dir: true });
      } else {
        const r = files.open(q, !!(oflags & O_CREAT), !!(oflags & O_EXCL), !!(oflags & O_TRUNC));
        if (r.err) return r.err;
        fds.set(nextFd, { path: q, h: r.h, pos: 0, append: !!(fdflags & FD_APPEND) });
      }
      dv().setUint32(out, nextFd++, true);
      return OK;
    }),
    fd_close: guard(fd => { const f = fds.get(fd); if (!f || fd === 3) return f ? OK : EBADF; if (!f.dir) files.close(f.h); fds.delete(fd); return OK; }),
    fd_read: guard((fd, iv, n, out) => {
      const f = fds.get(fd);
      if (!f || f.dir) return EBADF;
      let tot = 0;
      for (const [p, l] of iovs(iv, n)) {
        const got = files.read(f.h, u8().subarray(p, p + l), f.pos);
        f.pos += got; tot += got;
        if (got < l) break;
      }
      dv().setUint32(out, tot, true);
      return OK;
    }),
    fd_pread: guard((fd, iv, n, off, out) => {
      const f = fds.get(fd);
      if (!f || f.dir) return EBADF;
      let tot = 0, pos = Number(off);
      for (const [p, l] of iovs(iv, n)) {
        const got = files.read(f.h, u8().subarray(p, p + l), pos);
        pos += got; tot += got;
        if (got < l) break;
      }
      dv().setUint32(out, tot, true);
      return OK;
    }),
    fd_write: guard((fd, iv, n, out) => {
      let tot = 0;
      if (fd === 1 || fd === 2) {
        for (const [p, l] of iovs(iv, n)) { const s = str(p, l); (fd === 1 ? console.log : console.error)(s.replace(/\n$/, '')); tot += l; }
        dv().setUint32(out, tot, true);
        return OK;
      }
      const f = fds.get(fd);
      if (!f || f.dir) return EBADF;
      if (f.append) f.pos = files.size(f.h);
      for (const [p, l] of iovs(iv, n)) { const k = files.write(f.h, u8().slice(p, p + l), f.pos); f.pos += k; tot += k; }
      dv().setUint32(out, tot, true);
      return OK;
    }),
    fd_seek: guard((fd, off, whence, out) => {
      const f = fds.get(fd);
      if (!f || f.dir) return EBADF;
      const base = whence === 0 ? 0 : whence === 1 ? f.pos : files.size(f.h);
      const np = base + Number(off);
      if (np < 0) return EINVAL;
      f.pos = np;
      dv().setBigUint64(out, BigInt(np), true);
      return OK;
    }),
    fd_tell: guard((fd, out) => { const f = fds.get(fd); if (!f) return EBADF; dv().setBigUint64(out, BigInt(f.pos), true); return OK; }),
    fd_filestat_get: guard((fd, buf) => {
      const f = fds.get(fd);
      if (!f) return EBADF;
      filestat(buf, f.dir ? { type: FT_DIR, size: 0 } : { type: FT_REG, size: files.size(f.h) });
      return OK;
    }),
    fd_filestat_set_size: guard((fd, size) => { const f = fds.get(fd); if (!f || f.dir) return EBADF; files.truncate(f.h, Number(size)); return OK; }),
    path_filestat_get: guard((dirfd, flags, p, n, buf) => {
      const q = at(dirfd, p, n);
      if (q === null) return EBADF;
      const st = files.stat(q);
      if (!st) return ENOENT;
      filestat(buf, st);
      return OK;
    }),
    fd_sync: guard(fd => { const f = fds.get(fd); if (f && !f.dir) files.sync(f.h); return OK; }),
    fd_datasync: guard(fd => { const f = fds.get(fd); if (f && !f.dir) files.sync(f.h); return OK; }),
    fd_advise: () => OK,
    fd_allocate: guard((fd, off, len) => {
      const f = fds.get(fd);
      if (!f || f.dir) return EBADF;
      const end = Number(off) + Number(len);
      if (end > files.size(f.h)) files.truncate(f.h, end);
      return OK;
    }),
    fd_readdir: guard((fd, buf, len, cookie, out) => {
      const f = fds.get(fd);
      if (!f || !f.dir) return EBADF;
      const ents = [{ name: '.', type: FT_DIR }, { name: '..', type: FT_DIR }, ...files.list(f.path)];
      let o = 0;
      for (let i = Number(cookie); i < ents.length && o < len; i++) {
        const name = enc.encode(ents[i].name), head = new DataView(new ArrayBuffer(24));
        head.setBigUint64(0, BigInt(i + 1), true);
        head.setBigUint64(8, BigInt(i + 1), true);
        head.setUint32(16, name.length, true);
        head.setUint8(20, ents[i].type);
        const rec = new Uint8Array(24 + name.length);
        rec.set(new Uint8Array(head.buffer)); rec.set(name, 24);
        const take = Math.min(rec.length, len - o);
        u8().set(rec.subarray(0, take), buf + o);
        o += take;
      }
      dv().setUint32(out, o, true);
      return OK;
    }),
    path_create_directory: guard((dirfd, p, n) => { const q = at(dirfd, p, n); return q === null ? EBADF : files.mkdir(q); }),
    path_remove_directory: guard((dirfd, p, n) => { const q = at(dirfd, p, n); return q === null ? EBADF : files.rmdir(q); }),
    path_unlink_file: guard((dirfd, p, n) => { const q = at(dirfd, p, n); return q === null ? EBADF : files.unlink(q); }),
    path_rename: guard((d1, p1, n1, d2, p2, n2) => {
      const a = at(d1, p1, n1), b = at(d2, p2, n2);
      return a === null || b === null ? EBADF : files.rename(a, b);
    }),
    poll_oneoff: (inp, out, nsubs, nout) => {    // only clock subscriptions come here (a sleep): they are all due now
      const v = dv();
      for (let i = 0; i < nsubs; i++) {
        v.setBigUint64(out + i * 32, v.getBigUint64(inp + i * 48, true), true);
        v.setUint16(out + i * 32 + 8, 0, true);
        v.setUint8(out + i * 32 + 10, v.getUint8(inp + i * 48 + 8));
      }
      v.setUint32(nout, nsubs, true);
      return OK;
    },
    proc_exit: code => { throw new Exit(code); },
    sched_yield: () => OK,
    random_get: (p, n) => {
      const b = u8().subarray(p, p + n);
      if (typeof crypto !== 'undefined' && crypto.getRandomValues) for (let i = 0; i < n; i += 65536) crypto.getRandomValues(b.subarray(i, Math.min(n, i + 65536)));
      else return ENOSYS;
      return OK;
    },
  };
  return {
    imports: { wasi_snapshot_preview1: w },
    bind(memory) { mem = memory; },
  };
}
