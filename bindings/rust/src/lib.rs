//! vecindex - a vector database in one C file, for RAG inside an application.
//!
//! ```no_run
//! let db = vecindex::Db::open("docs.vxi", 4).unwrap();
//! db.add_text(1, &[0.1, 0.2, 0.3, 0.4], "the chunk the embedding came from").unwrap();
//! for (id, score) in db.search(&[0.1, 0.2, 0.3, 0.4], 5).unwrap() {
//!     println!("{} {} {:?}", id, score, db.text(id).unwrap());
//! }
//! ```
//!
//! A store is a file and its journal: every change is in the journal at once ([`Db::sync`] makes it durable), and
//! damaged blocks of the file are rebuilt from its parity when it is opened. The C library (vecindex's
//! `include/vecindex.h`) is compiled into the crate by `build.rs`; there are no Rust dependencies.
//!
//! Every method may be called from many threads at once on one `Db` (the C library takes its own locks), so `Db` is
//! `Send` and `Sync`.
use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;

/// The raw C API, as `include/vecindex.h` declares it.
pub mod ffi {
    use std::os::raw::{c_char, c_float, c_int, c_void};
    pub type ViDb = c_void;
    extern "C" {
        pub fn vi_version() -> *const c_char;
        pub fn vi_error() -> *const c_char;
        pub fn vi_open(path: *const c_char, dim: c_int, flags: c_int) -> *mut ViDb;
        pub fn vi_close(db: *mut ViDb);
        pub fn vi_dim(db: *const ViDb) -> c_int;
        pub fn vi_count(db: *mut ViDb) -> i64;
        pub fn vi_add(db: *mut ViDb, id: i64, vec: *const c_float) -> c_int;
        pub fn vi_update(db: *mut ViDb, id: i64, vec: *const c_float) -> c_int;
        pub fn vi_add_batch(db: *mut ViDb, n: c_int, ids: *const i64, vecs: *const c_float) -> c_int;
        pub fn vi_add_text(db: *mut ViDb, id: i64, vec: *const c_float, text: *const c_char, len: c_int) -> c_int;
        pub fn vi_get(db: *mut ViDb, id: i64, out: *mut c_float) -> c_int;
        pub fn vi_get_text(db: *mut ViDb, id: i64, buf: *mut c_char, cap: c_int) -> c_int;
        pub fn vi_delete(db: *mut ViDb, id: i64) -> c_int;
        pub fn vi_delete_range(db: *mut ViDb, lo: i64, hi: i64) -> c_int;
        pub fn vi_key(s: *const c_char, len: c_int) -> i64;
        pub fn vi_label(db: *mut ViDb, name: *const c_char, parent: *const c_char) -> c_int;
        pub fn vi_label_add(db: *mut ViDb, id: i64, labels: *const c_char) -> c_int;
        pub fn vi_label_remove(db: *mut ViDb, id: i64, labels: *const c_char) -> c_int;
        pub fn vi_set_tag(db: *mut ViDb, id: i64, tag: i64) -> c_int;
        pub fn vi_get_tag(db: *mut ViDb, id: i64, tag: *mut i64) -> c_int;
        pub fn vi_set_num(db: *mut ViDb, id: i64, num: i64) -> c_int;
        pub fn vi_get_num(db: *mut ViDb, id: i64, num: *mut i64) -> c_int;
        pub fn vi_set_ns(db: *mut ViDb, id: i64, ns: i64) -> c_int;
        pub fn vi_get_ns(db: *mut ViDb, id: i64, ns: *mut i64) -> c_int;
        pub fn vi_count_ns(db: *mut ViDb, ns: i64) -> i64;
        pub fn vi_delete_ns(db: *mut ViDb, ns: i64) -> i64;
        pub fn vi_search(db: *mut ViDb, q: *const c_float, k: c_int, ids: *mut i64, sc: *mut c_float) -> c_int;
        pub fn vi_search_ns(db: *mut ViDb, ns: i64, q: *const c_float, k: c_int, ids: *mut i64, sc: *mut c_float) -> c_int;
        pub fn vi_search_in(db: *mut ViDb, ns: *const i64, n: c_int, q: *const c_float, k: c_int, ids: *mut i64, sc: *mut c_float) -> c_int;
        pub fn vi_search_range(db: *mut ViDb, q: *const c_float, k: c_int, lo: i64, hi: i64, ids: *mut i64, sc: *mut c_float) -> c_int;
        pub fn vi_search_labeled(db: *mut ViDb, q: *const c_float, k: c_int, all: *const c_char, any: *const c_char,
                                 none: *const c_char, ids: *mut i64, sc: *mut c_float) -> c_int;
        pub fn vi_search_hybrid(db: *mut ViDb, q: *const c_float, text: *const c_char, tlen: c_int, k: c_int,
                                ids: *mut i64, sc: *mut c_float) -> c_int;
        pub fn vi_search_grouped(db: *mut ViDb, q: *const c_float, k: c_int, shift: c_int, ids: *mut i64, sc: *mut c_float) -> c_int;
        pub fn vi_search_exact(db: *mut ViDb, q: *const c_float, k: c_int, ids: *mut i64, sc: *mut c_float, checked: *mut i64) -> c_int;
        pub fn vi_build(db: *mut ViDb);
        pub fn vi_save(db: *mut ViDb, path: *const c_char) -> c_int;
        pub fn vi_sync(db: *mut ViDb) -> c_int;
    }
}

/// A call the C library refused, with its reason (`vi_error()`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn last(what: &str) -> Error {
    let m = unsafe { ffi::vi_error() };
    let m = if m.is_null() { String::new() } else { unsafe { CStr::from_ptr(m) }.to_string_lossy().into_owned() };
    Error(format!("{}: {}", what, if m.is_empty() { "failed" } else { &m }))
}

/// The C library's version.
pub fn version() -> &'static str {
    unsafe { CStr::from_ptr(ffi::vi_version()) }.to_str().unwrap_or("?")
}

/// A 64-bit key for a name - a namespace, or an id made from a string - the same in every language.
pub fn key(name: &str) -> i64 {
    unsafe { ffi::vi_key(name.as_ptr() as *const c_char, name.len() as c_int) }
}

/// How rows are compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    Cosine,
    Dot,
    L2,
}

/// One condition on a search (one at a time).
#[derive(Debug, Clone)]
pub enum Filter<'a> {
    /// The rows of one namespace.
    Ns(i64),
    /// The rows of any of these namespaces.
    AnyNs(&'a [i64]),
    /// The rows whose number is in [lo, hi].
    Num(i64, i64),
    /// Labels: every one of `all`, at least one of `any`, none of `none` (each "a,b"; "" for no condition).
    Labels { all: &'a str, any: &'a str, none: &'a str },
    /// Words too: BM25 over the rows' texts fused with the vector.
    Text(&'a str),
    /// One hit a group, the group of an id being id >> shift (ids made as doc << 20 | chunk: shift 20).
    Group(u32),
}

/// One store: a file and its journal, or memory only.
pub struct Db {
    db: *mut ffi::ViDb,
    dim: usize,
}

unsafe impl Send for Db {}
unsafe impl Sync for Db {}

fn cpath(p: &Path) -> Result<CString> {
    CString::new(p.to_string_lossy().as_bytes()).map_err(|_| Error("a path with a 0 byte".into()))
}

fn cstr(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| Error("a string with a 0 byte".into()))
}

impl Db {
    /// The store in `path`, made with `dim` dimensions when it is not there yet (`dim` 0: it must exist).
    pub fn open<P: AsRef<Path>>(path: P, dim: usize) -> Result<Db> {
        Db::open_with(Some(path.as_ref()), dim, Metric::Cosine)
    }

    /// A store in memory only.
    pub fn memory(dim: usize) -> Result<Db> {
        Db::open_with(None, dim, Metric::Cosine)
    }

    /// `open` with the metric of a new store (an existing one keeps its own).
    pub fn open_with(path: Option<&Path>, dim: usize, metric: Metric) -> Result<Db> {
        let flags = match metric { Metric::Cosine => 0, Metric::Dot => 4, Metric::L2 => 8 };
        let p = match path { Some(p) => Some(cpath(p)?), None => None };
        let db = unsafe { ffi::vi_open(p.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()), dim as c_int, flags) };
        if db.is_null() {
            return Err(last(&format!("open {}", path.map_or("(memory)".into(), |p| p.display().to_string()))));
        }
        let dim = unsafe { ffi::vi_dim(db) } as usize;
        Ok(Db { db, dim })
    }

    /// Dimensions of the vectors.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Live rows.
    pub fn len(&self) -> usize {
        unsafe { ffi::vi_count(self.db) as usize }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn check(&self, v: &[f32]) -> Result<()> {
        if v.len() != self.dim {
            return Err(Error(format!("a vector of {} floats was expected, not {}", self.dim, v.len())));
        }
        Ok(())
    }

    fn ok(r: c_int, what: &str) -> Result<()> {
        if r == 0 { Ok(()) } else { Err(last(what)) }
    }

    /// Adds the row `id`, or replaces it.
    pub fn add(&self, id: i64, vec: &[f32]) -> Result<()> {
        self.check(vec)?;
        Db::ok(unsafe { ffi::vi_add(self.db, id, vec.as_ptr()) }, "add")
    }

    /// `add` with a text kept beside the row (a chunk of a document): it comes back with [`Db::text`] and feeds
    /// the words side of a [`Filter::Text`] search.
    pub fn add_text<T: AsRef<[u8]>>(&self, id: i64, vec: &[f32], text: T) -> Result<()> {
        self.check(vec)?;
        let t = text.as_ref();
        Db::ok(unsafe { ffi::vi_add_text(self.db, id, vec.as_ptr(), t.as_ptr() as *const c_char, t.len() as c_int) }, "add_text")
    }

    /// Many rows at once: `vecs` is `ids.len()` vectors one after another.
    pub fn add_many(&self, ids: &[i64], vecs: &[f32]) -> Result<()> {
        if vecs.len() != ids.len() * self.dim {
            return Err(Error(format!("{} ids need {} floats, not {}", ids.len(), ids.len() * self.dim, vecs.len())));
        }
        let n = unsafe { ffi::vi_add_batch(self.db, ids.len() as c_int, ids.as_ptr(), vecs.as_ptr()) };
        if n as usize == ids.len() { Ok(()) } else { Err(last("add_many")) }
    }

    /// A new vector for the row `id`; its text, tag, number and namespace stay.
    pub fn update(&self, id: i64, vec: &[f32]) -> Result<()> {
        self.check(vec)?;
        Db::ok(unsafe { ffi::vi_update(self.db, id, vec.as_ptr()) }, "update")
    }

    /// The row's vector, `None` when there is no such row.
    pub fn get(&self, id: i64) -> Option<Vec<f32>> {
        let mut v = vec![0.0f32; self.dim];
        if unsafe { ffi::vi_get(self.db, id, v.as_mut_ptr()) } == 0 { Some(v) } else { None }
    }

    /// The row's text as bytes, `None` when it has none or there is no row.
    pub fn text_bytes(&self, id: i64) -> Option<Vec<u8>> {
        let mut buf = vec![0u8; 4096];
        let mut n = unsafe { ffi::vi_get_text(self.db, id, buf.as_mut_ptr() as *mut c_char, buf.len() as c_int) };
        if n > buf.len() as c_int {
            buf = vec![0u8; n as usize];
            n = unsafe { ffi::vi_get_text(self.db, id, buf.as_mut_ptr() as *mut c_char, buf.len() as c_int) };
        }
        if n <= 0 {
            return None;
        }
        buf.truncate(n as usize);
        Some(buf)
    }

    /// The row's text (invalid UTF-8 replaced), `None` when it has none or there is no row.
    pub fn text(&self, id: i64) -> Option<String> {
        self.text_bytes(id).map(|b| String::from_utf8_lossy(&b).into_owned())
    }

    /// Deletes the row `id`; `true` when there was one.
    pub fn delete(&self, id: i64) -> bool {
        unsafe { ffi::vi_delete(self.db, id) == 1 }
    }

    /// Deletes every row with an id in [lo, hi]; how many.
    pub fn delete_range(&self, lo: i64, hi: i64) -> Result<usize> {
        let n = unsafe { ffi::vi_delete_range(self.db, lo, hi) };
        if n < 0 { Err(last("delete_range")) } else { Ok(n as usize) }
    }

    fn get64(&self, f: unsafe extern "C" fn(*mut c_void, i64, *mut i64) -> c_int, id: i64) -> Option<i64> {
        let mut x = 0i64;
        if unsafe { f(self.db, id, &mut x) } == 0 { Some(x) } else { None }
    }

    /// The row's 64-bit tag (labels share its bits).
    pub fn set_tag(&self, id: i64, tag: i64) -> Result<()> {
        Db::ok(unsafe { ffi::vi_set_tag(self.db, id, tag) }, "set_tag")
    }
    pub fn tag(&self, id: i64) -> Option<i64> {
        self.get64(ffi::vi_get_tag, id)
    }

    /// The row's number, for [`Filter::Num`] (a date, a price, a version).
    pub fn set_num(&self, id: i64, num: i64) -> Result<()> {
        Db::ok(unsafe { ffi::vi_set_num(self.db, id, num) }, "set_num")
    }
    pub fn num(&self, id: i64) -> Option<i64> {
        self.get64(ffi::vi_get_num, id)
    }

    /// Puts the row in a namespace (any i64, or [`key`] of a name).
    pub fn set_ns(&self, id: i64, ns: i64) -> Result<()> {
        Db::ok(unsafe { ffi::vi_set_ns(self.db, id, ns) }, "set_ns")
    }
    pub fn ns(&self, id: i64) -> Option<i64> {
        self.get64(ffi::vi_get_ns, id)
    }
    pub fn count_ns(&self, ns: i64) -> usize {
        unsafe { ffi::vi_count_ns(self.db, ns).max(0) as usize }
    }
    /// Deletes every row of a namespace; how many.
    pub fn delete_ns(&self, ns: i64) -> usize {
        unsafe { ffi::vi_delete_ns(self.db, ns).max(0) as usize }
    }

    /// Makes a label, under `parent` when given.
    pub fn label(&self, name: &str, parent: Option<&str>) -> Result<()> {
        let n = cstr(name)?;
        let p = match parent { Some(p) => Some(cstr(p)?), None => None };
        Db::ok(unsafe { ffi::vi_label(self.db, n.as_ptr(), p.as_ref().map_or(std::ptr::null(), |c| c.as_ptr())) }, "label")
    }
    /// Puts labels on a row ("red,big").
    pub fn add_labels(&self, id: i64, labels: &str) -> Result<()> {
        let l = cstr(labels)?;
        Db::ok(unsafe { ffi::vi_label_add(self.db, id, l.as_ptr()) }, "add_labels")
    }
    pub fn remove_labels(&self, id: i64, labels: &str) -> Result<()> {
        let l = cstr(labels)?;
        Db::ok(unsafe { ffi::vi_label_remove(self.db, id, l.as_ptr()) }, "remove_labels")
    }

    /// The `k` nearest rows, best first, as (id, score).
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(i64, f32)>> {
        self.search_where(Some(query), k, None)
    }

    /// `search` under one condition; `query` may be `None` only with [`Filter::Text`] (words alone).
    pub fn search_where(&self, query: Option<&[f32]>, k: usize, filter: Option<Filter<'_>>) -> Result<Vec<(i64, f32)>> {
        if let Some(q) = query {
            self.check(q)?;
        } else if !matches!(filter, Some(Filter::Text(_))) {
            return Err(Error("a query vector, or Filter::Text".into()));
        }
        if k == 0 {
            return Ok(Vec::new());
        }
        let q = query.map_or(std::ptr::null(), |q| q.as_ptr());
        let mut ids = vec![0i64; k];
        let mut sc = vec![0.0f32; k];
        let (ip, sp, kk) = (ids.as_mut_ptr(), sc.as_mut_ptr(), k as c_int);
        let n = unsafe {
            match filter {
                None => ffi::vi_search(self.db, q, kk, ip, sp),
                Some(Filter::Ns(ns)) => ffi::vi_search_ns(self.db, ns, q, kk, ip, sp),
                Some(Filter::AnyNs(set)) => ffi::vi_search_in(self.db, set.as_ptr(), set.len() as c_int, q, kk, ip, sp),
                Some(Filter::Num(lo, hi)) => ffi::vi_search_range(self.db, q, kk, lo, hi, ip, sp),
                Some(Filter::Labels { all, any, none }) => {
                    let (a, b, c) = (cstr(all)?, cstr(any)?, cstr(none)?);
                    ffi::vi_search_labeled(self.db, q, kk, a.as_ptr(), b.as_ptr(), c.as_ptr(), ip, sp)
                }
                Some(Filter::Text(t)) => ffi::vi_search_hybrid(self.db, q, t.as_ptr() as *const c_char, t.len() as c_int, kk, ip, sp),
                Some(Filter::Group(shift)) => ffi::vi_search_grouped(self.db, q, kk, shift as c_int, ip, sp),
            }
        };
        if n < 0 {
            return Err(last("search"));
        }
        Ok(ids.into_iter().zip(sc).take(n as usize).collect())
    }

    /// The `k` rows with the highest exact score - the same as scoring every row - and how many rows were scored
    /// exactly: the fast scan's error bound (proved in `lean4/Vecindex/Exact.lean`) rules the others out.
    pub fn search_exact(&self, query: &[f32], k: usize) -> Result<(Vec<(i64, f32)>, i64)> {
        self.check(query)?;
        if k == 0 {
            return Ok((Vec::new(), 0));
        }
        let mut ids = vec![0i64; k];
        let mut sc = vec![0.0f32; k];
        let mut chk = 0i64;
        let n = unsafe { ffi::vi_search_exact(self.db, query.as_ptr(), k as c_int, ids.as_mut_ptr(), sc.as_mut_ptr(), &mut chk) };
        if n < 0 {
            return Err(last("search_exact"));
        }
        Ok((ids.into_iter().zip(sc).take(n as usize).collect(), chk))
    }

    /// Indexes what was added (rows are indexed as they come by default; this does it now).
    pub fn build(&self) {
        unsafe { ffi::vi_build(self.db) }
    }

    /// Writes the whole store (to `path`, or its own file) and starts a fresh journal.
    pub fn save<P: AsRef<Path>>(&self, path: Option<P>) -> Result<()> {
        let p = match path { Some(p) => Some(cpath(p.as_ref())?), None => None };
        Db::ok(unsafe { ffi::vi_save(self.db, p.as_ref().map_or(std::ptr::null(), |c| c.as_ptr())) }, "save")
    }

    /// Flushes the journal to the disk: everything before it survives a crash.
    pub fn sync(&self) -> Result<()> {
        Db::ok(unsafe { ffi::vi_sync(self.db) }, "sync")
    }
}

impl Drop for Db {
    /// Flushes the journal and frees the store; the file is not rewritten (the next open replays the journal).
    fn drop(&mut self) {
        unsafe { ffi::vi_close(self.db) }
    }
}

impl fmt::Debug for Db {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Db {{ dim: {}, rows: {} }}", self.dim, self.len())
    }
}
