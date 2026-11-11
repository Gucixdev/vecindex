"""vecindex - a vector database in one C file, for RAG inside an application.

    import vecindex
    with vecindex.open("docs.vxi", dim=384) as db:
        db.add(1, embedding, text="the chunk the embedding came from")
        for id, score in db.search(query, k=5):
            print(id, score, db.text(id))

A store is a file and its journal: every change is in the journal at once, a crash loses nothing that was synced,
and damaged blocks of the file are rebuilt from its parity when it is opened. Vectors are any sequence of floats
(a list, array.array("f"), a float32 numpy array - the last two without a copy per element).

The C library is built into the package and loaded with ctypes; there is nothing else to install. The C API it wraps
is include/vecindex.h of vecindex.
"""
import array
import ctypes
import glob
import os

__all__ = ["DB", "open", "key", "version", "VecindexError"]
__version__ = "0.4.0"

_i64 = ctypes.c_int64
_f32p = ctypes.POINTER(ctypes.c_float)
_i64p = ctypes.POINTER(_i64)
_vp = ctypes.c_void_p
_cp = ctypes.c_char_p


def _load():
    here = os.path.dirname(os.path.abspath(__file__))
    found = sorted(glob.glob(os.path.join(here, "_libvecindex*.so")) + glob.glob(os.path.join(here, "_libvecindex*.pyd"))
                   + glob.glob(os.path.join(here, "_libvecindex*.dylib")))
    if not found:
        raise ImportError("vecindex: the C library was not built into the package (pip install it, or "
                          "python setup.py build_ext --inplace in bindings/python)")
    return ctypes.CDLL(found[0])


_L = _load()


def _sig(name, res, *args):
    f = getattr(_L, name)
    f.restype = res
    f.argtypes = list(args)
    return f


_sig("vi_version", _cp)
_sig("vi_error", _cp)
_sig("vi_open", _vp, _cp, ctypes.c_int, ctypes.c_int)
_sig("vi_close", None, _vp)
_sig("vi_dim", ctypes.c_int, _vp)
_sig("vi_count", _i64, _vp)
_sig("vi_add", ctypes.c_int, _vp, _i64, _f32p)
_sig("vi_update", ctypes.c_int, _vp, _i64, _f32p)
_sig("vi_add_batch", ctypes.c_int, _vp, ctypes.c_int, _i64p, _f32p)
_sig("vi_add_text", ctypes.c_int, _vp, _i64, _f32p, _cp, ctypes.c_int)
_sig("vi_get", ctypes.c_int, _vp, _i64, _f32p)
_sig("vi_get_text", ctypes.c_int, _vp, _i64, _cp, ctypes.c_int)
_sig("vi_delete", ctypes.c_int, _vp, _i64)
_sig("vi_delete_range", ctypes.c_int, _vp, _i64, _i64)
_sig("vi_key", _i64, _cp, ctypes.c_int)
_sig("vi_label", ctypes.c_int, _vp, _cp, _cp)
_sig("vi_label_add", ctypes.c_int, _vp, _i64, _cp)
_sig("vi_label_remove", ctypes.c_int, _vp, _i64, _cp)
_sig("vi_set_tag", ctypes.c_int, _vp, _i64, _i64)
_sig("vi_get_tag", ctypes.c_int, _vp, _i64, _i64p)
_sig("vi_set_num", ctypes.c_int, _vp, _i64, _i64)
_sig("vi_get_num", ctypes.c_int, _vp, _i64, _i64p)
_sig("vi_set_ns", ctypes.c_int, _vp, _i64, _i64)
_sig("vi_get_ns", ctypes.c_int, _vp, _i64, _i64p)
_sig("vi_count_ns", _i64, _vp, _i64)
_sig("vi_delete_ns", _i64, _vp, _i64)
_sig("vi_search", ctypes.c_int, _vp, _f32p, ctypes.c_int, _i64p, _f32p)
_sig("vi_search_ns", ctypes.c_int, _vp, _i64, _f32p, ctypes.c_int, _i64p, _f32p)
_sig("vi_search_in", ctypes.c_int, _vp, _i64p, ctypes.c_int, _f32p, ctypes.c_int, _i64p, _f32p)
_sig("vi_search_range", ctypes.c_int, _vp, _f32p, ctypes.c_int, _i64, _i64, _i64p, _f32p)
_sig("vi_search_labeled", ctypes.c_int, _vp, _f32p, ctypes.c_int, _cp, _cp, _cp, _i64p, _f32p)
_sig("vi_search_hybrid", ctypes.c_int, _vp, _f32p, _cp, ctypes.c_int, ctypes.c_int, _i64p, _f32p)
_sig("vi_search_grouped", ctypes.c_int, _vp, _f32p, ctypes.c_int, ctypes.c_int, _i64p, _f32p)
_sig("vi_build", None, _vp)
_sig("vi_save", ctypes.c_int, _vp, _cp)
_sig("vi_sync", ctypes.c_int, _vp)

LIVE, MANUAL, DOT, L2 = 1, 2, 4, 8
_METRICS = {"cos": 0, "cosine": 0, "dot": DOT, "l2": L2}


class VecindexError(Exception):
    """A call the C library refused; the message is vi_error()'s."""


def _err(what):
    msg = _L.vi_error()
    return VecindexError("%s: %s" % (what, msg.decode("utf-8", "replace") if msg else "failed"))


def version():
    """The C library's version (VI_VERSION)."""
    return _L.vi_version().decode()


def key(name):
    """A 64-bit key for a name: a namespace or an id made from a string, the same in every language."""
    b = name.encode() if isinstance(name, str) else bytes(name)
    return _L.vi_key(b, len(b))


def _b(s):
    if s is None:
        return None
    if isinstance(s, (list, tuple, set)):
        s = ",".join(s)
    return s.encode() if isinstance(s, str) else bytes(s)


class DB:
    """One store: a file (path) and its journal, or memory only (path None). dim 0 opens an existing file."""

    def __init__(self, path=None, dim=0, metric="cos", manual=False):
        if metric not in _METRICS:
            raise ValueError("metric is one of cos, dot, l2")
        flags = _METRICS[metric] | (MANUAL if manual else 0)
        self._db = _L.vi_open(os.fsencode(path) if path is not None else None, int(dim), flags)
        if not self._db:
            raise _err("open %s" % (path,))
        self.dim = _L.vi_dim(self._db)
        self._ids = None
        self._sc = None

    # -- the handle --
    def close(self):
        """Flushes the journal and frees the store (the file is not rewritten: the next open replays the journal)."""
        if getattr(self, "_db", None):
            _L.vi_close(self._db)
            self._db = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _h(self):
        if not self._db:
            raise VecindexError("the store is closed")
        return self._db

    def __len__(self):
        return int(_L.vi_count(self._h()))

    # -- vectors in and out --
    def _vec(self, v, n=1):
        """v as n * dim float32 values, in a ctypes array (no copy per element when v is a float32 buffer)."""
        want = n * self.dim
        try:
            mv = memoryview(v)
            if mv.format == "f" and mv.c_contiguous and mv.nbytes == want * 4:
                return (ctypes.c_float * want).from_buffer_copy(mv)
        except TypeError:
            pass
        if n > 1 and not isinstance(v, (array.array, bytes, bytearray)) and hasattr(v, "__len__") \
                and len(v) == n and hasattr(v[0], "__len__"):
            flat = array.array("f")
            for row in v:
                flat.extend(array.array("f", row))
            a = flat
        else:
            a = array.array("f", v)
        if len(a) != want:
            raise ValueError("a vector of %d floats was expected, not %d" % (want, len(a)))
        return (ctypes.c_float * want).from_buffer(a)

    def add(self, id, vec, text=None):
        """Adds the row id (or replaces it), with an optional text kept beside it (str or bytes)."""
        v = self._vec(vec)
        if text is None:
            r = _L.vi_add(self._h(), int(id), v)
        else:
            t = text.encode() if isinstance(text, str) else bytes(text)
            r = _L.vi_add_text(self._h(), int(id), v, t, len(t))
        if r != 0:
            raise _err("add %d" % id)

    def add_many(self, ids, vecs):
        """Many rows at once: ids, and vecs as len(ids) vectors (a list of lists, or one flat float32 buffer)."""
        n = len(ids)
        ia = (_i64 * n)(*[int(i) for i in ids])
        r = _L.vi_add_batch(self._h(), n, ia, self._vec(vecs, n))
        if r != n:
            raise _err("add_many: %d of %d went in" % (max(r, 0), n))

    def update(self, id, vec):
        """A new vector for the row id; its text, tag, number and namespace stay."""
        if _L.vi_update(self._h(), int(id), self._vec(vec)) != 0:
            raise _err("update %d" % id)

    def get(self, id):
        """The row's vector (a list of floats), or None when there is no such row."""
        out = (ctypes.c_float * self.dim)()
        if _L.vi_get(self._h(), int(id), out) != 0:
            return None
        return list(out)

    def text(self, id, raw=False):
        """The row's text (str, or bytes with raw), None when it has none or there is no row."""
        buf = ctypes.create_string_buffer(4096)
        n = _L.vi_get_text(self._h(), int(id), buf, len(buf))
        if n > len(buf):
            buf = ctypes.create_string_buffer(n)
            n = _L.vi_get_text(self._h(), int(id), buf, len(buf))
        if n <= 0:
            return None
        b = buf.raw[:n]
        return b if raw else b.decode("utf-8", "replace")

    def __contains__(self, id):
        return self.get(id) is not None

    def delete(self, id):
        """Deletes the row id; True when there was one."""
        return _L.vi_delete(self._h(), int(id)) == 1

    def delete_range(self, lo, hi):
        """Deletes every row with an id in [lo, hi]; returns how many."""
        n = _L.vi_delete_range(self._h(), int(lo), int(hi))
        if n < 0:
            raise _err("delete_range")
        return n

    # -- what a row carries --
    def _set(self, fn, what, id, v):
        if fn(self._h(), int(id), int(v)) != 0:
            raise _err("%s %d" % (what, id))

    def _get(self, fn, id):
        x = _i64()
        return x.value if fn(self._h(), int(id), ctypes.byref(x)) == 0 else None

    def set_tag(self, id, tag): self._set(_L.vi_set_tag, "set_tag", id, tag)
    def tag(self, id): return self._get(_L.vi_get_tag, id)
    def set_num(self, id, num): self._set(_L.vi_set_num, "set_num", id, num)
    def num(self, id): return self._get(_L.vi_get_num, id)

    def set_ns(self, id, ns):
        """Puts the row in a namespace: an int, or a name (str)."""
        self._set(_L.vi_set_ns, "set_ns", id, key(ns) if isinstance(ns, str) else ns)

    def ns(self, id): return self._get(_L.vi_get_ns, id)

    def count_ns(self, ns):
        return int(_L.vi_count_ns(self._h(), key(ns) if isinstance(ns, str) else int(ns)))

    def delete_ns(self, ns):
        """Deletes every row of a namespace; returns how many."""
        return int(_L.vi_delete_ns(self._h(), key(ns) if isinstance(ns, str) else int(ns)))

    def label(self, name, parent=None):
        """Makes a label, under parent when given (a label stands for itself and everything below it)."""
        if _L.vi_label(self._h(), _b(name), _b(parent)) != 0:
            raise _err("label %s" % name)

    def add_labels(self, id, labels):
        """Puts labels on a row: "red,big" or a list."""
        if _L.vi_label_add(self._h(), int(id), _b(labels)) != 0:
            raise _err("add_labels %d" % id)

    def remove_labels(self, id, labels):
        if _L.vi_label_remove(self._h(), int(id), _b(labels)) != 0:
            raise _err("remove_labels %d" % id)

    # -- search --
    def search(self, vec=None, k=10, *, text=None, ns=None, num=None, labels=None, group=None):
        """The k nearest rows as [(id, score)], best first. One condition at a time:
        text: words too (BM25 fused with the vector; vec may be None for words only);
        ns: a namespace (int or name) or a list of them; num: (lo, hi) of the rows' numbers;
        labels: {"all": ..., "any": ..., "none": ...} (each "a,b" or a list); group: a shift, one hit per id >> shift.
        """
        k = int(k)
        if k <= 0:
            return []
        given = [x is not None for x in (text, ns, num, labels, group)]
        if sum(given) > 1:
            raise ValueError("search takes one of text, ns, num, labels, group at a time")
        db = self._h()
        ids = (_i64 * k)()
        sc = (ctypes.c_float * k)()
        q = self._vec(vec) if vec is not None else None
        if q is None and text is None:
            raise ValueError("a query vector, or text")
        if text is not None:
            t = text.encode() if isinstance(text, str) else bytes(text)
            n = _L.vi_search_hybrid(db, q, t, len(t), k, ids, sc)
        elif ns is not None:
            if isinstance(ns, (list, tuple, set)):
                ks = [key(x) if isinstance(x, str) else int(x) for x in ns]
                arr = (_i64 * len(ks))(*ks)
                n = _L.vi_search_in(db, arr, len(ks), q, k, ids, sc)
            else:
                n = _L.vi_search_ns(db, key(ns) if isinstance(ns, str) else int(ns), q, k, ids, sc)
        elif num is not None:
            lo, hi = num
            n = _L.vi_search_range(db, q, k, int(lo), int(hi), ids, sc)
        elif labels is not None:
            n = _L.vi_search_labeled(db, q, k, _b(labels.get("all")), _b(labels.get("any")), _b(labels.get("none")), ids, sc)
        elif group is not None:
            n = _L.vi_search_grouped(db, q, k, int(group), ids, sc)
        else:
            n = _L.vi_search(db, q, k, ids, sc)
        if n < 0:
            raise _err("search")
        return [(ids[i], sc[i]) for i in range(n)]

    # -- the file --
    def build(self):
        """Indexes what was added (only needed with manual=True; by default rows are indexed as they come)."""
        _L.vi_build(self._h())

    def save(self, path=None):
        """Writes the whole store (to path, or its own file) and starts a fresh journal."""
        if _L.vi_save(self._h(), os.fsencode(path) if path is not None else None) != 0:
            raise _err("save")

    def sync(self):
        """Flushes the journal to the disk: everything before it survives a crash."""
        if _L.vi_sync(self._h()) != 0:
            raise _err("sync")


def open(path=None, dim=0, metric="cos", manual=False):
    """vecindex.open("docs.vxi", dim=384): the store in path, made with dim dimensions if it is not there yet."""
    return DB(path, dim, metric, manual)
