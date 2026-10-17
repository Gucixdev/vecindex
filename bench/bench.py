#!/usr/bin/env python3
"""One method of the vecindex benchmark on a prepared dataset (prepare_glove.py); prints JSON lines.

usage: bench.py DATA_DIR METHOD [--threads T] [--lat N]
methods: vecindex, vecindex-i8, vecindex-2bit, vecindex-2bit-i8, turbovec4, turbovec2, faiss-hnsw, faiss-ivfflat,
         faiss-ivfpqfs, qdrant

Per method: build (timed, T threads), index size, save and load (timed, warm page cache), then for every
search setting, on the LOADED index: recall@10 over all queries, single-thread QPS (one query per call,
first N queries) and T-thread QPS (all queries). Every library is driven from Python the same way, so the
per-call overhead is the same for all of them.
"""
import argparse
import concurrent.futures
import ctypes
import json
import os
import platform
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
K = 10


def recall_at_k(ids, gt, k=K):
    return float((ids[:, :k, None] == gt[:, None, :k]).any(axis=2).sum()) / (k * len(gt))


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


class VecIndex:
    """vecindex: the store's index (rotated 4-bit fast-scan, IVF from 600k rows, exact re-score of the candidates).
    bits=2: 2-bit codes; vbytes=1: int8 rows (the compact modes for very large stores)."""
    pins_threads = True

    def __init__(self, dim, threads, bits=4, vbytes=4):
        lib = ctypes.CDLL(os.path.join(HERE, "libvi.so"))
        P, I, L, LL, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_long, ctypes.c_longlong, ctypes.c_char_p
        lib.vi_new.restype = P; lib.vi_new.argtypes = [I]
        lib.vi_new2.restype = P; lib.vi_new2.argtypes = [I, I, I]
        lib.vi_set_rerank.argtypes = [P, I]
        lib.vi_file_bytes.restype = LL; lib.vi_file_bytes.argtypes = [P]
        lib.vi_add.restype = I; lib.vi_add.argtypes = [P, P, L, L]
        lib.vi_build.argtypes = [P]
        lib.vi_set_probe.argtypes = [P, I]
        lib.vi_set_threads.argtypes = [P, I]
        lib.vi_save.restype = I; lib.vi_save.argtypes = [P, F]
        lib.vi_load.restype = P; lib.vi_load.argtypes = [F]
        lib.vi_mem_bytes.restype = LL; lib.vi_mem_bytes.argtypes = [P]
        lib.vi_search.restype = I; lib.vi_search.argtypes = [P, P, I, P]
        lib.vi_search_batch.argtypes = [P, P, I, I, P, I]
        self.lib, self.dim, self.threads, self.s = lib, dim, threads, None
        self.bits, self.vbytes = bits, vbytes
        self.version = os.popen(f"git -C {HERE} rev-parse --short HEAD 2>/dev/null").read().strip()

    def build(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.s = self.lib.vi_new2(self.dim, self.bits, self.vbytes)
        self.lib.vi_add(self.s, X.ctypes.data, len(X), 0)
        self.lib.vi_build(self.s)

    def size(self):
        return int(self.lib.vi_mem_bytes(self.s)) + int(self.lib.vi_file_bytes(self.s))

    def size_split(self):
        """(RAM-only bytes, file-backed bytes): the codes stay hot; rows are paged in for re-scoring"""
        return int(self.lib.vi_mem_bytes(self.s)), int(self.lib.vi_file_bytes(self.s))

    def save(self, path):
        if self.lib.vi_save(self.s, path.encode()) != 0:
            raise RuntimeError("vi_save failed")

    def load(self, path):
        self.s = self.lib.vi_load(path.encode())
        if not self.s:
            raise RuntimeError("vi_load failed")

    def sweep(self):
        if self.bits == 2:                               # coarser codes: re-score more candidates
            return [{"nprobe": p, "rerank": r} for p in (16, 32, 64, 96, 128, 192) for r in (256, 1000)]
        return [{"nprobe": p} for p in (8, 16, 24, 32, 48, 64, 96, 128, 192)]

    def set(self, prm):
        self.lib.vi_set_probe(self.s, prm["nprobe"])
        self.lib.vi_set_rerank(self.s, prm.get("rerank", 0))
        self.lib.vi_set_threads(self.s, 1)

    def single_thread(self, on):
        pass

    def search_one(self, q, k):
        out = np.empty(k, np.int64)
        self.lib.vi_search(self.s, q.ctypes.data, k, out.ctypes.data)
        return out

    def search_batch(self, Q, k, threads):
        out = np.empty((len(Q), k), np.int64)
        self.lib.vi_search_batch(self.s, Q.ctypes.data, len(Q), k, out.ctypes.data, threads)
        return out


class Faiss:
    pins_threads = True

    def __init__(self, dim, threads, factory, sweep, ef_construction=None):
        import faiss
        self.faiss, self.threads, self.dim = faiss, threads, dim
        faiss.omp_set_num_threads(threads)
        self.index = faiss.index_factory(dim, factory, faiss.METRIC_INNER_PRODUCT)
        if ef_construction:
            self.index.hnsw.efConstruction = ef_construction
        self._sweep, self.version = sweep, faiss.__version__

    def build(self, X):
        if not self.index.is_trained:
            self.index.train(X)
        self.index.add(X)

    def size(self):
        return int(self.faiss.serialize_index(self.index).nbytes)

    def save(self, path):
        self.faiss.write_index(self.index, path)

    def load(self, path):
        self.index = self.faiss.read_index(path)

    def sweep(self):
        return self._sweep

    def set(self, prm):
        ps = self.faiss.ParameterSpace()
        for key, val in prm.items():
            if key == "k_factor":
                self.faiss.downcast_index(self.index).k_factor = float(val)
            else:
                ps.set_index_parameter(self.index, key, val)

    def single_thread(self, on):
        self.faiss.omp_set_num_threads(1 if on else self.threads)

    def search_one(self, q, k):
        return self.index.search(q[None, :], k)[1][0]

    def search_batch(self, Q, k, threads):
        self.faiss.omp_set_num_threads(threads)
        return self.index.search(Q, k)[1]


class TurboVec:
    """turbovec (TurboQuant): no knobs; its thread pool is fixed at start, so run once per thread count.
    It needs dim % 8 == 0: vectors (and queries) are zero-padded to the next multiple of 8, which leaves every
    inner product unchanged (GloVe 100 -> 104)."""
    pins_threads = False

    def __init__(self, dim, threads, bits):
        os.environ["RAYON_NUM_THREADS"] = str(threads)
        import turbovec
        self.mod, self.dim, self.bits, self.version = turbovec, dim, bits, getattr(turbovec, "__version__", "?")
        self.pdim = (dim + 7) // 8 * 8

    def _pad(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if self.pdim == self.dim:
            return X
        return np.ascontiguousarray(np.pad(X, [(0, 0)] * (X.ndim - 1) + [(0, self.pdim - self.dim)]))

    def build(self, X):
        self.index = self.mod.TurboQuantIndex(dim=self.pdim, bit_width=self.bits)
        self.index.add(self._pad(X))

    def size(self):
        return len(self.index.to_bytes())

    def save(self, path):
        self.index.write(path)

    def load(self, path):
        self.index = self.mod.TurboQuantIndex.load(path)

    def sweep(self):
        return [{}]

    def set(self, prm):
        pass

    def single_thread(self, on):
        pass

    def search_one(self, q, k):
        return self.index.search(self._pad(q[None, :]), k)[1][0]

    def search_batch(self, Q, k, threads):
        return self.index.search(self._pad(Q), k)[1]


class Qdrant:
    """Qdrant's engine embedded in-process (qdrant-edge-py, by the Qdrant team): default HNSW m=16,
    ef_construct=100, cosine. Batch QPS: one query per call from T Python threads."""
    pins_threads = True

    def __init__(self, dim, threads):
        import qdrant_edge as qe
        self.qe, self.dim, self.threads, self.ef = qe, dim, threads, 64
        self.version = "qdrant-edge-py " + __import__("importlib.metadata").metadata.version("qdrant-edge-py")
        self.path = None

    def _cfg(self, search_threads):
        qe = self.qe
        return qe.EdgeConfig(
            vectors=qe.EdgeVectorParams(size=self.dim, distance=qe.Distance.Cosine),
            hnsw_config=qe.HnswIndexConfig(m=16, ef_construct=100, full_scan_threshold=10000,
                                           max_indexing_threads=self.threads),
            max_search_threads=search_threads)

    def build(self, X):
        qe = self.qe
        self.path = tempfile.mkdtemp(prefix="qdrant-", dir=os.environ.get("BENCH_TMP"))
        self.shard = qe.EdgeShard.create(self.path, self._cfg(self.threads))
        for i in range(0, len(X), 5000):
            pts = [qe.Point(j, X[j].tolist()) for j in range(i, min(i + 5000, len(X)))]
            self.shard.update(qe.UpdateOperation.upsert_points(pts))
        self.shard.optimize()
        self.shard.flush()

    def size(self):
        total = 0
        for root, _, files in os.walk(self.path):
            if os.sep + "wal" in root + os.sep:
                continue
            total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
        return total

    def save(self, path):
        self.shard.flush()
        self.shard.close()

    def load(self, path):
        self.shard = self.qe.EdgeShard.load(self.path, self._cfg(self.threads))

    def sweep(self):
        return [{"hnsw_ef": e} for e in (16, 32, 48, 64, 96, 128, 192, 256, 384)]

    def set(self, prm):
        self.ef = prm["hnsw_ef"]

    def single_thread(self, on):
        pass

    def search_one(self, q, k):
        qe = self.qe
        res = self.shard.search(qe.SearchRequest(query=qe.Query.Nearest(q.tolist()), limit=k,
                                                 params=qe.SearchParams(hnsw_ef=self.ef)))
        ids = [int(p.id) for p in res][:k]
        return np.array(ids + [-1] * (k - len(ids)), np.int64)

    def search_batch(self, Q, k, threads):
        with concurrent.futures.ThreadPoolExecutor(threads) as ex:
            return np.stack(list(ex.map(lambda q: self.search_one(q, k), Q)))

    def cleanup(self):
        if self.path:
            shutil.rmtree(self.path, ignore_errors=True)


def make(method, dim, threads):
    if method == "vecindex":
        return VecIndex(dim, threads)
    if method == "vecindex-i8":
        return VecIndex(dim, threads, 4, 1)
    if method == "vecindex-2bit":
        return VecIndex(dim, threads, 2, 4)
    if method == "vecindex-2bit-i8":
        return VecIndex(dim, threads, 2, 1)
    if method == "turbovec4":
        return TurboVec(dim, threads, 4)
    if method == "turbovec2":
        return TurboVec(dim, threads, 2)
    if method == "faiss-hnsw":
        return Faiss(dim, threads, "HNSW32", [{"efSearch": e} for e in (16, 32, 48, 64, 96, 128, 192, 256, 384)],
                     ef_construction=200)
    if method == "faiss-ivfflat":
        return Faiss(dim, threads, "IVF4096,Flat", [{"nprobe": p} for p in (8, 16, 32, 48, 64, 96, 128, 192)])
    if method == "faiss-ivfpqfs":
        return Faiss(dim, threads, "IVF4096,PQ50x4fs,RFlat",
                     [{"nprobe": p, "k_factor": f} for p in (8, 16, 32, 64, 128) for f in (4, 16)])
    if method == "qdrant":
        return Qdrant(dim, threads)
    raise SystemExit(f"unknown method {method}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("method")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--lat", type=int, default=2000, help="queries timed one at a time on one thread")
    ap.add_argument("--sweep", help="override the search settings, e.g. nprobe=32,96 (one parameter)")
    a = ap.parse_args()

    with open(os.path.join(a.data, "meta.json")) as f:
        meta = json.load(f)
    n, nq, dim = meta["n"], meta["nq"], meta["dim"]
    train = np.fromfile(os.path.join(a.data, "train.f32"), np.float32).reshape(n, dim)
    test = np.fromfile(os.path.join(a.data, "test.f32"), np.float32).reshape(nq, dim)
    gt = np.fromfile(os.path.join(a.data, "gt.i32"), np.int32).reshape(nq, meta["gt_k"])

    m = make(a.method, dim, a.threads)
    tmp = tempfile.mkdtemp(prefix="bench-", dir=os.environ.get("BENCH_TMP"))
    path = os.path.join(tmp, "index")
    try:
        t0 = time.perf_counter(); m.build(train); t_build = time.perf_counter() - t0
        t0 = time.perf_counter(); m.save(path); t_save = time.perf_counter() - t0
        t0 = time.perf_counter(); m.load(path); t_load = time.perf_counter() - t0
        size = m.size()                                   # what the loaded, ready-to-search index holds
        base = {"dataset": meta["name"], "method": a.method, "version": m.version, "n": n, "dim": dim,
                "threads": a.threads, "cpu": cpu_model(), "build_s": round(t_build, 3),
                "save_s": round(t_save, 3), "load_s": round(t_load, 3), "index_bytes": size,
                "bytes_per_vec": round(size / n, 1)}
        if hasattr(m, "size_split"):
            ram, fil = m.size_split()
            base.update(ram_bytes_per_vec=round(ram / n, 1), file_bytes_per_vec=round(fil / n, 1))
        print(json.dumps(dict(base, kind="build")), flush=True)
        sweep = m.sweep()
        if a.sweep:
            key, vals = a.sweep.split("=", 1)
            sweep = [dict(p, **{key: int(v)}) for v in vals.split(",") for p in ([{}] if key in sweep[0] or len(sweep[0]) == 1 else [dict((k2, v2) for k2, v2 in sweep[0].items() if k2 != key)])]
        for prm in sweep:
            m.set(prm)
            m.single_thread(False)
            t0 = time.perf_counter(); ids = m.search_batch(test, K, a.threads); t_mt = time.perf_counter() - t0
            rec = recall_at_k(np.asarray(ids), gt)
            qps_1t = None
            if m.pins_threads or a.threads == 1:
                m.single_thread(True)
                t0 = time.perf_counter()
                for q in test[:a.lat]:
                    m.search_one(q, K)
                qps_1t = round(a.lat / (time.perf_counter() - t0), 1)
                m.single_thread(False)
            print(json.dumps(dict(base, kind="search", params=prm, recall10=round(rec, 4),
                                  qps_1t=qps_1t, qps_mt=round(nq / t_mt, 1))), flush=True)
    finally:
        if hasattr(m, "cleanup"):
            m.cleanup()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
