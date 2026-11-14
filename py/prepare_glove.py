#!/usr/bin/env python3
"""glove-100-angular, prepared the way ann-benchmarks does it.

GloVe Twitter, 100 dimensions, 1,193,514 words. Rows are normalised to unit length (angular =
cosine = inner product on unit vectors), 10,000 random rows are held out as queries, and the exact
top-100 neighbours of every query are computed by brute force (FAISS IndexFlatIP).

usage: prepare_glove.py GLOVE_TWITTER_100_GZ OUT_DIR
writes OUT_DIR/{train.f32, test.f32, gt.i32, meta.json}: raw little-endian arrays, row-major.
"""
import gzip
import json
import os
import sys
import time

import faiss
import numpy as np


def load(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        n, d = map(int, f.readline().split())
        rows, chunk, skipped = [], [], 0
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if len(parts) != d + 1:
                skipped += 1
                continue
            chunk.append(" ".join(parts[1:]))
            if len(chunk) == 100000:
                rows.append(np.array(" ".join(chunk).split(), dtype=np.float32).reshape(-1, d))
                chunk = []
        if chunk:
            rows.append(np.array(" ".join(chunk).split(), dtype=np.float32).reshape(-1, d))
    return np.concatenate(rows), n, skipped


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    X, n_header, skipped = load(src)
    norms = np.linalg.norm(X, axis=1)
    zero = int((norms == 0).sum())
    X = X[norms > 0] / norms[norms > 0, None]
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(X))
    test = np.ascontiguousarray(X[perm[:10000]])
    train = np.ascontiguousarray(X[perm[10000:]])
    print(f"parsed {len(X) + zero} rows ({n_header} in header, {skipped} malformed, {zero} zero) in {time.time() - t0:.1f}s")

    t0 = time.time()
    index = faiss.IndexFlatIP(train.shape[1])
    index.add(train)
    _, gt = index.search(test, 100)
    print(f"ground truth: {time.time() - t0:.1f}s")

    train.tofile(os.path.join(out, "train.f32"))
    test.tofile(os.path.join(out, "test.f32"))
    gt.astype(np.int32).tofile(os.path.join(out, "gt.i32"))
    meta = {"name": "glove-100-angular", "n": int(train.shape[0]), "nq": int(test.shape[0]),
            "dim": int(train.shape[1]), "gt_k": 100, "metric": "angular (unit vectors, inner product)",
            "source": "GloVe Twitter 27B 100d via gensim-data", "split_seed": 1,
            "dropped_zero_rows": zero, "dropped_malformed_rows": skipped}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
