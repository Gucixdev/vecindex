#!/usr/bin/env python3
"""turbovec (Google TurboQuant) on the vectors vcmp dumped: python3 vcmp_turbovec.py OUT_DIR"""
import os, sys, time
sys.path.insert(0, os.path.expanduser("~/.cache/vecdb/pylibs"))
import numpy as np
from turbovec import TurboQuantIndex

d = sys.argv[1]; DIM = 256
data = np.fromfile(d + "/data.bin", np.float32).reshape(-1, DIM)
q = np.fromfile(d + "/queries.bin", np.float32).reshape(-1, DIM)
truth = np.fromfile(d + "/truth.bin", np.int32).reshape(-1, 10)
for bits in (4, 2):
    idx = TurboQuantIndex(dim=DIM, bit_width=bits)
    t0 = time.time(); idx.add(data); tb = time.time() - t0
    hit = 0; t0 = time.time()
    for i in range(len(q)):
        _, ids = idx.search(q[i:i + 1], k=10)
        hit += len(set(np.asarray(ids).ravel().tolist()) & set(truth[i].tolist()))
    ts = (time.time() - t0) / len(q)
    print("turbovec %d-bit: add %.2f s, recall@10 %.1f%%, %.3f ms/query, %.1f MB codes" % (bits, tb, 100 * hit / (10 * len(q)), ts * 1e3, len(data) * DIM * bits / 8 / 1e6))
