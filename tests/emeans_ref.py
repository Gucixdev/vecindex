#!/usr/bin/env python3
"""emeans_ref.py - the store's e-means (src/tq.c) held to Meta's reference (faiss/contrib/e_means, MIT).

  EMEANS_REF=/path/to/faiss/contrib python3 -W error tests/emeans_ref.py .build/emeans_test [GLOVE_TRAIN_BIN]

1. The recipe: batch size, first step and the cosine schedule from the C code against the reference's own functions
   (base._resolve_recipe, base._scheduled_alpha) on a grid of n, k, passes and dimensions - the batch must be equal,
   the steps equal to 1e-12.
2. The result: the k-means error of the C e-means against the reference's NumpyEmeans(spherical=True) on the same
   rows, same k and passes, three seeds each. The two start from different rows and shuffle differently, so no bit
   equality: the C mean must not be more than 1% above the reference's.
Without numpy, faiss or EMEANS_REF it prints "skip" and exits 0; the run script counts that as a skip, not a pass.
"""
import os, subprocess, sys, tempfile

def skip(why):
    print("skip " + why)
    sys.exit(0)

ref = os.environ.get("EMEANS_REF")
if not ref or not os.path.isdir(os.path.join(ref, "e_means")):
    skip("EMEANS_REF does not point at faiss/contrib (with e_means/)")
try:
    import numpy as np
    import faiss  # noqa: F401  (the reference assigns with faiss.knn)
except ImportError as e:
    skip("no %s" % e.name)
sys.path.insert(0, ref)
from e_means import base                          # noqa: E402
from e_means.e_means import NumpyEmeans           # noqa: E402

exe = sys.argv[1]
fails = 0

def check(ok, msg):
    global fails
    print(("  ok   " if ok else "  FAIL ") + msg)
    fails += not ok

# 1. the recipe
grid = [(n, k, T, d) for n in (1000, 17345, 65536, 200001, 1110000, 40000000)
        for k in (2, 16, 67, 1024, 4350) for T in (1.0, 2.0, 3.0, 5.0, 2.5) for d in (2, 16, 17, 100, 768) if k < n]
out = subprocess.run([exe, "recipe"], input="".join("%d %d %r %d\n" % g for g in grid),
                     capture_output=True, text=True, check=True).stdout.split("\n")
bad_b = bad_a = bad_s = 0
for (n, k, T, d), line in zip(grid, out):
    B, a0, s7, s20 = line.split()
    law, _ = base._resolve_batch_law("auto", d)
    rb, ra = base._resolve_recipe(n, k, T, law, None, None)
    bad_b += int(B) != rb
    bad_a += abs(float(a0) - ra) > 1e-12 * ra
    bad_s += abs(float(s7) - base._scheduled_alpha(ra, 7, 20)) > 1e-12 or abs(float(s20) - base._scheduled_alpha(ra, 20, 20)) > 1e-12
check(len(out) - 1 == len(grid), "recipe: %d grid points answered" % len(grid))
check(bad_b == 0, "recipe: batch size equal to the reference's on all %d points (%d differ)" % (len(grid), bad_b))
check(bad_a == 0, "recipe: first step equal to 1e-12 on all points (%d differ)" % bad_a)
check(bad_s == 0, "schedule: the cosine steps equal to 1e-12 (%d differ)" % bad_s)

# 2. the result, on clustered synthetic rows and (when given) a GloVe sample
def sets():
    rs = np.random.RandomState(2026)
    k, d = 120, 64
    cen = rs.randn(k, d); cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    sizes = rs.randint(50, 900, size=k)
    x = np.concatenate([cen[i] + 0.06 * rs.randn(sizes[i], d) for i in range(k)]).astype("float32")
    x = x[rs.permutation(len(x))]
    yield "synthetic %d rows, %d clusters" % (len(x), k), x, k
    if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
        g = np.fromfile(sys.argv[2], dtype="float32").reshape(-1, 100)
        yield "GloVe-100 sample, 60000 rows", g[rs.choice(len(g), 60000, replace=False)], 256

for name, x, k in sets():
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    x = np.ascontiguousarray(x, dtype="float32")
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        np.array(x.shape, dtype="int32").tofile(f); x.tofile(f); path = f.name
    for T in (2.0, 3.0):
        cs, ls, refs = [], [], []
        for seed in (1, 2, 3):
            o = subprocess.run([exe, "train", path, str(k), repr(T), str(seed)], capture_output=True, text=True, check=True).stdout.split()
            cs.append(float(o[1])); ls.append(float(o[3]))
            em = NumpyEmeans(x.shape[1], k, nepochs=T, spherical=True, seed=seed)
            cen, _ = em.train(x)
            refs.append(float(np.mean(2 - 2 * (x @ cen.T).max(axis=1))))
        c, r, l = float(np.mean(cs)), float(np.mean(refs)), float(np.mean(ls))
        check(c <= 1.01 * r,
              "%s, %g passes: C e-means error %.5f (seeds 1-3: %s) no worse than 1%% over the reference's %.5f (%s); Lloyd 8 rounds %.5f"
              % (name, T, c, " ".join("%.5f" % v for v in cs), r, " ".join("%.5f" % v for v in refs), l))
    os.unlink(path)

print("all ok" if not fails else "FAILED")
sys.exit(1 if fails else 0)
