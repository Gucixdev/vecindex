"""The README tables from bench.py output: python3 tables.py GLOVE.jsonl [OTHERS.jsonl [OLD.jsonl]]
vecindex rows come from the first file, every other library from the second (default: the same file).
QPS cells: batch of 10k queries on T threads (T = the most threads any row in the file used) / one query per call on 1 thread, best setting at recall@10 >= x."""
import json, sys, collections

TH = (0.85, 0.90, 0.95, 0.97)
T = 2   # set from the data below
NAMES = collections.OrderedDict([
    ("vecindex", "vecindex 4-bit + f32 rows"),
    ("vecindex-i8", "vecindex 4-bit + int8 rows"),
    ("vecindex-2bit", "vecindex 2-bit + f32 rows"),
    ("vecindex-2bit-i8", "vecindex 2-bit + int8 rows"),
    ("vecindex-1bit", "vecindex 1-bit + f32 rows"),
    ("vecindex-1bit-i8", "vecindex 1-bit + int8 rows"),
    ("vecindex-cascade", "vecindex cascade: 1-bit, 4-bit, f32 rows"),
    ("vecindex-cascade-i8", "vecindex cascade: 1-bit, 4-bit, int8 rows"),
    ("faiss-ivfpqfs", "FAISS IVF4096,PQ50x4fs,RFlat"),
    ("faiss-hnsw", "FAISS HNSW32"),
    ("leann", "LEANN (HNSW, vectors stored)"),
    ("vecgrid", "VecGrid (hnswlib, 271 partitions)"),
    ("faiss-ivfflat", "FAISS IVF4096,Flat"),
    ("qdrant", "Qdrant (edge, HNSW)"),
    ("turbovec4", "turbovec 4-bit"),
    ("turbovec2", "turbovec 2-bit"),
])

def load(files, only=None):
    builds, search = {}, collections.defaultdict(list)
    for f in files:
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            m = r["method"]
            if only and not only(m): continue
            if r["kind"] == "build":
                if r["threads"] == T or m not in builds: builds[m] = r
            else: search[m].append(r)
    return builds, search

def pick(search, m, x, mode):
    c = []
    for s in search[m]:
        if s["recall10"] < x: continue
        if mode == "2t" and s["threads"] == T and s.get("qps_mt"): c.append((s["qps_mt"], s))
        if mode == "1t" and s.get("qps_1t"): c.append((s["qps_1t"], s))
    return max(c, key=lambda z: z[0]) if c else (None, None)

def fmt(q):
    return "-" if q is None else f"{q:,.0f}"

T = max((json.loads(l).get("threads", 1) for l in open(sys.argv[1]) if l.startswith("{")), default=2)
vi = sys.argv[1]; others = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]
b1, s1 = load([vi], lambda m: m.startswith("vecindex"))
b2, s2 = load([others], lambda m: not m.startswith("vecindex"))
builds = dict(b1); builds.update(b2)
search = collections.defaultdict(list); search.update(s1); search.update(s2)

print("| method | build | load | bytes/vec | best recall@10 |" + "".join(f" >= {x:.2f} |" for x in TH))
print("|---|---:|---:|---:|---:|" + "---:|" * len(TH))
for m, name in NAMES.items():
    if m not in builds: continue
    b = builds[m]
    rmax = max((s["recall10"] for s in search[m]), default=0)
    cells = []
    for x in TH:
        q2, _ = pick(search, m, x, "2t"); q1, _ = pick(search, m, x, "1t")
        cells.append("-" if q2 is None and q1 is None else f"{fmt(q2)} / {fmt(q1)}")
    bt = f"{b['build_s']:.1f} s"; lt = f"{b['load_s']*1000:.0f} ms" if b['load_s'] < 1 else f"{b['load_s']:.2f} s"
    print(f"| {name} | {bt} | {lt} | {b['bytes_per_vec']:.0f} | {rmax:.4f} |" + "".join(f" {c} |" for c in cells))

print()
print("settings:")
for m in NAMES:
    if m not in builds: continue
    out = []
    for x in TH:
        q2, a = pick(search, m, x, "2t"); q1, b = pick(search, m, x, "1t")
        f = lambda s: ",".join(f"{k}={v}" for k, v in s["params"].items()) + f" ({s['recall10']})" if s else "-"
        out.append(f">={x}: {f(a)} | {f(b)}")
    print(" ", m, "; ".join(out))

print()
ours = [m for m in NAMES if m.startswith("vecindex") and m in builds]
theirs = [m for m in NAMES if not m.startswith("vecindex") and m in builds]
print(f"| recall@10 | vecindex, batch {T} threads | best other | x | vecindex, 1 thread per call | best other | x |")
print("|---|---:|---|---:|---:|---|---:|")
for x in TH:
    row = f"| >= {x:.2f} |"
    for mode in ("2t", "1t"):
        a = max(((pick(search, m, x, mode)[0] or 0, m) for m in ours))
        o = max(((pick(search, m, x, mode)[0] or 0, m) for m in theirs), default=(0, ""))
        row += f" {fmt(a[0])} | {fmt(o[0]) + ' (' + NAMES[o[1]] + ')' if o[0] else 'none reaches it'} | {a[0] / o[0]:.2f}x |" if o[0] else f" {fmt(a[0])} | none reaches it | - |"
    print(row)


# scorecard: every verdict is computed from the rows above, nothing written by hand
def scorecard(label, mine):
    print()
    print(f"scorecard - {label} vs the best of the others (build/load/bytes: only methods reaching recall@10 >= 0.90; lower is better there, higher for the rest)")
    print("| metric | vecindex | best other | verdict |")
    print("|---|---:|---|---|")
    def reach(m): return max((x["recall10"] for x in search[m]), default=0) >= 0.90
    def low(key, f, unit):   # only methods that reach recall@10 0.90 at all: a quantizer with no rerank is not a fair build-time rival
        a = min(((builds[m][key] * f, m) for m in mine if m in builds and reach(m)), default=None)
        o = min(((builds[m][key] * f, m) for m in theirs if reach(m)), default=None)
        return a, o, unit
    def row(name, a, o, unit, lower):
        if not a or not o: return
        if lower and a[0] == 0: print(f"| {name} | 0 {unit} | {o[0]:,.2f} {unit} ({NAMES[o[1]]}) | vecindex better (0) |"); return
        if lower and o[0] == 0: print(f"| {name} | {a[0]:,.2f} {unit} ({NAMES[a[1]]}) | 0 {unit} ({NAMES[o[1]]}) | other has none to pay (mmap, nothing to load) |"); return
        r = (o[0] / a[0]) if lower else (a[0] / o[0] if o[0] else 0)
        if r == 0: print(f"| {name} | {a[0]:,.2f} | {o[0]:,.2f} | other is higher by an unmeasurable factor |"); return
        win = r > 1.0
        v = f"vecindex better, {r:.2f}x" if win else (f"SAME" if abs(r - 1) < 0.005 else f"vecindex WORSE, {1 / r:.2f}x behind")
        print(f"| {name} | {a[0]:,.2f} {unit} ({NAMES[a[1]]}) | {o[0]:,.2f} {unit} ({NAMES[o[1]]}) | {v} |")
    row("build time", *low("build_s", 1, "s")[:2], "s", True)
    row("load time", *low("load_s", 1000, "ms")[:2], "ms", True)
    row("bytes per vector", *low("bytes_per_vec", 1, "B")[:2], "B", True)
    ra = max(((max((x["recall10"] for x in search[m]), default=0), m) for m in mine if m in builds), default=None)
    ro = max(((max((x["recall10"] for x in search[m]), default=0), m) for m in theirs), default=None)
    row("best recall@10 reached", ra, ro, "", False)
    for x in (0.90, 0.95):
        for mode, nm in (("2t", f"QPS, batch {T} threads"), ("1t", "QPS, 1 thread per call")):
            a = max(((pick(search, m, x, mode)[0] or 0, m) for m in mine if m in builds), default=None)
            o = max(((pick(search, m, x, mode)[0] or 0, m) for m in theirs), default=None)
            if a and o and o[0]: row(f"{nm} at recall@10 >= {x:.2f}", a, o, "", False)
            elif a and a[0]: print(f"| {nm} at recall@10 >= {x:.2f} | {a[0]:,.0f} | none reaches it | vecindex only |")
scorecard("default vecindex (4-bit codes + f32 rows)", ["vecindex"])
scorecard(f"best of the {len(ours)} vecindex variants", ours)
print()
print(f"caveats: one host ({__import__('platform').node()}), {T} threads for the batch columns; vecindex has {len(ours)} variants against 1-2 settings per other library,")
print("so 'best of variants' flatters it - read the default-config scorecard first. Recall is vs exact top-10 on 10k held-out queries.")

if len(sys.argv) > 3:
    print()
    bo, so = load([sys.argv[3]])
    print("old code on the same host:")
    for m in bo:
        b = bo[m]
        cells = []
        for x in TH:
            q2, _ = pick(so, m, x, "2t"); q1, _ = pick(so, m, x, "1t")
            cells.append(f"{fmt(q2)} / {fmt(q1)}")
        print(f"| {m} (old) | {b['build_s']:.1f} s | {b['bytes_per_vec']:.0f} | {max(s['recall10'] for s in so[m]):.4f} |" + "".join(f" {c} |" for c in cells))
