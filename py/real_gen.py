#!/usr/bin/env python3
"""Build a REAL RAG retrieval benchmark for vecindex.

corpus  = unique SQuAD contexts (real Wikipedia passages) + AG-News snippets
queries = SQuAD questions (question -> its passage is a genuine asymmetric
          retrieval task, not synthetic near-duplicates)
GT      = exact cosine top-10 over the embedded corpus (what an ANN index
          should return) AND the gold passage id for each question (the
          actual RAG metric).

writes  real_data.bin  <IIII dim n nq k> f32[n*dim] f32[nq*dim]
        real_gt.bin     <II nq k> u64[nq*k]         (cosine top-k)
        real_gold.bin   <I nq>    i32[nq]            (gold passage row, -1 if none)
"""
import sys, struct, numpy as np, datasets, torch
from transformers import AutoTokenizer, AutoModel

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
N_CTX_CAP = 40000       # squad unique contexts (~18.9k) + agnews to top up
N_AGNEWS  = 30000
NQ        = 1000
K         = 10
BATCH     = 256

tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModel.from_pretrained(MODEL).eval()
dev = "cuda" if torch.cuda.is_available() else "cpu"
mdl = mdl.to(dev)

def embed(texts):
    out = np.empty((len(texts), 384), dtype=np.float32)
    for i in range(0, len(texts), BATCH):
        b = texts[i:i+BATCH]
        enc = tok(b, padding=True, truncation=True, max_length=192, return_tensors="pt").to(dev)
        with torch.no_grad():
            h = mdl(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            v = torch.nn.functional.normalize(v, dim=1)
        out[i:i+len(b)] = v.cpu().numpy()
        if i % (BATCH*20) == 0:
            print(f"  embed {i}/{len(texts)}", flush=True)
    return out

print("loading squad...", flush=True)
sq = datasets.load_dataset("rajpurkar/squad", split="train")
seen, ctx, cmap = {}, [], {}
for row in sq:
    c = row["context"]
    if c not in seen:
        seen[c] = len(ctx); ctx.append(c)
    cmap.setdefault(id(row), None)
ctx = ctx[:N_CTX_CAP]
ctx_index = {c: i for i, c in enumerate(ctx)}
print(f"  {len(ctx)} unique squad contexts", flush=True)

# questions + their gold context row (only where the context survived the cap)
q_txt, q_gold = [], []
for row in sq:
    if len(q_txt) >= NQ: break
    gi = ctx_index.get(row["context"], -1)
    q_txt.append(row["question"]); q_gold.append(gi)
print(f"  {len(q_txt)} questions, {sum(1 for g in q_gold if g>=0)} with gold in corpus", flush=True)

print("loading ag_news to top up the corpus...", flush=True)
ag = datasets.load_dataset("fancyzhx/ag_news", split="train")
extra = [r["text"] for r in ag.select(range(min(N_AGNEWS, len(ag))))]
base_texts = ctx + extra
print(f"  corpus = {len(base_texts)} passages ({len(ctx)} squad + {len(extra)} agnews)", flush=True)

print("embedding corpus...", flush=True)
B = embed(base_texts)
print("embedding queries...", flush=True)
Q = embed(q_txt)

n, dim, nq = B.shape[0], B.shape[1], Q.shape[0]
print(f"n={n} dim={dim} nq={nq}  computing exact cosine top-{K}...", flush=True)
gt = np.empty((nq, K), dtype=np.uint64)
for i in range(0, nq, 64):
    sims = Q[i:i+64] @ B.T
    idx = np.argpartition(-sims, K, axis=1)[:, :K]
    for r in range(idx.shape[0]):
        o = idx[r][np.argsort(-sims[r, idx[r]])]
        gt[i+r] = o.astype(np.uint64)

with open(f"{OUT}/real_data.bin", "wb") as f:
    f.write(struct.pack("<IIII", dim, n, nq, K))
    f.write(B.tobytes()); f.write(Q.tobytes())
with open(f"{OUT}/real_gt.bin", "wb") as f:
    f.write(struct.pack("<II", nq, K)); f.write(gt.tobytes())
with open(f"{OUT}/real_gold.bin", "wb") as f:
    f.write(struct.pack("<I", nq))
    f.write(np.array(q_gold, dtype=np.int32).tobytes())

# gold-in-cosine-top-k sanity: how often the gold passage IS the cosine NN
hit = sum(1 for i in range(nq) if q_gold[i] >= 0 and q_gold[i] in set(int(x) for x in gt[i]))
haveg = sum(1 for g in q_gold if g >= 0)
pair = (B[np.random.randint(0, n, 4000)] * B[np.random.randint(0, n, 4000)]).sum(1)
print(f"wrote real_data.bin real_gt.bin real_gold.bin", flush=True)
print(f"  random-pair cosine mean={pair.mean():.3f} std={pair.std():.3f}", flush=True)
print(f"  gold passage in exact cosine top-{K}: {hit}/{haveg} ({100.0*hit/max(haveg,1):.1f}%)", flush=True)
