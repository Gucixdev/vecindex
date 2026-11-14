#!/usr/bin/env python3
"""Real before/after measurement for rerank_srv.py's cross-encoder, against real_val.c's own
SQuAD-question benchmark data (real_gt.bin top-10 cosine neighbours, real_gold.bin gold passage
row, real_texts.json for the actual text - see real_texts.py). Only meaningful on queries where
the gold passage IS already in the top-10 cosine neighbours (a rerank stage can only reorder
what vector search already retrieved - it cannot fix a retrieval miss, so those are reported
separately, not folded into the "did reranking help" number).

Metric: MRR (mean reciprocal rank of the gold passage within its own top-10) and gold@1 (fraction
where gold is EXACTLY the top pick), before (raw cosine order) and after (cross-encoder rerank).
"""
import json, os, struct, sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DIR = sys.argv[1] if len(sys.argv) > 1 else "."

with open(f"{DIR}/real_gt.bin", "rb") as f:
    nq, k = struct.unpack("<II", f.read(8))
    gt = struct.unpack(f"<{nq*k}Q", f.read(8 * nq * k))
with open(f"{DIR}/real_gold.bin", "rb") as f:
    (ngold,) = struct.unpack("<I", f.read(4))
    gold = struct.unpack(f"<{ngold}i", f.read(4 * ngold))
with open(f"{DIR}/real_texts.json") as f:
    texts = json.load(f)
corpus, queries = texts["corpus"], texts["queries"]

print(f"nq={nq} k={k} corpus={len(corpus)} queries={len(queries)}", flush=True)

cand_lists = []
for q in range(nq):
    row = list(gt[q * k:(q + 1) * k])
    g = gold[q]
    if g >= 0 and g in row:
        cand_lists.append((q, row, row.index(g)))  # 0-based rank of gold in raw cosine order

print(f"queries with gold in top-{k} (rerank-eligible): {len(cand_lists)}/{nq}", flush=True)

name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
print(f"loading {name}...", flush=True)
tok = AutoTokenizer.from_pretrained(name)
mdl = AutoModelForSequenceClassification.from_pretrained(name).eval()

before_mrr = sum(1.0 / (r + 1) for _, _, r in cand_lists) / len(cand_lists)
before_at1 = sum(1 for _, _, r in cand_lists if r == 0) / len(cand_lists)

after_ranks = []
B = 16
with torch.no_grad():
    for start in range(0, len(cand_lists), B):
        chunk = cand_lists[start:start + B]
        for q, row, gold_r in chunk:
            qtext = queries[q]
            pairs = [[qtext, corpus[c]] for c in row]
            enc = tok(pairs, padding=True, truncation=True, max_length=256, return_tensors="pt")
            scores = mdl(**enc).logits.squeeze(-1).tolist()
            order = sorted(range(len(row)), key=lambda i: -scores[i])
            new_rank = order.index(gold_r)
            after_ranks.append(new_rank)
        if start % (B * 10) == 0:
            print(f"  reranked {start}/{len(cand_lists)}", flush=True)

after_mrr = sum(1.0 / (r + 1) for r in after_ranks) / len(after_ranks)
after_at1 = sum(1 for r in after_ranks if r == 0) / len(after_ranks)

print()
print(f"gold-in-top-{k} but not rank-1 before rerank (rerank's actual opportunity): "
      f"{sum(1 for _,_,r in cand_lists if r>0)}/{len(cand_lists)}")
print(f"BEFORE rerank (raw cosine order):  MRR={before_mrr:.4f}  gold@1={before_at1*100:.1f}%")
print(f"AFTER  rerank (cross-encoder):     MRR={after_mrr:.4f}  gold@1={after_at1*100:.1f}%")
print(f"delta: MRR {after_mrr-before_mrr:+.4f}  gold@1 {(after_at1-before_at1)*100:+.1f} points")
