#!/usr/bin/env python3
"""Companion to real_gen.py: reconstructs the SAME corpus/query text lists, in the SAME order,
so they line up with real_data.bin's rows/real_gold.bin's query indices - real_gen.py itself
never saved the raw text (only vectors), and re-running it would mean re-embedding everything.
This only re-walks the (now HF-cache-local, so fast) datasets loading code, no model involved.
Writes real_texts.json: {"corpus": [...], "queries": [...]} (same N_CTX_CAP/N_AGNEWS/NQ as
real_gen.py - keep them in sync if either changes).
"""
import json, sys, datasets

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
N_CTX_CAP = 40000
N_AGNEWS = 30000
NQ = 1000

print("loading squad...", flush=True)
sq = datasets.load_dataset("rajpurkar/squad", split="train")
seen, ctx = {}, []
for row in sq:
    c = row["context"]
    if c not in seen:
        seen[c] = len(ctx)
        ctx.append(c)
ctx = ctx[:N_CTX_CAP]
ctx_index = {c: i for i, c in enumerate(ctx)}

q_txt = []
for row in sq:
    if len(q_txt) >= NQ:
        break
    q_txt.append(row["question"])

print("loading ag_news...", flush=True)
ag = datasets.load_dataset("fancyzhx/ag_news", split="train")
extra = [r["text"] for r in ag.select(range(min(N_AGNEWS, len(ag))))]
corpus = ctx + extra

with open(f"{OUT}/real_texts.json", "w") as f:
    json.dump({"corpus": corpus, "queries": q_txt}, f)
print(f"wrote real_texts.json: {len(corpus)} corpus texts, {len(q_txt)} queries", flush=True)
