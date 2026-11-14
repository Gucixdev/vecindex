#!/usr/bin/env python3
"""Cross-encoder reranking server for vecdb: keeps a model loaded and scores (query, candidate)
pairs over stdin/stdout - a second-stage pass over the hybrid search's own top-K, the piece
README's "Honest gaps" called out as missing (RRF fusion is a real substitute for a lot of what
reranking buys, but not the same as an actual cross-encoder reading query and candidate together).

  python3 rerank_srv.py [model]   default cross-encoder/ms-marco-MiniLM-L-6-v2 (English; MS MARCO
                                   trained). For multilingual content swap in e.g.
                                   cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 - this file does not
                                   hardcode the language, same as embedsrv.py's own model arg.

Environment: RERANK_DEVICE=cuda (default cpu), RERANK_MAXLEN tokens per pair (default 256),
RERANK_THREADS.

Protocol (native endian, see rerank.c): the server first writes u8 1 (ready, one byte so a
mismatched embedsrv.py can't be started here by mistake - that protocol writes a u32 dim first).
Then per request: u32 qlen, qlen bytes (the query), u32 ncand, then ncand x (u32 clen, clen bytes)
(the candidates, in the order to score). Reply: u8 status (0 ok, 1 failed - the client must not
trust any scores that follow a 1), then (status 0) ncand x f32 (relevance score, higher = more
relevant, NOT necessarily in [0,1] - a raw cross-encoder logit unless the model itself applies a
sigmoid; ms-marco-MiniLM does not, so treat scores as a ranking signal, not a probability)."""
import os, struct, sys, warnings

if os.environ.get("RERANK_DEVICE", "cpu") != "cuda":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~/.cache/vecdb/pylibs"))
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.set_num_threads(int(os.environ.get("RERANK_THREADS", os.cpu_count() or 4)))
name = sys.argv[1] if len(sys.argv) > 1 else "cross-encoder/ms-marco-MiniLM-L-6-v2"
dev = os.environ.get("RERANK_DEVICE", "cpu")
maxlen = int(os.environ.get("RERANK_MAXLEN", "256"))

tok = AutoTokenizer.from_pretrained(name)
mdl = AutoModelForSequenceClassification.from_pretrained(name).eval().to(dev)

stdin = sys.stdin.buffer
stdout = sys.stdout.buffer


def read_exact(n):
    buf = b""
    while len(buf) < n:
        chunk = stdin.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def read_str():
    hdr = read_exact(4)
    if hdr is None:
        return None
    n = struct.unpack("<I", hdr)[0]
    if n == 0:
        return ""
    b = read_exact(n)
    if b is None:
        return None
    return b.decode("utf-8", "replace")


stdout.write(struct.pack("<B", 1))
stdout.flush()

while True:
    q = read_str()
    if q is None:
        break
    nch = read_exact(4)
    if nch is None:
        break
    ncand = struct.unpack("<I", nch)[0]
    cands = []
    ok = True
    for _ in range(ncand):
        c = read_str()
        if c is None:
            ok = False
            break
        cands.append(c)
    if not ok or len(cands) != ncand:
        break
    try:
        pairs = [[q, c] for c in cands]
        scores = []
        B = 32
        with torch.no_grad():
            for i in range(0, len(pairs), B):
                batch = pairs[i:i + B]
                enc = tok(batch, padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to(dev)
                out = mdl(**enc).logits.squeeze(-1)
                if out.dim() == 0:
                    out = out.unsqueeze(0)
                scores.extend(out.float().cpu().tolist())
        stdout.write(struct.pack("<B", 0))
        stdout.write(struct.pack("<%df" % len(scores), *scores))
        stdout.flush()
    except Exception as e:
        print(f"rerank_srv: request failed: {e}", file=sys.stderr, flush=True)
        stdout.write(struct.pack("<B", 1))
        stdout.flush()
