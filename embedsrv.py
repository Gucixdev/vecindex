#!/usr/bin/env python3
"""Embedding server for vecdb: keeps a model loaded and answers over stdin/stdout.

  python3 embedsrv.py [text-model]          default intfloat/multilingual-e5-small (384 d, Polish ok)
  python3 embedsrv.py --clip [clip-model]   default openai/clip-vit-base-patch32 (512 d, images + text)

Environment: EMBED_DEVICE=cuda (default cpu), EMBED_MAXLEN tokens per text (default 128), EMBED_INT8=1 (cpu),
EMBED_THREADS, EMBED_BATCH.

Protocol (native endian, see vecdb.c): the server first writes u32 dim; then per request
  u8 kind ('D' document, 'Q' query, 'I' image path, 'B'/'C' batch of documents/queries), u32 len, len bytes
(a batch payload is u32 count, then count x (u32 len, bytes)) and answers u8 status: 0 then the dim float32
vectors (unit length), or 1 and nothing (the request failed; the client must not store zeros).
Run it with the interpreter that has torch and transformers; Pillow is only needed for --clip (it is picked
up from ~/.cache/vecdb/pylibs if you pip install --target there)."""
import os, struct, sys, warnings

if os.environ.get("EMBED_DEVICE", "cpu") != "cuda":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""    # CPU unless asked for (EMBED_DEVICE=cuda): the GPU is the desktop's
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~/.cache/vecdb/pylibs"))
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

torch.set_num_threads(int(os.environ.get("EMBED_THREADS", os.cpu_count() or 4)))
args = sys.argv[1:]
clip = bool(args) and args[0] == "--clip"
if clip:
    args = args[1:]
name = args[0] if args else ("openai/clip-vit-base-patch32" if clip else "intfloat/multilingual-e5-small")
dev = os.environ.get("EMBED_DEVICE", "cpu")


def norm(v):
    v = v.astype(np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


if clip:
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(name).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(name, use_fast=False)
    dim = model.config.projection_dim

    def embed(kind, payload):
        with torch.no_grad():
            if kind == "I":
                img = Image.open(payload.decode("utf-8", "replace")).convert("RGB")
                x = proc(images=img, return_tensors="pt").to(dev)
                return norm(model.get_image_features(**x)[0].cpu().numpy())
            x = proc(text=[payload.decode("utf-8", "replace")], return_tensors="pt", padding=True, truncation=True, max_length=77).to(dev)
            return norm(model.get_text_features(**x)[0].cpu().numpy())

    embed_many = None
else:
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(dev).eval()
    dim = model.config.hidden_size
    e5 = "e5" in name.lower()
    maxlen = int(os.environ.get("EMBED_MAXLEN", "128"))
    if os.environ.get("EMBED_INT8") == "1":    # dynamic int8 linear layers: ~20% faster on CPU, tiny quality cost
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    # texts per forward pass: about 4096 tokens in flight, so a GPU shared with the desktop (a few hundred MB
    # free) does not run out of memory at long lengths
    BS = int(os.environ.get("EMBED_BATCH", str(max(1, min(32, 4096 // maxlen)))))

    def forward(texts):
        x = tok(texts, return_tensors="pt", truncation=True, max_length=maxlen, padding=True).to(dev)
        with torch.no_grad():
            h = model(**x).last_hidden_state
        m = x["attention_mask"].unsqueeze(-1).to(h.dtype)
        return ((h * m).sum(1) / m.sum(1)).cpu().numpy()

    def forward_safe(texts):
        """out of memory: free the cache and retry in halves; a single text that still fails raises"""
        try:
            return forward(texts)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(texts) == 1:
                raise
            h = len(texts) // 2
            return np.concatenate([forward_safe(texts[:h]), forward_safe(texts[h:])])

    def embed_many(kind, payloads):
        """unit vectors for many texts; sorted by length so batches carry little padding"""
        texts = [p.decode("utf-8", "replace") for p in payloads]
        if e5:
            pre = "query: " if kind == "Q" else "passage: "
            texts = [pre + t for t in texts]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [None] * len(texts)
        for s in range(0, len(order), BS):
            idx = order[s:s + BS]
            v = forward_safe([texts[i] for i in idx])
            for j, i in enumerate(idx):
                out[i] = norm(v[j])
        return out

    def embed(kind, payload):
        return embed_many(kind, [payload])[0]


out = sys.stdout.buffer
inp = sys.stdin.buffer
out.write(struct.pack("=I", dim)); out.flush()


def reply(vs):
    out.write(b"\x00" + b"".join(v.astype(np.float32).tobytes() for v in vs)); out.flush()


def fail(msg):
    sys.stderr.write("embedsrv: %s\n" % msg)
    out.write(b"\x01"); out.flush()            # status 1: no vectors follow; the client must not store zeros


while True:
    hdr = inp.read(5)
    if len(hdr) < 5:
        break
    kind, n = hdr[:1].decode(), struct.unpack("=I", hdr[1:])[0]
    payload = inp.read(n) if n else b""
    try:
        if kind in ("B", "C") and embed_many is not None:
            cnt = struct.unpack_from("=I", payload, 0)[0]; pos = 4; items = []
            for _ in range(cnt):
                ln = struct.unpack_from("=I", payload, pos)[0]; pos += 4
                items.append(payload[pos:pos + ln]); pos += ln
            reply(embed_many("Q" if kind == "C" else "D", items))
        else:
            reply([embed(kind, payload)])
    except Exception as ex:                    # a bad image or an unrecoverable OOM must not kill the server
        fail(str(ex)[:200])
