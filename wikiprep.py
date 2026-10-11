#!/usr/bin/env python3
"""Turn a Wikipedia XML dump into the record stream vbench_wiki reads.

  bzcat simplewiki.xml.bz2 | python3 wikiprep.py > wiki.rec  [max_articles]

Record: "<title>\\n<byte length>\\n<text>\\n". The text is plain wikitext reduced to prose:
[[links]] kept as wikilinks (so they become graph edges), [[Category:X]] turned into #X
tags, templates/tables/refs/html stripped. Redirects and non-article namespaces skipped."""
import re, sys, xml.etree.ElementTree as ET

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
tpl = re.compile(r"\{\{[^{}]*\}\}")
ref = re.compile(r"<ref[^>]*?/>|<ref[^>]*>.*?</ref>", re.S)
tag = re.compile(r"<[^>]+>")
cat = re.compile(r"\[\[Category:([^\]|]+)[^\]]*\]\]", re.I)
filelink = re.compile(r"\[\[(?:File|Image|Media):[^\]]*(?:\[\[[^\]]*\]\][^\]]*)*\]\]", re.I)
other = re.compile(r"\[\[[a-z]{2,3}:[^\]]*\]\]")
tbl = re.compile(r"\{\|.*?\|\}", re.S)
head = re.compile(r"^=+\s*(.*?)\s*=+\s*$", re.M)
quote = re.compile(r"'{2,}")

def clean(t):
    tags = ["#" + re.sub(r"[^\w]+", "_", c.strip().lower()) for c in cat.findall(t)]
    t = cat.sub("", t)
    t = ref.sub("", t)
    t = tbl.sub("", t)
    for _ in range(4):
        t = tpl.sub("", t)
    t = filelink.sub("", t)
    t = other.sub("", t)
    t = tag.sub("", t)
    t = head.sub(r"\1.", t)
    t = quote.sub("", t)
    t = re.sub(r"\n{2,}", "\n", t).strip()
    return t, tags

n = 0
out = sys.stdout.buffer
for _, el in ET.iterparse(sys.stdin.buffer, events=("end",)):
    if not el.tag.endswith("page"):
        continue
    ns = el.find("./{*}ns")
    title = el.findtext("./{*}title")
    text = el.findtext("./{*}revision/{*}text") or ""
    el.clear()
    if ns is None or ns.text != "0" or text.lstrip().lower().startswith("#redirect") or len(text) < 400:
        continue
    body, tags = clean(text)
    if len(body) < 300:
        continue
    body = body[:6000] + "\n" + " ".join(sorted(set(tags))[:12])
    b = body.encode("utf-8")
    out.write(title.encode("utf-8") + b"\n" + str(len(b)).encode() + b"\n" + b + b"\n")
    n += 1
    if n >= limit:
        break
sys.stderr.write("%d articles\n" % n)
