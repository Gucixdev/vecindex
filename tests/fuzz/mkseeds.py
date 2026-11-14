#!/usr/bin/env python3
"""Seed inputs for the fuzz targets: one request of every kind in each protocol, for a store of dimension 4.

    python3 tests/fuzz/mkseeds.py          writes tests/fuzz/proto/*

The fuzzer starts from these and mutates; inputs it finds that crash go next to them as regression cases."""
import os, struct

D = 4
here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "proto")
os.makedirs(out, exist_ok=True)
vec = [0.1, -0.2, 0.3, 0.4]
fv = struct.pack("<%df" % D, *vec)
js = "[0.1, -0.2, 0.3, 0.4]"
seeds = {}

def frame(op, k, payload):
    return struct.pack("<BBHI", 0xB5, op, k, len(payload)) + payload

seeds["bin_add_search"] = frame(2, 0, struct.pack("<q", 1) + fv) + frame(2, 0, struct.pack("<q", 2) + fv[::-1]) + frame(1, 3, fv)
seeds["bin_get_del"] = frame(2, 0, struct.pack("<q", 7) + fv) + frame(8, 0, struct.pack("<q", 7)) + frame(3, 0, struct.pack("<q", 7)) + frame(8, 0, struct.pack("<q", 7))
seeds["bin_maint"] = frame(4, 0, b"") + frame(5, 0, b"") + frame(6, 0, b"") + frame(7, 0, b"")
seeds["bin_range_delrange"] = frame(2, 0, struct.pack("<q", 3) + fv) + frame(12, 10, struct.pack("<qq", 0, 100)) + frame(9, 0, struct.pack("<qq", 0, 100))
seeds["bin_text"] = frame(10, 0, struct.pack("<q", 5) + fv + b"hello world") + frame(11, 0, struct.pack("<q", 5)) \
    + frame(13, 3, struct.pack("<I", 1) + fv + b"hello") + frame(13, 3, struct.pack("<I", 0) + b"world")
seeds["bin_use_unknown"] = frame(14, 0, b"acme") + frame(99, 0, b"") + frame(1, 0, b"short")
seeds["line_basic"] = b"ping\nstats\nadd 1 0.1 -0.2 0.3 0.4\nadd 2 0.5 0.1 -0.9 0.2\nsearch 2 0.1 -0.2 0.3 0.4\n"
seeds["line_get_del"] = b"add 7 1 0 0 0\nget 7\ndel 7\nget 7\ndel 1 100\n"
seeds["line_batch"] = b"begin\nadd 3 0 1 0 0\ndel 3\ncommit\nbegin\nadd 4 0 0 1 0\nabort\ncommit sync\n"
seeds["line_text"] = b"addt 5 0.1 -0.2 0.3 0.4 hello\\nworld \\t tab\ntext 5\nwords 3 hello\nhybrid 3 0.1 -0.2 0.3 0.4 hello world\nrange 0 100 10\n"
seeds["line_maint"] = b"build\nsave\nsnapshot\nhelp\nuse acme\nquit\nping\n"
seeds["line_crlf"] = b"add 9 0.1 0.2 0.3 0.4\r\nsearch 1 0.1 0.2 0.3 0.4\r\n"

def http(method, path, body=b"", ctype=None, keep=True):
    h = "%s %s HTTP/1.1\r\nHost: x\r\n" % (method, path)
    if ctype:
        h += "Content-Type: %s\r\n" % ctype
    if body or method == "POST":
        h += "Content-Length: %d\r\n" % len(body)
    if not keep:
        h += "Connection: close\r\n"
    return (h + "\r\n").encode() + body

seeds["http_health_stats"] = http("GET", "/health") + http("GET", "/stats")
seeds["http_add_search"] = http("POST", "/add?id=3", js.encode()) + http("POST", "/search?k=3", js.encode()) + http("GET", "/get?id=3")
seeds["http_json_text"] = http("POST", "/add?id=4", b'{"vector": %s, "text": "hi \\"there\\"\\n"}' % js.encode()) \
    + http("GET", "/text?id=4") + http("POST", "/hybrid?k=3", b'{"vector": %s, "text": "hi"}' % js.encode())
seeds["http_octet"] = http("POST", "/add?id=6", fv, "application/octet-stream") + http("POST", "/search?k=2", fv, "application/octet-stream")
seeds["http_delete_range"] = http("GET", "/range?lo=0&hi=10") + http("POST", "/delete?id=3") + http("POST", "/delete?lo=0&hi=9")
seeds["http_maint"] = http("POST", "/build") + http("POST", "/save") + http("POST", "/snapshot") + http("GET", "/nothing", keep=False)
seeds["http_tenant"] = http("POST", "/t/acme/search?k=1", js.encode())
seeds["mixed"] = seeds["line_basic"] + seeds["bin_add_search"] + seeds["http_add_search"]

for name, data in seeds.items():
    with open(os.path.join(out, name), "wb") as f:
        f.write(data)
print("%d seeds in %s" % (len(seeds), out))
