#!/usr/bin/env python3
"""End-to-end test of vserv: both protocols, both socket kinds, persistence, bad input, concurrency.

    python3 bench/vserv_test.py [path/to/vserv]

Starts its own server on a temporary store; exits non-zero on the first failure."""
import json, os, random, signal, socket, struct, subprocess, sys, tempfile, threading, time

VSERV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", ".build", "vserv")
DIM, N = 32, 2000
tmp = tempfile.mkdtemp(prefix="vserv-test-")
SOCK, STORE = os.path.join(tmp, "v.sock"), os.path.join(tmp, "v.vxi")
PORT = 20000 + os.getpid() % 20000
rng = random.Random(7)
vecs = [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(N)]
fails = 0


def check(cond, what):
    global fails
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        fails += 1


def start():
    p = subprocess.Popen([VSERV, "-s", SOCK, "-p", str(PORT), "-t", "2", "-d", str(DIM), "-x", STORE], stderr=subprocess.PIPE)
    for _ in range(200):
        if os.path.exists(SOCK):
            try:
                socket.create_connection(("127.0.0.1", PORT), timeout=1).close()
                return p
            except OSError:
                pass
        time.sleep(0.02)
    raise SystemExit("vserv did not start: " + p.stderr.read().decode(errors="replace"))


socket.setdefaulttimeout(20)


def unix():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(SOCK)
    return s


def recv_exact(s, n):
    b = b""
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c:
            raise ConnectionError("closed")
        b += c
    return b


def bin_call(s, op, k=0, payload=b""):
    s.sendall(struct.pack("<BBHI", 0xB5, op, k, len(payload)) + payload)
    magic, status, count, nb = struct.unpack("<BBHI", recv_exact(s, 8))
    return status, count, recv_exact(s, nb) if nb else b""


def bin_search(s, v, k):
    st, cnt, body = bin_call(s, 1, k, struct.pack("<%df" % DIM, *v))
    return st, [struct.unpack_from("<qf", body, 12 * i) for i in range(cnt)]


def http(method, path, body=None, raw=False, sock=None):
    s = sock or socket.create_connection(("127.0.0.1", PORT))
    data = b"" if body is None else (struct.pack("<%df" % DIM, *body) if raw else json.dumps(body).encode())
    head = "%s %s HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n" % (method, path, len(data))
    if raw:
        head += "Content-Type: application/octet-stream\r\n"
    s.sendall(head.encode() + b"\r\n" + data)
    resp = read_http(s)
    if sock is None:
        s.close()
    return resp


_left = {}   # bytes read past the end of the previous response, per socket (pipelined answers can share a recv)


def read_http(s):
    buf = _left.pop(s, b"")
    while b"\r\n\r\n" not in buf:
        c = s.recv(65536)
        if not c:
            raise ConnectionError("closed")
        buf += c
    head, rest = buf.split(b"\r\n\r\n", 1)
    lines = head.decode().split("\r\n")
    code = int(lines[0].split()[1])
    clen = int([l.split(":")[1] for l in lines if l.lower().startswith("content-length")][0])
    while len(rest) < clen:
        c = s.recv(65536)
        if not c:
            raise ConnectionError("closed")
        rest += c
    if len(rest) > clen:
        _left[s] = rest[clen:]
    return code, json.loads(rest[:clen]) if clen else None


print("vserv end-to-end (%s)" % VSERV)
p = start()
try:
    # binary protocol over the Unix socket: bulk add, build, search
    s = unix()
    t0 = time.time()
    for i, v in enumerate(vecs):
        st, _, _ = bin_call(s, 2, 0, struct.pack("<q%df" % DIM, i, *v))
        if st != 0:
            break
    check(st == 0, "binary: %d adds over the Unix socket (%.0f/s)" % (N, N / (time.time() - t0)))
    check(bin_call(s, 4)[0] == 0, "binary: build")
    st, hits = bin_search(s, vecs[17], 5)
    check(st == 0 and hits and hits[0][0] == 17, "binary: search finds row 17 first %s" % (hits[:2],))
    st, cnt, body = bin_call(s, 6)
    live = struct.unpack("<4q", body)[0] if body else -1
    check(live == N, "binary: stats live=%d" % live)
    check(bin_call(s, 1, 5, b"\0" * 8)[0] == 2, "binary: wrong vector size -> status 2")
    check(bin_call(s, 99)[0] == 3, "binary: unknown op -> status 3")
    s.close()

    # HTTP over TCP
    check(http("GET", "/health") == (200, {"ok": True}), "http: GET /health")
    code, st = http("GET", "/stats")
    check(code == 200 and st["count"] == N and st["dim"] == DIM and st["built"], "http: GET /stats %s" % st)
    code, r = http("POST", "/search?k=3", vecs[42])
    check(code == 200 and r["ids"][0] == 42 and len(r["ids"]) == 3, "http: POST /search (JSON) -> %s" % r["ids"])
    code, r = http("POST", "/search?k=2", vecs[43], raw=True)
    check(code == 200 and r["ids"][0] == 43, "http: POST /search (raw f32) -> %s" % r["ids"])
    newv = [rng.uniform(-1, 1) for _ in range(DIM)]
    check(http("POST", "/add?id=5000", newv)[0] == 200, "http: POST /add?id=5000")
    code, r = http("POST", "/search?k=1", newv)
    check(code == 200 and r["ids"] == [5000], "http: the added row is searchable before a build (pending tail)")
    check(http("POST", "/delete?id=3") == (200, {"removed": 1}), "http: POST /delete?id=3")
    code, r = http("POST", "/search?k=10", vecs[3])
    check(code == 200 and 3 not in r["ids"], "http: deleted row gone")
    check(http("POST", "/search", [1.0, 2.0])[0] == 400, "http: short vector -> 400")
    check(http("POST", "/search", "nonsense")[0] == 400, "http: not a vector -> 400")
    check(http("GET", "/nope")[0] == 404, "http: unknown path -> 404")
    # pipelining: two requests in one write, two answers in order
    s = socket.create_connection(("127.0.0.1", PORT))
    s.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\nGET /stats HTTP/1.1\r\nHost: x\r\n\r\n")
    a, b = read_http(s), read_http(s)
    check(a[0] == 200 and b[0] == 200 and "count" in b[1], "http: two pipelined requests answered in order")
    # a request split across many tiny writes
    req = b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n"
    for ch in req:
        s.sendall(bytes([ch]))
        time.sleep(0.001)
    check(read_http(s)[0] == 200, "http: request trickled in byte by byte")
    s.close()
    # an oversized body is refused before it is read
    s = socket.create_connection(("127.0.0.1", PORT))
    s.sendall(b"POST /search HTTP/1.1\r\nContent-Length: 999999999\r\n\r\n")
    check(read_http(s)[0] == 413, "http: huge Content-Length -> 413")
    s.close()

    # concurrency: 8 clients x 150 searches on keep-alive connections, mixed protocols
    errs = []

    def client(t):
        try:
            if t % 2:
                c = unix()
                for i in range(150):
                    j = (t * 151 + i) % N
                    if j == 3:
                        continue
                    st, h = bin_search(c, vecs[j], 1)
                    if st or not h or h[0][0] != j:
                        errs.append((t, j, h))
            else:
                c = socket.create_connection(("127.0.0.1", PORT))
                for i in range(150):
                    j = (t * 151 + i) % N
                    if j == 3:
                        continue
                    code, r = http("POST", "/search?k=1", vecs[j], raw=True, sock=c)
                    if code != 200 or r["ids"] != [j]:
                        errs.append((t, j, r))
            c.close()
        except Exception as e:  # noqa: BLE001
            errs.append((t, repr(e)))

    th = [threading.Thread(target=client, args=(t,)) for t in range(8)]
    t0 = time.time()
    [x.start() for x in th]
    [x.join() for x in th]
    check(not errs, "concurrency: 8 clients x 150 searches, %.0f queries/s, errors %s" % (1200 / (time.time() - t0), errs[:3]))
finally:
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=30)
check(p.returncode == 0 and os.path.exists(STORE), "SIGTERM: clean exit, store saved (-x)")
check(not os.path.exists(SOCK), "socket file removed on exit")

# restart: everything comes back from the file
p = start()
try:
    code, st = http("GET", "/stats")
    check(code == 200 and st["count"] == N, "restart: count %s (added one, deleted one)" % st.get("count"))
    code, r = http("POST", "/search?k=1", newv)
    check(code == 200 and r["ids"] == [5000], "restart: row added over HTTP survived")
    code, r = http("POST", "/search?k=10", vecs[3])
    check(code == 200 and 3 not in r["ids"], "restart: deleted row stays deleted")
finally:
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=30)
print("vserv:", "OK" if not fails else "%d FAILED" % fails)
sys.exit(1 if fails else 0)
