#!/usr/bin/env python3
"""TLS end-to-end test of vserv: HTTPS and the binary protocol inside TLS, plaintext refused on the TLS port,
session resumption, large replies across many records, concurrency, mutual TLS (client certificates), an RSA
server key, and curl.

    python3 bench/vserv_tls_test.py [path/to/vserv]

Makes its own throwaway CA and certificates with the openssl command line tool (skips if there is none)."""
import json, os, random, shutil, signal, socket, ssl, struct, subprocess, sys, tempfile, threading, time

VSERV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", ".build", "vserv")
if not shutil.which("openssl"):
    print("vserv tls: openssl command not found - skipped")
    sys.exit(0)
DIM, N = 16, 500
tmp = tempfile.mkdtemp(prefix="vserv-tls-")
P = lambda f: os.path.join(tmp, f)
PORT = 21000 + os.getpid() % 20000
socket.setdefaulttimeout(20)
rng = random.Random(3)
vecs = [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(N)]
fails = 0


def check(cond, what):
    global fails
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        fails += 1


def sh(*a):
    subprocess.run(a, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# a CA, a server certificate for localhost / 127.0.0.1, a client certificate, an RSA server pair, a stranger CA
CA_EXT = ["-addext", "basicConstraints=critical,CA:TRUE", "-addext", "keyUsage=critical,keyCertSign,cRLSign"]
sh("openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes", "-keyout", P("ca.key"),
   "-out", P("ca.pem"), "-days", "2", "-subj", "/CN=vecindex test CA", *CA_EXT)
KEYID = "subjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid,issuer\n"
open(P("san.ext"), "w").write("subjectAltName=DNS:localhost,IP:127.0.0.1\nkeyUsage=critical,digitalSignature,keyEncipherment\n"
                              "extendedKeyUsage=serverAuth\n" + KEYID)
open(P("cli.ext"), "w").write("keyUsage=critical,digitalSignature\nextendedKeyUsage=clientAuth\n" + KEYID)
for name, ext, newkey in (("server", "san.ext", ["ec", "-pkeyopt", "ec_paramgen_curve:P-256"]),
                          ("client", "cli.ext", ["ec", "-pkeyopt", "ec_paramgen_curve:P-256"]),
                          ("rsa", "san.ext", ["rsa:2048"])):
    sh("openssl", "req", "-newkey", *newkey, "-nodes", "-keyout", P(name + ".key"), "-out", P(name + ".csr"),
       "-subj", "/CN=" + ("localhost" if name != "client" else "trusted client"))
    sh("openssl", "x509", "-req", "-in", P(name + ".csr"), "-CA", P("ca.pem"), "-CAkey", P("ca.key"), "-CAcreateserial",
       "-out", P(name + ".pem"), "-days", "2", "-extfile", P(ext))
sh("openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes", "-keyout", P("evil.key"),
   "-out", P("evil.pem"), "-days", "2", "-subj", "/CN=trusted client", "-addext", "keyUsage=critical,digitalSignature")   # same name, wrong issuer


def start(*extra):
    store = P("t%d.vxi" % random.randrange(1 << 30))
    p = subprocess.Popen([VSERV, "-p", str(PORT), "-s", P("v.sock"), "-t", "2", "-d", str(DIM), *extra, store], stderr=subprocess.PIPE)
    for _ in range(300):
        try:
            socket.create_connection(("127.0.0.1", PORT), timeout=1).close()
            return p
        except OSError:
            time.sleep(0.02)
    raise SystemExit("vserv did not start: " + p.stderr.read().decode(errors="replace"))


def stop(p):
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=30)


def ctx(client=None):
    c = ssl.create_default_context(cafile=P("ca.pem"))
    if client:
        c.load_cert_chain(P(client + ".pem"), P(client + ".key"))
    return c


def tls_conn(c=None, session=None):
    s = (c or ctx()).wrap_socket(socket.create_connection(("127.0.0.1", PORT)), server_hostname="localhost", session=session)
    return s


_left = {}


def read_http(s):
    buf = _left.pop(s, b"")
    while b"\r\n\r\n" not in buf:
        c = s.recv(65536)
        if not c:
            raise ConnectionError("closed")
        buf += c
    head, rest = buf.split(b"\r\n\r\n", 1)
    lines = head.decode().split("\r\n")
    clen = int([l.split(":")[1] for l in lines if l.lower().startswith("content-length")][0])
    while len(rest) < clen:
        c = s.recv(65536)
        if not c:
            raise ConnectionError("closed")
        rest += c
    if len(rest) > clen:
        _left[s] = rest[clen:]
    return int(lines[0].split()[1]), json.loads(rest[:clen])


def http(s, method, path, body=None, raw=False):
    data = b"" if body is None else (struct.pack("<%df" % DIM, *body) if raw else json.dumps(body).encode())
    head = "%s %s HTTP/1.1\r\nHost: localhost\r\nContent-Length: %d\r\n" % (method, path, len(data))
    if raw:
        head += "Content-Type: application/octet-stream\r\n"
    s.sendall(head.encode() + b"\r\n" + data)
    return read_http(s)


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
    _, status, count, nb = struct.unpack("<BBHI", recv_exact(s, 8))
    return status, count, recv_exact(s, nb) if nb else b""


print("vserv TLS end-to-end (%s)" % VSERV)
p = start("-c", P("server.pem"), "-k", P("server.key"))
try:
    C = ctx()                                          # one context: a session can only be resumed through it
    s = tls_conn(C)
    check(s.version() == "TLSv1.2", "handshake: %s, cipher %s" % (s.version(), s.cipher()[0]))
    check(any(x in s.cipher()[0] for x in ("GCM", "CHACHA20")) and "ECDHE" in s.cipher()[0], "AEAD suite with ECDHE (forward secrecy)")
    check(http(s, "GET", "/health") == (200, {"ok": True}), "HTTPS: GET /health")
    for i, v in enumerate(vecs):                       # the binary protocol inside the same TLS connection
        st, _, _ = bin_call(s, 2, 0, struct.pack("<q%df" % DIM, i, *v))
        if st:
            break
    check(st == 0, "binary over TLS: %d adds on one connection" % N)
    check(bin_call(s, 4)[0] == 0, "binary over TLS: build")
    st, cnt, body = bin_call(s, 1, 3, struct.pack("<%df" % DIM, *vecs[7]))
    check(st == 0 and struct.unpack_from("<q", body, 0)[0] == 7, "binary over TLS: search finds row 7")
    code, r = http(s, "POST", "/search?k=400", vecs[9])      # a ~15 KB JSON reply: several TLS records
    check(code == 200 and r["ids"][0] == 9 and len(r["ids"]) == 400, "HTTPS: k=400 reply (%d ids) across records" % len(r.get("ids", [])))
    sess = s.session
    s.close()
    s2 = tls_conn(C, session=sess)
    check(s2.session_reused, "session resumed from the server's cache")
    check(http(s2, "GET", "/stats")[1]["count"] == N, "HTTPS on the resumed session")
    s2.close()
    # plaintext on the TLS port is refused; the Unix socket stays plaintext
    raw = socket.create_connection(("127.0.0.1", PORT))
    raw.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
    code, r = read_http(raw)
    check(code == 400 and "TLS" in r["error"], "plaintext HTTP on the TLS port -> 400 %s" % r)
    raw.close()
    u = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    u.connect(P("v.sock"))
    check(http(u, "GET", "/health")[0] == 200, "Unix socket: plaintext as before")
    u.close()
    # a client that does not trust our CA fails the handshake
    try:
        ssl.create_default_context().wrap_socket(socket.create_connection(("127.0.0.1", PORT)), server_hostname="localhost")
        check(False, "a client without our CA is refused")
    except ssl.SSLError:
        check(True, "a client without our CA rejects the certificate")
    # concurrency: 6 TLS clients x 100 searches
    errs = []

    def client(t):
        try:
            c = tls_conn()
            for i in range(100):
                j = (t * 101 + i) % N
                code, r = http(c, "POST", "/search?k=1", vecs[j], raw=True)
                if code != 200 or r["ids"] != [j]:
                    errs.append((t, j, r))
            c.close()
        except Exception as e:  # noqa: BLE001
            errs.append((t, repr(e)))

    th = [threading.Thread(target=client, args=(t,)) for t in range(6)]
    t0 = time.time()
    [x.start() for x in th]
    [x.join() for x in th]
    check(not errs, "6 TLS clients x 100 searches, %.0f queries/s, errors %s" % (600 / (time.time() - t0), errs[:2]))
    # curl, if present
    if shutil.which("curl"):
        out = subprocess.run(["curl", "-s", "--cacert", P("ca.pem"), "https://localhost:%d/health" % PORT], capture_output=True, text=True)
        check(out.stdout.strip() == '{"ok":true}', "curl --cacert https://localhost/health -> %s" % out.stdout.strip())
finally:
    stop(p)

# mutual TLS: only clients with a certificate from our CA
p = start("-c", P("server.pem"), "-k", P("server.key"), "-a", P("ca.pem"))
try:
    try:
        s = tls_conn()
        http(s, "GET", "/health")
        check(False, "mTLS: a client without a certificate is refused")
    except (ssl.SSLError, ConnectionError, OSError):
        check(True, "mTLS: a client without a certificate is refused")
    try:
        s = tls_conn(ctx("evil"))
        http(s, "GET", "/health")
        check(False, "mTLS: a self-signed look-alike certificate is refused")
    except (ssl.SSLError, ConnectionError, OSError):
        check(True, "mTLS: a self-signed look-alike certificate is refused")
    s = tls_conn(ctx("client"))
    check(http(s, "GET", "/health") == (200, {"ok": True}), "mTLS: a client with our certificate gets in")
    s.close()
finally:
    stop(p)

# an RSA server key
p = start("-c", P("rsa.pem"), "-k", P("rsa.key"))
try:
    s = tls_conn()
    check(http(s, "GET", "/health")[0] == 200 and "RSA" in s.cipher()[0], "RSA server key: %s" % s.cipher()[0])
    s.close()
finally:
    stop(p)
shutil.rmtree(tmp, ignore_errors=True)
print("vserv tls:", "OK" if not fails else "%d FAILED" % fails)
sys.exit(1 if fails else 0)
