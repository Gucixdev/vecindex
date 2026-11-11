# vserv: the store over a socket

`src/vserv.c` serves one store file: Linux epoll, one worker thread per CPU the process may use (its affinity mask and
cgroup quota, times `-g`), a request buffer per connection (allocated on its first byte, reused after; it grows for a
request that carries a big text and shrinks back), replies built in a per-thread buffer - a search allocates nothing.
The store is opened with `store_load` (mmap), so a 100M-row file is ready in milliseconds.

```sh
./run                                    # builds .build/vserv (BearSSL into .build/libbearssl.a first)
.build/vserv -s /run/vx.sock -p 8077 -t 2 -d 384 docs.vxi        # Unix socket (0600) + TCP on 127.0.0.1; a new store
.build/vserv -p 8077 -L -r 0.95 -l syslog docs.vxi               # live: adds indexed as they come; log to syslog
.build/vserv -p 8077 -g 0.5 docs.vxi                             # half the CPUs; background work at nice 14, disk when idle
.build/vserv -p 8077 -m 1 docs.vxi                               # the scanned codes locked in RAM, rows read from the disk
.build/vserv -p 8443 -c cert.pem -k key.pem [-a clients-ca.pem] [-T] docs.vxi   # TLS, optional mutual TLS
curl -s localhost:8077/stats
curl -s -X POST 'localhost:8077/search?k=5' -d '[0.1, 0.2, ...]'
```

## Three protocols on every socket

They are told apart by the first byte of a request, so one connection can mix them.

**Lines** (a lowercase letter first): one line in, one line out, for `nc`, `socat` and a readline in any language.
A session over TCP, word for word:

```
add 1 1 0 0 0                 ok
add 2 0 1 0 0                 ok
search 2 0.9 0.1 0 0          ok 2 1 0.993884 2 0.110432
stats                         ok count=2 dim=4 pending=2 built=0 snapshot=idle last=none hot=8 cold=72 memory=0 damaged=0 rebuilt=0 rows_bad=0 rows_healed=0 journal=ok
del 1 2                       ok 2
quit                          ok bye
```

The commands: `ping`, `stats`, `search K X1 .. XD`, `add ID X1 .. XD`, `get ID`, `del ID`, `del LO HI` (every id in
the range), `build`, `save`, `snapshot`, `help`, `quit`. A reply is `ok ...` or `err WHAT`; scores are the cosine
(1 the same direction, -1 the opposite), with 6 decimals - or, for a store made with `-M dot` or `-M l2`, the inner
product or the Euclidean distance.

Texts (see [api.md](api.md#texts-from-a-small-chunk-to-the-whole-document)): `addt ID X1 .. XD TEXT` adds a row with
a text, the rest of the line, with `\n` `\r` `\t` `\0` `\\` for the bytes a line cannot carry; `text ID` answers
`ok LEN TEXT` escaped the same way; `range LO HI [MAX]` lists the ids in the range, ascending; `hybrid K X1 .. XD
WORDS` and `words K WORDS` search with the words against the texts (BM25), with or without a vector, and answer like
`search` with fused scores (bigger is better):

```
addt 7340032 1 0 0 0 Oolong is partly oxidised.   ok
text 7340032                                       ok 26 Oolong is partly oxidised.
range 7340032 8388607                              ok 1 7340032
words 3 oxidised                                   ok 1 7340032 0.049180
```

Grouped search, numbers and updates (see [api.md](api.md)): `group K SHIFT X1 .. XD` gives one hit an `id >> SHIFT`
group - with ids `doc << 20 | chunk`, `group 10 20 ...` is the best chunk of 10 documents; `num ID N` sets a row's
number (a time, a price) and `num ID` reads it; `between K LO HI X1 .. XD` searches the rows whose number is in
`[LO, HI]`; `update ID X1 .. XD` gives a row a new vector and keeps its tag, number and text. In a batch, `num` and
`update` wait like `add` (`ok queued`).

```
num 7340032 1731250000                             ok
between 5 1731000000 1732000000 1 0 0 0            ok 1 7340032 1.000000
group 3 20 1 0 0 0                                 ok 1 7340032 1.000000
update 7340032 0.9 0.1 0 0                         ok
```

**Many users, one server** (`-K keys.tsv`): a line of the file is `KEY TENANT`. A client on the TCP port shows its
key first - `auth KEY` (lines), op 20 (binary), `Authorization: Bearer KEY` on every HTTP request - and from then on
reaches only its tenant. With one store a tenant is a number (1 to 524287): its rows go into a namespace of the store
with ids of its own (0 to 2^44 - 1), so two tenants may both have an id 1, and no search, get, delete or range of one
ever touches the other's rows. With a tenant directory a tenant is a name, and its key opens only its file. A key
cannot build, save, snapshot or scrub - that stays with whoever runs the server - and its `stats` show only its own
row count. The Unix socket needs no key: who may connect to it is the socket file's mode.

```
# keys.tsv
k_9f2c41d07e3a4b55a1f0   7
k_1b7e0c9da2f84c3e8d61   8

vserv -K keys.tsv -c cert.pem -k key.pem -p 8443 data.vxi
curl -H 'Authorization: Bearer k_9f2c41d07e3a4b55a1f0' -d '[0.1, 0.2, ...]' https://host:8443/search?k=10
```

A line may add limits: `KEY TENANT ROWS QPS` - at most ROWS rows (counted once a second; replacing a row the tenant
has is never refused) and QPS requests a second (past it: `err rate limit`, binary status 7, HTTP 429); 0 or nothing
is no limit. A key with `*` for its tenant is the operator's: the whole store, any tenant, the housekeeping.

**Numbers:** `GET /metrics` answers JSON, `GET /prometheus` the Prometheus text format - the same numbers in both:
uptime, searches, writes, refused requests, search latency in buckets (0.5 ms to 1 s), and for every open store its
rows, rows not indexed yet, memory (hot, cold), blocks checked, damaged and rebuilt, rows that failed their check
and were healed, the journal, snapshots and the lens. Only for the operator: no `-K`, the `*` key, or the Unix
socket; a tenant's key gets 403. A Prometheus scrape with keys sends the `*` key (`authorization: credentials:`).

**Binary** (`0xB5` first): little-endian frames, nothing to parse - `u8 0xB5, u8 op, u16 k, u32 nbytes, payload`.
Ops: 1 search, 2 add, 3 delete, 4 build, 5 save, 6 stats, 7 snapshot, 8 get, 9 delete a range of ids, 10 add with a
text, 11 get a text, 12 the ids in a range, 13 hybrid search, 14 use a tenant, 15 grouped search (`u32 shift`, then
the vector), 16 a number range (`i64 lo, i64 hi`, then the vector), 17 update (`i64 id`, then the vector), 18 set a
number (`i64 id, i64 number`), 19 get a number (`i64 id`, reply `i64`).

**HTTP/1.1** (anything else): `GET /health`, `GET /stats`, `GET /get?id=N`, `GET /text?id=N`,
`GET /range?lo=&hi=&max=`, `GET /num?id=`, `POST /search?k=` (with `&group=SHIFT` one hit a group, with `&lo=&hi=`
the rows whose number is in the range), `/hybrid?k=`, `/update?id=`, `/num?id=&num=`, `/add?id=`, `/delete?id=` or `/delete?lo=&hi=`, `/build`,
`/save`, `/snapshot`; vectors as a JSON array or raw little-endian f32 (an add's text after the floats), or
`{"vector": [...], "text": "..."}` for an add with a text and for `/hybrid` (either key); keep-alive, pipelining. The header scan looks for the blank line 16 bytes at a time with SSE2 (the picohttpparser
idea, cut down to what this server needs).

In all three an add replaces the row with the same id: one row per id. A request may carry up to 8 MB of text: the
buffer of a connection starts small, grows for a big request and goes back once it is answered. The full reference is the comment at the top
of `src/vserv_proto.c`.

## Durability: a journal and background snapshots

Every write goes into the journal next to the file (`docs.vxi.j`), which is flushed to the disk every second (`-f
SECONDS`; 0 leaves it to the kernel), so a crash loses at most that second. A new store's file is created at start,
so the journal covers it from the first write. A background snapshot folds the journal into the file: a child process
writes a copy-on-write image of the store while the server keeps answering and taking writes; the writes made
meanwhile go to both journals, and the new one takes over when the file is complete. One starts whenever the journal
has outgrown the file (and 64 MB), every `-S SECONDS` when something changed, and on `snapshot` from a client.
`tests/vserv_test.c` kills the server with `kill -9` a second after a burst of adds and finds every one of them after
the restart; `tests/scrash.c` does the same to the store at random moments, with snapshots running.

Several writes go in together with a batch, begun and ended on the line protocol; what comes in between, in any
of the three protocols on that connection, waits ("ok queued") and searches do not see it yet:

```
begin                         ok
del 2097152 3145727           ok queued
addt 2097152 0.1 ... first chunk of the new version     ok queued
addt 2097153 0.2 ... second chunk                       ok queued
commit sync                   ok 3
```

`commit` applies all of it at once and journals it as one record with a CRC-32C, so a crash leaves all or none of it;
`commit sync` replies once it is on the disk (commits that arrive during an fsync share the next one). `abort`, or
closing the connection, drops it. A tenant cannot be switched while a batch is open.

Every file carries BLAKE3 digests of its 64 KiB blocks and Reed-Solomon parity (2 blocks for every 32, groups
interleaved: [api.md](api.md#integrity-checksums-reed-solomon-parity-scrub)). At start the server checks every
data block, rebuilds damaged ones and writes them back, and says so on stderr (for a tenant, in the log: `check file=...
blocks= damaged= rebuilt= journal=`); a file with damage beyond the parity in its rows does not start. `scrub` on the
line protocol reads every block of the current store's file from the disk, rebuilds and rewrites what it can, logs
`scrub file= damaged= rebuilt= ms=` and replies `ok damaged=N rebuilt=M`; `stats` ends with `damaged= rebuilt=
rows_bad= rows_healed= journal=ok|damaged`: what the load found, then the rows that failed their check when a search
read them - rebuilt from the parity on the spot, or left out of the results. A cron line that runs `scrub` weekly is the ZFS habit.

A copy of the store is the file plus its journal, both written atomically (`rename`), so `rsync` or `cp` of the two
at any moment is a consistent backup; a standby machine that replays them starts in milliseconds (mmap). A snapshot
to another path (`store_snapshot` / `vi_snapshot` with a path) writes a point-in-time copy without touching the live
journal. For a copy that keeps up by itself, see the replica below.

## A replica on a second machine: `vserv -F`

```sh
# on the standby: copy everything the primary has flushed, every second, until stopped
vserv -F 10.0.0.5:8077 /srv/vx/docs.vxi          # a one-store primary (plaintext TCP: -P if it has TLS)
vserv -F /run/vx/primary.sock /srv/vx/           # a tenant server: every tenant, NAME.vxi in the directory
VSERV_KEY=op_key_... vserv -F 10.0.0.5:8077 ...  # the operator's key, when the primary has -K
```

The replica copies what `store_load` reads, the file and its journal, and serves nothing. The journal comes over byte
for byte, but only up to what the primary's last fsync covered, which always ends on a whole record - so the replica
never holds a write the primary could still lose in a crash, and a write reaches it within about a second (`-f` on
the primary; with `-f 0` the replica sees writes only after a `commit sync` or a snapshot). When a save or a snapshot
gives the primary's file a new stamp, the replica fetches the new file in 4 MB chunks into `NAME.vxi.part`, checks every
block against its digest (rebuilding from the parity what it can), renames it into place and starts the journal over;
in between the old file and its journal stay a matching pair. A torn tail on the replica's own journal (it was
killed mid-write) is cut when it starts and fetched again. Every file copied is a line on stderr and in the log (`-l`).

Failover is by hand or by the init system: stop the replica, start `vserv` on the same path. It replays the journal
like any restart and has every row the primary had flushed - `tests/vserv_test.c` kills the primary with `kill -9`
and checks exactly that. Never run a replica and a server on the same files at once. There is one primary per store:
the replica is a standby, not a second place to write.

The primary side is three binary ops (21 journal, 22 file, 23 the tenants), refused to tenant keys: the Unix socket
or the operator's key (`*` in `-K`). The replica speaks plaintext. Across machines either let the primary's TCP port
take plaintext on a private network (`-P` with TLS, or no TLS) or carry the Unix socket over ssh, which encrypts it:

```sh
ssh -N -L /run/vx/primary.sock:/srv/vx/vecindex.sock vx@primary &   # a local socket that is the primary's
vserv -F /run/vx/primary.sock /srv/vx/
```

A tenant server's replica opens every tenant on the primary (`use NAME`), which keeps them open there afterwards.

## Encryption at rest: the operating system's

The server writes plain files and leaves encryption to the layer below, which does it at disk speed with AES-NI and
already solves the hard part, the keys (a passphrase at boot, a TPM, a network key server). Its own encryption would
gain nothing: the store is memory-mapped, so its pages sit in the page cache in the clear while it runs whatever the
file holds. Pick one:

- **The whole volume, LUKS.** `cryptsetup luksFormat /dev/nvme0n1p3`, `cryptsetup open /dev/nvme0n1p3 vx`,
  `mkfs.ext4 /dev/mapper/vx`, mount it where the stores live. Unattended boot: `systemd-cryptenroll --tpm2-device=auto`
  (the key in the TPM) or clevis with a tang server (the key only on your network). Covers a stolen disk, a disk sent
  back under warranty, a disk thrown away.
- **A dataset, ZFS native encryption.** `zfs create -o encryption=aes-256-gcm -o keyformat=passphrase tank/vx`.
  `zfs send --raw` replicates or backs it up still encrypted: the backup machine never sees the key.
- **A directory, fscrypt** (ext4 with `-O encrypt`, or f2fs): the key of one directory, locked again by removing the
  key. With a server per directory, deleting that directory's key makes its data unreadable everywhere, backups
  included (crypto-shredding). Tenants in one directory share its key.

What else holds the data and must be on the same encrypted layer: the snapshot's temporary file (`NAME.vxi.snap`, in
the same directory), the replica's files and every backup copy (encrypted only if where they land is), swap (encrypt
it, or keep the store in RAM with `-m`), and core dumps (`LimitCORE=0` in the unit). In transit there is TLS (`-c`/`-k`
below) for clients and ssh for the replica.

## Memory: hot codes in RAM, cold rows on the disk

A search scans the hot part of a store every time: the index's codes (1, 2 or 4 bits a dimension), its slot table and
centroids. It reads the cold part only for the few hundred candidates it re-scores and the hits it returns: the rows
(f32 or int8), ids, tags, their checks and texts. The file is mapped, so by default the kernel keeps what searches touch and drops
the rarely read rows first under pressure. `-m` takes that decision away from it:

| `-m` | locked in RAM | on the disk, read when needed |
|---|---|---|
| 0 (default) | nothing; the codes are read ahead at start | everything, cached as the kernel sees fit |
| 1 | the hot part | rows, ids, tags, texts |
| 2 | the hot part, rows, ids, tags and checks | texts |

`stats` reports both sizes (`hot=` and `cold=` in bytes) and the tier. For 100M rows of 768 dimensions with 1-bit
codes and f32 rows that is about 10 GB hot (96 bytes of codes and 4 of slot table a row) and 307 GB cold: `-m 1`
keeps the 10 GB in RAM and the exact f32 rows on an SSD, read only for the candidates. The lock needs a memlock limit
that large (`ulimit -l`, or `LimitMEMLOCK=infinity` in a systemd unit); what it refuses is only read ahead, and the
server says so at start. In C: `vi_set_memory` and `vi_memory` ([api.md](api.md)).

## Access and accountability: the operating system and the log, no users and roles

Who may connect is decided outside the server: the Unix socket is created mode 0600 (owner only; change its group and
mode to share it), the TCP port binds to 127.0.0.1 unless `-b` says otherwise, and `-a CA` lets in only clients with a
certificate from that CA. What each connection did goes to the log, one line per event (`-l syslog`, `-l -` for
stderr, `-l FILE`), with the peer of every line: `uid`, `gid` and `pid` on the Unix socket (`SO_PEERCRED`), address
and port on TCP, and the certificate's name under mutual TLS - kept across session resumptions. From the session
above, with `-l - -A`:

```
2026-10-30T00:05:54Z vserv[21369] start file=/tmp/demo.vxi rows=0 dim=4 threads=2 created journal=on memory=0 hot=0 cold=0
2026-10-30T00:05:55Z vserv[21369] conn=1 peer=tcp:127.0.0.1:52036 open
2026-10-30T00:05:55Z vserv[21369] conn=1 peer=tcp:127.0.0.1:52036 add id=1 ok
2026-10-30T00:05:55Z vserv[21369] conn=1 peer=tcp:127.0.0.1:52036 add id=2 ok
2026-10-30T00:05:55Z vserv[21369] conn=1 peer=tcp:127.0.0.1:52036 search k=2 hits=2 us=7
2026-10-30T00:05:55Z vserv[21369] conn=1 peer=tcp:127.0.0.1:52036 del lo=1 hi=2 removed=2
2026-10-30T00:05:55Z vserv[21369] conn=1 peer=tcp:127.0.0.1:52036 close requests=6
2026-10-30T00:05:56Z vserv[21369] stop rows=0
```

Writes, connections, refused requests, failed TLS handshakes and snapshots are logged once `-l` is given; searches
only with `-A`. A bulk load logs one line per row: for millions of rows prefer `-l FILE` to syslog, which journald
rate-limits.

## Tenants: a store file each

Given a directory instead of a file, the server keeps a store per tenant in it, `DIR/NAME.vxi` with its own journal,
index and snapshots; workers, the log and the housekeeping are shared. A client names its tenant before anything else:

```sh
.build/vserv -p 8077 -d 384 -l syslog /srv/vx/                # every tenant's store has 384 dimensions
printf 'use acme\nsearch 5 0.1 ...\n' | nc localhost 8077      # lines: use NAME, then as on a one-store server
curl -s -X POST 'localhost:8077/t/acme/search?k=5' -d '[0.1, ...]'   # HTTP: /t/NAME/ before every path
```

The binary protocol has op 14 with the name as its payload; until a tenant is chosen every other request gets status
5 (lines: `err no tenant: use NAME first`; HTTP: 400). A tenant opens on first use (mmap, milliseconds), or is created
there when its file does not exist yet, and stays open; up to 4096 at once. Names are 1 to 64 of `A-Z a-z 0-9 _ -`,
so a name never leaves the directory. Every log line of a connection carries its tenant (`tenant=acme`), and opening
or creating one is logged.

Which client may use which tenant is, again, not a table of users inside the server. With mutual TLS (`-a CA`) and
`-U`, a client may use only the tenant its certificate names, so issuing the certificate is granting the access;
clients without a certificate (local ones on the Unix socket, whose file mode decides who connects) may use any.
Another split is a server per group of tenants, each with its own directory and socket, whose file mode and group
decide who connects.

## TLS

`-c`/`-k`: TLS 1.2 through the vendored [BearSSL](https://bearssl.org) 0.6 (MIT, unmodified, no crypto code of our
own): ECDHE only (forward secrecy), AEAD suites only (AES-GCM with AES-NI, ChaCha20-Poly1305), EC or RSA keys,
sessions resumed from one cache shared by every worker, keys wiped when a connection closes. `-a CA` requires client
certificates (mutual TLS). TLS is sniffed per connection (`0x16`): the TCP port takes only TLS (`-P` lets plaintext in
too); the Unix socket takes TLS or, for local tools, plaintext (`-T`: TLS only).

## Measured

2.4 MB idle RSS; 4.2 KB per open plaintext connection, 20.7 KB per TLS connection (300 open). From Python clients on
2 cores: ~15-19k searches/s plaintext, ~10.7k over TLS on the first 2-core host; 8.9-10.6k and 5.7-7.5k on the Cascade
Lake host of [benchmarks.md](benchmarks.md) (2k rows; the Python clients are the bottleneck). `-L` puts the store in
live mode (added rows are indexed by the server's own thread and the index is rebuilt in the background as it ages -
no `/build` calls) and `-r 0.95` keeps recall@10 at 0.95 (the probes tuned on the queries it answers); see "Live
ingest" in [store.md](store.md).

## Tests

`py/vserv_test.py` (29 checks: the binary protocol and HTTP, both sockets, pipelining, a request trickled byte by
byte, oversized and malformed input, 8 concurrent clients, SIGTERM saves, restart, live mode: 3000 adds over the
socket and no build call) and `py/vserv_tls_test.py` (22 checks: resumption on every reconnect, multi-record replies,
plaintext refused, untrusted CA, mTLS accept/reject, RSA key, curl, TLS on the Unix socket), and `tests/vserv_test.c`
(the line protocol and its errors, the binary and HTTP snapshot, get and range-delete calls, replace-on-add, the log
and its peers, kill -9 against the journal, timed snapshots, API keys and their limits, /metrics and /prometheus, a
replica (`-F`) through a snapshot, a kill -9 of the primary and a failover, over TCP with the operator's key, on a
tenant directory and past a torn tail, and the certificate name of a curl client under mutual TLS in the log) run in
`./run`.
