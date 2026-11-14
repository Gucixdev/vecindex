/* vserv_log.c - the log: who connected, from where, what they changed and what failed (-l), every search too (-A).
 * Part of vserv.c, which includes it at this point; not compiled on its own.
 *
 * There are no users or roles inside the server: who may connect at all is decided by the operating system (the Unix
 * socket's file mode and group, the address the TCP port is bound to) and by TLS client certificates (-a); what each
 * connection did is written here, one line per event, to syslog (-l syslog), stderr (-l -) or a file (-l PATH,
 * appended, mode 0600). Every line names the connection and its peer:
 *
 *   conn=7 peer=unix:uid=1000,gid=1000,pid=4242 add id=42 ok
 *   conn=9 peer=tcp:10.0.0.5:51234,cn="ingest-1" del lo=1048576 hi=2097151 removed=37
 *   snapshot ok file=docs.vxi ms=812
 *
 * A tenant server (a directory of stores) adds the tenant a connection uses to each of its lines, and logs every
 * tenant it opens or creates:
 *
 *   tenant created name=acme rows=0
 *   conn=12 peer=tcp:10.0.0.7:40112,cn="acme" tenant=acme add id=7 ok
 *
 * Writes (add, delete, build, save, snapshot), connections opening and closing, refused requests and failed TLS
 * handshakes are always logged once -l is given; searches only with -A. A line is built in a local buffer and goes
 * out in one write, so lines from different workers never interleave. */

static i32 vs_log_fd = -1, vs_log_sys, vs_log_all;
static unsigned long vs_conn_seq;

/* -l DEST: "syslog", "-" for stderr, or a file; 0, or -1 when the file cannot be opened */
static i32 vs_log_open_(const char *dest)
{
    if (!strcmp(dest, "syslog")) { openlog("vserv", LOG_PID | LOG_NDELAY, LOG_DAEMON); vs_log_sys = 1; return 0; }
    if (!strcmp(dest, "-")) { vs_log_fd = 2; return 0; }
    vs_log_fd = open(dest, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    return vs_log_fd >= 0 ? 0 : -1;
}

static i32 vs_logging_(void) { return vs_log_sys || vs_log_fd >= 0; }

/* one line; c names the connection (NULL: a server event); warn: something was refused or failed */
static void vs_log_(const VsConn *c, i32 warn, const char *fmt, ...)
{
    char line[1024];
    i32 n = 0;
    va_list ap;
    if (!vs_logging_()) return;
    if (!vs_log_sys) {                                           /* syslog stamps its own lines */
        time_t now = time((time_t *)0);
        struct tm tm;
        gmtime_r(&now, &tm);
        n = (i32)strftime(line, 32, "%Y-%m-%dT%H:%M:%SZ ", &tm);
        n += sprintf(line + n, "vserv[%d] ", (int)getpid());
    }
    if (c) n += snprintf(line + n, sizeof(line) - (size_t)n, "conn=%lu peer=%s ", c->id, c->peer);
    if (c && vs_dir && c->t) n += snprintf(line + n, sizeof(line) - (size_t)n, "tenant=%s ", c->t->name);
    va_start(ap, fmt);
    n += vsnprintf(line + n, sizeof(line) - (size_t)n - 1, fmt, ap);
    va_end(ap);
    if (n > (i32)sizeof(line) - 2) n = (i32)sizeof(line) - 2;
    if (vs_log_sys) { line[n] = 0; syslog(warn ? LOG_WARNING : LOG_INFO, "%s", line); return; }
    line[n++] = '\n';
    if (write(vs_log_fd, line, (size_t)n) < 0) {}                /* a full disk must not stop the server */
}

/* who is on the other end of a fresh connection, for every line about it */
static void vs_peer_(VsConn *c, const VsListen *l)
{
    if (!l->tcp) {
#ifdef SO_PEERCRED
        struct ucred u;
        socklen_t n = sizeof(u);
        if (getsockopt(c->fd, SOL_SOCKET, SO_PEERCRED, &u, &n) == 0) {
            sprintf(c->peer, "unix:uid=%u,gid=%u,pid=%d", (unsigned)u.uid, (unsigned)u.gid, (int)u.pid);
            return;
        }
#endif
        strcpy(c->peer, "unix");
        return;
    }
    {
        struct sockaddr_in a;
        socklen_t n = sizeof(a);
        char ip[INET_ADDRSTRLEN];
        if (getpeername(c->fd, (struct sockaddr *)&a, &n) == 0 && inet_ntop(AF_INET, &a.sin_addr, ip, sizeof(ip)))
            sprintf(c->peer, "tcp:%s:%u", ip, (unsigned)ntohs(a.sin_port));
        else
            strcpy(c->peer, "tcp");
    }
}

#ifdef VSERV_TLS
/* the name a client certificate carries, added to the connection's peer once its handshake is done: printable
 * ASCII only (anything else becomes '?'), quoted */
static void vs_peer_name_(VsConn *c, const char *cn, i32 resumed)
{
    size_t n = strlen(c->peer), i;
    if (n + 12 >= sizeof(c->peer)) return;
    n += (size_t)sprintf(c->peer + n, ",cn=\"");
    for (i = 0; cn[i] && n + 3 < sizeof(c->peer); ++i) c->peer[n++] = cn[i] >= 0x20 && cn[i] < 0x7f && cn[i] != '"' ? cn[i] : '?';
    c->peer[n++] = '"'; c->peer[n] = 0;
    vs_log_(c, 0, "tls %s", resumed ? "resumed" : "verified");
}
#endif
