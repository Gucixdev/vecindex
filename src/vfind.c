/* vfind.c - search a store's texts as you type, like fzf: the hits under the prompt, a few lines of each, and Enter
 * opens the whole document (its chunks, found by id range). Enter there prints the choice on stdout.
 *
 *   vfind [-b BITS] [-k N] STORE.vxi
 *     -b BITS   ids are document << BITS | chunk (default 20, as in docs/api.md)
 *     -k N      hits to fetch per search (default 100)
 *   stdout: the chosen row as ID<TAB>DOCUMENT<TAB>CHUNK; exit 0 chosen, 1 nothing chosen, 2 bad usage or no store
 *
 *   vfind docs.vxi                        # type, Up/Down, Enter to read the document, Enter again to choose
 *   id=$(vfind docs.vxi | cut -f1)
 *
 * The list: type to search the words of the texts (BM25; the word being typed counts as a prefix), Up/Down or
 * Ctrl-P/Ctrl-N move, PgUp/PgDn jump, Enter opens the document, Ctrl-U clears, Ctrl-W drops a word, Esc quits. The
 * document: Up/Down, PgUp/PgDn, Home/End scroll, Enter chooses the chunk shown first, Esc or q goes back to the list.
 * The store is opened read-only (store_load_ro): a server may be writing to it meanwhile; what it wrote after the
 * open is not seen. Green on dark; with NO_COLOR, reverse video only. The screen is drawn on /dev/tty, so stdout
 * carries the choice alone.
 *
 *   cc -std=c89 -O2 -D_POSIX_C_SOURCE=200112L -Isrc src/vfind.c -lm -lpthread -o vfind
 */
#include "store.c"
#include <termios.h>
#include <signal.h>
#include <poll.h>
#include <sys/ioctl.h>

#define VF_MAXK   1000
#define VF_QMAX   512
#define VF_LINE   4096                                  /* bytes of one screen line, escapes included */

typedef struct { i64 id; f32 score; char *text; i32 len; } VfHit;
typedef struct { const char *p; i32 n, chunk, first; } VfLine;      /* a wrapped line of the document view */

static RagStore *vf_s;
static i32 vf_bits = 20, vf_k = 100, vf_color = 1, vf_tty = -1;
static struct termios vf_saved;
static volatile sig_atomic_t vf_resized;
static i32 vf_rows = 24, vf_cols = 80;
static char vf_q[VF_QMAX + 1];
static i32 vf_ql;
static VfHit vf_hit[VF_MAXK];
static i32 vf_nhit, vf_sel, vf_top, vf_qcol = 3;         /* hits; the selected one; the first shown; the cursor's column */
static char **vf_old;                                   /* the lines on the screen now, for the diff */
static i32 vf_oldn;
static char vf_out[1 << 16];                            /* bytes for the terminal, written in one go */
static i32 vf_on;

/* v as decimal at o; returns the end (i64 whatever the width of long) */
static char *vf_dec_(char *o, i64 v)
{
    char t[24];
    i32 n = 0;
    u64 u = v < 0 ? (u64)0 - (u64)v : (u64)v;
    do { t[n++] = (char)('0' + (i32)(u % 10u)); u /= 10u; } while (u);
    if (v < 0) *o++ = '-';
    while (n) *o++ = t[--n];
    *o = 0;
    return o;
}

/* ---- the terminal ---- */

static void vf_flush_(void)
{
    i32 at = 0;
    while (at < vf_on) { ssize_t k = write(vf_tty, vf_out + at, (size_t)(vf_on - at)); if (k <= 0) break; at += (i32)k; }
    vf_on = 0;
}
static void vf_put_(const char *p, i32 n)
{
    if (vf_on + n > (i32)sizeof(vf_out)) vf_flush_();
    if (n > (i32)sizeof(vf_out)) { (void)!write(vf_tty, p, (size_t)n); return; }
    memcpy(vf_out + vf_on, p, (size_t)n); vf_on += n;
}
static void vf_puts_(const char *s) { vf_put_(s, (i32)strlen(s)); }

static void vf_size_(void)
{
    struct winsize ws;
    if (ioctl(vf_tty, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0 && ws.ws_col > 0) { vf_rows = ws.ws_row; vf_cols = ws.ws_col; }
    if (vf_cols > VF_LINE / 16) vf_cols = VF_LINE / 16;
    if (vf_rows < 4) vf_rows = 4;
}

static void vf_restore_(void)
{
    if (vf_tty < 0) return;
    vf_puts_("\033[0m\033[?25h\033[?1049l");
    vf_flush_();
    tcsetattr(vf_tty, TCSAFLUSH, &vf_saved);
}
static void vf_on_winch_(int sig) { (void)sig; vf_resized = 1; }
static void vf_on_term_(int sig) { (void)sig; vf_restore_(); _exit(1); }

static i32 vf_raw_(void)
{
    struct termios t;
    struct sigaction sa;
    if ((vf_tty = open("/dev/tty", O_RDWR)) < 0 || tcgetattr(vf_tty, &vf_saved) != 0) return -1;
    t = vf_saved;
    t.c_iflag &= ~(tcflag_t)(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
    t.c_oflag &= ~(tcflag_t)OPOST;
    t.c_cflag |= CS8;
    t.c_lflag &= ~(tcflag_t)(ECHO | ICANON | IEXTEN | ISIG);
    t.c_cc[VMIN] = 1; t.c_cc[VTIME] = 0;
    if (tcsetattr(vf_tty, TCSAFLUSH, &t) != 0) return -1;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = vf_on_winch_; sigaction(SIGWINCH, &sa, (struct sigaction *)0);
    sa.sa_handler = vf_on_term_; sigaction(SIGTERM, &sa, (struct sigaction *)0); sigaction(SIGHUP, &sa, (struct sigaction *)0);
    vf_puts_("\033[?1049h\033[H\033[2J");
    vf_size_();
    return 0;
}

enum { VK_NONE = 0, VK_UP = 256, VK_DOWN, VK_PGUP, VK_PGDN, VK_HOME, VK_END, VK_ESC };

/* one key: a byte, or an arrow / page / home / end sequence, or a lone Esc (nothing follows within 30 ms) */
static i32 vf_key_(void)
{
    u8 c, s[8];
    i32 n = 0;
    struct pollfd p;
    if (read(vf_tty, &c, 1) != 1) return VK_NONE;
    if (c != 27) return c;
    p.fd = vf_tty; p.events = POLLIN;
    while (n < 7 && poll(&p, 1, 30) == 1 && read(vf_tty, &s[n], 1) == 1) { ++n; if (n >= 2 && ((s[n - 1] >= 'A' && s[n - 1] <= 'Z') || s[n - 1] == '~')) break; }
    if (n == 0) return VK_ESC;
    if (n >= 2 && (s[0] == '[' || s[0] == 'O')) {
        switch (s[n - 1]) {
        case 'A': return VK_UP;
        case 'B': return VK_DOWN;
        case 'H': return VK_HOME;
        case 'F': return VK_END;
        case '~': return s[1] == '5' ? VK_PGUP : s[1] == '6' ? VK_PGDN : (s[1] == '1' || s[1] == '7') ? VK_HOME : (s[1] == '4' || s[1] == '8') ? VK_END : VK_NONE;
        }
    }
    return VK_NONE;
}

/* ---- the words of the query, for highlighting ---- */

static i32 vf_isword_(u8 c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80; }
static u8 vf_lower_(u8 c) { return c >= 'A' && c <= 'Z' ? (u8)(c + 32) : c; }

/* the length of a query word starting at text[i] (case folded for ASCII), or 0 */
static i32 vf_match_(const char *t, i32 n, i32 i)
{
    i32 a = 0;
    if (i > 0 && vf_isword_((u8)t[i - 1])) return 0;         /* only at the start of a word */
    while (a < vf_ql) {
        i32 b = a, j;
        while (b < vf_ql && !vf_isword_((u8)vf_q[b])) ++b;
        a = b;
        while (a < vf_ql && vf_isword_((u8)vf_q[a])) ++a;
        if (a == b) break;
        for (j = 0; b + j < a && i + j < n && vf_lower_((u8)t[i + j]) == vf_lower_((u8)vf_q[b + j]); ++j) {}
        if (b + j == a) return a - b;
    }
    return 0;
}

/* ---- lines ---- */

/* columns of the UTF-8 bytes [p, p + n) (one a code point) */
static i32 vf_width_(const char *p, i32 n) { i32 w = 0, i; for (i = 0; i < n; ++i) if (((u8)p[i] & 0xc0) != 0x80) ++w; return w; }

/* appends up to w columns of text (newlines and tabs as spaces, query words highlighted) to line l; returns bytes used */
static i32 vf_span_(char *l, i32 *ln, const char *t, i32 n, i32 w, const char *base)
{
    i32 i = 0, col = 0, hl = 0;
    while (i < n && col < w) {
        i32 m = hl ? 0 : vf_match_(t, n, i), cl = 1;
        u8 c = (u8)t[i];
        if (m && vf_color) { memcpy(l + *ln, "\033[1;92m", 7); *ln += 7; hl = m; }
        else if (m) { memcpy(l + *ln, "\033[1;4m", 6); *ln += 6; hl = m; }
        if (c >= 0xc0) while (i + cl < n && ((u8)t[i + cl] & 0xc0) == 0x80) ++cl;
        if (c == '\n' || c == '\t' || c == '\r' || c < 0x20) l[(*ln)++] = ' ';
        else { memcpy(l + *ln, t + i, (size_t)cl); *ln += cl; }
        i += cl; ++col;
        if (hl && (hl -= cl) <= 0) { hl = 0; memcpy(l + *ln, base, strlen(base)); *ln += (i32)strlen(base); }
        if (*ln > VF_LINE - 64) break;
    }
    if (hl) { memcpy(l + *ln, base, strlen(base)); *ln += (i32)strlen(base); }
    return i;
}

/* the diff renderer: line r of the new frame goes out only when it differs from what the screen shows */
static void vf_line_(i32 r, const char *l, i32 ln)
{
    char at[32];
    if (r < vf_oldn && vf_old[r] && (i32)strlen(vf_old[r]) == ln && !memcmp(vf_old[r], l, (size_t)ln)) return;
    sprintf(at, "\033[%d;1H", (int)r + 1);
    vf_puts_(at); vf_put_(l, ln); vf_puts_("\033[0m\033[K");
    if (r < vf_oldn) { free(vf_old[r]); vf_old[r] = (char *)malloc((size_t)ln + 1); if (vf_old[r]) { memcpy(vf_old[r], l, (size_t)ln); vf_old[r][ln] = 0; } }
}

static void vf_frame_reset_(void)
{
    i32 i;
    for (i = 0; i < vf_oldn; ++i) free(vf_old[i]);
    free(vf_old);
    vf_oldn = vf_rows;
    vf_old = (char **)calloc((size_t)vf_oldn, sizeof(char *));
    vf_puts_("\033[H\033[2J");
}

/* the key bar: each key a reverse-video cap beside its label */
static void vf_keybar_(char *l, i32 *ln, const char *const *keys, i32 nk)
{
    i32 i, col = 0;
    for (i = 0; i + 1 < nk; i += 2) {
        i32 kw = vf_width_(keys[i], (i32)strlen(keys[i])), lw = vf_width_(keys[i + 1], (i32)strlen(keys[i + 1]));
        if (col + kw + lw + 4 > vf_cols) break;
        *ln += sprintf(l + *ln, "%s\033[7m %s \033[0m%s %s ", vf_color ? "\033[32m" : "", keys[i], vf_color ? "\033[32m" : "", keys[i + 1]);
        col += kw + lw + 4;
    }
}

static i64 vf_doc_(i64 id) { return id >> vf_bits; }
static i64 vf_chunk_(i64 id) { return id & (((i64)1 << vf_bits) - 1); }

/* ---- the list ---- */

static void vf_search_(void)
{
    static i64 ids[VF_MAXK];
    static f32 sc[VF_MAXK];
    i32 i;
    for (i = 0; i < vf_nhit; ++i) free(vf_hit[i].text);
    vf_nhit = vf_sel = vf_top = 0;
    if (vf_ql == 0) return;
    {   char q[sizeof(vf_q) + 2];                      /* the word being typed counts as a prefix: "zebr" -> "zebr*" */
        i32 ql = vf_ql;
        memcpy(q, vf_q, (size_t)ql);
        if (ql > 0 && q[ql - 1] != ' ' && q[ql - 1] != '*') q[ql++] = '*';
        vf_nhit = store_search_hybrid(vf_s, (const f32 *)0, q, ql, vf_k, ids, sc);
    }
    for (i = 0; i < vf_nhit; ++i) {
        i32 n = store_get_text(vf_s, ids[i], (char *)0, 0);
        vf_hit[i].id = ids[i]; vf_hit[i].score = sc[i];
        vf_hit[i].len = n > 0 ? n : 0;
        vf_hit[i].text = (char *)malloc((size_t)vf_hit[i].len + 1);
        if (vf_hit[i].text && n > 0) store_get_text(vf_s, ids[i], vf_hit[i].text, n);
    }
}

/* where a hit's snippet starts: a little before the first query word in it */
static i32 vf_snip_(const VfHit *h)
{
    i32 i, s;
    for (i = 0; i < h->len; ++i) if (vf_match_(h->text, h->len, i)) break;
    if (i >= h->len) return 0;
    for (s = i, i = 0; s > 0 && i < 24; --s) if (((u8)h->text[s - 1] & 0xc0) != 0x80) ++i;
    while (s > 0 && s < h->len && vf_isword_((u8)h->text[s - 1]) && i < 40) { --s; ++i; }
    return s;
}

static void vf_draw_list_(void)
{
    static char l[VF_LINE];
    static const char *const keys[] = { "Up/Down", "move", "Enter", "open", "^U", "clear", "Esc", "quit" };
    const char *base = vf_color ? "\033[0;32m" : "\033[0m";
    i32 per = 3, fit = (vf_rows - 2) / per, r, ln;
    if (fit < 1) fit = 1;
    if (vf_sel < vf_top) vf_top = vf_sel;
    if (vf_sel >= vf_top + fit) vf_top = vf_sel - fit + 1;
    ln = sprintf(l, "%s> %s", vf_color ? "\033[1;32m" : "", vf_color ? "\033[0;92m" : "");
    {   i32 q0 = 0, used, info = 0;
        char tail[64];
        while (q0 < vf_ql && vf_width_(vf_q + q0, vf_ql - q0) > vf_cols - 4) {          /* a long query: its end */
            ++q0;
            while (q0 < vf_ql && ((u8)vf_q[q0] & 0xc0) == 0x80) ++q0;
        }
        used = 2 + vf_width_(vf_q + q0, vf_ql - q0);
        vf_qcol = used + 1;
        if (vf_ql == 0) info = sprintf(tail, "type to search the texts");
        else info = sprintf(tail, "%d hit%s", (int)vf_nhit, vf_nhit == 1 ? "" : "s");
        memcpy(l + ln, vf_q + q0, (size_t)(vf_ql - q0)); ln += vf_ql - q0;
        if (used + info + 2 < vf_cols) {
            ln += sprintf(l + ln, "%s%*s", vf_color ? "\033[2;32m" : "", (int)(vf_cols - used - info), "");
            memcpy(l + ln, tail, (size_t)info); ln += info;
        }
    }
    vf_line_(0, l, ln);
    for (r = 1; r < vf_rows - 1; ++r) {
        i32 h = vf_top + (r - 1) / per, part = (r - 1) % per, mark;
        ln = 0;
        if (h >= vf_nhit || (r - 1) / per >= fit) { vf_line_(r, l, 0); continue; }
        mark = h == vf_sel;
        ln += sprintf(l + ln, "%s%s", mark ? (vf_color ? "\033[1;92m\xe2\x96\x8c" : "\033[7m>\033[0m") : " ", base);
        if (part == 0) {
            char *o = l + ln;
            o += sprintf(o, "%s doc ", mark ? "\033[1m" : ""); o = vf_dec_(o, vf_doc_(vf_hit[h].id));
            o += sprintf(o, "  chunk "); o = vf_dec_(o, vf_chunk_(vf_hit[h].id));
            o += sprintf(o, "  %sid ", vf_color ? "\033[2;32m" : ""); o = vf_dec_(o, vf_hit[h].id);
            o += sprintf(o, "  %.4f", (double)vf_hit[h].score);
            ln = (i32)(o - l);
        }
        else {
            const VfHit *x = &vf_hit[h];
            i32 w = vf_cols - 4, s = vf_snip_(x), off = 0;
            ln += sprintf(l + ln, "   ");
            if (part == 2) { i32 col = 0; for (off = s; off < x->len && col < w; ++off) if (((u8)x->text[off] & 0xc0) != 0x80) ++col;
                             while (off < x->len && ((u8)x->text[off] & 0xc0) == 0x80) ++off; }
            else off = s;
            if (off < x->len) vf_span_(l, &ln, x->text + off, x->len - off, w, base);
        }
        vf_line_(r, l, ln);
    }
    ln = 0; vf_keybar_(l, &ln, keys, 8);
    vf_line_(vf_rows - 1, l, ln);
    {   char at[32]; sprintf(at, "\033[1;%dH", (int)vf_qcol); vf_puts_(at); }
    vf_puts_("\033[?25h");
    vf_flush_();
}

/* ---- the document ---- */

typedef struct { char *buf; VfLine *line; i32 nline, cap, first, nchunk; i64 doc; } VfDoc;

/* the chunks of the selected hit's document, wrapped to the screen; first: the line where the hit's chunk starts */
static void vf_doc_load_(VfDoc *d, i64 id)
{
    static i64 ids[4096];
    i32 n, i, total = 0, at = 0;
    d->doc = vf_doc_(id);
    n = store_range(vf_s, d->doc << vf_bits, (d->doc << vf_bits) | (((i64)1 << vf_bits) - 1), ids, 4096);
    if (n < 0) n = 0;
    for (i = 0; i < n; ++i) { i32 m = store_get_text(vf_s, ids[i], (char *)0, 0); total += (m > 0 ? m : 0) + 1; }
    d->buf = (char *)malloc((size_t)total + 1);
    d->nline = 0; d->cap = 64; d->first = 0; d->nchunk = n;
    d->line = (VfLine *)malloc((size_t)d->cap * sizeof(VfLine));
    for (i = 0; i < n && d->buf && d->line; ++i) {
        i32 m = store_get_text(vf_s, ids[i], d->buf + at, total - at), w = vf_cols - 2, p = at, e;
        if (m < 0) m = 0;
        e = at + m;
        if (ids[i] == id) d->first = d->nline;
        if (d->nline + 1 >= d->cap) { d->cap *= 2; d->line = (VfLine *)realloc(d->line, (size_t)d->cap * sizeof(VfLine)); }
        d->line[d->nline].p = (const char *)0; d->line[d->nline].n = 0; d->line[d->nline].chunk = (i32)vf_chunk_(ids[i]); d->line[d->nline].first = 1; ++d->nline;
        while (p < e) {                                  /* wrap at a space when one is near, at the newlines always */
            i32 q = p, col = 0, cut = -1;
            while (q < e && d->buf[q] != '\n' && col < w) { if (d->buf[q] == ' ') cut = q; if (((u8)d->buf[q] & 0xc0) != 0x80) ++col; ++q; }
            while (q < e && ((u8)d->buf[q] & 0xc0) == 0x80) ++q;         /* never inside a UTF-8 character */
            if (q < e && d->buf[q] != '\n' && cut > p + w / 2) q = cut + 1;
            if (d->nline + 1 >= d->cap) { d->cap *= 2; d->line = (VfLine *)realloc(d->line, (size_t)d->cap * sizeof(VfLine)); }
            d->line[d->nline].p = d->buf + p; d->line[d->nline].n = q - p; d->line[d->nline].chunk = (i32)vf_chunk_(ids[i]); d->line[d->nline].first = 0;
            ++d->nline;
            p = q < e && d->buf[q] == '\n' ? q + 1 : q;
        }
        at = e;
    }
}

static void vf_draw_doc_(const VfDoc *d, i32 top, i32 hit_chunk)
{
    static char l[VF_LINE];
    static const char *const keys[] = { "Up/Down", "scroll", "PgUp/PgDn", "page", "Enter", "choose", "Esc", "back" };
    const char *base = vf_color ? "\033[0;32m" : "\033[0m";
    i32 r, ln;
    {   char *o = l;
        o += sprintf(o, "%sdoc ", vf_color ? "\033[1;92m" : "\033[1m"); o = vf_dec_(o, d->doc);
        o += sprintf(o, "%s  %d chunk%s  line %d of %d", vf_color ? "\033[0;2;32m" : "\033[0m", (int)d->nchunk, d->nchunk == 1 ? "" : "s",
                     (int)(top + 1), (int)d->nline);
        ln = (i32)(o - l); }
    vf_line_(0, l, ln);
    for (r = 1; r < vf_rows - 1; ++r) {
        i32 k = top + r - 1;
        ln = 0;
        if (k < d->nline) {
            const VfLine *x = &d->line[k];
            if (x->first) ln = sprintf(l, "%s\xe2\x94\x80\xe2\x94\x80 chunk %d %s", vf_color ? "\033[2;32m" : "\033[2m", (int)x->chunk,
                                       x->chunk == hit_chunk ? "(the hit) " : "");
            else {
                ln = sprintf(l, "%s%s ", x->chunk == hit_chunk ? (vf_color ? "\033[92m\xe2\x96\x8c" : "|") : " ", base);
                vf_span_(l, &ln, x->p, x->n, vf_cols - 2, base);
            }
        }
        vf_line_(r, l, ln);
    }
    ln = 0; vf_keybar_(l, &ln, keys, 8);
    vf_line_(vf_rows - 1, l, ln);
    vf_puts_("\033[?25l");
    vf_flush_();
}

/* the document of the selected hit until Esc (0) or Enter (1) */
static i32 vf_doc_view_(i64 id)
{
    VfDoc d;
    i32 top, key, page;
    memset(&d, 0, sizeof(d));
    vf_doc_load_(&d, id);
    top = d.first;
    vf_frame_reset_();
    for (;;) {
        page = vf_rows - 2;
        if (top > d.nline - page) top = d.nline - page;
        if (top < 0) top = 0;
        vf_draw_doc_(&d, top, (i32)vf_chunk_(id));
        key = vf_key_();
        if (vf_resized) { vf_resized = 0; vf_size_(); free(d.buf); free(d.line); vf_doc_load_(&d, id); vf_frame_reset_(); continue; }
        if (key == VK_ESC || key == 'q') break;
        if (key == '\r' || key == '\n') { free(d.buf); free(d.line); return 1; }
        if (key == VK_UP || key == 16 || key == 'k') --top;
        else if (key == VK_DOWN || key == 14 || key == 'j') ++top;
        else if (key == VK_PGUP) top -= page;
        else if (key == VK_PGDN || key == ' ') top += page;
        else if (key == VK_HOME) top = 0;
        else if (key == VK_END) top = d.nline;
        else if (key == 3) break;
    }
    free(d.buf); free(d.line);
    vf_frame_reset_();
    return 0;
}

/* One string per line: C89 compilers need not take a literal longer than 509 characters. */
static const char *const vf_help[] = {
    "vfind - search a store's texts as you type; Enter opens the document, Enter again prints the choice",
    "",
    "  vfind [-b BITS] [-k N] STORE.vxi",
    "    -b BITS   ids are document << BITS | chunk (default 20)",
    "    -k N      hits to fetch per search (default 100)",
    "  stdout: the chosen row as ID<TAB>DOCUMENT<TAB>CHUNK; exit 0 chosen, 1 nothing chosen, 2 bad usage or no store",
    "",
    "  vfind docs.vxi",
    "  id=$(vfind docs.vxi | cut -f1)",
    "",
    "The list: type to search the words of the texts (BM25; the word being typed counts as a prefix), Up/Down or",
    "Ctrl-P/Ctrl-N move, PgUp/PgDn jump, Enter opens the document, Ctrl-U clears, Ctrl-W drops a word, Esc quits.",
    "The document: Up/Down, PgUp/PgDn, Home/End scroll, Enter chooses the hit, Esc or q goes back to the list.",
    "The store is opened read-only; a server may be writing to it. NO_COLOR: no colours.",
    (const char *)0
};

static void vf_usage_(void)
{
    fprintf(stderr, "usage: vfind [-b BITS] [-k N] STORE.vxi   (vfind -h: the details)\n");
}


int main(int argc, char **argv)
{
    static Arena arena;
    const char *path = (const char *)0;
    i32 i, key, chosen = -1;
    for (i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            for (key = 0; vf_help[key]; ++key) puts(vf_help[key]);
            return 0;
        }
        if (!strcmp(argv[i], "-b") && i + 1 < argc) vf_bits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-k") && i + 1 < argc) vf_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--")) { if (i + 1 < argc) path = argv[++i]; }
        else if (argv[i][0] == '-' || path) { vf_usage_(); return 2; }
        else path = argv[i];
    }
    if (!path || vf_bits < 1 || vf_bits > 62) { vf_usage_(); return 2; }
    if (vf_k < 1) vf_k = 1;
    if (vf_k > VF_MAXK) vf_k = VF_MAXK;
    if (getenv("NO_COLOR") && *getenv("NO_COLOR")) vf_color = 0;
    if (!(vf_s = store_load_ro(&arena, path))) { fprintf(stderr, "vfind: cannot open %s as a vecindex store\n", path); return 2; }
    if (!vf_s->has_text) fprintf(stderr, "vfind: %s holds no texts (add rows with vi_add_text) - nothing to search\n", path);
    else {                                               /* the word index, before the first key: one pass over the texts */
        i64 id1; f32 sc1;
        store_search_hybrid(vf_s, (const f32 *)0, "a", 1, 1, &id1, &sc1);
    }
    if (vf_raw_() != 0) { fprintf(stderr, "vfind: needs a terminal (/dev/tty)\n"); store_destroy(vf_s); return 2; }
    vf_frame_reset_();
    for (;;) {
        vf_draw_list_();
        key = vf_key_();
        if (vf_resized) { vf_resized = 0; vf_size_(); vf_frame_reset_(); continue; }
        if (key == VK_ESC || key == 3) break;
        if (key == VK_UP || key == 16) { if (vf_sel > 0) --vf_sel; }
        else if (key == VK_DOWN || key == 14) { if (vf_sel + 1 < vf_nhit) ++vf_sel; }
        else if (key == VK_PGUP) { vf_sel -= (vf_rows - 2) / 3; if (vf_sel < 0) vf_sel = 0; }
        else if (key == VK_PGDN) { vf_sel += (vf_rows - 2) / 3; if (vf_sel >= vf_nhit) vf_sel = vf_nhit - 1; if (vf_sel < 0) vf_sel = 0; }
        else if (key == '\r' || key == '\n') { if (vf_nhit > 0 && vf_doc_view_(vf_hit[vf_sel].id)) { chosen = vf_sel; break; } }
        else if (key == 21) { vf_ql = 0; vf_search_(); }                               /* Ctrl-U */
        else if (key == 23) {                                                            /* Ctrl-W */
            while (vf_ql > 0 && vf_q[vf_ql - 1] == ' ') --vf_ql;
            while (vf_ql > 0 && vf_q[vf_ql - 1] != ' ') --vf_ql;
            vf_search_();
        } else if (key == 127 || key == 8) {
            while (vf_ql > 0 && ((u8)vf_q[vf_ql - 1] & 0xc0) == 0x80) --vf_ql;         /* a whole UTF-8 character */
            if (vf_ql > 0) --vf_ql;
            vf_search_();
        } else if (key >= 32 && key < 256 && vf_ql < VF_QMAX) { vf_q[vf_ql++] = (char)key; vf_search_(); }
    }
    vf_restore_();
    if (chosen >= 0) {
        char out[96], *o = out;
        i64 id = vf_hit[chosen].id;
        o = vf_dec_(o, id); *o++ = '\t'; o = vf_dec_(o, vf_doc_(id)); *o++ = '\t'; o = vf_dec_(o, vf_chunk_(id)); *o++ = '\n';
        (void)!write(1, out, (size_t)(o - out));
    }
    store_destroy(vf_s);
    return chosen >= 0 ? 0 : 1;
}
