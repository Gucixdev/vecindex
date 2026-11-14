/* vecdb_cli.c - the vecdb command line:  vecdb DB.vdb index DIR | update DIR | remove TITLE | stats | search Q |
 *                 images Q | similar PIC | explore Q | links T | backlinks T | tag TAG | tags | ghosts
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -D_POSIX_C_SOURCE=200112L -O2 -march=native -Isrc src/vecdb_cli.c -lm -lpthread -o vecdb
 */
#include "vecdb.c"
#include <stdio.h>

static char *cli_slurp_(const char *path)
{
    FILE *f = fopen(path, "rb");
    long n; char *b;
    if (!f) return (char *)0;
    fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET);
    b = (char *)malloc((size_t)n + 1);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return (char *)0; }
    b[n] = 0; fclose(f);
    return b;
}

static const char *cli_extract_script_ = "extract.py";

/* text of a file: read as is for text formats, through extract.py for pdf/docx/... */
static char *cli_text_(const char *path, const char *name)
{
    size_t l = strlen(name);
    static const char *plain[] = { ".md", ".txt", ".org", ".rst", ".markdown", 0 };
    static const char *rich[] = { ".pdf", ".docx", ".odt", ".pptx", ".epub", ".html", ".htm", 0 };
    i32 i;
    for (i = 0; plain[i]; ++i) { size_t k = strlen(plain[i]); if (l > k && !strcmp(name + l - k, plain[i])) return cli_slurp_(path); }
    for (i = 0; rich[i]; ++i) {
        size_t k = strlen(rich[i]);
        if (l > k && !strcmp(name + l - k, rich[i])) {
            char cmd[8192], *b = (char *)malloc(1 << 16), *q = cmd;
            size_t n = 0, cap = 1 << 16, r;
            FILE *p;
            const char *s;
            q += sprintf(q, "python3 '%s' '", cli_extract_script_);
            for (s = path; *s && q < cmd + 8000; ++s) { if (*s == '\'') { memcpy(q, "'\\''", 4); q += 4; } else *q++ = *s; }
            strcpy(q, "'");
            p = popen(cmd, "r");
            if (!p) { free(b); return (char *)0; }
            while ((r = fread(b + n, 1, cap - n - 1, p)) > 0) { n += r; if (n + 1 >= cap) { cap *= 2; b = (char *)realloc(b, cap); } }
            pclose(p);
            b[n] = 0;
            return b;
        }
    }
    return (char *)0;
}

/* chunk size in characters: what the model actually reads (EMBED_MAXLEN tokens, ~3.5 chars each), so no chunk is
 * embedded from its first part only; VECDB_CHUNK overrides */
static i32 cli_chunk_ = 448;
#define CLI_CHUNK cli_chunk_

/* notes are queued and added 1024 at a time (so several model servers can share a batch), so the model server sees batches */
static char *cli_qt_[1024], *cli_qx_[1024];
static i32 cli_qn_ = 0;

static void cli_flush_(VecDB *db)
{
    i32 i;
    if (cli_qn_ == 0) return;
    vecdb_add_notes(db, cli_qn_, (const char **)cli_qt_, (const char **)cli_qx_);
    for (i = 0; i < cli_qn_; ++i) { free(cli_qt_[i]); free(cli_qx_[i]); }
    cli_qn_ = 0;
}

static void cli_queue_(VecDB *db, const char *title, const char *text)
{
    cli_qt_[cli_qn_] = (char *)malloc(strlen(title) + 1); strcpy(cli_qt_[cli_qn_], title);
    cli_qx_[cli_qn_] = (char *)malloc(strlen(text) + 1); strcpy(cli_qx_[cli_qn_], text);
    if (++cli_qn_ == 1024) cli_flush_(db);
}

/* a long text becomes a document note that links to its chunks; every chunk links back and to its
 * neighbours, so the graph (and MCTS) can walk through a document */
static void cli_add_doc_(VecDB *db, const char *name, const char *text, i32 *count)
{
    i32 len = (i32)strlen(text), nch = 0, cap = 64, i;
    i32 *st, *en;
    char title[4200], *body;
    const char *p = text;
    if (len <= CLI_CHUNK + CLI_CHUNK / 4) { cli_queue_(db, name, text); ++*count; return; }
    st = (i32 *)malloc((size_t)cap * sizeof(i32)); en = (i32 *)malloc((size_t)cap * sizeof(i32));
    while (p < text + len) {
        const char *e = p + CLI_CHUNK, *stop = text + len;
        if (e >= stop) e = stop;
        else {
            const char *q = e;                                   /* prefer a paragraph break, then a sentence, then a blank */
            while (q > p + CLI_CHUNK / 2 && !(q[0] == '\n' && q[-1] == '\n')) --q;
            if (q <= p + CLI_CHUNK / 2) { q = e; while (q > p + CLI_CHUNK / 2 && !(q[-1] == '.' && q[0] == ' ')) --q; }
            if (q <= p + CLI_CHUNK / 2) { q = e; while (q > p + CLI_CHUNK / 2 && q[0] != ' ') --q; }
            if (q > p + CLI_CHUNK / 2) e = q;
        }
        if (nch == cap) { cap *= 2; st = (i32 *)realloc(st, (size_t)cap * sizeof(i32)); en = (i32 *)realloc(en, (size_t)cap * sizeof(i32)); }
        st[nch] = (i32)(p - text); en[nch] = (i32)(e - text); ++nch;
        p = e; while (p < text + len && (*p == '\n' || *p == ' ')) ++p;
    }
    body = (char *)malloc((size_t)len + (size_t)nch * 8 + 256 + (size_t)nch * (strlen(name) + 24));
    for (i = 0; i < nch; ++i) {                                  /* chunks */
        char *w = body;
        sprintf(title, "%s \xc2\xa7%d", name, i + 1);
        memcpy(w, text + st[i], (size_t)(en[i] - st[i])); w += en[i] - st[i];
        w += sprintf(w, "\n\n[[%s]]", name);
        if (i > 0) w += sprintf(w, " [[%s \xc2\xa7%d]]", name, i);
        if (i + 1 < nch) w += sprintf(w, " [[%s \xc2\xa7%d]]", name, i + 2);
        cli_queue_(db, title, body);
    }
    { char *w = body;                                            /* the document: its opening + links to every chunk */
      i32 head = en[0] < 700 ? en[0] : 700;
      memcpy(w, text, (size_t)head); w += head;
      w += sprintf(w, "\n\n");
      for (i = 0; i < nch; ++i) w += sprintf(w, "[[%s \xc2\xa7%d]] ", name, i + 1);
      cli_queue_(db, name, body); }
    *count += nch + 1;
    free(body); free(st); free(en);
}

static i32 cli_is_image_(const char *name)
{
    static const char *ext[] = { ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", 0 };
    size_t l = strlen(name); i32 i;
    for (i = 0; ext[i]; ++i) { size_t k = strlen(ext[i]); if (l > k) { size_t j; i32 same = 1; for (j = 0; j < k; ++j) if ((name[l - k + j] | 32) != ext[i][j]) same = 0; if (same) return 1; } }
    return 0;
}

static void cli_index_(VecDB *db, const char *dir, i32 *count)
{
    DIR *d = opendir(dir);
    struct dirent *e;
    if (!d) return;
    while ((e = readdir(d)) != (struct dirent *)0) {
        char path[4096]; struct stat st;
        char *txt;
        if (e->d_name[0] == '.') continue;
        sprintf(path, "%s/%s", dir, e->d_name);
        if (stat(path, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) { cli_index_(db, path, count); continue; }
        if (db->store2 && cli_is_image_(e->d_name)) {
            char cap[512]; size_t i;
            strncpy(cap, e->d_name, 400); cap[400] = 0;
            for (i = 0; cap[i]; ++i) if (cap[i] == '_' || cap[i] == '-' || cap[i] == '.') cap[i] = ' ';
            if (vecdb_add_image(db, e->d_name, path, cap) >= 0) ++*count;
            continue;
        }
        txt = cli_text_(path, e->d_name);
        if (txt) { if (txt[0]) cli_add_doc_(db, e->d_name, txt, count); free(txt); }
    }
    closedir(d);
}

static void cli_hits_(const VecDB *db, const VHit *h, i32 n)
{
    i32 i;
    for (i = 0; i < n; ++i) {
        printf("%2d. %-36s score %.4f  cos %.2f  bm25 %.2f", i + 1, db->nodes[h[i].node].name, h[i].score, h[i].sim, h[i].lex);
        if (h[i].via >= 0) printf("  via %s (depth %d)", db->nodes[h[i].via].name, h[i].depth);
        putchar('\n');
    }
}

/* chunks of one document ("name §3") are shown once, under the document's name, at the rank of their best chunk */
static void cli_hits_grouped_(const VecDB *db, const VHit *h, i32 n, i32 want)
{
    i32 i, j, shown = 0;
    char base[512], other[512];
    for (i = 0; i < n && shown < want; ++i) {
        const char *nm = db->nodes[h[i].node].name;
        const char *p = strstr(nm, " \xc2\xa7");
        i32 bl = p ? (i32)(p - nm) : (i32)strlen(nm), dup = 0, more = 0;
        if (bl > 500) bl = 500;
        memcpy(base, nm, (size_t)bl); base[bl] = 0;
        for (j = 0; j < i; ++j) {                                /* an earlier hit of the same document already shown */
            const char *q = db->nodes[h[j].node].name, *pq = strstr(q, " \xc2\xa7");
            i32 ql = pq ? (i32)(pq - q) : (i32)strlen(q);
            if (ql == bl && memcmp(q, base, (size_t)bl) == 0) { dup = 1; break; }
        }
        if (dup) continue;
        for (j = i + 1; j < n; ++j) {
            const char *q = db->nodes[h[j].node].name, *pq = strstr(q, " \xc2\xa7");
            i32 ql = pq ? (i32)(pq - q) : (i32)strlen(q);
            if (ql == bl && memcmp(q, base, (size_t)bl) == 0) ++more;
        }
        (void)other;
        printf("%2d. %-36s score %.4f  cos %.2f  bm25 %.2f", ++shown, nm, h[i].score, h[i].sim, h[i].lex);
        if (more) printf("  (+%d more parts)", more);
        if (h[i].via >= 0) printf("  via %s (depth %d)", db->nodes[h[i].via].name, h[i].depth);
        putchar('\n');
    }
}

static void cli_names_(const VecDB *db, const i32 *ids, i32 n)
{
    i32 i;
    for (i = 0; i < n; ++i) printf("  %s%s\n", db->nodes[ids[i]].kind == VN_TAG ? "#" : "", db->nodes[ids[i]].name);
}

static const char *cli_path_ = "";
static VEmbedPool *cli_pool_ = (VEmbedPool *)0;

/* stop the model servers on the way out, so a pipe reading our output is not held open by them */
static void cli_close_(void) { vecdb_embpool_close(cli_pool_); cli_pool_ = (VEmbedPool *)0; }

static i32 cli_run_(VecDB *db, const char *cmd, const char *arg)
{
    VHit hits[64];
    i32 buf[4096], n, i;
    (void)n;
    if (!strcmp(cmd, "search")) cli_hits_grouped_(db, hits, vecdb_query(db, arg, 40, 0, hits), 10);
    else if (!strcmp(cmd, "vec")) cli_hits_grouped_(db, hits, vecdb_search(db, arg, 40, 1, -1, hits), 10);       /* vector only */
    else if (!strcmp(cmd, "lex")) cli_hits_grouped_(db, hits, vecdb_search(db, arg, 40, 2, -1, hits), 10);       /* words only */
    else if (!strcmp(cmd, "remove")) {
        i32 id = vecdb_find(db, arg);
        if (id < 0 || !vecdb_remove_note(db, id)) { fprintf(stderr, "no such note: %s\n", arg); return 1; }
        if (vecdb_sync(db, cli_path_) != 0) { fprintf(stderr, "vecdb: could not save\n"); return 1; }
        printf("removed %s\n", arg);
    }
    else if (!strcmp(cmd, "stats"))
        printf("%d notes, %d pictures, %d ghosts (linked, not yet written), %d tags, embedding cache %d\n", vecdb_count(db, VN_NOTE), db->store2 ? store_count(db->store2) : 0, vecdb_count(db, VN_GHOST), vecdb_count(db, VN_TAG), vecdb_embed_cache_count(db));
    else if (!strcmp(cmd, "images")) cli_hits_(db, hits, vecdb_search_images(db, arg, 10, hits));
    else if (!strcmp(cmd, "similar")) {
        i32 id = vecdb_find(db, arg);
        if (id < 0) { fprintf(stderr, "no such picture\n"); return 1; }
        cli_hits_(db, hits, vecdb_similar_images(db, id, 10, hits));
    }
    else if (!strcmp(cmd, "explore")) cli_hits_grouped_(db, hits, vecdb_query(db, arg, 40, 1, hits), 10);
    else if (!strcmp(cmd, "links") || !strcmp(cmd, "backlinks")) {
        i32 id = vecdb_find(db, arg);
        if (id < 0) { fprintf(stderr, "no such note\n"); return 1; }
        n = cmd[0] == 'l' ? vecdb_links(db, id, buf, 4096) : vecdb_backlinks(db, id, buf, 4096);
        cli_names_(db, buf, n);
    } else if (!strcmp(cmd, "tag")) {
        i32 id = vecdb_find_tag(db, arg);
        if (id < 0) { fprintf(stderr, "no such tag\n"); return 1; }
        cli_names_(db, buf, vecdb_notes_with_tag(db, id, buf, 4096));
    } else if (!strcmp(cmd, "tags")) {
        for (i = 0; i < db->nn; ++i) if (db->nodes[i].kind == VN_TAG) printf("  #%-24s %d\n", db->nodes[i].name, db->nodes[i].in.n);
    } else if (!strcmp(cmd, "ghosts")) {
        for (i = 0; i < db->nn; ++i) if (db->nodes[i].kind == VN_GHOST) printf("  %-30s linked from %d\n", db->nodes[i].name, db->nodes[i].in.n);
    } else return 2;
    return 0;
}

int main(int argc, char **argv)
{
    static Arena arena;
    VecDB *db;
    const char *cmd;

    if (argc < 3) { fprintf(stderr, "usage: vecdb DB.vdb index DIR | update DIR | remove TITLE | stats | search Q | images Q | similar PIC | explore Q | links T | backlinks T | tag TAG | tags | ghosts\n"); return 2; }
    cmd = argv[2];
    { const char *ch = getenv("VECDB_CHUNK"), *ml = getenv("EMBED_MAXLEN");
      if (ch && atoi(ch) > 100) cli_chunk_ = atoi(ch); else cli_chunk_ = (ml && atoi(ml) > 16 ? atoi(ml) : 128) * 7 / 2; }
    { static char sp[4096]; char *sl = strrchr(argv[0], '/'); if (getenv("VECDB_EXTRACT")) cli_extract_script_ = getenv("VECDB_EXTRACT"); else if (sl) { *sl = 0; sprintf(sp, "%s/../py/extract.py", argv[0]); *sl = '/'; cli_extract_script_ = sp; } }
    if (!strcmp(cmd, "index")) {
        i32 count = 0;
        struct stat sb;
        if (argc < 4 || stat(argv[3], &sb) != 0 || !S_ISDIR(sb.st_mode)) { fprintf(stderr, "vecdb: %s is not a directory\n", argc < 4 ? "(none)" : argv[3]); return 2; }
        { const char *sm = getenv("VECDB_SMOOTH"), *es = getenv("VECDB_EMBED_SERVER");
          VEmbedPool *ep = es ? vecdb_embpool_open(es) : (VEmbedPool *)0;
          if (es && !ep) { fprintf(stderr, "vecdb: embedding server failed to start: %s\n", es); return 1; }
          cli_pool_ = ep; atexit(cli_close_);
          db = vecdb_new(&arena, ep ? ep->dim : 256);
          if (ep) { const char *cp = getenv("VECDB_EMBED_CACHE"); char *cpath = (char *)malloc(strlen(argv[1]) + 6); vecdb_use_embpool(db, ep); if (cp) strcpy(cpath, cp); else sprintf(cpath, "%s.emb", argv[1]); vecdb_embed_cache_open(db, cpath, es); free(cpath); }
          { const char *cs = getenv("VECDB_CLIP_SERVER");
            if (cs) { VEmbedProc *cp = vecdb_embproc_open(cs, (const char *)0); if (cp) vecdb_enable_images(db, cp); else fprintf(stderr, "vecdb: image model failed to start, images skipped\n"); } }
          cli_index_(db, argv[3], &count);
          cli_flush_(db);
          if (sm) vecdb_smooth(db, (f32)atof(sm)); }
        vecdb_save(db, argv[1]);
        printf("%d notes (%d pictures), %d ghosts, %d tags -> %s\n", vecdb_count(db, VN_NOTE), db->store2 ? store_count(db->store2) : 0, vecdb_count(db, VN_GHOST), vecdb_count(db, VN_TAG), argv[1]);
        return 0;
    }
    { const char *es = getenv("VECDB_EMBED_SERVER");
      VEmbedPool *ep = es ? vecdb_embpool_open(es) : (VEmbedPool *)0;
      db = vecdb_load_with(&arena, argv[1], ep ? vdb_pool_embed_ : (void (*)(void *, const char *, i32, i32, f32 *, i32))0, ep);
      if (db && ep) vecdb_use_embpool(db, ep);
      cli_pool_ = ep; atexit(cli_close_); }
    if (!db) { fprintf(stderr, "vecdb: cannot open %s (run: vecdb %s index DIR)\n", argv[1], argv[1]); return 1; }
    { const char *cs = getenv("VECDB_CLIP_SERVER");
      if (cs && db->store2) { VEmbedProc *cp = vecdb_embproc_open(cs, (const char *)0); if (cp) vecdb_set_image_model(db, cp); } }
    vecdb_build(db);
    cli_path_ = argv[1];
    if (!strcmp(cmd, "update")) {                                /* re-read DIR: changed files re-embedded (unchanged ones come from the cache), vanished ones removed */
        i32 count = 0, i, gone = 0, *toff_len;
        i64 *toff;
        const char *es = getenv("VECDB_EMBED_SERVER");
        if (argc < 4) { fprintf(stderr, "usage: vecdb DB.vdb update DIR\n"); return 2; }
        if (es && db->embed) { const char *cp = getenv("VECDB_EMBED_CACHE"); char *cpath = (char *)malloc(strlen(argv[1]) + 6); if (cp) strcpy(cpath, cp); else sprintf(cpath, "%s.emb", argv[1]); vecdb_embed_cache_open(db, cpath, es); free(cpath); }
        { const char *cs = getenv("VECDB_CLIP_SERVER");
          if (cs && !db->store2) { VEmbedProc *cp = vecdb_embproc_open(cs, (const char *)0); if (cp) vecdb_enable_images(db, cp); } }
        toff = (i64 *)malloc((size_t)(db->nn + 1) * sizeof(i64)); toff_len = (i32 *)malloc((size_t)(db->nn + 1) * sizeof(i32));
        for (i = 0; i < db->nn; ++i) { toff[i] = db->nodes[i].toff; toff_len[i] = db->nodes[i].tlen; }
        { i32 before = db->nn;
          cli_index_(db, argv[3], &count);
          cli_flush_(db);
          for (i = 0; i < before; ++i)                            /* a note whose body was not rewritten was not in DIR any more */
              if (db->nodes[i].kind == VN_NOTE && db->nodes[i].toff == toff[i] && db->nodes[i].tlen == toff_len[i]) { vecdb_remove_note(db, i); ++gone; } }
        free(toff); free(toff_len);
        if (vecdb_sync(db, argv[1]) != 0) { fprintf(stderr, "vecdb: could not save\n"); return 1; }
        printf("%d files read, %d notes removed; now %d notes, %d ghosts, %d tags\n", count, gone, vecdb_count(db, VN_NOTE), vecdb_count(db, VN_GHOST), vecdb_count(db, VN_TAG));
        return 0;
    }
    if (argc < 4 && strcmp(cmd, "tags") && strcmp(cmd, "ghosts") && strcmp(cmd, "repl") && strcmp(cmd, "stats")) return 2;
    if (!strcmp(cmd, "repl")) {                                  /* keep the models loaded: one command per line */
        char line[2048];
        fprintf(stderr, "vecdb> ");
        while (fgets(line, sizeof line, stdin)) {
            char *arg = strchr(line, ' '), *nl = strchr(line, '\n');
            if (nl) *nl = 0;
            if (arg) *arg++ = 0;
            if (!strcmp(line, "quit") || !strcmp(line, "exit")) break;
            if (line[0]) cli_run_(db, line, arg ? arg : "");
            fflush(stdout); fprintf(stderr, "vecdb> ");
        }
        return 0;
    }
    return cli_run_(db, cmd, argc > 3 ? argv[3] : "");
}
