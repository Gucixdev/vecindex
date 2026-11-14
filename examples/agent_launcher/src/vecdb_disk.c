/* vecdb_disk.c - saving and loading: PATH.vdb (nodes), PATH.vdt (note bodies), PATH.vxi (the vectors).
 * Part of vecdb.c, which includes it at this point; not compiled on its own. */

/* ------------------------------------------------------------------ */
/* save / load: PATH.vdb  nodes (kind, title, where the body sits)      */
/*              PATH.vdt  the note bodies (this becomes the live text   */
/*                        file: bodies stay on disk, never held in RAM) */
/*              PATH.vxi  the vector index (rows, meta and the graph)   */
/* Links, tags and the lexicon are rebuilt from the bodies on load       */
/* (~8 us per note); if the .vxi is missing the vectors are recomputed   */
/* with the embedder you pass to vecdb_load_with.                        */
/* ------------------------------------------------------------------ */

#define VDB_MAGIC_V3 0x33424456u  /* 'VDB3': the first release, header = magic dim nn dim2 */
#define VDB_MAGIC 0x58424456u     /* 'VDBX': header = magic version dim nn dim2 flags */
#define VDB_VERSION 5u            /* what this build writes; it reads 3, 4 (legacy) and 5. A higher number is a file from a newer build. */

static char vdb_err_[400] = "";
/* why the last vecdb_load failed ("" if it did not) */
const char *vecdb_error(void) { return vdb_err_; }

static i32 vdb_wi_(FILE *f, i32 x) { return fwrite(&x, 4, 1, f) == 1 ? 0 : -1; }
static i32 vdb_ri_(FILE *f, i32 *x) { return fread(x, 4, 1, f) == 1 ? 0 : -1; }

static char *vdb_ext_(const char *path, const char *ext)
{
    char *p = (char *)malloc(strlen(path) + strlen(ext) + 1);
    strcpy(p, path); strcat(p, ext);
    return p;
}

/* PATH+ext, or PATH+ext+".gXXXXXXXXXXXXXXXX" when gen != 0 - see VecDB.bodygen's comment */
static char *vdb_ext_gen_(const char *path, const char *ext, u64 gen)
{
    char *p;                                                      /* C89 has no 64-bit printf length modifier: format as two u32 halves */
    if (gen == 0) return vdb_ext_(path, ext);
    p = (char *)malloc(strlen(path) + strlen(ext) + 20);
    sprintf(p, "%s%s.g%08lx%08lx", path, ext, (unsigned long)(gen >> 32), (unsigned long)(gen & 0xffffffffu));
    return p;
}

static i32 vdb_write_nodes_(const VecDB *db, const char *path)
{
    char *tmp = vdb_ext_(path, ".tmp");
    FILE *f = fopen(tmp, "wb");                                  /* write beside, fsync, rename: the old table is valid until the new one is complete */
    i32 i;
    if (!f) { free(tmp); return -1; }
    vdb_wi_(f, (i32)VDB_MAGIC); vdb_wi_(f, (i32)VDB_VERSION); vdb_wi_(f, db->dim); vdb_wi_(f, db->nn); vdb_wi_(f, db->store2 ? db->dim2 : 0); vdb_wi_(f, 0);
    fwrite(&db->bodygen, sizeof(u64), 1, f);
    for (i = 0; i < db->nn; ++i) {
        const VNode *v = &db->nodes[i];
        i32 nl = (i32)strlen(v->name);
        vdb_wi_(f, v->kind); vdb_wi_(f, v->space); vdb_wi_(f, nl); fwrite(v->name, 1, (size_t)nl, f);
        fwrite(&v->toff, sizeof(i64), 1, f); vdb_wi_(f, v->tlen);
    }
    if (fflush(f) != 0 || fsync(fileno(f)) != 0 || fclose(f) != 0 || rename(tmp, path) != 0) { remove(tmp); free(tmp); return -1; }
    free(tmp);
    store_fsync_dir_(path);
    return 0;
}

/* True cross-file atomicity: PATH.vdt and PATH.vxi are tightly coupled (byte offsets, row<->id order) and a
 * full rewrite of one without the other mid-crash would let an old node table silently misread a new body file
 * (or vice versa) - two independently-atomic renames are not atomic AS A PAIR. So a full rewrite that would have
 * to replace an ALREADY-EXISTING PATH.vdt/.vxi (re-indexing an existing database from scratch: vecdb_new + save,
 * not vecdb_load) never touches those files at all: it writes the new set under a fresh generation suffix
 * (PATH.vdt.g<hex>, PATH.vxi.g<hex>, in lockstep - db->bodygen), and only the LAST step, the node table (the
 * real commit record, itself atomic), starts pointing at that generation. A crash before it leaves the old
 * generation's files completely untouched (loads as the old database); a crash after leaves the new generation's
 * files complete and fsynced (loads as the new one). Either way, never a mismatched pairing. The one-time cost:
 * an old, now-orphaned generation's files, cleaned up best-effort right after the commit succeeds.
 * A brand-new path (nothing there yet) skips all of this and just uses the bare names - cheap common case. */
static i32 vdbl_save(VecDB *db, const char *path)
{
    i32 i, r, need_full = !db->dbpath || strcmp(db->dbpath, path) != 0;
    char *vp, *old_vdt = (char *)0, *old_vxi = (char *)0, *old_ivxi = (char *)0, *old_vxij = (char *)0;

    if (need_full) {                                              /* attaching to `path` for the first time in this object's life */
        char *barename = vdb_ext_(path, ".vdt");
        struct stat sb;
        u64 gen = 0;
        char *newname, *tmp;
        FILE *nf;
        i64 pos = 0;

        if (stat(barename, &sb) == 0) {                           /* something already lives under the canonical name */
            u64 h = vdb_hash64_((const char *)&sb, sizeof(sb), (u64)time((time_t *)0) ^ (u64)(size_t)db ^ (u64)getpid());
            gen = h ? h : 1;                                       /* 0 is reserved to mean "bare name" */
            old_vdt = barename;                                    /* remember it: delete it once the new generation is safely committed */
            old_vxi = vdb_ext_(path, ".vxi");
            old_ivxi = vdb_ext_(path, ".img.vxi");
            old_vxij = vdb_ext_(path, ".vxi.j");
        } else free(barename);

        newname = vdb_ext_gen_(path, ".vdt", gen);
        tmp = vdb_ext_(newname, ".tmp");
        nf = fopen(tmp, "w+b");
        if (!nf) { free(tmp); free(newname); free(old_vdt); free(old_vxi); free(old_ivxi); free(old_vxij); return -1; }
        for (i = 0; i < db->nn; ++i) {
            char *t;
            if (db->nodes[i].kind != VN_NOTE) continue;
            t = vdbl_text(db, i);
            if (!t) { db->nodes[i].tlen = -1; continue; }
            if (db->nodes[i].tlen > 0) fwrite(t, 1, (size_t)db->nodes[i].tlen, nf);
            db->nodes[i].toff = pos; pos += db->nodes[i].tlen;
            free(t);
        }
        if (store_fsync_(nf) != 0 || rename(tmp, newname) != 0) {
            fclose(nf); remove(tmp); free(tmp); free(newname); free(old_vdt); free(old_vxi); free(old_ivxi); free(old_vxij);
            return -1;
        }
        store_fsync_dir_(newname);
        free(tmp);
        if (db->tf) fclose(db->tf);                                /* nf keeps working after the rename: same fd, same inode, new name */
        db->tf = nf; free(db->tpath); db->tpath = newname; db->tend = pos;
        db->bodygen = gen;
        free(db->dbpath); db->dbpath = (char *)malloc(strlen(path) + 1); strcpy(db->dbpath, path);
    }
    if (db->tf) store_fsync_(db->tf);

    /* commit order: bodies (already fsynced above), then the vector files (own atomic tmp+rename), then the
     * node table LAST - it is the commit record. A crash before it leaves the previous table valid; load then
     * sees the counts disagree and re-embeds (the embedding cache makes that cheap). */
    vdbl_build(db);
    vp = vdb_ext_gen_(path, ".vxi", db->bodygen);
    r = store_save(db->store, vp);
    if (r == 0 && db->store2) { char *ip = vdb_ext_gen_(path, ".img.vxi", db->bodygen); r = store_save(db->store2, ip); free(ip); }
    free(vp);
    if (r == 0) r = vdb_write_nodes_(db, path);
    if (r == 0 && old_vdt) { remove(old_vdt); remove(old_vxi); remove(old_ivxi); remove(old_vxij); }   /* best-effort: an orphan left behind is harmless */
    free(old_vdt); free(old_vxi); free(old_ivxi); free(old_vxij);
    return r == 0 ? 0 : -1;
}

/* Durable save that costs what changed, not what exists: bodies are already appended in place, the small node
 * table is rewritten, and the vector stores only flush their journals (see store_sync). The first call for a
 * path (or a path this object was not already attached to) does a full vecdb_save. */
static i32 vdbl_sync(VecDB *db, const char *path)
{
    char *vp;
    i32 r, warm = db->dbpath && strcmp(db->dbpath, path) == 0;
    if (warm) { vp = vdb_ext_gen_(path, ".vxi", db->bodygen); warm = db->store->jpath && strcmp(db->store->jpath, vp) == 0; if (!warm) free(vp); }
    if (!warm) return vdbl_save(db, path);
    r = db->tf ? store_fsync_(db->tf) : 0;                       /* bodies, then journals, then the node table (the commit record) */
    if (r == 0) r = store_sync(db->store, vp);
    if (r == 0 && db->store2) { char *ip = vdb_ext_gen_(path, ".img.vxi", db->bodygen); r = store_sync(db->store2, ip); free(ip); }
    free(vp);
    if (r != 0) return -1;
    return vdb_write_nodes_(db, path) == 0 ? 0 : -1;
}

VecDB *vecdb_load_with(Arena *a, const char *path, void (*embed)(void *, const char *, i32, i32, f32 *, i32), void *ctx)
{
    FILE *f = fopen(path, "rb");
    VecDB *db = (VecDB *)0;
    i32 magic, dim = 0, nn = 0, i, nnotes = 0, nimg = 0, use, use2 = 1, dim2 = 0;
    char **names = (char **)0, *tp, *vp;
    u8 *kinds = (u8 *)0, *spaces = (u8 *)0;
    i64 *toff = (i64 *)0; i32 *tlen = (i32 *)0;
    RagStore *st;
    i64 tsize;
    u64 bodygen = 0;

    vdb_err_[0] = 0;
    if (!f) { sprintf(vdb_err_, "cannot open %.100s", path); return db; }
    if (vdb_ri_(f, &magic)) { sprintf(vdb_err_, "%.100s is empty or truncated", path); fclose(f); return db; }
    if ((u32)magic == VDB_MAGIC) {
        i32 ver, flags;
        if (vdb_ri_(f, &ver) || vdb_ri_(f, &dim) || vdb_ri_(f, &nn) || vdb_ri_(f, &dim2) || vdb_ri_(f, &flags)) { sprintf(vdb_err_, "%.100s: header truncated", path); fclose(f); return db; }
        if ((u32)ver > VDB_VERSION) { sprintf(vdb_err_, "%.100s was written by a newer vecdb (format %d, this build reads up to %u)", path, ver, VDB_VERSION); fclose(f); return db; }
        if (ver >= 5 && fread(&bodygen, sizeof(u64), 1, f) != 1) { sprintf(vdb_err_, "%.100s: header truncated", path); fclose(f); return db; }
    } else if ((u32)magic == VDB_MAGIC_V3) {                     /* legacy: same node table, no version/flags/bodygen in the header */
        if (vdb_ri_(f, &dim) || vdb_ri_(f, &nn) || vdb_ri_(f, &dim2)) { sprintf(vdb_err_, "%.100s: header truncated", path); fclose(f); return db; }
    } else { sprintf(vdb_err_, "%.100s is not a vecdb file", path); fclose(f); return db; }
    if (dim <= 0 || nn < 0 || dim2 < 0) { sprintf(vdb_err_, "%.100s: corrupt header", path); fclose(f); return db; }
    names = (char **)calloc((size_t)nn + 1, sizeof(char *)); kinds = (u8 *)calloc((size_t)nn + 1, 1); spaces = (u8 *)calloc((size_t)nn + 1, 1);
    toff = (i64 *)calloc((size_t)nn + 1, sizeof(i64)); tlen = (i32 *)calloc((size_t)nn + 1, sizeof(i32));
    for (i = 0; i < nn; ++i) {
        i32 kind, space, nl;
        if (vdb_ri_(f, &kind) || vdb_ri_(f, &space) || vdb_ri_(f, &nl) || nl < 0) goto bad;
        names[i] = (char *)malloc((size_t)nl + 1);
        if (fread(names[i], 1, (size_t)nl, f) != (size_t)nl) goto bad;
        names[i][nl] = 0; kinds[i] = (u8)kind; spaces[i] = (u8)space;
        if (fread(&toff[i], sizeof(i64), 1, f) != 1 || vdb_ri_(f, &tlen[i])) goto bad;
    }
    fclose(f); f = (FILE *)0;

    db = vecdb_new(a, dim);
    db->embed = embed; db->embed_ctx = ctx;
    db->bodygen = bodygen;
    db->dbpath = (char *)malloc(strlen(path) + 1); strcpy(db->dbpath, path);
    tp = vdb_ext_gen_(path, ".vdt", bodygen); vp = vdb_ext_gen_(path, ".vxi", bodygen);
    if (db->tf) fclose(db->tf);
    db->tf = fopen(tp, "r+b"); db->tpath = tp;
    if (db->tf) { fseeko(db->tf, 0, SEEK_END); tsize = (i64)ftello(db->tf); db->tend = tsize; }

    /* recreate the nodes in their original order so ids (= store doc ids) stay valid */
    for (i = 0; i < nn; ++i) {
        char *key = (char *)malloc(strlen(names[i]) + 1);
        i32 kl;
        if (kinds[i] == VN_TAG) { kl = vdb_norm_(names[i], (i32)strlen(names[i]), key, 0); vdb_tag_node_(db, key, kl); }
        else if (kinds[i] == VN_DEAD) { vdb_newnode_(db, VN_DEAD, names[i], (i32)strlen(names[i])); }
        else { kl = vdb_norm_(names[i], (i32)strlen(names[i]), key, 1); vdb_title_node_(db, key, kl, 1); }
        free(key);
        if (kinds[i] == VN_NOTE) { db->nodes[i].toff = toff[i]; db->nodes[i].tlen = tlen[i]; db->nodes[i].space = spaces[i]; if (spaces[i]) ++nimg; else ++nnotes; }
    }
    st = store_load(a, vp);
    use = st && store_count(st) == nnotes;
    if (use) { store_destroy(db->store); db->store = st; store_set_fast(st, 0.97f); }
    else if (st) store_destroy(st);
    if (dim2 > 0) {                                              /* the image space (needs its model to be searched, not to be loaded) */
        char *ip = vdb_ext_gen_(path, ".img.vxi", bodygen);
        RagStore *s2 = store_load(a, ip);
        db->dim2 = dim2;
        use2 = s2 && store_count(s2) == nimg;
        if (use2) { db->store2 = s2; store_set_fast(s2, 0.97f); }
        else { if (s2) store_destroy(s2); db->store2 = store_new(a, dim2); store_set_fast(db->store2, 0.97f); }
        free(ip);
    }
    for (i = 0; i < nn; ++i) {
        char *t;
        if (kinds[i] != VN_NOTE) continue;
        t = vdbl_text(db, i);
        if (!t) continue;
        if (spaces[i]) vdb_add_(db, names[i], t, (const f32 *)0, 0, 0, 1);   /* image vectors cannot be recomputed from text: an image index without its .img.vxi is lost */
        else if (use) vdb_add_(db, names[i], t, (const f32 *)0, 0, 0, 0);
        else { f32 *vec = (f32 *)malloc((size_t)dim * sizeof(f32)); vdb_embed_doc_(db, names[i], t, vec); vdb_add_(db, names[i], t, vec, 1, 0, 0); free(vec); }
        free(t);
    }
    (void)use2;
    free(vp);
    for (i = 0; i < nn; ++i) free(names[i]);
    free(names); free(kinds); free(spaces); free(toff); free(tlen);
    return db;
bad:
    if (f) fclose(f);
    for (i = 0; i < nn; ++i) free(names[i]);
    free(names); free(kinds); free(spaces); free(toff); free(tlen);
    return (VecDB *)0;
}

VecDB *vecdb_load(Arena *a, const char *path) { return vecdb_load_with(a, path, (void (*)(void *, const char *, i32, i32, f32 *, i32))0, (void *)0); }
