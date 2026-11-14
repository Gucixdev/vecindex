/* mem_disk.c - persistence: PATH.mem (the side-table) and PATH.mem.cold (the archive).
 * Part of mem.c, which includes it at this point; not compiled on its own. */

/* ------------------------------------------------------------------ */
/* persistence: PATH.mem - the side-table; PATH.mem.cold - see mem.c  */
/* ------------------------------------------------------------------ */

#define MEM_MAGIC 0x314d454du   /* 'MEM1' */

/* Save the side-table: written beside the target, fsynced, renamed in - same discipline as
 * vecdb's own files (see "Cross-file atomicity" in docs/vecdb.md). This file is a pure cache
 * of derived bookkeeping (decay clocks, provenance, corroboration) keyed by vecdb node id, not
 * itself a source of truth - a missing or unreadable PATH.mem just means every node starts back
 * at MK_EVENT with a fresh clock and no history, never a corrupt READ (mem_load fails closed).
 * Archived (already-removed) entries are not written - nothing legitimate should still reference
 * them once mem_fix has run. Returns 0 on success. */
i32 mem_save(Mem *m, const char *path)
{
    char *tmp = (char *)malloc(strlen(path) + 6);
    FILE *f;
    i32 i, live = 0;
    sprintf(tmp, "%s.tmp", path);
    f = fopen(tmp, "wb");
    if (!f) { free(tmp); return -1; }
    { u32 magic = MEM_MAGIC; fwrite(&magic, 4, 1, f); }
    pthread_rwlock_rdlock(&m->lk);                                /* a consistent snapshot of the table while it is serialised */
    fwrite(m->half_life, sizeof(f32), 3, f);
    for (i = 0; i < m->n; ++i) if (!m->meta[i].archived) ++live;
    fwrite(&live, 4, 1, f);
    for (i = 0; i < m->n; ++i) {
        MemMeta *e = &m->meta[i];
        i32 kind;
        if (e->archived) continue;
        kind = (i32)e->kind;
        fwrite(&e->vid, 4, 1, f); fwrite(&kind, 4, 1, f);
        fwrite(&e->born_at, 8, 1, f); fwrite(&e->last_touch, 8, 1, f);
        fwrite(&e->access_count, 4, 1, f); fwrite(&e->corroboration, 4, 1, f); fwrite(&e->contradiction, 4, 1, f);
        fwrite(&e->nprov, 4, 1, f);
        if (e->nprov > 0) fwrite(e->prov, 4, (size_t)e->nprov, f);
    }
    pthread_rwlock_unlock(&m->lk);
    if (fflush(f) != 0 || fsync(fileno(f)) != 0 || fclose(f) != 0 || rename(tmp, path) != 0) { remove(tmp); free(tmp); return -1; }
    free(tmp);
    store_fsync_dir_(path);
    return 0;
}

/* Load the side-table; call after vecdb_load so vids already exist. A vid the table names that
 * is not (any more) a live vecdb note is silently dropped (this is exactly what mem_fix would
 * do to it anyway). Returns 0 on success, -1 if the file is absent/unreadable (the caller then
 * has an empty, working Mem - see mem_save's comment on why that is a safe default). */
i32 mem_load(Mem *m, const char *path)
{
    FILE *f = fopen(path, "rb");
    u32 magic;
    i32 live, i;
    if (!f) return -1;
    if (fread(&magic, 4, 1, f) != 1 || magic != MEM_MAGIC || fread(m->half_life, sizeof(f32), 3, f) != 3 || fread(&live, 4, 1, f) != 1) { fclose(f); return -1; }
    for (i = 0; i < live; ++i) {
        i32 vid, kind, ac, cor, con, nprov;
        i64 born, touch;
        if (fread(&vid, 4, 1, f) != 1 || fread(&kind, 4, 1, f) != 1 || fread(&born, 8, 1, f) != 1 || fread(&touch, 8, 1, f) != 1
            || fread(&ac, 4, 1, f) != 1 || fread(&cor, 4, 1, f) != 1 || fread(&con, 4, 1, f) != 1 || fread(&nprov, 4, 1, f) != 1 || nprov < 0) { fclose(f); return -1; }
        if (vid >= 0 && vid < m->db->nn && m->db->nodes[vid].kind == VN_NOTE) {
            MemMeta *e = mem_new_meta_(m, vid, (MemKind)kind);
            e->born_at = born; e->last_touch = touch; e->access_count = ac; e->corroboration = cor; e->contradiction = con;
            if (nprov > 0) { e->prov = (i32 *)malloc((size_t)nprov * sizeof(i32)); if (fread(e->prov, 4, (size_t)nprov, f) != (size_t)nprov) { fclose(f); return -1; } e->nprov = e->provcap = nprov; }
        } else if (nprov > 0) { fseeko(f, (off_t)nprov * 4, SEEK_CUR); }
    }
    fclose(f);
    return 0;
}
