/* vecdb_search.c - hybrid search: vector kNN and BM25 fused by reciprocal rank, and the image space.
 * Part of vecdb.c, which includes it at this point; not compiled on its own. */

/* ------------------------------------------------------------------ */
/* search                                                              */
/* ------------------------------------------------------------------ */

#define VDB_RRF_K 60.0f

static f32 vdb_dot_(const f32 *a, const f32 *b, i32 n)
{
    f32 s = 0.0f; i32 i;
    for (i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

/* mode: 0 hybrid (vector + BM25, reciprocal rank fusion), 1 vector only, 2 lexical only.
 * tag >= 0 keeps only notes carrying that tag node. Returns the number of hits. */
/* candidate list of one query: at most a few dozen ids, so a linear find beats any dense O(nodes) array */
typedef struct { i32 id[256]; f32 fused[256], sim[256], lex[256]; i32 n; } VCand;

static i32 vdb_cand_(VCand *c, i32 id)
{
    i32 i;
    for (i = 0; i < c->n; ++i) if (c->id[i] == id) return i;
    if (c->n >= 256) return -1;
    c->id[c->n] = id; c->fused[c->n] = 0.0f; c->sim[c->n] = 0.0f; c->lex[c->n] = 0.0f;
    return c->n++;
}

/* search that also hands back the lexical hits (explore reuses them as rewards) */
static i32 vdb_search_(VecDB *db, const char *query, i32 k, i32 mode, i32 tag, VHit *out, i32 lexcap, i64 *lid_out, f32 *lsc_out, i32 *nl_out, f32 *qout)
{
    i32 over = k * 3 + 10, nv = 0, nl = 0, i, n, exact = -1;
    i64 vid[256], lid[256];
    f32 vsc[256], lsc[256], *q = (f32 *)malloc((size_t)db->dim * sizeof(f32));
    VCand *c = (VCand *)malloc(sizeof(VCand));
    i32 qlen = (i32)strlen(query);

    if (over < lexcap) over = lexcap;
    if (over > 250) over = 250;
    c->n = 0;
    vdb_embed_(db, query, qlen, 1, q);
    if (mode != 2 && !(db->embed && vdb_zero_(q, db->dim))) {          /* a failed query embedding falls back to words only */
        i32 bit = tag >= 0 ? db->nodes[tag].tagbit : -1;
        nv = bit >= 0 ? store_search_filter(db->store, q, over, vid, vsc, (i64)1 << bit, 0) : store_search(db->store, q, over, vid, vsc);
    }
    if (mode != 1) nl = lex_search(&db->lex, query, qlen, over, lid, lsc);
    /* Rank fusion (RRF), weighted. Base weights favour the words (1.5 vs 0.6): a query taken from deep inside a
     * note is invisible to a vector that only read the first tokens, while exact words are never wrong in that
     * way. A side whose best hit clearly stands out from its runner-up (a rare word, a near-duplicate) gets up
     * to +2, so a confident vector still wins. Plain RRF ties the sides and loses exact hits. */
    { f32 wv = 0.6f, wl = 1.5f;      /* base weights: measured on Wikipedia with two embedders, words beat a blind vector */
      if (nv > 1 && vsc[0] < 0.0f && vsc[1] < 0.0f) { f32 m = 1.0f - vsc[0] / vsc[1]; wv += 2.0f * (m > 0 ? (m < 1 ? m : 1.0f) : 0.0f); }
      if (nl == 1) wl = 3.0f;
      else if (nl > 1 && lsc[0] > 0.0f) wl += 2.0f * (1.0f - lsc[1] / lsc[0]);
      for (i = 0; i < nv; ++i) {
          i32 ci = vdb_cand_(c, (i32)vid[i]);
          if (ci < 0) continue;
          c->fused[ci] += wv / (VDB_RRF_K + (f32)i + 1.0f);
      }
      for (i = 0; i < nl; ++i) {
          i32 ci = vdb_cand_(c, (i32)lid[i]);
          if (ci < 0) continue;
          c->lex[ci] = lsc[0] > 0 ? lsc[i] / lsc[0] : 0.0f;
          c->fused[ci] += wl / (VDB_RRF_K + (f32)i + 1.0f);
      } }
    /* the hash dictionary: a query that IS a title goes straight to that note, first */
    if (mode != 1) {
        char *key = (char *)malloc((size_t)qlen + 1);
        i32 kl = vdb_norm_(query, qlen, key, 1), tid = kl > 0 ? hs_get(&db->names, key, kl) : -1;
        free(key);
        if (tid >= 0 && db->nodes[tid].kind == VN_NOTE) { i32 ci = vdb_cand_(c, tid); if (ci >= 0) { c->fused[ci] += 10.0f; exact = tid; } }
    }
    /* the vector side only knows the cosine of its own hits; fill in the rest */
    for (i = 0; i < c->n; ++i)
        if (c->sim[i] == 0.0f && vdb_vec_(db, c->id[i]) && db->nodes[c->id[i]].kind == VN_NOTE) c->sim[i] = vdb_dot_(q, vdb_vec_(db, c->id[i]), db->dim);
    if (tag >= 0) {                                             /* keep notes that carry the tag */
        i32 w = 0;
        for (i = 0; i < c->n; ++i) if (vivec_has_(&db->nodes[c->id[i]].out, tag)) {
            c->id[w] = c->id[i]; c->fused[w] = c->fused[i]; c->sim[w] = c->sim[i]; c->lex[w] = c->lex[i]; ++w;
        }
        c->n = w;
    }
    (void)exact;
    /* selection sort of the top k */
    n = 0;
    for (i = 0; i < c->n && n < k; ++i) {
        i32 j, best = i;
        f32 tf;
        for (j = i + 1; j < c->n; ++j) if (c->fused[j] > c->fused[best]) best = j;
        if (best != i) {
            i32 ti = c->id[i]; c->id[i] = c->id[best]; c->id[best] = ti;
            tf = c->fused[i]; c->fused[i] = c->fused[best]; c->fused[best] = tf;
            tf = c->sim[i]; c->sim[i] = c->sim[best]; c->sim[best] = tf;
            tf = c->lex[i]; c->lex[i] = c->lex[best]; c->lex[best] = tf;
        }
        out[n].node = c->id[i]; out[n].sim = c->sim[i]; out[n].lex = c->lex[i];
        out[n].score = c->fused[i]; out[n].via = -1; out[n].depth = 0; ++n;
    }
    if (lid_out) { memcpy(lid_out, lid, (size_t)nl * sizeof(i64)); memcpy(lsc_out, lsc, (size_t)nl * sizeof(f32)); *nl_out = nl; }
    if (qout) memcpy(qout, q, (size_t)db->dim * sizeof(f32));
    free(q); free(c);
    return n;
}

static i32 vdbl_search(VecDB *db, const char *query, i32 k, i32 mode, i32 tag, VHit *out)
{ return vdb_search_(db, query, k, mode, tag, out, 0, (i64 *)0, (f32 *)0, (i32 *)0, (f32 *)0); }


/* ---- images: a second vector space, filled by an image model (CLIP) that also embeds text queries into it ---- */

/* start the image space; p is a server started with py/embedsrv.py --clip */
void vecdb_enable_images(VecDB *db, VEmbedProc *p)
{
    db->proc2 = p; db->dim2 = p->dim;
    if (!db->store2) { db->store2 = store_new(db->arena, p->dim); store_set_fast(db->store2, 0.97f); }
}

/* after vecdb_load: attach the model so text queries can search a loaded image space */
void vecdb_set_image_model(VecDB *db, VEmbedProc *p) { if (db->store2 && p && p->dim == db->dim2) db->proc2 = p; }

/* Index a picture. `title` is how notes link to it ([[title]], ![[title]], ![](dir/title)); `caption`
 * (may be NULL) is searchable text. Returns the node id, or -1 if the model could not read the file. */
i32 vecdb_add_image(VecDB *db, const char *title, const char *path, const char *caption)
{
    VEmbedProc *p = (VEmbedProc *)db->proc2;
    f32 *vec;
    char *body;
    i32 id;
    if (!p || !db->store2) return -1;
    vec = (f32 *)malloc((size_t)db->dim2 * sizeof(f32));
    if (vecdb_embproc_ask(p, 'I', path, (i32)strlen(path), vec) != 0) { free(vec); return -1; }
    { i32 i; f32 s = 0.0f; for (i = 0; i < db->dim2; ++i) s += vec[i] * vec[i]; if (s < 1e-6f) { free(vec); return -1; } }
    body = (char *)malloc(strlen(caption ? caption : "") + strlen(path) + 32);
    sprintf(body, "%s\nfile: %s", caption ? caption : "", path);
    pthread_rwlock_wrlock(&db->lk);
    id = vdb_add_(db, title, body, vec, 1, 1, 1);
    pthread_rwlock_unlock(&db->lk);
    free(vec); free(body);
    return id;
}

/* text -> pictures: the query goes through the image model's text tower; BM25 over captions and file
 * names is fused in, so a word in a file name still wins. Only image notes are returned. */
static i32 vdbl_search_images(VecDB *db, const char *query, i32 k, VHit *out)
{
    VEmbedProc *p = (VEmbedProc *)db->proc2;
    i64 vid[128], lid[128]; f32 vsc[128], lsc[128], *q;
    VCand *c;
    i32 nv, nl, i, n = 0, over = k * 4 + 10;
    if (!p || !db->store2) return 0;
    if (over > 120) over = 120;
    q = (f32 *)malloc((size_t)db->dim2 * sizeof(f32));
    if (vecdb_embproc_ask(p, 'Q', query, (i32)strlen(query), q) != 0) { free(q); return 0; }
    c = (VCand *)malloc(sizeof(VCand)); c->n = 0;
    nv = store_search(db->store2, q, over, vid, vsc);
    nl = lex_search(&db->lex, query, (i32)strlen(query), over, lid, lsc);
    for (i = 0; i < nv; ++i) { i32 ci = vdb_cand_(c, (i32)vid[i]); if (ci >= 0) { c->fused[ci] += 1.0f / (VDB_RRF_K + (f32)i + 1.0f); c->sim[ci] = vdb_dot_(q, store_vec_ptr(db->store2, vid[i]), db->dim2); } }
    for (i = 0; i < nl; ++i) if (db->nodes[(i32)lid[i]].space) { i32 ci = vdb_cand_(c, (i32)lid[i]); if (ci >= 0) { c->fused[ci] += 1.0f / (VDB_RRF_K + (f32)i + 1.0f); c->lex[ci] = lsc[0] > 0 ? lsc[i] / lsc[0] : 0.0f; } }
    for (i = 0; i < c->n && n < k; ++i) {
        i32 j, best = i;
        for (j = i + 1; j < c->n; ++j) if (c->fused[j] > c->fused[best]) best = j;
        if (best != i) {
            i32 ti = c->id[i]; f32 tf;
            c->id[i] = c->id[best]; c->id[best] = ti;
            tf = c->fused[i]; c->fused[i] = c->fused[best]; c->fused[best] = tf;
            tf = c->sim[i]; c->sim[i] = c->sim[best]; c->sim[best] = tf;
            tf = c->lex[i]; c->lex[i] = c->lex[best]; c->lex[best] = tf;
        }
        out[n].node = c->id[i]; out[n].sim = c->sim[i]; out[n].lex = c->lex[i]; out[n].score = c->fused[i]; out[n].via = -1; out[n].depth = 0; ++n;
    }
    free(q); free(c);
    return n;
}

/* picture -> similar pictures (no model needed: uses the stored vector) */
static i32 vdbl_similar_images(VecDB *db, i32 id, i32 k, VHit *out)
{
    const f32 *q;
    i64 vid[64]; f32 vsc[64];
    i32 nv, i, n = 0;
    if (!db->store2 || id < 0 || id >= db->nn || !db->nodes[id].space) return 0;
    q = store_vec_ptr(db->store2, (i64)id);
    if (!q) return 0;
    nv = store_search(db->store2, q, k + 1 > 64 ? 64 : k + 1, vid, vsc);
    for (i = 0; i < nv && n < k; ++i) {
        if ((i32)vid[i] == id) continue;
        out[n].node = (i32)vid[i]; out[n].sim = vdb_dot_(q, store_vec_ptr(db->store2, vid[i]), db->dim2); out[n].lex = 0.0f;
        out[n].score = out[n].sim; out[n].via = -1; out[n].depth = 0; ++n;
    }
    for (i = 1; i < n; ++i) {                                    /* the index returns candidates, not an exact order */
        VHit x = out[i]; i32 j;
        for (j = i; j > 0 && out[j - 1].score < x.score; --j) out[j] = out[j - 1];
        out[j] = x;
    }
    return n;
}

