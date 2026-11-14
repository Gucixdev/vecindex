/* vecdb_graph.c - Monte Carlo tree search over links, backlinks and tags, and the query language that mixes words with graph conditions.
 * Part of vecdb.c, which includes it at this point; not compiled on its own. */

/* ------------------------------------------------------------------ */
/* MCTS over the graph                                                 */
/* ------------------------------------------------------------------ */

typedef struct {
    i32 g;              /* graph node */
    i32 parent;         /* tree parent, -1 for the seeds' virtual root */
    i32 depth;
    i32 first_child, next_sib, nchild;
    i32 nact, tried, off;   /* neighbours of g: how many, how many were tried, random start offset */
    i32 visits;
    f64 wsum;
} VTree;

static u32 vdb_rng_(u32 *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }

static i32 vdb_nb_(const VecDB *db, i32 g, i32 idx)
{
    const VNode *v = &db->nodes[g];
    return idx < v->out.n ? v->out.v[idx] : v->in.v[idx - v->out.n];
}

static i32 vdb_on_path_(const VTree *t, i32 node, i32 g)
{
    for (; node >= 0; node = t[node].parent) if (t[node].g == g) return 1;
    return 0;
}

/* per-query state of the nodes MCTS touched; sparse (hash id -> slot), so a query costs
 * O(work done), never O(nodes in the database) */
typedef struct {
    const VecDB *db; const f32 *q;
    const i64 *lid; const f32 *lsc; i32 nl;
    H64 map;
    i32 *node, *vis, *via, *dep; f32 *rw; f64 *val;
    i32 n, cap;
} VSlots;

static i32 vs_slot_(VSlots *s, i32 g)
{
    i32 i = h64_get(&s->map, (i64)g);
    if (i >= 0) return i;
    if (s->n >= s->cap) return s->cap - 1;                    /* never happens: cap is a proven bound */
    i = s->n++;
    { const VNode *v = &s->db->nodes[g];
      f32 r = 0.0f;
      i32 j;
      for (j = 0; j < s->nl; ++j) if ((i32)s->lid[j] == g) { r = s->lsc[0] > 0 ? 0.4f * s->lsc[j] / s->lsc[0] : 0.0f; break; }
      if (v->kind == VN_NOTE && vdb_vec_(s->db, g)) { f32 c = vdb_dot_(s->q, vdb_vec_(s->db, g), s->db->dim); if (c > 0) r += 0.6f * c; }
      s->node[i] = g; s->rw[i] = r; s->vis[i] = 0; s->via[i] = -1; s->dep[i] = 0; s->val[i] = 0.0;
      h64_put(&s->map, (i64)g, i); }
    return i;
}
#define RW(g) (S.rw[vs_slot_(&S, (g))])

void vecdb_explore_defaults(VExplore *o)
{
    o->seeds = 6; o->iters = 600; o->maxdepth = 4; o->rollout = 3;
    o->c = 1.0f; o->gamma = 0.8f; o->seed = 1u;
}

/* Monte Carlo tree search from the query's hybrid hits. Returns up to k notes,
 * best first; hit.via / hit.depth say how each was reached. */
static i32 vdbl_explore(VecDB *db, const char *query, i32 k, const VExplore *opt, VHit *out)
{
    VExplore o;
    i32 nseed, i, it, ntree = 0, cap, n = 0, nlx = 0, scap;
    f32 *q = (f32 *)malloc((size_t)db->dim * sizeof(f32));
    i64 lid[256]; f32 lsc[256];
    VSlots S;
    VHit *seeds;
    VTree *t;
    i32 first_root = -1, last_root = -1, nroots = 0, *ord;
    u32 rng;
    f64 lg;

    if (opt) o = *opt; else vecdb_explore_defaults(&o);
    rng = o.seed ? o.seed : 1u;

    seeds = (VHit *)malloc((size_t)(o.seeds + 1) * sizeof(VHit));
    nseed = vdb_search_(db, query, o.seeds, 0, -1, seeds, 64, lid, lsc, &nlx, q);   /* hybrid seeds; its BM25 hits become rewards */
    scap = o.seeds + o.iters * (1 + o.rollout) + 16;
    S.db = db; S.q = q; S.lid = lid; S.lsc = lsc; S.nl = nlx; S.n = 0; S.cap = scap;
    h64_init(&S.map, scap);
    S.node = (i32 *)malloc((size_t)scap * sizeof(i32)); S.vis = (i32 *)malloc((size_t)scap * sizeof(i32));
    S.via = (i32 *)malloc((size_t)scap * sizeof(i32)); S.dep = (i32 *)malloc((size_t)scap * sizeof(i32));
    S.rw = (f32 *)malloc((size_t)scap * sizeof(f32)); S.val = (f64 *)malloc((size_t)scap * sizeof(f64));
    cap = nseed + o.iters + 8;
    t = (VTree *)calloc((size_t)cap, sizeof(VTree));
    for (i = 0; i < nseed; ++i) {                               /* the seeds are the root's children */
        VTree *r = &t[ntree];
        r->g = seeds[i].node; r->parent = -1; r->depth = 0; r->first_child = r->next_sib = -1;
        r->nact = db->nodes[r->g].out.n + db->nodes[r->g].in.n;
        r->off = r->nact ? (i32)(vdb_rng_(&rng) % (u32)r->nact) : 0;
        { i32 sl = vs_slot_(&S, r->g); S.vis[sl] += 1; S.val[sl] += S.rw[sl]; }
        if (first_root < 0) first_root = ntree; else t[last_root].next_sib = ntree;
        last_root = ntree; ++nroots; ++ntree;
    }
    for (it = 1; it <= o.iters && nroots > 0; ++it) {
        i32 cur = -1, node = -1, leafg, step, prevg, depth;
        f64 rew, best;
        lg = log((f64)it + 1.0);
        /* selection: walk down while the node is fully expanded */
        { i32 c0 = first_root;
          for (;;) {
              i32 c, bestc = -1;
              f64 bestv = -1.0;
              for (c = c0; c >= 0; c = t[c].next_sib) {           /* children of cur (or the roots) */
                  f64 v = t[c].visits == 0 ? 1e18 : t[c].wsum / t[c].visits + (f64)o.c * sqrt(lg / t[c].visits);
                  if (v > bestv) { bestv = v; bestc = c; }
              }
              if (bestc < 0) break;
              cur = bestc;
              if (t[cur].visits == 0) { node = cur; break; }      /* an unvisited seed or child: simulate it */
              if (t[cur].tried < t[cur].nact && t[cur].depth < o.maxdepth) {
                  /* expansion: the next neighbour not yet on the path */
                  i32 add = -1;
                  while (t[cur].tried < t[cur].nact) {
                      i32 g = vdb_nb_(db, t[cur].g, (t[cur].off + t[cur].tried++) % t[cur].nact);
                      if (db->nodes[g].kind != VN_DEAD && !vdb_on_path_(t, cur, g)) { add = g; break; }
                  }
                  if (add >= 0 && ntree < cap) {
                      VTree *nw = &t[ntree];
                      nw->g = add; nw->parent = cur; nw->depth = t[cur].depth + 1; nw->first_child = nw->next_sib = -1;
                      nw->nact = db->nodes[add].out.n + db->nodes[add].in.n;
                      nw->off = nw->nact ? (i32)(vdb_rng_(&rng) % (u32)nw->nact) : 0;
                      nw->next_sib = t[cur].first_child; t[cur].first_child = ntree; ++t[cur].nchild;
                      node = ntree++;
                      break;
                  }
              }
              if (t[cur].first_child < 0) { node = cur; break; }  /* terminal */
              c0 = t[cur].first_child;
          } }
        if (node < 0) continue;
        leafg = t[node].g;
        /* simulation: a short random walk; the value is the best (discounted) reward met on it */
        rew = RW(leafg); best = rew; prevg = -1; depth = t[node].depth;
        { i32 g = leafg;
          f64 disc = 1.0;
          for (step = 0; step < o.rollout; ++step) {
              i32 tot = db->nodes[g].out.n + db->nodes[g].in.n, tries, ng = -1;
              if (tot == 0) break;
              for (tries = 0; tries < 4; ++tries) {
                  i32 cand = vdb_nb_(db, g, (i32)(vdb_rng_(&rng) % (u32)tot));
                  if (cand != prevg && db->nodes[cand].kind != VN_DEAD) { ng = cand; break; }
              }
              if (ng < 0) break;
              prevg = g; g = ng; disc *= (f64)o.gamma;
              { f64 rg = disc * RW(g); if (rg > best) best = rg; }
          } }
        rew = best;
        { i32 sl = vs_slot_(&S, leafg);
          if (t[node].visits == 0 && S.vis[sl] == 0) { S.via[sl] = t[node].parent >= 0 ? t[t[node].parent].g : -1; S.dep[sl] = depth; }
          S.vis[sl] += 1; S.val[sl] += rew; }
        for (cur = node; cur >= 0; cur = t[cur].parent) { t[cur].visits += 1; t[cur].wsum += rew; }
    }
    /* rank the notes that were reached: half its own answer, half what the tree found through it */
    ord = (i32 *)malloc((size_t)(S.n + 1) * sizeof(i32));
    { f32 *sc = (f32 *)malloc((size_t)(S.n + 1) * sizeof(f32));
      for (i = 0; i < S.n; ++i) {
          sc[i] = 0.0f;
          if (db->nodes[S.node[i]].kind == VN_NOTE && S.vis[i] > 0) {
              sc[i] = 0.5f * S.rw[i] + 0.5f * (f32)(S.val[i] / S.vis[i]);
              ord[n++] = i;
          }
      }
      for (i = 1; i < n; ++i) {                                /* ord holds slots */
          i32 x = ord[i], j;
          for (j = i; j > 0 && sc[ord[j - 1]] < sc[x]; --j) ord[j] = ord[j - 1];
          ord[j] = x;
      }
      for (it = 0; it < n && it < k; ++it) {
          i32 sl = ord[it];
          out[it].node = S.node[sl]; out[it].sim = vdb_vec_(db, S.node[sl]) ? vdb_dot_(q, vdb_vec_(db, S.node[sl]), db->dim) : 0.0f;
          out[it].lex = S.rw[sl]; out[it].score = sc[sl]; out[it].via = S.via[sl]; out[it].depth = S.dep[sl];
      }
      if (n > k) n = k;
      if (nseed > 0 && seeds[0].score >= 5.0f && n > 0) {         /* the query is a title: that note stays first */
          VHit ex = seeds[0]; i32 j;
          for (j = 0; j < n && out[j].node != ex.node; ++j) {}
          if (j == n && n < k) ++n;
          if (j == n) j = n - 1;
          for (; j > 0; --j) out[j] = out[j - 1];
          ex.score = out[1 < n ? 1 : 0].score + 1.0f;
          out[0] = ex;
      }
      free(sc); }
    free(ord); free(t); free(seeds); free(q); h64_free(&S.map);
    free(S.node); free(S.vis); free(S.via); free(S.dep); free(S.rw); free(S.val);
    return n;
}

/* ---- one query string that mixes words with graph and tag conditions ----
 *   quantum entanglement                 words: hybrid search (vector + BM25)
 *   #physics  -#draft  tag:physics/x     only notes with / without that tag (nested: #a also matches #a/b)
 *   [[Bell tests]]                       notes linked to or from that note
 *   to:[[Bell tests]]                    notes that link to it (its backlinks)
 *   from:[[Bell tests]]                  notes it links to
 * Conditions combine with AND. Words are optional: "#physics to:[[X]]" lists matching notes, most linked first.
 * mcts != 0 runs the MCTS explorer for the words instead of plain hybrid search. */
typedef struct { i32 node, mode, neg; } VCond;   /* mode: 0 neighbour, 1 to, 2 from, 3 tag */

static i32 vdb_cond_ok_(const VecDB *db, i32 c, const VCond *f)
{
    const VNode *v = &db->nodes[c];
    i32 has = 0;
    /* mode 3 (#tag) and mode 1 (to:[[X]]) really do share this exact check, not a copy-paste
     * slip: a tag is stored as an ordinary out-edge to a VN_TAG-kind node, same list a wikilink
     * out-edge to a VN_NOTE-kind node lives in - what distinguishes them is f->node's kind at
     * the call site (vdb_query_'s parser resolves "#tag" via vecdb_find_tag, "to:[[X]]" via
     * vecdb_find), not a different edge list here. Kept as two arms (not merged with ||) so that
     * stays visible instead of looking like one of the two modes was dropped by mistake. */
    if (f->mode == 3) has = vivec_has_(&v->out, f->node);
    else if (f->mode == 1) has = vivec_has_(&v->out, f->node);
    else if (f->mode == 2) has = vivec_has_(&db->nodes[f->node].out, c);
    else has = vivec_has_(&v->out, f->node) || vivec_has_(&db->nodes[f->node].out, c);
    return f->neg ? !has : has;
}

static i32 vdbl_query(VecDB *db, const char *query, i32 k, i32 mcts, VHit *out)
{
    i32 ql = (i32)strlen(query), i = 0, nc = 0, nw = 0, n = 0, j, seedtag = -1, gi = -1;
    VCond *cond = (VCond *)malloc(((size_t)ql / 2 + 2) * sizeof(VCond));
    char *words = (char *)malloc((size_t)ql + 2), *key = (char *)malloc((size_t)ql + 1);   /* a word may end in the separator we add */
    VHit *hits;
    i32 want, ok = 1;

    while (i < ql) {
        char c = query[i];
        i32 neg = 0, mode = -1, st;
        if (c == ' ' || c == '\t') { ++i; continue; }
        st = i;
        if (c == '-' && i + 1 < ql && (query[i + 1] == '#' || query[i + 1] == '[')) { neg = 1; ++i; c = query[i]; }
        if (!strncmp(query + i, "to:", 3) && query[i + 3] == '[') { mode = 1; i += 3; c = query[i]; }
        else if (!strncmp(query + i, "from:", 5) && query[i + 5] == '[') { mode = 2; i += 5; c = query[i]; }
        else if (!strncmp(query + i, "tag:", 4)) { mode = 3; i += 4; c = query[i]; }
        if (c == '[' && i + 1 < ql && query[i + 1] == '[') {
            const char *e = strstr(query + i + 2, "]]");
            if (e) {
                i32 tl = (i32)(e - (query + i + 2)), m = vdb_norm_(query + i + 2, tl, key, 1), id = m > 0 ? hs_get(&db->names, key, m) : -1;
                if (id < 0) ok = 0;                                  /* an unknown note matches nothing */
                else { cond[nc].node = id; cond[nc].mode = mode < 0 ? 0 : mode; cond[nc].neg = neg; ++nc; }
                i = (i32)(e - query) + 2; continue;
            }
        }
        if (c == '#' || mode == 3) {
            i32 t0 = i + (c == '#' ? 1 : 0), t1 = t0, m, id;
            while (t1 < ql && vdb_tagchar_((u8)query[t1])) ++t1;
            m = vdb_norm_(query + t0, t1 - t0, key, 0);
            id = m > 0 ? hs_get(&db->tags, key, m) : -1;
            if (m > 0) { if (id < 0 && !neg) ok = 0; else if (id >= 0) { cond[nc].node = id; cond[nc].mode = 3; cond[nc].neg = neg; ++nc; } }
            i = t1 > t0 ? t1 : i + 1; continue;
        }
        i = st;                                                        /* an ordinary word */
        while (i < ql && query[i] != ' ' && query[i] != '\t') words[nw++] = query[i++];
        words[nw++] = ' ';
    }
    words[nw] = 0;
    free(key);
    if (!ok) { free(cond); free(words); return 0; }
    for (j = 0; j < nc; ++j) if (!cond[j].neg) { if (cond[j].mode == 3 && seedtag < 0) seedtag = cond[j].node; if (cond[j].mode != 3 && gi < 0) gi = j; }

    want = nc ? k * 12 + 40 : k;
    if (want > 250) want = 250;
    hits = (VHit *)malloc((size_t)(want + 1) * sizeof(VHit));
    if (nw > 0) {
        i32 got = mcts ? vdbl_explore(db, words, want, (const VExplore *)0, hits) : vdb_search_(db, words, want, 0, seedtag, hits, 0, (i64 *)0, (f32 *)0, (i32 *)0, (f32 *)0);
        for (j = 0; j < got && n < k; ++j) {
            i32 g, pass = 1;
            for (g = 0; g < nc && pass; ++g) pass = vdb_cond_ok_(db, hits[j].node, &cond[g]);
            if (pass && db->nodes[hits[j].node].kind == VN_NOTE) out[n++] = hits[j];
        }
    }
    if (gi >= 0 && n < k) {                                            /* the graph condition names the candidates: top up from them */
        const VCond *g0 = &cond[gi];
        const VNode *x = &db->nodes[g0->node];
        i32 total = (g0->mode == 1) ? x->in.n : (g0->mode == 2 ? x->out.n : x->in.n + x->out.n), t;
        f32 *qv = (nw > 0) ? (f32 *)malloc((size_t)db->dim * sizeof(f32)) : (f32 *)0;
        if (qv) vdb_embed_(db, words, nw, 1, qv);
        for (t = 0; t < total; ++t) {
            i32 c = g0->mode == 1 ? x->in.v[t] : (g0->mode == 2 ? x->out.v[t] : vdb_nb_(db, g0->node, t)), g, pass = 1, dup = 0;
            const f32 *cv;
            VHit h;
            if (db->nodes[c].kind != VN_NOTE) continue;
            for (g = 0; g < nc && pass; ++g) pass = vdb_cond_ok_(db, c, &cond[g]);
            if (!pass) continue;
            for (g = 0; g < n; ++g) if (out[g].node == c) { dup = 1; break; }
            if (dup) continue;
            cv = vdb_vec_(db, c);
            h.node = c; h.via = -1; h.depth = 0; h.lex = 0.0f;
            h.sim = (qv && cv) ? vdb_dot_(qv, cv, db->dim) : 0.0f;
            h.score = nw > 0 ? h.sim : (f32)db->nodes[c].in.n;        /* no words: the most linked first */
            if (n < k) out[n++] = h;                                    /* keep the k best of the candidates */
            else { i32 w = 0; for (g = 1; g < n; ++g) if (out[g].score < out[w].score) w = g; if (h.score > out[w].score) out[w] = h; }
        }
        free(qv);
        for (j = 1; j < n; ++j) { VHit x2 = out[j]; i32 z; for (z = j; z > 0 && out[z - 1].score < x2.score; --z) out[z] = out[z - 1]; out[z] = x2; }
    }
    if (nw == 0 && gi < 0 && nc > 0 && seedtag >= 0) {                 /* tags only: the notes carrying them, most linked first */
        const VNode *tn = &db->nodes[seedtag];
        for (j = 0; j < tn->in.n; ++j) {
            i32 c = tn->in.v[j], g, pass = 1;
            VHit h;
            if (db->nodes[c].kind != VN_NOTE) continue;
            for (g = 0; g < nc && pass; ++g) pass = vdb_cond_ok_(db, c, &cond[g]);
            if (!pass) continue;
            h.node = c; h.sim = 0.0f; h.lex = 0.0f; h.score = (f32)db->nodes[c].in.n; h.via = -1; h.depth = 0;
            if (n < k) out[n++] = h;
            else { i32 w = 0, z; for (z = 1; z < n; ++z) if (out[z].score < out[w].score) w = z; if (h.score > out[w].score) out[w] = h; }
        }
        for (j = 1; j < n; ++j) { VHit x2 = out[j]; i32 z; for (z = j; z > 0 && out[z - 1].score < x2.score; --z) out[z] = out[z - 1]; out[z] = x2; }
    }
    free(hits); free(cond); free(words);
    return n;
}
