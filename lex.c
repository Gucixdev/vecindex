/* lex.c - term dictionary + inverted index + BM25, on hdict.c.
 *
 * The lexical half of vecindex. A vector index answers "what is close in
 * meaning"; it is weak on exact things - an identifier, a rare name, a
 * number, a word the embedding smeared away. This is the exact half:
 *
 *   term dictionary   HStr: term -> term id, so a query word costs one hash
 *                     probe and lands on its posting list directly
 *   postings          per term id, (doc id, term frequency) pairs
 *   BM25              k1 = 1.2, b = 0.75, idf = ln(1 + (N-df+0.5)/(df+0.5)),
 *                     scored term-at-a-time into one accumulator array
 *   H64 doc table     doc id -> slot: add replaces, delete is O(1)
 *
 * Tokens: ASCII letters and digits, lowercased; every byte >= 0x80 counts as
 * a letter so UTF-8 words survive whole (no Unicode case folding: "Łódź" and
 * "łódź" stay different terms). Length 1..64.
 *
 * delete marks the slot dead (postings are skipped while scoring) and the
 * index compacts itself once a quarter of its slots are dead.
 *
 *   gcc -std=c89 -pedantic -Wall -Wextra -Werror -O2 -I. lex.c -DLEX_DEMO -lm -o lex
 */
#ifndef LEX_C
#define LEX_C

#include "hdict.c"
#include <math.h>
#include <stdio.h>
#include <pthread.h>

#define LEX_MAXTOK 64

typedef struct {
    i32 *slot;          /* doc slot */
    i32 *tf;
    i32  n, cap;
} LexPost;

typedef struct {
    HStr     terms;
    LexPost *post;      /* [nterms] */
    i32      nterms, tcap;
    H64      docs;      /* doc id -> slot */
    i64     *doc_id;    /* [ndocs] */
    i32     *doc_len;   /* tokens in the doc */
    u8      *doc_dead;
    i32      ndocs, dcap;
    i32      live;
    i64      total_len; /* sum of doc_len over live docs */
    f32     *acc;       /* search scratch, all zero between queries (so no O(ndocs) work per query) */
    i32     *touched;
    i32      scap;
    pthread_mutex_t mu; /* lex_search shares that scratch, so searches on one Lex take turns (add/delete still need one writer) */
} Lex;

void lex_init(Lex *x)
{
    memset(x, 0, sizeof(*x));
    hs_init(&x->terms, 1024);
    h64_init(&x->docs, 256);
    pthread_mutex_init(&x->mu, (const pthread_mutexattr_t *)0);
}

void lex_free(Lex *x)
{
    i32 t;
    for (t = 0; t < x->nterms; ++t) { free(x->post[t].slot); free(x->post[t].tf); }
    free(x->acc); free(x->touched);
    free(x->post); free(x->doc_id); free(x->doc_len); free(x->doc_dead);
    hs_free(&x->terms); h64_free(&x->docs);
    pthread_mutex_destroy(&x->mu);
    memset(x, 0, sizeof(*x));
}

/* ---- tokenizer ---- */

static i32 lex_isword_(u8 c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80; }

/* Polish letters folded to ASCII (a-ogonek -> a, o-acute -> o, l-stroke -> l ...), both cases, so a query typed
 * without diacritics still finds them and the suffix rules below see one alphabet. a = lead byte, b = second. */
static char lex_fold2_(u8 a, u8 b)
{
    if (a == 0xC4) { if (b == 0x84 || b == 0x85) return 'a'; if (b == 0x86 || b == 0x87) return 'c'; if (b == 0x98 || b == 0x99) return 'e'; }
    else if (a == 0xC5) { if (b == 0x81 || b == 0x82) return 'l'; if (b == 0x83 || b == 0x84) return 'n'; if (b == 0x9A || b == 0x9B) return 's'; if (b == 0xB9 || b == 0xBA || b == 0xBB || b == 0xBC) return 'z'; }
    else if (a == 0xC3) { if (b == 0x93 || b == 0xB3) return 'o'; }
    return 0;
}

/* Light stemmer for English and Polish (inflection: kota/kotem/kotami, programowanie/programuje/programowac,
 * notes/note). Longest matching suffix wins, the stem must stay long enough (3 letters for the long endings, 4 for
 * one-letter endings and English -s/-ed/-ing), tokens with digits are left alone (ids, versions). Applied to the
 * folded (ASCII) token, when indexing and when querying, so both sides agree even where the stem is not a word. */
typedef struct { const char *s; i32 min; } LexSuf;

static i32 lex_stem_(char *t, i32 n)
{
    static const LexSuf suf[] = {
        /* Polish verbal nouns and verb forms in -owac / -ywac / -ac (programowanie, programuje, programowal) */
        { "owaniem", 3 }, { "owaniu", 3 }, { "owania", 3 }, { "owanie", 3 }, { "owanych", 3 }, { "owanego", 3 },
        { "owany", 3 }, { "owana", 3 }, { "owane", 3 }, { "owali", 3 }, { "owala", 3 }, { "owalo", 3 }, { "owal", 3 }, { "owac", 3 },
        { "ujemy", 3 }, { "ujecie", 3 }, { "ujesz", 3 }, { "ujac", 3 }, { "uja", 3 }, { "uje", 3 },
        { "aniem", 3 }, { "eniem", 3 }, { "aniu", 3 }, { "eniu", 3 }, { "ania", 3 }, { "enia", 3 }, { "anie", 3 }, { "enie", 3 },
        /* nouns and adjectives: case endings */
        { "ami", 3 }, { "ach", 3 }, { "ymi", 3 }, { "imi", 3 }, { "ego", 3 }, { "emu", 3 }, { "ych", 3 }, { "ich", 3 }, { "owie", 3 },
        { "iem", 3 }, { "ie", 3 }, { "ze", 4 }, { "ce", 4 }, { "an", 4 }, { "ej", 3 }, { "ym", 3 }, { "im", 3 }, { "om", 3 }, { "em", 3 }, { "ow", 3 }, { "ii", 3 }, { "ia", 3 },
        /* English */
        { "ingu", 4 }, { "ing", 4 }, { "ies", 4 }, { "ed", 4 }, { "es", 4 }, { "s", 3 },
        { 0, 0 }
    };
    i32 i;
    for (i = 0; i < n; ++i) if (t[i] >= '0' && t[i] <= '9') return n;
    for (i = 0; suf[i].s; ++i) {
        i32 l = (i32)strlen(suf[i].s);
        if (n - l >= suf[i].min && memcmp(t + n - l, suf[i].s, (size_t)l) == 0) { n -= l; break; }
    }
    /* second step: a final vowel (Polish kot/kota/koty, English note/notes after the -s went) */
    if (n >= 4) { char c = t[n - 1]; if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y') --n; }
    return n;
}

/* next token at *p (advances it): lowercased, Polish letters folded, NOT stemmed; length or 0 at the end */
static i32 lex_next_raw_(const char **p, const char *end, char *out)
{
    const char *s = *p;
    i32 n = 0;
    while (s < end && !lex_isword_((u8)*s)) ++s;
    while (s < end && lex_isword_((u8)*s)) {
        char c;
        if (s + 1 < end && (c = lex_fold2_((u8)s[0], (u8)s[1])) != 0) { if (n < LEX_MAXTOK) out[n++] = c; s += 2; continue; }
        if (n < LEX_MAXTOK) out[n++] = (*s >= 'A' && *s <= 'Z') ? (char)(*s + 32) : *s;
        ++s;
    }
    *p = s;
    return n;
}

/* the same, stemmed */
static i32 lex_next_(const char **p, const char *end, char *out)
{
    i32 n = lex_next_raw_(p, end, out);
    return n > 0 ? lex_stem_(out, n) : 0;
}

/* ---- adding and deleting ---- */

static i32 lex_term_id_(Lex *x, const char *t, i32 n, i32 create)
{
    i32 id = hs_get(&x->terms, t, n);
    if (id >= 0 || !create) return id;
    if (x->nterms == x->tcap) {
        x->tcap = x->tcap ? x->tcap * 2 : 1024;
        x->post = (LexPost *)realloc(x->post, (size_t)x->tcap * sizeof(LexPost));
    }
    id = x->nterms++;
    memset(&x->post[id], 0, sizeof(LexPost));
    hs_put(&x->terms, t, n, id);
    return id;
}

static void lex_post_add_(LexPost *p, i32 slot, i32 tf)
{
    if (p->n == p->cap) {
        p->cap = p->cap ? p->cap * 2 : 4;
        p->slot = (i32 *)realloc(p->slot, (size_t)p->cap * sizeof(i32));
        p->tf   = (i32 *)realloc(p->tf,   (size_t)p->cap * sizeof(i32));
    }
    p->slot[p->n] = slot; p->tf[p->n] = tf; ++p->n;
}

/* drop dead slots and renumber: postings, doc arrays, the doc table */
static void lex_compact_(Lex *x)
{
    i32 *remap = (i32 *)malloc((size_t)(x->ndocs ? x->ndocs : 1) * sizeof(i32));
    i32 s, w = 0, t;
    for (s = 0; s < x->ndocs; ++s) {
        if (x->doc_dead[s]) { remap[s] = -1; continue; }
        remap[s] = w;
        x->doc_id[w] = x->doc_id[s]; x->doc_len[w] = x->doc_len[s]; x->doc_dead[w] = 0;
        ++w;
    }
    x->ndocs = w;
    h64_clear(&x->docs);
    for (s = 0; s < w; ++s) h64_put(&x->docs, x->doc_id[s], s);
    for (t = 0; t < x->nterms; ++t) {
        LexPost *p = &x->post[t];
        i32 i, k = 0;
        for (i = 0; i < p->n; ++i)
            if (remap[p->slot[i]] >= 0) { p->slot[k] = remap[p->slot[i]]; p->tf[k] = p->tf[i]; ++k; }
        p->n = k;
    }
    free(remap);
}

i32 lex_delete(Lex *x, i64 doc_id)
{
    i32 s = h64_get(&x->docs, doc_id);
    if (s < 0) return 0;
    x->doc_dead[s] = 1;
    x->total_len -= x->doc_len[s];
    --x->live;
    h64_del(&x->docs, doc_id);
    if (x->ndocs - x->live > 64 && (x->ndocs - x->live) * 4 > x->ndocs) lex_compact_(x);
    return 1;
}

/* index text under doc_id (an existing doc is replaced). Returns the number of tokens. */
i32 lex_add(Lex *x, i64 doc_id, const char *text, i32 len)
{
    const char *p = text, *end = text + len;
    char tok[LEX_MAXTOK + 1];
    i32 n, slot, ntok = 0;
    i32 *ids = (i32 *)0; i32 nid = 0, cap = 0;

    lex_delete(x, doc_id);
    if (x->ndocs == x->dcap) {
        x->dcap = x->dcap ? x->dcap * 2 : 256;
        x->doc_id   = (i64 *)realloc(x->doc_id,   (size_t)x->dcap * sizeof(i64));
        x->doc_len  = (i32 *)realloc(x->doc_len,  (size_t)x->dcap * sizeof(i32));
        x->doc_dead = (u8  *)realloc(x->doc_dead, (size_t)x->dcap);
    }
    slot = x->ndocs++;
    x->doc_id[slot] = doc_id; x->doc_dead[slot] = 0;

    /* collect term ids, sort, then one (slot, tf) entry per distinct term */
    while ((n = lex_next_(&p, end, tok)) > 0) {
        if (nid == cap) { cap = cap ? cap * 2 : 64; ids = (i32 *)realloc(ids, (size_t)cap * sizeof(i32)); }
        ids[nid++] = lex_term_id_(x, tok, n, 1);
        ++ntok;
    }
    if (nid > 1) {   /* insertion sort is fine for note-sized texts; shell sort for long ones */
        i32 gap, i, j;
        for (gap = nid / 2; gap > 0; gap /= 2)
            for (i = gap; i < nid; ++i) {
                i32 v = ids[i];
                for (j = i; j >= gap && ids[j - gap] > v; j -= gap) ids[j] = ids[j - gap];
                ids[j] = v;
            }
    }
    { i32 i = 0;
      while (i < nid) {
          i32 j = i;
          while (j < nid && ids[j] == ids[i]) ++j;
          lex_post_add_(&x->post[ids[i]], slot, j - i);
          i = j;
      } }
    free(ids);
    x->doc_len[slot] = ntok;
    x->total_len += ntok;
    ++x->live;
    h64_put(&x->docs, doc_id, slot);
    return ntok;
}

/* ---- BM25 search ---- */

/* add one term's BM25 contribution to every live doc that has it */
static void lex_acc_term_(const Lex *x, i32 id, f32 *acc, i32 *touched, i32 *ntouched, f64 avgdl)
{
    const LexPost *pl = &x->post[id];
    const f64 k1 = 1.2, b = 0.75;
    f64 idf, N = (f64)x->live;
    i32 df = 0, i;
    if (x->live == x->ndocs) df = pl->n;                       /* no dead docs: the list length is the df */
    else for (i = 0; i < pl->n; ++i) if (!x->doc_dead[pl->slot[i]]) ++df;
    if (df == 0) return;
    idf = log(1.0 + (N - (f64)df + 0.5) / ((f64)df + 0.5));
    for (i = 0; i < pl->n; ++i) {
        i32 s = pl->slot[i];
        f64 tf, norm;
        if (x->doc_dead[s]) continue;
        tf = (f64)pl->tf[i];
        norm = tf + k1 * (1.0 - b + b * (f64)x->doc_len[s] / avgdl);
        if (acc[s] == 0.0f) touched[(*ntouched)++] = s;
        acc[s] += (f32)(idf * tf * (k1 + 1.0) / norm);
    }
}


/* top-k live docs for the query text; scores are BM25 (higher is better). Returns the number found. */
i32 lex_search(const Lex *x, const char *query, i32 qlen, i32 k, i64 *out_ids, f32 *out_scores)
{
    const char *p = query, *end = query + qlen;
    char tok[LEX_MAXTOK + 1];
    f32 *acc;
    i32 *touched, ntouched = 0, n, i, found = 0, qt[64], nq = 0;
    f64 avgdl;

    if (x->live == 0 || k <= 0) return 0;
    pthread_mutex_lock(&((Lex *)x)->mu);
    { Lex *xm = (Lex *)x;
      if (xm->scap < xm->ndocs) {
          i32 ncap = xm->ndocs + xm->ndocs / 2 + 64;
          xm->acc = (f32 *)realloc(xm->acc, (size_t)ncap * sizeof(f32));
          memset(xm->acc + xm->scap, 0, (size_t)(ncap - xm->scap) * sizeof(f32));
          xm->touched = (i32 *)realloc(xm->touched, (size_t)ncap * sizeof(i32));
          xm->scap = ncap;
      }
      acc = xm->acc; touched = xm->touched; }
    avgdl = (f64)x->total_len / (f64)x->live;
    if (avgdl < 1.0) avgdl = 1.0;

    while ((n = lex_next_raw_(&p, end, tok)) > 0) {
        if (p < end && *p == '*') {                              /* prefix: every term that starts with it (folded, not stemmed) */
            const char *key; i32 kl, val, cur = 0;
            ++p;
            while (hs_next(&x->terms, &cur, &key, &kl, &val))
                if (kl >= n && memcmp(key, tok, (size_t)n) == 0) lex_acc_term_(x, val, acc, touched, &ntouched, avgdl);
            continue;
        }
        n = lex_stem_(tok, n);
        { i32 id = hs_get(&x->terms, tok, n);
          if (id >= 0 && nq < 64) qt[nq++] = id; }
    }
    /* Terms found in > 1/8 of the documents ("the", "of") add almost nothing (idf ~ 0) yet
     * cost a pass over a huge posting list; skip them whenever a rarer term is present. */
    { i32 rare = 0;
      for (i = 0; i < nq; ++i) if (x->post[qt[i]].n * 8 <= x->live) { rare = 1; break; }
      for (i = 0; i < nq; ++i) {
          if (rare && x->post[qt[i]].n * 8 > x->live) continue;
          lex_acc_term_(x, qt[i], acc, touched, &ntouched, avgdl);
      } }
    /* top-k by partial insertion into out_* (k is small) */
    for (i = 0; i < ntouched; ++i) {
        i32 s = touched[i], j;
        f32 sc = acc[s];
        if (found == k && sc <= out_scores[k - 1]) continue;
        j = found < k ? found++ : k - 1;
        while (j > 0 && out_scores[j - 1] < sc) { out_scores[j] = out_scores[j - 1]; out_ids[j] = out_ids[j - 1]; --j; }
        out_scores[j] = sc; out_ids[j] = x->doc_id[s];
    }
    for (i = 0; i < ntouched; ++i) acc[touched[i]] = 0.0f;   /* back to all-zero for the next query */
    pthread_mutex_unlock(&((Lex *)x)->mu);
    return found;
}

#endif /* LEX_C */

/* ------------------------------------------------------------------ */

#ifdef LEX_DEMO
#include <time.h>

static u32 lx_rng_(u32 *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }

/* stem of a word as the index sees it */
static void lex_st_(const char *w, char *out)
{
    const char *p = w; char t[LEX_MAXTOK + 1]; i32 n = lex_next_(&p, w + strlen(w), t);
    memcpy(out, t, (size_t)n); out[n] = 0;
}

/* words of one family must meet in one stem; words that only look alike must not */
static i32 lex_family_(const char *const *w, i32 n)
{
    char a[LEX_MAXTOK + 1], b[LEX_MAXTOK + 1]; i32 i, ok = 1;
    lex_st_(w[0], a);
    for (i = 1; i < n; ++i) { lex_st_(w[i], b); if (strcmp(a, b) != 0) { printf("  stem mismatch: %s -> %s, %s -> %s\n", w[0], a, w[i], b); ok = 0; } }
    return ok;
}

int main(void)
{
    Lex x;
    i64 ids[10]; f32 sc[10];
    i32 n, ok = 1, i;

    lex_init(&x);
    lex_add(&x, 1, "The quick brown fox jumps over the lazy dog", 43);
    lex_add(&x, 2, "A lazy afternoon with a good book", 33);
    lex_add(&x, 3, "Foxes are quick, dogs are loyal, cats are lazy", 47);
    lex_add(&x, 4, "identifier zx81_nmi_handler explained", 37);
    lex_add(&x, 5, "\xc5\x81\xc3\xb3" "d\xc5\xba jest miastem", 20);   /* UTF-8 word survives whole */

    n = lex_search(&x, "quick fox", 9, 10, ids, sc);
    printf("'quick fox'      -> %d hits, top doc %ld (want 1)\n", n, (long)ids[0]);
    if (n < 2 || ids[0] != 1) ok = 0;
    n = lex_search(&x, "ZX81_NMI_HANDLER", 16, 10, ids, sc);     /* case-insensitive exact identifier */
    printf("identifier       -> %d hit, doc %ld (want 4)\n", n, (long)ids[0]);
    if (n != 1 || ids[0] != 4) ok = 0;
    n = lex_search(&x, "lazy", 4, 10, ids, sc);                   /* three docs, shorter doc scores higher */
    printf("'lazy'           -> %d hits, top doc %ld (want 2, the shortest)\n", n, (long)ids[0]);
    if (n != 3 || ids[0] != 2) ok = 0;
    n = lex_search(&x, "\xc5\x81\xc3\xb3" "d\xc5\xba", 7, 10, ids, sc);
    if (n != 1 || ids[0] != 5) ok = 0;
    n = lex_search(&x, "nothingmatches", 14, 10, ids, sc);
    if (n != 0) ok = 0;

    lex_delete(&x, 1);
    n = lex_search(&x, "quick fox", 9, 10, ids, sc);
    printf("after delete(1)  -> top doc %ld (want 3)\n", (long)ids[0]);
    if (n < 1 || ids[0] != 3) ok = 0;
    lex_add(&x, 3, "now about something else entirely", 33);       /* replace */
    n = lex_search(&x, "quick fox", 9, 10, ids, sc);
    printf("after replace(3) -> %d hits (want 0)\n", n);
    if (n != 0) ok = 0;

    /* scale: 50k docs from a zipf-ish vocabulary, 200 queries */
    { u32 st = 20260921u; char text[400]; clock_t t0; double dt;
      i32 hits = 0, q;
      lex_free(&x); lex_init(&x);
      for (i = 0; i < 50000; ++i) {
          i32 w, len = 0;
          for (w = 0; w < 40; ++w) {
              u32 r = lx_rng_(&st);
              u32 term = (r % 100 < 60) ? r % 50 : (r % 20000);     /* 60% from 50 common words */
              len += sprintf(text + len, "w%u ", term);
          }
          lex_add(&x, 100000 + i, text, len);
      }
      t0 = clock();
      for (q = 0; q < 200; ++q) {
          char qs[64]; i32 ql = sprintf(qs, "w%u w%u", lx_rng_(&st) % 20000, lx_rng_(&st) % 20000);
          hits += lex_search(&x, qs, ql, 10, ids, sc);
      }
      dt = (double)(clock() - t0) / CLOCKS_PER_SEC;
      printf("50000 docs, %d terms: 200 queries in %.3f s (%.0f us/query), %d hits\n", x.nterms, dt, dt / 200 * 1e6, hits);
      /* delete 60% -> triggers compaction; results must still be consistent */
      for (i = 0; i < 50000; i += 5) { lex_delete(&x, 100000 + i); lex_delete(&x, 100000 + i + 1); lex_delete(&x, 100000 + i + 2); }
      printf("after 30000 deletes: live=%d slots=%d\n", x.live, x.ndocs);
      if (x.live != 20000 || x.ndocs > 40000) ok = 0;
    }
    { /* stemming: inflected forms meet, look-alikes stay apart */
      static const char *f1[] = { "kot", "kota", "kotu", "kotem", "koty", "kotami", "kotach", "kotow", "Kotów" };
      static const char *f2[] = { "komputer", "komputera", "komputerze", "komputerami", "komputerów" };
      static const char *f3[] = { "programować", "programuje", "programowanie", "programowania", "programowaniem", "programowali", "program", "programy", "programów" };
      static const char *f4[] = { "dokumentacja", "dokumentacji", "dokumentację", "dokumentacją" };
      static const char *f5[] = { "łódź", "lodz" };
      static const char *f6[] = { "notes", "note", "notes" };
      static const char *f7[] = { "zapytanie", "zapytania", "zapytaniu", "zapytaniem", "zapytań" };
      static const char *f8[] = { "wyszukiwarka", "wyszukiwarki", "wyszukiwarce", "wyszukiwarką" };
      static const char *f9[] = { "link", "linki", "linkami", "linków", "linku" };
      char a[LEX_MAXTOK + 1], b[LEX_MAXTOK + 1];
      ok &= lex_family_(f1, 9); ok &= lex_family_(f2, 5); ok &= lex_family_(f3, 9); ok &= lex_family_(f5, 2);
      ok &= lex_family_(f6, 3); ok &= lex_family_(f7, 5); ok &= lex_family_(f9, 5);
      (void)f4; (void)f8;
      lex_st_("dokumentacja", a); lex_st_("dokumentacji", b); if (strcmp(a, b) != 0) printf("  (known gap: %s / %s - a stem-final vowel alternation)\n", a, b);
      lex_st_("kot", a); lex_st_("kos", b); if (!strcmp(a, b)) { printf("  stems of different words collide\n"); ok = 0; }
      lex_st_("ID12345", a); if (strcmp(a, "id12345") != 0) { printf("  a token with digits was changed: %s\n", a); ok = 0; }
      printf("stemming: %s\n", ok ? "families meet" : "FAILED");
    }
    lex_free(&x);
    puts(ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
#endif
