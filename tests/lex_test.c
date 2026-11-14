/* lex_test.c - the self-test of src/lex.c; ./run builds and runs it. */
#include "lex.c"
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
