/* vfind_demo.c - a small store with texts for tests/vfind_test.sh: 3 documents of 3-4 chunks, ids doc << 20 | chunk.
 *   vfind_demo PATH.vxi */
#include "store.c"

int main(int argc, char **argv)
{
    static Arena arena;
    static const char *const doc[3][4] = {
        { "Tea came to Europe in the 1600s, by Dutch ships.", "Green and black tea come from one plant, Camellia sinensis.",
          "Oolong is partly oxidised, between green and black.", (const char *)0 },
        { "The cat sleeps sixteen hours a day.", "A cat purrs at 25 to 150 hertz.", "Cats were kept in Cyprus 9500 years ago.",
          "Most cats do not taste sweetness." },
        { "Zebras have black skin under white fur.", "The zebrafish genome was sequenced in 2013;\nit has 26,000 genes.",
          "Zebrafish regrow heart muscle after an injury.", (const char *)0 } };
    RagStore *s;
    f32 v[8];
    u32 st = 7u;
    i32 d, c, i;
    if (argc != 2) { fprintf(stderr, "usage: vfind_demo PATH.vxi\n"); return 2; }
    s = store_new(&arena, 8);
    for (d = 0; d < 3; ++d)
        for (c = 0; c < 4 && doc[d][c]; ++c) {
            for (i = 0; i < 8; ++i) { st ^= st << 13; st ^= st >> 17; st ^= st << 5; v[i] = (f32)(st % 2001u) / 1000.0f - 1.0f; }
            if (store_put_text(s, v, ((i64)d << 20) | c, 0, doc[d][c], (i32)strlen(doc[d][c])) != 0) return 1;
        }
    if (store_save(s, argv[1]) != 0) return 1;
    store_destroy(s);
    return 0;
}
