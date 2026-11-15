/* probe.c - the native build reading a store the WebAssembly build wrote: prints "COUNT TEXT-OF-ROW-20" (test.mjs) */
#include <stdio.h>
#include "vecindex.h"
int main(int argc, char **argv)
{
    char t[64];
    int n;
    vi_db *db = argc > 1 ? vi_open(argv[1], 0, 0) : (vi_db *)0;
    if (!db) { fprintf(stderr, "probe: %s\n", vi_error()); return 1; }
    n = vi_get_text(db, 20, t, (int)sizeof(t) - 1);
    t[n > 0 ? n : 0] = 0;
    printf("%ld %s\n", (long)vi_count(db), t);
    vi_close(db);
    return 0;
}
