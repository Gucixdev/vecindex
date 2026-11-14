/* hdict_test.c - the self-test of src/hdict.c; ./run builds and runs it. */
#include "hdict.c"
#include <stdio.h>
#include <time.h>

int main(void)
{
    enum { N = 200000 };
    H64 h; HStr d;
    i32 i, ok = 1, seen = 0, cur = 0, vlen, vv;
    const char *kp;
    char key[32];
    clock_t t0;
    double t_h, t_scan;
    i64 *scanids;

    /* H64: insert, get, update, delete, tombstone reuse, growth */
    h64_init(&h, 4);
    for (i = 0; i < N; ++i) if (h64_put(&h, (i64)i * 7919 + 13, i) != -1) ok = 0;
    for (i = 0; i < N; ++i) if (h64_get(&h, (i64)i * 7919 + 13) != i) ok = 0;
    if (h.n != N) ok = 0;
    if (h64_put(&h, 13, 42) != 0 || h64_get(&h, 13) != 42) ok = 0;         /* update returns old */
    for (i = 1; i < N; i += 2) if (h64_del(&h, (i64)i * 7919 + 13) != i) ok = 0;
    for (i = 1; i < N; i += 2) if (h64_get(&h, (i64)i * 7919 + 13) != -1) ok = 0;
    for (i = 0; i < N; i += 2) if (h64_get(&h, (i64)i * 7919 + 13) != (i == 0 ? 42 : i)) ok = 0;
    if (h.n != N / 2) ok = 0;
    for (i = 1; i < N; i += 2) h64_put(&h, (i64)i * 7919 + 13, -i - 1 < 0 ? i : 0);   /* reinsert over tombstones */
    if (h.n != N) ok = 0;
    if (h64_get(&h, -5) != -1 || h64_del(&h, -5) != -1) ok = 0;            /* absent */
    printf("H64   n=%d cap=%d used=%d  %s\n", h.n, h.cap, h.used, ok ? "ok" : "FAIL");

    /* HStr: interning, update, delete, iteration, empty and long keys */
    hs_init(&d, 4);
    for (i = 0; i < N; ++i) { sprintf(key, "term-%d", i); if (hs_put(&d, key, (i32)strlen(key), i) != -1) ok = 0; }
    for (i = 0; i < N; ++i) { sprintf(key, "term-%d", i); if (hs_get(&d, key, (i32)strlen(key)) != i) ok = 0; }
    if (hs_get(&d, "term-1", 6) != 1 || hs_get(&d, "term-1x", 7) != -1 || hs_get(&d, "", 0) != -1) ok = 0;
    if (hs_put(&d, "", 0, 7) != -1 || hs_get(&d, "", 0) != 7) ok = 0;
    { char big[600]; memset(big, 'k', sizeof big);                       /* a key past the 1-byte length */
      if (hs_put(&d, big, 600, 600) != -1 || hs_get(&d, big, 600) != 600 || hs_get(&d, big, 599) != -1) ok = 0; }
    if (hs_put(&d, "term-5", 6, 555) != 5 || hs_get(&d, "term-5", 6) != 555) ok = 0;
    for (i = 0; i < N; i += 3) { sprintf(key, "term-%d", i); if (hs_del(&d, key, (i32)strlen(key)) < 0) ok = 0; }
    for (i = 0; i < N; ++i) { sprintf(key, "term-%d", i); if ((hs_get(&d, key, (i32)strlen(key)) == -1) != (i % 3 == 0)) ok = 0; }
    while (hs_next(&d, &cur, &kp, &vlen, &vv)) { ++seen; if ((i32)strlen(kp) != vlen) ok = 0; }
    if (seen != d.n) ok = 0;
    printf("HStr  n=%d cap=%d pool=%lu B  iterated=%d  %s\n", d.n, d.cap, (unsigned long)d.plen, seen, ok ? "ok" : "FAIL");

    /* the point of it: doc id -> row by hash vs by scanning every row */
    scanids = (i64 *)malloc(sizeof(i64) * N);
    for (i = 0; i < N; ++i) scanids[i] = (i64)i * 7919 + 13;
    t0 = clock();
    for (i = 0; i < N; i += 100) { i32 r; i64 want = (i64)(N - 1 - i) * 7919 + 13; for (r = 0; r < N; ++r) if (scanids[r] == want) break; if (r == N) ok = 0; }
    t_scan = (double)(clock() - t0) / CLOCKS_PER_SEC;
    t0 = clock();
    for (i = 0; i < N; i += 100) if (h64_get(&h, (i64)(N - 1 - i) * 7919 + 13) < 0) ok = 0;
    t_h = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("lookup of %d ids among %d: scan %.4f s, hash %.6f s  (%s)\n", N / 100, N, t_scan, t_h, t_h < t_scan ? "hash faster" : "?");
    free(scanids);
    h64_free(&h); hs_free(&d);
    puts(ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
