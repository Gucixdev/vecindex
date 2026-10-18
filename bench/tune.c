/* tune.c - ask the store for a recall@10 and check that it keeps it, on GloVe-100's held-out queries.
 *
 *   gcc -std=gnu99 -O2 -march=native -I.. tune.c -lm -lpthread -o tune && ./tune DATA_DIR 0.95
 *
 * The build tunes on the store's own rows (no traffic yet); then 2000 held-out queries are answered and timed, the
 * store re-tunes on the queries it just answered, and the other 8000 are answered with that. DATA_DIR comes from
 * prepare_glove.py. */
#include "store.c"
static Arena pa;
static double now_(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static i32 *GT; static f32 *Q; static i32 nq = 10000, dim = 100;
static double recall(RagStore *s, i32 from, i32 cnt, double *qps) {
  i32 i, hit = 0; double t0 = now_();
  for (i = from; i < from + cnt; ++i) { i64 r[10]; f32 c[10]; i32 g = store_search(s, Q + (size_t)i*dim, 10, r, c), x, z; for (x = 0; x < g; ++x) for (z = 0; z < 10; ++z) if (GT[(size_t)i*100+z] == r[x]) { ++hit; break; } }
  *qps = cnt / (now_() - t0); return hit / (10.0 * cnt); }
int main(int argc, char **argv){
  i32 n = 1183514, i; FILE *f; char p[512]; f32 *X = malloc((size_t)n*dim*4); RagStore *s; double t0, qps, r; f32 tgt = (f32)atof(argv[2]);
  Q = malloc((size_t)nq*dim*4); GT = malloc((size_t)nq*100*4);
  sprintf(p, "%s/train.f32", argv[1]); f = fopen(p, "rb"); if (fread(X, 4, (size_t)n*dim, f) != (size_t)n*dim) return 1; fclose(f);
  sprintf(p, "%s/test.f32", argv[1]); f = fopen(p, "rb"); if (fread(Q, 4, (size_t)nq*dim, f) != (size_t)nq*dim) return 1; fclose(f);
  sprintf(p, "%s/gt.i32", argv[1]); f = fopen(p, "rb"); if (fread(GT, 4, (size_t)nq*100, f) != (size_t)nq*100) return 1; fclose(f);
  s = store_new(&pa, dim); for (i = 0; i < n; ++i) store_add(s, X + (size_t)i*dim, i);
  store_set_target_recall(s, tgt); store_set_threads(s, 1);
  t0 = now_(); store_build(s);
  printf("target %.2f: build+tune %.2fs, tuned on its own rows -> %d probes (measured %.4f)\n", tgt, now_()-t0, s->tq->nprobe, store_tuned_recall(s));
  r = recall(s, 0, 2000, &qps); printf("  held-out queries 0-1999: recall@10 %.4f, %.0f QPS (1 thread)\n", r, qps);
  t0 = now_(); store_tune(s);
  printf("  re-tuned on the 2000 queries it answered: %d probes (measured %.4f) in %.2fs\n", s->tq->nprobe, store_tuned_recall(s), now_()-t0);
  r = recall(s, 2000, 8000, &qps); printf("  held-out queries 2000-9999: recall@10 %.4f, %.0f QPS (1 thread)\n", r, qps);
  return 0; }
