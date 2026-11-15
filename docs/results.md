# Results

GloVe-100 (1,183,514 vectors, 10,000 held-out queries, recall@10 against the exact top 10), 2 vCPUs of an Intel
Xeon (Cascade Lake). Queries per second at the fastest setting that reaches each recall, as one batch on 2 threads /
one query per call on 1 thread:

| method | build | bytes/vec | best recall@10 | >= 0.90 | >= 0.95 | >= 0.97 |
|---|---:|---:|---:|---:|---:|---:|
| vecindex 4-bit + f32 rows | 8.0 s | 475 | 0.9884 | 7,996 / 2,277 | 4,897 / 1,433 | 2,606 / 862 |
| vecindex 2-bit + int8 rows | 8.3 s | 153 | 0.9712 | 7,235 / 2,654 | 2,420 / 949 | 1,896 / 715 |
| FAISS IVF4096,PQ50x4fs,RFlat | 124.5 s | 436 | 0.9785 | 4,456 / 1,081 | 2,822 / 871 | 1,846 / 526 |
| turbovec 4-bit | 2.2 s | 56 | 0.8465 | - | - | - |

The method, every row, the scale runs (100M vectors in 5.84 GiB), what each optimisation bought and the bottlenecks
left are in [docs/benchmarks.md](docs/benchmarks.md).
