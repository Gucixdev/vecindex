# lean4 - vecindex, machine-checked

Lean 4 proofs about the kernels of vecindex's C code, and a test that holds the C code to the proved model. The
proofs cover the integrity of the store file (GF(2^8), the Cauchy Reed-Solomon parity and its rebuild, CRC-32C), the
integer arithmetic of the fast-scan kernels, what the codes mean and what the table scans add up, the rotation and
the top-k insertion; and on the server's side, the isolation of tenants' ids, the key file's tenant number, the
replica's journal after a crash and the request limit of an API key. Every theorem rests on Lean's three standard axioms only. The C code is compared with the Lean
definitions on every input of the GF(2^8) and digit tables, on ranges and grids of the other finite kernels and on
random inputs of the rest.

## How it holds together

A proof about Lean definitions says nothing about C code by itself. The chain from the C code to the theorems has
four links, and each is either proved or checked by a program:

1. **The model.** Each Lean definition follows one C function step by step, and its file names the function.
   `GF256.mul` is `gf_mul_`, `Crc.upTab` is `store_crc32c_up_`, `Gauss.invert` is `gf_invert_`, `RS.recover` is the
   rebuild in `ssum_group_`, `FastScan.avx2Score` is the AVX2 arithmetic of `tq_block_`, and so on (table below).
2. **The theorems** are about those definitions, and Lean's kernel checks every proof. `Audit.lean` takes every
   theorem the `Vecindex` modules define - 213 written in the sources and 351 that Lean generates there (equation
   lemmas and the like) - and fails if one rests on anything but `propext`, `Classical.choice` and `Quot.sound`: no
   `sorry`, no `native_decide` or `bv_decide` (they add `Lean.ofReduceBool` and `Lean.trustCompiler`, which trust the
   compiler), no axiom of a library. It selects theorems by the module that defines them, so a theorem outside the
   `Vecindex` namespace is audited too; a planted `sorry` and a planted `native_decide` both make it fail.
3. **The model against the C code.** `vecindex-oracle` runs the Lean definitions the theorems are about; `c/diff.c`
   runs the real C functions (it includes `src/store.c`, so the static functions are in reach). `run` compares their
   output: every entry of the GF(2^8) tables, all 65,536 products and a hash of every Cauchy coefficient, every byte
   of the q1-q5 digit tables, the column counts of every even `dpad` up to 4096, a grid of 1,050 file layouts and the
   Walsh-Hadamard chunks of every `dpad` up to 1024; random inputs for CRC-32C, the fast-scan block kernel and the
   top-k insertion. The C side is built three times (`-march=native`, `haswell`, `x86-64`), so the AVX-512, AVX2
   and scalar kernels and both CRC paths (SSE4.2 and the table) meet the same oracle.
4. **Does the test catch a wrong kernel?** `mutants` breaks the C code on purpose, one small change at a time in a
   copy of `src/`, and checks that the step covering it fails. 25 of 27 mutants are caught; the two that survive change
   code no input reaches: the row swap of `gf_invert_`, and the replica walk's cap on a frame's length, which only a
   journal past 1 GB could exercise (see Findings).

What stays trusted: Lean's kernel, the Lean compiler (it builds the oracle), gcc and the libc. The step from the model
to the C code is tested, not proved: exhaustively for the tables, on ranges, grids and random cases for the rest. The
theorems about sums and rotations hold in exact arithmetic, and the C code adds in f32, so for those `c/diff.c`
compares the C results with a double-precision computation, within a tolerance for f32 rounding.

## Build and run

```sh
cd holyc/vecindex
./run lean               # = lean4/run: build, axiom audit, gen/trustlean.c, differential test
lean4/run quick          # the same without rebuilding the proofs
N=2000 SEED=7 lean4/run  # more random cases, another seed (defaults: 300, the date)
lean4/mutants            # the mutation test (after a run)
```

Needs elan with `leanprover/lean4:v4.26.0`, gcc and bash. `lake` fetches Mathlib v4.26.0, TrustLean and AMO-Lean at
the commits in `lake-manifest.json` into `.lake/`. Mathlib's prebuilt cache (`lake exe cache get`) saves most of the
first build. Without it lake compiles the part of Mathlib these files need, with its dependencies: about 1,400
modules, which took about 30 minutes on the 2 vCPUs used here. `./run all` runs `lean4/run` too when `lake` is
installed.

On a 2-vCPU Xeon with AVX-512, 2026-11-13:

```
=================== lean4 summary ===================
  STEP                   RESULT  WHAT IT CHECKED
  lake build             ok      2785 jobs, proofs and oracle
  axiom audit            ok      753 theorems (270 written in the sources)
  source scan            ok      no sorry, admit, native_decide, bv_decide or axiom
  gen/trustlean.c        ok      printed by TrustLean from the proved programs
  gf native              ok      260 lines: every table entry and product, the coefficients hashed
  digits native          ok      1280 lines: every byte, q1-q5
  vd native              ok      3 lines: every even dpad to 4096, 1, 2 and 4 bits
  layout native          ok      1050 lines: a grid of sizes, block sizes, k, m
  chunks native          ok      1024 lines: every dpad from 1 to 1024
  crc native             ok      302 random cases, seed 20261111
  block native           ok      300 random cases, seed 20261111
  topk native            ok      300 random cases, seed 20261111
  props native           ok      9 checks
  ... the same nine steps for haswell and x86-64 ...
  tenant                 ok      2666 cases: edges of ids, ranges, tenants, numbers in the key file; and random, seed 20261111
  journal                ok      300 cases: journals torn, flipped, past the cap, wrong stamp: random, seed 20261111
  rate                   ok      300 cases: runs of 200 requests, f64 exact: random, seed 20261111
  ... nsset (3000 random cases) on every -march as well ...
  37 steps, all ok   (122 s)
```

The nine checks in C alone (`c/diff.c props`): `tq_code_` counts the boundaries below the value (random values,
the boundaries, specials, NaN); bit 3 of `tq_code_` is `v > 0`; the TrustLean kernels against vecindex's C on every
input (the `gf_init_` walk, the CRC-32C byte step, the lane write); the q1-q5 table scans against the inner product
with the dequantized vector; `tq_wht_` keeps inner products; and the store end to end: files written by
`ssum_write_file_` with at most m damaged blocks per group (data or parity) come back byte for byte, and a group with
m + 1 is reported as damaged, not passed off as rebuilt.

## What is proved

| C code | Lean | Theorem |
|---|---|---|
| `gf_init_`, `gf_mul_`, `gf_inv_` (`store_sum.c`) | `GF256/Tables.lean`, `Algebra.lean`, `Field.lean` | The tables are those of GF(2^8) with polynomial 0x11d and generator 2 (order 255); `gf_mul_` is `xt` applied `gf_log_[a]` times; with `^` as addition the bytes form a field (`instance : Field GF256`, built from the C tables, not assumed) |
| the Cauchy coefficients `gf_coef_` | `RS/Cauchy.lean`, `RS/StoreSum.lean` | `cauchy_kernel`: a Cauchy matrix has no kernel; `coef_mds`: every square part of the coefficient matrix of `store_sum.c` is invertible |
| `gf_invert_` | `RS/Invert.lean` | `invert_mul`: when it returns 0 the result is a left inverse; `invert_none`: it returns -1 only for a matrix with a nonzero kernel vector; `invert_ok`: it succeeds on every matrix without one |
| the rebuild of `ssum_group_` | `RS/Group.lean`, `RS/StoreSum.lean` | `recover_gf`: for every k and m that `ssum_layout_` accepts (k <= 128, m <= 64), every group of at most k data blocks and every choice of at most m damaged blocks among its data and parity, the rebuild returns the damaged data exactly |
| `ssum_layout_` | `RS/StoreSum.lean` | `layout_ng_le`: no group holds more than k data blocks, so `md[128]`, `mp[64]`, `use[64]` and `a[64 * 64]` are large enough; `layout_cover`, `layout_slot`: the groups deal out the data blocks without gaps or overlaps; `layout_burst`: a run of L neighbouring blocks puts at most ceil(L / G) of them into one group, so a run of at most m G is within the parity (`burst_within_parity`) and one of m G + 1 is not (`burst_beyond_parity`), the limit `tests/battle_damage.c` tests on real files; `layout_regions`: the digest tables and the trailers do not overlap, trailer A and table B start on 4 KiB boundaries, trailer B is the last 128 bytes |
| `store_crc_init_`, `store_crc32c_up_` (`store_batch.c`) | `Crc32c.lean` | `upTab_eq_upBit`: the table path is CRC-32C bit by bit; `check_value`: "123456789" gives 0xE3069283; `detects_byte_error`: two messages of the same length that differ in one byte never share a CRC (so every single-bit error is caught) |
| `tq_block_` (`tq_scan.c`) | `FastScan.lean` | `avx2_eq_ref`: the AVX2 / AVX-512 16-bit word trick (even rows as `a0 - (a1 << 8)`) gives the scalar kernel's sums for every block, table and column count; `no_overflow`: with `lutmax = min(255, 65535 / vd)` no sum wraps; `gt_flip`: the sign-flipped signed compare is the unsigned `v > thr` |
| `tq_vd_`, the kernel loads | `FastScan.lean` | `vd_even`, `vd_enough`: the column count is even and covers every coordinate; `avx2_loads_in_block`, `avx512_loads_in_block`: every 32- and 64-byte load stays inside the block |
| `tq_get_lane_`, `tq_put_lane_` | `FastScan.lean`, `FastScanLayout.lean` | `get_put_lane`, `get_put_lane_other`: writing one row's codes leaves the other rows alone; `laneByte_lt`, `laneByte_inj`: the block layout is AMO-Lean's stride permutation, so every code byte is inside the block and no two (row pair, column) share one |
| `tq_code_`, `tq_codes_`, `tq_c1_row_` (`tq.c`) | `Codes.lean` | `code_eq_count`: the four branch-free steps give the number of boundaries below the value, for any monotone comparisons (NaN included); `code_bit3`: bit 3 is the comparison with `TQ_B[7] = 0`, the coordinate's sign, which `tq_c1_row_` takes as the cascade's 1-bit code |
| the 2-bit and 1-bit columns of `tq_lut_` | `Codes.lean` | `col2_decode`, `lut2_at_code`, `lut1_at_code`: the 16-entry table of a column adds the products of its coordinates with their levels |
| `tq_lut_` | `Codes.lean` | `quantized_rank`: before rounding, the u8 tables rank the rows exactly as the scores |
| q1-q5 packing, `q*_lut_resid`, `vs_lut_gather_dot`, `q*_reconstruct_into` | `Codes.lean` | `digit_pack`, `or_pack`, `pack_lt`: packing levels into a byte and reading them back gives the levels; `scan_eq_dot`: the table scan is the inner product of the residual with the dequantized vector |
| `tq_wht_`, `tq_rot_`, the shuffle in `tq_build_skip` | `Hadamard.lean` | `pass_dot`, `wht_dot`, `whtN_dot`: each butterfly pass doubles a chunk's inner product, so the scaled transform keeps it; `chunks_cover`: the chunks are powers of two that lie inside `[0, dpad)` and add up to `dpad`; `signs_dot`, `perm_dot`, `shuffle_bijective`: the signs and the permutation keep inner products |
| `q1_topk_insert_` (and q2-q5) | `TopK.lean` | `topk_correct`: after any sequence of candidates the array holds `min(k, n)` of them, sorted, and no candidate left out is better than one kept; `bubble_eq_ins`: ties keep their arrival order |
| `vs_in_`, `vs_out_`, `vs_rin_`, `vs_own_`, the tenant number of `vs_keys_load_` (`vserv_keys.c`) | `Tenant.lean` | `out_in`: a wire id comes back as itself; `in_inj`: no two tenants share a store id; `in_fits`: tenant << 44 stays positive inside i64; `rin_exact`, `rin_none`: a tenant's range reaches exactly its own ids in the range; `own_iff`: the hybrid post-filter keeps exactly the tenant's rows; `parse_iff`: the key file takes a tenant number exactly when it is digits of a value 1 to 524287, `parse_bound`: no intermediate value past 5242879 |
| `vs_rjtrim_` (`vserv_repl.c`), the frame of `store_jframe_` (`store_batch.c`) | `Journal.lean` | `walk_frames`: a journal of whole frames is kept whole; `walk_take`, `good_take`, `trim_take`: any prefix of the primary's journal (a replica killed mid-write) keeps exactly the frames that arrived whole - the primary's header and first k frames, never a piece of a record; `resume`: the rest from that offset gives the primary's journal back. For any checksum into u32 |
| `vs_bucket_` (`vserv_keys.c`) | `Rate.lean` | `step_inv`: the tokens stay in [0, qps]; `window`: of any run of requests from t1 to t2 at most qps + qps (t2 - t1) get through, whatever came before; `wait_ok`: after 1/qps of quiet a request is let through |
| `tq_in_set_` (`tq.c`), the namespace set of `store_search_nss` | `NsSet.lean` | `inSet_iff`: on a set sorted ascending without repeats, the branchless lower bound answers exactly membership; `go_lt`: it reads only inside the set |
| `tq_em_update_`, `tq_em_revive_`, `tq_em_alpha_` (`tq.c`, e-means) | `Emeans.lean` | `pos_eq_ratio`: the C code's position-and-mass update equals the paper's running count and sum, ((1-a) S + a s) / ((1-a) N + a n); `lloyd_step`: a step of 1 makes a centroid the mean of its batch, one Lloyd round on a batch of every row; `weight_mem`, `pos_between`: for 0 < a <= 1 a step moves a centroid towards its batch mean and never past it; `mass_pos`: the mass stays positive; `split_mass`: a revival halves the donor's mass; `alpha_mem`: every step of the schedule lies in [a0 / 10, a0] for any cosine in [-1, 1]. Over any ordered field: the f32 code is held to it by `tests/emeans_test.c`, the whole algorithm to Meta's reference by `tests/emeans_ref.py` |
| `store_search_exact`, `store_ex_b_`, `store_ex_byte_` (`store_exact.c`, search with a proof) | `Exact.lean` | `dot_split`, `cs_abs`: a row's cosine is the code's part plus q·e, and \|q·e\| <= \|q\| \|e\| (Cauchy-Schwarz); `byte_covers`: the error byte, ceil(\|e\| / step), covers \|e\|; `round_sum`, `table_sum`: u8 table entries each within 1/2 of exact add up to within n / (2 sc) of the exact sum; `est_bound`: together, \|cosine - estimate\| <= (n / (2 sc) + \|q\| byte step) / dim + margin; `skip_sound`: every metric's score is a line a c + b0 with a >= 0, so a row whose line at estimate + bound is under the threshold scores under it; `skipped_not_topk`: such a row has k rows strictly better; `answer_top`: the k best of the rows scored exactly, every other row skipped under a threshold each answer row reaches, is the store's exact top k; `full_or_all`: an answer short of k means no row was skipped. Over any ordered field: `tests/exact_test.c` checks each term of the bound on every row, from codes decoded apart from the encoder, and the answer against every row scored |

## TrustLean and AMO-Lean

**TrustLean** ([manuelpuebla/trust-lean](https://github.com/manuelpuebla/trust-lean)) gives a formal semantics of a
small C subset (MicroC, with 32-bit wrapping in `evalMicroC_uint32`) and a printer from it to C text.
`Vecindex/CodeGen.lean` writes three C kernels as MicroC programs - one turn of `gf_init_`'s loop, one byte of
CRC-32C bit by bit, one byte of `tq_put_lane_` - and proves that under that semantics they compute the same Lean
definitions as above (`xtime_correct`, `crcByte_correct`, `putNib_correct`). TrustLean prints them to
`gen/trustlean.c`, and `c/diff.c` runs them against vecindex's C on every input of the kernel: a second,
independently written implementation of each kernel, tied to the specification by a proof. `run` fails if
`gen/trustlean.c` is not what the proved programs print.

**AMO-Lean** ([manuelpuebla/amo-lean](https://github.com/manuelpuebla/amo-lean)) supplies the stride permutation of
its matrix layer (`AmoLean.Matrix.strideIndex`, with `strideIndex_lt` and `stride_inverse_eq`). The fast-scan block
stores its codes column by column, which is that permutation, so AMO-Lean's lemmas give the two layout facts in
`FastScanLayout.lean`. The rest of AMO-Lean (NTT, FRI, Poseidon2, its e-graph optimizer) is about prime-field
cryptography and has no counterpart in vecindex; it is not imported.

## Findings

- **The whole test suite of vecindex runs and passes.** vecindex's README had several rows as "tests written, not
  yet run" (integrity, memory tiers, tenants, transactions, ASan/UBSan). `./run` on the same machine, 2026-11-06: 26
  steps, all ok, the store under TSan and ASan/UBSan and the kill -9 test included. The rows now say so.
- **The row swap of `gf_invert_` is never needed.** Every square part of a Cauchy matrix is invertible, so its leading
  minors are nonzero and the pivot of each column is already on the diagonal; the mutant that deletes the swap
  survives every test. This is an argument, not one of the Lean theorems. The swap is correct defensive code, and the
  proofs here cover it.
- **`gf_inv_(0)` is 1, not 0.** It is never called with 0: the denominators `(k + p) ^ i` are nonzero because
  `k + p > i`, and `gf_invert_` inverts only nonzero pivots. `RS/StoreSum.lean` relies on exactly that.
- **`tq_code_` never reads its sentinel.** The search compares with `TQ_B[0..14]` only (`code_congr`);
  `TQ_B[15] = 1e30f` is never read.
- **`q1_topk_insert_` needs `k >= 1`.** With `k = 0` it would read `scores[-1]`. The entry points checked refuse or
  replace a k below 1: `vi_search_ok_` (vecindex.c), the server protocol (vserv_proto.c), the store search.
- **TrustLean's printer-parser round trip does not build** at the pinned commit on Lean 4.26.0:
  `TrustLean/MicroC/RoundtripExpr.lean` stops with 27 errors (missing cases for the two cast operators, failing
  `change` steps for the bitwise ones). `CodeGen.lean` therefore does not use `master_roundtrip`; the differential
  test stands in for it.
- **AMO-Lean declares 8 axioms**, all in its Radix-4 NTT. None of the modules used here depends on them; the axiom
  audit would fail if one did.
- **A tenant's hybrid search can return fewer than k rows.** `vs_hybrid_` asks the whole store for 4 k hits and keeps
  the tenant's own (`own_iff` proves that it keeps exactly those): with many other tenants' rows ranked higher, fewer
  than k of 4 k are its own. Isolation holds; recall can suffer. The vector and range searches do not have this: they
  search the tenant's namespace itself.
- **The key file once read a long line as two.** A line past the reader's 511 bytes had its tail parsed as a line of
  its own, so a long comment could carry a key. Found while modelling the parser; `vs_keys_load_` now refuses the file,
  and `tests/vserv_test.c` fails on the old code.
- **The cap of the replica's walk is unreachable in tests.** A frame length past `STORE_BATCH_MAX` (1 GB) is also past
  the bytes of any test journal, so removing the check stops the walk at the same place (the mutant survives, listed).
- **Both libraries use `native_decide` widely** (675 uses in TrustLean, 511 in AMO-Lean, outside comments). The lemmas
  used here do not depend on it (audit).

## Not covered

Threads and locks, the journal replay and the file I/O around the parity, BLAKE3 (tested against the reference
implementation's digests in `tests/store_test.c`), the k-means and HNSW routers, BM25, the server's protocols and TLS
(only its id mapping, key file number, token bucket and the replica's journal walk are modelled), the primary's side
of the replica (that the bytes served end on a whole record is the write lock's business, tested in `tests/vserv_test.c`),
and the rounding of f32 arithmetic (bounded by tests, not proved). The proofs assume that a damaged block's BLAKE3
digest differs from the stored one; a collision is outside the model.

## Layout

| path | what |
|---|---|
| `Vecindex/` | the model and the proofs, one file per kernel (`Vecindex.lean` imports them all) |
| `Oracle.lean` | `vecindex-oracle`: the model on concrete inputs; prints `gen/trustlean.c` |
| `Audit.lean` | the axiom audit |
| `c/diff.c` | the C side of the differential test (the store's kernels) |
| `c/vsdiff.c` | the same for the server's kernels: it includes `vserv.c` with its main renamed |
| `gen/trustlean.c` | the C that TrustLean prints from the proved programs (generated, committed for review) |
| `run`, `mutants` | the test and the mutation test |
| `lakefile.lean`, `lake-manifest.json`, `lean-toolchain` | Lean 4.26.0, Mathlib v4.26.0, TrustLean and AMO-Lean pinned |

## License

MIT, as vecindex. Fetched by lake, not vendored: Mathlib (Apache-2.0), TrustLean (MIT per its README; the repository
has no LICENSE file), AMO-Lean (MIT).
