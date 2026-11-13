/-
  Search with a proof, `store_search_exact` in `src/store_exact.c`. The fast scan reads a row's 4-, 2- or 1-bit code
  x̂ through a u8 table and gives an estimate of the cosine; the row is x = x̂ + e. Proved over any ordered field:

  * `dot_split`: q·x = q·x̂ + q·e - the estimate's error is the table's rounding plus q·e;
  * `cs_abs` (Cauchy-Schwarz): |q·e| <= |q| |e|, with |q| and |e| given as the roots of the sums of squares
    (`StoreExQ.qr` is |rotated q|; the row's byte holds |e| in steps);
  * `byte_covers`: a byte of ceil(|e| / step) steps covers |e| (`store_ex_byte_` rounds up, never down);
  * `table_sum`: when every table entry is within 1/2 of its exact value times the scale sc, the sum read from the
    table is within n / (2 sc) of the exact sum over n columns (`StoreExQ.lutb`);
  * `est_bound`: together, |cosine - estimate| <= (n/(2 sc) + |q| byte step) / dim + margin (`store_ex_b_`);
  * `skip_sound`: every metric's score is a line a c + b0 in the cosine with a >= 0 (cosine 2c - 2, dot
    |q| |x| c, L2 2 |q| |x| c - |q|^2 - |x|^2), so a row whose line at estimate + bound is below the threshold has a
    score below it - the row the scan skips cannot beat the threshold;
  * `skipped_not_topk`: with k rows at or above the threshold, a skipped row has k rows strictly better: it is in no
    top k;
  * `answer_top`: the answer is the k best of the rows scored exactly, and every skipped row was below a threshold
    that each answer row reaches - so no row anywhere beats an answer row: the answer is exact.

  The C code computes in f32 and f64; `STORE_EX_MARGIN` (1e-4) covers the rounding of the estimate, and
  `tests/exact_test.c` checks each term of the bound on every row against codes decoded independently of the encoder.
  Trusted: Lean's kernel, Mathlib, and the step from this model to the C.
-/
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Algebra.Order.Floor.Semiring
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

namespace Vecindex.Exact

open Finset

section field

variable {F : Type*} [Field F] [LinearOrder F] [IsStrictOrderedRing F]
variable {ι : Type*}

omit [LinearOrder F] [IsStrictOrderedRing F] in
/-- The dot product with the row splits into the code's part and the coding error's part. -/
theorem dot_split (s : Finset ι) (q xh e : ι → F) :
    ∑ i ∈ s, q i * (xh i + e i) = ∑ i ∈ s, q i * xh i + ∑ i ∈ s, q i * e i := by
  rw [← sum_add_distrib]; exact sum_congr rfl fun i _ => mul_add _ _ _

/-- Cauchy-Schwarz with the lengths as roots: `|q·e| <= |q| |e|`. -/
theorem cs_abs (s : Finset ι) (q e : ι → F) {qr en : F} (hq : 0 ≤ qr) (he : 0 ≤ en)
    (hq2 : qr ^ 2 = ∑ i ∈ s, q i ^ 2) (he2 : en ^ 2 = ∑ i ∈ s, e i ^ 2) :
    |∑ i ∈ s, q i * e i| ≤ qr * en := by
  apply abs_le_of_sq_le_sq _ (mul_nonneg hq he)
  rw [mul_pow, hq2, he2]
  exact sum_mul_sq_le_sq_mul_sq s q e

/-- The error byte: ceil(|e| / step) steps cover |e|. -/
theorem byte_covers [FloorSemiring F] {e step : F} (hs : 0 < step) :
    e ≤ (⌈e / step⌉₊ : F) * step := by
  have h := Nat.le_ceil (e / step)
  rwa [div_le_iff₀ hs] at h

/-- `n` roundings of at most 1/2 each add up to at most n/2. -/
theorem round_sum (s : Finset ι) (r : ι → F) (h : ∀ i ∈ s, |r i| ≤ 1 / 2) :
    |∑ i ∈ s, r i| ≤ s.card / 2 := by
  calc |∑ i ∈ s, r i| ≤ ∑ i ∈ s, |r i| := abs_sum_le_sum_abs _ _
    _ ≤ ∑ _i ∈ s, (1 / 2 : F) := sum_le_sum h
    _ = s.card / 2 := by rw [sum_const, nsmul_eq_mul]; ring

/-- The table: column i holds `t i`, within 1/2 of `(x i - m i) sc` (the C subtracts the column's minimum m and
    rounds half up into a u8). The sum read from the table, rescaled, is within n / (2 sc) of the exact sum. -/
theorem table_sum (s : Finset ι) (x m t : ι → F) {sc : F} (hsc : 0 < sc)
    (h : ∀ i ∈ s, |t i - (x i - m i) * sc| ≤ 1 / 2) :
    |∑ i ∈ s, x i - ((∑ i ∈ s, t i) / sc + ∑ i ∈ s, m i)| ≤ s.card / (2 * sc) := by
  have key : ∑ i ∈ s, x i - ((∑ i ∈ s, t i) / sc + ∑ i ∈ s, m i)
      = -(∑ i ∈ s, (t i - (x i - m i) * sc)) / sc := by
    rw [sum_sub_distrib, ← sum_mul, sum_sub_distrib]; field_simp; ring
  rw [key, abs_div, abs_neg, abs_of_pos hsc, div_le_div_iff₀ hsc (by positivity)]
  have := round_sum s _ h
  nlinarith

/-- The whole bound, `store_ex_b_`: the cosine is (h + d) / dim where h is the code's dot product and d = q·e; the
    estimate is h' / dim, with |h - h'| <= L from the table and |d| <= |q| |e| <= |q| byte step. -/
theorem est_bound {h h' d L qr en byte step inv margin : F} (hinv : 0 < inv) (hm : 0 ≤ margin) (hq : 0 ≤ qr)
    (hL : |h - h'| ≤ L) (hd : |d| ≤ qr * en) (hb : en ≤ byte * step) :
    |(h + d) * inv - h' * inv| ≤ (L + qr * byte * step) * inv + margin := by
  have hd' : |d| ≤ qr * byte * step := hd.trans (by rw [mul_assoc]; exact mul_le_mul_of_nonneg_left hb hq)
  have e1 : (h + d) * inv - h' * inv = ((h - h') + d) * inv := by ring
  rw [e1, abs_mul, abs_of_pos hinv]
  have := (abs_add_le (h - h') d).trans (add_le_add hL hd')
  nlinarith

/-- A row whose score line at estimate + bound is below the threshold scores below it. -/
theorem skip_sound {a b0 c est bd tau : F} (ha : 0 ≤ a) (hc : |c - est| ≤ bd) (hskip : a * (est + bd) + b0 < tau) :
    a * c + b0 < tau := by
  have : c ≤ est + bd := by linarith [le_abs_self (c - est)]
  nlinarith

end field

section order

variable {G : Type*} [LinearOrder G] {ι : Type*}

/-- A skipped row is in no top k: k rows reach the threshold, so k rows are strictly better than it. -/
theorem skipped_not_topk (rows W : Finset ι) (sc : ι → G) {x : ι} {k : ℕ} {tau : G} (hW : W ⊆ rows)
    (hk : k ≤ W.card) (hge : ∀ w ∈ W, tau ≤ sc w) (hx : sc x < tau) :
    k ≤ (rows.filter fun w => sc x < sc w).card := by
  refine hk.trans (card_le_card fun w hw => ?_)
  exact mem_filter.mpr ⟨hW hw, hx.trans_le (hge w hw)⟩

/-- The answer is exact. `C` are the rows scored exactly, `A` the answer: the best of `C`. Every row outside `C` was
    skipped below a threshold that every answer row reaches (the threshold only rises, and the answer holds the best
    rows seen). Then no row of the store beats an answer row. -/
theorem answer_top (rows C A : Finset ι) (sc : ι → G)
    (hbest : ∀ a ∈ A, ∀ c ∈ C, c ∉ A → sc c ≤ sc a)
    (hskip : ∀ x ∈ rows, x ∉ C → ∃ tau, sc x < tau ∧ ∀ a ∈ A, tau ≤ sc a) :
    ∀ a ∈ A, ∀ x ∈ rows, x ∉ A → sc x ≤ sc a := by
  intro a ha x hx hxA
  by_cases hxC : x ∈ C
  · exact hbest a ha x hxC hxA
  · obtain ⟨tau, hlt, hall⟩ := hskip x hx hxC
    exact (hlt.trans_le (hall a ha)).le

/-- When the answer is not full (fewer than k rows), nothing was skipped: the threshold is only set once k rows are
    held, so every row was scored, and the answer is every row that passed. Here as: a skip needs k answer rows. -/
theorem full_or_all (rows C A : Finset ι) {k : ℕ} (hCr : C ⊆ rows)
    (hskip : ∀ x ∈ rows, x ∉ C → k ≤ A.card) (hsmall : A.card < k) : C = rows := by
  refine le_antisymm hCr fun x hx => ?_
  by_contra hxC
  exact absurd (hskip x hx hxC) (not_le.mpr hsmall)

end order

end Vecindex.Exact
