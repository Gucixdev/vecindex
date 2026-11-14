/-
  Why the configurations are what they are - the rules behind them, proved; what only a measurement can say is named
  as such. Over any ordered field (the C computes in floats; these are the exact-arithmetic facts its choices rest on).

  1. The cluster count (`store_nclusters_` in `src/store.c`, 4 sqrt(n)). Model: a query compares itself with all c
     centroids (cost a each) and scans the rows of the p clusters it probes, n / c rows each on average (cost b a
     row): a c + b p n / c. `cost_ge` and `cost_eq`: the least cost is 2 m with m^2 = a b p n, reached where a c = m,
     that is c = sqrt(b p n / a). So the count grows as sqrt(n); the factor is sqrt(b p / a): 4 when a query probes
     16 clusters and a centroid costs what a row's code does (`four_sqrt`). The C's 4 is the usual rule of thumb, not
     derived from measured a, b and p - with the probes the tuner picks, the best factor is sqrt(b p / a) for that p,
     which a sweep over the cluster count has to show.
  2. Re-scoring precision. `order_kept`: two scores each known within eps keep their order when they differ by more
     than 2 eps. `topk_kept`: a true top set whose every row beats every other row by more than 2 eps is still on top
     by the approximate scores. `dot_err`: per coordinate errors of at most step / 2 move a dot product with q by at
     most step / 2 times the sum of |q_i|. `step_order`: on the same range, a uniform quantizer's step is range /
     (levels - 1): int8 (255 steps) is 17 times finer than 4-bit (15), 4-bit 5 times finer than 2-bit (3), and
     ternary (2 steps) is coarser than 2-bit - which is why rows for re-scoring are int8 or f32 and never ternary.
  3. Probes. `probe_mono`: probing more clusters of the same ranking keeps every cluster probed before, so every
     candidate. `found_mono`: when every candidate is re-scored, a row of the true top k among the candidates stays
     in the top k of a bigger candidate set - recall never falls as probes rise. `rerank_can_lose`: with a re-rank
     depth R below the candidate count it can: a new candidate with a higher estimate takes the place of a true
     neighbour. So the tuner's search over probe counts is safe only with R big enough, or by measuring each count.
  4. Memory. `payload`: the bytes a row's code and row take, dim * bits / 8 + dim * row bytes; the rest (id, slot,
     the 1-bit cascade, an int8 row's scale) is counted by `tq_ram_` and measured, not modelled here.

  Not provable here, and not claimed: which recall a RAG application needs, and which codes reach which recall on
  given embeddings - those are properties of the data, measured (docs/benchmarks.md) and tuned (store_tune.c).
-/
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

namespace Vecindex.Config

open Finset

section field

variable {F : Type*} [Field F] [LinearOrder F] [IsStrictOrderedRing F]

/-- The cost model of one query with c clusters. -/
def cost (a b p n c : F) : F := a * c + b * p * n / c

/-- No cluster count costs less than 2 m, m being the root of a b p n. -/
theorem cost_ge {a b p n c m : F} (ha : 0 < a) (hc : 0 < c) (hm : m ^ 2 = a * b * p * n) :
    2 * m ≤ cost a b p n c := by
  unfold cost
  have hac : 0 < a * c := mul_pos ha hc
  have key : a * c + b * p * n / c - 2 * m = (a * c - m) ^ 2 / (a * c) := by
    field_simp
    rw [show (a * c - m) ^ 2 = a ^ 2 * c ^ 2 - 2 * a * c * m + m ^ 2 by ring, hm]
    ring
  have : 0 ≤ (a * c - m) ^ 2 / (a * c) := div_nonneg (sq_nonneg _) hac.le
  linarith

omit [IsStrictOrderedRing F] in
/-- ... and the count with a c = m, that is c = sqrt(b p n / a), costs exactly 2 m: the best. -/
theorem cost_eq {a b p n c m : F} (ha : 0 < a) (hc : 0 < c) (hm : m ^ 2 = a * b * p * n) (hopt : a * c = m) :
    cost a b p n c = 2 * m := by
  unfold cost
  have hc0 : c ≠ 0 := hc.ne'
  have ha0 : a ≠ 0 := ha.ne'
  have : b * p * n / c = m := by
    rw [div_eq_iff hc0]
    have h2 : a * (b * p * n) = a * (m * c) := by
      calc a * (b * p * n) = m ^ 2 := by rw [hm]; ring
        _ = m * (a * c) := by rw [hopt]; ring
        _ = a * (m * c) := by ring
    exact mul_left_cancel₀ ha0 h2
  rw [this, hopt]; ring

omit [IsStrictOrderedRing F] in
/-- The factor 4: probing 16 clusters with a centroid as dear as a row (b = a), the best count c has c^2 = 16 n,
    c = 4 sqrt(n). -/
theorem four_sqrt {a n c : F} (ha : 0 < a) (hc : 0 < c) (h16 : c ^ 2 = 16 * n) :
    cost a a 16 n c = 2 * (a * c) := by
  apply cost_eq ha hc _ rfl
  calc (a * c) ^ 2 = a ^ 2 * c ^ 2 := by ring
    _ = a * a * 16 * n := by rw [h16]; ring

/-- Two scores each known within eps keep their order when they differ by more than 2 eps. -/
theorem order_kept {s t s' t' eps : F} (hs : |s' - s| ≤ eps) (ht : |t' - t| ≤ eps) (gap : t + 2 * eps < s) :
    t' < s' := by
  have h1 := neg_abs_le (s' - s)
  have h2 := le_abs_self (t' - t)
  linarith

variable {ι : Type*}

/-- A true top set whose rows beat every other row by more than 2 eps stays on top by scores known within eps. -/
theorem topk_kept (A : Finset ι) (s s' : ι → F) {eps : F} (herr : ∀ i, |s' i - s i| ≤ eps)
    (gap : ∀ a ∈ A, ∀ y, y ∉ A → s y + 2 * eps < s a) :
    ∀ a ∈ A, ∀ y, y ∉ A → s' y < s' a :=
  fun a ha y hy => order_kept (herr a) (herr y) (gap a ha y hy)

/-- Per coordinate errors of at most step / 2 move a dot product with q by at most step / 2 times the sum of |q_i|. -/
theorem dot_err (S : Finset ι) (q e : ι → F) {step : F} (h : ∀ i ∈ S, |e i| ≤ step / 2) :
    |∑ i ∈ S, q i * e i| ≤ step / 2 * ∑ i ∈ S, |q i| := by
  calc |∑ i ∈ S, q i * e i| ≤ ∑ i ∈ S, |q i * e i| := abs_sum_le_sum_abs _ _
    _ = ∑ i ∈ S, |q i| * |e i| := sum_congr rfl fun i _ => abs_mul _ _
    _ ≤ ∑ i ∈ S, |q i| * (step / 2) := sum_le_sum fun i hi => mul_le_mul_of_nonneg_left (h i hi) (abs_nonneg _)
    _ = step / 2 * ∑ i ∈ S, |q i| := by rw [← sum_mul]; ring

/-- A uniform quantizer of `levels` levels on a range of width w steps by w / (levels - 1). -/
def step (w : F) (levels : ℕ) : F := w / ((levels : F) - 1)

/-- int8 is 17 times finer than 4-bit, 4-bit 5 times finer than 2-bit, 2-bit finer than ternary. -/
theorem step_order {w : F} (hw : 0 < w) :
    step w 16 = 17 * step w 256 ∧ step w 4 = 5 * step w 16 ∧ step w 4 < step w 3 := by
  unfold step
  norm_num
  refine ⟨by field_simp; norm_num, by field_simp; norm_num, ?_⟩
  rw [div_lt_div_iff_of_pos_left hw (by norm_num) (by norm_num)]
  norm_num

end field

section probes

variable {ι : Type*} [DecidableEq ι]

omit [DecidableEq ι] in
/-- Probing more clusters of one ranking keeps every cluster probed before. -/
theorem probe_mono (rank : List ι) {p p' : ℕ} (h : p ≤ p') : ∀ x ∈ rank.take p, x ∈ rank.take p' := by
  intro x hx
  have : rank.take p = (rank.take p').take p := by rw [List.take_take, Nat.min_eq_left h]
  rw [this] at hx
  exact List.mem_of_mem_take hx

/-- The rows of the probed clusters: the candidates. More probes, a superset of candidates. -/
theorem cand_mono (rank : List ι) (rows : ι → Finset ℕ) {p p' : ℕ} (h : p ≤ p') :
    ((rank.take p).toFinset.biUnion rows) ⊆ ((rank.take p').toFinset.biUnion rows) := by
  intro r hr
  simp only [mem_biUnion, List.mem_toFinset] at hr ⊢
  obtain ⟨c, hc, hr⟩ := hr
  exact ⟨c, probe_mono rank h c hc, hr⟩

end probes

section rerank

variable {G : Type*} [LinearOrder G]

/-- In the top k of a set: fewer than k of its rows score strictly higher. -/
def inTop (S : Finset ℕ) (s : ℕ → G) (k : ℕ) (x : ℕ) : Prop := (S.filter fun y => s x < s y).card < k

/-- With every candidate re-scored, a row in the top k of the candidates stays in the top k of a bigger set of
    candidates as long as it is in the true top k of the whole store: recall does not fall as probes rise. -/
theorem found_mono (all S T : Finset ℕ) (s : ℕ → G) (k : ℕ) (x : ℕ) (hST : S ⊆ T) (hT : T ⊆ all)
    (hx : x ∈ S) (htrue : inTop all s k x) : x ∈ T ∧ inTop T s k x := by
  refine ⟨hST hx, lt_of_le_of_lt (card_le_card ?_) htrue⟩
  intro y hy
  simp only [mem_filter] at hy ⊢
  exact ⟨hT hy.1, hy.2⟩

/-- The R best by estimate, for R = 1: the candidates no other candidate's estimate beats. -/
def best1 (S : Finset ℕ) (est : ℕ → ℕ) : Finset ℕ := S.filter fun y => ∀ z ∈ S, est z ≤ est y

/-- With a re-rank depth below the candidate count, more candidates can lose a neighbour: row 0 is kept from {0}
    and lost from {0, 1}, where row 1's estimate is higher. -/
theorem rerank_can_lose :
    ¬ ∀ (S T : Finset ℕ) (est : ℕ → ℕ) (x : ℕ), S ⊆ T → x ∈ best1 S est → x ∈ best1 T est := by
  intro h
  have := h {0} {0, 1} id 0 (by decide) (by decide)
  revert this
  decide

end rerank

/-- The bytes a row's code and its row take: dim * bits / 8 + dim * row bytes (4-bit + f32 at dim 100: 450). -/
def payload (dim bits rowBytes : ℕ) : ℕ := dim * bits / 8 + dim * rowBytes

theorem payload_glove : payload 100 4 4 = 450 ∧ payload 100 2 1 = 125 ∧ payload 100 4 1 = 150 := by decide

end Vecindex.Config
