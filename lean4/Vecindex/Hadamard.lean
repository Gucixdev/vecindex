/-
  The rotation of `src/tq.c`: random signs, Walsh-Hadamard, a permutation, random signs, Walsh-Hadamard, scale.
  What the search needs from it is that it keeps inner products: a query and a row rotated the same way score the
  same as before, times one constant, so the ranking does not change. Proved here in exact arithmetic (any
  commutative ring; the C code adds in f32):

  * `pass_dot`: one butterfly pass `x[j], x[j+h] = x[j] + x[j+h], x[j] - x[j+h]` over a chunk doubles the inner
    product on that chunk;
  * `wht_dot`: the `k` passes of a chunk of `2^k` multiply it by `2^k`, and with the scale `1/sqrt(len)` of
    `tq_wht_` (any `s` with `s * s * len = 1`) the chunk keeps its inner product (`whtN_dot`);
  * `pass_outside`: a pass touches nothing outside its chunk;
  * `chunks_cover`: the chunks of `tq_wht_` (largest power of two first, the binary decomposition of `dpad`) are
    consecutive powers of two that cover `[0, dpad)` exactly;
  * `signs_dot`, `perm_dot`: the sign flips (`s[i] = +-1`) and the gather through a permutation keep the inner
    product; `fisher_yates_perm`: the shuffle of `tq_build` gives a permutation whatever its random numbers.
-/
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Logic.Equiv.Basic
import Mathlib.Tactic.Ring

open Finset

namespace Vecindex.Hadamard

variable {α : Type*} [CommRing α]

/-- `Σ_{i < m n} f i = Σ_{g < m} Σ_{s < n} f (g n + s)`. -/
theorem sum_range_mul (f : ℕ → α) (m n : ℕ) :
    ∑ i ∈ range (m * n), f i = ∑ g ∈ range m, ∑ s ∈ range n, f (g * n + s) := by
  induction m with
  | zero => simp
  | succ m ih => rw [Nat.succ_mul, sum_range_add, ih, sum_range_succ]

/-- The inner product over the chunk `[base, base + len)`. -/
def dot (base len : ℕ) (x y : ℕ → α) : α := ∑ i ∈ range len, x (base + i) * y (base + i)

/-- One pass of span `h` over the chunk `[base, base + len)`: the lower index of each pair gets `a + b`, the upper
    one `a - b` (`x[j] = a + b; x[j + h] = a - b;`). -/
def pass (base len h : ℕ) (x : ℕ → α) : ℕ → α := fun i =>
  if base ≤ i ∧ i < base + len then
    if (i - base) / h % 2 = 0 then x i + x (i + h) else x (i - h) - x i
  else x i

theorem pass_outside (base len h : ℕ) (x : ℕ → α) (i : ℕ) (hi : ¬ (base ≤ i ∧ i < base + len)) :
    pass base len h x i = x i := by
  unfold pass; rw [if_neg hi]

/-- **A pass doubles the inner product of its chunk.** -/
theorem pass_dot (base h m : ℕ) (hh : 0 < h) (x y : ℕ → α) :
    dot base (m * (2 * h)) (pass base (m * (2 * h)) h x) (pass base (m * (2 * h)) h y) =
      2 * dot base (m * (2 * h)) x y := by
  set L := m * (2 * h) with hL
  have hin : ∀ g, g < m → ∀ t, t < 2 * h → base ≤ base + (g * (2 * h) + t) ∧ base + (g * (2 * h) + t) < base + L := by
    intro g hg t ht
    refine ⟨by omega, ?_⟩
    have : g * (2 * h) + t < (g + 1) * (2 * h) := by rw [Nat.add_mul, Nat.one_mul]; omega
    have : (g + 1) * (2 * h) ≤ m * (2 * h) := Nat.mul_le_mul_right _ hg
    omega
  have key : ∀ g, g < m → ∀ o, o < h →
      pass base L h x (base + (g * (2 * h) + o)) * pass base L h y (base + (g * (2 * h) + o)) +
      pass base L h x (base + (g * (2 * h) + (h + o))) * pass base L h y (base + (g * (2 * h) + (h + o))) =
      2 * (x (base + (g * (2 * h) + o)) * y (base + (g * (2 * h) + o)) +
           x (base + (g * (2 * h) + (h + o))) * y (base + (g * (2 * h) + (h + o)))) := by
    intro g hg o ho
    have hlo : (g * (2 * h) + o) / h % 2 = 0 := by
      have : g * (2 * h) + o = h * (2 * g) + o := by ring
      rw [this, Nat.mul_add_div hh, Nat.div_eq_of_lt ho]; omega
    have hup : (g * (2 * h) + (h + o)) / h % 2 = 1 := by
      have : g * (2 * h) + (h + o) = h * (2 * g + 1) + o := by ring
      rw [this, Nat.mul_add_div hh, Nat.div_eq_of_lt ho]; omega
    have e1 : ∀ z : ℕ → α, pass base L h z (base + (g * (2 * h) + o)) =
        z (base + (g * (2 * h) + o)) + z (base + (g * (2 * h) + (h + o))) := by
      intro z
      unfold pass
      rw [if_pos (hin g hg o (by omega)), Nat.add_sub_cancel_left, if_pos hlo]
      congr 2; omega
    have e2 : ∀ z : ℕ → α, pass base L h z (base + (g * (2 * h) + (h + o))) =
        z (base + (g * (2 * h) + o)) - z (base + (g * (2 * h) + (h + o))) := by
      intro z
      unfold pass
      rw [if_pos (hin g hg (h + o) (by omega)), Nat.add_sub_cancel_left, if_neg (by omega)]
      congr 2; omega
    rw [e1 x, e1 y, e2 x, e2 y]
    ring
  have split : ∀ F : ℕ → α, ∑ i ∈ range L, F i =
      ∑ g ∈ range m, ∑ o ∈ range h, (F (g * (2 * h) + o) + F (g * (2 * h) + (h + o))) := by
    intro F
    rw [hL, sum_range_mul]
    refine sum_congr rfl fun g _ => ?_
    rw [show 2 * h = h + h by ring, sum_range_add, ← sum_add_distrib]
  unfold dot
  rw [split (fun i => pass base L h x (base + i) * pass base L h y (base + i)),
    split (fun i => x (base + i) * y (base + i)), mul_sum]
  refine sum_congr rfl fun g hg => ?_
  rw [mul_sum]
  exact sum_congr rfl fun o ho => key g (mem_range.mp hg) o (mem_range.mp ho)

/-- The passes of spans `1, 2, 4, ..., 2^(t-1)` (`for (h = 1; h < len; h *= 2)`). -/
def passes (base len : ℕ) : ℕ → (ℕ → α) → (ℕ → α)
  | 0, x => x
  | t + 1, x => pass base len (2 ^ t) (passes base len t x)

/-- The unnormalized transform of a chunk of `2 ^ k`. -/
def wht (base k : ℕ) (x : ℕ → α) : ℕ → α := passes base (2 ^ k) k x

theorem passes_dot (base k : ℕ) (x y : ℕ → α) : ∀ t, t ≤ k →
    dot base (2 ^ k) (passes base (2 ^ k) t x) (passes base (2 ^ k) t y) = 2 ^ t * dot base (2 ^ k) x y
  | 0, _ => by simp [passes]
  | t + 1, ht => by
    have hk : 2 ^ k = 2 ^ (k - t - 1) * (2 * 2 ^ t) := by
      rw [← pow_succ', ← pow_add]; congr 1; omega
    show dot base (2 ^ k) (pass base (2 ^ k) (2 ^ t) (passes base (2 ^ k) t x))
      (pass base (2 ^ k) (2 ^ t) (passes base (2 ^ k) t y)) = _
    have := pass_dot base (2 ^ t) (2 ^ (k - t - 1)) (Nat.two_pow_pos t) (passes base (2 ^ k) t x)
      (passes base (2 ^ k) t y)
    rw [← hk] at this
    rw [this, passes_dot base k x y t (by omega)]
    ring

/-- **The chunk transform multiplies the inner product by the chunk length.** -/
theorem wht_dot (base k : ℕ) (x y : ℕ → α) :
    dot base (2 ^ k) (wht base k x) (wht base k y) = (2 ^ k : α) * dot base (2 ^ k) x y := by
  unfold wht; rw [passes_dot base k x y k le_rfl]

/-- `x[i] *= sc` over the chunk after the passes. -/
def whtN (base k : ℕ) (s : α) (x : ℕ → α) : ℕ → α := fun i =>
  if base ≤ i ∧ i < base + 2 ^ k then s * wht base k x i else wht base k x i

/-- **With `sc = 1 / sqrt(len)` the chunk keeps its inner product.** -/
theorem whtN_dot (base k : ℕ) (s : α) (hs : s * s * 2 ^ k = 1) (x y : ℕ → α) :
    dot base (2 ^ k) (whtN base k s x) (whtN base k s y) = dot base (2 ^ k) x y := by
  have h1 : dot base (2 ^ k) (whtN base k s x) (whtN base k s y) =
      s * s * dot base (2 ^ k) (wht base k x) (wht base k y) := by
    unfold dot whtN
    rw [mul_sum]
    refine sum_congr rfl fun i hi => ?_
    have hc : base ≤ base + i ∧ base + i < base + 2 ^ k := ⟨by omega, by have := mem_range.mp hi; omega⟩
    rw [if_pos hc, if_pos hc]
    ring
  rw [h1, wht_dot, ← mul_assoc, hs, one_mul]

/-! ## The chunks of `tq_wht_` -/

/-- `for (len = 1; len * 2 <= dpad; len *= 2) {}`: the exponent of the largest power of two at most `n`. Each later
    chunk is the largest power of two of what is left too: after a chunk of `2^e`, less than `2^e` is left, and
    `while (len > dpad - base) len /= 2` halves down to it. -/
def topExp (n : ℕ) : ℕ := n.log2

/-- The chunks `(base, k)` of `tq_wht_` from `base` on, `rem` entries left: `while (len > dpad - base) len /= 2`
    keeps the largest power of two that fits, so each chunk is the largest power of two of what is left. -/
def chunks : ℕ → ℕ → List (ℕ × ℕ)
  | _, 0 => []
  | base, rem + 1 =>
    (base, topExp (rem + 1)) :: chunks (base + 2 ^ topExp (rem + 1)) (rem + 1 - 2 ^ topExp (rem + 1))
termination_by _ rem => rem
decreasing_by
  have : 2 ^ (rem + 1).log2 ≤ rem + 1 := Nat.log2_self_le (Nat.succ_ne_zero rem)
  have := Nat.two_pow_pos (topExp (rem + 1))
  simp only [topExp] at *
  omega

/-- **The chunks are consecutive powers of two that cover `[base, base + rem)` exactly.** -/
theorem chunks_cover : ∀ rem base, ((chunks base rem).map fun c => 2 ^ c.2).sum = rem ∧
    ∀ c ∈ chunks base rem, base ≤ c.1 ∧ c.1 + 2 ^ c.2 ≤ base + rem
  | 0, base => by simp [chunks]
  | rem + 1, base => by
    have hle : 2 ^ (rem + 1).log2 ≤ rem + 1 := Nat.log2_self_le (Nat.succ_ne_zero rem)
    have hpos := Nat.two_pow_pos (topExp (rem + 1))
    simp only [topExp] at hle hpos
    have ih := chunks_cover (rem + 1 - 2 ^ topExp (rem + 1)) (base + 2 ^ topExp (rem + 1))
    rw [chunks]
    simp only [List.map_cons, List.sum_cons, List.mem_cons, forall_eq_or_imp]
    simp only [topExp] at ih ⊢
    refine ⟨by rw [ih.1]; omega, ⟨by omega, by omega⟩, fun c hc => ?_⟩
    have := ih.2 c hc
    omega
termination_by rem => rem
decreasing_by
  have : 2 ^ (rem + 1).log2 ≤ rem + 1 := Nat.log2_self_le (Nat.succ_ne_zero rem)
  have := Nat.two_pow_pos (Nat.log2 (rem + 1))
  omega

/-! ## Signs and the permutation -/

/-- `tmp[i] = in[i] * s1[i]` with `s1[i] = +-1` keeps inner products. -/
theorem signs_dot (n : ℕ) (s x y : ℕ → α) (hs : ∀ i, s i * s i = 1) :
    ∑ i ∈ range n, (x i * s i) * (y i * s i) = ∑ i ∈ range n, x i * y i :=
  sum_congr rfl fun i _ => by
    calc (x i * s i) * (y i * s i) = x i * y i * (s i * s i) := by ring
      _ = x i * y i := by rw [hs i, mul_one]

/-- `out[i] = tmp[perm[i]]` through a permutation keeps inner products. -/
theorem perm_dot (n : ℕ) (σ : Equiv.Perm (Fin n)) (x y : Fin n → α) :
    ∑ i, x (σ i) * y (σ i) = ∑ i, x i * y i :=
  Equiv.sum_comp σ (fun i => x i * y i)

/-- One step of the shuffle of `tq_build`, `x = perm[i]; perm[i] = perm[k]; perm[k] = x;`, is composition with the
    transposition of `i` and `k` (both below `dpad`: `k = rng() % (i + 1) <= i`). -/
def swapStep {n : ℕ} (p : Equiv.Perm (Fin n)) (i k : Fin n) : Equiv.Perm (Fin n) := (Equiv.swap i k).trans p

theorem swapStep_array {n : ℕ} (p : Equiv.Perm (Fin n)) (i k : Fin n) :
    swapStep p i k i = p k ∧ swapStep p i k k = p i ∧ ∀ j, j ≠ i → j ≠ k → swapStep p i k j = p j := by
  refine ⟨by simp [swapStep], by simp [swapStep], fun j hi hk => ?_⟩
  simp only [swapStep, Equiv.trans_apply, Equiv.swap_apply_of_ne_of_ne hi hk]

/-- The whole shuffle, for any sequence of steps the random numbers pick: still a permutation of `0 .. dpad - 1`,
    so `out[i] = tmp[perm[i]]` reads every coordinate exactly once. -/
def shuffle {n : ℕ} (steps : List (Fin n × Fin n)) : Equiv.Perm (Fin n) :=
  steps.foldl (fun p st => swapStep p st.1 st.2) (Equiv.refl _)

theorem shuffle_bijective {n : ℕ} (steps : List (Fin n × Fin n)) : Function.Bijective (shuffle steps) :=
  (shuffle steps).bijective

end Vecindex.Hadamard
