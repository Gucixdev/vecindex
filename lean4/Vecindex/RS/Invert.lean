/-
  `gf_invert_` of `src/store_sum.c`, over any field: Gauss-Jordan elimination with the first nonzero pivot.

      for (c = 0; c < n; ++c) {
          for (r = c; r < n && !a[r * n + c]; ++r) {}
          if (r == n) return -1;
          if (r != c) swap rows r and c of a and of inv;
          f = gf_inv_(a[c * n + c]);
          row c of a and of inv times f;
          for (r = 0; r < n; ++r) {
              if (r == c || !(f = a[r * n + c])) continue;
              row r of a ^= f * row c of a; row r of inv ^= f * row c of inv;
          }
      }
      return 0;

  `^=` is subtraction in GF(2^8). Each row update reads row r and row c only, and row c does not change while the
  other rows are updated, so the loop over r is one simultaneous step here.

  Proved: when it returns 0, `inv * a = 1` (`invert_mul`); it returns -1 only for a matrix with a nonzero kernel
  vector (`invert_none`). Only the left inverse is needed by the rebuild: `x = inv * (a * d) = d`.
-/
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Field.Basic
import Mathlib.Logic.Equiv.Basic
import Mathlib.Tactic.Ring

open Finset

namespace Vecindex.Gauss

set_option linter.unusedSectionVars false

variable {F : Type*} [Field F] [DecidableEq F] {n : ℕ}

/-- An `n x n` matrix, row-major as in C: `M i j` is `a[i * n + j]`. -/
abbrev Mat (n : ℕ) (F : Type*) := Fin n → Fin n → F

def mul (A B : Mat n F) : Mat n F := fun i j => ∑ k, A i k * B k j

def app (A : Mat n F) (v : Fin n → F) : Fin n → F := fun i => ∑ k, A i k * v k

def one : Mat n F := fun i j => if i = j then 1 else 0

/-- Rows `r` and `c` exchanged (nothing happens for `r = c`). -/
def swap (M : Mat n F) (r c : Fin n) : Mat n F := fun i => M (Equiv.swap r c i)

/-- Row `c` times `f`. -/
def scale (M : Mat n F) (c : Fin n) (f : F) : Mat n F := fun i j => if i = c then M i j * f else M i j

/-- Every row `i ≠ c` with `g i ≠ 0` minus `g i` times row `c`. -/
def elim (M : Mat n F) (c : Fin n) (g : Fin n → F) : Mat n F :=
  fun i j => if i = c ∨ g i = 0 then M i j else M i j - g i * M c j

/-! ## The row operations act the same on `inv` and on `inv * A` -/

theorem mul_swap (B A : Mat n F) (r c : Fin n) : mul (swap B r c) A = swap (mul B A) r c := rfl

theorem mul_scale (B A : Mat n F) (c : Fin n) (f : F) : mul (scale B c f) A = scale (mul B A) c f := by
  funext i j
  by_cases h : i = c
  · simp only [mul, scale, h, if_true, sum_mul]; exact sum_congr rfl fun k _ => by ring
  · simp [mul, scale, h]

theorem mul_elim (B A : Mat n F) (c : Fin n) (g : Fin n → F) : mul (elim B c g) A = elim (mul B A) c g := by
  funext i j
  by_cases h : i = c ∨ g i = 0
  · simp only [mul, elim, h, if_true]
  · simp only [mul, elim, if_neg h, mul_sum, ← sum_sub_distrib]
    exact sum_congr rfl fun k _ => by ring

/-! ## ... and they lose no kernel vector -/

theorem app_swap (M : Mat n F) (r c : Fin n) (v : Fin n → F) (h : app (swap M r c) v = 0) : app M v = 0 := by
  funext i
  have := congrFun h (Equiv.swap r c i)
  simpa [app, swap, Equiv.swap_apply_self] using this

theorem app_scale (M : Mat n F) (c : Fin n) (f : F) (hf : f ≠ 0) (v : Fin n → F)
    (h : app (scale M c f) v = 0) : app M v = 0 := by
  funext i
  have := congrFun h i
  by_cases hi : i = c
  · subst hi
    have h1 : ∑ x, M i x * f * v x = 0 := by simpa [app, scale] using this
    have h2 : (∑ x, M i x * v x) * f = 0 := by
      rw [sum_mul, ← h1]; exact sum_congr rfl fun x _ => by ring
    exact (mul_eq_zero.mp h2).resolve_right hf
  · simpa [app, scale, hi] using this

theorem app_elim (M : Mat n F) (c : Fin n) (g : Fin n → F) (v : Fin n → F)
    (h : app (elim M c g) v = 0) : app M v = 0 := by
  have hc : app M v c = 0 := by
    have := congrFun h c
    simpa [app, elim] using this
  funext i
  have := congrFun h i
  by_cases hi : i = c ∨ g i = 0
  · simpa [app, elim, hi] using this
  · simp only [app, elim, if_neg hi, Pi.zero_apply] at this
    have hc' : ∑ k, M c k * v k = 0 := hc
    have e : ∑ k, (M i k - g i * M c k) * v k = ∑ k, M i k * v k - g i * ∑ k, M c k * v k := by
      rw [mul_sum, ← sum_sub_distrib]; exact sum_congr rfl fun k _ => by ring
    rw [e, hc', mul_zero, sub_zero] at this
    exact this

/-- `for (r = c; r < n && !a[r * n + c]; ++r) {}` -/
def pivot (a : Mat n F) (c : Fin n) : Option (Fin n) :=
  (List.finRange n).find? (fun r => decide (c ≤ r) && decide (a r c ≠ 0))

/-- One turn of the outer loop: `none` is `return -1`. -/
def step (c : Fin n) (s : Mat n F × Mat n F) : Option (Mat n F × Mat n F) :=
  match pivot s.1 c with
  | none => none
  | some r =>
    let a2 := scale (swap s.1 r c) c (swap s.1 r c c c)⁻¹
    let i2 := scale (swap s.2 r c) c (swap s.1 r c c c)⁻¹
    some (elim a2 c (fun i => a2 i c), elim i2 c (fun i => a2 i c))

/-- The state after the first `c` columns. -/
def run (A : Mat n F) : ℕ → Option (Mat n F × Mat n F)
  | 0 => some (A, one)
  | c + 1 => (run A c).bind fun s => if h : c < n then step ⟨c, h⟩ s else some s

/-- `gf_invert_`: `some inv` for `return 0`, `none` for `return -1`. -/
def invert (A : Mat n F) : Option (Mat n F) := (run A n).map Prod.snd

/-! ## Pivot -/

theorem pivot_some (a : Mat n F) (c r : Fin n) (h : pivot a c = some r) : c ≤ r ∧ a r c ≠ 0 := by
  have := List.find?_some h
  simpa using this

theorem pivot_none (a : Mat n F) (c : Fin n) (h : pivot a c = none) : ∀ r : Fin n, c ≤ r → a r c = 0 := by
  intro r hr
  have := List.find?_eq_none.mp h r (List.mem_finRange r)
  by_contra h0
  exact this (by simp [hr, h0])

/-! ## The invariant of the outer loop -/

/-- After `c` columns: `inv * A = a`, every kernel vector of `a` is one of `A`, and columns below `c` of `a` are
    those of the identity. -/
structure Inv (A : Mat n F) (c : ℕ) (s : Mat n F × Mat n F) : Prop where
  left : mul s.2 A = s.1
  ker : ∀ v, app s.1 v = 0 → app A v = 0
  cols : ∀ j : Fin n, j.val < c → ∀ i, s.1 i j = one i j

theorem one_mul' (A : Mat n F) : mul one A = A := by
  funext i j; simp [mul, one]

theorem inv_zero (A : Mat n F) : Inv A 0 (A, one) where
  left := one_mul' A
  ker := fun _ h => h
  cols := fun _ h => absurd h (Nat.not_lt_zero _)

theorem step_inv (A : Mat n F) (c : Fin n) (s t : Mat n F × Mat n F) (hs : Inv A c.val s)
    (ht : step c s = some t) : Inv A (c.val + 1) t := by
  unfold step at ht
  cases hp : pivot s.1 c with
  | none => rw [hp] at ht; exact absurd ht (by simp)
  | some r =>
    rw [hp] at ht
    simp only [Option.some.injEq] at ht
    subst ht
    obtain ⟨hcr, hrc⟩ := pivot_some s.1 c r hp
    have hpiv : swap s.1 r c c c = s.1 r c := by simp [swap, Equiv.swap_apply_right]
    have hf : (swap s.1 r c c c)⁻¹ ≠ 0 := inv_ne_zero (by rw [hpiv]; exact hrc)
    refine ⟨?_, ?_, ?_⟩
    · dsimp only
      rw [mul_elim, mul_scale, mul_swap, hs.left]
    · intro v hv
      exact hs.ker v (app_swap _ r c v (app_scale _ c _ hf v (app_elim _ c _ v hv)))
    · intro j hj i
      -- the columns below c stay unit columns, and column c becomes one
      have hsw : ∀ j : Fin n, j.val < c.val → ∀ i, swap s.1 r c i j = one i j := by
        intro j hj i
        have hjr : j ≠ r := fun h => by
          have h1 := Fin.le_iff_val_le_val.mp hcr
          rw [← h] at h1; omega
        have hjc : j ≠ c := fun h => by rw [h] at hj; exact absurd hj (lt_irrefl _)
        simp only [swap, one]
        rw [hs.cols j hj]
        simp only [one]
        by_cases hij : i = j
        · subst hij; rw [Equiv.swap_apply_of_ne_of_ne hjr hjc]
        · rw [if_neg hij, if_neg]
          intro h
          apply hij
          rw [← h, Equiv.swap_apply_of_ne_of_ne] <;>
            · intro h'
              rw [h'] at h
              first
                | exact hjc (by rw [← h, Equiv.swap_apply_left])
                | exact hjr (by rw [← h, Equiv.swap_apply_right])
      have hcc : scale (swap s.1 r c) c (swap s.1 r c c c)⁻¹ c c = 1 := by
        simp only [scale, if_true]; exact mul_inv_cancel₀ (by rw [hpiv]; exact hrc)
      have hcj : ∀ j : Fin n, j.val < c.val → scale (swap s.1 r c) c (swap s.1 r c c c)⁻¹ c j = 0 := by
        intro j hj
        have hjc : j ≠ c := fun h => by rw [h] at hj; exact absurd hj (lt_irrefl _)
        simp only [scale, if_true]; rw [hsw j hj c]; simp [one, Ne.symm hjc]
      dsimp only
      simp only [elim]
      by_cases hjc : j.val < c.val
      · -- an old unit column
        have hsc : scale (swap s.1 r c) c (swap s.1 r c c c)⁻¹ i j = one i j := by
          by_cases hic : i = c
          · subst hic
            rw [hcj j hjc]; simp only [one]
            rw [if_neg]
            intro h; rw [h] at hjc; exact lt_irrefl _ hjc
          · simp only [scale, if_neg hic]; exact hsw j hjc i
        split_ifs with h
        · exact hsc
        · rw [hsc, hcj j hjc, mul_zero, sub_zero]
      · -- column c
        have hjc' : j = c := by apply Fin.ext; omega
        subst hjc'
        by_cases hic : i = j
        · subst hic; rw [if_pos (Or.inl rfl), hcc]; simp [one]
        · split_ifs with h
          · rcases h with h | h
            · exact absurd h hic
            · rw [h]; simp [one, hic]
          · rw [hcc, mul_one, sub_self]; simp [one, hic]

theorem run_inv (A : Mat n F) : ∀ c s, run A c = some s → Inv A (min c n) s
  | 0, s, h => by
    simp only [run, Option.some.injEq] at h; subst h; simpa using inv_zero A
  | c + 1, s, h => by
    simp only [run] at h
    cases hr : run A c with
    | none => rw [hr] at h; exact absurd h (by simp)
    | some s0 =>
      rw [hr] at h
      simp only [Option.bind_some] at h
      have h0 := run_inv A c s0 hr
      by_cases hc : c < n
      · rw [dif_pos hc] at h
        have := step_inv A ⟨c, hc⟩ s0 s (by simpa [Nat.min_eq_left (Nat.le_of_lt hc)] using h0) h
        simpa [Nat.min_eq_left (Nat.succ_le_of_lt hc)] using this
      · rw [dif_neg hc] at h
        simp only [Option.some.injEq] at h
        subst h
        simpa [Nat.min_eq_right (by omega : n ≤ c), Nat.min_eq_right (by omega : n ≤ c + 1)] using h0

/-- When `gf_invert_` returns 0, `inv` is a left inverse of `a`. -/
theorem invert_mul (A B : Mat n F) (h : invert A = some B) : mul B A = one := by
  unfold invert at h
  cases hr : run A n with
  | none => rw [hr] at h; exact absurd h (by simp)
  | some s =>
    rw [hr] at h
    simp only [Option.map_some, Option.some.injEq] at h
    subst h
    have hi := run_inv A n s hr
    rw [Nat.min_self] at hi
    rw [hi.left]
    funext i j
    exact hi.cols j j.isLt i

/-- A failed pivot search at column `c` exhibits a nonzero kernel vector. -/
theorem kernel_of_stuck (a : Mat n F) (c : Fin n) (hcols : ∀ j : Fin n, j.val < c.val → ∀ i, a i j = one i j)
    (hz : ∀ r : Fin n, c ≤ r → a r c = 0) : ∃ v : Fin n → F, v ≠ 0 ∧ app a v = 0 := by
  let v : Fin n → F := fun k => if k = c then 1 else if k.val < c.val then -a k c else 0
  refine ⟨v, fun h => by have := congrFun h c; simp [v] at this, ?_⟩
  funext i
  have hterm : ∀ k, a i k * v k = (if k = c then a i c else 0) + (if k = i ∧ i.val < c.val then -a i c else 0) := by
    intro k
    by_cases hkc : k = c
    · subst hkc
      have hn : ¬ (k = i ∧ i.val < k.val) := fun ⟨h1, h2⟩ => by rw [h1] at h2; exact lt_irrefl _ h2
      simp only [v, if_pos rfl, mul_one]
      rw [if_neg hn, add_zero, if_pos trivial]
    · by_cases hk : k.val < c.val
      · simp only [v, if_neg hkc, if_pos hk, hcols k hk i, one]
        by_cases hik : i = k
        · subst hik; simp [hk]
        · simp [hik, Ne.symm hik]
      · simp only [v, if_neg hkc, if_neg hk, mul_zero, zero_add]
        rw [if_neg]
        rintro ⟨rfl, hi⟩; exact hk hi
  simp only [app]
  rw [sum_congr rfl fun k _ => hterm k, sum_add_distrib, sum_ite_eq', if_pos (mem_univ _)]
  rw [show (∑ k : Fin n, if k = i ∧ i.val < c.val then -a i c else 0) = if i.val < c.val then -a i c else 0 by
    by_cases hi : i.val < c.val
    · simp [hi]
    · simp [hi]]
  by_cases hi : i.val < c.val
  · rw [if_pos hi, add_neg_cancel]; rfl
  · rw [if_neg hi, add_zero]; exact hz i (Fin.le_iff_val_le_val.mpr (by omega))

theorem run_none (A : Mat n F) : ∀ c, c ≤ n → run A c = none → ∃ v : Fin n → F, v ≠ 0 ∧ app A v = 0
  | 0, _, h => by simp [run] at h
  | c + 1, hc, h => by
    simp only [run] at h
    cases hr : run A c with
    | none => exact run_none A c (by omega) hr
    | some s =>
      rw [hr] at h
      simp only [Option.bind_some, dif_pos (show c < n by omega)] at h
      have hs := run_inv A c s hr
      rw [Nat.min_eq_left (by omega)] at hs
      unfold step at h
      cases hp : pivot s.1 ⟨c, by omega⟩ with
      | some r => rw [hp] at h; exact absurd h (by simp)
      | none =>
        obtain ⟨v, hv, hav⟩ := kernel_of_stuck s.1 ⟨c, by omega⟩ hs.cols (pivot_none s.1 _ hp)
        exact ⟨v, hv, hs.ker v hav⟩

/-- `gf_invert_` returns -1 only for a matrix with a nonzero kernel vector. -/
theorem invert_none (A : Mat n F) (h : invert A = none) : ∃ v : Fin n → F, v ≠ 0 ∧ app A v = 0 := by
  unfold invert at h
  cases hr : run A n with
  | none => exact run_none A n le_rfl hr
  | some s => rw [hr] at h; exact absurd h (by simp)

/-- For a matrix with no kernel, `gf_invert_` succeeds and returns a left inverse. -/
theorem invert_ok (A : Mat n F) (hA : ∀ v, app A v = 0 → v = 0) : ∃ B, invert A = some B ∧ mul B A = one := by
  cases h : invert A with
  | none =>
    obtain ⟨v, hv, hav⟩ := invert_none A h
    exact absurd (hA v hav) hv
  | some B => exact ⟨B, rfl, invert_mul A B h⟩

end Vecindex.Gauss
