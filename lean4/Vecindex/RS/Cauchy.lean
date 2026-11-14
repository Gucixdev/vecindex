/-
  Cauchy matrices have no kernel. `store_sum.c` takes its parity coefficients from the Cauchy matrix
  `c(p, i) = 1 / ((k + p) xor i)`; every square part of it - any set of intact parity rows against as many damaged
  data columns - must be invertible for the rebuild to work. This is the statement for any field.

  Proof: if `∑ j, v j / (x i - y j) = 0` for every `i`, the polynomial `P t = ∑ j, v j ∏_{l ≠ j} (t - y l)` has
  degree below `n` and the `n` distinct roots `x i`, so it is zero; evaluated at `y j` it is
  `v j ∏_{l ≠ j} (y j - y l)`, so `v j = 0`.
-/
import Mathlib.Algebra.Polynomial.Roots
import Mathlib.Algebra.Polynomial.BigOperators
import Mathlib.Tactic.Ring

open Polynomial Finset

namespace Vecindex

variable {F : Type*} [Field F]

theorem cauchy_kernel {n : ℕ} (x y : Fin n → F) (hx : Function.Injective x) (hy : Function.Injective y)
    (hxy : ∀ i j, x i ≠ y j) (v : Fin n → F) (hv : ∀ i, ∑ j, v j * (x i - y j)⁻¹ = 0) : v = 0 := by
  classical
  rcases Nat.eq_zero_or_pos n with hn | hn
  · subst hn; funext j; exact j.elim0
  set P : F[X] := ∑ j, C (v j) * ∏ l ∈ univ.erase j, (X - C (y l)) with hPdef
  have hdeg : P.natDegree < n := by
    have : P.natDegree ≤ n - 1 := by
      rw [hPdef]
      apply natDegree_sum_le_of_forall_le
      intro j _
      refine (natDegree_C_mul_le _ _).trans ?_
      refine (natDegree_prod_le _ _).trans ?_
      calc ∑ l ∈ univ.erase j, (X - C (y l)).natDegree
          ≤ ∑ _l ∈ univ.erase j, (1 : ℕ) :=
            sum_le_sum (f := fun l => (X - C (y l)).natDegree) (g := fun _ => 1)
              (fun l _ => natDegree_X_sub_C_le (y l))
        _ = n - 1 := by simp [card_erase_of_mem (mem_univ j)]
    omega
  have hroot : ∀ i, P.eval (x i) = 0 := by
    intro i
    have hne : ∀ l, x i - y l ≠ 0 := fun l => sub_ne_zero.mpr (hxy i l)
    have hsplit : ∀ j, ∏ l ∈ univ.erase j, (x i - y l) = (∏ l, (x i - y l)) * (x i - y j)⁻¹ := by
      intro j
      have h := mul_prod_erase univ (fun l => x i - y l) (mem_univ j)
      simp only at h
      rw [← h, mul_comm (x i - y j), mul_assoc, mul_inv_cancel₀ (hne j), mul_one]
    simp only [hPdef, eval_finset_sum, eval_mul, eval_C, eval_prod, eval_sub, eval_X]
    simp_rw [hsplit]
    calc ∑ j, v j * ((∏ l, (x i - y l)) * (x i - y j)⁻¹)
        = (∏ l, (x i - y l)) * ∑ j, v j * (x i - y j)⁻¹ := by
          rw [mul_sum]; refine sum_congr rfl fun j _ => ?_; ring
      _ = 0 := by rw [hv i, mul_zero]
  have hP0 : P = 0 := eq_zero_of_natDegree_lt_card_of_eval_eq_zero P hx hroot (by simpa using hdeg)
  funext j
  have hj := congrArg (Polynomial.eval (y j)) hP0
  simp only [hPdef, eval_finset_sum, eval_mul, eval_C, eval_prod, eval_sub, eval_X, eval_zero] at hj
  rw [sum_eq_single j] at hj
  · have hprod : ∏ l ∈ univ.erase j, (y j - y l) ≠ 0 := by
      rw [prod_ne_zero_iff]
      intro l hl
      exact sub_ne_zero.mpr fun h => (ne_of_mem_erase hl) (hy h).symm
    exact (mul_eq_zero.mp hj).resolve_right hprod
  · intro j' _ hj'
    have : ∏ l ∈ univ.erase j', (y j - y l) = 0 :=
      prod_eq_zero (mem_erase.mpr ⟨fun h => hj' h.symm, mem_univ j⟩) (sub_self _)
    rw [this, mul_zero]
  · intro h; exact absurd (mem_univ j) h

end Vecindex
