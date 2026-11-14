/-
  The arithmetic of `gf_mul_` / `gf_inv_` on bytes: every field law, from the facts about the tables in
  `Tables.lean` and one symbolic fact, that a turn of the table loop (`xt`) is linear over xor.

  The key step: for `a ≠ 0`, `gf_mul_(a, b)` is `xt` applied `gf_log_[a]` times to `b`. Commutativity and
  associativity then come from adding logarithms, distributivity from the linearity of `xt`.
-/
import Vecindex.GF256.Tables

namespace Vecindex.GF256

/-! ## Iterates -/

theorem expList_getD (i : Nat) (h : i < 255) : expList.getD i 0 = iter xt i 1 :=
  iterList_getD xt 255 1 i h

/-- The generator's powers repeat with period 255. -/
theorem iter_xt_mod (n : Nat) : iter xt n 1 = iter xt (n % 255) 1 := by
  induction n using Nat.strongRecOn with
  | _ n ih =>
    by_cases h : n < 255
    · rw [Nat.mod_eq_of_lt h]
    · have h1 : n = (n - 255) + 255 := by omega
      have h2 : n % 255 = (n - 255) % 255 := by omega
      have h3 : iter xt n 1 = iter xt (n - 255) 1 := by
        conv => lhs; rw [h1]
        rw [iter_add, xt_iter_255]
      rw [h3, h2]
      exact ih (n - 255) (by omega)

/-! ## The loop step is linear -/

theorem xt_zero : xt 0 = 0 := by decide

theorem xt_form : ∀ x, x < 256 → xt x = ((x <<< 1) % 256) ^^^ (if x.testBit 7 = true then 0x1d else 0) := by
  decide +kernel

theorem xor_lt (a b : Nat) (ha : a < 256) (hb : b < 256) : a ^^^ b < 256 :=
  Nat.xor_lt_two_pow (n := 8) ha hb

theorem xor_swap (a b c d : Nat) : (a ^^^ b) ^^^ (c ^^^ d) = (a ^^^ c) ^^^ (b ^^^ d) := by
  rw [Nat.xor_assoc, ← Nat.xor_assoc b c d, Nat.xor_comm b c, Nat.xor_assoc c b d, ← Nat.xor_assoc]

theorem xt_xor (a b : Nat) (ha : a < 256) (hb : b < 256) : xt (a ^^^ b) = xt a ^^^ xt b := by
  rw [xt_form _ (xor_lt a b ha hb), xt_form a ha, xt_form b hb, Nat.shiftLeft_xor_distrib,
    show (256 : Nat) = 2 ^ 8 from rfl, Nat.xor_mod_two_pow, Nat.testBit_xor, xor_swap]
  congr 1
  cases a.testBit 7 <;> cases b.testBit 7 <;> decide

theorem iter_xt_lt : ∀ n x, x < 256 → iter xt n x < 256
  | 0, _, h => h
  | n + 1, x, h => iter_xt_lt n (xt x) (xt_lt x h)

theorem iter_xt_zero : ∀ n, iter xt n 0 = 0
  | 0 => rfl
  | n + 1 => by show iter xt n (xt 0) = 0; rw [xt_zero]; exact iter_xt_zero n

theorem iter_xt_xor : ∀ n a b, a < 256 → b < 256 → iter xt n (a ^^^ b) = iter xt n a ^^^ iter xt n b
  | 0, _, _, _, _ => rfl
  | n + 1, a, b, ha, hb => by
    show iter xt n (xt (a ^^^ b)) = iter xt n (xt a) ^^^ iter xt n (xt b)
    rw [xt_xor a b ha hb]
    exact iter_xt_xor n (xt a) (xt b) (xt_lt a ha) (xt_lt b hb)

/-! ## Logarithms -/

theorem log_eq (a : Nat) : log a = logLit.getD a 0 := by
  unfold log; rw [logTab_eq]

theorem log_lt (a : Nat) (ha : a < 256) : log a < 255 := by
  rw [log_eq]; exact logLit_lt a ha

theorem exp_lt512 (i : Nat) (h : i < 512) : exp i = iter xt (i % 255) 1 := by
  unfold exp; rw [expTab_getD i h, expList_getD _ (Nat.mod_lt _ (by decide))]

/-- `a = 2 ^ gf_log_[a]` for every nonzero byte. -/
theorem iter_log (a : Nat) (ha : a < 256) (h0 : a ≠ 0) : iter xt (log a) 1 = a := by
  rw [← expList_getD _ (log_lt a ha), log_eq]; exact expList_logLit a ha h0

/-- `gf_log_[2 ^ i] = i`. -/
theorem log_iter (i : Nat) (hi : i < 255) : log (iter xt i 1) = i := by
  rw [log_eq, ← expList_getD i hi]; exact logLit_expList i hi

theorem iter_one_lt (i : Nat) : iter xt i 1 < 256 := iter_xt_lt i 1 (by decide)

theorem iter_one_ne_zero (i : Nat) : iter xt i 1 ≠ 0 := by
  rw [iter_xt_mod, ← expList_getD _ (Nat.mod_lt _ (by decide))]
  exact expList_ne_zero _ (Nat.mod_lt _ (by decide))

/-- Two powers of the generator agree exactly when their exponents agree mod 255. -/
theorem iter_one_inj (i j : Nat) (h : iter xt i 1 = iter xt j 1) : i % 255 = j % 255 := by
  rw [iter_xt_mod i, iter_xt_mod j] at h
  have := congrArg log h
  rwa [log_iter _ (Nat.mod_lt _ (by decide)), log_iter _ (Nat.mod_lt _ (by decide))] at this

/-! ## Multiplication is a power of `xt` -/

/-- For `a ≠ 0`, `gf_mul_(a, b)` is `xt` applied `gf_log_[a]` times to `b`. -/
theorem mul_eq_iter (a b : Nat) (ha : a < 256) (ha0 : a ≠ 0) (hb : b < 256) : mul a b = iter xt (log a) b := by
  unfold mul
  by_cases hb0 : b = 0
  · subst hb0; simp [iter_xt_zero]
  · rw [if_pos ⟨ha0, hb0⟩, exp_lt512 _ (by have := log_lt a ha; have := log_lt b hb; omega),
      ← iter_xt_mod, iter_add, iter_log b hb hb0]

theorem mul_lt (a b : Nat) (hb : b < 256) (ha : a < 256) : mul a b < 256 := by
  by_cases ha0 : a = 0
  · simp [mul, ha0]
  · rw [mul_eq_iter a b ha ha0 hb]; exact iter_xt_lt _ _ hb

theorem mul_zero (a : Nat) : mul a 0 = 0 := by simp [mul]

theorem zero_mul (b : Nat) : mul 0 b = 0 := by simp [mul]

theorem mul_comm (a b : Nat) : mul a b = mul b a := by
  unfold mul
  by_cases h : a ≠ 0 ∧ b ≠ 0
  · rw [if_pos h, if_pos ⟨h.2, h.1⟩, Nat.add_comm]
  · rw [if_neg h, if_neg (fun h' => h ⟨h'.2, h'.1⟩)]

theorem one_mul (b : Nat) (hb : b < 256) : mul 1 b = b := by
  rw [mul_eq_iter 1 b (by decide) (by decide) hb, log_eq, logLit_one]; rfl

theorem mul_one (a : Nat) (ha : a < 256) : mul a 1 = a := by
  rw [mul_comm]; exact one_mul a ha

theorem mul_ne_zero (a b : Nat) (ha : a < 256) (hb : b < 256) (ha0 : a ≠ 0) (hb0 : b ≠ 0) : mul a b ≠ 0 := by
  rw [mul_eq_iter a b ha ha0 hb, ← iter_log b hb hb0, ← iter_add]; exact iter_one_ne_zero _

theorem mul_assoc (a b c : Nat) (ha : a < 256) (hb : b < 256) (hc : c < 256) :
    mul (mul a b) c = mul a (mul b c) := by
  by_cases ha0 : a = 0
  · subst ha0; simp [zero_mul]
  by_cases hb0 : b = 0
  · subst hb0; simp [zero_mul, mul_zero]
  by_cases hc0 : c = 0
  · subst hc0; simp [mul_zero]
  have hab := mul_lt a b hb ha
  rw [mul_comm (mul a b) c, mul_eq_iter c _ hc hc0 hab, mul_eq_iter a b ha ha0 hb, ← iter_add,
    mul_comm b c, mul_eq_iter a _ ha ha0 (mul_lt c b hb hc), mul_eq_iter c b hc hc0 hb, ← iter_add,
    Nat.add_comm]

theorem mul_xor (a b c : Nat) (ha : a < 256) (hb : b < 256) (hc : c < 256) :
    mul a (b ^^^ c) = mul a b ^^^ mul a c := by
  by_cases ha0 : a = 0
  · subst ha0; simp [zero_mul]
  rw [mul_eq_iter a _ ha ha0 (xor_lt b c hb hc), mul_eq_iter a b ha ha0 hb, mul_eq_iter a c ha ha0 hc]
  exact iter_xt_xor _ b c hb hc

/-! ## Inverses -/

theorem inv_eq (a : Nat) : inv a = iter xt ((255 - log a) % 255) 1 := by
  unfold inv; rw [exp_lt512 _ (by omega)]

theorem inv_lt (a : Nat) : inv a < 256 := by rw [inv_eq a]; exact iter_one_lt _

theorem inv_ne_zero (a : Nat) : inv a ≠ 0 := by rw [inv_eq a]; exact iter_one_ne_zero _

theorem mul_inv (a : Nat) (ha : a < 256) (ha0 : a ≠ 0) : mul a (inv a) = 1 := by
  rw [mul_eq_iter a _ ha ha0 (inv_lt a), inv_eq a, ← iter_add, iter_xt_mod]
  have hl := log_lt a ha
  have : (log a + (255 - log a) % 255) % 255 = 0 := by omega
  rw [this]; rfl

theorem inv_mul (a : Nat) (ha : a < 256) (ha0 : a ≠ 0) : mul (inv a) a = 1 := by
  rw [mul_comm]; exact mul_inv a ha ha0

/-- `gf_coef_` is never zero (the inverse of a byte is never zero; `gf_inv_(0)` is 1). -/
theorem coef_ne_zero (k p i : Nat) : coef k p i ≠ 0 :=
  inv_ne_zero _

end Vecindex.GF256
