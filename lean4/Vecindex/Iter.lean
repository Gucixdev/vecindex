/-
  `f` applied `n` times, and the list of the first `n` iterates: the loops `for (i = 0; i < n; ++i) x = f(x);` of
  the C code.
-/

namespace Vecindex

/-- `f` applied `n` times. -/
def iter (f : Nat → Nat) : Nat → Nat → Nat
  | 0, x => x
  | n + 1, x => iter f n (f x)

/-- `[x, f x, f (f x), ...]`, `n` entries. -/
def iterList (f : Nat → Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | n + 1, x => x :: iterList f n (f x)

theorem iter_comm (f : Nat → Nat) : ∀ m x, iter f m (f x) = f (iter f m x)
  | 0, _ => rfl
  | m + 1, x => iter_comm f m (f x)

theorem iter_succ' (f : Nat → Nat) (n x : Nat) : iter f (n + 1) x = f (iter f n x) := iter_comm f n x

theorem iter_add (f : Nat → Nat) : ∀ m n x, iter f (m + n) x = iter f m (iter f n x)
  | _, 0, _ => rfl
  | m, n + 1, x => by
    rw [← Nat.add_assoc, iter_succ', iter_succ', iter_add f m n x, iter_comm]

theorem iterList_getD (f : Nat → Nat) : ∀ n x i, i < n → (iterList f n x).getD i 0 = iter f i x
  | 0, _, _, h => absurd h (Nat.not_lt_zero _)
  | _ + 1, _, 0, _ => rfl
  | n + 1, x, i + 1, h => iterList_getD f n (f x) i (by omega)

/-- A property kept by every step is kept by the iterates. -/
theorem iter_induct (f : Nat → Nat) (P : Nat → Prop) (h : ∀ x, P x → P (f x)) : ∀ n x, P x → P (iter f n x)
  | 0, _, hx => hx
  | n + 1, x, hx => iter_induct f P h n (f x) (h x hx)

end Vecindex
