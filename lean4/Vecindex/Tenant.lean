/-
  The ids of a tenant in one store, `src/vserv_keys.c` and `src/vserv_proto.c`. A key of `-K` names a tenant, a
  number 1 to 524287; its rows live in the one store with the tenant above bit 44 of the id:

      vs_in_   (wire id -> store id)   x in [0, 2^44): x | tn << 44, else refused; tn = 0 (no tenant): x as it is
      vs_out_  (store id -> wire id)   y & (2^44 - 1)
      vs_rin_  (a range of wire ids)   [lo, hi] clipped to [0, 2^44), both ends moved up; refused when nothing is left
      own      (the hybrid post-filter in vs_hybrid_)   keep y when y >> 44 == tn, as y & (2^44 - 1)
      the tenant number in the key file: `for (i = 0; i < tl && isdigit(t[i]) && n <= TMAXN; ++i) n = n * 10 + d`,
      then refused unless every character was read and 1 <= n <= TMAXN

  Proved: a wire id maps to exactly one store id and back (`out_in`), no two tenants share a store id (`in_inj`),
  store ids stay positive and inside i64 (`in_fits`), a tenant's range covers exactly its own ids in the range and
  nobody else's (`rin_exact`, `rin_none`), the post-filter keeps exactly the tenant's own rows (`own_iff`), and the
  key file's number is accepted exactly when it is a decimal number in [1, 524287] (`parse_iff`), with no
  intermediate value past 5242879 (`parse_bound`: no overflow of the C's i64).

  Ids are `Int` (the C's i64); the bit operations are those of the C code, on the natural numbers they act on, and
  `in_or` shows that the OR is the sum. `vs_out_` is only ever applied to ids of the tenant's own namespace, so it is
  modelled for ids >= 0. `y >> 44` of a negative i64 is gcc's arithmetic shift, a floor division: `Int./` here.
-/
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

namespace Vecindex.Tenant

set_option linter.dupNamespace false

/-- 2^44, the first id that is not a tenant's. -/
def B : Int := 17592186044416

theorem B_eq : B = 2 ^ 44 := by decide

/-- `VS_TMASK`. -/
def mask : Int := B - 1

/-- `VS_TMAXN`: the last tenant number, so that tenant << 44 stays inside an i64. -/
def tmax : Nat := 524287

theorem tmax_eq : (tmax : Int) = 2 ^ 19 - 1 := by decide

/-- `vs_in_`: `none` is the C's -1 (an id outside [0, 2^44)). -/
def inId (tn x : Int) : Option Int :=
  if tn = 0 then some x
  else if x < 0 ∨ x > mask then none
  else some (Int.ofNat (x.toNat ||| (tn.toNat <<< 44)))

/-- `vs_out_` on an id of the tenant's namespace (`y >= 0`). -/
def outId (tn y : Int) : Int :=
  if tn = 0 then y else Int.ofNat (y.toNat &&& (2 ^ 44 - 1))

/-- `vs_rin_`. -/
def rin (tn lo hi : Int) : Option (Int × Int) :=
  if tn = 0 then some (lo, hi)
  else if hi < 0 ∨ lo > mask ∨ lo > hi then none
  else
    let a := max lo 0
    let b := min hi mask
    some (Int.ofNat (a.toNat ||| (tn.toNat <<< 44)), Int.ofNat (b.toNat ||| (tn.toNat <<< 44)))

/-- The post-filter of a tenant's hybrid search: `(y >> 44) == tn ? y & TMASK : dropped`. -/
def own (tn y : Int) : Option Int :=
  if y / B = tn then some (Int.ofNat (y.toNat &&& (2 ^ 44 - 1))) else none

/-- The OR of the two fields is their sum: the id's bits below 44, the tenant's above. -/
theorem in_or (tn x : Int) (ht : 0 ≤ tn) (h0 : 0 ≤ x) (h1 : x ≤ mask) :
    Int.ofNat (x.toNat ||| (tn.toNat <<< 44)) = tn * B + x := by
  have hx : x.toNat < 2 ^ 44 := by
    have : x < 2 ^ 44 := by rw [← B_eq]; unfold mask at h1; omega
    omega
  rw [Nat.or_comm, Nat.shiftLeft_eq, Nat.mul_comm, ← Nat.two_pow_add_eq_or_of_lt hx]
  rw [B_eq, Int.ofNat_eq_natCast]
  push_cast
  rw [Int.toNat_of_nonneg ht, Int.toNat_of_nonneg h0]
  ring

theorem and_mask (y : Int) (h : 0 ≤ y) : Int.ofNat (y.toNat &&& (2 ^ 44 - 1)) = y % B := by
  rw [Nat.and_two_pow_sub_one_eq_mod, B_eq, Int.ofNat_eq_natCast, Int.natCast_emod]
  push_cast
  rw [Int.toNat_of_nonneg h]

theorem in_eq (tn x : Int) (ht : 1 ≤ tn) (h0 : 0 ≤ x) (h1 : x ≤ mask) : inId tn x = some (tn * B + x) := by
  unfold inId
  rw [if_neg (by omega), if_neg (by omega), in_or tn x (by omega) h0 h1]

theorem in_none (tn x : Int) (ht : tn ≠ 0) (h : x < 0 ∨ x > mask) : inId tn x = none := by
  unfold inId; rw [if_neg ht, if_pos h]

/-- Exactly the ids in [0, 2^44) map, and what they map to. -/
theorem in_iff (tn x y : Int) (ht : 1 ≤ tn) : inId tn x = some y ↔ 0 ≤ x ∧ x ≤ mask ∧ y = tn * B + x := by
  by_cases h : x < 0 ∨ x > mask
  · rw [in_none tn x (by omega) h]; simp only [reduceCtorEq, false_iff]; omega
  · rw [in_eq tn x ht (by omega) (by omega)]
    constructor
    · intro e; cases e; exact ⟨by omega, by omega, rfl⟩
    · rintro ⟨_, _, rfl⟩; rfl

/-- Back from the store: the wire id the tenant wrote. -/
theorem out_in (tn x : Int) (ht : 1 ≤ tn) (h0 : 0 ≤ x) (h1 : x ≤ mask) : outId tn (tn * B + x) = x := by
  unfold outId
  rw [if_neg (by omega), and_mask _ (by unfold B; positivity)]
  unfold mask at h1
  rw [Int.add_comm, Int.add_mul_emod_self_right]
  exact Int.emod_eq_of_lt h0 (by omega)

/-- No two tenants share a store id; nor do two ids of one tenant. -/
theorem in_inj (t₁ t₂ x₁ x₂ y : Int) (h₁ : 1 ≤ t₁) (h₂ : 1 ≤ t₂)
    (e₁ : inId t₁ x₁ = some y) (e₂ : inId t₂ x₂ = some y) : t₁ = t₂ ∧ x₁ = x₂ := by
  rw [in_iff _ _ _ h₁] at e₁; rw [in_iff _ _ _ h₂] at e₂
  unfold mask at e₁ e₂; unfold B at e₁ e₂
  omega

/-- The store id of a key's tenant is positive and fits an i64: tenant << 44 never overflows. -/
theorem in_fits (tn x y : Int) (h1 : 1 ≤ tn) (h2 : tn ≤ tmax) (e : inId tn x = some y) : 0 < y ∧ y < 2 ^ 63 := by
  rw [in_iff _ _ _ h1] at e
  have : (2 : Int) ^ 63 = 9223372036854775808 := by norm_num
  unfold mask at e; unfold B at e; unfold tmax at h2
  omega

/-- A tenant's range delete or range listing reaches exactly the tenant's ids that are in the range. -/
theorem rin_exact (tn lo hi a b : Int) (ht : 1 ≤ tn) (e : rin tn lo hi = some (a, b)) :
    ∀ y, (a ≤ y ∧ y ≤ b) ↔ ∃ x, lo ≤ x ∧ x ≤ hi ∧ inId tn x = some y := by
  unfold rin at e
  rw [if_neg (by omega)] at e
  split at e
  · cases e
  · rename_i hn
    simp only [Option.some.injEq, Prod.mk.injEq] at e
    obtain ⟨ea, eb⟩ := e
    simp only [not_or, not_lt] at hn
    rw [in_or tn _ (by omega) (le_max_right _ _) (max_le hn.2.1 (by unfold mask B; norm_num))] at ea
    rw [in_or tn _ (by omega) (le_min hn.1 (by unfold mask B; norm_num)) (min_le_right _ _)] at eb
    intro y
    simp only [in_iff _ _ _ ht]
    constructor
    · rintro ⟨h1, h2⟩
      refine ⟨y - tn * B, ?_, ?_, ?_, ?_, by ring⟩
      · have := le_max_left lo 0; linarith
      · have := min_le_left hi mask; linarith
      · have := le_max_right lo 0; linarith
      · have := min_le_right hi mask; linarith
    · rintro ⟨x, h1, h2, h3, h4, rfl⟩
      rw [← ea, ← eb]
      exact ⟨by have := max_le h1 h3; linarith, by have := le_min h2 h4; linarith⟩

/-- A refused range holds none of the tenant's ids. -/
theorem rin_none (tn lo hi : Int) (ht : 1 ≤ tn) (e : rin tn lo hi = none) :
    ∀ x, lo ≤ x → x ≤ hi → inId tn x = none := by
  unfold rin at e
  rw [if_neg (by omega)] at e
  split at e
  · rename_i h
    intro x h1 h2
    exact in_none tn x (by omega) (by omega)
  · cases e

/-- The hybrid post-filter: a row comes back iff it is the tenant's own, under the id the tenant gave it. -/
theorem own_iff (tn y x : Int) (ht : 1 ≤ tn) : own tn y = some x ↔ inId tn x = some y := by
  rw [in_iff _ _ _ ht]
  unfold own
  have hB : (0 : Int) < B := by unfold B; norm_num
  constructor
  · intro e
    split at e
    · rename_i hq
      cases e
      have hy : 0 ≤ y := by
        have := Int.ediv_mul_le y (show B ≠ 0 by omega)
        nlinarith [Int.emod_nonneg y (show B ≠ 0 by omega), Int.mul_ediv_add_emod y B]
      rw [and_mask y hy]
      refine ⟨Int.emod_nonneg _ (by omega), ?_, ?_⟩
      · unfold mask; have := Int.emod_lt_of_pos y hB; omega
      · have := Int.mul_ediv_add_emod y B; rw [hq] at this; linarith
    · cases e
  · rintro ⟨h0, h1, rfl⟩
    unfold mask at h1
    have hq : (tn * B + x) / B = tn := by
      rw [Int.add_comm, Int.add_mul_ediv_right _ _ (by omega), Int.ediv_eq_zero_of_lt h0 (by omega)]; ring
    rw [if_pos hq, and_mask _ (by nlinarith)]
    congr 1
    rw [Int.add_comm, Int.add_mul_emod_self_right]
    exact Int.emod_eq_of_lt h0 (by omega)

/-! ## The tenant's number in the key file -/

def isDigit (c : Nat) : Bool := 48 ≤ c && c ≤ 57

/-- The loop of `vs_keys_load_`: how many characters it read and the number it built. -/
def loop : List Nat → Nat → Nat × Nat
  | [], n => (0, n)
  | c :: cs, n =>
    if isDigit c ∧ n ≤ tmax then
      let r := loop cs (n * 10 + (c - 48))
      (r.1 + 1, r.2)
    else (0, n)

/-- Accepted (`some n`) or refused, as `vs_keys_load_` decides after the loop. -/
def parse (s : List Nat) : Option Nat :=
  let r := loop s 0
  if r.1 < s.length ∨ r.2 < 1 ∨ r.2 > tmax then none else some r.2

/-- The decimal value of digits, from n. -/
def val : List Nat → Nat → Nat
  | [], n => n
  | c :: cs, n => val cs (n * 10 + (c - 48))

theorem val_ge (s : List Nat) (n : Nat) : n ≤ val s n := by
  induction s generalizing n with
  | nil => exact Nat.le_refl _
  | cons c cs ih => unfold val; exact Nat.le_trans (by omega) (ih _)

theorem loop_ok (s : List Nat) (n : Nat) (hd : ∀ c ∈ s, isDigit c) (hv : val s n ≤ tmax) :
    loop s n = (s.length, val s n) := by
  induction s generalizing n with
  | nil => rfl
  | cons c cs ih =>
    have hc : isDigit c := hd c (List.mem_cons_self ..)
    have hn : n ≤ tmax := Nat.le_trans (val_ge (c :: cs) n) hv
    unfold loop
    rw [if_pos ⟨hc, hn⟩]
    unfold val at hv
    rw [ih _ (fun x hx => hd x (List.mem_cons_of_mem _ hx)) hv]
    rfl

theorem loop_bad (s : List Nat) (n : Nat) (hn : n ≤ tmax) (h : ¬((∀ c ∈ s, isDigit c) ∧ val s n ≤ tmax)) :
    (loop s n).1 < s.length ∨ (loop s n).2 > tmax := by
  induction s generalizing n with
  | nil => exact absurd ⟨fun c hc => absurd hc List.not_mem_nil, hn⟩ h
  | cons c cs ih =>
    unfold loop
    by_cases hc : isDigit c
    · rw [if_pos ⟨hc, hn⟩]
      simp only [List.length_cons]
      by_cases hn' : n * 10 + (c - 48) ≤ tmax
      · have : ¬((∀ x ∈ cs, isDigit x) ∧ val cs (n * 10 + (c - 48)) ≤ tmax) := by
          rintro ⟨h1, h2⟩
          exact h ⟨fun x hx => by
            rcases List.mem_cons.mp hx with rfl | hx
            · exact hc
            · exact h1 x hx, by unfold val; exact h2⟩
        rcases ih _ hn' this with h1 | h1
        · left; omega
        · right; exact h1
      · cases cs with
        | nil => right; simp only [loop]; omega
        | cons d ds =>
          left
          unfold loop
          rw [if_neg (fun h => hn' h.2)]
          simp
    · rw [if_neg (fun h => hc h.1)]
      left; simp

/-- The key file takes a tenant number exactly when it is digits only, of a value 1 to 524287. -/
theorem parse_iff (s : List Nat) (n : Nat) :
    parse s = some n ↔ (∀ c ∈ s, isDigit c) ∧ val s 0 = n ∧ 1 ≤ n ∧ n ≤ tmax := by
  unfold parse
  dsimp only
  constructor
  · intro e
    split at e
    · cases e
    · rename_i hok
      cases e
      by_cases h : (∀ c ∈ s, isDigit c) ∧ val s 0 ≤ tmax
      · rw [loop_ok s 0 h.1 h.2] at hok ⊢
        exact ⟨h.1, rfl, by omega, by omega⟩
      · rcases loop_bad s 0 (by unfold tmax; omega) h with h1 | h1 <;> omega
  · rintro ⟨hd, rfl, h1, h2⟩
    rw [loop_ok s 0 hd h2]
    simp only [Nat.lt_irrefl, false_or]
    rw [if_neg (by omega)]

/-- Every number the loop builds stays below 10 * 524287 + 10: the C's i64 never comes close to overflowing. -/
theorem parse_bound (s : List Nat) (n : Nat) (hn : n ≤ tmax * 10 + 9) : (loop s n).2 ≤ tmax * 10 + 9 := by
  induction s generalizing n with
  | nil => exact hn
  | cons c cs ih =>
    unfold loop
    split
    · rename_i h
      have hc := h.1
      unfold isDigit at hc
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
      exact ih _ (by have := h.2; omega)
    · exact hn

end Vecindex.Tenant
