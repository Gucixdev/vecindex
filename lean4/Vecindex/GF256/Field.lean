/-
  The bytes with `^` as addition and `gf_mul_` as multiplication form a field: a Mathlib `Field` instance on
  `GF256`, built only from the laws proved in `Algebra.lean`. Subtraction is xor too (characteristic 2), so the
  `x ^= f * y` updates of `store_sum.c` are field subtractions.
-/
import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.Ring.Defs
import Vecindex.GF256.Algebra

namespace Vecindex

/-- A byte, with the arithmetic of `store_sum.c`. -/
@[ext] structure GF256 where
  val : Nat
  lt : val < 256
  deriving DecidableEq

namespace GF256

open Vecindex.GF256 (xor_lt mul_lt)

instance : Zero GF256 := ⟨⟨0, by decide⟩⟩
instance : One GF256 := ⟨⟨1, by decide⟩⟩
instance : Add GF256 := ⟨fun a b => ⟨a.val ^^^ b.val, xor_lt _ _ a.lt b.lt⟩⟩
instance : Neg GF256 := ⟨fun a => a⟩
instance : Sub GF256 := ⟨fun a b => a + b⟩
instance : Mul GF256 := ⟨fun a b => ⟨GF256.mul a.val b.val, mul_lt _ _ b.lt a.lt⟩⟩
/-- `gf_inv_`, with the field convention `0⁻¹ = 0` (the C code never inverts 0). -/
instance : Inv GF256 := ⟨fun a => if a.val = 0 then 0 else ⟨GF256.inv a.val, GF256.inv_lt _⟩⟩
instance : Inhabited GF256 := ⟨0⟩

@[simp] theorem val_zero : (0 : GF256).val = 0 := rfl
@[simp] theorem val_one : (1 : GF256).val = 1 := rfl
@[simp] theorem val_add (a b : GF256) : (a + b).val = a.val ^^^ b.val := rfl
@[simp] theorem val_neg (a : GF256) : (-a).val = a.val := rfl
@[simp] theorem val_sub (a b : GF256) : (a - b).val = a.val ^^^ b.val := rfl
@[simp] theorem val_mul (a b : GF256) : (a * b).val = GF256.mul a.val b.val := rfl

theorem val_inv (a : GF256) (h : a.val ≠ 0) : (a⁻¹).val = GF256.inv a.val := by
  show (if a.val = 0 then (0 : GF256) else ⟨GF256.inv a.val, GF256.inv_lt _⟩).val = _
  rw [if_neg h]

/-- The cast of a natural number: its parity, characteristic 2. -/
def ofNat' (n : Nat) : GF256 := if n % 2 = 0 then 0 else 1

instance : NatCast GF256 := ⟨ofNat'⟩
instance : IntCast GF256 := ⟨fun z => if z % 2 = 0 then 0 else 1⟩

theorem add_assoc' (a b c : GF256) : a + b + c = a + (b + c) := by ext; simp [Nat.xor_assoc]
theorem add_comm' (a b : GF256) : a + b = b + a := by ext; simp [Nat.xor_comm]
theorem zero_add' (a : GF256) : 0 + a = a := by ext; simp
theorem add_zero' (a : GF256) : a + 0 = a := by ext; simp
theorem neg_add_cancel' (a : GF256) : -a + a = 0 := by ext; simp
theorem mul_assoc' (a b c : GF256) : a * b * c = a * (b * c) := by
  ext; simp only [val_mul]; exact GF256.mul_assoc _ _ _ a.lt b.lt c.lt
theorem mul_comm' (a b : GF256) : a * b = b * a := by ext; simp only [val_mul]; exact GF256.mul_comm _ _
theorem one_mul' (a : GF256) : 1 * a = a := by ext; simp only [val_mul, val_one]; exact GF256.one_mul _ a.lt
theorem mul_one' (a : GF256) : a * 1 = a := by ext; simp only [val_mul, val_one]; exact GF256.mul_one _ a.lt
theorem zero_mul' (a : GF256) : 0 * a = 0 := by ext; simp only [val_mul, val_zero]; exact GF256.zero_mul _
theorem mul_zero' (a : GF256) : a * 0 = 0 := by ext; simp only [val_mul, val_zero]; exact GF256.mul_zero _
theorem left_distrib' (a b c : GF256) : a * (b + c) = a * b + a * c := by
  ext; simp only [val_mul, val_add]; exact GF256.mul_xor _ _ _ a.lt b.lt c.lt
theorem right_distrib' (a b c : GF256) : (a + b) * c = a * c + b * c := by
  rw [mul_comm' (a + b), left_distrib', mul_comm' c, mul_comm' c]

theorem natCast_succ' (n : Nat) : ((n + 1 : Nat) : GF256) = (n : GF256) + 1 := by
  show ofNat' (n + 1) = ofNat' n + 1
  unfold ofNat'
  by_cases h : n % 2 = 0
  · rw [if_neg (by omega), if_pos h]; ext; simp
  · rw [if_pos (by omega), if_neg h]; ext; simp

instance : CommRing GF256 where
  add_assoc := add_assoc'
  zero_add := zero_add'
  add_zero := add_zero'
  add_comm := add_comm'
  mul_assoc := mul_assoc'
  one_mul := one_mul'
  mul_one := mul_one'
  zero_mul := zero_mul'
  mul_zero := mul_zero'
  left_distrib := left_distrib'
  right_distrib := right_distrib'
  mul_comm := mul_comm'
  neg_add_cancel := neg_add_cancel'
  nsmul := nsmulRec
  zsmul := zsmulRec
  natCast_zero := rfl
  natCast_succ := natCast_succ'
  intCast_ofNat n := by
    show (if ((n : Int) % 2 = 0) then (0 : GF256) else 1) = ofNat' n
    unfold ofNat'
    by_cases h : n % 2 = 0
    · rw [if_pos (by omega), if_pos h]
    · rw [if_neg (by omega), if_neg h]
  intCast_negSucc n := by
    show (if (Int.negSucc n % 2 = 0) then (0 : GF256) else 1) = -(ofNat' (n + 1))
    unfold ofNat'
    by_cases h : (n + 1) % 2 = 0
    · rw [if_pos (by omega), if_pos h]; rfl
    · rw [if_neg (by omega), if_neg h]; rfl

theorem mul_inv_cancel' (a : GF256) (h : a ≠ 0) : a * a⁻¹ = 1 := by
  have h0 : a.val ≠ 0 := fun h' => h (GF256.ext h')
  ext; rw [val_mul, val_inv a h0]; exact GF256.mul_inv _ a.lt h0

instance : Field GF256 where
  exists_pair_ne := ⟨0, 1, by decide⟩
  mul_inv_cancel := mul_inv_cancel'
  inv_zero := rfl
  nnqsmul := _
  qsmul := _

/-- Subtraction is xor: `x ^= f * y` in C is `x - f * y` here. -/
theorem sub_eq_add (a b : GF256) : a - b = a + b := rfl

end GF256
end Vecindex
