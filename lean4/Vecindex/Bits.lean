/-
  Bit facts shared by the CRC, GF(2^8) and fast-scan files.
-/

namespace Vecindex

theorem xor_swap (a b c d : Nat) : (a ^^^ b) ^^^ (c ^^^ d) = (a ^^^ c) ^^^ (b ^^^ d) := by ac_rfl

theorem xor_cancel (a b : Nat) : (a ^^^ b) ^^^ b = a := by rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]

theorem xor_eq_zero {a b : Nat} (h : a ^^^ b = 0) : a = b := by
  rw [← xor_cancel a b, h, Nat.zero_xor]

/-- A number below `2 ^ i` and a multiple of `2 ^ i` share no bits: their xor is their sum. -/
theorem xor_high (a w i : Nat) (ha : a < 2 ^ i) : a ^^^ (2 ^ i * w) = 2 ^ i * w + a := by
  apply Nat.eq_of_testBit_eq
  intro j
  rw [Nat.testBit_xor, Nat.testBit_two_pow_mul_add w ha j]
  have h0 : (2 ^ i * w).testBit j = (2 ^ i * w + 0).testBit j := by rw [Nat.add_zero]
  rw [h0, Nat.testBit_two_pow_mul_add w (Nat.two_pow_pos i) j]
  by_cases hj : j < i
  · simp [hj]
  · have : a.testBit j = false :=
      Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le ha (Nat.pow_le_pow_right (by decide) (by omega)))
    simp [hj, this]

/-- The same for `|`: bits of different ranges add. -/
theorem or_high (a w i : Nat) (ha : a < 2 ^ i) : a ||| (2 ^ i * w) = 2 ^ i * w + a := by
  apply Nat.eq_of_testBit_eq
  intro j
  rw [Nat.testBit_or, Nat.testBit_two_pow_mul_add w ha j]
  have h0 : (2 ^ i * w).testBit j = (2 ^ i * w + 0).testBit j := by rw [Nat.add_zero]
  rw [h0, Nat.testBit_two_pow_mul_add w (Nat.two_pow_pos i) j]
  by_cases hj : j < i
  · simp [hj]
  · have : a.testBit j = false :=
      Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le ha (Nat.pow_le_pow_right (by decide) (by omega)))
    simp [hj, this]

end Vecindex
