/-
  CRC-32C (Castagnoli) of `src/store_batch.c`, which guards every journal record and both trailers of a store file.

      static void store_crc_init_(void)
      {   for (i = 0; i < 256; ++i) { for (c = i, j = 0; j < 8; ++j) c = c & 1u ? (c >> 1) ^ 0x82f63b78u : c >> 1;
                                       store_crc_tab_[i] = c; } }
      static u32 store_crc32c_up_(u32 crc, const u8 *p, size_t n)
      {   ... while (n--) crc = store_crc_tab_[(crc ^ *p++) & 0xffu] ^ (crc >> 8); return crc; }
      static u32 store_crc32c_(const u8 *p, size_t n) { return ~store_crc32c_up_(0xffffffffu, p, n); }

  Proved: the table path computes the bit-at-a-time definition (`upTab_eq_upBit`); the standard check value
  (`check_value`); and two messages of the same length that differ in one byte - any change inside one byte, every
  single-bit error among them - never have the same CRC (`detects_byte_error`). The SSE4.2 path (`_mm_crc32_u64`)
  is Intel's instruction for the same CRC; the differential test runs both paths of the C code against this model.
-/
import Vecindex.Iter
import Vecindex.Bits

namespace Vecindex.Crc

/-- The reflected Castagnoli polynomial. -/
def POLY : Nat := 0x82f63b78

/-- `c & 1u ? (c >> 1) ^ 0x82f63b78u : c >> 1` -/
def bit (c : Nat) : Nat := if c % 2 = 1 then (c >>> 1) ^^^ POLY else c >>> 1

/-- `store_crc_tab_[i]`. -/
def tab (i : Nat) : Nat := iter bit 8 i

/-- `store_crc32c_up_`, the table path. -/
def upTab (crc : Nat) : List Nat → Nat
  | [] => crc
  | b :: bs => upTab (tab ((crc ^^^ b) &&& 0xff) ^^^ (crc >>> 8)) bs

/-- CRC-32C by its definition: each byte into the register, then eight one-bit steps. -/
def upBit (crc : Nat) : List Nat → Nat
  | [] => crc
  | b :: bs => upBit (iter bit 8 (crc ^^^ b)) bs

/-- `store_crc32c_`: `~` of a 32-bit register is xor with `0xffffffff`. -/
def crc32c (bs : List Nat) : Nat := upTab 0xffffffff bs ^^^ 0xffffffff

/-! ## One bit step is linear -/

theorem bit_xor (x y : Nat) : bit (x ^^^ y) = bit x ^^^ bit y := by
  have hs : (x ^^^ y) >>> 1 = (x >>> 1) ^^^ (y >>> 1) := Nat.shiftRight_xor_distrib
  have hm := @Nat.xor_mod_two_eq_one x y
  unfold bit
  rcases Nat.mod_two_eq_zero_or_one x with hx | hx <;> rcases Nat.mod_two_eq_zero_or_one y with hy | hy
  · have h1 : ¬ (x ^^^ y) % 2 = 1 := fun h => hm.mp h ⟨fun _ => by omega, fun _ => by omega⟩
    rw [if_neg h1, if_neg (by omega : ¬ x % 2 = 1), if_neg (by omega : ¬ y % 2 = 1), hs]
  · have h1 : (x ^^^ y) % 2 = 1 := hm.mpr fun h => by have := h.mpr hy; omega
    rw [if_pos h1, if_neg (by omega : ¬ x % 2 = 1), if_pos hy, hs]; ac_rfl
  · have h1 : (x ^^^ y) % 2 = 1 := hm.mpr fun h => by have := h.mp hx; omega
    rw [if_pos h1, if_pos hx, if_neg (by omega : ¬ y % 2 = 1), hs]; ac_rfl
  · have h1 : ¬ (x ^^^ y) % 2 = 1 := fun h => hm.mp h ⟨fun _ => hy, fun _ => hx⟩
    rw [if_neg h1, if_pos hx, if_pos hy, hs]
    calc (x >>> 1) ^^^ (y >>> 1) = ((x >>> 1) ^^^ (y >>> 1)) ^^^ (POLY ^^^ POLY) := by
          rw [Nat.xor_self, Nat.xor_zero]
      _ = ((x >>> 1) ^^^ POLY) ^^^ ((y >>> 1) ^^^ POLY) := by ac_rfl

theorem iter_bit_xor : ∀ n x y, iter bit n (x ^^^ y) = iter bit n x ^^^ iter bit n y
  | 0, _, _ => rfl
  | n + 1, x, y => by
    show iter bit n (bit (x ^^^ y)) = iter bit n (bit x) ^^^ iter bit n (bit y)
    rw [bit_xor]; exact iter_bit_xor n _ _

/-- Steps on a register whose low bits are zero only shift it. -/
theorem iter_bit_high : ∀ j w, iter bit j (2 ^ j * w) = w
  | 0, w => by simp [iter]
  | j + 1, w => by
    show iter bit j (bit (2 ^ (j + 1) * w)) = w
    have h : bit (2 ^ (j + 1) * w) = 2 ^ j * w := by
      unfold bit
      rw [if_neg (by rw [Nat.pow_succ, Nat.mul_comm (2 ^ j) 2, Nat.mul_assoc]; omega),
        Nat.shiftRight_eq_div_pow, Nat.pow_one, Nat.pow_succ, Nat.mul_comm (2 ^ j) 2, Nat.mul_assoc,
        Nat.mul_div_cancel_left _ (by decide)]
    rw [h]; exact iter_bit_high j w

/-- One table lookup is eight bit steps. -/
theorem tab_step (crc b : Nat) (hb : b < 256) :
    tab ((crc ^^^ b) &&& 0xff) ^^^ (crc >>> 8) = iter bit 8 (crc ^^^ b) := by
  have hsplit : crc ^^^ b = ((crc ^^^ b) % 256) ^^^ (2 ^ 8 * ((crc ^^^ b) / 256)) := by
    rw [xor_high _ _ 8 (Nat.mod_lt _ (by decide))]
    omega
  have hhi : (crc ^^^ b) / 256 = crc >>> 8 := by
    rw [show (crc ^^^ b) / 256 = (crc ^^^ b) >>> 8 by rw [Nat.shiftRight_eq_div_pow],
      Nat.shiftRight_xor_distrib, show b >>> 8 = 0 by rw [Nat.shiftRight_eq_div_pow]; omega, Nat.xor_zero]
  conv => rhs; rw [hsplit]
  rw [iter_bit_xor, iter_bit_high, hhi,
    show (crc ^^^ b) &&& 0xff = (crc ^^^ b) % 256 from Nat.and_two_pow_sub_one_eq_mod _ 8]
  rfl

/-- **The table path is CRC-32C.** -/
theorem upTab_eq_upBit : ∀ (bs : List Nat) (crc : Nat), (∀ b ∈ bs, b < 256) → upTab crc bs = upBit crc bs
  | [], _, _ => rfl
  | b :: bs, crc, h => by
    show upTab (tab ((crc ^^^ b) &&& 0xff) ^^^ (crc >>> 8)) bs = upBit (iter bit 8 (crc ^^^ b)) bs
    rw [tab_step crc b (h b (List.mem_cons_self ..))]
    exact upTab_eq_upBit bs _ (fun b' hb' => h b' (List.mem_cons_of_mem _ hb'))

/-- The check value of CRC-32C ("123456789"). -/
theorem check_value : crc32c [0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39] = 0xE3069283 := by
  decide +kernel

/-! ## Error detection -/

theorem bit_lt (x : Nat) (hx : x < 2 ^ 32) : bit x < 2 ^ 32 := by
  unfold bit
  have h1 : x >>> 1 < 2 ^ 32 := by rw [Nat.shiftRight_eq_div_pow]; omega
  split
  · exact Nat.xor_lt_two_pow h1 (by decide)
  · exact h1

theorem iter_bit_lt (n x : Nat) (hx : x < 2 ^ 32) : iter bit n x < 2 ^ 32 :=
  iter_induct bit (· < 2 ^ 32) bit_lt n x hx

theorem bit_eq_zero (x : Nat) (hx : x < 2 ^ 32) (h : bit x = 0) : x = 0 := by
  unfold bit at h
  have hd : x >>> 1 = x / 2 := by rw [Nat.shiftRight_eq_div_pow, Nat.pow_one]
  split at h
  · -- odd: x / 2 = POLY, but x / 2 < 2^31 < POLY
    have : x >>> 1 = POLY := by
      calc x >>> 1 = (x >>> 1 ^^^ POLY) ^^^ POLY := by rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
        _ = POLY := by rw [h, Nat.zero_xor]
    rw [hd] at this
    unfold POLY at this
    omega
  · rw [hd] at h; omega

/-- One bit step loses nothing on a 32-bit register (the polynomial has its top bit set). -/
theorem bit_inj (x y : Nat) (hx : x < 2 ^ 32) (hy : y < 2 ^ 32) (h : bit x = bit y) : x = y := by
  apply xor_eq_zero
  apply bit_eq_zero _ (Nat.xor_lt_two_pow hx hy)
  rw [bit_xor, h, Nat.xor_self]

theorem iter_bit_inj : ∀ n x y, x < 2 ^ 32 → y < 2 ^ 32 → iter bit n x = iter bit n y → x = y
  | 0, _, _, _, _, h => h
  | n + 1, x, y, hx, hy, h =>
    bit_inj x y hx hy (iter_bit_inj n (bit x) (bit y) (bit_lt x hx) (bit_lt y hy) h)

theorem upBit_append (crc : Nat) (u w : List Nat) : upBit crc (u ++ w) = upBit (upBit crc u) w := by
  induction u generalizing crc with
  | nil => rfl
  | cons b u ih => exact ih _

theorem upBit_lt : ∀ (bs : List Nat) (crc : Nat), crc < 2 ^ 32 → (∀ b ∈ bs, b < 256) → upBit crc bs < 2 ^ 32
  | [], _, h, _ => h
  | b :: bs, crc, h, hb =>
    upBit_lt bs _ (iter_bit_lt 8 _ (Nat.xor_lt_two_pow h (by have := hb b (List.mem_cons_self ..); omega)))
      (fun b' h' => hb b' (List.mem_cons_of_mem _ h'))

/-- Two different registers stay different through any bytes. -/
theorem upBit_inj : ∀ (bs : List Nat) (c c' : Nat), c < 2 ^ 32 → c' < 2 ^ 32 → (∀ b ∈ bs, b < 256) →
    c ≠ c' → upBit c bs ≠ upBit c' bs
  | [], _, _, _, _, _, h => h
  | b :: bs, c, c', hc, hc', hb, hne => by
    have hb0 : b < 2 ^ 32 := by have := hb b (List.mem_cons_self ..); omega
    apply upBit_inj bs _ _ (iter_bit_lt 8 _ (Nat.xor_lt_two_pow hc hb0))
      (iter_bit_lt 8 _ (Nat.xor_lt_two_pow hc' hb0)) (fun b' h' => hb b' (List.mem_cons_of_mem _ h'))
    intro h
    apply hne
    have := iter_bit_inj 8 _ _ (Nat.xor_lt_two_pow hc hb0) (Nat.xor_lt_two_pow hc' hb0) h
    calc c = (c ^^^ b) ^^^ b := by rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
      _ = (c' ^^^ b) ^^^ b := by rw [this]
      _ = c' := by rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]

/-- **Every error inside one byte is caught**: two messages of the same length that differ only in one byte have
    different CRCs. This covers every single-bit error. -/
theorem detects_byte_error (u w : List Nat) (b b' : Nat) (hb : b < 256) (hb' : b' < 256) (hne : b ≠ b')
    (hu : ∀ x ∈ u, x < 256) (hw : ∀ x ∈ w, x < 256) :
    crc32c (u ++ b :: w) ≠ crc32c (u ++ b' :: w) := by
  have hall : ∀ c, (∀ x ∈ u ++ c :: w, x < 256) ↔ c < 256 := by
    intro c
    constructor
    · intro h; exact h c (by simp)
    · intro hc x hx
      simp only [List.mem_append, List.mem_cons] at hx
      rcases hx with hx | hx | hx
      · exact hu x hx
      · rw [hx]; exact hc
      · exact hw x hx
  unfold crc32c
  rw [upTab_eq_upBit _ _ ((hall b).mpr hb), upTab_eq_upBit _ _ ((hall b').mpr hb'), upBit_append, upBit_append]
  intro h
  have h' := congrArg (· ^^^ 0xffffffff) h
  simp only [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero] at h'
  have hr : upBit 0xffffffff u < 2 ^ 32 := upBit_lt u _ (by decide) hu
  generalize upBit 0xffffffff u = r at h' hr
  have hr0 : ∀ c, c < 256 → r ^^^ c < 2 ^ 32 := fun c hc => Nat.xor_lt_two_pow hr (by omega)
  refine upBit_inj w (iter bit 8 (r ^^^ b)) (iter bit 8 (r ^^^ b')) (iter_bit_lt 8 _ (hr0 b hb))
    (iter_bit_lt 8 _ (hr0 b' hb')) hw ?_ h'
  intro h2
  apply hne
  have := iter_bit_inj 8 _ _ (hr0 b hb) (hr0 b' hb') h2
  calc b = r ^^^ (r ^^^ b) := by rw [← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]
    _ = r ^^^ (r ^^^ b') := by rw [this]
    _ = b' := by rw [← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]

end Vecindex.Crc
