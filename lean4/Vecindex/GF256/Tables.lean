/-
  GF(2^8) exactly as `src/store_sum.c` builds it.

      static void gf_init_(void)
      {
          u32 i, x = 1;
          for (i = 0; i < 255; ++i) { gf_exp_[i] = (u8)x; gf_log_[x] = (u8)i; x <<= 1; if (x & 0x100u) x ^= 0x11du; }
          for (i = 255; i < 512; ++i) gf_exp_[i] = gf_exp_[i - 255];
      }
      static u8 gf_mul_(u32 a, u32 b) { return (u8)(a && b ? gf_exp_[gf_log_[a] + gf_log_[b]] : 0); }
      static u8 gf_inv_(u32 a) { return gf_exp_[255 - gf_log_[a]]; }
      static u8 gf_coef_(u32 k, u32 p, u32 i) { return gf_inv_((k + p) ^ i); }

  Every value is a natural number below 256 (a byte). The facts the field proof needs about the two tables are
  finite and are checked here by the kernel (`decide`), with no `native_decide`.
-/
import Vecindex.Iter

namespace Vecindex.GF256

set_option maxRecDepth 100000

/-- One turn of the loop in `gf_init_`: `x <<= 1; if (x & 0x100u) x ^= 0x11du;` -/
def xt (x : Nat) : Nat :=
  if (x <<< 1) &&& 0x100 ≠ 0 then (x <<< 1) ^^^ 0x11d else x <<< 1

/-- `gf_exp_[0..254]`: the values `x` takes, in order. -/
def expList : List Nat := iterList xt 255 1

/-- All 512 entries of `gf_exp_`: the second loop copies entry `i - 255` into entry `i`, so entries 255..509 repeat
    the first 255 and entries 510, 511 repeat entries 0 and 1. -/
def expTab : List Nat := expList ++ expList ++ expList.take 2

/-- `gf_log_` as the first loop leaves it: entry `gf_exp_[i]` set to `i` for `i = 0..254` in order, entry 0 never
    written (static storage starts at zero). -/
def logTab : List Nat :=
  (List.range 255).foldl (fun t i => t.set (expList.getD i 0) i) (List.replicate 256 0)

/-- The same table written out, so that the kernel does not rebuild it in every `decide`. -/
def logLit : List Nat := [
  0, 0, 1, 25, 2, 50, 26, 198, 3, 223, 51, 238, 27, 104, 199, 75, 4, 100, 224, 14, 52, 141, 239, 129, 28, 193, 105,
  248, 200, 8, 76, 113, 5, 138, 101, 47, 225, 36, 15, 33, 53, 147, 142, 218, 240, 18, 130, 69, 29, 181, 194, 125,
  106, 39, 249, 185, 201, 154, 9, 120, 77, 228, 114, 166, 6, 191, 139, 98, 102, 221, 48, 253, 226, 152, 37, 179, 16,
  145, 34, 136, 54, 208, 148, 206, 143, 150, 219, 189, 241, 210, 19, 92, 131, 56, 70, 64, 30, 66, 182, 163, 195, 72,
  126, 110, 107, 58, 40, 84, 250, 133, 186, 61, 202, 94, 155, 159, 10, 21, 121, 43, 78, 212, 229, 172, 115, 243, 167,
  87, 7, 112, 192, 247, 140, 128, 99, 13, 103, 74, 222, 237, 49, 197, 254, 24, 227, 165, 153, 119, 38, 184, 180, 124,
  17, 68, 146, 217, 35, 32, 137, 46, 55, 63, 209, 91, 149, 188, 207, 205, 144, 135, 151, 178, 220, 252, 190, 97, 242,
  86, 211, 171, 20, 42, 93, 158, 132, 60, 57, 83, 71, 109, 65, 162, 31, 45, 67, 216, 183, 123, 164, 118, 196, 23, 73,
  236, 127, 12, 111, 246, 108, 161, 59, 82, 41, 157, 85, 170, 251, 96, 134, 177, 187, 204, 62, 90, 203, 89, 95, 176,
  156, 169, 160, 81, 11, 245, 22, 235, 122, 117, 44, 215, 79, 174, 213, 233, 230, 231, 173, 232, 116, 214, 244, 234,
  168, 80, 88, 175]

theorem logTab_eq : logTab = logLit := by decide +kernel

/-- `gf_exp_[i]`. -/
def exp (i : Nat) : Nat := expTab.getD i 0

/-- `gf_log_[a]`. -/
def log (a : Nat) : Nat := logTab.getD a 0

/-- `gf_mul_(a, b)`: the index `gf_log_[a] + gf_log_[b]` is at most 508, inside the 512 entries. -/
def mul (a b : Nat) : Nat := if a ≠ 0 ∧ b ≠ 0 then exp (log a + log b) else 0

/-- `gf_inv_(a)`. -/
def inv (a : Nat) : Nat := exp (255 - log a)

/-- `gf_coef_(k, p, i)`: the Cauchy coefficient of data block `i` in parity block `p`. -/
def coef (k p i : Nat) : Nat := inv ((k + p) ^^^ i)

/-! ## Finite facts, checked by the kernel -/

theorem expTab_length : expTab.length = 512 := by decide +kernel

theorem xt_lt : ∀ x, x < 256 → xt x < 256 := by decide +kernel

theorem xt_ne_zero : ∀ x, x < 256 → x ≠ 0 → xt x ≠ 0 := by decide +kernel

/-- The generator 2 has order 255: `x` comes back to 1 after 255 turns, and not before. -/
theorem xt_iter_255 : iter xt 255 1 = 1 := by decide +kernel

theorem expList_ne_one : ∀ i, i < 255 → 0 < i → expList.getD i 0 ≠ 1 := by decide +kernel

/-- The second half of `gf_exp_` repeats the first: entry `i` is entry `i mod 255` of the loop. -/
theorem expTab_getD : ∀ i, i < 512 → expTab.getD i 0 = expList.getD (i % 255) 0 := by decide +kernel

theorem expList_lt : ∀ i, i < 255 → expList.getD i 0 < 256 := by decide +kernel

theorem expList_ne_zero : ∀ i, i < 255 → expList.getD i 0 ≠ 0 := by decide +kernel

theorem logLit_lt : ∀ a, a < 256 → logLit.getD a 0 < 255 := by decide +kernel

theorem logLit_expList : ∀ i, i < 255 → logLit.getD (expList.getD i 0) 0 = i := by decide +kernel

theorem expList_logLit : ∀ a, a < 256 → a ≠ 0 → expList.getD (logLit.getD a 0) 0 = a := by decide +kernel

theorem logLit_one : logLit.getD 1 0 = 0 := by decide +kernel

end Vecindex.GF256
