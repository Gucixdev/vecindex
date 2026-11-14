/-
  What a code means and what a scan adds up: the quantizers `q1.c` - `q5.c` and the codes of `tq.c`.

  * `tq_code_` (and the AVX-512 `tq_codes_`) find a coordinate's 4-bit level with four branch-free steps. Whatever
    the comparisons `v > TQ_B[i]` return, as long as they are monotone in `i` (true for a number against sorted
    boundaries, and for NaN, which compares false with all of them), the result is the number of boundaries below
    `v` (`code_eq_count`), and bit 3 of it is the comparison with `TQ_B[7] = 0`, the sign the 1-bit codes keep
    (`code_bit3`): the cascade's 1-bit codes taken from the 4-bit ones are the codes a 1-bit index would compute.
  * The 2-bit column `ca | cb << 2` gives `ca` and `cb` back (`col2_decode`), so the 16-entry table of a 2-bit column
    adds the two coordinates' products (`lut2_at_code`); the 1-bit table adds `+-x` by the four sign bits
    (`lut1_at_code`).
  * q1-q5 pack `DPB` levels of radix `R` into a byte (`b |= lvl << (BITS * s)`, or `+= lvl * 3^s` for q3) and read
    them back with a digit table; packing then reading gives the levels (`digit_pack`). The scan adds one table
    entry per byte (`vs_lut_gather_dot`); in exact arithmetic that is the inner product of the residual with the
    dequantized vector of `q*_reconstruct_into` (`scan_eq_dot`). f32 adds in another order, so the two differ by
    rounding only.
  * `tq_lut_` stores each table entry as `round((v - min_d) * scale)`; without the rounding, the sums are
    `scale * score - scale * sum min_d` for every row, so they rank the rows exactly as the scores do
    (`quantized_rank`).
-/
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Nat.Find
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Vecindex.Bits

open Finset

namespace Vecindex.Codes

/-! ## `tq_code_` -/

/-- `c = (v > B[7]) << 3; c += (v > B[c + 3]) << 2; c += (v > B[c + 1]) << 1; c += (v > B[c]);` with
    `gt i` standing for `v > TQ_B[i]`. -/
def code (gt : ℕ → Bool) : ℕ :=
  let c1 := if gt 7 then 8 else 0
  let c2 := c1 + if gt (c1 + 3) then 4 else 0
  let c3 := c2 + if gt (c2 + 1) then 2 else 0
  c3 + if gt c3 then 1 else 0

/-- The search reads boundaries 0..14 only: the sentinel `TQ_B[15]` is never compared. -/
theorem code_congr (f g : ℕ → Bool) (h : ∀ i, i < 15 → f i = g i) : code f = code g := by
  unfold code
  rw [h 7 (by decide)]
  by_cases h7 : g 7 = true
  · simp only [h7, if_true]
    rw [h 11 (by decide)]
    by_cases h11 : g 11 = true
    · simp only [h11, if_true]; rw [h 13 (by decide)]
      by_cases h13 : g 13 = true
      · simp only [h13, if_true]; rw [h 14 (by decide)]
      · simp only [h13, Bool.false_eq_true, if_false]; rw [h 12 (by decide)]
    · simp only [h11, Bool.false_eq_true, if_false]; rw [h 9 (by decide)]
      by_cases h9 : g 9 = true
      · simp only [h9, if_true]; rw [h 10 (by decide)]
      · simp only [h9, Bool.false_eq_true, if_false]; rw [h 8 (by decide)]
  · simp only [h7, Bool.false_eq_true, if_false]
    rw [h 3 (by decide)]
    by_cases h3 : g 3 = true
    · simp only [h3, if_true]; rw [h 5 (by decide)]
      by_cases h5 : g 5 = true
      · simp only [h5, if_true]; rw [h 6 (by decide)]
      · simp only [h5, Bool.false_eq_true, if_false]; rw [h 4 (by decide)]
    · simp only [h3, Bool.false_eq_true, if_false]; rw [h 1 (by decide)]
      by_cases h1 : g 1 = true
      · simp only [h1, if_true]; rw [h 2 (by decide)]
      · simp only [h1, Bool.false_eq_true, if_false]; rw [h 0 (by decide)]

theorem code_prefix : ∀ c, c ≤ 15 → code (fun i => decide (i < c)) = c ∧
    (List.range 15).countP (fun i => decide (i < c)) = c := by decide

/-- Monotone comparisons are a prefix: the first `c` boundaries are below `v`, the rest are not. -/
theorem antitone_prefix (gt : ℕ → Bool) (h : ∀ i j, i ≤ j → j < 15 → gt j = true → gt i = true) :
    ∃ c ≤ 15, ∀ i, i < 15 → gt i = decide (i < c) := by
  classical
  have hex : ∃ c, c = 15 ∨ gt c = false := ⟨15, Or.inl rfl⟩
  refine ⟨Nat.find hex, ?_, ?_⟩
  · exact Nat.find_le (Or.inl rfl)
  · intro i hi
    by_cases hic : i < Nat.find hex
    · have := Nat.find_min hex hic
      simp only [not_or, Bool.not_eq_false] at this
      simp [this.2, hic]
    · have hspec := Nat.find_spec hex
      have hc15 : Nat.find hex ≠ 15 := by omega
      have hgc : gt (Nat.find hex) = false := hspec.resolve_left hc15
      simp only [hic, decide_false]
      by_contra hgi
      simp only [Bool.not_eq_false] at hgi
      have := h (Nat.find hex) i (by omega) hi hgi
      rw [hgc] at this; exact absurd this (by decide)

/-- **The 4-bit code is the number of boundaries below the value.** -/
theorem code_eq_count (gt : ℕ → Bool) (h : ∀ i j, i ≤ j → j < 15 → gt j = true → gt i = true) :
    code gt = (List.range 15).countP gt := by
  obtain ⟨c, hc, hgt⟩ := antitone_prefix gt h
  rw [code_congr gt (fun i => decide (i < c)) hgt, (code_prefix c hc).1]
  have : (List.range 15).countP gt = (List.range 15).countP (fun i => decide (i < c)) :=
    List.countP_congr fun i hi => by rw [hgt i (List.mem_range.mp hi)]
  rw [this, (code_prefix c hc).2]

/-- Bit 3 of the code (`cd4 & 8u`) is the comparison with `TQ_B[7]`, which is `0.0f`: the sign of the
    coordinate, which is what `tq_encode_row_` keeps as a 1-bit code (`rot > 0.0f`). -/
theorem code_bit3 (gt : ℕ → Bool) (h : ∀ i j, i ≤ j → j < 15 → gt j = true → gt i = true) :
    (code gt &&& 8 ≠ 0) ↔ gt 7 = true := by
  obtain ⟨c, hc, hgt⟩ := antitone_prefix gt h
  rw [code_congr gt (fun i => decide (i < c)) hgt, (code_prefix c hc).1, hgt 7 (by decide)]
  clear hgt h
  revert c
  decide

theorem code_lt (gt : ℕ → Bool) : code gt < 16 := by
  unfold code; dsimp only; split <;> split <;> split <;> split <;> omega

/-! ## The 2-bit and 1-bit columns -/

/-- `cd[v] = (u8)(ca | (cb << 2))`, read back as `c & 3` and `c >> 2` by `tq_lut_`. -/
theorem col2_decode : ∀ ca, ca < 4 → ∀ cb, cb < 4 → (ca ||| (cb <<< 2)) &&& 3 = ca ∧ (ca ||| (cb <<< 2)) >>> 2 = cb ∧
    (ca ||| (cb <<< 2)) < 16 := by decide

variable {α : Type*} [CommRing α]

/-- `lf[d * 16 + c] = a * TQ_CENT2[c & 3] + b * TQ_CENT2[c >> 2]` -/
def lut2 (a b : α) (cent : ℕ → α) (c : ℕ) : α := a * cent (c &&& 3) + b * cent (c >>> 2)

/-- At the stored column the table holds the two coordinates' products with their levels. -/
theorem lut2_at_code (a b : α) (cent : ℕ → α) (ca cb : ℕ) (ha : ca < 4) (hb : cb < 4) :
    lut2 a b cent (ca ||| (cb <<< 2)) = a * cent ca + b * cent cb := by
  have := col2_decode ca ha cb hb
  unfold lut2; rw [this.1, this.2.1]

/-- `v = (c & 1 ? x[0] : -x[0]) + (c & 2 ? x[1] : -x[1]) + (c & 4 ? x[2] : -x[2]) + (c & 8 ? x[3] : -x[3])` -/
def lut1 (x : ℕ → α) (c : ℕ) : α := ∑ i ∈ range 4, if c.testBit i then x i else -x i

/-- The 1-bit column of signs `s i` (bit i set when coordinate 4v+i is positive). -/
def col1 (s : ℕ → Bool) : ℕ := ∑ i ∈ range 4, if s i then 2 ^ i else 0

theorem col1_testBit : ∀ s0 s1 s2 s3 : Bool, ∀ i, i < 4 →
    (col1 (fun j => [s0, s1, s2, s3].getD j false)).testBit i = [s0, s1, s2, s3].getD i false := by decide

/-- At the stored column the table holds `+x_i` for a positive coordinate and `-x_i` for the others: the inner
    product with the row's signs. -/
theorem lut1_at_code (x : ℕ → α) (s0 s1 s2 s3 : Bool) :
    lut1 x (col1 (fun j => [s0, s1, s2, s3].getD j false)) =
      ∑ i ∈ range 4, if [s0, s1, s2, s3].getD i false then x i else -x i := by
  unfold lut1
  exact sum_congr rfl fun i hi => by rw [col1_testBit s0 s1 s2 s3 i (mem_range.mp hi)]

/-! ## q1 - q5: packing -/

/-- Digit `s` of byte `b` in radix `R`: the digit tables (`(b >> (BITS * s)) & MASK` for radix `2^BITS`,
    `rem % 3; rem /= 3` for q3). -/
def digit (R b s : ℕ) : ℕ := b / R ^ s % R

/-- A byte of levels `l 0 .. l (n-1)`, little end first: `Σ l s * R ^ s`. -/
def pack (R n : ℕ) (l : ℕ → ℕ) : ℕ := ∑ s ∈ range n, l s * R ^ s

theorem pack_succ (R n : ℕ) (l : ℕ → ℕ) : pack R (n + 1) l = l 0 + R * pack R n (fun s => l (s + 1)) := by
  unfold pack
  rw [sum_range_succ', pow_zero, mul_one, add_comm, mul_sum]
  congr 1
  exact sum_congr rfl fun s _ => by ring

theorem pack_lt (R n : ℕ) (l : ℕ → ℕ) (hl : ∀ s, s < n → l s < R) : pack R n l < R ^ n := by
  induction n generalizing l with
  | zero => simp [pack]
  | succ n ih =>
    rw [pack_succ]
    have h0 := hl 0 (by omega)
    have h1 := ih (fun s => l (s + 1)) (fun s hs => hl (s + 1) (by omega))
    rw [pow_succ]
    calc l 0 + R * pack R n (fun s => l (s + 1)) < R + R * pack R n (fun s => l (s + 1)) := by omega
      _ = R * (pack R n (fun s => l (s + 1)) + 1) := by ring
      _ ≤ R * R ^ n := Nat.mul_le_mul_left _ h1
      _ = R ^ n * R := by ring

/-- **Packing then reading a digit gives the level back.** -/
theorem digit_pack (R n : ℕ) (l : ℕ → ℕ) (hl : ∀ s, s < n → l s < R) : ∀ t, t < n → digit R (pack R n l) t = l t := by
  induction n generalizing l with
  | zero => intro t ht; omega
  | succ n ih =>
    intro t ht
    have hR : 0 < R := Nat.lt_of_le_of_lt (Nat.zero_le _) (hl 0 (by omega))
    rw [pack_succ]
    cases t with
    | zero =>
      unfold digit
      rw [pow_zero, Nat.div_one, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt (hl 0 (by omega))]
    | succ t =>
      unfold digit
      have hdiv : (l 0 + R * pack R n (fun s => l (s + 1))) / R = pack R n (fun s => l (s + 1)) := by
        rw [Nat.add_mul_div_left _ _ hR, Nat.div_eq_of_lt (hl 0 (by omega)), zero_add]
      rw [pow_succ, mul_comm (R ^ t) R, ← Nat.div_div_eq_div_mul, hdiv]
      exact ih (fun s => l (s + 1)) (fun s hs => hl (s + 1) (by omega)) t (by omega)

/-- The bit-packed quantizers OR each level into place (`out |= lvl << (BITS * s)`, in increasing `s`); every
    level lands above the bits already set, so the OR is the sum `pack (2 ^ BITS)`. -/
theorem or_pack (B n : ℕ) (l : ℕ → ℕ) (hl : ∀ s, s < n → l s < 2 ^ B) :
    (List.range n).foldl (fun acc s => acc ||| (l s <<< (B * s))) 0 = pack (2 ^ B) n l := by
  have key : ∀ m, m ≤ n → (List.range m).foldl (fun acc s => acc ||| (l s <<< (B * s))) 0 = pack (2 ^ B) m l := by
    intro m
    induction m with
    | zero => intro _; simp [pack]
    | succ m ih =>
      intro hm
      rw [List.range_succ, List.foldl_append, ih (by omega)]
      simp only [List.foldl_cons, List.foldl_nil]
      have hlt : pack (2 ^ B) m l < 2 ^ (B * m) := by
        rw [pow_mul]; exact pack_lt _ _ _ (fun s hs => hl s (by omega))
      rw [Nat.shiftLeft_eq, Nat.mul_comm (l m), or_high _ _ _ hlt]
      unfold pack
      rw [sum_range_succ, ← pow_mul]
      ring
  exact key n le_rfl

/-! ## q1 - q5: the scan -/

/-- Entry `bv` of the table of byte group `g` (`q*_lut_resid`): the sum over the group's `DPB` dimensions of the
    residual times the codebook value of that dimension's digit; dimensions past `dim` add 0. -/
def lutEntry (R DPB dim : ℕ) (r : ℕ → α) (cb : ℕ → ℕ → α) (g bv : ℕ) : α :=
  ∑ s ∈ range DPB, if g * DPB + s < dim then r (g * DPB + s) * cb (g * DPB + s) (digit R bv s) else 0

/-- `vs_lut_gather_dot(lut, row, cbytes)`: one table entry per code byte. -/
def scan (R DPB dim cbytes : ℕ) (r : ℕ → α) (cb : ℕ → ℕ → α) (row : ℕ → ℕ) : α :=
  ∑ g ∈ range cbytes, lutEntry R DPB dim r cb g (row g)

/-- The level of dimension `d` in a code row, as `q*_reconstruct_into` reads it. -/
def level (R DPB : ℕ) (row : ℕ → ℕ) (d : ℕ) : ℕ := digit R (row (d / DPB)) (d % DPB)

/-- **The table scan is the inner product with the dequantized vector**: with `cbytes * DPB >= dim` bytes, the sum
    of the table entries of a row is `Σ_d r[d] * codebook[d][level(row, d)]`, i.e. the residual against
    `q*_reconstruct_into(row) - centroid`. -/
theorem scan_eq_dot (R DPB dim cbytes : ℕ) (r : ℕ → α) (cb : ℕ → ℕ → α) (row : ℕ → ℕ) (hD : 0 < DPB)
    (hcb : dim ≤ cbytes * DPB) :
    scan R DPB dim cbytes r cb row = ∑ d ∈ range dim, r d * cb d (level R DPB row d) := by
  let F : ℕ → α := fun d => if d < dim then r d * cb d (level R DPB row d) else 0
  have hgrp : ∀ g s, s < DPB → (if g * DPB + s < dim then r (g * DPB + s) * cb (g * DPB + s)
      (digit R (row g) s) else 0) = F (g * DPB + s) := by
    intro g s hs
    have h1 : (g * DPB + s) / DPB = g := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ hD, Nat.div_eq_of_lt hs, zero_add]
    have h2 : (g * DPB + s) % DPB = s := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hs]
    simp only [F, level, h1, h2]
  have hsplit : ∀ n, ∑ g ∈ range n, ∑ s ∈ range DPB, F (g * DPB + s) = ∑ d ∈ range (n * DPB), F d := by
    intro n
    induction n with
    | zero => simp
    | succ n ih => rw [sum_range_succ, ih, Nat.succ_mul, sum_range_add]
  unfold scan lutEntry
  rw [sum_congr rfl fun g _ => sum_congr rfl fun s hs => hgrp g s (mem_range.mp hs), hsplit]
  rw [← sum_subset (range_mono hcb) (fun d _ hd => by
    have : ¬ d < dim := fun h => hd (mem_range.mpr h)
    simp only [F, if_neg this])]
  exact sum_congr rfl fun d hd => by simp only [F, if_pos (mem_range.mp hd)]

/-! ## The u8 tables keep the ranking -/

variable {β : Type*} [Field β] [LinearOrder β] [IsStrictOrderedRing β]

/-- `tq_lut_` (before rounding): entry `c` of column `d` is `(lf d c - mn d) * scale`. A row's sum over the
    columns is `scale * (its score) - scale * Σ mn`, the same affine map for every row; with `scale > 0` the order
    of two rows is the order of their scores. -/
theorem quantized_rank (vd : ℕ) (lf : ℕ → ℕ → β) (mn : ℕ → β) (scale : β) (hs : 0 < scale) (c c' : ℕ → ℕ) :
    (∑ d ∈ range vd, (lf d (c d) - mn d) * scale) < ∑ d ∈ range vd, (lf d (c' d) - mn d) * scale ↔
      (∑ d ∈ range vd, lf d (c d)) < ∑ d ∈ range vd, lf d (c' d) := by
  simp only [← sum_mul, sum_sub_distrib]
  rw [mul_lt_mul_iff_of_pos_right hs]
  constructor <;> intro h <;> linarith

end Vecindex.Codes
