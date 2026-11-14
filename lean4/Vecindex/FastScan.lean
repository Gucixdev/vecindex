/-
  The integer arithmetic of the fast-scan (`src/tq.c`, `src/tq_scan.c`).

  A block holds 32 rows; code column `d` is 16 bytes, byte `k` carrying row `k` in its low half and row `k + 16` in
  its high half (`tq_get_lane_` / `tq_put_lane_`). A query's table has 16 bytes per column (`tq_lut_`). The scalar
  kernel of `tq_block_` is the definition:

      out[j] = (u16)(out[j] + lut[d * 16 + c])   for every column d,  c = row j's half-byte of column d

  The AVX2 / AVX-512 kernel looks the table up for 32 rows at once with a byte shuffle and adds 16-bit words without
  masking: word k holds row 2k + 256 * row 2k+1, and the even rows come back as `a0 - (a1 << 8)` (FAISS's trick).
  Proved here, for every block, table and column count:

  * `avx2_eq_ref`: the AVX2 kernel's 32 sums are the scalar kernel's, modulo 2^16;
  * `no_overflow`: with `lutmax = min(255, 65535 / vd)` (as `tq_lut_` sets it) the sums never wrap, so they are the
    exact sums of the table entries;
  * `gt_flip`: `v > thr` on unsigned 16-bit words is `(v ^ 0x8000) > (thr ^ 0x8000)` on signed ones (the AVX2 mask);
  * `vd_even`, `avx2_loads_in_block`: the column count is always even, so the two-column steps stay in the block;
  * `get_put_lane`, `get_put_lane_other`: writing one row's codes leaves the other 31 rows alone.
-/
import Vecindex.Bits

namespace Vecindex.FastScan

/-! ## Nibble lanes: `tq_get_lane_` / `tq_put_lane_` on one byte -/

/-- Row `k`'s half of a code byte: low half for `k < 16`, high half for `k >= 16`. -/
def getNib (byte k : Nat) : Nat := if k < 16 then byte &&& 15 else byte >>> 4

/-- `*q = (*q & 0xf0) | cd` or `*q = (*q & 0x0f) | (cd << 4)`. -/
def putNib (byte k cd : Nat) : Nat := if k < 16 then (byte &&& 0xf0) ||| cd else (byte &&& 0x0f) ||| (cd <<< 4)

theorem get_put_byte : ∀ byte, byte < 256 → ∀ cd, cd < 16 →
    getNib (putNib byte 0 cd) 0 = cd ∧ getNib (putNib byte 16 cd) 16 = cd ∧
    getNib (putNib byte 0 cd) 16 = getNib byte 16 ∧ getNib (putNib byte 16 cd) 0 = getNib byte 0 ∧
    putNib byte 0 cd < 256 ∧ putNib byte 16 cd < 256 := by
  decide +kernel

/-- Column `d`, byte `k mod 16` of a block: the byte that carries row `k`. -/
def laneByte (d k : Nat) : Nat := d * 16 + k % 16

/-- `tq_put_lane_` on a block (a byte array as a function). -/
def putLane (blk : Nat → Nat) (vd k : Nat) (cd : Nat → Nat) : Nat → Nat := fun x =>
  if x < vd * 16 ∧ x % 16 = k % 16 then putNib (blk x) k (cd (x / 16)) else blk x

/-- `tq_get_lane_`: row `k`'s code in column `d`. -/
def getLane (blk : Nat → Nat) (d k : Nat) : Nat := getNib (blk (laneByte d k)) k

theorem laneByte_div (d k : Nat) : laneByte d k / 16 = d := by unfold laneByte; omega
theorem laneByte_mod (d k : Nat) : laneByte d k % 16 = k % 16 := by unfold laneByte; omega

theorem getNib_norm (byte k : Nat) :
    getNib byte k = getNib byte (if k < 16 then 0 else 16) := by
  unfold getNib; split <;> simp_all

theorem putNib_norm (byte k cd : Nat) :
    putNib byte k cd = putNib byte (if k < 16 then 0 else 16) cd := by
  unfold putNib; split <;> simp_all

/-- Writing row `k` and reading it back gives the codes written. -/
theorem get_put_lane (blk : Nat → Nat) (vd k d : Nat) (cd : Nat → Nat) (hd : d < vd)
    (hb : blk (laneByte d k) < 256) (hc : cd d < 16) : getLane (putLane blk vd k cd) d k = cd d := by
  unfold getLane putLane
  rw [if_pos ⟨by unfold laneByte; omega, laneByte_mod d k⟩, laneByte_div, getNib_norm _ _,
    putNib_norm _ _ _]
  have := get_put_byte _ hb _ hc
  split
  · exact this.1
  · exact this.2.1

/-- Writing row `k` leaves every other row of the block as it was. -/
theorem get_put_lane_other (blk : Nat → Nat) (vd k k' d : Nat) (cd : Nat → Nat) (hk : k < 32) (hk' : k' < 32)
    (hne : k ≠ k') (hb : blk (laneByte d k') < 256) (hc : cd d < 16) :
    getLane (putLane blk vd k cd) d k' = getLane blk d k' := by
  unfold getLane putLane
  by_cases hx : laneByte d k' < vd * 16 ∧ laneByte d k' % 16 = k % 16
  · rw [if_pos hx, laneByte_div]
    -- same byte, so k' is the other half of it
    have hsame : k % 16 = k' % 16 := by rw [← hx.2, laneByte_mod]
    have := get_put_byte _ hb _ hc
    rw [getNib_norm _ _, putNib_norm _ _ _, getNib_norm _ k']
    by_cases h1 : k < 16
    · have h2 : ¬ k' < 16 := by omega
      simp only [h1, h2, if_true, if_false]; exact this.2.2.1
    · have h2 : k' < 16 := by omega
      simp only [h1, h2, if_true, if_false]; exact this.2.2.2.1
  · rw [if_neg hx]

/-! ## The column count -/

/-- `tq_vd_(dpad, bits)`: code columns of a row. -/
def vdOf (dpad bits : Nat) : Nat :=
  if bits = 2 then ((dpad / 2) + 1) / 2 * 2 else if bits = 1 then ((dpad + 3) / 4 + 1) / 2 * 2 else dpad

theorem and_not_one (x : Nat) : (x + 1) / 2 * 2 = x + 1 - (x + 1) % 2 := by omega

/-- `dpad = dim + (dim & 1)` is even, and so is `vd` for every code width. -/
theorem vd_even (dim bits : Nat) : vdOf (dim + dim % 2) bits % 2 = 0 := by
  unfold vdOf; split
  · omega
  · split <;> omega

/-- Enough columns for every coordinate of an even `dpad`: one per coordinate (4-bit), per pair (2-bit), per four
    (1-bit). -/
theorem vd_enough (dpad bits : Nat) (he : dpad % 2 = 0) :
    (bits = 2 → (dpad + 1) / 2 ≤ vdOf dpad bits) ∧ (bits = 1 → (dpad + 3) / 4 ≤ vdOf dpad bits) ∧
    (bits ≠ 1 → bits ≠ 2 → dpad ≤ vdOf dpad bits) := by
  unfold vdOf
  refine ⟨fun h => ?_, fun h => ?_, fun h1 h2 => ?_⟩
  · subst h; rw [if_pos rfl]; omega
  · subst h; rw [if_neg (by decide), if_pos rfl]; omega
  · rw [if_neg h2, if_neg h1]; exact Nat.le_refl _

/-- With an even column count the AVX2 kernel's 32-byte loads at `d * 16` (d even, `d < vd`) and the AVX-512
    kernel's 64-byte loads (`d + 4 <= vd`) stay inside the block's `vd * 16` bytes. -/
theorem avx2_loads_in_block (vd d : Nat) (hvd : vd % 2 = 0) (hd : d % 2 = 0) (hlt : d < vd) :
    d * 16 + 32 ≤ vd * 16 := by omega

theorem avx512_loads_in_block (vd d : Nat) (h : d + 4 ≤ vd) : d * 16 + 64 ≤ vd * 16 := by omega

/-! ## The 16-bit sums -/

/-- Sum of a list, 16-bit wrapping at every add (`_mm256_add_epi16`, `(u16)(out[j] + x)`). -/
def sum16 : List Nat → Nat
  | [] => 0
  | x :: xs => (x + sum16 xs) % 65536

theorem sum16_eq (xs : List Nat) : sum16 xs = xs.sum % 65536 := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp only [sum16, List.sum_cons, ih]; omega

/-- One 16-bit word of the shuffled table, rows `2k` (low byte `e`) and `2k + 1` (high byte `o`). -/
def word (e o : Nat) : Nat := e + 256 * o

theorem word_shr (e o : Nat) (he : e < 256) : word e o >>> 8 = o := by
  unfold word; rw [Nat.shiftRight_eq_div_pow]; omega

theorem sum_word (es os : List Nat) (h : es.length = os.length) :
    (List.zipWith word es os).sum = es.sum + 256 * os.sum := by
  induction es generalizing os with
  | nil => cases os <;> simp_all
  | cons e es ih =>
    cases os with
    | nil => simp at h
    | cons o os =>
      simp only [List.zipWith_cons_cons, List.sum_cons, List.length_cons] at h ⊢
      rw [ih os (by omega)]; unfold word; omega

theorem sum_word_shr (es os : List Nat) (h : es.length = os.length) (he : ∀ e ∈ es, e < 256) :
    ((List.zipWith word es os).map (· >>> 8)).sum = os.sum := by
  induction es generalizing os with
  | nil => cases os <;> simp_all
  | cons e es ih =>
    cases os with
    | nil => simp at h
    | cons o os =>
      simp only [List.zipWith_cons_cons, List.map_cons, List.sum_cons, List.length_cons] at h ⊢
      rw [word_shr e o (he e (List.mem_cons_self ..)), ih os (by omega) (fun x hx => he x (List.mem_cons_of_mem _ hx))]

/-- **FAISS's trick.** For the table bytes `es` (even row) and `os` (odd row) of every column: the 16-bit sums of
    the words and of the words shifted right by 8 give the even row as `a0 - (a1 << 8)` and the odd row as `a1`,
    both modulo 2^16, exactly the scalar kernel's sums. -/
theorem avx2_pair (es os : List Nat) (h : es.length = os.length) (he : ∀ e ∈ es, e < 256) :
    let a0 := sum16 (List.zipWith word es os)
    let a1 := sum16 ((List.zipWith word es os).map (· >>> 8))
    (a0 + 65536 - (a1 * 256) % 65536) % 65536 = sum16 es ∧ a1 = sum16 os := by
  show (sum16 (List.zipWith word es os) + 65536 - (sum16 ((List.zipWith word es os).map (· >>> 8)) * 256) % 65536)
      % 65536 = sum16 es ∧ sum16 ((List.zipWith word es os).map (· >>> 8)) = sum16 os
  rw [sum16_eq, sum16_eq, sum16_eq, sum16_eq, sum_word es os h, sum_word_shr es os h he]
  constructor <;> omega

/-! ## The rows of a block: the AVX2 kernel against the scalar one -/

/-- Row `r`'s table entry for column `d`: `lut[d * 16 + c]`, `c` row `r`'s half-byte of column `d`. -/
def col (blk lut : Nat → Nat) (r d : Nat) : Nat := lut (d * 16 + getNib (blk (laneByte d r)) r)

/-- The scalar kernel: `out[j]` after all `vd` columns. -/
def refScore (blk lut : Nat → Nat) (vd j : Nat) : Nat := sum16 ((List.range vd).map (col blk lut j))

/-- The AVX2 kernel for row `j`: the shuffle of half-bytes puts rows `2k` and `2k + 1` of a half (`e = j - j % 2`
    and `e + 1`) into one 16-bit word; `a0` sums the words, `a1` the words shifted right by 8; after the halves of
    the register are folded, `_mm_unpacklo/hi_epi16` hand row `e` the value `a0 - (a1 << 8)` and row `e + 1` the
    value `a1`. -/
def avx2Score (blk lut : Nat → Nat) (vd j : Nat) : Nat :=
  let e := j / 2 * 2
  let es := (List.range vd).map (col blk lut e)
  let os := (List.range vd).map (col blk lut (e + 1))
  let a0 := sum16 (List.zipWith word es os)
  let a1 := sum16 ((List.zipWith word es os).map (· >>> 8))
  if j % 2 = 0 then (a0 + 65536 - (a1 * 256) % 65536) % 65536 else a1

/-- **The AVX2 kernel computes the scalar kernel's sums**, for every block, table and column count. -/
theorem avx2_eq_ref (blk lut : Nat → Nat) (vd j : Nat) (hl : ∀ x, lut x < 256) :
    avx2Score blk lut vd j = refScore blk lut vd j := by
  have hp := avx2_pair ((List.range vd).map (col blk lut (j / 2 * 2)))
    ((List.range vd).map (col blk lut (j / 2 * 2 + 1))) (by simp)
    (by intro e he; simp only [List.mem_map] at he; obtain ⟨_, _, rfl⟩ := he; exact hl _)
  unfold avx2Score refScore
  by_cases h : j % 2 = 0
  · rw [if_pos h, hp.1, show j / 2 * 2 = j by omega]
  · rw [if_neg h, hp.2, show j / 2 * 2 + 1 = j by omega]

/-! ## No overflow -/

/-- `lutmax = 65535 / vd; if (lutmax > 255) lutmax = 255;` -/
def lutmax (vd : Nat) : Nat := min (65535 / vd) 255

theorem sum_le (xs : List Nat) (b : Nat) (h : ∀ x ∈ xs, x ≤ b) : xs.sum ≤ xs.length * b := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
    simp only [List.sum_cons, List.length_cons, Nat.succ_mul]
    have := h x (List.mem_cons_self ..)
    have := ih (fun y hy => h y (List.mem_cons_of_mem _ hy))
    omega

/-- **No sum ever wraps.** `tq_lut_` clamps every entry to `lutmax`, so a row's sum over `vd` columns is at most
    `vd * (65535 / vd) <= 65535`: the 16-bit kernel computes the exact sum. -/
theorem no_overflow (xs : List Nat) (vd : Nat) (hlen : xs.length = vd) (h : ∀ x ∈ xs, x ≤ lutmax vd) :
    sum16 xs = xs.sum := by
  rw [sum16_eq]
  apply Nat.mod_eq_of_lt
  have h1 := sum_le xs (lutmax vd) h
  have h2 : vd * lutmax vd ≤ 65535 := by
    unfold lutmax
    calc vd * min (65535 / vd) 255 ≤ vd * (65535 / vd) := Nat.mul_le_mul_left _ (Nat.min_le_left _ _)
      _ ≤ 65535 := Nat.mul_div_le 65535 vd
  rw [hlen] at h1
  omega

/-! ## The threshold mask -/

/-- A 16-bit word read as a signed one. -/
def toSigned (x : Nat) : Int := if x < 32768 then (x : Int) else (x : Int) - 65536

theorem xor_8000 (x : Nat) (hx : x < 65536) : toSigned (x ^^^ 0x8000) = (x : Int) - 32768 := by
  have h8 : (0x8000 : Nat) = 2 ^ 15 * 1 := rfl
  by_cases h : x < 32768
  · have : x ^^^ 0x8000 = 32768 + x := by rw [h8, xor_high x 1 15 (by omega)]
    rw [this]; unfold toSigned; rw [if_neg (by omega)]; omega
  · have hr : x = (x - 32768) ^^^ (2 ^ 15 * 1) := by rw [xor_high _ 1 15 (by omega)]; omega
    have : x ^^^ 0x8000 = x - 32768 := by
      conv => lhs; rw [hr]
      rw [h8, xor_cancel]
    rw [this]; unfold toSigned; rw [if_pos (by omega)]; omega

/-- `_mm_cmpgt_epi16` on sign-flipped words is the unsigned `v > thr` that the scalar kernel tests. -/
theorem gt_flip (v thr : Nat) (hv : v < 65536) (ht : thr < 65536) :
    toSigned (v ^^^ 0x8000) > toSigned (thr ^^^ 0x8000) ↔ v > thr := by
  rw [xor_8000 v hv, xor_8000 thr ht]; omega

end Vecindex.FastScan
