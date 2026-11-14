/-
  The fast-scan block layout is a stride permutation. A block keeps its 32 rows' codes column by column: byte
  `d * 16 + k mod 16` holds row `k`'s code of column `d` (low half for `k < 16`, high half for `k >= 16`). Think of the
  16 row pairs times `vd` columns as a 16 x vd matrix stored by rows; the block stores it by columns, which is
  AMO-Lean's stride permutation `L^{16 vd}_16` (`AmoLean.Matrix.strideIndex`, the transposition step of the
  Cooley-Tukey FFT). AMO-Lean's lemmas then give the layout's two safety facts:

  * `laneByte_lt` (from `strideIndex_lt`): every row's code byte of every column is inside the block's `vd * 16`
    bytes;
  * `laneByte_inj` (from `stride_inverse_eq`): no two (row pair, column) share a byte, so each byte is shared by
    exactly the rows `k` and `k + 16`, which `get_put_lane_other` keeps apart.
-/
import AmoLean.Matrix.Perm
import Vecindex.FastScan

namespace Vecindex.FastScan

open AmoLean.Matrix (strideIndex strideIndex_lt stride_inverse_eq)

/-- Row `k`'s byte of column `d` is the stride permutation at `(k mod 16) * vd + d`. -/
theorem laneByte_stride (vd d k : Nat) (hd : d < vd) : laneByte d k = strideIndex vd 16 ((k % 16) * vd + d) := by
  unfold laneByte strideIndex
  have h1 : ((k % 16) * vd + d) % vd = d := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hd]
  have h2 : ((k % 16) * vd + d) / vd = k % 16 := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hd, Nat.zero_add]
  rw [h1, h2]

theorem stride_arg_lt (vd d k : Nat) (hd : d < vd) : (k % 16) * vd + d < vd * 16 := by
  have hk : k % 16 < 16 := Nat.mod_lt _ (by decide)
  have : (k % 16) * vd + d < (k % 16 + 1) * vd := by rw [Nat.succ_mul]; omega
  have : (k % 16 + 1) * vd ≤ 16 * vd := Nat.mul_le_mul_right _ hk
  rw [Nat.mul_comm vd 16]; omega

/-- **Every code byte is inside the block.** -/
theorem laneByte_lt (vd d k : Nat) (hd : d < vd) : laneByte d k < vd * 16 := by
  rw [laneByte_stride vd d k hd]
  exact strideIndex_lt vd 16 _ (stride_arg_lt vd d k hd)

/-- **No two (row pair, column) share a byte.** -/
theorem laneByte_inj (vd d d' k k' : Nat) (hd : d < vd) (hd' : d' < vd) (h : laneByte d k = laneByte d' k') :
    d = d' ∧ k % 16 = k' % 16 := by
  rw [laneByte_stride vd d k hd, laneByte_stride vd d' k' hd'] at h
  have h' := congrArg (strideIndex 16 vd) h
  rw [stride_inverse_eq vd 16 _ (stride_arg_lt vd d k hd), stride_inverse_eq vd 16 _ (stride_arg_lt vd d' k' hd')] at h'
  have e1 : ((k % 16) * vd + d) % vd = d := by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hd]
  have e2 : ((k' % 16) * vd + d') % vd = d' := by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hd']
  have e3 : ((k % 16) * vd + d) / vd = k % 16 := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hd, Nat.zero_add]
  have e4 : ((k' % 16) * vd + d') / vd = k' % 16 := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hd', Nat.zero_add]
  exact ⟨by rw [← e1, ← e2, h'], by rw [← e3, ← e4, h']⟩

end Vecindex.FastScan
