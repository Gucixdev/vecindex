/-
  The rebuild of one parity group, `ssum_group_` in `src/store_sum.c`, and the proof that it gives back every
  damaged data block whenever no more than `m` blocks of the group (data and parity together) are damaged.

  Everything in the C code works byte by byte (`gf_madd_` over B bytes), so the model is one byte position: `d i` is
  byte `x` of data block `i`, `par p` the same byte of parity block `p`. The proof holds for every `x`, so it holds
  for whole blocks.

      md[] = damaged data blocks i < ng, ascending; mp[] = damaged parity blocks p < m, ascending
      if (nmd + nmp > m) goto fail;
      use[q] = the q-th parity index not in mp                          (q < nmd)
      r[q]   = parity use[q] ^ sum over intact i of c(use[q], i) d[i]
      a[q][p] = c(use[q], md[p]);  if (gf_invert_(a, ai, nmd)) goto fail;
      x[p]   = sum over q of ai[p][q] r[q]

  `recover_eq` is the statement for any field and any coefficients whose square parts have no kernel;
  `RS.recover_gf` instantiates it with GF(2^8) and the Cauchy coefficients of the C code.
-/
import Vecindex.RS.Invert
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.List.GetD

open Finset

namespace Vecindex.RS

set_option linter.unusedSectionVars false

variable {F : Type*} [Field F] [DecidableEq F]

/-- `use[]`: the first `t` parity indices below `m` that are not damaged. -/
def useList (m t : ℕ) (mp : List ℕ) : List ℕ := ((List.range m).filter (fun p => p ∉ mp)).take t

/-- The model of `ssum_group_` from the damaged lists to the rebuilt bytes `x[0..nmd)`. -/
def recover (c : ℕ → ℕ → F) (m ng : ℕ) (d par : ℕ → F) (md mp : List ℕ) : Option (Fin md.length → F) :=
  if md.length + mp.length > m then none else
    let use : Fin md.length → ℕ := fun q => (useList m md.length mp).getD q 0
    let r : Fin md.length → F := fun q =>
      par (use q) - ∑ i ∈ (range ng).filter (fun i => i ∉ md), c (use q) i * d i
    let A : Gauss.Mat md.length F := fun q p => c (use q) (md.get p)
    (Gauss.invert A).map fun ai => Gauss.app ai r

/-! ## Facts about the index lists -/

theorem useList_length (m t : ℕ) (mp : List ℕ) (hmp : mp.Nodup) (hlt : ∀ p ∈ mp, p < m)
    (ht : t + mp.length ≤ m) : (useList m t mp).length = t := by
  unfold useList
  rw [List.length_take]
  apply Nat.min_eq_left
  -- the filtered range has m - |mp| entries
  have hsub : mp.toFinset ⊆ range m := fun p hp => mem_range.mpr (hlt p (List.mem_toFinset.mp hp))
  have hcard : ((range m).filter (fun p => p ∉ mp)).card = m - mp.length := by
    have : (range m).filter (fun p => p ∉ mp) = range m \ mp.toFinset := by
      ext p; simp [mem_sdiff]
    rw [this, card_sdiff, inter_eq_left.mpr hsub, card_range, List.toFinset_card_of_nodup hmp]
  have hlen : ((List.range m).filter (fun p => p ∉ mp)).length = ((range m).filter (fun p => p ∉ mp)).card := by
    rw [← List.toFinset_card_of_nodup ((List.nodup_range).filter _)]
    congr 1
    ext p; simp
  omega

theorem useList_nodup (m t : ℕ) (mp : List ℕ) : (useList m t mp).Nodup :=
  ((List.nodup_range).filter _).sublist (List.take_sublist _ _)

theorem useList_lt (m t : ℕ) (mp : List ℕ) : ∀ p ∈ useList m t mp, p < m := by
  intro p hp
  have := (List.take_sublist _ _).subset hp
  simpa using (List.mem_filter.mp this).1

/-! ## The algebra of the rebuild -/

theorem app_app {t : ℕ} (B A : Gauss.Mat t F) (v : Fin t → F) :
    Gauss.app B (Gauss.app A v) = Gauss.app (Gauss.mul B A) v := by
  funext i
  simp only [Gauss.app, Gauss.mul, mul_sum, sum_mul]
  rw [Finset.sum_comm]
  exact sum_congr rfl fun k _ => sum_congr rfl fun l _ => by ring

theorem app_one {t : ℕ} (v : Fin t → F) : Gauss.app Gauss.one v = v := by
  funext i; simp [Gauss.app, Gauss.one]

/-- The parity of a group minus its intact blocks' share is the damaged blocks' share. -/
theorem residual (c : ℕ → ℕ → F) (ng p : ℕ) (d : ℕ → F) (md : List ℕ) (hmd : md.Nodup)
    (hlt : ∀ i ∈ md, i < ng) :
    (∑ i ∈ range ng, c p i * d i) - ∑ i ∈ (range ng).filter (fun i => i ∉ md), c p i * d i =
      ∑ q : Fin md.length, c p (md.get q) * d (md.get q) := by
  rw [← sum_filter_add_sum_filter_not (range ng) (fun i => i ∈ md), add_sub_cancel_right]
  have hfin : (range ng).filter (fun i => i ∈ md) = md.toFinset := by
    ext i; simp only [mem_filter, mem_range, List.mem_toFinset]
    exact ⟨fun h => h.2, fun h => ⟨hlt i h, h⟩⟩
  rw [hfin, List.sum_toFinset _ hmd, ← Fin.sum_univ_fun_getElem md (fun i => c p i * d i)]
  rfl

/-- **The rebuild gives the data back.** For any field and coefficients whose square parts (distinct parity rows
    below `m`, distinct data columns below `ng`) have no kernel: when no more than `m` blocks of a group are damaged,
    `recover` returns exactly the damaged data blocks. -/
theorem recover_eq (c : ℕ → ℕ → F) (m ng : ℕ) (d par : ℕ → F) (md mp : List ℕ)
    (hpar : ∀ p, p < m → par p = ∑ i ∈ range ng, c p i * d i)
    (hmd : md.Nodup) (hmdlt : ∀ i ∈ md, i < ng)
    (hmp : mp.Nodup) (hmplt : ∀ p ∈ mp, p < m)
    (hcount : md.length + mp.length ≤ m)
    (hMDS : ∀ (rows cols : Fin md.length → ℕ), Function.Injective rows → Function.Injective cols →
      (∀ q, rows q < m) → (∀ p, cols p < ng) →
      ∀ v : Fin md.length → F, (∀ q, ∑ p, c (rows q) (cols p) * v p = 0) → v = 0) :
    ∃ x, recover c m ng d par md mp = some x ∧ ∀ p, x p = d (md.get p) := by
  have hlen := useList_length m md.length mp hmp hmplt hcount
  set t := md.length
  let use : Fin t → ℕ := fun q => (useList m t mp).getD q 0
  have huse : ∀ q : Fin t, use q = (useList m t mp).get ⟨q, by rw [hlen]; exact q.isLt⟩ := by
    intro q; simp only [use]; rw [List.getD_eq_getElem]; rfl
  have huse_lt : ∀ q, use q < m := fun q => by
    rw [huse]; exact useList_lt m t mp _ (List.get_mem _ _)
  have huse_inj : Function.Injective use := by
    intro q q' h
    rw [huse, huse] at h
    have := (List.Nodup.get_inj_iff (useList_nodup m t mp)).mp h
    exact Fin.ext (by simpa using this)
  have hmd_inj : Function.Injective md.get := (List.nodup_iff_injective_get.mp hmd)
  let A : Gauss.Mat t F := fun q p => c (use q) (md.get p)
  have hA : ∀ v, Gauss.app A v = 0 → v = 0 := by
    intro v hv
    apply hMDS use md.get huse_inj hmd_inj huse_lt (fun p => hmdlt _ (List.get_mem _ _)) v
    intro q
    exact congrFun hv q
  obtain ⟨ai, hai, hmul⟩ := Gauss.invert_ok A hA
  let dv : Fin t → F := fun p => d (md.get p)
  have hr : (fun q : Fin t => par (use q) - ∑ i ∈ (range ng).filter (fun i => i ∉ md), c (use q) i * d i)
      = Gauss.app A dv := by
    funext q
    rw [hpar _ (huse_lt q), residual c ng (use q) d md hmd hmdlt]
    rfl
  refine ⟨Gauss.app ai (Gauss.app A dv), ?_, ?_⟩
  · have hrec : recover c m ng d par md mp = (Gauss.invert A).map (fun ai => Gauss.app ai
        (fun q => par (use q) - ∑ i ∈ (range ng).filter (fun i => i ∉ md), c (use q) i * d i)) := by
      unfold recover; rw [if_neg (by omega)]
    rw [hrec, hai, Option.map_some, hr]
  · intro p; rw [app_app, hmul, app_one]

end Vecindex.RS
