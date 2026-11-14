/-
  The top-k insertion of the quantizer scans (`q1_topk_insert_` in `src/q1.c`, the same in q2-q5):

      if (*have < k) { j = *have; put (id, score) at j; while (j > 0 && scores[j - 1] < scores[j]) swap down; ++*have; }
      else if (score > scores[k - 1]) { j = k - 1; put it at j; while (j > 0 && scores[j - 1] < scores[j]) swap down; }

  Proved for scores from any linear order (f32 without NaN): after any sequence of candidates the array holds
  `min(k, n)` of them, sorted from the best, and no candidate that is not in it scores higher than one that is
  (`topk_correct`). The bubbling loop is the insertion of the new candidate after every entry that is not worse
  (`bubble_eq_ins`), so ties keep their arrival order. `k >= 1`: with `k = 0` the C code would read
  `scores[-1]`; the callers refuse `k < 1` first.
-/
import Mathlib.Order.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Induction

namespace Vecindex.TopK

set_option linter.dupNamespace false
set_option linter.unusedSectionVars false

variable {ι β : Type*} [LinearOrder β]

/-- Sorted from the best score down. -/
def Sorted (L : List (ι × β)) : Prop := L.Pairwise fun a b => b.2 ≤ a.2

/-- Insertion after every entry that is not worse than `x`. -/
def ins (x : ι × β) : List (ι × β) → List (ι × β)
  | [] => [x]
  | y :: ys => if y.2 < x.2 then x :: y :: ys else y :: ins x ys

/-- `while (j > 0 && scores[j - 1] < scores[j]) swap`: `x` written after the entries `L` and moved down past every
    entry that scores less. Defined on the reversed prefix, the side the loop walks. -/
def bubbleRev (x : ι × β) : List (ι × β) → List (ι × β)
  | [] => [x]
  | y :: ys => if y.2 < x.2 then y :: bubbleRev x ys else x :: y :: ys

def bubble (L : List (ι × β)) (x : ι × β) : List (ι × β) := (bubbleRev x L.reverse).reverse

/-- One call of `q1_topk_insert_`. -/
def step (k : ℕ) (L : List (ι × β)) (x : ι × β) : List (ι × β) :=
  if L.length < k then bubble L x
  else match L.getLast? with
    | some last => if last.2 < x.2 then bubble L.dropLast x else L
    | none => L

theorem ins_perm (x : ι × β) : ∀ L : List (ι × β), (ins x L).Perm (x :: L)
  | [] => List.Perm.refl _
  | y :: ys => by
    unfold ins
    split
    · exact List.Perm.refl _
    · exact ((ins_perm x ys).cons y).trans (List.Perm.swap x y ys)

theorem ins_sorted (x : ι × β) : ∀ L : List (ι × β), Sorted L → Sorted (ins x L)
  | [] => fun _ => List.pairwise_singleton _ _
  | y :: ys => by
    intro h
    have hy := List.pairwise_cons.mp h
    unfold ins
    split
    · rename_i hlt
      refine List.pairwise_cons.mpr ⟨fun z hz => ?_, h⟩
      rcases List.mem_cons.mp hz with rfl | hz
      · exact le_of_lt hlt
      · exact le_trans (hy.1 z hz) (le_of_lt hlt)
    · rename_i hge
      refine List.pairwise_cons.mpr ⟨fun z hz => ?_, ins_sorted x ys hy.2⟩
      rcases List.mem_cons.mp ((ins_perm x ys).subset hz) with rfl | hz
      · exact not_lt.mp hge
      · exact hy.1 z hz

theorem ins_append_min (x y : ι × β) (hlt : y.2 < x.2) :
    ∀ L : List (ι × β), ins x L ++ [y] = ins x (L ++ [y])
  | [] => by simp [ins, hlt]
  | z :: zs => by
    simp only [List.cons_append, ins]
    split
    · rfl
    · rw [List.cons_append, ins_append_min x y hlt zs]

theorem ins_append_end (x : ι × β) : ∀ L : List (ι × β), (∀ z ∈ L, ¬ z.2 < x.2) → ins x L = L ++ [x]
  | [] => fun _ => rfl
  | z :: zs => fun hx => by
    simp only [ins, if_neg (hx z (List.mem_cons_self ..)), List.cons_append]
    rw [ins_append_end x zs (fun w hw => hx w (List.mem_cons_of_mem _ hw))]

/-- On a sorted array the bubbling loop is `ins`. -/
theorem bubble_eq_ins (x : ι × β) (L : List (ι × β)) (hL : Sorted L) : bubble L x = ins x L := by
  induction L using List.reverseRecOn with
  | nil => rfl
  | append_singleton L y ih =>
    have hs : Sorted L := (List.pairwise_append.mp hL).1
    have hyL : ∀ z ∈ L, y.2 ≤ z.2 :=
      fun z hz => (List.pairwise_append.mp hL).2.2 z hz y (List.mem_singleton_self y)
    unfold bubble
    rw [List.reverse_append, List.reverse_singleton, List.singleton_append]
    unfold bubbleRev
    split
    · rename_i hlt
      rw [List.reverse_cons]
      have := ih hs
      unfold bubble at this
      rw [this]
      exact ins_append_min x y hlt L
    · rename_i hge
      have hx : ∀ z ∈ L ++ [y], ¬ z.2 < x.2 := by
        intro z hz
        rcases List.mem_append.mp hz with hz | hz
        · exact fun h => hge (lt_of_le_of_lt (hyL z hz) h)
        · rw [List.mem_singleton.mp hz]; exact hge
      rw [ins_append_end x _ hx, List.reverse_cons, List.reverse_cons, List.reverse_reverse]

/-- The invariant: the array is sorted, holds `min(k, n)` candidates, and together with the candidates left out
    it is a rearrangement of all of them, none of the left-out ones better than any kept one. -/
structure TopK (k : ℕ) (xs L : List (ι × β)) : Prop where
  sorted : Sorted L
  length : L.length = min k xs.length
  split : ∃ rest, (L ++ rest).Perm xs ∧ ∀ y ∈ L, ∀ z ∈ rest, z.2 ≤ y.2

theorem dropLast_append_getLast (L : List (ι × β)) (last : ι × β) (h : L.getLast? = some last) :
    L.dropLast ++ [last] = L := by
  have hne : L ≠ [] := by intro h'; subst h'; simp at h
  rw [List.getLast?_eq_some_getLast hne, Option.some.injEq] at h
  rw [← h]; exact List.dropLast_append_getLast hne

theorem step_topk (k : ℕ) (hk : 1 ≤ k) (xs L : List (ι × β)) (x : ι × β) (h : TopK k xs L) :
    TopK k (xs ++ [x]) (step k L x) := by
  obtain ⟨hs, hlen, rest, hperm, hrest⟩ := h
  unfold step
  by_cases hlt : L.length < k
  · rw [if_pos hlt, bubble_eq_ins x L hs]
    have hrest0 : rest = [] := by
      have := hperm.length_eq
      rw [List.length_append] at this
      have : L.length = xs.length := by omega
      cases rest with
      | nil => rfl
      | cons _ _ => simp at *; omega
    subst hrest0
    refine ⟨ins_sorted x L hs, ?_, ⟨[], ?_, fun _ _ _ h => absurd h (by simp)⟩⟩
    · rw [(ins_perm x L).length_eq, List.length_cons, List.length_append, List.length_singleton]; omega
    · rw [List.append_nil] at hperm ⊢
      exact (ins_perm x L).trans ((hperm.cons x).trans (List.perm_append_singleton x xs).symm)
  · rw [if_neg hlt]
    have hLk : L.length = k := by omega
    have hne : L ≠ [] := by intro h'; subst h'; simp at hLk; omega
    obtain ⟨last, hlast⟩ : ∃ last, L.getLast? = some last := ⟨L.getLast hne, List.getLast?_eq_some_getLast hne⟩
    have hsplitL := dropLast_append_getLast L last hlast
    have hlastmin : ∀ y ∈ L, last.2 ≤ y.2 := by
      intro y hy
      rw [← hsplitL] at hy hs
      rcases List.mem_append.mp hy with hy | hy
      · exact (List.pairwise_append.mp hs).2.2 y hy last (List.mem_singleton_self _)
      · rw [List.mem_singleton.mp hy]
    rw [hlast]
    simp only []
    by_cases hbetter : last.2 < x.2
    · rw [if_pos hbetter]
      have hsd : Sorted L.dropLast := by rw [← hsplitL] at hs; exact (List.pairwise_append.mp hs).1
      rw [bubble_eq_ins x _ hsd]
      refine ⟨ins_sorted x _ hsd, ?_, ⟨last :: rest, ?_, ?_⟩⟩
      · rw [(ins_perm x _).length_eq, List.length_cons, List.length_dropLast, List.length_append,
          List.length_singleton]
        omega
      · -- x :: dropLast L ++ last :: rest is xs ++ [x] rearranged
        have h1 : (ins x L.dropLast ++ last :: rest).Perm (x :: (L.dropLast ++ [last] ++ rest)) := by
          rw [List.append_assoc, List.singleton_append]
          exact ((ins_perm x _).append (List.Perm.refl _))
        rw [hsplitL] at h1
        exact h1.trans ((hperm.cons x).trans (List.perm_append_singleton x xs).symm)
      · intro y hy z hz
        rcases List.mem_cons.mp ((ins_perm x _).subset hy) with rfl | hy'
        · rcases List.mem_cons.mp hz with rfl | hz'
          · exact le_of_lt hbetter
          · exact le_trans (hrest last (by rw [← hsplitL]; simp) z hz') (le_of_lt hbetter)
        · have hyL : y ∈ L := by rw [← hsplitL]; exact List.mem_append_left _ hy'
          rcases List.mem_cons.mp hz with rfl | hz'
          · exact hlastmin y hyL
          · exact hrest y hyL z hz'
    · rw [if_neg hbetter]
      refine ⟨hs, by rw [List.length_append, List.length_singleton]; omega, ⟨x :: rest, ?_, ?_⟩⟩
      · exact (List.perm_middle.trans (hperm.cons x)).trans (List.perm_append_singleton x xs).symm
      · intro y hy z hz
        rcases List.mem_cons.mp hz with rfl | hz'
        · exact le_trans (not_lt.mp hbetter) (hlastmin y hy)
        · exact hrest y hy z hz'

/-- **The scan keeps the k best candidates.** -/
theorem topk_correct (k : ℕ) (hk : 1 ≤ k) (xs : List (ι × β)) : TopK k xs (xs.foldl (step k) []) := by
  induction xs using List.reverseRecOn with
  | nil => exact ⟨List.Pairwise.nil, by simp, ⟨[], by simp, fun _ h => absurd h (by simp)⟩⟩
  | append_singleton xs x ih =>
    rw [List.foldl_append, List.foldl_cons, List.foldl_nil]
    exact step_topk k hk xs _ x ih

end Vecindex.TopK
