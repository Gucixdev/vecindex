/-
  The namespace set of a search over several namespaces, `tq_in_set_` in `src/tq.c` (used by the scan in
  `src/tq_scan.c` and by `store_cnf_ok_` in `src/store.c`):

      const i64 *b = set; i32 len = n;
      while (len > 1) { i32 h = len >> 1; b += (b[h - 1] < v) ? h : 0; len -= h; }
      return *b == v;

  a lower bound without a branch on the data. Proved: on a set sorted ascending without repeats (what
  `store_search_nss` builds: qsort, then repeats dropped) and n >= 1, it answers exactly whether v is in the set
  (`inSet_iff`), and it reads only indices below n (`go_lt`: the final b is inside the set).
-/
import Mathlib.Tactic.Linarith

namespace Vecindex.NsSet

set_option linter.dupNamespace false

/-- The loop: from offset b with len elements left, the offset it ends at. -/
def go (a : List Int) (v : Int) (b len : Nat) : Nat :=
  if h : len > 1 then go a v (b + if a.getD (b + len / 2 - 1) 0 < v then len / 2 else 0) (len - len / 2) else b
termination_by len
decreasing_by omega

/-- `tq_in_set_`. -/
def inSet (a : List Int) (v : Int) : Bool := a.getD (go a v 0 a.length) 0 == v

/-- Ascending without repeats. -/
def Strict (a : List Int) : Prop := ∀ i j, i < j → j < a.length → a.getD i 0 < a.getD j 0

theorem go_lt (a : List Int) (v : Int) (b len : Nat) (hl : 1 ≤ len) : go a v b len < b + len := by
  induction len using Nat.strong_induction_on generalizing b with
  | _ len ih =>
    unfold go
    split
    · rename_i h
      by_cases hc : a.getD (b + len / 2 - 1) 0 < v
      · simp only [if_pos hc]
        have := ih (len - len / 2) (by omega) (b + len / 2) (by omega)
        omega
      · simp only [if_neg hc, Nat.add_zero]
        have := ih (len - len / 2) (by omega) b (by omega)
        omega
    · omega

/-- The invariant: everything before b is below v, and v, if it is there at all, is in [b, b + len). -/
theorem go_spec (a : List Int) (v : Int) (hs : Strict a) (b len : Nat) (hl : 1 ≤ len) (hb : b + len ≤ a.length)
    (hlow : ∀ i, i < b → a.getD i 0 < v) (hin : ∀ i, i < a.length → a.getD i 0 = v → b ≤ i ∧ i < b + len) :
    ∀ i, i < a.length → a.getD i 0 = v → i = go a v b len := by
  induction len using Nat.strong_induction_on generalizing b with
  | _ len ih =>
    intro i hi hv
    unfold go
    split
    · rename_i h
      have hmid : b + len / 2 - 1 < a.length := by omega
      by_cases hc : a.getD (b + len / 2 - 1) 0 < v
      · simp only [if_pos hc]
        refine ih (len - len / 2) (by omega) (b + len / 2) (by omega) (by omega) ?_ ?_ i hi hv
        · intro j hj
          by_cases hjb : j < b
          · exact hlow j hjb
          · rcases Nat.lt_or_ge j (b + len / 2 - 1) with h1 | h1
            · exact lt_trans (hs j _ h1 hmid) hc
            · have : j = b + len / 2 - 1 := by omega
              rw [this]; exact hc
        · intro j hj hjv
          have := hin j hj hjv
          refine ⟨?_, by omega⟩
          by_contra hlt
          have : a.getD j 0 < v := by
            rcases Nat.lt_or_ge j (b + len / 2 - 1) with h1 | h1
            · exact lt_trans (hs j _ h1 hmid) hc
            · have : j = b + len / 2 - 1 := by omega
              rw [this]; exact hc
          omega
      · simp only [if_neg hc, Nat.add_zero]
        refine ih (len - len / 2) (by omega) b (by omega) (by omega) hlow ?_ i hi hv
        intro j hj hjv
        have := hin j hj hjv
        refine ⟨this.1, ?_⟩
        by_contra hge
        have : a.getD (b + len / 2 - 1) 0 < a.getD j 0 := hs _ j (by omega) hj
        omega
    · have := hin i hi hv
      omega

/-- On a set sorted ascending without repeats, `tq_in_set_` is membership. -/
theorem inSet_iff (a : List Int) (v : Int) (hs : Strict a) (hn : 1 ≤ a.length) :
    inSet a v = true ↔ ∃ i, i < a.length ∧ a.getD i 0 = v := by
  unfold inSet
  constructor
  · intro h
    refine ⟨go a v 0 a.length, by have := go_lt a v 0 a.length hn; omega, ?_⟩
    simpa using h
  · rintro ⟨i, hi, hv⟩
    have := go_spec a v hs 0 a.length hn (by omega) (fun j hj => absurd hj (Nat.not_lt_zero _))
      (fun j hj _ => ⟨Nat.zero_le _, by omega⟩) i hi hv
    rw [← this, hv]
    simp

end Vecindex.NsSet
