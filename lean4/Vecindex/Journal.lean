/-
  The journal's frames and the replica's copy of them: `vs_rjtrim_` in `src/vserv_repl.c` (the walk the replica does
  when it starts) and the frame `store_jframe_` writes in `src/store_batch.c`.

  A journal is a 20-byte header (magic, dimension, the stamp of its file, a CRC-32C of the three) and then frames,
  each `'G' | u32 length | u32 CRC-32C of the payload | payload`. The walk reads frames while one is whole: 9 bytes,
  'G', a length up to `STORE_BATCH_MAX` (2^30) that the bytes left cover, and a CRC that matches; the good part ends
  before the first frame that is not. The replica then cuts its journal there.

  The replica copies the primary's journal byte for byte, so after a crash of the replica (its process killed in a
  write, the bytes of one chunk landed partly) its journal is a prefix of the primary's: `take r J`. Proved: a journal
  of whole frames is kept whole (`walk_frames`), and for any r the walk of `take r J` keeps exactly the frames that
  arrived whole and nothing of the torn one (`walk_take`), so what the replica keeps is the primary's header and its
  first k frames (`good_take`, `trim_take`): replaying it applies a prefix of the primary's changes, never a piece of a
  record. Appending the rest of the primary's journal from that point gives the primary's journal back (`resume`).

  The checksum is a parameter: any function into u32 (the oracle in `run` uses `Crc.crc32c`, which is `store_crc32c_`).
  That a damaged payload is caught is CRC-32C's business (any change of up to 32 consecutive bits), not proved here.
  Assumed, as in the C code: a crash leaves a prefix (ext4's and XFS's ordered data: no garbage past the bytes written).
-/
import Mathlib.Data.List.Basic

namespace Vecindex.Journal

set_option linter.dupNamespace false

/-- `STORE_BATCH_MAX`. -/
def bmax : Nat := 2 ^ 30

/-- The four little-endian bytes of a u32. -/
def le32 (n : Nat) : List Nat := [n % 256, n / 256 % 256, n / 65536 % 256, n / 16777216 % 256]

def de32 (a b c d : Nat) : Nat := a + 256 * b + 65536 * c + 16777216 * d

theorem de32_le32 (n : Nat) (h : n < 2 ^ 32) :
    de32 (n % 256) (n / 256 % 256) (n / 65536 % 256) (n / 16777216 % 256) = n := by
  unfold de32; omega

/-- One frame of `store_jframe_`, around a payload. -/
def frame (crc : List Nat → Nat) (p : List Nat) : List Nat := 71 :: (le32 p.length ++ le32 (crc p) ++ p)

/-- Frames one after another. -/
def frames (crc : List Nat → Nat) : List (List Nat) → List Nat
  | [] => []
  | p :: ps => frame crc p ++ frames crc ps

/-- The loop of `vs_rjtrim_` after the header: how many bytes of whole frames there are (fuel: at most that many
    frames; the length of the input is always enough). -/
def walk (crc : List Nat → Nat) : Nat → List Nat → Nat
  | 0, _ => 0
  | f + 1, b =>
    match b with
    | 71 :: a0 :: a1 :: a2 :: a3 :: c0 :: c1 :: c2 :: c3 :: tl =>
      let n := de32 a0 a1 a2 a3
      if n ≤ bmax ∧ n ≤ tl.length ∧ crc (tl.take n) = de32 c0 c1 c2 c3 then 9 + n + walk crc f (tl.drop n) else 0
    | _ => 0

/-- The whole of `vs_rjtrim_`: the header (checked by `head`, which sees its 20 bytes) and the frames after it. -/
def good (crc : List Nat → Nat) (head : List Nat → Bool) (b : List Nat) : Nat :=
  if b.length ≥ 20 ∧ head (b.take 20) then 20 + walk crc b.length (b.drop 20) else 0

/-- Bytes of whole frames within the first r bytes of `frames ps`, and how many frames that is. -/
def fit : List (List Nat) → Nat → Nat
  | [], _ => 0
  | p :: ps, r => if 9 + p.length ≤ r then 9 + p.length + fit ps (r - (9 + p.length)) else 0

def cnt : List (List Nat) → Nat → Nat
  | [], _ => 0
  | p :: ps, r => if 9 + p.length ≤ r then cnt ps (r - (9 + p.length)) + 1 else 0

variable (crc : List Nat → Nat)

theorem frame_length (p : List Nat) : (frame crc p).length = 9 + p.length := by
  simp [frame, le32]; omega

theorem frames_length_ge (ps : List (List Nat)) : ps.length ≤ (frames crc ps).length := by
  induction ps with
  | nil => simp [frames]
  | cons p ps ih => simp only [frames, List.length_append, frame_length, List.length_cons]; omega

/-- The walk over one whole frame and what follows it: the frame is taken, the walk goes on after it. -/
theorem walk_frame (hc : ∀ q, crc q < 2 ^ 32) (p rest : List Nat) (hp : p.length ≤ bmax) (f : Nat) :
    walk crc (f + 1) (frame crc p ++ rest) = 9 + p.length + walk crc f rest := by
  have hl : p.length < 2 ^ 32 := by unfold bmax at hp; omega
  simp only [frame, le32, List.cons_append, List.nil_append, walk]
  rw [de32_le32 _ hl, de32_le32 _ (hc p)]
  simp [hp, List.take_left', List.drop_left']

/-- A journal of whole frames is kept whole. -/
theorem walk_frames (hc : ∀ q, crc q < 2 ^ 32) (ps : List (List Nat)) (hp : ∀ p ∈ ps, p.length ≤ bmax)
    (f : Nat) (hf : ps.length ≤ f) : walk crc f (frames crc ps) = (frames crc ps).length := by
  induction ps generalizing f with
  | nil => cases f <;> simp [frames, walk]
  | cons p ps ih =>
    simp only [List.length_cons] at hf
    obtain ⟨g, rfl⟩ : ∃ g, f = g + 1 := ⟨f - 1, by omega⟩
    simp only [frames]
    rw [walk_frame crc hc p _ (hp p (List.mem_cons_self ..)), ih (fun q hq => hp q (List.mem_cons_of_mem _ hq)) g
      (by omega)]
    simp [frame_length]

/-- The torn frame: fewer bytes of a frame than the whole are never taken. -/
theorem walk_torn (p : List Nat) (hp : p.length ≤ bmax) (r : Nat) (hr : r < 9 + p.length) (f : Nat) :
    walk crc f ((frame crc p).take r) = 0 := by
  cases f with
  | zero => rfl
  | succ g =>
    by_cases h9 : r < 9
    · rcases r with _ | _ | _ | _ | _ | _ | _ | _ | _ | r <;> first | omega | simp [frame, le32, walk]
    · obtain ⟨s, rfl⟩ : ∃ s, r = s + 9 := ⟨r - 9, by omega⟩
      have hl : p.length < 2 ^ 32 := by unfold bmax at hp; omega
      simp only [frame, le32, List.cons_append, List.nil_append]
      simp only [List.take_succ_cons, walk]
      rw [de32_le32 _ hl]
      simp only [List.length_take]
      rw [if_neg (fun h => by have := h.2.1; omega)]

/-- Any prefix of a journal of whole frames: the walk keeps exactly the frames that are whole in it. -/
theorem walk_take (hc : ∀ q, crc q < 2 ^ 32) (ps : List (List Nat)) (hp : ∀ p ∈ ps, p.length ≤ bmax)
    (r f : Nat) (hf : min r (frames crc ps).length ≤ f) : walk crc f ((frames crc ps).take r) = fit ps r := by
  induction ps generalizing r f with
  | nil => cases f <;> simp [frames, walk, fit]
  | cons p ps ih =>
    have hp0 := hp p (List.mem_cons_self ..)
    simp only [frames, fit]
    simp only [frames, List.length_append, frame_length] at hf
    split_ifs with h
    · obtain ⟨g, rfl⟩ : ∃ g, f = g + 1 := ⟨f - 1, by omega⟩
      rw [List.take_append, List.take_of_length_le (by rw [frame_length]; omega), frame_length,
        walk_frame crc hc p _ hp0, ih (fun q hq => hp q (List.mem_cons_of_mem _ hq)) _ g (by omega)]
    · rw [List.take_append_of_le_length (by rw [frame_length]; omega)]
      exact walk_torn crc p hp0 r (by omega) f

/-- The frames that fit are a prefix of the frames: the first `cnt` of them. -/
theorem fit_eq (ps : List (List Nat)) (r : Nat) : (frames crc ps).take (fit ps r) = frames crc (ps.take (cnt ps r)) := by
  induction ps generalizing r with
  | nil => simp [frames, fit, cnt]
  | cons p ps ih =>
    simp only [fit, cnt]
    split_ifs with h
    · simp only [frames, List.take_succ_cons]
      rw [List.take_append, List.take_of_length_le (by rw [frame_length]; omega), frame_length,
        show 9 + p.length + fit ps (r - (9 + p.length)) - (9 + p.length) = fit ps (r - (9 + p.length)) by omega, ih]
    · simp [frames]

theorem fit_le (ps : List (List Nat)) (r : Nat) : fit ps r ≤ r := by
  induction ps generalizing r with
  | nil => simp [fit]
  | cons p ps ih => simp only [fit]; split_ifs <;> [have := ih (r - (9 + p.length)); skip] <;> omega

/-- The replica's journal after a crash, `take r` of the primary's (header h, frames ps): what the walk keeps. -/
theorem good_take (hc : ∀ q, crc q < 2 ^ 32) (head : List Nat → Bool) (h : List Nat) (hh : h.length = 20)
    (hok : head h) (ps : List (List Nat)) (hp : ∀ p ∈ ps, p.length ≤ bmax) (r : Nat) :
    good crc head ((h ++ frames crc ps).take r) = if r < 20 then 0 else 20 + fit ps (r - 20) := by
  have hlen : ((h ++ frames crc ps).take r).length = min r (20 + (frames crc ps).length) := by
    simp [List.length_take, hh]
  have hhead : 20 ≤ r → ((h ++ frames crc ps).take r).take 20 = h := by
    intro hr
    rw [List.take_take, show min 20 r = 20 by omega, List.take_left' hh]
  have hrest : ((h ++ frames crc ps).take r).drop 20 = (frames crc ps).take (r - 20) := by
    rw [List.drop_take, List.drop_left' hh]
  unfold good
  by_cases hr : r < 20
  · rw [if_neg (by rw [hlen]; omega), if_pos hr]
  · rw [if_pos ⟨by rw [hlen]; have := frames_length_ge crc ps; omega, by rw [hhead (by omega)]; exact hok⟩, if_neg hr,
      hrest, walk_take crc hc ps hp _ _ (by rw [hlen]; omega)]

/-- What the replica keeps after cutting: the primary's header and its first k frames, whole. -/
theorem trim_take (hc : ∀ q, crc q < 2 ^ 32) (head : List Nat → Bool) (h : List Nat) (hh : h.length = 20)
    (hok : head h) (ps : List (List Nat)) (hp : ∀ p ∈ ps, p.length ≤ bmax) (r : Nat) (hr : 20 ≤ r) :
    ((h ++ frames crc ps).take r).take (good crc head ((h ++ frames crc ps).take r))
      = h ++ frames crc (ps.take (cnt ps (r - 20))) := by
  rw [good_take crc hc head h hh hok ps hp r, if_neg (by omega), List.take_take,
    show min (20 + fit ps (r - 20)) r = 20 + fit ps (r - 20) by have := fit_le ps (r - 20); omega,
    List.take_append, hh, List.take_of_length_le (by omega),
    show 20 + fit ps (r - 20) - 20 = fit ps (r - 20) by omega, fit_eq]

/-- From the cut, the rest of the primary's journal makes the primary's journal again: the replica resumes at
    offset `good` and loses nothing. -/
theorem resume (b : List Nat) (g r : Nat) (hg : g ≤ r) :
    (b.take r).take g ++ b.drop g = b := by
  rw [List.take_take, show min g r = g by omega, List.take_append_drop]

end Vecindex.Journal
