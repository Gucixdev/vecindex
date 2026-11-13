/-
  The update step of e-means, `tq_em_update_` in `src/tq.c`, one coordinate of one cluster:

      N' = (1 - a) N + a n;   w = a n / N';   c' = c + w (s / n - c)

  where the cluster has mass N at position c, the batch gives it n rows summing to s, and a is the step. The paper
  (Meta FAIR, arXiv 2611.09501) defines the step on a running count and a running sum instead,
  N' = (1 - a) N + a n and S' = (1 - a) S + a s, with the centroid S' / N'. Proved over any ordered field:

  * `pos_eq_ratio`: from c = S / N, the position form gives exactly S' / N' - the C code and the paper agree;
  * `lloyd_step`: with a = 1 the new position is s / n, the plain mean of the cluster's rows - on a batch of every
    row that is one round of Lloyd's k-means;
  * `weight_mem`, `pos_between`: for 0 < a <= 1 and N >= 0 the weight lies in [0, 1], so a step moves the centroid
    towards its batch mean and never past it (`tests/emeans_test.c` checks the same on the f32 code);
  * `mass_pos`: the mass stays positive, so the next step's division is defined;
  * `split_mass`: a revival halves a donor's mass between it and the revived cluster, nothing lost or made.

  The cosine schedule `tq_em_alpha_`: `alpha_mem` - every step lies between a0 / 10 and a0, for any cosine in [-1, 1].

  The C code computes in f32 and f64, so these are facts about the model; the tests compare the code with it within
  rounding. Trusted: Lean's kernel, Mathlib, and the step from this model to the C.
-/
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

namespace Vecindex.Emeans

set_option linter.dupNamespace false

variable {F : Type*} [Field F] [LinearOrder F] [IsStrictOrderedRing F]

/-- The new mass of a cluster: `mm` in `tq_em_update_`. -/
def mass (a N n : F) : F := (1 - a) * N + a * n

/-- How far the centroid moves towards the batch mean: `w` in `tq_em_update_`. -/
def weight (a N n : F) : F := a * n / mass a N n

/-- The new position, the C code's form. -/
def pos (a c N n s : F) : F := c + weight a N n * (s / n - c)

theorem mass_pos {a N n : F} (ha : 0 < a) (ha1 : a ≤ 1) (hN : 0 ≤ N) (hn : 0 < n) : 0 < mass a N n := by
  unfold mass
  have h1 : 0 ≤ (1 - a) * N := mul_nonneg (by linarith) hN
  have h2 : 0 < a * n := mul_pos ha hn
  linarith

/-- The position form equals the paper's sum-and-count form. -/
theorem pos_eq_ratio {a S N n s : F} (ha : 0 < a) (ha1 : a ≤ 1) (hN : 0 < N) (hn : 0 < n) :
    pos a (S / N) N n s = ((1 - a) * S + a * s) / mass a N n := by
  have hm : mass a N n ≠ 0 := (mass_pos ha ha1 hN.le hn).ne'
  have hN0 : N ≠ 0 := hN.ne'
  have hn0 : n ≠ 0 := hn.ne'
  unfold pos weight
  rw [eq_div_iff hm]
  have key : a * n / mass a N n * mass a N n = a * n := div_mul_cancel₀ _ hm
  calc (S / N + a * n / mass a N n * (s / n - S / N)) * mass a N n
      = S / N * mass a N n + a * n / mass a N n * mass a N n * (s / n - S / N) := by ring
    _ = S / N * mass a N n + a * n * (s / n - S / N) := by rw [key]
    _ = (1 - a) * S + a * s := by unfold mass; field_simp; ring

omit [IsStrictOrderedRing F] in
/-- A step of 1 makes the centroid the mean of its batch: one Lloyd round when the batch is every row. -/
theorem lloyd_step {c N n s : F} (hn : 0 < n) : pos 1 c N n s = s / n := by
  have hn0 : n ≠ 0 := hn.ne'
  unfold pos weight mass
  simp [hn0]

theorem weight_mem {a N n : F} (ha : 0 < a) (ha1 : a ≤ 1) (hN : 0 ≤ N) (hn : 0 < n) :
    0 ≤ weight a N n ∧ weight a N n ≤ 1 := by
  have hm := mass_pos ha ha1 hN hn
  unfold weight
  refine ⟨div_nonneg (mul_pos ha hn).le hm.le, ?_⟩
  rw [div_le_one hm]
  unfold mass
  have : 0 ≤ (1 - a) * N := mul_nonneg (by linarith) hN
  linarith

/-- The new position lies between the old one and the batch mean. -/
theorem pos_between {a c N n s : F} (ha : 0 < a) (ha1 : a ≤ 1) (hN : 0 ≤ N) (hn : 0 < n) :
    min c (s / n) ≤ pos a c N n s ∧ pos a c N n s ≤ max c (s / n) := by
  obtain ⟨w0, w1⟩ := weight_mem ha ha1 hN hn
  have hp : pos a c N n s = (1 - weight a N n) * c + weight a N n * (s / n) := by unfold pos; ring
  rw [hp]
  set w := weight a N n
  constructor
  · have h1 : (1 - w) * min c (s / n) ≤ (1 - w) * c := mul_le_mul_of_nonneg_left (min_le_left _ _) (by linarith)
    have h2 : w * min c (s / n) ≤ w * (s / n) := mul_le_mul_of_nonneg_left (min_le_right _ _) w0
    nlinarith
  · have h1 : (1 - w) * c ≤ (1 - w) * max c (s / n) := mul_le_mul_of_nonneg_left (le_max_left _ _) (by linarith)
    have h2 : w * (s / n) ≤ w * max c (s / n) := mul_le_mul_of_nonneg_left (le_max_right _ _) w0
    nlinarith

/-- Revival splits a donor's mass in two halves, one stays, one goes to the revived cluster. -/
theorem split_mass (m : F) : m / 2 + m / 2 = m := add_halves m

/-- The schedule of `tq_em_alpha_`: a0 (0.1 + 0.9 (1 + cos (π f)) / 2), with `cs` for the cosine. -/
def alpha (a0 cs : F) : F := a0 * (1 / 10 + 9 / 10 * ((1 + cs) / 2))

/-- Every step lies between a0 / 10 and a0, for any cosine value in [-1, 1] (the one property of libm's `cos` this
    needs, and the only one trusted here). -/
theorem alpha_mem {a0 cs : F} (h : 0 ≤ a0) (c0 : -1 ≤ cs) (c1 : cs ≤ 1) : a0 / 10 ≤ alpha a0 cs ∧ alpha a0 cs ≤ a0 := by
  unfold alpha
  constructor <;> nlinarith

end Vecindex.Emeans
