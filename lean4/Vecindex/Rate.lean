/-
  The request limit of an API key, `vs_bucket_` in `src/vserv_keys.c`:

      *tokens += (now - *last) * qps;  if (*tokens > qps) *tokens = qps;  *last = now;
      ok = *tokens >= 1.0;  if (ok) *tokens -= 1.0;

  A key starts with `qps` tokens. Proved over the rationals, for any qps >= 0 and request times that never go back
  (CLOCK_MONOTONIC): the tokens stay in [0, qps] (`step_inv`), and of any run of requests from t₁ to t₂ at most
  qps + qps * (t₂ - t₁) get through, whatever came before (`window`). A key that waited 1/qps since its last request
  is let through (`wait_ok`, qps >= 1). The C code computes in f64; `run` checks it against this model on times that
  are multiples of 1/1024 s and integer rates, where every f64 operation is exact, so the two must agree bit for bit.
-/
import Mathlib.Algebra.Order.Field.Rat
import Mathlib.Data.Rat.Defs
import Mathlib.Tactic.Linarith

namespace Vecindex.Rate

set_option linter.dupNamespace false

structure St where
  tokens : ℚ
  last : ℚ

/-- One request at time `now`: let through or not, and the bucket after it. -/
def step (q : ℚ) (s : St) (now : ℚ) : Bool × St :=
  let t := min q (s.tokens + (now - s.last) * q)
  if 1 ≤ t then (true, ⟨t - 1, now⟩) else (false, ⟨t, now⟩)

/-- A run of requests: how many got through, and the bucket after the last. -/
def run (q : ℚ) : St → List ℚ → Nat × St
  | s, [] => (0, s)
  | s, t :: ts =>
    let r := step q s t
    let r' := run q r.2 ts
    ((if r.1 then 1 else 0) + r'.1, r'.2)

/-- The decisions, one a request (for the oracle). -/
def decide' (q : ℚ) : St → List ℚ → List Bool
  | _, [] => []
  | s, t :: ts => (step q s t).1 :: decide' q (step q s t).2 ts

/-- Times that never go back, from `t0` on. -/
def Mono : ℚ → List ℚ → Prop
  | _, [] => True
  | t0, t :: ts => t0 ≤ t ∧ Mono t ts

theorem step_inv (q : ℚ) (hq : 0 ≤ q) (s : St) (now : ℚ) (h0 : 0 ≤ s.tokens) (hl : s.last ≤ now) :
    0 ≤ (step q s now).2.tokens ∧ (step q s now).2.tokens ≤ q := by
  have hm : 0 ≤ (now - s.last) * q := mul_nonneg (by linarith) hq
  unfold step
  simp only
  split_ifs with h
  · exact ⟨by linarith, by have := min_le_left q (s.tokens + (now - s.last) * q); linarith⟩
  · exact ⟨le_min hq (by linarith), min_le_left _ _⟩

theorem step_last (q : ℚ) (s : St) (now : ℚ) : (step q s now).2.last = now := by
  unfold step; simp only; split_ifs <;> rfl

/-- Let through or not, the request and the tokens left add up to the bucket after the refill. -/
theorem step_sum (q : ℚ) (s : St) (now : ℚ) :
    (if (step q s now).1 then (1 : ℚ) else 0) + (step q s now).2.tokens = min q (s.tokens + (now - s.last) * q) := by
  unfold step
  dsimp only
  by_cases h : 1 ≤ min q (s.tokens + (now - s.last) * q)
  · rw [if_pos h]; simp
  · rw [if_neg h]; simp

/-- The time of the last request of a run (`t0` for none). -/
def endT : ℚ → List ℚ → ℚ
  | t0, [] => t0
  | _, t :: ts => endT t ts

theorem run_nonneg (q : ℚ) (hq : 0 ≤ q) : ∀ (ts : List ℚ) (s : St), 0 ≤ s.tokens → Mono s.last ts →
    0 ≤ (run q s ts).2.tokens
  | [], s, h0, _ => by simpa [run] using h0
  | t :: ts, s, h0, hm => by
    simp only [run]
    exact run_nonneg q hq ts _ (step_inv q hq s t h0 hm.1).1 (by rw [step_last]; exact hm.2)

/-- The count and the tokens left never exceed the tokens at the start plus the refill since. -/
theorem run_budget (q : ℚ) (hq : 0 ≤ q) : ∀ (ts : List ℚ) (s : St), 0 ≤ s.tokens → Mono s.last ts →
    ((run q s ts).1 : ℚ) + (run q s ts).2.tokens ≤ s.tokens + (endT s.last ts - s.last) * q
  | [], s, _, _ => by simp [run, endT]
  | t :: ts, s, h0, hm => by
    obtain ⟨hl, hm'⟩ := hm
    have hinv := step_inv q hq s t h0 hl
    have ih := run_budget q hq ts (step q s t).2 hinv.1 (by rw [step_last]; exact hm')
    rw [step_last] at ih
    have hs := step_sum q s t
    have hmin := min_le_right q (s.tokens + (t - s.last) * q)
    simp only [run, endT]
    push_cast
    linarith

/-- Of any run of requests from t₁ to the last, at most qps + qps * (last - t₁) get through, whatever came before. -/
theorem window (q : ℚ) (hq : 0 ≤ q) (s : St) (t₁ : ℚ) (ts : List ℚ) (h0 : 0 ≤ s.tokens) (hl : s.last ≤ t₁)
    (hm : Mono t₁ ts) :
    ((run q s (t₁ :: ts)).1 : ℚ) ≤ q + (endT t₁ ts - t₁) * q := by
  have hinv := step_inv q hq s t₁ h0 hl
  have hb := run_budget q hq ts (step q s t₁).2 hinv.1 (by rw [step_last]; exact hm)
  rw [step_last] at hb
  have hfin := run_nonneg q hq ts _ hinv.1 (by rw [step_last]; exact hm)
  have hs := step_sum q s t₁
  have hmin := min_le_left q (s.tokens + (t₁ - s.last) * q)
  simp only [run]
  push_cast
  linarith

/-- A key with qps >= 1 that waited 1/qps since its last request is let through. -/
theorem wait_ok (q : ℚ) (hq : 1 ≤ q) (s : St) (now : ℚ) (h0 : 0 ≤ s.tokens) (hw : s.last + 1 / q ≤ now) :
    (step q s now).1 = true := by
  have hq0 : 0 < q := by linarith
  have : 1 ≤ (now - s.last) * q := by
    have h := mul_le_mul_of_nonneg_right (show 1 / q ≤ now - s.last by linarith) hq0.le
    rwa [one_div, inv_mul_cancel₀ hq0.ne'] at h
  unfold step
  simp only
  rw [if_pos (le_min hq (by linarith))]

end Vecindex.Rate
