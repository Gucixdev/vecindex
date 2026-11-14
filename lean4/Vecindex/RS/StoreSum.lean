/-
  `store_sum.c` end to end: GF(2^8) from its tables, its Cauchy coefficients, its rebuild, and its file layout.

  * `recover_gf`: for every k and m that `ssum_layout_` accepts (1 <= k <= 128, m <= 64), every group of at most k
    data blocks, every choice of at most m damaged blocks among its data and parity blocks, the rebuild returns the
    damaged data blocks exactly. Which blocks are damaged comes from their BLAKE3 digests; a damaged block whose
    digest still matches would be a BLAKE3 collision, which this model does not cover.
  * `layout_*`: the blocks of a file are dealt into groups without gaps or overlaps, a group never holds more than k
    data blocks (the C arrays `md[128]`, `mp[64]`, `use[64]` and `a[64 * 64]` are large enough), a run of damaged
    neighbouring blocks puts at most ceil(L / G) of them into one group, and the tables and trailers do not overlap.
-/
import Vecindex.RS.Group
import Vecindex.RS.Cauchy
import Vecindex.GF256.Field

open Finset

namespace Vecindex.RS

/-- `gf_coef_(k, p, i)` as a field element. -/
def coefF (k p i : ℕ) : GF256 := ⟨GF256.coef k p i, GF256.inv_lt _⟩

theorem xor_ne_zero_of_ne {a b : ℕ} (h : a ≠ b) : a ^^^ b ≠ 0 := by
  intro hx
  apply h
  calc a = (a ^^^ b) ^^^ b := by rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
    _ = b := by rw [hx, Nat.zero_xor]

/-- The coefficient is the Cauchy entry `1 / (x_p - y_i)` with `x_p = k + p`, `y_i = i`. -/
theorem coefF_eq (k p i : ℕ) (hkp : k + p < 256) (hi : i < k) :
    coefF k p i = ((⟨k + p, hkp⟩ : GF256) - ⟨i, by omega⟩)⁻¹ := by
  have hne : (k + p) ^^^ i ≠ 0 := xor_ne_zero_of_ne (by omega)
  have hv : ((⟨k + p, hkp⟩ : GF256) - ⟨i, by omega⟩).val = (k + p) ^^^ i := rfl
  apply GF256.ext
  rw [GF256.val_inv _ (by rw [hv]; exact hne), hv]
  simp only [coefF, GF256.coef]

/-- Every square part of the coefficient matrix has no kernel. -/
theorem coef_mds (k m ng : ℕ) (hkm : k + m ≤ 256) (hng : ng ≤ k) {t : ℕ} (rows cols : Fin t → ℕ)
    (hr : Function.Injective rows) (hc : Function.Injective cols) (hrl : ∀ q, rows q < m)
    (hcl : ∀ p, cols p < ng) (v : Fin t → GF256) (hv : ∀ q, ∑ p, coefF k (rows q) (cols p) * v p = 0) :
    v = 0 := by
  have hx : ∀ q, k + rows q < 256 := fun q => by have := hrl q; omega
  have hy : ∀ p, cols p < 256 := fun p => by have := hcl p; omega
  apply cauchy_kernel (fun q => (⟨k + rows q, hx q⟩ : GF256)) (fun p => (⟨cols p, hy p⟩ : GF256))
  · intro q q' h
    apply hr
    have := congrArg GF256.val h
    simp only at this
    omega
  · intro p p' h
    apply hc
    exact congrArg GF256.val h
  · intro q p h
    have := congrArg GF256.val h
    simp only at this
    have := hcl p
    omega
  · intro q
    rw [← hv q]
    refine sum_congr rfl fun p _ => ?_
    rw [coefF_eq k (rows q) (cols p) (hx q) (by have := hcl p; omega), mul_comm]

/-- **Any m damaged blocks of a group come back.** -/
theorem recover_gf (k m ng : ℕ) (hk : k ≤ 128) (hm : m ≤ 64) (hng : ng ≤ k)
    (d par : ℕ → GF256) (md mp : List ℕ)
    (hpar : ∀ p, p < m → par p = ∑ i ∈ range ng, coefF k p i * d i)
    (hmd : md.Nodup) (hmdlt : ∀ i ∈ md, i < ng) (hmp : mp.Nodup) (hmplt : ∀ p ∈ mp, p < m)
    (hcount : md.length + mp.length ≤ m) :
    ∃ x, recover (coefF k) m ng d par md mp = some x ∧ ∀ p, x p = d (md.get p) :=
  recover_eq (coefF k) m ng d par md mp hpar hmd hmdlt hmp hmplt hcount
    (fun rows cols hr hc hrl hcl v hv => coef_mds k m ng (by omega) hng rows cols hr hc hrl hcl v hv)

/-! ## The layout of `ssum_layout_` -/

/-- `SSUM_AL`: up to the next multiple of 4096. -/
def al (x : ℕ) : ℕ := (x + 4095) / 4096 * 4096

/-- `ssum_layout_(z, D, blog, k, m)` for the arguments it accepts: blocks of `2 ^ blog` bytes, `N` data blocks,
    `G` groups, `P` parity blocks, then table A at `ta`, trailer A at `tra`, table B at `tb`, trailer B last. -/
structure Layout where
  B : ℕ
  N : ℕ
  G : ℕ
  P : ℕ
  ta : ℕ
  T : ℕ
  tra : ℕ
  tb : ℕ
  fsize : ℕ

def layout (D blog k m : ℕ) : Layout :=
  let B := 2 ^ blog
  let N := (D + B - 1) / B
  let G := (N + k - 1) / k
  let P := G * m
  let ta := (N + P) * B
  let T := (N + P) * 32
  let tra := al (ta + T)
  let tb := al (tra + 128)
  { B, N, G, P, ta, T, tra, tb, fsize := tb + T + 128 }

/-- The data blocks of group `g`: `ng = (N - g + G - 1) / G` of them, at `g, g + G, g + 2G, ...`. -/
def ngOf (N G g : ℕ) : ℕ := (N - g + G - 1) / G

theorem layout_G_mul (D blog k m : ℕ) (hk : 1 ≤ k) : (layout D blog k m).N ≤ k * (layout D blog k m).G := by
  simp only [layout]
  have := Nat.lt_mul_div_succ ((D + 2 ^ blog - 1) / 2 ^ blog) (show 0 < k by omega)
  have h2 : ((D + 2 ^ blog - 1) / 2 ^ blog + k - 1) / k * k + k > (D + 2 ^ blog - 1) / 2 ^ blog + k - 1 :=
    Nat.lt_div_mul_add (by omega)
  rw [Nat.mul_comm]
  omega

/-- A group never holds more than `k` data blocks: `md[128]` is large enough. -/
theorem layout_ng_le (D blog k m g : ℕ) (hk : 1 ≤ k) (hD : 0 < D) :
    ngOf (layout D blog k m).N (layout D blog k m).G g ≤ k := by
  have hNG := layout_G_mul D blog k m hk
  set L := layout D blog k m
  have hN : 0 < L.N := by
    simp only [L, layout]
    apply Nat.div_pos _ (Nat.two_pow_pos blog)
    have := Nat.one_le_two_pow (n := blog)
    omega
  have hG : 0 < L.G := by
    rcases Nat.eq_zero_or_pos L.G with h | h
    · rw [h] at hNG; omega
    · exact h
  unfold ngOf
  rw [Nat.div_le_iff_le_mul_add_pred hG]
  have : L.N - g + L.G - 1 ≤ L.N + L.G - 1 := by omega
  calc L.N - g + L.G - 1 ≤ L.N + L.G - 1 := this
    _ ≤ k * L.G + (L.G - 1) := by omega
    _ = L.G * k + (L.G - 1) := by rw [Nat.mul_comm]

/-- Every data block `b < N` is block `b / G` of group `b % G`, and that index is in range. -/
theorem layout_cover (N G b : ℕ) (hG : 0 < G) (hb : b < N) :
    b % G < G ∧ b / G < ngOf N G (b % G) ∧ b % G + b / G * G = b := by
  refine ⟨Nat.mod_lt _ hG, ?_, by rw [Nat.mul_comm]; exact Nat.mod_add_div b G⟩
  unfold ngOf
  rw [Nat.lt_div_iff_mul_lt hG]
  have := Nat.mod_add_div b G
  have : b / G * G = b - b % G := by rw [Nat.mul_comm]; omega
  have hm := Nat.mod_lt b hG
  omega

/-- ... and every slot of a group is a data block: no gaps, no overlaps. -/
theorem layout_slot (N G g i : ℕ) (hG : 0 < G) (hi : i < ngOf N G g) : g + i * G < N := by
  unfold ngOf at hi
  rw [Nat.lt_div_iff_mul_lt hG] at hi
  omega

/-- A run of `L` neighbouring blocks puts at most `ceil(L / G)` of them into any group, so a run of up to `m * G`
    damaged data blocks is rebuilt (with the parity intact). -/
theorem layout_burst (G g s L : ℕ) (hG : 0 < G) :
    ((range (s + L)).filter (fun b => s ≤ b ∧ b % G = g % G)).card ≤ (L + G - 1) / G := by
  set S := (range (s + L)).filter (fun b => s ≤ b ∧ b % G = g % G)
  -- two blocks of one group are at least G apart, so b ↦ (b - s) / G is injective on S
  have key : ∀ b ∈ S, ∀ b' ∈ S, b < b' → (b - s) / G < (b' - s) / G := by
    intro b hb b' hb' hlt
    simp only [S, mem_filter, mem_range] at hb hb'
    have hmod : (b' - b) % G = 0 := Nat.sub_mod_eq_zero_of_mod_eq (by rw [hb.2.2, hb'.2.2])
    have hdvd : G ≤ b' - b := Nat.le_of_dvd (by omega) (Nat.dvd_of_mod_eq_zero hmod)
    calc (b - s) / G < (b - s) / G + 1 := Nat.lt_succ_self _
      _ = (b - s + G) / G := (Nat.add_div_right _ hG).symm
      _ ≤ (b' - s) / G := Nat.div_le_div_right (by omega)
  have hmaps : ∀ b ∈ S, (b - s) / G ∈ range ((L + G - 1) / G) := by
    intro b hb
    simp only [S, mem_filter, mem_range] at hb ⊢
    have h1 : (b - s) / G ≤ (L - 1) / G := Nat.div_le_div_right (by omega)
    have h2 : (L - 1 + G) / G = (L - 1) / G + 1 := Nat.add_div_right _ hG
    have h3 : L + G - 1 = L - 1 + G := by omega
    rw [h3, h2]
    omega
  calc S.card ≤ (range ((L + G - 1) / G)).card := by
        apply card_le_card_of_injOn (fun b => (b - s) / G) (fun b hb => hmaps b hb)
        intro b hb b' hb' heq
        rcases lt_trichotomy b b' with h | h | h
        · exact absurd heq (ne_of_lt (key b hb b' hb' h))
        · exact h
        · exact absurd heq.symm (ne_of_lt (key b' hb' b hb h))
    _ = (L + G - 1) / G := card_range _

/-- A run of at most m G neighbouring blocks puts at most m of them into any one group, so with m parity blocks a
group every such run of damaged data blocks is rebuilt (`recover_gf`). `tests/battle_damage.c` destroys a run of
exactly m G data blocks and expects the file back byte for byte. -/
theorem burst_within_parity (G g s L m : ℕ) (hG : 0 < G) (hL : L ≤ m * G) :
    ((range (s + L)).filter (fun b => s ≤ b ∧ b % G = g % G)).card ≤ m := by
  have h : (L + G - 1) / G < m + 1 := by
    rw [Nat.div_lt_iff_lt_mul hG]
    have : (m + 1) * G = m * G + G := by ring
    omega
  exact (layout_burst G g s L hG).trans (Nat.lt_succ_iff.mp h)

/-- One block more and the run is beyond the parity: m G + 1 neighbouring blocks put m + 1 of them (s, s + G, ...,
s + m G) into the group of the first one. `tests/battle_damage.c` destroys such a run and expects a refusal. -/
theorem burst_beyond_parity (G s m : ℕ) (hG : 0 < G) :
    m + 1 ≤ ((range (s + (m * G + 1))).filter (fun b => s ≤ b ∧ b % G = s % G)).card := by
  have hsub : (range (m + 1)).image (fun i => s + i * G)
      ⊆ (range (s + (m * G + 1))).filter (fun b => s ≤ b ∧ b % G = s % G) := by
    intro b hb
    simp only [mem_image, mem_range] at hb
    obtain ⟨i, hi, rfl⟩ := hb
    simp only [mem_filter, mem_range]
    have him : i * G ≤ m * G := Nat.mul_le_mul_right G (Nat.lt_succ_iff.mp hi)
    exact ⟨Nat.add_lt_add_left (Nat.lt_succ_of_le him) s, Nat.le_add_right s (i * G),
      Nat.add_mul_mod_self_right s i G⟩
  have hinj : Set.InjOn (fun i => s + i * G) (range (m + 1)) := by
    intro i _ j _ h
    exact Nat.eq_of_mul_eq_mul_right hG (Nat.add_left_cancel h)
  calc m + 1 = ((range (m + 1)).image (fun i => s + i * G)).card := by
        rw [card_image_of_injOn hinj, card_range]
    _ ≤ _ := card_le_card hsub

/-- The two tables and two trailers follow the blocks without overlapping, and trailer B is the last 128 bytes. -/
theorem layout_regions (D blog k m : ℕ) :
    let L := layout D blog k m
    L.ta + L.T ≤ L.tra ∧ L.tra + 128 ≤ L.tb ∧ L.tb + L.T + 128 = L.fsize ∧ L.tra % 4096 = 0 ∧ L.tb % 4096 = 0 := by
  intro L
  have htra : L.tra = al (L.ta + L.T) := rfl
  have htb : L.tb = al (L.tra + 128) := rfl
  refine ⟨?_, ?_, rfl, ?_, ?_⟩
  · rw [htra]; unfold al; omega
  · rw [htb]; unfold al; omega
  · rw [htra]; unfold al; omega
  · rw [htb]; unfold al; omega

end Vecindex.RS
