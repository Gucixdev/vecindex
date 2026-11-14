/-
  A second implementation of three C kernels, written as TrustLean MicroC programs, proved against the same Lean
  definitions as the C code, and printed as C by TrustLean's MicroC printer (`lean4/gen/trustlean.c`, made by
  `lake exe vecindex-oracle gen`). The differential test (`lean4/run`) runs vecindex's C and the printed C on every
  input of the kernel and compares.

      gf_init_ step      x <<= 1; if (x & 0x100u) x ^= 0x11du;                 = GF256.xt
      CRC-32C byte       crc ^= byte; 8 x (c & 1u ? (c >> 1) ^ 0x82f63b78u : c >> 1)   = Crc.iter bit 8
      tq_put_lane_ byte  (q & 0xf0) | cd  or  (q & 0x0f) | (cd << 4)                = FastScan.putNib

  The semantics is TrustLean's `evalMicroC_uint32`: every operation wraps to 32 bits, as on `uint32_t` in C (the
  shifts here are by 1 and 4, well below 32). What is trusted: that the printed text, compiled by gcc, means what
  `evalMicroC_uint32` says. The differential test checks that on every input. TrustLean's printer-parser round-trip
  theorem does not build for the bitwise operators at the pinned commit, so it is not used.
-/
import TrustLean.MicroC.UnsignedEval
import TrustLean.MicroC.PrettyPrint
import Vecindex.GF256.Tables
import Vecindex.Crc32c
import Vecindex.FastScan

open TrustLean

namespace Vecindex.CodeGen

def var (s : String) : MicroCExpr := .varRef s
def lit (n : Int) : MicroCExpr := .litInt n
def bin (op : MicroCBinOp) (a b : MicroCExpr) : MicroCExpr := .binOp op a b

/-! ## The programs -/

/-- `x <<= 1; if (x & 0x100u) x ^= 0x11du;` -/
def xtimeProg : MicroCStmt :=
  .seq (.assign "x" (bin .bshl (var "x") (lit 1)))
    (.ite (bin .ltOp (lit 0) (bin .band (var "x") (lit 256)))
      (.assign "x" (bin .bxor (var "x") (lit 285)))
      .skip)

/-- `c = c & 1u ? (c >> 1) ^ 0x82f63b78u : c >> 1;` -/
def bitProg : MicroCStmt :=
  .ite (bin .eqOp (bin .band (var "c") (lit 1)) (lit 1))
    (.assign "c" (bin .bxor (bin .bshr (var "c") (lit 1)) (lit 0x82f63b78)))
    (.assign "c" (bin .bshr (var "c") (lit 1)))

def rep : ℕ → MicroCStmt → MicroCStmt
  | 0, _ => .skip
  | n + 1, s => .seq s (rep n s)

/-- One byte of CRC-32C, bit at a time. -/
def crcByteProg : MicroCStmt := .seq (.assign "c" (bin .bxor (var "c") (var "b"))) (rep 8 bitProg)

/-- One byte of `tq_put_lane_`. -/
def putNibProg : MicroCStmt :=
  .ite (bin .ltOp (var "k") (lit 16))
    (.assign "q" (bin .bor (bin .band (var "q") (lit 240)) (var "cd")))
    (.assign "q" (bin .bor (bin .band (var "q") (lit 15)) (bin .bshl (var "cd") (lit 4))))

/-! ## Evaluation helpers -/

theorem wrap_nat (n : ℕ) (h : n < 2 ^ 32) : wrapUInt32 (n : Int) = n := by
  unfold wrapUInt32 wrapWidth
  exact Int.emod_eq_of_lt (Int.natCast_nonneg n) (by exact_mod_cast h)

theorem upd_same (env : MicroCEnv) (n : String) (v : Value) : env.update n v n = v := by
  simp [MicroCEnv.update]

theorem upd_other (env : MicroCEnv) (n m : String) (v : Value) (h : m ≠ n) : env.update n v m = env m := by
  simp [MicroCEnv.update, h]

theorem eval_assign (f : ℕ) (env : MicroCEnv) (n : String) (e : MicroCExpr) (v : Value)
    (h : evalMicroCExpr_uint32 env e = some v) :
    evalMicroC_uint32 f env (.assign n e) = some (.normal, env.update n v) := by
  simp [evalMicroC_uint32, h]

theorem eval_seq (f : ℕ) (env env1 : MicroCEnv) (s1 s2 : MicroCStmt)
    (h : evalMicroC_uint32 f env s1 = some (.normal, env1)) :
    evalMicroC_uint32 f env (.seq s1 s2) = evalMicroC_uint32 f env1 s2 := by
  simp [evalMicroC_uint32, h]

theorem eval_ite (f : ℕ) (env : MicroCEnv) (c : MicroCExpr) (s1 s2 : MicroCStmt) (b : Bool)
    (h : evalMicroCExpr_uint32 env c = some (.bool b)) :
    evalMicroC_uint32 f env (.ite c s1 s2) = if b then evalMicroC_uint32 f env s1 else evalMicroC_uint32 f env s2 := by
  cases b <;> simp [evalMicroC_uint32, h]

/-! ## gf_init_'s step -/

theorem xtime_correct (env : MicroCEnv) (x : ℕ) (hx : x < 256) (h : env "x" = .int x) :
    ∃ env', evalMicroC_uint32 0 env xtimeProg = some (.normal, env') ∧ env' "x" = .int (GF256.xt x) := by
  have hs : x <<< 1 < 2 ^ 32 := by rw [Nat.shiftLeft_eq]; omega
  have e1 : evalMicroCExpr_uint32 env (bin .bshl (var "x") (lit 1)) = some (.int ((x <<< 1 : ℕ) : Int)) := by
    simp only [bin, var, lit, evalMicroCExpr_uint32, h, evalMicroCBinOp_uint32]
    rw [show (1 : Int).toNat % 64 = 1 from rfl, show Int.shiftLeft (x : Int) 1 = ((x <<< 1 : ℕ) : Int) from rfl,
      wrap_nat _ hs]
  set env1 := env.update "x" (.int ((x <<< 1 : ℕ) : Int))
  have hb : ((x <<< 1) &&& 256) < 2 ^ 32 := lt_of_le_of_lt Nat.and_le_right (by decide)
  have e2 : evalMicroCExpr_uint32 env1 (bin .ltOp (lit 0) (bin .band (var "x") (lit 256))) =
      some (.bool (decide ((0 : Int) < (((x <<< 1) &&& 256 : ℕ) : Int)))) := by
    simp only [bin, var, lit, evalMicroCExpr_uint32, env1, upd_same, evalMicroCBinOp_uint32]
    rw [show Int.land ((x <<< 1 : ℕ) : Int) 256 = (((x <<< 1) &&& 256 : ℕ) : Int) from rfl, wrap_nat _ hb]
  rw [xtimeProg, eval_seq 0 env env1 _ _ (eval_assign 0 env "x" _ _ e1), eval_ite 0 env1 _ _ _ _ e2]
  unfold GF256.xt
  have hx2 : (x <<< 1) ^^^ 0x11d < 2 ^ 32 := Nat.xor_lt_two_pow (n := 32) hs (by decide)
  have e3 : evalMicroCExpr_uint32 env1 (bin .bxor (var "x") (lit 285)) =
      some (.int (((x <<< 1) ^^^ 0x11d : ℕ) : Int)) := by
    simp only [bin, var, lit, evalMicroCExpr_uint32, env1, upd_same, evalMicroCBinOp_uint32]
    rw [show Int.xor ((x <<< 1 : ℕ) : Int) 285 = (((x <<< 1) ^^^ 0x11d : ℕ) : Int) from rfl, wrap_nat _ hx2]
  split_ifs with h1 h2 h2
  · exact ⟨_, eval_assign 0 env1 "x" _ _ e3, upd_same _ _ _⟩
  · exfalso; simp at h1; omega
  · exfalso; simp at h1; omega
  · exact ⟨env1, by simp, upd_same _ _ _⟩

/-! ## CRC-32C, one byte -/

theorem bit_step (env : MicroCEnv) (w : ℕ) (hw : w < 2 ^ 32) (h : env "c" = .int w) :
    evalMicroC_uint32 0 env bitProg = some (.normal, env.update "c" (.int (Crc.bit w))) := by
  have ha : w &&& 1 < 2 ^ 32 := lt_of_le_of_lt Nat.and_le_right (by decide)
  have hr : w >>> 1 < 2 ^ 32 := by rw [Nat.shiftRight_eq_div_pow]; omega
  have e1 : evalMicroCExpr_uint32 env (bin .eqOp (bin .band (var "c") (lit 1)) (lit 1)) =
      some (.bool ((((w &&& 1 : ℕ) : Int)) == 1)) := by
    simp only [bin, var, lit, evalMicroCExpr_uint32, h, evalMicroCBinOp_uint32]
    rw [show Int.land (w : Int) 1 = ((w &&& 1 : ℕ) : Int) from rfl, wrap_nat _ ha]
  rw [bitProg, eval_ite 0 env _ _ _ _ e1]
  have hand : w &&& 1 = w % 2 := Nat.and_one_is_mod w
  unfold Crc.bit
  by_cases hodd : w % 2 = 1
  · have hc : ((((w &&& 1 : ℕ) : Int)) == 1) = true := by rw [hand, hodd]; rfl
    rw [if_pos hc, if_pos hodd]
    have hx : (w >>> 1) ^^^ Crc.POLY < 2 ^ 32 := Nat.xor_lt_two_pow (n := 32) hr (by decide)
    apply eval_assign
    simp only [bin, var, lit, evalMicroCExpr_uint32, h, evalMicroCBinOp_uint32]
    rw [show (1 : Int).toNat % 64 = 1 from rfl, show Int.shiftRight (w : Int) 1 = ((w >>> 1 : ℕ) : Int) from rfl,
      wrap_nat _ hr, show Int.xor ((w >>> 1 : ℕ) : Int) 0x82f63b78 = (((w >>> 1) ^^^ Crc.POLY : ℕ) : Int) from rfl,
      wrap_nat _ hx]
  · have hc : ¬ ((((w &&& 1 : ℕ) : Int)) == 1) = true := by
      rw [hand]; have : w % 2 = 0 := by omega
      rw [this]; decide
    rw [if_neg hc, if_neg hodd]
    apply eval_assign
    simp only [bin, var, lit, evalMicroCExpr_uint32, h, evalMicroCBinOp_uint32]
    rw [show (1 : Int).toNat % 64 = 1 from rfl, show Int.shiftRight (w : Int) 1 = ((w >>> 1 : ℕ) : Int) from rfl,
      wrap_nat _ hr]

theorem rep_bits : ∀ (n : ℕ) (env : MicroCEnv) (w : ℕ), w < 2 ^ 32 → env "c" = .int w →
    ∃ env', evalMicroC_uint32 0 env (rep n bitProg) = some (.normal, env') ∧ env' "c" = .int (iter Crc.bit n w)
  | 0, env, w, _, h => ⟨env, by simp [rep], h⟩
  | n + 1, env, w, hw, h => by
    obtain ⟨env', h1, h2⟩ := rep_bits n (env.update "c" (.int (Crc.bit w))) (Crc.bit w) (Crc.bit_lt w hw)
      (upd_same _ _ _)
    refine ⟨env', ?_, ?_⟩
    · show evalMicroC_uint32 0 env (.seq bitProg (rep n bitProg)) = _
      rw [eval_seq 0 env _ _ _ (bit_step env w hw h), h1]
    · rw [h2]; rfl

/-- **The printed CRC kernel computes one byte of CRC-32C.** -/
theorem crcByte_correct (env : MicroCEnv) (c b : ℕ) (hc : c < 2 ^ 32) (hb : b < 256)
    (h1 : env "c" = .int c) (h2 : env "b" = .int b) :
    ∃ env', evalMicroC_uint32 0 env crcByteProg = some (.normal, env') ∧
      env' "c" = .int (iter Crc.bit 8 (c ^^^ b)) := by
  have hx : c ^^^ b < 2 ^ 32 := Nat.xor_lt_two_pow (n := 32) hc (by omega)
  have e1 : evalMicroCExpr_uint32 env (bin .bxor (var "c") (var "b")) = some (.int ((c ^^^ b : ℕ) : Int)) := by
    simp only [bin, var, evalMicroCExpr_uint32, h1, h2, evalMicroCBinOp_uint32]
    rw [show Int.xor (c : Int) (b : Int) = ((c ^^^ b : ℕ) : Int) from rfl, wrap_nat _ hx]
  obtain ⟨env', h3, h4⟩ := rep_bits 8 (env.update "c" (.int ((c ^^^ b : ℕ) : Int))) (c ^^^ b) hx (upd_same _ _ _)
  exact ⟨env', by rw [crcByteProg, eval_seq 0 env _ _ _ (eval_assign 0 env "c" _ _ e1), h3], h4⟩

/-! ## One byte of `tq_put_lane_` -/

/-- **The printed lane kernel writes one row's half-byte as `tq_put_lane_` does.** -/
theorem putNib_correct (env : MicroCEnv) (q k cd : ℕ) (hcd : cd < 16) (hk : k < 2 ^ 32)
    (h1 : env "q" = .int q) (h2 : env "k" = .int k) (h3 : env "cd" = .int cd) :
    ∃ env', evalMicroC_uint32 0 env putNibProg = some (.normal, env') ∧
      env' "q" = .int (FastScan.putNib q k cd) := by
  have e1 : evalMicroCExpr_uint32 env (bin .ltOp (var "k") (lit 16)) = some (.bool (decide ((k : Int) < 16))) := by
    simp only [bin, var, lit, evalMicroCExpr_uint32, h2, evalMicroCBinOp_uint32]
  rw [putNibProg, eval_ite 0 env _ _ _ _ e1]
  unfold FastScan.putNib
  by_cases hk16 : k < 16
  · rw [if_pos (by simp; omega), if_pos hk16]
    have hv : (q &&& 0xf0) ||| cd < 2 ^ 32 :=
      Nat.or_lt_two_pow (lt_of_le_of_lt Nat.and_le_right (by decide)) (by omega)
    refine ⟨_, eval_assign 0 env "q" _ (.int (((q &&& 0xf0) ||| cd : ℕ) : Int)) ?_, upd_same _ _ _⟩
    simp only [bin, var, lit, evalMicroCExpr_uint32, h1, h3, evalMicroCBinOp_uint32]
    rw [show Int.land (q : Int) 240 = ((q &&& 0xf0 : ℕ) : Int) from rfl,
      wrap_nat _ (lt_of_le_of_lt Nat.and_le_right (by decide)),
      show Int.lor ((q &&& 0xf0 : ℕ) : Int) (cd : Int) = (((q &&& 0xf0) ||| cd : ℕ) : Int) from rfl, wrap_nat _ hv]
  · rw [if_neg (by simp; omega), if_neg hk16]
    have hsh : cd <<< 4 < 2 ^ 32 := by rw [Nat.shiftLeft_eq]; omega
    have hv : (q &&& 0x0f) ||| (cd <<< 4) < 2 ^ 32 :=
      Nat.or_lt_two_pow (lt_of_le_of_lt Nat.and_le_right (by decide)) hsh
    refine ⟨_, eval_assign 0 env "q" _ (.int (((q &&& 0x0f) ||| (cd <<< 4) : ℕ) : Int)) ?_, upd_same _ _ _⟩
    simp only [bin, var, lit, evalMicroCExpr_uint32, h1, h3, evalMicroCBinOp_uint32]
    rw [show Int.land (q : Int) 15 = ((q &&& 0x0f : ℕ) : Int) from rfl,
      wrap_nat _ (lt_of_le_of_lt Nat.and_le_right (by decide)),
      show (4 : Int).toNat % 64 = 4 from rfl, show Int.shiftLeft (cd : Int) 4 = ((cd <<< 4 : ℕ) : Int) from rfl,
      wrap_nat _ hsh,
      show Int.lor ((q &&& 0x0f : ℕ) : Int) ((cd <<< 4 : ℕ) : Int) = (((q &&& 0x0f) ||| (cd <<< 4) : ℕ) : Int)
        from rfl, wrap_nat _ hv]

/-! ## The C text -/

/-- A `uint32_t` C function around a printed program body. -/
def cFunction (name : String) (params : List String) (body : MicroCStmt) (result : String) : String :=
  "static uint32_t " ++ name ++ "(" ++ ", ".intercalate (params.map ("uint32_t " ++ ·)) ++ ")\n{\n    " ++
    microCToString body ++ "\n    return " ++ result ++ ";\n}\n"

def cFile : String :=
  "/* trustlean.c - generated by lean4/Vecindex/CodeGen.lean (lake exe vecindex-oracle gen) from TrustLean MicroC\n" ++
  " * programs proved there; do not edit. */\n#include <stdint.h>\n\n" ++
  cFunction "tl_gf_xtime" ["x"] xtimeProg "x" ++ "\n" ++
  cFunction "tl_crc32c_byte" ["c", "b"] crcByteProg "c" ++ "\n" ++
  cFunction "tl_put_nib" ["q", "k", "cd"] putNibProg "q"

end Vecindex.CodeGen
