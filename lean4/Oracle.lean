/-
  vecindex-oracle - the Lean model of each kernel, run on concrete inputs, for the differential test in `run`.
  Every value printed comes from the definitions the theorems are about, not from a second implementation.

    vecindex-oracle gen       the TrustLean-printed C kernels (gen/trustlean.c)
    vecindex-oracle gf        gf_exp_, gf_log_, every gf_mul_, gf_inv_, and an FNV-1a hash of every gf_coef_
    vecindex-oracle digits    the digit tables of q1 - q5
    vecindex-oracle vd        tq_vd_ for every even dpad up to 4096 and 1, 2, 4 bits
    vecindex-oracle layout    ssum_layout_ over a grid of sizes, block sizes, k and m
    vecindex-oracle chunks    the Walsh-Hadamard chunks of every dpad up to 1024
    vecindex-oracle crc       stdin: messages as hex lines; out: their CRC-32C
    vecindex-oracle block     stdin: cases "vd, 16 vd code bytes, 16 vd table bytes"; out: the 32 row sums
    vecindex-oracle topk      stdin: cases "k n, then n (id score)"; out: the kept (id score) pairs
    vecindex-oracle tenant    stdin: "in TN X", "out TN Y", "rin TN LO HI", "own TN Y", "parse HEX"; out: the results
    vecindex-oracle journal   stdin: "journal STAMP HEX" (the stamp of the file beside it, the journal's bytes);
                              out: the good part vs_rjtrim_ keeps
    vecindex-oracle rate      stdin: "rate QPS T1 .. TN" (times in 1/1024 s); out: which requests get through
-/
import Vecindex

open Vecindex

def joinNat (xs : List Nat) : String := " ".intercalate (xs.map toString)

def fnv (h : UInt64) (b : Nat) : UInt64 := (h ^^^ (UInt64.ofNat b)) * 1099511628211

def gfOut : IO Unit := do
  IO.println ("exp " ++ joinNat ((List.range 512).map GF256.exp))
  IO.println ("log " ++ joinNat ((List.range 256).map GF256.log))
  for a in List.range 256 do
    IO.println ("mul " ++ toString a ++ " " ++ joinNat ((List.range 256).map (GF256.mul a)))
  IO.println ("inv " ++ joinNat ((List.range 256).map GF256.inv))
  let mut h : UInt64 := 14695981039346656037
  for k in List.range' 1 128 do
    for p in List.range 64 do
      for i in List.range k do
        h := fnv h (GF256.coef k p i)
  IO.println ("coef " ++ toString h)

def digitsOut : IO Unit := do
  for (R, D) in [(2, 8), (4, 4), (3, 5), (16, 2), (256, 1)] do
    for b in List.range 256 do
      IO.println ("digits " ++ toString R ++ " " ++ toString b ++ " " ++
        joinNat ((List.range D).map (Codes.digit R b)))

def vdOut : IO Unit := do
  for bits in [1, 2, 4] do
    IO.println ("vd " ++ toString bits ++ " " ++
      joinNat ((List.range 2048).map fun i => FastScan.vdOf (2 * i + 2) bits))

def layoutOut : IO Unit := do
  for D in [1, 4095, 4096, 65535, 65536, 65537, 1000000, 123456789, 4294967296, 1099511627777] do
    for blog in [12, 16, 24] do
      for k in [1, 2, 3, 7, 32, 100, 128] do
        for m in [0, 1, 2, 5, 64] do
          let L := RS.layout D blog k m
          IO.println ("layout " ++ joinNat [D, blog, k, m, L.N, L.G, L.P, L.ta, L.tra, L.tb, L.fsize])

def chunksOut : IO Unit := do
  for dpad in List.range' 1 1024 do
    IO.println ("chunks " ++ toString dpad ++ " " ++
      " ".intercalate ((Hadamard.chunks 0 dpad).map fun c => toString c.1 ++ ":" ++ toString (2 ^ c.2)))

def hexByte (c1 c2 : Char) : Nat :=
  let v (c : Char) : Nat := if c.isDigit then c.toNat - '0'.toNat else c.toLower.toNat - 'a'.toNat + 10
  v c1 * 16 + v c2

def parseHex : List Char → List Nat
  | c1 :: c2 :: cs => hexByte c1 c2 :: parseHex cs
  | _ => []

def hex8 (n : Nat) : String :=
  let s := String.ofList (Nat.toDigits 16 n)
  String.ofList (List.replicate (8 - s.length) (Char.ofNat 48)) ++ s

def numbers (s : String) : List Nat :=
  (s.splitToList Char.isWhitespace).filterMap fun w => w.toNat?

def ints (s : String) : List Int :=
  (s.splitToList Char.isWhitespace).filterMap fun w => w.toInt?

def crcOut (input : String) : IO Unit := do
  for line in input.splitOn "\n" do
    let l := line.trim
    if l.isEmpty then continue
    let bytes := if l == "-" then [] else parseHex l.toList
    IO.println ("crc " ++ hex8 (Crc.crc32c bytes))

partial def blockOut (xs : List Nat) : IO Unit := do
  match xs with
  | vd :: rest =>
    let blkA := (rest.take (16 * vd)).toArray
    let lutA := ((rest.drop (16 * vd)).take (16 * vd)).toArray
    let blk : Nat → Nat := fun i => blkA.getD i 0
    let lut : Nat → Nat := fun i => lutA.getD i 0
    IO.println ("block " ++ joinNat ((List.range 32).map (FastScan.refScore blk lut vd)))
    blockOut (rest.drop (32 * vd))
  | [] => pure ()

partial def topkOut (xs : List Int) : IO Unit := do
  match xs with
  | k :: n :: rest =>
    let pairs := (List.range n.toNat).map fun i => (rest.getD (2 * i) 0, rest.getD (2 * i + 1) 0)
    let kept := pairs.foldl (TopK.step k.toNat) []
    IO.println ("topk" ++ String.join (kept.map fun p => " " ++ toString p.1 ++ ":" ++ toString p.2))
    topkOut (rest.drop (2 * n.toNat))
  | _ => pure ()


def showOpt (o : Option Int) : String := match o with | some v => toString v | none => "none"

def tenantOut (input : String) : IO Unit := do
  for line in input.splitOn "\n" do
    let w := (line.trim.splitToList Char.isWhitespace).filter (· ≠ "")
    match w with
    | ["in", a, b] => match a.toInt?, b.toInt? with
      | some tn, some x => IO.println ("in " ++ a ++ " " ++ b ++ " " ++ showOpt (Tenant.inId tn x))
      | _, _ => pure ()
    | ["out", a, b] => match a.toInt?, b.toInt? with
      | some tn, some y => IO.println ("out " ++ a ++ " " ++ b ++ " " ++ toString (Tenant.outId tn y))
      | _, _ => pure ()
    | ["own", a, b] => match a.toInt?, b.toInt? with
      | some tn, some y => IO.println ("own " ++ a ++ " " ++ b ++ " " ++ showOpt (Tenant.own tn y))
      | _, _ => pure ()
    | ["rin", a, b, c] => match a.toInt?, b.toInt?, c.toInt? with
      | some tn, some lo, some hi => IO.println ("rin " ++ a ++ " " ++ b ++ " " ++ c ++ " " ++
          (match Tenant.rin tn lo hi with | some (x, y) => toString x ++ " " ++ toString y | none => "none"))
      | _, _, _ => pure ()
    | ["parse", h] => IO.println ("parse " ++ h ++ " " ++
        (match Tenant.parse (parseHex h.toList) with | some n => toString n | none => "none"))
    | _ => pure ()

def le32At (b : List Nat) (i : Nat) : Nat :=
  b.getD i 0 + 256 * b.getD (i + 1) 0 + 65536 * b.getD (i + 2) 0 + 16777216 * b.getD (i + 3) 0

/-- The header check of `vs_rjtrim_`: magic VXJ3, its CRC-32C, the stamp of the file beside it. -/
def jhead (fst : Nat) (h : List Nat) : Bool :=
  le32At h 0 == 0x334a5856 && Crc.crc32c ((h.drop 4).take 12) == le32At h 16 && fst != 0 &&
    le32At h 8 + 4294967296 * le32At h 12 == fst

def journalOut (input : String) : IO Unit := do
  for line in input.splitOn "\n" do
    match (line.trim.splitToList Char.isWhitespace).filter (· ≠ "") with
    | ["journal", st, hx] => match st.toNat? with
      | some fst =>
        let b := if hx == "-" then [] else parseHex hx.toList
        IO.println ("journal " ++ toString (Journal.good Crc.crc32c (jhead fst) b))
      | none => pure ()
    | _ => pure ()

def rateOut (input : String) : IO Unit := do
  for line in input.splitOn "\n" do
    match (line.trim.splitToList Char.isWhitespace).filter (· ≠ "") with
    | "rate" :: q :: ts =>
      match q.toNat? with
      | some qn =>
        let times : List ℚ := ts.filterMap fun t => t.toNat?.map fun k => (k : ℚ) / 1024
        let d := Rate.decide' (qn : ℚ) ⟨(qn : ℚ), 0⟩ times
        IO.println ("rate " ++ String.ofList (d.map fun x => if x then '1' else '0'))
      | none => pure ()
    | _ => pure ()

def main (args : List String) : IO UInt32 := do
  match args with
  | ["gen"] => IO.print CodeGen.cFile; pure 0
  | ["gf"] => gfOut; pure 0
  | ["digits"] => digitsOut; pure 0
  | ["vd"] => vdOut; pure 0
  | ["layout"] => layoutOut; pure 0
  | ["chunks"] => chunksOut; pure 0
  | ["crc"] => crcOut (← (← IO.getStdin).readToEnd); pure 0
  | ["block"] => blockOut (numbers (← (← IO.getStdin).readToEnd)); pure 0
  | ["topk"] => topkOut (ints (← (← IO.getStdin).readToEnd)); pure 0
  | ["tenant"] => tenantOut (← (← IO.getStdin).readToEnd); pure 0
  | ["journal"] => journalOut (← (← IO.getStdin).readToEnd); pure 0
  | ["rate"] => rateOut (← (← IO.getStdin).readToEnd); pure 0
  | _ => IO.eprintln "usage: vecindex-oracle gen|gf|digits|vd|layout|chunks|crc|block|topk|tenant|journal|rate"; pure 2
