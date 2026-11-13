/-
  vecindex, machine-checked: every module below proves facts about a kernel of the C code in ../src. The README
  of this folder lists each theorem next to the C function it is about.
-/
import Vecindex.Iter
import Vecindex.Bits
import Vecindex.GF256.Tables
import Vecindex.GF256.Algebra
import Vecindex.GF256.Field
import Vecindex.RS.Cauchy
import Vecindex.RS.Invert
import Vecindex.RS.Group
import Vecindex.RS.StoreSum
import Vecindex.Crc32c
import Vecindex.FastScan
import Vecindex.FastScanLayout
import Vecindex.Codes
import Vecindex.Hadamard
import Vecindex.TopK
import Vecindex.CodeGen
import Vecindex.Tenant
import Vecindex.Journal
import Vecindex.Rate
import Vecindex.NsSet
import Vecindex.Emeans
