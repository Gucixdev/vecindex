import Lake
open Lake DSL

/- Machine-checked model of vecindex: specifications of its kernels, the theorems that they hold,
   and the executables that feed the same inputs to the C code and to the Lean model. -/
package «vecindex» where
  leanOptions := #[⟨`autoImplicit, false⟩]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.26.0"

require TrustLean from git
  "https://github.com/manuelpuebla/trust-lean" @ "aaf67f6ce98690042318f90573c2b3077b88123b"

require «amo-lean» from git
  "https://github.com/manuelpuebla/amo-lean" @ "555e5db8732ae644e930ff1541e20e00dc2322e1"

@[default_target]
lean_lib «Vecindex» where
  roots := #[`Vecindex]

/-- The Lean model run on concrete inputs (differential test) and the printer of the TrustLean kernels. -/
lean_exe «vecindex-oracle» where
  root := `Oracle
