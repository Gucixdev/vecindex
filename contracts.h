#ifndef HOLYC_CONTRACTS_H_
#define HOLYC_CONTRACTS_H_

/* Design-by-contract macros with TWO independently real backends, not
 * one decorative one:
 *
 *   1. Every normal `./build`/`./build test` compile: HOLYC_REQUIRES/
 *      ENSURES/INVARIANT become ordinary HOLYC_ASSERT runtime checks -
 *      real, immediate value on every test run, for every contributor,
 *      with nothing extra to install. This is "checked" - true for
 *      whatever inputs a test run actually exercises, no more.
 *
 *   2. Under CBMC (github.com/diffblue/cbmc, a real bounded model
 *      checker backed by an SMT solver - this machine has z3
 *      installed for it), the SAME macros become __CPROVER_assume/
 *      __CPROVER_assert: REQUIRES narrows the input space CBMC
 *      symbolically explores, ENSURES/INVARIANT become proof
 *      obligations CBMC must discharge for EVERY input in that space,
 *      not just the ones a test happened to try. This is "confirmed" -
 *      true for all bounded inputs, exhaustively. CBMC predefines
 *      __CPROVER__ automatically when it compiles a file, so a plain
 *      `./build test` of the exact same source never takes this
 *      branch - proof_cbmc/'s own harnesses (see proof_cbmc/README.md)
 *      are what actually gets run through cbmc, not the test suite.
 *
 * NOT the same tool as project/proof/ (a standalone SMT claim
 * verifier: `//verify vars:`/`//verify is:` comment annotations ->
 * SMT-LIB2 -> z3 directly, no C code execution modeled at all) -
 * genuinely complementary, not overlapping:
 *   - project/proof proves standalone ALGEBRAIC CLAIMS the developer
 *     writes (e.g. "no_overflow_add(a,b) -> (a+b)>a"), fast, exhaustive
 *     over the claim's declared scalar types, but knows nothing about
 *     any actual C function's real control flow/loops/pointers.
 *   - this file's contracts + CBMC prove properties of the ACTUAL C
 *     function BODY (real branches, real pointer arithmetic, real
 *     loops up to an unwind bound) - a stronger, slower, more
 *     "does this real code actually do what it claims" tool.
 * A function can reasonably use both: a project/proof claim for the
 * pure arithmetic property, HOLYC_ENSURES here for the same property
 * as actually computed by the real implementation.
 *
 * Frama-C + ACSL (its own "requires"/"ensures" comment-annotation
 * language, a real, separate third verification tier) is NOT wired up
 * here yet - Frama-C isn't packaged for this distro (would need opam)
 * and ACSL annotations can't be generated through this macro system
 * anyway (they have to be literal comments Frama-C's own frontend
 * reads directly - C macros can only emit tokens, never comments). A
 * real, documented gap, not a silent one.
 *
 * Not every function gets contracts - only ones where "what must be
 * true" is a genuinely useful, checkable property (a bound, an
 * invariant, a non-aliasing guarantee), the same selective judgment
 * this project already applies to HOLYC_ASSERT itself. A contract that
 * just restates the function's own body in different words is noise,
 * not verification. */

#include "base.h"

#if defined(__CPROVER__)
    #define HOLYC_REQUIRES(cond) __CPROVER_assume(cond)
    #define HOLYC_ENSURES(cond) __CPROVER_assert((cond), #cond)
    #define HOLYC_INVARIANT(cond) __CPROVER_assert((cond), #cond)
#else
    #define HOLYC_REQUIRES(cond) HOLYC_ASSERT(cond)
    #define HOLYC_ENSURES(cond) HOLYC_ASSERT(cond)
    #define HOLYC_INVARIANT(cond) HOLYC_ASSERT(cond)
#endif

#endif /* HOLYC_CONTRACTS_H_ */
