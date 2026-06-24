# disc speedup #1 -- mixed-precision CG (impl plan)

## STATUS (2026-06-23): CODE DONE, NOT BUILT
- NEW variant `disc_multipleGamma_binary_mixedprec_claude.cc` written; `Make.inc`
  target added (`bin_PROGRAMS` + `_SOURCES`/`_LDADD`).
- NOT built yet. USER builds (Claude never compiles): `cd Grid && automake
  examples/Makefile` -> `cd build && ./config.status examples/Makefile` ->
  `make -C build/examples disc_multipleGamma_binary_mixedprec_claude`.
- PENDING: first build + 1-config correctness/speed check (Chunks C-D below):
  traces agree ~1e-7 vs the double binary, expect ~0.5x wall.
- Ground rules: Claude NEVER submits jobs / NEVER rm / NEVER compiles (user does).


## Algorithm source (mandatory citation)

Mixed-precision (single inner / double outer) Conjugate Gradient with reliable
update of the residual:

- M. A. Clark, R. Babich, K. Barros, R. Brower, C. Rebbi, "Solving Lattice QCD
  systems of equations using mixed precision solvers on GPUs," Comput. Phys.
  Commun. 181 (2010) 1517, arXiv:0911.3191.
- G. L. G. Sleijpen, H. A. van der Vorst, "Reliable updated residuals in hybrid
  Bi-CG methods," Computing 56 (1996) 141 (the reliable-update idea).

Grid implementation: `Grid/algorithms/iterative/ConjugateGradientMixedPrec.h`,
class `MixedPrecisionConjugateGradient<FieldD,FieldF>`. The implementing code
comment must cite arXiv:0911.3191.

## Goal / physics summary

Disconnected-loop measurement `disc_multipleGamma_binary_claude.cc` inverts the
Mobius operator $D_\text{ov}$ ($L_s=16$, $24^3\times48$) once per
(timeslice, even/odd, spin, color) = $48\times2\times4\times4 = 1536$ solves per
config, all at `tol=1e-8` in DOUBLE precision. On MI300A, FP32 throughput is
~2x FP64 and the operator is bandwidth-bound, so running the CG inner iterations
in SINGLE precision with a DOUBLE reliable-update outer loop roughly HALVES the
per-config wall time, bit-reproducible to the SAME 1e-8 outer residual.

Constraints carried from the discussion:
- Keep the OUTER tolerance at `1e-8` (do NOT relax it -- that was option #4, ruled
  out). Mixed precision keeps 1e-8 cheaply; only the inner CG runs at single.
- Even/odd Schur preconditioning (`SchurDiagMooee`) stays exactly as now.
- No change to the dilution scheme, source, contraction, I/O, or the wall-time
  blocker.

## Canonical idiom we mirror

`Grid/tests/Test_dwf_mixedcg_prec.cc` (lines ~95-145):
- single-precision grids `UGrid_f, UrbGrid_f, FGrid_f, FrbGrid_f`
  (`vComplexF::Nsimd()`),
- single gauge field `Umu_f`, filled per config via `precisionChange(Umu_f,Umu)`,
- double + single Mobius actions `FermAct` / `FermAct_f`,
- dual Schur operators
  `SchurDiagMooeeOperator<MobiusFermionD,LatticeFermionD> HermOpEO(FermAct)` and
  `SchurDiagMooeeOperator<MobiusFermionF,LatticeFermionF> HermOpEO_f(FermAct_f)`,
- `MixedPrecisionConjugateGradient<LatticeFermionD,LatticeFermionF>
   mCG(1.0e-8, MaxInner, MaxOuter, FrbGrid_f, HermOpEO_f, HermOpEO)`.

The current code drives the solve through
`SchurRedBlackDiagMooeeSolve<LatticeFermion> schur(CG)`. `mCG` is a
`LinearFunction<LatticeFermionD>`, so it drops straight in as the inner solver:
`SchurRedBlackDiagMooeeSolve<LatticeFermionD> schur(mCG); schur(FermAct, src5,
result5, ZG);` -- the `SchurRedBlack` wrapper still builds its own (cheap) double
Mooee operator for the source reconstruction; `mCG` does the heavy iterations.

## Files to modify

- `Grid/examples/disc_multipleGamma_binary_claude.cc`  -- the kernel (see OPEN
  QUESTION on in-place vs new file).
- (only if a NEW binary is chosen) `Grid/examples/Make.inc` +
  automake/configure regen so `build/examples` learns the new target.
- driver/scripts: NONE. Same binary name + same CLI/env, so
  `submit_disc_tuolumne.sh`, the finalizer, and the new-ens driver are untouched.

## Ordered implementation chunks

### Chunk A -- single-precision grids + gauge (main)
Files: `disc_multipleGamma_binary_claude.cc`
- After the double grids (line ~135) add `UGrid_f, UrbGrid_f, FGrid_f, FrbGrid_f`
  with `GridDefaultSimd(Nd,vComplexF::Nsimd())`.
- Allocate `LatticeGaugeFieldF Umu_f(UGrid_f)` once (outside the conf loop).
- Inside the conf loop, after `NerscIO::readConfiguration(Umu,...)` (line ~243),
  add `precisionChange(Umu_f, Umu);`.
- Build the single action per config next to the double one (line ~246):
  `MobiusFermionF FermAct_f(Umu_f,*FGrid_f,*FrbGrid_f,*UGrid_f,*UrbGrid_f, mass,
   M5, b, c, Params);`  (same b,c,M5,boundary).

### Chunk B -- mixed-precision solver in Solve()
Files: `disc_multipleGamma_binary_claude.cc`
- Extend `Solve(...)` to take both actions (`Action &D, ActionF &D_f`) and the
  single rb grid.
- Inside, COMMENT OUT (keep as reference) the old:
  `// ConjugateGradient<LatticeFermion> CG(1.0e-8,100000);`
  `// SchurRedBlackDiagMooeeSolve<LatticeFermion> schur(CG);`
  and add beneath:
  `SchurDiagMooeeOperator<Action,LatticeFermionD>  HermOpEO(D);`
  `SchurDiagMooeeOperator<ActionF,LatticeFermionF> HermOpEO_f(D_f);`
  `MixedPrecisionConjugateGradient<LatticeFermionD,LatticeFermionF>
     mCG(1.0e-8, MAXINNER, MAXOUTER, D_f.FermionRedBlackGrid(), HermOpEO_f,
         HermOpEO);  // arXiv:0911.3191`
  `SchurRedBlackDiagMooeeSolve<LatticeFermionD> schur(mCG);`
- Keep `ZeroGuesser<LatticeFermionD> ZG;` and `schur(D,src5,result5,ZG);`
  unchanged (deflation guesser is option #3, later).
- Update the one call site (line ~258) to pass `FermAct, FermAct_f`.

### Chunk C -- build + smoke test
Files: (build only)
- Rebuild: `make -C /usr/workspace/lsd/matsumoto5/su4_32c/build/examples
  disc_multipleGamma_binary_claude` (non-incremental as usual).
- Handed off to the user via `tmp_claude.sh` teeing to a `*_claude.log` (no rm).

### Chunk D -- correctness + speed validation
Files: (run only, user-run handoff script)
- Run on ONE already-measured config into a SCRATCH obsdir, compare the 10
  `traces.<gam>.<conf>` against the existing double-precision outputs:
  agreement to ~1e-7 (relative) confirms the mixed-prec path.
- Record per-config wall time vs the ~1000 s double baseline (expect ~0.5x).

## Open questions (RESOLVED 2026-06-23)

1. SOURCE ORGANIZATION -> (b) NEW variant binary
   `disc_multipleGamma_binary_mixedprec_claude.cc`, added to `Make.inc`
   `bin_PROGRAMS`; needs automake/configure regen. Original binary left untouched
   as the double-precision reference for same-config A/B.
2. ITERATION CAPS -> adopt `MAXINNER=10000`, `MAXOUTER=50` (test idiom). Outer
   tol fixed at 1e-8 (no #4).
3. SCHUR OPERATOR -> PLAIN `SchurDiagMooeeOperator` (not Paranoid).

## Constructor (verified against the header)

`MixedPrecisionConjugateGradient<FieldD,FieldF>(tol, maxinner, maxouter, sp_grid,
Linop_f, Linop_d)`. Optional `useGuesser(LinearFunction<FieldF>&)` injects a
single-precision guesser -- this is the hook for the #3 deflation step later.
