# 24^4 fp32 production comparison -- implementation plan

## Physics / goal

Run the production DWF comparison on a 24^4 config, in **single precision** so the deflated
Krylov solve fits one GPU:

- Config: `configs_iwasaki_24_b2.6/ckpoint_lat.1100` (Iwasaki, $\beta=2.6$).
- Mass scan: $m = 0.1,\ 0.01,\ 0.001$ (frame is mass-independent -> built once, reused).
- Solvers compared: **RB-CGNE** (honest baseline) vs **deflated FGMRES = GMRES-DR** at the
  production window $(m_\text{restart}, k) = (16, 32)$. No outer/external deflation.
- Metric: $D_W$ applies. RB-CGNE $= 2 L_s \cdot \text{iters}$; GMRES-DR $= L_s \cdot \text{iters}$.
- Tolerance: $10^{-6}$ for BOTH solvers (fp32 CG/GMRES floor is $\sim 3\times10^{-7}$, so $10^{-8}$
  is unreachable in single).

## Why fp32 (the memory blocker)

GMRES-DR stores $m_\text{restart} + k \approx 48$ full 5D vectors. On $24^4 \times L_s{=}8$ one fp64
5D fermion is $24^4 \cdot 8 \cdot 12 \cdot 16\,\text{B} \approx 509$ MB, so the basis alone is
$\sim 24$ GB, and with the Mobius temporaries + gauge + F buffers the fp64 run is $\sim 63$ GB --
over the GPU. fp32 halves the basis to $\sim 12.7$ GB; the whole run fits ($\sim 32$ GB).

## Key design decision: prerot folds the frame -> M0 is the *bare* free inverse

Production already uses Peter Boyle's $\Omega$ pre-rotation (`--prerot`): gauge-transform
$U \to U^\Omega = \Omega U \Omega^\dagger$ once, rotate the source by $\Omega_5$, and precondition
with the **bare** free inverse $F$ (identity frame). Unitarily equivalent -> identical iteration
counts, no per-apply $\Omega$.

This is what makes the fp32 leg clean: with prerot the preconditioner is literally the bare free
inverse $F$ (no $\Omega$ multiply). So we **do NOT touch** the shared `FreeLimitPreconditioner`
(whose `Omega5` is hard-coded `LatticeColourMatrixD` and would not compile against a single field).

**Why not `FreeMobius5DInverse<WilsonImplF>` directly?** Its double barrel path (compiled even under
`FREEMOBIUS5D_USE_FP32`) assigns `ComplexD` literals into `FermionField` site elements
(`w()(a)(col) = ComplexD(...)`), which does not compile for a single field
(`std::complex<float> = std::complex<double>`). Editing that shared production header is out of scope.
Instead we keep $F$ as the **tested** `FreeMobius5DInverse<WilsonImplD>` (double field, single momentum
core -- exactly what production runs) and wrap it in a tiny single-precision adapter
`FreeInvF32Adapter : LinearFunction<LatticeFermionF>` that does single -> double, applies $F$, double ->
single. The memory hog -- the GMRES-DR Krylov basis -- stays single (fits); $F$'s own buffers are
modest. Two extra `precisionChange` passes per apply; $F$ already solves in single internally, so no
accuracy is lost.

Frame construction (WilsonFlow, FA-Landau fix, topological charge, the $\Omega$ gauge transform,
source rotation) stays in **double** -- it is cheap, mass-independent, and precision-insensitive.
Only the per-mass solve block (D, F, source, solvers) is single precision.

## Files

New dedicated driver (mirrors the split single-purpose drivers already in this project):

- `tests/solver/Test_dwf_freeprec_fp32_claude.cc` -- NEW. Frame in fp64, then RB-CGNE + GMRES-DR
  in fp32 with the frame folded via prerot.

Build + submit scripts (mirror the existing `grid_twolevel_*`/`grid_dwfrb_*` pairs):

- `scripts_nm/grid_freeprec_fp32_build_scc_gpu_claude.sh` -- NEW. Single-file nvcc build with
  `-DFREEMOBIUS5D_FP32` (F core already fp32) into `build_merged/Test_dwf_freeprec_fp32_claude`.
- `scripts_nm/grid_freeprec_fp32_gpu_qsub_claude.sh` -- NEW. qsub driver; CONFIG/GRID/masses/
  (m,k)/tol env-overridable; `--flowcache` reused so the flowed config is cached.

No shared-header edits. No external deflation-vector dump needed (GMRES-DR builds its $k=32$
deflation space adaptively from the Krylov space).

## Solver source citations

- GMRES-DR / recycling GMRES: `RecyclingGeneralisedMinimalResidual_claude.h`; Morgan,
  "GMRES with deflated restarting", SIAM J. Sci. Comput. 24 (2002) 20.
- RB (Schur even-odd) CG: Grid `SchurRedBlackDiagMooeeSolve`.
- FA Landau gauge fix: C.T.H. Davies et al., PRD 37 (1988) 1581.
- Free Mobius inverse (FFT + per-momentum $(4L_s)^2$ block inverse): project note
  `grid_packonce_fft_impl_plan_claude.md`.

## Ordered chunks

1. **fp32 driver** (`Test_dwf_freeprec_fp32_claude.cc`). Frame block copied from the production
   driver (double). Then per mass: build `UrotD` (double gauge transform), `precisionChange` ->
   `LatticeGaugeFieldF UrotF`; `MobiusFermionF D(UrotF, ...)`; `FreeMobius5DInverse<WilsonImplD>
   Ffree` (double, on FGrid) wrapped by `FreeInvF32Adapter M0`; source gaussian in double, rotate by
   `Om5` (double), `precisionChange` -> `LatticeFermionF`;
   RB-CGNE via `SchurRedBlackDiagMooeeSolve<LatticeFermionF>` (tol 1e-6); GMRES-DR via
   `RecyclingGeneralisedMinimalResidual<LatticeFermionF>` (m=16,k=32,recycle=false,tol 1e-6).
   Print iters + $D_W$ applies + wall + RB/GMRES-DR ratio per mass.
2. **Build script** + regenerate nothing (single-file nvcc recipe, no Make.inc).
3. **qsub script** (single 24^4 config, mass scan inside one job; flowcache on).
4. Hand both scripts to the user to run; read the log back.

## Open questions (resolve before coding)

- Q1: "deflated FGMRES" = **GMRES-DR(m=16,k=32)** (adaptive inner deflation, no external
  vectors)? Assumed yes. (The SVD-external-deflation `deflprec` path is a separate study and
  needs a 24^4 deflation-vector dump; not part of this comparison.)
- Q2: Dedicated fp32 driver (this plan) vs a `-DFREEPREC_OUTER_FP32` compile switch on the
  existing production driver? Dedicated driver recommended -- keeps the fp64 production driver
  untouched and matches the split-driver pattern.
- Q3: Also include the plain (un-preconditioned) FGMRES(M0) `run_m0` leg, or just RB-CGNE vs
  GMRES-DR? Assumed just the two the user named.
