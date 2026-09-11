# Staggered Fourier + red-black preconditioner — implementation plan

Owner: the dedicated Fourier/RB thread (job 6d173710). Direction:
`scripts_nm/fourier_rb_preconditioner_staggered_claude.md` (handoff from the gauge-fixing thread,
2026-09-10). Status: chunk 1 in progress.

## Sources (cite prominently)

- Even-odd / Schur preconditioning: T. DeGrand, P. Rossi; Grid `SchurRedBlack.h` /
  `SchurStaggeredOperator` (`Grid/algorithms/LinearOperator.h:575`).
- Block-matrix inverse identity $S_c^{-1} = (M^{-1})_{cc}$: Y. Saad, "Iterative Methods for Sparse
  Linear Systems", Schur-complement section.
- Staggered fermions / phases: J. Kogut, L. Susskind; N. Kawamoto, J. Smit (spin diagonalization);
  M. Golterman, J. Smit (even-odd structure).
- Free-limit Fourier preconditioner + frame lineage: the DWF line (`Grid/qcd/utils/FreeMobius5D_claude.h`,
  `FreeWilson_claude.h`); R. Brower, H. Neff, K. Orginos, arXiv:1206.5214 (Mobius background).

## Physics / goal summary

Combine the free-limit Fourier preconditioner $M_0 = \Omega^\dagger F \Omega$ with red-black (even-odd)
preconditioning, using staggered as the clean prototype (no spin index). Key identity (Saad): the free
RB-Schur inverse is the checkerboard block of the free full inverse,
$$
S_{c,\text{free}}^{-1} = (D_\text{free}^{-1})_{cc} = P_c F P_c ,
$$
so no sublattice FFT is needed — the FULL-lattice FFT $F$ is reused and simply projected with $P_c$
(this sidesteps the Brillouin-zone folding of the doubled-spacing sublattice).

Grid conventions pinned down for the free symbol (verified in source):

- `NaiveStaggeredFermion::M` is $D = m + D_\text{hop}$
  (`NaiveStaggeredFermionImplementation.h`, `axpy(out, mass, in, out)`), with the conventional
  $(1/2)\,c_1/u_0$ folded into the links by `ImportGauge`; we use $c_1 = u_0 = 1$.
- `SchurStaggeredOperator::Mpc` is exactly $m^2 - D_{eo} D_{oe}$
  (`Grid/algorithms/LinearOperator.h:598`, `axpby(out, -1.0, mass*mass, tmp, in)`).
- Free limit: the staggered-phase cross terms in $D_\text{hop}^2$ cancel identically
  ($\eta_\mu(x)\eta_\nu(x+\hat\mu) = -\eta_\nu(x)\eta_\mu(x+\hat\nu)$ for $\mu \ne \nu$, translations
  commute), so on the FULL lattice
  $$
  D^\dagger D_\text{free} = m^2 - D_\text{hop}^2 = m^2 + \sum_\mu \sin^2 p_\mu
  $$
  is FFT-diagonal with a SCALAR symbol, on both checkerboards. Its even block IS `Mpc`.
- Therefore the free RB preconditioner for staggered CG is
  $$
  \text{Mpc}_\text{free}^{-1} = P_e \; \text{FFT}^{-1} \, \frac{1}{m^2 + \sum_\mu \sin^2 p_\mu} \, \text{FFT} \; P_e .
  $$
- Grid's backward `FFT_all_dim` is volume-normalized (verified in `tests/core/Test_fft.cc` invertible
  check), so no extra $1/V$ factor.
- Boundary conditions: `StaggeredImpl` is periodic (no boundary-phase params). Chunk 1 validates with
  periodic BC in all directions; the AP-time twist enters later chunks (via the position-space twist
  phase, as in `FreeWilson_claude.h`) if/when we match production BC.

## Files

- `tests/solver/staggered_fourier_rb_impl_plan_claude.md` — this plan.
- `tests/solver/Test_staggered_freeprec_rb_claude.cc` — chunk 1 validation test (new).
- `log/tmp_claude.sh` — build+run handoff (new; compiles against `build_mpi` via `grid-config`,
  pattern of `scripts_nm/grid_freeprec_build_scc_mpi_claude.sh`).
- Later chunks: `Grid/qcd/utils/FreeStaggered_claude.h` (promoted preconditioner class),
  a driver `tests/solver/Test_staggered_freeprec_rb_solve_claude.cc`, and qsub scripts.

## Chunks

1. **Free-symbol validation (no gauge).** CPU, small lattice (8^4), unit gauge, masses {0.5, 0.05}.
   - Test A (full lattice): apply $D^\dagger D$ (Grid `M` then `Mdag`) to
     $\text{FFT}^{-1} [\,(m^2+\sum_\mu \sin^2 p_\mu)^{-1}\, \text{FFT}\, \eta\,]$; recover $\eta$ to
     machine precision. Validates the scalar symbol and the $\eta$-phase cancellation.
   - Test B (RB): same with $\eta_e$ on the even checkerboard, preconditioner $P_e F_\text{norm} P_e$
     (zero-pad to full grid, FFT-diagonal apply, project even), operator
     `SchurStaggeredOperator::Mpc`; recover $\eta_e$ to machine precision. This is the bit-level
     $S^{-1} = (M^{-1})_{cc}$ check.
   - Files: `tests/solver/Test_staggered_freeprec_rb_claude.cc`, `log/tmp_claude.sh`.
2. **Interacting, no frame.** On a quenched config (this project's `ckpoint_lat.*`), run staggered
   RB-CG preconditioned with $P_e F_\text{norm} P_e$ ($\Omega = 1$) vs plain RB-CG; metric = D-apply
   count to fixed relative residual. Requires a preconditioned Krylov driver (flexible CG or the FGMRES
   used on the DWF line). Files: promote preconditioner to `FreeStaggered_claude.h`, new solve driver.
3. **Add the frame.** $\Omega$ = flowed-Landau frame; $M_0^{\rm rb} = P_e \Omega^\dagger F \Omega P_e$
   (site-local $\Omega$ commutes with $P_e$). Measure extra clustering; report flowed-fixed Landau
   functional per config. Check the handoff note's caveat: staggered frame response is NOT assumed.
4. **Map back to Wilson/DWF.** Same $P_c F P_c$ construction with the spin/$L_s$-carrying $F$
   (`FreeMobius5D_claude.h`) as a Fourier-preconditioned RB-CGNE; compare against the two-level
   deflation baseline (`scripts_nm/svd_deflation_topological_twolevel_claude.md`).

## Open questions

- Chunk 2: which gauge ensemble/mass point is the headline target for the staggered interacting test
  (quenched 16^4 `ckpoint_lat.640` at which staggered mass)?
- Chunk 2: preconditioned solver choice — flexible CG vs FGMRES (DWF line used FGMRES; symbol is SPD
  here so PCG is natural). To be decided with Nobu when chunk 2 starts.
- AP-time BC: `StaggeredImplParams` carries no boundary phases; if production comparison needs AP time,
  add the twist by hand (position-space phase) as in `FreeWilson_claude.h`.

## Status log

- 2026-09-10: plan written; chunk 1 test + handoff script drafted (this session). Awaiting user build/run.
- 2026-09-10 (first run): PHYSICS PASSED at $m=0.5$ — `STAGFREE_FULL rel2 = 1.6e-31`,
  `STAGFREE_RB rel2 = 1.4e-31` (log `log/staggered_freeprec_rb_chunk1_claude.log`). Process then
  aborted with `free(): invalid size` at teardown of the first mass iteration (re-constructing a
  second `NaiveStaggeredFermion` in-process). Workaround: one mass per process — test takes
  `--mass`, `tmp_claude.sh` invokes once per mass, single-threaded. $m=0.05$ still to be confirmed.
- 2026-09-10 (second run): **CHUNK 1 PHYSICS COMPLETE** — both masses at machine precision:
  $m=0.5$: FULL `1.6e-31`, RB `1.4e-31`; $m=0.05$: FULL `5.6e-29`, RB `5.8e-29` (growth tracks the
  $1/m^2$ condition number). The `free(): invalid size` abort persists at scope exit even with ONE
  operator per process, i.e. it is the `NaiveStaggeredFermion`/`SchurStaggeredOperator` DESTRUCTOR
  path (never exercised before program end in upstream Grid tests), not the mass loop. Workaround in
  the test: heap-allocate both and intentionally leak (commented originals kept in place). Exit-code
  confirmation of the leak workaround pending an optional rerun; physics needs no rerun.
- 2026-09-10: Nobu decided to SKIP staggered chunks 2-4 and go to DWF directly. The DWF line
  continues in `dwf_fourier_rb_impl_plan_claude.md`; this staggered plan is closed at chunk 1.
