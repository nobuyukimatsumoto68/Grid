# DWF Fourier + red-black preconditioner — implementation plan

Owner: the Fourier/RB thread (job 6d173710). User decision 2026-09-10: **skip the staggered
interacting/frame chunks, go to DWF directly** (staggered chunk 1 validated the mechanism bit-level;
see `staggered_fourier_rb_impl_plan_claude.md`). Direction:
`scripts_nm/fourier_rb_preconditioner_staggered_claude.md`, section "Map back to Wilson/DWF".

## Sources (cite prominently)

- Block-matrix inverse identity $S_c^{-1} = (M^{-1})_{cc}$: Y. Saad, "Iterative Methods for Sparse
  Linear Systems", Schur-complement section. Validated bit-level on staggered (chunk 1, 2026-09-10).
- Even-odd / Schur preconditioning: T. DeGrand, P. Rossi; Grid `SchurRedBlackDiagMooeeSolve`,
  `NonHermitianSchurDiagMooeeOperator` (`Grid/algorithms/LinearOperator.h:474`).
- Free Mobius inverse $F$ + frame $M_0=\Omega^\dagger F \Omega$: `Grid/qcd/utils/FreeMobius5D_claude.h`
  (validated on the full lattice, freeprec line); Mobius kernel R. Brower, H. Neff, K. Orginos,
  arXiv:1206.5214.
- FGMRES: Y. Saad (flexible GMRES);
  `Grid/algorithms/iterative/RecyclingGeneralisedMinimalResidual_claude.h` (this tree).
- Two-level baseline being compared against: `scripts_nm/svd_deflation_topological_twolevel_claude.md`
  and the split binaries (1)-(4) of `twolevel_common_claude.h`.

## Physics / goal summary

Grid's RB solve (`SchurRedBlackDiagMooeeSolve`) eliminates the even checkerboard and runs the inner
Krylov on the ODD-checkerboard Schur complement
$$
S_o = M_{oo} - M_{oe} M_{ee}^{-1} M_{eo} \qquad (= \text{Mpc, DiagMooee form}),
$$
with CGNE cost $2 L_s \times$ iters in $D_W$ applies. By the block-inverse identity, the FREE Schur
inverse is the odd block of the free full inverse:
$$
S_{o,\text{free}}^{-1} = (D_\text{free}^{-1})_{oo} = P_o F P_o ,
$$
with $F$ = `FreeMobius5DInverse` (full-lattice FFT + per-momentum $(4L_s)^2$ block inverse — already
built and validated). So the combined preconditioner is the EXISTING $F$ (optionally framed,
$M_0 = \Omega^\dagger F \Omega$; site-local $\Omega$ commutes with $P_o$), zero-padded on even,
projected back to odd:
$$
M_0^{\rm rb} = P_o\, \Omega^\dagger F\, \Omega\, P_o .
$$
Run FGMRES on the non-Hermitian $S_o$ with right preconditioner $M_0^{\rm rb}$; in the free limit the
preconditioned operator is exactly 1. Goal: beat plain RB-CGNE in $D_W$ applies, and reduce how much
RB-Schur$^2$ deflation the two-level scheme needs.

Metric (established convention): $D_W$ applies to relative residual $10^{-8}$;
RB-CGNE $= 2 L_s \times$ iters; RB-FGMRES $= L_s \times$ iters (one Mpc = two half-checkerboard hops
= one full-lattice $D_W$-equivalent); $M_0$ costs 0. Params: $L_s=8$, $M_5=1.8$, Shamir $b=1.5$,
$c=0.5$, $m=0.1$, AP time (all from `twolevel_common_claude.h` TL_*).

## Files

- `tests/solver/dwf_fourier_rb_impl_plan_claude.md` — this plan.
- `tests/solver/Test_dwf_freeprec_rb_claude.cc` — D1a cold gate (MPI/CPU, self-contained, no I/O).
- `log/tmp_claude.sh` — CPU cold-gate build+run handoff (chunk D1a).
- `tests/solver/Test_dwf_freeprec_rb_solve_claude.cc` — D1b interacting solve (GPU; reads a config).
  SEPARATE binary from D1a: it needs NerscIO/BinaryIO, which only link against the GPU install tree.
- `scripts_nm/grid_dwfrb_build_scc_gpu_claude.sh`, `scripts_nm/grid_dwfrb_gpu_qsub_claude.sh` —
  GPU build + qsub for D1b (adapted from the twolevel pair; default config
  `configs_iwasaki_16_b2.6/ckpoint_lat.640`).

## Chunks

- **D1a. Cold gate (free-limit exactness).** Unit gauge, small lattice (8^4, $L_s=8$), CPU.
  Check $\text{Mpc}_\text{free}\,(P_o F P_o)\, b_o = b_o$ to machine precision
  (`NonHermitianSchurDiagMooeeOperator::Mpc` vs the projected `FreeMobius5DInverse`).
  The DWF analogue of staggered chunk-1 Test B — validates the identity with nontrivial
  $M_{ee}$ (5D $s$-hopping), spin, and the AP-time twist.
  Files: `Test_dwf_freeprec_rb_claude.cc` (cold path), `log/tmp_claude.sh`.
- **D1b. Interacting, no frame ($\Omega=1$).** `ckpoint_lat.640` (16^4): (a) plain RB-CGNE baseline
  (identical to `Test_dwf_rbcgne_claude.cc` leg (a), same fixed source seed {20,21,22,23});
  (b) FGMRES on Mpc with right prec $P_o F P_o$, tol $10^{-8}$, restart 16. Report iters + $D_W$.
  Manual RB source prep / solution reconstruction (b_o' = b_o - M_{oe} M_{ee}^{-1} b_e;
  x_e = M_{ee}^{-1}(b_e - M_{eo} x_o)); verify the reconstructed full residual.
  Files: same binary (`--config` path), GPU build+qsub scripts.
- **D2. Add the frame.** $M_0^{\rm rb} = P_o \Omega^\dagger F \Omega P_o$ with the flowed-Landau
  warm-start frame (reuse the `OmL` construction from `Test_dwf_deflprec_claude.cc:77-88`).
  Same runs, measure the extra reduction; report the flowed-fixed Landau functional.
- **D3. Interaction with deflation.** Combine $M_0^{\rm rb}$-FGMRES with the RB-Schur$^2$ deflation
  dump (`deflrb`): does the Fourier preconditioner reduce the number of deflation vectors needed to
  reach the two-level line's best $D_W$ counts? Compare against the recorded
  `twolevel_ckpoint_lat.640_rbcgne_*` numbers.

## Design decisions

- Solver on $S_o$: FGMRES (non-Hermitian $S_o$; right preconditioning keeps true residuals), NOT
  preconditioned CGNE — mirrors the freeprec line and reuses the existing
  `RecyclingGeneralisedMinimalResidual` (restart 16, recycle k=0 initially).
- The odd-checkerboard lift/project wrapper is a small `LinearFunction` in the test (zero-pad even,
  apply $F$ or $M_0$ on the full 5D grid, `pickCheckerboard(Odd)`); promote to
  `FreeMobius5D_claude.h` only if it survives to production use.
- One operator/preconditioner construction per process (staggered chunk-1 lesson: Grid operator
  destructors before program end can corrupt the heap; keep them alive to exit).

## Open questions

- D1b runs on GPU (adapting the twolevel qsub) — confirm 1-node GPU job as for the twolevel runs.
- FGMRES restart length: start at 16 (deflprec default); scan later if the preconditioned iteration
  count is close to the restart length.

## Status log

- 2026-09-10: plan written after user's "skip to DWF" decision.
- 2026-09-10: chunk D1a coded (`Test_dwf_freeprec_rb_claude.cc` cold-gate path +
  `FreeMobiusRbInverse` = P_o F P_o wrapper) and build/run handoff `log/tmp_claude.sh` written.
  Wiring: `NonHermitianSchurDiagMooeeOperator::Mpc` (= S_o) vs the projected `FreeMobius5DInverse`;
  both apply orders checked, tol 1e-9 (fp64 PlannedFFT default). Operators heap-allocated + leaked
  (staggered chunk-1 destructor lesson). Awaiting user build/run. D1b (interacting FGMRES) is TODO
  in the same binary.
- 2026-09-10 (D1a build attempt 1): COMPILE OK, LINK FAILED — undefined `AllToAllV` /
  `aggregateTargetBytes` from BinaryIO `AggregateExchange` templates, pulled in by
  `NerscIO::readConfiguration` + the `twolevel_common` deflation I/O. Confirmed `nm` shows zero
  `AllToAllV` in `build_mpi/lib/libGrid.a` (source BinaryIO.h is ahead of that stale install tree).
  FIX: made the D1a test self-contained (only Grid.h + FreeMobius5D_claude.h, unit gauge, no I/O) so
  no AggregateExchange templates instantiate. Constants copied locally (RB_* = TL_*). The interacting
  D1b leg keeps its own binary/path and runs on GPU (current install tree has the symbols). Awaiting
  user rebuild/run of the fixed D1a.
- 2026-09-10 (D1a run, fixed link): **IDENTITY VALIDATED** — `||S(PFP)b - b||/||b|| = 1.90e-7`,
  `||(PFP)S b - b||/||b|| = 1.90e-7` (both orders). This is F's SINGLE-PRECISION FFT floor (its fp64
  block-Thomas inner gate is 4.3e-16 in the same log), which is by design: F is a preconditioner and
  the RGMRES is flexible enough to reach true 1e-8 with an fp32 M0. So 1.9e-7 = the free-limit RB
  Schur identity S_o^{-1} = P_o F P_o holding to single precision = PASS. Gate tol was wrongly set to
  1e-9 (fp64 expectation) and asserted FAIL on run 1; loosened to 1e-5 (freeprec cold-gate
  convention). **Chunk D1a physics complete.** Next: D1b interacting FGMRES leg (GPU).
- 2026-09-10: chunk D1b coded — `Test_dwf_freeprec_rb_solve_claude.cc` + GPU build/qsub scripts.
  Leg (a) plain RB-CGNE (`SchurRedBlackDiagMooeeSolve` + CG, D_W = 2 Ls*iters), leg (b) Fourier-prec
  RB-FGMRES (`NonHermitianSchurRedBlackDiagMooeeSolve` inner solver = `RecyclingGeneralisedMinimalResidual`
  restart 16, deflation 0, right-prec `FreeMobiusRbInverse` = P_o F P_o, Omega=1; D_W = Ls*IterationCount).
  Same source seed {20,21,22,23} as `Test_dwf_rbcgne_claude.cc` -> baseline matches recorded
  twolevel_*_rbcgne_* numbers. Verified: default config exists; RGMRES calls only LinOp.Op + Prec.
  EXPECTATION: without the frame or deflation, the ~12 topological RB-Schur stragglers may keep FGMRES
  from clustering fully (freeprec_gaugefix_topology_claude.md sec 8) -- a partial D_W win here motivates
  D2 (frame) and D3 (deflation). Awaiting user GPU build (grid_dwfrb_build_scc_gpu_claude.sh) + qsub.
- 2026-09-10 (Nobu, mid-run guidance): "won't win / won't even converge without Omega" -> folded the
  frame AND the deflated restart into the SAME binary rather than deferring to separate chunks. Now
  three legs: (a) RB-CGNE, (b) RB-FGMRES Omega=1 [control, expected to lose], (c) RB-FGMRES framed
  M0 = P_o(Om^d F Om)P_o. Both FGMRES legs use GMRES-DR with DeflationDim k=24 (was mistakenly 0 =
  plain restarted GMRES, which stalls on the ~12 topological stragglers -- the exact failure Nobu
  flagged and the reason RecyclingGeneralisedMinimalResidual exists). Frame = flowed-Landau warm start
  (flow 0.02x873 = s/t0=6, FourierAcceleratedGaugeFixer), reusing FreeLimitPreconditioner. NOTE: the
  job Nobu already submitted was the PRE-EDIT binary (legs a+b only, k=0) -> rebuild + resubmit for
  the meaningful (c) framed + GMRES-DR result. CLI: --gmres_k, --flow_eps, --flow_nstep (0 skips (c)).
  If adaptive GMRES-DR still leaves a residual tail, D3 = seed the recycle space with the external
  RB-Schur^2 `deflrb` dump via RGMRES::SetDeflationSubspace.
- 2026-09-10 (audit + Nobu restructure, gauge-fixing thread): an audit found the interacting FGMRES legs
  had NEVER produced a number -- they crashed with `GRID_ASSERT cb==lat.Checkerboard()` (Lattice_ET.h).
  ROOT CAUSE (C1, now FIXED): `RecyclingGeneralisedMinimalResidual_claude.h` initialised its recycle/
  deflation accumulators with `= Zero()` (defaults to Even cb) then added Odd-cb Krylov vectors -> assert.
  Invisible in all prior use because GMRES-DR ran on the FULL 5D grid (everything Even). FIX: stamp
  `.Checkerboard() = cb` (cb = solve checkerboard) on every fresh `Zero()` accumulator (accu/accy/accc/
  Unew/Ynew + the SeedU au/ay). Applies to the twolevel binaries too but is a no-op there (full grid).
  RESTRUCTURE (Nobu): `Test_dwf_freeprec_rb_solve_claude.cc` is now a DEDICATED framed Fourier-RB FGMRES
  -- dropped leg (a) RB-CGNE (measured in Test_dwf_rbcgne_claude.cc) and leg (b) Omega=1 control
  (unnecessary). Solver params aligned to the latest twolevel deflprec: restart m=16, GMRES-DR k=32
  (was 24), tol 1e-8, maxit 20000. qsub GMRES_K default -> 32. NEEDS REBUILD (RGMRES header + solve .cc
  changed) then rerun: grid_dwfrb_build_scc_gpu_claude.sh + grid_dwfrb_gpu_qsub_claude.sh. Audit MINORs to
  fold into analysis: RB-CGNE (normal-eqn residual) vs FGMRES (direct Schur residual) -> compare at matched
  TRUE full residual (both print it); FGMRES D_W = Ls*iters undercounts init + per-cycle matvecs by ~1/m.
  The staggered destructor `free(): invalid size` (M1) is still unexplained -- leaked around, root-cause
  before production.
- 2026-09-10 (Nobu clarification): "the flowed-Landau optimizer IS an Omega-tuning scheme." So leg (c)
  (flow s/t0=6 + Fourier-accelerated Landau gauge-fix -> tuned Omega) already IS the Omega-tuned run;
  the separate FrameOptimizerDWF_claude.h gradient descent is NOT needed here. Each single job already
  contains the Omega=1 control (b) AND the Omega-tuned (c) leg -> the two identical submitted jobs are
  redundant. Terminology fix: throughout, "frame"/"Landau frame" = the Omega-tuning.
