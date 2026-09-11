# Two-level deflation CHECK -- implementation plan (Nobu 2026-09-08)

Design context: [[freeprec_twolevel_deflation_design_claude.md]], [[project-deflation-gmres]].
This plan = the concrete first CHECK Nobu specified. MAJOR addition (new DWF direct-Omega optimizer).

## Goal / hypothesis
Deflate the ~10 lowest 5D HERMITIAN domain-wall modes, then frame-optimize $M_0 D_{DW}$ DIRECTLY in the
deflated complement. HOPE: this lifts the low OUTLIERS of the preconditioned spectrum $M_0 D_{DW}$ (the
Re~0.3 stragglers) toward 1 -> smaller quality metric $C$ on the low modes -> fewer GMRES inverse counts.
Quality metric (confirmed, the pivot object): $C(v) = \lVert(1-M_0 D)v\rVert/\lVert v\rVert$ (=$|1-\mu|$
on an eigenmode; Test_wilson_frameopt_claude.cc:149).

## The check (per config; 8 thermalized configs)
1. Compute the lowest **24** modes of the 5D Hermitian DW operator (deflate the lowest **~10**; keep all
   24 for the $C$ diagnostic). Operator = $|D_{DW}|^2$ (MdagM) via the run_sv Chebyshev-IRL (already
   written; extend to RETURN the eigenvectors). $V_\text{low}$ = lowest 10; $V_{24}$ = lowest 24.
2. Build **frame A (no deflation)**: direct-$\Omega$ opt of $M_0 D_{DW}$ on the FULL space (no projector),
   $L[\Omega]=\sum_v \lVert (M_0(\Omega)D-1) v\rVert^2$. (LANDAU is a SEPARATE reference -- taken from the
   CURRENT Phase B run's GMRES-DR(16,32) counts, not recomputed here.)
3. Build **frame B (with deflation)**: SAME direct-$\Omega$ opt but restricted to the complement
   $P = 1 - V_\text{low}V_\text{low}^\dagger$: minimise $L_P[\Omega]=\sum_v \lVert P(M_0(\Omega)D-1)P v\rVert^2$.
   A vs B isolates the DEFLATION's effect on the optimised frame (identical optimizer, only P differs);
   Landau -> A isolates the direct-opt gain (the old ~1.28x).
4. For EACH frame, measure the BARE preconditioned-operator quality on all **24** low modes:
   $C(\psi_i) = \lVert(1 - M_0 D)\psi_i\rVert / \lVert\psi_i\rVert$ (NO projector -- deflating a deflated
   mode is trivial; the nontrivial data is how $M_0 D$ actually acts on them). The interesting comparison
   is on modes ~10-24 (the near-zeros the frame B opt was free to target, having handed the lowest ~10 to
   the deflation): does frame B give lower $C$ there than frame A? (On the deflated 0-10, frame B may be
   worse -- it ignored them -- but the solve deflates them, so that is fine.)
5. For EACH frame, measure the **inverse count** with GMRES-DR at the FIXED $(m,k)=(16,32)$ (isolates the
   frame effect; the solve is identical, only $\Omega$ differs). Masses 0.1, 0.01, 0.001.
COMPARE: does frame B lower $C$ on the non-deflated low modes (lift the outliers) and cut the inverse count
vs frame A?

## 8 thermalized configs (span |Q|; all plaquette-thermalised, traj>=100)
|Q|: 60(8), 100(6), 180(5), 200(4), 300(3), 640(3, reference), 740(2), 760(1). (Adjust as needed.)
|Q| known per config from flow data (topological core = chi_Q); deflate a FIXED ~10 (covers <|Q|>+3sigma,
sigma_Q~1.8, |Q| range 1-8).

## Files
- NEW `Grid/Grid/qcd/utils/FrameOptimizerDWF_claude.h` -- DWF direct-$\Omega$ optimizer (port of the
  Wilson fo_loss/fo_loss_force/fo_descend, Test_wilson_frameopt_claude.cc:220-423) with an OPTIONAL
  projector $P$ (deflation subspace). Loss/force/descend on $M_0=\Omega^\dagger F \Omega$, $F=$
  FreeMobius5DInverse, interacting op $D_{DW}=$ MobiusFermionD.
- NEW `Grid/tests/solver/Test_dwf_twolevel_claude.cc` -- the check driver (modes -> frames A/B -> C ->
  GMRES-DR count), reuses run_sv (eigvecs), FreeMobius5DInverse, RecyclingGeneralisedMinimalResidual.
- Build/run scripts: `grid_twolevel_build_scc_gpu_claude.sh`, `grid_twolevel_gpu_qsub_claude.sh`
  (8-config loop, or one job per config via a wrapper).

## Ordered chunks
1. **[DONE 2026-09-08, GPU-run PENDING] 5D Hermitian eigenvectors.** `Test_dwf_twolevel_claude.cc` +
   `compute_low_modes()` (Chebyshev-IRL, ODD order, cheb_hi auto; ports run_sv). Loads config, builds
   MobiusFermionD, computes + STORES the lowest N (default 24) |D_DW|^2 modes, prints sigma_i + chi_q.
   VALIDATED on 640 (job 7502205): sigma reproduces the saved sv data EXACTLY (0.1229,0.1232,...),
   lambda_max 133.28 -> cheb_hi 146.6. Nstop=24 reveals the cluster runs to sigma~0.17 (broader than the
   old Nstop=12 view). FIXED a reporting bug: |Q| = NET INDEX = sum chi_q (~3), NOT the count of
   |chi_q|>0.8 modes (=7, over-counts because +/- non-topological pairs are individually chiral).
   Machinery CONFIRMED -> green-light chunk 2.
2. **[DONE + FD-GATE PASSED 2026-09-09] DWF direct-$\Omega$ optimizer (the MAJOR piece).**
   CRITICAL FINDING (numerical gradient check earned its keep): the Mobius kernel (b!=c) is NOT
   g5R5-Hermitian -- [D_minus, P_pm] != 0 (Nobu confirmed analytically; verified numerically: Shamir c=0
   -> Gamma5 M Gamma5 = M^dag to 3e-15, Mobius c=0.5 -> 0.42; Grid's own Gamma5R5HermitianLinearOperator
   agrees). So F^dag != Gamma5 F Gamma5 (my first assumption, gave a 10% gradient error). FIX: F^dag =
   TRUE adjoint (D_free.Mdag)^{-1} = D_free (D_free^dag D_free)^{-1} (Grid's consistent adjoint, as CGNE
   uses) via CG on the free MdagM (well-conditioned) -- `FreeMobiusAdjInverse` functor in the test. FD
   gate on 4^4 now PASSES: rel.err O(eps^2) 8.7e-4 -> 4e-10 (floor) as eps 1e-2 -> 1e-5. Analytic went
   -2427.6 (wrong) -> -2175.79 (= the FD all along). PRECONDITIONER UNAFFECTED: F = D_free^{-1} exactly
   (3.8e-16), used forward only in M0/GMRES-DR/Phase B. (Speed TODO: FFT-fast applyDag = Minv^dag +
   conj-twist-phases, validate vs D_free.Mdag; the CG-F^dag is fine for the 8-config check.)
   `Grid/Grid/qcd/utils/FrameOptimizerDWF_claude.h`: fodwf_loss / fodwf_loss_force / fodwf_grad_check /
   fodwf_descend, free functions on concrete LatticeFermionD. Omega applied as the 5D broadcast Omega5
   (matches FreeLimitPreconditioner); $F^\dagger=\Gamma_5 F\Gamma_5$ via G5R5 (Mobius $\Gamma_5$-Herm);
   4D gradient = $Ta(\sum_s M5)$ (sum the 5D colour-matrix gradient over the 5th dim, since $\Omega_4$ is
   s-independent). Optional projector P = 1 - sum psi psi^dag (Vlow empty -> full space = frame A). Gate:
   $s=\Omega^\dagger\Gamma_5 F\Gamma_5\Omega\,Pr$, $M5{+}{=}\mathrm{traceSpin}(w{\otimes}s)-\mathrm{traceSpin}(c{\otimes}Pr)$.
   Wired into Test_dwf_twolevel_claude.cc `--fdgate` (random SU(3) frame, nprobe probes, eps sweep 1e-2..1e-4,
   PASS if rel.err<1e-3). CPU fsyntax-only clean. Build + run `grid_twolevel_gpu_qsub_claude.sh GATE=1`.
3+4. **[DONE code + 4^4 smoke-run 2026-09-09, GPU-run PENDING] pipeline.** Test_dwf_twolevel_claude.cc
   `--pipeline`: Landau warm-start (flow s/t0=6 + FA gauge-fix) -> frame A (fodwf_descend, Vempty=full
   space) vs frame B (fodwf_descend, Vlow=lowest defl_k=10 of the 24 modes) -> bare C(psi_i)=
   ||psi_i - M0 D psi_i||/||psi_i|| on all 24 (report mean/max over ALL and over the NON-deflated 10-24) +
   GMRES-DR(16,32) inverse count, for Landau / A / B. Same probes for A,B; F^dag = FreeMobiusAdjInverse
   (CG on free MdagM). CLI --defl_k/--fo_iter/--fo_nprobe/--flow_nstep. Ran clean on 4^4 (numbers not
   physical: hot/tiny). Build fp32 (grid_twolevel_build_scc_gpu_claude.sh), run PIPELINE=1
   (grid_twolevel_gpu_qsub_claude.sh) over 8 cfgs spanning |Q|. NOTE the CG-F^dag makes the descent the
   slow part (6h wall); FFT-fast applyDag is the later speed win.

3old. **C-diagnostic (bare preconditioned operator).** $C(\psi_i)=\lVert(1-M_0 D)\psi_i\rVert/\lVert\psi_i
   \rVert$ (NO projector), over all 24 modes, per frame; print per-mode (flag the non-deflated 10-24 = the
   interesting ones) + summary (mean/max on 10-24, count $C\ge1$). Store 24, deflate 10. Files:
   Test_dwf_twolevel_claude.cc.
4. **Pipeline + solve.** Per config: modes -> frame A (full-space direct-opt) -> frame B (deflated-opt, P)
   -> C (both) -> GMRES-DR(16,32) count (both), masses 0.1/0.01/0.001. Landau reference = from the Phase B
   run (not recomputed). Build + qsub over the 8 configs. Files: Test_dwf_twolevel_claude.cc, scripts.
5. **Analysis.** $C_A$ vs $C_B$ on the 24 modes (outlier lift?), inverse-count A vs B vs |Q|, across the
   8 configs. Plot (colour-blind: marker+colour). Files: plot .gp + a summary .md.

## Open questions (resolve before/within the chunk)
- Q_grad: confirm FreeMobius5DInverse Hermiticity ($\Gamma_5 F \Gamma_5 = F^\dagger$?) for the force. FD
  gate is the safety net regardless.
- Q_frameA: RESOLVED (Nobu) -- frame A = full-space direct-$\Omega$ opt (same optimizer as B, no P).
  Landau is a SEPARATE reference taken from the current Phase B run, NOT recomputed in this test.
- Q_solve: GMRES-DR(16,32) with frame only (clean isolation, DEFAULT), or ALSO feed $V_\text{low}$ as a
  fixed solve-deflation subspace (the full layer-3)? Default = frame-only; add solve-deflation as a variant.
- Q_defl_count: fixed 10 for the frame opt (Nobu: 3sigma<~10). C on 24. Revisit per-config (|Q|+N_bulk)
  later.
- Q_modes_op: |D_DW|^2 (smallest sigma, run_sv ready) vs g5R5 D_DW (signed) -- use |D_DW|^2.

## SVD-deflation + M0-complement -- the correct two-level scheme (Nobu 2026-09-09)

Supersedes the failed "seed the D5^H D5 low modes as a FROZEN GMRES-DR harmonic-Ritz recycle" attempt
(precond run j7510214: 192 iters adaptive -> NON-convergent when the 10 singular vectors were frozen into
the Ritz recycle). Root cause: $D_5$ is NON-normal, so its low modes are SINGULAR vectors, not eigenvectors
of $M_0 D_5$; feeding right-singular vectors into an eigen-recycle is ill-posed. The correct realization is
Saad's deflation done as an SVD (deflate_impl_plan_claude.md solve D), with $M_0$ preconditioning ONLY the
complement.

### Setup (once per config)
From Chebyshev-IRL on $|D_5|^2 = D_5^\dagger D_5$: the $k$ smallest RIGHT singular vectors $v_i$
($D_5^\dagger D_5\, v_i = \lambda_i v_i$, $\sigma_i = \sqrt{\lambda_i} \approx m$ for the topological ones),
and the matching LEFT singular vectors
$$
u_i = D_5 v_i / \sigma_i, \qquad D_5 v_i = \sigma_i u_i, \quad D_5^\dagger u_i = \sigma_i v_i .
$$
Orthonormalise $U = [u_1..u_k]$ (Gram-Schmidt; $V$ already orthonormal from IRL). $k$ = would-be-topological
count ($=|Q|=3$ on 640, or the low cluster $\sim 10$). Projectors $P_V = 1 - VV^\dagger$, $P_U = 1 - UU^\dagger$.

### Solve $D_5 x = b$
1. **Pre-solve the topological modes, NO preconditioner** (exact spectral inverse on the singular subspace):
$$
x_\text{lo} = \sum_{i=1}^{k} \frac{u_i^\dagger b}{\sigma_i}\, v_i = V\,\Sigma^{-1} U^\dagger b .
$$
2. **Deflated remainder RHS**: $b_\perp = P_U\, b = b - U U^\dagger b$ (note $D_5 x_\text{lo} = U U^\dagger b$).
3. **$M_0$-preconditioned GMRES on the complement**: solve
$$
\hat A\, x_\perp = b_\perp, \qquad \hat A \equiv P_U\, D_5\, P_V ,
$$
right-preconditioned by $M_0$ ($B = \hat A M_0$), $x_\perp \perp V$. $\hat A$ has smallest singular value
$\sigma_{k+1}$ (the near-null directions removed), and $M_0 \approx D_5^{-1}$ on exactly this complement, so
convergence is fast. Exactness: with true singular subspaces, $D_5 x_\perp \perp U$ whenever $x_\perp \perp V$
($D_5 = \sum_j \sigma_j u_j v_j^\dagger$), so the two blocks DECOUPLE and the scheme is exact.
4. $x = x_\text{lo} + P_V x_\perp$.

Metric: $D_W$ applies = $L_s \times$ (GMRES iters on $\hat A$) + $2k$ setup (forming $U$). Compare to plain
$M_0$-GMRES (192) and to RB-CGNE deflated by the RB-Schur$^2$ modes (the apples-to-apples partner).

### Why this JUSTIFIES direct-$\Omega$ opt of $\|1 - M_0 D_5\|$ in the deflated space (frame B)
$M_0 = \Omega^\dagger F \Omega$ is translation-invariant (FFT-diagonal free kernel in the frame), so it
CANNOT represent the sharply wall-localized would-be-topological modes -- they are the $\mathrm{Re}\approx
0.3$ stragglers of $\mathrm{spec}(M_0 D_5)$ and the residual difficulty the frame cannot remove. Once those
$k$ modes are handed to the EXACT deflation (step 1), the frame's job is only to make $M_0 D_5 \approx 1$ on
the COMPLEMENT $P_V(\cdot)P_V$ -- a target $M_0$ CAN meet. So minimising
$$
L_P[\Omega] = \sum_v \big\lVert P\,(M_0(\Omega) D_5 - 1)\,P\, v \big\rVert^2, \qquad P = P_V ,
$$
(frame B) is now well-posed: the optimiser is no longer fighting the modes it structurally cannot fix. Frame
A (full-space) vs B (deflated-complement) then isolates exactly this: does removing the topological modes
from the objective let the frame push the remaining spectrum closer to 1 (lower $C$ on modes $k$--24, fewer
complement-GMRES iters)?

### Open decisions (confirm before coding)
- $k$: 3 (strict topological, $|Q|$) or $\sim$10 (the low cluster to $\sigma\sim 0.18$)? Suggest running both.
- Replace the (broken) `gmresdr` stage with this `svddefl` solve, or add it as a new `--stage svddefl`?
- Include the projector in the operator (rigorous, above) vs init-guess-only (like RB-CGNE's DeflatedGuesser)
  -- suggest the projected operator for GMRES (init-guess alone does not remove the stalling eigenvalues).

## Sources
Frame optimiser: dwf4 dwf4_frameopt_claude.h (grid_frame_optimizer_impl_plan_claude.md). Deflation: Morgan
2002 (GMRES-DR), Parks/de Sturler 2006 (GCRO-DR), Stathopoulos-Orginos 2007 (eigCG low-mode deflation),
Banks-Casher 1980 (spectral density). SVD/singular-subspace deflation of the non-Hermitian solve: Saad,
"Iterative Methods" 2nd ed. ch. 6.5 (deflated CG/GMRES); the singular-subspace mirror is in
deflate_impl_plan_claude.md (solve D).
