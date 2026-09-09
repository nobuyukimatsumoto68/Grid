# Deflated / recycling GMRES for the free-limit preconditioner -- implementation plan (Nobu 2026-09-08)

## Goal / physics

Fix the restart-20 non-convergence of FGMRES($M_0$) (analysis: `freeprec_gaugefix_topology_claude.md`
\S8, \S8.1) by DEFLATING the $\sim 12$ isolated near-null eigenvalues of $M_0 D_\text{DW}$ (measured on
640, m0devals: 12 conj. pairs at $\mathrm{Re}\approx 0.3$) so a small restart window recovers
no-restart convergence at a memory budget that scales to $24^4$.

Deliverable: BOTH deflated variants, tested against plain FGMRES and RB-CGNE:
- **GMRES-DR$(m,k)$** -- deflation within one solve.
- **GCRO-DR** -- recycling the deflation subspace ACROSS solves (mass loop; later HMC MD sequence).

## Algorithm sources (cite in code + here, prominently)

- **GMRES-DR:** R. B. Morgan, "GMRES with deflated restarting", SIAM J. Matrix Anal. Appl. 24 (2002) 20.
- **GCRO-DR:** M. L. Parks, E. de Sturler, G. Mackey, D. D. Johnson, S. Maiti, "Recycling Krylov
  subspaces for sequences of linear systems", SIAM J. Sci. Comput. 28 (2006) 1651.
- Background: Saad & Schultz 1986 (GMRES); Saad, "Iterative Methods for Sparse Linear Systems" 2nd ed.,
  ch. 6.5 (harmonic Ritz / DEFLATION); Embree 2003 (restart stagnation).

## Key design decisions

1. **$M_0$ is a FIXED linear operator** ($\Omega^\dagger F \Omega$; $F$ = exact FFT inverse, no inner
   solve) -> use STANDARD right-preconditioned deflated GMRES on the constant operator $B \equiv A M_0$,
   solve $B u = b$, recover $x = M_0 u$ (one extra $M_0$ apply at the end). No flexible bookkeeping. The
   textbook GMRES-DR / GCRO-DR formulas apply verbatim. (Same holds for $M_1$ if ever wanted.)
2. **ONE class, a flag selects the variant.** GCRO-DR with the recycle space RESET each `operator()`
   call reproduces GMRES-DR for a single system (Parks et al.: the two give identical iterates for one
   system). So implement GCRO-DR once, expose `bool RecycleAcrossSolves`:
   - `false` -> GMRES-DR (subspace rebuilt from a first plain cycle each solve),
   - `true`  -> GCRO-DR (subspace $U,C$ persists across calls -> the mass loop / MD sequence).
   File: `Grid/Grid/algorithms/iterative/RecyclingGeneralisedMinimalResidual_claude.h`.
3. **Deflation selection = HARMONIC Ritz** (standard for GMRES-DR; targets interior/near-0). Computed
   from the small dense Hessenberg via Eigen (machinery already in
   `ImplicitlyRestartedArnoldi_claude.h`). Keep an UNROTATED copy of $\bar H_m$ (the stock FGMRES Givens
   pass overwrites $H$).
4. **Currency unchanged.** Per Arnoldi step: one $M_0$ apply ($D_W$-free) + one $A$ apply. $D_W$ count =
   $L_s \times$ (outer iters), identical bookkeeping to the current FGMRES($M_0$) count, so ratios stay
   comparable to the scan.

## Math to implement (harmonic Ritz + GCRO-DR restart)

Right-preconditioned operator $B = A M_0$. After an $m$-step Arnoldi on the (deflated) $B$:
$$
B V_m = V_{m+1}\bar H_m = V_m H_m + h_{m+1,m}\, v_{m+1} e_m^\top .
$$
Harmonic Ritz pairs $(\theta_i, g_i)$ (keep the $k$ smallest $|\theta_i|$):
$$
\big(H_m + h_{m+1,m}^2\, H_m^{-H} e_m e_m^\top\big)\, g_i = \theta_i\, g_i .
$$
GMRES-DR next cycle: new leading basis $Y_k = V_m G_k$ (orthonormalised), with images $B Y_k =
V_{m+1}\bar H_m G_k$; continue Arnoldi for $m-k$ more steps; the combined $[Y_k\,V_{m-k}]$ obeys
$B[Y_k\,V] = [\,\cdot\,]\bar G$ with $\bar G$ = (full $k\times k$ block) + Hessenberg tail; minimise
$\lVert \beta e_1 - \bar G d\rVert$.
GCRO-DR: keep $U_k$ (solution space), $C_k = B U_k$ with $C_k^H C_k = I$ (from a thin QR of $B U_k$);
project $x \mathrel{+}= U_k C_k^H r$, $r \leftarrow (I - C_kC_k^H) r$; inner GMRES on $\hat B = (I -
C_kC_k^H)B$ recording the coupling $B_{\text{cpl}} = C_k^H B V_m$; solve the $(m+k)$ combined LSQ; refresh
$U_k$ from harmonic Ritz of $[U_k, V_m]$; PERSIST $U_k,C_k$ to the next system.

## Files

- NEW `Grid/Grid/algorithms/iterative/RecyclingGeneralisedMinimalResidual_claude.h` -- the solver
  (template `<class Field>`, mirrors FGMRES interface + `RecycleAcrossSolves`, `DeflationDim k`).
- EDIT `Grid/tests/solver/Test_dwf_freeprec_claude.cc` -- add ops `gmresdr`, `gcrodr`; CLI
  `--deflate-k`; reuse `--restart`; report $D_W$ count + wall vs RB-CGNE and vs plain FGMRES; for
  `gcrodr` run the mass loop with the recycle space carried across masses and report the small-mass
  gain.
- NEW build/run scripts: `grid_freeprec_build_scc_gpu_merged_claude.sh` reused (header-only add ->
  just rebuild); `grid_freeprec_deflate_gpu_qsub_claude.sh` (single config, sweep $k$).
- (opt) seed $U_k$ from a one-off IRA on $B=A M_0$ via `ImplicitlyRestartedArnoldi_claude.h`.

## Ordered chunks

1. **[DONE 2026-09-08] Harmonic-Ritz + dense helpers** (Eigen): `HarmonicRitz_claude.h` (split out,
   Eigen-only, unit-testable). Unit test `scripts_nm/harmonic_ritz_unittest_claude.cc` PASSES: (T1) the
   $h^2 f e_m^\top$ correction matches an independently-formed reference; (T2) a known near-null pair
   $0.3\pm0.1i$ is recovered as the two smallest harmonic Ritz values.
2. **[DONE 2026-09-08, GPU-VALIDATION PENDING] GMRES-DR core** (`RecycleAcrossSolves=false`).
   Implemented as the GCRO-DR core (covers both; residual kept $\perp C$ so the per-cycle solve reduces
   to a standard GMRES LSQ for $d_v$ plus $d_u=-B_\text{mat}d_v$) in
   `RecyclingGeneralisedMinimalResidual_claude.h`, with the combined-space harmonic-Ritz refresh
   (`updateRecycleSpace`). CPU compile of the wired test = clean (fsyntax-only, exit 0). VALIDATE on 640
   via the $k$-sweep qsub: GMRES-DR(20,k) should approach the no-restart floor (175 iters @ m=0.1) once
   $k$ exceeds the knee.
   Files: `RecyclingGeneralisedMinimalResidual_claude.h`, `Test_dwf_freeprec_claude.cc`.
3. **GCRO-DR recycling** (`RecycleAcrossSolves=true`): persist $U_k,C_k$ across `operator()`. TODO not
   yet coded: at a new solve with a CHANGED operator (different mass), recompute $C=BU$ + re-orthonormalise
   (QR) + adjust $U$ BEFORE the projection (the recycle vectors are fields, valid across masses, but $C$
   is operator-dependent). Then hoist the solver object OUTSIDE the mass loop in the test so $U,C$ carry
   $0.1\to0.01\to0.001$. Files: same two.
4. **[PARTIAL: wiring DONE] Test wiring + scripts**: ops `gmresdr`/`gcrodr`, `--deflate-k`,
   `--deflate-sweep`, report vs RB-CGNE (count + wall) -- DONE. qsub `grid_freeprec_deflate_gpu_qsub_
   claude.sh` (k-sweep on 640) DONE. Remaining: broaden to the config set once the knee is fixed.
   Files: `Test_dwf_freeprec_claude.cc`, scripts.
5. **(optional) IRA seeding** of $U_k$. Files: `Test_dwf_freeprec_claude.cc`.

## Production plan (Nobu 2026-09-08): 2D (m,k) scan -> fixed (m,k) ensemble loop

GMRES-DR VALIDATED (job 7500570, config 640, restart m=20, ALL true 1e-8): deflation restores
no-restart convergence (18477 -> 180 iters), D_W-count win vs RB-CGNE 2.3--2.8x (m=0.1) / 3.8x (m=0.01);
win GROWS toward small mass because deflation removes the near-zero modes -> GMRES-DR iters ~mass-INDEPENDENT
(k=16: 540@m=0.1, 600@m=0.01) while RB-CGNE grows 252->1150. Mass-scaling is the headline.

TWO KNOBS: m = RestartLength (`--restart`, restart window, cost O(m^2)/cyc); k = DeflationDim
(`--deflate-k`, saved low modes, the physical near-null count). Realistic production: m~20, k~20.

PHASE A -- 2D (m,k) scan on the hard config 640, masses {0.1,0.01,0.001}, pick a fixed (m,k):
  test now loops `--restart-sweep` (m) x `--deflate-sweep` (k) in the gmresdr block (2D scan).
  qsub `grid_freeprec_2dscan_gpu_qsub_claude.sh` (default MSWEEP 16:24:32, KSWEEP 16:24:32, masses
  0.1:0.01:0.001). Choose (m,k) = best D_W-count win at acceptable memory/wall (sweet spot ~ m20,k24).
PHASE B -- loop the FIXED (m,k) over the b2.6 Iwasaki ensemble (42 cfgs), 3 masses:
  run script `grid_freeprec_run_gpu_qsub_claude.sh` now forwards RESTART + DEFLATEK; wrapper forwards them.
  SUBMIT: `OPS=gmresdr RESTART=<m> DEFLATEK=<k> MASSLIST=0.1:0.01:0.001
           CONFIGDIR=.../configs_iwasaki_16_b2.6 bash grid_freeprec_wrapper_claude.sh`  (DRYRUN=1 to preview).
  -> the deflated production win-vs-config table at true 1e-8 (generality check).
Both need the GPU binary REBUILT for the 2D-loop change (backward-compatible; single m/k if no sweep).
Next after Phase B: chunk 5 mixed precision (fp32 bulk + fp64 reliable update) for the WALL win; and the
3-set (3k+2m) storage optimisation for 24^4.

## Chunk 5 -- MIXED PRECISION (Nobu 2026-09-08: do AFTER Phase A/B settles (m,k))

SEQUENCING (approved): finish Phase A (2D m,k scan on 640, 3 masses) + Phase B (fixed (m,k) over the
ensemble) FIRST -> that fixes the production (m,k). THEN do chunk 5. Do NOT start chunk 5 before (m,k) is set.

WHAT: reliable-update mixed precision. Keep solution x + true residual r = src - A x in fp64; run the bulk
(Krylov V/Z, D_W applies, M0, deflation vectors U/Y/C) in fp32; recompute the true residual in fp64 each
cycle (my existing per-cycle `r = src - A psi` recompute IS the reliable-update hook). Delivered solution
accuracy UNCHANGED (true 1e-8) -- only intermediate storage/arithmetic goes fp32. This is NOT a change to
the problem; naive all-fp32 without the fp64 net is exactly the ~3e-7 floor bug we fixed.
WHY: ~2x memory (fp32 fields) AND ~2x wall (fp32 D_W applies -- the wall<1 lever). On 4-GPU it brings 32^4
deflation onto 40 GB cards (k=40 -> 32 GB/GPU). The dense harmonic-Ritz eigenproblem stays fp64 (Eigen).
HOW: solver already templated on Field -> instantiate RecyclingGeneralisedMinimalResidual<LatticeFermionF>
for fp32 storage inside a fp64 reliable-update outer (Grid pattern = ConjugateGradientMixedPrec). Deflation
vectors fp32 = fine (accelerator only).
BASELINE (Nobu, firm): the comparison is against MIXED-PREC RB-CGNE (ConjugateGradientMixedPrec /
SchurRedBlack with fp32 inner + fp64 reliable update), NOT the current fp64 RB-CGNE -- apples-to-apples wall.
The D_W-COUNT metric is precision-agnostic so the count wins are unchanged; this is purely for the honest
WALL comparison. Item 1 (3-set -> 2-set storage) = optional headroom only, deferred (needs harmonic-Ritz
reformulation, correctness risk, saves only 1/3; fp32 already fits 32^4/4-GPU).

## Decisions (RESOLVED 2026-09-08, Nobu)

- **Q1 -> ONE combined class** (`RecyclingGeneralisedMinimalResidual_claude.h`) + `RecycleAcrossSolves`
  flag: false = GMRES-DR, true = GCRO-DR.
- **Q2 -> default $k=24$**, CLI `--deflate-k`; final default set from the restart-scan knee.
- **Q3 -> HARMONIC Ritz** for deflation-vector selection.
- **Q4 -> $M_0$ ONLY** (M1 already converges at restart-20; leave it on plain FGMRES).
- **Q5 -> restart window $m=20$** paired with deflation (win is purely from deflation).

Gating: coding begins AFTER the restart-N scan on 640 lands (sets $k$ knee + gives the plain restart-20
baseline to beat). Chunk 1 (harmonic-Ritz dense helper + synthetic unit test) is scan-independent.
