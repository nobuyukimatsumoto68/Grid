# disc LMA ESTIMATOR bench + shared header -- impl plan

## STATUS: chunks 1-4 DONE + validated (2026-06-26)
Shared header `disc_lma_v2_common_claude.h` + thin `disc_lma_bench_v2_claude.cc` (refactored,
behavior-identical) + new `disc_lma_estimator_bench_v2_claude.cc` (Make.inc + tmp_claude.sh Phase 6).
DET GATE = 1.6e-8 (projection UNBIASED to solver tol for ANY modes -- the construction $u_i=M_{pc}v_i/\sigma_i$
in double makes the split cancel term-by-term; RR is REDUNDANT for correctness, only buys variance).
4^4 HOT variance ratio var(lma)/var(plain) ~ 5-13 (>1) -- EXPECTED: gapped spectrum, no near-zero modes,
so LMA has no low-mode variance to remove. Eigensolve is MIXED-precision (single Lanczos subspace ->
double RR/A2A/solve), NOT single-prec. Next: chunk D production binary, OR a smooth/lighter local config
to demonstrate ratio<1.

## Goal
Now that the eigensolve (Cheby + RR) and the A2A reconstruction work
(`disc_lma_bench_v2_claude.cc`, see `disc_lma_cheby_v2_impl_plan_claude.md` STATUS: WORKING),
build the actual **LMA variance-reduction estimator** (chunk B/C of
`disc_lma_impl_plan_claude.md`) as a SEPARATE bench `.cc`, reusing the eigensolve/A2A machinery
through a shared HEADER so nothing is duplicated.

Physics (from `disc_lma_impl_plan_claude.md`, unchanged): split the loop
$$L_\Gamma(t,\vec p)=\underbrace{L_\Gamma^\text{low}}_{\text{exact, all }x,\text{ no noise}}
                    +\underbrace{L_\Gamma^\text{high,stoch}}_{\text{stochastic, low modes projected out}}.$$
- Exact low: $L_\Gamma^\text{low}(x)=\sum_i \sigma_i^{-1}\,\mathrm{tr}_{sc}[\Gamma\,a_i(x)\,b_i(x)^\dagger]$.
- Source projection (chunk B): per Z4 dilution source $\eta$, per column $sc$: Import + RedBlack
  reduction to the odd-cb source $b_o' = \texttt{src}_o - M_{oe}M_{ee}^{-1}\texttt{src}_e$
  (BEFORE $M_{pc}^\dagger$); project out the low modes onto the ORTHONORMAL $u_i$,
  $b_o'^\perp = b_o' - \sum_i u_i\langle u_i,b_o'\rangle$; solve the now well-conditioned
  $M_{pc}x_o = b_o'^\perp$ (via $M_{pc}^\dagger M_{pc}x_o = M_{pc}^\dagger b_o'^\perp$); back-
  substitute + Export $\to S_\text{high}\eta$. Key identity $M_{pc}^{-1}u_i=v_i/\sigma_i$
  $\Rightarrow S_\text{high}=S-S_\text{low}$, unbiased.
- LMA estimate $=L^\text{low}+\mathrm{tr}[\Gamma\,(S_\text{high}\eta)\,\eta^\dagger]$; plain estimate
  $=\mathrm{tr}[\Gamma\,(S_\text{full}\eta)\,\eta^\dagger]$ (existing path). Per-noise DETERMINISTIC
  cross-check: $S_\text{full}\eta = S_\text{high}\eta + S_\text{low}\eta$ to solver tol.

## Mode accuracy & estimator form (design Q, 2026-06-26)
Two LMA forms, with DIFFERENT sensitivity to mode accuracy:
- **Source projection (deflated solve) -- the plan, gives SPEED+variance.** Project the low
  modes out of the odd-cb source, solve the well-conditioned high system, add the exact A2A
  low part. UNBIASED requires the EXACT identity $M_{pc}^{-1}u_i=v_i/\sigma_i$. Total solve
  $$x_\text{tot}=\underbrace{M_{pc}^{-1}(1-\sum_i u_iu_i^\dagger)b_o}_{x_\text{high}}
    +\underbrace{\sum_i \tfrac{v_i}{\sigma_i}\langle u_i,b_o\rangle}_{x_\text{low,A2A}},$$
  and $x_\text{tot}=M_{pc}^{-1}b_o$ ONLY if $M_{pc}^{-1}u_i=v_i/\sigma_i$. With APPROXIMATE
  $(v_i,u_i,\sigma_i)$ there is a BIAS $\sim\|M_{pc}^{-1}u_i-v_i/\sigma_i\|\sim$ mode residual.
- **Recombination -- UNBIASED for ANY modes, but variance-only.**
  $$L=\underbrace{L^\text{low}_\text{A2A}}_{\text{noise-free, all }x}
     +\big[\,\mathrm{tr}\,\Gamma S_\text{full}\eta\eta^\dagger
            -\mathrm{tr}\,\Gamma S_\text{low,A2A}\eta\eta^\dagger\,\big].$$
  The approx low part CANCELS between the noise-free and stochastic pieces, so the central
  value is exact for any modes; only the VARIANCE gain depends on mode quality. Cost: it keeps
  the FULL (ill-conditioned) solve -> no speed, and it is the natural CROSS-CHECK of the
  projected form (it directly MEASURES the projected form's mode-accuracy bias).

Accuracy split (why RR is usually enough): RR (Rayleigh-Ritz) gives the EVALS very accurately
(Ritz values converge as residual$^2$ -> ~1e-10 even from a rough subspace), and evals are the
sensitive part ($1/\sigma_i$ weighting of $L^\text{low}$). RR EVECS are only SUBSPACE-accurate
(~2e-5 in our 4^4 run). The projected-form bias is then ~2e-5 of the low part -- far below the
disc STOCHASTIC error (~%), so RR likely suffices. To PERFECT eval+evec to double (the user's
point -- we recompute evals anyway and need both): do NOT run a fresh double Lanczos; instead
a FEW DOUBLE subspace-iteration rounds SEEDED by the single subspace -- apply the Cheby filter
(cheap matvecs) in double, re-orthonormalize, RR -- converges both to double. RR is the
0-iteration limit; add a `RR_ITERS` knob. (Block Lanczos only matters for EXACT/tight clusters
-- not needed on hot/real non-degenerate spectra; Grid's IRL is single-vector and converged.)

CORRECTION (2026-06-26, MEASURED): the projection is UNBIASED to SOLVER tol for ANY modes, NOT
biased by the mode residual. Reason: we DEFINE $u_i=M_{pc}v_i/\sigma_i$, so $M_{pc}^{-1}u_i=v_i/\sigma_i$
holds EXACTLY by construction (it does not need $v_i$ to be a true eigenvector). Hence the
projected-out part $M_{pc}^{-1}\sum_i u_iu_i^\dagger b_o$ exactly equals the A2A low part
$\sum_i\sigma_i^{-1}v_i\langle u_i,b_o\rangle$, and $S_\text{full}=S_\text{high}+S_\text{low}$ to
solver tol. The DET GATE confirmed this: **1.6e-8** (= solver 1e-8), not ~2e-5. So mode accuracy
affects the VARIANCE only (how much of the low part $S_\text{low}$ captures), NEVER the bias --
`RR_ITERS`/perfecting modes is never needed for correctness, only for extra variance reduction.

DECISION (2026-06-26): use the **PROJECTION** form as the estimator, and CODE-CHECK it with the
DETERMINISTIC per-noise identity
$$\big\|\,S_\text{full}\eta-(S_\text{high}\eta+S_\text{low}\eta)\,\big\| / \|S_\text{full}\eta\|.$$
$S_\text{full}\eta$ is one ordinary (un-deflated) solve; $S_\text{high}\eta$ is the projected
solve; $S_\text{low}\eta=\sum_i\sigma_i^{-1}a_i\langle b_i,\eta\rangle$. With EXACT modes this is
zero to solver tol; with APPROXIMATE modes it equals the mode residual -- so ONE check both
validates the projection arithmetic AND measures the projection bias (no statistics needed). This
is sharper than the stochastic recombination, so we do NOT separately code recombination (it
remains the conceptual sibling). If the measured residual ever approaches the target precision,
turn on `RR_ITERS` (double subspace-iteration polish) -- otherwise plain RR.

## Files
- NEW `disc_lma_v2_common_claude.h` -- shared machinery (the "separation"):
  `env_int/env_double`, `InverseHermOp`, `ReadEigref`, `ComputeChebWindow`,
  `RayleighRitzRefine`, `BuildPhysicalA2A`, a `LMAEigParams` knob struct, and the high-level
  drivers `BuildLowModes(...)` (eigensolve + subspace extract + RR -> `sub`,`eval_use`) and
  `BuildA2ASet(...)` (per-mode -> `a`,`b`,`u`,`sigma`).
- EDIT `disc_lma_bench_v2_claude.cc` -- include the header; delete the now-shared definitions;
  keep its eigensolve-validation behavior (GATES + L^low). Must build + run identically.
- NEW `disc_lma_estimator_bench_v2_claude.cc` -- the estimator bench (chunks B/C below).
- EDIT `Make.inc` -- register `disc_lma_estimator_bench_v2_claude`.
- EDIT repo-root run handoff (extend `tmp_claude.sh`, or NEW `run_lma_estimator_bench_claude.sh`).

## Ordered chunks
### 1 -- Extract shared header; refactor the v2 bench onto it (NO behavior change)
Files: `disc_lma_v2_common_claude.h`, `disc_lma_bench_v2_claude.cc`, `Make.inc`.
Move the shared pieces into the header behind `LMAEigParams` + `BuildLowModes`/`BuildA2ASet`.
Gate: the v2 bench still prints the same order/amplification/converged/GATE/L^low lines.

### 2 -- Estimator bench skeleton (exact low + PLAIN stochastic)
Files: `disc_lma_estimator_bench_v2_claude.cc`, `Make.inc`.
Build eigenbasis (`BuildLowModes`) + A2A set (`BuildA2ASet`) -> exact `L^low[gamma](t,p)`.
Reuse the original disc binary's `StochasticDilutedSource` (Z4, time+eo dilution) + `Solve`
(`SchurRedBlackDiagMooeeSolve`) for the PLAIN full loop. Print central values for id/g5.

### 3 -- Source-projected high solve (chunk B core) + unbiasedness
Files: `disc_lma_estimator_bench_v2_claude.cc`.
`SolveHighProjected(D, HermOpEO, u[], eta4 -> Shigh4)` per the projection above. LMA estimate
= L^low + tr[Gamma Shigh eta eta^dag]. DETERMINISTIC per-noise gate:
$\|S_\text{full}\eta-(S_\text{high}\eta+S_\text{low}\eta)\|/\|S_\text{full}\eta\|$ to solver tol
(the decisive correctness check; $S_\text{low}\eta=\sum_i\sigma_i^{-1}a_i\langle b_i,\eta\rangle$).

### 4 -- Variance (the deliverable)
Files: `disc_lma_estimator_bench_v2_claude.cc`.
Accumulate over `NSRC` independent noise samples; per gamma/t report mean (LMA vs plain agree
within errors -- check $\Delta t\neq0$) and VARIANCE ratio var(LMA)/var(plain) vs $N_{ev}$.

## Knobs / decisions
- New bench name: `disc_lma_estimator_bench_v2_claude.cc` (open to a shorter name).
- Testbed: same 4^4 HOT, m=0.01 (mechanics + rough variance); bigger lattice later for real variance.
- New knobs: `NSRC` (noise samples), outer solve tol fixed 1e-8 (NEVER relaxed), `DO_PLAIN` (on).
- Header is `_claude.h`; one-statement-per-line; no Unicode (LaTeX macros in comments).

## Open questions
1. OK to refactor `disc_lma_bench_v2_claude.cc` onto the header in-place (chunk 1), or keep it
   untouched and only have the NEW bench use the header (less churn, some duplication)?
2. Bench name `disc_lma_estimator_bench_v2_claude` acceptable?
3. Variance test on 4^4 HOT is mechanics-only (tiny volume); fine for now, or pick a larger local
   lattice for a meaningful variance number?
