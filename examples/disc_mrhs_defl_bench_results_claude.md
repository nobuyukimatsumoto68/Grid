# disc speedup #2+#3 -- benchmark results (measured spectrum + Nev sweep)

Companion data file for `disc_mrhs_deflation_impl_plan_claude.md`. Records the
measured low spectrum and deflation timings from the bench
`disc_mrhs_defl_bench_claude.cc`.

## Run / setup
- Run id: `disc_bench_f3GCqKSc2NW3` (2026-06-23), output at su4_32c ROOT.
- Config: `conf_nc4nf1_2448_b10p800_m0p0100_lat.758`, mass $m=0.01$, $M_5=1.5$,
  Mobius $L_s=16$, $b=1.5$, $c=0.5$, anti-periodic T.
- Layout: 8 nodes / 32 ranks, grid $24^3\times48$, mpi `2.2.2.4` (production layout).
- Operator probed: the even/odd Schur operator $\hat H = $ `SchurDiagMooee` (the
  SQUARED Dirac op; eigenvalues $\sim$ singular-value$^2$), single precision.

## Spectrum
- $\lambda_\text{max} \approx 81.8$ (power method).
- $\lambda_\text{min} \approx 4.83\times10^{-4}$ (= eval[0] below; the earlier
  inverse-power probe gave $\sim5\times10^{-4}$, consistent).
- $\kappa = \lambda_\text{max}/\lambda_\text{min} \approx 1.7\times10^5$,
  $\sqrt\kappa \approx 410$ (matches the $\sim$3000-iter CG baseline).

### Lowest eigenvalues of $\hat H$ (shift-invert Lanczos, $N_\text{conv}=100$)
Only the first 20 were printed in this run (the bench now prints all $N_\text{conv}$
in the next build). The bottom is DENSE -- 20 modes barely double:

| i | eval[i] | i | eval[i] |
|---|---------|---|---------|
| 0 | 4.832e-4 | 10 | 6.512e-4 |
| 1 | 4.925e-4 | 11 | 6.692e-4 |
| 2 | 4.989e-4 | 12 | 6.882e-4 |
| 3 | 5.095e-4 | 13 | 7.001e-4 |
| 4 | 5.160e-4 | 14 | 7.323e-4 |
| 5 | 5.359e-4 | 15 | 7.464e-4 |
| 6 | 5.573e-4 | 16 | 7.588e-4 |
| 7 | 5.688e-4 | 17 | 7.998e-4 |
| 8 | 6.048e-4 | 18 | 8.563e-4 |
| 9 | 6.054e-4 | 19 | 9.052e-4 |

eval[0..19] span $[4.83, 9.05]\times10^{-4}$ -- a dense near-zero cluster (light-mass
Banks-Casher regime). FULL 100-mode spectrum (run `disc_bench_f3GHQ9uSeMyh`,
2026-06-24): grows $\sim$20x over 100 modes (densest at the very bottom, spreading
higher). eval[i] (i:value):
```
 0:4.832e-4   5:5.359e-4  10:6.512e-4  15:7.464e-4  20:9.823e-4
25:1.199e-3  30:1.523e-3  35:1.734e-3  40:2.237e-3  45:2.718e-3
50:3.126e-3  55:3.662e-3  60:4.328e-3  65:4.945e-3  70:5.625e-3
75:6.144e-3  80:6.849e-3  85:7.775e-3  90:8.612e-3  95:9.209e-3  99:9.728e-3
```
(full per-index list in the log). eval[100] would be $\sim10^{-2}$ -> deflating 100
modes lifts the effective floor from $4.8\times10^{-4}$ to $\sim10^{-2}$ (a $\sim$20x
lift -> $\sqrt{}\sim4.5$x in the deflated-mode condition, but the SOLVE only needs to
converge the undeflated tail, see sweep).

## Eigensolve (shift-invert / inverse Lanczos)
- Method: `EIG_METHOD=2`, Lanczos on $\hat H^{-1}$ (inner CG to `INV_TOL=1e-4`),
  Nstop=Nk=100, Nm=180. NO window tuning.
- Result: `#modes converged: >= 100/100`, $N_\text{conv}=100$, wall $= 643$ s
  ($\approx$ 11 min). Chebyshev IRL by contrast converged 0/24 -- shift-invert is the
  robust choice for this wide/dense spectrum.

## Nev sweep -- 16-RHS batched mixed-prec solve (outer tol 1e-8)
Run `disc_bench_f3GHQ9uSeMyh` (2026-06-24), MAXPATCH=1000, full sweep completed.
| Nev | wall (16 RHS) | speedup vs Nev=0 |
|-----|---------------|------------------|
| 0   | 168.0 s | 1.00x |
| 25  | 131.9 s | 1.27x |
| 50  | 115.6 s | 1.45x |
| 100 | 106.4 s | 1.58x |
Solve decrements: 168.0 -> 131.9 (-36.1) -> 115.6 (-16.3) -> 106.4 (-9.2 over 50):
decelerating but NOT yet plateaued.

## Break-even (per config = setup + 96 source groups), setup(Nev) ~ (Nev/100)*627 s
| Nev | setup | 96*solve | T(Nev) | per-config speedup |
|-----|-------|----------|--------|--------------------|
| 0   | 0 s    | 16132 s | 16132 s | 1.00x |
| 25  | 157 s  | 12663 s | 12820 s | 1.26x |
| 50  | 314 s  | 11102 s | 11416 s | 1.41x |
| 100 | 627 s  | 10213 s | 10840 s | 1.49x |
Marginal (50->100): solve saves 889 s/config, setup adds 313 s -> NET -576 s, i.e.
benefit 17.8 s/mode > cost 6.3 s/mode -> the SPEED optimum is somewhat BEYOND 100,
but absolute per-config gain is modest (~1.5x) and shrinking. A bigger Nev is the
variance/LMA lever (future), not a speed win.

## Implications
- Deflation works but with DIMINISHING RETURNS here: the dense low spectrum means
  deflating $N_\text{ev}$ modes only lifts the effective $\lambda_\text{min}$ to
  eval[$N_\text{ev}$], which grows slowly. Nev=25 -> 1.26x; Nev=100 likely
  $\sim$1.5-2x, not 5-10x. A 5x cut ($\lambda_\text{cut}\approx0.012$) would need
  $O(\text{several hundred})$ modes at this density.
- Break-even still favorable: setup (643 s for 100 modes) is amortized over the 96
  source groups/config, so even a modest per-solve speedup nets a per-config win.
- OPEN EFFICIENCY ISSUE (chunk #1): the mixed-prec batched baseline is 163.7 s /
  16 RHS $= 10.2$ s/solve; the reliable update runs 3 FULL outer iterations because
  `InnerTolerance=1e-8`. Loosening InnerTolerance ($\sim$1e-4) should cut the outer
  work substantially -- revisit before trusting absolute speedups vs the production
  double-prec solve.

## Solver comparison (run disc_bench_f3GJdphTkejq, 2026-06-24, Nev=0, InnerTol=1e-4)
| solver | wall | per-RHS |
|--------|------|---------|
| plain DOUBLE CG (1 RHS)      | 12.71 s | 12.71 s |
| mixed-prec CG (1 RHS)        |  8.86 s |  8.86 s |
| mixed-prec BATCHED (16 RHS)  | 139.6 s |  **8.72 s** |
=> **mixed-prec is ~1.43x over plain double** (chunk #1 pays off). **BATCHING gives
~NO gain** (8.72 vs 8.86 s/RHS) -- on this APU/operator the multi-RHS solve does not
amortize (chunk #2 not helping at Nev=0). Corrects an earlier worry: production
double at m=0.01 is ~12.7 s/solve (the memory "950s/config" was a HEAVIER ensemble),
so mixed-prec is a genuine speedup, not a regression.
OPEN: does batching help WITH deflation? (deflation projection is batched) -> being
measured via DO_BATCHCMP (batch=1 vs 16 at Nev=100) in the next run.

## NaN breakdown at 200 modes (same run)
shift-invert Lanczos with INV_TOL=1e-4 broke down at step 200: `alpha[200]=(-nan)
beta[200]=nan`, converged 0/200, then the inner CG spun on the NaN source to maxit.
100 modes was fine -> the inexact inner solve (1e-4) loses orthogonality pushing to
200. FIX: INV_TOL=1e-5, NSTOP capped at 150, + NaN guard in InverseHermOp (bail on
non-finite/zero source). The whole Nev sweep produced nothing that run.

## Pending / open
- DECISION (Chunk B): per-config speed gain is only ~1.5x at Nev=100 and the
  shift-invert setup is 627 s/config -- modest. Go/no-go on the production variant
  (Chunk D) hinges on the EFFICIENCY issue below + the variance/LMA benefit.
- **CRITICAL open**: the mixed-prec batched solve is 106-168 s / 16 RHS
  = 6.6-10.5 s/solve, doing 3 FULL outer iters (InnerTolerance=1e-8). Must (a)
  compare to the CURRENT production PLAIN-DOUBLE CG at m=0.01 (the disc binary's
  SchurRedBlackDiagMooee, tol 1e-8) -- if production double is much faster, the
  mixed-prec batched path is a REGRESSION; and (b) loosen InnerTolerance (~1e-4)
  and re-measure the baseline before trusting any speedup. THIS dominates the
  go/no-go, more than Nev tuning.
- If pursued: Chunk D production variant + (future) LMA for the disc estimator
  variance (see impl-plan "FUTURE DIRECTION").
