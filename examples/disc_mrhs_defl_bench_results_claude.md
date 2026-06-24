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
Banks-Casher regime). eval[20..99] PENDING (next run prints the full 100).

## Eigensolve (shift-invert / inverse Lanczos)
- Method: `EIG_METHOD=2`, Lanczos on $\hat H^{-1}$ (inner CG to `INV_TOL=1e-4`),
  Nstop=Nk=100, Nm=180. NO window tuning.
- Result: `#modes converged: >= 100/100`, $N_\text{conv}=100$, wall $= 643$ s
  ($\approx$ 11 min). Chebyshev IRL by contrast converged 0/24 -- shift-invert is the
  robust choice for this wide/dense spectrum.

## Nev sweep -- 16-RHS batched mixed-prec solve (outer tol 1e-8)
| Nev | wall (16 RHS) | speedup vs Nev=0 | notes |
|-----|---------------|------------------|-------|
| 0   | 163.7 s | 1.00x | baseline, $\sim$3000 inner iters/RHS, 3 outer iters |
| 25  | 129.9 s | 1.26x | inner iters $\sim$2690 |
| 50  | PENDING | -- | aborted: `MAXPATCH=50` too small (needed 51); fixed -> 1000 |
| 100 | PENDING | -- | -- |

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

## Pending
- Rebuild + resubmit (MAXPATCH=1000, full-spectrum print) -> Nev=50/100 timings +
  the complete 100-mode eval list (append here).
- Then go/no-go on the production deflation variant (Chunk D) given the modest
  speedup, weighed against the disc-estimator stabilization benefit at light mass.
