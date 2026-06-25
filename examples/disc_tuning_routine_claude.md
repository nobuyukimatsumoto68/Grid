# disc eigensolve + solver TUNING ROUTINE (per physics parameter)

Reusable procedure to tune the deflation/LMA eigensolve and the mixed-precision
solver for EACH new ensemble (every mass $m$ / coupling $\beta$). The spectrum
($\lambda_\text{min}$, density, condition number) changes with $m,\beta$, so this
routine is re-run per ensemble. The tool is the bench
`disc_mrhs_defl_bench_claude.cc` driven by `submit_disc_bench_claude.sh` (env knobs).

Reference numbers below are for m=0.01 @ b10.8, config lat.758 ($24^3\times48$).

## Hard constraints (do NOT tune)
- **Outer solve tol = 1e-8** -- the physics target; NEVER relaxed (this was "#4",
  ruled out). Everything else below is machinery to reach it faster/quieter.
- **Layout = production 8 nodes / 32 ranks, mpi 2.2.2.4** for $24^3\times48$.
  1 node does NOT fit the MI300A host arena (per-rank fields too big). Same OPTIONS
  as `submit_disc_tuolumne.sh`.

## The knobs and what they control
| knob | controls | tuned value (m=0.01) |
|------|----------|----------------------|
| (power method) | $\lambda_\text{max}$ (UV edge $\alpha$) | ~81.8 |
| `CHEB_HI_FAC` | UV safety factor, $\alpha=\text{fac}\cdot\lambda_\text{max}$ | 1.1 (good UV est) |
| `CHEB_LO` | IR edge $\beta$ of the Chebyshev window | (Cheby only; see below) |
| `EIG_METHOD` | 1=Chebyshev IRL (cheap matvec), 2=shift-invert IRL (inner CG/matvec) | 2 default; PREFER 1 when it converges (shift-invert ~50% of wall) |
| `INV_TOL` | shift-invert inner-CG tol | 1e-5 (1e-4 NaN'd at 200) |
| `INNER_TOL` | mixed-prec inner single-CG tol | **1e-4** (1e-8 masks deflation) |
| `MAXPATCH` | mixed-prec final double patch-up cap | 1000 (50 aborted at Nev=50) |
| `ERESID`/`MAXITER` | Lanczos convergence resid / restarts | 1e-5 / 300 |
| `NSTOP`/`NK`/`NM` | eigensolve sought / Krylov sizes | sweep; Nm ~ 1.6 Nstop |
| `NEV_LIST` | deflation sweep points | -> break-even knee |

## ROUTINE (ordered), per ensemble
1. **$\lambda_\text{min}$ + $\lambda_\text{max}$** : `DO_LMIN=1 DO_DEFL=0` (inverse
   power iteration, ~8 steps) + power method. Gives $\lambda_\text{min}$,
   $\lambda_\text{max}$, $\kappa=\lambda_\text{max}/\lambda_\text{min}$ and the IR
   scale. (m=0.01: $\lambda_\text{min}\approx5\times10^{-4}$, $\kappa\approx1.6\times10^5$.)
2. **Eigensolve choice** : `EIG_METHOD=2` (shift-invert) is the robust default and
   needs no window. BUT it is EXPENSIVE: shift-invert makes every Lanczos matvec a
   full inner CG of $\hat H$ to `INV_TOL` -- measured **960 inner solves** (~450 iters
   each) to build 150 modes at m=0.05, i.e. the eigensolve is ~**50% of the per-config
   wall** even amortized over the 1536 dilution solves. So PREFER Chebyshev IRL
   (`EIG_METHOD=1`, cheap polynomial matvecs, NO inner solve) WHENEVER it converges.
   Chebyshev only failed at m=0.01 because the spectrum is compressed AND the window
   was wrong -- it is worth RE-TRYING per ensemble with a window set from the freshly
   measured spectrum (recipe below). The eigensolve only fills the IRL **PolyOp slot**;
   switching method swaps `InverseHermOp` (shift-invert) <-> `FunctionHermOp(Chebyshev)`
   in that one slot, everything else (IRL, the `PlainHermOp(H)` convergence test,
   eval/evec) identical.

   **Chebyshev window recipe (lower edge = the CUT, NOT $\lambda_\text{min}$).**
   Grid `Chebyshev(lo, hi, N)` amplifies eigenvalues BELOW `lo` and suppresses the
   bulk `[lo, hi]`. So:
   $$\text{hi}=1.1\,\lambda_\text{max},\qquad \text{lo}=\lambda_{N_\text{stop}}
     \ \ (\text{the cut at the highest wanted mode}),\qquad N\sim\sqrt{\text{hi}/\text{lo}}.$$
   Setting $\text{lo}=\lambda_\text{min}$ is WRONG (amplifies nothing). $N$ by rule of
   thumb is fine; raise it until modes converge. $\lambda_{N_\text{stop}}$ comes from a
   spectrum (a one-time shift-invert calibration run, OR a coarse Cheby run).
   **Calibration is ONE-TIME PER ENSEMBLE**: $\lambda_\text{max}$ and
   $\lambda_{N_\text{stop}}$ are gauge-stable (the cut is steadier than individual
   modes), so fix the window once and run per-config Cheby-IRL with NO per-config
   probe (optional few-second power-method $\lambda_\text{max}$ refresh). That one
   calibration run also DOUBLES as the Cheby go/no-go -- if Cheby will not converge
   even here, keep shift-invert and lean on `ImplicitlyRestartedBlockLanczos` (split
   grids, batched matvecs) for the speedup instead.
3. **Eigensolve stability** : `INV_TOL=1e-5`. Start NSTOP modest (~100), confirm
   `#modes converged: Nconv/Nstop` and a sane increasing eval list. If breakdown
   (NaN $\beta_k$, 0 converged) at large NSTOP -> tighten `INV_TOL` (1e-5->1e-6) or
   lower NSTOP. NaN guard in `InverseHermOp` prevents the 50000-iter spin.
4. **Solver baseline** : `DO_SOLVERCMP=1` at `INNER_TOL=1e-4` -> double vs mixed vs
   mixed-batched at Nev=0. Confirms the mixed-prec gain (~1.4x here) and that
   batching alone is ~neutral at Nev=0.
5. **Deflation sweep + break-even** : `DO_DEFL=1`, `NEV_LIST=0,...` -> per-config
   $T(N_{ev})=\text{setup}(N_{ev})+96\,t_\text{solve}(N_{ev})$; pick the knee. Add
   `DO_BATCHCMP=1` once (batching DOES help with deflation: 1.65x here). At
   `INNER_TOL=1e-4`, m=0.01 gave Nev=150 -> ~5x over production double, still
   climbing -> search 200/250.
6. **Record** per-ensemble: $\lambda_\text{min},\lambda_\text{max}$, eval density,
   chosen $N_{ev}$, setup time, speedup -> in `disc_mrhs_defl_bench_results_claude.md`.

## Key lessons (why the routine is shaped this way)
- **Shift-invert is robust but COSTLY -- prefer Chebyshev when it converges**: each
  shift-invert matvec is a full inner CG (~960 inner solves / 150 modes -> ~50% of the
  per-config wall). Chebyshev's matvec is a cheap polynomial (no solve). Cheby only
  failed at m=0.01 (compressed spectrum + wrong window, lo set like $\lambda_\text{min}$
  instead of the cut $\lambda_{N_\text{stop}}$); retry per ensemble with the window
  recipe in step 2. LMA wants MORE modes (variance) -> the eigensolve gets heavier ->
  a cheap eigensolve is a PREREQUISITE, not a nicety. Fallback if Cheby still fails:
  `ImplicitlyRestartedBlockLanczos` (batched matvecs, split grids).
- **`INNER_TOL` is the hidden lever**: at 1e-8 the mixed solver runs 3 full outer
  iters and deflation barely helps (Nev=100 was 106s); at 1e-4 the deflation guess
  pays off (Nev=100 -> 33s). Always loosen the INNER tol; the OUTER tol stays 1e-8.
- **`MAXPATCH`** must be generous (1000); the default 50 aborts mid-sweep.
- Per-config eigenvectors are the expensive asset -> save + reuse for deflation AND
  LMA (see `disc_lma_impl_plan_claude.md` production directory memo).

## For LMA (variance) the routine extends with one more axis
After the speed tuning above, the LMA step adds a VARIANCE-driven $N_{ev}$ scan
(chunk C of the LMA plan): measure the disc-correlator variance vs $N_{ev}$ -- the
variance-optimal $N_{ev}$ is generally LARGER than the speed knee.
