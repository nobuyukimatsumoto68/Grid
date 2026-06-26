# Chebyshev-filtered low-mode eigensolve -- v2 bench impl plan

## STATUS: WORKING (2026-06-25)
Cheby IRL converges end-to-end on the HOT 4^4 testbed: auto order 83, amplification 1e4, 12/12
converged (reldiff 1.4e-8), single Cheby -> double Rayleigh-Ritz refine, reconstruction completes,
no crash. Working env: `EIG_PREC=1 EIG_METHOD=1 RR_REFINE=1 NSTOP=12 NK=24 NM=120 CHEB_GAIN=1e4
CHEB_LO_FAC=1.02 ERESID=1e-4`. The fixes (even degree, auto-order, single+RR, GATE gating, no
store-free) are documented in the sections below. Next: chunk-D production binary
(shift-invert per ensemble, Cheby+RR per config).

## Goal
Get the **Chebyshev-filtered IRL** eigensolve to CONVERGE for the low modes of the
EO-Schur squared Mobius operator $\hat H = M_{pc}^\dagger M_{pc}$ (eigenvalues
$\lambda_i=\sigma_i^2$). It has never converged in past attempts (see
`disc_lma_HANDOFF_claude.md` Sec. 5). Develop and tune it on a SMALL,
LOW-MEMORY testbed -- $8^4$ FREE theory (cold/unit gauge) -- then carry the tuned
routine to a larger lattice / real config on a cluster.

Work happens entirely in the COPY `disc_lma_bench_v2_claude.cc` (never edit
`disc_lma_bench_claude.cc`, the chunk-A reference).

## Sources (cite in code + here)
- N. Matsumoto, "Large-scale eigenvalue problem" (`lanczos.pdf`, this repo) -- the
  governing note. Chebyshev map (Sec. 3.2): with
  $$q(\lambda;\alpha,\beta)=\frac{2\lambda^2-(\alpha^2+\beta^2)}{\alpha^2-\beta^2},$$
  monotonic in $\lambda>0$, mapping $[\beta,\alpha]\to[-1,1]$. To filter the $m$
  smallest, pick $|\lambda_\text{cut}|\lesssim\beta$ and $|\lambda_\text{max}|\le\alpha$;
  then $p(\lambda)=T_n(q)$ amplifies $\lambda<\beta$. KEY note remark (Sec. 3.1): use
  Chebyshev only "when the rough landscape of the entire spectrum is known in
  advance" -- otherwise the robust choice is the CG/shift-invert inverse iteration.
- Saad, "Numerical Methods for Large Eigenvalue Problems" (the note's main ref).
- Grid `ImplicitlyRestartedLanczos.h` (PolyOp = filter, HermOp = convergence test).

## Why it stalls (diagnosis, now understood)
Grid's `Chebyshev(lo,hi,order)` is the pure polynomial $T_{order-1}(y)$ with the
LINEAR map $y=(\lambda-\tfrac{hi+lo}{2})/\tfrac{hi-lo}{2}$ (since $\hat H$ already has
$\lambda=\sigma^2$, linear-in-$\lambda$ here == the note's $\lambda^2$ map in $\sigma$).
For low modes set $lo=\lambda_\text{cut}$ (just above the highest WANTED $\lambda_{Nstop}$)
and $hi\ge\lambda_\text{max}$ (to keep the whole unwanted bulk inside $[-1,1]$, since
modes ABOVE $hi$ also blow up). The wanted band $[\lambda_\text{min},lo]$ then maps to
$y\in[-\tfrac{hi+lo}{hi-lo},-1]$, an interval of width
$$\Delta y=\frac{2\,lo}{hi-lo}\approx\frac{2\,lo}{hi}\quad(hi\gg lo).$$
At the real disc mass $m=0.01$, $\kappa=\lambda_\text{max}/\lambda_\text{min}\sim10^5$,
so $hi\gg lo$ and $\Delta y\to0$: ALL low modes collapse onto $y\approx-1$ with nearly
equal filter value -> no SEPARATION between them -> Lanczos cannot resolve the cluster
(the handoff's "one tight cluster at filtered-value ~34.8, 0/100 converged").
Amplification $\ne$ separation. Pushing order to ~340 to steepen near $y=-1$ costs as
much as a shift-invert inner solve -> no win. **Conclusion: Cheby is intrinsically bad
for the ill-conditioned ($\kappa\sim10^5$) light-mass case; it is VIABLE only where the
spectrum is well-conditioned / gapped.** FREE theory at $8^4$ is exactly that regime
($\kappa\sim O(10\text{-}100)$, no near-zero modes) -> the right place to get the Cheby
mechanics + tuning ROUTINE working, then reuse on gapped/heavier ensembles.

## Files
- `disc_lma_bench_v2_claude.cc` (copy of the chunk-A bench; edited here).
- `Make.inc` -- register `disc_lma_bench_v2_claude` (and, missing on this machine,
  `disc_lma_bench_claude`).
- this plan.

## Chunks
### V2-1 -- free-theory + small-lattice testbed (run locally, low memory)
Files: `disc_lma_bench_v2_claude.cc`, `Make.inc`
- Add `--free` (CLI) -> `SU<Nc>::ColdConfiguration(Umu)` (unit gauge), skip the NERSC
  read, so the bench runs on `--grid 8.8.8.8` with no config and tiny memory.
- Everything downstream (eigensolve, A2A gates, $L^\text{low}$) already works on any
  gauge field -- no other change needed for a local run.

### V2-2 -- spectral-landscape probe (the note's prerequisite for Cheby)
Files: `disc_lma_bench_v2_claude.cc`
- $\lambda_\text{max}$: `PowerMethod` on $\hat H$ (already in the Cheby branch) -> set
  $hi = \texttt{CHEB\_HI\_FAC}\cdot\lambda_\text{max}$.
- $\lambda_\text{cut}$ (= $lo$): on the small testbed, get the TRUE low spectrum from a
  one-off shift-invert reference eigensolve (EIG_METHOD=2 is cheap at $8^4$), then set
  $lo$ from $\lambda_{Nstop}$ with a margin. Knob `CHEB_LO_AUTO` (default on for the
  bench): `lo = CHEB_LO_FAC * lambda_ref[Nstop-1]`. Manual override stays via `CHEB_LO`.
  (On a cluster where the reference is too dear, the user supplies `CHEB_LO` from a
  prior run -- the "landscape known in advance" workflow.)

### V2-3 -- filter-shape diagnostic (SEE amplification vs separation)
Files: `disc_lma_bench_v2_claude.cc`
- After building the `Chebyshev` object, print $p(\lambda)=$ `Cheby.approx(lambda)` at:
  (i) a log-spaced grid over $[\lambda_\text{min,ref}, hi]$, and (ii) each reference
  eigenvalue $\lambda_i$. Report the AMPLIFICATION RATIO
  $|p(\lambda_\text{min})| / |p(lo)|$ and the SEPARATION across the wanted band
  $|p(\lambda_i)-p(\lambda_{i+1})|$. This makes the stall visible (all $p(\lambda_i)$
  equal -> compressed) and gives a direct tuning target (want the wanted band spread
  over many orders of magnitude, bulk $|p|\le1$).

### V2-4 -- converge + validate on $8^4$ free
Files: `disc_lma_bench_v2_claude.cc`
- Tune `(CHEB_LO, CHEB_HI_FAC, CHEB_ORD, NK, NM)` until Cheby `Nconv == Nstop` with
  GATE1 eigres at the eresid target.
- VALIDATION: Cheby eigenvalues match the shift-invert reference to eresid; GATE2 lift
  residual stays at machine precision (unchanged from chunk A). Record the converged
  parameter set + the filter diagnostic in a short results note.

## Testbed gauge LESSON (2026-06-25): FREE is pathological -> use HOT
FREE/cold gauge is EXACTLY degenerate: the reference spectrum at $8^4$, $m=0.01$ has
$\lambda_{min}\approx0.78$ with **multiplicity ~24** (lattice-momentum shells). Single-vector
Lanczos can extract at most ONE eigenvector per distinct eigenvalue, so seeking
$N_{stop}=20$ modes that are all the SAME eigenvalue never converges (`0/20`), and after
~90 restarts the residual collapses ($\beta\to0$, divide-by-zero) -> `(nan,nan)` Lanczos
steps. The shift-invert reference only limps to 46/50 for the same reason. This is intrinsic
to free theory; a real config has no exact degeneracy. CONFLICT: free is well-conditioned
(Cheby-friendly) but exactly degenerate (Lanczos-fatal).
**Resolution: HOT (random SU(N)) gauge** (`--hot`, `SU<Nc>::HotConfiguration`) -- lifts the
degeneracy like a real config, still small/low-memory. A FIXED RNG seed `{1,2,3,4}` is used
in BOTH the eigref pre-calc and the bench so they build the IDENTICAL config (reproducible
per `--grid`/`--mpi`); their spectra must agree for the auto window + validation.

## KEY BUG (2026-06-25): Chebyshev degree must be EVEN
Grid's IRL keeps the LARGEST filtered eigenvalues (`partial_sort(...,std::greater)`,
ImplicitlyRestartedLanczos.h:319). For LOW modes the filter must therefore map them to the
most POSITIVE value. `Chebyshev(lo,hi,order)` builds $T_{order-1}$; for $y<-1$ (the wanted
band), $T_n(y)=(-1)^n T_n(|y|)$, so:
- EVEN degree ($order$ odd): wanted modes -> $+$large -> IRL keeps them (CORRECT).
- ODD degree ($order$ even): wanted modes -> $-$large -> IRL keeps BULK modes near $+1$
  instead (spurious eval, e.g. ~32 in a spectrum whose true low modes are ~1.2; never
  converges -> restarts forever -> nan, even in double).
This is why every Grid example uses even degree (order 101->$T_{100}$, 4001->$T_{4000}$).
The bench now FORCES `CHEB_ORD` odd (`cheb_o = raw|1`) and the diagnostic warns if
$p(\lambda_{min})<0$. This single bug produced both the spurious eval=32 AND the recurring
nan across the order-80/250 runs.

## Reconstruction crash + single+RR (2026-06-25)
After Cheby CONVERGED (12/12, reldiff 5.9e-8, GATE1/2 ~1e-14), the bench SEGFAULTed in the
reconstruction loop -- a GPU fault (cudaMemcpy <- MemoryManager::Clone <- AcceleratorViewOpen
<- LatticeFermion::operator=), NOT HDF5. Crash mode WANDERED (5 -> 13 with launch-blocking) =>
accumulating device-memory pressure, not a fixed-mode logic bug. Two causes fixed:
1. The GATE1/2/3 validation ran for EVERY mode but is only printed for i<Ncheck; GATE2 alone
   allocates ~5 full-5D double fields + 2 full operator applies per mode -> churn. Now GATES
   run ONLY for i<Ncheck. (L^low, the deliverable, still runs for all modes.)
2. The full Nm-mode eigensolve store (evec/evec_d) stayed resident through reconstruction.
   Now we extract only the Nuse-mode subspace and `clear()+shrink_to_fit()` the big store.
SINGLE-prec Cheby + DOUBLE Rayleigh-Ritz (user's idea, RR_REFINE=1): a cheap single Cheby
(relaxed ERESID so it converges the SUBSPACE without chasing past the single floor) supplies
the subspace; `RayleighRitzRefine` re-orthonormalizes in double, forms A_jk=<V_j,H V_k>,
diagonalizes (Eigen SelfAdjoint), and rotates -> double-accurate eigenpairs. No high-order
double eigensolve, half the eigensolve memory. So we do NOT need order-251 in DOUBLE.

## Precision (2026-06-25)
High-order Chebyshev needs DOUBLE: $T_{249}$ in single amplifies rounding (dynamic range
~$3\times10^5$), so the Ritz residual floors ~$10^{-4}$, above the IRL target
$\|r\|/\lambda_{max}<$ eresid (eresid$^2=10^{-12}$ normalized) -> restart breakdown. The
bench now has a DOUBLE Cheby path (`EIG_PREC=2 EIG_METHOD=1`); window+diagnostic factored
into `ComputeChebWindow` shared by both precisions.

## Decisions (resolved 2026-06-25)
1. Local testbed = HOT (random) gauge, `--hot` -> `SU<Nc>::HotConfiguration` (seed {1,2,3,4}).
   `--free` (cold) kept available but is exactly degenerate -> not usable for Lanczos here.
2. Cheby cut $lo$ = AUTO from a reference spectrum, but the reference is computed in a
   SEPARATE pre-calc file `disc_lma_eigref_v2_claude.cc` (shift-invert IRL + PowerMethod)
   and written to disk; the Cheby bench READS it. `lo = CHEB_LO_FAC*lambda_ref[Nstop-1]`,
   `hi = CHEB_HI_FAC*lambda_max`. Manual `CHEB_LO` override via `CHEB_LO_AUTO=0`.
3. The reference landscape is stored in **HDF5** (`eigref_*.h5`): datasets `lambda`,
   `sigma` + scalars `lambda_max`, `Nconv`, and `grid/Ls/mass/M5/b/c/free` metadata.
   Grid `Hdf5Writer`/`Hdf5Reader`; h5py-readable for offline landscape plots.

## Files delivered
- `disc_lma_eigref_v2_claude.cc` -- the pre-calc (robust shift-invert reference -> .h5).
- `disc_lma_bench_v2_claude.cc` -- the Cheby bench (reads .h5, auto window, filter
  diagnostic, Cheby-vs-reference eval validation; inherits chunk-A gates + L^low).
- `Make.inc` -- registers both + the previously-missing `disc_lma_bench_claude`.
- `tmp_claude.sh` (repo root) -- build + 8^4-free validation handoff (user runs).
