# disc LMA + eigensolve -- HANDOFF (2026-06-25)

## REPLY (2026-06-25, local session) -- CHEBYSHEV EIGENSOLVE NOW CONVERGES
Picked this up locally and resolved the long-standing "Chebyshev never converges" item (Sec 5).
The Cheby IRL now converges cleanly; full state in machine memory
`project_disc_lma_cheby_v2_state.md` and design doc `examples/disc_lma_cheby_v2_impl_plan_claude.md`.
New files (never edited the originals): `examples/disc_lma_bench_v2_claude.cc` (Cheby bench),
`examples/disc_lma_eigref_v2_claude.cc` (shift-invert pre-calc -> eigref HDF5), repo-root
`tmp_claude.sh` (build+run). Tested on a SMALL local testbed: **HOT** (random) gauge, 4^4, Ls16,
m=0.01 (free/cold is unusable -- see below). What was wrong, in order found:
1. getopt: leading `+` + ParseArgs-before-Grid_init silently dropped CLI flags. (Grid_init first; drop `+`.)
2. FREE/cold gauge is EXACTLY degenerate (lambda_min mult ~24) -> single-vector Lanczos breaks
   down to nan. Use HOT (random) gauge to lift it (like a real config).
3. **Chebyshev DEGREE must be EVEN.** Grid `Chebyshev(lo,hi,order)` = T_{order-1}; IRL keeps the
   LARGEST filtered evals (partial_sort greater). T_odd(y<-1) -> -large, so IRL locked onto BULK
   modes near +1 (spurious eval ~32, nan). Force ODD order (=even degree). THIS was the core bug.
4. High-order single Cheby hits the single-prec residual floor (eresid^2=1e-12 normalized
   unreachable) -> single Cheby supplies the SUBSPACE only, then **double Rayleigh-Ritz refine**.
5. Order MUST be auto-derived from a target gain (CHEB_GAIN~1e4): a fixed order over-amplified to
   ~7e12 on 4^4 -> single-prec Lanczos fluctuated. degree ~ acosh(gain)/acosh|y_min|.
6. Reconstruction GPU crash: ran GATE2 (5 full-5D fields/mode) for every mode though only printed
   for i<Ncheck -> gate the GATES behind i<Ncheck; and don't clear()/shrink_to_fit() the evec store
   (tripped Grid AccCache.bytes==bytes) -- emplace_back/swap instead.
Working recipe (bench env): `EIG_PREC=1 EIG_METHOD=1 RR_REFINE=1 NSTOP=12 NK=24 NM=120 CHEB_GAIN=1e4
CHEB_LO_FAC=1.02 ERESID=1e-4`. Result: auto order 83, amplification 1e4, 12/12 converged
(reldiff 1.4e-8), RR refine, reconstruction completes (GATE1 ~2e-5), no crash.
VERDICT: Cheby IS viable with the above, but for DENSE/clustered low spectra (the real m=0.01 SDM
case) it is fragile vs shift-invert. So: **shift-invert = per-ENSEMBLE eigensolve; Cheby+RR = cheap
per-CONFIG path** (read the ensemble eigref + mild margins).

UPDATE (2026-06-26): chunk B (source-projection) DONE + validated. Refactored shared machinery into
`examples/disc_lma_v2_common_claude.h`; built `examples/disc_lma_estimator_bench_v2_claude.cc` (exact
$L^\text{low}$ + plain full-solve loop + PROJECTED high solve + NSRC variance). DET GATE
$\|S_\text{full}-(S_\text{high}+S_\text{low})\|/\|S_\text{full}\|=1.6e-8$ => the projection LMA is
UNBIASED to solver tol for ANY modes (the $u_i=M_{pc}v_i/\sigma_i$ construction makes the split cancel
term-by-term; eigensolve is MIXED-prec single-Lanczos->double-RR/A2A/solve; RR is redundant for
correctness, only buys variance). 4^4 HOT variance ratio>1 (EXPECTED: gapped, no near-zero modes --
LMA needs the light/near-zero regime).

CHUNK D DONE (2026-06-26): `disc_multipleGamma_binary_lma_claude.cc` written + COMPILES (never edit the
original disc binary). Config loop + self-skip + wall-blocker + per-config mixed-prec Cheby+RR (reading
the per-ensemble eigref) + EVEC CHECKPOINT (`evec.<conf>.scidac`/`eval.<conf>.h5`, reload to skip the
eigensolve on rerun) + LMA loop (exact $L^\text{low}$ + `SolvePropProjected` high) -> `traces.<gam>.<conf>`
Scidac (drop-in; pipeline unchanged). RNG4 = `SeedUniqueString(config_path)`. NOT YET RUN in production --
needs a real config dir + a per-ensemble eigref (RUN recipe in `disc_lma_production_impl_plan_claude.md`:
eigref once on a real config (no --hot), then the binary). PENDING physics: run on a real near-zero-mode
ensemble to measure the actual variance ratio (<1 expected there). Full state: machine memory
`project_disc_lma_cheby_v2_state.md`.

### RELEVANT FILES for the remote agent (all in `examples/` of this fork, all `_claude`)
- **`disc_lma_v2_common_claude.h`** -- READ FIRST. Shared machinery (header-only, inline): `BuildLowModes`
  (eigensolve Cheby/shift-invert, single/double, + RR -> refined subspace), `BuildA2ASet` (-> per-mode
  `a_i,b_i,u_i,sigma_i`), `BuildPhysicalA2A` (the v/w lift), `ComputeChebWindow` (auto window + auto order
  from `CHEB_GAIN`), `RayleighRitzRefine`, `SetupGauge`, `ReadEigref`, `LMAEigParams`/`ReadLMAEigParams`.
- **`disc_lma_eigref_v2_claude.cc`** -- per-ENSEMBLE shift-invert eigref pre-calc -> `eigref_<grid>_m<mass>.h5`
  (HDF5 landscape: `lambda`, `sigma`, `lambda_max`). Run ONCE per ensemble.
- **`disc_lma_bench_v2_claude.cc`** -- eigensolve VALIDATION bench (thin; `#include`s the header): GATES 1-3
  + exact `L^low`. Use to sanity-check the eigenbasis on a new ensemble.
- **`disc_lma_estimator_bench_v2_claude.cc`** -- the LMA ESTIMATOR validation bench (chunk B/C): plain
  full-solve loop + `SolveHighProjected` + exact `L^low` + the DET GATE + `NSRC` variance loop.
- **`disc_multipleGamma_binary_lma_claude.cc`** -- the PRODUCTION binary (chunk D): config loop +
  self-skip + wall-blocker + per-config Cheby+RR (eigref) + EVEC checkpoint (`SaveEvecs`/`LoadEvecs`) +
  the projected-LMA loop -> `traces.<gam>.<conf>` Scidac (drop-in). Run recipe in
  `disc_lma_production_impl_plan_claude.md`. NEVER edit the original `disc_multipleGamma_binary_claude.cc`.
- `disc_lma_cheby_v2_impl_plan_claude.md` -- eigensolve design + every Grid bug/lesson (degree parity,
  auto-order, AccCache, degeneracy nan, getopt).
- `disc_lma_estimator_bench_impl_plan_claude.md` -- estimator design: projection vs recombination,
  unbiasedness derivation, mode-accuracy/mixed-precision.
- `disc_lma_production_impl_plan_claude.md` -- chunk-D production design + DECISIONS (RNG, output,
  evec checkpoint) + the RUN recipe (eigref once per ensemble, then the production binary).
- Build/run: repo-root **`tmp_claude.sh`** (`REGEN=1` only when Make.inc changes; phases: eigref ->
  eigensolve bench -> estimator bench; 4^4 HOT testbed). `examples/Make.inc` registers the binaries.

---

Pick-up doc for continuing the disconnected-loop **low-mode averaging (LMA)** work and
the **eigensolve** investigation in a local environment. All paths below are in the
`nobuyukimatsumoto68/Grid` fork unless noted; the disc/LMA code lives in `examples/`.
Companion design doc with full math: **`examples/disc_lma_impl_plan_claude.md`** (read
that for the derivations; this file is the status + decisions + how-to-run).

Project memory (richer, machine-local): `~/.claude/projects/-g-g91-matsumoto5/memory/project_disc_speedup.md`.

---

## 1. Goal
Speed up AND variance-reduce the disconnected loop $L_\Gamma(x)=\mathrm{tr}_{sc}[\Gamma\,S(x,x)]$
($S=M^{-1}$ the physical quark propagator) for SU(4) stealth-dark-matter, Mobius DWF,
$L_s=16$, $M_5=1.5$, $b/c=1.5/0.5$, $24^3\times48$, on tuolumne (MI300A, flux). The disc
binary does $N_t(48)\times eo(2)\times sc(16)=1536$ Mobius solves/config.
- **Speed** = mixed-prec + multi-RHS + low-mode deflation (done/benchmarked elsewhere).
- **Variance** = LMA: compute the low-mode part of the loop EXACTLY from eigenvectors,
  stochastic only on the high-mode remainder. THIS doc.

The eigenbasis (low modes of the EO-Schur squared Mobius operator) is reused THREE ways:
deflation (speed) + disc LMA (variance) + conn LMA (variance). Computing it once and
reusing is the key amortization.

---

## 2. File map (all `_claude`, in `examples/` unless noted)
- `disc_lma_bench_claude.cc` -- CHUNK A bench (correctness). Eigensolves, reconstructs the
  physical 4D A2A pair $(a_i,b_i)$, runs 3 gates, prints the exact low loop. Has env knobs
  `EIG_METHOD` (1 Cheby / 2 shift-invert), `EIG_PREC` (1 single / 2 double), `CHEB_LO/ORD/HI_FAC`,
  `NSTOP/NK/NM/NEV/NCHECK/INV_TOL/ERESID/MAXITER`. Helper `BuildPhysicalA2A` does the v/w lift.
- `disc_lma_impl_plan_claude.md` -- the design doc: KEY DERIVATION (v/w), all 4 chunks
  (A reconstruction, B source projection, C variance validation, D production), conn LMA.
- `disc_mrhs_defl_bench_claude.cc` -- the deflation/eigensolve benchmark (shift-invert IRL,
  Nev sweep). Shares the eigensolve pattern.
- `disc_multipleGamma_binary_claude.cc` -- ORIGINAL disc binary (double CG). Reference; never edit.
- `disc_multipleGamma_binary_defl_claude.cc` -- the ~5x speed stack (eigensolve once + batched
  mixed-prec deflated solve). Production reference; shift-invert hardcoded.
- `disc_multipleGamma_binary_mixedprec_claude.cc` -- mixed-prec variant (no eigensolve).
- `disc_tuning_routine_claude.md` -- per-ensemble eigensolve/solver tuning routine.
- Scripts at repo ROOT (`/usr/workspace/lsd/matsumoto5/su4_32c/`, == workdir):
  `build_disc_speedup_claude.sh` (builds all 4 disc binaries), `submit_disc_lma_bench_claude.sh`
  (flux launcher for the chunk-A bench).
- Make.inc: `examples/Make.inc` registers `disc_lma_bench_claude` (+ the others).

---

## 3. The reconstruction (KEY DERIVATION) -- VALIDATED
Full details in the plan; summary. Eigenpairs are of $\hat H=M_{pc}^\dagger M_{pc}$
(SchurDiagMooee, ODD checkerboard, 5D), $\hat H v_i=\sigma_i^2 v_i$. The disc loop uses
$S=E\,M^{-1}\,I$ with $E=$`ExportPhysicalFermionSolution`, $I=$`ImportPhysicalFermionSource`.
- $u_i=M_{pc}v_i/\sigma_i$ (odd-cb left singular vector).
- Full 5D $\mathcal V_i$: odd $=v_i$, even $=-M_{ee}^{-1}M_{eo}v_i$.
- Full 5D $\mathcal U_i$: odd $=u_i$, even $=-M_{ee}^{-\dagger}M_{oe}^\dagger u_i$.
- Physical A2A: $a_i=E\mathcal V_i$, $b_i=I^\dagger\mathcal U_i=P^\dagger(\texttt{DminusDag}\,\mathcal U_i)$
  ($P^\dagger\chi=P_+\chi[0]+P_-\chi[L_s{-}1]$ -- chiral projectors SWAPPED vs Export).
- $S_\text{low}=\sum_i\sigma_i^{-1}a_ib_i^\dagger$; loop
  $L_\Gamma^\text{low}(x)=\sum_i\sigma_i^{-1}\mathrm{tr}_{sc}[\Gamma\,a_i b_i^\dagger](x)$, NO solves.

**Chunk-A gate run CONFIRMED the lift:** on m=0.01 lat.758, shift-invert, Nconv=100:
- **GATE2 `r1 = ||M V_i - sigma_i u_hat_i|| = 2.5e-14`** -> the $\mathcal V_i$ even fill-in +
  $a_i=E\mathcal V_i$ surface map are CORRECT to machine precision. **Decisive.**
- GATE1 eigres = 0.13 (uniform); GATE2 `r2 = ||Mdag U - sigma v_hat||` = 0.13 (mirrors GATE1).
- GATE3 `||b - g5 a|| / ||b||` ~0.21 (modes 0-3) / 1.78 (mode 4): INCONCLUSIVE while modes
  are rough; this is the only check on the b-side ($b_i$) so far -- settle it with clean modes.

---

## 4. The 0.13 GATE1 -- diagnosis (the live open item)
eval[0]=4.83e-4 .. eval[99]=9.7e-3 = the KNOWN m=0.01 spectrum, so the eigenVALUES are right;
the 0.13 is per-mode eigenvector ROUGHNESS. Two non-exclusive causes, NOT yet disentangled:
1. **Single-prec inner-CG floor.** The shift-invert inner CG logs "Computed 9.8e-6 / True
   2.4e-3" -- single-precision residual drift at $\kappa\sim1.6\times10^5$ caps $\hat H^{-1}$
   accuracy at ~2e-3 -> rough modes. Fine for deflation; for LMA the low part is then ~13%
   accurate -> degrades the VARIANCE gain (central value stays unbiased).
2. **Near-degeneracy / too few Krylov vectors.** The absolute residual is only ~6e-5
   (= 0.13*lambda_0), ~6x the single floor; and the run used `Nk=Nstop=100, Nm=140` (only a
   40-vector buffer; rule of thumb wants $N_m\sim1.5\text{-}2 N_\text{stop}$, $N_k>N_\text{stop}$).
   The lowest modes are near-degenerate (eval[0],eval[1] ~2% apart; gap 9e-6 < residual 6e-5),
   so individual eigenvectors mix within clusters -> large per-mode residual, but $S_\text{low}$
   (the projector-weighted sum, what LMA uses) is ROBUST to intra-cluster mixing. So 0.13
   likely OVERSTATES the LMA-relevant error.

**Cheap NEXT tests to disentangle (both in the bench, env-only or one rebuild):**
- (a) Larger $N_m$, single prec: `NSTOP=100 NK=140 NM=240` (fits pdebug). If GATE1 drops ->
  it was hierarchy. (Configured but not yet run -- superseded by the Cheby detour below.)
- (b) Double inner CG: `EIG_PREC=2` (the bench supports it; SLOW -> pbatch, -t 240m). Removes
  the single floor. If GATE1 drops -> it was precision.
- (c) The decisive LMA metric is NOT the per-mode residual but $S_\text{low}$ itself: compare
  $S_\text{low}\eta$ on a point source, single-prec vs double-prec modes. If ~1% agreement,
  single is fine for LMA and 0.13 is a red herring. (NOT yet implemented in the bench.)
- Production fix if precision matters: **mixed-prec (reliable-update) inner CG** beats the
  single floor cheaply (single storage); or a double Rayleigh-Ritz / double-IRL **refine**
  of the single-prec subspace (converge cheap in single, polish in double).

---

## 5. Eigensolve method -- shift-invert vs Chebyshev (RESOLVED for m=0.01)
**Shift-invert is the right tool for m=0.01.** Investigation:
- Shift-invert IRL: PolyOp $=\hat H^{-1}$ via inner CG. Robust ($1/\lambda$ stretches the dense
  bottom -> good separation) but EXPENSIVE (each matvec = ~450-iter inner CG; ~960 inner solves
  for 150 modes -> the eigensolve is ~50% of the per-config wall).
- Chebyshev IRL: PolyOp $=T_N(\hat H)$, cheap matvec ($N$ $\hat H$-applies, no inner solve).
  TESTED on m=0.01 with a tuned window (`lo=0.02 > eval[99]`, `hi~100`, `ord=150`):
  amplification worked (alpha ~-16, modes lifted above bulk) but **STALLED, 0/100 converged**.
  Per-mode residuals FLAT at ~0.05 across restarts; the filter mapped the near-degenerate
  bottom (eval[0..k] ~2% apart) into ONE tight cluster at filtered-value ~34.8 that Lanczos
  cannot resolve. Amplification != separation.
- Quant: to amplify the bottom strongly enough you'd need `CHEB_ORD`~340, at which point the
  Cheby matvec costs ~the same as a shift-invert inner solve -> no win. The compressed,
  near-degenerate spectrum ($\kappa\sim10^5$, $\lambda_\text{min}\sim5\times10^{-4}$) defeats Cheby.
- CONTRAST -- eye4 (M5-flow/mres, `Grid_sdm_build/src/gauge_gen_Nc4/eye4_anti.cc`): Cheby IRL
  works there because it's a DIFFERENT regime -- 4D Wilson $H_W^2$, only ~10 modes, and at
  M5=1.5 a GAPPED isolated low cluster (|H_W|min~5.9, NO near-zero modes). Their own tuning
  notes even warn: "validated only where there are NO near-zero modes." Our disc operator is
  exactly the near-zero case their caveat flags. So Cheby's prior success does NOT transfer;
  it confirms the diagnosis. (eye4 uses `ChebyParams{alpha,beta,Npoly=101}`, IRL Nstop=10/Nk=15/Nm=50.)

**Bottom line:** for the disc m=0.01 eigensolve, use shift-invert; attack the 0.13 via
mixed-prec/double inner CG or larger $N_m$ (Sec. 4), NOT Cheby. Keep the `EIG_METHOD` knob for
heavier masses / future ensembles where the spectrum may be gapped enough for Cheby.

---

## 6. Chunk B = SOURCE PROJECTION (decided; not yet coded)
The variance-reduction step. The loop split $L_\Gamma=L_\Gamma^\text{low,exact}+L_\Gamma^\text{high,stoch}$.
Per dilution source $\eta$: Import + RedBlackSource -> $\tilde b_o$ (odd cb); project out the low
modes onto the ORTHONORMAL $u_i$ (NO Gram inverse -- the obliquity worry only applies if you
project in 4D physical $a_i/b_i$ space, which you do NOT):
$$\tilde b_o^\perp=\tilde b_o-\sum_i u_i\langle u_i,\tilde b_o\rangle,$$
solve the now WELL-CONDITIONED high system (low modes gone -> light-mass slowness removed),
RedBlackSolution + Export -> $S_\text{high}\eta$; add the exact $L^\text{low}$. KEY identity:
$M_{pc}v_i=\sigma_i u_i\Rightarrow M_{pc}^{-1}u_i=v_i/\sigma_i$, so projecting onto $u_i$ removes
EXACTLY the low part of the inverse -> $S_\text{high}=S-S_\text{low}$, unbiased (uses only
$\langle\eta\eta^\dagger\rangle=1$). Variance AND speed from one eigenbasis. The $u_i$ are
already built in chunk A's `BuildPhysicalA2A`.
- Recombination ($L^\text{low}+L^\text{full,stoch}-L^\text{low,stoch}$, full ill-conditioned
  solve) is the SAME object via a different route -> variance only; keep as a one-time
  validation cross-check (= old estimator + a mean-zero correction).
- New production binary (chunk D): `disc_multipleGamma_binary_lma_claude.cc` (separate file;
  never edit the originals). Conn LMA: `baryons_0000_dirac_lma_claude.cc` + `A2Autils::MesonField`,
  shares the eigenbasis.

---

## 7. Build + run (USER does; ground rules in Sec. 9)
Rebuild (the bench `.cc` changed -- EIG_PREC/EIG_METHOD + memory fix):
```
cd /usr/workspace/lsd/matsumoto5/su4_32c
bash build_disc_speedup_claude.sh        # automake regen + build all 4 disc binaries; tees .log
```
Run the chunk-A bench (8 nodes / 32 ranks, mpi 2.2.2.4 -- 1 node host-OOMs):
```
flux batch submit_disc_lma_bench_claude.sh   # config/knobs at the top of the script
```
Writes NO data; gates + `L^low(t,p=0)` go to the flux output `disc_lma_bench_<id>`.
Memory note: only the chosen-precision evec store is allocated (single ~0.68 GB, double
~1.36 GB each; at NM=240 the unused store would be ~10 GB/rank -> OOM, so it's avoided).

---

## 8. State + immediate next steps
- chunk A: lift CONFIRMED (GATE2 r1=2.5e-14). GATE1 0.13 + GATE3 still open (mode quality).
- Eigensolve: shift-invert is the method for m=0.01; Cheby ruled out (dense near-degenerate).
- chunk B: form decided (source projection); NOT coded.
- **Do next:**
  1. Settle the 0.13: run the larger-$N_m$ single-prec test (Sec 4a), then if needed the
     double inner CG (4b). Decide if single-prec modes suffice for LMA via the $S_\text{low}$
     point-source metric (4c) -- which ALSO settles GATE3 (clean modes -> judge $b=\gamma_5 a$).
  2. Write the chunk-B source-projection bench (Sec 6), validate central value (== plain
     stochastic, $\Delta t\neq0$ unbiased) + VARIANCE drop vs $N_{ev}$.
  3. chunk D production binary + per-ensemble evec+disc directory (save/reload evecs as the
     expensive per-config checkpoint -- see the plan's PRODUCTION DIRECTORY memo).

---

## 9. Ground rules (do NOT break)
- Files get `_claude` before the extension. No Unicode in files (LaTeX: `$...$` in .md, bare
  macros in code comments).
- Outer solve tol = 1e-8 NEVER relaxed.
- Production layout 8 nodes / 32 ranks (mpi 2.2.2.4); 1 node does not fit the MI300A host arena.
- (These applied to the Claude-assisted workflow on the remote machine: Claude never submitted
  jobs, compiled, or wrote rm/destructive commands -- the user did all builds/submits. In a
  local environment adapt as needed.)
