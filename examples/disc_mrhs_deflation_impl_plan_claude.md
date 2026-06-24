# disc speedup #2+#3 -- multi-RHS + exact low-mode deflation (no multigrid)

## STATUS (2026-06-23): BENCH RUNS END-TO-END; measured spectrum recorded
- Bench `disc_mrhs_defl_bench_claude.cc` BUILT + runs at 8-node/32-rank (mpi
  2.2.2.4). Results in `disc_mrhs_defl_bench_results_claude.md`.
- Eigensolve = SHIFT-INVERT Lanczos (EIG_METHOD=2): 100/100 modes, 643s (Cheby
  IRL failed 0/24). Spectrum DENSE: eval[0]=4.83e-4 ... eval[19]=9.05e-4.
- Nev=0 baseline 163.7s/16-RHS; Nev=25 -> 129.9s (1.26x). Nev=50/100 pending
  (was MAXPATCH=50 abort -> fixed 1000; bench now prints all Nconv -> rebuild).
- Decisions locked: batch width 16; primary m=0.01 @ b10.8; NO multigrid.
- Speed-Nev SATURATES (dense spectrum) -> pick the MODEST knee from
  T(Nev)=setup+96*solve(Nev). Bigger Nev is a FUTURE/variance lever (see end).
- Ground rules: Claude NEVER submits jobs / NEVER rm / NEVER compiles (user does).


Combines the original chunks #2 (multi-RHS / batched solve) and #3 (deflation)
into ONE track, on the FINE operator only. Multigrid (HDCG /
GeneralCoarsenedMatrix / Aggregation / coarse Lanczos) is explicitly NOT used --
decided 2026-06-23. Builds on chunk #1 (mixed precision), file
`disc_mixedprec_impl_plan_claude.md`.

## Algorithm sources (mandatory citation)

- Exact eigenvector deflation across multiple right-hand sides:
  A. Stathopoulos, K. Orginos, "Computing and deflating eigenvalues while solving
  multiple right-hand side linear systems ... QCD," SIAM J. Sci. Comput. 32
  (2010) 439, arXiv:0707.0131.
- Low-mode deflation context: M. Luscher, "Local coherence and deflation of the
  low quark modes in lattice QCD," JHEP 0707 (2007) 081, arXiv:0706.2298.
- Mixed-precision reliable-update CG: M. A. Clark et al., arXiv:0911.3191.
- Grid classes: `ImplicitlyRestartedBlockLanczos`,
  `algorithms/deflation/MultiRHSDeflation.h`,
  `algorithms/iterative/ConjugateGradientMixedPrecBatched.h`. Implementing code
  must cite arXiv:0707.0131 (deflation) and arXiv:0911.3191 (mixed prec).

## Goal

Cut the dominant cost (1536 fine Mobius solves/config -- see chunk #1 plan) two
ways at once on the LIGHT ensemble m=0.01 (b10.8), where CG critically slows:
1. BATCH the 16 spin-color solves of one (t,eo) as a multi-RHS solve (amortizes
   gauge-field memory traffic on the APU).
2. DEFLATE the low modes (exact, fine grid). Eigenvectors depend only on the
   gauge field, so they are built ONCE per config and reused across all
   Nt*2 = 96 batched solves (1536 single solves).

Constraints carried forward: outer residual stays 1e-8 (no #4); even/odd Schur
preconditioning unchanged; source/dilution/contraction/I-O/wall-blocker
unchanged; NO multigrid.

## Decisions (2026-06-23)

- Batch width = 16 (spin-color of one (t,eo)). Wider blocks across (t,eo) are a
  later experiment, not now.
- Primary (and first) target ensemble: m=0.01 @ b10.8. Heavier ensembles likely
  keep chunk-#1 (mixed prec, no deflation) -- revisit after benchmarks.
- BENCHMARK FIRST on one config before any production wiring (user runs on a GPU
  node). Nev and the Lanczos iteration/Chebyshev window stay measured knobs.

## Building blocks (verified present in this Grid)

- `MixedPrecisionConjugateGradientBatched<FieldD,FieldF>` -- batched
  `operator()(const std::vector<FieldD>& src, std::vector<FieldD>& sol)`, single
  inner / double reliable-update + patch-up; `useGuesser(LinearFunction<FieldF>&)`
  is invoked on the WHOLE single-prec batch (header line ~168).
- `MultiRHSDeflation<Field>` -- standalone (NOT a LinearFunction):
  `ImportEigenBasis(evec,eval)` and
  `DeflateSources(std::vector<Field>& src, std::vector<Field>& guess)` (batched
  nev.nrhs BLAS GEMM).
- `ImplicitlyRestartedBlockLanczos<Field>` -- block (multi-RHS) Lanczos for the
  fine eigenbasis.

## The one adaptor we must write

`MultiRHSDeflation` is not a `LinearFunction`, so to use it as the batched-CG
guesser we add a thin wrapper (file-scope, no lambda):

    template<class FieldF>
    class MrhsDeflationGuesser : public LinearFunction<FieldF> {
      MultiRHSDeflation<FieldF> &defl;
    public:
      using LinearFunction<FieldF>::operator();
      MrhsDeflationGuesser(MultiRHSDeflation<FieldF> &d) : defl(d) {}
      void operator()(const std::vector<FieldF> &in, std::vector<FieldF> &out){
        // out = deflated guess for the whole batch (arXiv:0707.0131)
        defl.DeflateSources(const_cast<std::vector<FieldF>&>(in), out);
      }
      void operator()(const FieldF &, FieldF &){ GRID_ASSERT(0); } // batch-only
    };

(Confirm DeflateSources' source arg constness when wiring; adjust signature if it
takes a non-const ref.)

## Files

- NEW benchmark binary `disc_mrhs_defl_bench_claude.cc` (examples/), + `Make.inc`
  target + automake/config.status regen (same mechanism as chunk #1).
- Later: production variant `disc_multipleGamma_binary_mrhs_defl_claude.cc`
  (only after benchmarks justify it). Original + mixedprec binaries untouched.
- Driver/scripts: none until production.

## Ordered chunks

### Chunk A -- benchmark harness (ONE config, m=0.01)   [GPU-node session]
Files: `disc_mrhs_defl_bench_claude.cc`, `Make.inc`
Single config of conf_nc4nf1_2448_b10p800_m0p0100. Build the double + single
SchurDiagMooee ops (chunk #1 style). Then MEASURE and print:
  (a) baseline: chunk-#1 mixed-prec CG, time for one 16-solve group (no defl);
  (b) block Lanczos setup time vs Nev (e.g. Nev = 0,48,96,192,384), report
      lowest eigenvalues + Chebyshev window used;
  (c) batched mixed-prec CG on the same 16 RHS WITH the deflation guesser, for
      each Nev: iteration count + wall time.
Output a tiny table so we can solve the break-even:
  setup(Nev) + 96 * solve_defl(Nev)  <  96 * solve_mixed(0)   [per config]
No traces written; correctness is checked separately (Chunk C).

### Chunk B -- pick Nev / params from the numbers
Files: (this md)
Record the chosen Nev, Lanczos params, and whether deflation wins on m=0.01.
Decide go/no-go for production and whether heavier ensembles get it.

### Chunk C -- correctness check
Files: (run only)
On one config, compare the 10 `traces.<gam>.<conf>` from the deflated-batched
path against the double-precision reference outputs: agree to ~1e-7. The 16-RHS
batched solution must equal the 16 single solves (deflation/guess only changes
the starting point, not the converged result at 1e-8).

### Chunk D -- production variant (only if Chunk B says go)
Files: `disc_multipleGamma_binary_mrhs_defl_claude.cc`, `Make.inc`
Per config: read gauge, precisionChange, build actions, run block Lanczos ->
ImportEigenBasis; then the (t,eo) loop calls the batched-16 deflated solve in
place of the current per-spin-color `Solve`. Reuse the wall-time blocker, but its
per-config estimate now includes the Lanczos setup -- bump DISC_TPT bootstrap
accordingly.

## Open questions (for the GPU-node benchmark session)

1. Nev sweep range -- start {48,96,192,384}? finer near break-even.
2. Block Lanczos Chebyshev window / Nstop for m=0.01 at 24^3x48 (the M5=1.5
   Mobius spectrum; reuse the |H_W|min calibration from the flow-scan work if
   relevant). Needs the measured low spectrum from Chunk A(b).
3. Memory: single-prec fine 5D rb evec ~0.68 GB each; Nev=384 ~260 GB,
   distributed over the 8-node (32 GCD) disc job -> fine, but confirm at runtime.
4. Does `DeflateSources` take source by non-const ref (adjust the adaptor)?

## FUTURE DIRECTION -- variance-driven Nev / low-mode averaging (not now)
The benchmark optimizes Nev for SPEED, where the gain saturates at a modest knee
(dense low spectrum: eval[0..19] = [4.83, 9.05]e-4). The future escalation grows
Nev for a DIFFERENT reason -- VARIANCE of the disconnected estimator -- using the
SAME eigenbasis we already compute:
- For the disc loop $\mathrm{Tr}[\Gamma M^{-1}]$ the low modes DOMINATE the
  stochastic-estimator variance. **Low-mode averaging (LMA / all-mode averaging,
  Blum-Izubuchi-Shintani)**: compute the low-mode part of the trace EXACTLY from the
  eigenvectors, stochastically estimate only the high-mode remainder. There more
  modes buy SIGNAL, not wall-clock -- justifying $N_\text{ev}$ of hundreds on
  physics grounds once the eigensolve is being paid for anyway.
- The exact low-mode contraction and the deflation guesser SHARE the eigenbasis, so
  LMA reuses this exact shift-invert + `MultiRHSDeflation` machinery. Keep the
  bench's `NSTOP`/`NEV_LIST` open knobs and the eigensolve scalable (it already is).
- Regime change also pushes Nev up later: lighter quark mass or larger volume ->
  higher near-zero density + lower spectrum (Banks-Casher) -> more low modes
  relevant for both speed and variance.
So: NOW pick the modest speed-knee Nev (and fix chunk #1's InnerTolerance); LATER,
if the disc signal is low-mode-variance-limited, push Nev up for LMA.
