# disc LMA PRODUCTION binary (chunk D) -- impl plan

## STATUS: D1 + D2 written (2026-06-26), pending build
- D1: moved StochasticDilutedSource/Solve/TraceField/SolveHighProjected/SolvePropProjected/
  LowPartOnSource into `disc_lma_v2_common_claude.h` (inline) + added `SaveEvecs`/`LoadEvecs`
  (Scidac evecs on odd-cb, set `Checkerboard()=Odd` on read; evals+Nuse in HDF5). Estimator bench
  refactored to use them (behavior-identical).
- D2: `disc_multipleGamma_binary_lma_claude.cc` -- config loop + per-config Cheby+RR (eigref) +
  evec checkpoint reload/save + LMA loop (exact L^low + SolvePropProjected high) -> `traces.<gam>.<conf>`.
  Registered in Make.inc; build target added to tmp_claude.sh. Build needs `REGEN=1` (new target).

## RUN recipe (production, on a REAL ensemble)
1. per-ENSEMBLE eigref ONCE (shift-invert on one real config, NOT hot):
   `EIG_PREC=1 NSTOP=<N> NK=<..> NM=<..> ./disc_lma_eigref_v2_claude --grid <L> --mpi <..> \
      --config <dir>/<prefix_lat.NNNN> --mass <m> --eigref eigref_<ens>.h5`
2. production over the ensemble:
   `EIG_PREC=1 EIG_METHOD=1 RR_REFINE=1 NSTOP=<N> NK=<..> NM=<..> CHEB_GAIN=1e4 CHEB_LO_FAC=1.02 ERESID=1e-4 \
      ./disc_multipleGamma_binary_lma_claude --grid <L> --mpi <..> --mass <m> --beta <b> \
      --dir <configdir> --obsdir <outdir> --eigref eigref_<ens>.h5`
   Writes `traces.<gam>.<conf>` (drop-in) + `evec.<conf>.scidac`/`eval.<conf>.h5` checkpoints in outdir.
   Self-skips done configs; reloads evecs to skip the eigensolve on rerun; `DISC_DEADLINE_EPOCH` blocker.

## Goal
`disc_multipleGamma_binary_lma_claude.cc` -- the production disconnected-loop binary with LMA,
combining the reference disc binary's config-loop infrastructure with the validated LMA estimator.
Per config: mixed-prec Cheby+RR eigensolve (reading the per-ENSEMBLE eigref) -> A2A set -> the LMA
loop = exact noise-free $L^\text{low}$ + source-PROJECTED stochastic high part, written in the SAME
Scidac format as `disc_multipleGamma_binary_claude.cc` so the downstream pipeline is unchanged.
Validated prototype: `disc_lma_estimator_bench_v2_claude.cc` (chunks 1-4). Shared machinery:
`disc_lma_v2_common_claude.h`. Never edit the original disc binary.

## What it reuses (no duplication)
- From `disc_lma_v2_common_claude.h`: `ReadEigref`, `BuildLowModes` (Cheby+RR), `BuildA2ASet`,
  and (after chunk D1, moved into the header) `StochasticDilutedSource`, `Solve`, `TraceField`,
  `SolveHighProjected`, `SolvePropProjected`, `LowPartOnSource`.
- From the reference `disc_multipleGamma_binary_claude.cc`: config discovery (dir scan for
  `*_lat.NNNN`), the self-skip (output exists), the graceful wall-time blocker (DISC_DEADLINE_EPOCH).

## RNG randomization (user-flagged)
The noise RNG (`RNG4`) must NOT carry the testbed's fixed seed `{11,12,13,14}` (that would give the
SAME noise for every config -> correlated estimates). Per config, seed it so the noise is independent
across configs. Options (decide below). The EIGENSOLVE RNG (Lanczos start, `{5,6,7,8}`) stays FIXED
-- the eigenvectors should be reproducible; only the stochastic noise is randomized.

## Ordered chunks
### D1 -- move the stochastic/solve/projection helpers into the header
Files: `disc_lma_v2_common_claude.h`, `disc_lma_estimator_bench_v2_claude.cc`.
Move `StochasticDilutedSource`, `Solve`, `TraceField`, `SolveHighProjected`, `SolvePropProjected`,
`LowPartOnSource` from the estimator bench `.cc` into the header (inline). Refactor the estimator
bench to use them from the header. NO behavior change (re-verify the estimator bench output).

### D2 -- the production binary
Files: `disc_multipleGamma_binary_lma_claude.cc`, `Make.inc`, run script.
- CLI like the disc binary: `--mass --beta --dir --obsdir` + `--eigref` (the per-ensemble landscape).
  Eigensolve knobs via env (`ReadLMAEigParams`).
- Config loop (discover `--dir`, self-skip on existing `--obsdir/traces.<gam>.<conf>`, wall blocker).
- Per config: read NERSC gauge; seed `RNG4` (randomized -- see decision); `BuildLowModes` +
  `BuildA2ASet`; accumulate exact $L^\text{low}$; the (t,eo)-diluted source loop with the PROJECTED
  high solve (`SolvePropProjected`) -> $L^\text{high,stoch}$; LMA loop $= L^\text{low}+L^\text{high}$.
- Write Scidac per gamma per config (format matching the disc binary). Optionally also write the
  noise-free $L^\text{low}$ separately (cheap, reusable) -- decision below.

## DECISIONS (2026-06-26)
1. **RNG**: `RNG4.SeedUniqueString(config_path)` per config (config-unique, DETERMINISTIC, reproducible
   -- the reference disc binary's approach; randomizes across configs, NOT the testbed's fixed seed).
   Eigensolve/gauge RNGs stay fixed (reproducible eigenvectors).
2. **Output**: combined LMA loop ONLY, in the `traces.<gam>.<conf>` Scidac format (drop-in; pipeline
   unchanged). No separate $L^\text{low}$ file.
3. **Evec checkpoint**: INCLUDE. After `BuildLowModes`, save the refined double subspace `sub` +
   `eval_use`; on rerun reload to SKIP the eigensolve. Format: Grid evec I/O for the fields (confirm
   the exact facility -- EigenPack / ScidacWriter on the rb grid) + HDF5 for the evals (eigref-style).
   Helpers `SaveEvecs`/`LoadEvecs` go in the header (reusable for conn LMA / deflation).
