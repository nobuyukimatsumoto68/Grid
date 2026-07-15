# Round-1 impl plan: conn + disc on the heavy 32^3x64 ensembles

Companion to the brainstorm `disc_conn_32c_brainstorm_claude.md` (rationale + variance
math + rejected levers). This plan is the AGREED, SCOPED work. Chunked; work top-down,
report after each chunk, get go-ahead before the next.

## Physics / goal

Connected ($C$) + disconnected ($D$) meson measurements on the four MATURE heavy
$32^3\times64$ ($L_s=16$, $M5=1.5$) ensembles, to combine as $C_\text{phys}=2D-C$:
`32_64_0.1/beta10.865m0.1` (60 cfg), `32_64_0.2/beta10.99m0.2` (79),
`32_64_0.3/beta11.035m0.3` (95), `32_64_0.4/beta11.045m0.4` (100); all stride 20 in
`/p/lustre5/matsumoto5/`. Light 0.01/0.05 = ROUND 2, DEFERRED (thermalizing).

Two scoped workstreams (everything else = "algorithm swampland", OUT):
1. **Dilution**: a tunable $K$-partition time dilution knob `DISC_TPART` (source on $K$
   equally-spaced slices $\{t+kN_t/K\}$, loop $t=0..N_t/K-1$ -> factor $K$ fewer solves).
   Default 2. Unbiased (cross term mean-zero over noise), variance $\sim K|S(N_t/K)|^2$
   exp-suppressed; SAFEST at heavy mass (short correlation length).
2. **Deflation/LMA bench-tune** for light + THRESHOLD masses: find where deflation/LMA
   stops paying. Bench NOW on mature m=0.1 (then 0.2 if 0.1 wins); light deferred.

## Binaries (all geometry-agnostic via `--grid 32.32.32.64`; only launch layout changes)
- Heavy disc baseline: `disc_multipleGamma_binary_mixedprec_claude.cc` (own
  StochasticDilutedSource :55, loop :311).
- Deflation/LMA disc: `disc_multipleGamma_binary_lma_v2_claude.cc` + header
  `disc_lma_v2_common_claude.h` (StochasticDilutedSource :518, conf loop :164, source
  loop :251).
- Conn: `baryons_0000_dirac_claude.cc` (loop mode, MobiusFermionD per config).
- Bench: `disc_lma_eigref_v2_claude.cc` (per-ensemble shift-invert -> spectrum h5),
  `disc_lma_estimator_bench_v2_claude.cc` (variance + break-even; NSRC knob :139),
  `disc_lma_bench_v2_claude.cc` (eigensolve gates).

## Chunks

### CHUNK 0 -- MPI layout for 32^3x64 (D6, prerequisite for every run)
Decide node count + rank grid. 24c used 8N/32r mpi 2.2.2.4 for 24^3x48 (per-rank
~20MB). 32^3x64 = 3.16x volume -> propose 16N/64r or 32N/128r so per-rank fits the host
arena AND the deflation evec store (NEV~100) avoids the AccCache cliff (the resident-NEV
ceiling rises with node count). Output: chosen `--grid`/`--mpi` + a 1-config disc startup
fit check (USER runs). No source change.
Files: (decision; feeds all submit_* scripts below). NO code.

### CHUNK 1 -- DISC_TPART K-partition time dilution (workstream 1)
1a. `disc_lma_v2_common_claude.h`: `StochasticDilutedSource` gains `(Nt, tpart, tbase)`;
    mask = OR over $k=0..tpart-1$ of `(t == tbase + k*Nt/tpart)`. tpart=1 -> current.
1b. `disc_multipleGamma_binary_lma_v2_claude.cc`: read `DISC_TPART` env; outer loop
    `tbase=0..Nt/tpart-1`; pass tpart. TraceField accumulation / wall-blocker / self-skip
    / output format UNCHANGED (traces localize on the source slices).
1c. `disc_multipleGamma_binary_mixedprec_claude.cc`: same knob in its standalone
    StochasticDilutedSource (:55) + loop (:311) -- this is the HEAVY production binary.
1d. variance-vs-K measurement: driver runs the heavy disc binary at K=1,2,4 on a few
    m=0.1 configs; jackknife the per-gamma disc loop variance vs per-config wall -> pick
    production K (expect 2). Output is the same trace format (different noise) -> fix one K.
Files: `disc_lma_v2_common_claude.h`, `disc_multipleGamma_binary_lma_v2_claude.cc`,
`disc_multipleGamma_binary_mixedprec_claude.cc`, new `run_disc_tpart_scan_claude.sh` +
`submit_disc_tpart_scan_claude.sh`, analysis cell in an ipynb / `*_tpart_results_claude.md`.

### CHUNK 2 -- deflation/LMA spectrum probe + break-even on heavy m=0.1 (then 0.2)
2a. New submit scripts at the chunk-0 layout pointing at the m=0.1 config path (binaries
    unchanged, just `--grid` + path): eigref + estimator bench + gates.
2b. eigref on one m=0.1 config -> spectrum (lambda_min, density below a cut, lambda_max)
    -> Cheby window; write `eigref_3264_b10p865_m0p1000.h5`.
2c. estimator bench at that spectrum: deflation break-even $T(N_\text{ev})=$ eig_setup
    $+ N_\text{solve}\cdot$solve$(N_\text{ev})$ vs plain mixedprec; LMA variance vs
    $N_\text{ev}$. VERDICT: does deflation/LMA win at m=0.1? If yes -> repeat at m=0.2.
    NEV stays HBM-resident (no evec disk-store this round -- see DEFERRED below).
Files: new `submit_disc_eigref_32c_claude.sh`, `submit_disc_estbench_32c_claude.sh`,
`run_disc_defl_probe_32c_claude.sh`, eigref h5 output, `*_defl_threshold_results_claude.md`.

### CHUNK 3 -- heavy production drivers (disc + conn), per the threshold finding
3a. disc: masses where deflation LOSES -> mixedprec binary (+ DISC_TPART); where it WINS
    -> lma_v2 binary (+ eigref + DISC_TPART). Driver loops 0.1-0.4, NATIVE stride 20,
    wall blocker, internal self-skip, flux auto-chain. New `run_disc_32c_claude.sh` +
    `submit_disc_32c_claude.sh` (+ a `..._lma_...` variant for winning masses).
3b. conn: loop-mode conn on 0.1-0.4, native stride 20, same config set as disc (for
    $2D-C$). Decide double vs mixedprec conn (conn is cheap; default double unless the
    bigger volume makes it slow). New `run_meson_32c_claude.sh` reusing
    `submit_meson_momproj_loop_claude.sh` with `--grid`.
Files: new disc + conn drivers/submits (mirror the 24c ones; 32c grid/paths/layout/dir).
Output dirs: NEW per-ensemble obs dirs under `/p/lustre5/matsumoto5/` (TBD naming).

### CHUNK 4 -- validation gate (before mass production)
4a. disc unbiasedness on one heavy config: LMA traces vs mixedprec traces agree within
    stochastic error (DET gate proven on 24c; re-confirm at 32c geometry); K=2 vs K=1
    agree within error.
4b. conn sanity: a heavy conn correlator vs a 24c-style reference channel.
Files: analysis cells / `*_validation_claude.md`.

## DEFERRED (explicitly NOT this round)
- Evec disk-store / `NEV_streamed` (user 2026-06-29: "let's not work on saving evecs for
  now"). NEV stays HBM-resident; the resident-NEV ceiling is managed by NODE COUNT
  (chunk 0). Disk-store analysis kept in the brainstorm doc Idea 7 for a future round.
- Round-2 light 0.01/0.05 + shared-basis conn; MADWF; Block CG; Schur DiagTwo; HP.

## Open questions
- OQ1 (chunk 0): target node count -- 16N/64r vs 32N/128r? Resolve with the fit check.
- OQ2 (chunk 2): if deflation wins at m=0.1 AND 0.2, do we also probe 0.3/0.4, or assume
  the trend and run mixedprec there? (user chose "0.1 then 0.2 if wins" -> revisit.)
- OQ3 (chunk 3): conn at heavy -- plain double (simple) or mixedprec? Measure conn
  per-config wall at 32c first.
- OQ4 output-dir naming for the 32c heavy obs (mirror `obs_nc4nf1_3264_b..._m...`?).

## Ground rules (inherited, DO NOT BREAK)
Claude NEVER submits/compiles/rm. User runs all builds+jobs. Outer solver tol 1e-8 NEVER
relaxed. Match original `.cu`/`.cc` formatting (2-space, one stmt/line). Preserve old
lines commented when swapping to a variant for A/B (DISC_TPART default 1 keeps old behavior).
