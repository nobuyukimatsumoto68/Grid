# Impl plan: LMA for the CONNECTED correlator (shared eigenbasis with disc)

Companion to `disc_conn_32c_round1_impl_plan_claude.md` (heavy mixed-prec round 1 —
unaffected by this work) and `disc_lma_HANDOFF_claude.md` (eigensolve machinery).
This plan is the round-2-facing work the user chose to start FIRST: bring LMA to the
connected measurement, reusing the per-config low-mode basis the disc LMA v2 stack
already builds. Chunked; work top-down, report after each chunk, get go-ahead before
the next.

## Physics / goal

Light-mass conn is 16 point-source solves per config (Nc=4 x Ns=4) and each solve is
slow at m=0.01 on 3.16x volume; the SAME per-config eigenbasis that disc LMA v2 builds
can (a) deflate those solves and (b) replace the low part of the correlator by an
exact, all-source-averaged low-mode 2pt. Two estimator phases, strictly staged:

- **Phase A — deflated conn (estimator UNCHANGED).** Solve the 16 columns with the
  projected mixed-prec batched solver: exact low part from the eigenpairs
  `x_low = sum_i (1/sigma_i) a_i <b_i, src>` + high part via
  `SolvePropProjectedMixedBatched` (`disc_lma_v2_common_claude.h:658`). Output is the
  same correlator to solver tolerance -> trivially validated against existing conn.
- **Phase B — full conn LMA (estimator CHANGES; own validation gate).**
  `C_LMA(t) = C_low^{all-src}(t) + [ C_exact^{point}(t) - C_low^{point}(t) ]`, where
  `C_low^{all-src}` is the translation-averaged low-mode 2pt built from the physical
  A2A pairs (`BuildPhysicalA2A`, `disc_lma_v2_common_claude.h:124`). Unbiased by
  construction (correction term subtracts the same low part at the same source);
  gains statistics from volume-averaging the IR-dominated low part.

**Development/validation bed = the 24c LIGHT ensembles NOW** (complete, eigrefs +
tuned windows at repo root, plain-conn references `mesons_conn.<conf>.h5` in
`/p/lustre5/matsumoto5/obs_nc4nf1_2448/obs_nc4nf1_2448_b10p800_m0p0100/` produced by
`baryons_0000_dirac_claude` at CG 1e-9 double). Deploy on 32c m=0.01/0.05 when they
thermalize (round 2). No dependency on / interference with heavy round-1.

## Eval / evec checkpoint design (D-IO)

Question raised 2026-07-20: are eval checkpoints needed, and do MULTIPLE evecs in
`.h5` make sense?

- **Evals -> HDF5: YES, always.** Tiny (KB), boss-rank write is fine; this is already
  the implemented format (`SaveEvecs` writes `Nuse` + `eval` datasets,
  `disc_lma_v2_common_claude.h:741`). Checkpoint `eval.<conf>.h5` per config
  unconditionally in every LMA run — costs nothing, gives spectrum QA / window
  re-tuning / low-mode saturation checks offline. NOTE: evals alone cannot seed a
  restart — the low-mode subspace needs the vectors.
- **Evecs -> HDF5: NO.** Grid's `Hdf5Writer` is a boss-rank serial writer (fine for a
  KB eval vector, wrong tool for O(100) x multi-GB fields — gather-to-boss + serial
  write). The implemented and correct path is **Scidac** (`ScidacWriter`, parallel
  I/O, per-field records, one file `evec.<conf>.scidac` holding all NEV vectors;
  `LoadEvecs` restores `Checkerboard()=Odd`). "Multiple evecs in one .h5" buys
  nothing over the existing single Scidac file and loses parallel I/O.
- **Evec checkpoint policy: TRANSIENT produce->consume only, OFF by default.**
  Sizes (odd-cb 5d fermion, Ls=16, Nc=4, double): 24c ~1.36 GB/evec -> ~136 GB/config
  @NEV=100; 32c ~4.3 GB/evec -> **~430 GB/config** -> a persisted light ensemble
  (~80-130 cfg) is O(40 TB): NOT a persistent store (consistent with the 2026-06-29
  "no evec disk-store" decision). What IS now justified: an `EVEC_SAVE=1`/auto-load
  knob so a conn job can reuse the basis a disc job built (or resume after a wall
  cut) without re-paying the eigensolve, with the driver DELETING `evec.<conf>.scidac`
  once both consumers have self-skipped that config. `eval.<conf>.h5` is kept forever.
- **Readout speed is the real deferral reason (user 2026-07-20) — fix it, don't swap
  containers.** The slow path is not Scidac-vs-HDF5: Grid HDF5 is boss-rank SERIAL
  (strictly slower for GB-scale fields); Scidac is already MPI-parallel into one file
  per config. The actual levers: (1) **Lustre striping** — driver runs
  `lfs setstripe -c -1 -S 16M` on `EVEC_DIR` before first write (default striping
  serializes a 136-430 GB file onto few OSTs and is the classic "readout is slow");
  (2) **single-prec storage** (halves bytes; modes are single-prec from the eigensolve
  anyway, optional RR re-refine on load); (3) later, batched mode-by-mode streamed
  reads overlapped with compute (= Idea 7 `NEV_streamed`). Budget: eigensolve
  ~150-220 s/config (3-5% of disc wall at 24c) vs a striped parallel read of 136 GB
  at 5-10 GB/s ~ 15-30 s -> reload wins ~10x IF striping is set. GATE the knob on a
  measured LoadEvecs-vs-BuildLowModes timing (chunk 2).
- **Production shape stays open (OQ-C): FUSE conn into the disc LMA job** (zero I/O,
  basis stays in HBM) vs **checkpoint chain** (disc EVEC_SAVE -> conn EVEC_LOAD ->
  delete). For STANDALONE conn the checkpoint is mandatory: 16 point solves can never
  amortize re-paying the eigensolve.

## Binaries / files

- Shared machinery: `disc_lma_v2_common_claude.h` (`BuildLowModes:362`,
  `BuildPhysicalA2A:124`, `SolvePropProjectedMixedBatched:658`, `SaveEvecs:741`,
  `LoadEvecs:758`, env knobs `ReadLMAEigParams:324`).
- Conn reference: `Grid_sdm_build/src/gauge_gen_Nc4/baryons_0000_dirac_claude.cc`
  (loop mode, `output_complete:394` self-skip, `MesonTrace_hdf:158`, wall blocker).
- NEW `Grid/examples/baryons_0000_dirac_lma_claude.cc` (name revives the pattern
  promised in the brainstorm; the 24c file never actually existed). Lives in
  `Grid/examples/` so it includes the common header and builds via
  `make -C build/examples`, like the disc LMA binaries.

## Chunks

### CHUNK 0 — eval checkpoints + evec save/load knob in the EXISTING disc binary
STATUS: CODE DONE (2026-07-20), NOT YET COMPILED (user builds:
`make -C build/examples disc_multipleGamma_binary_lma_v2_claude`). Implementation notes:
new header pieces are `SaveEvals` / `SaveEvecRecords` / `LoadEvecRecords` (records-only,
prec switch; v1 `SaveEvecs`/`LoadEvecs` untouched so the v1 binary keeps compiling);
precision is encoded in the FILENAME (`.scidacf` single / `.scidac` double) so a prec
mismatch on load is a miss -> eigensolve fallback; single-prec load RR-re-refines
(gated on `RR_REFINE`) and refreshes the eval h5; `eval.<conf>.h5` + `evec.<conf>.*`
both land in `EVEC_DIR` (default obsdir) so a consumer finds them together; the
striping line ships with chunk 2's submit script.
0a. `disc_multipleGamma_binary_lma_v2_claude.cc`: always write `eval.<conf>.h5` to
    obsdir after `BuildLowModes` (boss-guarded; reuse the `SaveEvecs` eval block or a
    small `SaveEvals` split out of it in the header).
0b. env `EVEC_SAVE=1` -> also `SaveEvecs` to `EVEC_DIR` (default obsdir);
    `EVEC_LOAD=1` -> try `LoadEvecs` before eigensolving (skip `BuildLowModes` on
    hit). Default both OFF: production behavior byte-identical.
0c. I/O made fast per D-IO: submit script does `lfs setstripe -c -1 -S 16M $EVEC_DIR`
    before first use; `EVEC_PREC=single|double` (default single) selects storage
    precision (single = `LatticeFermionF` records, RR re-refine on load); timing
    printout `LoadEvecs wall` vs `BuildLowModes wall` per config.
Files: `disc_lma_v2_common_claude.h`, `disc_multipleGamma_binary_lma_v2_claude.cc`.
Cheap; also immediately useful to disc reruns.

### CHUNK 1 — Phase-A conn binary (deflated solves, estimator unchanged)
STATUS: CODE DONE (2026-07-20), NOT YET COMPILED (user runs `build_conn_lma_claude.sh`
at repo root -- Make.inc gained the target so the automake regen steps are required).
Implementation notes: lives in `Grid/examples/` (includes the common header); output
schema pieces copied verbatim from the reference; the solver stack is the disc-v2 one
at outer 1e-8 (reference conn was 1e-9 -- gate accordingly); extra knob
`CONN_COMPARE_PLAIN=1` runs the header's plain 1e-8 Solve too and prints per-column
reldiff_max + walls (in-binary apples-to-apples gate + break-even, feeds chunk 2);
per-config timing line prints basis/low/high walls (1b).
1a. `baryons_0000_dirac_lma_claude.cc`: loop-mode skeleton copied from
    `baryons_0000_dirac_claude.cc` (config auto-detect, `output_complete` self-skip,
    wall blocker, `mesons_conn.<conf>.h5` schema UNCHANGED — drop-in for the
    `2D - C` analysis). Per config: `LoadEvecs`-else-`BuildLowModes` (same env knobs
    + `--eigref` as disc), then the 16 point-source columns through
    exact-low + projected-high mixed-prec batched solve, outer tol 1e-8.
1b. Timing printout per config: eigensolve wall vs solve wall vs plain-conn wall (the
    conn-only break-even; informs whether standalone conn-LMA ever runs unfused).
Files: new `baryons_0000_dirac_lma_claude.cc`.

### CHUNK 2 — Phase-A validation on 24c m=0.01 (USER runs)
STATUS: DONE -- ALL GATES PASSED (2026-07-20, pdebug f3MSqEYXqycs, 24c m=0.01
lat.100/104, `submit_conn_lma_validate_claude.sh` two-pass + `compare_conn_lma_claude.py`):
- CORRECTNESS: LMA conn vs 1e-9 reference h5: reldiff_max 1.8e-7 over all 190 datasets
  (the expected 1e-8-vs-1e-9 solver level). pass1 (fresh basis) vs pass2 (reloaded
  single-prec basis): 1.3e-8 -- basis provenance irrelevant, as designed. In-binary
  COMPARE_PLAIN propagator-column reldiff 2.1e-5 = solver tol at m=0.01 conditioning
  (kappa~1e6 -> sqrt(kappa)*1e-8; correlator level is the 1e-7 above).
- I/O GATE (D-IO): LoadEvecRecords 24 s (~68 GB single @ ~2.8 GB/s striped) + RR
  re-refine 5 s = 29 s vs BuildLowModes 121-137 s -> reload wins ~4.5x. Write 41 s.
  CHECKPOINT PATH CONFIRMED VIABLE; OQ-A RESOLVED: single-prec storage works
  (gates above passed through a single-prec round-trip).
- BREAK-EVEN (1b): standalone conn with FRESH eigensolve LOSES (167-185 s vs plain
  147-165 s at the same 1e-8) -- as predicted, 16 solves cannot amortize. With the
  reloaded basis conn is 86-90 s/config => ~1.8x vs plain-1e-8 (more vs the 1e-9
  reference path). => standalone conn REQUIRES the checkpoint (or fusion).
2a. New `submit_conn_lma_validate_claude.sh` (24c layout `-N 8 -n 32`, grid
    `24.24.24.48`, mpi `2.2.2.4`, eigref `eigref_2448_b10p800_m0p0100.h5`, LATDIR
    `conf_nc4nf1_2448_b10p800_m0p0100`, fresh OBSDIR under
    `/p/lustre5/matsumoto5/validate_conn_lma/`) on ~3 configs.
2b. GATE: every channel of `mesons_conn.<conf>.h5` agrees with the existing 24c
    reference h5 at the solver-tolerance level (expect ~1e-7 relative, as in the
    disc defl/LMA gates). Small python compare script alongside the submit script.
2c. I/O GATE: same run exercises disc `EVEC_SAVE` -> conn `EVEC_LOAD` on the striped
    dir; record `LoadEvecs` vs `BuildLowModes` walls (D-IO predicts ~10x in favor of
    reload; if reload does NOT clearly win with striping+single, drop the checkpoint
    path and commit to FUSED for production).
Files: new `submit_conn_lma_validate_claude.sh`, `compare_conn_lma_claude.py`,
`*_conn_lma_validation_claude.md`.

### CHUNK 3 — Phase-B full conn LMA (all-source low-mode 2pt)
STATUS: GATES PASSED on 2 configs (2026-07-21, pdebug f3MXtHhZ9jV1, lat.100/104):
DET gate 7e-15 (exact); term identity 0; `_t_point` vs Phase A 1e-8; phaseB wall
62-65 s/config (pair matrices); conn total ~150 s/config with reloaded basis.
3c VERDICT (2026-07-21, 12 cfgs lat.100-144, blocker stopped before 148; DET gates
10/10 PASS): **Phase B VALIDATED, UNBIASED, LARGE WIN.**
- UNBIASEDNESS (decisive test): combined mean vs the 165-config PLAIN 24c reference
  (obs dir, cfgs>=100): |z| <= 1.9 at all probed t, both signs, no trend. The naive
  12-cfg paired test had shown 3-8 "sigma" in G5/I_I -- diagnosed as the POINT term's
  heavy-tail small-sample bias (12-cfg point mean sits LOW vs the 165-cfg truth, e.g.
  t=41 1.21e-2 vs 1.71e-2; lowall/lowpt ratio wanders 0.8-1.5 not a constant; lowall
  exactly t-symmetric). Phase B REMOVES that tail, it does not introduce bias.
- VARIANCE: same-12-cfg var-ratio med: G5_G5 0.074, most channels 0.15-0.5, GT_GT ~2.6
  (vector, low part tiny -- consider using `_t_point` for that channel in analysis).
  vs the plain-reference variance the pion gain is far larger: err(t=24) 1.9e-5 @12cfg
  combined vs 1.5e-4 @165cfg plain => ~30x error at equal configs (~1e3 in statistics)
  at mid-t. Light-mass conn LMA is decisively worth it.
- Remaining (optional, for the record): resubmit the same command to finish cfgs
  148-176 -> 20-cfg set.
(2026-07-21 build fix: std::norm -> real/imag sums; Grid Complex is thrust::complex
on HIP. Stale-binary guard added to the submit script after a stale Phase-A run.)
SCOPE NOTES: MESONS ONLY (user 2026-07-20: no baryon work -- the binary never carried
the reference's disabled baryon sector; the filename is lineage). Phase B applies to
the ZERO-MOMENTUM time correlators (`_t`, `_t_p0`); `_x` and p!=0 datasets stay the
point estimator (phased slice sums deferred). Implementation: header
`SlicePairSpinMatrix` (colour-traced spin outer product, time-sliced; contraction
identity documented there) + `GammaMat`; binary retains the A2A set under
`CONN_LMA_FULL=1`, builds Nev^2 pair matrices for a and b (one kernel + sliceSum per
pair, ~245 MB/rank host at Nev=100 Nt=48), assembles lowall/lowpt per channel from the
shared ChGammas table, and runs an in-binary DET gate (pair-matrix lowpt vs PropLow
contraction, exact identity, FAIL >= 1e-8). Writer `MesonTrace_hdf_lmafull` swaps
`_t`/`_t_p0` for the combined estimator and adds `_t_point`/`_t_lowall`/`_t_lowpt`.
Run Phase B into a SEPARATE obsdir (self-skip would trip on Phase-A files).
3c variance verdict needs ~10+ configs (extend CONF_MAX on pbatch);
`analyze_conn_phaseb_claude.py` prints the term identity, the Phase-A cross-check,
and var(combined)/var(point) per channel.
3a. Header: `LowModeMeson2pt(...)` — from the physical A2A pairs `(a_i, b_i, 1/sigma_i)`
    build the meson field `M_ij(t) = sum_x  b_i^dag(x) Gamma a_j(x)` per gamma and
    contract to `C_low^{all-src}(t)`, normalized per source slice. Stream i,j blocks
    to bound HBM (same discipline as `BuildPhysicalA2A`).
3b. Binary: env `CONN_LMA_FULL=1` switches output to
    `C_low^{all-src} + (C_exact^{point} - C_low^{point})`; also write the three terms
    separately into the h5 (new datasets, additive — schema stays a superset).
3c. GATE on 24c m=0.01, ~10 configs: mean agrees with Phase-A/plain conn within
    stochastic error per channel+t; measure the variance reduction on the pion/IR
    channels vs cost. VERDICT: is Phase B worth it, or is Phase A (deflation only)
    the production setting?
Files: `disc_lma_v2_common_claude.h`, `baryons_0000_dirac_lma_claude.cc`, analysis
cells / `*_conn_lma_phaseB_results_claude.md`.

### CHUNK 4 — production shape for 32c round 2 (blocked on light thermalization)
STATUS: **REVISED TO FUSION (2026-07-22, user).** The standalone plan paid the light
NEV=100 eigensolve TWICE per config (separate disc + conn jobs). Fusion does it ONCE:
new binary `disc_conn_lma_fused_claude.cc` builds the basis per config, then produces
BOTH disc traces (exact low + K=2 mixed-prec projected high) AND conn Phase-B off the
one resident A2A set -- zero evec I/O, ~one eigensolve/config saved. The two standalone
binaries are UNTOUCHED (heavy round 1 keeps using them at NEV=20). Files:
`disc_conn_lma_fused_claude.cc` (+ Make.inc), `build_fused_lma_claude.sh`,
`submit_fused_32c_claude.sh`, `run_fused_32c_light_claude.sh` (replaces the separate
light disc/conn drivers), `submit_fused_fit_test_claude.sh` (8N memory+timing test at
the real NEV=100/NM=240 + full-A2A-retain + Phase-B footprint). NOT YET COMPILED.
Memory driver = BuildA2ASet retains all a,b,u at NEV=100 (like conn Phase B) ON TOP of
the NM=240 eigensolve peak -> that is exactly what the 8N fit test checks (16N default
until it passes). Bring-up: fit test prints PHASEB DET GATE (conn self-check) + wall +
--debug-mem peak; optionally cross-check fused disc traces vs a standalone disc run on
the same config before trusting production.
(Earlier standalone reasoning, for the record: at 32c NEV=100 the evec round-trip ~ the
eigensolve, so save/load was a wash -- but that ignored the doubled eigensolve across
two jobs, which fusion removes outright.) Thermalization check (plaquette headers, 2026-07-21): both light ensembles
creep until traj ~200 -> CONF_MIN=200. Files: `run_disc_32c_light_claude.sh` +
`run_conn_32c_light_claude.sh` driving the (now env-parameterized) generic 32c
submits with NSTOP/NK/NM/NEV=100/140/240/100, K=2, 16N/64r (kept for the NM=240
store), measure-as-we-go chains; light eigrefs via the parameterized
`submit_disc_eigref_32c_claude.sh` (env override example in its header; pbatch 120m).
4a. Preferred: FUSED binary/driver — extend the disc LMA v2 conf loop to run conn
    contractions after the trace while the basis is resident (env `CONN_FUSED=1`,
    conn self-skip independent of disc self-skip), OR chain disc(EVEC_SAVE) ->
    conn(EVEC_LOAD) -> driver deletes evec file. Decide from chunk-1 timing + the
    32c MPI layout / NEV ceiling determined in round-1 CHUNK 0.
4b. New `run_conn_lma_32c_claude.sh` + submit for m=0.01/0.05 at the chunk-0 layout;
    eigrefs `eigref_3264_b10p800_m0p0100.h5` / `..._b10p840_m0p0500.h5` via the
    existing `disc_lma_eigref_v2_claude` (may already exist by then from round-1
    chunk 2 work at the light masses — reuse).
Files: new driver/submit; possibly `disc_multipleGamma_binary_lma_v2_claude.cc`.

## DEFERRED / OUT
- Persistent evec store (sizes above; revisit only with single-prec + block
  compression and a real reuse case beyond conn+disc).
- Conn LMA at HEAVY masses (conn is cheap there; round 1 stays plain mixed-prec).
- Baryon (3-quark) LMA; momentum-projected A2A meson fields beyond p=0; MADWF etc.
  (algorithm swampland per the brainstorm).

## Open questions
- OQ-A (chunk 0): store evecs double (bit-faithful restart, 2x size+read time) or
  single + RR re-refine on load (half the I/O — the point of the exercise)?
  Default now SINGLE, since readout speed is the binding constraint (D-IO).
- OQ-B (chunk 3): Phase-B low part from the SAME NEV as deflation, or fewer
  (NEV_LMA <= NEV) to cut the all-source 2pt cost? Default: same NEV.
- OQ-C (chunk 4): fused vs chained default for 32c production (decide on timing).
- OQ-D: obsdir naming for 32c light conn (mirror `obs_nc4nf1_3264_b..._m...`?
  same open question as round-1 OQ4 — settle once for both).

## Ground rules (inherited, DO NOT BREAK)
Claude NEVER submits/compiles/rm. User runs all builds+jobs. Outer solver tol 1e-8
NEVER relaxed (conn reference was 1e-9 double: keep the Phase-A gate at the level
the mixed tolerances imply, document the achieved agreement). Match original `.cc`
formatting (2-space, one stmt/line). Old behavior preserved under default-off env
knobs (`EVEC_SAVE/EVEC_LOAD/CONN_LMA_FULL/CONN_FUSED` all default OFF).
