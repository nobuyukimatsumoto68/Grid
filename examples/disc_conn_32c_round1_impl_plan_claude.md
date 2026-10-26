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
STATUS: CODE + SCRIPTS DONE (2026-07-21), NOT YET COMPILED/RUN. Implemented:
header `StochasticDilutedSource` gains optional (tpart=1, Nt=0) args (default =
bit-identical original, same bernoulli stream; K>1 puts the SAME Z4 field on the K
slices {t + k*Nt/K}); both disc binaries read `DISC_TPART` (assert K|Nt, tbase loop
Nt/K) + `DISC_SEED_STR` (RNG seed suffix, K-scan resampling only, ""=production
unchanged) + always print `# LOOPP0 <gam> conf <c> t <t> re im` after the trace write
(log-based analysis; also a free unbiasedness monitor in production). 1d scan =
`run_disc_tpart_scan_claude.sh` (3 pbatch jobs K=1/2/4, NSEEDS=4 noise streams on
lat.1200 @ lma_v2 NEV=20) + `analyze_tpart_scan_claude.py` (noise var-ratio, FoM
F_K = (var_K/var_1)(T_K/T_1), bias-z; gauge variance only favors larger K; PLUS the
same-noise PAIRING-SPIKE check -- slices sharing a solve share noise, so the disc 2pt
<L(t)L(t+dt)> is BIASED at dt = multiples of Nt/K; the analyzer estimates the noise
covariance vs dt over seeds and flags the spike vs background).
K POLICY BY MASS (user 2026-07-21): HEAVY m>=0.1 -> K from this scan (spike
exponentially dead, but measured not assumed). LIGHT m=0.01/0.05 (24c production AND
32c round 2) -> K=1 DEFAULT; K=4 EXCLUDED (user: no contact spike at Nt/4); K=2 only
if a dedicated LIGHT-mass K-scan (same scripts pointed at 24c m0.01; conservative
proxy for 32c whose larger Nt separates slices further) shows the dt=Nt/2 spike
invisible.
Build: `make -C build/examples disc_multipleGamma_binary_lma_v2_claude
disc_multipleGamma_binary_mixedprec_claude baryons_0000_dirac_lma_claude` (header
touched -> conn binary rebuild too; no Make.inc change).
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
VERDICT (2026-07-21, eigref f3MYawbisfqh + bench f3MYvtaUBy9q, m=0.1 lat.1200,
16N/64r mpi 2.2.2.8): **MIXED-PREC WINS AT HEAVY MASS -- use it for ALL of m=0.1-0.4.**
- Spectrum: lambda_min=0.0239 (sigma_min=0.155 ~ the quark-mass scale), lambda_max=81.6,
  Nconv=114 in 1030 s shift-invert. GAPPED: NEV=100 covers only lambda<=0.028 -> no
  low-mode dominance -> no variance win possible (contrast m=0.01: same machinery = 
  ~30x error win on conn, ~5x disc speedup).
- Timing (8-group DISC_NGROUPS samples): T_mp=28.9 s/group; T_proj=26.1 s/group (both
  NEV=50 and 100 -- deflation shaves ~9%); T_eig+T_A2A = 121 s (NEV=50) / 247 s (NEV=100).
  Per config (128 groups): mixedprec 3700 s vs LMA 3460-3590 s -> <=6% net, single-sample
  jitter territory. NOT worth the eigref/window/NEV machinery at heavy mass.
- OQ2 ANSWERED: gap grows with mass -> no bench needed at 0.2-0.4, mixedprec everywhere.
- CHUNK-0/OQ1 ANSWERED: 16N/64r (2.2.2.8) fits INCLUDING the NM=240 eigensolve store
  (no AccCache trip) -> heavy production layout = 16N/64r. Heavy disc cost: ~62 min/config
  at 16N (K=1) -> ~31 min with DISC_TPART=2; 4-ensemble campaign ~170 node-days.
SHALLOW-NEV SWEEP DONE (2026-07-21, f3MZJtQhAqy9). Full m=0.1 table (s/config, 128
groups, 16N): mixedprec 3699 | NEV=10: 31+128*27.2=3509 (-5.1%) | **NEV=20:
48+128*26.6=3448 (-6.8%, SWEET SPOT)** | NEV=50: 3462 | NEV=100: 3588. T_proj plateau
(26.1-26.6) holds down to NEV=20, degrades at 10 -> the deflation win is the lowest
~20 modes. FINAL BINARY MATRIX (user-confirmed 2026-07-21):
- disc m=0.1  = lma_v2 @ NEV=20 (NSTOP=20 NK=40 NM=80; env only, traces drop-in).
- disc m=0.2-0.4 = mixedprec.
- conn m=0.1  = baryons_0000_dirac_lma_claude Phase A @ NEV=20 (STANDALONE: own 37 s
  eigensolve per config, NO evec checkpointing/coupling to disc -- transfer costs more
  than the eigensolve at this depth; CONN_LMA_FULL=0, gapped spectrum). ~70 s/config vs
  ~60-120 s plain; outer tol 1e-8 = the 32c conn standard (vs 1e-9 24c refs; measured
  ~2e-7 on correlators, far below stats). First config runs CONN_COMPARE_PLAIN=1.
- conn m=0.2-0.4 = plain reference binary (win would be minutes; keep uniform; can flip
  to the deflated binary by submit-script edit if m=0.2 conn measures slow).
Validation gate: one m=0.1 config LMA vs mixedprec disc traces within stochastic error
(chunk 4). NO evec disk I/O anywhere in round 1.
NEXT: chunk 1 (DISC_TPART + K-scan) then chunk 3 production drivers.
Original approach note: scripts were READY 2026-07-21 morning, run same day. Approach revised from
the original estimator-bench plan: the estimator bench's NSRC samples are full
dilution passes with UNBATCHED double CGs (both estimators) -- affordable on the 4^4
testbed, prohibitive on a real 32c config. The 32c break-even is therefore
TIMING-FIRST via DISC_NGROUPS timing mode (added to the mixedprec binary 2026-07-21,
default OFF, along with CONF_MIN/MAX/STRIDE env -- needed for chunk 3 anyway):
- `submit_disc_eigref_32c_claude.sh`: eigref on latest m=0.1 config, 16N/64r
  (2.2.2.8) = the chunk-0 fit check for the eigensolve (~8 GB/GPU evec store);
  pdebug -N 16, pbatch fallback; -> eigref_3264_b10p865_m0p1000.h5. The SPECTRUM
  (gap size vs m=0.01) is the variance-side verdict.
- `submit_disc_bench_32c_claude.sh`: one-config symlink dir, 3 phases at 8 groups:
  mixedprec baseline / lma NEV=50 / lma NEV=100 -> LMA wins iff
  T_eig+T_A2A+128*T_proj < 128*T_mp. Stale-binary guard included.
Build: `make -C build/examples disc_multipleGamma_binary_mixedprec_claude
disc_multipleGamma_binary_lma_v2_claude` (no Make.inc change, no regen).
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
STATUS: SCRIPTS DONE (2026-07-21), awaiting launch. K=2 EVERYWHERE (user 2026-07-21:
"K=2 for both" heavy AND light; at K=2 the same-noise pairing bias sits only at the
ANTIPODAL dt=Nt/2 -- flag/drop in analysis; K-scan jobs cancelled as moot; the RUNNING
24c chains stay K=1 as launched). OQ4 RESOLVED: obs dirs =
/p/lustre5/matsumoto5/obs_nc4nf1_3264/obs_nc4nf1_3264_b<beta>_m<mass>, conn+disc
SHARE the dir (24c pattern). Files: generic env-driven `submit_disc_32c_claude.sh` +
`run_disc_32c_claude.sh` (chain driver, 4 ensembles, binary matrix + TPT via --env,
16N/64r, -t 240m backfill-friendly, stale-binary guards) and `submit_conn_32c_claude.sh`
+ `run_conn_32c_claude.sh` (same pattern; first-ever m0.1 conn job auto-enables
CONN_COMPARE_PLAIN=1 -> in-binary gate on every config it completes, chain continues
without it once mesons_conn files exist). OQ3 RESOLVED: conn m0.1 deflated @1e-8,
conn m0.2-0.4 plain 1e-9 reference binary. CONF_MIN not set (starts at lat.20) --
run the plaquette thermalization check before/alongside the first jobs; overridable
per ensemble via extra --env=CONF_MIN=... without script edits (disc binaries only;
plain conn uses --stride/native range).
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
