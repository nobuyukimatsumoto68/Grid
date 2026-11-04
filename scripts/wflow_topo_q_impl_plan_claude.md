# Gluonic topological charge via Wilson flow (24c SU(4) ensembles) -- impl plan

## Goal
Measure the gluonic topological charge $Q$ (and the flow energy density) on every gauge configuration used in the
meson analysis, with Grid's native Wilson-flow program, to replace / check the fermionic proxy
$q = m\sum_x\mathrm{tr}[\gamma_5 S]$ (`meson_v2_topo_q_claude.py`, not integer-like). User request 2026-11-03.

Methods:
- Wilson (gradient) flow, RK3 integrator: M. Luescher, JHEP 08 (2010) 071, arXiv:1006.4518
  (Grid `Grid/qcd/smearing/WilsonFlow.h`, `evolve_step`).
- $Q$ clover definition `WilsonLoops::TopologicalCharge`; improved `TopologicalCharge5Li`:
  P. de Forcrand, M. Garcia Perez, I.-O. Stamatescu, NPB 499 (1997) 409, hep-lat/9701012.

## Grid native program: `Grid/HMC/ComputeWilsonFlow.cc` (built with the Grid tree, `build/HMC/ComputeWilsonFlow`)
- Reads `input.xml` from the CWD: `WilsonFlow{steps, step_size, meas_interval, maxTau, meas_interval_density,
  flow_type, path}` and `Configurations{conf_path, conf_prefix, conf_smr_prefix, rng_prefix, StartConfiguration,
  EndConfiguration, Skip}`; loops conf = Start..End step Skip, file `conf_path/conf_prefix.<conf>` (NERSC).
- Prints per measurement: `[WilsonFlow] Saved energy density (clover) & topo. charge density: conf step tau
  (E_avg,T_sum) E Q ...`, default `[WilsonFlow] Top. charge : step Q`, and at the end `TopologicalCharge conf Q`,
  `TopologicalCharge5Li conf Q5Li`.
- SU(4) TRAP: `flow_type = "Wilson"` sets `WilsonGaugeAction(3.0)`. The flow kernel is $\partial S$ with
  `deriv factor = 0.5 beta/Nc`; Luescher's normalization needs beta = Nc (library default
  `WilsonGaugeAction(Gimpl::num_colours)`). With 3.0 and Nc = 4 the flow time is rescaled by 3/4. Fix: leave
  `flow_type` EMPTY (no match -> library default, correct) -- no code change -- or a `_claude` copy with
  `WilsonGaugeAction(Nc)`. (Q at large flow time is unaffected either way; t-labels / energy density are.)
- A missing config file aborts the loop -> ranges must follow existing files (no gaps inside a range).

## Config sets (union of connected and disc configs used in the analysis)
- m0.1 b10.865: 1000..4640 step 20; m0.2 b10.99: 1000..4680/20; m0.3 b11.035: 1000..7980/20; m0.4 b11.045:
  1000..9980/20 (Sungwoo dirs /p/lustre5/matsumoto5/conf_nc4nf1_2448/conf_nc4nf1_2448_b<b>_m<m>/).
- m0.01 b10.8: 4..756 step 4 (conn from 4, LMA disc from 100) + continuation 760..latest step 4
  (/p/lustre5/matsumoto5/conf_nc4nf1_2448_cont/...); m0.05 b10.84: 4..96/4, 500..2136/4 (gap 100-496 has no
  configs) + continuation 2140..latest/4.

## Files
- `Grid/scripts/submit_wflow_q_tuolumne_claude.sh` (new): flux batch launcher, 1 node, writes input.xml into a
  per-(ensemble, range) run dir, runs ComputeWilsonFlow, log in the run dir.
- `Grid/scripts/run_wflow_q_tuo_claude.sh` (new): driver over the ensembles / ranges above.
- `Grid_sdm_build/src/gauge_gen_Nc4/wflow_q_parse_claude.py` (new): logs -> npz (conf, tau, E, Q, Q5Li) + plots
  (Q time series, histogram, Q vs flow time, comparison with the proxy q).
- `rsync_pull_wflow_q_claude.sh` (new, repo root): pull the logs.

## Chunks
1. Check the cluster binary exists (`build/HMC/ComputeWilsonFlow`; else `make -C build/HMC ComputeWilsonFlow`)
   + 1-config debug run (m0.4 lat.1000) to get the per-config time and check the output lines / normalization.
   Files: submit_wflow_q_tuolumne_claude.sh
2. Production driver over all ensembles.  Files: run_wflow_q_tuo_claude.sh
3. Pull + parse + plots.  Files: rsync_pull_wflow_q_claude.sh, wflow_q_parse_claude.py

## Open questions (user)
- Q1: flow action: Wilson flow (matches our Wilson gauge action) with the library default normalization
  (flow_type empty)? Or Symanzik / Zeuthen?
- Q2: flow range: step 0.01, steps 600 (t/a^2 up to 6, smearing radius sqrt(8t) ~ 6.9 < L/2 = 12), measurements
  every 20 steps? (Q should plateau well before.)
- Q3: config sets as above (union of conn + disc)? Include the continuation configs (HMC still running)?
- Q4: nodes per job (1 node / 4 GPUs suggested; the flow is cheap) and wall time.

## Decisions (user 2026-11-03)
- Q1: Wilson flow, library-default normalization (flow_type EMPTY -> WilsonGaugeAction(Nc)).
- Q2: step 0.01, 600 steps (t/a^2 to 6), measurements every 20 steps.
- Q3: config sets as listed, INCLUDING the continuation (extended) ensembles.
- Q4: 1 node per job.

## Chunk 1 status (2026-11-03): launcher written, NOT run
`submit_wflow_q_tuolumne_claude.sh`: writes input.xml (flow_type empty, steps/step_size/meas from env), checks binary +
first config, runs ComputeWilsonFlow on 1 node (mpi 1.1.2.2), log `wflow_q_<prefix>_<start>_<end>_claude.log` in RUNDIR.
Debug: m0.4 lat.1000 on pdebug 30m (command in the script header). Check: timing per config, the `[WilsonFlow]`
lines, Q vs flow time plateau, and that no "[WilsonGaugeAction] Beta: 3" is printed by the flow (default Nc).

## Chunk 1 result (debug, m0.4 lat.1000, 2026-11-03)
~0.33 s/flow step + ~1.6 s per measurement -> ~225 s/config (600 steps). Q5Li: -8.3 (t=0) -> -24.58 (t=1.0) ->
-24.97 (t=3) -> -24.98 (t=5.4): integer-like, plateau from t ~ 3. Clover Q -23.3 at t=5.4, still drifting
(O(a^2), ~7% low) -> use Q5Li. No "Beta" print (flow uses the library default WilsonGaugeAction(Nc)).
t^2<E>(plaq) at t=5: 0.571.

## Chunk 2 (2026-11-03): driver written, NOT run
`run_wflow_q_tuo_claude.sh`: stream table (orig + cont, incl. m0.1 cont from 4660), existing-config grid
start/step, contiguous runs cut into chunks of NPER = 55 configs (4h jobs), one 1-node job per chunk, job name
wflow_<tag>, rundir /p/lustre5/matsumoto5/wflow_nc4nf1_2448/<conf_dir_kind>/<tag>/<first>_<last>/; chunks whose
log holds "TopologicalCharge5Li <last>" are skipped (resume). DRYRUN=1 lists chunks.

## Production round 1 (2026-11-03): 28 chunks COMPLETED, 14 TIMEOUT (HANGS mid-flow, normal speed before)
Preview parse of the pulled logs (`wflow_q_parse_claude.py`, complete flows only): configs 171/390/145/185/277/251
(m0.01..m0.4); Q5Li at t = 6 INTEGER-LIKE: <|Q - round Q|> = 0.052/0.048/0.026/0.024/0.021/0.014.
Fermionic proxy q = m sum tr[g5 S] vs Q5Li: corr -0.990/-0.978/-0.965/-0.541/-0.507/-0.437, slope dq/dQ
-0.956/-0.954/-0.974/-1.066/-1.163/-1.053 -> q = -Q (sign convention) x ~0.95 at light mass (m_res) + stochastic
noise at heavy mass.

## Round 2 plan (user 2026-11-03): RERUN ENTIRELY with h5 output
- New binary `Grid_sdm_build/src/gauge_gen_Nc4/wflow_q_h5_claude.cc` (copy of Grid HMC/ComputeWilsonFlow logic,
  library-default flow action WilsonGaugeAction(Nc)), built with `make -f Makefile_claude wflow_q_h5_claude`
  (generic %.cc rule, no automake). CLI: --dir --prefix --start --end --step --outdir (+ STEPS/STEP_SIZE/MEAS env).
- Per config `<outdir>/wflow.<conf>.h5` (Grid Hdf5Writer): t, t2E_plaq, t2E_clover, Q_clover, Q5Li at every
  measurement step (+ plaquette at t = 0). Written to a temp name and renamed when complete -> a hang never leaves
  a partial file; self-skip of existing files (rerun = resume); graceful wall blocker (deadline env) as in the
  disc binaries.
- Driver + launcher replace the round-1 ones (side-by-side _h5 versions; round-1 files kept).
Open: Qr1 rank layout per job: (a) one flow on 4 ranks (as round 1, hangs seen) or (b) 4 concurrent
single-rank flows per node (1 GPU each, different configs; no MPI comms). Qr2 one h5 per config (above) or one
per ensemble.
Decisions (user 2026-11-03): Qr1 (b) 4 concurrent single-rank flows per node; Qr2 one h5 per config.

## Round 2 code (2026-11-03): written, NOT compiled
- `Grid_sdm_build/src/gauge_gen_Nc4/wflow_q_h5_claude.cc`: --dir --prefix --start --end --step --outdir --part --nparts;
  env WFLOW_STEPS/STEP_SIZE/MEAS/DEADLINE_EPOCH/TPT_SECONDS; WilsonFlow with default action, measurements at step 0
  and every MEAS: t, t2E_plaq, t2E_clover, Q_clover, Q5Li (+ plaq0, conf) -> wflow.<conf>.h5 via .tmp + rename;
  self-skip, missing-gauge skip, graceful blocker. Build: make -f Makefile_claude wflow_q_h5_claude.
- `Grid/scripts/submit_wflow_q_h5_tuolumne_claude.sh`: 1 node, 4 x `flux run -n 1 -g 1` (--mpi 1.1.1.1), logs in OUTDIR/logs.
- `Grid/scripts/run_wflow_q_h5_tuo_claude.sh`: ranges of NPER (60, to be set from the debug) existing configs per job,
  OUTDIR /p/lustre5/matsumoto5/wflow_nc4nf1_2448_h5/<tag>/, a range with all h5 present is skipped.
- Next: debug on m0.4 1000..1060 (4 configs, one per GPU), pdebug 30m -> single-rank time per config -> NPER.
Round-2 debug (2026-11-04, m0.4 1000..1060, 4 single-rank flows): ~955 s/config per GPU (node throughput ~239 s/config,
~ the 4-rank 225 s, no MPI); h5 written for all 4; Q5Li(t=6) conf 1000 = -24.983 (= round 1). NPER -> 52 (13/GPU/4h).
