# Shrink-flow frame driver -- implementation plan

Goal: test the two-stage "shrink flow" frame (discussion:
`Grid/scripts_nm/freeprec_gaugefix_topology_claude.md` Sec. 4) -- Stage 1 Wilson flow to $t_0$ kills
the perturbative dof; Stage 2 switches the flow force to an UNDER-improved action ($\varepsilon > 1$,
$c_1 > 0$) whose $O(a^2)$ artifact shrinks the instanton radius, ideally until the lump(s) fall
through and the frame copy becomes trivial-sector; Stage 3 short Wilson re-smooth; then Landau fix
and measure the $M_0$ win on the ORIGINAL config vs the frame-independent CGNE baseline.

**Algorithm sources (mandatory citations):** instanton-size control via the $O(a^2)$ term of the
smearing/cooling action: Garcia Perez, Gonzalez-Arroyo, Snippe, van Baal (hep-lat/9309009);
de Forcrand, Garcia Perez, Stamatescu, improved cooling (hep-lat/9701012); Moran, Leinweber,
over-improved stout smearing and the $\varepsilon$-parametrisation $c_1 = -(1-\varepsilon)/12$,
$S(\rho) = S_0[1 - (\varepsilon/5)(a/\rho)^2 + O(a^4)]$ (arXiv:0801.1165); cooling/gradient-flow
equivalence: Bonati, D'Elia (arXiv:1401.2441). Shrink flow = $\varepsilon \in (1, 5/2)$, stability
needs $c_0 = 1 - 8 c_1 > 0$, i.e. $c_1 < 1/8$; default here $c_1 = +1/12$ ($\varepsilon = 2$).

## Files

- `Grid/tests/solver/Test_dwf_shrinkflow_claude.cc` -- NEW dedicated driver (mimics
  `Test_dwf_flowscan_claude.cc` idioms: flow-force swap via `setGaugeAction(RBCGaugeAction(Nc,c1))`,
  divergence guard, Landau fix, M0-resid, capped FGMRES(M0), optional M1).
- `Grid/scripts_nm/grid_shrinkflow_build_scc_gpu_claude.sh` -- targeted GPU build into build_merged
  (copy of the spectrum build recipe).
- `Grid/scripts_nm/grid_shrinkflow_gpu_qsub_claude.sh` -- SGE GPU batch script (Nobu prefers qsub
  over interactive for GPU jobs; copy of grid_spectrum_gpu_qsub_claude.sh; 1 GPU + omp 8, 12h).
  C1 env is a SPACE-separated list (SGE -v splits on commas; script converts to the --c1 comma list).
- `Grid/tests/solver/Test_dwf_flowscan_claude.cc` -- BUG FIX only: `--fgmres_restart` was parsed in
  `main` into a variable that exists only inside `run_flowscan` (added 09-04 after the last rebuild;
  will not compile as-is). Fix = file-scope `g_fgmres_restart` (idiom of Test_dwf_freeprec_claude.cc:68).

## Driver design

Physics point identical to the flowscan (Ls=8, M5=1.8, Mobius b=1.5 c=0.5, m=0.1, AP time), so all
$D_W$ counts are comparable. Sequence:

1. Load config; CGNE baseline once (frame-independent).
2. **Stage 1**: Wilson flow, `--nstep1` (default 146 = $\tau = t_0$ at eps 0.02, the current default
   frame). Print plaq, $Q_{5Li}$.
3. **Baseline frame** (default ON, `--no_baseline` to skip): Landau-fix a COPY of the stage-1 config
   and run FGMRES(M0) -- the current-default frame on the SAME config+source, so the run is
   self-contained A/B.
4. **Stage 2**: flow with `RBCGaugeAction(Nc, c1)`; `--c1` is a COMMA LIST (shrink-strength SCAN;
   each value gets its own stage2+3+frame+solve from the shared stage-1 config; default +1/12,
   trend scan e.g. `0,0.0417,0.0833,0.1167` = $\varepsilon$ 1, 1.5, 2, 2.4). Chunks of `--qchunk`
   (default 10) steps up to `--nstep2` (default 2000; Nobu: pure gauge is inert under the flow, so
   long stage 2 is safe -- tau2 up to 40 ~ 14 t0); after each chunk print tau, plaq, $Q_{5Li}$;
   STOP early when $|Q| <$ `--qstop` (default 0.5; <=0 disables). Divergence guard per chunk (plaq
   out of range -> revert to previous chunk, stop stage 2).
5. **Stage 3**: `--nstep3` (default 20) plain Wilson steps to erase the fall-through ripple.
6. **Shrink frame** (per c1): Landau-fix, print Landau functional + M0-resid, FGMRES(M0) [+ M1 if
   in `--ops`], win ratios vs CGNE.

Per-c1 trend observables: tau2 to reach $|Q| < 0.5$ (the shrink RATE, expect $\propto \varepsilon$),
plaquette trace (stability), Landau functional + M0-resid + win of the resulting frame.

**Checkpointing (Nobu 2026-09-05):** `--ckp_dir <dir>` (qsub env CKPDIR, default
`$ROOT/shrinkflow_ckp`; empty disables) writes the flowed config as NERSC (3x3, fp32) at every
$k \cdot t_0$ crossing of the TOTAL flow time (chunk-boundary resolution): `shrink_<cfg>_k1` = end
of stage 1 (c1-independent), `shrink_<cfg>_c1<tok>_k<k>` during stage 2, `..._c1<tok>_final` after
stage 3 (the actual frame config). ~19 MB each, up to ~15/c1 at the full nstep2. These feed the
**instanton-size probe** (next chunk): clover topological charge density $q(x)$ per checkpoint,
find lumps (peaks of $|q|$), estimate the radius from the BPST core value
$$
q(0) = \frac{6}{\pi^2 \rho^4}
\quad\Rightarrow\quad
\rho = \left( \frac{6}{\pi^2\, |q_\text{peak}|} \right)^{1/4},
$$
track $\rho_i(\tau)$ per lump vs the predicted $d\rho/dt \propto -\varepsilon a^2/\rho^3$, and
correlate with the frame quality (Landau functional / M0-resid / win) at the same $\tau$.

Other CLI (mirrors flowscan): `--config --t0 --flow_eps --ops cgne,m0[,m1] --solve_tol (1e-6)
--fgmres_restart (256) --gf_maxit (3000)`.

## Quick test (qsub, 1 GPU + omp 8)

```
qsub -v CONFIG=/projectnb/qfe/nmatsum/dwf/configs_iwasaki_16_b2.6/ckpoint_lat.640 \
  /projectnb/qfe/nmatsum/dwf/Grid/scripts_nm/grid_shrinkflow_gpu_qsub_claude.sh
# c1 trend scan: add C1='0 0.0417 0.0833 0.1167'
```
Watch: (a) does $Q_{5Li}$ on the frame copy actually walk to 0 (and in how much tau, per c1);
(b) does the plaquette stay sane (guard quiet); (c) shrink-frame M0 win vs the baseline-frame M0 win.

## Open questions

- If the shrink stalls (fat lumps, artifact-suppressed rate): raise c1 toward 1/8 (epsilon 2.4) or
  extend nstep2; if the UV goes unstable even with c0>0, interleave Wilson re-smooth chunks.
- Q-monitor uses $Q_{5Li}$ each chunk (cost fine at 16^4); switch to clover-only if it matters.
