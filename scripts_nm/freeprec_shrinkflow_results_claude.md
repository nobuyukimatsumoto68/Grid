# Shrink-flow frame results

Results of the two-stage shrink-flow frame experiment (Grid/tests/solver/Test_dwf_shrinkflow_claude.cc;
design in shrinkflow_impl_plan_claude.md; physics motivation freeprec_gaugefix_topology_claude.md Sec. 4).
Config = b2.6 16^4 traj 640 (the hard config, |Q|=3). CGNE baseline = 379 iters = 6064 D_W applies.

## Experiment 1 -- TWO-STAGE: Wilson to $t_0$, then switch to under-improved $c_1$ flow (2026-09-05)

Job 7472042. Stage 1 = Wilson to $\tau_1 = t_0$ (nstep1=146, eps 0.02). Stage 2 = switch flow force to
RBCGaugeAction($N_c, c_1$), scan $c_1$, chunked to $\tau_2 = 40 \approx 14\,t_0$ with a $Q_{5Li}$ monitor
(never triggered qstop -- $Q$ never fell). Stage 3 = 20 Wilson re-smooth steps. Baseline = Landau fix
straight after stage 1 (= the current default $s/t_0 = 1$ frame). Log
log/shrinkflow_ckpoint_lat.640_claude.log.

$\varepsilon = 1 + 12 c_1$ (Moran-Leinweber, 0801.1165). $c_1 = 0$ is the CONTROL (switch to Wilson-again
= no real switch, isolates flow-time from kernel).

| frame | $\varepsilon$ | $Q_{5Li}$ | Landau func | M0-resid | FGMRES iters | $D_W$ | win CGNE/M0 |
|---|---|---|---|---|---|---|---|
| baseline $s/t_0{=}1$ | -- | -2.992 | 0.013651 | 0.378524 | 171 | 1368 | 4.433x |
| shrink $c_1{=}0$ | 1.0 | -3.000 | 0.0079075 | 0.384420 | 147 | 1176 | 5.156x |
| shrink $c_1{=}0.0417$ | 1.5 | -3.000 | 0.0078965 | 0.384390 | 147 | 1176 | 5.156x |
| shrink $c_1{=}0.0833$ | 2.0 | -3.000 | 0.0078840 | 0.384350 | 147 | 1176 | 5.156x |
| shrink $c_1{=}0.1167$ | 2.4 | -3.000 | 0.0078725 | 0.384320 | 147 | 1176 | 5.156x |

### Findings
1. **The $c_1$ knob does essentially nothing.** All four $c_1$ give IDENTICAL 147 iters / 5.156x.
   Landau and M0-resid move monotonically in the predicted direction (more under-improvement -> lower
   both) but it is a ~0.1% effect -- four orders of magnitude too weak to change the iteration count.
2. **$Q$ is preserved at -3 out to $14\,t_0$ for all $c_1$.** Within the stable window ($c_1 < 1/8$,
   $\varepsilon \le 2.4$) under-improved flow does NOT drive the instantons through the lattice.
3. **Extended flow past $t_0$ does help the frame (4.43x -> 5.16x), but NOT by shrinking.** The gain
   comes from $I\bar I$ pair annihilation + bulk smoothing (see the size probe below), independent of
   $c_1$. This overturns the earlier "$\Omega$ saturates at $t_0$" premise for this hard config: more
   flow helps -- just not via topology.

### Instanton-size probe (Test_instsize_claude.cc, clover $q(x)$, BPST $\rho = (6/(\pi^2 |q_{peak}|))^{1/4}$)
Leading-lump $\rho_0$ (strongest peak = smallest $\rho$) and number of lumps vs total flow $\tau$.
The value AT $t_0$ (checkpoint k1, shared by all $c_1$): $\rho_0 = 3.19a$, 10 lumps, $Q_{clover} = -2.69$.

| $\tau/t_0$ | $c_1{=}0$ $\rho_0$ | nlumps | $c_1{=}0.1167$ $\rho_0$ | nlumps |
|---|---|---|---|---|
| 1  | 3.19 | 10 | 3.19 | 10 |
| 2  | 3.67 | 7  | 3.55 | 7  |
| 3  | 3.87 | 6  | 3.70 | 6  |
| 5  | 4.23 | 5  | 4.05 | 5  |
| 8  | 4.67 | 3  | 4.45 | 3  |
| 11 | 4.83 | 3  | 4.56 | 3  |
| 14 | 4.86 | 2  | 4.61 | 2  |

- Under BOTH Wilson and the strongest stable under-improvement, the surviving lump **BROADENS**
  ($\rho_0$ grows $3.2 \to 4.6$-$4.9a$ past $t_0$); it does NOT shrink. Under-improvement only slightly
  resists the broadening (4.61 vs 4.86 at $14\,t_0$).
- The **smallest lump is at $t_0$ itself**; any further flow (any kernel, stable window) broadens the
  survivor. The nlumps 10 -> 2 drop = the small ($I\bar I$) lumps annihilate, and $Q_{clover}$ sharpens
  toward -3.
- Caveat: $\rho_0$ is the single-site strongest-peak lump; the early leading lump (among 10, some
  dislocation-scale) is not strictly the same object as the late survivor. Direction is unambiguous.

### Verdict
Net NEGATIVE for the two-stage shrink-flow idea on fine b2.6: the $O(a^2)$ shrink force
$\sim \varepsilon a^2/\rho^3$ is negligible at $\rho \sim 4a$, and smoothing/broadening dominates --
exactly the framework's pessimistic caveat (no local 1-parameter flow beats ~2.4x Wilson at $O(a^2)$,
and here even 2.4x is far too little). The win is still 5.16x (up from 4.43x) from annihilation+smoothing.

## Experiment 2 -- $c_1$ FLOW FROM THE BEGINNING (planned next, 2026-09-05)

Motivation: in Experiment 1 the instanton had already relaxed to $\rho \sim 3a$ by $t_0$, PAST the
size where the $1/\rho^3$ shrink force bites. Idea: run the under-improved flow from $\tau = 0$ (skip
the Wilson stage 1), so it acts while the config is rough and the lumps are still near dislocation
scale ($\rho \sim a$, where $\varepsilon a^2/\rho^3$ is largest) -- catch and shrink them BEFORE they
grow. Same $c_1$ scan, checkpoint every $t_0$, probe $\rho(\tau)$, compare to Experiment 1.

Setup: --nstep1 0 (no Wilson pre-flow) + --no_baseline (Landau on the bare config is meaningless) +
stage 2 = $c_1$ from $\tau=0$. DISTINCT log + checkpoint dir so Experiment 1 is preserved:
```
qsub -v CONFIG=/projectnb/qfe/nmatsum/dwf/configs_iwasaki_16_b2.6/ckpoint_lat.640,\
NSTEP1=0,NOBASELINE=1,C1='0 0.0417 0.0833 0.1167',\
TAG=640_c1fromstart,CKPDIR=/projectnb/qfe/nmatsum/dwf/shrinkflow_ckp_fromstart \
grid_shrinkflow_gpu_qsub_claude.sh
# log -> log/shrinkflow_640_c1fromstart_claude.log ; checkpoints -> shrinkflow_ckp_fromstart/
```
Watch: does $\rho_0(\tau)$ DECREASE early (unlike Experiment 1's monotone growth), and does $Q$ drop?
$c_1{=}0$ here = pure Wilson flow from scratch = the standard-flow control.
