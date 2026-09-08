# $q$-squeeze flow ($\Vert q \Vert_n$ localization) -- implementation plan

**FIRST RESULT (2026-09-06, job on 640, stage1=Wilson to t0, stage2=||q||_n for 10 t0, eps 0.02):**
q-squeeze WORKS in the right direction -- FIRST method to SHRINK not broaden the lump. n=2: rho0
3.193->3.158 (-1.1%), qmax +4.5%, PR 1997->1873 (-6%), plaq FROZEN (bulk untouched), Q_clover
preserved. n=4: rho0 3.193->3.092 (-3.2%), qmax +14% -- higher n shrinks ~3x FASTER (6c overlap
penalty not yet biting at n=4). BUT the shrink is far too small over 10 t0 to help the win: n=2 and
n=4 both 4.36x, vs baseline 4.43x (slightly WORSE -- pure q-flow freezes the bulk so it forgoes the
long-Wilson bulk-smoothing gain that gave 5.16x). Mechanism validated, magnitude ~100x too slow at
eps 0.02 / 10 t0. n=6 pending.

n=6 done: rho0 3.193->2.979 (-6.7%), qmax +32%, PR -21% -- MONOTONIC in n, no turnover through n=6,
all still 4.36x. Higher n strictly shrinks faster (6c overlap penalty not yet biting).

LONG-SQUEEZE FOLLOW-UP LAUNCHED as TWO independent jobs (Nobu 2026-09-06), both eps2=0.4 (20x), stage1
Wilson eps=0.02 (t0 scale; new --flow_eps2 knob), nstep2<=2000 (tau2<=800~275 t0), qstop=0.5, config 640:
 (1) nstep1 SCAN at fixed n=6: grid_shrinkflow_qsqlong_gpu_qsub_claude.sh, NSTEP1=146,730,1460 (1/5/10
     t0), TAG _qsqlong_n1scan. Tests bulk-smoothing(banked by Wilson) vs starting-lump-size tension.
 (2) n SCAN at fixed nstep1=146: grid_shrinkflow_qsqlong_nscan_gpu_qsub_claude.sh, QN=6,8,10,12, TAG
     _qsqlong_nscan. Finds the turnover where force width rho/sqrt(n) < profile (~n>10) = manufactures
     a single-site spike instead of shrinking the instanton. FD check PASSED for n=6,8,10,12 (1e-8).
Both NEED the GPU rebuild first (grid_shrinkflow_build_scc_gpu_claude.sh: flow_eps2 + nstep1-list new).

NEXT (Nobu 2026-09-06), two follow-ups:
1. LONGER re-run: eps *= 20-50 (0.4-1.0) and s_max *= ~100 (tau2 ~ 10^3 t0) so the 1/rho^5 runaway
   actually reaches fall-through. lambda is absorbed into flow time, so this is just "flow more".
   Watch RK3 stability at large eps (q-force is small, ~3e-3, so eps=0.4-1.0 should hold; guard reverts).
2. nstep1 SCAN: vary the Wilson pre-flow time before the squeeze -- long Wilson banks the bulk-
   smoothing win (5.16x channel), q^n squeezes the lump on top. Tension: more Wilson broadens the lump
   (bigger starting rho -> slower squeeze). BUILT: --nstep1 now takes a COMMA LIST (qsqueeze mode);
   outer loop reflows stage 1 per value, per-nstep1 baseline + inner n-scan; tokens/labels carry
   _n1<val> when scanned. CPU-compiled OK 2026-09-06 (NEEDS GPU rebuild before submit).

**STATUS (2026-09-06): Chunks A+B+C DONE.** QSqueezeFlowAction_claude.h written (pure $\Vert q\Vert_n$
action, no Wilson term, $\lambda$ absorbed); **FD check PASSED** (central difference, ratio -> 1 to
$10^{-9}$ at dt=1e-5 for n=2,4,6; one net sign fixed relative to the naive bookkeeping, normalization
$c_0/8$ exact). Driver has `--fdcheck [--fddt]` and `--qsqueeze --qn <list>` modes; qsub has
QSQUEEZE/QN. CPU build (build_mpi_merged) used for the FD check; GPU rebuild + submit = Nobu.

Goal (approved by Nobu 2026-09-06): implement the 6b/6c flow of
`Grid/scripts_nm/freeprec_gaugefix_topology_claude.md` -- gradient flow of

$$
S_\text{flow} = S_W - \lambda\, \Vert q \Vert_n ,
\qquad
\Vert q \Vert_n = \Big( \sum_x |q(x)|^n \Big)^{1/n} ,
$$

whose extra force squeezes the topological lump toward the cutoff at an $O(1)$, $\lambda$-tunable
rate ($d\rho/dt \propto -\lambda/\rho^5$ on the BPST moduli space), with the norm-gradient weight

$$
w_n(x) = \mathrm{sgn}\,q(x) \left( \frac{|q(x)|}{\Vert q \Vert_n} \right)^{n-1} \le 1
$$

(self-normalized: bounded force, no $10^{-3n}$ underflow; $n$ even -> $w_n = q^{n-1}/\Vert q\Vert_n^{n-1}$,
smooth). **Scan $n = 2, 4, 6$** (Nobu; the $L^2$-vs-soft-$L^\infty$ trade-off of Sec. 6c: larger $n$ =
more core-concentrated force, safer bulk, but force support $\sim \rho/\sqrt n$ must still cover the
profile). **Monitor the plaquette AND the $q(x)$ distribution** (lump size/number) along the flow.

## Algorithm sources (mandatory)
- Localization cost + norm analysis: this project, freeprec_gaugefix_topology_claude.md Sec. 6b/6c.
- The $F\tilde F$ (topological-charge) force = imaginary-$\theta$ HMC force: M. D'Elia, F. Negro,
  arXiv:1306.2919 (here with the site-dependent weight $\theta(x) \to \lambda\, w_n(x)$).
- Clover-leaf derivative with insertion: Grid `CloverHelpers::Cmunu` (WilsonCloverHelpers.h:42),
  Eq. (B.39) of Z. Sroczynski's PhD thesis -- the same object used by the Wilson-clover fermion force
  (WilsonCloverFermionImplementation.h:312).
- Clover $q(x)$: density of WilsonLoops::TopologicalCharge (WilsonLoops.h:641).

## Design

New header `Grid/Grid/qcd/utils/QSqueezeFlowAction_claude.h` (header-only, like FreeMobius5D_claude.h):

`template<class Gimpl> class QSqueezeGaugeAction : public Action<typename Gimpl::GaugeField>`
- members: `RealD lambda; int qn;` an internal `WilsonGaugeAction<Gimpl>` (beta = 2 Nc so that
  `deriv` matches the WilsonFlow default force normalization -- CHECK against WilsonFlow.h's default
  SG construction and mimic exactly);
- `S(U)` = Wilson S - lambda*||q||_n (for logging/FD-check);
- `deriv(U, dSdU)`:
  1. Wilson part: internal WilsonGaugeAction::deriv.
  2. Clover field strengths F_{munu} (6 planes, WilsonLoops::FieldStrength) -> q(x) (probe formula)
     -> global ||q||_n -> weight field w_n(x) (LatticeComplex).
  3. For each (mu,nu) pair: insertion Lambda_{munu}(x) = c_q * w_n(x) * Fdual_{munu}(x) where
     Fdual_{munu} = (1/2) eps_{munurhosig} F_{rhosig} (i.e. the PARTNER clover F in the
     tr(F Fdual) contraction; c_q collects 8/(32 pi^2), the 1/8 of FieldStrength, and the factor 2
     from the two F's in q -- fixed by the FD check, not trusted from the derivation).
  4. Force: for each mu, sum over nu != mu of CloverHelpers-style Cmunu(U, Lambda_{munu}, mu, nu)
     terms; combine as in the clover-fermion force (U[mu]*force_mu, Ta() projection so the flow force
     stays algebra-valued); dSdU_mu -= lambda * that (SIGN such that ||q||_n GROWS along the flow --
     verified numerically, flipped if wrong).
  Note Cmunu lives in WilsonCloverHelpers<Impl> (fermion Impl); either instantiate via WilsonImplD or
  copy the ~40-line function into the new header with Gimpl shift primitives (decide at coding time;
  copying avoids dragging fermion Impl types into a gauge action -- preferred, with citation comment).
- diagnostics exposed after each deriv: q_max, ||q||_n, Q_clover, ||Z_W||, ||Z_q|| (force norms).

Plugs into the EXISTING shrinkflow driver via `wf.setGaugeAction(&qsqueeze)` -- WilsonFlow's RK3 uses
only SG->deriv (WilsonFlow.h:253), so the whole Exp-1 harness (chunked stage 2, Q monitor, per-$t_0$
checkpoints, stage 3, Landau + FGMRES(M0) A/B, CGNE-shared) is reused unchanged.

## Chunks

**A. Force header + finite-difference validation.**
Files: Grid/Grid/qcd/utils/QSqueezeFlowAction_claude.h, driver `--fdcheck` mode.
FD check (idiom of tests/forces): random algebra direction dU at scattered sites, compare
(S(U e^{eps dU}) - S(U))/eps vs the deriv contraction, on a small lattice (8^4 or the 640 config).
MUST pass before any flow runs; fixes c_q and the sign.

**B. Driver wiring (Test_dwf_shrinkflow_claude.cc).**
`--qsqueeze` mode: stage 1 Wilson to t0 UNCHANGED (squeeze acts on the clean config where the lump
is at its rho-minimum); stage 2 force = QSqueezeGaugeAction, scanning `--qn <comma list>` (2,4,6) at
fixed `--lambda`; stage 3 + frames + solves unchanged. Per-chunk monitor line gains: Q_clover, q_max
(+ the implied rho0 = (6/(pi^2 q_max))^{1/4}), nsites with |q| > qmax/2 (core volume proxy), nlumps
proxy (sites above 0.5e-3 threshold clustered is offline work -- in-driver keep it to counts), and
the force-ratio ||Z_q||/||Z_W|| (the stability dial). Checkpoints every t0 as before (full lump
distribution offline via Test_instsize_claude + SciDAC dumps when LIME lands).

**C. Build + qsub.**
Rebuild grid_shrinkflow_build_scc_gpu_claude.sh (driver + new header). qsub gains QSQUEEZE=1,
QN='2:4:6' (colon-safe), LAMBDA. Distinct TAG/CKPDIR per run (keep Exp 1/2 logs).

**D. Lambda calibration then the n-scan.**
First submit = CALIBRATION: n=2 only, LAMBDA list (e.g. 0.01 / 0.1 / 1), SHORT stage 2 (nstep2 ~ 300),
read the force-ratio + q_max trend, pick lambda where the squeeze is active (q_max rising) but the
bulk plaq is stable. Then the full n=2,4,6 scan at that lambda, full nstep2, on 640.

## DECISIONS (Nobu 2026-09-06)
1. **$\lambda = 1$** -- for the PURE $\Vert q \Vert_n$ flow the coupling is absorbable into the flow
   time, so no lambda knob at all.
2. **Stage 2 = PURE $\Vert q \Vert_n$ flow** (no Wilson term in the generator; "deal with the
   smearing part later"). Stage 1 Wilson-to-$t_0$ retained as frame prep (on the bare config $q(x)$
   is UV noise -- the squeeze needs the smoothed config), stage 3 short re-smooth retained as gauge-
   fix prep. **Stage-2 flow time = $10\,t_0$** ($\tau_2 \approx 29$, nstep2 $\approx 1455$ at eps
   0.02). NOTE: pure $q$-flow has no UV damping -- watch the plaquette monitor; and its intrinsic
   timescale is unknown a priori (force $\sim$ clover force $\times\, w_n$), so the per-chunk
   force-norm + $q_{max}$ prints are the calibration; adjust EPS if nothing moves / too stiff.
3. **Checkpoint every $t_0$** (as already implemented); lump size/number monitored in-driver via
   cheap $q(x)$ stats each chunk (q_max -> implied rho0, core-site count), full peak-finder offline
   on the checkpoints via Test_instsize_claude.
