# Free-limit DWF preconditioner -- future directions (Nobu's roadmap)

Communication left by Nobu (2026-09-02) on where this project is heading, to be picked up once the
current $\beta$/volume generation + the M1 study land. Two big threads: (1) refine the frame $\Omega$,
(2) the spectrum transform for $M_0$ and $M_1$ + deflatability. Keep this file updated as the plan firms.

Context / status pointers: current results and the M1 wiring live in memory `project-freeprec-r2`; config
generation + the $\beta$ scan in `project-r2-config-gen`; the M1 apply is `FreeLimitPreconditioner1` in
`Grid/Grid/qcd/utils/FreeMobius5D_claude.h`; the spectrum/Arnoldi code was started by a detached agent
(`Grid/Grid/algorithms/iterative/ImplicitlyRestartedArnoldi_claude.h`,
`Grid/tests/solver/spectrum_transform_impl_plan_claude.md`).

---

## Direction 1 -- $\Omega$ refinement (the frame determination)

Once multiple $\beta$/volume ensembles land, systematically study how the frame $\Omega$ is best found.

### 1a. Set the scale with $t_0$ (units for everything)
For each ensemble: run the flow and identify BOTH the topological charge $Q$ AND the flow scale $t_0$.
$t_0$ is defined (Luscher 1006.4518) by
$$
t^2 \langle E(t) \rangle \big|_{t = t_0} = 0.3,
$$
with $E(t)$ the (clover) action density along the flow. **$t_0$ becomes THE unit in which everything is
discussed** -- flow times, step sizes, and the frame-flow time are all quoted as $t/t_0$ (dimensionless),
so results are comparable across $\beta$ and volume at a fixed physical scale. (Also compute/record $w_0$
from $t\,\frac{d}{dt}[t^2\langle E\rangle]|_{w_0^2}=0.3$ if convenient -- BMW 1203.4469 -- as a cross-check
scale.) DONE (2026-09-02): $t_0$ set with NO driver edit -- reconstructed $t^2\langle E\rangle_\text{plaq}
= 36\,t^2(1-\bar P)$ from the `plaq` column already in the flow logs (Grid `WilsonFlow.h:161-169`), ensemble
mean crossing 0.3 (`grid_t0_reconstruct_claude.sh`). Nobu: "Iwasaki flow is fine". Results in memory
`project-r2-config-gen` (b2.6 $t_0/a^2$=2.91; scan 0.76/1.14/1.70). Clover $t_0$ (tighter) would need a
driver edit -- not done, not needed.

### 1b. Scan the frame-flow choices -- find what is optimal
Vary, and measure the resulting preconditioner quality (the $D_W$-apply ratio CGNE / free-prec, i.e. the
headline metric; secondarily the flowed-fixed Landau functional and the low-mode structure from Dir. 2):
- **flow time** $\tau/t_0$ (currently fixed $\tau=2$ Wilson): is there an optimum, and is it universal in
  $t_0$ units across $\beta$/volume?
- **step size** (RK3 eps; currently 0.01 fixed-step) -- integration-error sensitivity of the frame.
- **flow type / smoother**: Wilson, Iwasaki, DBW2, "anti-Iwasaki" (opposite-sign rectangle coeff $c_1$),
  and stout smearing as an alternative to gradient flow. Grid has these gauge actions
  (`WilsonGaugeAction`, `IwasakiGaugeAction`, `DBW2GaugeAction`, plus `Smear_Stout`); the flow-force swap
  idiom already used for Iwasaki-vs-Wilson flow (`setGaugeAction`) generalizes -- $c_1$: Wilson $0$,
  Iwasaki $-0.331$, DBW2 $-1.4088$, anti-Iwasaki $+0.331$ (all with $c_0 + 8 c_1 = 1$ to fix the
  flow-time scale). Question: which smoother gives the best frame (smoothest $U^L$, best win) at fixed
  $\tau/t_0$, and does the ranking depend on $\beta$ (coarse vs fine)?

### 1b-Stage-1 -- FLOW-TIME SCAN (ACTIVE, staged 2026-09-02; Nobu approved)

First stage of 1b: fix the flow TYPE to **Wilson**, scan the frame-flow TIME, find the optimum, and test
whether it is universal in $t_0$ units across $\beta$. $t_0$ was set (memory `project-r2-config-gen`):
16^4 $t_0/a^2$ = 0.759 (b2.13), 1.140 (b2.25), 1.697 (b2.37), 2.91 (b2.6). Our runs SO FAR sat at
$\tau=2$ lattice = $\tau/t_0 = 0.69$ on b2.6 only.

Parameters (Nobu):
- **$s/t_0 \in \{0.4, 0.5, 0.6, ..., 1.2\}$** (9 flow times; $s$ = frame-flow time). Per ensemble the
  LATTICE flow time is $\tau = (s/t_0)\,t_0$, and $\mathrm{nstep} = \tau/\mathrm{eps}$ with eps $=0.02$:
  - b2.13: $\tau$ = 0.30..0.91  (nstep 15..46)
  - b2.25: $\tau$ = 0.46..1.37  (nstep 23..68)
  - b2.37: $\tau$ = 0.68..2.04  (nstep 34..102)
  - b2.6 : $\tau$ = 1.16..3.49  (nstep 58..175)
- Flow = **Wilson** (baseline; other types are later stages of 1b).
- **10 configs per ensemble**. 16^4 ensembles first (b2.13/2.25/2.37 have 34-37 configs; b2.6 has 35).
  24^4 DEFERRED for this scan (only 4-9 configs, and ~5x slower per solve -> a 24^4 config's 9-tau scan
  will not fit 12h).
- Metric = **M0 AND M1** $D_W$-apply ratio (CGNE / FGMRES) vs $s/t_0$ (Nobu chose both). CGNE is
  FRAME-INDEPENDENT -> compute ONCE per config, reuse for all 9 $\tau$ and both ops. **Solve tol 1e-6**
  (Nobu; faster scan -- the RATIO is ~tol-robust, but absolute counts won't match the 1e-8 headline data).

Job structure (Nobu: loop in the WRAPPER, NO array jobs, 12h):
- The wrapper LOOPS over configs and fires a plain per-config `qsub` (10 jobs/ensemble; no `-t` array).
- Each per-config job loads its config, runs CGNE once, then loops the 9 $\tau$: Wilson-flow to $\tau$,
  Landau-fix -> $\Omega$, FGMRES(M0) + FGMRES(M1), print one line per $\tau$ ($s/t_0$, iters, ratios,
  Landau). `#$ -l h_rt=12:00:00`. Est. at 16^4, tol 1e-6, M0+M1: ~9 x ~55 min + CGNE ~= 8-9 h -> fits 12h
  (watch the coarse/off-optimum frames where FGMRES needs more iters; each $\tau$ line is flushed as it
  finishes, so a wall still yields the completed $\tau$ points).

CODE NEEDED (implement after Nobu's OK; mirrors existing CLI):
- Expose the frame-flow time SCAN in `Test_dwf_freeprec_claude.cc`: a `--flow_nsteps <comma list>` (eps
  fixed 0.02) so run_headline computes CGNE once then loops the frame+M0 over the list, printing one line
  per $\tau$ (`s/t0=... tau=... M0 iters=... ratio=... landau=...`). The per-ensemble nstep list is passed
  by the wrapper (it knows $t_0$).
- New wrapper `grid_freeprec_flowscan_wrapper_claude.sh`: per ensemble, look up $t_0$, build the nstep
  list, pick 10 configs (MINTRAJ/SKIP), submit the array job. Logs -> log/, `#$ -m n`, DATDIR per ensemble.
- Output: `ratio(s/t0)` per config -> ensemble mean -> plot ratio vs $s/t_0$, one series per $\beta$;
  identify the optimum and whether it collapses in $t_0$ units.

Open choices to confirm: (i) M0-only (recommended) vs also M1; (ii) solve tol 1e-8 (as now) vs 1e-6 to
speed the scan (the RATIO is ~tol-robust); (iii) exact 10-config selection (MINTRAJ/SKIP).

### 1b-Stage-2 -- FLOW-KERNEL scan (ACTIVE, 2026-09-03; Nobu approved)
At a FIXED $s/t_0$ (the M0-optimum from Stage-1, or 0.4 pending), vary the flow SMOOTHER: round 1 =
**wilson, iwasaki, anti-iwasaki** (baseline / stronger-smoothing / roughening control -- a sign test that
"smoother frame -> better win"). Implemented in the DEDICATED driver Test_dwf_flowscan_claude.cc via
`--frame_flows <comma list>` (swaps the flow force with `setGaugeAction(RBCGaugeAction(Nc,c1))`; $c_1$:
Wilson 0, Symanzik $-1/12$, Iwasaki $-0.331$, DBW2 $-1.4067$, anti-Iwasaki $+0.331$). anti-Iwasaki has
$c_0<0$ -> the flow maximizes the action and can DIVERGE -> GUARDED (skips solves + logs DIVERGED if the
flowed plaquette is NaN/out-of-range). GPU binary built (build_merged, fp32). Submit via the flowscan
wrapper with `FLOWS=wilson,iwasaki,antiiwasaki SOT="<fixed>"`. Metric = same M0/M1 $D_W$ ratio, now vs
flow kernel, per $\beta$; expect Iwasaki $\gtrsim$ Wilson $\gtrsim$ anti-Iwasaki if smoothness is the lever.

### 1b-Stage-3 -- STEP-SIZE scan (NEXT, after the kernel study; Nobu 2026-09-03)
At a fixed $s/t_0$ AND fixed kernel, vary the flow integration step: a COARSE eps and a FINE eps (e.g.
$\{0.04, 0.01\}$ vs the baseline 0.02) -- test the integration-error sensitivity of the frame (does a
coarser/cheaper flow give an as-good frame?). CODE NEEDED: add `--flow_eps` to Test_dwf_flowscan_claude.cc
(currently hardcoded 0.02 in main; run_flowscan already takes flow_eps) + wrapper passes EPS as
`--flow_eps` (the wrapper's nstep = $s/t_0 \cdot t_0/$eps ALREADY uses EPS, so at fixed $s/t_0$ the LATTICE
$\tau$ is held while nstep scales with 1/eps). Then run eps $\in \{0.04, 0.02, 0.01\}$.

### 1c. Direct gradient-descent determination of $\Omega$
Compare the flow+Landau frame against a DIRECT group-valued gradient-descent determination of $\Omega$
(minimize the target functional on the group manifold, no flow). Reference implementation exists in
`qed2/dwf4_qcd_claude` (norm-optimized frame; `dwf4_group_claude.h`, right-invariant derivative /
group-valued GD). Question: does direct GD match or beat flow+Landau, and does it dodge the Gribov-copy
issue (dwf4 found flow+Landau usually beats direct Landau by avoiding Gribov copies)? This ties into the
already-planned norm-optimized $\Omega$ (with and without the $M_1$ correction).

---

## Direction 2 -- spectrum transform ($M_0$ and $M_1$) + deflatability

Quantify HOW the preconditioner moves the Dirac spectrum, for BOTH the leading frame $M_0$ AND the
$D_W$-corrected $M_1$:
- Compute the preconditioned complex spectrum $\mathrm{spec}(M_0 D_{DW})$ and $\mathrm{spec}(M_1 D_{DW})$
  (non-Hermitian -> Implicitly Restarted Arnoldi; the detached agent's IRA solver). $M_0/M_1$ should
  cluster the bulk near $1$; watch where the near-$0$ stragglers ($\sim |Q|$ would-be zero modes) land,
  and whether $M_1$ pulls them in further than $M_0$.
- Also the UNpreconditioned near-$0$ / singular-value spectrum of $D_{DW}$ (Hermitian IRL on $M^\dagger M$)
  as the baseline the transform acts on.
- **Key question: is the residual difficulty in the LOW MODES, and if so is it DEFLATABLE?** If the win
  degrades at high $|Q|$ because a handful of topology-tied low modes are not preconditioned, then
  deflating those few modes (adding them to the frame / a deflation subspace on top of $M_0$ or $M_1$)
  could recover the win. Test: (i) confirm the stragglers are few and $Q$-tied; (ii) build a small
  deflation space from them; (iii) measure the $D_W$-apply ratio with deflation + $M_0$/$M_1$.
  (dwf2 caution, memory `dwf2-*`: low-mode deflation was found REDUNDANT with the free kernel in 2D --
  so this needs an honest measurement here, not an assumption either way.)

---

## Direction 3 -- DOUBLE preconditioning: zMobius (MADWF) x free-limit frame (Nobu 2026-09-03)

Because the target solve is VALENCE (we only want the propagator of the true $D_{DW}$, no sea
determinant), we are free to precondition with a DIFFERENT operator as long as the outer solve targets
the true $D_{DW}$ and a flexible solver (FGMRES) / a MADWF reconstruction makes the answer EXACT. Idea:
stack TWO preconditioners --
  (i) **zMobius / MADWF** (Ls-reduction): approximate the target $L_s$ operator with a cheap reduced-$L_s'$
      zMobius (complex per-slice $b_s,c_s$ tuned to the sign function). Grid HAS this: `ZMobiusFermion`
      (Fermion.h) + `MADWF.h` (Moebius Accelerated DWF -- reduced-$L_s'$ inner solve + exact reconstruction).
  (ii) **free-limit frame** $M_0$/$M_1$ (gauge/spatial): our $\Omega^\dagger F \Omega$.
-> the "double-preconditioned operator": zMobius cuts the $L_s$/chiral cost, the free frame cuts the
gauge/spatial cost. Cost synergy (local agent's profile: $F$ = FFT 61% + dense-$(4L_s)^2$ solve 29%): a
smaller $L_s'$ shrinks BOTH the dense-$L_s$ block and the per-slice work.

TWO composition options (CONFIRM WITH NOBU which he means):
  (a) MADWF OUTER (true Mobius $L_s{=}8$ target, inner cheap zMobius $L_s'\sim4$ solve) with the inner
      zMobius solve PRECONDITIONED by our free-limit $M_0/M_1$.
  (b) Replace the free kernel $F$ INSIDE $M_0$ with a reduced-$L_s'$ FREE-zMobius inverse (cheaper $M_0$
      apply), still outer-solving the true $D_{DW}$.
Nobu (2026-09-03): this is a TRIAL-AND-ERROR decision; leans (b), open to (a). "We can even DECREASE $L_s$
for $M_0$" -- i.e. the reduced-$L_s'$ free kernel is the real lever, and it can be a PLAIN reduced-$L_s'$
Mobius before even going to zMobius. KEY WRINKLE: $M_0$ preconditions $D_{DW}(L_s{=}8)$, so it must map the
$L_s{=}8$ space to itself -- a reduced-$L_s'$ kernel cannot be applied directly; it needs an $L_s$-TRANSFER
(chiral projection to the reduced space + reconstruction, the MADWF $P/P^{-1}$ structure). For the FREE
kernel this transfer is ANALYTIC / FFT-diagonal -> a cheap, self-contained "FREE MADWF" preconditioner
(no inner gauge solve, unlike (a)). Hierarchy to try (all inside $M_0$, valence-exact via flexible outer):
  (b0) reduced-$L_s'$ PLAIN free Mobius $F$ + analytic $L_s$-transfer (simplest);
  (b1) reduced-$L_s'$ free zMobius $F$ (complex $b_s,c_s$, better sign approx at the same $L_s'$).
CODE: (a) wires Grid's MADWF + ZMobiusFermion with $M_0/M_1$ as the inner preconditioner; (b) generalizes
FreeMobius5DInverse to reduced $L_s'$ + the analytic transfer (b0), then per-slice COMPLEX $b_s,c_s$ (b1) +
$\omega_s$ tuning (Zolotarev-Remez / Yin-Mawhinney). Ref: R.C. Brower, H. Neff, K. Orginos (Mobius,
arXiv:1206.5214); zMobius/MADWF (Yin-Mawhinney; McGlynn). PLAN before coding; measure win per variant.

## Cross-cutting: the metric for "optimal"
Whenever we say "optimal" (flow time, flow type, direct-GD, deflation), the primary yardstick is the
honest $D_W$-apply ratio CGNE / free-prec (as in `freeprec_summary_claude.md`), reported in $t_0$ units
and vs $Q$. Secondary diagnostics: flowed-fixed Landau functional, and the low-mode spectrum from Dir. 2.

## Open questions to settle with Nobu before coding each piece
- Dir 1a: is $t_0$ (clover $E$) the scale, or also $w_0$? Which flow's $E$ defines $t_0$ -- always Wilson,
  or the same flow used for the frame?
- Dir 1b: "optimal" judged purely by the $D_W$ ratio, or a combined figure of merit?
- Dir 2: IRA plain-vs-shift-invert for the interior near-$0$ region (the agent's open fork); how many
  low modes to target; deflation subspace built from left+right vectors (non-Hermitian) vs the Hermitian
  $M^\dagger M$ singular vectors.
