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
scale.) ACTION ITEM: extend the flow driver to print $t^2\langle E\rangle(t)$ and solve for $t_0$ (and
report $\sqrt{t_0}$, $Q$ vs $t$) per config; average $t_0$ over the ensemble.

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
