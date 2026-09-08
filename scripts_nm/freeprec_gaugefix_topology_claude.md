# Improving $\Omega$ on a flowed $Q \neq 0$ config: gauge fixing vs. pure-gauge extraction

Discussion note (2026-09-05), answering Nobu's question: at $t \sim t_0$ the flowed config is
essentially pure gauge + topological lump(s); Landau fixing is only "correct" for the pure-gauge part,
so the lump degrades the frame $\Omega$. Can we (a) improve the gauge fixing, and/or (b) extract the
pure-gauge dof from such a config?

Sources cited below: Luscher 1006.4518 (Wilson flow, $t_0$, flow with gauge-fixing/damping term);
BPST instanton regular vs. singular gauge (standard, e.g. Vandoren-van Nieuwenhuizen 0802.1862);
Atiyah-Singer index / would-be zero modes at finite $m$.

---

## 0. Sharpen the premise: what is fixable and what is not

A frame with $U^L_\mu = \Omega\, U_\mu\, \Omega^\dagger \approx 1$ everywhere exists iff $U$ is flat
(zero field strength, trivial holonomy). $Q \neq 0$ is the exact obstruction: no gauge transform
removes genuine curvature. So the question splits into two channels with very different headroom:

1. **The lump core.** There the curvature is physical. No choice of $\Omega$ makes $U^L \approx 1$
   there, and — independently — the index theorem puts $|Q|$ would-be zero modes at
   $\sigma \sim O(m)$ (we SAW them: 3 chiral surface modes at $\sigma \approx 0.123 \approx m$ on
   traj 640). These modes are gauge-invariant. $M_0$ leaves them as the $\sim 12$ stragglers at
   $\mathrm{Re}\,\lambda \sim 0.3$ in the measured $\mathrm{spec}(M_0 D_{DW})$. **No gauge-fixing
   improvement can touch this floor.** The only cures are the $M_1$ correction term or deflation of
   those few modes.

2. **The bulk / lump tail.** Here $U$ *is* near pure gauge, and the question is whether the Landau
   fixer spends the gauge freedom well — or whether accommodating the lump makes it compromise the
   bulk (a delocalised frame residual). This is where better gauge fixing has headroom.

Our own R2 data already hints the split matters: the flowed-fixed Landau functional is ~constant
($0.014$-$0.016$) across all $Q$ and does NOT track the win — i.e. the *global* frame-quality number
is blind to the degradation channel. Either the degradation is channel-1 (physical, spectral —
functional rightly blind) or it is channel-2 but hiding in the *spatial distribution* of the residual
rather than its volume average. Section 3 gives the one-plot diagnostic that decides this.

---

## 1. Why the lump can poison the frame globally: the Gribov / gauge-copy mechanism

For a single continuum instanton, both standard representatives are transverse
($\partial_\mu A_\mu = 0$), i.e. both are Landau-gauge stationary points, but they are very
different frames:

- **Regular gauge:**

$$
A_\mu^a(x) = \frac{2\,\eta^a_{\mu\nu}\, x_\nu}{x^2 + \rho^2},
\qquad |A| \sim 1/x \ \text{at large } x .
$$

- **Singular gauge:**

$$
A_\mu^a(x) = \frac{2\,\rho^2\, \bar\eta^a_{\mu\nu}\, x_\nu}{x^2\,(x^2 + \rho^2)},
\qquad |A| \sim \rho^2/x^3 \ \text{at large } x .
$$

The physics (curvature, $q(x)$) is identical; the *frame quality away from the core* is not: the
singular-gauge copy is localised ($\Vert A \Vert^2$ integrable), the regular-gauge copy pollutes the
whole volume with a slowly-decaying $1/x$ tail. On the lattice these are distinct Gribov copies of
the same Landau condition. A local relaxation algorithm (our Fourier-accelerated fixer) can land on
either basin — and on a *regular-like* copy the free-limit mismatch $U^L - 1$ is spread over the
bulk, degrading $M_0$ everywhere, not just at the core.

Note the volume-averaged functional barely distinguishes them: the regular-gauge tail contributes
$\sim \rho^2 / L^2$ per unit volume — small, consistent with our "functional ~constant" observation.
So the functional is the wrong meter for exactly this failure mode.

**Consequences for improving the fixing itself (option a):**

- **a1. Copy search.** On hard configs, restart the Landau fixing from several random gauge starts
  (and/or overrelaxation + simulated annealing), keep the copy with the smallest functional — but
  *judge* copies by the localisation diagnostic of Sec. 3 (and ultimately the $D_W$-win), not the
  functional alone. Cheap test: one hard config (640 or 80), ~8 random starts, look at the spread of
  functional AND win.
- **a2. Gauge-fix along the flow instead of at the endpoint.** Luscher's flow with a gauge
  damping term (1006.4518, the $\alpha_0\, \partial_\mu B_\mu$ term added to the flow generator)
  drives the configuration continuously toward Landau gauge *while it smooths*. Starting the frame
  from the (already near-trivial) early-flow config and transporting it continuously avoids the
  copy-hopping a cold endpoint fixing can do, and should systematically select the "smooth" copy.
  Implementation in Grid: add the damping term to the WilsonFlow generator on a copy, or
  equivalently accumulate $\Omega_t$ by Landau-fixing at a few intermediate flow times, each warm-
  started from the previous $\Omega$ (a poor-man's version, no new flow code needed: just reuse the
  previous xform as the starting gauge).
- **a3. Loss-function change (norm-optimised frame — roadmap Dir 1c).** Landau minimises the
  uniform-weight $\sum_x \Vert A \Vert^2$. The quantity that actually controls $M_0$ is the
  free-propagator-weighted mismatch,

$$
\text{loss}(\Omega) \;=\; \big\Vert\, \big( D_{DW}[U^L] - D_{\text{free}} \big)\, F \,\big\Vert^2 ,
$$

  estimated stochastically ($v \mapsto \Vert M_0 D_{DW} v - v \Vert^2$ over a few Gaussian $v$), and
  minimised over the group manifold by the dwf4-style right-invariant gradient descent
  (`qed2/dwf4_qcd_claude/dwf4_group_claude.h`). Because nothing can be gained at the lump core, the
  free-propagator weight automatically *deweights* the core and spends the gauge freedom in the bulk
  — exactly the reallocation Landau refuses to make. This is the principled version of "improve the
  gauge fixing"; it is also the most expensive option.

---

## 2. Extracting the pure-gauge dof (option b): factor out the lump

Goal: write $U = U_{\text{top}} \cdot V$ with $V$ in the trivial sector, Landau-fix $V$ (clean, no
obstruction) to get $\Omega$, and let the localised $U_{\text{top}}$ be handled by $M_1$/deflation.

- **b1. Deep-flow quotient (recommended first try — no ansatz fitting).** Flow a COPY far past
  $t_0$ (e.g. $s/t_0 = 3$-$4$, staying inside the measured $Q$-plateau so no dislocation falls
  through): perturbative dressing dies, the config approaches the multi-instanton solution
  $U^{\text{deep}} \approx U_{\text{top}}$. Then define the link-ratio field

$$
V_\mu(x) \;=\; U^{\text{deep}\,\dagger}_\mu(x)\; U^{(t_0)}_\mu(x) ,
$$

  which is small everywhere (two smooth configs differing by the not-yet-decayed dressing) and
  carries no lump, hence trivial-sector in practice. Landau-fix $V$ to get $\Omega$. The flow itself
  performs the topological/pure-gauge separation; cost = one longer flow per config (cheap at
  $16^4$, eps can be the rough 0.1 per the step-size result).
  *Caveat:* the factorisation is only link-wise, not an exact non-abelian gauge decomposition
  ($U \neq U_{\text{top}} \cdot V$ as bundles); its value is empirical — does the $V$-frame beat the
  direct Landau frame on hard configs? One-config A/B is a day of compute.
- **b2. Instanton-ansatz subtraction.** Locate lumps from the flowed clover charge density $q(x)$
  (peak position; size from the BPST core value $q(0) = 6/(\pi^2 \rho^4)$; colour orientation is the
  hard part), build lattice BPST links, quotient as in b1. Strictly more work than b1 for the same
  end; only worth it if b1's residual dressing turns out to matter. The `visualisation/` tools can
  render $q(x)$ overlays for the fitting sanity check.
- **b3. 2D prototype (exact).** In the qed2/dwf4 U(1) testbed the decomposition IS exact (Hodge:
  $A = A_{\text{top}} + \partial \lambda$ + harmonic, uniform-flux part carries $Q$). Prototype
  "frame from the $Q$-stripped part + $M_1$/deflation for the rest" there, where every step is
  controlled, before deciding how hard to push the SU(3) version.

**What b1/b2 buy and what they do not:** the framed config $U^L$ still contains the lump — $M_0$'s
free kernel never matches it. The gain is confined to channel 2: the frame no longer compromises the
bulk. The lump itself must still be eaten by:

- **$M_1$** — its correction $F\, D(\tilde A)\, F$ becomes *localised* when the frame residual is
  localised, which is exactly the regime where a first-order correction is accurate. Consistent with
  the data: $M_1$ already lifts precisely the high-$|Q|$ configs (80: 2.47 -> 3.37x, 120: 2.32 ->
  3.22x). A lump-localised frame should improve $M_1$ further.
- **Deflation of the $\sim |Q|$ stragglers** on the FGMRES side (the m0devals cluster at
  $\mathrm{Re} \sim 0.3$ is few-dimensional and well-separated — an ideal deflation space; the
  earlier abandoned attempt deflated the WRONG operator, bare $D_{DW}$ under CGNE).

---

## 3. The one-plot diagnostic that decides where to invest

Before building any of the above, measure WHERE the frame residual lives. Per site,

$$
r(x) \;=\; 1 - \frac{1}{4 N_c} \sum_\mu \mathrm{Re}\, \mathrm{tr}\, U^L_\mu(x) ,
$$

on one hard config (640 or 80) at the current default frame ($s/t_0 = 1.0$), overlaid with the
flowed $q(x)$ (or just its lump positions). Two outcomes:

- **$r(x)$ localised at the lumps** -> the Landau frame is already bulk-optimal; channel 2 is empty;
  gauge-fixing improvements (a1-a3, b1-b2) have little headroom. Invest in deflation + $M_1$
  (channel 1). 
- **$r(x)$ smeared over the volume** (regular-gauge-like copy) -> the fixer IS being poisoned by the
  lump; a1 (copy search) and b1 (deep-flow quotient) are cheap and directly targeted; a3 is the
  principled endgame.

This is a ~30-line addition to the flowscan/freeprec driver (dump $r(x)$ site-wise to a .dat on the
boss node, plus its lump-window vs. bulk averages as two scalars per config), or even just the two
scalar averages if we want a number before a picture.

Also connected: the pending decisive re-check (configs 80/120 re-run at the new $s/t_0 = 1.0$ frame)
measures the same split at the win level — if their low win lifts, frame fragility (channel 2 was
real); if it stays, genuine topological floor (channel 1). Run it first; it is already queued.

---

## 4. Shrink-flow: engineering a flow that quenches the instanton radius (Nobu 2026-09-05)

Question: numerical evidence says the flow stops improving $\Omega$ beyond $t \sim t_0$ (perturbative
dof are dead by then). If a flow could also SHRINK the instanton radius $\rho$ toward a point, the
config would approach pure gauge + point defect = the ideal frame config. Known framework?

**Yes: (over/under-)improved cooling / smearing / flow.** The key fact: in the continuum, $\rho$ is
an exact modulus (scale invariance) — a flat direction of $S$ — so any continuum-exact flow leaves
$\rho$ marginal. What moves $\rho$ is the $O(a^2)$ Symanzik artifact of the chosen flow action, and
its SIGN AND SIZE ARE A TUNABLE KNOB. This is the classic instanton-stability engineering of
Garcia Perez–Gonzalez-Arroyo–Snippe–van Baal (hep-lat/9309009), de Forcrand–Garcia Perez–Stamatescu
"improved cooling" (hep-lat/9701012), and Moran–Leinweber "over-improved stout" (arXiv:0801.1165);
cooling/flow equivalence: Bonati–D'Elia (arXiv:1401.2441).

Moran–Leinweber parametrisation of the plaquette+rectangle family (our `RBCGaugeAction(Nc, c1)`),
with $c_0 + 8 c_1 = 1$:

$$
c_0 = \frac{5 - 2\varepsilon}{3},
\qquad
c_1 = -\frac{1-\varepsilon}{12}
\quad\Longleftrightarrow\quad
\varepsilon = 1 + 12\, c_1 .
$$

Checkpoints: Wilson $c_1{=}0 \Rightarrow \varepsilon{=}1$; tree Symanzik $c_1{=}-1/12 \Rightarrow
\varepsilon{=}0$; over-improved $\varepsilon{=}-1$; Iwasaki $c_1{=}-0.331 \Rightarrow \varepsilon
\approx -3.0$; DBW2 $\Rightarrow \varepsilon \approx -15.9$; our anti-Iwasaki $c_1{=}+0.331
\Rightarrow \varepsilon \approx +5.0$. The lattice action of an instanton of size $\rho$:

$$
S(\rho) = S_0 \left[ 1 - \frac{\varepsilon}{5} \frac{a^2}{\rho^2} + O(a^4) \right],
$$

so gradient flow acting on the modulus gives

$$
\frac{d\rho}{dt} \;\propto\; -\frac{\partial S}{\partial \rho}
\;\propto\; -\,\varepsilon\, \frac{a^2}{\rho^3} .
$$

- $\varepsilon > 0$ (Wilson and beyond): instantons SHRINK, accelerating as $\rho \to a$, and fall
  through the lattice ($Q$ jumps, config becomes trivial-sector). This is exactly the mechanism we
  already watched on b2.13/140 (Wilson-flow $Q$: 3.5 -> 2 -> 0).
- $\varepsilon < 0$ (Iwasaki, DBW2): instantons GROW / are stabilised — which is precisely why we
  chose Iwasaki flow for the $Q$-measurement plateaus. For the FRAME flow it is the wrong sign of
  the knob.

**Stability window.** The same $\varepsilon$ that shrinks instantons de-improves the UV smoothing:
$c_0 > 0$ requires $\varepsilon < 5/2$, i.e. $c_1 < 1/8$. Our anti-Iwasaki ($\varepsilon \approx 5$,
$c_0 \approx -1.65$) is far outside — hence the observed divergence (the guard in
`Test_dwf_flowscan_claude.cc` was catching exactly this). The maximal-stable shrink flow inside the
one-parameter family is $\varepsilon \to 5/2^-$; practical picks:

$$
\varepsilon = 2 \Rightarrow c_1 = +\tfrac{1}{12} \approx +0.083,\ c_0 = \tfrac13 ;
\qquad
\varepsilon = 2.4 \Rightarrow c_1 \approx +0.117,\ c_0 \approx 0.067\ (\text{marginal}) .
$$

i.e. ~2-2.4x the Wilson shrink rate with a still-positive plaquette term. ($c_0>0$ is necessary,
not proven sufficient — the empirical divergence guard stays.)

**Proposed schedule (two-stage frame flow):**
1. Wilson (or current) flow to $t_0$ — kills perturbative dof; $\Omega$ quality saturates here per
   the numerical evidence.
2. Switch the flow force to $\varepsilon \approx 2$-$2.4$ ($c_1 = +1/12$...) and continue: bulk
   stays smooth (it is already near-flat, and the force is action-decreasing), lumps shrink at the
   artifact-driven rate; optionally run until the frame-copy $Q$ drops through the lumps
   (fall-through = the obstruction leaves the config entirely), then
3. a short Wilson re-smooth to erase the fall-through ripple, then Landau fix.
Because the $Q$-measurement runs on its OWN copy and the solve is on the ORIGINAL config, losing $Q$
on the frame copy is not a bug — it is the point. The frame becomes trivial-sector pure gauge,
Landau fixing is unobstructed, and the entire lump mismatch moves to channel 1 (zero modes —
deferred; $M_1$/deflation).

**Honest caveats:**
- The shrink force is an $O(a^2)$ artifact: $t_{\text{shrink}} \sim \rho_0^4 / (\varepsilon a^2)$
  up to an unknown $O(1)$ constant. Fat lumps on the fine b2.6 lattice may need $t \gg t_0$ (their
  Iwasaki-$Q$ robustness to $\tau \sim 4$ is this slowness seen from the $\varepsilon < 0$ side).
  The flow itself is cheap (rough eps 0.1 is validated); the per-config cost is still Landau-fix
  dominated, and Landau on a smoother, more trivial config should converge no worse.
- Multi-lump configs ($|Q|$ up to 8): each lump shrinks at its own $\rho$-dependent rate — smallest
  die first; a long-enough stage 2 clears them sequentially. Watch for slow stragglers (one fat
  lump can dominate the time).
- $\varepsilon$ in $(1, 5/2)$ only buys ~2.4x over plain Wilson; if that is too slow, the framework
  answer is that NO local one-parameter flow does better at $O(a^2)$ — a genuinely $O(1)$ shrink
  needs explicit scale-breaking in the flow functional (no established lattice framework; the
  moduli-space reason above is why).
- Implementation is ~already in place: `--frame_flows` swaps the flow force via
  `setGaugeAction(RBCGaugeAction(Nc, c1))`; needed additions are (i) an arbitrary-$c_1$ /
  $\varepsilon$ value (not just the named set), (ii) a two-stage schedule (switch action at a given
  $\tau$), (iii) the existing divergence guard + a $Q$-monitor on the frame copy.

---

## 6. $q(x)$-guided flow: use the charge density itself to localize the lump (Nobu 2026-09-06)

Context: Experiment 1 (freeprec_shrinkflow_results_claude.md) showed the $c_1$ artifact knob is
hopeless ($0.1\%$ effects) and that plain flow BROADENS the surviving lump past $t_0$ while the bulk
keeps improving. Nobu's proposal: we can measure $q(x)$ cheaply -- feed it back into the flow. Two
concrete realizations, in increasing ambition.

### 6a. Masked (q-weighted) flow -- freeze the lump, smooth the bulk

Multiply the flow force by a gauge-invariant local weight built from the charge density:

$$
\partial_t U_\mu(x) = -\,w(x)\; \{ \partial S_W[U] \}_{x,\mu}\, U_\mu(x),
\qquad
w(x) = \frac{1}{1 + |q(x)|/q_\text{ref}} ,
$$

with $q_\text{ref}$ set between the bulk noise level and the lump core value (Exp 1 numbers: core
$|q| \sim 3$-$6 \times 10^{-3}$, bulk after $t_0$ orders below -- a wide window, e.g.
$q_\text{ref} = 5\times10^{-4}$). Where $q(x)$ is large the flow force is suppressed (the lump is
FROZEN at its current size); in the bulk $w \approx 1$ and full Wilson smoothing proceeds.

- **Why it is aimed right:** Exp 1 measured that (i) the smallest lump size occurs AT $t_0$
  ($\rho_0 = 3.19a$) and only broadens after, and (ii) the win keeps growing with bulk
  smoothing/annihilation ($4.43 \to 5.16$x). The masked flow decouples the two: hold the lump at its
  $t_0$ minimum while flowing the bulk arbitrarily long. It does not shrink below $\rho(t_0)$, but it
  stops paying the broadening cost.
- **Legality:** $q(x)$ is gauge invariant, so $w(x)$ is a scalar field and $w \cdot Z$ transforms
  covariantly -- the flow stays gauge covariant and smooth (hence $Q$-preserving). It is NOT a
  gradient flow of any action (the weighted force is not integrable); that is fine -- nothing we use
  requires a variational structure.
- **Implementation (cheap):** cannot use WilsonFlow's internal stepper (fixed force), but our driver
  already chunks the flow; replace the chunk by an explicit RK3/Euler step: compute the Wilson force
  $Z$ (SG.deriv), compute clover $q(x)$ (probe code, GPU-ready), scale $Z \to w Z$ sitewise, exponentiate.
  ~50-100 lines in Test_dwf_shrinkflow_claude.cc; $w$ can be refreshed every chunk rather than every
  step (lump moves slowly).
- Variant: binary mask from the probe's lump balls (freeze inside $2\rho$, flow outside) -- simpler
  conceptually, but the smooth $w(q)$ needs no lump-finding in the loop.

### 6b. $q^2$-squeeze flow -- genuine $O(1)$ shrink force

The moduli-space argument (Sec. 4) says an $O(1)$ shrink rate needs explicit scale-breaking in the
flow functional. A LINEAR $q$ term is useless: $\int q = Q$ is topological, (quasi-)zero force. The
leading useful term is quadratic:

$$
S_\text{flow} = S_W - \lambda \int d^4x\; q(x)^2 .
$$

For a BPST instanton $\int q^2 \propto 1/\rho^4$, so

$$
S_\text{eff}(\rho) = 8\pi^2/g^2 - \frac{\lambda\, c}{\rho^4}
\quad\Rightarrow\quad
\frac{d\rho}{dt} \propto -\,\frac{4 \lambda c}{\rho^5} :
$$

an unsuppressed, $\lambda$-tunable squeeze that RUNS AWAY once started -- which is the point: the
lump localizes to the cutoff and falls through ($Q$ drops on the frame copy; the solve still targets
the original). Stability structure is favourable: at small fields $S_W \sim F^2$ dominates the
$q^2 \sim F^4$ term, so the bulk is protected; only above a $\lambda$-dependent amplitude threshold
(the lump core) does the squeeze term win. Risks: $\lambda$ too large -> bulk $I\bar{I}$ pair
nucleation from noise (the $-q^2$ term rewards charge fluctuations once they cross threshold);
integrator stiffness as the core sharpens (needs adaptive eps or small eps near fall-through).
- **Force:** $\delta \int q^2 / \delta U$ = the clover-leaf force of $F\tilde F$ weighted locally by
  $2 q(x)$ -- the same object as the imaginary-$\theta$ HMC force (d'Elia, Negro arXiv:1306.2919;
  Panagopoulos-Vicari line of work), with $\theta(x) = 2\lambda q(x)$. Not in upstream Grid; ~200
  lines following the clover field-strength code (WilsonLoops::FieldStrength).
- Schedule idea: Wilson to $t_0$ (annihilate pairs, reach $\rho_{\min}$), then switch on $-\lambda q^2$
  with $\lambda$ ramped until the survivors fall through, then short Wilson re-smooth, Landau fix.
  I.e. Exp 1's schedule with a force that actually bites.

### 6c. Which norm is the right localization cost? ($L^2$ vs $L^\infty$, Nobu 2026-09-06)

Nobu's question: why should the $L^2$ cost $-\lambda \int q^2$ localize at all -- isn't the honest
localization objective the $L^\infty$ norm $\max_x |q(x)|$, regularized as $L^n$ with large $n$?

**Why any $p>1$ works on the moduli space.** For a BPST lump $q(x) = \rho^{-4} f(x/\rho)$ at fixed
$Q = \int q = 1$:

$$
\Vert q \Vert_p^p = \int |q|^p\, d^4x \;\propto\; \rho^{4(1-p)} = \rho^{-4(p-1)} ,
\qquad
\Vert q \Vert_\infty = q_\text{max} \;\propto\; \rho^{-4} .
$$

Every $p>1$ breaks the scale invariance in the shrink direction ($p=1$ is $Q$ itself: no force).
$p=2$ is simply the minimal even power; amusingly its moduli potential $\rho^{-4}$ has the SAME
$\rho$-scaling as the $L^\infty$ one. So the choice of norm does not change the collective shrink
scaling -- it changes WHERE in field space the force acts and what else it rewards.

**What large $n$ buys (Nobu is right).** The flow force density of $-\lambda \Vert q \Vert_n$ is the
clover $F\tilde F$ force times the local weight

$$
w_n(x) = \mathrm{sgn}\,q(x)\, \left( \frac{|q(x)|}{\Vert q \Vert_n} \right)^{n-1} \le 1 ,
$$

(using the NORM, not its $n$-th power, so the weight is self-normalized and bounded -- this also
kills the $q_\text{max}^{\,n-1}$ amplitude runaway and the $10^{-3n}$ underflow). For $n=2$,
$w \propto q(x)$: the force is spread over everything, including bulk fluctuations -- the pair-
nucleation risk. For large $n$, $w_n$ is a soft-max: exponentially concentrated at the lump core,
NEGLIGIBLE in the bulk. Large $n$ is therefore the cure for the $L^2$ cost's main defect -- it makes
the squeeze surgical.

**Why $n$ must stay finite and moderate (the real trade-off).** The support of $w_n$ has width
$\sim \rho/\sqrt{n}$ (for $f(u) = (1+u^2)^{-4}$, $f^{\,n-1}$ falls to $1/e$ at $u \sim 1/\sqrt{n}$).
The deformation we want to excite is the $\rho$-modulus -- a COLLECTIVE mode with support over the
whole profile. The overlap of the force with that mode degrades as $n$ grows: at very large $n$ the
flow just spikes the few sites at the peak -- it manufactures a dislocation ON TOP of a broad
shoulder instead of adiabatically shrinking the instanton (an $L^\infty$ gradient is argmax-supported
= a single-site force, the degenerate limit of this). So:

- $n$ LARGE enough that $w_n$ is bulk-safe (no nucleation reward),
- $n$ SMALL enough that the force width $\rho/\sqrt{n}$ still covers the profile ($\sqrt{n} \ll
  \rho/a$; with $\rho \sim 3a$ that caps $n \lesssim 10$).

Sweet spot expectation: $n \sim 4$-$8$, possibly scheduled (raise $n$ as the bulk cleans up). $n$
should be a SCAN KNOB, not a fixed choice; even $n$ (or $|q|$ with sign factor) handles anti-lumps.

**Unified implementation:** one code path -- the clover $dq/dU$ force kernel times the local weight
$w_n(x)$; $n=2$ recovers the $L^2$ flow, large $n$ approaches the regularized $L^\infty$ flow. The
$n$-scan then measures the overlap-vs-safety trade-off directly (shrink rate of $\rho_0(\tau)$ from
the probe vs bulk plaq/pair count).

### Recommendation
6a first: it is safe, ~an afternoon of code, reuses the whole Exp-1 harness (checkpoints, probe,
baseline A/B on 640), and directly converts Exp 1's measured mechanism into a frame gain (bulk
smoothing without lump broadening). 6b second if 6a's frozen-lump frame shows the lump tail still
limits the win -- 6b is the only known-physics route to an actual point-like lump, at the cost of a
new force term and a stability study; per 6c, implement it as the $\Vert q \Vert_n$ family with $n$
a scan knob ($n=2$ = the original $L^2$; $n \sim 4$-$8$ expected optimal). Both are one-config (640)
experiments first.

---

## 7. The HOLONOMY (Polyakov-loop) obstruction -- likely what the ~0.38 M0-resid floor IS (Nobu/Taku 2026-09-07)

Nobu's framing of the maximal-tree residual: the Polyakov loop is a global obstruction that flow
cannot remove. This is exactly right and, combined with the Q=0-vs-Q!=0 data, points to a NEW and
better explanation of the win floor than "generic near-zero Dirac modes".

**The physics.** Trivialising a gauge field to $U^L_\mu(x)=1$ everywhere has TWO obstructions:
1. LOCAL curvature $F_{\mu\nu}\neq0$ (removed by flow, except the topological lump -- Secs. 1-6);
2. GLOBAL holonomy: even a FLAT connection ($F=0$) on a torus is NOT gauge-trivial. It is classified
   by the Wilson lines wrapping the $N_d$ non-contractible cycles -- the Polyakov loops
   $$
   P_\mu(x_\perp) = \mathcal{P}\exp\!\oint dx_\mu\, A_\mu ,
   $$
   which are gauge invariant (up to conjugation) and CANNOT be removed by any gauge transform.

**Flow DOES change the holonomy (correction, Nobu 2026-09-07).** Earlier I wrongly called the holonomy a
flow-protected flat direction. It is NOT: the Polyakov loop is a continuous, non-topological observable,
and Wilson flow moves it -- $|P|$ grows / smooths under flow (standard; used in P-loop renormalisation).
A flat connection IS a flow fixed point ($F=0 \Rightarrow$ zero staple force), but real configs only
approach flat asymptotically and the holonomy relaxes continuously along the way; it is not frozen at
its $t=0$ value. What matters for the frame is simply that the config WE gauge-fix (at flow time $t$)
carries a definite, generally non-trivial holonomy that $F$ (trivial-holonomy) ignores. Because flow
moves $P$ toward less-disordered (larger $|P|$), this predicts the win should RISE with flow time and
then plateau (as $P$ saturates below 1 with residual spatial variation) -- which is exactly what the
data show (747: 4.0->4.9x over 1-6 $t_0$; 640 baseline 4.4->5.3x over 1-5 $t_0$). So the holonomy
mechanism explains BOTH the floor and the flow-time dependence.

**Maximal tree makes it explicit.** The tree trivialises every link EXCEPT the $N_d$ Polyakov loops;
the off-tree residual IS the holonomy. So the maximal-tree frame is the cleanest probe of this piece.

**Why it likely IS the M0 floor.** The free kernel $F$ is built for $U=1$ (trivial holonomy, up to the
fixed AP-time BC). If the flowed config's holonomy differs, then even a PERFECT frame gives
$U^L=$ (flat + holonomy) $\neq 1$, so $M_0=\Omega^\dagger F\Omega$ mismatches $D_{DW}$ by the holonomy --
a floor NO $\Omega$ can beat. Key consistency check with the data: the ~0.38 M0-resid floor is the SAME
on $Q=0$ (747) and $Q=-3$ (640). If the floor were the instanton zero modes it would DIFFER with $Q$;
it does not. A generic holonomy is present at BOTH $Q$ -> holonomy fits the data where topology does not.

**Classification of F=0 configs mod gauge (Nobu's question, 2026-09-07).** Flat connections on $T^4$
are classified by the holonomy homomorphism $\pi_1(T^4)=\mathbb{Z}^4 \to G$ = 4 COMMUTING group
elements (the Polyakov loops $P_\mu$) modulo simultaneous conjugation. For $SU(N)$ these diagonalise
into the Cartan torus, so the moduli space is
$$
\mathcal{M}_\text{flat} = (T^{N-1})^4 / \text{Weyl},
\qquad \dim_\mathbb{R} = 4(N-1)\ \ (=8\ \text{for }SU(3)).
$$
It is a CONTINUUM, not a finite set of sectors; $U=1$ is one point (trivial holonomy). CONFIRMED
numerically (Test_holonomy_check_claude.cc): a constant diagonal $SU(3)$ config has avgPlaquette=1
(F=0) with Polyakov loops $|P_\mu|$=0.59/0.94/0.48/0.40, UNCHANGED by Landau fixing -> not gauge-
connected to $U=1$. Notes: (i) gauge-connected != moduli-connected -- the constant config is not in
the gauge orbit of $U=1$, but IS reachable by a non-gauge path through $\mathcal{M}_\text{flat}$
(dial holonomy $\to 0$); (ii) holonomy ($\pi_1$) is ORTHOGONAL to the instanton charge $Q$ ($\pi_3$)
-- a config carries both; the $Q$-independent M0 floor is the holonomy piece; (iii) 't Hooft twisted
BC would add discrete components $n_{\mu\nu}\in\mathbb{Z}_N$, irrelevant for our periodic/untwisted BC.
The free kernel $F$ sits at the ORIGIN ($U=1$) of $\mathcal{M}_\text{flat}$; $\Omega$ moves within a
gauge orbit but cannot change the moduli coordinate, so $M_0$ can only match $D_{DW}$ if the frame's
holonomy is already ~origin. That is the floor.

**The fix (concrete, FFT-compatible): holonomy-matched free kernel.** A CONSTANT holonomy background is
translation invariant, so a global gauge rotation to the Cartan turns it into $N_c$ abelian phases per
direction -> the free Dirac/Mobius operator with that background is STILL FFT-diagonal (momenta shifted
$p_\mu \to p_\mu + a_\mu$, i.e. twisted BCs). So:
1. measure the flowed-frame config's spatially-averaged Polyakov loop per direction, diagonalise to
   phases $a_\mu^{(c)}$ ($c=1..N_c$, $\sum_c a=0$);
2. build $F$ (FreeMobius5DInverse) with those twisted BCs instead of the plain AP-time/periodic ones;
3. $M_0=\Omega^\dagger F_{\text{twist}}\Omega$ -- same cost (still one FFT + dense-$L_s$ solve), now
   matching the flat+holonomy background the frame actually reaches.
This captures the ZERO-MODE of the holonomy field (the spatial average); the residual spatial variation
of $P_\mu(x_\perp)$ remains, but the average is the dominant piece. If holonomy is the floor, this
should drop M0-resid and lift the win on BOTH 640 and 747.

**Closed Fourier form (Nobu's question, 2026-09-07).** For a CONSTANT holonomy $U_\mu=V_\mu=
\mathrm{diag}(e^{i\theta_\mu^{(c)}})$ (average, Cartan-rotated), the free operator stays FFT-diagonal at
TWISTED momenta -- each colour's momentum is shifted:
$$
\tilde D_\text{free}^{[\text{hol}]}(p)\big|_c = \tilde D_\text{free}(p+\theta^{(c)}),
\qquad
\sin p_\mu \to \sin(p_\mu+\theta_\mu^{(c)}),\ \ (1-\cos p_\mu)\to(1-\cos(p_\mu+\theta_\mu^{(c)})),
$$
with $p_\mu=(2\pi n_\mu+\varphi_\mu)/L_\mu$; the holonomy is an extra per-colour twist on the BC phase
$\varphi_\mu$. Same FFT, same $(4L_s)^2$ block solve, same cost -- only the per-colour momentum in each
block changes (a small edit to FreeMobius5DInverse: 3 colour phase-shifts instead of 1 BC phase). The
$L_s$/Mobius structure is untouched (no gauge link in the 5th dim). EXACT only for constant, commuting
$V_\mu$: (i) real $P_\mu(x_\perp)$ varies -> the form captures its spatial AVERAGE (zero mode), residual
variation is not FFT-diagonal; (ii) averaged $V_\mu$ generically don't commute -> simultaneous Cartan
diagonalisation is approximate, so match the TEMPORAL holonomy exactly (tied to the AP-BC + low modes)
and treat the rest perturbatively.

**Diagonalising the Polyakov loop vs. gauge fixing -- how it acts on fermions (Nobu's question, 2026-09-07).**
A gauge transform $g(x)$ acts on the fermion as $\psi(x)\to g(x)\psi(x)$ and conjugates the Polyakov
loop at its base point, $P(x_\perp)\to g(x_\perp,0)\,P(x_\perp)\,g(x_\perp,0)^\dagger$. So "diagonalise
$P$" is just an ADDITIONAL gauge rotation $\Omega_\text{diag}$ that rotates the holonomy into the Cartan
-- an ordinary colour rotation on $\psi$, no new structure. It is needed because the twisted free
kernel is colour-diagonal ONLY in the basis where the holonomy is diagonal: $F_\text{twist}$ shifts
colour $c$'s momentum by $\theta^{(c)}$, which matches the config only in $P$'s eigenbasis. So the frame
is $\Omega = \Omega_\text{diag}\,\Omega_\text{Landau}$ and $M_0=\Omega^\dagger F_\text{twist}\Omega$; on
a fermion it reads: rotate colour ($\Omega$), apply per-colour-twisted free inverse (a per-colour phase
in the momentum -- i.e. the temporal holonomy becomes a per-colour shift of the AP-time BC phase,
$e^{i(\pi+\theta_t^{(c)})}$, and likewise spatial), rotate back ($\Omega^\dagger$). Compatibility:
$\Omega_\text{diag}$ for the AVERAGE holonomy is a CONSTANT rotation, and a constant rotation preserves
Landau transversality ($\partial_\mu(gA_\mu g^\dagger)=g(\partial_\mu A_\mu)g^\dagger=0$), so it uses
Landau's residual global-gauge freedom -- no conflict.
THE TENSION (why it is only approximate): FFT-diagonality of $F$ needs the twist CONSTANT in $x$.
Diagonalising $P(x_\perp)$ EXACTLY gives $x_\perp$-dependent eigenphases (and eigenvectors) -> a
spatially-varying twist -> NOT FFT-diagonal. So the practical match is to the SPATIAL-AVERAGE Polyakov
loop (one constant $SU(N)$ -> constant Cartan twist -> FFT preserved, captures the holonomy zero mode);
the residual spatial variation of $P(x_\perp)$ stays as a non-FFT piece. This is the same
average-vs-variation caveat as the closed-form section, now seen from the fermion side: the constant
twist is exactly "the action on fermion vectors" of the average holonomy.

**Order:** (i) the maximal-tree run (already coded) gives a first look -- its off-tree holonomy content
and whether it changes the win; (ii) measure Polyakov loops on the flowed 747/640 frames (cheap, a
WilsonLoops call); (iii) if the holonomy is non-trivial, implement the twisted free kernel (a BC change
in FreeMobius5DInverse) and re-test. This is the most promising lead yet because it is specific,
measured, and cheap to fix -- unlike the lump, which we exhausted.

1. **Decisive re-check** 80/120 at $s/t_0 = 1.0$ (queued; needs the build_mpi_merged freeprec
   rebuild) — sizes channel 2 vs channel 1 at the win level.
2. **$r(x)$ localisation diagnostic** (Sec. 3) on 640 + 80 — tells us if the Landau frame is
   bulk-optimal already.
3. If channel 2 is real: **b1 deep-flow quotient** A/B on one hard config, and **a1 copy search**
   (both cheap); a3 norm-optimised GD as the principled follow-up (roadmap Dir 1c machinery).
4. Channel 1 regardless: **deflation of the m0d stragglers** on the FGMRES side (we already have
   the 640 modes), and note $M_1$ is expected to gain from any frame-localisation improvement.
