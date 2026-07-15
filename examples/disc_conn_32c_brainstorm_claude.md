# Conn + Disc on the $32^3\times64$ ensembles -- brainstorm (pre-plan)

Status: BRAINSTORM. No code yet. Goal of this doc: agree the idea set and the
open decisions before writing `*_impl_plan_claude.md`.

## Target

Connected + disconnected meson measurements on ALL SIX new $32^3\times64$ ($L_s=16$,
$M5=1.5$) ensembles in `/p/lustre5/matsumoto5/` (configs are `ckpoint_lat.<n>` sitting
directly in `beta<...>m<m>/`; the `32c_<hash>` subdirs are separate, not configs):

| ensemble dir                 | $\beta$ | $m$  | configs (2026-06-29)               | maturity        | estimator        |
|------------------------------|---------|------|------------------------------------|-----------------|------------------|
| `32_64_0.01/beta10.8m0.01`   | 10.8    | 0.01 | ~21, lat.4-84, stride 4, growing   | early (thermalizing) | LMA v2 (light)   |
| `32_64_0.05/beta10.84m0.05`  | 10.84   | 0.05 | ~29, lat.4-116, stride 4, growing  | early (thermalizing) | LMA v2 (light)   |
| `32_64_0.1/beta10.865m0.1`   | 10.865  | 0.1  | 60, lat.20-1200, stride 20         | mature          | mixed-prec       |
| `32_64_0.2/beta10.99m0.2`    | 10.99   | 0.2  | 79, lat.20-1580, stride 20         | mature          | mixed-prec       |
| `32_64_0.3/beta11.035m0.3`   | 11.035  | 0.3  | 95, lat.20-1900, stride 20         | mature          | mixed-prec       |
| `32_64_0.4/beta11.045m0.4`   | 11.045  | 0.4  | 100, lat.20-2000, stride 20        | mature          | mixed-prec       |

This mirrors the $24^3\times48$ program; the same physics combine $C_\text{phys}=2D-C$.
Light targets (0.01, 0.05) -> deflation/LMA. Heavy (0.1-0.4) -> mixed-prec only
(24c verdict: eigensolve overhead not worth it at heavy mass).

## RESOLVED DECISIONS (2026-06-29)

- D1 time dilution: implement the $K$-partition KNOB (`DISC_TPART`, default 2),
  expect to RUN at $K=2$ but keep $K$ tunable + measure. Applies to ALL masses (the
  $t+N_t/2$ cross term is short-range/suppressed at heavy mass too -- safest there).
- D2 hierarchical probing: DEFERRED (LMA + dilution-2 + mixed-prec first).
- D3 conn at light mass: SHARE the per-config disc eigenbasis -> deflate/LMA conn
  (revive `baryons_0000_dirac_lma_claude.cc` pattern); heavy conn -> mixed-prec.
- D4 scope: ALL SIX ensembles. Light = LMA v2 (re-tuned per ensemble); heavy =
  mixed-prec, both conn + disc.

## STAGING (2026-06-29)

- **ROUND 1 = HEAVY 0.1-0.4 only** (mature, ready now): plain MIXED-PREC conn + disc,
  with the $K$-partition time-dilution knob on disc. NO eigensolve/LMA/eigref. This is
  the immediate deliverable.
- **ROUND 2 = LIGHT 0.01, 0.05** (LMA v2 + shared-basis conn): DEFERRED until those
  ensembles thermalize (user: "don't touch the light masses until they thermalize").
  HMC still running; revisit when lat numbers are deep + plaquette equilibrated.

## STILL-OPEN (resolve in the impl plan)

- D5 RESOLVED: light deferred (above). Heavy ready now.
- D6 MPI layout for $32^3\times64$ (3.16x volume): node count + rank grid + the
  Grid AccCache device-mem ceiling (24c tripped at NEV>=123 / ~102GB) shifts with
  volume -> needs a fit/eigensolve-memory check. Affects NEV ceiling for light LMA.
- D7 heavy ensembles are mature + stride 20 already -> run at native stride 20 (no
  thinning)? conn+disc share the SAME config set to combine $2D-C$.

## What ports directly from the 24c work

- Disc binary `disc_multipleGamma_binary_lma_v2_claude.cc` is geometry-agnostic:
  lattice from `GridDefaultLatt()` (`--grid 32.32.32.64`), `Nt` read from the grid,
  only `Ls=16` is literal (unchanged). So the SAME binary runs on 32c; what changes
  is the launch (`--grid`, MPI layout) and a per-ensemble eigref/tuning.
- Mixed-prec + batched + Cheby/RR deflation (LMA v2) is the production stack already.
- Conn binary `baryons_0000_dirac_claude.cc` loop-mode is geometry-agnostic too.

Main porting cost is NOT code -- it is (a) MPI layout for the 3.16x bigger volume
($32^3\times64$ vs $24^3\times48$ = 3.16x sites; per-rank field 3.16x -> the 24c
8N/32r host-arena fit no longer holds, will need ~16-32 nodes), and (b) a one-time
per-ensemble eigref + Cheby-window calibration on a real 32c config (spectrum shifts
with volume/$\beta$/$m$).

## Idea 1 -- t-direction dilution: source at $t$ and $t+N_t/2$ (user's idea)

Current high-mode stochastic loop (`disc_..._lma_v2_claude.cc:251`): full time
dilution, one Z4 source per timeslice, loop $t=0..N_t-1$ (x eo x spin x color).
`StochasticDilutedSource` (`disc_lma_v2_common_claude.h:537`) keeps the random field
on a single slice: `xi = where(t==tslice, xi, zz)`.

Proposed: put the SAME diluted source on two slices $t$ and $t+N_t/2$ in one solve,
loop $t=0..N_t/2-1$. Edit is one line: `where((t==tslice)||(t==tslice+Nt/2), xi, zz)`.
Factor 2 fewer solves.

### Why it is unbiased + cheap (variance)

The disc loop is the LOCAL field $L_\Gamma(x)=\mathrm{tr}[\Gamma\,\psi(x)\,\eta^\dagger(x)]$,
$\psi=D^{-1}\eta$, and `TraceField` keeps it only where $\eta(x)\neq0$ (the source
slices). At sink slice $t$ with $\eta=\eta_t+\eta_{t+N_t/2}$:
$$
L_\Gamma(t) = \underbrace{\eta_t^\dagger\,\Gamma\,[D^{-1}\eta_t](t)}_{\text{wanted diagonal}}
            + \underbrace{\eta_t^\dagger\,\Gamma\,[D^{-1}\eta_{t+N_t/2}](t)}_{\text{cross term}} .
$$
$\eta_t$ and $\eta_{t+N_t/2}$ are independent draws (different sites of the random
field) so $E[\text{cross}]=0$ over noise -> the estimator stays UNBIASED. The cross
term only adds variance, of size $\sim |S(t,t+N_t/2)|^2$, a propagator across the
maximal separation $N_t/2=32$ -> exponentially suppressed.

### Synergy with LMA (the real justification)

LMA already computes the LOW-mode (long-range) part of the loop EXACTLY (no noise).
The stochastic estimator only carries the HIGH-mode remainder $S_\text{high}$, which
is SHORT-ranged -> the cross propagator $S_\text{high}(t,t+N_t/2)$ is suppressed far
MORE than the full propagator. So precisely because we run LMA, the $t+N_t/2$ pairing
is essentially free. This is the clean argument for the factor 2.

### Generalization (a knob, not a fixed 2)

Same logic with $K$ equally-spaced slices $\{t+kN_t/K\}_{k=0}^{K-1}$, loop
$t=0..N_t/K-1$ -> factor $K$, variance penalty $\sim K\,|S_\text{high}(N_t/K)|^2$.
With LMA, $K=4$ (separation 16) may still be cheap. Proposal: make the number of time
partitions $K$ a build/env knob (`DISC_TPART`, default 2) and MEASURE variance vs $K$
on a couple of configs rather than fix $K=2$ blindly. $K=1$ recovers the current code.

## Idea 2 -- mixed-prec + deflation, by mass (ports from 24c verdicts)

24c economics: mixed-prec ~1.4x at all masses (no overhead); deflation/LMA pays off
only at LIGHT mass (eigensolve overhead amortizes against slow solves), marginal at
$m=0.05$ but WINS ~1.25x once the eigensolve is the cheap Cheby+RR. Both live 32c
masses are light -> LMA v2 stack for both. Heavier 0.1-0.4 (when generated) -> plain
mixed-prec, no eigensolve. Re-tune eigref per ensemble (spectrum is volume/m/beta
dependent; the 24c windows are NOT valid at 32c -- one calibration run each).

## Additional ideas (menu -- decide what is in scope)

1. **Hierarchical probing (HP)** -- Stathopoulos-Laeuchli-Orginos, arXiv:1302.4018.
   Hadamard/coloring vectors that cancel the dominant near-diagonal off-diagonal
   contributions to the trace variance; STACKS on top of dilution+LMA and is the
   state-of-the-art disc variance reducer. Cost: needs a coloring over the lattice +
   $2^k$ probing vectors. Biggest extra win, biggest extra complexity. IN or OUT?

2. **$K$-partition time dilution knob** (Idea 1 generalized) -- cheap to add, gives a
   measured variance-vs-cost curve; recommend YES regardless of HP.

3. **Hopping-parameter / first-terms subtraction** -- subtract a few exactly-known
   (traceless for several $\Gamma$) low-order terms to cut noise. Less standard for
   DWF than Wilson; LMA already removes the dominant low-mode variance, so likely LOW
   marginal value here. Probably OUT.

4. **NSRC / number of Z4 hits as an explicit knob** -- with LMA + reduced dilution the
   per-config cost drops; may want >1 hit to rebalance noise. Make it an env knob and
   tune from production variance. Cheap. Recommend YES.

5. **Conn deflation/LMA sharing the eigenbasis** -- conn is cheap (point source, 12
   solves) but at $m=0.01$ on the bigger volume the solves are slower; the SAME
   per-config eigenbasis can deflate conn and (A2A) give conn LMA. 24c had
   `baryons_0000_dirac_lma_claude.cc` sharing the basis. Worth it for 32c m=0.01? Or
   just mixed-prec conn (simple, already ~1.4x)?

## Idea 5b -- is deflation/LMA REALLY useless at heavy mass on the BIGGER volume? (revisit)

The "heavy -> mixed-prec only" verdict above is PORTED from 24c and is NOT safe to
assume at $32^3\times64$. Two separate quantities, do not conflate:

- **DEFLATION = speed** (cut CG iterations). Governed by the condition number
  $\kappa=\lambda_\text{max}/\lambda_\text{min}$.
- **LMA = variance** (remove the low-mode noise of $\mathrm{Tr}[\Gamma D^{-1}]$).
  Governed by how much of the disc-loop signal/variance sits in the low modes.

Volume / mass scaling (the user's point):
- Spectral DENSITY scales with the 4-volume: $N(\lambda_c)\sim V\!\int_0^{\lambda_c}\!\rho(\lambda)\,d\lambda$.
  So $32^3\times64$ has ~3.16x MORE eigenmodes below any fixed cut than $24^3\times48$.
- Smallest Dirac eigenvalue: near-zero modes scale as $\sim 1/(V\Sigma)$ (spectrum
  stretches DOWN with volume). But a MASSIVE operator lifts the low edge to
  $\lambda_\text{min}(M_\text{pc}^\dagger M_\text{pc})\sim \mathcal{O}(m^2)$ once
  $m \gg 1/(V\Sigma)$. At HEAVY $m=0.1$-$0.4$ the mass dominates -> $\lambda_\text{min}$
  is mass-set, only MILDLY volume-dependent; at light $m$ the volume stretch matters.

Net for HEAVY mass on the bigger volume:
- (speed) per-solve $\kappa$ is mass-dominated -> deflation speedup PER MODE ~ similar
  to 24c; but to reach the same $\lambda_\text{cut}$ you need ~3.16x MORE modes
  ($N_\text{ev}\sim V$) and the eigensolve costs more -> pure-speed economics do NOT
  automatically flip. MEASURE the break-even, do not assume.
- (variance) ~3.16x more low modes CONTRIBUTE to the loop -> LMA could remove MORE
  variance than at 24c; countered by the heavy mass gap suppressing the RELATIVE
  low-mode weight. Net is EMPIRICAL.

CRUCIAL stale-verdict caveat: the 24c "not worth it at heavy mass" (m=0.1: 1948s,
eigensolve 386s) used the EXPENSIVE shift-invert eigensolve. The CHEAP Cheby+RR
per-config eigensolve (~95s on 24c m=0.01) already FLIPPED m=0.05 from "tied" to
"wins ~1.25x" -- so the heavy verdict may be stale even at 24c. With cheap Cheby
eigensolve + 3.16x denser spectrum, deflation/LMA at $m=0.1$-$0.4$ on 32c is
plausibly worth it. => DO NOT hard-code "mixed-prec only" for heavy; gate on a cheap
per-ensemble spectrum probe (lambda_min + mode density below a cut + defl break-even),
which is the eigref step we run anyway. The LMA-v2 binary is geometry-agnostic, so
switching a heavy ensemble to deflation/LMA is a binary swap, not new code.

## Idea 6 -- other Grid solver/preconditioning features (verified present in this checkout)

Current solver = `SchurDiagMooee` + `MixedPrecisionConjugateGradientBatched` (batched
INDEPENDENT mixed-prec CG, NOT true block). Unused Grid levers:

- **MADWF** (`Grid/qcd/action/fermion/MADWF.h`, `tests/solver/Test_zMADWF_prec.cc`):
  outer physics kept at $L_s=16$ EXACT, inner solve at reduced $L_s'$ (ZMobius coeffs)
  + Pauli-Villars correction -> ~$L_s/L_s'$ fewer 5D matvecs with NO action/mres change.
  Strong at LIGHT mass (round 2), stacks on top of deflation (deflation kept for LMA
  anyway). HIGH effort: new ZMobius inner-coeff tuning (one-time per $L_s',m,\beta$) +
  PVinverter + inner Schur solver + guesser. NOT built in this project. Refs: Yin-
  Mawhinney; Cossu et al arXiv:1706.05843; Brower-Neff-Orginos (Mobius).
- **Schur DiagTwo** (`SchurRedBlackDiagTwoSolve`, `SchurDiagTwoOperator`): better-
  conditioned red-black decomposition than DiagMooee; ~1-line drop-in. Cheap A/B,
  no-regret. NOTE: changing the Schur op also changes the deflation operator (round 2
  eigenbasis must match) -- keep consistent.
- **True Block CG** (`BlockConjugateGradient.h`, `BlockCGrQ`): shares ONE block-Krylov
  space across the 16 spin-color RHS. CONFIRMED we do NOT use this -- current
  `MixedPrecisionConjugateGradientBatched` (ConjugateGradientMixedPrecBatched.h) is a
  "mixed precision restarted defect correction CG" that solves the RHS INDEPENDENTLY,
  batching only the dslash for GPU THROUGHPUT (no shared Krylov, no iteration cut).
  Block CG's shared subspace cuts iterations only when RHS share low-mode content ->
  MODEST for random stochastic disc RHS, MORE for the conn POINT source.
  **MEMORY (user's point): Block CG + DEFLATION rehits the 24c AccCache cliff.** The
  cliff is dominated by the resident NEV evec store (must stay resident -- deflation
  projects every RHS every solve); Block CG adds a width-$N_\text{rhs}$ block-Krylov
  working set (P, AP, R, X in working prec) ON TOP -> raises peak where we were already
  at ~102GB. Current batched+deflation already coexists with the store WITHOUT a
  block-Krylov set -> it is the memory-aware sweet spot for the disc DEFLATION path.
  RULE: keep batched+deflation for disc deflation (rd2/heavy-if-probed); reserve Block
  CG for NON-deflated / low-NEV paths -- round-1 heavy disc (no evecs) and CONN (point
  source, loose memory). On 32c node count is free (D6) -> block+defl, if ever wanted,
  is bought with NODES (smaller per-rank fields), not byte-shaving.
- **Polynomial/Chebyshev preconditioning** of the inner CG: redundant with deflation;
  low value here.
- **Runtime perf flags** (`--shm`, `--comms-overlap`, dslash asm, rank-grid mapping):
  wall-time not iterations; real on MI300A -> folds into D6 (MPI layout).

ASSESSMENT: Round 1 (HEAVY, well-conditioned) -> mixed-prec is the lever; only cheap
no-regret add = Schur-DiagTwo A/B + correct runtime/rank-grid flags. Do NOT block heavy
production on MADWF/BlockCG. Round 2 (LIGHT) -> MADWF is the strong unused lever (stacks
w/ deflation); BlockCG a secondary A/B. MADWF = parallel investigation, not in round 1.

## Idea 7 -- disk-store evecs: helps VARIANCE-NEV, not the deflation-solve ceiling

User idea: evecs are expensive -> disk-store them, "probably solves the memory issue."
Refined -- the basis is used two ways with OPPOSITE memory behavior:
- **Per-SOLVE deflation guesser (speed)**: projects every RHS onto ALL modes, 768-1536x
  per config -> modes MUST be RESIDENT in HBM (streaming 215GB/solve from Lustre is
  hopeless). This is what tripped the NEV>=123 AccCache cliff. Disk does NOT help; the
  resident-NEV ceiling is HBM-bound, lifted by NODE COUNT (free on 32c, D6).
- **Once-per-CONFIG LMA low part + source projection (variance)**: L^low = sum_i
  (1/sigma_i) <b_i, Gamma a_i> and the source projection onto u_i touch each mode ONCE
  per config and are pure SUMS over modes -> STREAMABLE mode-by-mode from disk, never
  all resident. HERE disk-store genuinely lifts the ceiling -> can use MORE modes than
  fit in HBM for variance.
=> Real payoff = DECOUPLE two NEV: NEV_resident (~100, HBM, deflation SPEED) vs
NEV_streamed (larger, disk, LMA VARIANCE). LMA wants more modes than the speed knee, and
32c's 3.16x denser low spectrum is exactly where extra variance modes help.
Recompute angle is WEAK: per-config eigensolve is only ~3-5% of config wall (cheap
eigref-anchored Cheby+RR; the expensive shift-invert is the PER-ENSEMBLE eigref, done
once + already saved as the SPECTRUM only). Per-config evecs are gauge-dependent ->
can't transfer across configs anyway. SaveEvecs/LoadEvecs exist in the header (unused).
CAVEAT: 215GB/config scratch + Lustre streaming bandwidth must be checked.
PLAN: treat as a MEASURED knob inside the deflation/LMA bench-tune (NEV_resident vs
NEV_streamed), not a foregone design.

## Open decisions (need user input before the impl plan)

- D1. Time dilution: fix $K=2$, or implement the $K$-partition knob and measure?
- D2. Hierarchical probing: in scope now, or defer (LMA + dilution-2 first)?
- D3. Conn at $m=0.01$: deflation/LMA (share basis) or plain mixed-prec only?
- D4. Scope now: only the two live light ensembles (0.01, 0.05), or also wire up the
      heavier 0.1-0.4 (mixed-prec) for when those streams populate?
- D5. Are the live ensembles equilibrated enough to start measuring, or measure-as-we-go
      / wait for more thermalization? (HMC still running per memory.)
- D6. MPI layout for $32^3\times64$ -- target node count (16N? 32N?) and rank grid; needs
      a quick fit/eigensolve-memory check (the 24c 102GB AccCache ceiling at NEV>=123
      will shift with volume).
