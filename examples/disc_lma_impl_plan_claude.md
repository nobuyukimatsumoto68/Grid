# variance reduction by low-mode averaging (LMA) -- DISC + CONN impl plan

## STATUS (2026-06-25): chunk-A RAN -- lift CONFIRMED; chunk B = SOURCE PROJECTION
chunk-A bench (disc_lma_bench_f3GVvKeN3LxX, m=0.01 lat.758, shift-invert, Nconv=100):
- **GATE2 r1 = ||MV-su|| = 2.5e-14 -> the v/w lift (V_i even fill-in + a_i=E.V_i
  surface map) is CORRECT to machine precision.** This is the decisive result.
- GATE1 eigres = 0.13 (uniform over all 100 modes); GATE2 r2 = 0.13 (mirrors GATE1).
  eval[0]=4.83e-4..eval[99]=9.73e-3 = the KNOWN m=0.01 spectrum (eigenVALUES right),
  so the 0.13 is mode ROUGHNESS, expected from the SINGLE-PRECISION inner CG (log:
  "Computed 9.8e-6 / True 2.4e-3" -- single-prec residual drift at kappa~1.6e5). Fine
  for deflation; for LMA the low part is then ~13% accurate -> degrades VARIANCE gain
  only (recombination/projection central value stays unbiased). User ACCEPTS the
  O(0.1) single-prec mode error.
- GATE3 ||b-g5a||/||b|| ~0.21 (modes 0-3), 1.78 (mode 4): inconclusive while modes are
  rough -- revisit after accurate modes. (b-side has no machine-precision gate yet.)
- For LMA-quality modes (production): double or MIXED-PREC (reliable-update) inner CG
  beats the single floor; Chebyshev IRL (pure matvecs, no CG-drift) likely accurate
  too. Optional `EIG_PREC=double` re-run would give clean GATE1 + a GATE3 verdict.
chunk B FORM DECIDED 2026-06-25 = **SOURCE PROJECTION** (project odd-cb source onto the
ORTHONORMAL $u_i$, solve the well-conditioned high system): variance AND speed, clean
(user pushed: given the deflated-solve path already exists, recombination has no
upside). See chunk B below.
Builds DIRECTLY on the eigenbasis from the deflation work
(`disc_mrhs_deflation_impl_plan_claude.md`): the SAME shift-invert eigenvectors
serve THREE uses on one per-config eigensolve:
  (1) DEFLATION  -> speed of the solves (done/benchmarked),
  (2) DISC LMA   -> variance of the disconnected loop (this plan, below),
  (3) CONN LMA   -> variance of the CONNECTED meson correlator (added 2026-06-24;
                    classic LMA -- DeGrand-Schaefer was connected mesons).
Computing the eigenbasis once and reusing it three ways is the key amortization;
the per-ensemble evec directory (memo below) is shared by disc + conn.

## Method: LMA (NOT AMA) -- and how they differ
- **LMA (what we do here)** -- DeGrand-Schaefer hep-lat/0401011; Giusti-Hoelbling-
  Luscher-Wittig hep-lat/0402002. Split $M^{-1}=M^{-1}_\text{low}+M^{-1}_\text{high}$;
  compute the LOW part EXACTLY from eigenvectors (noise-free, all sites), estimate
  the HIGH part stochastically (as now, but with the low modes projected out).
- **AMA (the generalization, NOT needed first)** -- Blum-Izubuchi-Shintani
  arXiv:1208.4349 (refined: Shintani et al. 1402.0244). A covariant CHEAP
  approximation (typically SLOPPY relaxed-CG solves, or low-mode-deflated) averaged
  over translations + a few exact bias corrections:
  $$O^\text{AMA}=\frac1{N_G}\sum_g O^{(\text{appx}),g}+\big(O^{(\text{exact})}-O^{(\text{appx})}\big).$$
  LMA is the special case appx = low-mode part. AMA's extra lever = sloppy solves +
  bias correction -- an ORTHOGONAL future option (e.g. relax the high-mode solve
  tolerance and correct), cite but defer.

## Physics / goal
The disconnected loop $L_\Gamma(x)=\mathrm{tr}_{sc}[\Gamma\,S(x,x)]$ ($S=M^{-1}$ the
physical quark propagator) has a stochastic-estimator variance DOMINATED by the low
Dirac modes at light mass ($\lambda_\text{min}\sim5\times10^{-4}$ here). LMA computes
that dominant piece EXACTLY:
$$L_\Gamma(x)=\underbrace{\mathrm{tr}_{sc}[\Gamma\,S_\text{low}(x,x)]}_{\text{exact, no noise, all }x}
            +\underbrace{\mathrm{tr}_{sc}[\Gamma\,S_\text{high}(x,x)]}_{\text{stochastic, low modes removed}} .$$
Momentum-projected (as the disc binary does):
$$L_\Gamma(t,\vec p)=\sum_{\vec x}e^{-i\vec p\cdot\vec x}\,\mathrm{tr}_{sc}[\Gamma\,S(x,x)].$$
UNBIASEDNESS: confirmed for the correlator at $\Delta t\neq0$ because the disc
binary uses FRESH time-diluted Z4 noise per source timeslice -> loops at different
$t$ use independent noise; the exact low part adds zero bias at any $\Delta t$.
Only the $\Delta t=0$ contact point keeps the standard same-noise bias (drop/split).

## The exact low-mode part (the new object)
With $S_\text{low}=\sum_{i=1}^{N_{ev}}\frac1{\lambda_i}\,v_i\,w_i^\dagger$ (the A2A
"v/w" pair for the low modes),
$$L_\Gamma^\text{low}(t,\vec p)=\sum_{\vec x}e^{-i\vec p\cdot\vec x}\sum_{i=1}^{N_{ev}}
   \frac1{\lambda_i}\,\mathrm{tr}_{sc}\!\big[\Gamma\,v_i(x)\,w_i^\dagger(x)\big].$$
NO solves -- just contractions of the eigenvectors we already computed.

## KEY DERIVATION (FINALIZED 2026-06-24): physical v/w from OUR Schur eigenpairs
Our eigenvectors are of the EVEN-ODD Schur, SQUARED Mobius operator
($\hat H=M_{pc}^\dagger M_{pc}$, ODD checkerboard, 5D, eigenvalues $\lambda_i=\sigma_i^2$),
with (verified against Grid `SchurDiagMooeeOperator::Mpc` and
`CayleyFermion5D::Import/ExportPhysicalFermion*`)
$$M_{pc}=M_{oo}-M_{oe}M_{ee}^{-1}M_{eo}\quad\text{(odd}\to\text{odd)},\qquad
  \hat H v_i=\sigma_i^2 v_i,\ \ \langle v_i,v_j\rangle=\delta_{ij}.$$
A2A is the literature recipe (Foley-Juge-O'Cais-Peardon-Ryan-Skullerud
hep-lat/0505023); the EO/Schur lift below is exact algebra, no approximation.

**Step 1 -- odd-cb singular pair of $M_{pc}$.** Left singular vector
$u_i=M_{pc}v_i/\sigma_i$ (orthonormal), so $M_{pc}^{-1}=\sum_i\sigma_i^{-1}v_i u_i^\dagger$
(low part = the lowest $N_{ev}$). In Grid: `HermOpEO.Mpc(v_i,u_i); u_i*=1/sigma_i;`
with $\sigma_i=\sqrt{\lambda_i}$.

**Step 2 -- lift odd pair to the FULL 5D singular pair of $M$.** Block Schur
factorization $M=L\,\mathrm{diag}(M_{ee},M_{pc})\,U$ with
$L=\big(\begin{smallmatrix}1&0\\ M_{oe}M_{ee}^{-1}&1\end{smallmatrix}\big)$,
$U=\big(\begin{smallmatrix}1&M_{ee}^{-1}M_{eo}\\0&1\end{smallmatrix}\big)$ gives
$M^{-1}=U^{-1}\mathrm{diag}(M_{ee}^{-1},M_{pc}^{-1})L^{-1}$. The low modes live only in
$M_{pc}^{-1}$, so $M^{-1}_\text{low}=\sum_i\sigma_i^{-1}\,\mathcal V_i\,\mathcal U_i^\dagger$ with the FULL 5D
$$\mathcal V_i=U^{-1}\binom{0}{v_i}=\binom{-M_{ee}^{-1}M_{eo}v_i}{v_i},\qquad
  \mathcal U_i=L^{-\dagger}\binom{0}{u_i}=\binom{-M_{ee}^{-\dagger}M_{oe}^\dagger u_i}{u_i}.$$
Grid calls (checkerboard inferred from the field):
- $\mathcal V_i$ even $=-$`MooeeInv(Meooe(v_i))` (this is exactly the intermediate Grid
  forms inside `Mpc`); odd $=v_i$. Assemble with `setCheckerboard`.
- $\mathcal U_i$ even $=-$`MooeeInvDag(MeooeDag(u_i))`; odd $=u_i$.

**Step 3 -- physical 4D A2A vectors.** The disc loop uses the physical propagator
$S=E\,M^{-1}\,I$ with $E=$`ExportPhysicalFermionSolution`, $I=$`ImportPhysicalFermionSource`.
From the Grid source: $I=D_-\,P$, $P\psi$ embeds the 4D field on the surface
($P_+\psi$ on $s{=}0$, $P_-\psi$ on $s{=}L_s{-}1$); $E\chi=P_-\chi[0]+P_+\chi[L_s{-}1]$.
Hence
$$S_\text{low}=\sum_i\frac1{\sigma_i}\,a_i\,b_i^\dagger,\qquad
  a_i=E\,\mathcal V_i,\qquad b_i=I^\dagger\mathcal U_i=P^\dagger\big(\texttt{DminusDag}\,\mathcal U_i\big),$$
where $P^\dagger\chi=P_+\chi[0]+P_-\chi[L_s{-}1]$ (chiral projectors SWAPPED vs $E$ --
$b$ is NOT built like $a$). Both $a_i,b_i$ are 4D spinor-color fields.

**$\gamma_5$-hermiticity cross-check (not assumed).** $S$ is $\gamma_5$-Hermitian, so
$b_i=\gamma_5 a_i$ should hold up to normalization -- used ONLY as an independent check
in chunk A, never relied on for the production value.

**The exact low loop.** On the diagonal, momentum-projected,
$$L_\Gamma^\text{low}(t,\vec p)=\sum_{\vec x}e^{-i\vec p\cdot\vec x}
   \sum_{i=1}^{N_{ev}}\frac1{\sigma_i}\,\mathrm{tr}_{sc}\!\big[\Gamma\,a_i(x)\,b_i(x)^\dagger\big].$$
NO solves -- contractions of vectors already in hand.

## Grid building blocks to MIMIC (found in repo)
- `algorithms/deflation/Deflation.h` `DeflatedGuesser::operator()`:
  `guess = sum_i v_i * innerProduct(v_i,src)/eval[i]` -- the $M^{-1}_\text{low}$-on-a-
  -source primitive (also used to PROJECT low modes OUT of the stochastic source).
- `qcd/utils/A2Autils.h` `A2Autils<Impl>::MesonField(mat, w_i, v_j, gammas, mom,
  orthogdim)` + `tests/Test_meson_field.cc` / `benchmarks/Benchmark_meson_field.cc`:
  the v/w + gamma + momentum contraction pattern. For the LOOP we need only the
  DIAGONAL ($i$=$j$, $x$=$y$) momentum-projected trace -- simpler than a full meson
  field, but A2Autils is the idiom to copy (and validates our contraction).
- Cayley hooks `ExportPhysicalFermionSolution` / `Dminus` (CayleyFermion5D.h:62-65)
  for the 5D->4D physical step.
- Our shift-invert eigensolve (bench `disc_mrhs_defl_bench_claude.cc`,
  `InverseHermOp` + IRL) already produces $\{\lambda_i,v_i\}$.

## CONNECTED correlator LMA (added 2026-06-24) -- same eigenbasis
The connected meson 2pt function
$$C_\Gamma(t)=-\sum_{\vec x}\mathrm{tr}\big[\Gamma\,S(x,0)\,\tilde\Gamma\,S(0,x)\big],
  \qquad S(0,x)=\gamma_5 S(x,0)^\dagger\gamma_5,\ \ \tilde\Gamma=\gamma_5\Gamma^\dagger\gamma_5,$$
currently uses ONE point-source propagator $S(\cdot,0)$ per config
(`baryons_0000_dirac_claude`). Its variance at light mass / large $t$ is dominated
by the low modes (pion). LMA (the ORIGINAL application -- DeGrand-Schaefer
hep-lat/0401011) splits each $S=S_\text{low}+S_\text{high}$ and treats the
low-low piece exactly, AVERAGED over all source points:
$$C_\Gamma^\text{LMA}(t)=\underbrace{\langle C^\text{ll}_\Gamma(t)\rangle_{\text{all }y}}_{\text{exact, vol-averaged}}
   +\underbrace{\big[C^\text{full}_\Gamma(t)-C^\text{ll}_\Gamma(t)\big]_{\text{point src }y_0}}_{\text{high-mode remainder, from the point source}} .$$
- $C^\text{ll}$ = both legs from $S_\text{low}$: an ALL-TO-ALL meson field built from the
  low-mode v/w vectors -> EXACTLY what `A2Autils<Impl>::MesonField(mat, w_i, v_j,
  gammas, mom, orthogdim)` computes (this is its canonical use; conn LMA fits
  A2Autils MORE directly than disc, which only needs the diagonal). Volume-averaged
  -> big variance cut, noise-free.
- The remainder $C^\text{full}-C^\text{ll}$ is evaluated on the SAME point source the
  conn binary already inverts -> captures the high modes; subtracting the
  point-source low-low avoids double counting. UNBIASED (deterministic split).
- Uses the SAME v/w reconstruction (the KEY DERIVATION above) and the SAME per-config
  eigenbasis as disc LMA + deflation. NEW binary
  `baryons_0000_dirac_lma_claude.cc` (do NOT edit the original conn binary); reads
  the saved evecs from the shared per-ensemble dir.
- Deflation ALSO speeds the conn point-source solve (reuse the bench solver), so the
  conn binary gets speed (deflation) + variance (LMA) from the one eigensolve too.

## Files -- ALWAYS NEW BINARIES (never edit the original disc binary)
- NEW `disc_lma_bench_claude.cc` (examples/) + Make.inc: on ONE config, compute the
  loop two ways -- (a) plain stochastic (as the disc binary), (b) LMA (exact low +
  projected stochastic high) -- and report central value (must agree) + VARIANCE.
- NEW production binary `disc_multipleGamma_binary_lma_claude.cc` (a SEPARATE
  binary, NOT a modification of `disc_multipleGamma_binary_claude.cc`): per config
  it eigensolves once, adds the exact low part to the loop field, and projects the
  low modes out of the stochastic part. Original disc + mixedprec binaries stay
  untouched as references.

## PRODUCTION DIRECTORY LAYOUT (memo, 2026-06-24)
The eigenvectors are EXPENSIVE and GAUGE-dependent (one set per config), and are
reused for BOTH deflation (speed) and LMA (variance) -- so they must be SAVED to
disk, computed once per config. In production create a NEW per-ensemble (or
per-config) directory that holds BOTH:
- the **eigenvector files** (`evec_<conf>.{h5,scidac}` + the `eval_<conf>` list),
  written with a Grid ScidacWriter/HDF5 writer right after the eigensolve;
- the **disc output h5** (the LMA loop data) for the same configs.
Layout sketch (per ensemble `..._b<betastr>_m<massstr>`):
```
.../lma_nc4nf1_2448_b<betastr>_m<massstr>/
    evec.<conf>.scidac   eval.<conf>.xml     # eigenbasis, per config (reused)
    disc_lma.<gam>.<conf>                     # the LMA loop output
```
So an interrupted/rerun job RELOADS the saved evecs (skip the eigensolve) and only
fills missing disc outputs -- same self-skip pattern as the current disc binary,
but now the evec files are the expensive checkpoint. Confirm the evec I/O format
(Grid Lanczos checkpoint uses ScidacWriter for evecs + XML for evals -- see
`tests/Test_compressed_lanczos_hot_start.cc` checkpointFine/Restore) and whether
the user wants HDF5 specifically for the evecs.

## Ordered chunks
### A -- v/w reconstruction + the exact low loop (correctness FIRST)
Files: `disc_lma_bench_claude.cc`, Make.inc
Reuse the bench eigensolve (shift-invert IRL -> $\{\lambda_i,v_i\}$ single-prec on
odd cb). Build $a_i,b_i$ per the KEY DERIVATION. Two correctness gates BEFORE any
variance claim:
1. **A2A == direct deflated solve, on a point source $\eta$** (the decisive check).
   - DIRECT: $I\eta\to$`RedBlackSource`$\to b_o$; apply the EXPLICIT low inverse
     $M_{pc}^{-1,\text{low}}b_o=\sum_i\sigma_i^{-1}v_i\langle u_i,b_o\rangle$;
     `RedBlackSolution`$\to$ full 5D; $E\to S_\text{low}\eta$ (4D).
   - A2A: $\sum_i\sigma_i^{-1}a_i\,\langle b_i,\eta\rangle$.
   - Must agree to eigen-precision. This validates the $\mathcal V/\mathcal U$ lift,
     the surface maps, AND the normalization in one shot. (Also report
     $\|b_i-\gamma_5 a_i\|$ as the independent $\gamma_5$-herm cross-check.)
2. **$L^\text{low}$ vs projected stochastic.** A high-statistics stochastic estimate
   of the loop with the high modes projected OUT should converge to $L^\text{low}$.
### B -- SOURCE PROJECTION (project the odd-cb source onto the orthonormal $u_i$)
**Form chosen: source projection -- variance AND speed from the one eigenbasis.**
(Recombination $L=L^\text{low}+L^\text{full,stoch}-L^\text{low,stoch}$ is the SAME
object computed differently -- $S_\text{high}\eta=S\eta-S_\text{low}\eta$ either way --
but it keeps the full ill-conditioned solve and gets ONLY variance. Since we already
have the deflated/projected-solve path, projection is strictly better.)

The loop is split $L_\Gamma=L_\Gamma^\text{low,exact}+L_\Gamma^\text{high,stoch}$, with
the high part from a solve whose source has the low modes removed. Per dilution Z4
source $\eta$ (a `LatticePropagator`), per column $sc$:
1. `ImportPhysicalFermionSource`$(\eta_{sc})\to$ 5D, `RedBlackSource`$\to\tilde b_o$ (odd cb).
2. Project out the low modes IN THE ODD-CB SPACE, where the singular vectors are
   ORTHONORMAL (no Gram inverse, no obliquity -- this is the whole point):
   $$\tilde b_o^\perp=\tilde b_o-\sum_{i=1}^{N_{ev}}u_i\,\langle u_i,\tilde b_o\rangle,
     \qquad u_i=M_{pc}v_i/\sigma_i\ \ (\text{orthonormal, from chunk A}).$$
3. Solve the PROJECTED system $x_o=M_{pc}^{-1}\tilde b_o^\perp$ -- now WELL-CONDITIONED
   (the small eigenvalues are gone) so the light-mass slowness is removed; deflated
   solver optional. `RedBlackSolution` + `ExportPhysicalFermionSolution`$\to S_\text{high}\eta$.
4. $L_\Gamma^\text{high,stoch}=\mathrm{tr}[\Gamma\,(S_\text{high}\eta)\,\eta^\dagger]$
   (the binary's existing `TraceField`, original $\eta$ in the bra), plus the exact
   $L_\Gamma^\text{low}$ from chunk A.

**Exactly unbiased, no projector subtlety** (the key identity): since
$M_{pc}v_i=\sigma_i u_i\Rightarrow M_{pc}^{-1}u_i=v_i/\sigma_i$,
$$M_{pc}^{-1}\Big(1-\sum_i u_iu_i^\dagger\Big)\tilde b_o
   =M_{pc}^{-1}\tilde b_o-\sum_i\frac{v_i}{\sigma_i}\langle u_i,\tilde b_o\rangle
   =M_{pc}^{-1,\text{high}}\,\tilde b_o,$$
so projecting onto the ORTHONORMAL $u_i$ removes exactly the low part of the inverse;
after back-substitution + Export, $S_\text{high}=S-S_\text{low}$, and
$\langle\mathrm{tr}[\Gamma S_\text{high}\eta\eta^\dagger]\rangle=\mathrm{tr}[\Gamma S_\text{high}]$
(uses only $\langle\eta\eta^\dagger\rangle=1$). The even-even contact $M_{ee}^{-1}$ that
the back-substitution carries is a HIGH object (correctly in $S_\text{high}$, absent
from the $a_ib_i$ low part). Variance drops because the low modes -- which dominate the
light-mass loop variance -- are gone from the stochastic part and supplied EXACTLY by
$L^\text{low}$. The $u_i$ are already built in chunk A (`BuildPhysicalA2A`).
NOTE: this is what deflation does for the SOLVE; here the same projection is used to
return the HIGH part of the propagator for the loop. Recombination remains a handy
ONE-TIME validation cross-check (it = old estimator + a mean-zero correction, so it
ties to the existing code per noise sample), not the production path.
### C -- VALIDATION (the deliverable = variance, not speed)
On a few configs: (i) central value LMA == plain stochastic within errors
(unbiased; check $\Delta t\neq0$), (ii) VARIANCE of the disc correlator drops.
Report the variance ratio vs $N_{ev}$ -- THIS sets the physics value and how far
$N_{ev}$ is worth pushing (variance-driven, unlike the speed knee).
### D -- production fold-in (only if C shows real variance reduction)
Add the exact low part to the loop field in the disc binary; optionally coarsen
dilution (SEPARATE choice, validate its own variance budget) for fewer solves.

## Open questions
1. The v/w recipe normalization for Mobius EO-Schur (the KEY DERIVATION) -- finalize
   vs A2A code/literature; validate by reconstructing $S_\text{low}$ against a direct
   deflated solve on a point source.
2. $N_{ev}$ for variance: how much of the loop variance lives in the lowest ~150-250
   modes? Measured in chunk C; likely larger than the speed-optimal $N_{ev}$.
3. Eigenvectors in single precision -- is single enough for the exact low part, or
   does $1/\lambda_i$ at $\lambda\sim5\times10^{-4}$ need double for the smallest modes?
4. Reuse the SAME eigenbasis for deflation (speed) AND LMA (variance) in one
   production pass -- compute once, use twice.
