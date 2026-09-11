# Topological spectral correction of the free-limit preconditioner

## Notation (Nobu's, this thread)

- $F := D_W(U=1)$ -- the free Wilson/Mobius Dirac **operator** after trivializing gauge fixing.
- $M0 := F^{-1}$ -- the preconditioner (the code object `FreeMobius5DInverse` applies $M0 = F^{-1}$;
  in older notes "$F$" meant this inverse -- here $F$ is the OPERATOR).
- Framed interacting operator $D_\text{int} := D_W(U_\Omega)$ -- the **4D** Wilson kernel at fixed $M5$
  (NOT the 5D Mobius $D5$). Premise: the frame makes $U_\Omega \approx 1$, so $F \approx D_\text{int}$.
- Would-be-topological mode (4D): right eigenpair $D_\text{int}\,\phi = \lambda\,\phi$, $\lambda \approx 0$.
  Left eigenpair comes FREE from $\gamma_5$-Hermiticity ($D_W^H = \gamma_5 D_W \gamma_5$): for real $\lambda$,
  $\chi = \gamma_5\phi$ satisfies $\chi^H D_W = \lambda\chi^H$. Biorthonormal $\chi^H\phi = 1$ by scaling.
  NO two-sided Arnoldi; this is WHY we stay in 4D (the 5D Mobius $D5$ is non-normal with no such structure).

## Mass-independence and the 4D -> 5D lift (the payoff)

The 4D kernel $D_W$ (fixed $M5$) does NOT see the quark mass $m$ -- $m$ enters only the 5th-direction
structure. So $(\phi,\chi{=}\gamma_5\phi,\lambda)$ is computed ONCE per config, mass-independent. The
correction is applied at the KERNEL level ($F \to D_\text{approx}$). Mobius is affine in the kernel,
$D5 = A(m,M5,b,c)\otimes D_W + B(m,M5,b,c)\otimes 1$, so the kernel correction lifts to a 5D correction
$A(m)\otimes C$ that composes with the EXISTING exact 5th-direction block inverse $M0(m)$ to generate the
correct $m$-dependent 5D near-zero behaviour AUTOMATICALLY. One 4D correction -> all masses. Expensive
part (4D eigensolve) once per config; per mass = re-lift through $A(m)$ + a few $M0(m)$ applies for the
Woodbury vectors.

## Why $F$ fails on the topological mode (the near-null outlier)

$F$ has a spectral gap ($\approx 1$ on the DW scale) -- it has NO near-zero mode. So $M0 = F^{-1}$ stays
bounded on $\phi$, while $D_\text{int}\phi \approx 0$:
$$
M0\, D_\text{int}\, \phi = \lambda\, F^{-1}\phi \approx 0 .
$$
Thus $M0\,D_\text{int}$ inherits a near-null eigenvalue exactly where $D_\text{int}$ had one. That outlier
is the cluster that stalls the Krylov solve, and what external deflation targets.

## The proposal: correct $F$ instead of deflating the Krylov space

Oblique spectral projector $P := \phi\chi^H$ (idempotent: $P^2 = \phi(\chi^H\phi)\chi^H = P$),
complement $Q := I - P$. Because $\phi,\chi$ are the true eigenpair,
$$
P D_\text{int} = D_\text{int} P = \lambda P,\qquad
D_\text{int} = Q D_\text{int} Q + \lambda P \quad(\text{exact, block-diagonal in }P\oplus Q).
$$
Replace ONLY the complement block $Q D_\text{int} Q \to Q F Q$ (use the free model where it is good):
$$
\boxed{\,D_\text{approx} = Q F Q + \lambda P\,}
$$
Exact on the $P$-block ($D_\text{approx}\phi = \lambda\phi$, $\chi^H D_\text{approx} = \lambda\chi^H$),
free-modelled on the $Q$-block. Strictly better than $F$; the corrected preconditioner
$M0_\text{new} := D_\text{approx}^{-1}$ has $M0_\text{new} D_\text{int} \approx 1$ INCLUDING on $\phi$,
so the near-null outlier is removed at the source.

## The inverse is $F^{-1}$ + rank-2 Woodbury (cheap)

Expand $C := D_\text{approx} - F = QFQ + \lambda P - F = -PF - FP + PFP + \lambda P$. With $P=\phi\chi^H$,
$a := F^H\chi$, $f := F\phi$, $s := \chi^H F\phi$ (scalar):
$$
C = -f\,\chi^H + \phi\big[(s+\lambda)\chi^H - a^H\big] = U V^H,\qquad
U = [\,\phi,\; f\,],\quad V^H = \begin{bmatrix}(s+\lambda)\chi^H - a^H\\ -\chi^H\end{bmatrix}.
$$
Rank 2. Woodbury:
$$
D_\text{approx}^{-1} = F^{-1} - F^{-1}U\,\big(I_2 + V^H F^{-1} U\big)^{-1} V^H F^{-1}.
$$
Precompute (once per config, per mass):
- $F^{-1}U = [\,F^{-1}\phi,\; F^{-1}f\,] = [\,F^{-1}\phi,\; \phi\,]$ -- ONE free-inverse apply ($F^{-1}\phi$);
  the second column collapses to $\phi$ since $F^{-1}F\phi = \phi$.
- $V^H F^{-1}$ -- two row-vectors; needs $F^{-H}$ on $\chi$ and $a=F^H\chi$ (the latter gives
  $F^{-H}F^H\chi=\chi$), i.e. one adjoint free-inverse apply $F^{-H}\chi$.
- the $2\times2$ matrix $(I_2 + V^H F^{-1}U)^{-1}$ -- inverted once.

Per-apply overhead of $M0_\text{new}$ vs $M0$: $F^{-1}b$ (unchanged) + a handful of global inner products
and axpys (the rank-2 update). Negligible next to the FFT + 5th-direction block solve. The correction
rides OUTSIDE the FFT, so the exact per-momentum $(4L_s)\times(4L_s)$ block inverse is untouched.

Multiple topological modes ($|Q|$ of them, plus near-degenerate near-zeros): stack
$U=[\phi_i, F\phi_i]$, rank $2|Q|$; same formula with a $(2|Q|)\times(2|Q|)$ inner solve. Still tiny.

## What this buys / why it beats outer Krylov deflation here

1. Removes the near-null outlier of $M0\,D_\text{int}$ at the SOURCE -> every RHS and every mass benefits
   without the Krylov space having to re-discover/hold the $k$ modes each solve ("refine, not find" made
   structural: the mode is baked into $M0$, not rediscovered).
2. Cleanly justifies **direct $\Omega$-optimization of $\|1 - M0_\text{new} D_\text{int}\|$ restricted to
   the $Q$-complement** -- the topological block is handled exactly by the correction, so the frame is
   optimized only where $F$ is the model. (This is the justification Nobu asked for earlier.)

This is a spectral / deflation preconditioner (Woodbury low-rank spectral correction); standard family
(e.g. deflated/augmented preconditioning, Giraud-Gratton; Nicolaides coarse correction). Cite on
implement.

## RESOLVED design points

1. $P$ built from the **4D** eigenset of $D_W[U_\Omega]$; correction at the kernel level; 5D preconditioner
   via the affine lift + existing exact 5th-direction block inverse. (No 5D eigensolve.)
2. Left vector FREE: $\chi = \gamma_5\phi$ (for real $\lambda$, from $\gamma_5$-Hermiticity). No two-sided Arnoldi.
3. Correction computed ONCE per config, mass-independent; per-mass = cheap $M0(m)$ re-lift.

## THE open question -- requires a test

Which 4D mode set feeds $P$?

- **(a) Non-Hermitian eigenset of $D_W[U_\Omega]$**: $D_W\phi = \lambda\phi$ (complex $\lambda$; topological one
  near-real-$0$), $\chi = \gamma_5\phi$. The projector then satisfies the EXACT identity
  $P D_W = D_W P = \lambda P$, so $D_\text{approx}$ is exact-on-the-block. Cost: complex Arnoldi
  (shift-invert near $0$); the $\chi=\gamma_5\phi$ shortcut degrades if $\lambda$ is appreciably complex.
- **(b) Hermitian eigenset of $H_W = \gamma_5 D_W[U_\Omega]$**: $H_W\psi = \mu\psi$, $\mu$ real
  (= right singular vectors of $D_W$, since $H_W^2 = D_W^H D_W$). Cheap + robust: Chebyshev-filtered
  Hermitian Lanczos, no complex arithmetic, $\chi=\psi$ trivially. BUT $\psi$ is NOT an eigenvector of
  the non-normal $D_W$, so $P$ only APPROXIMATELY block-diagonalizes $D_W$ -> $D_\text{approx}$ approximate.

(a) and (b) COINCIDE at an exact zero mode (chirality eigenstate); they DIVERGE for the non-normal
off-zero part. Whether (b)'s cheaper-but-approximate projector removes the outlier as well as (a) is
empirical.

### Discriminating test (fixed config, e.g. 16^4 ckpoint_lat.640, $|Q|=3$)

For a small rank ($|Q|$ modes, plus a couple of near-zeros):
1. Compute set (a): shift-invert Arnoldi of $D_W[U_\Omega]$ near $0$ -> $\{\phi_i,\lambda_i\}$, $\chi_i=\gamma_5\phi_i$.
2. Compute set (b): Chebyshev-Lanczos of $H_W[U_\Omega]$ near $0$ -> $\{\psi_i,\mu_i\}$.
3. Build $M0_\text{new}$ (Woodbury) each way; MEASURE:
   - the magnitude of the smallest few eigenvalues of $M0_\text{new}\,D5$ (how well the outlier collapses),
     via the existing `Test_dwf_m0d5_overlap` shift-invert IRA; AND
   - the GMRES-DR / FGMRES iteration count with $M0_\text{new}$ vs plain $M0$ (the bottom line).
   Winner = whichever collapses the low-mode cluster / cuts iterations more, per unit eigensolve cost.

## Relationship to what we already have

- `Test_dwf_m0d5_overlap` -- reuse to measure how well $M0_\text{new}\,D5$'s outlier collapses (the test metric).
- Need a NEW small 4D eigensolve on $D_W[U_\Omega]$ (a) and/or $H_W[U_\Omega]$ (b). The frame $U_\Omega$
  (flow + Landau, prerot-folded) is already produced by the production driver -> reuse the flow cache.
- `Test_dwf_svddump_deflprec` ($D5^\dagger D5$ singular vectors) is the 5D analogue -- not what we need;
  the 4D $H_W$ route is the cheaper cousin.
