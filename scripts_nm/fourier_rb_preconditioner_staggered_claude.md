# Fourier (free-limit) preconditioner combined with red-black preconditioning — staggered as the clean case

Handoff note (Nobu, 2026-09-10). Owner: the new dedicated thread. Origin: the DWF free-limit
preconditioner line (memory [[project-freeprec-r2]], [[project-deflation-gmres]]); design sibling
[[svd_deflation_topological_twolevel_claude.md]]. This note is self-contained.

## The question

We have a "Fourier" preconditioner $M_0 = \Omega^\dagger F \Omega$: $\Omega$ is the Landau-fixed frame of a
Wilson-flowed config, and $F = D_\text{free}^{-1}$ is the FREE Dirac inverse, which is diagonal in
momentum space (built by FFT). Separately, production solves use red-black (even-odd) preconditioning:
Schur-complement onto one checkerboard and run CG there. **Can the Fourier preconditioner be combined with
red-black preconditioning — cleanly via FFT — especially on the staggered lattice?**

Answer: yes. The free RB-Schur operator is still translation-invariant, hence Fourier-diagonal, so the two
compose. Staggered is the cleanest case because it has no spin index and its even-odd Schur complement is a
scalar Laplacian-type operator.

## General mechanism (any Wilson-type or staggered kernel)

Even-odd block form
$$
M = \begin{pmatrix} M_{ee} & M_{eo} \\ M_{oe} & M_{oo} \end{pmatrix},
\qquad S_c = M_{cc} - M_{c\bar c} M_{\bar c\bar c}^{-1} M_{\bar c c}
$$
is the Schur complement on checkerboard $c$ (the operator RB-CG runs on). The block-inverse identity gives
the crucial simplification
$$
S_c^{-1} = \big(M^{-1}\big)_{cc}.
$$
Therefore the **free** RB-Schur inverse is exactly the **checkerboard block of the free full inverse**
$F = D_\text{free}^{-1}$, which we already build by FFT:
$$
S_{c,\text{free}}^{-1} = \big(D_\text{free}^{-1}\big)_{cc} = P_c\, F\, P_c ,
$$
with $P_c$ the checkerboard projector. Since $\Omega$ is a site-local gauge rotation it commutes with
$P_c$, so the combined **Fourier + RB preconditioner** is just the existing $F$, framed and projected:
$$
\boxed{\,M_0^{\rm rb} = P_c\,\Omega^\dagger F\,\Omega\,P_c\,}
$$
applied to the RB-Schur system that RB-CG(NE) solves. No new FFT machinery — reuse $F$, restrict I/O to one
checkerboard. This clusters the RB-Schur spectrum near 1 (as $M_0$ does for the full operator) on top of the
factor-$\sim2$ conditioning that red-black already buys.

## Why staggered is the cleanest case

Staggered fermions have **no spin index** (one complex component per site) and $D_{ee}=D_{oo}=m$. The
operator staggered CG runs on is the even block of the normal operator,
$$
(D^\dagger D)_{ee} = m^2 - D_{eo}D_{oe},
$$
a **scalar** (Laplacian-type) operator. On the free lattice its momentum symbol is scalar,
$$
\widehat{(D^\dagger D)_{ee}}(p) = m^2 + \sum_{\mu}\sin^2 p_\mu
$$
(up to the staggered phase bookkeeping $\eta_\mu(x)=\pm1$, which is a position-space sign absorbed cleanly).
So the free staggered RB preconditioner is a **single scalar per momentum** — the FFT carries no spin/$L_s$
tensor structure at all, unlike the Wilson/DWF case. This makes staggered the ideal prototype: minimal
code, exact closed-form free symbol, trivial per-momentum divide.

## Two subtleties

1. **Brillouin-zone folding.** The even sublattice has doubled spacing, so a sublattice FFT folds $p$ with
   $p+\pi$. The block-inverse route above avoids writing a sublattice FFT: use the FULL-lattice FFT for
   $F$ and simply project with $P_c$. (If a genuine sublattice FFT is ever wanted, handle the fold
   explicitly.)
2. **The frame $\Omega$ for staggered.** The "free-limit in a frame" idea should transfer, but staggered's
   gauge structure differs from the Wilson kernel; whether the flowed-Landau frame produces the same
   near-1 clustering for staggered needs to be checked, not assumed.

## Suggested first steps

1. **Free-symbol validation (staggered, no gauge).** On a free staggered lattice, build $(D^\dagger D)_{ee}$,
   confirm it is diagonalized by the FFT with symbol $m^2+\sum_\mu\sin^2 p_\mu$, and that $P_c F P_c$
   inverts it to machine precision. (Bit-level check, like the $4^4$ DWF cross-check that anchored the DWF
   port.)
2. **Interacting, no frame.** On a gauge config, precondition RB-CG for staggered with $P_c F P_c$ (free RB
   inverse, $\Omega=1$) and measure the $D_W$-apply reduction vs plain RB-CG.
3. **Add the frame.** Build $\Omega$ (flowed-Landau) and test $M_0^{\rm rb}=P_c\Omega^\dagger F\Omega P_c$;
   measure the extra clustering / count win, and the flowed-fixed Landau functional per config.
4. **Map back to Wilson/DWF** if promising: same $P_c F P_c$ construction, but $F$ carries spin+$L_s$
   (reuse `Grid/qcd/utils/FreeMobius5D_claude.h`); this is a Fourier-preconditioned RB-CGNE for the DWF line
   and could reduce how much RB-Schur$^2$ deflation the two-level scheme needs (see
   [[svd_deflation_topological_twolevel_claude.md]]).

## Metric and conventions (inherit from the DWF line)

$D_W$-apply count to a fixed relative residual (RB-CGNE $= 2 L_s\times$iters, FGMRES $= L_s\times$iters, $M_0$
costs 0 $D_W$). Report the flowed-fixed Landau functional per config. Anti-periodic time BC. For the DWF
mapping: $L_s=8$, $M_5=1.8$, Shamir $b=1.5,c=0.5$.

## Sources

Even-odd / Schur preconditioning: DeGrand-Rossi; Grid `SchurRedBlack.h`. Free-limit Fourier preconditioner
+ frame: the DWF line (`FreeMobius5D_claude.h`, memory [[project-freeprec-r2]]). Staggered even-odd
structure: Golterman-Smit; standard staggered CG references. Block-matrix inverse identity
$S_c^{-1}=(M^{-1})_{cc}$: standard (e.g. Saad, "Iterative Methods", Schur-complement section).
