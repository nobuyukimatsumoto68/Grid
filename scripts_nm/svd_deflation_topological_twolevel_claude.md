# Two-level solve: SVD-deflate the would-be-topological modes, then $M_0 D_5$-precondition the rest

Dedicated design note (Nobu 2026-09-09). Sibling docs: [[twolevel_deflation_check_impl_plan_claude.md]]
(the running check), [[deflate_impl_plan_claude.md]] (the singular-subspace deflation origin),
[[gmres_deflation_impl_plan_claude.md]] (GMRES-DR), [[freeprec_future_directions_claude.md]] Dir. 2.
Memory: [[project-deflation-gmres]].

## The idea in one line

Identify the would-be-topological low modes of $D_5$, **pre-solve them exactly without any
preconditioner** (closed-form singular-subspace inverse), then solve the remainder with the free-limit
$M_0 D_5$ preconditioner. The topological modes are exactly the ones $M_0$ structurally cannot fix, so
handing them to an exact deflation and letting $M_0$ handle the rest is the natural division of labour --
and it makes direct-$\Omega$ optimization of $\lVert 1 - M_0 D_5\rVert$ **in the deflated complement**
a well-posed target.

## Physics: where the topological modes are, and why $M_0$ fails on them

For domain-wall fermions the would-be topological zero modes are **surface modes bound to the fifth-
dimension walls** ($P_+$ on $s=L_s-1$, $P_-$ on $s=0$), decaying into the gapped bulk. They sit at the
**bottom of the $D_5$ spectrum, at singular value $\sigma \approx m$** (not at zero -- the quark mass is
the floor). On config 640 ($16^4$ Iwasaki b2.6, $m=0.1$, Shamir $b{=}1.5,c{=}0.5,M_5{=}1.8,L_s{=}8$):
the three lowest singular triplets are at $\sigma \approx 0.123 \approx m$ with $\chi_q \approx +1$ (same
sign $\Rightarrow Q=3$); the rest of the low cluster ($\sigma \approx 0.125$--$0.18$) are non-topological
$\pm$ near-pairs.

$M_0 = \Omega^\dagger F \Omega$ is **translation-invariant** (FFT-diagonal free kernel in the frame), so it
is tuned for delocalized bulk/plane-wave modes and is blind to the sharply wall-localized topological
modes. They are the $\mathrm{Re}\approx 0.3$ stragglers of $\mathrm{spec}(M_0 D_5)$ -- the residual
difficulty the frame cannot remove, and the source of the restart stall / the high-$|Q|$ degradation of
the win.

## Singular vs eigen: what we have, and why singular is the right object

`compute_low_modes` gives the lowest eigenvectors of $D_5^\dagger D_5$ -- i.e. the **right singular
vectors** $v_i$ of $D_5$ ($D_5^\dagger D_5\, v_i = \sigma_i^2 v_i$), NOT eigenvectors of $D_5$. Möbius
$D_5$ is non-normal (and not even $\gamma_5 R_5$-Hermitian for $c\neq 0$), so singular $\neq$ eigen. We
use the SINGULAR subspace on purpose:

- $M_0$ is invertible and well conditioned, so $M_0 D_5\, x \approx 0 \iff D_5\, x\approx 0$: the near-zero
  RIGHT eigenvectors of $M_0 D_5$ ARE (approximately) the small right singular vectors $v_i$ of $D_5$.
  Moreover $|\lambda_{\min}| \le \sigma_{\min}$, so **every** small-eigenvalue direction lies inside the
  small-singular subspace $\mathrm{span}\{v_i\}$. Deflating it removes the stalling directions; at worst
  we harmlessly over-deflate.
- The singular route is exact and cheap: it uses only the Hermitian $D_5^\dagger D_5$ modes (robust IRL),
  never the ill-conditioned eigenvectors of the non-normal operator.

## The scheme (exact)

Left singular vectors from the right ones, $k$ applies of $D_5$:
$$
u_i = D_5 v_i / \sigma_i,\qquad D_5 v_i = \sigma_i u_i,\quad D_5^\dagger u_i = \sigma_i v_i .
$$
(The $u_i$ are orthonormal automatically: $u_i^\dagger u_j = v_i^\dagger D_5^\dagger D_5 v_j/(\sigma_i
\sigma_j) = \delta_{ij}$, since $V$ are orthonormal eigenvectors of $D_5^\dagger D_5$.) Projectors
$P_V = 1 - VV^\dagger$, $P_U = 1 - UU^\dagger$.

Solve $D_5 x = b$:

1. **Pre-solve the topological subspace, NO preconditioner** (this is the whole ill-conditioned part,
   the $1/\sigma$ amplification):
$$
x_\text{lo} = \sum_{i\le k} \frac{u_i^\dagger b}{\sigma_i}\, v_i = V\,\Sigma^{-1} U^\dagger b .
$$
   Check: $D_5 x_\text{lo} = \sum_i (u_i^\dagger b)\,u_i = U U^\dagger b$ -- exactly the part of $b$ in
   $\mathrm{span}\{u_i\}$.
2. **$M_0$-GMRES on the deflated complement**: with $b_\perp = P_U b$, solve
$$
\hat A\, x_\perp = b_\perp,\qquad \hat A \equiv P_U\, D_5\, P_V ,
$$
   right-preconditioned by $M_0$; $\hat A$ has smallest singular value $\sigma_{k+1}$ (well conditioned)
   and $M_0 \approx D_5^{-1}$ precisely on this complement.
3. $x = x_\text{lo} + P_V x_\perp$.

**Exactness** (needs only the SVD, no eigenvectors): $D_5=\sum_j\sigma_j u_j v_j^\dagger$ maps $V^\perp \to
U^\perp$, so for $x_\perp\perp V$ we have $D_5 x_\perp\in U^\perp$ and $\hat A x_\perp = D_5 x_\perp$; the
low/complement blocks decouple and $D_5 x = b$ to the GMRES tolerance (floored only by the IRL accuracy of
the $v_i$). $\hat A$ is invariant under $P_V$ on input, so the final $P_V x_\perp$ projection does not
disturb the residual.

This is the SVD-deflation of [[deflate_impl_plan_claude.md]] "solve D", now with $M_0$ preconditioning the
complement.

## Why this justifies direct-$\Omega$ opt of $\lVert 1 - M_0 D_5\rVert$ in the deflated space

Because $M_0$ (translation-invariant) cannot represent the wall-localized topological modes, minimizing
$\lVert 1 - M_0 D_5\rVert$ over the FULL space forces the frame to chase modes it structurally cannot fix.
Once those $k$ modes are handed to the exact deflation (step 1), the frame's only job is the COMPLEMENT:
$$
L_P[\Omega] = \sum_v \big\lVert P\,(M_0(\Omega) D_5 - 1)\,P\, v\big\rVert^2,\qquad P = P_V ,
$$
which $M_0$ CAN meet. So "frame B" (deflated-complement direct-$\Omega$ opt, in
[[FrameOptimizerDWF_claude.h]]) becomes well-posed, and frame A (full-space) vs frame B isolates exactly
whether removing the topological modes from the objective lets the frame push the remaining spectrum
closer to 1 (lower $C$ on modes $k$--24, fewer complement-GMRES iters).

## Contrast with the failed frozen-seed attempt

The earlier `gmresdr` stage FROZE the $v_i$ into GMRES-DR's harmonic-Ritz recycle
(`SetDeflationSubspace(Vfull, true)`) -> non-convergent (j7510214: 20000 iters, res 0.034). Wrong on two
counts: (i) GMRES-DR's recycle expects approximate EIGENvectors of $B=A M_0$, but $v_i$ are singular
vectors of $D_5$; (ii) freezing killed the adaptation that gives the working 192-iter run. The SVD scheme
above is the correct use of the same $v_i$.

## Implementation status (2026-09-09)

Wired as `--stage svddefl` in `Test_dwf_twolevel_claude.cc`:
- `project_complement()` (orthogonal-complement projector, `axpy`-based) and `class DeflatedD5Op`
  ($\hat A = P_U D_5 P_V$, `LinearOperatorBase`).
- `do_svd` block: for each $k$ in `--svd_klist` (default `3,10`) build $U$, pre-solve $x_\text{lo}$,
  $M_0$-GMRES(16,32) on $\hat A$, recombine, and print iters + **true rel.res of $D_5 x=b$** (the
  correctness gate). Same fixed-seed source $b$ as RB-CGNE -> apples-to-apples.
- `do_gmres` (the frozen-seed version) kept opt-in (`--stage gmresdr`) for reference; dropped from
  `all`/`precond`.
- Deflation vectors $v_i$ checkpointed (`deflfull_*`) for offline analysis (LIME-free BinaryIO).
- Baseline to beat: adaptive GMRES-DR(16,32) on bare $D_5$ = **192 iters** (Landau frame, m=0.1, cfg 640).

Build: `grid_twolevel_build_scc_gpu_claude.sh`. Run: `grid_twolevel_gpu_qsub_claude.sh` with
`PIPELINE=1 STAGE=svddefl` (or `precond`).

## Open questions / next steps

1. Does SVD-defl($k{=}3$) already beat 192, or is $k{=}10$ needed? (The low cluster runs to $\sigma\sim
   0.18$; the 3 topological modes may not be the whole difficulty.)
2. Compare against deflated RB-CGNE (rbcgne stage, RB-Schur$^2$ modes) at matched $k$ -- the headline
   apples-to-apples, and its mass-scaling ($m=0.1/0.01/0.001$).
3. True rel.res floor vs IRL accuracy: if it floors near the $10^{-5}$ IRL residual rather than $10^{-8}$,
   tighten `sv_resid` (currently 1e-5) or re-orthonormalize $U$ via the $k\times k$ Gram matrix
   $G=W^\dagger W$, $W_i=D_5 v_i$ (robust to degeneracy; the fallback to the per-mode $\sigma$ formula).
4. Then: run frame B (deflated-complement $\Omega$-opt) and measure $C$ / complement-GMRES vs frame A.
5. Mixed precision for the wall win (chunk 5), and per-config $k$ from $\chi_Q$ once validated.

## Sources

Singular-subspace / deflated GMRES: Saad, "Iterative Methods for Sparse Linear Systems" 2nd ed. ch. 6.5;
Morgan, "GMRES with deflated restarting", SIAM J. Matrix Anal. Appl. 24 (2002) 20. Deflated CG:
Saad, Yeung, Erhel, Guyomarc'h, SIAM J. Sci. Comput. 21 (2000) 1909. DWF wall modes / index:
Kaplan 1992; Furman-Shamir 1995; Edwards-Heller-Narayanan (overlap/DWF zero modes).
