# Deflation of the |D_DW|^2 low modes — GMRES vs CGNE (config 640)

> STATUS NOTE (2026-09-03): the M0^{-1} D_DW shift-invert spectrum run (WHICH=m0d) was TOO SLOW
> (GMRES inner solve at sigma=0) -> Nobu stopped it -> SHELVED. This deflation experiment is the pivot.
> Nobu's framing: deflate the ~zero mode of the SQUARED operator |D_DW|^2 = M^dagger M; it is a
> SPECTRUM TRANSFORMATION (lift the low eigenvalues), analogous to what M0 does. So the operator to
> deflate is the squared/bare D_DW (fork #1 RESOLVED -> bare, not preconditioned).

## Spectrum transformation (the deflation, stated explicitly)

$A = M^\dagger M$ (Hermitian PD) with the $k$ lowest eigenpairs $(\lambda_i, v_i)$ from IRL. The
low-rank transformed operator
$$
A' = A + \sum_{i=1}^{k}(\mu-\lambda_i)\,v_i v_i^\dagger
$$
moves each $\lambda_i \to \mu$ and leaves the orthogonal complement untouched, so the condition number
seen by CG drops from $\lambda_{\max}/\lambda_1$ to $\lambda_{\max}/\lambda_{k+1}$. Target $\mu$: use
$\mu=\lambda_{\max}$ (push the deflated modes to the TOP; standard) — or $\mu=$ any value $\ge\lambda_{k+1}$;
report is insensitive as long as $\mu\gg\lambda_1$. This IS deflated-CG (Saad 2000); the recovered true
solution of $A x = M^\dagger b$ uses the deflated-CG split $x=x_{\rm defl}+x_\perp$,
$x_{\rm defl}=\sum_i (v_i^\dagger M^\dagger b/\lambda_i)v_i$. The $D_{DW}$-level image (shift the small
SINGULAR values $\sigma_i\to\sqrt{\mu}$) is the SVD-deflated GMRES below.

## Physics / goal

On the hard config traj 640 (16^4 Iwasaki b2.6, |Q|=3) the sv diagnostic found **exactly 3**
topology-tied low modes of $|D_{DW}|^2$ (singular triplets at $\sigma\approx 0.123\approx m$,
chi_q$\approx+1$, one wall). These 3 stragglers are the residual difficulty the free-limit
preconditioner cannot remove. Question: if we **deflate** those 3 modes, how much does the solve
accelerate, and **which outer solver — GMRES or CGNE — benefits more**?

Metric (project standard): honest $D_W$-apply count = number of Mobius (D_DW) applies to reach a
fixed relative residual (CGNE = 2 applies/iter via $M$ and $M^\dagger$; GMRES = 1 apply/iter on
$D_{DW}$). Count with an apply-counter on the operator wrapper.

## Algorithm sources (cite in code + here)

- Deflated CG (project out an invariant subspace of the Hermitian PD normal operator):
  Saad, Yeung, Erhel, Guyomarc'h, "A deflated version of the conjugate gradient algorithm", SIAM J.
  Sci. Comput. 21 (2000) 1909. Lattice near-zero-mode deflation context: Stathopoulos & Orginos,
  "Computing and deflating eigenvalues while solving multiple RHS ... in lattice QCD", SIAM J. Sci.
  Comput. 32 (2010) 439 (eigCG).
- SVD/singular-subspace deflation for the non-Hermitian GMRES solve (truncated-SVD correction +
  projected operator): parallels the deflated-CG projector; related to Morgan, "GMRES with deflated
  restarting" (GMRES-DR), SIAM J. Matrix Anal. Appl. 24 (2002) 20 — but here the deflated subspace is
  the SINGULAR subspace, not a Ritz/eigen subspace (D_DW is gapped, no small eigenvalues).

## The deflation subspace

From Chebyshev-accelerated IRL on $|D_{DW}|^2 = M^\dagger M$ (exactly as run_sv): the $k=3$ lowest
right singular vectors $v_i$ ($M^\dagger M v_i = \lambda_i v_i$, $\sigma_i=\sqrt{\lambda_i}$), and the
matching left singular vectors $u_i = M v_i/\sigma_i$ ($M M^\dagger u_i=\lambda_i u_i$). Orthonormalise
each set (they should already be ~orthonormal; degenerate triplet -> re-Gram-Schmidt to be safe).

$V=[v_1,v_2,v_3]$, $U=[u_1,u_2,u_3]$, so $M v_i=\sigma_i u_i$ and $M^\dagger u_i=\sigma_i v_i$.

## The four solves (same rhs $b$, Z2 random on FGrid; tol 1e-8)

Solve $D_{DW} x = b$ four ways, all to the same relative residual, count Mobius applies:

- **A. CGNE plain**: CG on $M^\dagger M x = M^\dagger b$. (baseline)
- **B. CGNE deflated** (Saad 2000): init in the subspace exactly,
  $x_0=\sum_i \tfrac{v_i^\dagger M^\dagger b}{\lambda_i}v_i$, then CG on the deflated normal operator
  with projector $P = I-\sum_i v_i v_i^\dagger$ applied to residuals; add $x_0$ back.
- **C. GMRES plain**: GMRES on $D_{DW}$, rhs $b$. (baseline)
- **D. GMRES deflated** (SVD-deflation, mirror of B): exact singular-subspace correction
  $x_{\rm defl}=\sum_i \tfrac{u_i^\dagger b}{\sigma_i}v_i$; then GMRES on the projected operator
  $(I-UU^\dagger)\,D_{DW}\,(I-VV^\dagger)$ with rhs $(I-UU^\dagger)b$, giving $x_\perp\perp V$; add
  $x_{\rm defl}$. The projected operator has the 3 smallest singular values removed -> better
  conditioned.

Report: A vs B (CGNE speedup), C vs D (GMRES speedup), and B vs D (which deflated solver wins).

## Files

- NEW `tests/solver/Test_dwf_deflate_claude.cc` — dedicated file (Nobu prefers dedicated over folding
  into the spectrum driver). Reuses the run_sv IRL setup (MdagMLinearOperator + Chebyshev + IRL, odd
  order, auto cheb_hi via PowerMethod) to get $V$; forms $U$; runs the 4 solves with an apply-counting
  operator wrapper. CLI: --config, --k (default 3), --tol (1e-8), --gmres_restart, the cheb_* / sv_*
  knobs, and --which_solves (default all four).
- Build: extend `scripts_nm/grid_spectrum_build_scc_gpu_claude.sh` pattern -> a
  `grid_deflate_build_scc_gpu_claude.sh` (same grid-config link recipe, build_merged).
- Run: `scripts_nm/run_deflate_claude.sh` (modules + libcublas LD_LIBRARY_PATH + tee to log/).

## Efficiency (the sv/IRL step is the cost — don't recompute)

The 3 low modes are the expensive part. Compute them ONCE (Chebyshev IRL, fp32) and reuse for all
four solves in one binary run. Optionally save $v_i$ (ScidacWriter) so re-runs skip the IRL entirely.
The solves themselves are cheap ($\kappa(A)\sim\lambda_{\max}/\lambda_1\sim 133/0.015\sim 9000$ ->
plain CGNE ~sqrt(kappa)~95 iters; deflated ~ much less). Keep everything fp32 (F/M0 already are).

## Resolved / open

1. RESOLVED (Nobu): deflate the SQUARED operator |D_DW|^2 (bare $D_{DW}$), not $M_0 D_{DW}$.
2. **Confirm** the GMRES-side deflation = the SVD-deflation (solve D), i.e. the $D_{DW}$-level image of
   the $A'$ spectrum transform — vs. augmented/recycled GMRES-DR that rebuilds its own Ritz subspace.
   (Recommend SVD-deflation: it's the exact mirror of the |D_DW|^2 transform, apples-to-apples with
   CGNE.)
3. $\mu$ target for the transform: default $\mu=\lambda_{\max}$ (report insensitive). OK?
