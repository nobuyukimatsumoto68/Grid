# Two-level deflation + Omega-in-deflated-space (design note)

Nobu 2026-09-08 (complementary project; PARKED behind Phase A/B -> mixed precision -> (L,m)-scaling).
Status: DESIGN ONLY, no code. This composes existing pieces; it does NOT change the fundamental
(L,m)-scaling (see [[project-deflation-gmres]], [[project-freeprec-r2]]).

## Goal / the key insight (reopens the frame side)

The free-limit preconditioner $M_0 = \Omega^\dagger F \Omega$ has a residual difficulty = the isolated
near-null cluster of $M_0 D$ (the topological stragglers at $\mathrm{Re}\approx 0.3$; the ~0.38 M0-resid
"floor"). The frame side was declared CLOSED (no $\Omega$ beats the floor) -- BUT that floor is a
NEAR-NULL-MODE floor, NOT a frame property. If we EXPLICITLY deflate those modes, the obstruction is gone
FROM THE COMPLEMENT, so re-optimising $\Omega$ restricted to the bulk is no longer floor-bound. The frame
side was closed only *conditional on the near-null modes being present*. Remove them -> $\Omega$ has room.

## The quality metric C (confirmed 2026-09-08 -- the pivot object)

Per-fermion-vector preconditioner quality (Test_wilson_frameopt_claude.cc:149, opscan_one/opscan_mx):
$$
C(v) = \frac{\lVert (1 - M_0 D)\,v \rVert}{\lVert v \rVert}, \qquad M_0 = \Omega^\dagger F \Omega .
$$
- $C=0$ = perfect ($M_0 D v = v$); on an eigenmode of $M_0 D$ with eigenvalue $\mu$, $C = |1-\mu|$.
- $C \ge 1$ = RED FLAG: the mode is amplified under stationary iteration ($\mathrm{Re}\,\mu<0$ or large
  $\mathrm{Im}\,\mu$).
Frame optimiser cost (fo_loss, :228) is the SAME object summed over probe vectors:
$$
L[\Omega] = \sum_v \lVert (M_0(\Omega) D)\,v - v \rVert^2 = \sum_v C(v)^2 \lVert v \rVert^2 .
$$
Direct-$\Omega$ gradient descent (dwf4 port, :220-) minimises $L$; gate-validated analytic gradient.
WHY the frame side floored: minimising $L$ over the FULL space is DOMINATED by the near-null modes (large
$C$, frame-irrelevant) -> only 1.28x over Landau. The fix below removes those modes from the average.

## The three-layer stack

1. **Explicit Hermitian deflation.** Compute the low modes of the HERMITIAN DW operator $H = \gamma_5 R_5
   D_{DW}$ (real spectrum, clean Lanczos/IRL). Machinery exists: run_sv, IRL on $M^\dagger M$, the G5R5
   idiom (Test_dwf_G5R5.cc). Low-mode space $V_\text{low}$, projector $P = 1 - V_\text{low}V_\text{low}^\dagger$.
2. **Re-optimise $\Omega$ in the DEFLATED (complement) space.** Minimise the SAME frame loss restricted to
   the complement:
   $$
   L_P[\Omega] = \sum_v \lVert P\,(M_0(\Omega) D - 1)\,P\,v \rVert^2 = \sum_v \big(C_P(v)\big)^2 \dots
   $$
   i.e. drive $C$ down on the BULK only (the near-null modes are handled exactly by layer 1). Use the
   existing direct-$\Omega$ norm-opt (dwf4_group_claude.h / the local-agent routine) with $P$ inserted.
3. **Deflated GMRES on top.** GMRES-DR ([[project-deflation-gmres]],
   RecyclingGeneralisedMinimalResidual_claude.h) mops up the RESIDUAL near-null of the framed
   non-Hermitian operator. Project its Krylov off $V_\text{low}$ to avoid double-counting.

## Why TWO deflations are non-redundant

$M_0 D$ is NON-Hermitian: via the $\gamma_5$ trick it is a product of two INDEFINITE Hermitian operators
$(\Omega^\dagger H_\text{free}^{-1}\Omega)\,H$ with $H=\gamma_5 D_{DW}$, so its spectrum is COMPLEX and its
near-null is NOT exactly $H$'s low modes. Layer 1 (Hermitian) cleanly removes the KNOWN near-zero cluster
(cheap, exact, real Lanczos); layer 3 (GMRES-DR, adaptive/harmonic-Ritz) handles the residual COMPLEX
near-null that the Hermitian projector cannot capture. Different objects -> principled, not belt-and-suspenders.

## Cost / reuse / subtleties

- One-off Hermitian eigensolve is the cost, BUT the $H$ low modes are ~mass-robust (topological +
  near-zero) -> reuse across the whole $m=0.1/0.01/0.001$ loop AND along an HMC trajectory (amortises like
  eigCG / production deflation). The mode COUNT is the (L,m)-scaling number (Banks-Casher $\rho(0)V$) --
  the fundamental measurement directly SIZES this method.
- Avoid double-counting: GMRES-DR (layer 3) must deflate the COMPLEMENT of $V_\text{low}$ (project Krylov
  off $V_\text{low}$), else it wastes $k$ re-finding modes already removed.
- Projector consistency: apply $P$ consistently to the $M_0 D$ ACTION; the $\gamma_5$ trick gives the map
  between the $H$-eigenbasis and the $M_0 D$ deflation.
- Complementary "shrink-k" mass knobs (m_prec/M5_prec/M_shift, [[project-deflation-gmres]]) also reduce
  the near-null count and thus the size of $V_\text{low}$.

## Machinery to reuse (this is mostly composition)

- Hermitian spectrum: run_sv / IRL(MdagM) / G5R5 (Test_dwf_spectrum_transform_claude.cc,
  ImplicitlyRestartedLanczos, Test_dwf_G5R5.cc idiom).
- Direct-$\Omega$ optimiser: dwf4_group_claude.h + fo_loss/fo_loss_force/fo_descend
  (Test_wilson_frameopt_claude.cc), add the projector $P$.
- Deflated GMRES: RecyclingGeneralisedMinimalResidual_claude.h (project Krylov off $V_\text{low}$).
- Quality assessment: C(v) above (opscan_one/opscan_mx) -- the diagnostic AND the optimiser target.

## Sources
GMRES-DR: Morgan SIMAX 24 (2002) 20. GCRO-DR: Parks/de Sturler SISC 28 (2006) 1651. eigCG / Hermitian
low-mode deflation for lattice: Stathopoulos-Orginos 2007 (arXiv:0707.0131); spectral density /
Banks-Casher: Banks-Casher 1980.
