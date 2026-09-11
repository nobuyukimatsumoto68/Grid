# Topological-corrected free preconditioner -- implementation plan

Design + derivation: `topological_precond_correction_claude.md` (read first).
Notation there is Nobu's: $F = D_W(U{=}1)$ operator, $M0 = F^{-1}$ preconditioner (code
`FreeMobius5DInverse`), $D_\text{int} = D_W(U_\Omega)$ the 4D framed Wilson kernel at fixed $M5$.

## Goal (one line)

Kill the near-null outlier of $M0\,D5[U_\Omega]$ by correcting the 4D free kernel $F$ on the
would-be-topological block: $D_\text{approx} = QFQ + \lambda P$, $P=\phi\chi^H$, $\chi=\gamma_5\phi$,
so $M0_\text{new}=D_\text{approx}^{-1}$ (Woodbury) has $M0_\text{new}D5 \approx I$ including on $\phi$.
One 4D eigensolve per config, mass-independent; the win transfers to every mass via the exact
5th-direction block inverse.

## Algorithm sources (cite in code + here, prominently)

- Sherman-Morrison-Woodbury low-rank inverse; deflated/augmented preconditioning
  (A. Gaul, M. Gutknecht, J. Liesen, R. Nabben, "A framework for deflated and augmented Krylov
  subspace methods", SIAM J. Matrix Anal. 34 (2013) 495; R.A. Nicolaides, SIAM J. Numer. Anal. 24 (1987) 355).
- Would-be-zero modes = low modes of the 4D Wilson kernel at the wall height $H_W(-M5)=\gamma_5 D_W(-M5)$
  (overlap-kernel low modes: R.G. Edwards, U.M. Heller, R. Narayanan, Nucl.Phys. B540 (1999) 457;
  H. Neuberger, Phys.Lett. B417 (1998) 141).
- Chebyshev-filtered implicitly restarted Lanczos: Grid `ImplicitlyRestartedLanczos` + `Chebyshev`.
- Shift-invert Arnoldi (non-Herm, largest-modulus of the shift-inverted operator): as already used in
  `Test_dwf_m0d5_overlap_claude.cc`.
- $\gamma_5$-Hermiticity of Wilson $D_W^H = \gamma_5 D_W \gamma_5$ (left vector free: $\chi=\gamma_5\phi$).

## The two candidate mode sets (the thing under test)

- (a) NON-Herm eigenset of $D_W(-M5)[U_\Omega]$: $D_W\phi_i=\lambda_i\phi_i$ near $0$, $\chi_i=\gamma_5\phi_i$.
  Exact block-diagonalization $P D_W = D_W P = \lambda P$. Cost: complex shift-invert Arnoldi.
- (b) Herm eigenset of $H_W(-M5)^2 = D_W^H D_W$ (= singular vectors of $D_W$): $\psi_i$, $|\mu_i|$, sign
  $\mu_i = $ chirality. Cheap Chebyshev-Lanczos, $\chi=\psi$. Projector only APPROXIMATELY diagonalizes
  the non-normal $D_W$. Coincides with (a) at an exact zero mode; diverges on marginal near-zeros.

## Test config

$16^4$ `configs_iwasaki_16_b2.6/ckpoint_lat.640`, $|Q|=3$ (already characterized). Frame = production
(flow $s/t0=6$ i.e. eps 0.02 x nstep 873, FA-Landau $\alpha=0.1/16$, gf_maxit 4000), prerot-folded;
reuse the flow cache in `flowcache/`.

## Chunk 1 findings + method decisions (2026-09-10, on-node, Nobu-confirmed)

- **Operator = MASSLESS 4D Wilson `D_W(0)` on the gauge-rotated ROUGH config `Urot = Omega U Omega^dag`**
  (NOT the smoothed/flowed config). The rough config is what we actually precondition, so its
  would-be-topological modes are the right ones -- even though they are "melted" (see below). Config
  NUMBER is irrelevant (240 vs 640 same method).
- **Masses are irrelevant to the modes.** `D_W(m) = D_W(0) + m` is a pure additive shift -> eigenVECTORS
  are mass-independent; only eigenVALUES move. So no mass tuning; the `-M5` shift was wrong only because
  I then took SINGULAR values of the shifted op (shift-DEPENDENT) which picked the bulk/mobility-edge.
- **Real eigenvalues of `D_W` carry the topology.** gamma5-Herm `D_W` -> complex eigenvalues come in
  conjugate pairs with `chi ~ 0` (non-topological); the REAL eigenvalues have definite-sign chirality =
  the would-be-topological modes. Index = sum of `sign(chi)` over the real eigenvalues (NOT a `|chi|`
  magnitude cut).
- **Left vector free:** for a real `lambda`, `chi_left = gamma5 phi` EXACTLY (`D_W^H gamma5 phi = gamma5
  D_W phi = lambda gamma5 phi`), for ANY chirality magnitude. So `P = phi (gamma5 phi)^H`, biorthonormal
  `chi^H phi = <phi|gamma5|phi>` (= the chirality, e.g. 0.6 rough / 0.95 smooth).
- **Eigensolver: DIRECT shift-invert IRA on `D_W` (sigma=0).** GMRES inner `D_W^{-1}` is fine (`D_W(0)`
  well-conditioned, `sigma_min ~ 0.6-0.7`); CGNE inner + `B^dag D_W B` RR-cleanup is an equally-valid more
  robust variant (peer thread's winner). ABANDONED: Rayleigh-Ritz in the Hermitian singular subspace --
  it scrambles the `+-sigma`-degenerate chirality partners into non-chiral combinations (resid ~ 0.25).
- **Rough vs smooth:** peer thread on a 4 t_0-reflowed SMOOTH config got clean `|Q|=3` (3 real modes,
  `chi ~ 0.94/0.96/0.93` at `|lambda| ~ 0.15-0.19`) -- a clean DEMONSTRATION. On our rough `Urot` the
  same modes sit at `|lambda| ~ 0.70` with `chi ~ 0.55-0.67` (melted by roughness + additive renorm), and
  THAT is the correct set for the correction. Test both 240 and 640.

## Ordered chunks

### Chunk 1 -- 4D would-be-zero-mode eigensolver + (a)/(b) divergence report  [DECISION DATA]
Standalone; independently valuable (decides a-vs-b and validates the premise) with NO Woodbury yet.
- Build frame $U_\Omega$ (reuse the frame block + flow cache from `Test_dwf_freeprec_claude.cc`).
- 4D operators on $U_\Omega$: `WilsonFermionD DW(U_Omega, ..., mass = -M5)`; $H_W = \gamma_5 D_W$.
- (a) shift-invert Arnoldi of `DW` near $0$ (reuse the `ShiftInvertB` + IRA idiom from
  `Test_dwf_m0d5_overlap_claude.cc`, operator = `DW` not `M0 D5`) -> $\{\phi_i,\lambda_i\}$, set
  $\chi_i=\gamma_5\phi_i$, scale $\chi_i^H\phi_i=1$.
- (b) Chebyshev-IRL of `MdagMLinearOperator(DW)` ($=D_W^H D_W$) for smallest $|\mu|$ -> $\{\psi_i,\mu_i^2\}$;
  $\mu_i$ sign from $\psi_i^H H_W \psi_i$.
- REPORT (the decision table): $|\lambda_i|$ vs $|\mu_i|$; chirality $\langle\phi_i|\gamma_5|\phi_i\rangle$
  (near $\pm1$ => topological, clean; away => marginal, where a/b diverge); overlaps
  $|\langle\phi_i|\psi_j\rangle|$; biorthogonality residual $\|D_W\phi_i-\lambda_i\phi_i\|$,
  $\|\chi_i^H D_W - \lambda_i\chi_i^H\|$.
- Dump both sets (BinaryIO, LIME-free) for chunks 2-4.
- Files: `tests/solver/Test_dwf_wtop_eigs_claude.cc` (new); `tests/solver/twolevel_common_claude.h`
  (add a 4D `DW[U_Omega]` builder + a `report_chirality4d` helper, beside existing);
  `scripts_nm/grid_wtop_eigs_build_scc_gpu_claude.sh`, `scripts_nm/grid_wtop_eigs_gpu_qsub_claude.sh` (new).

### Chunk 2 -- premise check: do the 4D modes ARE the 5D outliers?  [GO/NO-GO]
- Reuse `Test_dwf_m0d5_overlap_claude.cc`: it already gets the low modes of $M0\,D5[U_\Omega]$
  (shift-invert IRA). Add: read the chunk-1 4D $\phi_i$, lift to 5D (wall embedding $P_\pm$ on
  $s=0,L_s{-}1$), and report $\|P_\text{4Dlift}\, w_j\|^2$ for each 5D outlier $w_j$ (how much of each
  5D near-null mode is spanned by the lifted 4D topological modes).
- GO iff the 5D outliers are well-covered by the lifted 4D modes (premise holds). If not, the 4D-kernel
  correction cannot remove them -> stop and rethink (maybe need a few marginal modes, or 5D data).
- Files: extend `tests/solver/Test_dwf_m0d5_overlap_claude.cc` (guarded new block, preserve original);
  reuse its qsub.

### Chunk 3 -- the corrected preconditioner $M0_\text{new}$ (Woodbury)  [THE BUILD]
- New class `FreeLimitPreconditionerTopo` in `Grid/qcd/utils/FreeMobius5D_claude.h`, ADDED BESIDE the
  existing `FreeLimitPreconditioner` (original untouched). Holds: base `FreeMobius5DInverse& F` (= $M0$),
  the 4D modes $\{\phi_i,\chi_i,\lambda_i\}$, and per-mass Woodbury data.
- Apply: `out = M0 in - M0 U (I + V^H M0 U)^{-1} V^H M0 in` where $U,V$ are the 5D lift of the 4D
  correction $C=D_\text{approx}-F$ through the Mobius affine structure $D5=A(m)\otimes D_W+B(m)$.
  Precompute per mass: assemble the lift columns (Mobius $s$-coupling $A(m)$ applied to
  $\{\phi_i, F\phi_i\}$ across $s$-slices with $P_\pm$), apply $M0(m)$ to each, invert the small inner
  matrix once. Per-apply overhead = $M0$ + rank-$(2|Q|\cdot\text{s-support})$ BLAS1.
- KEY SUBTLETY (resolve on-node with chunk-1 data in hand): the exact 4D->5D affine lift. Two routes,
  pick after inspecting the modes:
  (i) matrix-free $\Delta D5\,v := D5[U_\Omega]v - D5[\text{free}]v$ RESTRICTED to the low-rank 4D image
      (apply the Mobius hopping only on the $\{\phi_i\}$ 4D content) -- avoids hand-deriving $A(m)$;
  (ii) explicit $A(m)$ $s$-structure from the Mobius $b_s,c_s,P_\pm$ (cleaner, needs the algebra).
  Start with (i) for correctness, optimize to (ii) if hot.
- Files: `Grid/qcd/utils/FreeMobius5D_claude.h` (new class beside original);
  `tests/solver/Test_dwf_freeprec_topo_claude.cc` (driver: frame -> read 4D modes -> build $M0_\text{new}$
  -> solve).

### Chunk 4 -- measure the win  [PAYOFF]
- With $M0_\text{new}$: FGMRES and GMRES-DR iteration count vs plain $M0$, and vs external Krylov
  deflation (the $w_o$ scan baseline). Run BOTH mode sets (a) and (b) from chunk 1 -> the a/b verdict.
- Re-measure outlier collapse: smallest eigenvalues of $M0_\text{new} D5$ via `Test_dwf_m0d5_overlap`.
- Mass scan to confirm the single 4D correction transfers ($m=0.1,0.01,0.001$).
- Files: reuse `Test_dwf_freeprec_topo_claude.cc` + `Test_dwf_m0d5_overlap_claude.cc`.

## Open questions (resolve with Nobu, mostly on-node after chunk 1)

1. Number of modes: $|Q|=3$ topological + how many marginal near-zeros? Read from the chunk-1 spectrum
   (the chirality column tells us where clean topological ends and marginal begins).
2. a vs b: settled empirically by chunk 1 (divergence/overlap) + chunk 4 (iterations). Prior: (b) suffices
   for the clean $|Q|$ modes (chirality $\Rightarrow \chi=\gamma_5\phi\approx\pm\phi$, projector already
   near-Hermitian there); (a) only needed if marginal modes matter.
3. Correct toward exact $\lambda$ (block $=\lambda P$) vs toward $D_\text{int}$'s Rayleigh action
   $\lambda_i=\chi_i^H D_W\phi_i$ -- same for the true eigenpair (a); differ for (b). Use the Rayleigh
   quotient (robust for both).
4. The 4D->5D affine lift implementation (chunk 3 route i vs ii).
5. Complex $\lambda_i$ in (a): if appreciably complex the $\chi=\gamma_5\phi$ shortcut degrades -> flag
   from chunk-1 $\text{Im}\,\lambda_i$; fall back to a genuine left-Arnoldi only if needed.

## What we reuse (no new frame/eigensolver plumbing from scratch)

- Frame + flow cache: `Test_dwf_freeprec_claude.cc` block (flow, FA-Landau, prerot).
- Shift-invert IRA idiom + `report_chirality`/dump helpers: `Test_dwf_m0d5_overlap_claude.cc`,
  `twolevel_common_claude.h`.
- Metric (outlier collapse): `Test_dwf_m0d5_overlap_claude.cc`.
