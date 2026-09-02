# Spectrum-transform diagnostic: how $M_0 = \Omega^\dagger F \Omega$ reshapes $\mathrm{spec}(D_\text{DW})$

Implementation plan (step 1: plan only, no code yet). Working repo: `/projectnb/qfe/nmatsum/dwf`,
Grid checkout `/projectnb/qfe/nmatsum/dwf/Grid` (source tree nested at `Grid/Grid/`), branch `dwf_prec`.

## 1. Physics / goal

Quantify HOW the free-limit Mobius preconditioner $M_0 = \Omega^\dagger F \Omega$ (Landau frame,
$\tau = 2$ Wilson flow, $\alpha = 0.00625$; only $M_0$, no $M_1$ correction) transforms the
domain-wall Dirac spectrum. Two spectra:

- **(A) Unpreconditioned near-zero / singular-value spectrum of $D_\text{DW}$.** We probe the smallest
  singular values of the Mobius operator $D_\text{DW}$ via the Hermitian positive-definite
  $H^2 = D_\text{DW}^\dagger D_\text{DW}$ (= `MdagM`). Its smallest eigenvalues are $\sigma_\text{min}^2$;
  the $\sim |Q|$ would-be zero modes show up as the softest $\sigma$. This is CLEAN and Hermitian.

- **(B) Preconditioned complex spectrum $\mathrm{spec}(M_0 D_\text{DW})$.** This is the genuinely new
  piece. The operator is NON-Hermitian. Proof (the $\gamma_5$ trick): with $H = \gamma_5 D_\text{DW}$ and
  $H_\text{free} = \gamma_5 D_\text{free}$ both Hermitian but INDEFINITE,
  $$
  M_0 D_\text{DW} = \left(\Omega^\dagger H_\text{free}^{-1}\, \Omega\right) H .
  $$
  A product of two Hermitian factors is Hermitian only if they commute or if one is definite; here
  $H_\text{free}$ is indefinite, so $M_0 D_\text{DW}$ is not Hermitian and its spectrum is COMPLEX.
  The physics question: $D_\text{DW}$'s near-$0$ modes sit near the origin; a good preconditioner should
  move the whole spectrum toward $1$. We want the low-lying complex eigenvalues (smallest modulus, near
  $0$) of $M_0 D_\text{DW}$ and to see how far $M_0$ pushes them toward $1$.

## 2. Algorithm choice and citations

### (A) Hermitian smallest singular values -- reuse Grid IRL

Reuse Grid's `ImplicitlyRestartedLanczos` (`Grid/algorithms/iterative/ImplicitlyRestartedLanczos.h`) on
`MdagMLinearOperator<MobiusFermionD, LatticeFermionD>`, wired exactly like `tests/lanczos/Test_dwf_lanczos.cc`:
a `PlainHermOp` for the bare `HermOp` and a `FunctionHermOp` wrapping a `Chebyshev` polynomial to amplify
the small end (standard Kalkreuter-Simma / Chebyshev-accelerated Lanczos). No new eigensolver code.

Sources (as Grid's IRL already cites): Lanczos restart a la Sorensen; Kalkreuter-Simma polynomial
acceleration, T. Kalkreuter, H. Simma, Comput. Phys. Commun. 93 (1996) 33, hep-lat/9507023.

### (B) Non-Hermitian low-lying complex spectrum -- new Implicitly Restarted Arnoldi (IRA)

Grid has NO general (non-Hermitian) eigensolver. New code required. Algorithm and citations (to appear
prominently at the top of the new header and in the test):

- **D. C. Sorensen**, "Implicit Application of Polynomial Filters in a $k$-Step Arnoldi Method",
  SIAM J. Matrix Anal. Appl. 13(1), 357-385 (1992). -- the IRA restart.
- **R. B. Lehoucq, D. C. Sorensen, C. Yang**, "ARPACK Users' Guide" (SIAM, 1998). -- exact-shift
  strategy, deflation, residual bounds.

**Arnoldi factorization.** After $m$ steps, $A V_m = V_m H_m + f_m e_m^\dagger$ with $V_m$ an
orthonormal Krylov basis ($V_m^\dagger V_m = I$), $H_m$ upper Hessenberg $= V_m^\dagger A V_m$, and $f_m$
the residual. Each step is one $A$-apply plus modified Gram-Schmidt against the current basis. This is
exactly the loop already coded in Grid's GMRES `arnoldiStep`
(`Grid/algorithms/iterative/GeneralisedMinimalResidual.h:188`), reused as a template.

**Restart method -- pick: implicitly shifted QR with EXACT shifts (classic IRA / ARPACK).**
Grow to $m = N_{kv}$ steps, compute the $m$ Ritz values $\theta_i$ = eigenvalues of $H_m$ (dense, via
Eigen), split into $k = N_{ev}$ WANTED (smallest modulus, near $0$) and $p = m - k$ UNWANTED. Use the $p$
unwanted $\theta_i$ as shifts and apply $p$ implicitly-shifted QR sweeps to $H_m$:
$$
(H_m - \theta_j I) = Q_j R_j,\qquad H_m \leftarrow Q_j^\dagger H_m Q_j,\qquad V_m \leftarrow V_m Q_j .
$$
Accumulating $Q = \prod_j Q_j$ applies the degree-$p$ filter polynomial
$\psi(A) = \prod_j (A - \theta_j I)$ implicitly to the starting vector, damping the unwanted directions.
Truncating back to the leading $k$ columns yields a valid $k$-step Arnoldi factorization to extend again.
The compressed residual update is $f_k \leftarrow V_m(:,k)\,\beta + f_m\,\sigma$ with
$\beta = H_m(k,k-1)$ (subdiagonal) and $\sigma = Q(m-1,k-1)$, mirroring the Hermitian IRL's
`f *= Qt(...); f += lme[...]*evec[...]` bookkeeping (`ImplicitlyRestartedLanczos.h:348-356`).

**Why exact-shift IRA (not Krylov-Schur, not a fixed polynomial filter):** Nobu explicitly asked for IRA.
Exact shifts are Sorensen's own recommended, parameter-free filter (no Chebyshev interval to tune, unlike
the Hermitian case). Krylov-Schur (Stewart 2001) is the mathematically equivalent modern reformulation and
is a candidate fallback if the explicit shifted-QR bulge bookkeeping proves fragile at complex arithmetic
-- raised as an open question below, not the primary path.

**Small dense linear algebra -- reuse Eigen (Grid bundles it).** For the $m \times m$ Hessenberg:
`Eigen::ComplexEigenSolver<Eigen::MatrixXcd>` for the Ritz values/vectors (already used in
`Test_dwf_freeprec_claude.cc`); `Eigen::HouseholderQR<Eigen::MatrixXcd>` for each shifted-QR sweep,
accumulating $Q$. No hand-rolled QR/Hessenberg.

**Ritz extraction and convergence (no extra matvecs).** Ritz values = eigenvalues $\theta_i$ of the
leading $k \times k$ $H_k$; Ritz vectors $x_i = V_k\, y_i$ with $H_k y_i = \theta_i y_i$. The a-posteriori
Arnoldi residual $\lVert A x_i - \theta_i x_i \rVert = |\beta_k|\,|e_k^\dagger y_i|$ is available from the
subdiagonal $\beta_k = \lVert f \rVert$ and the last component of $y_i$ -- convergence is tested on this
bound WITHOUT forming $A x_i$. A final explicit residual check on the accepted modes closes the loop.

**Arithmetic / field type.** `LatticeFermionD` is already complex ($\mathrm{ComplexD}$); `innerProduct`
returns $\mathrm{ComplexD}$. Arnoldi in native complex arithmetic is the natural fit -- no real embedding.

## 3. What Grid already provides vs what is new

Reusable (found by grepping `Grid/Grid/algorithms/` and `lattice/`):

- `LinearOperatorBase<Field>::Op` / `LinearFunction<Field>` interfaces (`algorithms/LinearOperator.h`).
  `NonHermitianLinearOperator<MobiusFermionD,...>` wraps $D_\text{DW}$; `M0` is ALREADY a
  `LinearFunction` (`FreeMobius5D_claude.h:338`).
- `basisOrthogonalize(basis, w, k)` = modified Gram-Schmidt (`lattice/Lattice_basis.h:37`).
- `basisRotate<VField,Matrix>(basis, Qt, ...)` -- templated on the matrix type, so it accepts a complex
  `Qt` for the $V \leftarrow V Q$ basis rotation (`lattice/Lattice_basis.h:51`). (`basisRotateJ` is hard-typed
  to `MatrixXd`; not needed for (B).)
- GMRES `arnoldiStep` (`GeneralisedMinimalResidual.h:188`) -- the exact Arnoldi/MGS + Hessenberg-fill loop
  to template the new code on.
- Grid IRL itself for (A); `FunctionHermOp` / `PlainHermOp` / `Chebyshev` adapters
  (`Test_dwf_lanczos.cc:145-151`).
- Eigen (bundled): `ComplexEigenSolver`, `HouseholderQR`, `ComplexSchur` if we switch to Krylov-Schur.

Genuinely new:

- `ImplicitlyRestartedArnoldi_claude.h` -- the non-Hermitian IRA driver (Arnoldi factorization, exact-shift
  implicit QR restart, complex Ritz extraction, residual-bound convergence). Nothing in Grid does this.
- A composite operator that applies $M_0 D_\text{DW}$ (apply $D_\text{DW}$ then $M_0$) as a single
  `LinearFunction<LatticeFermionD>` for the Arnoldi $A$-apply.

## 4. Files to create / modify

- **Create** `Grid/tests/solver/spectrum_transform_impl_plan_claude.md` -- THIS plan.
  Files: `Grid/tests/solver/spectrum_transform_impl_plan_claude.md`.
- **Create** `Grid/Grid/algorithms/iterative/ImplicitlyRestartedArnoldi_claude.h` -- new IRA header,
  `template<class Field> class ImplicitlyRestartedArnoldi` taking a `LinearFunction<Field>&`, following
  Grid IRL idioms (field allocation, `basisOrthogonalize`, `basisRotate`, Eigen dense algebra).
  Files: `Grid/Grid/algorithms/iterative/ImplicitlyRestartedArnoldi_claude.h`.
- **Create** `Grid/tests/solver/Test_dwf_spectrum_transform_claude.cc` -- driver: reuse the frame/operator
  assembly from `Test_dwf_freeprec_claude.cc` (Wilson flow + Landau `xform`, `MobiusFermionD D`,
  `FreeMobius5DInverse Ffree`, `FreeLimitPreconditioner M0`), then run (A) Grid IRL on `MdagM` and (B) the
  new IRA on the composite $M_0 D_\text{DW}$; dump both spectra to stderr (repo convention). Composite
  operator defined here as a named `LinearFunction` subclass (no lambda).
  Files: `Grid/tests/solver/Test_dwf_spectrum_transform_claude.cc`.
- **Modify** `Grid/tests/solver/Makefile.am` (add the new test target) and, if the header needs to be in an
  umbrella, confirm `Grid/algorithms/iterative/Iterative.h` include list. Files:
  `Grid/tests/solver/Makefile.am`, possibly `Grid/Grid/algorithms/iterative/Iterative.h`.

Naming: every new/edited file carries `_claude` before the extension per the global rule; `Makefile.am` /
`Iterative.h` are pre-existing non-`_claude` files and (per rule 1) keep their names when edited in place --
FLAGGED as an open question below since editing them technically conflicts with the `_claude` convention.

## 5. Ordered implementation chunks (small, reviewable)

1. **IRA header, Arnoldi core.** `ImplicitlyRestartedArnoldi_claude.h`: constructor (Nev, Nkv, tol,
   maxiter, which-Ritz selector), the $k$-step Arnoldi factorization (MGS via `basisOrthogonalize`, fill
   `Eigen::MatrixXcd H`), and dense Ritz values via `ComplexEigenSolver`. Unit-testable on a small
   synthetic dense operator (a fixed non-normal matrix wrapped as a `LinearFunction`) with known spectrum.
   Files: `Grid/Grid/algorithms/iterative/ImplicitlyRestartedArnoldi_claude.h`.
2. **IRA restart + convergence.** Add exact-shift implicit QR sweeps (Eigen `HouseholderQR`), basis
   rotation (`basisRotate` with complex `Qt`), residual-bound convergence test, and the final explicit
   residual check + sorted eigenvalue output. Validate on the synthetic matrix vs Eigen's dense spectrum.
   Files: `Grid/Grid/algorithms/iterative/ImplicitlyRestartedArnoldi_claude.h`.
3. **Composite operator + test skeleton.** `Test_dwf_spectrum_transform_claude.cc`: reuse the freeprec
   assembly; define the $M_0 D_\text{DW}$ composite `LinearFunction`; run IRA on a cold/unit gauge as a
   sanity gate (spectrum should be exactly $1$ there, since $M_0 D_\text{DW} = I$). Files:
   `Grid/tests/solver/Test_dwf_spectrum_transform_claude.cc`, `Grid/tests/solver/Makefile.am`.
4. **(A) IRL on MdagM.** Wire Grid IRL (Chebyshev-accelerated) for the unpreconditioned smallest singular
   values; dump $\sigma_\text{min}$ and $\sim |Q|$ soft modes. Files:
   `Grid/tests/solver/Test_dwf_spectrum_transform_claude.cc`.
5. **Physics run on a real config.** Load a `configs_iwasaki_16_b2.6/ckpoint_lat.NNN`, build the flow+Landau
   frame, run (A) and (B), and compare where $D_\text{DW}$'s near-$0$ modes are vs where $M_0 D_\text{DW}$
   moves them. Hand off as a `tmp_claude.sh` + `*_claude.log` for Nobu to run on a CPU node
   (`build_mpi`). Files: driver + a handoff script.

## 6. Open questions for Nobu

1. **(B) target region -- plain IRA vs shift-invert.** Plain Arnoldi resolves EXTERIOR (largest-modulus)
   eigenvalues; "near $0$" (smallest modulus) is interior and can converge slowly. Since $M_0$ clusters the
   bulk near $1$, the near-$0$ stragglers may be isolated enough on the small-modulus side that plain IRA
   with a "smallest modulus" selection works at small volume ($\sim |Q|$ soft modes). If resolution is
   poor, the robust fix is shift-invert Arnoldi at $\sigma = 0$ (Arnoldi on $(M_0 D_\text{DW})^{-1}$ via an
   inner FGMRES), which is heavier. Start plain and escalate, or build shift-invert up front?
2. **Restart algorithm.** OK to implement exact-shift IRA (Sorensen) as asked, with Krylov-Schur held as a
   fallback -- or would you rather I go straight to Krylov-Schur (equivalent, simpler complex bookkeeping)?
3. **Do (A) at all now, or (B) first?** (A) is mostly wiring Grid IRL; (B) is the new code. Suggest landing
   (B) with the unit-gauge sanity gate first, then (A). Agree?
4. **Volumes and configs.** Start at $8^4$ (interactive/CPU), then $16^4$? Which
   `configs_iwasaki_16_b2.6/ckpoint_lat.NNN` (or a $Q \neq 0$ config from the R2 generation) should the
   headline run use? Mobius params -- reuse the freeprec headline $(b,c,m,M_5,L_s) = (1.5,0.5,0.1,1.8,8)$,
   anti-periodic time?
5. **How many modes.** $N_{ev} \sim 20$-$40$ near $0$, $N_{kv} \sim 2$-$3\times N_{ev}$? Confirm sizing.
6. **Precision.** Double throughout (matches freeprec)? The dense per-momentum $F$ is double; keep the whole
   diagnostic double.
7. **Editing pre-existing non-`_claude` files.** Adding the test needs edits to `Makefile.am` (and maybe the
   `Iterative.h` umbrella). Global rule 1 says new/edited files get `_claude`, but these build files must keep
   their exact names to work. I will edit them in place (append target / include) and NOT rename -- confirm
   that is what you want, or should the test be built via a standalone compile line instead of the autotools
   `Makefile.am`?
