# Holonomy in the frame norm-optimization -- implementation plan

**STATUS (2026-09-07): decisions = separate header / joint / Wilson-first. Chunk A DONE + gated; chunk
B DONE + compiles; chunk C (run) in progress.**
- A: Grid/qcd/utils/FreeWilsonTwisted_claude.h -- standalone per-colour twisted free Wilson inverse
  (theta[Nc][Nd], setTheta rebuilds; per-colour momentum blocks + colour-diagonal position phase,
  single FFT). GATE Test_freewilson_twist_gate_claude.cc PASSED: theta=0 reproduces FreeWilsonInverse
  BIT-FOR-BIT (||Ft-F||=0); theta!=0 changes it (3% for small twist). Local agent's FreeWilson_claude.h
  untouched.
- B: Test_wilson_frameopt_claude.cc -- fo_loss/fo_loss_force/fo_grad_check/fo_descend TEMPLATED on the
  kernel type (existing FreeWilsonInverse calls unchanged; twisted F is g5-Hermitian so the analytic
  Omega-gradient still holds). New fo_descend_joint (per iter: 1 backtracked Omega group-step + 1
  backtracked FD theta-step; theta traceless: theta[0],theta[1] free, theta[2]=-(t0+t1)). New M0TwistW
  adapter (Omega^dag F_theta Omega) for the FGMRES win. Wired as --frameopt --opt_theta [--theta-eta].
  Also guarded a pre-existing --save-modes ScidacWriter block with #ifdef HAVE_LIME (merged CPU build
  has no LIME). Compiles on build_mpi_merged.
- C: functional test running on 747 (--frameopt --opt_theta). Compare L(Landau,theta=0) vs joint
  (Omega,theta), and FGMRES iters Landau vs Omega-only vs joint.


Goal (Nobu 2026-09-07): add the CONSTANT holonomy twist theta_mu^(c) to the set of variables optimized
alongside the group-valued frame Omega, in the direct norm-optimization (Test_wilson_frameopt_claude.cc,
merged from local), and see if it improves the frame residual ||M0 D_W v - v||. This is the principled
version of the twisted free kernel: rather than MEASURE and match a holonomy, we OPTIMIZE theta jointly
with Omega, so the optimizer finds whatever constant twist best lowers the loss (sidesteps the "average
holonomy ~ 0" problem -- the optimizer is not restricted to the spatial average).

Why this is the right home for the idea: M0 = Omega^dag F Omega preconditions D_W[U_orig]; Omega enters
by CONJUGATION so it cannot change the framed config's holonomy. But F is OURS to choose -- giving F a
tunable per-colour twist theta lets it MATCH the holonomy the frame cannot remove. Joint (Omega,theta)
optimization is the clean way to exploit it.

## The physics of the twist (closed FFT form, already derived -- see freeprec_gaugefix_topology_claude.md Sec.7)
Free Wilson block per colour c: D_W^{(c)}(p) = A(p+theta^c) I + i sum_mu sin(p_mu+theta_mu^c) gamma_mu.
So the twist is a per-colour momentum shift p_mu -> p_mu + theta_mu^c, on top of the AP-time BC shift tw.
SU(3): theta_mu^(c), c=1..3 with sum_c theta_mu^c = 0 -> 2 free phases per direction x 4 dirs = 8 params.

## Files
- Grid/qcd/utils/FreeWilson_claude.h -- make the kernel per-colour-twist capable. IT IS THE LOCAL
  AGENT'S SHARED FILE -> add theta as an OPTIONAL member defaulting to 0 (theta=0 reproduces the current
  color-blind kernel bit-for-bit; no behaviour change for existing callers). Changes:
  * add `std::vector<std::array<double,Nd>> theta_c` (Nc entries; default all 0);
  * Build(): Minv becomes PER COLOUR -- `std::vector<Eigen::MatrixXcd> Minv[Nc]` (or Minv indexed
    [c*V4+slot]); block at momentum p + tw[mu] + theta_c[c][mu];
  * position phase becomes per colour: phase_neg_c/phase_pos_c (compensate tw + theta_c);
  * apply(): colour loop selects the colour's block set + phase. theta=0 -> identical to now.
  * `void setTheta(const std::vector<std::array<double,Nd>>&)` -> rebuild (cheap: V4 x Nc 4x4 inverts).
- Test_wilson_frameopt_claude.cc -- add theta to the optimization:
  * fo_loss / fo_loss_force already take F by ref -> F now carries theta; loss is L[Omega,theta].
  * NEW theta optimiser. MVP = FINITE-DIFFERENCE gradient on the 8 theta params (loss is cheap; 16
    evals/step), coordinate or steepest descent with backtracking, reusing fo_loss. theta rebuild via
    setTheta each eval.
  * OUTER loop: alternate -- (a) fo_descend on Omega at fixed theta (existing), (b) theta FD-descent at
    fixed Omega. Repeat a few rounds. Report L and ||M0 D v-v|| for theta=0 (baseline) vs optimised theta.
  * CLI --opt_theta (default off -> current behaviour), --theta_iter, --theta_eta.

## Chunks
A. FreeWilson theta-capable + a theta=0 regression gate (residual identical to current within 1e-14).
B. theta FD-gradient + alternating optimiser in the frameopt driver; --opt_theta wiring; FD-gate the
   theta gradient like fo_grad_check.
C. Run on 640 + 747: baseline (theta=0, Omega-only) vs joint (Omega,theta). Report frame residual +, if
   cheap, the FGMRES win. Does a nonzero optimal theta lower the residual, and by how much?

## Open questions (CONFIRM before coding)
1. Edit FreeWilson_claude.h IN PLACE (theta optional, default 0 = no-op) -- OR a separate
   FreeWilsonTwisted_claude.h subclass to avoid touching the local agent's file? (Leaning in-place with
   the default-0 guarantee, since the color loop is the natural place; but it is dwms-af's file.)
2. theta optimisation: alternating (Omega then theta, simplest, reuses fo_descend) vs fully joint
   gradient? (Leaning alternating for the MVP.)
3. This is the WILSON frameopt (Test_wilson_frameopt, FreeWilsonInverse). Do it here first (cheap, no
   Ls), then port to the Mobius/DWF freeprec (FreeMobius5DInverse) if it helps? (Leaning yes -- Wilson
   is the fast testbed.)
