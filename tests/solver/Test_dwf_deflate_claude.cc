// Deflation of the |D_DW|^2 low modes as a SPECTRUM TRANSFORMATION -- GMRES vs CGNE (Direction 2).
// On a hard config (traj 640, |Q|=3) the sv diagnostic found k=3 topology-tied low singular triplets of
// D_DW at sigma ~ m. This test deflates them and asks: how much does the solve accelerate, and which outer
// solver -- GMRES or CGNE -- benefits more?
//
// Spectrum transform: with the k lowest right/left singular vectors v_i,u_i (D v_i = sigma_i u_i) and a
// target sqrt_mu = sigma_max, define
//   D' = D + sum_i (sqrt_mu - sigma_i) u_i v_i^dag      (lifts sigma_i -> sqrt_mu; unchanged on the rest)
// so A' = D'^dag D' lifts the |D|^2 low eigenvalues lambda_i=sigma_i^2 -> mu. Both are better conditioned.
// The TRUE solution of D x = b is recovered from x' = D'^{-1} b by the analytic low-mode correction
//   x = x' + sum_i (1/sigma_i - 1/sqrt_mu)(u_i^dag b) v_i     [same for the CGNE and GMRES deflated solves].
// Ref: deflated CG -- Saad, Yeung, Erhel, Guyomarc'h, SIAM J. Sci. Comput. 21 (2000) 1909; lattice near-
// zero deflation -- Stathopoulos & Orginos, SIAM J. Sci. Comput. 32 (2010) 439. The GMRES side is the
// singular-value image of the same transform.
//
// Metric: honest D_W-apply count = Mobius (D_DW) applies to reach a fixed relative residual (CGNE = 2/iter
// via M and M^dag; GMRES = 1/iter on D_DW). Counted by the global g_dw_applies incremented in the wrappers.
//
// Run:  Test_dwf_deflate_claude --grid 16.16.16.16 --mpi 1.1.1.1 --config <nersc> \
//         --k 3 --tol 1e-8 --cheb_lo 0.5 --cheb_ord 21 --gmres_restart 50

#include <Grid/Grid.h>

#include <vector>
#include <string>
#include <algorithm>

using namespace Grid;

// Global Mobius(D_DW)-apply counter. Reset before each solve; every D.M / D.Mdag goes through the wrappers.
static long g_dw_applies = 0;

// D' apply: out = D in + sum_i (sqrt_mu - sigma_i) u_i (v_i^dag in).
static void applyDprime(MobiusFermionD& D, const std::vector<LatticeFermionD>& V,
                        const std::vector<LatticeFermionD>& U, const std::vector<RealD>& sig,
                        RealD sqrt_mu, const LatticeFermionD& in, LatticeFermionD& out) {
  D.M(in, out);
  g_dw_applies++;
  for (int i = 0; i < (int)V.size(); ++i) {
    ComplexD c = innerProduct(V[i], in);          // v_i^dag in
    ComplexD coeff = c * (sqrt_mu - sig[i]);
    out = out + coeff * U[i];
  }
}

// D'^dag apply: out = D^dag in + sum_i (sqrt_mu - sigma_i) v_i (u_i^dag in)  [coeff real].
static void applyDprimeDag(MobiusFermionD& D, const std::vector<LatticeFermionD>& V,
                           const std::vector<LatticeFermionD>& U, const std::vector<RealD>& sig,
                           RealD sqrt_mu, const LatticeFermionD& in, LatticeFermionD& out) {
  D.Mdag(in, out);
  g_dw_applies++;
  for (int i = 0; i < (int)V.size(); ++i) {
    ComplexD c = innerProduct(U[i], in);          // u_i^dag in
    ComplexD coeff = c * (sqrt_mu - sig[i]);
    out = out + coeff * V[i];
  }
}

// --- CGNE plain: Hermitian PD normal operator A = D^dag D ---
class CountMdagMOp : public LinearOperatorBase<LatticeFermionD> {
public:
  MobiusFermionD& D;
  LatticeFermionD t;
  CountMdagMOp(MobiusFermionD& D_, GridBase* g) : D(D_), t(g) {}
  void HermOp(const LatticeFermionD& in, LatticeFermionD& out) {
    D.M(in, t);
    g_dw_applies++;
    D.Mdag(t, out);
    g_dw_applies++;
  }
  void HermOpAndNorm(const LatticeFermionD& in, LatticeFermionD& out, RealD& n1, RealD& n2) {
    HermOp(in, out);
    ComplexD dot = innerProduct(in, out);
    n1 = real(dot);
    n2 = norm2(out);
  }
  void Op(const LatticeFermionD& in, LatticeFermionD& out) { HermOp(in, out); }
  void AdjOp(const LatticeFermionD& in, LatticeFermionD& out) { HermOp(in, out); }
  void OpDiag(const LatticeFermionD& in, LatticeFermionD& out) { assert(0); }
  void OpDir(const LatticeFermionD& in, LatticeFermionD& out, int dir, int disp) { assert(0); }
  void OpDirAll(const LatticeFermionD& in, std::vector<LatticeFermionD>& out) { assert(0); }
};

// --- CGNE deflated: A' = D'^dag D' (low |D|^2 eigenvalues lifted to mu) ---
class CountAprimeOp : public LinearOperatorBase<LatticeFermionD> {
public:
  MobiusFermionD& D;
  const std::vector<LatticeFermionD>& V;
  const std::vector<LatticeFermionD>& U;
  const std::vector<RealD>& sig;
  RealD sqrt_mu;
  LatticeFermionD t;
  CountAprimeOp(MobiusFermionD& D_, const std::vector<LatticeFermionD>& V_,
               const std::vector<LatticeFermionD>& U_, const std::vector<RealD>& sig_,
               RealD sqrt_mu_, GridBase* g)
    : D(D_), V(V_), U(U_), sig(sig_), sqrt_mu(sqrt_mu_), t(g) {}
  void HermOp(const LatticeFermionD& in, LatticeFermionD& out) {
    applyDprime(D, V, U, sig, sqrt_mu, in, t);
    applyDprimeDag(D, V, U, sig, sqrt_mu, t, out);
  }
  void HermOpAndNorm(const LatticeFermionD& in, LatticeFermionD& out, RealD& n1, RealD& n2) {
    HermOp(in, out);
    ComplexD dot = innerProduct(in, out);
    n1 = real(dot);
    n2 = norm2(out);
  }
  void Op(const LatticeFermionD& in, LatticeFermionD& out) { HermOp(in, out); }
  void AdjOp(const LatticeFermionD& in, LatticeFermionD& out) { HermOp(in, out); }
  void OpDiag(const LatticeFermionD& in, LatticeFermionD& out) { assert(0); }
  void OpDir(const LatticeFermionD& in, LatticeFermionD& out, int dir, int disp) { assert(0); }
  void OpDirAll(const LatticeFermionD& in, std::vector<LatticeFermionD>& out) { assert(0); }
};

// --- GMRES plain: Op = D_DW ---
class CountDOp : public LinearOperatorBase<LatticeFermionD> {
public:
  MobiusFermionD& D;
  CountDOp(MobiusFermionD& D_) : D(D_) {}
  void Op(const LatticeFermionD& in, LatticeFermionD& out) {
    D.M(in, out);
    g_dw_applies++;
  }
  void AdjOp(const LatticeFermionD& in, LatticeFermionD& out) {
    D.Mdag(in, out);
    g_dw_applies++;
  }
  void HermOp(const LatticeFermionD& in, LatticeFermionD& out) { assert(0); }
  void HermOpAndNorm(const LatticeFermionD& in, LatticeFermionD& out, RealD& n1, RealD& n2) { assert(0); }
  void OpDiag(const LatticeFermionD& in, LatticeFermionD& out) { assert(0); }
  void OpDir(const LatticeFermionD& in, LatticeFermionD& out, int dir, int disp) { assert(0); }
  void OpDirAll(const LatticeFermionD& in, std::vector<LatticeFermionD>& out) { assert(0); }
};

// --- GMRES deflated: Op = D' (low singular values lifted to sqrt_mu) ---
class CountDprimeOp : public LinearOperatorBase<LatticeFermionD> {
public:
  MobiusFermionD& D;
  const std::vector<LatticeFermionD>& V;
  const std::vector<LatticeFermionD>& U;
  const std::vector<RealD>& sig;
  RealD sqrt_mu;
  CountDprimeOp(MobiusFermionD& D_, const std::vector<LatticeFermionD>& V_,
               const std::vector<LatticeFermionD>& U_, const std::vector<RealD>& sig_, RealD sqrt_mu_)
    : D(D_), V(V_), U(U_), sig(sig_), sqrt_mu(sqrt_mu_) {}
  void Op(const LatticeFermionD& in, LatticeFermionD& out) {
    applyDprime(D, V, U, sig, sqrt_mu, in, out);
  }
  void AdjOp(const LatticeFermionD& in, LatticeFermionD& out) {
    applyDprimeDag(D, V, U, sig, sqrt_mu, in, out);
  }
  void HermOp(const LatticeFermionD& in, LatticeFermionD& out) { assert(0); }
  void HermOpAndNorm(const LatticeFermionD& in, LatticeFermionD& out, RealD& n1, RealD& n2) { assert(0); }
  void OpDiag(const LatticeFermionD& in, LatticeFermionD& out) { assert(0); }
  void OpDir(const LatticeFermionD& in, LatticeFermionD& out, int dir, int disp) { assert(0); }
  void OpDirAll(const LatticeFermionD& in, std::vector<LatticeFermionD>& out) { assert(0); }
};

// true relative residual ||D x - b|| / ||b|| on the ORIGINAL operator (correctness check; not counted).
static RealD true_resid(MobiusFermionD& D, const LatticeFermionD& x, const LatticeFermionD& b, RealD bn) {
  LatticeFermionD r(b.Grid());
  D.M(x, r);
  r = r - b;
  return std::sqrt(norm2(r) / bn);
}

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  const double M5 = 1.8;
  const int Ls = 8;
  const double bb = 1.5;
  const double cc = 0.5;
  const double mm = 0.1;
  std::vector<Complex> boundary = {1, 1, 1, -1};

  int kdefl = 3;
  RealD tol = 1.0e-8;
  int cg_maxit = 20000;
  int gmres_maxit = 20000;
  int gmres_restart = 50;
  // low-mode eigensolve (same Chebyshev IRL as run_sv): cheb_hi<=0 -> auto (power-method lambda_max),
  // cheb_ord forced ODD (IRL keeps largest filtered value), cheb_lo = cut ABOVE the target cluster.
  RealD cheb_lo = 0.5;
  RealD cheb_hi = -1.0;
  int cheb_ord = 21;
  RealD sv_resid = 1.0e-5;
  int sv_maxit = 200;
  int Nstop = 12;
  int Nk = 24;
  int Nm = 48;

  if (GridCmdOptionExists(argv, argv + argc, "--k")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--k");
    GridCmdOptionInt(a, kdefl);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--tol")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--tol");
    GridCmdOptionFloat(a, tol);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--cg_maxit")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--cg_maxit");
    GridCmdOptionInt(a, cg_maxit);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--gmres_maxit")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--gmres_maxit");
    GridCmdOptionInt(a, gmres_maxit);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--gmres_restart")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--gmres_restart");
    GridCmdOptionInt(a, gmres_restart);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--cheb_lo")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--cheb_lo");
    GridCmdOptionFloat(a, cheb_lo);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--cheb_hi")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--cheb_hi");
    GridCmdOptionFloat(a, cheb_hi);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--cheb_ord")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--cheb_ord");
    GridCmdOptionInt(a, cheb_ord);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--sv_resid")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--sv_resid");
    GridCmdOptionFloat(a, sv_resid);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--sv_maxit")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--sv_maxit");
    GridCmdOptionInt(a, sv_maxit);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--nstop")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--nstop");
    GridCmdOptionInt(a, Nstop);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--nk")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--nk");
    GridCmdOptionInt(a, Nk);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--nm")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--nm");
    GridCmdOptionInt(a, Nm);
  }
  if (!GridCmdOptionExists(argv, argv + argc, "--config")) {
    std::cout << "ERROR: need --config <NERSC gauge file>" << std::endl;
    Grid_finalize();
    return 1;
  }
  std::string cfgfile = GridCmdOptionPayload(argv, argv + argc, "--config");

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplexD::Nsimd());
  Coordinate mpi = GridDefaultMpi();
  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(latt, simd, mpi);
  GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian* FGrid = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
  GridRedBlackCartesian* FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);

  GridParallelRNG RNG5(FGrid);
  RNG5.SeedFixedIntegers(std::vector<int>({1, 2, 3, 4}));

  LatticeGaugeFieldD U(UGrid);
  FieldMetaData rheader;
  NerscIO::readConfiguration(U, rheader, cfgfile);
  Real plaq = WilsonLoops<PeriodicGimplD>::avgPlaquette(U);
  std::cout << "loaded " << cfgfile << "  plaq=" << plaq << std::endl;

  WilsonImplD::ImplParams Params(boundary);
  MobiusFermionD D(U, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mm, M5, bb, cc, Params);

  // ---- k lowest singular modes of D_DW via Chebyshev IRL on |D_DW|^2 (same setup as run_sv) ----
  MdagMLinearOperator<MobiusFermionD, LatticeFermionD> HermMdagM(D);
  if (cheb_ord % 2 == 0) {
    cheb_ord += 1;
    std::cout << "  cheb_ord even -> bumped to " << cheb_ord << " (IRL needs ODD order)" << std::endl;
  }
  if (cheb_hi <= 0.0) {
    LatticeFermionD pmsrc(FGrid);
    gaussian(RNG5, pmsrc);
    PowerMethod<LatticeFermionD> PM;
    RealD lmax = PM(HermMdagM, pmsrc);
    cheb_hi = 1.1 * lmax;
    std::cout << "  power-method lambda_max(|D|^2) = " << lmax << "  -> cheb_hi = " << cheb_hi << std::endl;
  }
  RealD mu = cheb_hi / 1.1;          // = lambda_max; the transform target lambda_i -> mu
  RealD sqrt_mu = std::sqrt(mu);
  std::cout << "  spectrum-transform target: mu = " << mu << "  sqrt_mu = " << sqrt_mu << std::endl;

  Chebyshev<LatticeFermionD> Cheby(cheb_lo, cheb_hi, cheb_ord);
  FunctionHermOp<LatticeFermionD> OpCheby(Cheby, HermMdagM);
  PlainHermOp<LatticeFermionD> Op(HermMdagM);
  ImplicitlyRestartedLanczos<LatticeFermionD> IRL(OpCheby, Op, Nstop, Nk, Nm, sv_resid, sv_maxit);
  std::vector<RealD> eval(Nm);
  std::vector<LatticeFermionD> evec(Nm, LatticeFermionD(FGrid));
  LatticeFermionD lsrc(FGrid);
  gaussian(RNG5, lsrc);
  int Nconv = 0;
  IRL.calc(eval, evec, lsrc, Nconv);
  std::cout << "  IRL Nconv=" << Nconv << std::endl;

  // select the k lowest-lambda converged modes (k tiny -> manual selection, no lambda)
  if (kdefl > Nconv) kdefl = Nconv;
  std::vector<int> order;
  std::vector<char> used(Nconv, 0);
  for (int p = 0; p < kdefl; ++p) {
    int best = -1;
    for (int i = 0; i < Nconv; ++i) {
      if (used[i]) continue;
      if (best < 0 || eval[i] < eval[best]) best = i;
    }
    used[best] = 1;
    order.push_back(best);
  }
  std::vector<LatticeFermionD> V(kdefl, LatticeFermionD(FGrid));
  std::vector<LatticeFermionD> Uv(kdefl, LatticeFermionD(FGrid));
  std::vector<RealD> sig(kdefl);
  LatticeFermionD tmp(FGrid);
  std::cout << "  deflation subspace (k=" << kdefl << "):" << std::endl;
  for (int i = 0; i < kdefl; ++i) {
    int idx = order[i];
    V[i] = evec[idx];
    RealD lam = eval[idx];
    sig[i] = std::sqrt(lam > 0.0 ? lam : 0.0);
    D.M(V[i], tmp);                 // u_i = D v_i / sigma_i  (auto-orthonormal for orthonormal v_i)
    Uv[i] = tmp * (1.0 / sig[i]);
    std::cout << "    mode[" << i << "] sigma=" << sig[i] << "  lambda=" << lam << std::endl;
  }

  // ---- right-hand side + analytic low-mode correction ----
  LatticeFermionD b(FGrid);
  gaussian(RNG5, b);
  RealD bn = norm2(b);
  // corr = sum_i (1/sigma_i - 1/sqrt_mu)(u_i^dag b) v_i ; recovers true x from x' = D'^{-1} b.
  LatticeFermionD corr(FGrid);
  corr = Zero();
  for (int i = 0; i < kdefl; ++i) {
    ComplexD ub = innerProduct(Uv[i], b);
    ComplexD c = ub * (1.0 / sig[i] - 1.0 / sqrt_mu);
    corr = corr + c * V[i];
  }

  CountMdagMOp Aop(D, FGrid);
  CountAprimeOp Apop(D, V, Uv, sig, sqrt_mu, FGrid);
  CountDOp Dop(D);
  CountDprimeOp Dpop(D, V, Uv, sig, sqrt_mu);

  LatticeFermionD x(FGrid);
  LatticeFermionD xp(FGrid);
  LatticeFermionD f(FGrid);
  long a_cgne = 0, b_cgne = 0, c_gmres = 0, d_gmres = 0;
  RealD r_cgne = 0, r_cgned = 0, r_gmres = 0, r_gmresd = 0;

  ConjugateGradient<LatticeFermionD> CG(tol, cg_maxit, false);
  ConjugateGradient<LatticeFermionD> CGd(tol, cg_maxit, false);
  GeneralisedMinimalResidual<LatticeFermionD> GMRES(tol, gmres_maxit, gmres_restart, false);
  GeneralisedMinimalResidual<LatticeFermionD> GMRESd(tol, gmres_maxit, gmres_restart, false);

  // A. CGNE plain: CG on A = D^dag D, rhs f = D^dag b.
  std::cout << "==== A. CGNE plain ====" << std::endl;
  g_dw_applies = 0;
  D.Mdag(b, f);
  g_dw_applies++;
  x = Zero();
  CG(Aop, f, x);
  a_cgne = g_dw_applies;
  r_cgne = true_resid(D, x, b, bn);

  // B. CGNE deflated: CG on A' = D'^dag D', rhs f' = D'^dag b; then x = x' + corr.
  std::cout << "==== B. CGNE deflated (|D|^2 low modes lifted to mu) ====" << std::endl;
  g_dw_applies = 0;
  applyDprimeDag(D, V, Uv, sig, sqrt_mu, b, f);
  xp = Zero();
  CGd(Apop, f, xp);
  x = xp + corr;
  b_cgne = g_dw_applies;
  r_cgned = true_resid(D, x, b, bn);

  // C. GMRES plain: GMRES on D_DW, rhs b.
  std::cout << "==== C. GMRES plain ====" << std::endl;
  g_dw_applies = 0;
  x = Zero();
  GMRES(Dop, b, x);
  c_gmres = g_dw_applies;
  r_gmres = true_resid(D, x, b, bn);

  // D. GMRES deflated: GMRES on D' (low singular values -> sqrt_mu), rhs b; then x = x' + corr.
  std::cout << "==== D. GMRES deflated (SVD-deflation) ====" << std::endl;
  g_dw_applies = 0;
  xp = Zero();
  GMRESd(Dpop, b, xp);
  x = xp + corr;
  d_gmres = g_dw_applies;
  r_gmresd = true_resid(D, x, b, bn);

  std::cout << "==== deflation summary (config " << cfgfile << ", k=" << kdefl << ", tol=" << tol
            << ") ====" << std::endl;
  std::cout << "  solve            D_W-applies    true rel resid" << std::endl;
  std::cout << "  A CGNE  plain    " << a_cgne << "    " << r_cgne << std::endl;
  std::cout << "  B CGNE  deflated " << b_cgne << "    " << r_cgned << std::endl;
  std::cout << "  C GMRES plain    " << c_gmres << "    " << r_gmres << std::endl;
  std::cout << "  D GMRES deflated " << d_gmres << "    " << r_gmresd << std::endl;
  if (b_cgne > 0) std::cout << "  CGNE speedup A/B = " << (double)a_cgne / (double)b_cgne << std::endl;
  if (d_gmres > 0) std::cout << "  GMRES speedup C/D = " << (double)c_gmres / (double)d_gmres << std::endl;

  Grid_finalize();
  return 0;
}
