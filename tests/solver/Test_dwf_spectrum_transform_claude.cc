// Spectrum-transform diagnostic (Direction 2): Implicitly Restarted Arnoldi on the NON-Hermitian
// operators D_DW and M0 D_DW for a (hard) config -- the smallest-modulus (near-zero) eigenvalues.
// D_DW near-0 = the topology-tied low modes that make a config hard; M0 D_DW near-0 = where the
// free-limit frame moves them (should cluster the bulk near 1). Uses the agent's IRA solver
// Grid/algorithms/iterative/ImplicitlyRestartedArnoldi_claude.h. M0 built exactly as in the flowscan
// driver (WilsonFlow + Landau -> Omega, FreeMobius5DInverse F, FreeLimitPreconditioner).
//
// Run:  Test_dwf_spectrum_transform_claude --grid 16.16.16.16 --mpi 1.1.1.1 --config <nersc> \
//         --which both --flow_nstep 58 --flow_eps 0.02 --nstop 12 --nk 24 --nm 48 --eresid 1e-4 --maxit 200
// (near-0 is INTERIOR for a non-Hermitian op -> plain smallest-modulus IRA may converge slowly; if so,
//  escalate to shift-invert at sigma=0. Trial-and-error the Nstop/Nk/Nm.)

#include <Grid/Grid.h>
#include <Grid/qcd/utils/FreeMobius5D_claude.h>
#include <Grid/algorithms/iterative/ImplicitlyRestartedArnoldi_claude.h>

#include <vector>
#include <string>
#include <complex>
#include <fstream>

using namespace Grid;

// A = D_DW : out = D.M(in)
template <class FOp, class Field>
class ApplyDop : public LinearFunction<Field> {
public:
  FOp& D;
  ApplyDop(FOp& D_) : D(D_) {}
  void operator()(const Field& in, Field& out) { D.M(in, out); }
};

// A = M0 D_DW : out = M0(D.M(in))   [spec(M0 D) = spec(D M0), so apply order is immaterial for evals]
template <class FOp, class Field>
class ApplyM0Dop : public LinearFunction<Field> {
public:
  FOp& D;
  LinearFunction<Field>& M0;
  Field tmp;
  ApplyM0Dop(FOp& D_, LinearFunction<Field>& M0_, GridBase* g) : D(D_), M0(M0_), tmp(g) {}
  void operator()(const Field& in, Field& out) {
    D.M(in, tmp);
    M0(tmp, out);
  }
};

// Shifted operator (A - sigma I) with Op() only (BiCGSTAB uses just the forward apply).
template <class FOp, class Field>
class ShiftedDop : public LinearOperatorBase<Field> {
public:
  FOp& D;
  RealD sigma;
  ShiftedDop(FOp& D_, RealD s) : D(D_), sigma(s) {}
  void Op(const Field& in, Field& out) {
    D.M(in, out);
    out = out - sigma * in;
  }
  void AdjOp(const Field& in, Field& out) { assert(0); }
  void OpDiag(const Field& in, Field& out) { assert(0); }
  void OpDir(const Field& in, Field& out, int dir, int disp) { assert(0); }
  void OpDirAll(const Field& in, std::vector<Field>& out) { assert(0); }
  void HermOp(const Field& in, Field& out) { assert(0); }
  void HermOpAndNorm(const Field& in, Field& out, RealD& n1, RealD& n2) { assert(0); }
};

// Shifted preconditioned operator (M0 D_DW - sigma I): Op() = M0(D.M(in)) - sigma in.
template <class FOp, class Field>
class ShiftedM0Dop : public LinearOperatorBase<Field> {
public:
  FOp& D;
  LinearFunction<Field>& M0;
  RealD sigma;
  Field tmp;
  ShiftedM0Dop(FOp& D_, LinearFunction<Field>& M0_, RealD s, GridBase* g)
    : D(D_), M0(M0_), sigma(s), tmp(g) {}
  void Op(const Field& in, Field& out) {
    D.M(in, tmp);
    M0(tmp, out);
    out = out - sigma * in;
  }
  void AdjOp(const Field& in, Field& out) { assert(0); }
  void OpDiag(const Field& in, Field& out) { assert(0); }
  void OpDir(const Field& in, Field& out, int dir, int disp) { assert(0); }
  void OpDirAll(const Field& in, std::vector<Field>& out) { assert(0); }
  void HermOp(const Field& in, Field& out) { assert(0); }
  void HermOpAndNorm(const Field& in, Field& out, RealD& n1, RealD& n2) { assert(0); }
};

// Shift-invert apply: out = (A - sigma)^{-1} in, via restarted GMRES (robust for the non-Hermitian,
// possibly-indefinite M0 D_DW at sigma~0 -- BiCGSTAB breaks down there). IRA on this returns the
// LARGEST-modulus theta = 1/(lambda - sigma), i.e. the eigenvalues of A CLOSEST to sigma; convert back.
template <class Field>
class ShiftInvert : public LinearFunction<Field> {
public:
  LinearOperatorBase<Field>& Ashift;
  RealD inner_tol;
  int inner_maxit;
  int inner_restart;
  ShiftInvert(LinearOperatorBase<Field>& A, RealD tol, int maxit, int restart)
    : Ashift(A), inner_tol(tol), inner_maxit(maxit), inner_restart(restart) {}
  void operator()(const Field& in, Field& out) {
    out = Zero();
    GeneralisedMinimalResidual<Field> GMRES(inner_tol, inner_maxit, inner_restart, /*err_on_no_conv=*/false);
    GMRES(Ashift, in, out);
  }
};

// Hermitian PD normal operator (D-sigma)^dag (D-sigma) for CGNE shift-invert (robust, no BiCGSTAB breakdown).
template <class FOp, class Field>
class ShiftedNormalOp : public LinearOperatorBase<Field> {
public:
  FOp& D;
  RealD sigma;
  Field t1;
  ShiftedNormalOp(FOp& D_, RealD s, GridBase* g) : D(D_), sigma(s), t1(g) {}
  void HermOp(const Field& in, Field& out) {
    D.M(in, t1);
    t1 = t1 - sigma * in;      // (D - sigma) in
    D.Mdag(t1, out);
    out = out - sigma * t1;    // (D - sigma)^dag [(D-sigma) in]   (sigma real)
  }
  void HermOpAndNorm(const Field& in, Field& out, RealD& n1, RealD& n2) {
    HermOp(in, out);
    ComplexD dot = innerProduct(in, out);
    n1 = real(dot);
    n2 = norm2(out);
  }
  void Op(const Field& in, Field& out) { HermOp(in, out); }
  void AdjOp(const Field& in, Field& out) { HermOp(in, out); }
  void OpDiag(const Field& in, Field& out) { assert(0); }
  void OpDir(const Field& in, Field& out, int dir, int disp) { assert(0); }
  void OpDirAll(const Field& in, std::vector<Field>& out) { assert(0); }
};

// Shift-invert via CGNE: out = (D-sigma)^{-1} in, solving (D-sigma)^dag(D-sigma) x = (D-sigma)^dag in with CG.
template <class FOp, class Field>
class ShiftInvertCGNE : public LinearFunction<Field> {
public:
  FOp& D;
  RealD sigma;
  ShiftedNormalOp<FOp, Field> Nop;
  RealD inner_tol;
  int inner_maxit;
  Field rhs;
  ShiftInvertCGNE(FOp& D_, RealD s, RealD tol, int maxit, GridBase* g)
    : D(D_), sigma(s), Nop(D_, s, g), inner_tol(tol), inner_maxit(maxit), rhs(g) {}
  void operator()(const Field& in, Field& out) {
    D.Mdag(in, rhs);
    rhs = rhs - sigma * in;    // rhs = (D-sigma)^dag in
    out = Zero();
    ConjugateGradient<Field> CG(inner_tol, inner_maxit, /*err_on_no_conv=*/false);
    CG(Nop, rhs, out);
  }
};

static void run_ira(const std::string& label, LinearFunction<LatticeFermionD>& Op, GridCartesian* FGrid,
                    GridParallelRNG& RNG5, int Nstop, int Nk, int Nm, RealD eresid, int MaxIter) {
  std::cout << "==== IRA smallest-modulus spectrum: " << label << " ====" << std::endl;
  ImplicitlyRestartedArnoldi<LatticeFermionD> IRA(Op, Nstop, Nk, Nm, eresid, MaxIter, IRAsmallestModulus);
  std::vector<ComplexD> eval(Nm);
  std::vector<LatticeFermionD> evec(Nm, LatticeFermionD(FGrid));
  LatticeFermionD src(FGrid);
  gaussian(RNG5, src);
  int Nconv = 0;
  IRA.calc(eval, evec, src, Nconv);
  std::cout << "  [" << label << "] Nconv=" << Nconv << "  near-zero (smallest |lambda|) eigenvalues:"
            << std::endl;
  int nshow = (Nconv < Nstop) ? Nconv : Nstop;
  for (int i = 0; i < nshow; ++i) {
    std::cout << "    lambda[" << i << "] = (" << eval[i].real() << ", " << eval[i].imag()
              << ")   |lambda| = " << std::hypot(eval[i].real(), eval[i].imag()) << std::endl;
  }
}

// Shift-invert IRA: eigenvalues of A NEAREST sigma. IRA (LARGEST modulus) on (A-sigma)^{-1} gives
// theta = 1/(lambda - sigma); convert lambda = sigma + 1/theta. sigma ~ -M5 = -1.8 targets the physical
// Wilson branch (gate 0a) where the topology-tied modes peel off on a Q!=0 config.
static void run_ira_si(const std::string& label, LinearFunction<LatticeFermionD>& SIop, RealD sigma,
                       GridCartesian* FGrid, GridParallelRNG& RNG5,
                       int Nstop, int Nk, int Nm, RealD eresid, int MaxIter,
                       const std::string& datfile) {
  std::cout << "==== IRA shift-invert (sigma=" << sigma << ") spectrum: " << label << " ====" << std::endl;
  ImplicitlyRestartedArnoldi<LatticeFermionD> IRA(SIop, Nstop, Nk, Nm, eresid, MaxIter, IRAlargestModulus);
  std::vector<ComplexD> eval(Nm);
  std::vector<LatticeFermionD> evec(Nm, LatticeFermionD(FGrid));
  LatticeFermionD src(FGrid);
  gaussian(RNG5, src);
  int Nconv = 0;
  IRA.calc(eval, evec, src, Nconv);
  std::cout << "  [" << label << "] Nconv=" << Nconv << "  eigenvalues of A nearest sigma=" << sigma
            << ":" << std::endl;
  int nshow = (Nconv < Nstop) ? Nconv : Nstop;
  // dump a clean .dat (Re/Im scatter) on the boss node, in addition to the log
  std::ofstream fout;
  bool wdat = (!datfile.empty()) && FGrid->IsBoss();
  if (wdat) {
    fout.open(datfile.c_str());
    fout << "# IRA shift-invert eigenvalues of " << label << " nearest sigma=" << sigma << std::endl;
    fout << "# i   Re(lambda)   Im(lambda)   |lambda-sigma|" << std::endl;
  }
  for (int i = 0; i < nshow; ++i) {
    ComplexD theta = eval[i];
    ComplexD lam = ComplexD(sigma, 0.0) + ComplexD(1.0, 0.0) / theta;  // lambda = sigma + 1/theta
    RealD dist = std::hypot(lam.real() - sigma, lam.imag());
    std::cout << "    lambda[" << i << "] = (" << lam.real() << ", " << lam.imag()
              << ")   |lambda-sigma| = " << dist << std::endl;
    if (wdat) {
      fout << i << "   " << lam.real() << "   " << lam.imag() << "   " << dist << std::endl;
    }
  }
  if (wdat) {
    fout.close();
    std::cout << "  [" << label << "] eigenvalues -> " << datfile << std::endl;
  }
}

// Singular-value + chirality (index) diagnostic on D_DW, done natively in 5D (Nobu's request).
// D_DW is gapped and Gamma5-R5-Hermitian (Gamma5 R5 D = D^dag Gamma5 R5) -> its eigenvalues are
// conjugate-paired and never hit zero for Q!=0, so topology is INVISIBLE in the D_DW eigenvalue spectrum
// (why the IRA runs above cannot see Q). It IS visible in the SINGULAR values: at finite physical mass m
// the would-be overlap zero mode is lifted only to sigma ~ O(m), sitting below the bulk gap. Lanczos on
// |D_DW|^2 = D^dag D (Hermitian PD) isolates that low cluster; the CHIRALITY of each low mode carries the
// index:
//   Q  ~  sum over near-zero modes of  <q|gamma5|q>,  q = physical surface field = P_- psi(0)+P_+ psi(Ls-1).
// Per mode we report sigma_i=sqrt(lambda_i), the physical-quark chirality chi_q (index proxy, ->\pm 1 for a
// zero mode), the 5D Gamma5-R5 chirality chi5 (cross-check), and the s=0 / s=Ls-1 wall weights (a surface
// mode localises on one wall). Chebyshev-accelerated IRL exactly as in tests/lanczos/Test_dwf_G5R5.cc; the
// window [cheb_lo,cheb_hi] must bracket spec(|D|^2) and cheb_lo must sit BELOW the smallest sigma^2 (~m^2)
// so the polynomial stays monotone on the spectrum (avoids the even/odd IRL reconstruction pitfall).
static void run_sv(MobiusFermionD& D, GridCartesian* UGrid, GridCartesian* FGrid, int Ls,
                   GridParallelRNG& RNG5, int Nstop, int Nk, int Nm,
                   RealD cheb_lo, RealD cheb_hi, int cheb_ord, RealD resid, int MaxIter, RealD mm) {
  std::cout << "==== Lanczos on |D_DW|^2 : singular values + chirality (index) ====" << std::endl;
  // ORDER PARITY (the Grid IRL Chebyshev pitfall): Chebyshev(lo,hi,order) is the single term T_{order-1}.
  // IRL keeps the LARGEST filtered value (partial_sort, std::greater in ImplicitlyRestartedLanczos.h). A
  // target mode sits BELOW cheb_lo at y<-1, where T_n(y) = (-1)^n T_n(|y|), n=order-1. For it to be large
  // POSITIVE (so IRL keeps it), need n EVEN, i.e. order ODD (Grid's own Test_dwf_lanczos uses order=4001).
  // An EVEN order makes the low modes large NEGATIVE -> sorted to the bottom and discarded. So force odd.
  if (cheb_ord % 2 == 0) {
    cheb_ord += 1;
    std::cout << "  [sv] cheb_ord was even -> bumped to " << cheb_ord << " (IRL needs ODD order)" << std::endl;
  }
  MdagMLinearOperator<MobiusFermionD, LatticeFermionD> HermMdagM(D);
  // The Chebyshev is bounded only INSIDE [cheb_lo, cheb_hi]; ABOVE cheb_hi it grows like cosh(ord*...) and
  // overflows (alpha ~ 1e128 garbage). So cheb_hi MUST bracket the TOP of spec(|D|^2). cheb_hi<=0 -> measure
  // lambda_max by power iteration and set cheb_hi = 1.1*lambda_max. (cheb_lo too low is harmless: it just
  // amplifies the small modes we are hunting.)
  if (cheb_hi <= 0.0) {
    LatticeFermionD pmsrc(FGrid);
    gaussian(RNG5, pmsrc);
    PowerMethod<LatticeFermionD> PM;
    RealD lmax = PM(HermMdagM, pmsrc);
    cheb_hi = 1.1 * lmax;
    std::cout << "  power-method lambda_max(|D|^2) = " << lmax << "  -> cheb_hi = " << cheb_hi << std::endl;
  }
  std::cout << "  Chebyshev(" << cheb_lo << ", " << cheb_hi << ", " << cheb_ord
            << ")  Nstop=" << Nstop << " Nk=" << Nk << " Nm=" << Nm << " resid=" << resid << std::endl;
  Chebyshev<LatticeFermionD> Cheby(cheb_lo, cheb_hi, cheb_ord);
  FunctionHermOp<LatticeFermionD> OpCheby(Cheby, HermMdagM);
  PlainHermOp<LatticeFermionD> Op(HermMdagM);
  ImplicitlyRestartedLanczos<LatticeFermionD> IRL(OpCheby, Op, Nstop, Nk, Nm, resid, MaxIter);

  std::vector<RealD> eval(Nm);
  std::vector<LatticeFermionD> evec(Nm, LatticeFermionD(FGrid));
  LatticeFermionD src(FGrid);
  gaussian(RNG5, src);
  int Nconv = 0;
  IRL.calc(eval, evec, src, Nconv);

  std::cout << "  [sv] Nconv=" << Nconv << "  (mass m=" << mm
            << " -> expect topological sigma ~ O(m), separated below the bulk gap)" << std::endl;
  Gamma g5(Gamma::Algebra::Gamma5);
  LatticeFermionD g5r5v(FGrid);
  LatticeFermionD q4(UGrid);
  LatticeFermionD g5q(UGrid);
  LatticeFermionD s0(UGrid);
  LatticeFermionD sL(UGrid);
  int nshow = (Nconv < Nstop) ? Nconv : Nstop;
  RealD sum_chiq = 0.0;
  RealD sum_chi5 = 0.0;
  for (int i = 0; i < nshow; ++i) {
    RealD lam = eval[i];
    RealD sigma = std::sqrt(lam > 0.0 ? lam : 0.0);
    RealD vn = norm2(evec[i]);
    // 5D Gamma5-R5 chirality  <v|Gamma5 R5|v>/<v|v>
    G5R5(g5r5v, evec[i]);
    RealD chi5 = real(innerProduct(evec[i], g5r5v)) / vn;
    // physical-quark chirality: q = P_- psi(0) + P_+ psi(Ls-1);  chi_q = <q|g5|q>/<q|q>
    D.ExportPhysicalFermionSolution(evec[i], q4);
    RealD qn = norm2(q4);
    g5q = g5 * q4;
    RealD chiq = real(innerProduct(q4, g5q)) / qn;
    // wall localisation: slice weights at s=0 and s=Ls-1 (5th dim is orthog=0)
    ExtractSlice(s0, evec[i], 0, 0);
    ExtractSlice(sL, evec[i], Ls - 1, 0);
    RealD n0 = norm2(s0) / vn;
    RealD nL = norm2(sL) / vn;
    std::cout << "    mode[" << i << "] sigma=" << sigma << "  lambda=" << lam
              << "  chi_q=" << chiq << "  chi5=" << chi5
              << "  wall(s=0)=" << n0 << "  wall(s=Ls-1)=" << nL << std::endl;
    sum_chiq += chiq;
    sum_chi5 += chi5;
  }
  std::cout << "  [sv] sum chi_q (index estimate ~ Q) = " << sum_chiq
            << "   sum chi5 = " << sum_chi5 << std::endl;
}

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  const double M5 = 1.8;
  const int Ls = 8;
  const double bb = 1.5;
  const double cc = 0.5;
  const double mm = 0.1;
  std::vector<Complex> boundary = {1, 1, 1, -1};

  // frame flow (default s/t0=0.4 on b2.6: eps 0.02 x nstep 58 = tau 1.16); IRA knobs; op selection.
  double flow_eps = 0.02;
  int flow_nstep = 58;
  int Nstop = 12;
  int Nk = 24;
  int Nm = 48;
  RealD eresid = 1.0e-2;  // loosened for now (Nobu); plain Arnoldi is stuck ~1e-2 on the interior modes
  int MaxIter = 200;
  int gf_maxit = 3000;    // same SMALL step alpha=0.1/16 (mimic existing); just more iters to finish Landau
  std::string which = "both";
  // shift-invert (default): find eigenvalues NEAREST sigma. D_DW -> physical Wilson branch sigma=-M5=-1.8;
  // M0 D_DW -> sigma~0 (near-zero stragglers + check no negative-real evals). --plain = old smallest-modulus.
  bool plain = false;
  RealD sigma_d = -1.8;
  RealD sigma_m0d = 0.0;
  RealD inner_tol = 1.0e-5;
  int inner_maxit = 500;
  int inner_restart = 50;   // GMRES restart length for the M0 D_DW shift-invert inner solve
  // --which sv : Lanczos on |D_DW|^2 (singular values) + chirality/index. Chebyshev filter:
  //   cheb_lo = CUT: modes with lambda < cheb_lo (sigma < sqrt(cheb_lo)) are extracted -> keep it ABOVE
  //             the target near-zero cluster (lambda ~ m^2 = 0.01). Raise to widen, lower to zoom in.
  //   cheb_hi <= 0 -> auto (power-method lambda_max); MUST bracket the top of spec(|D|^2) or it overflows.
  //   cheb_ord = polynomial order; forced ODD (see run_sv). Reuses Nstop/Nk/Nm.
  RealD cheb_lo = 0.5;
  RealD cheb_hi = -1.0;
  int cheb_ord = 21;
  RealD sv_resid = 1.0e-5;
  int sv_maxit = 200;

  if (GridCmdOptionExists(argv, argv + argc, "--flow_eps")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--flow_eps");
    GridCmdOptionFloat(a, flow_eps);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--flow_nstep")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--flow_nstep");
    GridCmdOptionInt(a, flow_nstep);
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
  if (GridCmdOptionExists(argv, argv + argc, "--eresid")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--eresid");
    GridCmdOptionFloat(a, eresid);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--maxit")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--maxit");
    GridCmdOptionInt(a, MaxIter);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--gf_maxit")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--gf_maxit");
    GridCmdOptionInt(a, gf_maxit);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--which")) {
    which = GridCmdOptionPayload(argv, argv + argc, "--which");
  }
  if (GridCmdOptionExists(argv, argv + argc, "--plain")) {
    plain = true;
  }
  if (GridCmdOptionExists(argv, argv + argc, "--sigma_d")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--sigma_d");
    GridCmdOptionFloat(a, sigma_d);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--sigma_m0d")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--sigma_m0d");
    GridCmdOptionFloat(a, sigma_m0d);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--inner_tol")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--inner_tol");
    GridCmdOptionFloat(a, inner_tol);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--inner_maxit")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--inner_maxit");
    GridCmdOptionInt(a, inner_maxit);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--inner_restart")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--inner_restart");
    GridCmdOptionInt(a, inner_restart);
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
  if (!GridCmdOptionExists(argv, argv + argc, "--config")) {
    std::cout << "ERROR: need --config <NERSC gauge file>" << std::endl;
    Grid_finalize();
    return 1;
  }
  std::string cfgfile = GridCmdOptionPayload(argv, argv + argc, "--config");
  // basename of the config, for the eigenvalue .dat dumps (absolute path, mirrors the sv .dat location)
  std::string cfgbase = cfgfile;
  size_t slashpos = cfgbase.find_last_of('/');
  if (slashpos != std::string::npos) cfgbase = cfgbase.substr(slashpos + 1);
  std::string dwevdat = "/projectnb/qfe/nmatsum/dwf/log/spectrum_dwevals_" + cfgbase + "_claude.dat";
  std::string m0devdat = "/projectnb/qfe/nmatsum/dwf/log/spectrum_m0devals_" + cfgbase + "_claude.dat";

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

  // ---- target operator D_DW on the ORIGINAL config ----
  WilsonImplD::ImplParams Params(boundary);
  MobiusFermionD D(U, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mm, M5, bb, cc, Params);

  // ---- singular-value + chirality (index) diagnostic on D_DW, native 5D (no frame/M0 needed) ----
  if (which == "sv") {
    run_sv(D, UGrid, FGrid, Ls, RNG5, Nstop, Nk, Nm, cheb_lo, cheb_hi, cheb_ord, sv_resid, sv_maxit, mm);
  }

  // ---- IRA on D_DW (no frame needed) ----
  if (which == "d" || which == "both") {
    if (plain) {
      ApplyDop<MobiusFermionD, LatticeFermionD> Aop(D);
      run_ira("D_DW", Aop, FGrid, RNG5, Nstop, Nk, Nm, eresid, MaxIter);
    } else {
      // CGNE shift-invert for D_DW (robust at sigma=-1.8 where (D-sigma) is near-singular).
      ShiftInvertCGNE<MobiusFermionD, LatticeFermionD> SI(D, sigma_d, inner_tol, inner_maxit, FGrid);
      run_ira_si("D_DW", SI, sigma_d, FGrid, RNG5, Nstop, Nk, Nm, eresid, MaxIter, dwevdat);
    }
  }

  // ---- M0 D_DW: only this branch needs the free-limit frame (flow -> Landau -> Omega -> M0) ----
  bool need_m0 = (which == "m0d" || which == "both");
  if (need_m0) {
    LatticeGaugeFieldD Uflowed(UGrid);
    WilsonFlow<PeriodicGimplD> wf(flow_eps, flow_nstep);
    wf.smear(Uflowed, U);
    LatticeColourMatrixD xform(UGrid);
    RealD gf_alpha = 0.1 / 16.0;
    FourierAcceleratedGaugeFixer<PeriodicGimplD>::SteepestDescentGaugeFix(
        Uflowed, xform, gf_alpha, gf_maxit, 1.0e-12, 1.0e-12, /*Fourier=*/true, /*orthog=*/-1,
        /*err_on_no_converge=*/false);
    Real landau = 1.0 - WilsonLoops<PeriodicGimplD>::linkTrace(Uflowed);
    Real Qflow = WilsonLoops<PeriodicGimplD>::TopologicalCharge5Li(Uflowed);
    std::cout << "frame: flow tau=" << (flow_eps * flow_nstep) << "  Landau=" << landau
              << "  Q_5Li(flowed)=" << Qflow << std::endl;
    FreeMobius5DInverse<WilsonImplD> Ffree(FGrid, Ls, M5, bb, cc, mm, boundary);
    FreeLimitPreconditioner<WilsonImplD> M0(Ffree, xform, FGrid);

    if (plain) {
      ApplyM0Dop<MobiusFermionD, LatticeFermionD> AopM0(D, M0, FGrid);
      run_ira("M0 D_DW", AopM0, FGrid, RNG5, Nstop, Nk, Nm, eresid, MaxIter);
    } else {
      ShiftedM0Dop<MobiusFermionD, LatticeFermionD> Ash0(D, M0, sigma_m0d, FGrid);
      ShiftInvert<LatticeFermionD> SI0(Ash0, inner_tol, inner_maxit, inner_restart);
      run_ira_si("M0 D_DW", SI0, sigma_m0d, FGrid, RNG5, Nstop, Nk, Nm, eresid, MaxIter, m0devdat);
    }
  }

  Grid_finalize();
  return 0;
}
