/*************************************************************************************
  Test_dwf_wtop_eigs_claude.cc -- CHUNK 1 of the topological-corrected free preconditioner.
  Plan: tests/solver/wtop_precond_impl_plan_claude.md ; design: topological_precond_correction_claude.md.

  Computes the would-be-topological low modes of the 4D framed MASSLESS Wilson kernel D_W(0)[U_Omega]
  (Nobu: only the massless Wilson kernel matters; the M5 shift only displaces eigenvalues, and the
  chiral zero modes sit at lambda ~ 0 for D_W(0)). Produces the DECISION DATA for "which mode set
  feeds the projector P":
    (b) Hermitian set: lowest eigenvectors psi_i of H_W^2 = D_W^H D_W (= right singular vectors of D_W),
        via Chebyshev-IRL. Cheap, robust, chi = psi. |mu_i| = sqrt(eval); sign(mu_i) = psi^H H_W psi.
    (a) Non-Herm set: DIRECT shift-invert IRA on D_W (sigma=0) -> smallest-|lambda| eigenpairs
        {phi_i, lambda_i} of D_W, with chirality chi_a = <phi|g5|phi> and residual ||D_W phi - lambda phi||.
        This is the honest non-Hermitian eigensolve (a Rayleigh-Ritz-in-the-singular-subspace attempt was
        abandoned: it scrambles the +-sigma-degenerate chirality partners into non-chiral combinations).

  Frame U_Omega = GaugeTransform(original U, xform), xform = FA-Landau fix of the flowed config
  (prerot picture: D_W(-M5) on the framed gauge, so bare F = D_W(1) is its free model). Flow cached.

  D_W here is the MASSLESS Wilson operator (mass = 0 default; --mass_w to scan the additively-renormalized
  critical point). gamma5-Hermiticity D_W^H = g5 D_W g5 gives the left vector free (chi = g5 phi for real
  lambda). Chebyshev-IRL: Grid ImplicitlyRestartedLanczos. (M5-shift refs, for the kernel connection:
  Edwards-Heller-Narayanan Nucl.Phys.B540 (1999) 457; Neuberger Phys.Lett.B417 (1998) 141.)

  CLI: --config <NERSC> [--nstop 16] [--nk 24] [--nm 48] [--flow_nstep 873] [--flow_eps 0.02]
       [--cheb_lo 0.01] [--cheb_ord 61] [--gf_maxit 4000] [--flowcache <dir>] [--ckpt_dir <dir>]
       [--nreport 12]
*************************************************************************************/
#include "twolevel_common_claude.h"
#include <Grid/algorithms/iterative/ImplicitlyRestartedArnoldi_claude.h>   // shift-invert IRA (Sorensen 1992)
#include <Grid/qcd/utils/FreeWilson_claude.h>                              // 4D free-Wilson inverse (M0) for C(phi)
#include <fstream>
#include <sstream>

using namespace Grid;

// B = D_W (4D, non-Hermitian) as a LinearOperatorBase: Op(in,out) = D_W in. Only Op is used (GMRES).
class DWOp : public LinearOperatorBase<LatticeFermionD> {
  WilsonFermionD& D;
public:
  DWOp(WilsonFermionD& D_) : D(D_) {}
  void Op(const LatticeFermionD& in, LatticeFermionD& out) { D.M(in, out); }
  void AdjOp(const LatticeFermionD& in, LatticeFermionD& out) { GRID_ASSERT(0); }
  void OpDiag(const LatticeFermionD& in, LatticeFermionD& out) { GRID_ASSERT(0); }
  void OpDir(const LatticeFermionD& in, LatticeFermionD& out, int dir, int disp) { GRID_ASSERT(0); }
  void OpDirAll(const LatticeFermionD& in, std::vector<LatticeFermionD>& out) { GRID_ASSERT(0); }
  void HermOpAndNorm(const LatticeFermionD& in, LatticeFermionD& out, RealD& n1, RealD& n2) { GRID_ASSERT(0); }
  void HermOp(const LatticeFermionD& in, LatticeFermionD& out) { GRID_ASSERT(0); }
};

// (D_W)^{-1} via plain GMRES (sigma=0). D_W(0) is well conditioned (sigma_min ~ 0.6) -> cheap inner solve.
// LinearFunction for IRA; IRA(largestModulus) on (D_W)^{-1} returns theta = 1/lambda -> smallest-|lambda|.
class ShiftInvertDW : public LinearFunction<LatticeFermionD> {
  LinearOperatorBase<LatticeFermionD>& B;
  RealD tol;
  int maxit;
  int restart;
public:
  ShiftInvertDW(LinearOperatorBase<LatticeFermionD>& B_, RealD tol_, int maxit_, int restart_)
    : B(B_), tol(tol_), maxit(maxit_), restart(restart_) {}
  void operator()(const LatticeFermionD& in, LatticeFermionD& out) {
    out = Zero();
    GeneralisedMinimalResidual<LatticeFermionD> GMRES(tol, maxit, restart, false);
    GMRES(B, in, out);   // solve D_W out = in
  }
};

// Quality measure C(phi) = ||(1 - M0 D_W) phi|| / ||phi||  (Test_wilson_frameopt_claude.cc:149).
// 4D operator D_W at mass_c (D_W(mass_c) phi = DW.M(phi) + mass_c phi, mass is additive), 4D preconditioner
// M0 = bare free-Wilson inverse Fw at mprec (prerot: frame folded into Urot -> no Omega multiply here).
// C -> 0 : phi is free-like (correction pointless there). C -> O(1) / >=1 : M0 fails on phi (correction relevant).
static RealD quality_C(const LatticeFermionD& phi, WilsonFermionD& DW, RealD mass_c,
                       LinearFunction<LatticeFermionD>& Fw) {
  LatticeFermionD dwphi(phi.Grid());
  LatticeFermionD m0dwphi(phi.Grid());
  LatticeFermionD r(phi.Grid());
  DW.M(phi, dwphi);
  axpy(dwphi, mass_c, phi, dwphi);   // dwphi = mass_c*phi + D_W(0)phi = D_W(mass_c) phi
  Fw(dwphi, m0dwphi);                // M0 (D_W phi)
  r = phi - m0dwphi;
  return std::sqrt(norm2(r) / norm2(phi));
}

// ---- flow cache (double NERSC, LIME-free) -- same helpers as Test_dwf_freeprec_claude.cc ----
static std::string flow_label(const std::string& cfg_base, double eps, int nstep) {
  std::ostringstream os;
  os << "flow_" << cfg_base << "_wilson_eps" << eps << "_n" << nstep;
  return os.str();
}
static std::string flow_cache_file(const std::string& dir, const std::string& cfg_base, double eps, int nstep) {
  return dir + "/" + flow_label(cfg_base, eps, nstep) + ".nersc";
}
// frame cache: Urot = Omega U Omega^dag keyed by (config, flow eps/nstep, gf_maxit) -> skip flow + Landau.
static std::string urot_file(const std::string& dir, const std::string& cfg_base, double eps, int nstep, int gfmax) {
  std::ostringstream os;
  os << dir << "/urot_" << cfg_base << "_wilson_eps" << eps << "_n" << nstep << "_gf" << gfmax << ".nersc";
  return os.str();
}
static bool flow_cache_exists(const std::string& path) {
  std::ifstream f(path.c_str());
  return f.good();
}

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  const double M5 = TL_M5;          // wall height 1.8 (kept for reference/labels)
  // UNSHIFTED Wilson (Nobu): the topological chiral zero modes cluster at lambda ~ 0. The mass in Grid's
  // Wilson op is a PURE additive shift D_W(m) = D_W(0) + m -> eigenVECTORS are shift-independent (same
  // topological modes), only eigenVALUES move. The -M5 shift pushes the chiral modes to lambda ~ -M5, and
  // the SMALL SINGULAR values of the SHIFTED op are then the bulk/mobility-edge (low chirality), NOT the
  // chiral zero modes. So identify the modes on D_W(0): chiral zero modes at lambda ~ 0, cleanly.
  double mass_W = 0.0;              // --mass_w to scan the additive-renormalized critical point if needed
  std::vector<Complex> boundary = {1, 1, 1, -1};   // anti-periodic time

  int Nstop = 16;
  int Nk = 24;
  int Nm = 48;
  int flow_nstep = 873;
  RealD flow_eps = 0.02;
  RealD cheb_lo = 0.01;
  int cheb_ord = 61;
  int gf_maxit = 4000;
  int nreport = 12;
  RealD inner_tol = 1.0e-8;      // (a) shift-invert IRA inner GMRES on D_W
  int inner_restart = 100;
  int inner_maxit = 2000;
  RealD ira_eresid = 1.0e-7;     // IRA outer convergence
  int ira_maxit = 200;
  // quality measure C(phi) = ||(1 - M0 D_W) phi||/||phi|| (Test_wilson_frameopt_claude.cc:149).
  // MASSLESS throughout (Nobu): 4D operator D_W(0) and 4D preconditioner M0 = bare free-Wilson inverse
  // Fw(0) (prerot: frame folded into Urot). Fw(0) is invertible because AP-time BC shifts the lowest
  // momentum to (0,0,0,+-pi/L) -> no exact free zero mode. Matches the massless kernel used in the IRA.
  double mass_c = 0.0;
  double mprec = 0.0;
  std::string flowcache_dir = "";
  std::string ckpt_dir = "";
  if (GridCmdOptionExists(argv, argv + argc, "--nstop")) Nstop = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--nstop"));
  if (GridCmdOptionExists(argv, argv + argc, "--nk")) Nk = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--nk"));
  if (GridCmdOptionExists(argv, argv + argc, "--nm")) Nm = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--nm"));
  if (GridCmdOptionExists(argv, argv + argc, "--flow_nstep")) flow_nstep = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--flow_nstep"));
  if (GridCmdOptionExists(argv, argv + argc, "--flow_eps")) flow_eps = std::stod(GridCmdOptionPayload(argv, argv + argc, "--flow_eps"));
  if (GridCmdOptionExists(argv, argv + argc, "--cheb_lo")) cheb_lo = std::stod(GridCmdOptionPayload(argv, argv + argc, "--cheb_lo"));
  if (GridCmdOptionExists(argv, argv + argc, "--cheb_ord")) cheb_ord = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--cheb_ord"));
  if (GridCmdOptionExists(argv, argv + argc, "--gf_maxit")) gf_maxit = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--gf_maxit"));
  if (GridCmdOptionExists(argv, argv + argc, "--nreport")) nreport = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--nreport"));
  if (GridCmdOptionExists(argv, argv + argc, "--mass_w")) mass_W = std::stod(GridCmdOptionPayload(argv, argv + argc, "--mass_w"));
  if (GridCmdOptionExists(argv, argv + argc, "--inner_tol")) inner_tol = std::stod(GridCmdOptionPayload(argv, argv + argc, "--inner_tol"));
  if (GridCmdOptionExists(argv, argv + argc, "--inner_restart")) inner_restart = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--inner_restart"));
  if (GridCmdOptionExists(argv, argv + argc, "--inner_maxit")) inner_maxit = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--inner_maxit"));
  if (GridCmdOptionExists(argv, argv + argc, "--mass_c")) mass_c = std::stod(GridCmdOptionPayload(argv, argv + argc, "--mass_c"));
  if (GridCmdOptionExists(argv, argv + argc, "--mprec")) mprec = std::stod(GridCmdOptionPayload(argv, argv + argc, "--mprec"));
  if (GridCmdOptionExists(argv, argv + argc, "--flowcache")) flowcache_dir = GridCmdOptionPayload(argv, argv + argc, "--flowcache");
  if (GridCmdOptionExists(argv, argv + argc, "--ckpt_dir")) ckpt_dir = GridCmdOptionPayload(argv, argv + argc, "--ckpt_dir");

  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(
      GridDefaultLatt(), GridDefaultSimd(Nd, vComplexD::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);

  GridParallelRNG RNG4(UGrid);
  RNG4.SeedFixedIntegers({1, 2, 3, 4});

  if (!GridCmdOptionExists(argv, argv + argc, "--config")) {
    std::cout << GridLogError << "--config required" << std::endl;
    Grid_finalize();
    return 1;
  }
  std::string cfg = GridCmdOptionPayload(argv, argv + argc, "--config");
  std::string cfg_base = cfg.substr(cfg.find_last_of('/') + 1);
  LatticeGaugeFieldD U(UGrid);
  FieldMetaData header;
  NerscIO::readConfiguration(U, header, cfg);
  std::cout << GridLogMessage << "loaded " << cfg << "  plaq="
            << WilsonLoops<PeriodicGimplD>::avgPlaquette(U) << std::endl;

  // ---- frame: Urot = Omega U Omega^dag, CACHED (skips flow + Landau on rerun; shared with precmodes) ----
  LatticeGaugeFieldD Urot(UGrid);
  bool urot_loaded = false;
  std::string urotpath;
  if (!flowcache_dir.empty()) {
    urotpath = urot_file(flowcache_dir, cfg_base, flow_eps, flow_nstep, gf_maxit);
    if (flow_cache_exists(urotpath)) {
      FieldMetaData h;
      NerscIO::readConfiguration(Urot, h, urotpath);
      urot_loaded = true;
      std::cout << GridLogMessage << "[urotcache] LOADED " << urotpath << " (flow + Landau skipped)" << std::endl;
    }
  }
  if (!urot_loaded) {
    LatticeGaugeFieldD Uflowed(UGrid);
    bool flow_loaded = false;
    if (!flowcache_dir.empty()) {
      std::string flowpath = flow_cache_file(flowcache_dir, cfg_base, flow_eps, flow_nstep);
      if (flow_cache_exists(flowpath)) {
        FieldMetaData h;
        NerscIO::readConfiguration(Uflowed, h, flowpath);
        flow_loaded = true;
        std::cout << GridLogMessage << "[flowcache] LOADED " << flowpath << " (flow skipped)" << std::endl;
      }
    }
    if (!flow_loaded) {
      WilsonFlow<PeriodicGimplD> wf(flow_eps, flow_nstep);
      wf.smear(Uflowed, U);
      if (!flowcache_dir.empty()) {
        std::string flowpath = flow_cache_file(flowcache_dir, cfg_base, flow_eps, flow_nstep);
        NerscIO::writeConfiguration(Uflowed, flowpath, "DWF", "FLOWCACHE", 1);
        std::cout << GridLogMessage << "[flowcache] SAVED " << flowpath << std::endl;
      }
    }
    LatticeColourMatrixD xform(UGrid);
    FourierAcceleratedGaugeFixer<PeriodicGimplD>::SteepestDescentGaugeFix(
        Uflowed, xform, 0.1 / 16.0, gf_maxit, 1.0e-12, 1.0e-12, true, -1, false);
    std::cout << GridLogMessage << "  Landau functional=" << (1.0 - WilsonLoops<PeriodicGimplD>::linkTrace(Uflowed))
              << "  Q_5Li=" << WilsonLoops<PeriodicGimplD>::TopologicalCharge5Li(Uflowed) << std::endl;
    Urot = U;
    SU<Nc>::GaugeTransform<PeriodicGimplD>(Urot, xform);   // U_Omega = Omega U Omega^dag
    if (!flowcache_dir.empty()) {
      NerscIO::writeConfiguration(Urot, urotpath, "DWF", "UROTCACHE", 1);
      std::cout << GridLogMessage << "[urotcache] SAVED " << urotpath << std::endl;
    }
  }

  // ---- 4D Wilson kernel D_W(-M5) on the framed gauge ----
  WilsonImplD::ImplParams Params(boundary);
  WilsonFermionD DW(Urot, *UGrid, *UrbGrid, mass_W, Params);
  Gamma g5(Gamma::Algebra::Gamma5);

  // ============ set (b): Hermitian H_W^2 = D_W^H D_W low modes (Chebyshev-IRL) ============
  std::cout << GridLogMessage << "==== (b) lowest " << Nstop << " modes of D_W^H D_W (Chebyshev-IRL) ====" << std::endl;
  int co = (cheb_ord % 2 == 0) ? cheb_ord + 1 : cheb_ord;   // IRL keeps LARGEST filtered -> ODD order
  MdagMLinearOperator<WilsonFermionD, LatticeFermionD> HermMdagM(DW);
  LatticeFermionD pmsrc(UGrid);
  gaussian(RNG4, pmsrc);
  PowerMethod<LatticeFermionD> PM;
  RealD lmax = PM(HermMdagM, pmsrc);
  RealD cheb_hi = 1.1 * lmax;
  std::cout << GridLogMessage << "  power-method lambda_max(D_W^H D_W)=" << lmax << "  cheb(" << cheb_lo
            << "," << cheb_hi << "," << co << ")" << std::endl;
  Chebyshev<LatticeFermionD> Cheby(cheb_lo, cheb_hi, co);
  FunctionHermOp<LatticeFermionD> OpCheby(Cheby, HermMdagM);
  PlainHermOp<LatticeFermionD> OpPlain(HermMdagM);
  ImplicitlyRestartedLanczos<LatticeFermionD> IRL(OpCheby, OpPlain, Nstop, Nk, Nm, 1.0e-9, 200);
  std::vector<RealD> eval(Nm);
  std::vector<LatticeFermionD> evec(Nm, LatticeFermionD(UGrid));
  LatticeFermionD src(UGrid);
  gaussian(RNG4, src);
  int Nconv = 0;
  IRL.calc(eval, evec, src, Nconv);
  int n = (Nconv < Nstop) ? Nconv : Nstop;
  std::cout << GridLogMessage << "  IRL Nconv=" << Nconv << "  keeping n=" << n << std::endl;

  // psi_i (orthonormal), mu_i = sqrt(eval_i) with sign from psi^H H_W psi ; chirality chi_i = <psi|g5|psi>
  std::vector<LatticeFermionD> psi(n, LatticeFermionD(UGrid));
  std::vector<RealD> muB(n);
  std::vector<RealD> chiB(n);
  std::vector<RealD> residHB(n);   // ||H_W psi - mu psi||  (verify psi is a clean H_W eigenvector)
  LatticeFermionD DWpsi(UGrid);
  LatticeFermionD g5psi(UGrid);
  LatticeFermionD rHW(UGrid);
  for (int i = 0; i < n; ++i) {
    psi[i] = evec[i];
    RealD nn = std::sqrt(norm2(psi[i]));
    psi[i] = psi[i] * (1.0 / nn);
    DW.M(psi[i], DWpsi);                       // D_W psi
    g5psi = g5 * DWpsi;                        // g5 D_W psi = H_W psi
    RealD hw = real(innerProduct(psi[i], g5psi));   // mu_i = psi^H H_W psi (real)
    muB[i] = hw;
    rHW = g5psi - muB[i] * psi[i];             // H_W psi - mu psi  (psi normalized)
    residHB[i] = std::sqrt(norm2(rHW));
    g5psi = g5 * psi[i];
    chiB[i] = real(innerProduct(psi[i], g5psi));    // <psi|g5|psi>
  }

  // ============ set (a): DIRECT shift-invert IRA on the non-Hermitian D_W (Nobu) ============
  // Smallest-|lambda| eigenpairs of D_W directly, via IRA(largestModulus) on (D_W)^{-1} (sigma=0).
  // D_W(0) is well conditioned here (sigma_min ~ 0.6) so the inner GMRES on D_W is cheap. Gives the TRUE
  // D_W eigenvectors -- no singular-subspace Rayleigh-Ritz detour (which scrambled the +-degenerate
  // chirality partners). IRA: Sorensen 1992; wrapper mirrors Test_dwf_m0d5_overlap_claude.cc.
  std::cout << GridLogMessage << "==== (a) shift-invert IRA on D_W (sigma=0, non-Hermitian) ====" << std::endl;
  DWOp Bop(DW);
  ShiftInvertDW SIop(Bop, inner_tol, inner_maxit, inner_restart);
  ImplicitlyRestartedArnoldi<LatticeFermionD> IRA(SIop, Nstop, Nk, Nm, ira_eresid, ira_maxit, IRAlargestModulus);
  std::vector<ComplexD> theta(Nm);
  std::vector<LatticeFermionD> Wv(Nm, LatticeFermionD(UGrid));
  LatticeFermionD asrc(UGrid);
  gaussian(RNG4, asrc);
  int NconvA = 0;
  IRA.calc(theta, Wv, asrc, NconvA);
  int na = (NconvA < Nstop) ? NconvA : Nstop;
  std::cout << GridLogMessage << "  IRA Nconv=" << NconvA << "  keeping na=" << na << std::endl;

  // lambda = 1/theta (sigma=0) ; normalize ; chirality ; residual ; then sort by |lambda| ascending
  std::vector<ComplexD> lamA(na);
  std::vector<RealD> chiA(na);
  std::vector<RealD> residA(na);    // ||D_W phi - lambda phi||        (RIGHT eigenvector)
  std::vector<RealD> residLA(na);   // ||D_W^H (g5 phi) - lambda* (g5 phi)||  (LEFT eigenvector chi=g5 phi)
  std::vector<LatticeFermionD> phi(na, LatticeFermionD(UGrid));
  LatticeFermionD tmp(UGrid);
  LatticeFermionD Dphi(UGrid);
  LatticeFermionD chiv(UGrid);
  LatticeFermionD DHchi(UGrid);
  for (int i = 0; i < na; ++i) {
    phi[i] = Wv[i];
    RealD nn = std::sqrt(norm2(phi[i]));
    phi[i] = phi[i] * (1.0 / nn);
    lamA[i] = ComplexD(1.0, 0.0) / theta[i];
    DW.M(phi[i], Dphi);
    tmp = Dphi - lamA[i] * phi[i];
    residA[i] = std::sqrt(norm2(tmp));               // ||D_W phi - lambda phi|| (phi normalized)
    chiv = g5 * phi[i];                              // chi = g5 phi (candidate LEFT eigenvector)
    chiA[i] = real(innerProduct(phi[i], chiv));      // <phi|g5|phi> = chi^H phi (biorthonormalization)
    DW.Mdag(chiv, DHchi);                            // D_W^H chi
    ComplexD lamconj(lamA[i].real(), -lamA[i].imag());   // lambda*
    tmp = DHchi - lamconj * chiv;                    // D_W^H chi - lambda* chi
    residLA[i] = std::sqrt(norm2(tmp));              // ||.||  (chi normalized: ||g5 phi||=||phi||=1)
  }
  std::vector<std::pair<double, int>> ord(na);
  for (int i = 0; i < na; ++i) {
    ord[i] = std::make_pair(std::hypot(lamA[i].real(), lamA[i].imag()), i);
  }
  std::sort(ord.begin(), ord.end());

  // 4D free-Wilson preconditioner M0 = bare Fw(mprec) for the quality measure C(phi) (prerot -> no frame).
  FreeWilsonInverse<WilsonImplD> Fw(UGrid, mprec, boundary);
  // M1 = next-order preconditioner Ω†{Fw - Fw D(tildeA) Fw}Ω (FreeWilson_claude.h). Prerot -> identity
  // frame; framed operator D(tildeA) uses DW = D_W[Urot] (already the framed config). Same mass (0).
  LatticeColourMatrixD idxf(UGrid);
  idxf = ComplexD(1.0, 0.0);   // identity ColourMatrix (Grid: matrix = scalar -> identity*scalar)
  FreeLimitPreconditionerW1<WilsonImplD> M1(Fw, idxf, DW, UGrid);
  std::cout << GridLogMessage << "==== quality measure C(phi) = ||(1 - M D_W) phi||/||phi||  (M = M0, M1) : mass_c="
            << mass_c << " mprec=" << mprec << " ====" << std::endl;

  // ============ report: set (b) singular + set (a) direct eigen, chirality + net index + C ============
  std::cout << GridLogMessage << "==== (b) Hermitian singular modes of D_W (|mu| = sigma) ====" << std::endl;
  std::cout << GridLogMessage << "  i    mu_i           |mu|          chi_b          C_M0          C_M1          residH(H_W psi-mu psi)" << std::endl;
  int nchiB = 0;
  RealD idxB = 0.0;
  for (int i = 0; i < n; ++i) {
    if (std::abs(chiB[i]) > 0.8) {
      nchiB++;
      idxB += (chiB[i] > 0 ? 1.0 : -1.0);
    }
    RealD Cb = quality_C(psi[i], DW, mass_c, Fw);
    RealD Cb1 = quality_C(psi[i], DW, mass_c, M1);
    std::cout << GridLogMessage << "  " << i << "   " << muB[i] << "   " << std::abs(muB[i])
              << "   " << chiB[i] << "   " << Cb << "   " << Cb1 << "   " << residHB[i]
              << (std::abs(chiB[i]) > 0.8 ? "  (chiral)" : "")
              << (Cb1 >= 1.0 ? "  C_M1>=1(RED)" : "") << std::endl;
  }
  std::cout << GridLogMessage << "  (b) #|chi|>0.8 = " << nchiB << "   net signed index = " << idxB << std::endl;

  // The INDEX lives on the REAL eigenvalues of D_W (gamma5-Herm -> complex evals come in conjugate pairs
  // with chi ~ 0; real evals carry definite-sign chirality = the would-be-topological modes). So the
  // honest index = sum of sign(chi) over real eigenvalues (|Im lambda| < im_tol), NOT a |chi| threshold.
  RealD im_tol = 1.0e-4;   // |Im lambda| below this -> treat as a real eigenvalue
  std::cout << GridLogMessage << "==== (a) DIRECT D_W eigenmodes (shift-invert IRA), sorted by |lambda| ====" << std::endl;
  std::cout << GridLogMessage << "  k    Re(lam)         Im(lam)        |lam|         chi_a         residR        residL        C_M0          C_M1     type" << std::endl;
  int nreal = 0;
  RealD idxA = 0.0;
  int nra = (nreport < na) ? nreport : na;
  for (int k = 0; k < na; ++k) {
    int i = ord[k].second;
    bool isreal = (std::abs(lamA[i].imag()) < im_tol);
    if (isreal) {
      nreal++;
      idxA += (chiA[i] > 0 ? 1.0 : -1.0);
    }
    if (k < nra) {
      RealD absl = std::hypot(lamA[i].real(), lamA[i].imag());
      RealD Ca = quality_C(phi[i], DW, mass_c, Fw);
      RealD Ca1 = quality_C(phi[i], DW, mass_c, M1);
      std::cout << GridLogMessage << "  " << k << "   " << lamA[i].real() << "   " << lamA[i].imag()
                << "   " << absl << "   " << chiA[i] << "   " << residA[i] << "   " << residLA[i]
                << "   " << Ca << "   " << Ca1
                << (isreal ? "  REAL(chiral)" : "  complex") << (Ca1 >= 1.0 ? "  C_M1>=1(RED)" : "") << std::endl;
    }
  }
  std::cout << GridLogMessage << "  (a) #real eigenvalues (|Im|<" << im_tol << ") = " << nreal
            << "   net index = sum sign(chi_real) = " << idxA
            << "   => |Q| ~ " << (int)std::lround(std::abs(idxA)) << std::endl;

  // ---- checkpoint both 4D sets for chunks 2-4 (LIME-free BinaryIO) ----
  if (!ckpt_dir.empty()) {
    std::vector<RealD> muBv(muB.begin(), muB.begin() + n);
    save_deflation_vectors(psi, muBv, ckpt_dir, "wtop_psi");   // (b) Hermitian singular vecs + |mu|
    std::vector<LatticeFermionD> phiSorted;
    std::vector<RealD> lamReSorted;
    for (int k = 0; k < na; ++k) {
      phiSorted.push_back(phi[ord[k].second]);
      lamReSorted.push_back(lamA[ord[k].second].real());
    }
    save_deflation_vectors(phiSorted, lamReSorted, ckpt_dir, "wtop_phi");   // (a) direct eigvecs + Re(lambda)
    std::string lf = ckpt_dir + "/wtop_phi_lambda.txt";
    std::ofstream los(lf);
    los << "# k  Re(lambda)  Im(lambda)  chi_a\n";
    for (int k = 0; k < na; ++k) {
      int i = ord[k].second;
      los << k << "  " << std::setprecision(16) << lamA[i].real() << "  " << lamA[i].imag()
          << "  " << chiA[i] << "\n";
    }
    los.close();
    std::cout << GridLogMessage << "  checkpointed wtop_psi (b) + wtop_phi (a) + wtop_phi_lambda.txt to " << ckpt_dir << std::endl;
  }

  std::cout << GridLogMessage << "wtop_eigs done" << std::endl;
  Grid_finalize();
  return 0;
}
