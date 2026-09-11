/*************************************************************************************
  Test_dwf_precmodes_claude.cc -- smallest-|mu| (near-null) modes of the PRECONDITIONED 4D operators
  M0 D_W and M1 D_W (both MASSLESS). Companion to Test_dwf_wtop_eigs_claude.cc.
  Plan: tests/solver/wtop_precond_impl_plan_claude.md.

  We care about the modes around mu=0: a near-null mode of M D_W is a straggler that stalls the solver.
  We look for them via IRA(largestModulus) on (M D_W)^{-1} (shift-invert, sigma=0): theta = largest
  modulus eigenvalue of the inverse = smallest |mu| of M D_W ; mu = 1/theta ; C = |1 - mu| ; chirality.
  Finding NOTHING near 0 (mu_min bounded away from 0) is the GOOD outcome: the preconditioner has no
  near-null straggler. The scan's real output is mu_min(mprec) -- which mprec keeps M D_W farthest from 0.
  At good mprec, M D_W ~ 1 -> the smallest-|mu| modes are a cluster (~0.5, far from 0); IRA converges the
  isolated one and the cluster is degenerate, so a SMALL Nstop + modest ira_maxit terminate cleanly.
  Quality C: Test_wilson_frameopt_claude.cc:149. IRA: Sorensen 1992. Wrapper: Test_dwf_m0d5_overlap.

  4D massless: D_W(0) on the gauge-rotated ROUGH config Urot = Omega U Omega^dag (prerot -> M0 = bare Fw,
  M1 = Fw - Fw D(tildeA) Fw, framed op DW = D_W[Urot]). Fw at mprec; AP-time BC regularizes the free p=0
  zero. Frame Urot CACHED (NERSC) keyed by (config, flow eps/nstep, gf_maxit) -> reruns skip flow+Landau.

  CLI: --config <NERSC> [--nstop 4] [--nk 8] [--nm 16] [--mprec 0.8] [--mass_c 0] [--flow_nstep 873]
       [--flow_eps 0.02] [--gf_maxit 4000] [--inner_tol 1e-6] [--inner_restart 200] [--inner_maxit 4000]
       [--ira_eresid 1e-4] [--ira_maxit 30] [--nreport 8] [--flowcache <dir>]
*************************************************************************************/
#include <Grid/Grid.h>
#include <Grid/qcd/utils/FreeWilson_claude.h>
#include <Grid/algorithms/iterative/ImplicitlyRestartedArnoldi_claude.h>
#include <fstream>
#include <sstream>

using namespace Grid;

// Preconditioned operator M D_W (non-Hermitian): Op(in) = M(D_W in). M = M0 (bare Fw) or M1.
class PrecDwOp : public LinearOperatorBase<LatticeFermionD> {
  WilsonFermionD& D;
  LinearFunction<LatticeFermionD>& M;
  LatticeFermionD t;
public:
  PrecDwOp(WilsonFermionD& D_, LinearFunction<LatticeFermionD>& M_, GridBase* g) : D(D_), M(M_), t(g) {}
  void Op(const LatticeFermionD& in, LatticeFermionD& out) { D.M(in, t); M(t, out); }
  void AdjOp(const LatticeFermionD& in, LatticeFermionD& out) { GRID_ASSERT(0); }
  void OpDiag(const LatticeFermionD& in, LatticeFermionD& out) { GRID_ASSERT(0); }
  void OpDir(const LatticeFermionD& in, LatticeFermionD& out, int dir, int disp) { GRID_ASSERT(0); }
  void OpDirAll(const LatticeFermionD& in, std::vector<LatticeFermionD>& out) { GRID_ASSERT(0); }
  void HermOpAndNorm(const LatticeFermionD& in, LatticeFermionD& out, RealD& n1, RealD& n2) { GRID_ASSERT(0); }
  void HermOp(const LatticeFermionD& in, LatticeFermionD& out) { GRID_ASSERT(0); }
};

// (M D_W)^{-1} via plain GMRES (sigma=0). LinearFunction for IRA; IRA(largestModulus) -> smallest |mu|.
class ShiftInvertB : public LinearFunction<LatticeFermionD> {
  LinearOperatorBase<LatticeFermionD>& B;
  RealD tol;
  int maxit;
  int restart;
public:
  ShiftInvertB(LinearOperatorBase<LatticeFermionD>& B_, RealD tol_, int maxit_, int restart_)
    : B(B_), tol(tol_), maxit(maxit_), restart(restart_) {}
  void operator()(const LatticeFermionD& in, LatticeFermionD& out) {
    out = Zero();
    GeneralisedMinimalResidual<LatticeFermionD> GMRES(tol, maxit, restart, false);
    GMRES(B, in, out);
  }
};

// ---- frame cache (Urot = Omega U Omega^dag, NERSC, LIME-free) ----
static std::string flow_label(const std::string& cfg_base, double eps, int nstep) {
  std::ostringstream os;
  os << "flow_" << cfg_base << "_wilson_eps" << eps << "_n" << nstep;
  return os.str();
}
static std::string urot_file(const std::string& dir, const std::string& cfg_base, double eps, int nstep, int gfmax) {
  std::ostringstream os;
  os << dir << "/urot_" << cfg_base << "_wilson_eps" << eps << "_n" << nstep << "_gf" << gfmax << ".nersc";
  return os.str();
}
static bool file_exists(const std::string& path) {
  std::ifstream f(path.c_str());
  return f.good();
}

// shift-invert IRA on B = M D_W (sigma=0) -> smallest-|mu| modes; report mu, C=|1-mu|, chirality, resid.
static void precmode_scan(const char* mname, WilsonFermionD& DW, LinearFunction<LatticeFermionD>& M,
                          GridCartesian* UGrid, GridParallelRNG& RNG4, Gamma& g5,
                          int Nstop, int Nk, int Nm, RealD inner_tol, int inner_maxit, int inner_restart,
                          RealD ira_eresid, int ira_maxit, int nreport) {
  std::cout << GridLogMessage << "==== near-null modes of " << mname
            << " : shift-invert IRA (sigma=0, smallest |mu|) ====" << std::endl;
  PrecDwOp B(DW, M, UGrid);
  ShiftInvertB SIop(B, inner_tol, inner_maxit, inner_restart);
  ImplicitlyRestartedArnoldi<LatticeFermionD> IRA(SIop, Nstop, Nk, Nm, ira_eresid, ira_maxit, IRAlargestModulus);
  std::vector<ComplexD> theta(Nm);
  std::vector<LatticeFermionD> W(Nm, LatticeFermionD(UGrid));
  LatticeFermionD src(UGrid);
  gaussian(RNG4, src);
  int Nconv = 0;
  IRA.calc(theta, W, src, Nconv);
  int nl = (Nconv < Nstop) ? Nconv : Nstop;
  std::cout << GridLogMessage << "  IRA Nconv=" << Nconv << "  keeping " << nl << " near-null modes of " << mname << std::endl;
  if (nl <= 0) {
    std::cout << GridLogMessage << "  (no converged near-null mode -> M D_W bounded away from 0; GOOD)" << std::endl;
    return;
  }

  // mu = 1/theta ; C = |1 - mu| ; chirality ; residual ||M D_W w - mu w|| ; sort by |mu| ASC (nearest 0 first)
  std::vector<ComplexD> mu(nl);
  std::vector<RealD> Cq(nl), chi(nl), resid(nl);
  std::vector<LatticeFermionD> w(nl, LatticeFermionD(UGrid));
  LatticeFermionD BW(UGrid), r(UGrid), g5w(UGrid);
  for (int i = 0; i < nl; ++i) {
    w[i] = W[i];
    RealD nn = std::sqrt(norm2(w[i]));
    w[i] = w[i] * (1.0 / nn);
    mu[i] = ComplexD(1.0, 0.0) / theta[i];
    Cq[i] = std::hypot(1.0 - mu[i].real(), mu[i].imag());
    B.Op(w[i], BW);
    r = BW - mu[i] * w[i];
    resid[i] = std::sqrt(norm2(r));
    g5w = g5 * w[i];
    chi[i] = real(innerProduct(w[i], g5w));
  }
  std::vector<std::pair<double, int>> ord(nl);
  for (int i = 0; i < nl; ++i) {
    ord[i] = std::make_pair(std::hypot(mu[i].real(), mu[i].imag()), i);
  }
  std::sort(ord.begin(), ord.end());

  std::cout << GridLogMessage << "  k    Re(mu)          Im(mu)         |mu|          C=|1-mu|      chi           resid" << std::endl;
  int nr = (nreport < nl) ? nreport : nl;
  for (int k = 0; k < nr; ++k) {
    int i = ord[k].second;
    RealD absm = std::hypot(mu[i].real(), mu[i].imag());
    std::cout << GridLogMessage << "  " << k << "   " << mu[i].real() << "   " << mu[i].imag()
              << "   " << absm << "   " << Cq[i] << "   " << chi[i] << "   " << resid[i]
              << (std::abs(chi[i]) > 0.5 ? "  (chiral)" : "") << std::endl;
  }
  RealD mu_min = std::hypot(mu[ord[0].second].real(), mu[ord[0].second].imag());
  std::cout << GridLogMessage << "  " << mname << "  smallest |mu| = " << mu_min
            << "   (near 0? " << (mu_min < 0.1 ? "YES - near-null straggler" : "no - bounded away from 0 (GOOD)")
            << ")" << std::endl;
}

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  std::vector<Complex> boundary = {1, 1, 1, -1};   // anti-periodic time

  int Nstop = 4;
  int Nk = 8;
  int Nm = 16;
  double mprec = 0.8;
  double mass_c = 0.0;
  int flow_nstep = 873;
  RealD flow_eps = 0.02;
  int gf_maxit = 4000;
  RealD inner_tol = 1.0e-6;
  int inner_restart = 200;
  int inner_maxit = 4000;
  RealD ira_eresid = 1.0e-4;
  int ira_maxit = 30;
  int nreport = 8;
  std::string flowcache_dir = "";
  if (GridCmdOptionExists(argv, argv + argc, "--nstop")) Nstop = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--nstop"));
  if (GridCmdOptionExists(argv, argv + argc, "--nk")) Nk = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--nk"));
  if (GridCmdOptionExists(argv, argv + argc, "--nm")) Nm = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--nm"));
  if (GridCmdOptionExists(argv, argv + argc, "--mprec")) mprec = std::stod(GridCmdOptionPayload(argv, argv + argc, "--mprec"));
  if (GridCmdOptionExists(argv, argv + argc, "--mass_c")) mass_c = std::stod(GridCmdOptionPayload(argv, argv + argc, "--mass_c"));
  if (GridCmdOptionExists(argv, argv + argc, "--flow_nstep")) flow_nstep = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--flow_nstep"));
  if (GridCmdOptionExists(argv, argv + argc, "--flow_eps")) flow_eps = std::stod(GridCmdOptionPayload(argv, argv + argc, "--flow_eps"));
  if (GridCmdOptionExists(argv, argv + argc, "--gf_maxit")) gf_maxit = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--gf_maxit"));
  if (GridCmdOptionExists(argv, argv + argc, "--inner_tol")) inner_tol = std::stod(GridCmdOptionPayload(argv, argv + argc, "--inner_tol"));
  if (GridCmdOptionExists(argv, argv + argc, "--inner_restart")) inner_restart = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--inner_restart"));
  if (GridCmdOptionExists(argv, argv + argc, "--inner_maxit")) inner_maxit = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--inner_maxit"));
  if (GridCmdOptionExists(argv, argv + argc, "--ira_eresid")) ira_eresid = std::stod(GridCmdOptionPayload(argv, argv + argc, "--ira_eresid"));
  if (GridCmdOptionExists(argv, argv + argc, "--ira_maxit")) ira_maxit = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--ira_maxit"));
  if (GridCmdOptionExists(argv, argv + argc, "--nreport")) nreport = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--nreport"));
  if (GridCmdOptionExists(argv, argv + argc, "--flowcache")) flowcache_dir = GridCmdOptionPayload(argv, argv + argc, "--flowcache");

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

  // ---- frame: Urot = Omega U Omega^dag, CACHED (skips flow + Landau on rerun) ----
  LatticeGaugeFieldD Urot(UGrid);
  bool urot_loaded = false;
  std::string urotpath;
  if (!flowcache_dir.empty()) {
    urotpath = urot_file(flowcache_dir, cfg_base, flow_eps, flow_nstep, gf_maxit);
    if (file_exists(urotpath)) {
      FieldMetaData h;
      NerscIO::readConfiguration(Urot, h, urotpath);
      urot_loaded = true;
      std::cout << GridLogMessage << "[urotcache] LOADED " << urotpath << " (flow + Landau skipped)" << std::endl;
    }
  }
  if (!urot_loaded) {
    LatticeGaugeFieldD Uflowed(UGrid);
    bool flow_loaded = false;
    std::string flowpath = flowcache_dir.empty() ? "" : (flowcache_dir + "/" + flow_label(cfg_base, flow_eps, flow_nstep) + ".nersc");
    if (!flowpath.empty() && file_exists(flowpath)) {
      FieldMetaData h;
      NerscIO::readConfiguration(Uflowed, h, flowpath);
      flow_loaded = true;
      std::cout << GridLogMessage << "[flowcache] LOADED " << flowpath << std::endl;
    }
    if (!flow_loaded) {
      WilsonFlow<PeriodicGimplD> wf(flow_eps, flow_nstep);
      wf.smear(Uflowed, U);
      if (!flowpath.empty()) {
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
    SU<Nc>::GaugeTransform<PeriodicGimplD>(Urot, xform);
    if (!flowcache_dir.empty()) {
      NerscIO::writeConfiguration(Urot, urotpath, "DWF", "UROTCACHE", 1);
      std::cout << GridLogMessage << "[urotcache] SAVED " << urotpath << std::endl;
    }
  }

  // ---- massless 4D operator + M0 (bare Fw) + M1 (identity frame, framed op = DW) ----
  WilsonImplD::ImplParams Params(boundary);
  WilsonFermionD DW(Urot, *UGrid, *UrbGrid, mass_c, Params);
  Gamma g5(Gamma::Algebra::Gamma5);
  FreeWilsonInverse<WilsonImplD> Fw(UGrid, mprec, boundary);
  LatticeColourMatrixD idxf(UGrid);
  idxf = ComplexD(1.0, 0.0);   // identity ColourMatrix
  FreeLimitPreconditionerW1<WilsonImplD> M1(Fw, idxf, DW, UGrid);

  std::cout << GridLogMessage << "==== massless: mass_c=" << mass_c << " mprec=" << mprec
            << "  (Nstop=" << Nstop << " Nk=" << Nk << " Nm=" << Nm << " ira_eresid=" << ira_eresid
            << " ira_maxit=" << ira_maxit << ") ====" << std::endl;

  precmode_scan("M0 D_W", DW, Fw, UGrid, RNG4, g5, Nstop, Nk, Nm,
                inner_tol, inner_maxit, inner_restart, ira_eresid, ira_maxit, nreport);
  precmode_scan("M1 D_W", DW, M1, UGrid, RNG4, g5, Nstop, Nk, Nm,
                inner_tol, inner_maxit, inner_restart, ira_eresid, ira_maxit, nreport);

  std::cout << GridLogMessage << "precmodes done" << std::endl;
  Grid_finalize();
  return 0;
}
