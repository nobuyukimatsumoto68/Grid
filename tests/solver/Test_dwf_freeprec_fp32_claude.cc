// 24^4 fp32 production comparison: RB-CGNE vs deflated FGMRES (GMRES-DR) on a real DWF config.
// Plan: tests/solver/dwf_freeprec_fp32_impl_plan_claude.md.
//
// WHY fp32: GMRES-DR stores m_restart + k ~ 48 full 5D vectors. On 24^4 x Ls=8 one fp64 5D fermion
// is ~509 MB, so the fp64 basis alone is ~24 GB (whole run ~63 GB, over one GPU). fp32 halves it to
// ~12.7 GB and the run fits (~32 GB). Single-precision CG/GMRES floor is ~3e-7, so tol = 1e-6.
//
// DESIGN: production uses Peter Boyle's Omega pre-rotation -- gauge-transform U -> U^Omega = Omega U
// Omega^dag ONCE, rotate the source by Omega5, and precondition with the BARE free inverse F (identity
// frame). Unitarily equivalent -> identical iteration counts, no per-apply Omega. With the frame folded
// into the gauge, the preconditioner IS F, and FreeMobius5DInverse<WilsonImplF> is already a
// LinearFunction<LatticeFermionF>. So we use F directly and never touch the shared FreeLimitPreconditioner
// (whose Omega5 is hard-coded LatticeColourMatrixD and would not compile against a single field).
//
// The FRAME (WilsonFlow, FA-Landau fix, topological charge, the Omega gauge transform, source rotation)
// is built in DOUBLE -- cheap, mass-independent, precision-insensitive. Only the per-mass solve block
// (D, F, source, solvers) is single precision.
//
// Metric (D_W applies): RB-CGNE = 2 Ls iters; GMRES-DR = Ls iters (one M0 + one D per Arnoldi step).
// Solver refs: GMRES-DR = R.B. Morgan, SIAM J. Sci. Comput. 24 (2002) 20
// (RecyclingGeneralisedMinimalResidual_claude.h); RB Schur CG = Grid SchurRedBlackDiagMooeeSolve;
// FA Landau fix = C.T.H. Davies et al., PRD 37 (1988) 1581.

#include <Grid/Grid.h>
#include <Grid/qcd/utils/FreeMobius5D_claude.h>
#include <Grid/algorithms/iterative/RecyclingGeneralisedMinimalResidual_claude.h>

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>

using namespace Grid;

// Adapter: present the DOUBLE free inverse F (FreeMobius5DInverse<WilsonImplD> -- the tested production
// instantiation) as a single-precision LinearFunction, so the SINGLE outer GMRES-DR (whose Krylov basis
// is the memory hog and must be single to fit the GPU) can use it as its right-preconditioner. Bounces
// single -> double, applies F, double -> single. F ALREADY runs its momentum solve in single internally
// (FREEMOBIUS5D_USE_FP32), so the extra cost is just two precisionChange passes per apply; the outer
// single basis is the real memory win. We do NOT instantiate FreeMobius5DInverse<WilsonImplF>: its double
// barrel path assigns ComplexD literals into FermionField site elements, which does not compile for a
// single field (std::complex<float> = std::complex<double>), and editing that shared header is out of scope.
class FreeInvF32Adapter : public LinearFunction<LatticeFermionF> {
public:
  FreeMobius5DInverse<WilsonImplD>& Fd;
  LatticeFermionD in_d;
  LatticeFermionD out_d;
  FreeInvF32Adapter(FreeMobius5DInverse<WilsonImplD>& Fd_, GridCartesian* FGrid_d)
    : Fd(Fd_), in_d(FGrid_d), out_d(FGrid_d) {}
  virtual void operator()(const LatticeFermionF& in, LatticeFermionF& out) {
    precisionChange(in_d, in);
    Fd(in_d, out_d);
    precisionChange(out, out_d);
  }
};

// ---- flow cache (double NERSC, LIME-free) -- same helpers as Test_dwf_freeprec_claude.cc ----
// Unique flow label: config basename + flow type (wilson) + eps + nstep. Same setup -> same file, so a
// later run of the SAME setup loads the flowed config and skips the (expensive) gradient flow.
static std::string flow_label(const std::string& cfg_base, double eps, int nstep) {
  std::ostringstream os;
  os << "flow_" << cfg_base << "_wilson_eps" << eps << "_n" << nstep;
  return os.str();
}
static std::string flow_cache_file(const std::string& dir, const std::string& cfg_base, double eps, int nstep) {
  return dir + "/" + flow_label(cfg_base, eps, nstep) + ".nersc";
}
static bool flow_cache_exists(const std::string& path) {
  std::ifstream f(path.c_str());
  return f.good();
}
static bool flow_cache_load(LatticeGaugeFieldD& Uflowed, const std::string& path) {
  if (!flow_cache_exists(path)) {
    return false;
  }
  FieldMetaData h;
  NerscIO::readConfiguration(Uflowed, h, path);
  return true;
}
static void flow_cache_save(LatticeGaugeFieldD& Uflowed, const std::string& path) {
  NerscIO::writeConfiguration(Uflowed, path, "DWF", "FLOWCACHE", 1);
}

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  const int Ls = 8;
  const double M5 = 1.8;
  const double bb = 1.5;   // Shamir point
  const double cc = 0.5;

  // ---- CLI ----
  if (!GridCmdOptionExists(argv, argv + argc, "--config")) {
    std::cout << GridLogError << "need --config <nersc-file>" << std::endl;
    Grid_finalize();
    return 1;
  }
  std::string cfgfile = GridCmdOptionPayload(argv, argv + argc, "--config");
  std::string cfg_base = cfgfile.substr(cfgfile.find_last_of('/') + 1);

  std::vector<double> masses;
  if (GridCmdOptionExists(argv, argv + argc, "--mass-list")) {
    std::stringstream mss(GridCmdOptionPayload(argv, argv + argc, "--mass-list"));
    std::string mtok;
    while (std::getline(mss, mtok, ',')) {
      if (!mtok.empty()) {
        masses.push_back(std::stod(mtok));
      }
    }
  }
  if (masses.empty()) {
    masses.push_back(0.1);
    masses.push_back(0.01);
    masses.push_back(0.001);
  }

  double flow_eps = 0.02;
  int flow_nstep = 873;   // frame flow s/t0=6 -- PRODUCTION default (matches Test_dwf_freeprec_claude.cc)
  if (GridCmdOptionExists(argv, argv + argc, "--flow_eps")) {
    flow_eps = std::stod(GridCmdOptionPayload(argv, argv + argc, "--flow_eps"));
  }
  if (GridCmdOptionExists(argv, argv + argc, "--flow_nstep")) {
    flow_nstep = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--flow_nstep"));
  }
  int gf_maxit = 4000;
  if (GridCmdOptionExists(argv, argv + argc, "--gf_maxit")) {
    gf_maxit = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--gf_maxit"));
  }

  int m_restart = 16;   // GMRES-DR restart window (production)
  int k_deflate = 32;   // GMRES-DR adaptive deflation dim (production)
  if (GridCmdOptionExists(argv, argv + argc, "--restart")) {
    m_restart = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--restart"));
  }
  if (GridCmdOptionExists(argv, argv + argc, "--deflate-k")) {
    k_deflate = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--deflate-k"));
  }

  double solve_tol = 1.0e-6;   // fp32 CG/GMRES floor ~3e-7; 1e-8 is unreachable in single
  if (GridCmdOptionExists(argv, argv + argc, "--tol")) {
    solve_tol = std::stod(GridCmdOptionPayload(argv, argv + argc, "--tol"));
  }
  int solve_maxit = 20000;

  std::string flowcache_dir = "";
  if (GridCmdOptionExists(argv, argv + argc, "--flowcache")) {
    flowcache_dir = GridCmdOptionPayload(argv, argv + argc, "--flowcache");   // dir must exist
  }

  // ---- grids: double (frame + config + gauge transform) and single (solve) ----
  Coordinate latt = GridDefaultLatt();
  Coordinate mpi = GridDefaultMpi();
  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(latt, GridDefaultSimd(Nd, vComplexD::Nsimd()), mpi);
  GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian* FGrid = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
  GridRedBlackCartesian* FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);

  GridCartesian* UGridF = SpaceTimeGrid::makeFourDimGrid(latt, GridDefaultSimd(Nd, vComplexF::Nsimd()), mpi);
  GridRedBlackCartesian* UrbGridF = SpaceTimeGrid::makeFourDimRedBlackGrid(UGridF);
  GridCartesian* FGridF = SpaceTimeGrid::makeFiveDimGrid(Ls, UGridF);
  GridRedBlackCartesian* FrbGridF = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGridF);

  std::vector<Complex> boundary = {1, 1, 1, -1};   // anti-periodic time

  GridParallelRNG RNG5(FGrid);
  RNG5.SeedFixedIntegers(std::vector<int>({1, 2, 3, 4}));

  // ---- load config (double) ----
  LatticeGaugeFieldD U(UGrid);
  FieldMetaData rheader;
  NerscIO::readConfiguration(U, rheader, cfgfile);
  Real plaq0 = WilsonLoops<PeriodicGimplD>::avgPlaquette(U);
  std::cout << GridLogMessage << "loaded " << cfgfile << "  plaq=" << plaq0
            << "  header=" << rheader.plaquette << std::endl;

  // ---- frame (DOUBLE): flow (cached) + FA-Landau fix ----
  LatticeGaugeFieldD Uflowed(UGrid);
  std::string flowpath;
  bool flow_loaded = false;
  if (!flowcache_dir.empty()) {
    flowpath = flow_cache_file(flowcache_dir, cfg_base, flow_eps, flow_nstep);
    flow_loaded = flow_cache_load(Uflowed, flowpath);
    if (flow_loaded) {
      std::cout << GridLogMessage << "[flowcache] LOADED " << flowpath << " (flow skipped)" << std::endl;
    }
  }
  if (!flow_loaded) {
    WilsonFlow<PeriodicGimplD> wf(flow_eps, flow_nstep);
    wf.smear(Uflowed, U);
    if (!flowcache_dir.empty()) {
      flow_cache_save(Uflowed, flowpath);
      std::cout << GridLogMessage << "[flowcache] SAVED " << flowpath << std::endl;
    }
  }
  Real plaq_flowed = WilsonLoops<PeriodicGimplD>::avgPlaquette(Uflowed);

  LatticeColourMatrixD xform(UGrid);
  // FA Landau: Grid's step weight is 16x the dwf4 reference; use alpha = 0.1/16 so it does NOT overshoot.
  RealD gf_alpha = 0.1 / 16.0;
  FourierAcceleratedGaugeFixer<PeriodicGimplD>::SteepestDescentGaugeFix(
      Uflowed, xform, gf_alpha, gf_maxit, 1.0e-12, 1.0e-12, /*Fourier=*/true, /*orthog=*/-1,
      /*err_on_no_converge=*/false);
  Real landau = 1.0 - WilsonLoops<PeriodicGimplD>::linkTrace(Uflowed);
  Real Qflow = WilsonLoops<PeriodicGimplD>::TopologicalCharge5Li(Uflowed);
  std::cout << GridLogMessage << "flowed plaq=" << plaq_flowed << "  Landau functional=" << landau
            << "  Q_5Li(tau=" << (flow_eps * flow_nstep) << ")=" << Qflow << std::endl;

  // ---- Omega pre-rotation: fold the frame into the gauge (DOUBLE), then precisionChange to SINGLE ----
  LatticeGaugeFieldD UrotD(UGrid);
  UrotD = U;
  SU<Nc>::GaugeTransform<PeriodicGimplD>(UrotD, xform);   // U^Omega = Omega U Omega^dag
  LatticeGaugeFieldF UrotF(UGridF);
  precisionChange(UrotF, UrotD);

  // Omega5 (double) broadcast onto the 5D grid, used to rotate the source into the frame.
  LatticeColourMatrixD Om5(FGrid);
  for (int s = 0; s < Ls; ++s) {
    InsertSlice(xform, Om5, s, 0);
  }

  WilsonImplF::ImplParams ParamsF(boundary);

  // ---- mass loop (frame reused; only the operator/source/solvers depend on the mass) ----
  for (size_t im = 0; im < masses.size(); ++im) {
    double mm = masses[im];
    std::cout << GridLogMessage << "======== mass m = " << mm << "  (fp32, frame reused) ========" << std::endl;

    MobiusFermionF D(UrotF, *FGridF, *FrbGridF, *UGridF, *UrbGridF, mm, M5, bb, cc, ParamsF);
    // bare free inverse = M0 in the identity frame (prerot). Built in DOUBLE on FGrid (tested
    // instantiation); wrapped by FreeInvF32Adapter so the single outer GMRES-DR can call it.
    FreeMobius5DInverse<WilsonImplD> Ffree(FGrid, Ls, M5, bb, cc, mm, boundary);
    FreeInvF32Adapter M0(Ffree, FGrid);

    // source: gaussian in double, rotate into the frame by Omega5 (double), precisionChange to single.
    LatticeFermionD bsrcD(FGrid);
    gaussian(RNG5, bsrcD);
    bsrcD = Om5 * bsrcD;
    LatticeFermionF bsrc(FGridF);
    precisionChange(bsrc, bsrcD);

    long dW_rbcgne = 0;
    double wall_rbcgne = 0.0;

    // ---- RB-CGNE (honest baseline): SchurRedBlack even-odd CG on Mpc^dag Mpc ----
    {
      ConjugateGradient<LatticeFermionF> CGrb(solve_tol, solve_maxit, /*err_on_no_conv=*/false);
      SchurRedBlackDiagMooeeSolve<LatticeFermionF> SchurSolver(CGrb);
      LatticeFermionF xrb(FGridF);
      xrb = Zero();
      double tw = -usecond();
      SchurSolver(D, bsrc, xrb);
      tw += usecond();
      int rb_iters = CGrb.IterationsToComplete;
      wall_rbcgne = tw;
      dW_rbcgne = (long)2 * Ls * rb_iters;
      std::cout << GridLogMessage << "  RB-CGNE:    iters=" << rb_iters << "  D_W applies=" << dW_rbcgne
                << "  WALL=" << tw / 1.0e6 << " s   [honest baseline]" << std::endl;
    }

    // ---- deflated FGMRES = GMRES-DR right-preconditioned by the bare free inverse F ----
    {
      NonHermitianLinearOperator<MobiusFermionF, LatticeFermionF> LinOp(D);
      RecyclingGeneralisedMinimalResidual<LatticeFermionF> RGM(solve_tol, solve_maxit, M0,
                                                              m_restart, k_deflate,
                                                              /*recycle_across_solves=*/false,
                                                              /*err_on_no_conv=*/false);
      LatticeFermionF xg(FGridF);
      xg = Zero();
      double tw = -usecond();
      RGM(LinOp, bsrc, xg);
      tw += usecond();
      int rg_iters = RGM.IterationCount;
      long dW_rgm = (long)Ls * rg_iters;
      std::cout << GridLogMessage << "  GMRES-DR(m=" << m_restart << ",k=" << k_deflate << "): iters="
                << rg_iters << "  D_W applies=" << dW_rgm << "  WALL=" << tw / 1.0e6 << " s" << std::endl;
      if (dW_rbcgne > 0) {
        double sp = (dW_rgm > 0) ? (double)dW_rbcgne / (double)dW_rgm : 0.0;
        std::cout << GridLogMessage << "  D_W-apply speedup (RB-CGNE / GMRES-DR) = " << sp << "x" << std::endl;
      }
      if (wall_rbcgne > 0.0 && tw > 0.0) {
        std::cout << GridLogMessage << "  WALL speedup (RB-CGNE / GMRES-DR) = " << wall_rbcgne / tw
                  << "x  [>1 = free-prec wins wall]" << std::endl;
      }
    }
  }

  Grid_finalize();
  return 0;
}
