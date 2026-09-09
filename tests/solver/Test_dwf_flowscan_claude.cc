// Frame-flow-TIME scan (free-limit DWF preconditioner, Direction 1b Stage 1).
// See Grid/scripts_nm/freeprec_future_directions_claude.md and memory project-r2-config-gen.
//
// STANDALONE driver (moved out of Test_dwf_freeprec_claude.cc so the flow-scan path is decoupled from
// the main benchmark file / the local upstream-merge developments there). Reuses the SAME free-limit
// kernel FreeMobius5D_claude.h (which now defaults to PlannedFFT + fp32 after the merge -- fp32 gives
// IDENTICAL FGMRES iters, so the D_W COUNTS match the pre-merge scan; only the wall time drops).
//
// Fix flow = Wilson, scan the frame-flow time. CGNE is FRAME-INDEPENDENT -> computed ONCE and reused for
// every tau. For each nstep (tau = flow_eps*nstep) we Wilson-flow, Landau-fix the frame Omega, and record
// the M0 (and M1) D_W-apply ratio vs s/t0 = tau/t0 (t0 measured per ensemble). One block per tau, flushed
// as it finishes so a walltime kill still leaves the completed tau points in the log.
//
// Run:  Test_dwf_flowscan_claude --grid 16.16.16.16 --mpi 1.1.1.1 --config <nersc> \
//         --flow_nsteps 58,73,87,... --t0 2.91 --solve_tol 1e-6 --ops cgne,m0,m1

#include <Grid/Grid.h>
#include <Grid/qcd/utils/FreeMobius5D_claude.h>
#include <Grid/qcd/utils/MaxTreeGaugeFix_claude.h>

#include <vector>
#include <complex>
#include <sstream>
#include <string>

using namespace Grid;

// FGMRES Krylov RestartLength. FILE-SCOPE (idiom of Test_dwf_freeprec_claude.cc:68): the --fgmres_restart
// parse in main() previously wrote to a variable that only existed inside run_flowscan -> did not compile
// (the knob was added after the last GPU rebuild). Default 256 = effectively no-restart.
static int g_fgmres_restart = 20;  // FGMRES restart length; default 20 (Nobu 2026-09-07, low-memory
                                   // production setting; was 256 = no-restart). --fgmres_restart overrides.

// Landau gauge-fix iteration cap. FILE-SCOPE + CLI (--gf_maxit) because 1000 UNDER-CONVERGES on
// longer-flow / harder frames: dmuAmu stalls at O(10^2) (target 1e-12) and the M0 win is then an
// underestimate (seen on the Q=0 config 747, 2026-09-06). Push well past 1000 and gate on the dmuAmu
// plateau (project memory gotcha). Default kept at 1000 for back-compatibility with earlier scans.
static int g_gf_maxit = 1000;

// Gauge-fix orthogonal direction: -1 = LANDAU (fix all Nd dirs), 0..Nd-1 = COULOMB (fix the dirs !=
// orthog, leaving `orthog` -- the time direction -- unfixed). Grid's FourierAcceleratedGaugeFixer
// handles both via this arg. Set with --gf_orthog (or --coulomb = time dir Nd-1). Testing whether a
// Coulomb frame beats the Landau frame for M0 (Nobu 2026-09-07).
static int g_gf_orthog = -1;

// Maximal-tree gauge frame (Taku's idea, MaxTreeGaugeFix_claude.h) instead of Landau/Coulomb, via
// --maxtree. Single-rank only (serial nested build). Mutually exclusive with the orthog gauges.
static bool g_maxtree = false;

// Frame-flow SMOOTHER (Direction 1b, flow-kernel variation). Returns the gauge action whose gradient
// drives the flow; beta = Nc keeps the Luscher flow-time normalization (c0+8c1=1 for the improved ones).
// "wilson" returns nullptr = WilsonFlow's built-in default (no swap). Swap idiom: Test_flowed_topocharge.
//   c1: Wilson 0, Symanzik -1/12, Iwasaki -0.331, DBW2 -1.4067, anti-Iwasaki +0.331 (c0<0 -> roughening
//   control; the gradient flow then MAXIMIZES the plaquette action and may DIVERGE -- guarded at use).
static Action<PeriodicGimplD::GaugeField>* make_flow_action(const std::string& name) {
  int nc = PeriodicGimplD::num_colours;
  if (name == "wilson") {
    return nullptr;
  }
  if (name == "symanzik") {
    return new SymanzikGaugeAction<PeriodicGimplD>(nc);
  }
  if (name == "iwasaki") {
    return new IwasakiGaugeAction<PeriodicGimplD>(nc);
  }
  if (name == "dbw2") {
    return new DBW2GaugeAction<PeriodicGimplD>(nc);
  }
  if (name == "antiiwasaki") {
    return new RBCGaugeAction<PeriodicGimplD>(nc, 0.331);
  }
  return nullptr;
}

static void run_flowscan(const std::string& tag, LatticeGaugeFieldD& U,
                         GridCartesian* UGrid, GridRedBlackCartesian* UrbGrid,
                         GridCartesian* FGrid, GridRedBlackCartesian* FrbGrid,
                         int Ls, double M5, double bb, double cc, double mm,
                         std::vector<Complex> boundary, double flow_eps,
                         const std::vector<int>& nsteps, const std::vector<std::string>& flow_actions,
                         double t0, GridParallelRNG& RNG5, bool run_cgne, bool run_m0, bool run_m1,
                         double solve_tol) {
  std::cout << "==== flow-time scan (Wilson flow)  [" << tag << "]  t0=" << t0
            << "  tol=" << solve_tol << " ====" << std::endl;

  WilsonImplD::ImplParams Params(boundary);
  MobiusFermionD D(U, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mm, M5, bb, cc, Params);
  FreeMobius5DInverse<WilsonImplD> Ffree(FGrid, Ls, M5, bb, cc, mm, boundary);

  LatticeFermionD bsrc(FGrid);
  gaussian(RNG5, bsrc);
  int solve_maxit = 4000;
  // int fgmres_restart = 256;  // no-restart: RestartLength >> expected iters (not literally maxit; OOM)
  int fgmres_restart = g_fgmres_restart;  // set by --fgmres_restart in main (file-scope; see top)
  NonHermitianLinearOperator<MobiusFermionD, LatticeFermionD> LinOp(D);

  Real plaq0 = WilsonLoops<PeriodicGimplD>::avgPlaquette(U);
  std::cout << "  plaq=" << plaq0 << std::endl;

  // CGNE baseline depends only on D (the ORIGINAL operator), NOT the frame -> compute once, reuse.
  long dW_cgne = 0;
  if (run_cgne) {
    MdagMLinearOperator<MobiusFermionD, LatticeFermionD> HermOp(D);
    LatticeFermionD bn(FGrid);
    D.Mdag(bsrc, bn);
    LatticeFermionD xcg(FGrid);
    xcg = Zero();
    ConjugateGradient<LatticeFermionD> CG(solve_tol, solve_maxit);
    CG(HermOp, bn, xcg);
    int cg_iters = CG.IterationsToComplete;
    dW_cgne = (long)2 * Ls * cg_iters;
    std::cout << "  CGNE (frame-independent): iters=" << cg_iters << "  D_W applies=" << dW_cgne
              << std::endl;
  }

  RealD gf_alpha = 0.1 / 16.0;  // Grid FA step is 16x too large (see run_headline); use 0.1/16
  // int gf_maxit = 1000;
  int gf_maxit = g_gf_maxit;  // set by --gf_maxit in main (file-scope; default 1000)
  for (size_t i = 0; i < nsteps.size(); ++i) {
    int nstep = nsteps[i];
    RealD tau = flow_eps * nstep;
    RealD s_over_t0 = (t0 > 0.0) ? (tau / t0) : 0.0;

    for (size_t j = 0; j < flow_actions.size(); ++j) {
    std::string fa = flow_actions[j];

    LatticeGaugeFieldD Uflowed(UGrid);
    WilsonFlow<PeriodicGimplD> wf(flow_eps, nstep);
    Action<PeriodicGimplD::GaugeField>* flowSG = make_flow_action(fa);  // nullptr = Wilson (default)
    if (flowSG) {
      wf.setGaugeAction(flowSG);  // swap the flow force (Symanzik/Iwasaki/DBW2/anti-Iwasaki)
    }
    wf.smear(Uflowed, U);
    Real plaq_flowed = WilsonLoops<PeriodicGimplD>::avgPlaquette(Uflowed);
    // GUARD: a roughening flow (anti-Iwasaki, c0<0) MAXIMIZES the action -> the flowed config diverges.
    // A sane flowed plaquette is in (0,1]; NaN or out-of-range means the flow blew up -> skip the solves.
    if (!(plaq_flowed >= 0.0 && plaq_flowed <= 1.0001)) {
      std::cout << "  ---- s/t0=" << s_over_t0 << "  flow=" << fa
                << "  DIVERGED (flowed plaq=" << plaq_flowed << ") -- skipping solves ----" << std::endl;
      if (flowSG) {
        delete flowSG;
      }
      std::cout.flush();
      continue;
    }
    LatticeColourMatrixD xform(UGrid);
    if (g_maxtree) {
      MaxTreeGaugeFix<PeriodicGimplD>::GaugeTransform(Uflowed, xform);  // Taku's maximal-tree frame
    } else {
      FourierAcceleratedGaugeFixer<PeriodicGimplD>::SteepestDescentGaugeFix(
          Uflowed, xform, gf_alpha, gf_maxit, 1.0e-12, 1.0e-12, /*Fourier=*/true, /*orthog=*/g_gf_orthog,
          /*err_on_no_converge=*/false);
    }
    Real landau = 1.0 - WilsonLoops<PeriodicGimplD>::linkTrace(Uflowed);
    Real Qflow = WilsonLoops<PeriodicGimplD>::TopologicalCharge5Li(Uflowed);

    std::cout << "  ---- s/t0=" << s_over_t0 << "  flow=" << fa << "  tau=" << tau << "  nstep=" << nstep
              << "  Landau=" << landau << "  Q_5Li(flowed)=" << Qflow << " ----" << std::endl;

    FreeLimitPreconditioner<WilsonImplD> M0(Ffree, xform, FGrid);
    // Frame-match yardstick (marlborough's shared metric across the flow-eps axis, which perturbs Omega,
    // and the Ls' axis, which perturbs F): the operator residual ||M0 D v - v||/||v|| = how far
    // M0 = Omega^dag F Omega is from D_DW^{-1} on the REAL config. The Landau functional only sees the
    // Omega axis (Ls'-invariant); this residual sees BOTH. Cheap: one D apply + one M0 apply on bsrc.
    LatticeFermionD Dv(FGrid);
    LatticeFermionD M0Dv(FGrid);
    D.M(bsrc, Dv);
    M0(Dv, M0Dv);
    double m0_resid = std::sqrt(norm2(M0Dv - bsrc) / norm2(bsrc));
    std::cout << "    M0-resid ||M0 D v - v||/||v|| = " << m0_resid << std::endl;
    if (run_m0) {
      M0.n_apply = 0;
      // Cap FGMRES(M0) iterations at 2*iter_CGNE = dW_cgne/Ls (M0 is D_W-free, CGNE = 2*Ls*iters, so M0
      // can only WIN below this bound). Bounds the runaway on bad/pathological frames; if it hits the cap
      // unconverged, ratio(CGNE/M0) <= 1 = "M0 did not beat CGNE". Falls back to solve_maxit if no CGNE.
      int fgmres_maxit = (dW_cgne > 0) ? (int)(dW_cgne / Ls) : solve_maxit;
      FlexibleGeneralisedMinimalResidual<LatticeFermionD> FGMRES(solve_tol, fgmres_maxit, M0,
                                                                fgmres_restart, /*err_on_no_conv=*/false);
      LatticeFermionD xg(FGrid);
      xg = Zero();
      FGMRES(LinOp, bsrc, xg);
      int fg_iters = FGMRES.IterationCount;
      long dW_fgmres = (long)Ls * fg_iters;
      double ratio = (dW_cgne > 0 && dW_fgmres > 0) ? (double)dW_cgne / (double)dW_fgmres : 0.0;
      std::cout << "    M0: iters=" << fg_iters << "  D_W=" << dW_fgmres << "  ratio(CGNE/M0)=" << ratio
                << "x" << std::endl;
    }
    if (run_m1) {
      LatticeGaugeFieldD UL(UGrid);
      UL = U;
      SU<Nc>::GaugeTransform<PeriodicGimplD>(UL, xform);  // U^L = Omega U Omega^dag
      MobiusFermionD Dframed(UL, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mm, M5, bb, cc, Params);
      FreeLimitPreconditioner1<WilsonImplD> M1(Ffree, xform, Dframed, FGrid);
      FlexibleGeneralisedMinimalResidual<LatticeFermionD> FGMRES1(solve_tol, solve_maxit, M1,
                                                                 fgmres_restart, /*err_on_no_conv=*/false);
      LatticeFermionD xg1(FGrid);
      xg1 = Zero();
      FGMRES1(LinOp, bsrc, xg1);
      int fg1_iters = FGMRES1.IterationCount;
      long dW_fgmres1 = (long)Ls * fg1_iters + (long)Ls * M1.n_dw;  // honest total (M1 not D_W-free)
      double ratio1 = (dW_cgne > 0 && dW_fgmres1 > 0) ? (double)dW_cgne / (double)dW_fgmres1 : 0.0;
      std::cout << "    M1: iters=" << fg1_iters << "  internal D_DW[U^L]=" << M1.n_dw
                << "  D_W(total)=" << dW_fgmres1 << "  ratio(CGNE/M1)=" << ratio1 << "x" << std::endl;
    }
    if (flowSG) {
      delete flowSG;
    }
    std::cout.flush();
    }  // flow_actions (kernel) loop
  }
}

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  // Physics point (matches the freeprec headline so counts are comparable): Mobius Shamir b=1.5, c=0.5,
  // m=0.1, M5=1.8, Ls=8, anti-periodic time.
  const double M5 = 1.8;
  const int Ls = 8;
  const double bb = 1.5;
  const double cc = 0.5;
  const double mm = 0.1;
  std::vector<Complex> boundary = {1, 1, 1, -1};
  double flow_eps = 0.02;  // RK3 flow step size; overridable via --flow_eps for the step-size study

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplexD::Nsimd());
  Coordinate mpi = GridDefaultMpi();
  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(latt, simd, mpi);
  GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian* FGrid = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
  GridRedBlackCartesian* FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);

  GridParallelRNG RNG5(FGrid);
  RNG5.SeedFixedIntegers(std::vector<int>({1, 2, 3, 4}));

  if (!GridCmdOptionExists(argv, argv + argc, "--config")) {
    std::cout << "ERROR: need --config <NERSC gauge file>" << std::endl;
    Grid_finalize();
    return 1;
  }
  std::string cfgfile = GridCmdOptionPayload(argv, argv + argc, "--config");

  // ADDITIVE op selection (default all three). cgne is the ratio baseline (frame-independent).
  bool run_cgne = true;
  bool run_m0 = true;
  bool run_m1 = true;
  if (GridCmdOptionExists(argv, argv + argc, "--ops")) {
    std::string ops = GridCmdOptionPayload(argv, argv + argc, "--ops");
    run_cgne = (ops.find("cgne") != std::string::npos);
    run_m0 = (ops.find("m0") != std::string::npos);
    run_m1 = (ops.find("m1") != std::string::npos);
  }

  // Flow-nstep list (tau = flow_eps*nstep), t0 for s/t0 printout, solver tolerance.
  std::vector<int> flow_nsteps;
  if (GridCmdOptionExists(argv, argv + argc, "--flow_nsteps")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--flow_nsteps");
    std::stringstream ss(a);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      if (!tok.empty()) {
        flow_nsteps.push_back(std::stoi(tok));
      }
    }
  }
  if (flow_nsteps.empty()) {
    std::cout << "ERROR: need --flow_nsteps <comma list of nsteps>" << std::endl;
    Grid_finalize();
    return 1;
  }
  double scan_t0 = 1.0;
  if (GridCmdOptionExists(argv, argv + argc, "--t0")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--t0");
    GridCmdOptionFloat(a, scan_t0);
  }
  double scan_tol = 1.0e-8;
  if (GridCmdOptionExists(argv, argv + argc, "--solve_tol")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--solve_tol");
    GridCmdOptionFloat(a, scan_tol);
  }
  // FGMRES(M0/M1) Krylov RestartLength. Default 256 = effectively no-restart (huge Krylov -> expensive
  // orthogonalisation per iter). Set small (e.g. 40) to restart -> cheap iters, more of them (bounded by
  // the 2*iter_CGNE cap). Changes the restarted-GMRES count, so not directly comparable to no-restart runs.
  if (GridCmdOptionExists(argv, argv + argc, "--fgmres_restart")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--fgmres_restart");
    // GridCmdOptionInt(a, fgmres_restart);
    GridCmdOptionInt(a, g_fgmres_restart);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--gf_maxit")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--gf_maxit");
    GridCmdOptionInt(a, g_gf_maxit);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--gf_orthog")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--gf_orthog");
    GridCmdOptionInt(a, g_gf_orthog);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--coulomb")) {
    g_gf_orthog = Nd - 1;  // Coulomb: leave the time direction unfixed
  }
  if (GridCmdOptionExists(argv, argv + argc, "--maxtree")) {
    g_maxtree = true;  // Taku's maximal-tree frame (single-rank; overrides the orthog gauges)
  }
  // Step-size study (Direction 1b Stage-3): --flow_eps overrides the RK3 flow step. At a fixed s/t0 the
  // caller keeps tau = flow_eps*nstep constant (nstep scales with 1/eps), so only the integration error
  // of the frame changes -- coarse vs fine eps.
  if (GridCmdOptionExists(argv, argv + argc, "--flow_eps")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--flow_eps");
    GridCmdOptionFloat(a, flow_eps);
  }

  // Frame-flow KERNELS to compare (comma list; default just wilson). Each is swapped in via setGaugeAction.
  // Round 1 (Nobu): wilson,iwasaki,antiiwasaki -- baseline / smoother / roughening control.
  std::vector<std::string> flow_actions;
  if (GridCmdOptionExists(argv, argv + argc, "--frame_flows")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--frame_flows");
    std::stringstream fss(a);
    std::string ftok;
    while (std::getline(fss, ftok, ',')) {
      if (!ftok.empty()) {
        flow_actions.push_back(ftok);
      }
    }
  }
  if (flow_actions.empty()) {
    flow_actions.push_back("wilson");
  }

  LatticeGaugeFieldD Ureal(UGrid);
  FieldMetaData rheader;
  NerscIO::readConfiguration(Ureal, rheader, cfgfile);
  Real plaqr = WilsonLoops<PeriodicGimplD>::avgPlaquette(Ureal);
  std::cout << "loaded " << cfgfile << "  plaq(computed)=" << plaqr << "  header=" << rheader.plaquette
            << std::endl;

  run_flowscan("NERSC SU(3) -- FLOW SCAN", Ureal, UGrid, UrbGrid, FGrid, FrbGrid,
               Ls, M5, bb, cc, mm, boundary, flow_eps, flow_nsteps, flow_actions, scan_t0, RNG5,
               run_cgne, run_m0, run_m1, scan_tol);

  Grid_finalize();
  return 0;
}
