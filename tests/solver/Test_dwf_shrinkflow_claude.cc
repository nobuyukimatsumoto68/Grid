// Shrink-flow frame driver (free-limit DWF preconditioner; gauge-fix/topology thread).
// See Grid/scripts_nm/freeprec_gaugefix_topology_claude.md Sec. 4 and shrinkflow_impl_plan_claude.md.
//
// Two-stage frame flow. Stage 1 = Wilson flow to tau1 ~ t0 (kills the perturbative dof; numerical
// evidence says the Omega quality saturates here). Stage 2 = UNDER-improved flow (RBC plaquette+
// rectangle action with c1 > 0, epsilon = 1 + 12 c1 > 1) whose O(a^2) artifact SHRINKS the instanton
// radius, S(rho) = S0 [1 - (epsilon/5)(a/rho)^2 + O(a^4)], so the lump(s) on the FRAME COPY shrink
// and fall through the lattice (|Q| -> 0, trivial sector). For a pure-gauge background the flow force
// vanishes, so Stage 2 can run as long as needed. Stage 3 = short Wilson re-smooth of the
// fall-through ripple. Then Landau-fix -> Omega and measure the M0 (and optionally M1) D_W-apply win
// on the ORIGINAL config vs the frame-independent CGNE baseline. A baseline frame (Landau fix
// straight after Stage 1 = the current default s/t0=1.0 frame) runs first on the same config+source,
// so one run is a self-contained A/B.
// Stability window: c0 = 1 - 8 c1 > 0 -> c1 < 1/8; default c1 = +1/12 (epsilon = 2). c1 = +0.331
// (anti-Iwasaki, epsilon ~ 5) has c0 < 0 and diverges (seen in the flowscan kernel study) -- guarded.
//
// Sources: instanton-size control via the O(a^2) term of the cooling/smearing action --
// Garcia Perez, Gonzalez-Arroyo, Snippe, van Baal hep-lat/9309009; de Forcrand, Garcia Perez,
// Stamatescu hep-lat/9701012 (improved cooling); Moran, Leinweber arXiv:0801.1165 (over-improved
// stout; the epsilon parametrisation c1 = -(1-epsilon)/12); Bonati, D'Elia arXiv:1401.2441
// (cooling / gradient-flow equivalence).
//
// Run: Test_dwf_shrinkflow_claude --grid 16.16.16.16 --mpi 1.1.1.1 --config <nersc> \
//        --t0 2.91 --nstep1 146 --c1 0,0.0417,0.0833,0.1167 --nstep2 2000 --qchunk 10 --qstop 0.5 \
//        --nstep3 20 --ops cgne,m0 --solve_tol 1e-6
// --c1 is a COMMA LIST (shrink-strength scan, epsilon = 1+12c1 per value); stage 1 / CGNE / the
// baseline frame are shared across the list.

#include <Grid/Grid.h>
#include <Grid/qcd/utils/FreeMobius5D_claude.h>
#include <Grid/qcd/utils/QSqueezeFlowAction_claude.h>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

using namespace Grid;

// FGMRES Krylov RestartLength (default 256 = effectively no-restart). File-scope so main's CLI parse
// and the solve helper share it (idiom of Test_dwf_freeprec_claude.cc).
static int g_fgmres_restart = 256;

// Finite-difference validation of QSqueezeGaugeAction::deriv (idiom of tests/forces/
// Test_rect_force.cc): hot config, random Lie-algebra momentum, compare (S(U') - S(U)) against
// -2 sum_mu tr(mom_mu UdSdU_mu) dt. Run with --fdcheck [--qn 2,4,6] on a SMALL --grid (e.g. 8.8.8.8);
// MUST print ratio ~ 1 for every n before any flow run is trusted.
static int run_fdcheck(GridCartesian* UGrid, const std::vector<int>& qns, RealD dt) {
  GridParallelRNG pRNG(UGrid);
  pRNG.SeedFixedIntegers(std::vector<int>({45, 12, 81, 9}));
  LatticeGaugeFieldD U(UGrid);
  SU<Nc>::HotConfiguration(pRNG, U);
  int fail = 0;
  for (size_t i = 0; i < qns.size(); ++i) {
    QSqueezeGaugeAction<PeriodicGimplD> QA(qns[i]);
    RealD S0 = QA.S(U);
    LatticeGaugeFieldD UdSdU(UGrid);
    QA.deriv(U, UdSdU);
    LatticeGaugeFieldD mom(UGrid);
    LatticeColourMatrixD mommu(UGrid);
    for (int mu = 0; mu < Nd; mu++) {
      SU<Nc>::GaussianFundamentalLieAlgebraMatrix(pRNG, mommu);
      PokeIndex<LorentzIndex>(mom, mommu, mu);
    }
    // CENTRAL difference U(+-dt) = (1 +- mom dt + (1/2) mom^2 dt^2) U -- the O(dt^2) term cancels in
    // S(+dt) - S(-dt), so the linear (force) term is tested cleanly.
    LatticeGaugeFieldD Uplus(UGrid);
    Uplus = U;
    LatticeGaugeFieldD Uminus(UGrid);
    Uminus = U;
    {
      autoView(Uplus_v, Uplus, CpuWrite);
      autoView(Uminus_v, Uminus, CpuWrite);
      autoView(U_v, U, CpuRead);
      autoView(mom_v, mom, CpuRead);
      thread_foreach(ss, mom_v, {
        for (int mu = 0; mu < Nd; mu++) {
          auto md = mom_v[ss](mu) * dt;
          auto md2 = 0.5 * md * md;
          Uplus_v[ss](mu) = U_v[ss](mu) + (md + md2) * U_v[ss](mu);
          Uminus_v[ss](mu) = U_v[ss](mu) + (md2 - md) * U_v[ss](mu);
        }
      });
    }
    RealD Splus = QA.S(Uplus);
    RealD Sminus = QA.S(Uminus);
    RealD S1 = Splus;
    LatticeComplexD dS(UGrid);
    dS = Zero();
    for (int mu = 0; mu < Nd; mu++) {
      LatticeColourMatrixD UdSdUmu = PeekIndex<LorentzIndex>(UdSdU, mu);
      mommu = PeekIndex<LorentzIndex>(mom, mu);
      dS = dS - trace(mommu * UdSdUmu) * dt * 2.0;
    }
    ComplexD dSpred = TensorRemove(sum(dS));
    double dSc = 0.5 * (Splus - Sminus);  // central difference; compare to dSpred (one-sided pred)
    double ratio = (dSpred.real() != 0.0) ? dSc / dSpred.real() : 0.0;
    std::cout << std::setprecision(12) << "fdcheck n=" << qns[i] << ": S=" << S0 << "  Sprime=" << S1
              << "  dS(central)=" << dSc << "  dS(pred)=" << dSpred.real()
              << "  ratio(actual/pred)=" << ratio << std::endl;
    if (std::fabs(ratio - 1.0) > 0.05) {
      fail = 1;
    }
  }
  std::cout << "fdcheck " << (fail ? "FAILED" : "PASSED") << std::endl;
  return fail;
}

// Landau-fix Uflowed (in place) -> frame Omega = xform, then measure the M0 (and optionally M1) win
// on the ORIGINAL operator D. Mirrors the per-tau block of Test_dwf_flowscan_claude.cc.
static void frame_and_solve(const std::string& label, LatticeGaugeFieldD& Uflowed,
                            LatticeGaugeFieldD& Uorig, MobiusFermionD& D,
                            FreeMobius5DInverse<WilsonImplD>& Ffree,
                            NonHermitianLinearOperator<MobiusFermionD, LatticeFermionD>& LinOp,
                            LatticeFermionD& bsrc,
                            GridCartesian* UGrid, GridRedBlackCartesian* UrbGrid,
                            GridCartesian* FGrid, GridRedBlackCartesian* FrbGrid,
                            int Ls, double M5, double bb, double cc, double mm,
                            WilsonImplD::ImplParams& Params,
                            long dW_cgne, double solve_tol, int solve_maxit, int gf_maxit,
                            bool run_m0, bool run_m1) {
  RealD gf_alpha = 0.1 / 16.0;  // Grid FA step is 16x too large (GaugeFix.h psqMax) -> 0.1/16
  LatticeColourMatrixD xform(UGrid);
  FourierAcceleratedGaugeFixer<PeriodicGimplD>::SteepestDescentGaugeFix(
      Uflowed, xform, gf_alpha, gf_maxit, 1.0e-12, 1.0e-12, /*Fourier=*/true, /*orthog=*/-1,
      /*err_on_no_converge=*/false);
  Real landau = 1.0 - WilsonLoops<PeriodicGimplD>::linkTrace(Uflowed);
  Real Qfix = WilsonLoops<PeriodicGimplD>::TopologicalCharge5Li(Uflowed);
  std::cout << "==== frame [" << label << "]  Landau=" << landau << "  Q_5Li(frame cfg)=" << Qfix
            << " ====" << std::endl;

  FreeLimitPreconditioner<WilsonImplD> M0(Ffree, xform, FGrid);
  // Frame-match yardstick: operator residual ||M0 D v - v||/||v|| on the REAL config.
  LatticeFermionD Dv(FGrid);
  LatticeFermionD M0Dv(FGrid);
  D.M(bsrc, Dv);
  M0(Dv, M0Dv);
  double m0_resid = std::sqrt(norm2(M0Dv - bsrc) / norm2(bsrc));
  std::cout << "  [" << label << "] M0-resid ||M0 D v - v||/||v|| = " << m0_resid << std::endl;

  if (run_m0) {
    M0.n_apply = 0;
    // Cap FGMRES(M0) at 2*iter_CGNE = dW_cgne/Ls (M0 is D_W-free; beyond this it cannot beat CGNE).
    int fgmres_maxit = (dW_cgne > 0) ? (int)(dW_cgne / Ls) : solve_maxit;
    FlexibleGeneralisedMinimalResidual<LatticeFermionD> FGMRES(solve_tol, fgmres_maxit, M0,
                                                               g_fgmres_restart,
                                                               /*err_on_no_conv=*/false);
    LatticeFermionD xg(FGrid);
    xg = Zero();
    FGMRES(LinOp, bsrc, xg);
    int fg_iters = FGMRES.IterationCount;
    long dW_fgmres = (long)Ls * fg_iters;
    double ratio = (dW_cgne > 0 && dW_fgmres > 0) ? (double)dW_cgne / (double)dW_fgmres : 0.0;
    std::cout << "  [" << label << "] M0: iters=" << fg_iters << "  D_W=" << dW_fgmres
              << "  ratio(CGNE/M0)=" << ratio << "x" << std::endl;
  }
  if (run_m1) {
    LatticeGaugeFieldD UL(UGrid);
    UL = Uorig;
    SU<Nc>::GaugeTransform<PeriodicGimplD>(UL, xform);  // U^L = Omega U Omega^dag (ORIGINAL config)
    MobiusFermionD Dframed(UL, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mm, M5, bb, cc, Params);
    FreeLimitPreconditioner1<WilsonImplD> M1(Ffree, xform, Dframed, FGrid);
    FlexibleGeneralisedMinimalResidual<LatticeFermionD> FGMRES1(solve_tol, solve_maxit, M1,
                                                                g_fgmres_restart,
                                                                /*err_on_no_conv=*/false);
    LatticeFermionD xg1(FGrid);
    xg1 = Zero();
    FGMRES1(LinOp, bsrc, xg1);
    int fg1_iters = FGMRES1.IterationCount;
    long dW_fgmres1 = (long)Ls * fg1_iters + (long)Ls * M1.n_dw;  // honest total (M1 not D_W-free)
    double ratio1 = (dW_cgne > 0 && dW_fgmres1 > 0) ? (double)dW_cgne / (double)dW_fgmres1 : 0.0;
    std::cout << "  [" << label << "] M1: iters=" << fg1_iters << "  internal D_DW[U^L]=" << M1.n_dw
              << "  D_W(total)=" << dW_fgmres1 << "  ratio(CGNE/M1)=" << ratio1 << "x" << std::endl;
  }
  std::cout.flush();
}

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  // Physics point (matches the freeprec/flowscan headline so D_W counts are comparable).
  const double M5 = 1.8;
  const int Ls = 8;
  const double bb = 1.5;
  const double cc = 0.5;
  const double mm = 0.1;
  std::vector<Complex> boundary = {1, 1, 1, -1};

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplexD::Nsimd());
  Coordinate mpi = GridDefaultMpi();
  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(latt, simd, mpi);
  GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian* FGrid = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
  GridRedBlackCartesian* FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);

  GridParallelRNG RNG5(FGrid);
  RNG5.SeedFixedIntegers(std::vector<int>({1, 2, 3, 4}));

  // Q-squeeze mode (qsqueeze_impl_plan_claude.md): stage 2 = PURE ||q||_n flow (QSqueezeGaugeAction),
  // scanning --qn (comma list, default 2). --fdcheck runs ONLY the force validation and exits.
  bool qsqueeze = GridCmdOptionExists(argv, argv + argc, "--qsqueeze");
  std::vector<int> qns;
  if (GridCmdOptionExists(argv, argv + argc, "--qn")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--qn");
    std::stringstream ss(a);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      if (!tok.empty()) {
        qns.push_back(std::stoi(tok));
      }
    }
  }
  if (qns.empty()) {
    qns.push_back(2);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--fdcheck")) {
    double fd_dt = 1.0e-3;
    if (GridCmdOptionExists(argv, argv + argc, "--fddt")) {
      std::string a = GridCmdOptionPayload(argv, argv + argc, "--fddt");
      GridCmdOptionFloat(a, fd_dt);
    }
    int rc = run_fdcheck(UGrid, qns, fd_dt);
    Grid_finalize();
    return rc;
  }

  if (!GridCmdOptionExists(argv, argv + argc, "--config")) {
    std::cout << "ERROR: need --config <NERSC gauge file>" << std::endl;
    Grid_finalize();
    return 1;
  }
  std::string cfgfile = GridCmdOptionPayload(argv, argv + argc, "--config");

  double t0 = 2.91;  // b2.6 16^4 (memory project-r2-config-gen); for s/t0 printout only
  if (GridCmdOptionExists(argv, argv + argc, "--t0")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--t0");
    GridCmdOptionFloat(a, t0);
  }
  double flow_eps = 0.02;
  if (GridCmdOptionExists(argv, argv + argc, "--flow_eps")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--flow_eps");
    GridCmdOptionFloat(a, flow_eps);
  }
  // Stage-2 flow step. For the ||q||_n squeeze we want a MUCH larger step (lambda is absorbed into
  // flow time) while stage 1 Wilson keeps eps=0.02 for a correct t0 scale. Default = flow_eps.
  double flow_eps2 = flow_eps;
  if (GridCmdOptionExists(argv, argv + argc, "--flow_eps2")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--flow_eps2");
    GridCmdOptionFloat(a, flow_eps2);
  }
  // Stage 1 Wilson steps: tau1 = eps*nstep1. COMMA LIST in qsqueeze mode = scan the Wilson pre-flow
  // time (how much bulk-smoothing to bank before the ||q||_n squeeze); default 146 = t0.
  std::vector<int> nstep1s;
  if (GridCmdOptionExists(argv, argv + argc, "--nstep1")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--nstep1");
    std::stringstream ss(a);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      if (!tok.empty()) {
        nstep1s.push_back(std::stoi(tok));
      }
    }
  }
  if (nstep1s.empty()) {
    nstep1s.push_back(146);
  }
  int nstep1 = nstep1s[0];  // scalar: the c1 path + the shared pre-branch stage 1 use the first value
  // Stage 2 rectangle coeff -- COMMA LIST to scan the shrink strength: epsilon = 1+12c1, stability
  // needs c1 < 1/8. E.g. --c1 0,0.0417,0.0833,0.1167 = epsilon 1 (Wilson), 1.5, 2, 2.4. Each c1 gets
  // its own stage2+stage3+frame+solve from the SHARED stage-1 config; CGNE + baseline frame run once.
  std::vector<double> c1s;
  std::vector<std::string> c1toks;  // original tokens, for labels + checkpoint filenames
  if (GridCmdOptionExists(argv, argv + argc, "--c1")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--c1");
    std::stringstream ss(a);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      if (!tok.empty()) {
        c1s.push_back(std::stod(tok));
        c1toks.push_back(tok);
      }
    }
  }
  if (c1s.empty()) {
    c1s.push_back(1.0 / 12.0);  // default epsilon = 2
    c1toks.push_back("0.0833");
  }
  // Checkpoint the flowed config every t0 of TOTAL flow time (tau1+tau2 crossing k*t0, chunk-boundary
  // resolution) as NERSC files <ckp_dir>/shrink_<cfg>_c1<tok>_k<k>, plus the post-stage-3 config as
  // ..._final. k=1 = end of stage 1 (c1-independent). Empty --ckp_dir disables. These are the inputs
  // for the instanton-size probe (q(x) lump analysis).
  std::string ckp_dir = "";
  if (GridCmdOptionExists(argv, argv + argc, "--ckp_dir")) {
    ckp_dir = GridCmdOptionPayload(argv, argv + argc, "--ckp_dir");
  }
  int nstep2 = 2000;  // Stage 2 max steps (tau2 up to 40 ~ 14 t0 at eps 0.02; pure gauge = inert, long is safe)
  if (GridCmdOptionExists(argv, argv + argc, "--nstep2")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--nstep2");
    GridCmdOptionInt(a, nstep2);
  }
  int qchunk = 10;  // Stage 2 steps between Q/plaq monitor prints
  if (GridCmdOptionExists(argv, argv + argc, "--qchunk")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--qchunk");
    GridCmdOptionInt(a, qchunk);
  }
  double qstop = 0.5;  // stop Stage 2 when |Q_5Li| < qstop (all lumps fell through); <=0 disables
  if (GridCmdOptionExists(argv, argv + argc, "--qstop")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--qstop");
    GridCmdOptionFloat(a, qstop);
  }
  int nstep3 = 20;  // Stage 3 Wilson re-smooth steps (erase the fall-through ripple)
  if (GridCmdOptionExists(argv, argv + argc, "--nstep3")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--nstep3");
    GridCmdOptionInt(a, nstep3);
  }
  double solve_tol = 1.0e-6;
  if (GridCmdOptionExists(argv, argv + argc, "--solve_tol")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--solve_tol");
    GridCmdOptionFloat(a, solve_tol);
  }
  int gf_maxit = 3000;
  if (GridCmdOptionExists(argv, argv + argc, "--gf_maxit")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--gf_maxit");
    GridCmdOptionInt(a, gf_maxit);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--fgmres_restart")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--fgmres_restart");
    GridCmdOptionInt(a, g_fgmres_restart);
  }
  bool run_cgne = true;
  bool run_m0 = true;
  bool run_m1 = false;
  if (GridCmdOptionExists(argv, argv + argc, "--ops")) {
    std::string ops = GridCmdOptionPayload(argv, argv + argc, "--ops");
    run_cgne = (ops.find("cgne") != std::string::npos);
    run_m0 = (ops.find("m0") != std::string::npos);
    run_m1 = (ops.find("m1") != std::string::npos);
  }
  bool run_baseline = true;  // Landau fix + solve straight after Stage 1 (the current default frame)
  if (GridCmdOptionExists(argv, argv + argc, "--no_baseline")) {
    run_baseline = false;
  }

  LatticeGaugeFieldD U(UGrid);
  FieldMetaData rheader;
  NerscIO::readConfiguration(U, rheader, cfgfile);
  Real plaq0 = WilsonLoops<PeriodicGimplD>::avgPlaquette(U);
  std::cout << "loaded " << cfgfile << "  plaq(computed)=" << plaq0 << "  header=" << rheader.plaquette
            << std::endl;
  std::cout << "shrinkflow: eps=" << flow_eps << "  stage1 nstep1=" << nstep1
            << " (tau1=" << flow_eps * nstep1 << ", s/t0=" << flow_eps * nstep1 / t0 << ")"
            << "  stage2 nstep2<=" << nstep2 << " qchunk=" << qchunk << " qstop=" << qstop
            << "  stage3 nstep3=" << nstep3 << std::endl;
  if (qsqueeze) {
    for (size_t iq = 0; iq < qns.size(); ++iq) {
      std::cout << "  qsqueeze: qn[" << iq << "]=" << qns[iq] << " (stage 2 = pure |q|_n flow)"
                << std::endl;
    }
  } else {
    for (size_t ic = 0; ic < c1s.size(); ++ic) {
      // Moran-Leinweber epsilon (0801.1165): c1 = -(1-epsilon)/12
      std::cout << "  c1[" << ic << "]=" << c1s[ic] << "  epsilon=" << 1.0 + 12.0 * c1s[ic]
                << "  c0=" << 1.0 - 8.0 * c1s[ic] << std::endl;
    }
  }

  WilsonImplD::ImplParams Params(boundary);
  MobiusFermionD D(U, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mm, M5, bb, cc, Params);
  FreeMobius5DInverse<WilsonImplD> Ffree(FGrid, Ls, M5, bb, cc, mm, boundary);
  NonHermitianLinearOperator<MobiusFermionD, LatticeFermionD> LinOp(D);
  LatticeFermionD bsrc(FGrid);
  gaussian(RNG5, bsrc);
  int solve_maxit = 4000;

  // CGNE baseline (frame-independent) -- once, reused for every frame's ratio.
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
    std::cout << "CGNE (frame-independent): iters=" << cg_iters << "  D_W applies=" << dW_cgne
              << std::endl;
  }

  // ---- Stage 1: Wilson flow to tau1 ~ t0 ----
  LatticeGaugeFieldD U1(UGrid);
  U1 = U;
  if (nstep1 > 0) {
    LatticeGaugeFieldD Utmp(UGrid);
    WilsonFlow<PeriodicGimplD> wf1(flow_eps, nstep1);
    wf1.smear(Utmp, U1);
    U1 = Utmp;
  }
  Real plaq1 = WilsonLoops<PeriodicGimplD>::avgPlaquette(U1);
  Real Q1 = WilsonLoops<PeriodicGimplD>::TopologicalCharge5Li(U1);
  std::cout << "stage1 done: tau1=" << flow_eps * nstep1 << "  plaq=" << plaq1 << "  Q_5Li=" << Q1
            << std::endl;
  std::string cfgbase = cfgfile.substr(cfgfile.find_last_of('/') + 1);
  if (!ckp_dir.empty() && !qsqueeze) {
    std::string f1 = ckp_dir + "/shrink_" + cfgbase + "_k1";
    NerscIO::writeConfiguration(U1, f1);
    std::cout << "checkpoint k=1 (end of stage 1, tau=" << flow_eps * nstep1 << ") -> " << f1
              << std::endl;
  }

  // ---- Baseline frame (A): the current default s/t0=1.0 Wilson frame, on a COPY ----
  // (qsqueeze mode does its OWN per-nstep1 baseline inside the scan branch below.)
  if (run_baseline && !qsqueeze) {
    LatticeGaugeFieldD Ubase(UGrid);
    Ubase = U1;
    frame_and_solve("baseline s/t0=1", Ubase, U, D, Ffree, LinOp, bsrc, UGrid, UrbGrid, FGrid, FrbGrid,
                    Ls, M5, bb, cc, mm, Params, dW_cgne, solve_tol, solve_maxit, gf_maxit,
                    run_m0, run_m1);
  }

  // ---- Q-SQUEEZE mode: OUTER scan over the Wilson pre-flow nstep1, INNER scan over n. ----
  // For each nstep1: Wilson-flow the bare U to tau1 (=U1v), baseline frame + solve, then per n the
  // pure ||q||_n stage 2 from U1v. Tokens/labels carry nstep1 only when it is scanned (>1 value).
  if (qsqueeze) {
   for (size_t in1 = 0; in1 < nstep1s.size(); ++in1) {
    int ns1 = nstep1s[in1];
    bool scan_n1 = (nstep1s.size() > 1);
    std::string n1tag = scan_n1 ? ("_n1" + std::to_string(ns1)) : "";
    std::string n1lab = scan_n1 ? (" n1=" + std::to_string(ns1)) : "";
    LatticeGaugeFieldD U1v(UGrid);
    U1v = U;
    if (ns1 > 0) {
      LatticeGaugeFieldD Utmp(UGrid);
      WilsonFlow<PeriodicGimplD> wf1v(flow_eps, ns1);
      wf1v.smear(Utmp, U1v);
      U1v = Utmp;
    }
    Real plaq1v = WilsonLoops<PeriodicGimplD>::avgPlaquette(U1v);
    Real Q1v = WilsonLoops<PeriodicGimplD>::TopologicalCharge5Li(U1v);
    std::cout << "== qsqueeze stage1" << n1lab << ": nstep1=" << ns1 << " tau1=" << flow_eps * ns1
              << " (s/t0=" << flow_eps * ns1 / t0 << ")  plaq=" << plaq1v << "  Q_5Li=" << Q1v
              << " ==" << std::endl;
    if (run_baseline) {
      LatticeGaugeFieldD Ubase(UGrid);
      Ubase = U1v;
      frame_and_solve("baseline" + n1lab, Ubase, U, D, Ffree, LinOp, bsrc, UGrid, UrbGrid, FGrid,
                      FrbGrid, Ls, M5, bb, cc, mm, Params, dW_cgne, solve_tol, solve_maxit, gf_maxit,
                      run_m0, run_m1);
    }
    for (size_t iq = 0; iq < qns.size(); ++iq) {
      int nq = qns[iq];
      std::string tok = "qn" + std::to_string(nq) + n1tag;
      std::string label = "qsq n=" + std::to_string(nq) + n1lab;
      int kckp = 2;
      QSqueezeGaugeAction<PeriodicGimplD> QSG(nq);
      LatticeGaugeFieldD Ucur(UGrid);
      Ucur = U1v;
      LatticeGaugeFieldD Uprev(UGrid);
      int steps_done = 0;
      bool diverged = false;
      while (steps_done < nstep2) {
        int chunk = qchunk;
        if (steps_done + chunk > nstep2) {
          chunk = nstep2 - steps_done;
        }
        Uprev = Ucur;
        LatticeGaugeFieldD Unext(UGrid);
        WilsonFlow<PeriodicGimplD> wf2(flow_eps2, chunk);
        wf2.setGaugeAction(&QSG);  // PURE ||q||_n force (lambda absorbed into flow time)
        wf2.smear(Unext, Ucur);
        Ucur = Unext;
        steps_done += chunk;
        Real plaq2 = WilsonLoops<PeriodicGimplD>::avgPlaquette(Ucur);
        if (!(plaq2 >= 0.0 && plaq2 <= 1.0001)) {
          std::cout << "[" << label << "] stage2 DIVERGED at step " << steps_done
                    << " (plaq=" << plaq2 << ") -- reverting to the previous chunk" << std::endl;
          Ucur = Uprev;
          steps_done -= chunk;
          diverged = true;
          break;
        }
        Real Q2 = WilsonLoops<PeriodicGimplD>::TopologicalCharge5Li(Ucur);
        QSG.qStats(Ucur);
        double rho0 = std::pow(6.0 / (M_PI * M_PI * QSG.last_qmax), 0.25);
        RealD tau2 = flow_eps2 * steps_done;
        RealD tau_tot = flow_eps * ns1 + tau2;  // stage 1 (eps) + stage 2 (eps2)
        std::cout << "[" << label << "] stage2 step=" << steps_done << "  tau2=" << tau2
                  << "  plaq=" << plaq2 << "  Q_5Li=" << Q2 << "  Qclover=" << QSG.last_Qclover
                  << "  qmax=" << QSG.last_qmax << "  rho0=" << rho0 << "  PR=" << QSG.last_PR
                  << "  |q|_n=" << QSG.last_qnorm << std::endl;
        std::cout.flush();
        while (!ckp_dir.empty() && tau_tot >= kckp * t0 - 1.0e-9) {
          std::string fck = ckp_dir + "/shrink_" + cfgbase + "_" + tok + "_k"
                            + std::to_string(kckp);
          NerscIO::writeConfiguration(Ucur, fck);
          std::cout << "[" << label << "] checkpoint k=" << kckp << " (tau_tot=" << tau_tot
                    << ", Q_5Li=" << Q2 << ") -> " << fck << std::endl;
          kckp = kckp + 1;
        }
        if (qstop > 0.0 && std::fabs(Q2) < qstop) {
          std::cout << "[" << label << "] stage2 STOP: |Q_5Li|=" << std::fabs(Q2) << " < qstop="
                    << qstop << " (lumps fell through)" << std::endl;
          break;
        }
      }
      Real plaq2f = WilsonLoops<PeriodicGimplD>::avgPlaquette(Ucur);
      Real Q2f = WilsonLoops<PeriodicGimplD>::TopologicalCharge5Li(Ucur);
      std::cout << "[" << label << "] stage2 done: steps=" << steps_done << "  tau2="
                << flow_eps2 * steps_done << "  plaq=" << plaq2f << "  Q_5Li=" << Q2f
                << "  diverged=" << (diverged ? 1 : 0) << std::endl;

      if (nstep3 > 0) {
        LatticeGaugeFieldD Utmp(UGrid);
        WilsonFlow<PeriodicGimplD> wf3(flow_eps, nstep3);
        wf3.smear(Utmp, Ucur);
        Ucur = Utmp;
        Real plaq3 = WilsonLoops<PeriodicGimplD>::avgPlaquette(Ucur);
        Real Q3 = WilsonLoops<PeriodicGimplD>::TopologicalCharge5Li(Ucur);
        std::cout << "[" << label << "] stage3 done: tau3=" << flow_eps * nstep3
                  << "  plaq=" << plaq3 << "  Q_5Li=" << Q3 << std::endl;
      }
      if (!ckp_dir.empty()) {
        std::string ffin = ckp_dir + "/shrink_" + cfgbase + "_" + tok + "_final";
        NerscIO::writeConfiguration(Ucur, ffin);
        std::cout << "[" << label << "] checkpoint final (post-stage3, the frame config) -> "
                  << ffin << std::endl;
      }
      frame_and_solve(label, Ucur, U, D, Ffree, LinOp, bsrc, UGrid, UrbGrid, FGrid, FrbGrid,
                      Ls, M5, bb, cc, mm, Params, dW_cgne, solve_tol, solve_maxit, gf_maxit,
                      run_m0, run_m1);
    }  // qn scan
   }  // nstep1 scan
    Grid_finalize();
    return 0;
  }

  // ---- Per-c1 scan: Stage 2 (shrink) + Stage 3 (re-smooth) + shrink frame (B), from the SHARED U1 ----
  for (size_t ic = 0; ic < c1s.size(); ++ic) {
    double c1 = c1s[ic];
    std::string label = "shrink c1=" + c1toks[ic];
    int kckp = 2;  // next t0-multiple checkpoint index (k=1 was the end of stage 1)

    // ---- Stage 2: under-improved shrink flow, chunked with a Q monitor ----
    Action<PeriodicGimplD::GaugeField>* shrinkSG =
        new RBCGaugeAction<PeriodicGimplD>(PeriodicGimplD::num_colours, c1);
    LatticeGaugeFieldD Ucur(UGrid);
    Ucur = U1;
    LatticeGaugeFieldD Uprev(UGrid);
    int steps_done = 0;
    bool diverged = false;
    while (steps_done < nstep2) {
      int chunk = qchunk;
      if (steps_done + chunk > nstep2) {
        chunk = nstep2 - steps_done;
      }
      Uprev = Ucur;
      LatticeGaugeFieldD Unext(UGrid);
      WilsonFlow<PeriodicGimplD> wf2(flow_eps, chunk);
      wf2.setGaugeAction(shrinkSG);  // swap the flow force to the under-improved (shrink) action
      wf2.smear(Unext, Ucur);
      Ucur = Unext;
      steps_done += chunk;
      Real plaq2 = WilsonLoops<PeriodicGimplD>::avgPlaquette(Ucur);
      // GUARD (as in the flowscan): a sane flowed plaquette is in (0,1]; NaN/out-of-range = blow-up.
      if (!(plaq2 >= 0.0 && plaq2 <= 1.0001)) {
        std::cout << "[" << label << "] stage2 DIVERGED at step " << steps_done << " (plaq=" << plaq2
                  << ") -- reverting to the previous chunk" << std::endl;
        Ucur = Uprev;
        steps_done -= chunk;
        diverged = true;
        break;
      }
      Real Q2 = WilsonLoops<PeriodicGimplD>::TopologicalCharge5Li(Ucur);
      RealD tau2 = flow_eps * steps_done;
      std::cout << "[" << label << "] stage2 step=" << steps_done << "  tau2=" << tau2
                << "  (tau1+tau2)/t0=" << (flow_eps * nstep1 + tau2) / t0 << "  plaq=" << plaq2
                << "  Q_5Li=" << Q2 << std::endl;
      std::cout.flush();
      while (!ckp_dir.empty() && flow_eps * (nstep1 + steps_done) >= kckp * t0 - 1.0e-9) {
        std::string fck = ckp_dir + "/shrink_" + cfgbase + "_c1" + c1toks[ic] + "_k"
                          + std::to_string(kckp);
        NerscIO::writeConfiguration(Ucur, fck);
        std::cout << "[" << label << "] checkpoint k=" << kckp << " (tau_tot="
                  << flow_eps * (nstep1 + steps_done) << ", Q_5Li=" << Q2 << ") -> " << fck
                  << std::endl;
        kckp = kckp + 1;
      }
      if (qstop > 0.0 && std::fabs(Q2) < qstop) {
        std::cout << "[" << label << "] stage2 STOP: |Q_5Li|=" << std::fabs(Q2) << " < qstop="
                  << qstop << " (lumps fell through)" << std::endl;
        break;
      }
    }
    delete shrinkSG;
    Real plaq2f = WilsonLoops<PeriodicGimplD>::avgPlaquette(Ucur);
    Real Q2f = WilsonLoops<PeriodicGimplD>::TopologicalCharge5Li(Ucur);
    std::cout << "[" << label << "] stage2 done: steps=" << steps_done << "  tau2="
              << flow_eps * steps_done << "  plaq=" << plaq2f << "  Q_5Li=" << Q2f
              << "  diverged=" << (diverged ? 1 : 0) << std::endl;

    // ---- Stage 3: short Wilson re-smooth ----
    if (nstep3 > 0) {
      LatticeGaugeFieldD Utmp(UGrid);
      WilsonFlow<PeriodicGimplD> wf3(flow_eps, nstep3);
      wf3.smear(Utmp, Ucur);
      Ucur = Utmp;
      Real plaq3 = WilsonLoops<PeriodicGimplD>::avgPlaquette(Ucur);
      Real Q3 = WilsonLoops<PeriodicGimplD>::TopologicalCharge5Li(Ucur);
      std::cout << "[" << label << "] stage3 done: tau3=" << flow_eps * nstep3 << "  plaq=" << plaq3
                << "  Q_5Li=" << Q3 << std::endl;
    }
    if (!ckp_dir.empty()) {
      std::string ffin = ckp_dir + "/shrink_" + cfgbase + "_c1" + c1toks[ic] + "_final";
      NerscIO::writeConfiguration(Ucur, ffin);
      std::cout << "[" << label << "] checkpoint final (post-stage3, the frame config) -> " << ffin
                << std::endl;
    }

    // ---- Shrink frame (B) for this c1 ----
    frame_and_solve(label, Ucur, U, D, Ffree, LinOp, bsrc, UGrid, UrbGrid, FGrid, FrbGrid,
                    Ls, M5, bb, cc, mm, Params, dW_cgne, solve_tol, solve_maxit, gf_maxit,
                    run_m0, run_m1);
  }  // c1 scan

  Grid_finalize();
  return 0;
}
