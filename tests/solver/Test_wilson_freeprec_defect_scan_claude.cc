// Free WILSON preconditioner test.
//  (no --config) Chunk 0 cold gate: ||F_W D_W v - v|| ~ eps at unit gauge/periodic vs Grid WilsonFermionD.
//  (--config <nersc>) Chunk 1+2 headline: flow+Landau -> Omega -> M0_W = Omega^dag F_W Omega; residual
//   proxy ||M0_W D_W[U] v - v||/||v||; then RB-CGNE (SchurRedBlackDiagMooee, honest baseline) vs
//   FGMRES(M0_W) on the AP-time interacting Wilson operator -- N_it + wall + speedup. Per-config mass via
//   --mass. Plan: dwf4_qcd_claude/grid_free_wilson_impl_plan_claude.md.
#include <Grid/Grid.h>
#include <Grid/qcd/utils/FreeWilson_claude.h>
#include <sstream>

using namespace Grid;

// one mass: build the mass-dependent operator + M0_W and solve. Omega (xform) is passed in (built ONCE
// per config in main, mass-independent -> reused every mass; the flow+Landau is the expensive part).
static void solve_wilson(LatticeGaugeFieldD& U, GridCartesian* UGrid, GridRedBlackCartesian* UrbGrid,
                         const LatticeColourMatrixD& xform, double mass, double mprec,
                         GridParallelRNG& RNG, int restart, int repeat) {
  // F_W built at m_prec (decoupled from the operator mass). The m_prec scan showed a Tikhonov optimum
  // (~0.8), roughly INDEPENDENT of the operator's physical mass -- so fix m_prec near the optimum and
  // scan the operator mass toward m_crit for the crossover.
  std::cout << GridLogMessage << "==== Wilson headline solve (m_bare=" << mass
            << ", m_prec=" << mprec << ", restart=" << restart << ") ====" << std::endl;
  // ---- AP-time interacting Wilson operator + framed free-Wilson preconditioner (Omega REUSED) ----
  std::vector<Complex> boundary(Nd, Complex(1.0, 0.0));
  boundary[Nd - 1] = Complex(-1.0, 0.0);  // anti-periodic time
  WilsonImplD::ImplParams params(boundary);
  WilsonFermionD Dw(U, *UGrid, *UrbGrid, mass, params);
  FreeWilsonInverse<WilsonImplD> Fw(UGrid, mprec, boundary);
  FreeLimitPreconditionerW<WilsonImplD> M0(Fw, xform, UGrid);

  LatticeFermionD bsrc(UGrid);
  gaussian(RNG, bsrc);
  RealD nv = std::sqrt(norm2(bsrc));

  // residual proxy ||M0_W D_W[U] v - v|| / ||v|| (the shared yardstick: how good an approx inverse is M0)
  LatticeFermionD Dv(UGrid);
  Dw.M(bsrc, Dv);
  LatticeFermionD M0Dv(UGrid);
  M0(Dv, M0Dv);
  RealD proxy = std::sqrt(norm2(M0Dv - bsrc)) / nv;
  std::cout << GridLogMessage << "  residual proxy ||M0_W D_W v - v||/||v|| = " << proxy << std::endl;

  RealD tol = 1.0e-8;
  int maxit = 20000;

  // RB-CGNE honest baseline
  ConjugateGradient<LatticeFermionD> CGrb(tol, maxit, false);
  SchurRedBlackDiagMooeeSolve<LatticeFermionD> Schur(CGrb);
  LatticeFermionD xrb(UGrid);
  // repeat the solve, take the MIN wall (discards the GPU warmup / JIT of the first call). Quiet-GPU timing.
  double tw_rb = 1.0e30;
  int rb_iters = 0;
  for (int r = 0; r < repeat; ++r) {
    xrb = Zero();
    double t = -usecond();
    Schur(Dw, bsrc, xrb);
    t += usecond();
    if (t < tw_rb) {
      tw_rb = t;
    }
    rb_iters = CGrb.IterationsToComplete;
  }
  std::cout << GridLogMessage << "  RB-CGNE(Wilson): iters=" << rb_iters
            << "  WALL=" << tw_rb / 1.0e6 << " s  (min of " << repeat << ")   [honest baseline]" << std::endl;

  // vanilla FULL CGNE (CG on the full MdagM, no even-odd) -- the SOFT baseline that many use unpreconditioned.
  MdagMLinearOperator<WilsonFermionD, LatticeFermionD> HermOp(Dw);
  LatticeFermionD bn(UGrid);
  Dw.Mdag(bsrc, bn);
  LatticeFermionD xcg(UGrid);
  xcg = Zero();
  ConjugateGradient<LatticeFermionD> CGf(tol, maxit, false);
  double tw_cgf = -usecond();
  CGf(HermOp, bn, xcg);
  tw_cgf += usecond();
  int cgf_iters = CGf.IterationsToComplete;
  std::cout << GridLogMessage << "  CGNE(full): iters=" << cgf_iters
            << "  WALL=" << tw_cgf / 1.0e6 << " s   [vanilla/soft baseline]" << std::endl;

  // FGMRES right-preconditioned by M0_W
  NonHermitianLinearOperator<WilsonFermionD, LatticeFermionD> LinOp(Dw);
  FlexibleGeneralisedMinimalResidual<LatticeFermionD> FGMRES(tol, maxit, M0, restart, false);
  LatticeFermionD xg(UGrid);
  double tw_fg = 1.0e30;
  int fg_iters = 0;
  for (int r = 0; r < repeat; ++r) {
    xg = Zero();
    double t = -usecond();
    FGMRES(LinOp, bsrc, xg);
    t += usecond();
    if (t < tw_fg) {
      tw_fg = t;
    }
    fg_iters = FGMRES.IterationCount;
  }
  std::cout << GridLogMessage << "  FGMRES(M0_W) restart=" << restart << ": iters=" << fg_iters
            << "  WALL=" << tw_fg / 1.0e6 << " s  (min of " << repeat << ")" << std::endl;
  if (tw_fg > 0.0) {
    std::cout << GridLogMessage << "  WALL speedup (RB-CGNE / FGMRES-M0_W)  = " << (tw_rb / tw_fg)
              << "x  [>1 = free-prec beats RB-CGNE (hard bar)]" << std::endl;
    std::cout << GridLogMessage << "  WALL speedup (CGNE-full / FGMRES-M0_W) = " << (tw_cgf / tw_fg)
              << "x  [>1 = free-prec beats VANILLA CGNE (soft bar)]" << std::endl;
  }
}

// PRECONDITIONER-MASS SCAN: fixed operator (m_bare), sweep the free-prec mass m_prec to find the empirical
// optimum (may prefer a slightly HEAVIER m_prec than the naive m - m_crit -- Tikhonov/mass-floor: a
// too-light F_W over-amplifies near-zero modes). RB-CGNE computed ONCE (operator fixed).
static void scan_mprec(LatticeGaugeFieldD& U, GridCartesian* UGrid, GridRedBlackCartesian* UrbGrid,
                       const LatticeColourMatrixD& xform, double mass, const std::vector<double>& mprec_list,
                       GridParallelRNG& RNG, int restart) {
  std::cout << GridLogMessage << "==== preconditioner-mass scan (operator m_bare=" << mass
            << ", restart=" << restart << ") ====" << std::endl;
  std::vector<Complex> boundary(Nd, Complex(1.0, 0.0));
  boundary[Nd - 1] = Complex(-1.0, 0.0);
  WilsonImplD::ImplParams params(boundary);
  WilsonFermionD Dw(U, *UGrid, *UrbGrid, mass, params);
  LatticeFermionD bsrc(UGrid);
  gaussian(RNG, bsrc);
  RealD nv = std::sqrt(norm2(bsrc));
  RealD tol = 1.0e-8;
  int maxit = 20000;

  ConjugateGradient<LatticeFermionD> CGrb(tol, maxit, false);
  SchurRedBlackDiagMooeeSolve<LatticeFermionD> Schur(CGrb);
  LatticeFermionD xrb(UGrid);
  xrb = Zero();
  double tw_rb = -usecond();
  Schur(Dw, bsrc, xrb);
  tw_rb += usecond();
  int rb_iters = CGrb.IterationsToComplete;
  std::cout << GridLogMessage << "  RB-CGNE(Wilson): iters=" << rb_iters << "  WALL=" << tw_rb / 1.0e6
            << " s   [baseline, fixed operator]" << std::endl;

  NonHermitianLinearOperator<WilsonFermionD, LatticeFermionD> LinOp(Dw);
  for (size_t i = 0; i < mprec_list.size(); ++i) {
    double mp = mprec_list[i];
    FreeWilsonInverse<WilsonImplD> Fw(UGrid, mp, boundary);
    FreeLimitPreconditionerW<WilsonImplD> M0(Fw, xform, UGrid);
    LatticeFermionD Dv(UGrid);
    Dw.M(bsrc, Dv);
    LatticeFermionD M0Dv(UGrid);
    M0(Dv, M0Dv);
    RealD proxy = std::sqrt(norm2(M0Dv - bsrc)) / nv;
    FlexibleGeneralisedMinimalResidual<LatticeFermionD> FGMRES(tol, maxit, M0, restart, false);
    LatticeFermionD xg(UGrid);
    xg = Zero();
    double tw_fg = -usecond();
    FGMRES(LinOp, bsrc, xg);
    tw_fg += usecond();
    int fg = FGMRES.IterationCount;
    std::cout << GridLogMessage << "  m_prec=" << mp << ": FGMRES(M0_W) iters=" << fg
              << "  WALL=" << tw_fg / 1.0e6 << " s  proxy=" << proxy
              << "  count-win(RB/FG)=" << (double)rb_iters / (double)fg
              << "  wall-win=" << tw_rb / tw_fg << std::endl;
  }
}

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplexD::Nsimd());
  Coordinate mpi = GridDefaultMpi();
  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(latt, simd, mpi);
  GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);

  GridParallelRNG RNG(UGrid);
  RNG.SeedFixedIntegers(std::vector<int>({1, 2, 3, 4}));

  double mass = 0.1;
  if (GridCmdOptionExists(argv, argv + argc, "--mass")) {
    mass = std::stod(GridCmdOptionPayload(argv, argv + argc, "--mass"));
  }
  int restart = 80;
  if (GridCmdOptionExists(argv, argv + argc, "--restart")) {
    restart = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--restart"));
  }
  int repeat = 1;   // --repeat N: run each solve N times, report the MIN wall (quiet-GPU timing, drop warmup)
  if (GridCmdOptionExists(argv, argv + argc, "--repeat")) {
    repeat = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--repeat"));
    if (repeat < 1) {
      repeat = 1;
    }
  }
  // frame-flow knobs (defect_scan): the frame Omega = WilsonFlow(flow_eps x flow_nstep) then Landau. For a
  // defect-flowed checkpoint pass --flow-nstep 0 (NO reflow -- the checkpoint IS the frame source). Default
  // 0.02 x 100 = tau 2 (the original hardcoded frame flow for raw configs).
  double flow_eps = 0.02;
  int flow_nstep = 100;
  if (GridCmdOptionExists(argv, argv + argc, "--flow-eps")) {
    flow_eps = std::stod(GridCmdOptionPayload(argv, argv + argc, "--flow-eps"));
  }
  if (GridCmdOptionExists(argv, argv + argc, "--flow-nstep")) {
    flow_nstep = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--flow-nstep"));
  }
  // OPERATOR reflow (defect_scan): standard Wilson flow applied to the LOADED config BEFORE building the
  // operator AND the frame. Use it to (a) conventionally flow a raw config to match the defect checkpoints'
  // smoothness (--op-reflow-nstep 232 = 4 t0, the missing smooth Q!=0 reference), and (b) heal the open-
  // boundary seam of a defect checkpoint with a short reflow (--op-reflow-nstep 58 = +1 t0). 0 = as-is.
  double op_reflow_eps = 0.02;
  int op_reflow_nstep = 0;
  if (GridCmdOptionExists(argv, argv + argc, "--op-reflow-eps")) {
    op_reflow_eps = std::stod(GridCmdOptionPayload(argv, argv + argc, "--op-reflow-eps"));
  }
  if (GridCmdOptionExists(argv, argv + argc, "--op-reflow-nstep")) {
    op_reflow_nstep = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--op-reflow-nstep"));
  }

  // ---- Chunk 0 cold gate: unit gauge, periodic ----
  LatticeGaugeFieldD Uunit(UGrid);
  Uunit = 1.0;
  std::vector<Complex> per(Nd, Complex(1.0, 0.0));
  WilsonImplD::ImplParams pparams(per);
  WilsonFermionD Dwf(Uunit, *UGrid, *UrbGrid, mass, pparams);
  FreeWilsonInverse<WilsonImplD> Fwf(UGrid, mass, per);
  LatticeFermionD v(UGrid);
  gaussian(RNG, v);
  RealD nv = std::sqrt(norm2(v));
  LatticeFermionD Dv(UGrid);
  Dwf.M(v, Dv);
  LatticeFermionD FDv(UGrid);
  Fwf(Dv, FDv);
  RealD e1 = std::sqrt(norm2(FDv - v)) / nv;
  std::cout << GridLogMessage << "==== Wilson cold gate (unit,periodic,m=" << mass << ") ||F_W D_W v - v||/||v|| = "
            << e1 << "  " << ((e1 < 1.0e-10) ? "PASS" : "FAIL") << std::endl;

  // ---- Chunk 1+2 headline (--config <nersc>) ----
  if (GridCmdOptionExists(argv, argv + argc, "--config")) {
    std::string cfg = GridCmdOptionPayload(argv, argv + argc, "--config");
    LatticeGaugeFieldD U(UGrid);
    FieldMetaData header;
    NerscIO::readConfiguration(U, header, cfg);
    std::cout << GridLogMessage << "  loaded " << cfg << std::endl;

    // OPERATOR reflow: standard-flow the loaded config in place -> the OPERATOR + frame both use the reflowed
    // config. Conventional-flow a raw config to match smoothness, or heal a defect checkpoint's seam.
    if (op_reflow_nstep > 0) {
      Real plaq_pre = WilsonLoops<PeriodicGimplD>::avgPlaquette(U);
      Real q_pre = WilsonLoops<PeriodicGimplD>::TopologicalCharge5Li(U);
      LatticeGaugeFieldD Ur(UGrid);
      WilsonFlow<PeriodicGimplD> wfop(op_reflow_eps, op_reflow_nstep);
      wfop.smear(Ur, U);
      U = Ur;
      Real plaq_post = WilsonLoops<PeriodicGimplD>::avgPlaquette(U);
      Real q_post = WilsonLoops<PeriodicGimplD>::TopologicalCharge5Li(U);
      std::cout << GridLogMessage << "  OP-REFLOW: standard flow tau=" << (op_reflow_eps * op_reflow_nstep)
                << " (nstep=" << op_reflow_nstep << ")  plaq " << plaq_pre << " -> " << plaq_post
                << "   Q_5Li " << q_pre << " -> " << q_post << std::endl;
    }

    // frame Omega: flow + Landau -- built ONCE (mass-independent), REUSED for every mass. --flow-nstep 0 =>
    // NO reflow (Uflowed = U): the defect-flowed checkpoint is the frame source directly.
    Real plaq0 = WilsonLoops<PeriodicGimplD>::avgPlaquette(U);
    LatticeGaugeFieldD Uflowed(UGrid);
    if (flow_nstep > 0) {
      WilsonFlow<PeriodicGimplD> wf(flow_eps, flow_nstep);
      wf.smear(Uflowed, U);
    } else {
      Uflowed = U;
    }
    LatticeColourMatrixD xform(UGrid);
    FourierAcceleratedGaugeFixer<PeriodicGimplD>::SteepestDescentGaugeFix(
        Uflowed, xform, 0.1 / 16.0, 1000, 1.0e-12, 1.0e-12, true, -1, false);
    Real landau = 1.0 - WilsonLoops<PeriodicGimplD>::linkTrace(Uflowed);
    Real Qflow = WilsonLoops<PeriodicGimplD>::TopologicalCharge5Li(Uflowed);
    std::cout << GridLogMessage << "  frame: plaq=" << plaq0 << "  Landau functional=" << landau
              << "  Q_5Li(frame-flow tau=" << (flow_eps * flow_nstep) << ")=" << Qflow
              << "  (flow_nstep=" << flow_nstep << "; Omega built once, reused per mass)" << std::endl;

    // masses: --mass-list "0.1,-0.5,-0.7" (comma-sep) or the single --mass.
    std::vector<double> masses;
    if (GridCmdOptionExists(argv, argv + argc, "--mass-list")) {
      std::stringstream ss(GridCmdOptionPayload(argv, argv + argc, "--mass-list"));
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        masses.push_back(std::stod(tok));
      }
    } else {
      masses.push_back(mass);
    }
    // --mprec-list "a,b,c": PRECONDITIONER-MASS SCAN at fixed operator mass (--mass). Takes precedence.
    if (GridCmdOptionExists(argv, argv + argc, "--mprec-list")) {
      std::vector<double> mprec_list;
      std::stringstream ss(GridCmdOptionPayload(argv, argv + argc, "--mprec-list"));
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        mprec_list.push_back(std::stod(tok));
      }
      scan_mprec(U, UGrid, UrbGrid, xform, mass, mprec_list, RNG, restart);
    } else {
      // preconditioner mass per operator mass: --mprec V = FIXED m_prec = V for all masses (the Tikhonov
      // optimum, ~0.8, is operator-mass-independent -> use this for the light-operator crossover). Else
      // --mcrit C -> m_prec = mass - C (physical mass). Else m_prec = mass (old bare behaviour).
      bool have_mprec = GridCmdOptionExists(argv, argv + argc, "--mprec");
      double mprec_fixed = have_mprec ? std::stod(GridCmdOptionPayload(argv, argv + argc, "--mprec")) : 0.0;
      double mcrit = 0.0;
      if (GridCmdOptionExists(argv, argv + argc, "--mcrit")) {
        mcrit = std::stod(GridCmdOptionPayload(argv, argv + argc, "--mcrit"));
      }
      for (size_t i = 0; i < masses.size(); ++i) {
        double mprec = have_mprec ? mprec_fixed : (masses[i] - mcrit);
        solve_wilson(U, UGrid, UrbGrid, xform, masses[i], mprec, RNG, restart, repeat);
      }
    }
  }

  Grid_finalize();
  return 0;
}
