// Free WILSON preconditioner test.
//  (no --config) Chunk 0 cold gate: ||F_W D_W v - v|| ~ eps at unit gauge/periodic vs Grid WilsonFermionD.
//  (--config <nersc>) Chunk 1+2 headline: flow+Landau -> Omega -> M0_W = Omega^dag F_W Omega; residual
//   proxy ||M0_W D_W[U] v - v||/||v||; then RB-CGNE (SchurRedBlackDiagMooee, honest baseline) vs
//   FGMRES(M0_W) on the AP-time interacting Wilson operator -- N_it + wall + speedup. Per-config mass via
//   --mass. Plan: dwf4_qcd_claude/grid_free_wilson_impl_plan_claude.md.
#include <Grid/Grid.h>
#include <Grid/qcd/utils/FreeWilson_claude.h>
#include <Grid/algorithms/iterative/ImplicitlyRestartedArnoldi_claude.h>
#include <sstream>

using namespace Grid;

// LinearFunction wrappers so the non-Hermitian Arnoldi (IRA) can be pointed at D_W and at the preconditioned
// M0 D_W (right-precond spectrum: D applied then M0; same spectrum as D M0). Used by the --spectrum mode.
struct DwLinOp : public LinearFunction<LatticeFermionD> {
  WilsonFermionD& Dw;
  DwLinOp(WilsonFermionD& d) : Dw(d) {}
  void operator()(const LatticeFermionD& in, LatticeFermionD& out) { Dw.M(in, out); }
};
struct M0DwLinOp : public LinearFunction<LatticeFermionD> {
  WilsonFermionD& Dw;
  FreeLimitPreconditionerW<WilsonImplD>& M0;
  mutable LatticeFermionD tmp;
  M0DwLinOp(WilsonFermionD& d, FreeLimitPreconditionerW<WilsonImplD>& m, GridBase* g) : Dw(d), M0(m), tmp(g) {}
  void operator()(const LatticeFermionD& in, LatticeFermionD& out) {
    Dw.M(in, tmp);
    M0(tmp, out);
  }
};

// ================= DIRECT FRAME OPTIMIZER (ported from dwf4 dwf4_frameopt_claude.h) =================
// Minimize L[Omega] = sum_v ||M0(Omega) D_W v - v||^2, M0 = Omega^dag F Omega, by gradient descent on the
// frame Omega in SU(3) (per config; warm-started at the Landau frame). Analytic gradient, FD-gated. Gradient:
//   a=Om w; b=F a; c=Om^dag b; r=c-v (w=D_W v); s = Om^dag g5 F g5 Om r  (F^dag = g5 F g5, Wilson g5-herm);
//   G(x) = Ta( sum_v [ traceSpin(outer(w,s)) - traceSpin(outer(c,r)) ] ). Descent: Om <- Om exp(+eta G)
// (dL/dt|_{X=G} = -2||G||^2 < 0; validated by the dwf4 8^4 FD gate). See grid_frame_optimizer_impl_plan_claude.md.

static RealD fo_loss(FreeWilsonInverse<WilsonImplD>& F, const LatticeColourMatrixD& Om,
                     const std::vector<LatticeFermionD>& w, const std::vector<LatticeFermionD>& v) {
  GridBase* g = Om.Grid();
  LatticeColourMatrixD Omdag = adj(Om);
  LatticeFermionD a(g), b(g), c(g), r(g);
  RealD L = 0.0;
  for (size_t p = 0; p < w.size(); ++p) {
    a = Om * w[p];
    F(a, b);
    c = Omdag * b;
    r = c - v[p];
    L += norm2(r);
  }
  return L;
}

static RealD fo_loss_force(FreeWilsonInverse<WilsonImplD>& F, const LatticeColourMatrixD& Om,
                           const std::vector<LatticeFermionD>& w, const std::vector<LatticeFermionD>& v,
                           LatticeColourMatrixD& Gf) {
  GridBase* g = Om.Grid();
  LatticeColourMatrixD Omdag = adj(Om);
  Gamma g5(Gamma::Algebra::Gamma5);
  LatticeFermionD a(g), b(g), c(g), r(g), t(g), fg(g), s(g);
  LatticeColourMatrixD M(g);
  M = Zero();
  RealD L = 0.0;
  for (size_t p = 0; p < w.size(); ++p) {
    a = Om * w[p];
    F(a, b);
    c = Omdag * b;
    r = c - v[p];
    L += norm2(r);
    // s = Om^dag g5 F g5 Om r
    t = Om * r;
    t = g5 * t;
    F(t, fg);
    fg = g5 * fg;
    s = Omdag * fg;
    // M += traceSpin(outer(w,s)) - traceSpin(outer(c,r))  (spin-summed colour outer products)
    M = M + traceSpin(outerProduct(w[p], s)) - traceSpin(outerProduct(c, r));
  }
  Gf = Ta(M);
  return L;
}

// FINITE-DIFFERENCE gate: random su(N) X; central-diff of L along Om exp(+-eps X) vs analytic 2 sum Re tr[X G].
static RealD fo_grad_check(FreeWilsonInverse<WilsonImplD>& F, const LatticeColourMatrixD& Om,
                           const std::vector<LatticeFermionD>& w, const std::vector<LatticeFermionD>& v,
                           GridParallelRNG& pRNG, RealD eps, RealD& fd, RealD& an) {
  GridBase* g = Om.Grid();
  LatticeColourMatrixD Gf(g);
  fo_loss_force(F, Om, w, v, Gf);
  LatticeColourMatrixD Mr(g);
  gaussian(pRNG, Mr);
  LatticeColourMatrixD X = Ta(Mr);
  ComplexD tr = TensorRemove(sum(trace(X * Gf)));
  an = 2.0 * real(tr);
  LatticeColourMatrixD Op = Om * expMat(X, eps, 12);
  LatticeColourMatrixD Omn = Om * expMat(X, -eps, 12);
  RealD Lp = fo_loss(F, Op, w, v);
  RealD Lm = fo_loss(F, Omn, w, v);
  fd = (Lp - Lm) / (2.0 * eps);
  return std::abs(fd - an) / (std::abs(an) + 1.0e-30);
}

// backtracking-line-search gradient descent. Updates Om in place; returns final L.
static RealD fo_descend(FreeWilsonInverse<WilsonImplD>& F, LatticeColourMatrixD& Om,
                        const std::vector<LatticeFermionD>& w, const std::vector<LatticeFermionD>& v,
                        int niter, RealD eta0, int maxls) {
  GridBase* g = Om.Grid();
  LatticeColourMatrixD Gf(g);
  RealD L = fo_loss_force(F, Om, w, v, Gf);
  RealD L0 = L;
  RealD eta = eta0;
  std::cout << GridLogMessage << "# it     L                  ||G||^2           eta" << std::endl;
  for (int it = 0; it < niter; ++it) {
    RealD gn2 = norm2(Gf);
    std::cout << GridLogMessage << "  " << it << "   " << L << "   " << gn2 << "   " << eta << std::endl;
    if (gn2 < 1.0e-22) {
      break;
    }
    bool acc = false;
    for (int ls = 0; ls < maxls; ++ls) {
      LatticeColourMatrixD Otry = Om * expMat(Gf, eta, 12);
      RealD Lt = fo_loss(F, Otry, w, v);
      if (Lt < L) {
        Om = Otry;
        L = Lt;
        eta *= 1.3;
        acc = true;
        break;
      }
      eta *= 0.5;
    }
    if (!acc) {
      std::cout << GridLogMessage << "  (line search stuck at it=" << it << "; floor)" << std::endl;
      break;
    }
    L = fo_loss_force(F, Om, w, v, Gf);
  }
  std::cout << GridLogMessage << "fo_descend: L " << L0 << " -> " << L << "  ("
            << ((L > 0.0) ? L0 / L : 0.0) << "x lower)" << std::endl;
  return L;
}

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

    // ---- SPECTRUM (--spectrum): IRA (non-Herm Arnoldi) low eigenvalues of D_W and of M0 D_W (Landau frame).
    // "Get the spectrum of D_W first" -- are there ~|Q| near-zero (IR/topological) modes, and does the free-
    // prec cluster them? Smallest-|lambda| wanted. Nstop/Nk/Nm via --spec-nstop/--spec-nk/--spec-nm. ----
    if (GridCmdOptionExists(argv, argv + argc, "--spectrum")) {
      int Nstop = 16, Nk = 32, Nm = 64, spmaxit = 300;
      if (GridCmdOptionExists(argv, argv + argc, "--spec-nstop")) {
        Nstop = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--spec-nstop"));
      }
      if (GridCmdOptionExists(argv, argv + argc, "--spec-nk")) {
        Nk = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--spec-nk"));
      }
      if (GridCmdOptionExists(argv, argv + argc, "--spec-nm")) {
        Nm = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--spec-nm"));
      }
      double mprec = mass;
      if (GridCmdOptionExists(argv, argv + argc, "--mprec")) {
        mprec = std::stod(GridCmdOptionPayload(argv, argv + argc, "--mprec"));
      }
      std::vector<Complex> bnd(Nd, Complex(1.0, 0.0));
      bnd[Nd - 1] = Complex(-1.0, 0.0);
      WilsonImplD::ImplParams psp(bnd);
      WilsonFermionD Dwsp(U, *UGrid, *UrbGrid, mass, psp);
      LatticeFermionD src(UGrid);
      gaussian(RNG, src);
      std::vector<ComplexD> eval;
      std::vector<LatticeFermionD> evec(Nm + 1, LatticeFermionD(UGrid));
      int Nconv = 0;

      std::cout << GridLogMessage << "==== SPECTRUM  m=" << mass << " mprec=" << mprec
                << "  Nstop=" << Nstop << " Nk=" << Nk << " Nm=" << Nm << " ====" << std::endl;
      // (1) D_W low spectrum
      std::cout << GridLogMessage << "-- D_W low spectrum (IRA, smallest |lambda|) --" << std::endl;
      DwLinOp dwop(Dwsp);
      ImplicitlyRestartedArnoldi<LatticeFermionD> iraD(dwop, Nstop, Nk, Nm, 1.0e-6, spmaxit);
      iraD.calc(eval, evec, src, Nconv);
      std::cout << GridLogMessage << "  D_W: Nconv=" << Nconv << std::endl;
      // FIX 1 -- RAYLEIGH-RITZ CLEANUP of the returned subspace: the IRA restart accumulates factorization
      // error (internal estimate optimistic vs the true residual), so extract the BEST eigenpairs from the
      // returned span {evec[0..Nconv-1]} directly: GS-orthonormalize -> Hs = B^dag D_W B (Nconv x Nconv) ->
      // small dense eigensolve -> refined Ritz vectors. Report refined lambda, CHIRALITY chi = Re<psi|g5|psi>
      // /<psi|psi> (|chi|~1 = topological chiral zero mode tied to Q; ~0 = non-chiral bulk), and the TRUE
      // residual ||D psi - lambda psi||/||psi||. Robust to the IRA restart bug (best-of-subspace + honest resid).
      // The refined modes are STORED (rr_mode/rr_lam/rr_chi) so we can then apply M0 D_W to each (below).
      std::vector<LatticeFermionD> rr_mode;
      std::vector<ComplexD> rr_lam;
      std::vector<RealD> rr_chi;
      if (Nconv > 0) {
        int m = Nconv;
        std::vector<LatticeFermionD> B;
        for (int i = 0; i < m; ++i) {
          B.push_back(evec[i]);
        }
        // modified Gram-Schmidt orthonormalize
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < i; ++j) {
            ComplexD o = innerProduct(B[j], B[i]);
            axpy(B[i], -o, B[j], B[i]);
          }
          RealD nn = std::sqrt(norm2(B[i]));
          B[i] = B[i] * (1.0 / nn);
        }
        std::vector<LatticeFermionD> AB;
        for (int i = 0; i < m; ++i) {
          LatticeFermionD t(UGrid);
          Dwsp.M(B[i], t);
          AB.push_back(t);
        }
        Eigen::MatrixXcd Hs(m, m);
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < m; ++j) {
            ComplexD o = innerProduct(B[i], AB[j]);
            Hs(i, j) = std::complex<double>(real(o), imag(o));
          }
        }
        Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es(Hs, true);
        // sort refined eigenvalues by |lambda| ascending
        std::vector<std::pair<double, int> > ord(m);
        for (int k = 0; k < m; ++k) {
          ord[k] = std::make_pair(std::abs(es.eigenvalues()(k)), k);
        }
        std::sort(ord.begin(), ord.end());
        Gamma g5(Gamma::Algebra::Gamma5);
        std::cout << GridLogMessage << "  D_W Rayleigh-Ritz refined low modes (lambda, chi, true resid):"
                  << std::endl;
        for (int s = 0; s < m; ++s) {
          int k = ord[s].second;
          std::complex<double> lam = es.eigenvalues()(k);
          ComplexD lamg(lam.real(), lam.imag());
          LatticeFermionD x(UGrid);
          x = Zero();
          for (int i = 0; i < m; ++i) {
            std::complex<double> c = es.eigenvectors()(i, k);
            ComplexD cg(c.real(), c.imag());
            axpy(x, cg, B[i], x);
          }
          RealD n2 = norm2(x);
          x = x * (1.0 / std::sqrt(n2));  // normalize the stored mode
          LatticeFermionD Ax(UGrid), rr_(UGrid), g5x(UGrid);
          Dwsp.M(x, Ax);
          axpy(rr_, -lamg, x, Ax);
          RealD rres = std::sqrt(norm2(rr_));
          g5x = g5 * x;
          ComplexD chi = innerProduct(x, g5x);
          std::cout << GridLogMessage << "  D_W RR mode[" << s << "]  lambda=(" << lam.real() << ","
                    << lam.imag() << ")  |lambda|=" << std::abs(lam) << "   chi=" << real(chi)
                    << "   ||r||/||psi||=" << rres << std::endl;
          rr_mode.push_back(x);
          rr_lam.push_back(lamg);
          rr_chi.push_back(real(chi));
        }
      }
      // (2) HOW THE PRECONDITIONER ACTS ON THE D_W EIGENMODES. For each D_W eigenmode phi (D_W phi = lambda
      // phi), apply the framed free-prec M0 to D_W phi and report |M0 D_W phi|/|phi| (=1 for a perfect prec
      // M0 = D_W^{-1}) AND the residual |M0 D_W phi - phi|/|phi|. The 3 chiral (topological) modes should be
      // POORLY handled (far from 1) while the non-chiral bulk should be ~1 -- the direct evidence that the
      // free-prec's blocker is the |Q| chiral modes. M0 = Landau frame (xform).
      FreeWilsonInverse<WilsonImplD> Ffo(UGrid, mprec, bnd);
      FreeLimitPreconditionerW<WilsonImplD> M0L(Ffo, xform, UGrid);
      std::cout << GridLogMessage << "-- M0 (Landau frame) acting on D_W eigenmodes --" << std::endl;
      std::cout << GridLogMessage << "   mode : |lambda|      chi        |M0 DW phi|/|phi|   |M0 DW phi - phi|/|phi|"
                << std::endl;
      {
        LatticeFermionD Dphi(UGrid), M0Dphi(UGrid), diff(UGrid);
        for (size_t k = 0; k < rr_mode.size(); ++k) {
          Dwsp.M(rr_mode[k], Dphi);
          M0L(Dphi, M0Dphi);            // M0 D_W phi
          RealD ratio = std::sqrt(norm2(M0Dphi));            // |phi| = 1 (normalized)
          diff = M0Dphi - rr_mode[k];
          RealD res = std::sqrt(norm2(diff));
          std::cout << GridLogMessage << "  M0act[" << k << "]  |lam|=" << std::sqrt(real(rr_lam[k])*real(rr_lam[k])+imag(rr_lam[k])*imag(rr_lam[k]))
                    << "   chi=" << rr_chi[k] << "   |M0DWphi|/|phi|=" << ratio
                    << "   |M0DWphi-phi|/|phi|=" << res << std::endl;
        }
      }
      Grid_finalize();
      return 0;
    }

    // ---- DIRECT FRAME OPTIMIZER (--frameopt): descend Omega on the TRUE prec mismatch vs the Landau frame ----
    if (GridCmdOptionExists(argv, argv + argc, "--frameopt")) {
      double mprec = mass;
      if (GridCmdOptionExists(argv, argv + argc, "--mprec")) {
        mprec = std::stod(GridCmdOptionPayload(argv, argv + argc, "--mprec"));
      }
      int nprobe = 4;
      if (GridCmdOptionExists(argv, argv + argc, "--fo-probes")) {
        nprobe = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--fo-probes"));
      }
      int fo_iter = 60;
      if (GridCmdOptionExists(argv, argv + argc, "--fo-iter")) {
        fo_iter = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--fo-iter"));
      }
      double fo_eta = 0.1;
      if (GridCmdOptionExists(argv, argv + argc, "--fo-eta")) {
        fo_eta = std::stod(GridCmdOptionPayload(argv, argv + argc, "--fo-eta"));
      }
      std::vector<Complex> bnd(Nd, Complex(1.0, 0.0));
      bnd[Nd - 1] = Complex(-1.0, 0.0);  // anti-periodic time
      FreeWilsonInverse<WilsonImplD> Ffo(UGrid, mprec, bnd);
      WilsonImplD::ImplParams pfo(bnd);
      WilsonFermionD Dwfo(U, *UGrid, *UrbGrid, mass, pfo);
      std::vector<LatticeFermionD> vlist, wlist;
      for (int p = 0; p < nprobe; ++p) {
        LatticeFermionD vv(UGrid), ww(UGrid);
        gaussian(RNG, vv);
        Dwfo.M(vv, ww);
        vlist.push_back(vv);
        wlist.push_back(ww);
      }
      std::cout << GridLogMessage << "==== frameopt: m=" << mass << " mprec=" << mprec
                << " probes=" << nprobe << " iter=" << fo_iter << " eta=" << fo_eta << " ====" << std::endl;
      RealD fd = 0.0, an = 0.0;
      RealD rel = fo_grad_check(Ffo, xform, wlist, vlist, RNG, 1.0e-4, fd, an);
      std::cout << GridLogMessage << "  FD gate: fd=" << fd << "  an=" << an << "  rel=" << rel
                << ((rel < 1.0e-4) ? "  PASS" : "  CHECK") << std::endl;
      LatticeColourMatrixD Omopt = xform;
      RealD L0 = fo_loss(Ffo, xform, wlist, vlist);
      std::cout << GridLogMessage << "  L(Landau frame) = " << L0 << std::endl;
      fo_descend(Ffo, Omopt, wlist, vlist, fo_iter, fo_eta, 25);
      // compare FGMRES N_it: Landau frame vs optimized frame (same operator, same RHS)
      NonHermitianLinearOperator<WilsonFermionD, LatticeFermionD> LinOp(Dwfo);
      LatticeFermionD bsrc(UGrid), xg(UGrid);
      gaussian(RNG, bsrc);
      RealD tol = 1.0e-8;
      int maxit = 20000;
      {
        FreeLimitPreconditionerW<WilsonImplD> M0L(Ffo, xform, UGrid);
        FlexibleGeneralisedMinimalResidual<LatticeFermionD> FG(tol, maxit, M0L, restart, false);
        xg = Zero();
        FG(LinOp, bsrc, xg);
        std::cout << GridLogMessage << "  FGMRES(M0, LANDAU frame):    iters=" << FG.IterationCount << std::endl;
      }
      {
        FreeLimitPreconditionerW<WilsonImplD> M0O(Ffo, Omopt, UGrid);
        FlexibleGeneralisedMinimalResidual<LatticeFermionD> FG(tol, maxit, M0O, restart, false);
        xg = Zero();
        FG(LinOp, bsrc, xg);
        std::cout << GridLogMessage << "  FGMRES(M0, OPTIMIZED frame): iters=" << FG.IterationCount << std::endl;
      }
      Grid_finalize();
      return 0;
    }

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
