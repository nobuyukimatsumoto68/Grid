// dwf_pvmg_test_claude.cc  -- Grid chunk 1 of the PVMG program (see dwf_pvmg_grid_impl_plan_claude.md).
//
// Uses Grid's FourierAcceleratedPV (C. Lehner / P. Boyle, Grid/qcd/action/fermion/FourierAcceleratedPV.h)
// = the s-Fourier Pauli-Villars inverse: antiperiodic s-twist + FFT along s, per-momentum twisted-mass
// Wilson solves (WilsonTMFermion5D), numerator rotation (pA +- pB \gamma_5)^{-1}, inverse FFT.
// This file: (1) validates pvInv as the exact M_PV^{-1} on this build's conventions;
// (2) uses pvInv with a LOOSE inner CG as right preconditioner in FGMRES on M(m);
// (3) baselines: unpreconditioned CGNE on M(m)^dag M(m), and red-black Schur CG (production-like).
// Free field (unit gauge) by default; --hot for a disordered SU(Nc) configuration.
//
// Testbed background and cost tables: ../../dwf_1p1_claude/dwf_1p1_summary_claude.md (Sec. 7-8).
// Claude never builds/runs -- user runs compile_dwf_pvmg_claude.sh / tmp_claude.sh and Claude reads the log.

#include <Grid/Grid.h>

using namespace std;
using namespace Grid;

// right preconditioner for FGMRES: one loose Fourier-accelerated PV inverse
template <class Field, class PV>
class PVPrecon : public LinearFunction<Field> {
public:
  using LinearFunction<Field>::operator();
  PV& fapv;
  int calls;

  PVPrecon(PV& fapv_) : fapv(fapv_), calls(0) {}

  void operator()(const Field& in, Field& out) {
    out = Zero();
    fapv.pvInv(in, out);
    calls++;
  }
};

static double parse_double(int argc, char** argv, const std::string& key, double def) {
  for (int i = 1; i < argc - 1; i++) {
    if (key == argv[i]) {
      return atof(argv[i + 1]);
    }
  }
  return def;
}

static int parse_int(int argc, char** argv, const std::string& key, int def) {
  for (int i = 1; i < argc - 1; i++) {
    if (key == argv[i]) {
      return atoi(argv[i + 1]);
    }
  }
  return def;
}

static bool parse_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; i++) {
    if (key == argv[i]) {
      return true;
    }
  }
  return false;
}

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  const int Ls = parse_int(argc, argv, "--Ls", 16);
  const RealD M5 = parse_double(argc, argv, "--M5", 1.5);
  const RealD b = parse_double(argc, argv, "--b", 1.0);      // user-selected b = c kernel
  const RealD c = parse_double(argc, argv, "--c", 1.0);      // (production disc code uses b=1.5, c=0.5; pvInv supports both)
  const RealD mass = parse_double(argc, argv, "--mass", 0.01);
  const RealD tol_inner = parse_double(argc, argv, "--tolinner", 0.3);
  const RealD tol_outer = parse_double(argc, argv, "--tolouter", 1.0e-8);
  const int restart = parse_int(argc, argv, "--restart", 24);
  const bool hot = parse_flag(argc, argv, "--hot");
  const int group_in_s = parse_int(argc, argv, "--sgroup", 2);

  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplexD::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian* FGrid = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
  GridRedBlackCartesian* FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);

  std::vector<int> seeds4({1, 2, 3, 4});
  std::vector<int> seeds5({5, 6, 7, 8});
  GridParallelRNG RNG4(UGrid);
  RNG4.SeedFixedIntegers(seeds4);
  GridParallelRNG RNG5(FGrid);
  RNG5.SeedFixedIntegers(seeds5);

  LatticeGaugeField Umu(UGrid);
  if (hot) {
    SU<Nc>::HotConfiguration(RNG4, Umu);
  } else {
    SU<Nc>::ColdConfiguration(Umu);
  }
  std::cout << GridLogMessage << "PVMG chunk1: Ls=" << Ls << " M5=" << M5 << " b=" << b << " c=" << c
            << " mass=" << mass << " tol_inner=" << tol_inner << " tol_outer=" << tol_outer
            << (hot ? " HOT" : " FREE") << std::endl;

  MobiusFermionD Ddwf(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mass, M5, b, c);
  MobiusFermionD Dpv(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid, 1.0, M5, b, c);

  LatticeFermionD src(FGrid);
  random(RNG5, src);
  LatticeFermionD tmp(FGrid);
  LatticeFermionD chk(FGrid);

  // ---------------- phase 1: validate pvInv against M_PV (tight inner tolerance)
  {
    ConjugateGradient<LatticeFermionD> cg_tight(1.0e-10, 30000);
    FourierAcceleratedPV<LatticeFermionD, MobiusFermionD, LatticeGaugeField> fapv(Dpv, Umu, cg_tight, group_in_s);
    LatticeFermionD x(FGrid);
    x = Zero();
    fapv.pvInv(src, x);
    Dpv.M(x, chk);
    chk = chk - src;
    std::cout << GridLogMessage << "PVMG [1] |M_PV pvInv(v) - v| / |v| = "
              << std::sqrt(norm2(chk) / norm2(src)) << std::endl;
  }

  // ---------------- phase 2: FGMRES on M(m), right-preconditioned by loose pvInv
  {
    ConjugateGradient<LatticeFermionD> cg_loose(tol_inner, 10000);
    FourierAcceleratedPV<LatticeFermionD, MobiusFermionD, LatticeGaugeField> fapv(Dpv, Umu, cg_loose, group_in_s);
    PVPrecon<LatticeFermionD, decltype(fapv)> prec(fapv);
    NonHermitianLinearOperator<MobiusFermionD, LatticeFermionD> LinOp(Ddwf);
    FlexibleGeneralisedMinimalResidual<LatticeFermionD> fgmres(tol_outer, 20000, prec, restart);
    LatticeFermionD x(FGrid);
    x = Zero();
    GridStopWatch w;
    w.Start();
    fgmres(LinOp, src, x);
    w.Stop();
    Ddwf.M(x, chk);
    chk = chk - src;
    std::cout << GridLogMessage << "PVMG [2] FGMRES+PV(cg " << tol_inner << ") : outer iterations "
              << fgmres.IterationCount << "  prec applications " << prec.calls
              << "  true resid " << std::sqrt(norm2(chk) / norm2(src))
              << "  time " << w.Elapsed() << std::endl;
  }

  // ---------------- phase 3a: baseline unpreconditioned CGNE on M^dag M
  {
    MdagMLinearOperator<MobiusFermionD, LatticeFermionD> HermOp(Ddwf);
    LatticeFermionD bne(FGrid);
    Ddwf.Mdag(src, bne);
    ConjugateGradient<LatticeFermionD> cg(tol_outer, 100000);
    LatticeFermionD x(FGrid);
    x = Zero();
    GridStopWatch w;
    w.Start();
    cg(HermOp, bne, x);
    w.Stop();
    Ddwf.M(x, chk);
    chk = chk - src;
    std::cout << GridLogMessage << "PVMG [3a] CGNE (unprec) : iterations " << cg.IterationsToComplete
              << "  true resid " << std::sqrt(norm2(chk) / norm2(src))
              << "  time " << w.Elapsed() << std::endl;
  }

  // ---------------- phase 3b: baseline red-black Schur CG (production-like)
  {
    ConjugateGradient<LatticeFermionD> cg(tol_outer, 100000);
    SchurRedBlackDiagMooeeSolve<LatticeFermionD> schur(cg);
    LatticeFermionD x(FGrid);
    x = Zero();
    GridStopWatch w;
    w.Start();
    schur(Ddwf, src, x);
    w.Stop();
    Ddwf.M(x, chk);
    chk = chk - src;
    std::cout << GridLogMessage << "PVMG [3b] SchurRedBlack CG : iterations " << cg.IterationsToComplete
              << "  true resid " << std::sqrt(norm2(chk) / norm2(src))
              << "  time " << w.Elapsed() << std::endl;
  }

  Grid_finalize();
  return 0;
}
