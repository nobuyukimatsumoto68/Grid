// dwf_pvmg2_test_claude.cc -- Grid chunk 2 of the PVMG program (see dwf_pvmg_grid_impl_plan_claude.md).
//
// Tests the slice-wise s-Fourier PV inverse with multigrid inner solves (dwf_pvmg2_claude.h)
// against the chunk-1 FourierAcceleratedPV (5D Schur CG) path and the CGNE / SchurRedBlack baselines.
// b = c required (s-independent Wilson kernel mass; see header).
//
// Phases:
//   [1] slice-wise pvInv, plain-CG inner, tight tol: |M_PV pvInv(v) - v| / |v|  (validates the
//       4D twisted-mass + numerator-rotation path independently of MG)
//   [2] MG setup, then MG pvInv at tight tol: same residual check + apply counts
//   [3] FGMRES on M(m) right-preconditioned by loose MG pvInv (restart high: no Krylov discard)
//   [4] FGMRES + loose chunk-1 pvInv (5D Schur CG) reference at the same restart
//   [5] baselines: unpreconditioned CGNE, SchurRedBlack CG
//
// Claude never builds/runs -- user runs compile_dwf_pvmg_claude.sh / tmp_claude.sh and Claude reads the log.

#include <Grid/Grid.h>
#include "dwf_pvmg2_claude.h"

using namespace std;
using namespace Grid;

#ifndef NBASIS
#define NBASIS 24
#endif

// GPU free-memory probe (diagnosis of the 16^3x8 device OOM)
static void print_gpu_mem(const char* tag) {
#ifdef GRID_CUDA
  size_t fr = 0;
  size_t tot = 0;
  cudaMemGetInfo(&fr, &tot);
  std::cout << GridLogMessage << "GPUMEM " << tag << " : free " << fr / (1024.0 * 1024.0)
            << " MB of " << tot / (1024.0 * 1024.0) << " MB" << std::endl;
#endif
}

// right preconditioner for FGMRES: one loose chunk-1 Fourier-accelerated PV inverse
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

// right preconditioner for FGMRES: one loose slice-wise (MG) PV inverse
template <class Field, class PV>
class PVSlicePrecon : public LinearFunction<Field> {
public:
  using LinearFunction<Field>::operator();
  PV& pvs;
  RealD tol;
  int maxit;
  int calls;

  PVSlicePrecon(PV& pvs_, RealD tol_, int maxit_) : pvs(pvs_), tol(tol_), maxit(maxit_), calls(0) {}

  void operator()(const Field& in, Field& out) {
    print_gpu_mem("prec-call");
    out = Zero();
    pvs.pvInv(in, out, tol, maxit);
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
  const RealD b = parse_double(argc, argv, "--b", 1.0);
  const RealD c = parse_double(argc, argv, "--c", 1.0);
  const RealD mass = parse_double(argc, argv, "--mass", 0.01);
  const RealD tol_inner = parse_double(argc, argv, "--tolinner", 0.3);
  const RealD tol_outer = parse_double(argc, argv, "--tolouter", 1.0e-8);
  const int restart = parse_int(argc, argv, "--restart", 200);
  const bool hot = parse_flag(argc, argv, "--hot");
  const RealD gdis = parse_double(argc, argv, "--gdis", 0.0);   // weak-coupling disorder: U = exp(i g \sum_a c_a T^a)
  const int group_in_s = parse_int(argc, argv, "--sgroup", 2);
  // MG knobs
  const RealD p_cut = parse_double(argc, argv, "--pcut", 1.0);
  const int sdeg = parse_int(argc, argv, "--sdeg", 8);
  const RealD slo_frac = parse_double(argc, argv, "--slofrac", 0.03);
  const RealD ctol = parse_double(argc, argv, "--ctol", 0.05);
  const int blocksize = parse_int(argc, argv, "--block", 2);
  const RealD sub_lofrac = parse_double(argc, argv, "--sublofrac", 1.0e-3);
  const int sub_order = parse_int(argc, argv, "--subord", 100);
  const int phase = parse_int(argc, argv, "--phase", 0);   // 0 = all; 1..5 = single phase (fresh process per phase to bound device memory)
  const bool fixed = parse_flag(argc, argv, "--fixed");    // fixed-application preconditioner: 1 V-cycle (light) / degree-hdeg Chebyshev (heavy), no inner Krylov
  const int hdeg = parse_int(argc, argv, "--hdeg", 4);

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
  if (gdis > 0.0) {
    // weak-coupling random configuration (testbed analog of U(1) phases with strength g):
    // per link U = exp(i g \sum_a c_a T^a), c_a uniform in [-1/2, 1/2] (SU<Nc>::LieRandomize)
    LatticeColourMatrix Umu1(UGrid);
    for (int mu = 0; mu < Nd; mu++) {
      SU<Nc>::LieRandomize(RNG4, Umu1, gdis);
      PokeIndex<LorentzIndex>(Umu, Umu1, mu);
    }
  } else if (hot) {
    SU<Nc>::HotConfiguration(RNG4, Umu);
  } else {
    SU<Nc>::ColdConfiguration(Umu);
  }
  std::cout << GridLogMessage << "PVMG chunk2: Ls=" << Ls << " M5=" << M5 << " b=" << b << " c=" << c
            << " mass=" << mass << " tol_inner=" << tol_inner << " tol_outer=" << tol_outer
            << " restart=" << restart
            << (gdis > 0.0 ? " WEAK g=" : (hot ? " HOT" : " FREE")) << (gdis > 0.0 ? std::to_string(gdis) : "")
            << std::endl;
  std::cout << GridLogMessage << "PVMG chunk2 MG knobs: nbasis=" << NBASIS << " block=" << blocksize
            << " pcut=" << p_cut << " sdeg=" << sdeg << " slofrac=" << slo_frac << " ctol=" << ctol
            << " sublofrac=" << sub_lofrac << " subord=" << sub_order
            << (fixed ? " FIXED hdeg=" : "") << (fixed ? std::to_string(hdeg) : "") << std::endl;

  std::cout << GridLogMessage << "PVMG chunk2: plaquette = "
            << WilsonLoops<PeriodicGimplD>::avgPlaquette(Umu) << std::endl;
  print_gpu_mem("after-config");

  MobiusFermionD Ddwf(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mass, M5, b, c);
  MobiusFermionD Dpv(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid, 1.0, M5, b, c);

  LatticeFermionD src(FGrid);
  random(RNG5, src);
  LatticeFermionD chk(FGrid);

  PVSliceMG<MobiusFermionD, NBASIS> pvs(Dpv, Umu);
  print_gpu_mem("after-operators");
  pvs.p_cut = p_cut;
  pvs.sdeg = sdeg;
  pvs.slo_frac = slo_frac;
  pvs.ctol = ctol;
  pvs.fixed_apply = fixed;
  pvs.hdeg = hdeg;
  {
    std::cout << GridLogMessage << "PVMG chunk2: mu_s values:";
    for (int s = 0; s < Ls / 2; s++) {
      std::cout << " " << pvs.mus[s];
    }
    std::cout << "  (MG below pcut=" << p_cut << ")" << std::endl;
  }

  // ---------------- phase 1: slice-wise pvInv with plain-CG inner solves, tight tolerance
  if (phase == 0 || phase == 1) {
    pvs.zero_counters();
    RealD save_pcut = pvs.p_cut;
    pvs.p_cut = 0.0;   // force plain CG on all slices (MG not set up yet anyway)
    LatticeFermionD x(FGrid);
    x = Zero();
    pvs.pvInv(src, x, 1.0e-10, 30000);
    pvs.p_cut = save_pcut;
    Dpv.M(x, chk);
    chk = chk - src;
    std::cout << GridLogMessage << "PVMG [1] slice-wise pvInv(cg tight): |M_PV pvInv(v) - v| / |v| = "
              << std::sqrt(norm2(chk) / norm2(src))
              << "  fine applies " << pvs.fine_applies
              << "  inner iters " << pvs.inner_iters << std::endl;
  }

  // ---------------- MG setup (needed by phases 2 and 3)
  if (phase == 0 || phase == 2 || phase == 3) {
    std::vector<int> block({blocksize, blocksize, blocksize, blocksize});
    GridStopWatch w;
    w.Start();
    pvs.setup_mg(RNG4, block, sub_lofrac, sub_order);
    w.Stop();
    std::cout << GridLogMessage << "PVMG [2] MG setup time " << w.Elapsed() << std::endl;
    print_gpu_mem("after-mg-setup");
  }

  // ---------------- phase 2: MG pvInv at tight tolerance
  if (phase == 0 || phase == 2) {
    pvs.zero_counters();
    LatticeFermionD x(FGrid);
    x = Zero();
    pvs.pvInv(src, x, 1.0e-10, 30000);
    Dpv.M(x, chk);
    chk = chk - src;
    std::cout << GridLogMessage << "PVMG [2] slice-wise pvInv(MG tight): |M_PV pvInv(v) - v| / |v| = "
              << std::sqrt(norm2(chk) / norm2(src))
              << "  fine applies " << pvs.fine_applies
              << "  inner iters " << pvs.inner_iters
              << "  coarse iters " << pvs.coarse_iters << std::endl;
  }

  // ---------------- phase 3: FGMRES on M(m), right-preconditioned by loose MG pvInv
  if (phase == 0 || phase == 3) {
    print_gpu_mem("phase3-start");
    pvs.zero_counters();
    PVSlicePrecon<LatticeFermionD, decltype(pvs)> prec(pvs, tol_inner, 10000);
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
    std::cout << GridLogMessage << "PVMG [3] FGMRES+PVslice(" << (fixed ? "MG-fixed" : "MG") << " "
              << tol_inner << ") : outer iterations "
              << fgmres.IterationCount << "  prec applications " << prec.calls
              << "  fine applies " << pvs.fine_applies
              << "  inner iters " << pvs.inner_iters
              << "  coarse iters " << pvs.coarse_iters
              << "  true resid " << std::sqrt(norm2(chk) / norm2(src))
              << "  time " << w.Elapsed() << std::endl;
  }

  // ---------------- phase 4: FGMRES + loose chunk-1 pvInv (5D Schur CG) at the same restart
  if (phase == 0 || phase == 4) {
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
    std::cout << GridLogMessage << "PVMG [4] FGMRES+PV(5D Schur cg " << tol_inner << ") : outer iterations "
              << fgmres.IterationCount << "  prec applications " << prec.calls
              << "  true resid " << std::sqrt(norm2(chk) / norm2(src))
              << "  time " << w.Elapsed() << std::endl;
  }

  // ---------------- phase 5a: baseline unpreconditioned CGNE on M^dag M
  if (phase == 0 || phase == 5) {
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
    std::cout << GridLogMessage << "PVMG [5a] CGNE (unprec) : iterations " << cg.IterationsToComplete
              << "  true resid " << std::sqrt(norm2(chk) / norm2(src))
              << "  time " << w.Elapsed() << std::endl;
  }

  // ---------------- phase 5b: baseline red-black Schur CG (production-like)
  if (phase == 0 || phase == 5) {
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
    std::cout << GridLogMessage << "PVMG [5b] SchurRedBlack CG : iterations " << cg.IterationsToComplete
              << "  true resid " << std::sqrt(norm2(chk) / norm2(src))
              << "  time " << w.Elapsed() << std::endl;
  }

  Grid_finalize();
  return 0;
}
