// G0 (quality gate) of the domain-decomposed (additive-Schwarz) free-limit Mobius DWF preconditioner,
// Grid port. See qed2/dwf4_qcd_claude/grid_dd_freeprec_impl_plan_claude.md.
//
// Replaces the exact global free inverse F inside M0 = Omega^dag F Omega with a NODE-LOCAL block one
// (BlockFreeMobius5DInverse), and measures the outer FGMRES D_W count vs the exact-F baseline and vs
// CGNE, scanning block core side + halo overlap. VALIDATION: core=L, halo=0 must reproduce the exact-F
// FGMRES iteration count BIT-FOR-BIT (the block-local kernel on the full-size ext grid == the exact
// kernel; this also confirms the block-local AP handling). SINGLE-RANK only (G0 gather uses the full
// per-rank lex array); the MPI node=block performance version is G1 (later).
//
// Metric (dwf4 convention): D_W applies = CGNE 2*Ls*iters (normal op = 2 M/iter), FGMRES Ls*iters
// (1 M/iter; M0 costs 0 D_W). Run e.g.:
//   ./Test_dwf_freeprec_dd_claude --grid 16.16.16.16 --config <nersc> --mass 0.1
// No --config => Hot-config smoke test (plumbing only, not the physics win).

#include <Grid/Grid.h>
#include <Grid/qcd/utils/FreeMobius5D_claude.h>
#include <Grid/qcd/utils/BlockFreeMobius5D_claude.h>

#include <string>
#include <vector>

using namespace Grid;

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  const double M5 = 1.8;
  const int Ls = 8;
  const double bb = 1.5;   // headline Shamir point
  const double cc = 0.5;
  double mm = 0.1;
  if (GridCmdOptionExists(argv, argv + argc, "--mass")) {
    mm = std::stod(GridCmdOptionPayload(argv, argv + argc, "--mass"));
  }
  int min_core = 2;   // skip block cores below this (the tiny blocks are slow AND guaranteed bad);
  if (GridCmdOptionExists(argv, argv + argc, "--min-core")) {
    min_core = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--min-core"));
  }
  std::vector<Complex> boundary = {1, 1, 1, -1};  // anti-periodic time

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplexD::Nsimd());
  Coordinate mpi = GridDefaultMpi();
  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(latt, simd, mpi);
  GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian* FGrid = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
  GridRedBlackCartesian* FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);

  assert(latt[0] == latt[1] && latt[1] == latt[2] && latt[2] == latt[3] &&
         "DD gate needs a symmetric lattice (cubic blocks)");
  int L = latt[0];
  bool single_rank = (FGrid->_Nprocessors == 1);

  std::cout << "==== G0: DD (block-local Schwarz) free-prec  M5=" << M5 << " Ls=" << Ls
            << " b=" << bb << " c=" << cc << " m=" << mm << " L=" << L << " ====" << std::endl;

  GridParallelRNG RNG5(FGrid);
  RNG5.SeedFixedIntegers(std::vector<int>({1, 2, 3, 4}));
  GridParallelRNG RNG4(UGrid);
  RNG4.SeedFixedIntegers(std::vector<int>({5, 6, 7, 8}));

  LatticeGaugeFieldD U(UGrid);
  bool have_cfg = GridCmdOptionExists(argv, argv + argc, "--config");
  if (have_cfg) {
    std::string cfgfile = GridCmdOptionPayload(argv, argv + argc, "--config");
    FieldMetaData rheader;
    NerscIO::readConfiguration(U, rheader, cfgfile);
    Real plaqr = WilsonLoops<PeriodicGimplD>::avgPlaquette(U);
    std::cout << "  loaded " << cfgfile << "  plaq(computed)=" << plaqr
              << "  plaq(header)=" << rheader.plaquette << std::endl;
  } else {
    SU<Nc>::HotConfiguration(RNG4, U);
    std::cout << "  no --config: Hot-config smoke test (plumbing, not the physics win)" << std::endl;
  }

  // frame = WilsonFlow then Fourier Landau (same recipe as Test_dwf_freeprec_claude run_headline).
  LatticeGaugeFieldD Uflowed(UGrid);
  WilsonFlow<PeriodicGimplD> wf(0.02, 100);
  wf.smear(Uflowed, U);
  Real plaq_flowed = WilsonLoops<PeriodicGimplD>::avgPlaquette(Uflowed);
  LatticeColourMatrixD xform(UGrid);
  RealD gf_alpha = 0.1 / 16.0;   // Grid's FA step is 16x too large (GaugeFix.h:198); 0.1/16 = dwf4 step
  FourierAcceleratedGaugeFixer<PeriodicGimplD>::SteepestDescentGaugeFix(
      Uflowed, xform, gf_alpha, 1000, 1.0e-12, 1.0e-12, /*Fourier=*/true, /*orthog=*/-1,
      /*err_on_no_converge=*/false);
  Real landau = 1.0 - WilsonLoops<PeriodicGimplD>::linkTrace(Uflowed);
  std::cout << "  flowed plaq=" << plaq_flowed << "  flowed-fixed Landau functional=" << landau
            << std::endl;

  WilsonImplD::ImplParams Params(boundary);
  MobiusFermionD D(U, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mm, M5, bb, cc, Params);
  NonHermitianLinearOperator<MobiusFermionD, LatticeFermionD> LinOp(D);

  LatticeFermionD bsrc(FGrid);
  gaussian(RNG5, bsrc);
  RealD solve_tol = 1.0e-8;
  int solve_maxit = 4000;
  int fgmres_restart = 256;   // no-restart (RestartLength >> expected iters)

  // ---- CGNE baseline ----
  MdagMLinearOperator<MobiusFermionD, LatticeFermionD> HermOp(D);
  LatticeFermionD bn(FGrid);
  D.Mdag(bsrc, bn);
  LatticeFermionD xcg(FGrid);
  xcg = Zero();
  ConjugateGradient<LatticeFermionD> CG(solve_tol, solve_maxit);
  CG(HermOp, bn, xcg);
  int cg_iters = CG.IterationsToComplete;
  long dW_cgne = (long)2 * Ls * cg_iters;
  std::cout << "  CGNE      : iters=" << cg_iters << "  D_W=" << dW_cgne << std::endl;

  // ---- exact-F FGMRES (reference) ----
  FreeMobius5DInverse<WilsonImplD> Ffree(FGrid, Ls, M5, bb, cc, mm, boundary);
  FreeLimitPreconditioner<WilsonImplD> M0ex(Ffree, xform, FGrid);
  FlexibleGeneralisedMinimalResidual<LatticeFermionD> FGMRESex(solve_tol, solve_maxit, M0ex,
                                                              fgmres_restart, /*err_on_no_conv=*/false);
  LatticeFermionD xex(FGrid);
  xex = Zero();
  FGMRESex(LinOp, bsrc, xex);
  int ex_iters = FGMRESex.IterationCount;
  long dW_ex = (long)Ls * ex_iters;
  double ex_speed = (dW_ex > 0) ? (double)dW_cgne / (double)dW_ex : 0.0;
  std::cout << "  exact F   : FGMRES iters=" << ex_iters << "  D_W=" << dW_ex << "  => " << ex_speed
            << "x" << std::endl;

  if (!single_rank) {
    std::cout << "  [skip DD scan] G0 block-local kernel is single-rank only; rerun at --mpi 1.1.1.1"
              << std::endl;
    Grid_finalize();
    return 0;
  }

  // ---- block-local scan: core in {L,8,4,2} x halo in {0,1,2} ----
  int cand[4] = {L, 8, 4, 2};
  int halos[3] = {0, 1, 2};
  int seen[8];
  int nseen = 0;
  bool gate_val = false;
  for (int ic = 0; ic < 4; ++ic) {
    int core = cand[ic];
    if (core > L || L % core != 0) {
      continue;
    }
    if (core < min_core) {
      continue;
    }
    bool dup = false;
    for (int k = 0; k < nseen; ++k) {
      if (seen[k] == core) {
        dup = true;
      }
    }
    if (dup) {
      continue;
    }
    seen[nseen] = core;
    nseen += 1;
    for (int ih = 0; ih < 3; ++ih) {
      int halo = halos[ih];
      if (core == L && halo > 0) {
        continue;   // ext would exceed / self-overlap the whole lattice
      }
      if (core + 2 * halo > L) {
        continue;   // extended block larger than the lattice: periodic double-wrap
      }
      BlockFreeMobius5DInverse<WilsonImplD> Fblk(FGrid, Ls, M5, bb, cc, mm, boundary, core, halo);
      BlockFreeLimitPreconditioner<WilsonImplD> M0b(Fblk, xform, Ls, FGrid);
      FlexibleGeneralisedMinimalResidual<LatticeFermionD> FGMRESb(solve_tol, solve_maxit, M0b,
                                                                 fgmres_restart, /*err_on_no_conv=*/false);
      LatticeFermionD xb(FGrid);
      xb = Zero();
      FGMRESb(LinOp, bsrc, xb);
      int b_iters = FGMRESb.IterationCount;
      long dW_b = (long)Ls * b_iters;
      double b_speed = (dW_b > 0) ? (double)dW_cgne / (double)dW_b : 0.0;
      int npd = L / core;
      int nblk = npd * npd * npd * npd;
      std::cout << "  block     (core=" << core << " halo=" << halo << " ext=" << core + 2 * halo
                << " nblk=" << nblk << "): FGMRES iters=" << b_iters << "  D_W=" << dW_b << "  => "
                << b_speed << "x  (exact " << ex_iters << " it)" << std::endl;
      if (core == L && halo == 0) {
        gate_val = (b_iters == ex_iters);
        std::cout << "  [validation] core=L,halo=0 iters == exact-F iters : "
                  << (gate_val ? "PASS" : "FAIL") << "  (" << b_iters << " vs " << ex_iters << ")"
                  << std::endl;
      }
    }
  }

  std::cout << "==== G0 DD scan done; core=L,halo=0 validation " << (gate_val ? "PASS" : "FAIL")
            << " ====" << std::endl;

  Grid_finalize();
  return gate_val ? 0 : 1;
}
