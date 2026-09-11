/*************************************************************************************
  Test_dwf_svddump_deflprec_claude.cc  -- (2) SINGULAR-VECTOR DUMP for the deflated preconditioner.

  Computes the lowest Nkeep RIGHT singular vectors v_i of D5 (= lowest eigenvectors of D5^H D5 = MdagM,
  sigma_i = sqrt(lambda_i) ~ m for the topological ones) via Chebyshev-IRL, and dumps them (+ eigenvalues)
  to --ckpt_dir as "deflfull". A separate binary (Test_dwf_deflprec_claude.cc) reads them, forms the left
  vectors u_i = D5 v_i/sigma_i, and runs the SVD-deflation + M0-FGMRES two-level solve. Shared helpers:
  twolevel_common_claude.h. Also prints sigma + chi_q (topology check).

  CLI: --config <NERSC> --ckpt_dir <dir> [--nmodes 24] [--cheb_ord 21] [--cheb_lo 0.5]
*************************************************************************************/
#include "twolevel_common_claude.h"

using namespace Grid;

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  const double M5 = TL_M5;
  const int Ls = TL_Ls;
  const double bb = TL_bb;
  const double cc = TL_cc;
  const double mm = TL_mm;
  std::vector<Complex> boundary = {1, 1, 1, -1};

  int Nkeep = 24;
  int Nk = 40;
  int Nm = 64;
  RealD cheb_lo = 0.5;
  RealD cheb_hi = -1.0;   // auto = 1.1*lambda_max
  int cheb_ord = 21;
  RealD sv_resid = 1.0e-5;
  int sv_maxit = 200;
  if (GridCmdOptionExists(argv, argv + argc, "--nmodes")) Nkeep = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--nmodes"));
  // size the IRL working/max space to Nkeep so larger dumps (w_o=48) converge (defaults 40/64 at Nkeep=24).
  Nk = Nkeep + 16;
  Nm = 2 * Nkeep + 16;
  if (GridCmdOptionExists(argv, argv + argc, "--cheb_ord")) cheb_ord = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--cheb_ord"));
  if (GridCmdOptionExists(argv, argv + argc, "--cheb_lo")) cheb_lo = std::stod(GridCmdOptionPayload(argv, argv + argc, "--cheb_lo"));
  std::string ckpt_dir = "";
  if (GridCmdOptionExists(argv, argv + argc, "--ckpt_dir")) ckpt_dir = GridCmdOptionPayload(argv, argv + argc, "--ckpt_dir");
  if (ckpt_dir.empty()) {
    std::cout << GridLogError << "--ckpt_dir is required (where to dump deflfull)" << std::endl;
    Grid_finalize();
    return 1;
  }

  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(
      GridDefaultLatt(), GridDefaultSimd(Nd, vComplexD::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian* FGrid = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
  GridRedBlackCartesian* FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);

  GridParallelRNG RNG5(FGrid);
  RNG5.SeedFixedIntegers({1, 2, 3, 4, 5});

  LatticeGaugeFieldD Umu(UGrid);
  if (GridCmdOptionExists(argv, argv + argc, "--config")) {
    std::string cfg = GridCmdOptionPayload(argv, argv + argc, "--config");
    FieldMetaData header;
    NerscIO::readConfiguration(Umu, header, cfg);
    std::cout << GridLogMessage << "loaded " << cfg << "  plaq=" << WilsonLoops<PeriodicGimplD>::avgPlaquette(Umu) << std::endl;
  } else {
    std::cout << GridLogMessage << "no --config: cold start (unit gauge)" << std::endl;
    SU<Nc>::ColdConfiguration(Umu);
  }

  MobiusFermionD::ImplParams Params(boundary);
  MobiusFermionD D(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mm, M5, bb, cc, Params);

  std::vector<LatticeFermionD> modes;
  std::vector<RealD> lam;
  int nkeep = compute_low_modes(D, FGrid, RNG5, Nkeep, Nk, Nm, cheb_lo, cheb_hi, cheb_ord, sv_resid, sv_maxit, modes, lam);
  report_chirality(D, UGrid, modes, lam);

  std::cout << GridLogMessage << "dumping " << nkeep << " D5^H D5 right singular vectors (full 5D):" << std::endl;
  save_deflation_vectors(modes, lam, ckpt_dir, "deflfull");

  std::cout << GridLogMessage << "svddump_deflprec done" << std::endl;
  Grid_finalize();
  return 0;
}
