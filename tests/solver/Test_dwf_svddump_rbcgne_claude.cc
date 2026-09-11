/*************************************************************************************
  Test_dwf_svddump_rbcgne_claude.cc  -- (1) SINGULAR-VECTOR DUMP for deflated RB-CGNE.

  Computes the lowest defl_k eigenvectors of the RB-Schur^2 operator (SchurDiagMooeeOperator = S^dag S,
  the operator RB-CGNE's CG actually runs on) via Chebyshev-IRL on the ODD checkerboard, embeds each into
  a full 5D field (even=0), and dumps them (+ eigenvalues) to --ckpt_dir as "deflrb". A separate binary
  (Test_dwf_rbcgne_claude.cc) reads them back and runs the deflated RB-CGNE solve, so the Lanczos never
  co-resides with the solve. Shared helpers: twolevel_common_claude.h.

  CLI: --config <NERSC> --ckpt_dir <dir> [--defl_k 10] [--cheb_ord 21] [--cheb_lo 0.5]
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

  int defl_k = 10;
  int cheb_ord = 21;
  RealD cheb_lo = 0.5;
  if (GridCmdOptionExists(argv, argv + argc, "--defl_k")) defl_k = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--defl_k"));
  if (GridCmdOptionExists(argv, argv + argc, "--cheb_ord")) cheb_ord = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--cheb_ord"));
  if (GridCmdOptionExists(argv, argv + argc, "--cheb_lo")) cheb_lo = std::stod(GridCmdOptionPayload(argv, argv + argc, "--cheb_lo"));
  std::string ckpt_dir = "";
  if (GridCmdOptionExists(argv, argv + argc, "--ckpt_dir")) ckpt_dir = GridCmdOptionPayload(argv, argv + argc, "--ckpt_dir");
  if (ckpt_dir.empty()) {
    std::cout << GridLogError << "--ckpt_dir is required (where to dump deflrb)" << std::endl;
    Grid_finalize();
    return 1;
  }

  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(
      GridDefaultLatt(), GridDefaultSimd(Nd, vComplexD::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian* FGrid = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
  GridRedBlackCartesian* FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);

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

  // Lowest defl_k modes of the RB-Schur^2 operator S^dag S on the ODD checkerboard, via Chebyshev-IRL.
  std::cout << GridLogMessage << "==== RB-Schur^2 low modes (Chebyshev-IRL, order " << cheb_ord << ") ====" << std::endl;
  SchurDiagMooeeOperator<MobiusFermionD, LatticeFermionD> SchurOp(D);
  // Draw RB (odd-cb) sources from a FULL-grid RNG + pickCheckerboard -- do NOT create a GridParallelRNG on
  // FrbGrid: that red-black-grid RNG allocation is what blew the address space to ~880 GB and crashed the
  // dump at the first host malloc (deflprec, working entirely on FGrid, stays at ~22 GB). j7514762/j7514784.
  GridParallelRNG RNG5(FGrid);
  RNG5.SeedFixedIntegers({7, 8, 9, 10});
  LatticeFermionD ftmp(FGrid);
  LatticeFermionD pmrb(FrbGrid);
  pmrb.Checkerboard() = Odd;
  gaussian(RNG5, ftmp);
  pickCheckerboard(Odd, pmrb, ftmp);
  PowerMethod<LatticeFermionD> PMrb;
  RealD lmax_rb = PMrb(SchurOp, pmrb);
  Chebyshev<LatticeFermionD> Cheby_rb(cheb_lo, 1.1 * lmax_rb, cheb_ord);
  FunctionHermOp<LatticeFermionD> OpCheby_rb(Cheby_rb, SchurOp);
  PlainHermOp<LatticeFermionD> Op_rb(SchurOp);
  ImplicitlyRestartedLanczos<LatticeFermionD> IRLrb(OpCheby_rb, Op_rb, defl_k, 2 * defl_k, 4 * defl_k, 1.0e-5, 300);
  std::vector<RealD> eval_rb(4 * defl_k);
  std::vector<LatticeFermionD> evec_rb(4 * defl_k, LatticeFermionD(FrbGrid));
  for (auto& e : evec_rb) e.Checkerboard() = Odd;
  LatticeFermionD rbsrc(FrbGrid);
  rbsrc.Checkerboard() = Odd;
  gaussian(RNG5, ftmp);
  pickCheckerboard(Odd, rbsrc, ftmp);
  int Nconv_rb = 0;
  IRLrb.calc(eval_rb, evec_rb, rbsrc, Nconv_rb);
  int kdrb = (defl_k < Nconv_rb) ? defl_k : Nconv_rb;
  std::cout << GridLogMessage << "  RB-Schur^2 low modes: Nconv=" << Nconv_rb << " lowest eval="
            << eval_rb[0] << "  (dumping " << kdrb << ")" << std::endl;

  // Embed each odd-cb mode into a full 5D field (even=0) and dump (uniform format with deflfull).
  std::vector<LatticeFermionD> Vrb_full;
  std::vector<RealD> Erb(eval_rb.begin(), eval_rb.begin() + kdrb);
  for (int i = 0; i < kdrb; ++i) {
    LatticeFermionD vf(FGrid);
    vf = Zero();
    setCheckerboard(vf, evec_rb[i]);
    Vrb_full.push_back(vf);
  }
  std::cout << GridLogMessage << "dumping " << kdrb << " RB-Schur^2 deflation vectors (embedded full 5D):" << std::endl;
  save_deflation_vectors(Vrb_full, Erb, ckpt_dir, "deflrb");

  std::cout << GridLogMessage << "svddump_rbcgne done" << std::endl;
  Grid_finalize();
  return 0;
}
