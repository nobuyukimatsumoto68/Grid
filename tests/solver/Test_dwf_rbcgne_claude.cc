/*************************************************************************************
  Test_dwf_rbcgne_claude.cc  -- (3) deflated RB-CGNE (reads the dump; NO Lanczos in-process).

  Reads the RB-Schur^2 low modes dumped by Test_dwf_svddump_rbcgne_claude.cc ("deflrb"), then runs
  RB-CGNE (SchurRedBlackDiagMooeeSolve) plain vs deflated (DeflatedGuesser). This is the known-good plain
  RB-CGNE path -- no eigensolver machinery co-resident. Source b is drawn from a DEDICATED fixed seed so it
  is byte-identical to the deflated-preconditioner binary (apples-to-apples). Shared: twolevel_common_claude.h.

  CLI: --config <NERSC> --ckpt_dir <dir>
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

  std::string ckpt_dir = "";
  if (GridCmdOptionExists(argv, argv + argc, "--ckpt_dir")) ckpt_dir = GridCmdOptionPayload(argv, argv + argc, "--ckpt_dir");
  if (ckpt_dir.empty()) {
    std::cout << GridLogError << "--ckpt_dir is required (where to read deflrb)" << std::endl;
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

  // shared solve source (dedicated fixed seed -> byte-identical to the deflprec binary's b)
  LatticeFermionD bsrc(FGrid);
  GridParallelRNG RNGsrc(FGrid);
  RNGsrc.SeedFixedIntegers({20, 21, 22, 23});
  gaussian(RNGsrc, bsrc);

  // read the RB-Schur^2 deflation modes (as odd-checkerboard FrbGrid fields) + eigenvalues
  std::vector<LatticeFermionD> Vrb;
  std::vector<RealD> Erb;
  int kdrb = read_deflation_vectors_odd(Vrb, Erb, FGrid, FrbGrid, ckpt_dir, "deflrb");
  std::cout << GridLogMessage << "RB-CGNE with deflation k=" << kdrb << " (lowest eval=" << (kdrb ? Erb[0] : 0.0) << ")" << std::endl;

  // (a) RB-CGNE plain
  ConjugateGradient<LatticeFermionD> CGp(1.0e-8, 60000, false);
  SchurRedBlackDiagMooeeSolve<LatticeFermionD> Sp(CGp);
  LatticeFermionD sp(FGrid);
  sp = Zero();
  Sp(D, bsrc, sp);
  std::cout << GridLogMessage << "  [RB-CGNE plain] iters=" << CGp.IterationsToComplete
            << "  D_W=" << (long)2 * Ls * CGp.IterationsToComplete << std::endl;

  // (b) RB-CGNE deflated -- OUTER-deflation scan: use the first w_o of the dumped modes, for each w_o in
  // --wo_list (default 16,24,32,40,48). Dump the largest (48) once; the smaller w_o are its prefixes.
  std::string wo_list = "16,24,32,40,48";
  if (GridCmdOptionExists(argv, argv + argc, "--wo_list")) wo_list = GridCmdOptionPayload(argv, argv + argc, "--wo_list");
  std::vector<int> wolist;
  std::stringstream wss(wo_list);
  std::string wtok;
  while (std::getline(wss, wtok, ',')) {
    int wv = std::stoi(wtok);
    if (wv > 0 && wv <= kdrb) wolist.push_back(wv);
  }
  for (size_t wi = 0; wi < wolist.size(); ++wi) {
    int wo = wolist[wi];
    std::vector<LatticeFermionD> Vwo(Vrb.begin(), Vrb.begin() + wo);
    std::vector<RealD> Ewo(Erb.begin(), Erb.begin() + wo);
    ConjugateGradient<LatticeFermionD> CGd(1.0e-8, 60000, false);
    SchurRedBlackDiagMooeeSolve<LatticeFermionD> Sd(CGd);
    DeflatedGuesser<LatticeFermionD> g_rb(Vwo, Ewo);
    LatticeFermionD sd(FGrid);
    sd = Zero();
    Sd(D, bsrc, sd, g_rb);
    std::cout << GridLogMessage << "  [RB-CGNE deflated(w_o=" << wo << ")] iters=" << CGd.IterationsToComplete
              << "  D_W=" << (long)2 * Ls * CGd.IterationsToComplete << std::endl;
  }

  std::cout << GridLogMessage << "rbcgne done" << std::endl;
  Grid_finalize();
  return 0;
}
