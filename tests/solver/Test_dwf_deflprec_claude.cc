/*************************************************************************************
  Test_dwf_deflprec_claude.cc  -- (4) deflated preconditioner, M0-FGMRES (reads the dump; NO Lanczos).

  Reads the D5^H D5 right singular vectors dumped by Test_dwf_svddump_deflprec_claude.cc ("deflfull"),
  forms the left vectors u_i = D5 v_i/sigma_i, builds the Landau warm-start frame Omega + free-limit
  preconditioner M0 = Omega^dag F Omega, and runs the two-level solve of D5 x = b:
    (1) pre-solve the topological subspace, NO preconditioner:  x_lo = sum_i (u_i^dag b/sigma_i) v_i
    (2) M0-FGMRES on the deflated complement A_hat = P_U D5 P_V,  rhs P_U b
    (3) x = x_lo + P_V x_perp ; report iters + true rel.res of D5 x = b.
  Loops --svd_klist (default 3,10). Source b uses the SAME dedicated fixed seed as Test_dwf_rbcgne_claude
  (apples-to-apples). Design: scripts_nm/svd_deflation_topological_twolevel_claude.md.

  CLI: --config <NERSC> --ckpt_dir <dir> [--svd_klist 3,10] [--flow_nstep 873] [--restart 16] [--gmres_k 32]
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

  RealD flow_eps = 0.02;
  int flow_nstep = 873;   // s/t0=6 frame (production default)
  int restart = 16;       // FGMRES restart length m
  int gmres_k = 32;       // GMRES-DR deflation dim on the complement (0 => plain restarted GMRES)
  std::string svd_klist = "3,10";
  std::string ckpt_dir = "";
  if (GridCmdOptionExists(argv, argv + argc, "--flow_nstep")) flow_nstep = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--flow_nstep"));
  if (GridCmdOptionExists(argv, argv + argc, "--restart")) restart = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--restart"));
  if (GridCmdOptionExists(argv, argv + argc, "--gmres_k")) gmres_k = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--gmres_k"));
  if (GridCmdOptionExists(argv, argv + argc, "--svd_klist")) svd_klist = GridCmdOptionPayload(argv, argv + argc, "--svd_klist");
  if (GridCmdOptionExists(argv, argv + argc, "--ckpt_dir")) ckpt_dir = GridCmdOptionPayload(argv, argv + argc, "--ckpt_dir");
  if (ckpt_dir.empty()) {
    std::cout << GridLogError << "--ckpt_dir is required (where to read deflfull)" << std::endl;
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

  // shared solve source (dedicated fixed seed -> byte-identical to the rbcgne binary's b)
  LatticeFermionD bsrc(FGrid);
  GridParallelRNG RNGsrc(FGrid);
  RNGsrc.SeedFixedIntegers({20, 21, 22, 23});
  gaussian(RNGsrc, bsrc);

  // read the D5^H D5 right singular vectors v_i + eigenvalues (lambda_i = sigma_i^2)
  std::vector<LatticeFermionD> modes;
  std::vector<RealD> lam;
  int nkeep = read_deflation_vectors_full(modes, lam, FGrid, ckpt_dir, "deflfull");

  // Landau warm-start frame (flow s/t0=6 + Fourier-accelerated Landau gauge-fix) -> Omega -> M0 = Om^dag F Om
  std::cout << GridLogMessage << "==== Landau warm-start frame (flow " << flow_eps << "x" << flow_nstep << ") ====" << std::endl;
  LatticeGaugeFieldD Uflowed(UGrid);
  WilsonFlow<PeriodicGimplD> wf(flow_eps, flow_nstep);
  wf.smear(Uflowed, Umu);
  LatticeColourMatrixD OmL(UGrid);
  FourierAcceleratedGaugeFixer<PeriodicGimplD>::SteepestDescentGaugeFix(
      Uflowed, OmL, 0.1 / 16.0, 4000, 1.0e-12, 1.0e-12, true, -1, false);
  std::cout << GridLogMessage << "  Landau frame: flowed-fixed functional="
            << (1.0 - WilsonLoops<PeriodicGimplD>::linkTrace(Uflowed)) << std::endl;
  FreeMobius5DInverse<WilsonImplD> F(FGrid, Ls, M5, bb, cc, mm, boundary);
  FreeLimitPreconditioner<WilsonImplD> M0L(F, OmL, FGrid);

  // parse the k-list
  std::vector<int> klist;
  std::stringstream kss(svd_klist);
  std::string ktok;
  while (std::getline(kss, ktok, ',')) {
    int kv = std::stoi(ktok);
    if (kv > 0 && kv <= nkeep) klist.push_back(kv);
  }

  for (size_t ki = 0; ki < klist.size(); ++ki) {
    int kd = klist[ki];
    std::vector<LatticeFermionD> Vs(modes.begin(), modes.begin() + kd);
    std::vector<LatticeFermionD> Us;
    Us.reserve(kd);
    for (int i = 0; i < kd; ++i) {
      RealD sig = std::sqrt(lam[i] > 0.0 ? lam[i] : 0.0);
      LatticeFermionD ui(FGrid);
      D.M(Vs[i], ui);           // D5 v_i
      ui = ui * (1.0 / sig);    // u_i = D5 v_i / sigma_i (orthonormal since V orthonormal eigvecs of D5^H D5)
      Us.push_back(ui);
    }
    // (1) pre-solve the topological subspace, NO preconditioner: x_lo = sum_i (u_i^dag b/sigma_i) v_i
    LatticeFermionD xlo(FGrid);
    xlo = Zero();
    for (int i = 0; i < kd; ++i) {
      RealD sig = std::sqrt(lam[i] > 0.0 ? lam[i] : 0.0);
      ComplexD ci = innerProduct(Us[i], bsrc) / sig;
      axpy(xlo, ci, Vs[i], xlo);
    }
    // (2) M0-FGMRES on the deflated complement A_hat = P_U D5 P_V, rhs P_U b
    LatticeFermionD bperp(FGrid);
    bperp = bsrc;
    project_complement(bperp, Us);
    DeflatedD5Op Ahat(D, Us, Vs);
    RecyclingGeneralisedMinimalResidual<LatticeFermionD> RGMs(1.0e-8, 20000, M0L, restart, gmres_k, false, false);
    LatticeFermionD xperp(FGrid);
    xperp = Zero();
    RGMs(Ahat, bperp, xperp);
    project_complement(xperp, Vs);   // keep x_perp perp V (V-component undetermined by A_hat)
    // (3) x = x_lo + P_V x_perp ; true residual of the ORIGINAL system (correctness gate)
    LatticeFermionD xfull(FGrid);
    xfull = xlo + xperp;
    LatticeFermionD chk(FGrid);
    D.M(xfull, chk);
    chk = chk - bsrc;
    RealD relres = std::sqrt(norm2(chk) / norm2(bsrc));
    std::cout << GridLogMessage << "  [SVD-defl(k=" << kd << ") M0-FGMRES(" << restart << "," << gmres_k
              << ")] iters=" << RGMs.IterationCount << "  D_W=" << (long)Ls * RGMs.IterationCount
              << "  true rel.res=" << relres << std::endl;
  }

  std::cout << GridLogMessage << "deflprec done" << std::endl;
  Grid_finalize();
  return 0;
}
