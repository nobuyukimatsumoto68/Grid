/*************************************************************************************
  Test_dwf_freeprec_rb_solve_claude.cc -- DWF Fourier + red-black preconditioner, INTERACTING solve.
  Plan: dwf_fourier_rb_impl_plan_claude.md.

  DEDICATED framed Fourier-RB solve (Nobu 2026-09-10): the Omega=1 control leg and the RB-CGNE baseline
  are removed -- RB-CGNE is measured separately (Test_dwf_rbcgne_claude.cc, same source seed), and the
  Omega=1 control is known to lose. This binary does ONE thing:

    Framed Fourier-preconditioned RB-FGMRES:  solve S_o x_o = b_o (odd-checkerboard Schur, non-Hermitian)
    with RIGHT preconditioner  M0^rb = P_o (Omega^dag F Omega) P_o,
    Omega = flowed-Landau frame (site-local, commutes with the checkerboard projector P_o),
    F = FreeMobius5DInverse. GMRES-DR (RecyclingGeneralisedMinimalResidual), restart m, deflation k.

  KEY IDENTITY (validated cold in Test_dwf_freeprec_rb_claude.cc, D1a): the free RB-Schur inverse is the
  odd block of the free full inverse, S_{o,free}^{-1} = P_o F P_o. In the free limit the preconditioned
  S_o is exactly 1; the frame clusters the interacting RB-Schur spectrum near 1.

  Source: same fixed seed {20,21,22,23} as Test_dwf_rbcgne_claude.cc -> the RB-CGNE baseline measured
  there is byte-identical (apples-to-apples). Metric: D_W applies = Ls * GMRES iterations (RB-CGNE there
  is 2*Ls*iters). Reports the TRUE full residual ||D5 x - b||/||b|| as the correctness gate.

  GPU build (reads a config -> NerscIO/BinaryIO -> needs the build_merged install tree).

  Even-odd / Schur preconditioning: T. DeGrand, P. Rossi; Grid NonHermitianSchurRedBlackDiagMooeeSolve.
  FGMRES / GMRES-DR: Y. Saad; R. B. Morgan; RecyclingGeneralisedMinimalResidual_claude.h.
  Free Mobius inverse F: FreeMobius5D_claude.h (Mobius kernel R. Brower, H. Neff, K. Orginos, 1206.5214).

  CLI: --config <NERSC> (required)  [--restart <m>=16]  [--gmres_k <k>=32]  [--maxit <N>=20000]
       [--tol <t>=1e-8]  [--flow_eps <e>=0.02]  [--flow_nstep <N>=873]
*************************************************************************************/
#include <Grid/Grid.h>
#include <Grid/qcd/utils/FreeMobius5D_claude.h>
#include <Grid/algorithms/iterative/RecyclingGeneralisedMinimalResidual_claude.h>

using namespace Grid;

// Shamir-Mobius DWF parameters (match twolevel_common_claude.h TL_*).
static const double RB_M5 = 1.8;
static const int    RB_Ls = 8;
static const double RB_bb = 1.5;
static const double RB_cc = 0.5;
static const double RB_mm = 0.1;

// M0^rb = P_o Mfull P_o : lift an odd field to the full 5D grid (even = 0), apply the full-lattice
// framed preconditioner Mfull = M0 = Omega^dag F Omega, project back onto odd. The site-local frame
// commutes with the checkerboard projection, so P_o M0 P_o is well defined.
class RbLiftProject : public LinearFunction<LatticeFermionD> {
  LinearFunction<LatticeFermionD>& Mfull;
  GridCartesian* FGrid;
  LatticeFermionD full;
  LatticeFermionD Mout;
public:
  RbLiftProject(LinearFunction<LatticeFermionD>& Mfull_, GridCartesian* FGrid_)
    : Mfull(Mfull_), FGrid(FGrid_), full(FGrid_), Mout(FGrid_) {}
  void operator()(const LatticeFermionD& in_o, LatticeFermionD& out_o) {
    full = Zero();
    setCheckerboard(full, in_o);
    Mfull(full, Mout);
    out_o.Checkerboard() = Odd;
    pickCheckerboard(Odd, out_o, Mout);
  }
};

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  const double M5 = RB_M5;
  const int Ls = RB_Ls;
  const double bb = RB_bb;
  const double cc = RB_cc;
  const double mm = RB_mm;
  std::vector<Complex> boundary = {1, 1, 1, -1};

  int restart = 16;
  if (GridCmdOptionExists(argv, argv + argc, "--restart"))
    restart = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--restart"));
  // GMRES-DR deflation dim k: adaptively deflates the isolated near-null cluster of B = S_o M0 (the
  // topological stragglers that STALL plain restarted GMRES). Default 32 = the latest twolevel deflprec
  // value (RecyclingGeneralisedMinimalResidual(...,16,32,...)); k=0 is plain restarted GMRES and stalls.
  int gmres_k = 32;
  if (GridCmdOptionExists(argv, argv + argc, "--gmres_k"))
    gmres_k = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--gmres_k"));
  int maxit = 20000;
  if (GridCmdOptionExists(argv, argv + argc, "--maxit"))
    maxit = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--maxit"));
  double tol = 1.0e-8;
  if (GridCmdOptionExists(argv, argv + argc, "--tol"))
    tol = std::stod(GridCmdOptionPayload(argv, argv + argc, "--tol"));
  double flow_eps = 0.02;
  if (GridCmdOptionExists(argv, argv + argc, "--flow_eps"))
    flow_eps = std::stod(GridCmdOptionPayload(argv, argv + argc, "--flow_eps"));
  int flow_nstep = 873;   // s/t0 = 6 at eps 0.02 (deflprec Landau warm-start convention)
  if (GridCmdOptionExists(argv, argv + argc, "--flow_nstep"))
    flow_nstep = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--flow_nstep"));

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
    std::cout << GridLogMessage << "loaded " << cfg << "  plaq="
              << WilsonLoops<PeriodicGimplD>::avgPlaquette(Umu) << std::endl;
  } else {
    std::cout << GridLogError << "--config is required" << std::endl;
    Grid_finalize();
    return 1;
  }

  MobiusFermionD::ImplParams Params(boundary);
  // heap-allocate, never destruct (staggered chunk-1 lesson: operator destructors before program end
  // can corrupt the heap).
  MobiusFermionD* Dp = new MobiusFermionD(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mm, M5, bb, cc, Params);
  MobiusFermionD& D = *Dp;
  FreeMobius5DInverse<WilsonImplD>* Fp = new FreeMobius5DInverse<WilsonImplD>(FGrid, Ls, M5, bb, cc, mm, boundary);

  // shared solve source (dedicated fixed seed -> byte-identical to Test_dwf_rbcgne_claude.cc)
  LatticeFermionD bsrc(FGrid);
  GridParallelRNG RNGsrc(FGrid);
  RNGsrc.SeedFixedIntegers({20, 21, 22, 23});
  gaussian(RNGsrc, bsrc);
  double nb = std::sqrt(norm2(bsrc));

  LatticeFermionD Dx(FGrid);
  LatticeFermionD r(FGrid);

  // ---- Landau warm-start frame Omega -> framed free-limit preconditioner M0 = Omega^dag F Omega ----
  std::cout << GridLogMessage << "==== Landau warm-start frame (flow " << flow_eps << "x" << flow_nstep
            << ") ====" << std::endl;
  LatticeGaugeFieldD Uflowed(UGrid);
  WilsonFlow<PeriodicGimplD> wf(flow_eps, flow_nstep);
  wf.smear(Uflowed, Umu);
  LatticeColourMatrixD* OmL_p = new LatticeColourMatrixD(UGrid);
  FourierAcceleratedGaugeFixer<PeriodicGimplD>::SteepestDescentGaugeFix(
      Uflowed, *OmL_p, 0.1 / 16.0, 4000, 1.0e-12, 1.0e-12, true, -1, false);
  std::cout << GridLogMessage << "  Landau frame: flowed-fixed functional="
            << (1.0 - WilsonLoops<PeriodicGimplD>::linkTrace(Uflowed)) << std::endl;
  FreeLimitPreconditioner<WilsonImplD>* M0L_p = new FreeLimitPreconditioner<WilsonImplD>(*Fp, *OmL_p, FGrid);
  RbLiftProject* Mrb_p = new RbLiftProject(*M0L_p, FGrid);

  // ---- framed Fourier-preconditioned RB-FGMRES: M0^rb = P_o (Omega^dag F Omega) P_o ----
  RecyclingGeneralisedMinimalResidual<LatticeFermionD> RGf(tol, maxit, *Mrb_p, restart, gmres_k, false, false);
  NonHermitianSchurRedBlackDiagMooeeSolve<LatticeFermionD> Sc(RGf);
  LatticeFermionD xc(FGrid);
  xc = Zero();
  Sc(D, bsrc, xc);
  long dw = (long)Ls * RGf.IterationCount;
  D.M(xc, Dx);
  r = Dx - bsrc;
  std::cout << GridLogMessage << "  [RB-FGMRES framed M0=P(Om^d F Om)P  m=" << restart << " k=" << gmres_k
            << "]  iters=" << RGf.IterationCount << "  D_W=" << dw
            << "  ||D5 x - b||/||b||=" << std::sqrt(norm2(r)) / nb << std::endl;
  std::cout << GridLogMessage << "  (compare D_W to plain RB-CGNE from Test_dwf_rbcgne_claude.cc, same source)"
            << std::endl;

  std::cout << GridLogMessage << "dwf_freeprec_rb_solve done" << std::endl;

  Grid_finalize();
  return 0;
}
