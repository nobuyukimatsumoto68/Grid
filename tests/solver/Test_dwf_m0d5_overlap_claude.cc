/*************************************************************************************
  Test_dwf_m0d5_overlap_claude.cc -- low modes of the PRECONDITIONED operator M0 D5 by shift-invert IRA,
  and their OVERLAP with the Hermitian |D5|^2 (D5^dag D5) low modes.

  M0 D5 is non-Hermitian (M0 = Omega^dag F Omega framed free-limit preconditioner). Its near-zero
  eigenvalues are the residual difficulty (the ~Re=0.3 stragglers). Because M0 D5 ~ 1, the shift-invert
  inner solve (M0 D5)^{-1} is CHEAP (plain GMRES on the well-conditioned M0 D5, ~O(200) iters) -- unlike
  shift-inverting the bare D5. IRA(largest-modulus) on (M0 D5)^{-1} returns theta = 1/lambda, i.e. the
  smallest-|lambda| eigenpairs (lambda_i, w_i) of M0 D5, WITH eigenvectors.

  Overlap analysis: v_j = lowest D5^dag D5 modes (= right singular vectors of D5; the "deflfull" dump,
  read from --ckpt_dir). Report per M0-D5 eigenvector w_i: its eigenvalue lambda_i, and the projection
  ||P_V w_i||^2 = sum_j |<w_i|v_j>|^2 onto the D5 low-mode subspace, plus the top per-mode overlaps.
  Tests the claim that the near-zero eigenvectors of M0 D5 ~ the small right singular vectors of D5.

  IRA: Sorensen 1992 (Grid/algorithms/iterative/ImplicitlyRestartedArnoldi_claude.h). Shift-invert wrapper
  mirrors Test_dwf_spectrum_transform_claude.cc.

  CLI: --config <NERSC> --ckpt_dir <dir with deflfull> [--nstop 24] [--flow_nstep 873] [--sigma 0]
       [--inner_tol 1e-6] [--inner_restart 200] [--inner_maxit 4000] [--nk 36] [--nm 64]
*************************************************************************************/
#include "twolevel_common_claude.h"
#include <Grid/algorithms/iterative/ImplicitlyRestartedArnoldi_claude.h>

using namespace Grid;

// B = M0 D5 (non-Hermitian) as a LinearOperatorBase: Op(in,out) = M0( D5 in ). Only Op is used (GMRES).
class M0D5Op : public LinearOperatorBase<LatticeFermionD> {
  MobiusFermionD& D;
  LinearFunction<LatticeFermionD>& M0;
public:
  M0D5Op(MobiusFermionD& D_, LinearFunction<LatticeFermionD>& M0_) : D(D_), M0(M0_) {}
  void Op(const LatticeFermionD& in, LatticeFermionD& out) {
    LatticeFermionD t(in.Grid());
    D.M(in, t);        // D5 in
    M0(t, out);        // M0 (D5 in)
  }
  void AdjOp(const LatticeFermionD& in, LatticeFermionD& out) { GRID_ASSERT(0); }
  void OpDiag(const LatticeFermionD& in, LatticeFermionD& out) { GRID_ASSERT(0); }
  void OpDir(const LatticeFermionD& in, LatticeFermionD& out, int dir, int disp) { GRID_ASSERT(0); }
  void OpDirAll(const LatticeFermionD& in, std::vector<LatticeFermionD>& out) { GRID_ASSERT(0); }
  void HermOpAndNorm(const LatticeFermionD& in, LatticeFermionD& out, RealD& n1, RealD& n2) { GRID_ASSERT(0); }
  void HermOp(const LatticeFermionD& in, LatticeFermionD& out) { GRID_ASSERT(0); }
};

// (B - sigma)^{-1} via plain GMRES (B ~ 1 -> well conditioned) -- LinearFunction for IRA.
class ShiftInvertB : public LinearFunction<LatticeFermionD> {
  LinearOperatorBase<LatticeFermionD>& B;
  RealD tol;
  int maxit;
  int restart;
public:
  ShiftInvertB(LinearOperatorBase<LatticeFermionD>& B_, RealD tol_, int maxit_, int restart_)
    : B(B_), tol(tol_), maxit(maxit_), restart(restart_) {}
  void operator()(const LatticeFermionD& in, LatticeFermionD& out) {
    out = Zero();
    GeneralisedMinimalResidual<LatticeFermionD> GMRES(tol, maxit, restart, false);
    GMRES(B, in, out);   // sigma=0: solve B out = in  (shift folded into B if sigma!=0 -- see main)
  }
};

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  const double M5 = TL_M5;
  const int Ls = TL_Ls;
  const double bb = TL_bb;
  const double cc = TL_cc;
  const double mm = TL_mm;
  std::vector<Complex> boundary = {1, 1, 1, -1};

  int Nstop = 24;
  int Nk = 36;
  int Nm = 64;
  int flow_nstep = 873;
  RealD flow_eps = 0.02;
  RealD sigma = 0.0;
  RealD inner_tol = 1.0e-6;
  int inner_restart = 200;
  int inner_maxit = 4000;
  RealD eresid = 1.0e-8;
  int MaxIter = 200;
  std::string ckpt_dir = "";
  if (GridCmdOptionExists(argv, argv + argc, "--nstop")) Nstop = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--nstop"));
  if (GridCmdOptionExists(argv, argv + argc, "--nk")) Nk = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--nk"));
  if (GridCmdOptionExists(argv, argv + argc, "--nm")) Nm = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--nm"));
  if (GridCmdOptionExists(argv, argv + argc, "--flow_nstep")) flow_nstep = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--flow_nstep"));
  if (GridCmdOptionExists(argv, argv + argc, "--sigma")) sigma = std::stod(GridCmdOptionPayload(argv, argv + argc, "--sigma"));
  if (GridCmdOptionExists(argv, argv + argc, "--inner_tol")) inner_tol = std::stod(GridCmdOptionPayload(argv, argv + argc, "--inner_tol"));
  if (GridCmdOptionExists(argv, argv + argc, "--inner_restart")) inner_restart = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--inner_restart"));
  if (GridCmdOptionExists(argv, argv + argc, "--inner_maxit")) inner_maxit = std::stoi(GridCmdOptionPayload(argv, argv + argc, "--inner_maxit"));
  if (GridCmdOptionExists(argv, argv + argc, "--ckpt_dir")) ckpt_dir = GridCmdOptionPayload(argv, argv + argc, "--ckpt_dir");

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
    std::cout << GridLogError << "--config required" << std::endl;
    Grid_finalize();
    return 1;
  }

  MobiusFermionD::ImplParams Params(boundary);
  MobiusFermionD D(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mm, M5, bb, cc, Params);

  // Landau warm-start frame -> M0 = Omega^dag F Omega
  std::cout << GridLogMessage << "==== Landau frame (flow " << flow_eps << "x" << flow_nstep << ") ====" << std::endl;
  LatticeGaugeFieldD Uflowed(UGrid);
  WilsonFlow<PeriodicGimplD> wf(flow_eps, flow_nstep);
  wf.smear(Uflowed, Umu);
  LatticeColourMatrixD OmL(UGrid);
  FourierAcceleratedGaugeFixer<PeriodicGimplD>::SteepestDescentGaugeFix(
      Uflowed, OmL, 0.1 / 16.0, 4000, 1.0e-12, 1.0e-12, true, -1, false);
  std::cout << GridLogMessage << "  Landau functional=" << (1.0 - WilsonLoops<PeriodicGimplD>::linkTrace(Uflowed)) << std::endl;
  FreeMobius5DInverse<WilsonImplD> F(FGrid, Ls, M5, bb, cc, mm, boundary);
  FreeLimitPreconditioner<WilsonImplD> M0(F, OmL, FGrid);

  // shift-invert IRA on B = M0 D5 (sigma=0 default). B ~ 1 -> inner GMRES is cheap.
  M0D5Op B(D, M0);
  ShiftInvertB SIop(B, inner_tol, inner_maxit, inner_restart);
  std::cout << GridLogMessage << "==== shift-invert IRA on M0 D5 (sigma=" << sigma << ", Nstop=" << Nstop
            << " Nk=" << Nk << " Nm=" << Nm << ") ====" << std::endl;
  ImplicitlyRestartedArnoldi<LatticeFermionD> IRA(SIop, Nstop, Nk, Nm, eresid, MaxIter, IRAlargestModulus);
  std::vector<ComplexD> theta(Nm);
  std::vector<LatticeFermionD> W(Nm, LatticeFermionD(FGrid));
  LatticeFermionD src(FGrid);
  gaussian(RNG5, src);
  int Nconv = 0;
  IRA.calc(theta, W, src, Nconv);
  int nlow = (Nconv < Nstop) ? Nconv : Nstop;
  std::cout << GridLogMessage << "  IRA Nconv=" << Nconv << "  keeping " << nlow << " low modes of M0 D5" << std::endl;

  // convert theta -> lambda = sigma + 1/theta ; normalise eigenvectors
  std::vector<ComplexD> lam(nlow);
  for (int i = 0; i < nlow; ++i) {
    lam[i] = ComplexD(sigma, 0.0) + ComplexD(1.0, 0.0) / theta[i];
    RealD n = std::sqrt(norm2(W[i]));
    W[i] = W[i] * (1.0 / n);
  }

  // read the Hermitian |D5|^2 low modes (deflfull = right singular vectors of D5), lowest nlow
  std::vector<LatticeFermionD> V;
  std::vector<RealD> lamH;
  int nH = 0;
  if (!ckpt_dir.empty()) {
    nH = read_deflation_vectors_full(V, lamH, FGrid, ckpt_dir, "deflfull");
  } else {
    std::cout << GridLogWarning << "no --ckpt_dir: skipping overlap (need deflfull)" << std::endl;
  }
  int kV = (nH < nlow) ? nH : nlow;   // use the lowest kV D5 modes for the projection

  // overlap analysis
  std::cout << GridLogMessage << "==== overlap: M0 D5 low eigenvectors vs D5^dag D5 low modes (kV=" << kV << ") ====" << std::endl;
  std::cout << GridLogMessage << "  i   Re(lambda)      Im(lambda)     |lambda|      ||P_V w_i||^2   top-overlap(j,|<w|v>|)" << std::endl;
  for (int i = 0; i < nlow; ++i) {
    RealD proj = 0.0;
    int jmax = -1;
    RealD omax = 0.0;
    for (int j = 0; j < kV; ++j) {
      ComplexD o = innerProduct(W[i], V[j]);   // <w_i | v_j>
      RealD a = std::sqrt(o.real() * o.real() + o.imag() * o.imag());
      proj += a * a;
      if (a > omax) {
        omax = a;
        jmax = j;
      }
    }
    RealD absl = std::hypot(lam[i].real(), lam[i].imag());
    std::cout << GridLogMessage << "  " << i << "   " << lam[i].real() << "   " << lam[i].imag()
              << "   " << absl << "   " << proj << "   (" << jmax << ", " << omax << ")" << std::endl;
  }

  std::cout << GridLogMessage << "m0d5_overlap done" << std::endl;
  Grid_finalize();
  return 0;
}
