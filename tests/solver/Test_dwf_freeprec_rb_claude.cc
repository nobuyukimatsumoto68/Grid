/*************************************************************************************
  Test_dwf_freeprec_rb_claude.cc -- DWF Fourier + red-black preconditioner.

  Combines the free-limit Fourier preconditioner F = FreeMobius5DInverse with red-black (even-odd)
  preconditioning. Plan: dwf_fourier_rb_impl_plan_claude.md. Direction:
  scripts_nm/fourier_rb_preconditioner_staggered_claude.md ("Map back to Wilson/DWF").

  KEY IDENTITY (Y. Saad, "Iterative Methods for Sparse Linear Systems", Schur-complement section;
  validated bit-level on staggered in Test_staggered_freeprec_rb_claude.cc):
  Grid's RB solve runs the inner Krylov on the odd-checkerboard Schur complement
     S_o = M_oo - M_oe M_ee^{-1} M_eo   (= NonHermitianSchurDiagMooeeOperator::Mpc),
  and the FREE Schur inverse is the odd block of the free full inverse F = D_free^{-1}:
     S_{o,free}^{-1} = (D_free^{-1})_oo = P_o F P_o .
  So the combined preconditioner is the EXISTING F, zero-padded on even and projected back to odd:
     M0^rb = P_o F P_o    (frame Omega, site-local, commutes with P_o; added in chunk D2).

  CHUNK D1a (this file): unit-gauge cold gate
     || S_{o,free} (P_o F P_o) b_o - b_o ||/||b_o|| ~ machine eps      (both apply orders),
  the DWF analogue of the staggered chunk-1 RB check, now with nontrivial M_ee (5D s-hopping),
  spin, and the AP-time twist.

  Deliberately self-contained: includes ONLY Grid.h + FreeMobius5D_claude.h, uses unit gauge, and
  does NO config / deflation-vector I/O. This avoids instantiating BinaryIO's AggregateExchange
  templates, whose AllToAllV / aggregateTargetBytes symbols are ABSENT from the current build_mpi
  libGrid.a (source BinaryIO.h is ahead of that install tree). The interacting FGMRES leg (D1b, needs
  NerscIO to read ckpoint_lat.640) is a separate 16^4 GPU run built against the current GPU install.

  Even-odd / Schur preconditioning: T. DeGrand, P. Rossi; Grid NonHermitianSchurDiagMooeeOperator.
  Free Mobius inverse F: FreeMobius5D_claude.h (Mobius kernel R. Brower, H. Neff, K. Orginos,
  arXiv:1206.5214).

  Analysis output goes to stderr (repo convention); Grid logs go to stdout.
  Run (CPU cold gate):  ./Test_dwf_freeprec_rb_claude --grid 8.8.8.8 --mpi 1.1.1.1   (Ls=8 set in code)
*************************************************************************************/
#include <Grid/Grid.h>
#include <Grid/qcd/utils/FreeMobius5D_claude.h>

using namespace Grid;

// Shamir-Mobius DWF parameters (match twolevel_common_claude.h TL_*).
static const double RB_M5 = 1.8;
static const int    RB_Ls = 8;
static const double RB_bb = 1.5;
static const double RB_cc = 0.5;
static const double RB_mm = 0.1;

// M0^rb = P_o F P_o : lift an odd-checkerboard field to the full 5D grid (even = 0), apply the free
// full inverse F, then project back onto the odd checkerboard. Site-local frame Omega (chunk D2) would
// wrap F here and commute with the projection.
class FreeMobiusRbInverse : public LinearFunction<LatticeFermionD> {
  FreeMobius5DInverse<WilsonImplD>& F;
  GridCartesian* FGrid;
  LatticeFermionD full;
  LatticeFermionD Ffull;
public:
  FreeMobiusRbInverse(FreeMobius5DInverse<WilsonImplD>& F_, GridCartesian* FGrid_)
    : F(F_), FGrid(FGrid_), full(FGrid_), Ffull(FGrid_) {}
  void operator()(const LatticeFermionD& in_o, LatticeFermionD& out_o) {
    full = Zero();
    setCheckerboard(full, in_o);   // even stays 0
    F(full, Ffull);
    out_o.Checkerboard() = Odd;
    pickCheckerboard(Odd, out_o, Ffull);
  }
};

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  const double M5 = RB_M5;
  const int Ls = RB_Ls;
  const double bb = RB_bb;
  const double cc = RB_cc;
  const double mm = RB_mm;
  std::vector<Complex> boundary = {1, 1, 1, -1};   // anti-periodic time

  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(
      GridDefaultLatt(), GridDefaultSimd(Nd, vComplexD::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian* FGrid = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
  GridRedBlackCartesian* FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);

  std::cout << GridLogMessage << "unit gauge, free-limit RB cold gate" << std::endl;
  LatticeGaugeFieldD Umu(UGrid);
  SU<Nc>::ColdConfiguration(Umu);

  MobiusFermionD::ImplParams Params(boundary);
  // heap-allocate, never destruct (staggered chunk-1 lesson: Grid fermion-operator destructors before
  // program end can corrupt the heap; standard Grid test mains keep them alive to exit).
  MobiusFermionD* Dp = new MobiusFermionD(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mm, M5, bb, cc, Params);
  MobiusFermionD& D = *Dp;

  // free full inverse F (unit-gauge free Mobius operator, FFT-diagonal), AP-time BC baked in.
  FreeMobius5DInverse<WilsonImplD>* Fp = new FreeMobius5DInverse<WilsonImplD>(FGrid, Ls, M5, bb, cc, mm, boundary);
  FreeMobius5DInverse<WilsonImplD>& F = *Fp;

  NonHermitianSchurDiagMooeeOperator<MobiusFermionD, LatticeFermionD>* Sop_p =
      new NonHermitianSchurDiagMooeeOperator<MobiusFermionD, LatticeFermionD>(D);
  NonHermitianSchurDiagMooeeOperator<MobiusFermionD, LatticeFermionD>& Sop = *Sop_p;

  FreeMobiusRbInverse* Mrb_p = new FreeMobiusRbInverse(F, FGrid);
  FreeMobiusRbInverse& Mrb = *Mrb_p;

  GridParallelRNG RNG5(FGrid);
  RNG5.SeedFixedIntegers(std::vector<int>({1, 2, 3, 4}));
  LatticeFermionD vfull(FGrid);
  gaussian(RNG5, vfull);

  LatticeFermionD bo(FrbGrid);
  bo.Checkerboard() = Odd;
  pickCheckerboard(Odd, bo, vfull);

  // order 1:  S_o (P_o F P_o) b_o == b_o
  LatticeFermionD chi_o(FrbGrid);
  Mrb(bo, chi_o);
  LatticeFermionD res_o(FrbGrid);
  res_o.Checkerboard() = Odd;
  Sop.Mpc(chi_o, res_o);
  double e1 = std::sqrt(norm2(res_o - bo) / norm2(bo));

  // order 2:  (P_o F P_o) S_o b_o == b_o
  LatticeFermionD sb_o(FrbGrid);
  sb_o.Checkerboard() = Odd;
  Sop.Mpc(bo, sb_o);
  LatticeFermionD chi2_o(FrbGrid);
  Mrb(sb_o, chi2_o);
  double e2 = std::sqrt(norm2(chi2_o - bo) / norm2(bo));

  std::cerr << "DWFFREE_RB_GATE  ||S(PFP)b - b||/||b|| = " << e1
            << "   ||(PFP)S b - b||/||b|| = " << e2 << std::endl;

  // F's default apply is a SINGLE-PRECISION FFT (the fp64 block-inverse inner gate is ~1e-16, but the
  // FFT round-trip floors the identity at ~1e-7). That is by design: F is a PRECONDITIONER, and the
  // RGMRES (twolevel_common) is explicitly flexible so an fp32 M0 still drives the true residual to
  // 1e-8. So the free-limit RB identity holding to single precision (~1e-7) is a PASS -- same
  // convention as the full-lattice freeprec cold gate. (fp64 F builds would tighten this.)
  double gtol = 1.0e-5;
  bool pass = (e1 < gtol) && (e2 < gtol);
  std::cerr << "DWFFREE_RB_GATE  " << (pass ? "PASS" : "FAIL") << " (tol " << gtol << ")" << std::endl;
  GRID_ASSERT(pass);

  std::cerr << "DWFFREE_RB_CHUNK_D1a PASSED" << std::endl;

  Grid_finalize();
  return 0;
}
