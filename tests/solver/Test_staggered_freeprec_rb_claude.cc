//
// Test_staggered_freeprec_rb_claude.cc
//
// Chunk 1 of the staggered Fourier + red-black preconditioner line
// (plan: staggered_fourier_rb_impl_plan_claude.md; direction:
// scripts_nm/fourier_rb_preconditioner_staggered_claude.md).
//
// Free-symbol validation on the FREE (unit-gauge) naive staggered operator, periodic BC:
//
//   Test A (full lattice): D^dag D = m^2 - D_hop^2 is FFT-diagonal with SCALAR symbol
//     m^2 + \sum_\mu \sin^2 p_\mu  (staggered-phase cross terms cancel identically),
//     so applying D^dag D to FFT^{-1} [ symbol^{-1} FFT \eta ] must recover \eta to
//     machine precision.
//
//   Test B (red-black): the even-block operator SchurStaggeredOperator::Mpc = m^2 - D_eo D_oe
//     is inverted by the PROJECTED full-lattice free inverse P_e F_norm P_e
//     (zero-pad even source to the full grid, FFT-diagonal apply, project even).
//     This is the bit-level check of the block-inverse identity S_c^{-1} = (M^{-1})_{cc}
//     (Y. Saad, "Iterative Methods for Sparse Linear Systems", Schur-complement section),
//     and shows NO sublattice FFT is needed (no Brillouin-zone folding to handle).
//
// Even-odd / Schur preconditioning: T. DeGrand, P. Rossi; Grid SchurStaggeredOperator.
// Staggered phases: N. Kawamoto, J. Smit; M. Golterman, J. Smit.
// Free-limit Fourier preconditioner lineage: FreeMobius5D_claude.h / FreeWilson_claude.h.
//
// Analysis output goes to stderr (repo convention); Grid logs go to stdout.
// Run (CPU is fine; once per mass):
//   ./Test_staggered_freeprec_rb_claude --grid 8.8.8.8 --mpi 1.1.1.1 --mass 0.5
//

#include <Grid/Grid.h>

using namespace Grid;

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplexD::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  Coordinate latt = GridDefaultLatt();

  GridParallelRNG RNG4(UGrid);
  std::vector<int> seeds({1, 2, 3, 4});
  RNG4.SeedFixedIntegers(seeds);

  typedef NaiveStaggeredFermionD::FermionField FermionField;

  // unit links = the FREE operator
  LatticeGaugeFieldD Umu(UGrid);
  SU<Nc>::ColdConfiguration(RNG4, Umu);

  FFT theFFT(UGrid);

  // momentum symbol denom(p) = m^2 + \sum_\mu \sin^2 p_\mu, p_\mu = 2 \pi k_\mu / L_\mu (periodic BC)
  LatticeComplexD one(UGrid);
  one = ComplexD(1.0, 0.0);
  LatticeComplexD sk2(UGrid);
  sk2 = Zero();
  LatticeComplexD kmu(UGrid);
  for (int mu = 0; mu < Nd; ++mu) {
    RealD TwoPiL = M_PI * 2.0 / latt[mu];
    LatticeCoordinate(kmu, mu);
    kmu = TwoPiL * kmu;
    sk2 = sk2 + sin(kmu) * sin(kmu);
  }

  // ONE mass per process: re-constructing a second NaiveStaggeredFermion in the same process
  // aborts with free(): invalid size at teardown of the first (seen 2026-09-10, build_mpi tree),
  // so the mass loop moved to the run script (one invocation per mass, --mass flag).
  RealD mass = 0.5;
  if (GridCmdOptionExists(argv, argv + argc, "--mass")) {
    std::string arg = GridCmdOptionPayload(argv, argv + argc, "--mass");
    mass = std::stod(arg);
  }
  {
    RealD c1 = 1.0;
    RealD u0 = 1.0;
    // NaiveStaggeredFermionD Ds(Umu, *UGrid, *UrbGrid, mass, c1, u0);
    // heap-allocate and intentionally LEAK: destructing the staggered operator (stencils /
    // doubled gauge fields) aborts with free(): invalid size even for a single construction
    // (2026-09-10 runs); standard Grid test mains never destruct it before exit, so this
    // teardown path is unexercised upstream. Leak at process end is harmless.
    NaiveStaggeredFermionD* Ds_p = new NaiveStaggeredFermionD(Umu, *UGrid, *UrbGrid, mass, c1, u0);
    NaiveStaggeredFermionD& Ds = *Ds_p;

    LatticeComplexD denom(UGrid);
    denom = sk2 + (mass * mass) * one;
    LatticeComplexD symbolinv = pow(denom, -1.0);

    FermionField src(UGrid);
    gaussian(RNG4, src);

    // ---- Test A: full-lattice free normal operator vs the scalar FFT symbol ----
    FermionField src_k(UGrid);
    theFFT.FFT_all_dim(src_k, src, FFT::forward);
    FermionField chi_k(UGrid);
    chi_k = symbolinv * src_k;
    FermionField chi(UGrid);
    theFFT.FFT_all_dim(chi, chi_k, FFT::backward);

    FermionField Mchi(UGrid);
    Ds.M(chi, Mchi);
    FermionField MdagMchi(UGrid);
    Ds.Mdag(Mchi, MdagMchi);

    FermionField diff(UGrid);
    diff = MdagMchi - src;
    RealD relA = norm2(diff) / norm2(src);
    std::cerr << "STAGFREE_FULL m= " << mass << " rel2= " << relA << std::endl;

    // ---- Test B: RB Schur operator Mpc vs the PROJECTED full-lattice inverse P_e F P_e ----
    FermionField src_e(UrbGrid);
    pickCheckerboard(Even, src_e, src);

    FermionField padded(UGrid);
    padded = Zero();
    setCheckerboard(padded, src_e);

    theFFT.FFT_all_dim(src_k, padded, FFT::forward);
    chi_k = symbolinv * src_k;
    theFFT.FFT_all_dim(chi, chi_k, FFT::backward);

    FermionField chi_e(UrbGrid);
    pickCheckerboard(Even, chi_e, chi);

    // SchurStaggeredOperator<NaiveStaggeredFermionD, FermionField> HermOpEO(Ds);
    // heap-allocate and LEAK, same reason as Ds above (holds an RB-grid member field).
    SchurStaggeredOperator<NaiveStaggeredFermionD, FermionField>* HermOpEO_p = new SchurStaggeredOperator<NaiveStaggeredFermionD, FermionField>(Ds);
    FermionField res_e(UrbGrid);
    res_e.Checkerboard() = Even;
    HermOpEO_p->Mpc(chi_e, res_e);

    FermionField diff_e(UrbGrid);
    diff_e = res_e - src_e;
    RealD relB = norm2(diff_e) / norm2(src_e);
    std::cerr << "STAGFREE_RB   m= " << mass << " rel2= " << relB << std::endl;

    GRID_ASSERT(relA < 1.0e-20);
    GRID_ASSERT(relB < 1.0e-20);
  }

  std::cerr << "STAGFREE_CHUNK1 PASSED m= " << mass << std::endl;

  Grid_finalize();
  return 0;
}
