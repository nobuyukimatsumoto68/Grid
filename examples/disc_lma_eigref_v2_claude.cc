#include <getopt.h>   // getopt_long for --config/--mass/--eigref/--free
#include <cstdlib>    // std::getenv / std::atoi / std::atof
#include <cmath>      // std::isfinite (InverseHermOp NaN guard), std::sqrt
#include <fstream>    // write the reference spectrum file
#include <Grid/Grid.h>

// disc LMA -- v2 REFERENCE spectrum pre-calculation (separate from the Cheby bench).
// On ONE gauge config (FREE/cold by default, or a NERSC config) it eigensolves the
// low modes of the EO-Schur squared Mobius operator  H = M_pc^dag M_pc  (eigenvalues
// lambda_i = sigma_i^2) by ROBUST shift-invert IRL, and a PowerMethod estimate of
// lambda_max, then WRITES the spectral landscape to a text file. The Chebyshev bench
// (disc_lma_bench_v2_claude.cc) READS this file to set its filter window
//   lo = CHEB_LO_FAC * lambda_ref[Nstop-1]   (the cut, highest WANTED mode),
//   hi = CHEB_HI_FAC * lambda_max            (top of the bulk to suppress),
// which is the note's prerequisite -- Chebyshev is usable only "when the rough
// landscape of the entire spectrum is known in advance" (Matsumoto, lanczos.pdf Sec.3).
//
// This is the EXPENSIVE one-time pre-calc: run it once per ensemble (or on the free
// testbed) and reuse the .dat for all Cheby tuning runs.
//
// Algorithm sources:
//   - shift-invert Lanczos eigensolve: as in disc_mrhs_defl_bench_claude.cc.
//   - IRL: Matsumoto "Large-scale eigenvalue problem" (lanczos.pdf); Saad,
//     "Numerical Methods for Large Eigenvalue Problems"; Grid ImplicitlyRestartedLanczos.
//
// Claude never builds/submits/rm's -- user runs the build script and any job.

using namespace std;
using namespace Grid;

// ---- small env helpers (tunables overridable at run time, no rebuild) --------
static int    env_int   (const char* k, int    d){ const char* e=std::getenv(k); return e? std::atoi(e):d; }
static double env_double(const char* k, double d){ const char* e=std::getenv(k); return e? std::atof(e):d; }

// ---- CLI: config path / mass / eigref output / free-theory flag --------------
static void ParseArgs(int argc, char** argv, std::string& cfg, double& mass,
                      std::string& eigref, int& freeflag, int& hotflag)
{
  // No leading '+': let getopt PERMUTE so our flags are found regardless of position
  // (Grid's own --grid/--mpi are unknown here -> '?' -> ignored). Call Grid_init FIRST
  // (it consumes its args) so this permutation does not disturb --grid value pairing.
  const char* const short_opts = ":c:m:e:fH";
  const option long_opts[] = {
    {"config", required_argument, nullptr, 'c'},
    {"mass",   required_argument, nullptr, 'm'},
    {"eigref", required_argument, nullptr, 'e'},
    {"free",   no_argument,       nullptr, 'f'},
    {"hot",    no_argument,       nullptr, 'H'},
    {nullptr,  no_argument,       nullptr,  0}
  };
  opterr = 0;
  int idx, opt;
  while((opt = getopt_long(argc, argv, short_opts, long_opts, &idx)) != -1){
    switch(opt){
      case 'c': cfg    = optarg;            break;
      case 'm': mass   = std::stod(optarg); break;
      case 'e': eigref = optarg;            break;
      case 'f': freeflag = 1;               break;
      case 'H': hotflag  = 1;               break;
      default: break;
    }
  }
}

// ---- shift-invert operator: out = H^{-1} in via CG (same as the defl bench) ---
template<class FieldF>
class InverseHermOp : public LinearFunction<FieldF>
{
  LinearOperatorBase<FieldF> &_H;
  RealD _tol;
  int   _maxit;
public:
  using LinearFunction<FieldF>::operator();
  InverseHermOp(LinearOperatorBase<FieldF> &H, RealD tol, int maxit)
    : _H(H), _tol(tol), _maxit(maxit) {}
  void operator()(const FieldF &in, FieldF &out)
  {
    RealD n = norm2(in);
    if(!std::isfinite(n) || n == 0.0){
      out = Zero();
      std::cout << GridLogMessage << "# InverseHermOp: non-finite/zero source (norm2="
                << n << "), skipping CG" << std::endl;
      return;
    }
    ConjugateGradient<FieldF> CG(_tol, _maxit, false);
    out = Zero();
    CG(_H, in, out);
  }
};


int main(int argc, char** argv)
{
  Grid_init(&argc, &argv);   // FIRST: consume Grid's --grid/--mpi/... args

  std::string config = "";
  std::string eigref = "";
  double mass = 0.01;
  int    freeflag = 0;
  int    hotflag  = 0;
  ParseArgs(argc, argv, config, mass, eigref, freeflag, hotflag);

  // ---- fixed Mobius parameters (match disc_multipleGamma_binary_claude) ------
  const int Ls = 16;
  RealD M5 = 1.5, b = 1.5, c = 0.5;
  std::vector<Complex> boundary = {1,1,1,-1};
  typedef MobiusFermionD FermionAction;
  typedef MobiusFermionF FermionActionF;
  FermionAction::ImplParams Params(boundary);

  // ---- Lanczos knobs (same defaults as the defl bench) -----------------------
  const int    Nstop  = env_int   ("NSTOP",  50);
  const int    Nk     = env_int   ("NK",     60);
  const int    Nm     = env_int   ("NM",     120);
  const double eresid = env_double("ERESID", 1.0e-6);
  const int    maxit  = env_int   ("MAXITER",2000);
  const double inv_tol   = env_double("INV_TOL",   1.0e-5);
  const int    inv_maxit = env_int   ("INV_MAXIT", 50000);
  // EIG_PREC: 1 = single eigensolve (default, fast), 2 = double (removes the single
  // inner-CG true-resid floor). The reference is the ROBUST shift-invert -- the whole
  // point is a trustworthy landscape for the Cheby window.
  const int    eig_prec  = env_int   ("EIG_PREC",  1);

  if(eigref.empty()){
    Coordinate latt = GridDefaultLatt();
    std::string gtag = std::to_string(latt[0]);
    for(int d=1; d<(int)latt.size(); d++) gtag += "." + std::to_string(latt[d]);
    eigref = "eigref_" + gtag + "_m" + std::to_string(mass) + ".h5";
  }

  // ---- double + single grids -------------------------------------------------
  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(),
                                       GridDefaultSimd(Nd,vComplex::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);

  GridCartesian         * UGrid_f   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(),
                                         GridDefaultSimd(Nd,vComplexF::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid_f = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid_f);
  GridCartesian         * FGrid_f   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid_f);
  GridRedBlackCartesian * FrbGrid_f = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid_f);

  std::cout << GridLogMessage << "# eigref pre-calc: hot=" << hotflag << " free=" << freeflag
            << " config=" << config << " mass=" << mass << " -> " << eigref << std::endl;

  // ---- gauge: HOT (random, non-degenerate testbed) / FREE (cold) / NERSC -----
  // HOT uses a FIXED RNG seed so the eigref pre-calc and the Cheby bench build the
  // IDENTICAL config (reproducible for matching --grid/--mpi) -> their spectra agree.
  // FREE (cold/unit) is exactly degenerate (lambda_min multiplicity ~24) which breaks
  // single-vector Lanczos; HOT lifts the degeneracy, as a real config does.
  LatticeGaugeField  Umu(UGrid);
  LatticeGaugeFieldF Umu_f(UGrid_f);
  if(hotflag){
    GridParallelRNG pRNG(UGrid); pRNG.SeedFixedIntegers({1,2,3,4});
    SU<Nc>::HotConfiguration(pRNG, Umu);
    std::cout << GridLogMessage << "# HOT (random) gauge, seed {1,2,3,4}" << std::endl;
  } else if(freeflag || config.empty()){
    SU<Nc>::ColdConfiguration(Umu);    // unit gauge -- FREE theory (exactly degenerate!)
    std::cout << GridLogMessage << "# FREE theory: cold (unit) gauge" << std::endl;
  } else {
    FieldMetaData header;
    NerscIO::readConfiguration(Umu, header, config);
  }
  precisionChange(Umu_f, Umu);

  FermionAction  D  (Umu,   *FGrid,   *FrbGrid,   *UGrid,   *UrbGrid,   mass, M5, b, c, Params);
  FermionActionF D_f(Umu_f, *FGrid_f, *FrbGrid_f, *UGrid_f, *UrbGrid_f, mass, M5, b, c, Params);

  SchurDiagMooeeOperator<FermionAction,  LatticeFermion>  HermOpEO  (D);
  SchurDiagMooeeOperator<FermionActionF, LatticeFermionF> HermOpEO_f(D_f);

  // ---- lambda_max estimate (PowerMethod on H) -- the Cheby upper window edge --
  RealD lambda_max = 0.0;
  {
    GridParallelRNG RNG5f(FGrid_f); RNG5f.SeedFixedIntegers({9,10,11,12});
    LatticeFermionF tmp_f(FGrid_f), pm_src(FrbGrid_f);
    random(RNG5f, tmp_f); pickCheckerboard(Odd, pm_src, tmp_f);
    PowerMethod<LatticeFermionF> PM;
    lambda_max = PM(HermOpEO_f, pm_src);
    std::cout << GridLogMessage << "# PowerMethod lambda_max = " << lambda_max << std::endl;
  }

  // ======================================================================
  //  Reference eigensolve -- shift-invert IRL (robust; this defines the landscape).
  // ======================================================================
  std::vector<RealD> eval(Nm);
  int Nconv = 0;

  if(eig_prec == 2){
    std::vector<LatticeFermion> evec_d(Nm, LatticeFermion(FrbGrid));
    GridParallelRNG RNG5(FGrid); RNG5.SeedFixedIntegers({5,6,7,8});
    LatticeFermion tmp(FGrid), lanc_src(FrbGrid);
    random(RNG5, tmp); pickCheckerboard(Odd, lanc_src, tmp);

    PlainHermOp<LatticeFermion>   HermOp(HermOpEO);
    InverseHermOp<LatticeFermion> Hinv(HermOpEO, inv_tol, inv_maxit);
    ImplicitlyRestartedLanczos<LatticeFermion> IRL(Hinv, HermOp, Nstop, Nk, Nm, eresid, maxit);

    std::cout << GridLogMessage << "# DOUBLE shift-invert IRL seeking " << Nstop << " low modes" << std::endl;
    double t0 = usecond();
    IRL.calc(eval, evec_d, lanc_src, Nconv);
    std::cout << GridLogMessage << "# LANCZOS prec=double Nconv=" << Nconv
              << " wall=" << (usecond()-t0)*1.0e-6 << " s" << std::endl;
  } else {
    std::vector<LatticeFermionF> evec(Nm, LatticeFermionF(FrbGrid_f));
    GridParallelRNG RNG5f(FGrid_f); RNG5f.SeedFixedIntegers({5,6,7,8});
    LatticeFermionF tmp_f(FGrid_f), lanc_src(FrbGrid_f);
    random(RNG5f, tmp_f); pickCheckerboard(Odd, lanc_src, tmp_f);

    PlainHermOp<LatticeFermionF>   HermOp(HermOpEO_f);
    InverseHermOp<LatticeFermionF> Hinv(HermOpEO_f, inv_tol, inv_maxit);
    ImplicitlyRestartedLanczos<LatticeFermionF> IRL(Hinv, HermOp, Nstop, Nk, Nm, eresid, maxit);

    std::cout << GridLogMessage << "# SINGLE shift-invert IRL seeking " << Nstop << " low modes" << std::endl;
    double t0 = usecond();
    IRL.calc(eval, evec, lanc_src, Nconv);
    std::cout << GridLogMessage << "# LANCZOS prec=single Nconv=" << Nconv
              << " wall=" << (usecond()-t0)*1.0e-6 << " s" << std::endl;
  }

  for(int i=0; i<Nconv; i++)
    std::cout << GridLogMessage << "#   eval[" << i << "] = " << eval[i]
              << "  sigma = " << std::sqrt(eval[i]) << std::endl;

  // ======================================================================
  //  Write the reference landscape to HDF5 (boss rank only). Datasets:
  //  lambda (vector<double>, ascending), lambda_max, Nconv + scalar metadata.
  //  The Cheby bench reads lambda + lambda_max to set its filter window; h5py can
  //  read these directly for offline plotting of the landscape.
  // ======================================================================
  if(UGrid->IsBoss()){
    Coordinate latt = GridDefaultLatt();
    std::string gtag = std::to_string(latt[0]);
    for(int d=1; d<(int)latt.size(); d++) gtag += "." + std::to_string(latt[d]);

    std::vector<RealD> lam(eval.begin(), eval.begin() + Nconv);
    std::vector<RealD> sig(Nconv);
    for(int i=0; i<Nconv; i++) sig[i] = std::sqrt(eval[i]);

    Hdf5Writer WR(eigref);
    write(WR, "grid",       gtag);
    write(WR, "Ls",         Ls);
    write(WR, "mass",       mass);
    write(WR, "M5",         M5);
    write(WR, "b",          b);
    write(WR, "c",          c);
    write(WR, "free",       freeflag);
    write(WR, "Nconv",      Nconv);
    write(WR, "lambda_max", lambda_max);
    write(WR, "lambda",     lam);
    write(WR, "sigma",      sig);
    std::cout << GridLogMessage << "# wrote " << Nconv << " evals + lambda_max="
              << lambda_max << " to " << eigref << " (HDF5)" << std::endl;
  }

  std::cout << GridLogMessage << "# eigref pre-calc done." << std::endl;
  Grid_finalize();
  return 0;
}
