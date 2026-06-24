#include <getopt.h>   // getopt_long for --config/--mass
#include <cstdlib>    // std::getenv / std::atoi / std::atof
#include <sstream>    // NEV_LIST parsing
#include <Grid/Grid.h>

// disc speedup #2+#3 -- BENCHMARK harness (Chunk A).
// Measures, on ONE gauge config (target m=0.01 @ b10.8), the break-even of
// multi-RHS + exact low-mode deflation against a plain multi-RHS mixed-precision
// solve, for the disconnected-loop solver. NO multigrid (fine operator only).
//
// For each Nev it prints the wall time + iteration count of a 16-RHS batched
// mixed-precision solve (single inner / double reliable-update, outer tol 1e-8)
// with the lowest-Nev modes deflated out as the initial guess, plus the one-off
// Lanczos setup time. The production loop does Nt*2 = 96 such 16-RHS solves per
// config, so deflation wins when
//     setup(Nev) + 96*solve_defl(Nev)  <  96*solve_mrhs(0)   [per config].
//
// Algorithm sources:
//   - exact mRHS eigenvector deflation: A. Stathopoulos, K. Orginos,
//     arXiv:0707.0131 (SIAM J. Sci. Comput. 32 (2010) 439).
//   - low-mode deflation: M. Luscher, arXiv:0706.2298.
//   - mixed-precision reliable-update CG: M. A. Clark et al., arXiv:0911.3191.
// Grid classes: ImplicitlyRestartedLanczos (production should use the faster
// ImplicitlyRestartedBlockLanczos -- see note at the Lanczos call),
// MultiRHSDeflation, MixedPrecisionConjugateGradientBatched.
//
// NOTE: writes NO traces; correctness (traces match the double binary to ~1e-7)
// is a separate check. Claude never builds/submits/rm's -- user runs the build
// script and any flux job.

using namespace std;
using namespace Grid;

// ---- small env helpers (tunables overridable at run time, no rebuild) --------
static int    env_int   (const char* k, int    d){ const char* e=std::getenv(k); return e? std::atoi(e):d; }
static double env_double(const char* k, double d){ const char* e=std::getenv(k); return e? std::atof(e):d; }

// parse "0,48,96,192" -> vector<int>
static std::vector<int> env_int_list(const char* k, const std::vector<int>& d)
{
  const char* e = std::getenv(k);
  if(!e) return d;
  std::vector<int> out;
  std::string s(e), tok;
  std::stringstream ss(s);
  while(std::getline(ss, tok, ',')) if(!tok.empty()) out.push_back(std::atoi(tok.c_str()));
  return out.empty()? d : out;
}

// ---- config path / mass from CLI (mirrors the disc binary's ParseArgs) -------
static void ParseArgs(int argc, char** argv, std::string& cfg, double& mass)
{
  const char* const short_opts = "+:c:m:";
  const option long_opts[] = {
    {"config", required_argument, nullptr, 'c'},
    {"mass",   required_argument, nullptr, 'm'},
    {nullptr,  no_argument,       nullptr,  0}
  };
  opterr = 0;
  int idx, opt;
  while((opt = getopt_long(argc, argv, short_opts, long_opts, &idx)) != -1){
    switch(opt){
      case 'c': cfg  = optarg;            break;
      case 'm': mass = std::stod(optarg); break;
      default: break;
    }
  }
}

// ---- adaptor: MultiRHSDeflation -> LinearFunction (batched guesser) ----------
// The batched CG calls the guesser as (*guesser)(src_f, sol_f) on std::vectors;
// LinearFunction's vector operator() is virtual (LinearOperator.h:638), so this
// override is dispatched. DeflateSources gives the deflated guess for the whole
// batch (arXiv:0707.0131).
template<class FieldF>
class MrhsDeflationGuesser : public LinearFunction<FieldF>
{
  MultiRHSDeflation<FieldF> &defl;
public:
  using LinearFunction<FieldF>::operator();
  MrhsDeflationGuesser(MultiRHSDeflation<FieldF> &d) : defl(d) {}

  void operator()(const FieldF &in, FieldF &out){ GRID_ASSERT(0); } // batch-only

  void operator()(const std::vector<FieldF> &in, std::vector<FieldF> &out)
  {
    // DeflateSources takes a non-const source ref but only reads it.
    defl.DeflateSources(const_cast<std::vector<FieldF>&>(in), out);
  }
};


// ---- shift-invert operator: out = H^{-1} in via CG ----------------------------
// Used as the "PolyOp" of IRL so Lanczos sees the LARGEST eigenvalues of H^{-1} =
// the LOWEST of H, with NO Chebyshev window to tune. The inner CG runs to a loose
// tol (inexact shift-invert -- fine for deflation vectors). The convergence test
// still uses the real H (PlainHermOp), so reported eigenvalues are the true lambda.
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
    ConjugateGradient<FieldF> CG(_tol, _maxit, false); // false: don't abort on non-converge
    out = Zero();
    CG(_H, in, out);                                   // out = H^{-1} in
  }
};


int main(int argc, char** argv)
{
  std::string config = "";
  double mass = 0.01;
  ParseArgs(argc, argv, config, mass);

  Grid_init(&argc, &argv);

  // ---- fixed Mobius parameters (match disc_multipleGamma_binary_claude) ------
  const int Ls = 16;
  RealD M5 = 1.5, b = 1.5, c = 0.5;
  std::vector<Complex> boundary = {1,1,1,-1};
  typedef MobiusFermionD FermionAction;
  typedef MobiusFermionF FermionActionF;
  FermionAction::ImplParams Params(boundary);

  const int nrhs = env_int("NRHS", 16);   // spin-color of one (t,eo)

  // ---- Lanczos / Chebyshev / deflation knobs (env-overridable) ---------------
  const int    Nstop  = env_int   ("NSTOP",  64);
  const int    Nk     = env_int   ("NK",     64);
  const int    Nm     = env_int   ("NM",     96);
  const double cheb_lo  = env_double("CHEB_LO", 0.02);   // TUNE from printed evals
  const int    cheb_o   = env_int   ("CHEB_ORD",60);
  const double cheb_hifc= env_double("CHEB_HI_FAC", 1.1); // safety factor on power-method
                                                          // lambda_max (a good UV estimate -> 1.1 suffices)
  const double eresid = env_double("ERESID", 1.0e-5);
  const int    maxit  = env_int   ("MAXITER",200);
  const std::vector<int> Nev_list = env_int_list("NEV_LIST", {0,16,32,64});

  // ---- batched mixed CG knobs ------------------------------------------------
  const int    maxinner = env_int("MAXINNER", 10000);
  const int    maxouter = env_int("MAXOUTER", 50);
  const int    maxpatch = env_int("MAXPATCH", 50);

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

  std::cout << GridLogMessage << "# bench config = " << config << " mass = " << mass << std::endl;
  std::cout << GridLogMessage << "# nrhs=" << nrhs << " Nstop=" << Nstop << " Nk=" << Nk
            << " Nm=" << Nm << " cheb_lo=" << cheb_lo << " cheb_ord=" << cheb_o << std::endl;
  std::cout << GridLogMessage << "# single-prec evec ~0.68 GB each on this grid; "
            << "Nm=" << Nm << " => size the node count accordingly." << std::endl;

  // ---- gauge: double, then single copy ---------------------------------------
  LatticeGaugeField  Umu(UGrid);
  LatticeGaugeFieldF Umu_f(UGrid_f);
  {
    FieldMetaData header;
    NerscIO::readConfiguration(Umu, header, config);
  }
  precisionChange(Umu_f, Umu);

  FermionAction  D  (Umu,   *FGrid,   *FrbGrid,   *UGrid,   *UrbGrid,   mass, M5, b, c, Params);
  FermionActionF D_f(Umu_f, *FGrid_f, *FrbGrid_f, *UGrid_f, *UrbGrid_f, mass, M5, b, c, Params);

  SchurDiagMooeeOperator<FermionAction,  LatticeFermion>  HermOpEO  (D);
  SchurDiagMooeeOperator<FermionActionF, LatticeFermionF> HermOpEO_f(D_f);

  // ---- optional: measure the smallest Schur eigenvalue (DO_LMIN) -------------
  // lambda_min is the production hazard: it drives the ~3000-iter light-mass solve
  // AND inflates the disconnected stochastic estimator's variance (the disc loop
  // Tr[Gamma M^{-1}] is dominated by the low modes). Inverse power iteration:
  // v <- H^{-1} v (a double CG, slow precisely because lambda_min is small), then
  // the Rayleigh quotient <v,Hv>/<v,v> -> lambda_min. With the power-method
  // lambda_max this fixes the true condition number and the right cheb_lo window.
  if(env_int("DO_LMIN", 0)){
    GridParallelRNG RNGl(FGrid); RNGl.SeedFixedIntegers({9,9,9,9});
    LatticeFermion full(FGrid), v(FrbGrid), w(FrbGrid), Hv(FrbGrid);
    random(RNGl, full); pickCheckerboard(Odd, v, full);
    v = v * (1.0/std::sqrt(norm2(v)));

    ConjugateGradient<LatticeFermion> cgInv(1.0e-5, 50000);
    const int nIPM = env_int("LMIN_STEPS", 8);
    RealD lmin = 0.0;
    for(int k=0; k<nIPM; k++){
      w = Zero();
      cgInv(HermOpEO, v, w);                       // w = H^{-1} v
      v = w * (1.0/std::sqrt(norm2(w)));
      HermOpEO.HermOp(v, Hv);
      lmin = real(innerProduct(v, Hv)) / norm2(v); // Rayleigh quotient
      std::cout << GridLogMessage << "# LMIN step " << k << "  lambda_min ~ " << lmin << std::endl;
    }
    std::cout << GridLogMessage << "# LAMBDA_MIN estimate = " << lmin << std::endl;
  }

  // ---- the 16 fixed double-precision RHS on the odd checkerboard -------------
  // Seed the RNG on the FULL grid (SeedFixedIntegers asserts on a red-black grid),
  // draw random full-grid fields, then pickCheckerboard onto the odd rb subspace.
  GridParallelRNG RNG5(FGrid); RNG5.SeedFixedIntegers({1,2,3,4});
  std::vector<LatticeFermion> rhs(nrhs, FrbGrid);
  std::vector<LatticeFermion> sol(nrhs, FrbGrid);
  {
    LatticeFermion tmp(FGrid);
    for(int r=0; r<nrhs; r++){
      random(RNG5, tmp);
      pickCheckerboard(Odd, rhs[r], tmp);
    }
  }

  // ======================================================================
  //  Eigenbasis (FANCY): low modes of the single-prec Schur operator, gated by
  //  DO_DEFL (default 1). DO_DEFL=0 = BASIC smoke test: skip Power-method +
  //  Lanczos + deflation entirely and run ONLY the plain Nev=0 batched
  //  mixed-prec solve (gauge read -> ops -> one solve).
  //  PRODUCTION (DO_DEFL=1): swap ImplicitlyRestartedLanczos ->
  //  ImplicitlyRestartedBlockLanczos (faster many-RHS eigensolve).
  // ======================================================================
  const int    do_defl    = env_int   ("DO_DEFL", 1);
  const int    eig_method = env_int   ("EIG_METHOD", 2);  // 1 = Chebyshev IRL, 2 = shift-invert IRL
  const double inv_tol    = env_double("INV_TOL",   1.0e-4);
  const int    inv_maxit  = env_int   ("INV_MAXIT", 50000);
  int    Nconv = 0;
  double tL    = 0.0;
  std::vector<RealD>           eval;
  std::vector<LatticeFermionF> evec;

  if(do_defl){
    GridParallelRNG RNG5f(FGrid_f); RNG5f.SeedFixedIntegers({5,6,7,8});
    LatticeFermionF tmp_f(FGrid_f);

    PlainHermOp<LatticeFermionF> HermOp(HermOpEO_f);   // real H for the convergence test
    eval.resize(Nm);
    evec.resize(Nm, FrbGrid_f);
    LatticeFermionF lanc_src(FrbGrid_f);
    random(RNG5f, tmp_f); pickCheckerboard(Odd, lanc_src, tmp_f);

    double tL0 = usecond();
    if(eig_method == 1){
      // ---- Chebyshev-filtered IRL: needs the [cheb_lo, cheb_hi] window ----
      LatticeFermionF pm_src(FrbGrid_f);
      random(RNG5f, tmp_f); pickCheckerboard(Odd, pm_src, tmp_f);
      PowerMethod<LatticeFermionF> PM;
      RealD cheb_hi = cheb_hifc * PM(HermOpEO_f, pm_src);
      std::cout << GridLogMessage << "# Chebyshev IRL window [" << cheb_lo << ", " << cheb_hi
                << "] order " << cheb_o << std::endl;
      Chebyshev<LatticeFermionF>      Cheby(cheb_lo, cheb_hi, cheb_o);
      FunctionHermOp<LatticeFermionF> PolyOp(Cheby, HermOpEO_f);
      ImplicitlyRestartedLanczos<LatticeFermionF> IRL(PolyOp, HermOp, Nstop, Nk, Nm, eresid, maxit);
      IRL.calc(eval, evec, lanc_src, Nconv);
    } else {
      // ---- shift-invert IRL: Lanczos on H^{-1} (no window tuning) ----
      // each Lanczos step is an inner CG solve to inv_tol (inexact shift-invert).
      std::cout << GridLogMessage << "# shift-invert IRL: inner CG tol=" << inv_tol
                << " maxit=" << inv_maxit << " (seeking " << Nstop << " low modes)" << std::endl;
      InverseHermOp<LatticeFermionF> Hinv(HermOpEO_f, inv_tol, inv_maxit);
      ImplicitlyRestartedLanczos<LatticeFermionF> IRL(Hinv, HermOp, Nstop, Nk, Nm, eresid, maxit);
      IRL.calc(eval, evec, lanc_src, Nconv);
    }
    tL = (usecond() - tL0) * 1.0e-6;
    std::cout << GridLogMessage << "# LANCZOS: method=" << (eig_method==1?"cheby":"shift-invert")
              << " Nconv=" << Nconv << " wall=" << tL << " s" << std::endl;
    for(int i=0; i<Nconv; i++)
      std::cout << GridLogMessage << "#   eval[" << i << "] = " << eval[i] << std::endl;
  } else {
    std::cout << GridLogMessage
              << "# DO_DEFL=0: BASIC mode -- skipping eigensolve/deflation; Nev=0 solve only"
              << std::endl;
  }

  // ======================================================================
  //  Nev sweep: batched mixed-prec solve of the same 16 RHS, deflating the
  //  lowest Nev modes as the guess. Nev=0 = plain mRHS+mixed (the baseline).
  // ======================================================================
  std::cout << GridLogMessage << "# ===== break-even table (16-RHS batched solve) =====" << std::endl;
  std::cout << GridLogMessage << "# Nev    wall_s(16RHS)   (Lanczos setup = " << tL << " s)" << std::endl;

  for(int Nev : Nev_list){
    if(Nev > Nconv){
      std::cout << GridLogMessage << "# skip Nev=" << Nev << " (> Nconv=" << Nconv << ")" << std::endl;
      continue;
    }

    // fresh solver each iteration so the guesser pointer is clean (Nev=0 -> none)
    MixedPrecisionConjugateGradientBatched<LatticeFermion,LatticeFermionF>
      bCG(1.0e-8, maxinner, maxouter, maxpatch, FrbGrid_f, HermOpEO_f, HermOpEO);

    // these must outlive the solve below
    MultiRHSDeflation<LatticeFermionF> defl;
    MrhsDeflationGuesser<LatticeFermionF> guesser(defl);
    if(Nev > 0){
      defl.ImportEigenBasis(evec, eval, 0, Nev);
      bCG.useGuesser(guesser);
    }

    for(int r=0; r<nrhs; r++) sol[r] = Zero();

    double t0 = usecond();
    bCG(rhs, sol);
    double t = (usecond() - t0) * 1.0e-6;

    std::cout << GridLogMessage << "# RESULT Nev=" << Nev << "  wall=" << t << " s" << std::endl;
  }

  std::cout << GridLogMessage << "# break-even: deflation wins if  setup + 96*solve_defl < 96*solve(Nev=0)"
            << std::endl;

  Grid_finalize();
  return 0;
}
