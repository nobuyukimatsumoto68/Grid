#include <getopt.h>   // getopt_long for --config/--mass
#include <cstdlib>    // std::getenv / std::atoi / std::atof
#include <cmath>      // std::isfinite (InverseHermOp NaN guard), std::sqrt
#include <Grid/Grid.h>

// disc LMA -- CHUNK A bench (correctness only; NO variance/production yet).
// On ONE gauge config it (1) eigensolves the low modes of the EO-Schur squared
// Mobius operator (shift-invert IRL, single precision -- reuses the deflation
// bench setup), (2) reconstructs, per low mode i, the PHYSICAL 4D A2A pair
// (a_i,b_i) so that  S_low = sum_i (1/sigma_i) a_i b_i^dag  reproduces the
// low-mode part of the physical propagator S = E M^{-1} I used by the disc
// binary, and (3) runs three correctness gates + prints the exact low-mode loop
// L^low_Gamma(t, p=0).
//
// Derivation / Grid-call mapping: disc_lma_impl_plan_claude.md "KEY DERIVATION".
//   sigma_i = sqrt(lambda_i),  u_i = Mpc v_i / sigma_i  (odd-cb singular pair),
//   V_i = (even:-Mee^{-1}Meo v_i, odd: v_i),  U_i = (even:-Mee^{-dag}Moe^{dag}u_i, odd:u_i),
//   a_i = E V_i = ExportPhysicalFermionSolution(V_i),
//   b_i = I^dag U_i = P^dag(DminusDag U_i),  P^dag chi = P+ chi[0] + P- chi[Ls-1].
//
// Algorithm sources:
//   - A2A "v/w" all-to-all: Foley-Juge-O'Cais-Peardon-Ryan-Skullerud hep-lat/0505023.
//   - LMA: DeGrand-Schaefer hep-lat/0401011; Giusti et al hep-lat/0402002.
//   - shift-invert Lanczos eigensolve: as in disc_mrhs_defl_bench_claude.cc.
//
// Three gates (NO solve needed for 1+2 -- they test the construction against the
// ACTUAL 5D operator M):
//   GATE 1  eigen-residual ||H v - lambda v||/lambda + u-orthonormality.
//   GATE 2  ||M V_i - sigma_i uhat_i|| and ||Mdag U_i - sigma_i vhat_i||  (the lift).
//   GATE 3  gamma5-herm cross-check ||b_i - gamma5 a_i||/||b_i|| (informational).
//
// Claude never builds/submits/rm's -- user runs the build script and any flux job.

using namespace std;
using namespace Grid;

// ---- small env helpers (tunables overridable at run time, no rebuild) --------
static int    env_int   (const char* k, int    d){ const char* e=std::getenv(k); return e? std::atoi(e):d; }
static double env_double(const char* k, double d){ const char* e=std::getenv(k); return e? std::atof(e):d; }

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

// ---- build the physical 4D A2A pair (a,b) for one low mode --------------------
// Inputs: the double-precision odd-cb Schur eigenvector v_o (eigenvalue sigma^2),
// the double Mobius action D, its Schur operator HermOpEO. Outputs a4 = E V (on
// the gauge grid) and b4 = I^dag U (on the gauge grid). See the KEY DERIVATION.
static void BuildPhysicalA2A(MobiusFermionD &D,
                             SchurDiagMooeeOperator<MobiusFermionD,LatticeFermion> &HermOpEO,
                             const LatticeFermion &v_o, RealD sigma,
                             LatticeFermion &a4, LatticeFermion &b4,
                             LatticeFermion &u_o_out)
{
  GridBase *FGrid   = D.FermionGrid();
  GridBase *FrbGrid = D.FermionRedBlackGrid();
  const int Ls = D.Ls;

  // u = Mpc v / sigma  (odd-cb left singular vector of Mpc)
  LatticeFermion u_o(FrbGrid);
  HermOpEO.Mpc(v_o, u_o);
  u_o = u_o * (1.0/sigma);
  u_o_out = u_o;

  // full 5D V: odd = v, even = -Mee^{-1} Meo v
  LatticeFermion Vfull(FGrid);
  Vfull = Zero();
  {
    LatticeFermion te(FrbGrid), te2(FrbGrid);
    D.Meooe(v_o, te);       // te (even) = Meo v
    D.MooeeInv(te, te2);    // te2 = Mee^{-1} Meo v
    te2 = -te2;
    setCheckerboard(Vfull, te2);
    setCheckerboard(Vfull, v_o);
  }

  // full 5D U: odd = u, even = -Mee^{-dag} Moe^{dag} u
  LatticeFermion Ufull(FGrid);
  Ufull = Zero();
  {
    LatticeFermion te(FrbGrid), te2(FrbGrid);
    D.MeooeDag(u_o, te);    // te (even) = Moe^{dag} u
    D.MooeeInvDag(te, te2); // te2 = Mee^{-dag} Moe^{dag} u
    te2 = -te2;
    setCheckerboard(Ufull, te2);
    setCheckerboard(Ufull, u_o);
  }

  // a = E V (physical solution export: P- V[0] + P+ V[Ls-1])
  D.ExportPhysicalFermionSolution(Vfull, a4);

  // b = I^dag U = P^dag (DminusDag U), with P^dag chi = P+ chi[0] + P- chi[Ls-1].
  // Mirror ExportPhysicalFermionSolution with the chiral projectors SWAPPED so the
  // projector convention matches Grid's surface maps exactly.
  {
    LatticeFermion chi5(FGrid), t5(FGrid);
    D.DminusDag(Ufull, chi5);
    t5 = chi5;
    axpby_ssp_pplus (t5, 0.0, chi5, 1.0, chi5, 0, 0);     // t5[0]  = P+ chi5[0]
    axpby_ssp_pminus(t5, 1.0, t5,   1.0, chi5, 0, Ls-1);  // t5[0] += P- chi5[Ls-1]
    ExtractSlice(b4, t5, 0, 0);
  }
}


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

  // ---- Lanczos knobs (same defaults as the defl bench) -----------------------
  const int    Nstop  = env_int   ("NSTOP",  100);
  const int    Nk     = env_int   ("NK",     100);
  const int    Nm     = env_int   ("NM",     140);
  const double eresid = env_double("ERESID", 1.0e-5);
  const int    maxit  = env_int   ("MAXITER",200);
  const double inv_tol   = env_double("INV_TOL",   1.0e-5);
  const int    inv_maxit = env_int   ("INV_MAXIT", 50000);
  const int    Nev    = env_int   ("NEV", 100);      // # low modes used in the loop/checks
  const int    Ncheck = env_int   ("NCHECK", 5);     // # modes to print gate1/2/3 detail for
  // EIG_METHOD: 2 = shift-invert IRL (robust, expensive inner-CG matvec; DEFAULT),
  //             1 = Chebyshev-filtered IRL (cheap polynomial matvec; needs a window).
  // The eigenbasis is identical either way -- only the IRL PolyOp slot changes. Keep
  // the knob so we can A/B Cheby vs shift-invert per ensemble without a rebuild (the
  // Cheby window lower edge is the CUT lambda_{Nstop}, NOT lambda_min -- see
  // disc_tuning_routine_claude.md).
  const int    eig_method = env_int   ("EIG_METHOD", 2);
  const double cheb_lo    = env_double("CHEB_LO",    0.02);     // IR edge = the cut (TUNE from evals)
  const int    cheb_o     = env_int   ("CHEB_ORD",   60);       // ~ sqrt(hi/lo) rule of thumb
  const double cheb_hifc  = env_double("CHEB_HI_FAC",1.1);      // hi = fac * power-method lambda_max
  // EIG_PREC: 1 = single eigensolve (default, fast), 2 = double eigensolve (prec-switch
  // TEST -- removes the single inner-CG true-resid floor; SLOW, use pbatch). Same
  // Nstop/Nk/Nm in both so a GATE1 change isolates precision from the larger-Nm effect.
  const int    eig_prec   = env_int   ("EIG_PREC",   1);

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

  const int Nt = UGrid->_fdimensions[Tdir];

  std::cout << GridLogMessage << "# LMA bench config = " << config << " mass = " << mass
            << " Nev = " << Nev << std::endl;

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

  // ======================================================================
  //  Eigenbasis: low modes of the Schur operator. EIG_PREC picks the eigensolve
  //  precision (1 = single, default; 2 = double), EIG_METHOD the IRL PolyOp
  //  (shift-invert / Chebyshev) for the single path. The reconstruction reads each
  //  mode into a double v_d (copied from evec_d, or converted from evec per-mode).
  //  EIG_PREC=2 is the prec-switch TEST: run the SAME Nstop/Nk/Nm in double so any
  //  GATE1 change is attributable to precision alone (vs the larger-Nm hypothesis).
  //  Double shift-invert is SLOW (inner CG no longer floors at the single true-resid
  //  ~2e-3 -> it iterates to inv_tol; expect several x the single wall) -> pbatch.
  // ======================================================================
  std::vector<RealD>           eval(Nm);
  std::vector<LatticeFermionF> evec;     // single-prec modes (single path only)
  std::vector<LatticeFermion>  evec_d;   // double-prec modes (double path only)
  int    Nconv = 0;
  // Only the chosen-precision store is allocated (each evec is ~0.68/1.36 GB single/
  // double; at NM=240 the unused store would be ~10 GB/rank -> OOM risk). The recon
  // loop reads from whichever is filled (see v_d below).

  if(eig_prec == 2){
    // ---- DOUBLE-precision eigensolve (shift-invert) ----
    evec_d.resize(Nm, LatticeFermion(FrbGrid));
    GridParallelRNG RNG5(FGrid); RNG5.SeedFixedIntegers({5,6,7,8});
    LatticeFermion tmp(FGrid), lanc_src(FrbGrid);
    random(RNG5, tmp); pickCheckerboard(Odd, lanc_src, tmp);

    PlainHermOp<LatticeFermion>   HermOp(HermOpEO);
    InverseHermOp<LatticeFermion> Hinv(HermOpEO, inv_tol, inv_maxit);
    ImplicitlyRestartedLanczos<LatticeFermion> IRL(Hinv, HermOp, Nstop, Nk, Nm, eresid, maxit);

    std::cout << GridLogMessage << "# DOUBLE shift-invert IRL: inner CG tol=" << inv_tol
              << " seeking " << Nstop << " low modes" << std::endl;
    double t0 = usecond();
    IRL.calc(eval, evec_d, lanc_src, Nconv);
    std::cout << GridLogMessage << "# LANCZOS prec=double method=shift-invert Nconv=" << Nconv
              << " wall=" << (usecond()-t0)*1.0e-6 << " s" << std::endl;
  } else {
    // ---- SINGLE-precision eigensolve (modes converted to double per-mode below) ----
    evec.resize(Nm, LatticeFermionF(FrbGrid_f));
    GridParallelRNG RNG5f(FGrid_f); RNG5f.SeedFixedIntegers({5,6,7,8});
    LatticeFermionF tmp_f(FGrid_f), lanc_src(FrbGrid_f);
    random(RNG5f, tmp_f); pickCheckerboard(Odd, lanc_src, tmp_f);

    PlainHermOp<LatticeFermionF> HermOp(HermOpEO_f);   // real H for the convergence test

    double t0 = usecond();
    if(eig_method == 1){
      // ---- Chebyshev-filtered IRL: PolyOp = T_N(H), cheap matvec, needs window ----
      LatticeFermionF pm_src(FrbGrid_f);
      random(RNG5f, tmp_f); pickCheckerboard(Odd, pm_src, tmp_f);
      PowerMethod<LatticeFermionF> PM;
      RealD cheb_hi = cheb_hifc * PM(HermOpEO_f, pm_src);
      std::cout << GridLogMessage << "# Chebyshev IRL window [" << cheb_lo << ", " << cheb_hi
                << "] order " << cheb_o << " seeking " << Nstop << " low modes" << std::endl;
      Chebyshev<LatticeFermionF>      Cheby(cheb_lo, cheb_hi, cheb_o);
      FunctionHermOp<LatticeFermionF> PolyOp(Cheby, HermOpEO_f);
      ImplicitlyRestartedLanczos<LatticeFermionF> IRL(PolyOp, HermOp, Nstop, Nk, Nm, eresid, maxit);
      IRL.calc(eval, evec, lanc_src, Nconv);
    } else {
      // ---- shift-invert IRL: PolyOp = H^{-1} via inner CG (robust, no window) ----
      std::cout << GridLogMessage << "# shift-invert IRL: inner CG tol=" << inv_tol
                << " seeking " << Nstop << " low modes" << std::endl;
      InverseHermOp<LatticeFermionF> Hinv(HermOpEO_f, inv_tol, inv_maxit);
      ImplicitlyRestartedLanczos<LatticeFermionF> IRL(Hinv, HermOp, Nstop, Nk, Nm, eresid, maxit);
      IRL.calc(eval, evec, lanc_src, Nconv);
    }
    std::cout << GridLogMessage << "# LANCZOS prec=single method=" << (eig_method==1?"cheby":"shift-invert")
              << " Nconv=" << Nconv << " wall=" << (usecond()-t0)*1.0e-6 << " s" << std::endl;
  }

  for(int i=0; i<Nconv; i++)
    std::cout << GridLogMessage << "#   eval[" << i << "] = " << eval[i]
              << "  sigma = " << std::sqrt(eval[i]) << std::endl;

  const int Nuse = std::min(Nev, Nconv);
  std::cout << GridLogMessage << "# using Nuse=" << Nuse << " low modes" << std::endl;

  // ======================================================================
  //  Per-mode reconstruction + GATES + exact low-mode loop accumulation.
  //  Modes are read in double from evec_d (built above in either precision).
  // ======================================================================
  const std::vector<Gamma::Algebra> gams = {
    Gamma::Algebra::Identity, Gamma::Algebra::Gamma5,
    Gamma::Algebra::GammaX,   Gamma::Algebra::GammaY,
    Gamma::Algebra::GammaZ,   Gamma::Algebra::GammaT,
    Gamma::Algebra::GammaXGamma5, Gamma::Algebra::GammaYGamma5,
    Gamma::Algebra::GammaZGamma5, Gamma::Algebra::GammaTGamma5,
  };
  const std::vector<std::string> gam_names = {"id","g5","gx","gy","gz","gt","gxg5","gyg5","gzg5","gtg5"};

  std::vector<LatticeComplex> Llow(gam_names.size(), LatticeComplex(UGrid));
  for(auto &r : Llow) r = Zero();

  double gate1_max = 0.0, gate2_max = 0.0, gate3_max = 0.0;

  LatticeFermion a4(UGrid), b4(UGrid), u_o(FrbGrid);
  LatticeFermion Ga(UGrid), g5a(UGrid);
  LatticeFermion v_d(FrbGrid);   // current mode in double

  for(int i=0; i<Nuse; i++){
    if(eig_prec == 2) v_d = evec_d[i];               // double path: copy
    else              precisionChange(v_d, evec[i]);  // single path: convert per-mode
    const RealD sigma = std::sqrt(eval[i]);

    BuildPhysicalA2A(D, HermOpEO, v_d, sigma, a4, b4, u_o);

    // ---- GATE 1: eigen-residual ||H v - lambda v|| / lambda ----
    {
      LatticeFermion Hv(FrbGrid);
      HermOpEO.HermOp(v_d, Hv);
      Hv = Hv - eval[i]*v_d;
      RealD r = std::sqrt(norm2(Hv) / (eval[i]*eval[i]*norm2(v_d)));
      if(r > gate1_max) gate1_max = r;
      if(i < Ncheck)
        std::cout << GridLogMessage << "# GATE1 mode " << i << " eigres="
                  << r << "  norm2(v)=" << norm2(v_d) << std::endl;
    }

    // ---- GATE 2: ||M V - sigma uhat|| and ||Mdag U - sigma vhat|| ----
    // Rebuild V,U here (the helper consumes them); cheap relative to the eigensolve.
    {
      GridBase *FG = FGrid;
      LatticeFermion Vfull(FG), Ufull(FG);
      Vfull = Zero(); Ufull = Zero();
      {
        LatticeFermion te(FrbGrid), te2(FrbGrid);
        D.Meooe(v_d, te);    D.MooeeInv(te, te2);    te2 = -te2;
        setCheckerboard(Vfull, te2); setCheckerboard(Vfull, v_d);
        D.MeooeDag(u_o, te); D.MooeeInvDag(te, te2); te2 = -te2;
        setCheckerboard(Ufull, te2); setCheckerboard(Ufull, u_o);
      }
      LatticeFermion uhat(FG), vhat(FG), MX(FG);
      uhat = Zero(); setCheckerboard(uhat, u_o);
      vhat = Zero(); setCheckerboard(vhat, v_d);

      D.M(Vfull, MX);
      MX = MX - sigma*uhat;
      RealD r1 = std::sqrt(norm2(MX) / (eval[i]*norm2(u_o)));

      D.Mdag(Ufull, MX);
      MX = MX - sigma*vhat;
      RealD r2 = std::sqrt(norm2(MX) / (eval[i]*norm2(v_d)));

      RealD r = std::max(r1, r2);
      if(r > gate2_max) gate2_max = r;
      if(i < Ncheck)
        std::cout << GridLogMessage << "# GATE2 mode " << i << " ||MV-su||="
                  << r1 << "  ||MdagU-sv||=" << r2 << std::endl;
    }

    // ---- GATE 3: gamma5-herm cross-check  b ?= gamma5 a (informational) ----
    {
      g5a = Gamma(Gamma::Algebra::Gamma5) * a4;
      LatticeFermion d = b4 - g5a;
      RealD r = std::sqrt(norm2(d) / norm2(b4));
      if(r > gate3_max) gate3_max = r;
      if(i < Ncheck)
        std::cout << GridLogMessage << "# GATE3 mode " << i << " ||b-g5a||/||b||="
                  << r << "  norm2(a)=" << norm2(a4) << " norm2(b)=" << norm2(b4) << std::endl;
    }

    // ---- accumulate the exact low-mode loop  L^low_Gamma(x) ----
    // L^low_Gamma(x) = sum_i (1/sigma_i) tr_sc[Gamma a_i b_i^dag](x)
    //               = sum_i (1/sigma_i) localInnerProduct(b_i, Gamma a_i)(x).
    for(int ig=0; ig<(int)gam_names.size(); ig++){
      Ga = Gamma(gams[ig]) * a4;
      Llow[ig] = Llow[ig] + (1.0/sigma) * localInnerProduct(b4, Ga);
    }
  }

  std::cout << GridLogMessage << "# ===== GATE SUMMARY (max over " << Nuse << " modes) ====="  << std::endl;
  std::cout << GridLogMessage << "# GATE1 eigres_max      = " << gate1_max << std::endl;
  std::cout << GridLogMessage << "# GATE2 lift_resid_max  = " << gate2_max << "  (should be ~eigres)" << std::endl;
  std::cout << GridLogMessage << "# GATE3 g5herm_max      = " << gate3_max << "  (informational)" << std::endl;

  // ======================================================================
  //  Sanity print: zero-momentum projection of the exact low-mode loop,
  //  L^low_Gamma(t, p=0) = sum_{x_vec} L^low_Gamma(x), for id and g5.
  // ======================================================================
  for(int ig=0; ig<(int)gam_names.size(); ig++){
    if(gam_names[ig] != "id" && gam_names[ig] != "g5") continue;
    std::vector<TComplex> tv;
    sliceSum(Llow[ig], tv, Tdir);
    std::cout << GridLogMessage << "# L^low[" << gam_names[ig] << "](t,p=0):" << std::endl;
    for(int t=0; t<Nt; t++){
      Complex z = TensorRemove(tv[t]);
      std::cout << GridLogMessage << "#   t=" << t << "  " << real(z) << "  " << imag(z) << std::endl;
    }
  }

  std::cout << GridLogMessage << "# chunk-A done: gates above must pass before chunks B/C." << std::endl;

  Grid_finalize();
  return 0;
}
