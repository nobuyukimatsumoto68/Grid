#pragma once
// disc LMA v2 -- SHARED machinery for the eigensolve + A2A reconstruction, factored out of
// disc_lma_bench_v2_claude.cc so the eigensolve bench AND the LMA-estimator bench
// (disc_lma_estimator_bench_v2_claude.cc) reuse it with no duplication. Contents:
//   - env_int/env_double, ParseArgs, ReadEigref (HDF5 spectral landscape)
//   - InverseHermOp (shift-invert), BuildPhysicalA2A (per-mode physical v/w pair)
//   - ComputeChebWindow (auto window + auto order from a target gain), RayleighRitzRefine
//   - LMAEigParams + ReadLMAEigParams (env knobs)
//   - BuildLowModes (eigensolve Cheby/shift-invert single/double + RR -> refined subspace)
//   - BuildA2ASet (refined subspace -> per-mode a_i,b_i,u_i,sigma_i)
// Derivations + the bugs behind these choices: disc_lma_cheby_v2_impl_plan_claude.md,
// disc_lma_impl_plan_claude.md. Algorithm sources: A2A Foley et al hep-lat/0505023; LMA
// DeGrand-Schaefer hep-lat/0401011; IRL Matsumoto lanczos.pdf / Saad.
#include <getopt.h>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <cassert>
#include <Grid/Grid.h>

using namespace Grid;

// ---- small env helpers (tunables overridable at run time, no rebuild) --------
inline int    env_int   (const char* k, int    d){ const char* e=std::getenv(k); return e? std::atoi(e):d; }
inline double env_double(const char* k, double d){ const char* e=std::getenv(k); return e? std::atof(e):d; }

// ---- config path / mass / eigref / free / hot from CLI -----------------------
// No leading '+' in the optstring: getopt PERMUTES so our flags are found regardless of
// position (Grid's own --grid/--mpi are unknown here -> '?' -> ignored). Call Grid_init
// FIRST (it consumes its args) so this permutation does not disturb --grid value pairing.
inline void ParseArgs(int argc, char** argv, std::string& cfg, double& mass,
                      std::string& eigref, int& freeflag, int& hotflag)
{
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
      case 'c': cfg      = optarg;            break;
      case 'm': mass     = std::stod(optarg); break;
      case 'e': eigref   = optarg;            break;
      case 'f': freeflag = 1;                 break;
      case 'H': hotflag  = 1;                 break;
      default: break;
    }
  }
}

// ---- gauge setup: HOT (random, non-degenerate testbed) / FREE (cold) / NERSC ----
// HOT uses a FIXED seed {1,2,3,4} -- the SAME as disc_lma_eigref_v2_claude -- so the eigref
// pre-calc and any bench build the IDENTICAL config (reproducible per --grid/--mpi); their
// spectra must agree for the auto window. FREE (cold) is exactly degenerate -> Lanczos nan.
inline void SetupGauge(LatticeGaugeField& Umu, LatticeGaugeFieldF& Umu_f,
                       int hotflag, int freeflag, const std::string& config)
{
  GridBase* UGrid = Umu.Grid();
  if(hotflag){
    GridParallelRNG pRNG(UGrid); pRNG.SeedFixedIntegers({1,2,3,4});
    SU<Nc>::HotConfiguration(pRNG, Umu);
    std::cout << GridLogMessage << "# HOT (random) gauge, seed {1,2,3,4}" << std::endl;
  } else if(freeflag || config.empty()){
    SU<Nc>::ColdConfiguration(Umu);   // unit gauge -- FREE theory (exactly degenerate!)
    std::cout << GridLogMessage << "# FREE theory: cold (unit) gauge" << std::endl;
  } else {
    FieldMetaData header;
    NerscIO::readConfiguration(Umu, header, config);
  }
  precisionChange(Umu_f, Umu);
}

// ---- read the reference spectrum landscape (HDF5) written by disc_lma_eigref_v2_claude.
// Returns lambda_ref (ascending) + lambda_max.
inline bool ReadEigref(const std::string& path, std::vector<RealD>& lambda_ref, RealD& lambda_max)
{
  std::ifstream test(path);
  if(!test.good()) return false;     // missing file -> caller falls back to PowerMethod/knobs
  test.close();
  Hdf5Reader RD(path);
  read(RD, "lambda_max", lambda_max);
  read(RD, "lambda",     lambda_ref);
  return !lambda_ref.empty();
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
// Inputs: the double-precision odd-cb Schur eigenvector v_o (eigenvalue sigma^2), the double
// Mobius action D, its Schur operator HermOpEO. Outputs a4 = E V, b4 = I^dag U (gauge grid),
// u_o_out = odd-cb left singular vector. See the KEY DERIVATION (disc_lma_impl_plan_claude.md):
//   u = Mpc v / sigma,  V_i=(even:-Mee^{-1}Meo v, odd:v),  U_i=(even:-Mee^{-dag}Moe^{dag}u, odd:u),
//   a = E V,  b = I^dag U = P^dag(DminusDag U),  P^dag chi = P+ chi[0] + P- chi[Ls-1].
inline void BuildPhysicalA2A(MobiusFermionD &D,
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

// ---- Chebyshev window (lo=cut, hi=top of bulk) + AUTO order + filter diagnostic ----
// lo/hi from the reference landscape (have_ref) + lambda_max; the order is AUTO-derived from a
// target amplification GAIN so the SAME gain is hit on any spectrum (a fixed order over-amplifies
// when lambda_min is far below the cut -> single-prec Lanczos fluctuates; under-amplifies when
// close). deg ~ acosh(gain)/acosh|y_min|, forced EVEN (=> odd order) so low modes map to +large
// (the end Grid IRL keeps, partial_sort greater). Diagnostic uses a scalar Chebyshev::approx.
inline void ComputeChebWindow(bool have_ref, const std::vector<RealD>& lambda_ref,
                              RealD lambda_max, int Nstop,
                              int cheb_lo_auto, double cheb_lo_fac, double cheb_lo_man,
                              double cheb_hifc, double cheb_gain, double cheb_atop,
                              RealD& cheb_lo, RealD& cheb_hi, int& cheb_o)
{
  cheb_hi = cheb_hifc * lambda_max;
  if(cheb_lo_auto && have_ref){
    int icut = std::min(Nstop, (int)lambda_ref.size()) - 1;
    cheb_lo = cheb_lo_fac * lambda_ref[icut];
    std::cout << GridLogMessage << "# Cheby lo AUTO = " << cheb_lo_fac
              << " * lambda_ref[" << icut << "]=" << lambda_ref[icut]
              << " -> " << cheb_lo << std::endl;
  } else {
    cheb_lo = cheb_lo_man;
    std::cout << GridLogMessage << "# Cheby lo MANUAL = " << cheb_lo << std::endl;
  }
  // AUTO order. PREFERRED anchor = the BAND-TOP wanted mode lambda_ref[Nstop-1] (the cut
  // boundary), targeting amplification CHEB_ATOP. This is config-ROBUST: the band-top is
  // stable across configs, whereas anchoring on lambda_min (the cheb_gain fallback) is
  // config-DEPENDENT and UNDER-orders near-zero-mode configs (smaller lambda_min -> larger
  // |y_min| -> smaller degree, i.e. less amplification exactly where it's needed). MEASURED
  // (m=0.01 lat.758): band-top ~4.6x (CHEB_ATOP~5, order 151) converged 100/100 in 95s with
  // eval reldiff 9e-7 -- vs gain-1e4 (order 397, 237s, reldiff 4.5e-5, single-prec
  // over-amplification). CHEB_ATOP<=0 falls back to the gain-on-lambda_min path.
  if(cheb_atop > 0.0 && have_ref){
    int icut = std::min(Nstop, (int)lambda_ref.size()) - 1;
    RealD lcut = lambda_ref[icut];
    RealD yt   = (lcut - 0.5*(cheb_hi + cheb_lo)) / (0.5*(cheb_hi - cheb_lo));
    RealD ay   = std::fabs(yt);
    if(ay > 1.0 + 1.0e-12){
      int deg = (int)std::ceil(std::acosh(cheb_atop) / std::acosh(ay));
      if(deg < 2)    deg = 2;
      if(deg > 4000) deg = 4000;          // sane cap
      if(deg % 2 != 0) deg += 1;          // EVEN degree
      cheb_o = deg + 1;                   // order = degree + 1 (ODD)
      std::cout << GridLogMessage << "# Cheby order AUTO (band-top anchor): A_top=" << cheb_atop
                << "  lambda_ref[" << icut << "]=" << lcut << "  |y|=" << ay
                << "  -> degree=" << deg << " (order=" << cheb_o << ")" << std::endl;
    }
  } else if(cheb_gain > 0.0 && have_ref){
    RealD lmin = lambda_ref.front();
    RealD ymin = (lmin - 0.5*(cheb_hi + cheb_lo)) / (0.5*(cheb_hi - cheb_lo));
    RealD ay   = std::fabs(ymin);
    if(ay > 1.0 + 1.0e-12){
      int deg = (int)std::ceil(std::acosh(cheb_gain) / std::acosh(ay));
      if(deg < 2)    deg = 2;
      if(deg > 4000) deg = 4000;          // sane cap
      if(deg % 2 != 0) deg += 1;          // EVEN degree
      cheb_o = deg + 1;                   // order = degree + 1 (ODD)
      std::cout << GridLogMessage << "# Cheby order AUTO (lambda_min gain, FALLBACK): gain=" << cheb_gain
                << "  |y_min|=" << ay << "  -> degree=" << deg
                << " (order=" << cheb_o << ")" << std::endl;
    }
  }
  // even-degree invariant (odd order); guaranteed by the AUTO path, asserted for MANUAL.
  assert((cheb_o % 2 == 1) && "Chebyshev order must be ODD (=> even degree T_{order-1}); "
         "odd degree maps low modes to negative filter values and IRL locks onto bulk modes.");
  std::cout << GridLogMessage << "# Chebyshev IRL window [" << cheb_lo << ", " << cheb_hi
            << "] order " << cheb_o << " (deg " << cheb_o-1 << ")" << std::endl;
  if(have_ref){
    Chebyshev<LatticeFermionF> Cheby(cheb_lo, cheb_hi, cheb_o);
    std::cout << GridLogMessage << "# --- Cheby filter at reference evals ---" << std::endl;
    int nshow = std::min((int)lambda_ref.size(), Nstop + 5);
    for(int i=0; i<nshow; i++){
      RealD p = Cheby.approx(lambda_ref[i]);
      const char* tag = (lambda_ref[i] <= cheb_lo) ? "WANT" : "bulk";
      std::cout << GridLogMessage << "#   [" << tag << "] lambda_ref[" << i << "]="
                << lambda_ref[i] << "  p=" << p << std::endl;
    }
    RealD p_min = Cheby.approx(lambda_ref.front());
    RealD p_cut = Cheby.approx(cheb_lo);
    std::cout << GridLogMessage << "# p(lambda_min)=" << p_min << " p(lo)=" << p_cut
              << "  amplification |p_min/p_cut|=" << std::fabs(p_min / p_cut)
              << "  (want >> 1, and p_min POSITIVE)" << std::endl;
    if(p_min < 0.0)
      std::cout << GridLogMessage << "# WARNING: p(lambda_min) < 0 -> ODD Chebyshev degree;"
                << " IRL keeps the LARGEST filtered evals so it will lock onto bulk modes"
                << " (spurious eval, never converges). Use EVEN degree (odd CHEB_ORD)." << std::endl;
  }
}

// ---- Rayleigh-Ritz refinement (double) of an approximate low-mode subspace ----
// Polish m approximate eigenvectors V (double, odd rb) to DOUBLE accuracy WITHOUT a double
// eigensolve: MGS re-orthonormalize, form A_jk=<V_j,H V_k>, diagonalize (Eigen Hermitian),
// rotate V. On return V = refined eigenvectors, eval = refined eigenvalues (ascending).
inline void RayleighRitzRefine(SchurDiagMooeeOperator<MobiusFermionD,LatticeFermion>& HermOpEO,
                               std::vector<LatticeFermion>& V, std::vector<RealD>& eval)
{
  const int m = (int)V.size();
  if(m == 0) return;
  for(int j=0; j<m; j++){
    for(int k=0; k<j; k++){
      ComplexD o = innerProduct(V[k], V[j]);
      V[j] = V[j] - o * V[k];
    }
    RealD nj = std::sqrt(norm2(V[j]));
    V[j] = V[j] * (1.0 / nj);
  }
  GridBase* g = V[0].Grid();
  LatticeFermion HVk(g);
  Eigen::MatrixXcd A(m, m);
  for(int k=0; k<m; k++){
    HermOpEO.HermOp(V[k], HVk);
    for(int j=0; j<m; j++){
      ComplexD a = innerProduct(V[j], HVk);
      A(j, k) = std::complex<double>(real(a), imag(a));
    }
  }
  A = 0.5 * (A + A.adjoint().eval());   // kill rounding asymmetry
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(A);
  Eigen::VectorXd  mu = es.eigenvalues();    // ascending
  Eigen::MatrixXcd C  = es.eigenvectors();   // column i = coeffs of refined mode i
  std::vector<LatticeFermion> W;             // emplace_back avoids a prototype-temporary free
  W.reserve(m);
  for(int i=0; i<m; i++) W.emplace_back(g);
  for(int i=0; i<m; i++){
    W[i] = ComplexD(C(0,i).real(), C(0,i).imag()) * V[0];
    for(int k=1; k<m; k++)
      W[i] = W[i] + ComplexD(C(k,i).real(), C(k,i).imag()) * V[k];
  }
  V.swap(W);
  eval.resize(m);
  for(int i=0; i<m; i++) eval[i] = mu(i);
}

// ---- eigensolve knobs (env) ---------------------------------------------------
struct LMAEigParams {
  int    Nstop, Nk, Nm, maxit, Nev, Ncheck;
  double eresid;
  double inv_tol;   int inv_maxit;
  int    eig_method, eig_prec, rr_refine;
  int    cheb_lo_auto; double cheb_lo_fac, cheb_lo_man, cheb_gain, cheb_atop, cheb_hifc;
  int    cheb_o;
};

inline LMAEigParams ReadLMAEigParams()
{
  LMAEigParams P;
  P.Nstop  = env_int   ("NSTOP",  100);
  P.Nk     = env_int   ("NK",     100);
  P.Nm     = env_int   ("NM",     140);
  P.eresid = env_double("ERESID", 1.0e-5);
  P.maxit  = env_int   ("MAXITER",200);
  P.inv_tol   = env_double("INV_TOL",   1.0e-5);
  P.inv_maxit = env_int   ("INV_MAXIT", 50000);
  P.Nev    = env_int   ("NEV", 100);
  P.Ncheck = env_int   ("NCHECK", 5);
  P.eig_method = env_int   ("EIG_METHOD", 1);   // 1 = Chebyshev, 2 = shift-invert
  P.cheb_lo_auto = env_int   ("CHEB_LO_AUTO", 1);
  P.cheb_lo_fac  = env_double("CHEB_LO_FAC",  1.5);
  P.cheb_lo_man  = env_double("CHEB_LO",      0.02);
  // Order precedence: CHEB_ATOP>0 (band-top anchor, PRIMARY) -> CHEB_GAIN>0 (lambda_min,
  // FALLBACK) -> CHEB_ORD (MANUAL, used only if both <=0). CHEB_ORD must be ODD (=> even
  // degree); asserted in ComputeChebWindow.
  P.cheb_o    = env_int   ("CHEB_ORD",   61);
  P.cheb_atop = env_double("CHEB_ATOP",  8.0);   // target band-top amplification (validated ~5)
  P.cheb_gain = env_double("CHEB_GAIN",  0.0);   // lambda_min gain fallback (off by default now)
  P.cheb_hifc = env_double("CHEB_HI_FAC",1.1);
  P.eig_prec  = env_int   ("EIG_PREC",   1);     // 1 = single, 2 = double
  P.rr_refine = env_int   ("RR_REFINE",  1);
  return P;
}

// ---- BuildLowModes: eigensolve (Cheby/shift-invert, single/double) + subspace extract + RR ----
// Out: sub = refined double odd-cb eigenvectors (size Nuse=min(Nev,Nconv)); eval_use = refined
// eigenvalues (ascending). have_ref/lambda_ref/lambda_max_ref carry the eigref landscape (for the
// Cheby window); if !have_ref the Cheby path estimates lambda_max via a single-prec PowerMethod.
inline void BuildLowModes(MobiusFermionD &D, MobiusFermionF &D_f,
                          SchurDiagMooeeOperator<MobiusFermionD,LatticeFermion>  &HermOpEO,
                          SchurDiagMooeeOperator<MobiusFermionF,LatticeFermionF> &HermOpEO_f,
                          const LMAEigParams &P,
                          bool have_ref, const std::vector<RealD> &lambda_ref, RealD lambda_max_ref,
                          std::vector<LatticeFermion> &sub, std::vector<RealD> &eval_use)
{
  GridBase *FGrid    = D.FermionGrid();
  GridBase *FrbGrid  = D.FermionRedBlackGrid();
  GridBase *FGrid_f  = D_f.FermionGrid();
  GridBase *FrbGrid_f= D_f.FermionRedBlackGrid();

  std::vector<RealD>           eval(P.Nm);
  std::vector<LatticeFermionF> evec;     // single-prec store (single path)
  std::vector<LatticeFermion>  evec_d;   // double-prec store (double path)
  int Nconv = 0;

  // Chebyshev window + auto order (precision-independent), computed once.
  RealD cheb_lo = 0.0, cheb_hi = 0.0;
  int   cheb_ord_use = P.cheb_o;
  if(P.eig_method == 1){
    RealD lmax = lambda_max_ref;
    if(!have_ref){
      GridParallelRNG RNGpm(FGrid_f); RNGpm.SeedFixedIntegers({5,6,7,8});
      LatticeFermionF tmp_pm(FGrid_f), pm_src(FrbGrid_f);
      random(RNGpm, tmp_pm); pickCheckerboard(Odd, pm_src, tmp_pm);
      PowerMethod<LatticeFermionF> PM;
      lmax = PM(HermOpEO_f, pm_src);
    }
    ComputeChebWindow(have_ref, lambda_ref, lmax, P.Nstop, P.cheb_lo_auto, P.cheb_lo_fac,
                      P.cheb_lo_man, P.cheb_hifc, P.cheb_gain, P.cheb_atop, cheb_lo, cheb_hi, cheb_ord_use);
  }

  if(P.eig_prec == 2){
    // ---- DOUBLE-precision eigensolve (Chebyshev OR shift-invert) ----
    evec_d.resize(P.Nm, LatticeFermion(FrbGrid));
    GridParallelRNG RNG5(FGrid); RNG5.SeedFixedIntegers({5,6,7,8});
    LatticeFermion tmp(FGrid), lanc_src(FrbGrid);
    random(RNG5, tmp); pickCheckerboard(Odd, lanc_src, tmp);

    PlainHermOp<LatticeFermion> HermOp(HermOpEO);
    double t0 = usecond();
    if(P.eig_method == 1){
      std::cout << GridLogMessage << "# DOUBLE Chebyshev IRL seeking " << P.Nstop << " low modes" << std::endl;
      Chebyshev<LatticeFermion>      Cheby(cheb_lo, cheb_hi, cheb_ord_use);
      FunctionHermOp<LatticeFermion> PolyOp(Cheby, HermOpEO);
      ImplicitlyRestartedLanczos<LatticeFermion> IRL(PolyOp, HermOp, P.Nstop, P.Nk, P.Nm, P.eresid, P.maxit);
      IRL.calc(eval, evec_d, lanc_src, Nconv);
    } else {
      std::cout << GridLogMessage << "# DOUBLE shift-invert IRL: inner CG tol=" << P.inv_tol
                << " seeking " << P.Nstop << " low modes" << std::endl;
      InverseHermOp<LatticeFermion> Hinv(HermOpEO, P.inv_tol, P.inv_maxit);
      ImplicitlyRestartedLanczos<LatticeFermion> IRL(Hinv, HermOp, P.Nstop, P.Nk, P.Nm, P.eresid, P.maxit);
      IRL.calc(eval, evec_d, lanc_src, Nconv);
    }
    std::cout << GridLogMessage << "# LANCZOS prec=double method=" << (P.eig_method==1?"cheby":"shift-invert")
              << " Nconv=" << Nconv << " wall=" << (usecond()-t0)*1.0e-6 << " s" << std::endl;
  } else {
    // ---- SINGLE-precision eigensolve (modes converted to double in the subspace extract) ----
    evec.resize(P.Nm, LatticeFermionF(FrbGrid_f));
    GridParallelRNG RNG5f(FGrid_f); RNG5f.SeedFixedIntegers({5,6,7,8});
    LatticeFermionF tmp_f(FGrid_f), lanc_src(FrbGrid_f);
    random(RNG5f, tmp_f); pickCheckerboard(Odd, lanc_src, tmp_f);

    PlainHermOp<LatticeFermionF> HermOp(HermOpEO_f);
    double t0 = usecond();
    if(P.eig_method == 1){
      std::cout << GridLogMessage << "# SINGLE Chebyshev IRL seeking " << P.Nstop << " low modes" << std::endl;
      Chebyshev<LatticeFermionF>      Cheby(cheb_lo, cheb_hi, cheb_ord_use);
      FunctionHermOp<LatticeFermionF> PolyOp(Cheby, HermOpEO_f);
      ImplicitlyRestartedLanczos<LatticeFermionF> IRL(PolyOp, HermOp, P.Nstop, P.Nk, P.Nm, P.eresid, P.maxit);
      IRL.calc(eval, evec, lanc_src, Nconv);
    } else {
      std::cout << GridLogMessage << "# SINGLE shift-invert IRL: inner CG tol=" << P.inv_tol
                << " seeking " << P.Nstop << " low modes" << std::endl;
      InverseHermOp<LatticeFermionF> Hinv(HermOpEO_f, P.inv_tol, P.inv_maxit);
      ImplicitlyRestartedLanczos<LatticeFermionF> IRL(Hinv, HermOp, P.Nstop, P.Nk, P.Nm, P.eresid, P.maxit);
      IRL.calc(eval, evec, lanc_src, Nconv);
    }
    std::cout << GridLogMessage << "# LANCZOS prec=single method=" << (P.eig_method==1?"cheby":"shift-invert")
              << " Nconv=" << Nconv << " wall=" << (usecond()-t0)*1.0e-6 << " s" << std::endl;
  }

  for(int i=0; i<Nconv; i++)
    std::cout << GridLogMessage << "#   eval[" << i << "] = " << eval[i]
              << "  sigma = " << std::sqrt(eval[i]) << std::endl;

  // Cheby/shift-invert evals vs the reference landscape (headline convergence check).
  if(have_ref){
    std::cout << GridLogMessage << "# ===== eigensolve vs reference eigenvalues =====" << std::endl;
    int ncmp = std::min(Nconv, (int)lambda_ref.size());
    RealD reldiff_max = 0.0;
    for(int i=0; i<ncmp; i++){
      RealD rd = std::fabs(eval[i] - lambda_ref[i]) / lambda_ref[i];
      if(rd > reldiff_max) reldiff_max = rd;
      if(i < P.Ncheck)
        std::cout << GridLogMessage << "#   i=" << i << " eig=" << eval[i]
                  << " ref=" << lambda_ref[i] << " reldiff=" << rd << std::endl;
    }
    std::cout << GridLogMessage << "# eval reldiff_max (first " << ncmp << ") = "
              << reldiff_max << "   Nconv=" << Nconv << " / Nstop=" << P.Nstop
              << (Nconv >= P.Nstop ? "  CONVERGED" : "  *** NOT all converged ***") << std::endl;
  }

  // Extract the Nuse subspace in DOUBLE (emplace_back; do NOT free the store -- see the
  // AccCache.bytes==bytes lesson), then optional Rayleigh-Ritz polish to double accuracy.
  const int Nuse = std::min(P.Nev, Nconv);
  sub.clear();
  sub.reserve(Nuse);
  for(int i=0; i<Nuse; i++) sub.emplace_back(FrbGrid);
  for(int i=0; i<Nuse; i++){
    if(P.eig_prec == 2) sub[i] = evec_d[i];
    else                precisionChange(sub[i], evec[i]);
  }
  eval_use.assign(eval.begin(), eval.begin() + Nuse);
  if(P.rr_refine){
    std::cout << GridLogMessage << "# Rayleigh-Ritz refining " << Nuse << " modes in double..." << std::endl;
    RayleighRitzRefine(HermOpEO, sub, eval_use);
    for(int i=0; i<std::min(Nuse,P.Ncheck); i++)
      std::cout << GridLogMessage << "#   RR eval[" << i << "]=" << eval_use[i]
                << " sigma=" << std::sqrt(eval_use[i]) << std::endl;
  }
  std::cout << GridLogMessage << "# using Nuse=" << Nuse << " low modes" << std::endl;
}

// ---- BuildA2ASet: refined subspace -> per-mode physical A2A (a_i,b_i on UGrid), odd-cb u_i,
// sigma_i. Used by BOTH L^low (a_i b_i^dag) and the source projection (the orthonormal u_i).
inline void BuildA2ASet(MobiusFermionD &D,
                        SchurDiagMooeeOperator<MobiusFermionD,LatticeFermion> &HermOpEO,
                        const std::vector<LatticeFermion> &sub, const std::vector<RealD> &eval_use,
                        std::vector<LatticeFermion> &a, std::vector<LatticeFermion> &b,
                        std::vector<LatticeFermion> &u, std::vector<RealD> &sigma)
{
  GridBase *UGrid   = D.GaugeGrid();
  GridBase *FrbGrid = D.FermionRedBlackGrid();
  const int n = (int)sub.size();
  a.clear(); b.clear(); u.clear();
  a.reserve(n); b.reserve(n); u.reserve(n);
  for(int i=0; i<n; i++){ a.emplace_back(UGrid); b.emplace_back(UGrid); u.emplace_back(FrbGrid); }
  sigma.assign(n, 0.0);
  for(int i=0; i<n; i++){
    sigma[i] = std::sqrt(eval_use[i]);
    BuildPhysicalA2A(D, HermOpEO, sub[i], sigma[i], a[i], b[i], u[i]);
  }
}

// ============================================================================
//  Stochastic source + solves + source projection (shared by the estimator bench AND the
//  production binary). StochasticDilutedSource/Solve/TraceField are from the reference disc
//  binary; SolveHighProjected/SolvePropProjected/LowPartOnSource are the LMA chunk-B pieces.
// ============================================================================

// ---- Z4 time+eo diluted source (from the reference disc binary) ----
inline void StochasticDilutedSource(GridParallelRNG &RNG, LatticePropagator &source,
                                    GridBase *rbgrid, const int tslice, const int eo)
{
  GridBase *grid = source.Grid();
  RealD nrm = 1.0/std::sqrt(2.0);

  LatticeInteger t(grid);
  LatticeCoordinate(t, Tdir);

  LatticeComplex zz(grid);    zz = Zero();
  LatticeComplex xi(grid);
  LatticeComplex xi_rb(rbgrid);
  LatticeComplex xi_eo(grid);

  source = Zero();
  for(int s=0; s<Nd; s++){
    for(int col=0; col<Nc; col++){
      bernoulli(RNG, xi);
      xi = (2.0*xi - Complex(1.0,1.0))*nrm;
      xi = where(t==Integer(tslice), xi, zz);

      xi_eo = Zero();
      pickCheckerboard(eo, xi_rb, xi);
      setCheckerboard(xi_eo, xi_rb);

      auto spin_block = peekSpin(source, s, s);
      pokeColour(spin_block, xi_eo, col, col);
      pokeSpin(source, spin_block, s, s);
    }
  }
}

// ---- full PLAIN solve (SchurRedBlackDiagMooeeSolve), all spin-colour columns ----
template<class Action>
void Solve(Action &D, LatticePropagator &source, LatticePropagator &propagator)
{
  GridBase *UGrid = D.GaugeGrid();
  GridBase *FGrid = D.FermionGrid();

  LatticeFermion src4   (UGrid);
  LatticeFermion src5   (FGrid);
  LatticeFermion result5(FGrid);
  LatticeFermion result4(UGrid);

  ConjugateGradient<LatticeFermion> CG(1.0e-8, 100000);   // outer tol 1e-8 -- NEVER relax
  SchurRedBlackDiagMooeeSolve<LatticeFermion> schur(CG);
  ZeroGuesser<LatticeFermion> ZG;
  for(int s=0; s<Nd; s++){
    for(int col=0; col<Nc; col++){
      PropToFerm<Action>(src4, source, s, col);
      D.ImportPhysicalFermionSource(src4, src5);
      result5 = Zero();
      schur(D, src5, result5, ZG);
      D.ExportPhysicalFermionSolution(result5, result4);
      FermToProp<Action>(propagator, result4, s, col);
    }
  }
}

// ---- disc trace: tr[ Gamma psi eta^dag ] ----
inline void TraceField(LatticeComplex& out, const Gamma::Algebra& gam,
                       LatticePropagator &psi, LatticePropagator &eta)
{
  out = trace(Gamma(gam)*psi*adj(eta));
}

// ---- source-PROJECTED high solve (one column): out4 = S_high eta = E M^{-1} (1-P_low) I eta.
// Reproduce SchurRedBlackDiagMooeeSolve's reduction but PROJECT the low modes out of the odd-cb
// source BEFORE Mpc^dag (where u_i live), so the well-conditioned solve returns the HIGH part.
// Key identity Mpc^{-1} u_i = v_i/sigma_i (u_i := Mpc v_i/sigma_i) -> the split is exact (chunk B).
inline void SolveHighProjected(MobiusFermionD &D,
                               SchurDiagMooeeOperator<MobiusFermionD,LatticeFermion> &HermOpEO,
                               const std::vector<LatticeFermion> &u,
                               const LatticeFermion &eta4, LatticeFermion &out4)
{
  GridBase *FGrid   = D.FermionGrid();
  GridBase *FrbGrid = D.FermionRedBlackGrid();

  LatticeFermion src5(FGrid);
  D.ImportPhysicalFermionSource(eta4, src5);

  LatticeFermion src_e(FrbGrid), src_o(FrbGrid);
  pickCheckerboard(Even, src_e, src5);
  pickCheckerboard(Odd , src_o, src5);

  // b_o' = src_o - Meooe(MooeeInv(src_e))   (pre-Mpc^dag part of RedBlackSource)
  LatticeFermion tmp_e(FrbGrid), Mtmp(FrbGrid), bo(FrbGrid);
  D.MooeeInv(src_e, tmp_e);
  D.Meooe(tmp_e, Mtmp);
  bo = src_o - Mtmp;

  // project the low modes out onto u_i
  for(int i=0; i<(int)u.size(); i++){
    ComplexD ci = innerProduct(u[i], bo);
    bo = bo - ci * u[i];
  }

  // src_tilde = Mpc^dag b_o'^perp ; CG solves Mpc^dag Mpc x = src_tilde -> x = Mpc^{-1} b_o'^perp
  LatticeFermion src_tilde(FrbGrid), sol_o(FrbGrid);
  HermOpEO.MpcDag(bo, src_tilde);
  ConjugateGradient<LatticeFermion> CG(1.0e-8, 100000);   // outer tol 1e-8 -- NEVER relax
  sol_o = Zero();
  CG(HermOpEO, src_tilde, sol_o);

  // reconstruct full 5D: sol_e = MooeeInv(src_e - Meooe(sol_o))
  LatticeFermion te(FrbGrid), sol_e(FrbGrid), sol5(FGrid);
  D.Meooe(sol_o, te);
  te = src_e - te;
  D.MooeeInv(te, sol_e);
  sol5 = Zero();
  setCheckerboard(sol5, sol_e);
  setCheckerboard(sol5, sol_o);

  D.ExportPhysicalFermionSolution(sol5, out4);
}

// ---- projected high solve over all spin-colour columns of a diluted source propagator ----
inline void SolvePropProjected(MobiusFermionD &D,
                               SchurDiagMooeeOperator<MobiusFermionD,LatticeFermion> &HermOpEO,
                               const std::vector<LatticeFermion> &u,
                               LatticePropagator &source, LatticePropagator &propagator)
{
  GridBase *UGrid = D.GaugeGrid();
  LatticeFermion src4(UGrid), res4(UGrid);
  for(int s=0; s<Nd; s++){
    for(int col=0; col<Nc; col++){
      PropToFerm<MobiusFermionD>(src4, source, s, col);
      SolveHighProjected(D, HermOpEO, u, src4, res4);
      FermToProp<MobiusFermionD>(propagator, res4, s, col);
    }
  }
}

// ---- exact low part on a 4D source column: S_low eta = sum_i (1/sigma_i) a_i <b_i, eta> ----
inline void LowPartOnSource(const std::vector<LatticeFermion> &a,
                            const std::vector<LatticeFermion> &b,
                            const std::vector<RealD> &sigma,
                            const LatticeFermion &eta4, LatticeFermion &out4)
{
  out4 = Zero();
  for(int i=0; i<(int)a.size(); i++){
    ComplexD bi = innerProduct(b[i], eta4);
    out4 = out4 + (bi/sigma[i]) * a[i];
  }
}

// ============================================================================
//  Evec CHECKPOINT: save/reload the refined double subspace (odd-cb) + evals. Evecs ->
//  Scidac (set Checkerboard()=Odd before reading); evals + count -> HDF5. Lets a rerun SKIP
//  the per-config eigensolve (the expensive step). Reusable for conn LMA / deflation later.
// ============================================================================
// sub is non-const: Grid's writeScidacFieldRecord takes a non-const Lattice& (view machinery).
inline void SaveEvecs(const std::string& evec_file, const std::string& eval_file,
                      std::vector<LatticeFermion>& sub, const std::vector<RealD>& eval_use)
{
  emptyUserRecord record;
  ScidacWriter WR(sub[0].Grid()->IsBoss());
  WR.open(evec_file);
  for(int k=0; k<(int)sub.size(); k++) WR.writeScidacFieldRecord(sub[k], record);
  WR.close();
  if(sub[0].Grid()->IsBoss()){
    int n = (int)eval_use.size();
    std::vector<RealD> ev(eval_use);
    Hdf5Writer EV(eval_file);
    write(EV, "Nuse", n);
    write(EV, "eval", ev);
  }
}

inline bool LoadEvecs(const std::string& evec_file, const std::string& eval_file,
                      GridBase* FrbGrid,
                      std::vector<LatticeFermion>& sub, std::vector<RealD>& eval_use)
{
  std::ifstream t1(evec_file), t2(eval_file);
  bool ok = t1.good() && t2.good();
  t1.close(); t2.close();
  if(!ok) return false;
  int n = 0;
  {
    Hdf5Reader EV(eval_file);
    read(EV, "Nuse", n);
    read(EV, "eval", eval_use);
  }
  sub.clear();
  sub.reserve(n);
  for(int k=0; k<n; k++) sub.emplace_back(FrbGrid);
  emptyUserRecord record;
  ScidacReader RD;
  RD.open(evec_file);
  for(int k=0; k<n; k++){
    sub[k].Checkerboard() = Odd;
    RD.readScidacFieldRecord(sub[k], record);
  }
  RD.close();
  return true;
}
