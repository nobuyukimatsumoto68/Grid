#include "disc_lma_v2_common_claude.h"

// disc LMA -- v2 EIGENSOLVE bench (now thin; shared machinery in disc_lma_v2_common_claude.h).
// On ONE gauge config it: (1) eigensolves the low modes of the EO-Schur squared Mobius operator
// via Chebyshev (auto window + auto order) or shift-invert IRL, single or double, then optional
// double Rayleigh-Ritz refine -> a refined low-mode subspace (BuildLowModes); (2) reconstructs
// the physical 4D A2A pair (a_i,b_i) + odd-cb u_i per mode (BuildA2ASet); (3) runs three
// correctness GATES on the first NCHECK modes and prints the exact low-mode loop L^low(t,p=0).
// The LMA variance-reduction ESTIMATOR (chunk B) lives in disc_lma_estimator_bench_v2_claude.cc.
//
// GATES (test the construction against the ACTUAL 5D operator M):
//   GATE 1  ||H v - lambda v||/lambda                          (eigenvector residual)
//   GATE 2  ||M V_i - sigma_i uhat_i|| / ||Mdag U_i - sigma_i vhat_i||  (the v/w lift)
//   GATE 3  ||b_i - gamma5 a_i||/||b_i||                        (gamma5-herm cross-check)
//
// Claude never builds/submits/rm's -- user runs the build script and any job.

using namespace std;
using namespace Grid;

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

  const LMAEigParams P = ReadLMAEigParams();

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

  std::cout << GridLogMessage << "# LMA v2 eigensolve bench: hot=" << hotflag << " free=" << freeflag
            << " config=" << config << " mass=" << mass << " Nev=" << P.Nev
            << " eigref=" << eigref << std::endl;

  // ---- reference spectral landscape (for the Chebyshev window) ----------------
  std::vector<RealD> lambda_ref;
  RealD lambda_max_ref = 0.0;
  bool have_ref = false;
  if(!eigref.empty()) have_ref = ReadEigref(eigref, lambda_ref, lambda_max_ref);
  if(have_ref)
    std::cout << GridLogMessage << "# eigref loaded: " << lambda_ref.size()
              << " evals, lambda_ref[0]=" << lambda_ref.front()
              << " lambda_ref[back]=" << lambda_ref.back()
              << " lambda_max=" << lambda_max_ref << std::endl;
  else
    std::cout << GridLogMessage << "# no eigref (path empty/unreadable) -> "
              << "hi from PowerMethod, lo from manual CHEB_LO" << std::endl;

  // ---- gauge + Mobius actions + Schur operators ------------------------------
  LatticeGaugeField  Umu(UGrid);
  LatticeGaugeFieldF Umu_f(UGrid_f);
  SetupGauge(Umu, Umu_f, hotflag, freeflag, config);

  FermionAction  D  (Umu,   *FGrid,   *FrbGrid,   *UGrid,   *UrbGrid,   mass, M5, b, c, Params);
  FermionActionF D_f(Umu_f, *FGrid_f, *FrbGrid_f, *UGrid_f, *UrbGrid_f, mass, M5, b, c, Params);

  SchurDiagMooeeOperator<FermionAction,  LatticeFermion>  HermOpEO  (D);
  SchurDiagMooeeOperator<FermionActionF, LatticeFermionF> HermOpEO_f(D_f);

  // ---- eigensolve (+ RR) -> refined subspace, then per-mode physical A2A ------
  std::vector<LatticeFermion> sub;
  std::vector<RealD>          eval_use;
  BuildLowModes(D, D_f, HermOpEO, HermOpEO_f, P, have_ref, lambda_ref, lambda_max_ref, sub, eval_use);

  std::vector<LatticeFermion> a, b4v, u;
  std::vector<RealD>          sigma;
  BuildA2ASet(D, HermOpEO, sub, eval_use, a, b4v, u, sigma);
  const int Nuse = (int)sub.size();

  // ======================================================================
  //  GATES 1-3 -- validation spot-checks on the first NCHECK modes only (GATE2 allocates
  //  ~5 full-5D double fields/mode; running it for all modes churned GPU memory).
  // ======================================================================
  double gate1_max = 0.0, gate2_max = 0.0, gate3_max = 0.0;
  const int ncheck = std::min(Nuse, P.Ncheck);
  for(int i=0; i<ncheck; i++){
    const RealD sig = sigma[i];

    // ---- GATE 1: eigen-residual ||H v - lambda v|| / lambda ----
    {
      LatticeFermion Hv(FrbGrid);
      HermOpEO.HermOp(sub[i], Hv);
      Hv = Hv - eval_use[i]*sub[i];
      RealD r = std::sqrt(norm2(Hv) / (eval_use[i]*eval_use[i]*norm2(sub[i])));
      if(r > gate1_max) gate1_max = r;
      std::cout << GridLogMessage << "# GATE1 mode " << i << " eigres="
                << r << "  norm2(v)=" << norm2(sub[i]) << std::endl;
    }

    // ---- GATE 2: ||M V - sigma uhat|| and ||Mdag U - sigma vhat|| ----
    {
      LatticeFermion Vfull(FGrid), Ufull(FGrid);
      Vfull = Zero(); Ufull = Zero();
      {
        LatticeFermion te(FrbGrid), te2(FrbGrid);
        D.Meooe(sub[i], te);   D.MooeeInv(te, te2);    te2 = -te2;
        setCheckerboard(Vfull, te2); setCheckerboard(Vfull, sub[i]);
        D.MeooeDag(u[i], te);  D.MooeeInvDag(te, te2); te2 = -te2;
        setCheckerboard(Ufull, te2); setCheckerboard(Ufull, u[i]);
      }
      LatticeFermion uhat(FGrid), vhat(FGrid), MX(FGrid);
      uhat = Zero(); setCheckerboard(uhat, u[i]);
      vhat = Zero(); setCheckerboard(vhat, sub[i]);

      D.M(Vfull, MX);
      MX = MX - sig*uhat;
      RealD r1 = std::sqrt(norm2(MX) / (eval_use[i]*norm2(u[i])));

      D.Mdag(Ufull, MX);
      MX = MX - sig*vhat;
      RealD r2 = std::sqrt(norm2(MX) / (eval_use[i]*norm2(sub[i])));

      RealD r = std::max(r1, r2);
      if(r > gate2_max) gate2_max = r;
      std::cout << GridLogMessage << "# GATE2 mode " << i << " ||MV-su||="
                << r1 << "  ||MdagU-sv||=" << r2 << std::endl;
    }

    // ---- GATE 3: gamma5-herm cross-check  b ?= gamma5 a (informational) ----
    {
      LatticeFermion g5a(UGrid);
      g5a = Gamma(Gamma::Algebra::Gamma5) * a[i];
      LatticeFermion d = b4v[i] - g5a;
      RealD r = std::sqrt(norm2(d) / norm2(b4v[i]));
      if(r > gate3_max) gate3_max = r;
      std::cout << GridLogMessage << "# GATE3 mode " << i << " ||b-g5a||/||b||="
                << r << "  norm2(a)=" << norm2(a[i]) << " norm2(b)=" << norm2(b4v[i]) << std::endl;
    }
  }
  std::cout << GridLogMessage << "# ===== GATE SUMMARY (max over first " << ncheck << " modes) ====="  << std::endl;
  std::cout << GridLogMessage << "# GATE1 eigres_max      = " << gate1_max << std::endl;
  std::cout << GridLogMessage << "# GATE2 lift_resid_max  = " << gate2_max << "  (should be ~eigres)" << std::endl;
  std::cout << GridLogMessage << "# GATE3 g5herm_max      = " << gate3_max << "  (informational)" << std::endl;

  // ======================================================================
  //  Exact low-mode loop  L^low_Gamma(x) = sum_i (1/sigma_i) tr_sc[Gamma a_i b_i^dag](x),
  //  then its zero-momentum projection L^low(t,p=0) for id and g5.
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

  LatticeFermion Ga(UGrid);
  for(int i=0; i<Nuse; i++){
    for(int ig=0; ig<(int)gam_names.size(); ig++){
      Ga = Gamma(gams[ig]) * a[i];
      Llow[ig] = Llow[ig] + (1.0/sigma[i]) * localInnerProduct(b4v[i], Ga);
    }
  }

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

  std::cout << GridLogMessage << "# eigensolve bench done (gates + L^low above)." << std::endl;

  Grid_finalize();
  return 0;
}
