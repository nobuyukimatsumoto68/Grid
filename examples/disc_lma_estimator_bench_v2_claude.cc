#include "disc_lma_v2_common_claude.h"

// disc LMA -- ESTIMATOR bench (chunk B/C of disc_lma_impl_plan_claude.md). Reuses the
// eigensolve/A2A machinery from disc_lma_v2_common_claude.h. On ONE gauge config it:
//   (1) builds the refined low-mode subspace (BuildLowModes) + physical A2A a_i,b_i,u_i,sigma_i
//       (BuildA2ASet);
//   (2) forms the EXACT low loop L^low_Gamma(x) = sum_i (1/sigma_i) tr_sc[Gamma a_i b_i^dag](x);
//   (3) [THIS chunk] the PLAIN stochastic loop -- full time+eo+spin+colour-diluted Z4 noise,
//       full (un-deflated) solve, tr[Gamma S_full eta eta^dag] -- the baseline + the S_full eta
//       the chunk-3 projection check needs.
// Chunk 3 adds the source-PROJECTED high solve (S_high) + the deterministic per-noise gate
//   || S_full eta - (S_high eta + S_low eta) || / || S_full eta ||  (== mode residual);
// chunk 4 adds the variance comparison var(LMA) vs var(plain).
//
// Source/solve/trace infrastructure copied from disc_multipleGamma_binary_claude.cc (the
// reference disc binary); outer solve tol 1e-8 (NEVER relaxed). LMA: DeGrand-Schaefer
// hep-lat/0401011; Giusti et al hep-lat/0402002.

using namespace std;
using namespace Grid;


// Stochastic source + solves + projection now live in disc_lma_v2_common_claude.h.



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

  std::cout << GridLogMessage << "# LMA estimator bench: hot=" << hotflag << " free=" << freeflag
            << " config=" << config << " mass=" << mass << " Nev=" << P.Nev
            << " eigref=" << eigref << std::endl;

  // ---- reference spectral landscape (for the Chebyshev window) ----------------
  std::vector<RealD> lambda_ref;
  RealD lambda_max_ref = 0.0;
  bool have_ref = false;
  if(!eigref.empty()) have_ref = ReadEigref(eigref, lambda_ref, lambda_max_ref);
  if(have_ref)
    std::cout << GridLogMessage << "# eigref loaded: " << lambda_ref.size()
              << " evals, lambda_max=" << lambda_max_ref << std::endl;
  else
    std::cout << GridLogMessage << "# no eigref -> hi from PowerMethod, lo from manual CHEB_LO" << std::endl;

  // ---- gauge + Mobius actions + Schur operators ------------------------------
  LatticeGaugeField  Umu(UGrid);
  LatticeGaugeFieldF Umu_f(UGrid_f);
  SetupGauge(Umu, Umu_f, hotflag, freeflag, config);

  FermionAction  D  (Umu,   *FGrid,   *FrbGrid,   *UGrid,   *UrbGrid,   mass, M5, b, c, Params);
  FermionActionF D_f(Umu_f, *FGrid_f, *FrbGrid_f, *UGrid_f, *UrbGrid_f, mass, M5, b, c, Params);

  SchurDiagMooeeOperator<FermionAction,  LatticeFermion>  HermOpEO  (D);
  SchurDiagMooeeOperator<FermionActionF, LatticeFermionF> HermOpEO_f(D_f);

  // ---- eigenbasis (eigensolve + RR) + per-mode physical A2A ------------------
  std::vector<LatticeFermion> sub;
  std::vector<RealD>          eval_use;
  BuildLowModes(D, D_f, HermOpEO, HermOpEO_f, P, have_ref, lambda_ref, lambda_max_ref, sub, eval_use);

  std::vector<LatticeFermion> a, b4v, u;
  std::vector<RealD>          sigma;
  BuildA2ASet(D, HermOpEO, sub, eval_use, a, b4v, u, sigma);
  const int Nuse = (int)sub.size();
  std::cout << GridLogMessage << "# A2A set built for Nuse=" << Nuse << " low modes" << std::endl;

  const std::vector<Gamma::Algebra> gams = {
    Gamma::Algebra::Identity, Gamma::Algebra::Gamma5,
    Gamma::Algebra::GammaX,   Gamma::Algebra::GammaY,
    Gamma::Algebra::GammaZ,   Gamma::Algebra::GammaT,
    Gamma::Algebra::GammaXGamma5, Gamma::Algebra::GammaYGamma5,
    Gamma::Algebra::GammaZGamma5, Gamma::Algebra::GammaTGamma5,
  };
  const std::vector<std::string> gam_names = {"id","g5","gx","gy","gz","gt","gxg5","gyg5","gzg5","gtg5"};
  const int ngam = (int)gam_names.size();

  // ======================================================================
  //  (2) EXACT low-mode loop  L^low_Gamma(x) = sum_i (1/sigma_i) tr_sc[Gamma a_i b_i^dag](x).
  // ======================================================================
  std::vector<LatticeComplex> Llow(ngam, LatticeComplex(UGrid));
  for(auto &r : Llow) r = Zero();
  {
    LatticeFermion Ga(UGrid);
    for(int i=0; i<Nuse; i++){
      for(int ig=0; ig<ngam; ig++){
        Ga = Gamma(gams[ig]) * a[i];
        Llow[ig] = Llow[ig] + (1.0/sigma[i]) * localInnerProduct(b4v[i], Ga);
      }
    }
  }

  // ======================================================================
  //  (3) PLAIN stochastic loop -- one full time+eo+spin+colour dilution pass, full solve,
  //  L^plain_Gamma(x) = sum_{t,eo} tr[Gamma (S_full eta) eta^dag](x).
  // ======================================================================
  // p=0 real part of the EXACT low loop per (gamma,t) -- constant across noise samples.
  std::vector<std::vector<double>> llow_t(ngam, std::vector<double>(Nt, 0.0));
  for(int ig=0; ig<ngam; ig++){
    std::vector<TComplex> tv;
    sliceSum(Llow[ig], tv, Tdir);
    for(int t=0; t<Nt; t++) llow_t[ig][t] = real(TensorRemove(tv[t]));
  }

  // Per (gamma,t) running sum/sumsq over NSRC independent noise SAMPLES (each sample = one full
  // time+eo+sc dilution pass) for the PLAIN loop and the LMA loop (= exact low + projected high).
  const int NSRC = env_int("NSRC", 10);
  GridParallelRNG RNG4(UGrid); RNG4.SeedFixedIntegers({11,12,13,14});
  std::vector<std::vector<double>> sp (ngam, std::vector<double>(Nt, 0.0)), sp2(ngam, std::vector<double>(Nt, 0.0));
  std::vector<std::vector<double>> sl (ngam, std::vector<double>(Nt, 0.0)), sl2(ngam, std::vector<double>(Nt, 0.0));

  bool gate_done = false;
  for(int isrc=0; isrc<NSRC; isrc++){
    std::vector<LatticeComplex> res_plain(ngam, LatticeComplex(UGrid));
    std::vector<LatticeComplex> res_high (ngam, LatticeComplex(UGrid));
    for(auto &r : res_plain) r = Zero();
    for(auto &r : res_high)  r = Zero();

    for(int t=0; t<Nt; t++){
      for(int eo=0; eo<=1; eo++){
        LatticePropagator source(UGrid);
        StochasticDilutedSource(RNG4, source, UrbGrid, t, eo);

        LatticePropagator StochProp(UGrid);        // S_full eta  (plain full solve)
        Solve(D, source, StochProp);

        LatticePropagator StochPropHigh(UGrid);     // S_high eta  (low modes projected out)
        SolvePropProjected(D, HermOpEO, u, source, StochPropHigh);

        LatticeComplex tr(UGrid);
        for(int ig=0; ig<ngam; ig++){
          TraceField(tr, gams[ig], StochProp,     source); res_plain[ig] = res_plain[ig] + tr;
          TraceField(tr, gams[ig], StochPropHigh, source); res_high[ig]  = res_high[ig]  + tr;
        }

        // ---- DETERMINISTIC per-noise gate (once): with the consistent u_i = Mpc v_i/sigma_i,
        //   S_full eta = S_high eta + S_low eta to SOLVER tol for ANY modes (the projection is
        //   unbiased; mode accuracy affects only variance). See the impl plan CORRECTION.
        if(!gate_done){
          LatticeFermion eta4(UGrid), full4(UGrid), high4(UGrid), low4(UGrid), resid(UGrid);
          PropToFerm<MobiusFermionD>(eta4,  source,        0, 0);
          PropToFerm<MobiusFermionD>(full4, StochProp,     0, 0);
          PropToFerm<MobiusFermionD>(high4, StochPropHigh, 0, 0);
          LowPartOnSource(a, b4v, sigma, eta4, low4);
          resid = full4 - high4 - low4;
          RealD g = std::sqrt(norm2(resid) / norm2(full4));
          std::cout << GridLogMessage << "# DET GATE ||S_full-(S_high+S_low)||/||S_full|| = " << g
                    << "   (unbiased to solver tol for any modes)" << std::endl;
          gate_done = true;
        }
      }
    }

    // accumulate the p=0 real loop per (gamma,t): plain, and LMA = exact low + projected high.
    for(int ig=0; ig<ngam; ig++){
      std::vector<TComplex> tp, th;
      sliceSum(res_plain[ig], tp, Tdir);
      sliceSum(res_high[ig],  th, Tdir);
      for(int t=0; t<Nt; t++){
        double vp = real(TensorRemove(tp[t]));
        double vl = llow_t[ig][t] + real(TensorRemove(th[t]));
        sp[ig][t] += vp;  sp2[ig][t] += vp*vp;
        sl[ig][t] += vl;  sl2[ig][t] += vl*vl;
      }
    }
    std::cout << GridLogMessage << "# sample " << (isrc+1) << "/" << NSRC << " done" << std::endl;
  }

  // ======================================================================
  //  VARIANCE report (id/g5, p=0, per t). mean_plain ~ mean_lma (unbiased, statistical check);
  //  var_lma/var_plain < 1 is the low-mode VARIANCE reduction (the physics deliverable).
  //  Sample variance (1/(N-1)); needs NSRC>=2.
  // ======================================================================
  std::cout << GridLogMessage << "# ===== VARIANCE over NSRC=" << NSRC << " samples =====" << std::endl;
  if(NSRC >= 2){
    const double Nm1 = (double)(NSRC - 1);
    const double Ninv = 1.0/(double)NSRC;
    for(int ig=0; ig<ngam; ig++){
      if(gam_names[ig] != "id" && gam_names[ig] != "g5") continue;
      std::cout << GridLogMessage << "# VAR[" << gam_names[ig]
                << "]: t  mean_plain mean_lma   var_plain var_lma   ratio(lma/plain)" << std::endl;
      for(int t=0; t<Nt; t++){
        double mp = sp[ig][t]*Ninv,  ml = sl[ig][t]*Ninv;
        double vp = (sp2[ig][t] - sp[ig][t]*sp[ig][t]*Ninv)/Nm1;
        double vl = (sl2[ig][t] - sl[ig][t]*sl[ig][t]*Ninv)/Nm1;
        double ratio = (vp > 0.0) ? vl/vp : 0.0;
        std::cout << GridLogMessage << "#   t=" << t << "  " << mp << " " << ml
                  << "   " << vp << " " << vl << "   " << ratio << std::endl;
      }
    }
  }

  std::cout << GridLogMessage << "# estimator bench chunk-4 done (LMA variance vs plain)."
            << std::endl;

  Grid_finalize();
  return 0;
}
