#include <filesystem>
#include <getopt.h>
#include <ctime>
#include <cstdlib>
#include <cmath>

#include <Grid/Grid.h>

// =====================================================================
// PRODUCTION REFERENCE: disconnected-loop binary with the measured ~5x speed
// stack = mixed precision (~1.43x) + multi-RHS batching (helps WITH deflation,
// ~1.65x) + shift-invert low-mode deflation (Nev~150 -> ~4-5x). Outer tol stays
// 1e-8. This is the deflation/SPEED variant of disc_multipleGamma_binary_claude.cc;
// source, dilution (t,eo,spin,color), contraction, I/O and the wall-time blocker
// are kept identical to the original. The ORIGINAL binary is left untouched.
//
// NOTE (not yet here): LMA (exact low-mode part of the loop for VARIANCE) is the
// planned extension -- it reuses the SAME eigenbasis computed below; see
// disc_lma_impl_plan_claude.md. Per-config eigenvectors should be SAVED/RELOADED
// in a per-ensemble directory (evecs + disc h5) -- see that plan's directory memo;
// left as a TODO here (this reference recomputes the eigenbasis each config).
//
// Tuning knobs are env vars, per disc_tuning_routine_claude.md:
//   NEV, NSTOP, NK, NM, INV_TOL (shift-invert inner), INNER_TOL (mixed-prec inner),
//   MAXINNER, MAXOUTER, MAXPATCH, ERESID, MAXITER.
// Algorithm sources: mixed-prec CG arXiv:0911.3191; mRHS deflation arXiv:0707.0131.
// =====================================================================

using namespace std;
using namespace Grid;

static int    env_int   (const char* k, int    d){ const char* e=std::getenv(k); return e? std::atoi(e):d; }
static double env_double(const char* k, double d){ const char* e=std::getenv(k); return e? std::atof(e):d; }

void ParseArgs(int argc, char** argv, double& mass, double& beta,
               std::string& dir, std::string& obsdir)
{
  const char* const short_opts = "+:m:b:d:o:";
  const option long_opts[] = {
    {"mass",   required_argument, nullptr, 'm'},
    {"beta",   required_argument, nullptr, 'b'},
    {"dir",    required_argument, nullptr, 'd'},
    {"obsdir", required_argument, nullptr, 'o'},
    {nullptr,  no_argument,       nullptr,  0}
  };
  opterr = 0;
  int idx, opt;
  while((opt = getopt_long(argc, argv, short_opts, long_opts, &idx)) != -1){
    switch(opt){
      case 'm': mass   = std::stod(optarg); break;
      case 'b': beta   = std::stod(optarg); break;
      case 'd': dir    = optarg;            break;
      case 'o': obsdir = optarg;            break;
      default: break;
    }
  }
}

// shift-invert operator (Lanczos on H^{-1} via inner CG) -- copied from the bench.
template<class FieldF>
class InverseHermOp : public LinearFunction<FieldF>
{
  LinearOperatorBase<FieldF> &_H;
  RealD _tol; int _maxit;
public:
  using LinearFunction<FieldF>::operator();
  InverseHermOp(LinearOperatorBase<FieldF> &H, RealD tol, int maxit) : _H(H), _tol(tol), _maxit(maxit) {}
  void operator()(const FieldF &in, FieldF &out){
    RealD n = norm2(in);
    if(!std::isfinite(n) || n==0.0){ out = Zero(); return; } // NaN/breakdown guard
    ConjugateGradient<FieldF> CG(_tol, _maxit, false);
    out = Zero();
    CG(_H, in, out);
  }
};

// MultiRHSDeflation -> LinearFunction adaptor (batched guesser) -- from the bench.
template<class FieldF>
class MrhsDeflationGuesser : public LinearFunction<FieldF>
{
  MultiRHSDeflation<FieldF> &defl;
public:
  using LinearFunction<FieldF>::operator();
  MrhsDeflationGuesser(MultiRHSDeflation<FieldF> &d) : defl(d) {}
  void operator()(const FieldF &, FieldF &){ GRID_ASSERT(0); } // batch-only
  void operator()(const std::vector<FieldF> &in, std::vector<FieldF> &out){
    defl.DeflateSources(const_cast<std::vector<FieldF>&>(in), out);
  }
};

void StochasticDilutedSource(GridParallelRNG &RNG, LatticePropagator &source,
                             GridBase *rbgrid, const int tslice, const int eo)
{
  GridBase *grid = source.Grid();
  RealD nrm = 1.0/sqrt(2.0);
  LatticeInteger t(grid);  LatticeCoordinate(t, Tdir);
  LatticeComplex zz(grid);    zz = Zero();
  LatticeComplex xi(grid), xi_rb(rbgrid), xi_eo(grid);
  source = Zero();
  for(int s=0; s<Nd; s++){
    for(int c=0; c<Nc; c++){
      bernoulli(RNG, xi);
      xi = (2.0*xi - Complex(1.0,1.0))*nrm;
      xi = where(t==Integer(tslice), xi, zz);
      xi_eo = Zero();
      pickCheckerboard(eo, xi_rb, xi);
      setCheckerboard(xi_eo, xi_rb);
      auto spin_block = peekSpin(source, s, s);
      pokeColour(spin_block, xi_eo, c, c);
      pokeSpin(source, spin_block, s, s);
    }
  }
}

void TraceField(LatticeComplex& meson_CF, const Gamma::Algebra& gam,
                LatticePropagator &psi, LatticePropagator &eta)
{
  meson_CF = trace(Gamma(gam)*psi*adj(eta));
}

// Batched, deflated, mixed-precision solve of all Nd*Nc spin-color columns of one
// diluted source. Replaces the original per-(s,c) SchurRedBlackDiagMooeeSolve loop.
//   D, D_f  : double / single Mobius actions
//   schur   : SchurRedBlackDiagMooeeSolve (used for batched red-black source/recon)
//   bCG     : MixedPrecisionConjugateGradientBatched with the deflation guesser set
template<class Action, class ActionF, class Schur, class BatchedCG>
void SolveBatchedDeflated(Action &D, ActionF &D_f, Schur &schur, BatchedCG &bCG,
                          LatticePropagator &source, LatticePropagator &propagator)
{
  GridBase *UGrid  = D.GaugeGrid();
  GridBase *FGrid  = D.FermionGrid();
  GridBase *FrbGrid= D.FermionRedBlackGrid();
  const int N = Nd*Nc;

  std::vector<LatticeFermion> src5 (N, FGrid);
  std::vector<LatticeFermion> sol5 (N, FGrid);
  std::vector<LatticeFermion> src_o(N, FrbGrid);
  std::vector<LatticeFermion> sol_o(N, FrbGrid);

  LatticeFermion src4(UGrid);
  for(int s=0; s<Nd; s++){
    for(int c=0; c<Nc; c++){
      int b = s*Nc + c;
      PropToFerm<Action>(src4, source, s, c);
      D.ImportPhysicalFermionSource(src4, src5[b]);
    }
  }

  // red-black prep -> odd-checkerboard sources. NB the base-class vector overloads
  // are HIDDEN by SchurRedBlackDiagMooeeSolve's single-field overrides, so do the
  // per-RHS prep here (identical to what the vector overload does internally) and
  // batch only the solve.
  LatticeFermion tmp_e(FrbGrid);
  for(int b=0; b<N; b++) schur.RedBlackSource(D, src5[b], tmp_e, src_o[b]);
  for(int b=0; b<N; b++) sol_o[b] = Zero();

  // the heavy step: 16-RHS batched mixed-prec solve, deflated via bCG's guesser
  bCG(src_o, sol_o);

  // reconstruct full 5d solution (per-RHS), then export the physical 4d solution
  for(int b=0; b<N; b++){
    pickCheckerboard(Even, tmp_e, src5[b]);
    schur.RedBlackSolution(D, sol_o[b], tmp_e, sol5[b]);
  }

  LatticeFermion result4(UGrid);
  for(int s=0; s<Nd; s++){
    for(int c=0; c<Nc; c++){
      int b = s*Nc + c;
      D.ExportPhysicalFermionSolution(sol5[b], result4);
      FermToProp<Action>(propagator, result4, s, c);
    }
  }
}

int main(int argc, char** argv)
{
  double mass = 0.4, beta = 11.08;
  std::string dir = "", obsdir = "";
  ParseArgs(argc, argv, mass, beta, dir, obsdir);

  const int Ls = 16;
  Grid_init(&argc, &argv);

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

  RealD M5=1.5, b=1.5, c=0.5;
  std::vector<Complex> boundary = {1,1,1,-1};
  typedef MobiusFermionD FermionAction;
  typedef MobiusFermionF FermionActionF;
  FermionAction::ImplParams Params(boundary);

  // ---- tuning knobs (see disc_tuning_routine_claude.md) ----
  const int    Nev      = env_int   ("NEV",      150);
  const int    Nstop    = env_int   ("NSTOP",    Nev);
  const int    Nk       = env_int   ("NK",       Nev);
  const int    Nm       = env_int   ("NM",       Nev*8/5);
  const double inv_tol  = env_double("INV_TOL",  1.0e-5);
  const int    inv_maxit= env_int   ("INV_MAXIT",50000);
  const double eresid   = env_double("ERESID",   1.0e-5);
  const int    eig_maxit= env_int   ("MAXITER",  300);
  const double inner_tol= env_double("INNER_TOL",1.0e-4);
  const int    maxinner = env_int   ("MAXINNER", 10000);
  const int    maxouter = env_int   ("MAXOUTER", 50);
  const int    maxpatch = env_int   ("MAXPATCH", 1000);

  WilsonGaugeActionR Waction(beta);
  std::filesystem::create_directories(obsdir);

  LatticeGaugeField  Umu(UGrid);
  LatticeGaugeFieldF Umu_f(UGrid_f);
  GridParallelRNG    RNG4(UGrid);

  int conf_min, conf_max, interval;
  std::string lat_prefix;
  {
    std::vector<int> confs;
    const std::string suffix = "_lat.";
    for(const auto& entry : std::filesystem::directory_iterator(dir)){
      const std::string fname = entry.path().filename().string();
      const auto pos = fname.rfind(suffix);
      if(pos == std::string::npos) continue;
      const std::string numstr = fname.substr(pos + suffix.size());
      if(numstr.empty() || !std::all_of(numstr.begin(), numstr.end(), ::isdigit)) continue;
      if(lat_prefix.empty()) lat_prefix = fname.substr(0, pos + suffix.size());
      confs.push_back(std::stoi(numstr));
    }
    assert(!confs.empty());
    std::sort(confs.begin(), confs.end());
    conf_min = confs.front();
    interval = (confs.size() >= 2) ? confs[1] - confs[0] : 1;
    conf_max = confs.back() + interval;
  }

  const std::vector<Gamma::Algebra> gams = {
    Gamma::Algebra::Identity, Gamma::Algebra::Gamma5,
    Gamma::Algebra::GammaX, Gamma::Algebra::GammaY, Gamma::Algebra::GammaZ, Gamma::Algebra::GammaT,
    Gamma::Algebra::GammaXGamma5, Gamma::Algebra::GammaYGamma5,
    Gamma::Algebra::GammaZGamma5, Gamma::Algebra::GammaTGamma5,
  };
  const std::vector<std::string> gam_names = {"id","g5","gx","gy","gz","gt","gxg5","gyg5","gzg5","gtg5"};

  // graceful wall blocker (identical scheme to the original disc binary)
  long deadline = 0;
  if(const char* e = std::getenv("DISC_DEADLINE_EPOCH")) deadline = std::atol(e);
  double max_dur = 0.0;
  if(const char* e = std::getenv("DISC_TPT_SECONDS")) max_dur = std::atof(e);
  const double margin = 1.2;

  for(int conf=conf_min; conf<conf_max; conf+=interval){
    {
      bool all_done = true;
      for(int ig=0; ig<(int)gam_names.size(); ig++){
        const std::string path = obsdir + "/traces." + gam_names[ig] + "." + std::to_string(conf);
        if(!std::filesystem::exists(path)){ all_done = false; break; }
      }
      if(all_done){ std::cout << GridLogMessage << "skipping conf " << conf << " (output exists)" << std::endl; continue; }
    }
    if(deadline > 0 && max_dur > 0.0){
      uint64_t stop = 0;
      if(UGrid->IsBoss() && (double)std::time(nullptr) + margin*max_dur > (double)deadline) stop = 1;
      UGrid->GlobalSum(stop);
      if(stop){ std::cout << GridLogMessage << "blocker: stopping before conf " << conf << std::endl; break; }
    }
    const long t_cfg_start = (long)std::time(nullptr);

    {
      const std::string path = dir+"/"+lat_prefix+std::to_string(conf);
      FieldMetaData header;
      NerscIO::readConfiguration(Umu, header, path);
      RNG4.SeedUniqueString(path);
    }
    precisionChange(Umu_f, Umu);

    FermionAction  FermAct  (Umu,   *FGrid,   *FrbGrid,   *UGrid,   *UrbGrid,   mass, M5, b, c, Params);
    FermionActionF FermAct_f(Umu_f, *FGrid_f, *FrbGrid_f, *UGrid_f, *UrbGrid_f, mass, M5, b, c, Params);

    SchurDiagMooeeOperator<FermionAction,  LatticeFermion>  HermOpEO  (FermAct);
    SchurDiagMooeeOperator<FermionActionF, LatticeFermionF> HermOpEO_f(FermAct_f);

    // ---------- eigensolve ONCE per config (shift-invert IRL) ----------
    // TODO(production): save evec/eval to the per-ensemble dir and RELOAD if present
    // (skip this eigensolve) -- see disc_lma_impl_plan_claude.md directory memo.
    std::vector<RealD>           eval(Nm);
    std::vector<LatticeFermionF> evec(Nm, FrbGrid_f);
    {
      GridParallelRNG RNG5f(FGrid_f); RNG5f.SeedUniqueString(dir+"/"+lat_prefix+std::to_string(conf)+".eig");
      LatticeFermionF tmp_f(FGrid_f), lanc_src(FrbGrid_f);
      random(RNG5f, tmp_f); pickCheckerboard(Odd, lanc_src, tmp_f);
      InverseHermOp<LatticeFermionF> Hinv(HermOpEO_f, inv_tol, inv_maxit);
      PlainHermOp<LatticeFermionF>   HermOp(HermOpEO_f);
      ImplicitlyRestartedLanczos<LatticeFermionF> IRL(Hinv, HermOp, Nstop, Nk, Nm, eresid, eig_maxit);
      int Nconv = 0;
      IRL.calc(eval, evec, lanc_src, Nconv);
      std::cout << GridLogMessage << "conf " << conf << " eigensolve Nconv=" << Nconv << std::endl;
    }
    MultiRHSDeflation<LatticeFermionF> defl;
    defl.ImportEigenBasis(evec, eval, 0, Nev);
    MrhsDeflationGuesser<LatticeFermionF> guesser(defl);

    // batched mixed-prec solver (deflated) + a Schur wrapper for red-black prep/recon
    MixedPrecisionConjugateGradientBatched<LatticeFermion,LatticeFermionF>
      bCG(1.0e-8, maxinner, maxouter, maxpatch, FrbGrid_f, HermOpEO_f, HermOpEO);
    bCG.InnerTolerance = inner_tol;
    bCG.useGuesser(guesser);
    ConjugateGradient<LatticeFermion> dummyCG(1.0e-8, 100000); // unused; SchurRedBlack ctor needs an OperatorFunction
    SchurRedBlackDiagMooeeSolve<LatticeFermion> schur(dummyCG);

    std::vector<LatticeComplex> res(gam_names.size(), LatticeComplex(UGrid));
    for(auto &r : res) r = Zero();

    for(int t=0; t<Nt; t++){
      for(int eo=0; eo<=1; eo++){
        LatticePropagator source(UGrid);
        StochasticDilutedSource(RNG4, source, UrbGrid, t, eo);
        LatticePropagator StochProp(UGrid);
        SolveBatchedDeflated(FermAct, FermAct_f, schur, bCG, source, StochProp);
        LatticeComplex Trace_CF(UGrid);
        for(int ig=0; ig<(int)gam_names.size(); ig++){
          TraceField(Trace_CF, gams[ig], StochProp, source);
          res[ig] = res[ig] + Trace_CF;
        }
      }
    }

    for(int ig=0; ig<(int)gam_names.size(); ig++){
      const std::string path = obsdir + "/traces." + gam_names[ig] + "." + std::to_string(conf);
      emptyUserRecord record;
      ScidacWriter WR(UGrid->IsBoss());
      WR.open(path);
      WR.writeScidacFieldRecord(res[ig], record);
      WR.close();
    }

    uint64_t dur = 0;
    if(UGrid->IsBoss()) dur = (uint64_t)((long)std::time(nullptr) - t_cfg_start);
    UGrid->GlobalSum(dur);
    if((double)dur > max_dur) max_dur = (double)dur;
    std::cout << GridLogMessage << "conf " << conf << " took " << dur << "s" << std::endl;
  }

  Grid_finalize();
}
