#include <filesystem>
#include <getopt.h>
#include <ctime>    // std::time for the wall-time blocker
#include <cstdlib>  // std::getenv / std::atol for the wall-time blocker

#include <Grid/Grid.h>

// Mixed-precision variant of disc_multipleGamma_binary_claude.cc.
// The ONLY physics change is the linear solver: the inner CG iterations run in
// SINGLE precision with a DOUBLE reliable-update outer loop
// (MixedPrecisionConjugateGradient), keeping the SAME 1e-8 outer residual. On
// MI300A this roughly halves the per-config wall time (FP32 ~2x FP64, the Mobius
// apply is bandwidth-bound). Source, dilution, contraction, I/O and the
// wall-time blocker are byte-for-byte identical to the double-precision binary.
//
// Mixed-precision reliable-update CG: M. A. Clark et al., arXiv:0911.3191
// (Comput.Phys.Commun. 181 (2010) 1517); reliable update: Sleijpen & van der
// Vorst, Computing 56 (1996) 141. Grid class MixedPrecisionConjugateGradient.

using namespace std;
using namespace Grid;

void ParseArgs(int argc, char** argv,
               double& mass,
               double& beta,
               std::string& dir,
               std::string& obsdir)
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
  int option_index, opt;
  while ((opt = getopt_long(argc, argv, short_opts, long_opts, &option_index)) != -1){
    switch (opt) {
    case 'm': mass   = std::stod(optarg); break;
    case 'b': beta   = std::stod(optarg); break;
    case 'd': dir    = optarg;            break;
    case 'o': obsdir = optarg;            break;
    default: break;
    }
  }
}


inline bool is_exist (const std::string& name) {
return ( access( name.c_str(), F_OK ) != -1 );
}

void StochasticDilutedSource(GridParallelRNG &RNG, LatticePropagator &source,
                             GridBase *rbgrid, const int tslice, const int eo)
{
  GridBase *grid = source.Grid();
  RealD nrm = 1.0/sqrt(2.0);

  LatticeInteger t(grid);
  LatticeCoordinate(t, Tdir);

  LatticeComplex zz(grid);    zz    = Zero();
  LatticeComplex xi(grid);
  LatticeComplex xi_rb(rbgrid);
  LatticeComplex xi_eo(grid);

  source = Zero();
  for(int s=0; s<Nd; s++){
    for(int c=0; c<Nc; c++){
      bernoulli(RNG, xi);
      xi = (2.0*xi - Complex(1.0,1.0))*nrm;
      xi = where(t==Integer(tslice), xi, zz);

      xi_eo = Zero();
      pickCheckerboard(eo, xi_rb, xi);
      setCheckerboard(xi_eo, xi_rb);

      // place xi_eo on the diagonal (s,s;c,c) of the source propagator
      auto spin_block = peekSpin(source, s, s);
      pokeColour(spin_block, xi_eo, c, c);
      pokeSpin(source, spin_block, s, s);
    }
  }
}


// Adapt a LinearFunction (e.g. MixedPrecisionConjugateGradient, which already
// owns its single+double operators) to the OperatorFunction interface that
// SchurRedBlack*Solve expects (3-arg (Linop,in,out)). The Linop the Schur
// wrapper passes is IGNORED -- mCG holds the identical Schur operator itself.
template<class Field>
class LinearFunctionAsOperatorFunction : public OperatorFunction<Field>
{
  LinearFunction<Field> &_fn;
public:
  using OperatorFunction<Field>::operator();
  LinearFunctionAsOperatorFunction(LinearFunction<Field> &fn) : _fn(fn) {}
  void operator()(LinearOperatorBase<Field> &Linop, const Field &in, Field &out){
    _fn(in, out);
  }
};

template<class Action, class ActionF>
void Solve(Action &D, ActionF &D_f, LatticePropagator &source, LatticePropagator &propagator)
{
  GridBase *UGrid = D.GaugeGrid();
  GridBase *FGrid = D.FermionGrid();

  LatticeFermion src4  (UGrid);
  LatticeFermion src5  (FGrid);
  LatticeFermion result5(FGrid);
  LatticeFermion result4(UGrid);

  // --- double-precision reference solver (kept for A/B; see the unchanged
  // --- double-precision binary disc_multipleGamma_binary_claude.cc) ---
  // ConjugateGradient<LatticeFermion> CG(1.0e-8,100000);
  // SchurRedBlackDiagMooeeSolve<LatticeFermion> schur(CG);

  // --- mixed-precision solver: single inner / double reliable-update outer,
  // --- same 1e-8 outer residual (arXiv:0911.3191). The Schur-preconditioned
  // --- (even/odd) operator is built in both precisions; the single one drives
  // --- the inner CG, the double one the outer correction.
  SchurDiagMooeeOperator<Action, LatticeFermion>   HermOpEO  (D);
  SchurDiagMooeeOperator<ActionF, LatticeFermionF> HermOpEO_f(D_f);
  MixedPrecisionConjugateGradient<LatticeFermion, LatticeFermionF>
    mCG(1.0e-8, 10000, 50, D_f.FermionRedBlackGrid(), HermOpEO_f, HermOpEO);
  LinearFunctionAsOperatorFunction<LatticeFermion> mCGop(mCG);
  SchurRedBlackDiagMooeeSolve<LatticeFermion> schur(mCGop);

  ZeroGuesser<LatticeFermion> ZG; // Could be a DeflatedGuesser if have eigenvectors
  for(int s=0;s<Nd;s++){
    for(int c=0;c<Nc;c++){
      PropToFerm<Action>(src4,source,s,c);

      D.ImportPhysicalFermionSource(src4,src5);

      result5=Zero();
      schur(D,src5,result5,ZG);
      std::cout<<GridLogMessage
               <<"spin "<<s<<" color "<<c
               <<" norm2(src5d) "   <<norm2(src5)
               <<" norm2(result5d) "<<norm2(result5)<<std::endl;

      D.ExportPhysicalFermionSolution(result5,result4);

      FermToProp<Action>(propagator,result4,s,c);
    }
  }
}

void TraceField(LatticeComplex& meson_CF,
                const Gamma::Algebra& gam,
                LatticePropagator &psi, LatticePropagator &eta)
{
  meson_CF = trace(Gamma(gam)*psi*adj(eta));
}


int main (int argc, char ** argv)
{
  double mass   = 0.4;
  double beta   = 11.08;
  std::string dir    = "/mnt/baracuda_14/grid_claude/16c";
  std::string obsdir = "/mnt/baracuda_14/grid_claude/16c_obs";
  ParseArgs(argc, argv, mass, beta, dir, obsdir);

  const int Ls=16;
  Grid_init(&argc,&argv);

  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(),
                                                                   GridDefaultSimd(Nd,vComplex::Nsimd()),
                                                                   GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);

  // single-precision grids for the inner CG of the mixed-precision solver.
  GridCartesian         * UGrid_f   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(),
                                                                     GridDefaultSimd(Nd,vComplexF::Nsimd()),
                                                                     GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid_f = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid_f);
  GridCartesian         * FGrid_f   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid_f);
  GridRedBlackCartesian * FrbGrid_f = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid_f);

  const int Nt = UGrid->_fdimensions[Tdir];

  std::cout << "# mass=" << mass << " beta=" << beta << " Nt=" << Nt << std::endl;
  std::cout << "# dir = " << dir << " obsdir = " << obsdir << std::endl;
  WilsonGaugeActionR Waction(beta);

  std::filesystem::create_directories(obsdir);

  RealD M5=1.5;
  RealD b=1.5;
  RealD c=0.5;
  std::vector<Complex> boundary = {1,1,1,-1};
  typedef MobiusFermionD FermionAction;
  typedef MobiusFermionF FermionActionF;
  FermionAction::ImplParams Params(boundary);

  LatticeGaugeField  Umu(UGrid);
  LatticeGaugeFieldF Umu_f(UGrid_f);   // single-precision copy, refreshed per config
  GridParallelRNG  RNG4(UGrid);

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
  std::cout << "conf_min=" << conf_min << " conf_max=" << conf_max
            << " interval=" << interval << std::endl;


  const std::vector<Gamma::Algebra> gams = {
    Gamma::Algebra::Identity,
    Gamma::Algebra::Gamma5,
    Gamma::Algebra::GammaX,
    Gamma::Algebra::GammaY,
    Gamma::Algebra::GammaZ,
    Gamma::Algebra::GammaT,
    Gamma::Algebra::GammaXGamma5,
    Gamma::Algebra::GammaYGamma5,
    Gamma::Algebra::GammaZGamma5,
    Gamma::Algebra::GammaTGamma5,
  };
  const std::vector<std::string> gam_names = {"id", "g5", "gx", "gy", "gz", "gt", "gxg5", "gyg5", "gzg5", "gtg5"};

  // ----------------- graceful wall-time blocker -----------------
  // submit_disc passes DISC_DEADLINE_EPOCH (epoch s, from `flux job timeleft`);
  // 0 disables. Rather than a hand-tuned per-config time we MEASURE each config's
  // wall time as we go and keep the largest seen; before starting the next
  // (compute) config we stop cleanly if now + margin*max_dur would pass the
  // deadline. DISC_TPT_SECONDS (optional) only bootstraps the estimate so the
  // FIRST config of a job is guarded too. The decision and the measured duration
  // are taken on the boss and broadcast (GlobalSum) so every rank breaks together
  // -- the loop body below is collective. Resubmit continues (completed configs
  // are self-skipped above).
  long deadline = 0;
  if(const char* e = std::getenv("DISC_DEADLINE_EPOCH")) deadline = std::atol(e);
  double max_dur = 0.0;       // largest per-config wall time seen so far (s)
  if(const char* e = std::getenv("DISC_TPT_SECONDS")) max_dur = std::atof(e); // optional bootstrap
  const double margin = 1.2;  // generous safety factor on the estimate

  for(int conf=conf_min; conf<conf_max; conf+=interval){
    {
      bool all_done = true;
      for(int ig=0; ig<(int)gam_names.size(); ig++){
        const std::string path = obsdir + "/traces." + gam_names[ig] + "." + std::to_string(conf);
        if(!std::filesystem::exists(path)){ all_done = false; break; }
      }
      if(all_done){
        std::cout << GridLogMessage << "skipping conf " << conf << " (output exists)" << std::endl;
        continue;
      }
    }
    // graceful stop if the next config would not finish before the deadline.
    // (max_dur == 0 with no bootstrap means "no estimate yet" -> always run the
    // first config so a fresh job makes progress.)
    if(deadline > 0 && max_dur > 0.0){
      uint64_t stop = 0;
      if(UGrid->IsBoss()){
        if((double)std::time(nullptr) + margin*max_dur > (double)deadline) stop = 1;
      }
      UGrid->GlobalSum(stop);
      if(stop){
        std::cout << GridLogMessage << "blocker: est " << (long)(margin*max_dur)
                  << "s for next config exceeds deadline; stopping gracefully before conf "
                  << conf << std::endl;
        break;
      }
    }

    const long t_cfg_start = (long)std::time(nullptr);

    {
      const std::string path = dir+"/"+lat_prefix+std::to_string(conf);
      FieldMetaData header;
      NerscIO::readConfiguration(Umu, header, path);
      RNG4.SeedUniqueString(path);
    }
    // refresh the single-precision gauge copy for this config's inner solves.
    precisionChange(Umu_f, Umu);

    FermionAction  FermAct  (Umu,   *FGrid,   *FrbGrid,   *UGrid,   *UrbGrid,   mass, M5, b, c, Params);
    FermionActionF FermAct_f(Umu_f, *FGrid_f, *FrbGrid_f, *UGrid_f, *UrbGrid_f, mass, M5, b, c, Params);

    std::vector<LatticeComplex> res(gam_names.size(), LatticeComplex(UGrid));
    for(auto &r : res) r = Zero();

    for(int t=0; t<Nt; t++){
      for(int eo=0; eo<=1; eo++){
        LatticePropagator source(UGrid);

        StochasticDilutedSource(RNG4, source, UrbGrid, t, eo);

        LatticePropagator StochProp(UGrid);
        Solve(FermAct, FermAct_f, source, StochProp);

        LatticeComplex Trace_CF( UGrid );
        for(int ig=0; ig<(int)gam_names.size(); ig++){
          TraceField(Trace_CF, gams[ig], StochProp, source);
          res[ig] = res[ig] + Trace_CF;
        }
      }
    } // for dilute

    for(int ig=0; ig<(int)gam_names.size(); ig++){
      const std::string path = obsdir + "/traces." + gam_names[ig] + "." + std::to_string(conf);
      emptyUserRecord record;
      ScidacWriter WR(UGrid->IsBoss());
      WR.open(path);
      WR.writeScidacFieldRecord(res[ig], record);
      WR.close();
    }

    // update the per-config wall-time estimate for the blocker. Measure on the
    // boss and broadcast (GlobalSum) so max_dur is identical on every rank for
    // the next iteration's collective stop decision.
    uint64_t dur = 0;
    if(UGrid->IsBoss()) dur = (uint64_t)((long)std::time(nullptr) - t_cfg_start);
    UGrid->GlobalSum(dur);
    if((double)dur > max_dur) max_dur = (double)dur;
    std::cout << GridLogMessage << "conf " << conf << " took " << dur
              << "s (max so far " << (long)max_dur << "s)" << std::endl;

  } // conf

  Grid_finalize();
}
