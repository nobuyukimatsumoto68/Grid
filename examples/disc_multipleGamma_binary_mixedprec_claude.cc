#include <filesystem>
#include <getopt.h>
#include <ctime>    // std::time for the wall-time blocker
#include <cstdlib>  // std::getenv / std::atol for the wall-time blocker
#include <cassert>  // DISC_TPART divisibility assert (chunk 1)

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

// CHUNK-1 (round-1 plan): K-partition TIME dilution knob, same semantics as the header
// version in disc_lma_v2_common_claude.h -- tpart=1 (default) is bit-identical to the
// original; tpart=K puts the same Z4 noise on the K slices {tslice + k*Nt/K}.
void StochasticDilutedSource(GridParallelRNG &RNG, LatticePropagator &source,
                             GridBase *rbgrid, const int tslice, const int eo,
                             const int tpart = 1, const int Nt = 0)
{
  GridBase *grid = source.Grid();
  RealD nrm = 1.0/sqrt(2.0);

  LatticeInteger t(grid);
  LatticeCoordinate(t, Tdir);

  LatticeComplex zz(grid);    zz    = Zero();
  LatticeComplex xi(grid);
  LatticeComplex xim(grid);
  LatticeComplex xi_rb(rbgrid);
  LatticeComplex xi_eo(grid);

  source = Zero();
  for(int s=0; s<Nd; s++){
    for(int c=0; c<Nc; c++){
      bernoulli(RNG, xi);
      xi = (2.0*xi - Complex(1.0,1.0))*nrm;
      // xi = where(t==Integer(tslice), xi, zz);   // original single-slice mask (tpart==1 path below)
      if(tpart <= 1){
        xi = where(t==Integer(tslice), xi, zz);
      } else {
        xim = Zero();
        for(int k=0; k<tpart; k++){
          const int tk = tslice + k*(Nt/tpart);
          xim = where(t==Integer(tk), xi, xim);
        }
        xi = xim;
      }

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
  // loosen the inner single-CG tolerance (default 1e-8 forces ~3 full outer iters
  // and kills the mixed-prec gain; 1e-4 gives the measured ~1.4x). Outer tol stays
  // 1e-8. Env-overridable per disc_tuning_routine_claude.md.
  { const char* e = std::getenv("INNER_TOL"); mCG.InnerTolerance = e ? std::atof(e) : 1.0e-4; }
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

  // ---- second (and further) noise hit: DISC_HIT=N>0 seeds the noise with cfg_path + "_hit<N>"
  // (independent of hit 0) and writes into <obsdir>_hit<N>/ (same file names), so a hit-N run can
  // never touch the hit-0 outputs. DISC_HIT unset/0 = original behaviour (same seed, same obsdir).
  int disc_hit = 0;
  if(const char* e = std::getenv("DISC_HIT")) disc_hit = std::atoi(e);
  std::string hit_tag = "";
  if(disc_hit > 0){
    hit_tag = "_hit" + std::to_string(disc_hit);
    while(obsdir.size() > 1 && obsdir.back() == '/') obsdir.pop_back();
    obsdir = obsdir + hit_tag;
  }

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
    // env overrides for the MEASUREMENT set (as in the lma_v2 binary): CONF_MIN skips
    // thermalization, CONF_STRIDE picks the decorrelated subset, CONF_MAX caps the range.
    if(const char* e = std::getenv("CONF_MIN"))     conf_min = std::atoi(e);
    if(const char* e = std::getenv("CONF_MAX"))     conf_max = std::atoi(e);
    if(const char* e = std::getenv("CONF_STRIDE")){ int s = std::atoi(e); if(s > 0) interval = s; }
  }
  std::cout << "conf_min=" << conf_min << " conf_max=" << conf_max
            << " interval=" << interval << std::endl;

  // DISC_NGROUPS: 0 = all Nt*2 (t,eo) dilution groups (PRODUCTION). >0 = TIMING MODE -- do only
  // the first N groups and SKIP the trace write (a partial loop is NOT a valid output; this
  // measures the per-group mixed-batched solve wall, e.g. for the 32c break-even bench).
  int ngroups_max = 0;
  { const char* e = std::getenv("DISC_NGROUPS"); if(e) ngroups_max = std::atoi(e); }

  // CHUNK-1 knobs (round-1 plan), same semantics as the lma_v2 binary: DISC_TPART=K
  // sources K equally-spaced slices per solve (default 1 = original); DISC_SEED_STR
  // (default empty) appended to the per-config RNG seed -- K-scan resampling only.
  int tpart = 1;
  { const char* e = std::getenv("DISC_TPART"); if(e) tpart = std::atoi(e); }
  assert(tpart >= 1 && (Nt % tpart) == 0 && "DISC_TPART must divide Nt");
  std::string seed_suffix = "";
  { const char* e = std::getenv("DISC_SEED_STR"); if(e) seed_suffix = e; }
  if(tpart > 1) std::cout << GridLogMessage << "# DISC_TPART=" << tpart << std::endl;


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
// <<<<<<< HEAD
//       // RNG4.SeedUniqueString(path);   // original; DISC_SEED_STR="" reproduces it exactly
//       RNG4.SeedUniqueString(path + seed_suffix);
// =======
      // RNG4.SeedUniqueString(path);
      RNG4.SeedUniqueString(path + hit_tag);
// >>>>>>> bc333a071a0b8b682ab813caeee564f98f568183
    }
    // refresh the single-precision gauge copy for this config's inner solves.
    precisionChange(Umu_f, Umu);

    FermionAction  FermAct  (Umu,   *FGrid,   *FrbGrid,   *UGrid,   *UrbGrid,   mass, M5, b, c, Params);
    FermionActionF FermAct_f(Umu_f, *FGrid_f, *FrbGrid_f, *UGrid_f, *UrbGrid_f, mass, M5, b, c, Params);

    std::vector<LatticeComplex> res(gam_names.size(), LatticeComplex(UGrid));
    for(auto &r : res) r = Zero();

    int ngroups_done = 0;
    // DISC_TPART=K: each solve sources K slices {t + k*Nt/K} -> tbase loop runs Nt/K.
    for(int t=0; t<Nt/tpart && (ngroups_max==0 || ngroups_done<ngroups_max); t++){
      for(int eo=0; eo<=1 && (ngroups_max==0 || ngroups_done<ngroups_max); eo++){
        LatticePropagator source(UGrid);

        // StochasticDilutedSource(RNG4, source, UrbGrid, t, eo);   // original single-slice call
        StochasticDilutedSource(RNG4, source, UrbGrid, t, eo, tpart, Nt);

        LatticePropagator StochProp(UGrid);
        Solve(FermAct, FermAct_f, source, StochProp);

        LatticeComplex Trace_CF( UGrid );
        for(int ig=0; ig<(int)gam_names.size(); ig++){
          TraceField(Trace_CF, gams[ig], StochProp, source);
          res[ig] = res[ig] + Trace_CF;
        }
        ngroups_done++;
      }
    } // for dilute

    // Skip the trace write in timing mode (partial loop -> not a valid measurement; do not
    // leave traces that would self-skip a later real run).
    if(ngroups_max == 0){
    for(int ig=0; ig<(int)gam_names.size(); ig++){
      const std::string path = obsdir + "/traces." + gam_names[ig] + "." + std::to_string(conf);
      emptyUserRecord record;
      ScidacWriter WR(UGrid->IsBoss());
      WR.open(path);
      WR.writeScidacFieldRecord(res[ig], record);
      WR.close();
    }
    // p=0 loop per (gamma,t) to the log: cheap; the K-scan variance analysis +
    // unbiasedness cross-check read these lines instead of the Scidac fields.
    for(int ig=0; ig<(int)gam_names.size(); ig++){
      std::vector<TComplex> tv;
      sliceSum(res[ig], tv, Tdir);
      for(int tt=0; tt<Nt; tt++)
        std::cout << GridLogMessage << "# LOOPP0 " << gam_names[ig] << " conf " << conf
                  << " t " << tt << " " << real(TensorRemove(tv[tt]))
                  << " " << imag(TensorRemove(tv[tt])) << std::endl;
    }
    } else {
      std::cout << GridLogMessage << "TIMING MODE: did " << ngroups_done << "/" << (Nt*2)
                << " (t,eo) groups, NO traces written" << std::endl;
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
