#include "disc_lma_v2_common_claude.h"
#include <filesystem>
#include <getopt.h>
#include <ctime>    // wall-time blocker
#include <cstdlib>  // getenv/atol for the blocker

// disc LMA PRODUCTION binary (chunk D). Config-loop disconnected-loop measurement with LMA:
// per config -> mixed-prec Cheby+RR eigensolve (reading the per-ENSEMBLE eigref) -> physical A2A
// -> the LMA loop = exact noise-free L^low + source-PROJECTED stochastic high part, written in the
// SAME Scidac format `traces.<gam>.<conf>` as disc_multipleGamma_binary_claude.cc (drop-in; the
// downstream pipeline is unchanged). Shared machinery: disc_lma_v2_common_claude.h. Validated
// prototype: disc_lma_estimator_bench_v2_claude.cc. Never edit the original disc binary.
//
// Decisions (disc_lma_production_impl_plan_claude.md): RNG4 = SeedUniqueString(config_path) per
// config (config-unique, deterministic); output = combined LMA loop only; evec CHECKPOINT
// (evec.<conf>.scidac + eval.<conf>.h5) saved after the eigensolve and reloaded on rerun to SKIP it.
//
// LMA: DeGrand-Schaefer hep-lat/0401011; Giusti et al hep-lat/0402002. A2A: Foley et al hep-lat/0505023.

using namespace std;
using namespace Grid;

// ---- CLI: mass / beta / config dir / obs dir / eigref (env knobs via ReadLMAEigParams) -------
static void ParseArgsProd(int argc, char** argv, double& mass, double& beta,
                          std::string& dir, std::string& obsdir, std::string& eigref)
{
  const char* const short_opts = ":m:b:d:o:e:";   // no '+' -> permute; Grid_init runs FIRST
  const option long_opts[] = {
    {"mass",   required_argument, nullptr, 'm'},
    {"beta",   required_argument, nullptr, 'b'},
    {"dir",    required_argument, nullptr, 'd'},
    {"obsdir", required_argument, nullptr, 'o'},
    {"eigref", required_argument, nullptr, 'e'},
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
      case 'e': eigref = optarg;            break;
      default: break;
    }
  }
}


int main(int argc, char** argv)
{
  Grid_init(&argc, &argv);   // FIRST: consume Grid's --grid/--mpi/... args

  double mass = 0.01;
  double beta = 11.08;
  std::string dir    = "/mnt/baracuda_14/grid_claude/16c";
  std::string obsdir = "/mnt/baracuda_14/grid_claude/16c_obs";
  std::string eigref = "";
  ParseArgsProd(argc, argv, mass, beta, dir, obsdir, eigref);

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

  std::cout << GridLogMessage << "# disc LMA production: mass=" << mass << " beta=" << beta
            << " Nt=" << Nt << " dir=" << dir << " obsdir=" << obsdir
            << " eigref=" << eigref << std::endl;

  std::filesystem::create_directories(obsdir);

  // ---- per-ENSEMBLE spectral landscape (Cheby window); computed once by disc_lma_eigref_v2 ----
  std::vector<RealD> lambda_ref;
  RealD lambda_max_ref = 0.0;
  bool have_ref = false;
  if(!eigref.empty()) have_ref = ReadEigref(eigref, lambda_ref, lambda_max_ref);
  if(have_ref)
    std::cout << GridLogMessage << "# eigref loaded: " << lambda_ref.size()
              << " evals, lambda_max=" << lambda_max_ref << std::endl;
  else
    std::cout << GridLogMessage << "# WARNING: no eigref -> Cheby hi from PowerMethod, lo from CHEB_LO"
              << std::endl;

  LatticeGaugeField  Umu(UGrid);
  LatticeGaugeFieldF Umu_f(UGrid_f);
  GridParallelRNG    RNG4(UGrid);   // noise RNG, re-seeded per config (SeedUniqueString)

  const std::vector<Gamma::Algebra> gams = {
    Gamma::Algebra::Identity, Gamma::Algebra::Gamma5,
    Gamma::Algebra::GammaX,   Gamma::Algebra::GammaY,
    Gamma::Algebra::GammaZ,   Gamma::Algebra::GammaT,
    Gamma::Algebra::GammaXGamma5, Gamma::Algebra::GammaYGamma5,
    Gamma::Algebra::GammaZGamma5, Gamma::Algebra::GammaTGamma5,
  };
  const std::vector<std::string> gam_names = {"id","g5","gx","gy","gz","gt","gxg5","gyg5","gzg5","gtg5"};
  const int ngam = (int)gam_names.size();

  // ---- config discovery (scan dir for *_lat.NNNN) -- as in the reference disc binary ----
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
  std::cout << GridLogMessage << "# conf_min=" << conf_min << " conf_max=" << conf_max
            << " interval=" << interval << std::endl;

  // ---- graceful wall-time blocker (as in the reference disc binary) ----
  long deadline = 0;
  if(const char* e = std::getenv("DISC_DEADLINE_EPOCH")) deadline = std::atol(e);
  double max_dur = 0.0;
  if(const char* e = std::getenv("DISC_TPT_SECONDS")) max_dur = std::atof(e);
  const double margin = 1.2;

  for(int conf=conf_min; conf<conf_max; conf+=interval){
    // self-skip: all gamma outputs already present
    {
      bool all_done = true;
      for(int ig=0; ig<ngam; ig++){
        const std::string path = obsdir + "/traces." + gam_names[ig] + "." + std::to_string(conf);
        if(!std::filesystem::exists(path)){ all_done = false; break; }
      }
      if(all_done){
        std::cout << GridLogMessage << "# skipping conf " << conf << " (output exists)" << std::endl;
        continue;
      }
    }
    // wall blocker: stop cleanly if the next config would overrun the deadline
    if(deadline > 0 && max_dur > 0.0){
      uint64_t stop = 0;
      if(UGrid->IsBoss()){
        if((double)std::time(nullptr) + margin*max_dur > (double)deadline) stop = 1;
      }
      UGrid->GlobalSum(stop);
      if(stop){
        std::cout << GridLogMessage << "# blocker: stopping before conf " << conf << std::endl;
        break;
      }
    }

    const long t_cfg_start = (long)std::time(nullptr);

    // read gauge + seed the NOISE rng from the config path (config-unique, deterministic)
    const std::string cfg_path = dir + "/" + lat_prefix + std::to_string(conf);
    {
      FieldMetaData header;
      NerscIO::readConfiguration(Umu, header, cfg_path);
      RNG4.SeedUniqueString(cfg_path);
    }
    precisionChange(Umu_f, Umu);

    FermionAction  D  (Umu,   *FGrid,   *FrbGrid,   *UGrid,   *UrbGrid,   mass, M5, b, c, Params);
    FermionActionF D_f(Umu_f, *FGrid_f, *FrbGrid_f, *UGrid_f, *UrbGrid_f, mass, M5, b, c, Params);
    SchurDiagMooeeOperator<FermionAction,  LatticeFermion>  HermOpEO  (D);
    SchurDiagMooeeOperator<FermionActionF, LatticeFermionF> HermOpEO_f(D_f);

    // ---- eigenbasis: reload the per-config checkpoint, else eigensolve (Cheby+RR) and SAVE ----
    const std::string evec_file = obsdir + "/evec." + std::to_string(conf) + ".scidac";
    const std::string eval_file = obsdir + "/eval." + std::to_string(conf) + ".h5";
    std::vector<LatticeFermion> sub;
    std::vector<RealD>          eval_use;
    if(LoadEvecs(evec_file, eval_file, FrbGrid, sub, eval_use)){
      std::cout << GridLogMessage << "# conf " << conf << " reloaded " << sub.size()
                << " evecs (skipped eigensolve)" << std::endl;
    } else {
      BuildLowModes(D, D_f, HermOpEO, HermOpEO_f, P, have_ref, lambda_ref, lambda_max_ref, sub, eval_use);
      SaveEvecs(evec_file, eval_file, sub, eval_use);
      std::cout << GridLogMessage << "# conf " << conf << " eigensolved + saved " << sub.size()
                << " evecs" << std::endl;
    }

    std::vector<LatticeFermion> a, b4v, u;
    std::vector<RealD>          sigma;
    BuildA2ASet(D, HermOpEO, sub, eval_use, a, b4v, u, sigma);
    const int Nuse = (int)sub.size();

    // ---- the LMA loop field per gamma: res = exact L^low + projected high stochastic ----
    std::vector<LatticeComplex> res(ngam, LatticeComplex(UGrid));
    for(auto &r : res) r = Zero();

    // exact noise-free low part: L^low_Gamma(x) = sum_i (1/sigma_i) tr_sc[Gamma a_i b_i^dag](x)
    {
      LatticeFermion Ga(UGrid);
      for(int i=0; i<Nuse; i++){
        for(int ig=0; ig<ngam; ig++){
          Ga = Gamma(gams[ig]) * a[i];
          res[ig] = res[ig] + (1.0/sigma[i]) * localInnerProduct(b4v[i], Ga);
        }
      }
    }

    // source-projected stochastic high part: full time+eo+sc dilution, low modes projected out
    for(int t=0; t<Nt; t++){
      for(int eo=0; eo<=1; eo++){
        LatticePropagator source(UGrid);
        StochasticDilutedSource(RNG4, source, UrbGrid, t, eo);

        LatticePropagator StochPropHigh(UGrid);
        SolvePropProjected(D, HermOpEO, u, source, StochPropHigh);

        LatticeComplex tr(UGrid);
        for(int ig=0; ig<ngam; ig++){
          TraceField(tr, gams[ig], StochPropHigh, source);
          res[ig] = res[ig] + tr;
        }
      }
    }

    // ---- write the combined LMA loop per gamma (Scidac, traces.<gam>.<conf>) ----
    for(int ig=0; ig<ngam; ig++){
      const std::string path = obsdir + "/traces." + gam_names[ig] + "." + std::to_string(conf);
      emptyUserRecord record;
      ScidacWriter WR(UGrid->IsBoss());
      WR.open(path);
      WR.writeScidacFieldRecord(res[ig], record);
      WR.close();
    }

    // update the wall-time estimate (boss-measured, broadcast)
    uint64_t dur = 0;
    if(UGrid->IsBoss()) dur = (uint64_t)((long)std::time(nullptr) - t_cfg_start);
    UGrid->GlobalSum(dur);
    if((double)dur > max_dur) max_dur = (double)dur;
    std::cout << GridLogMessage << "# conf " << conf << " took " << dur
              << "s (max so far " << (long)max_dur << "s)" << std::endl;
  } // conf

  std::cout << GridLogMessage << "# disc LMA production done." << std::endl;
  Grid_finalize();
  return 0;
}
