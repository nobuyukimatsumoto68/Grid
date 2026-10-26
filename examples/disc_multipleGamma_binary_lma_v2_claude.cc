#include "disc_lma_v2_common_claude.h"
#include <filesystem>
#include <getopt.h>
#include <ctime>    // wall-time blocker
#include <cstdlib>  // getenv/atol for the blocker

// disc LMA PRODUCTION binary -- v2 (FASTER projected solve). Identical to
// disc_multipleGamma_binary_lma_claude.cc EXCEPT the per-config stochastic high part uses the
// MIXED-PRECISION BATCHED projected solve (SolvePropProjectedMixedBatched): single-inner/double-
// reliable-update CG (arXiv:0911.3191) batched over the 16 spin-colour columns (arXiv:0707.0131),
// instead of 16 plain double CGs. Outer tol 1e-8 unchanged -> bit-compatible traces.<gam>.<conf>
// (drop-in). Expected ~1.4-2x over the v1 projected solve (#1 mixed-prec + #2 batching), on top of
// the ~4x deflation-by-projection already in v1. Per config -> Cheby+RR eigensolve (eigref) ->
// physical A2A -> exact noise-free L^low + the mixed-batched projected high -> Scidac traces.
// Shared machinery: disc_lma_v2_common_claude.h. Never edit the original disc binary.
//
// Decisions: RNG4 = SeedUniqueString(config_path) per config; output = combined LMA loop only;
// evecs stay in memory BY DEFAULT. Knobs: INNER_TOL/MAXINNER/MAXOUTER/MAXPATCH (mixed CG)
// + the eigensolve env (ReadLMAEigParams).
//
// CHUNK 0 (conn_lma_impl_plan_claude.md), all default OFF -> default behavior unchanged:
//   eval.<conf>.h5 ALWAYS written to obsdir after the eigensolve (tiny; spectrum QA offline).
//   EVEC_LOAD=1  try the evec checkpoint before eigensolving (skip BuildLowModes on hit);
//   EVEC_SAVE=1  write the evec checkpoint after the eigensolve (Scidac records, EVEC_DIR
//                default obsdir -- the submit script MUST `lfs setstripe -c -1 -S 16M` it);
//   EVEC_PREC    single (default: .scidacf records, half the I/O, RR re-refine on load) | double.
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

  // ---- fixed Mobius parameters (match disc_multipleGamma_binary_claude) ------
  const int Ls = 16;
  RealD M5 = 1.5, b = 1.5, c = 0.5;
  std::vector<Complex> boundary = {1,1,1,-1};
  typedef MobiusFermionD FermionAction;
  typedef MobiusFermionF FermionActionF;
  FermionAction::ImplParams Params(boundary);

  const LMAEigParams P = ReadLMAEigParams();
  // DISC_NGROUPS: 0 = all Nt*2 (t,eo) dilution groups (PRODUCTION). >0 = TIMING MODE -- do only
  // the first N groups and SKIP the trace write (a partial loop is NOT a valid output; this is
  // just to measure the per-projected-solve wall quickly on pdebug, confirming the ~3-4x speedup).
  const int ngroups_max = env_int("DISC_NGROUPS", 0);

  // ---- CHUNK-1 knobs (round-1 plan): K-partition time dilution + scan seed suffix ----
  // DISC_TPART=K sources K equally-spaced slices per solve (factor K fewer solves);
  // default 1 = original behavior. DISC_SEED_STR (default empty) is appended to the
  // per-config RNG seed string -- ONLY for the K-scan noise resampling; production
  // leaves it unset so the noise stream is unchanged.
  const int tpart = env_int("DISC_TPART", 1);
  const char* seed_suffix_e = std::getenv("DISC_SEED_STR");
  const std::string seed_suffix = seed_suffix_e ? seed_suffix_e : "";

  // ---- CHUNK-0 evec-checkpoint knobs (see header comment; all default OFF) ----
  const int evec_save = env_int("EVEC_SAVE", 0);
  const int evec_load = env_int("EVEC_LOAD", 0);
  int evec_prec = 1;   // 1 = single (.scidacf, D-IO default), 2 = double (.scidac)
  if(const char* e = std::getenv("EVEC_PREC")){
    const std::string s(e);
    if(s == "double" || s == "2") evec_prec = 2;
  }
  std::string evec_dir = obsdir;
  if(const char* e = std::getenv("EVEC_DIR")) evec_dir = e;

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

  assert(tpart >= 1 && (Nt % tpart) == 0 && "DISC_TPART must divide Nt");

  std::cout << GridLogMessage << "# disc LMA production v2 (mixed-batched): mass=" << mass << " beta=" << beta
            << " Nt=" << Nt << " tpart=" << tpart << " dir=" << dir << " obsdir=" << obsdir
            << " eigref=" << eigref << std::endl;

  std::filesystem::create_directories(obsdir);
  if(evec_dir != obsdir) std::filesystem::create_directories(evec_dir);   // eval/evec files land here
  if(evec_save || evec_load)
    std::cout << GridLogMessage << "# evec checkpoint: save=" << evec_save << " load=" << evec_load
              << " prec=" << (evec_prec==1?"single":"double") << " dir=" << evec_dir << std::endl;

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
    // env overrides for the MEASUREMENT set: the new light ensembles are RAW HMC histories
    // (stride-4 from a COLD lat.4, plaq ~0.39 vs thermalized ~0.56) -- looping all of them both
    // hangs the eigensolve on the cold config and is far too many. CONF_MIN skips thermalization,
    // CONF_STRIDE picks the decorrelated subset (match the conn-meson set), CONF_MAX caps the range.
    if(const char* e = std::getenv("CONF_MIN"))     conf_min = std::atoi(e);
    if(const char* e = std::getenv("CONF_MAX"))     conf_max = std::atoi(e);
    if(const char* e = std::getenv("CONF_STRIDE")){ int s = std::atoi(e); if(s > 0) interval = s; }
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
// <<<<<<< HEAD
//       // RNG4.SeedUniqueString(cfg_path);   // original; DISC_SEED_STR="" reproduces it exactly
//       RNG4.SeedUniqueString(cfg_path + seed_suffix);
// =======
      // RNG4.SeedUniqueString(cfg_path);
      RNG4.SeedUniqueString(cfg_path + hit_tag);
// >>>>>>> bc333a071a0b8b682ab813caeee564f98f568183
    }
    precisionChange(Umu_f, Umu);

    FermionAction  D  (Umu,   *FGrid,   *FrbGrid,   *UGrid,   *UrbGrid,   mass, M5, b, c, Params);
    FermionActionF D_f(Umu_f, *FGrid_f, *FrbGrid_f, *UGrid_f, *UrbGrid_f, mass, M5, b, c, Params);
    SchurDiagMooeeOperator<FermionAction,  LatticeFermion>  HermOpEO  (D);
    SchurDiagMooeeOperator<FermionActionF, LatticeFermionF> HermOpEO_f(D_f);

    // ---- eigenbasis: default UNCHANGED -- eigensolve (Cheby+RR) per config, evecs in MEMORY.
    // (The old "checkpoint is a net loss" verdict assumed DOUBLE evecs on an UNSTRIPED dir;
    // with single-prec records + striping the read is ~10x cheaper than the eigensolve --
    // conn_lma_impl_plan_claude.md D-IO. Default stays OFF; the chunk-2 timing gate decides.)
    // The checkpoint exists for CONSUMERS that cannot amortize the eigensolve (standalone conn
    // LMA: 16 point solves) and for reruns. The LMA split stays exact for WHATEVER subspace is
    // used (u_i rebuilt from sub in the A2A stream), so a single-prec-stored basis changes the
    // variance split only, never the mean (estimator-bench mode-accuracy result).
    std::vector<LatticeFermion> sub;
    std::vector<RealD>          eval_use;
    const std::string evec_file = evec_dir + "/evec." + std::to_string(conf)
                                + (evec_prec == 1 ? ".scidacf" : ".scidac");
    // evals live NEXT TO the evec records so a consumer pointing EVEC_DIR at the producer's
    // dir finds both; with the default EVEC_DIR=obsdir this is the plan's obsdir eval.<conf>.h5.
    const std::string eval_file = evec_dir + "/eval." + std::to_string(conf) + ".h5";
    bool loaded = false;
    if(evec_load){
      double t0 = usecond();
      loaded = LoadEvecRecords(evec_file, eval_file, evec_prec, FrbGrid, FrbGrid_f, sub, eval_use);
      if(loaded){
        std::cout << GridLogMessage << "# conf " << conf << " LoadEvecRecords: " << sub.size()
                  << " modes (prec=" << (evec_prec==1?"single":"double")
                  << ") wall=" << (usecond()-t0)*1.0e-6 << " s" << std::endl;
        if(evec_prec == 1 && P.rr_refine){
          std::cout << GridLogMessage << "# RR re-refining single-prec checkpoint in double..." << std::endl;
          double t1 = usecond();
          RayleighRitzRefine(HermOpEO, sub, eval_use);
          std::cout << GridLogMessage << "# RR re-refine wall=" << (usecond()-t1)*1.0e-6 << " s" << std::endl;
          SaveEvals(eval_file, eval_use, UGrid);   // refresh evals with the refined values
        }
      } else {
        std::cout << GridLogMessage << "# conf " << conf << " no evec checkpoint (" << evec_file
                  << ") -> eigensolving" << std::endl;
      }
    }
    if(!loaded){
      double t0 = usecond();
      BuildLowModes(D, D_f, HermOpEO, HermOpEO_f, P, have_ref, lambda_ref, lambda_max_ref, sub, eval_use);
      std::cout << GridLogMessage << "# conf " << conf << " eigensolved " << sub.size()
                << " modes, BuildLowModes wall=" << (usecond()-t0)*1.0e-6 << " s" << std::endl;
      SaveEvals(eval_file, eval_use, UGrid);   // eval checkpoint: ALWAYS (tiny)
      if(evec_save){
        double t1 = usecond();
        SaveEvecRecords(evec_file, evec_prec, FrbGrid_f, sub);
        std::cout << GridLogMessage << "# conf " << conf << " SaveEvecRecords wall="
                  << (usecond()-t1)*1.0e-6 << " s -> " << evec_file << std::endl;
      }
    }

    // Per-mode physical A2A, STREAMED to bound peak device memory. The exact-low part L^low
    // consumes each (a_i,b_i) only transiently (one inner-product accumulation), but the
    // deflated solve needs ALL u_i across the dilution loop. So hold ALL u_i (odd-cb, ~5x
    // cheaper than a full-4D field) and build a_i,b_i ONE pair at a time, discarding them after
    // accumulating L^low. This drops the physical-A2A footprint from O(Nuse) full-4D fields to
    // O(1), letting Nuse reach Nconv (the deflation-depth lever) without tripping the AccCache
    // eviction assert that hit at high mode count when all a_i,b_i were materialized at once.
    // Replaces the former BuildA2ASet(...) one-shot build of {a,b,u}.
    const int Nuse = (int)sub.size();
    std::vector<LatticeFermion> u;
    std::vector<RealD>          sigma(Nuse, 0.0);
    u.reserve(Nuse);
    for(int i=0; i<Nuse; i++) u.emplace_back(FrbGrid);

    // ---- the LMA loop field per gamma: res = exact L^low + projected high stochastic ----
    std::vector<LatticeComplex> res(ngam, LatticeComplex(UGrid));
    for(auto &r : res) r = Zero();

    // exact noise-free low part: L^low_Gamma(x) = sum_i (1/sigma_i) tr_sc[Gamma a_i b_i^dag](x),
    // streaming a_i,b_i (full 4D) one mode at a time; u_i (odd-cb) is retained for the solve.
    {
      LatticeFermion a_i(UGrid), b_i(UGrid), Ga(UGrid);
      for(int i=0; i<Nuse; i++){
        sigma[i] = std::sqrt(eval_use[i]);
        BuildPhysicalA2A(D, HermOpEO, sub[i], sigma[i], a_i, b_i, u[i]);
        for(int ig=0; ig<ngam; ig++){
          Ga = Gamma(gams[ig]) * a_i;
          res[ig] = res[ig] + (1.0/sigma[i]) * localInnerProduct(b_i, Ga);
        }
      }
    }

    // source-projected stochastic high part: full time+eo+sc dilution, low modes projected out.
    // DISC_NGROUPS>0 stops after the first N (t,eo) groups (timing mode).
    // DISC_TPART=K: each solve sources K slices {t + k*Nt/K} -> tbase loop runs Nt/K.
    int ngroups_done = 0;
    for(int t=0; t<Nt/tpart && (ngroups_max==0 || ngroups_done<ngroups_max); t++){
      for(int eo=0; eo<=1 && (ngroups_max==0 || ngroups_done<ngroups_max); eo++){
        LatticePropagator source(UGrid);
        // StochasticDilutedSource(RNG4, source, UrbGrid, t, eo);   // original single-slice call
        StochasticDilutedSource(RNG4, source, UrbGrid, t, eo, tpart, Nt);

        LatticePropagator StochPropHigh(UGrid);
        SolvePropProjectedMixedBatched(D, D_f, HermOpEO, HermOpEO_f, u, source, StochPropHigh,
                                       P.inner_tol, P.maxinner, P.maxouter, P.maxpatch);

        LatticeComplex tr(UGrid);
        for(int ig=0; ig<ngam; ig++){
          TraceField(tr, gams[ig], StochPropHigh, source);
          res[ig] = res[ig] + tr;
        }
        ngroups_done++;
      }
    }

    // ---- write the combined LMA loop per gamma (Scidac, traces.<gam>.<conf>) ----
    // Skip in timing mode (partial loop -> not a valid measurement; do not leave traces that
    // would self-skip a later real run).
    if(ngroups_max == 0){
      for(int ig=0; ig<ngam; ig++){
        const std::string path = obsdir + "/traces." + gam_names[ig] + "." + std::to_string(conf);
        emptyUserRecord record;
        ScidacWriter WR(UGrid->IsBoss());
        WR.open(path);
        WR.writeScidacFieldRecord(res[ig], record);
        WR.close();
      }
      // p=0 loop per (gamma,t) to the log: cheap, and the K-scan variance analysis +
      // unbiasedness cross-check read these lines instead of the Scidac fields.
      for(int ig=0; ig<ngam; ig++){
        std::vector<TComplex> tv;
        sliceSum(res[ig], tv, Tdir);
        for(int tt=0; tt<Nt; tt++)
          std::cout << GridLogMessage << "# LOOPP0 " << gam_names[ig] << " conf " << conf
                    << " t " << tt << " " << real(TensorRemove(tv[tt]))
                    << " " << imag(TensorRemove(tv[tt])) << std::endl;
      }
    } else {
      std::cout << GridLogMessage << "# TIMING MODE: did " << ngroups_done << "/" << (Nt*2)
                << " (t,eo) groups, NO traces written (projected-solve wall is in the CG logs above)"
                << std::endl;
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
