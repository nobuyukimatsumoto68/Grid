#include "disc_lma_v2_common_claude.h"
#include <hdf5.h>       // conn output_complete (H5Fopen / H5Lexists)
#include <filesystem>   // config loop + disc self-skip
#include <ctime>        // wall blocker
#include <fstream>      // NERSC header pre-check
#include <cstdlib>

// FUSED disc + connected-meson LMA binary for the ROUND-2 LIGHT ensembles
// (conn_lma_impl_plan_claude.md chunk 4, OQ-C = FUSION). Per config it builds the
// low-mode basis ONCE (kept resident in HBM) and produces BOTH observables off it:
//   - DISCONNECTED loops -> traces.<gam>.<conf>  (Scidac; disc_multipleGamma_binary_lma_v2
//     estimator: exact L^low + DISC_TPART time-diluted mixed-prec projected high),
//   - CONNECTED mesons   -> mesons_conn.<conf>.h5 (Phase-B full LMA when CONN_LMA_FULL=1),
// sharing the SAME obsdir so 2D-C pairs per config. This removes the doubled eigensolve
// of running the two standalone binaries as separate jobs (~one eigensolve/config saved,
// zero evec I/O). The two standalone binaries are UNTOUCHED (heavy round 1 uses them).
//
// All physics pieces are the VALIDATED ones from the standalone binaries:
//   - eigensolve/A2A/projected solve: disc_lma_v2_common_claude.h,
//   - disc estimator: disc_multipleGamma_binary_lma_v2_claude.cc,
//   - conn Phase A/B + output schema: baryons_0000_dirac_lma_claude.cc (copied verbatim).
// Outer solver tol 1e-8 (NEVER relaxed). Independent self-skip per observable: a config
// where only one of {traces, mesons_conn} exists recomputes just the missing one (still
// paying the shared eigensolve, which the other needs). Both present -> skip.
//
// CLI: --dir <cfgdir> --obsdir <out> --mass <m> [--M5 1.5] [--beta b] [--eigref <h5>]
// Env: eigensolve knobs (ReadLMAEigParams) + CONF_MIN/MAX/STRIDE + DISC_TPART (default 1)
//   + CONN_LMA_FULL (default 1 here -- the light campaign wants Phase B) + DISC_SEED_STR
//   + wall blocker DISC_DEADLINE_EPOCH/DISC_TPT_SECONDS (disc dominates the fused wall).

using namespace std;
using namespace Grid;

// ============================ CONN output machinery (verbatim from
// baryons_0000_dirac_lma_claude.cc) ============================
static const int n_mom = 17;
static const int mom_table[n_mom][3] = {
  { 0, 0, 0}, { 1, 0, 0}, { 0, 1, 0}, { 0, 0, 1}, { 1, 1, 0}, { 1, 0, 1}, { 0, 1, 1},
  { 1,-1, 0}, { 1, 0,-1}, { 0, 1,-1}, { 1, 1, 1}, {-1, 1, 1}, { 1,-1, 1}, { 1, 1,-1},
  { 2, 0, 0}, { 0, 2, 0}, { 0, 0, 2},
};

static const int n_channel = 10;
static const char *channel_names[n_channel] = {
  "I_I","G5_G5","GTG5_GTG5","GXG5_GXG5","GYG5_GYG5","GZG5_GZG5",
  "GT_GT","GX_GX","GY_GY","GZ_GZ",
};

static const Gamma::Algebra ChGammas[n_channel][2] = {
  {Gamma::Algebra::Identity    ,Gamma::Algebra::Identity},
  {Gamma::Algebra::Gamma5      ,Gamma::Algebra::Gamma5},
  {Gamma::Algebra::GammaTGamma5,Gamma::Algebra::GammaTGamma5},
  {Gamma::Algebra::GammaXGamma5,Gamma::Algebra::GammaXGamma5},
  {Gamma::Algebra::GammaYGamma5,Gamma::Algebra::GammaYGamma5},
  {Gamma::Algebra::GammaZGamma5,Gamma::Algebra::GammaZGamma5},
  {Gamma::Algebra::GammaT      ,Gamma::Algebra::GammaT},
  {Gamma::Algebra::GammaX      ,Gamma::Algebra::GammaX},
  {Gamma::Algebra::GammaY      ,Gamma::Algebra::GammaY},
  {Gamma::Algebra::GammaZ      ,Gamma::Algebra::GammaZ},
};

void MakePhase(Coordinate mom,LatticeComplex &phase)
{
  GridBase *grid = phase.Grid();
  auto latt_size = grid->GlobalDimensions();
  ComplexD ci(0.0,1.0);
  phase=Zero();
  LatticeComplex coor(phase.Grid());
  for(int mu=0;mu<Nd;mu++){
    RealD TwoPiL =  M_PI * 2.0/ latt_size[mu];
    LatticeCoordinate(coor,mu);
    phase = phase + (TwoPiL * mom[mu]) * coor;
  }
  phase = exp(phase*ci);
}

void PointSource(Coordinate &coor,LatticePropagator &source)
{
  source=Zero();
  SpinColourMatrix kronecker; kronecker=1.0;
  pokeSite(kronecker,source,coor);
}

class MesonFile: Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(MesonFile, std::vector<Complex>,  data);
};

void MesonTrace_hdf(Hdf5Writer *WR,LatticePropagator &q1,LatticePropagator &q2)
{
  const int nchannel=n_channel;
  const Gamma::Algebra (*Gammas)[2] = ChGammas;
  std::vector<std::string> channel_name(channel_names, channel_names + n_channel);
  Gamma G5(Gamma::Algebra::Gamma5);
  LatticeComplex meson_CF(q1.Grid());
  for(int ch=0;ch<nchannel;ch++){
    MesonFile MF;
    Gamma Gsrc(Gammas[ch][0]);
    Gamma Gsnk(Gammas[ch][1]);
    meson_CF = trace(G5*adj(q1)*G5*Gsnk*q2*adj(Gsrc));
    std::vector<TComplex> meson_X;
    sliceSum(meson_CF,meson_X, Xdir);
    int nx=meson_X.size();
    std::vector<Complex> corr_X(nx);
    for(int t=0;t<nx;t++) corr_X[t] = TensorRemove(meson_X[t]);
    MF.data=corr_X;
    if (WR) write(*WR,channel_name[ch]+"_x",MF);
    std::vector<TComplex> meson_T;
    sliceSum(meson_CF,meson_T, Tdir);
    int nt=meson_T.size();
    std::vector<Complex> corr(nt);
    for(int t=0;t<nt;t++) corr[t] = TensorRemove(meson_T[t]);
    MF.data=corr;
    if (WR) write(*WR,channel_name[ch]+"_t",MF);
    LatticeComplex phase(q1.Grid());
    LatticeComplex meson_CFp(q1.Grid());
    for(int ip=0;ip<n_mom;ip++){
      Coordinate mom({mom_table[ip][0],mom_table[ip][1],mom_table[ip][2],0});
      MakePhase(mom,phase);
      meson_CFp = meson_CF*phase;
      std::vector<TComplex> meson_Tp;
      sliceSum(meson_CFp,meson_Tp, Tdir);
      int ntp=meson_Tp.size();
      std::vector<Complex> corr_p(ntp);
      for(int t=0;t<ntp;t++) corr_p[t] = TensorRemove(meson_Tp[t]);
      MesonFile MFp;
      MFp.data=corr_p;
      std::stringstream ss;
      ss<<channel_name[ch]<<"_t_p"<<ip;
      if (WR) write(*WR,ss.str(),MFp);
    }
  }
}

static void MesonCorrT(LatticePropagator &q1, LatticePropagator &q2, int ch,
                       std::vector<Complex> &corr)
{
  Gamma G5(Gamma::Algebra::Gamma5);
  Gamma Gsrc(ChGammas[ch][0]);
  Gamma Gsnk(ChGammas[ch][1]);
  LatticeComplex meson_CF(q1.Grid());
  meson_CF = trace(G5*adj(q1)*G5*Gsnk*q2*adj(Gsrc));
  std::vector<TComplex> meson_T;
  sliceSum(meson_CF, meson_T, Tdir);
  corr.resize(meson_T.size());
  for(size_t t=0; t<meson_T.size(); t++) corr[t] = TensorRemove(meson_T[t]);
}

void MesonTrace_hdf_lmafull(Hdf5Writer *WR, LatticePropagator &q1, LatticePropagator &q2,
                            const std::vector<std::vector<Complex>> &lowall,
                            const std::vector<std::vector<Complex>> &lowpt)
{
  const int nchannel=n_channel;
  const Gamma::Algebra (*Gammas)[2] = ChGammas;
  std::vector<std::string> channel_name(channel_names, channel_names + n_channel);
  Gamma G5(Gamma::Algebra::Gamma5);
  LatticeComplex meson_CF(q1.Grid());
  for(int ch=0;ch<nchannel;ch++){
    MesonFile MF;
    Gamma Gsrc(Gammas[ch][0]);
    Gamma Gsnk(Gammas[ch][1]);
    meson_CF = trace(G5*adj(q1)*G5*Gsnk*q2*adj(Gsrc));
    std::vector<TComplex> meson_X;
    sliceSum(meson_CF,meson_X, Xdir);
    int nx=meson_X.size();
    std::vector<Complex> corr_X(nx);
    for(int t=0;t<nx;t++) corr_X[t] = TensorRemove(meson_X[t]);
    MF.data=corr_X;
    if (WR) write(*WR,channel_name[ch]+"_x",MF);
    std::vector<TComplex> meson_T;
    sliceSum(meson_CF,meson_T, Tdir);
    int nt=meson_T.size();
    std::vector<Complex> corr(nt);
    std::vector<Complex> combined(nt);
    for(int t=0;t<nt;t++){
      corr[t]     = TensorRemove(meson_T[t]);
      combined[t] = lowall[ch][t] + corr[t] - lowpt[ch][t];
    }
    MF.data=combined;
    if (WR) write(*WR,channel_name[ch]+"_t",MF);
    MF.data=corr;    if (WR) write(*WR,channel_name[ch]+"_t_point",MF);
    MF.data=lowall[ch]; if (WR) write(*WR,channel_name[ch]+"_t_lowall",MF);
    MF.data=lowpt[ch];  if (WR) write(*WR,channel_name[ch]+"_t_lowpt",MF);
    LatticeComplex phase(q1.Grid());
    LatticeComplex meson_CFp(q1.Grid());
    for(int ip=0;ip<n_mom;ip++){
      MesonFile MFp;
      if(ip==0){
        MFp.data=combined;
      } else {
        Coordinate mom({mom_table[ip][0],mom_table[ip][1],mom_table[ip][2],0});
        MakePhase(mom,phase);
        meson_CFp = meson_CF*phase;
        std::vector<TComplex> meson_Tp;
        sliceSum(meson_CFp,meson_Tp, Tdir);
        int ntp=meson_Tp.size();
        std::vector<Complex> corr_p(ntp);
        for(int t=0;t<ntp;t++) corr_p[t] = TensorRemove(meson_Tp[t]);
        MFp.data=corr_p;
      }
      std::stringstream ss;
      ss<<channel_name[ch]<<"_t_p"<<ip;
      if (WR) write(*WR,ss.str(),MFp);
    }
  }
}

static bool conn_output_complete(const std::string &outfile)
{
  std::ifstream probe(outfile.c_str());
  if (!probe.good()) return false;
  probe.close();
  if (H5Fis_hdf5(outfile.c_str()) <= 0) return false;
  hid_t file = H5Fopen(outfile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) return false;
  bool complete = true;
  for(int ch=0; ch<n_channel && complete; ch++){
    std::vector<std::string> keys;
    keys.push_back(std::string(channel_names[ch]) + "_x");
    keys.push_back(std::string(channel_names[ch]) + "_t");
    for(int ip=0; ip<n_mom; ip++){
      std::stringstream ss; ss << channel_names[ch] << "_t_p" << ip;
      keys.push_back(ss.str());
    }
    for(size_t k=0; k<keys.size(); k++){
      std::string grp = "/" + keys[k];
      std::string dat = grp + "/data";
      if (H5Lexists(file,grp.c_str(),H5P_DEFAULT) <= 0 ||
          H5Lexists(file,dat.c_str(),H5P_DEFAULT) <= 0){ complete = false; break; }
    }
  }
  H5Fclose(file);
  return complete;
}

// ============================ DISC pieces ============================
static const std::vector<Gamma::Algebra> disc_gams = {
  Gamma::Algebra::Identity, Gamma::Algebra::Gamma5,
  Gamma::Algebra::GammaX,   Gamma::Algebra::GammaY,
  Gamma::Algebra::GammaZ,   Gamma::Algebra::GammaT,
  Gamma::Algebra::GammaXGamma5, Gamma::Algebra::GammaYGamma5,
  Gamma::Algebra::GammaZGamma5, Gamma::Algebra::GammaTGamma5,
};
static const std::vector<std::string> disc_gam_names =
  {"id","g5","gx","gy","gz","gt","gxg5","gyg5","gzg5","gtg5"};

static bool disc_output_complete(const std::string &obsdir, int conf)
{
  for(const auto &g : disc_gam_names)
    if(!std::filesystem::exists(obsdir + "/traces." + g + "." + std::to_string(conf)))
      return false;
  return true;
}


int main(int argc, char** argv)
{
  Grid_init(&argc, &argv);

  std::string dir, obsdir, eigref;
  RealD M5 = 1.5, mass = 0.01, beta = 10.8;
  for(int i=1; i<argc; i++){
    std::string a = argv[i];
    if      (a=="--dir"    && i+1<argc){ dir    = argv[++i]; }
    else if (a=="--obsdir" && i+1<argc){ obsdir = argv[++i]; }
    else if (a=="--mass"   && i+1<argc){ mass   = std::stod(argv[++i]); }
    else if (a=="--M5"     && i+1<argc){ M5     = std::stod(argv[++i]); }
    else if (a=="--beta"   && i+1<argc){ beta   = std::stod(argv[++i]); }
    else if (a=="--eigref" && i+1<argc){ eigref = argv[++i]; }
  }
  assert(!dir.empty() && !obsdir.empty() && "usage: --dir <cfgdir> --obsdir <out> --mass <m> [--M5] [--beta] [--eigref]");

  const int Ls = 16;
  RealD b = 1.5, c = 0.5;
  std::vector<Complex> boundary = {1,1,1,-1};
  typedef MobiusFermionD FermionAction;
  typedef MobiusFermionF FermionActionF;
  FermionAction::ImplParams Params(boundary);

  const LMAEigParams P = ReadLMAEigParams();
  const int lma_full = env_int("CONN_LMA_FULL", 1);   // fused light default = Phase B ON
  const int tpart    = env_int("DISC_TPART", 1);
  // DISC_NGROUPS>0 = FIT/TIMING mode: do only the first N disc (t,eo) groups and SKIP the
  // disc trace write (partial loop is not a valid measurement). The device memory peak
  // (eigensolve + A2A retain + Phase-B) is reached before/at the first group, so N=2 is a
  // full memory fit test. Conn is unaffected (always complete). 0 = production.
  const int ngroups_max = env_int("DISC_NGROUPS", 0);
  const char* seed_suffix_e = std::getenv("DISC_SEED_STR");
  const std::string seed_suffix = seed_suffix_e ? seed_suffix_e : "";
  const int evec_save = env_int("EVEC_SAVE", 0);
  const int evec_load = env_int("EVEC_LOAD", 0);
  int evec_prec = 1;
  if(const char* e = std::getenv("EVEC_PREC")){ std::string s(e); if(s=="double"||s=="2") evec_prec = 2; }
  std::string evec_dir = obsdir;
  if(const char* e = std::getenv("EVEC_DIR")) evec_dir = e;

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

  std::cout << GridLogMessage << "# FUSED disc+conn LMA: mass=" << mass << " beta=" << beta
            << " M5=" << M5 << " Nt=" << Nt << " tpart=" << tpart
            << " conn_full=" << lma_full << " dir=" << dir << " obsdir=" << obsdir
            << " eigref=" << eigref << std::endl;

  std::filesystem::create_directories(obsdir);
  if(evec_dir != obsdir) std::filesystem::create_directories(evec_dir);

  std::vector<RealD> lambda_ref;
  RealD lambda_max_ref = 0.0;
  bool have_ref = false;
  if(!eigref.empty()) have_ref = ReadEigref(eigref, lambda_ref, lambda_max_ref);
  if(have_ref)
    std::cout << GridLogMessage << "# eigref loaded: " << lambda_ref.size()
              << " evals, lambda_max=" << lambda_max_ref << std::endl;
  else
    std::cout << GridLogMessage << "# WARNING: no eigref -> Cheby hi from PowerMethod, lo from CHEB_LO" << std::endl;

  LatticeGaugeField  Umu(UGrid);
  LatticeGaugeFieldF Umu_f(UGrid_f);
  GridParallelRNG    RNG4(UGrid);   // disc noise, re-seeded per config

  // config discovery + env overrides (as in the standalone binaries)
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
    if(const char* e = std::getenv("CONF_MIN"))     conf_min = std::atoi(e);
    if(const char* e = std::getenv("CONF_MAX"))     conf_max = std::atoi(e);
    if(const char* e = std::getenv("CONF_STRIDE")){ int s = std::atoi(e); if(s > 0) interval = s; }
  }
  std::cout << GridLogMessage << "# conf_min=" << conf_min << " conf_max=" << conf_max
            << " interval=" << interval << std::endl;

  // wall blocker (disc dominates -> DISC_* env, as the disc binary)
  long deadline = 0;
  if(const char* e = std::getenv("DISC_DEADLINE_EPOCH")) deadline = std::atol(e);
  double max_dur = 0.0;
  if(const char* e = std::getenv("DISC_TPT_SECONDS")) max_dur = std::atof(e);
  const double margin = 1.2;

  // point source (gauge-independent) + its 16 spin-colour columns for the conn low part
  LatticePropagator point_source(UGrid);
  Coordinate Origin({0,0,0,0});
  PointSource(Origin, point_source);
  const int Nsc = Nd*Nc;
  std::vector<LatticeFermion> eta4;
  eta4.reserve(Nsc);
  for(int sc=0; sc<Nsc; sc++) eta4.emplace_back(UGrid);
  for(int sc=0; sc<Nsc; sc++)
    PropToFerm<FermionAction>(eta4[sc], point_source, sc/Nc, sc%Nc);

  for(int conf=conf_min; conf<conf_max; conf+=interval){

    const std::string conn_outfile = obsdir + "/mesons_conn." + std::to_string(conf) + ".h5";

    // per-observable self-skip (boss decides, broadcast so ranks agree)
    uint64_t cdone = 0, ddone = 0;
    if(UGrid->IsBoss()){
      cdone = conn_output_complete(conn_outfile) ? 1 : 0;
      ddone = disc_output_complete(obsdir, conf)  ? 1 : 0;
    }
    UGrid->GlobalSum(cdone);
    UGrid->GlobalSum(ddone);
    if(cdone && ddone){
      std::cout << GridLogMessage << "# skipping conf " << conf << " (disc+conn complete)" << std::endl;
      continue;
    }

    if(deadline > 0 && max_dur > 0.0){
      uint64_t stop = 0;
      if(UGrid->IsBoss())
        if((double)std::time(nullptr) + margin*max_dur > (double)deadline) stop = 1;
      UGrid->GlobalSum(stop);
      if(stop){ std::cout << GridLogMessage << "# blocker: stopping before conf " << conf << std::endl; break; }
    }

    const long t_cfg_start = (long)std::time(nullptr);

    const std::string cfg_path = dir + "/" + lat_prefix + std::to_string(conf);
    {
      uint64_t bad_cfg = 0;
      if(UGrid->IsBoss()){
        std::ifstream chk(cfg_path);
        std::string first_line;
        if(!chk || !std::getline(chk, first_line) || first_line != "BEGIN_HEADER") bad_cfg = 1;
      }
      UGrid->GlobalSum(bad_cfg);
      if(bad_cfg){
        std::cout << GridLogMessage << "# skipping conf " << conf
                  << " (NERSC header check failed; config may be incomplete)" << std::endl;
        continue;
      }
      std::cout << GridLogMessage << "# conf " << conf << " (disc_done=" << ddone
                << " conn_done=" << cdone << ")" << std::endl;
      FieldMetaData header;
      NerscIO::readConfiguration(Umu, header, cfg_path);
      RNG4.SeedUniqueString(cfg_path + seed_suffix);
    }
    precisionChange(Umu_f, Umu);

    FermionAction  D  (Umu,   *FGrid,   *FrbGrid,   *UGrid,   *UrbGrid,   mass, M5, b, c, Params);
    FermionActionF D_f(Umu_f, *FGrid_f, *FrbGrid_f, *UGrid_f, *UrbGrid_f, mass, M5, b, c, Params);
    SchurDiagMooeeOperator<FermionAction,  LatticeFermion>  HermOpEO  (D);
    SchurDiagMooeeOperator<FermionActionF, LatticeFermionF> HermOpEO_f(D_f);

    // ---- ONE eigensolve per config (Load or Build); shared by disc + conn ----
    std::vector<LatticeFermion> sub;
    std::vector<RealD>          eval_use;
    const std::string evec_file = evec_dir + "/evec." + std::to_string(conf)
                                + (evec_prec == 1 ? ".scidacf" : ".scidac");
    const std::string eval_file = evec_dir + "/eval." + std::to_string(conf) + ".h5";
    bool loaded = false;
    double t_basis = 0.0;
    if(evec_load){
      double t0 = usecond();
      loaded = LoadEvecRecords(evec_file, eval_file, evec_prec, FrbGrid, FrbGrid_f, sub, eval_use);
      if(loaded && evec_prec == 1 && P.rr_refine) RayleighRitzRefine(HermOpEO, sub, eval_use);
      t_basis = (usecond()-t0)*1.0e-6;
      if(loaded) std::cout << GridLogMessage << "# conf " << conf << " LoadEvecRecords " << sub.size()
                           << " modes wall=" << t_basis << " s" << std::endl;
    }
    if(!loaded){
      double t0 = usecond();
      BuildLowModes(D, D_f, HermOpEO, HermOpEO_f, P, have_ref, lambda_ref, lambda_max_ref, sub, eval_use);
      t_basis = (usecond()-t0)*1.0e-6;
      std::cout << GridLogMessage << "# conf " << conf << " eigensolved " << sub.size()
                << " modes, BuildLowModes wall=" << t_basis << " s" << std::endl;
      SaveEvals(eval_file, eval_use, UGrid);
      if(evec_save){ SaveEvecRecords(evec_file, evec_prec, FrbGrid_f, sub); }
    }

    // ---- physical A2A set (a_i,b_i on UGrid, u_i odd-cb) retained -- both observables use it ----
    const int Nuse = (int)sub.size();
    std::vector<LatticeFermion> a, bb, u;
    std::vector<RealD>          sigma;
    double t_a2a = 0.0;
    { double t0 = usecond();
      BuildA2ASet(D, HermOpEO, sub, eval_use, a, bb, u, sigma);
      t_a2a = (usecond()-t0)*1.0e-6; }

    double t_conn = 0.0, t_disc = 0.0, t_phaseb = 0.0;

    // ========================= CONNECTED =========================
    if(!cdone){
      double tc0 = usecond();
      // exact point-source low part per column + projected high (batched, outer 1e-8)
      std::vector<LatticeFermion> low4;
      low4.reserve(Nsc);
      for(int sc=0; sc<Nsc; sc++){ low4.emplace_back(UGrid); LowPartOnSource(a, bb, sigma, eta4[sc], low4[sc]); }
      LatticePropagator PropHigh(UGrid);
      SolvePropProjectedMixedBatched(D, D_f, HermOpEO, HermOpEO_f, u, point_source, PropHigh,
                                     P.inner_tol, P.maxinner, P.maxouter, P.maxpatch);
      LatticePropagator prop(UGrid);
      { LatticeFermion h4(UGrid), tot4(UGrid);
        for(int sc=0; sc<Nsc; sc++){
          PropToFerm<FermionAction>(h4, PropHigh, sc/Nc, sc%Nc);
          tot4 = low4[sc] + h4;
          FermToProp<FermionAction>(prop, tot4, sc/Nc, sc%Nc);
        } }

      // Phase B: all-source low 2pt from the pair spin matrices + DET gate (copied logic)
      std::vector<std::vector<Complex>> lowall(n_channel, std::vector<Complex>(Nt, Complex(0.,0.)));
      std::vector<std::vector<Complex>> lowpt (n_channel, std::vector<Complex>(Nt, Complex(0.,0.)));
      if(lma_full && Nuse > 0){
        double tp0 = usecond();
        LatticeSpinColourMatrix scm(UGrid);
        LatticeSpinMatrix       smat(UGrid);
        std::vector<std::vector<std::vector<SpinMatrix>>> SPa(Nuse), SPb(Nuse);
        for(int p=0; p<Nuse; p++){ SPa[p].resize(Nuse); SPb[p].resize(Nuse); }
        for(int p=0; p<Nuse; p++)
          for(int q=0; q<Nuse; q++){
            SlicePairSpinMatrix(a[p],  a[q],  scm, smat, SPa[p][q]);
            SlicePairSpinMatrix(bb[p], bb[q], scm, smat, SPb[p][q]);
          }
        std::vector<SpinColourVector> b0(Nuse);
        for(int i=0; i<Nuse; i++) peekSite(b0[i], bb[i], Origin);
        const RealD Vs = (RealD)UGrid->gSites() / (RealD)Nt;
        for(int ch=0; ch<n_channel; ch++){
          SpinMatrix Gt = GammaMat(Gamma::Algebra::Gamma5) * GammaMat(ChGammas[ch][1]);
          SpinMatrix Xs = adj(GammaMat(ChGammas[ch][0])) * GammaMat(Gamma::Algebra::Gamma5);
          std::vector<std::vector<Complex>> A(Nuse*Nuse, std::vector<Complex>(Nt));
          std::vector<std::vector<Complex>> B(Nuse*Nuse, std::vector<Complex>(Nt));
          std::vector<Complex>              B0(Nuse*Nuse);
          for(int i=0; i<Nuse; i++){
            for(int j=0; j<Nuse; j++){
              for(int t=0; t<Nt; t++){
                Complex za(0.,0.), zb(0.,0.);
                for(int s=0; s<Ns; s++) for(int s1=0; s1<Ns; s1++){
                  za += Gt()(s,s1)() * SPa[j][i][t]()(s1,s)();
                  zb += Xs()(s,s1)() * SPb[i][j][t]()(s1,s)();
                }
                A[i*Nuse+j][t] = za;
                B[j*Nuse+i][t] = zb;
              }
              Complex z0(0.,0.);
              for(int s=0; s<Ns; s++) for(int s1=0; s1<Ns; s1++){
                Complex m0(0.,0.);
                for(int cc=0; cc<Nc; cc++) m0 += b0[i]()(s1)(cc) * conj(b0[j]()(s)(cc));
                z0 += Xs()(s,s1)() * m0;
              }
              B0[j*Nuse+i] = z0;
            }
          }
          for(int i=0; i<Nuse; i++) for(int j=0; j<Nuse; j++){
            const RealD w = 1.0/(sigma[i]*sigma[j]);
            const std::vector<Complex> &Aij = A[i*Nuse+j];
            const std::vector<Complex> &Bji = B[j*Nuse+i];
            const Complex b0ji = B0[j*Nuse+i];
            for(int t=0; t<Nt; t++) lowpt[ch][t] += w * Aij[t] * b0ji;
            for(int t0=0; t0<Nt; t0++){
              const Complex wb = w * Bji[t0];
              for(int t=0; t<Nt; t++) lowall[ch][t] += wb * Aij[(t0+t)%Nt];
            }
          }
          for(int t=0; t<Nt; t++) lowall[ch][t] = lowall[ch][t] * (1.0/(Nt*Vs));
        }
        // DET gate
        LatticePropagator PropLow(UGrid);
        for(int sc=0; sc<Nsc; sc++) FermToProp<FermionAction>(PropLow, low4[sc], sc/Nc, sc%Nc);
        RealD gate_max = 0.0; int gate_ch = 0;
        for(int ch=0; ch<n_channel; ch++){
          std::vector<Complex> ref;
          MesonCorrT(PropLow, PropLow, ch, ref);
          RealD n2r=0., n2d=0.;
          for(int t=0; t<Nt; t++){
            const Complex d = ref[t]-lowpt[ch][t];
            n2r += real(ref[t])*real(ref[t]) + imag(ref[t])*imag(ref[t]);
            n2d += real(d)*real(d) + imag(d)*imag(d);
          }
          RealD rd = std::sqrt(n2d/(n2r>0.?n2r:1.));
          if(rd > gate_max){ gate_max = rd; gate_ch = ch; }
        }
        std::cout << GridLogMessage << "# PHASEB DET GATE conf " << conf
                  << " reldiff_max=" << gate_max << " (worst ch " << channel_names[gate_ch] << ")"
                  << (gate_max < 1.0e-8 ? "  PASS" : "  *** FAIL ***") << std::endl;
        t_phaseb = (usecond()-tp0)*1.0e-6;
      }

      std::unique_ptr<Hdf5Writer> WR;
      if(UGrid->IsBoss()) WR = std::make_unique<Hdf5Writer>(conn_outfile);
      if(lma_full) MesonTrace_hdf_lmafull(WR.get(), prop, prop, lowall, lowpt);
      else         MesonTrace_hdf(WR.get(), prop, prop);
      t_conn = (usecond()-tc0)*1.0e-6;
    }

    // ========================= DISCONNECTED =========================
    if(!ddone){
      double td0 = usecond();
      const int ngam = (int)disc_gam_names.size();
      std::vector<LatticeComplex> res(ngam, LatticeComplex(UGrid));
      for(auto &r : res) r = Zero();
      // exact noise-free low part L^low_Gamma(x) = sum_i (1/sigma_i) tr_sc[Gamma a_i b_i^dag](x)
      { LatticeFermion Ga(UGrid);
        for(int i=0; i<Nuse; i++)
          for(int ig=0; ig<ngam; ig++){
            Ga = Gamma(disc_gams[ig]) * a[i];
            res[ig] = res[ig] + (1.0/sigma[i]) * localInnerProduct(bb[i], Ga);
          } }
      // source-projected stochastic high part; DISC_TPART=K sources K slices/solve
      int ngroups_done = 0;
      for(int t=0; t<Nt/tpart && (ngroups_max==0 || ngroups_done<ngroups_max); t++){
        for(int eo=0; eo<=1 && (ngroups_max==0 || ngroups_done<ngroups_max); eo++){
          LatticePropagator source(UGrid);
          StochasticDilutedSource(RNG4, source, UrbGrid, t, eo, tpart, Nt);
          LatticePropagator StochPropHigh(UGrid);
          SolvePropProjectedMixedBatched(D, D_f, HermOpEO, HermOpEO_f, u, source, StochPropHigh,
                                         P.inner_tol, P.maxinner, P.maxouter, P.maxpatch);
          LatticeComplex tr(UGrid);
          for(int ig=0; ig<ngam; ig++){ TraceField(tr, disc_gams[ig], StochPropHigh, source); res[ig] = res[ig] + tr; }
          ngroups_done++;
        }
      }
      // FIT/TIMING mode (ngroups_max>0): partial loop -> do NOT write traces (would self-skip
      // a later real run) and do NOT emit LOOPP0.
      if(ngroups_max == 0){
        for(int ig=0; ig<ngam; ig++){
          const std::string path = obsdir + "/traces." + disc_gam_names[ig] + "." + std::to_string(conf);
          emptyUserRecord record;
          ScidacWriter WR(UGrid->IsBoss());
          WR.open(path);
          WR.writeScidacFieldRecord(res[ig], record);
          WR.close();
        }
        for(int ig=0; ig<ngam; ig++){
          std::vector<TComplex> tv;
          sliceSum(res[ig], tv, Tdir);
          for(int tt=0; tt<Nt; tt++)
            std::cout << GridLogMessage << "# LOOPP0 " << disc_gam_names[ig] << " conf " << conf
                      << " t " << tt << " " << real(TensorRemove(tv[tt]))
                      << " " << imag(TensorRemove(tv[tt])) << std::endl;
        }
      } else {
        std::cout << GridLogMessage << "# FIT MODE: did " << ngroups_done << " disc groups, NO traces written" << std::endl;
      }
      t_disc = (usecond()-td0)*1.0e-6;
    }

    uint64_t dur = 0;
    if(UGrid->IsBoss()) dur = (uint64_t)((long)std::time(nullptr) - t_cfg_start);
    UGrid->GlobalSum(dur);
    if((double)dur > max_dur) max_dur = (double)dur;
    std::cout << GridLogMessage << "# conf " << conf << " walls: basis=" << t_basis
              << " A2A=" << t_a2a << " conn=" << t_conn << " phaseB=" << t_phaseb
              << " disc=" << t_disc << " total=" << dur << " s (max " << (long)max_dur << "s)" << std::endl;
  } // conf

  std::cout << GridLogMessage << "# FUSED disc+conn LMA done." << std::endl;
  Grid_finalize();
  return 0;
}
