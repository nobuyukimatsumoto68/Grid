#include "disc_lma_v2_common_claude.h"
// HDF5 C API for the output-completeness check (H5Fopen / H5Lexists), as in the reference.
#include <hdf5.h>
#include <filesystem>  // directory_iterator for the --dir config loop
#include <ctime>       // wall-time blocker
#include <fstream>     // NERSC header pre-check

// CONNECTED meson binary with LMA deflation -- Phase A of conn_lma_impl_plan_claude.md
// (chunk 1). Loop-mode clone of baryons_0000_dirac_claude.cc (config auto-detect,
// output_complete self-skip, MESON_DEADLINE_EPOCH blocker, IDENTICAL mesons_conn.<conf>.h5
// schema -> drop-in for the 2D-C analysis) with the 16 point-source solves replaced by the
// disc LMA v2 solver stack: per-config low-mode basis (LoadEvecRecords checkpoint OR
// Cheby+RR BuildLowModes, same env knobs + --eigref as disc), then per column
//   S eta = S_low eta + S_high eta,
// exact low part streamed from the physical A2A pairs (a_i,b_i) and the high part via
// SolvePropProjectedMixedBatched (outer tol 1e-8 -- NEVER relax). The low/high split is
// EXACT for whatever subspace is used, so the estimator is the plain propagator to solver
// tolerance -- Phase A changes NO physics, only the solver. (Full conn LMA with the
// all-source-averaged low 2pt = Phase B, chunk 3.)
//
// NOTE the reference conn binary solves at 1e-9 double; this stack is 1e-8 (the disc
// production tolerance, hardwired). The chunk-2 gate documents the achieved agreement.
//
// CLI (loop mode ONLY -- no positional single-config path):
//   --dir <configdir> --obsdir <outdir> --mass <m> [--M5 1.5] [--stride k] [--eigref <h5>]
// Env: eigensolve knobs (ReadLMAEigParams) + CONF_MIN/CONF_MAX/CONF_STRIDE +
//   EVEC_LOAD/EVEC_SAVE/EVEC_PREC/EVEC_DIR (chunk-0 checkpoint; point EVEC_DIR at the disc
//   obsdir to REUSE the disc job's basis -- 16 point solves can never amortize an
//   eigensolve, so standalone conn wants EVEC_LOAD=1) +
//   CONN_COMPARE_PLAIN=1 -> ALSO run the plain mixed-tol-free Solve (header, 1e-8) and
//   print per-column reldiffs + walls (the in-binary apples-to-apples gate + break-even;
//   output h5 is still written from the LMA propagator).
//
// CHUNK 3 -- Phase B (CONN_LMA_FULL=1, default OFF): full conn LMA estimator
//   C_LMA(t) = C_low^{all-src}(t) + [ C^{point}(t) - C_low^{point}(t) ]
// for the ZERO-MOMENTUM time correlators (`_t` and its `_t_p0` cross-check dataset);
// `_x` and the p!=0 momentum datasets stay the point estimator (extending them needs
// phased slice sums -- deferred). The all-source low part is assembled from the
// time-sliced pair spin matrices (SlicePairSpinMatrix in the header) of the FULL A2A
// set, which Phase B therefore RETAINS in memory (~2*Nuse 4D fields; the 5D evec store
// still dominates). Extra datasets written per channel: `_t_point`, `_t_lowall`,
// `_t_lowpt` (the three terms, so the analysis can reweight/inspect). Unbiased by
// construction: <C_low^{all-src}> = <C_low^{point}> by translation invariance, so the
// correction term subtracts exactly what the average adds -- gate 3c checks this and
// the in-binary Phase-B DET gate checks the pair-matrix machinery EXACTLY per config
// (pair-matrix C_low^{point} vs the PropLow MesonTrace contraction, same 10 channels).
//
// Output-schema pieces (mom_table, channel_names, MakePhase, PointSource, MesonFile,
// MesonTrace_hdf, output_complete) are copied VERBATIM from baryons_0000_dirac_claude.cc.

using namespace std;
using namespace Grid;

// 17-momentum table matching average_trace2.f90 (where entry 3 denotes -1).
// Columns are (px,py,pz) in units of 2\pi/NS; the connected meson is projected
// onto each of these at the sink to match the disconnected loop momenta.
static const int n_mom = 17;
static const int mom_table[n_mom][3] = {
  { 0, 0, 0}, //  0
  { 1, 0, 0}, //  1
  { 0, 1, 0}, //  2
  { 0, 0, 1}, //  3
  { 1, 1, 0}, //  4
  { 1, 0, 1}, //  5
  { 0, 1, 1}, //  6
  { 1,-1, 0}, //  7
  { 1, 0,-1}, //  8
  { 0, 1,-1}, //  9
  { 1, 1, 1}, // 10
  {-1, 1, 1}, // 11
  { 1,-1, 1}, // 12
  { 1, 1,-1}, // 13
  { 2, 0, 0}, // 14
  { 0, 2, 0}, // 15
  { 0, 0, 2}, // 16
};

// Channel names, single source of truth shared by MesonTrace_hdf (which writes
// the datasets) and output_complete (which checks they are all present). The
// order and entries must match the Gammas table in MesonTrace_hdf.
static const int n_channel = 10;
static const char *channel_names[n_channel] = {
  "I_I","G5_G5",
  "GTG5_GTG5",
  "GXG5_GXG5",
  "GYG5_GYG5",
  "GZG5_GZG5",
  "GT_GT",
  "GX_GX",
  "GY_GY",
  "GZ_GZ",
};

// (Gsrc, Gsnk) per channel -- hoisted to file scope (from inside MesonTrace_hdf) so the
// Phase-B pair-matrix assembly uses the SAME table as the contractions. Order matches
// channel_names.
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

// Build the phase exp(i p.x). Copied from Mobius_mesons_xt.cc:50.
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
  const Gamma::Algebra (*Gammas)[2] = ChGammas;   // file-scope table (shared with Phase B)

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
    for(int t=0;t<nx;t++){
      corr_X[t] = TensorRemove(meson_X[t]);
      std::cout << " channel "<<ch<<" x "<<t<<" " <<corr_X[t]<<std::endl;
    }
    MF.data=corr_X;

    if (WR) write(*WR,channel_name[ch]+"_x",MF);

    std::vector<TComplex> meson_T;
    sliceSum(meson_CF,meson_T, Tdir);

    int nt=meson_T.size();

    std::vector<Complex> corr(nt);
    for(int t=0;t<nt;t++){
      corr[t] = TensorRemove(meson_T[t]);
      std::cout << " channel "<<ch<<" t "<<t<<" " <<corr[t]<<std::endl;
    }
    MF.data=corr;

    if (WR) write(*WR,channel_name[ch]+"_t",MF);

    // momentum-projected connected correlator C(p_idx,t), one dataset per
    // momentum in mom_table. Index 0 (zero momentum) reproduces the "_t"
    // dataset above as a cross-check. Phase projects the sink onto p.
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
      for(int t=0;t<ntp;t++){
        corr_p[t] = TensorRemove(meson_Tp[t]);
      }

      MesonFile MFp;
      MFp.data=corr_p;

      std::stringstream ss;
      ss<<channel_name[ch]<<"_t_p"<<ip;
      if (WR) write(*WR,ss.str(),MFp);

    }

  }

}

// ---- single-channel zero-momentum t-correlator from two propagators (the SAME
// contraction line as MesonTrace_hdf) -- used by the Phase-B DET gate on PropLow. ----
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

// ---- Phase-B writer: SAME schema as MesonTrace_hdf, but the zero-momentum time
// correlators (`_t` and its `_t_p0` duplicate) carry the FULL-LMA estimator
//   combined(t) = lowall(t) + point(t) - lowpt(t),
// and the three terms are written as EXTRA datasets `_t_point`/`_t_lowall`/`_t_lowpt`
// (schema superset; output_complete checks the standard subset). `_x` and p!=0 stay the
// point estimator (phased slice sums deferred -- see the file header). ----
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
    for(int t=0;t<nx;t++){
      corr_X[t] = TensorRemove(meson_X[t]);
    }
    MF.data=corr_X;

    if (WR) write(*WR,channel_name[ch]+"_x",MF);

    std::vector<TComplex> meson_T;
    sliceSum(meson_CF,meson_T, Tdir);

    int nt=meson_T.size();

    std::vector<Complex> corr(nt);          // the POINT estimator (term 2)
    std::vector<Complex> combined(nt);      // the Phase-B estimator -> `_t`
    for(int t=0;t<nt;t++){
      corr[t]     = TensorRemove(meson_T[t]);
      combined[t] = lowall[ch][t] + corr[t] - lowpt[ch][t];
      std::cout << " channel "<<ch<<" t "<<t<<" " <<combined[t]
                <<" (point "<<corr[t]<<")"<<std::endl;
    }
    MF.data=combined;
    if (WR) write(*WR,channel_name[ch]+"_t",MF);

    MF.data=corr;
    if (WR) write(*WR,channel_name[ch]+"_t_point",MF);
    MF.data=lowall[ch];
    if (WR) write(*WR,channel_name[ch]+"_t_lowall",MF);
    MF.data=lowpt[ch];
    if (WR) write(*WR,channel_name[ch]+"_t_lowpt",MF);

    // momentum datasets: p0 = the combined estimator (preserving the _t_p0 == _t
    // cross-check invariant); p!=0 = point estimator, as in MesonTrace_hdf.
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
        for(int t=0;t<ntp;t++){
          corr_p[t] = TensorRemove(meson_Tp[t]);
        }
        MFp.data=corr_p;
      }

      std::stringstream ss;
      ss<<channel_name[ch]<<"_t_p"<<ip;
      if (WR) write(*WR,ss.str(),MFp);

    }

  }

}

// Completeness check for an existing output file (verbatim from the reference).
// A finished run writes, for every channel, the datasets <channel>_x, <channel>_t and
// <channel>_t_p0 .. _t_p{n_mom-1}. Called on the boss rank only.
static bool output_complete(const std::string &outfile)
{
  std::ifstream probe(outfile.c_str());
  if (!probe.good()){
    return false;
  }
  probe.close();

  if (H5Fis_hdf5(outfile.c_str()) <= 0){
    return false;
  }

  hid_t file = H5Fopen(outfile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0){
    return false;
  }

  bool complete = true;
  for(int ch=0; ch<n_channel && complete; ch++){

    std::vector<std::string> keys;
    keys.push_back(std::string(channel_names[ch]) + "_x");
    keys.push_back(std::string(channel_names[ch]) + "_t");
    for(int ip=0; ip<n_mom; ip++){
      std::stringstream ss;
      ss << channel_names[ch] << "_t_p" << ip;
      keys.push_back(ss.str());
    }

    for(size_t k=0; k<keys.size(); k++){
      std::string grp = "/" + keys[k];
      std::string dat = grp + "/data";
      if (H5Lexists(file,grp.c_str(),H5P_DEFAULT) <= 0 ||
          H5Lexists(file,dat.c_str(),H5P_DEFAULT) <= 0){
        complete = false;
        break;
      }
    }
  }

  H5Fclose(file);
  return complete;
}


int main(int argc, char** argv)
{
  Grid_init(&argc, &argv);   // FIRST: consume Grid's --grid/--mpi/... args

  // ---- CLI (loop mode only; manual scan like the reference loop-mode block) ----
  std::string dir, obsdir, eigref;
  RealD M5 = 1.5, mass = 0.1;
  int stride_mult = 1;
  for(int i=1; i<argc; i++){
    std::string a = argv[i];
    if      (a=="--dir"    && i+1<argc){ dir    = argv[++i]; }
    else if (a=="--obsdir" && i+1<argc){ obsdir = argv[++i]; }
    else if (a=="--mass"   && i+1<argc){ mass   = std::stod(argv[++i]); }
    else if (a=="--M5"     && i+1<argc){ M5     = std::stod(argv[++i]); }
    else if (a=="--stride" && i+1<argc){ stride_mult = std::stoi(argv[++i]); }
    else if (a=="--eigref" && i+1<argc){ eigref = argv[++i]; }
  }
  assert(!dir.empty() && !obsdir.empty() && "usage: --dir <configdir> --obsdir <outdir> --mass <m> [--M5] [--stride] [--eigref]");

  // ---- fixed Mobius parameters (match the reference conn + disc binaries) ----
  const int Ls = 16;
  RealD b = 1.5, c = 0.5;
  std::vector<Complex> boundary = {1,1,1,-1};
  typedef MobiusFermionD FermionAction;
  typedef MobiusFermionF FermionActionF;
  FermionAction::ImplParams Params(boundary);

  const LMAEigParams P = ReadLMAEigParams();

  // ---- CHUNK-0 evec-checkpoint knobs (same semantics as the disc v2 binary) ----
  const int evec_save = env_int("EVEC_SAVE", 0);
  const int evec_load = env_int("EVEC_LOAD", 0);
  int evec_prec = 1;   // 1 = single (.scidacf, D-IO default), 2 = double (.scidac)
  if(const char* e = std::getenv("EVEC_PREC")){
    const std::string s(e);
    if(s == "double" || s == "2") evec_prec = 2;
  }
  std::string evec_dir = obsdir;
  if(const char* e = std::getenv("EVEC_DIR")) evec_dir = e;

  const int compare_plain = env_int("CONN_COMPARE_PLAIN", 0);
  const int lma_full      = env_int("CONN_LMA_FULL", 0);   // Phase B (chunk 3), default OFF

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

  std::cout << GridLogMessage << "# conn LMA " << (lma_full ? "Phase B (FULL)" : "Phase A")
            << ": mass=" << mass << " M5=" << M5
            << " dir=" << dir << " obsdir=" << obsdir << " eigref=" << eigref << std::endl;
  if(evec_save || evec_load)
    std::cout << GridLogMessage << "# evec checkpoint: save=" << evec_save << " load=" << evec_load
              << " prec=" << (evec_prec==1?"single":"double") << " dir=" << evec_dir << std::endl;

  std::filesystem::create_directories(obsdir);
  if(evec_dir != obsdir) std::filesystem::create_directories(evec_dir);   // eval/evec files land here

  // ---- per-ENSEMBLE spectral landscape (Cheby window), as in disc v2 ----
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

  // ---- config discovery (scan dir for *_lat.NNNN) + env overrides, as in disc v2 ----
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
  const int step = interval * (stride_mult > 0 ? stride_mult : 1);
  std::cout << GridLogMessage << "# conf_min=" << conf_min << " conf_max=" << conf_max
            << " interval=" << interval << " step=" << step << std::endl;

  // ---- graceful wall-time blocker (MESON_* env, as in the reference conn binary) ----
  long deadline = 0;
  if(const char* e = std::getenv("MESON_DEADLINE_EPOCH")) deadline = std::atol(e);
  double max_dur = 0.0;
  if(const char* e = std::getenv("MESON_TPT_SECONDS")) max_dur = std::atof(e);
  const double margin = 1.2;

  // point source (delta at origin) is gauge-independent -> build ONCE; the 16 spin-colour
  // source columns are extracted once for the low-part inner products.
  LatticePropagator point_source(UGrid);
  Coordinate Origin({0,0,0,0});
  PointSource(Origin, point_source);

  const int Nsc = Nd*Nc;   // 16 spin-colour columns
  std::vector<LatticeFermion> eta4;
  eta4.reserve(Nsc);
  for(int sc=0; sc<Nsc; sc++) eta4.emplace_back(UGrid);
  for(int sc=0; sc<Nsc; sc++)
    PropToFerm<FermionAction>(eta4[sc], point_source, sc/Nc, sc%Nc);

  for(int conf=conf_min; conf<conf_max; conf+=step){

    const std::string outfile = obsdir + "/mesons_conn." + std::to_string(conf) + ".h5";

    // cheap skip of a complete output (boss decides, broadcast so ranks agree).
    uint64_t done = 0;
    if(UGrid->IsBoss()) done = output_complete(outfile) ? 1 : 0;
    UGrid->GlobalSum(done);
    if(done){
      std::cout << GridLogMessage << "# skipping conf " << conf << " (output complete)" << std::endl;
      continue;
    }

    // graceful wall stop before a config we cannot finish
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

    {
      const std::string cfgpath = dir + "/" + lat_prefix + std::to_string(conf);

      // guard against partially-written configs (HMC may still be running).
      uint64_t bad_cfg = 0;
      if(UGrid->IsBoss()){
        std::ifstream chk(cfgpath);
        std::string first_line;
        if(!chk || !std::getline(chk, first_line) || first_line != "BEGIN_HEADER") bad_cfg = 1;
      }
      UGrid->GlobalSum(bad_cfg);
      if(bad_cfg){
        std::cout << GridLogMessage << "# skipping conf " << conf
                  << " (NERSC header check failed; config may be incomplete)" << std::endl;
        continue;
      }

      std::cout << GridLogMessage << "# conf " << conf << " -> " << outfile << std::endl;
      FieldMetaData header;
      NerscIO::readConfiguration(Umu, header, cfgpath);
    }
    precisionChange(Umu_f, Umu);

    FermionAction  D  (Umu,   *FGrid,   *FrbGrid,   *UGrid,   *UrbGrid,   mass, M5, b, c, Params);
    FermionActionF D_f(Umu_f, *FGrid_f, *FrbGrid_f, *UGrid_f, *UrbGrid_f, mass, M5, b, c, Params);
    SchurDiagMooeeOperator<FermionAction,  LatticeFermion>  HermOpEO  (D);
    SchurDiagMooeeOperator<FermionActionF, LatticeFermionF> HermOpEO_f(D_f);

    // ---- eigenbasis: LoadEvecRecords checkpoint -> else BuildLowModes (chunk-0 block,
    // same as disc v2; standalone conn wants EVEC_LOAD=1 pointed at the disc obsdir). ----
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
      if(loaded){
        std::cout << GridLogMessage << "# conf " << conf << " LoadEvecRecords: " << sub.size()
                  << " modes (prec=" << (evec_prec==1?"single":"double")
                  << ") wall=" << (usecond()-t0)*1.0e-6 << " s" << std::endl;
        if(evec_prec == 1 && P.rr_refine){
          std::cout << GridLogMessage << "# RR re-refining single-prec checkpoint in double..." << std::endl;
          double t1 = usecond();
          RayleighRitzRefine(HermOpEO, sub, eval_use);
          std::cout << GridLogMessage << "# RR re-refine wall=" << (usecond()-t1)*1.0e-6 << " s" << std::endl;
        }
      } else {
        std::cout << GridLogMessage << "# conf " << conf << " no evec checkpoint (" << evec_file
                  << ") -> eigensolving" << std::endl;
      }
      t_basis = (usecond()-t0)*1.0e-6;
    }
    if(!loaded){
      double t0 = usecond();
      BuildLowModes(D, D_f, HermOpEO, HermOpEO_f, P, have_ref, lambda_ref, lambda_max_ref, sub, eval_use);
      t_basis = (usecond()-t0)*1.0e-6;
      std::cout << GridLogMessage << "# conf " << conf << " eigensolved " << sub.size()
                << " modes, BuildLowModes wall=" << t_basis << " s" << std::endl;
      SaveEvals(eval_file, eval_use, UGrid);   // eval checkpoint: ALWAYS (tiny)
      if(evec_save){
        double t1 = usecond();
        SaveEvecRecords(evec_file, evec_prec, FrbGrid_f, sub);
        std::cout << GridLogMessage << "# conf " << conf << " SaveEvecRecords wall="
                  << (usecond()-t1)*1.0e-6 << " s -> " << evec_file << std::endl;
      }
    }

    // ---- A2A stream: accumulate the exact low part on the 16 point-source columns while
    // building the odd-cb u_i for the projected high solve (a_i,b_i transient, as disc v2). ----
    const int Nuse = (int)sub.size();
    std::vector<LatticeFermion> u;
    std::vector<RealD>          sigma(Nuse, 0.0);
    u.reserve(Nuse);
    for(int i=0; i<Nuse; i++) u.emplace_back(FrbGrid);

    std::vector<LatticeFermion> low4;
    low4.reserve(Nsc);
    for(int sc=0; sc<Nsc; sc++) low4.emplace_back(UGrid);
    for(int sc=0; sc<Nsc; sc++) low4[sc] = Zero();

    // Phase B retains the full physical A2A set for the pair matrices (2*Nuse 4D fields;
    // the 5D evec store still dominates memory). Phase A streams as before.
    std::vector<LatticeFermion> av, bv;
    if(lma_full){
      av.reserve(Nuse); bv.reserve(Nuse);
      for(int i=0; i<Nuse; i++){ av.emplace_back(UGrid); bv.emplace_back(UGrid); }
    }
    double t_low = 0.0;
    {
      double t0 = usecond();
      LatticeFermion a_s(UGrid), b_s(UGrid);
      for(int i=0; i<Nuse; i++){
        LatticeFermion &a_i = lma_full ? av[i] : a_s;
        LatticeFermion &b_i = lma_full ? bv[i] : b_s;
        sigma[i] = std::sqrt(eval_use[i]);
        BuildPhysicalA2A(D, HermOpEO, sub[i], sigma[i], a_i, b_i, u[i]);
        for(int sc=0; sc<Nsc; sc++){
          ComplexD bi = innerProduct(b_i, eta4[sc]);
          low4[sc] = low4[sc] + (bi/sigma[i]) * a_i;
        }
      }
      t_low = (usecond()-t0)*1.0e-6;
    }

    // ---- projected high solve, batched mixed-prec over the 16 columns (outer 1e-8) ----
    LatticePropagator PropHigh(UGrid);
    double t_high = 0.0;
    {
      double t0 = usecond();
      SolvePropProjectedMixedBatched(D, D_f, HermOpEO, HermOpEO_f, u, point_source, PropHigh,
                                     P.inner_tol, P.maxinner, P.maxouter, P.maxpatch);
      t_high = (usecond()-t0)*1.0e-6;
    }

    // ---- total propagator: S = S_low + S_high, column by column ----
    LatticePropagator prop(UGrid);
    {
      LatticeFermion h4(UGrid), tot4(UGrid);
      for(int sc=0; sc<Nsc; sc++){
        PropToFerm<FermionAction>(h4, PropHigh, sc/Nc, sc%Nc);
        tot4 = low4[sc] + h4;
        FermToProp<FermionAction>(prop, tot4, sc/Nc, sc%Nc);
      }
    }

    // ---- Phase B (CONN_LMA_FULL=1): all-source low-mode 2pt from the pair spin
    // matrices, the point-low term, and the in-binary DET gate (pair-matrix lowpt vs
    // the PropLow contraction -- an EXACT identity, expect ~1e-12). ----
    std::vector<std::vector<Complex>> lowall(n_channel, std::vector<Complex>(Nt, Complex(0.,0.)));
    std::vector<std::vector<Complex>> lowpt (n_channel, std::vector<Complex>(Nt, Complex(0.,0.)));
    double t_phaseb = 0.0;
    if(lma_full && Nuse > 0){
      double t0 = usecond();
      // pair matrices: SPa[p][q](t) from (l=a_p, r=a_q), SPb likewise over b.
      // tr[X * SP(f_l,f_r)] = f_r^dag X f_l (header identity).
      LatticeSpinColourMatrix scm(UGrid);
      LatticeSpinMatrix       smat(UGrid);
      std::vector<std::vector<std::vector<SpinMatrix>>> SPa(Nuse), SPb(Nuse);
      for(int p=0; p<Nuse; p++){ SPa[p].resize(Nuse); SPb[p].resize(Nuse); }
      for(int p=0; p<Nuse; p++){
        for(int q=0; q<Nuse; q++){
          SlicePairSpinMatrix(av[p], av[q], scm, smat, SPa[p][q]);
          SlicePairSpinMatrix(bv[p], bv[q], scm, smat, SPb[p][q]);
        }
      }
      // origin values of b (the point source sits at Origin) for the point-low term
      std::vector<SpinColourVector> b0(Nuse);
      for(int i=0; i<Nuse; i++) peekSite(b0[i], bv[i], Origin);
      const RealD Vs = (RealD)UGrid->gSites() / (RealD)Nt;   // spatial volume

      for(int ch=0; ch<n_channel; ch++){
        // spin structures from the SAME channel table as the contractions:
        // sink G5*Gsnk, source adj(Gsrc)*G5 (from C = tr[G5 S^dag G5 Gsnk S adj(Gsrc)])
        SpinMatrix Gt = GammaMat(Gamma::Algebra::Gamma5) * GammaMat(ChGammas[ch][1]);
        SpinMatrix Xs = adj(GammaMat(ChGammas[ch][0])) * GammaMat(Gamma::Algebra::Gamma5);
        // A_ij(t) = a_i^dag Gt a_j = tr[Gt*SPa[j][i](t)]
        // B_ji(t) = b_j^dag Xs b_i = tr[Xs*SPb[i][j](t)] ; B0_ji likewise at the origin site
        std::vector<std::vector<Complex>> A(Nuse*Nuse, std::vector<Complex>(Nt));
        std::vector<std::vector<Complex>> B(Nuse*Nuse, std::vector<Complex>(Nt));
        std::vector<Complex>              B0(Nuse*Nuse);
        for(int i=0; i<Nuse; i++){
          for(int j=0; j<Nuse; j++){
            for(int t=0; t<Nt; t++){
              Complex za(0.,0.), zb(0.,0.);
              for(int s=0; s<Ns; s++){
                for(int s1=0; s1<Ns; s1++){
                  za += Gt()(s,s1)() * SPa[j][i][t]()(s1,s)();
                  zb += Xs()(s,s1)() * SPb[i][j][t]()(s1,s)();
                }
              }
              A[i*Nuse+j][t] = za;
              B[j*Nuse+i][t] = zb;
            }
            Complex z0(0.,0.);
            for(int s=0; s<Ns; s++){
              for(int s1=0; s1<Ns; s1++){
                Complex m0(0.,0.);   // M0_{ij}^{s1 s} = sum_c b_i(0)^{s1 c} conj(b_j(0)^{s c})
                for(int c=0; c<Nc; c++) m0 += b0[i]()(s1)(c) * conj(b0[j]()(s)(c));
                z0 += Xs()(s,s1)() * m0;
              }
            }
            B0[j*Nuse+i] = z0;
          }
        }
        // assemble:  lowpt(t)  = sum_ij w_ij A_ij(t) B0_ji         (w_ij = 1/(sigma_i sigma_j))
        //            lowall(t) = (1/(Nt*Vs)) sum_t0 sum_ij w_ij A_ij((t0+t)%Nt) B_ji(t0)
        for(int i=0; i<Nuse; i++){
          for(int j=0; j<Nuse; j++){
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
        }
        for(int t=0; t<Nt; t++) lowall[ch][t] = lowall[ch][t] * (1.0/(Nt*Vs));
      } // ch

      // DET gate: pair-matrix lowpt vs the (already-validated) PropLow contraction
      LatticePropagator PropLow(UGrid);
      for(int sc=0; sc<Nsc; sc++) FermToProp<FermionAction>(PropLow, low4[sc], sc/Nc, sc%Nc);
      RealD gate_max = 0.0;
      int   gate_ch  = 0;
      for(int ch=0; ch<n_channel; ch++){
        std::vector<Complex> ref;
        MesonCorrT(PropLow, PropLow, ch, ref);
        RealD n2r=0., n2d=0.;
        for(int t=0; t<Nt; t++){
          const Complex d = ref[t]-lowpt[ch][t];
          n2r += real(ref[t])*real(ref[t]) + imag(ref[t])*imag(ref[t]);   // std::norm rejects thrust::complex (HIP build)
          n2d += real(d)*real(d) + imag(d)*imag(d);
        }
        RealD rd = std::sqrt(n2d/(n2r>0.?n2r:1.));
        if(rd > gate_max){ gate_max = rd; gate_ch = ch; }
      }
      std::cout << GridLogMessage << "# PHASEB DET GATE conf " << conf
                << " lowpt(pair-matrix) vs PropLow contraction reldiff_max=" << gate_max
                << " (worst ch " << channel_names[gate_ch] << ")"
                << (gate_max < 1.0e-8 ? "  PASS" : "  *** FAIL ***") << std::endl;
      t_phaseb = (usecond()-t0)*1.0e-6;
    }

    // ---- optional in-binary gate + break-even: plain solve (header Solve, SAME 1e-8) ----
    if(compare_plain){
      LatticePropagator prop_plain(UGrid);
      double t0 = usecond();
      Solve(D, point_source, prop_plain);
      double t_plain = (usecond()-t0)*1.0e-6;
      LatticeFermion p4(UGrid), l4(UGrid), d4(UGrid);
      RealD reldiff_max = 0.0;
      for(int sc=0; sc<Nsc; sc++){
        PropToFerm<FermionAction>(p4, prop_plain, sc/Nc, sc%Nc);
        PropToFerm<FermionAction>(l4, prop,       sc/Nc, sc%Nc);
        d4 = l4 - p4;
        RealD rd = std::sqrt(norm2(d4)/norm2(p4));
        if(rd > reldiff_max) reldiff_max = rd;
      }
      std::cout << GridLogMessage << "# COMPARE_PLAIN conf " << conf
                << " column reldiff_max=" << reldiff_max
                << " plain_wall=" << t_plain << " s vs lma(basis+low+high)="
                << t_basis+t_low+t_high << " s" << std::endl;
    }

    // ---- write the h5 (schema identical to the reference; Phase B adds term datasets
    // and swaps the zero-momentum `_t`/`_t_p0` for the combined estimator) ----
    std::unique_ptr<Hdf5Writer> WR;
    if(UGrid->IsBoss()) WR = std::make_unique<Hdf5Writer>(outfile);
    if(lma_full) MesonTrace_hdf_lmafull(WR.get(), prop, prop, lowall, lowpt);
    else         MesonTrace_hdf(WR.get(), prop, prop);

    // per-config timing breakdown (the chunk-1b conn break-even numbers) + blocker update
    uint64_t dur = 0;
    if(UGrid->IsBoss()) dur = (uint64_t)((long)std::time(nullptr) - t_cfg_start);
    UGrid->GlobalSum(dur);
    if((double)dur > max_dur) max_dur = (double)dur;
    std::cout << GridLogMessage << "# conf " << conf << " walls: basis=" << t_basis
              << " s  low(A2A)=" << t_low << " s  high(batched)=" << t_high
              << " s  phaseB=" << t_phaseb
              << " s  total=" << dur << " s (max so far " << (long)max_dur << "s)" << std::endl;
  } // conf

  std::cout << GridLogMessage << "# conn LMA Phase A done." << std::endl;
  Grid_finalize();
  return 0;
}
