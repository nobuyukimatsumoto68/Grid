/*************************************************************************************
    Quenched SU(3) Iwasaki HMC for the R2 topology scan (SCC). _claude variant of the stock
    tests/hmc/Test_hmc_IwasakiGauge.cc with:
      - MinimumNorm2 (Omelyan) integrator, same as the stock test, and
      - trajectory length, MD steps, and save interval read from the command line
        (--trajL, --mdsteps, --save_interval) so they can be tuned WITHOUT recompiling.
    beta=2.6. Saves ckpoint_lat.<t> + ckpoint_rng.<t> (resumable). Stock defaults were MDsteps=20, trajL=1.0.
*************************************************************************************/
/*  END LEGAL */
#include <Grid/Grid.h>

NAMESPACE_BEGIN(Grid);

// Q=0 checkpoint observable (Nobu): after each accepted trajectory, do a SHORT gradient flow (tau~1) to
// kill UV noise, measure the clover topological charge, and -- if |Q| < Qthresh -- write an EXTRA NERSC
// config (a DISTINCT prefix, not ckpoint_lat.*, so the gen script's AUTORESUME frontier detection and the
// NerscHmcCheckpointer are untouched). Catches the brief/rare trivial-Q excursions that the every-N-traj
// checkpointer misses. Short flow + clover-Q per trajectory is cheap (pure gauge). The saved config is the
// UNFLOWED gauge; treat it as a Q~0 CANDIDATE -- re-verify with a full flow-Q (tau=4) before use.
template <class Impl>
class QZeroCheckpointLogger : public HmcObservable<typename Impl::Field> {
public:
  INHERIT_GIMPL_TYPES(Impl);
  typedef typename Impl::Field Field;
  RealD flow_eps;
  int   flow_nstep;
  RealD Qthresh;
  std::string prefix;
  QZeroCheckpointLogger(RealD eps, int nstep, RealD qth, std::string pfx)
    : flow_eps(eps), flow_nstep(nstep), Qthresh(qth), prefix(pfx) {}

  void TrajectoryComplete(int traj, Field& U, GridSerialRNG& sRNG, GridParallelRNG& pRNG) override {
    GaugeField Uflow(U.Grid());
    WilsonFlow<Impl> wf(flow_eps, flow_nstep, /*meas_interval=*/ flow_nstep + 1);  // suppress per-step logs
    wf.smear(Uflow, U);
    RealD Q = WilsonLoops<Impl>::TopologicalCharge(Uflow);
    std::cout << GridLogMessage << "Qmonitor: [ " << traj << " ] flowed clover-Q(tau="
              << (flow_eps * flow_nstep) << ") = " << Q << std::endl;
    if (std::abs(Q) < Qthresh) {
      std::string fn = prefix + "." + std::to_string(traj);
      NerscIO::writeConfiguration(U, fn, 0, 0);   // UNFLOWED config, IEEE64BIG (two_row=0, bits32=0)
      std::cout << GridLogMessage << "Qmonitor: |Q|=" << std::abs(Q) << " < " << Qthresh
                << " -> saved trivial-Q CANDIDATE " << fn << std::endl;
    }
  }
};

template <class Impl>
class QZeroCheckpointMod : public ObservableModule<QZeroCheckpointLogger<Impl>, NoParameters> {
  typedef ObservableModule<QZeroCheckpointLogger<Impl>, NoParameters> ObsBase;
  RealD eps_;
  int   nstep_;
  RealD qth_;
  std::string pfx_;
  virtual void initialize() {
    this->ObservablePtr.reset(new QZeroCheckpointLogger<Impl>(eps_, nstep_, qth_, pfx_));
  }
public:
  QZeroCheckpointMod(RealD eps, int nstep, RealD qth, std::string pfx)
    : ObsBase(NoParameters()), eps_(eps), nstep_(nstep), qth_(qth), pfx_(pfx) {}
};

NAMESPACE_END(Grid);

int main(int argc, char **argv)
{
  using namespace Grid;

  Grid_init(&argc, &argv);
  int threads = GridThread::GetThreads();
  std::cout << GridLogMessage << "Grid is setup to use " << threads << " threads" << std::endl;

  // MinimumNorm2 (Omelyan) integrator -- the standard 2nd-order minimum-norm (stock test's choice).
  typedef GenericHMCRunner<MinimumNorm2> HMCWrapper;
  HMCWrapper TheHMC;

  TheHMC.Resources.AddFourDimGrid("gauge");

  // saveInterval from the command line: #configs = Trajectories / saveInterval. Default 20.
  // Saves ckpoint_lat.<traj> AND ckpoint_rng.<traj> -> resumable via --StartingType CheckpointStart.
  int save_interval = 20;
  if (GridCmdOptionExists(argv, argv + argc, "--save_interval")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--save_interval");
    GridCmdOptionInt(a, save_interval);
  }
  CheckpointerParameters CPparams;
  CPparams.config_prefix = "ckpoint_lat";
  CPparams.rng_prefix    = "ckpoint_rng";
  CPparams.saveInterval  = save_interval;
  CPparams.format        = "IEEE64BIG";
  TheHMC.Resources.LoadNerscCheckpointer(CPparams);

  RNGModuleParameters RNGpar;
  RNGpar.serial_seeds   = "1 2 3 4 5";
  RNGpar.parallel_seeds = "6 7 8 9 10";
  TheHMC.Resources.SetRNGSeeds(RNGpar);

  typedef PlaquetteMod<HMCWrapper::ImplPolicy> PlaqObs;
  TheHMC.Resources.AddObservable<PlaqObs>();

  // Q=0 checkpoint monitor: short flow (tau = q0_eps*q0_nstep ~ 1) + clover Q each trajectory; if
  // |Q| < q0_thresh, saves an EXTRA config <q0_prefix>.<traj> (distinct prefix -> AUTORESUME/checkpointer
  // unaffected). Catches the brief trivial-Q excursions the every-N-traj save misses. --no_q0 to disable.
  RealD q0_eps    = 0.02;
  int   q0_nstep  = 50;      // tau = 1.0
  RealD q0_thresh = 0.5;
  std::string q0_prefix = "trivialQ_lat";
  if (GridCmdOptionExists(argv, argv + argc, "--q0_eps")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--q0_eps");
    GridCmdOptionFloat(a, q0_eps);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--q0_nstep")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--q0_nstep");
    GridCmdOptionInt(a, q0_nstep);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--q0_thresh")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--q0_thresh");
    GridCmdOptionFloat(a, q0_thresh);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--q0_prefix")) {
    q0_prefix = GridCmdOptionPayload(argv, argv + argc, "--q0_prefix");
  }
  if (!GridCmdOptionExists(argv, argv + argc, "--no_q0")) {
    std::cout << GridLogMessage << "Q=0 monitor ON: flow tau=" << (q0_eps * q0_nstep)
              << "  |Q|<" << q0_thresh << " -> save " << q0_prefix << ".<traj>" << std::endl;
    TheHMC.Resources.AddObservable<QZeroCheckpointMod<HMCWrapper::ImplPolicy>>(q0_eps, q0_nstep, q0_thresh, q0_prefix);
  }

  // Iwasaki gauge coupling from the command line (no recompile to scan lattice spacing). Default 2.6;
  // RBC/UKQCD Iwasaki gauge anchors: 2.13 (24I, a~0.11 fm), 2.25 (32I, a~0.083 fm), 2.37 (32Ifine,
  // a~0.063 fm) -- NB those a are for the DYNAMICAL 2+1f theory; quenched at the same beta is finer.
  RealD beta = 2.6;
  if (GridCmdOptionExists(argv, argv + argc, "--beta")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--beta");
    GridCmdOptionFloat(a, beta);
  }
  std::cout << GridLogMessage << "Iwasaki gauge action beta=" << beta << std::endl;
  IwasakiGaugeActionR Iaction(beta);
  ActionLevel<HMCWrapper::Field> Level1(1);
  Level1.push_back(&Iaction);
  TheHMC.TheAction.push_back(Level1);

  // MD trajectory from the command line (no recompile to tune). Defaults: trajL=1.6, MDsteps=20 -> dt=0.08.
  // Tune MDsteps by the acceptance rate (target \sim 0.7-0.9). The stock test used MinimumNorm2 with
  // 20 steps at trajL=1.0 (dt=0.05).
  RealD trajL   = 1.6;
  int   mdsteps = 20;
  if (GridCmdOptionExists(argv, argv + argc, "--trajL")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--trajL");
    GridCmdOptionFloat(a, trajL);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--mdsteps")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--mdsteps");
    GridCmdOptionInt(a, mdsteps);
  }
  TheHMC.Parameters.MD.MDsteps = mdsteps;
  TheHMC.Parameters.MD.trajL   = trajL;
  std::cout << GridLogMessage << "MD integrator=MinimumNorm2  trajL=" << trajL
            << "  MDsteps=" << mdsteps << "  dt=" << trajL / mdsteps << std::endl;

  TheHMC.ReadCommandLine(argc, argv);
  TheHMC.Run();

  Grid_finalize();
}
