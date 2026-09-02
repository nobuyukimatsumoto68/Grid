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
