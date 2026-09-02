/*************************************************************************************
    R2 phase (SCC): flowed topological charge for binning quenched configs by Q.

    Loads a NERSC SU(3) config, Wilson-flows a COPY over a long trajectory, and prints
    the topological charge Q(tau) along the flow -- both the clover charge
    (WilsonLoops::TopologicalCharge) and the 5Li Luscher-Weisz improved charge
    (WilsonLoops::TopologicalCharge5Li, cleaner integer, plateaus faster). Bin a config
    as Q!=0 only if the 5Li charge plateaus clearly away from 0 (unsmeared clover Q is
    too noisy to bin -- dwf_prec handoff).

    IMPORTANT (dwf_prec handoff): this LONG flow is ONLY to MEASURE Q. It is a DIFFERENT
    flow from the FRAME flow (tau=2) used to build Omega for the preconditioner, and it
    runs on a COPY -- the preconditioner solve always uses the ORIGINAL unflowed config.

    Flow action (--flow_action): default is IWASAKI gradient flow, not the plain Wilson flow. The
    Iwasaki (rectangle-improved, c1=-0.331) flow suppresses dislocations and preserves the topological
    lump better (a small instanton survives the flow instead of falling through the lattice), giving a
    cleaner near-integer Q plateau -- the reason to prefer it here. Implemented by keeping Grid's
    Luscher RK3 flow integrator and swapping the flow force via WilsonFlow::setGaugeAction to
    IwasakiGaugeAction(beta=Nc); the improved normalization c0+8c1=1 keeps the flow-time scale standard.

    Gradient flow: M. Luscher, arXiv:1006.4518. Analysis output -> stderr (repo convention);
    Grid log messages -> stdout.
*************************************************************************************/
/*  END LEGAL */
#include <Grid/Grid.h>

using namespace Grid;

int main(int argc, char** argv)
{
  Grid_init(&argc, &argv);

  typedef PeriodicGimplD Gimpl;
  typedef Gimpl::GaugeField GaugeField;

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplexD::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(latt, simd, mpi);
  const bool boss = UGrid->IsBoss();   // only rank 0 writes the data table (else MPI duplicates it)

  // ---- command-line knobs -------------------------------------------------------
  if (!GridCmdOptionExists(argv, argv + argc, "--config")) {
    std::cerr << "ERROR: need --config <nersc-file>" << std::endl;
    Grid_finalize();
    return 1;
  }
  std::string cfgfile = GridCmdOptionPayload(argv, argv + argc, "--config");

  RealD eps = 0.01;
  int nstep = 800;
  int meas  = 20;
  if (GridCmdOptionExists(argv, argv + argc, "--flow_eps")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--flow_eps");
    GridCmdOptionFloat(a, eps);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--flow_nstep")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--flow_nstep");
    GridCmdOptionInt(a, nstep);
  }
  if (GridCmdOptionExists(argv, argv + argc, "--meas_interval")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--meas_interval");
    GridCmdOptionInt(a, meas);
  }
  std::string flow_action = "iwasaki";
  if (GridCmdOptionExists(argv, argv + argc, "--flow_action")) {
    flow_action = GridCmdOptionPayload(argv, argv + argc, "--flow_action");
  }

  // Flow force: Iwasaki (default, preserves topology) or plain Wilson. beta=Nc = Luscher flow
  // normalization; Iwasaki's c0+8c1=1 keeps the same flow-time scale. No WilsonFlowBase dtor deletes
  // SG, so one heap action reused across the per-chunk flow objects is safe (only the built-in Wilson
  // SG leaks, once per flow object -- negligible for this short-lived measurement).
  Action<Gimpl::GaugeField>* flowSG = nullptr;
  if (flow_action == "iwasaki") {
    flowSG = new IwasakiGaugeAction<Gimpl>(Gimpl::num_colours);
  } else if (flow_action != "wilson") {
    std::cerr << "ERROR: --flow_action must be iwasaki or wilson" << std::endl;
    Grid_finalize();
    return 1;
  }

  // ---- load the config ----------------------------------------------------------
  GaugeField U(UGrid);
  GaugeField Uflow(UGrid);
  GaugeField tmp(UGrid);
  FieldMetaData header;
  NerscIO::readConfiguration(U, header, cfgfile);

  RealD plaq0 = WilsonLoops<Gimpl>::avgPlaquette(U);
  RealD qc0   = WilsonLoops<Gimpl>::TopologicalCharge(U);
  RealD q50   = WilsonLoops<Gimpl>::TopologicalCharge5Li(U);

  if (boss) {
    std::cerr << "==== flowed topological charge: " << cfgfile << " ====" << std::endl;
    std::cerr << "# lattice " << latt << "   flow_action=" << flow_action << "   flow eps=" << eps
              << " nstep=" << nstep << " meas_interval=" << meas << std::endl;
    std::cerr << "# tau        plaq            Q_clover        Q_5Li" << std::endl;
    std::cerr << std::setprecision(10);
    std::cerr << 0.0 << "   " << plaq0 << "   " << qc0 << "   " << q50 << std::endl;
  }

  // ---- flow the COPY in chunks of meas steps, print Q(tau) each chunk -----------
  Uflow = U;
  RealD tau = 0.0;
  int nchunk = nstep / meas;
  for (int c = 0; c < nchunk; c++) {
    WilsonFlow<Gimpl> wf(eps, meas, meas);   // meas_interval=meas -> one internal print per chunk
    if (flowSG) {
      wf.setGaugeAction(flowSG);             // swap Wilson force -> Iwasaki (default)
    }
    wf.smear(tmp, Uflow);
    Uflow = tmp;
    tau += eps * meas;
    RealD plaq = WilsonLoops<Gimpl>::avgPlaquette(Uflow);
    RealD qc   = WilsonLoops<Gimpl>::TopologicalCharge(Uflow);
    RealD q5   = WilsonLoops<Gimpl>::TopologicalCharge5Li(Uflow);
    if (boss) {
      std::cerr << tau << "   " << plaq << "   " << qc << "   " << q5 << std::endl;
    }
  }

  if (boss) {
    std::cerr << "==== done: bin as Q!=0 only if Q_5Li plateaus clearly away from 0 ====" << std::endl;
  }

  if (flowSG) {
    delete flowSG;
  }
  Grid_finalize();
  return 0;
}
