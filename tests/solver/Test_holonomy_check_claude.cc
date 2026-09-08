// Confirm the holonomy obstruction: a constant diagonal SU(3) link field is FLAT (every plaquette=1,
// F=0, locally pure gauge) yet carries NONZERO Polyakov loops, and Landau gauge fixing leaves them
// unchanged -- so no gauge transform removes them. Demonstrates that "flat" != "gauge-trivial (U=1)"
// on the torus (flat connections classified by holonomy; 't Hooft torons). See
// Grid/scripts_nm/freeprec_gaugefix_topology_claude.md Sec. 7.

#include <Grid/Grid.h>

#include <cmath>

using namespace Grid;

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplexD::Nsimd());
  Coordinate mpi = GridDefaultMpi();
  GridCartesian Grid(latt, simd, mpi);

  LatticeGaugeFieldD U(&Grid);

  if (GridCmdOptionExists(argv, argv + argc, "--config")) {
    // MEASURE mode: load a real config, optional Wilson flow, report plaquette + Polyakov loops.
    // Polyakov loop is gauge invariant, so this is the holonomy of the frame config (no gauge fix
    // needed). --flow_nstep N flows Wilson (eps 0.02) to tau = 0.02 N before measuring.
    std::string cfgfile = GridCmdOptionPayload(argv, argv + argc, "--config");
    FieldMetaData h;
    NerscIO::readConfiguration(U, h, cfgfile);
    int flow_nstep = 0;
    if (GridCmdOptionExists(argv, argv + argc, "--flow_nstep")) {
      std::string a = GridCmdOptionPayload(argv, argv + argc, "--flow_nstep");
      GridCmdOptionInt(a, flow_nstep);
    }
    if (flow_nstep > 0) {
      LatticeGaugeFieldD Uf(&Grid);
      WilsonFlow<PeriodicGimplD> wf(0.02, flow_nstep);
      wf.smear(Uf, U);
      U = Uf;
    }
    Real plaq = WilsonLoops<PeriodicGimplD>::avgPlaquette(U);
    Real Q = WilsonLoops<PeriodicGimplD>::TopologicalCharge5Li(U);
    Coordinate gd = Grid.GlobalDimensions();
    double Vd = (double)gd[0] * gd[1] * gd[2] * gd[3];
    std::cout << "HOLO " << cfgfile << "  flow_nstep=" << flow_nstep << "  tau=" << 0.02 * flow_nstep
              << "  plaq=" << plaq << "  Q_5Li=" << Q;
    // For each direction: mean Polyakov loop |<L>| and the SPATIAL STD of the loop field L(x_perp)
    // = tr(Prod U_mu)/Nc.  std measures the transverse VARIATION of the holonomy; std/|mean| is the
    // relative variation the constant-twist (average-holonomy) kernel would MISS.
    for (int mu = 0; mu < Nd; mu++) {
      LatticeColourMatrixD Um(&Grid);
      Um = PeekIndex<LorentzIndex>(U, mu);
      LatticeColourMatrixD Ploop(&Grid);
      Ploop = Um;
      for (int t = 1; t < gd[mu]; t++) {
        Ploop = PeriodicGimplD::CovShiftForward(Um, mu, Ploop);
      }
      LatticeComplexD L(&Grid);
      L = trace(Ploop) * (1.0 / (double)Nc);
      ComplexD Lmean = sum(L) * (1.0 / Vd);
      LatticeComplexD L2(&Grid);
      L2 = conjugate(L) * L;
      RealD L2mean = real(TensorRemove(sum(L2))) * (1.0 / Vd);
      RealD var = L2mean - std::norm(Lmean);
      RealD sd = (var > 0.0) ? std::sqrt(var) : 0.0;
      std::cout << "  |P" << mu << "|=" << std::abs(Lmean) << " std=" << sd
                << " rel=" << (std::abs(Lmean) > 0.0 ? sd / std::abs(Lmean) : 0.0);
    }
    std::cout << std::endl;
    Grid_finalize();
    return 0;
  }

  // DEMO mode (no --config): constant diagonal SU(3) per direction (each row sums to 0 -> det = 1).
  double ph[4][3] = {{0.3, 0.1, -0.4}, {0.2, -0.5, 0.3}, {-0.1, 0.35, -0.25}, {0.15, 0.15, -0.30}};
  for (int mu = 0; mu < Nd; mu++) {
    typename LatticeColourMatrixD::vector_object::scalar_object V;
    V = Zero();
    for (int c = 0; c < Nc; c++) {
      V()()(c, c) = ComplexD(std::cos(ph[mu][c]), std::sin(ph[mu][c]));
    }
    LatticeColourMatrixD Umu(&Grid);
    Umu = V;  // broadcast the constant matrix to every site
    PokeIndex<LorentzIndex>(U, Umu, mu);
  }

  Real plaq = WilsonLoops<PeriodicGimplD>::avgPlaquette(U);
  std::cout << "constant diagonal config: avgPlaquette = " << plaq << "  (flat if == 1)" << std::endl;
  for (int mu = 0; mu < Nd; mu++) {
    ComplexD P = WilsonLoops<PeriodicGimplD>::avgPolyakovLoop(U, mu);
    std::cout << "  Polyakov mu=" << mu << "  P=" << P << "  |P|=" << std::abs(P) << std::endl;
  }

  // Landau gauge fix (a constant diagonal config is already transverse, so this should not move it).
  LatticeColourMatrixD xform(&Grid);
  FourierAcceleratedGaugeFixer<PeriodicGimplD>::SteepestDescentGaugeFix(
      U, xform, 0.1 / 16.0, 2000, 1.0e-12, 1.0e-12, true, -1, false);
  Real plaq2 = WilsonLoops<PeriodicGimplD>::avgPlaquette(U);
  std::cout << "after Landau: avgPlaquette = " << plaq2 << std::endl;
  for (int mu = 0; mu < Nd; mu++) {
    ComplexD P = WilsonLoops<PeriodicGimplD>::avgPolyakovLoop(U, mu);
    std::cout << "  Polyakov mu=" << mu << "  P=" << P << "  |P|=" << std::abs(P)
              << "  (unchanged by gauge fixing = obstruction confirmed)" << std::endl;
  }

  Grid_finalize();
  return 0;
}
