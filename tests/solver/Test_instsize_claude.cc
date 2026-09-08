// Instanton-size probe for the shrink-flow checkpoints (gauge-fix/topology thread).
// See shrinkflow_impl_plan_claude.md and Grid/scripts_nm/freeprec_gaugefix_topology_claude.md Sec. 4.
//
// Reads ONE (flowed) NERSC config, computes the site-wise CLOVER topological charge density q(x)
// (the density inside WilsonLoops::TopologicalCharge, WilsonLoops.h:641, kept as a lattice instead
// of summed), then finds the topological lumps by greedy peak extraction on |q(x)|:
//   - take the largest unmasked |q| site as a lump centre;
//   - BPST radius estimate from the core value  q(0) = 6/(\pi^2 \rho^4)  ->
//       \rho = (6/(\pi^2 |q_peak|))^{1/4}
//     (regular/singular gauge agree at the core; standard, e.g. Vandoren-van Nieuwenhuizen
//     arXiv:0802.1862);
//   - mask a periodic ball of radius max(2, mask_fac*\rho) around it, recording the ball-integrated
//     charge Q_ball (should be ~ +-1 for a clean isolated lump);
//   - repeat until |q_peak| < min_qpeak or max_lumps found.
// Prints per-lump lines and writes a .dat: cfg  lump  x y z t  qpeak  rho  maskrad  Qball.
// Global cross-checks: Q_clover = sum q, sum |q|, max |q|.
//
// SINGLE-RANK analysis (the site gather assumes the whole lattice on one rank) -- run --mpi 1.1.1.1.
// CPU build (build_mpi_merged); cheap pure-gauge analysis, no solver.
//
// Run:  mpirun -np 1 Test_instsize_claude --grid 16.16.16.16 --mpi 1.1.1.1 \
//         --config shrinkflow_ckp/shrink_ckpoint_lat.640_c10.0833_k3 --dat log/instsize_..._claude.dat

#include <Grid/Grid.h>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

using namespace Grid;

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplexD::Nsimd());
  Coordinate mpi = GridDefaultMpi();
  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(latt, simd, mpi);

  if (!GridCmdOptionExists(argv, argv + argc, "--config")) {
    std::cout << "ERROR: need --config <NERSC gauge file>" << std::endl;
    Grid_finalize();
    return 1;
  }
  std::string cfgfile = GridCmdOptionPayload(argv, argv + argc, "--config");
  std::string cfgbase = cfgfile.substr(cfgfile.find_last_of('/') + 1);

  double min_qpeak = 1.0e-4;  // stop below this core density (rho ~ 8.8a at 1e-4: no longer a lump)
  if (GridCmdOptionExists(argv, argv + argc, "--min_qpeak")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--min_qpeak");
    GridCmdOptionFloat(a, min_qpeak);
  }
  double mask_fac = 2.0;  // mask radius = max(2, mask_fac*rho)
  if (GridCmdOptionExists(argv, argv + argc, "--mask_fac")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--mask_fac");
    GridCmdOptionFloat(a, mask_fac);
  }
  int max_lumps = 30;
  if (GridCmdOptionExists(argv, argv + argc, "--max_lumps")) {
    std::string a = GridCmdOptionPayload(argv, argv + argc, "--max_lumps");
    GridCmdOptionInt(a, max_lumps);
  }
  std::string datfile = "instsize_" + cfgbase + "_claude.dat";
  if (GridCmdOptionExists(argv, argv + argc, "--dat")) {
    datfile = GridCmdOptionPayload(argv, argv + argc, "--dat");
  }
  // SciDAC dump of the q(x) LatticeComplexD field for the VTK visualisation tools
  // (Grid/visualisation/, e.g. FieldDensityAnimate; format = site_plaquette.cc writeFile idiom).
  // Needs a --with-lime Grid build; empty --scidac_dir disables.
  std::string scidac_dir = "";
  if (GridCmdOptionExists(argv, argv + argc, "--scidac_dir")) {
    scidac_dir = GridCmdOptionPayload(argv, argv + argc, "--scidac_dir");
  }

  LatticeGaugeFieldD U(UGrid);
  FieldMetaData rheader;
  NerscIO::readConfiguration(U, rheader, cfgfile);
  Real plaq = WilsonLoops<PeriodicGimplD>::avgPlaquette(U);
  std::cout << "loaded " << cfgfile << "  plaq=" << plaq << std::endl;

  // Clover q(x): the density of WilsonLoops::TopologicalCharge (WilsonLoops.h:641), kept site-wise.
  LatticeColourMatrixD Bx(UGrid);
  LatticeColourMatrixD By(UGrid);
  LatticeColourMatrixD Bz(UGrid);
  WilsonLoops<PeriodicGimplD>::FieldStrength(Bx, U, Ydir, Zdir);
  WilsonLoops<PeriodicGimplD>::FieldStrength(By, U, Zdir, Xdir);
  WilsonLoops<PeriodicGimplD>::FieldStrength(Bz, U, Xdir, Ydir);
  LatticeColourMatrixD Ex(UGrid);
  LatticeColourMatrixD Ey(UGrid);
  LatticeColourMatrixD Ez(UGrid);
  WilsonLoops<PeriodicGimplD>::FieldStrength(Ex, U, Tdir, Xdir);
  WilsonLoops<PeriodicGimplD>::FieldStrength(Ey, U, Tdir, Ydir);
  WilsonLoops<PeriodicGimplD>::FieldStrength(Ez, U, Tdir, Zdir);
  double coeff = 8.0 / (32.0 * M_PI * M_PI);
  LatticeComplexD qfield = coeff * trace(Bx * Ex + By * Ey + Bz * Ez);

  if (!scidac_dir.empty()) {
    std::string qfile = scidac_dir + "/q_" + cfgbase + "_claude.scidac";
#ifdef HAVE_LIME
    emptyUserRecord record;
    ScidacWriter WR(qfield.Grid()->IsBoss());
    WR.open(qfile);
    WR.writeScidacFieldRecord(qfield, record, 0, Grid::BinaryIO::BINARYIO_LEXICOGRAPHIC);
    WR.close();
    std::cout << "scidac q(x) -> " << qfile << std::endl;
#else
    std::cout << "WARNING: --scidac_dir requested but this Grid build has no LIME (HAVE_LIME unset)"
              << " -- skipping " << qfile << std::endl;
#endif
  }

  // Gather q(x) to the host (single rank -> the peekSite loop sees the whole lattice).
  int Lx = latt[0];
  int Ly = latt[1];
  int Lz = latt[2];
  int Lt = latt[3];
  long V = (long)Lx * Ly * Lz * Lt;
  std::vector<double> qh(V);
  Coordinate coor(4);
  for (int t = 0; t < Lt; ++t) {
    for (int z = 0; z < Lz; ++z) {
      for (int y = 0; y < Ly; ++y) {
        for (int x = 0; x < Lx; ++x) {
          coor[0] = x;
          coor[1] = y;
          coor[2] = z;
          coor[3] = t;
          TComplexD qs;
          peekSite(qs, qfield, coor);
          long idx = x + (long)Lx * (y + (long)Ly * (z + (long)Lz * t));
          qh[idx] = TensorRemove(qs).real();
        }
      }
    }
  }

  double Qtot = 0.0;
  double Qabs = 0.0;
  double qmax = 0.0;
  for (long i = 0; i < V; ++i) {
    Qtot += qh[i];
    Qabs += std::fabs(qh[i]);
    if (std::fabs(qh[i]) > qmax) {
      qmax = std::fabs(qh[i]);
    }
  }
  std::cout << "q(x) clover: Q_clover=" << Qtot << "  sum|q|=" << Qabs << "  max|q|=" << qmax
            << std::endl;

  std::ofstream dat(datfile.c_str());
  dat << "# instanton-size probe  cfg=" << cfgbase << "  plaq=" << plaq << "  Q_clover=" << Qtot
      << "  sum|q|=" << Qabs << "  max|q|=" << qmax << "\n";
  dat << "# cfg  lump  x y z t  qpeak  rho  maskrad  Qball\n";

  // Greedy lump extraction: argmax |q| -> BPST rho from the core value -> mask a periodic ball.
  std::vector<char> masked(V, 0);
  int nlump = 0;
  double rhosum = 0.0;
  while (nlump < max_lumps) {
    long imax = -1;
    double amax = 0.0;
    for (long i = 0; i < V; ++i) {
      if (!masked[i] && std::fabs(qh[i]) > amax) {
        amax = std::fabs(qh[i]);
        imax = i;
      }
    }
    if (imax < 0 || amax < min_qpeak) {
      break;
    }
    double qpk = qh[imax];
    double rho = std::pow(6.0 / (M_PI * M_PI * amax), 0.25);
    double maskrad = mask_fac * rho;
    if (maskrad < 2.0) {
      maskrad = 2.0;
    }
    int px = (int)(imax % Lx);
    int py = (int)((imax / Lx) % Ly);
    int pz = (int)((imax / ((long)Lx * Ly)) % Lz);
    int pt = (int)(imax / ((long)Lx * Ly * Lz));
    double mask2 = maskrad * maskrad;
    double Qball = 0.0;
    for (int t = 0; t < Lt; ++t) {
      int dt = std::abs(t - pt);
      if (Lt - dt < dt) {
        dt = Lt - dt;
      }
      for (int z = 0; z < Lz; ++z) {
        int dz = std::abs(z - pz);
        if (Lz - dz < dz) {
          dz = Lz - dz;
        }
        for (int y = 0; y < Ly; ++y) {
          int dy = std::abs(y - py);
          if (Ly - dy < dy) {
            dy = Ly - dy;
          }
          for (int x = 0; x < Lx; ++x) {
            int dx = std::abs(x - px);
            if (Lx - dx < dx) {
              dx = Lx - dx;
            }
            double d2 = (double)dx * dx + (double)dy * dy + (double)dz * dz + (double)dt * dt;
            if (d2 <= mask2) {
              long idx = x + (long)Lx * (y + (long)Ly * (z + (long)Lz * t));
              if (!masked[idx]) {
                Qball += qh[idx];
                masked[idx] = 1;
              }
            }
          }
        }
      }
    }
    std::cout << "lump " << nlump << ": pos=(" << px << "," << py << "," << pz << "," << pt
              << ")  qpeak=" << qpk << "  rho=" << rho << "  maskrad=" << maskrad
              << "  Qball=" << Qball << std::endl;
    dat << cfgbase << "  " << nlump << "  " << px << " " << py << " " << pz << " " << pt << "  "
        << qpk << "  " << rho << "  " << maskrad << "  " << Qball << "\n";
    rhosum += rho;
    nlump = nlump + 1;
  }
  std::cout << "lumps found: " << nlump << "  (min_qpeak=" << min_qpeak << ", mask_fac=" << mask_fac
            << ")  Q_clover=" << Qtot << std::endl;
  dat << "# nlumps=" << nlump << "  mean_rho=" << (nlump > 0 ? rhosum / nlump : 0.0) << "\n";
  dat.close();
  std::cout << "dat -> " << datfile << std::endl;

  Grid_finalize();
  return 0;
}
