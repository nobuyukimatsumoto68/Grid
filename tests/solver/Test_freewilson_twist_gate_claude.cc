// Chunk-A gate for the holonomy-twisted free Wilson kernel (holonomy_frameopt_impl_plan_claude.md):
// (1) theta=0 must reproduce FreeWilsonInverse bit-for-bit (regression); (2) theta!=0 must change the
// result (the twist is live). Cheap CPU check on 8^4.

#include <Grid/Grid.h>
#include <Grid/qcd/utils/FreeWilson_claude.h>
#include <Grid/qcd/utils/FreeWilsonTwisted_claude.h>

using namespace Grid;

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplexD::Nsimd());
  Coordinate mpi = GridDefaultMpi();
  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(latt, simd, mpi);

  double mass = 0.1;
  std::vector<Complex> boundary = {1, 1, 1, -1};

  GridParallelRNG RNG(UGrid);
  RNG.SeedFixedIntegers(std::vector<int>({1, 2, 3, 4}));
  LatticeFermionD v(UGrid);
  gaussian(RNG, v);

  FreeWilsonInverse<WilsonImplD> F(UGrid, mass, boundary);
  FreeWilsonTwistedInverse<WilsonImplD> Ft(UGrid, mass, boundary);  // theta = 0

  LatticeFermionD a(UGrid);
  LatticeFermionD b(UGrid);
  F(v, a);
  Ft(v, b);
  double rn = std::sqrt(norm2(a));
  double d0 = std::sqrt(norm2(b - a)) / rn;
  std::cout << "theta=0 regression: ||Ft - F||/||F|| = " << d0
            << (d0 < 1.0e-12 ? "  PASS" : "  FAIL") << std::endl;

  // theta != 0 (traceless per direction: theta[0],theta[1] free, theta[2] = -(theta[0]+theta[1]))
  std::vector<std::array<double, Nd>> th(Nc, std::array<double, Nd>());
  for (int mu = 0; mu < Nd; ++mu) {
    th[0][mu] = 0.10 + 0.01 * mu;
    th[1][mu] = -0.05 + 0.02 * mu;
    th[2][mu] = -(th[0][mu] + th[1][mu]);
  }
  Ft.setTheta(th);
  LatticeFermionD c(UGrid);
  Ft(v, c);
  double d1 = std::sqrt(norm2(c - a)) / rn;
  std::cout << "theta!=0 live: ||Ft(theta) - F||/||F|| = " << d1
            << (d1 > 1.0e-3 ? "  PASS (twist changes the kernel)" : "  FAIL (no effect)") << std::endl;

  Grid_finalize();
  return 0;
}
