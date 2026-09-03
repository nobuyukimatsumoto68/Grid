// C0 gate for the custom DOF-payload FFT (DofFFT_claude.h): direct radix-L DFT along ONE unsplit dim.
//
// Two checks, both must pass to ~1e-12 (double) before C1 (GPU + measure):
//   (1) roundtrip : iFFT(FFT(v)) == v   (self-consistency of fwd/bwd + the 1/L scale).
//   (2) oracle    : DofFFT.FFT_dim(x)  ==  PlannedFFT.FFT_dim_mask(mask={0,1,0,0,0})  (fwd AND bwd),
//                   pinning sign, ordering (freq=coord), and scale against native Grid.
// x = dim 1 of the 5D grid (s=dim0, x=1,y=2,z=3,t=4); x is UNSPLIT with the default SIMD layout.

#include <Grid/Grid.h>
#include <Grid/algorithms/DofFFT_claude.h>

using namespace Grid;

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  const int Ls = 8;
  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplexD::Nsimd());
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(latt, simd, mpi);
  GridCartesian* FGrid = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);

  // C0 handles one UNSPLIT spacetime dim. On a GPU build the default SIMD layout may split dim 1, so
  // pick the first spacetime dim (1..4) that is unsplit rather than hard-coding x.
  int xdim = -1;
  for (int d = 1; d <= 4; ++d) {
    if (FGrid->_simd_layout[d] == 1) {
      xdim = d;
      break;
    }
  }
  GRID_ASSERT(xdim > 0);

  std::cout << GridLogMessage << "==== DofFFT C0 : grid " << latt[0] << "." << latt[1] << "."
            << latt[2] << "." << latt[3] << "  Ls=" << Ls
            << "  simd5=" << FGrid->_simd_layout << "  rdim5=" << FGrid->_rdimensions << " ====" << std::endl;
  std::cout << GridLogMessage << "  transforming dim " << xdim << " (simd_layout=" << FGrid->_simd_layout[xdim]
            << ", unsplit)" << std::endl;

  GridParallelRNG rng(FGrid);
  rng.SeedFixedIntegers(std::vector<int>({1, 2, 3, 4}));
  LatticeFermionD v(FGrid);
  random(rng, v);
  RealD nv = norm2(v);

  DofFFT_claude<typename LatticeFermionD::vector_object> dfft(FGrid);
  PlannedFFT<typename LatticeFermionD::vector_object> pfft(FGrid);
  Coordinate mask(Nd + 1, 0);
  mask[xdim] = 1;

  // ---- (1) roundtrip : DofFFT fwd then bwd on dim x. ----
  LatticeFermionD k(FGrid);
  LatticeFermionD back(FGrid);
  dfft.FFT_dim(k, v, xdim, DofFFT_claude<typename LatticeFermionD::vector_object>::forward);
  dfft.FFT_dim(back, k, xdim, DofFFT_claude<typename LatticeFermionD::vector_object>::backward);
  LatticeFermionD dr = back - v;
  RealD rt = norm2(dr) / nv;
  std::cout << GridLogMessage << "  (1) roundtrip  ||iFFT(FFT(v)) - v||^2 / ||v||^2 = " << rt << std::endl;

  // ---- (2a) oracle forward : DofFFT.FFT_dim(x) vs PlannedFFT masked to x. ----
  LatticeFermionD kf_dof(FGrid);
  LatticeFermionD kf_ora(FGrid);
  dfft.FFT_dim(kf_dof, v, xdim, DofFFT_claude<typename LatticeFermionD::vector_object>::forward);
  pfft.FFT_dim_mask(kf_ora, v, mask, FFT::forward);
  LatticeFermionD df = kf_dof - kf_ora;
  RealD of = norm2(df) / nv;
  std::cout << GridLogMessage << "  (2a) oracle fwd  ||DofFFT - PlannedFFT||^2 / ||v||^2 = " << of << std::endl;

  // ---- (2b) oracle backward : same, backward. ----
  LatticeFermionD kb_dof(FGrid);
  LatticeFermionD kb_ora(FGrid);
  dfft.FFT_dim(kb_dof, v, xdim, DofFFT_claude<typename LatticeFermionD::vector_object>::backward);
  pfft.FFT_dim_mask(kb_ora, v, mask, FFT::backward);
  LatticeFermionD db = kb_dof - kb_ora;
  RealD ob = norm2(db) / nv;
  std::cout << GridLogMessage << "  (2b) oracle bwd  ||DofFFT - PlannedFFT||^2 / ||v||^2 = " << ob << std::endl;

  const RealD tol = 1.0e-24;  // norm2 (squared) relative tol; ~1e-12 elementwise
  bool pass = (rt < tol) && (of < tol) && (ob < tol);
  std::cout << GridLogMessage << "  ==== C0 " << (pass ? "PASS" : "FAIL")
            << "  (tol " << tol << " on squared relative norm) ====" << std::endl;

  Grid_finalize();
  return pass ? 0 : 1;
}
