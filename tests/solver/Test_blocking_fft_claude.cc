// A/B validation harness for the SIMD-blocking FFT (grid_blocking_fft_impl_plan_claude.md, Chunk A).
// Compares BlockingFFT::FFT_spacetime against Grid's FFT_dim_mask (the reference) on a random 5D fermion
// field -- forward, backward, and forward-then-backward = identity -- and reports max rel diffs. Every
// later chunk (strided cuFFT for x,y,z; the t butterfly) must keep this at machine eps. Also prints the
// simd_layout so we can confirm the assumed [.,.,.,2] (t-split, sd=2).

#include <Grid/Grid.h>
#include <Grid/algorithms/BlockingFFT_claude.h>

using namespace Grid;

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  const int Ls = 8;
  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplexD::Nsimd());
  Coordinate mpi = GridDefaultMpi();
  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(latt, simd, mpi);
  GridCartesian* FGrid = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);

  std::cout << GridLogMessage << "grid5 fdim = " << FGrid->_fdimensions
            << "  simd_layout = " << FGrid->_simd_layout
            << "  Nsimd = " << FGrid->Nsimd() << std::endl;

  GridParallelRNG RNG5(FGrid);
  RNG5.SeedFixedIntegers(std::vector<int>({1, 2, 3, 4}));
  LatticeFermionD src(FGrid);
  gaussian(RNG5, src);

  // reference: Grid FFT_dim_mask (s = dim 0 not transformed)
  FFT theFFT(FGrid);
  Coordinate mask(Nd + 1, 1);
  mask[0] = 0;
  LatticeFermionD ref_k(FGrid);
  theFFT.FFT_dim_mask(ref_k, src, mask, FFT::forward);

  // candidate
  BlockingFFT<LatticeFermionD> bfft(FGrid);
  LatticeFermionD cand_k(FGrid);
  bfft.FFT_spacetime(cand_k, src, FFT::forward);

  double dfwd = std::sqrt(norm2(cand_k - ref_k) / norm2(ref_k));
  std::cout << GridLogMessage << "[blocking-fft] fwd rel diff vs FFT_dim_mask = " << dfwd << std::endl;

  // backward vs reference
  LatticeFermionD ref_x(FGrid);
  theFFT.FFT_dim_mask(ref_x, ref_k, mask, FFT::backward);
  LatticeFermionD cand_x(FGrid);
  bfft.FFT_spacetime(cand_x, cand_k, FFT::backward);
  double dbwd = std::sqrt(norm2(cand_x - ref_x) / norm2(ref_x));
  std::cout << GridLogMessage << "[blocking-fft] bwd rel diff vs FFT_dim_mask = " << dbwd << std::endl;

  // forward-then-backward = identity (candidate alone)
  double drt = std::sqrt(norm2(cand_x - src) / norm2(src));
  std::cout << GridLogMessage << "[blocking-fft] fwd-then-bwd identity rel = " << drt << std::endl;

  bool pass = (dfwd < 1e-12) && (dbwd < 1e-12) && (drt < 1e-12);
  std::cout << GridLogMessage << "==== blocking-fft A/B " << (pass ? "PASS" : "FAIL") << " ===="
            << std::endl;

  Grid_finalize();
  return pass ? 0 : 1;
}
