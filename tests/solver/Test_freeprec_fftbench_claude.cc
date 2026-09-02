// Chunk 3: standalone FFT scaling bench for the free-limit Mobius DWF preconditioner.
//
// Isolates the (A) FFT lever WITHOUT the preconditioner: it times the same masked 4D FFT the
// preconditioner uses (s = dim 0 NOT transformed, mask[0]=0) applied fwd+bwd to a 5D fermion
// field, and times a Mobius Dhop apply as the "D_W-apply" currency this project measures wins in.
// Reports fwd+bwd FFT us/apply, Dhop us/apply, and FFT/Dhop. Run with `--log Performance` to also
// get FFT.h's per-dim t_shift[comm]/t_fft[kernel]/t_copy/t_insert split -- the comm-vs-compute
// breakdown that decides whether the Cshift-ring (FFT.h:367-417) needs a distributed-transpose
// rewrite. Compare across --mpi decompositions for weak/strong scaling (see fftbench_run_claude.sh).
//
// Grid FFT engine: Grid/algorithms/FFT.h. Mobius/Cayley kernel Brower-Neff-Orginos arXiv:1206.5214.

#include <Grid/Grid.h>

using namespace Grid;

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  const int Ls = 8;
  const int Nit = 50;

  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplexD::Nsimd());
  Coordinate mpi = GridDefaultMpi();
  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(latt, simd, mpi);
  GridRedBlackCartesian* UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian* FGrid = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
  GridRedBlackCartesian* FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);

  int Nranks = UGrid->_Nprocessors;
  std::cout << GridLogMessage << "==== FFT scaling bench: grid " << latt << " Ls " << Ls
            << " mpi " << mpi << " (" << Nranks << " ranks), Nit " << Nit << " ====" << std::endl;

  // ---- random 5D fermion source ----
  GridParallelRNG RNG5(FGrid);
  RNG5.SeedFixedIntegers(std::vector<int>({1, 2, 3, 4}));
  LatticeFermionD src(FGrid);
  gaussian(RNG5, src);
  LatticeFermionD kbuf(FGrid);
  LatticeFermionD out(FGrid);

  // ---- masked 4D FFT (s = dim 0 not transformed), matching the preconditioner apply ----
  FFT theFFT(FGrid);
  Coordinate mask(Nd + 1, 1);
  mask[0] = 0;

  // warm up: FFTW/cuFFT plan creation + first-touch allocation, excluded from the timing
  theFFT.FFT_dim_mask(kbuf, src, mask, FFT::forward);
  theFFT.FFT_dim_mask(out, kbuf, mask, FFT::backward);

  double tfft = -usecond();
  for (int i = 0; i < Nit; ++i) {
    theFFT.FFT_dim_mask(kbuf, src, mask, FFT::forward);
    theFFT.FFT_dim_mask(out, kbuf, mask, FFT::backward);
  }
  tfft += usecond();
  double fft_per = tfft / (double)Nit;  // us per (fwd + bwd)

  // ---- D_W currency: Mobius Dhop on unit gauge (Dhop cost is gauge-value-independent) ----
  LatticeGaugeFieldD Umu(UGrid);
  SU<Nc>::ColdConfiguration(Umu);  // identity links
  const double M5 = 1.8;
  const double bb = 1.5;
  const double cc = 0.5;
  const double mm = 0.1;
  std::vector<Complex> boundary = {1, 1, 1, -1};  // anti-periodic time
  MobiusFermionD::ImplParams Params(boundary);
  MobiusFermionD D(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mm, M5, bb, cc, Params);

  LatticeFermionD dwsrc(FGrid);
  gaussian(RNG5, dwsrc);
  LatticeFermionD dwout(FGrid);
  D.Dhop(dwsrc, dwout, 0);  // warm up
  double tdw = -usecond();
  for (int i = 0; i < Nit; ++i) {
    D.Dhop(dwsrc, dwout, 0);
  }
  tdw += usecond();
  double dw_per = tdw / (double)Nit;  // us per Dhop

  std::cout << GridLogMessage << "[fftbench] ranks=" << Nranks
            << "  fwd+bwd FFT us/apply = " << fft_per
            << "  Dhop us/apply = " << dw_per
            << "  FFT/Dhop = " << (fft_per / dw_per) << std::endl;
  std::cout << GridLogMessage << "[fftbench] run with --log Performance for the per-dim "
            << "t_shift[comm] / t_fft[kernel] split from FFT.h" << std::endl;

  Grid_finalize();
  return 0;
}
