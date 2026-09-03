// C1b PERFORMANCE gate for the custom DOF-payload FFT (DofFFT_claude.h): does the real butterfly (with
// twiddle arithmetic) confirm the memory-pattern proxy (Test_dof_fft_probe: GATHER8/16 vs PlannedFFT)?
//
// Times, back-to-back on the SAME field, ONE transformed (unsplit) dim, forward:
//   - DofFFT.FFT_dim(dim)                 : the real direct-DFT (twiddle matvec, one stage).
//   - PlannedFFT.FFT_dim_mask(mask=1 dim) : native Grid oracle, same single dim.
//   - GATHER8 / GATHER16                  : the probe's per-stage strided memory pattern (the bracket).
//   - COPY                                : streaming floor.
// Reports DofFFT/GATHER ratio (twiddle overhead over the pure pattern) and PlannedFFT/DofFFT (the win).
// TIMING NEEDS A QUIET GPU. Correctness is covered by Test_dof_fft_c0_claude (run that too).
//
// At L=8 the relevant bracket is GATHER8 (radix-8, 1 stage/dim); at L=16 it is GATHER16 (radix-16, 1
// stage/dim). DofFFT.FFT_dim is ONE stage at both (direct L-point DFT). Plan sec 3c/3d/4b.

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

  const uint64_t nOsites = FGrid->oSites();
  const int Nsimd = (int)FGrid->Nsimd();
  const int reps = 50;
  const int warm = 5;
  const int L = latt[0];

  // pick the first UNSPLIT spacetime dim (1..4), as in C0.
  int xdim = -1;
  for (int d = 1; d <= 4; ++d) {
    if (FGrid->_simd_layout[d] == 1) {
      xdim = d;
      break;
    }
  }
  GRID_ASSERT(xdim > 0);

  std::cout << GridLogMessage << "==== DofFFT C1b perf : grid " << latt[0] << "." << latt[1] << "."
            << latt[2] << "." << latt[3] << "  Ls=" << Ls << "  oSites=" << nOsites << "  Nsimd=" << Nsimd
            << "  simd5=" << FGrid->_simd_layout << " ====" << std::endl;
  std::cout << GridLogMessage << "  transformed dim = " << xdim << " (unsplit); L = " << L << std::endl;

  GridParallelRNG rng(FGrid);
  rng.SeedFixedIntegers(std::vector<int>({1, 2, 3, 4}));
  LatticeFermionD in(FGrid);
  LatticeFermionD out(FGrid);
  random(rng, in);
  out = in;

  // ---- DofFFT one-dim (forward), timed. ----
  DofFFT_claude<typename LatticeFermionD::vector_object> dfft(FGrid);
  LatticeFermionD kdof(FGrid);
  for (int r = 0; r < warm; ++r)
    dfft.FFT_dim(kdof, in, xdim, DofFFT_claude<typename LatticeFermionD::vector_object>::forward);
  accelerator_barrier();
  double tdof = -usecond();
  for (int r = 0; r < reps; ++r)
    dfft.FFT_dim(kdof, in, xdim, DofFFT_claude<typename LatticeFermionD::vector_object>::forward);
  accelerator_barrier();
  tdof += usecond();
  tdof /= reps;

  // ---- PlannedFFT one-dim (forward), same dim, timed. ----
  PlannedFFT<typename LatticeFermionD::vector_object> pfft(FGrid);
  Coordinate mask(Nd + 1, 0);
  mask[xdim] = 1;
  LatticeFermionD kpln(FGrid);
  for (int r = 0; r < warm; ++r)
    pfft.FFT_dim_mask(kpln, in, mask, FFT::forward);
  accelerator_barrier();
  double tpln = -usecond();
  for (int r = 0; r < reps; ++r)
    pfft.FFT_dim_mask(kpln, in, mask, FFT::forward);
  accelerator_barrier();
  tpln += usecond();
  tpln /= reps;

  // ---- COPY : streaming floor (1 stage, coalesced). ----
  for (int r = 0; r < warm; ++r) {
    autoView(iv, in, AcceleratorRead);
    autoView(ov, out, AcceleratorWrite);
    accelerator_for(ss, nOsites, Nsimd, { coalescedWrite(ov[ss], coalescedRead(iv[ss])); });
  }
  accelerator_barrier();
  double tcopy = -usecond();
  for (int r = 0; r < reps; ++r) {
    autoView(iv, in, AcceleratorRead);
    autoView(ov, out, AcceleratorWrite);
    accelerator_for(ss, nOsites, Nsimd, { coalescedWrite(ov[ss], coalescedRead(iv[ss])); });
  }
  accelerator_barrier();
  tcopy += usecond();
  tcopy /= reps;

  // ---- GATHER8 : radix-8 memory pattern (8 strided vObjs/thread, 1 read + 1 write). ----
  const uint64_t D8 = nOsites / 8;
  for (int r = 0; r < warm; ++r) {
    autoView(iv, in, AcceleratorRead);
    autoView(ov, out, AcceleratorWrite);
    accelerator_for(g, D8, Nsimd, {
      decltype(coalescedRead(iv[0])) a[8];
      for (int k = 0; k < 8; ++k) a[k] = coalescedRead(iv[g + (uint64_t)k * D8]);
      auto s = a[0];
      for (int k = 1; k < 8; ++k) s = s + a[k];
      for (int k = 0; k < 8; ++k) coalescedWrite(ov[g + (uint64_t)k * D8], s - a[k]);
    });
  }
  accelerator_barrier();
  double tg8 = -usecond();
  for (int r = 0; r < reps; ++r) {
    autoView(iv, in, AcceleratorRead);
    autoView(ov, out, AcceleratorWrite);
    accelerator_for(g, D8, Nsimd, {
      decltype(coalescedRead(iv[0])) a[8];
      for (int k = 0; k < 8; ++k) a[k] = coalescedRead(iv[g + (uint64_t)k * D8]);
      auto s = a[0];
      for (int k = 1; k < 8; ++k) s = s + a[k];
      for (int k = 0; k < 8; ++k) coalescedWrite(ov[g + (uint64_t)k * D8], s - a[k]);
    });
  }
  accelerator_barrier();
  tg8 += usecond();
  tg8 /= reps;

  // ---- GATHER16 : radix-16 memory pattern (16 strided vObjs/thread). ----
  const uint64_t D16 = nOsites / 16;
  for (int r = 0; r < warm; ++r) {
    autoView(iv, in, AcceleratorRead);
    autoView(ov, out, AcceleratorWrite);
    accelerator_for(g, D16, Nsimd, {
      decltype(coalescedRead(iv[0])) a[16];
      for (int k = 0; k < 16; ++k) a[k] = coalescedRead(iv[g + (uint64_t)k * D16]);
      auto s = a[0];
      for (int k = 1; k < 16; ++k) s = s + a[k];
      for (int k = 0; k < 16; ++k) coalescedWrite(ov[g + (uint64_t)k * D16], s - a[k]);
    });
  }
  accelerator_barrier();
  double tg16 = -usecond();
  for (int r = 0; r < reps; ++r) {
    autoView(iv, in, AcceleratorRead);
    autoView(ov, out, AcceleratorWrite);
    accelerator_for(g, D16, Nsimd, {
      decltype(coalescedRead(iv[0])) a[16];
      for (int k = 0; k < 16; ++k) a[k] = coalescedRead(iv[g + (uint64_t)k * D16]);
      auto s = a[0];
      for (int k = 1; k < 16; ++k) s = s + a[k];
      for (int k = 0; k < 16; ++k) coalescedWrite(ov[g + (uint64_t)k * D16], s - a[k]);
    });
  }
  accelerator_barrier();
  tg16 += usecond();
  tg16 /= reps;

  // ---- report. bracket = GATHER8 at L=8, GATHER16 at L=16 (each is 1 stage/dim there). ----
  double tgather = (L <= 8) ? tg8 : tg16;
  const char* gname = (L <= 8) ? "GATHER8" : "GATHER16";
  std::cout << GridLogMessage << "  DofFFT one-dim   = " << tdof  << " us   (real direct-DFT, 1 stage)" << std::endl;
  std::cout << GridLogMessage << "  PlannedFFT 1-dim = " << tpln  << " us   (native, same dim)" << std::endl;
  std::cout << GridLogMessage << "  COPY  (floor)    = " << tcopy << " us" << std::endl;
  std::cout << GridLogMessage << "  GATHER8          = " << tg8   << " us" << std::endl;
  std::cout << GridLogMessage << "  GATHER16         = " << tg16  << " us" << std::endl;
  std::cout << GridLogMessage << "  --- ratios (bracket = " << gname << " for L=" << L << ") ---" << std::endl;
  std::cout << GridLogMessage << "  DofFFT / " << gname << "  = " << tdof / tgather
            << " x   (twiddle-arith overhead over pure memory pattern; want <~1.5x)" << std::endl;
  std::cout << GridLogMessage << "  PlannedFFT / DofFFT = " << tpln / tdof
            << " x   (the one-dim win; want clearly >1)" << std::endl;

  Grid_finalize();
  return 0;
}
