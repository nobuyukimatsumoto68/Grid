// C1b PERFORMANCE gate, FP32 twin (LatticeFermionF) of Test_dof_fft_c1_perf_claude.cc. fp32 is the F
// deployment precision (FreeMobius5D default), so THIS is the honest F-path number. On V100 fp32 is ~2x
// throughput + half the bytes vs fp64 -> helps BOTH the arithmetic (the binding constraint for the dense
// O(L^2) DFT) and memory. Times DofFFT one-dim vs PlannedFFT one-dim vs GATHER8/16 + COPY, same field,
// L=8 and 16, forward. TIMING NEEDS A QUIET GPU. Plan sec 3c/4; fp32 default = memory grid-fft-profile.

#include <Grid/Grid.h>
#include <Grid/algorithms/DofFFT_claude.h>

using namespace Grid;

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  const int Ls = 8;
  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplexF::Nsimd());   // fp32 SIMD layout
  Coordinate mpi  = GridDefaultMpi();
  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(latt, simd, mpi);
  GridCartesian* FGrid = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);

  const uint64_t nOsites = FGrid->oSites();
  const int Nsimd = (int)FGrid->Nsimd();
  const int reps = 50;
  const int warm = 5;
  const int L = latt[0];

  // pick the first UNSPLIT spacetime dim (1..4). fp32 has a larger Nsimd, so more dims may be split.
  int xdim = -1;
  for (int d = 1; d <= 4; ++d) {
    if (FGrid->_simd_layout[d] == 1) {
      xdim = d;
      break;
    }
  }
  GRID_ASSERT(xdim > 0);

  std::cout << GridLogMessage << "==== DofFFT C1b perf FP32 : grid " << latt[0] << "." << latt[1] << "."
            << latt[2] << "." << latt[3] << "  Ls=" << Ls << "  oSites=" << nOsites << "  Nsimd=" << Nsimd
            << "  simd5=" << FGrid->_simd_layout << " ====" << std::endl;
  std::cout << GridLogMessage << "  transformed dim = " << xdim << " (unsplit); L = " << L << std::endl;

  GridParallelRNG rng(FGrid);
  rng.SeedFixedIntegers(std::vector<int>({1, 2, 3, 4}));
  LatticeFermionF in(FGrid);
  LatticeFermionF out(FGrid);
  random(rng, in);
  out = in;

  // ---- DofFFT one-dim (forward), timed. ----
  DofFFT_claude<typename LatticeFermionF::vector_object> dfft(FGrid);
  LatticeFermionF kdof(FGrid);
  for (int r = 0; r < warm; ++r)
    dfft.FFT_dim(kdof, in, xdim, DofFFT_claude<typename LatticeFermionF::vector_object>::forward);
  accelerator_barrier();
  double tdof = -usecond();
  for (int r = 0; r < reps; ++r)
    dfft.FFT_dim(kdof, in, xdim, DofFFT_claude<typename LatticeFermionF::vector_object>::forward);
  accelerator_barrier();
  tdof += usecond();
  tdof /= reps;

  // ---- PlannedFFT one-dim (forward), same dim, timed. ----
  PlannedFFT<typename LatticeFermionF::vector_object> pfft(FGrid);
  Coordinate mask(Nd + 1, 0);
  mask[xdim] = 1;
  LatticeFermionF kpln(FGrid);
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

  double tgather = (L <= 8) ? tg8 : tg16;
  const char* gname = (L <= 8) ? "GATHER8" : "GATHER16";
  std::cout << GridLogMessage << "  DofFFT one-dim   = " << tdof  << " us   (real direct-DFT, fp32, 1 stage)" << std::endl;
  std::cout << GridLogMessage << "  PlannedFFT 1-dim = " << tpln  << " us   (native fp32, same dim)" << std::endl;
  std::cout << GridLogMessage << "  COPY  (floor)    = " << tcopy << " us" << std::endl;
  std::cout << GridLogMessage << "  GATHER8          = " << tg8   << " us" << std::endl;
  std::cout << GridLogMessage << "  GATHER16         = " << tg16  << " us" << std::endl;
  std::cout << GridLogMessage << "  --- ratios (bracket = " << gname << " for L=" << L << ", FP32) ---" << std::endl;
  std::cout << GridLogMessage << "  DofFFT / " << gname << "  = " << tdof / tgather
            << " x   (twiddle-arith overhead over pure memory pattern; want <~1.5x)" << std::endl;
  std::cout << GridLogMessage << "  PlannedFFT / DofFFT = " << tpln / tdof
            << " x   (the one-dim win; want clearly >1)" << std::endl;

  Grid_finalize();
  return 0;
}
