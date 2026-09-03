// DOF-payload FFT proxy benchmark (GO/NO-GO for the custom radix DOF-FFT, BEFORE building it).
//
// The custom FFT's cost is ~98% memory movement; correctness (twiddles/ordering/cross-lane) is irrelevant
// to whether it's WORTH building. So this probe measures only the MEMORY PATTERNS, on the same 5D field
// PlannedFFT uses, and brackets the radix-8 FFT cost between two per-stage kernels:
//   COPY    : streaming read+write of the whole field (1 stage, fully coalesced) = the OPTIMISTIC floor.
//   GATHER8 : each thread reads 8 vObjs at a large stride (nOsites/8), combines, writes 8 (1 stage,
//             strided = the radix-8 butterfly access pattern) = the PESSIMISTIC per-stage cost.
// A radix-8 FFT of L=8 is 1 stage/dim -> 4 stages fwd. Estimate_fwd in [4*COPY, 4*GATHER8]. Compare to the
// measured PlannedFFT fft_fwd. Verdict:
//   4*GATHER8 < PlannedFFT_fwd  -> radix wins even worst-case  -> strong GO
//   4*COPY    > PlannedFFT_fwd  -> radix cannot win            -> NO-GO
//   otherwise -> build the C1 prototype to know.
// Run at L=8,16,32 (the radix advantage shrinks with L -- see grid_custom_dof_fft_impl_plan_claude.md).

#include <Grid/Grid.h>

using namespace Grid;

int main(int argc, char** argv) {
  Grid_init(&argc, &argv);

  const int Ls = 8;
  Coordinate latt = GridDefaultLatt();
  Coordinate simd = GridDefaultSimd(Nd, vComplexD::Nsimd());
  Coordinate mpi = GridDefaultMpi();
  GridCartesian* UGrid = SpaceTimeGrid::makeFourDimGrid(latt, simd, mpi);
  GridCartesian* FGrid = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);

  GridParallelRNG rng(FGrid);
  rng.SeedFixedIntegers(std::vector<int>({1, 2, 3, 4}));
  LatticeFermionD in(FGrid);
  LatticeFermionD out(FGrid);
  random(rng, in);
  out = in;

  const uint64_t nOsites = FGrid->oSites();
  const int Nsimd = (int)FGrid->Nsimd();
  const int reps = 50;
  const int warm = 5;

  std::cout << GridLogMessage << "==== DOF-FFT probe : grid " << latt[0] << "." << latt[1] << "."
            << latt[2] << "." << latt[3] << "  Ls=" << Ls << "  oSites=" << nOsites
            << "  Nsimd=" << Nsimd << " ====" << std::endl;

  // ---- COPY : streaming read+write, one stage (2 field-touches). Optimistic floor. ----
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

  // ---- GATHER8 : radix-8 butterfly memory pattern. Each of nOsites/8 threads reads 8 vObjs at stride
  // D=nOsites/8, combines (mock, distinct per output to defeat DCE), writes 8. One read + one write of the
  // field, STRIDED. Pessimistic per-stage cost. ----
  const uint64_t D = nOsites / 8;
  for (int r = 0; r < warm; ++r) {
    autoView(iv, in, AcceleratorRead);
    autoView(ov, out, AcceleratorWrite);
    accelerator_for(g, D, Nsimd, {
      auto a0 = coalescedRead(iv[g + 0 * D]);
      auto a1 = coalescedRead(iv[g + 1 * D]);
      auto a2 = coalescedRead(iv[g + 2 * D]);
      auto a3 = coalescedRead(iv[g + 3 * D]);
      auto a4 = coalescedRead(iv[g + 4 * D]);
      auto a5 = coalescedRead(iv[g + 5 * D]);
      auto a6 = coalescedRead(iv[g + 6 * D]);
      auto a7 = coalescedRead(iv[g + 7 * D]);
      auto s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
      coalescedWrite(ov[g + 0 * D], s);
      coalescedWrite(ov[g + 1 * D], s - a1);
      coalescedWrite(ov[g + 2 * D], s - a2);
      coalescedWrite(ov[g + 3 * D], s - a3);
      coalescedWrite(ov[g + 4 * D], s - a4);
      coalescedWrite(ov[g + 5 * D], s - a5);
      coalescedWrite(ov[g + 6 * D], s - a6);
      coalescedWrite(ov[g + 7 * D], s - a7);
    });
  }
  accelerator_barrier();
  double tg8 = -usecond();
  for (int r = 0; r < reps; ++r) {
    autoView(iv, in, AcceleratorRead);
    autoView(ov, out, AcceleratorWrite);
    accelerator_for(g, D, Nsimd, {
      auto a0 = coalescedRead(iv[g + 0 * D]);
      auto a1 = coalescedRead(iv[g + 1 * D]);
      auto a2 = coalescedRead(iv[g + 2 * D]);
      auto a3 = coalescedRead(iv[g + 3 * D]);
      auto a4 = coalescedRead(iv[g + 4 * D]);
      auto a5 = coalescedRead(iv[g + 5 * D]);
      auto a6 = coalescedRead(iv[g + 6 * D]);
      auto a7 = coalescedRead(iv[g + 7 * D]);
      auto s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
      coalescedWrite(ov[g + 0 * D], s);
      coalescedWrite(ov[g + 1 * D], s - a1);
      coalescedWrite(ov[g + 2 * D], s - a2);
      coalescedWrite(ov[g + 3 * D], s - a3);
      coalescedWrite(ov[g + 4 * D], s - a4);
      coalescedWrite(ov[g + 5 * D], s - a5);
      coalescedWrite(ov[g + 6 * D], s - a6);
      coalescedWrite(ov[g + 7 * D], s - a7);
    });
  }
  accelerator_barrier();
  tg8 += usecond();
  tg8 /= reps;

  // ---- GATHER16 : radix-16 memory pattern (16 strided vObjs/thread). Tests fewer stages (1/dim at L=16)
  // vs heavier register load (16 vObjs -> possible spill). One field read + write, strided by nOsites/16. ----
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

  // ---- PlannedFFT reference : masked 4D spacetime FFT (s = dim 0 not transformed), fwd + bwd. ----
  PlannedFFT<typename LatticeFermionD::vector_object> pfft(FGrid);
  Coordinate mask(Nd + 1, 1);
  mask[0] = 0;
  LatticeFermionD ktmp(FGrid);
  for (int r = 0; r < warm; ++r) {
    pfft.FFT_dim_mask(ktmp, in, mask, FFT::forward);
  }
  accelerator_barrier();
  double tfwd = -usecond();
  for (int r = 0; r < reps; ++r) {
    pfft.FFT_dim_mask(ktmp, in, mask, FFT::forward);
  }
  accelerator_barrier();
  tfwd += usecond();
  tfwd /= reps;

  double tbwd = -usecond();
  for (int r = 0; r < reps; ++r) {
    pfft.FFT_dim_mask(out, ktmp, mask, FFT::backward);
  }
  accelerator_barrier();
  tbwd += usecond();
  tbwd /= reps;

  // ---- report + verdict. radix-8 at L: stages/dim = ceil(log8(L)); here report for L (assume power of 2). ----
  int L = latt[0];
  int stages_per_dim = 1;
  {
    int LL = L;
    while (LL > 8) {
      LL /= 8;
      stages_per_dim++;
    }
    if (L > 1 && L < 8) stages_per_dim = 1;  // small L -> one stage
  }
  int stages_fwd = 4 * stages_per_dim;  // 4 spacetime dims
  double est_opt = stages_fwd * tcopy;
  double est_pes = stages_fwd * tg8;

  int stages16_per_dim = 1;
  {
    int LL = L;
    while (LL > 16) {
      LL /= 16;
      stages16_per_dim++;
    }
  }
  int stages16_fwd = 4 * stages16_per_dim;
  double est16_pes = stages16_fwd * tg16;

  std::cout << GridLogMessage << "  COPY     (1 stage, streaming)       = " << tcopy << " us" << std::endl;
  std::cout << GridLogMessage << "  GATHER8  (1 stage, strided radix8)  = " << tg8 << " us" << std::endl;
  std::cout << GridLogMessage << "  GATHER16 (1 stage, strided radix16) = " << tg16 << " us   (per-stage; "
            << tg16 / tg8 << "x GATHER8 -- <2x => radix-16 halves stages profitably)" << std::endl;
  std::cout << GridLogMessage << "  PlannedFFT fft_fwd                = " << tfwd << " us" << std::endl;
  std::cout << GridLogMessage << "  PlannedFFT fft_bwd                = " << tbwd << " us" << std::endl;
  std::cout << GridLogMessage << "  radix-8 stages/dim=" << stages_per_dim << " -> " << stages_fwd
            << " stages fwd" << std::endl;
  std::cout << GridLogMessage << "  EST radix fwd in [" << est_opt << " (copy floor), " << est_pes
            << " (gather8 ceiling)] us   vs PlannedFFT fwd " << tfwd << " us" << std::endl;
  std::cout << GridLogMessage << "  ratio PlannedFFT/EST in [" << tfwd / est_pes << " (worst), "
            << tfwd / est_opt << " (best)] x" << std::endl;
  std::cout << GridLogMessage << "  radix-16 stages/dim=" << stages16_per_dim << " -> " << stages16_fwd
            << " stages fwd; EST radix-16 fwd (gather16 ceiling) = " << est16_pes << " us  (vs radix-8 "
            << est_pes << ")  -> radix-" << (est16_pes < est_pes ? "16" : "8") << " better (pessimistic)"
            << std::endl;
  if (est_pes < tfwd) {
    std::cout << GridLogMessage << "  VERDICT: strong GO (radix wins even worst-case)" << std::endl;
  } else if (est_opt > tfwd) {
    std::cout << GridLogMessage << "  VERDICT: NO-GO (radix cannot beat PlannedFFT)" << std::endl;
  } else {
    std::cout << GridLogMessage << "  VERDICT: build the C1 prototype (bracket straddles PlannedFFT)" << std::endl;
  }

  Grid_finalize();
  return 0;
}
