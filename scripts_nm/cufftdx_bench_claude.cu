// cuFFTDx prototype gate (grid_blocking_fft_impl_plan_claude.md): is an in-kernel cuFFTDx length-8
// C2C double FFT competitive with cuFFT for our workload? First gate = raw batched throughput on
// contiguous data. If cuFFTDx is not competitive here, the fused-apply direction is not worth building.
// Standalone (no Grid). Idea/direction: user. cuFFTDx 1.2.1 (MathDx 24.08), sm_70.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftdx.hpp>

using namespace cufftdx;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA err %s:%d : %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

static const int NP = 8;   // FFT length (matches our lattice extent)
static const int FPB = 128;  // FFTs per block

using FFT = decltype(Size<NP>() + Precision<double>() + Type<fft_type::c2c>()
                     + Direction<fft_direction::forward>()
                     + ElementsPerThread<NP>() + FFTsPerBlock<FPB>() + SM<700>() + Block());
using cpx = typename FFT::value_type;

template <class F>
__launch_bounds__(F::max_threads_per_block)
__global__ void dxk(typename F::value_type* data, typename F::workspace_type ws) {
  using c = typename F::value_type;
  c td[F::storage_size];
  const unsigned lfid = threadIdx.y;
  const unsigned gfid = blockIdx.x * F::ffts_per_block + lfid;
  const unsigned off = size_of<F>::value * gfid;
  const unsigned st = F::stride;
  unsigned idx = off + threadIdx.x;
  for (unsigned i = 0; i < F::elements_per_thread; i++) {
    if ((i * st + threadIdx.x) < size_of<F>::value) {
      td[i] = data[idx];
      idx += st;
    }
  }
  extern __shared__ __align__(alignof(double2)) c smem[];
  F().execute(td, smem, ws);
  idx = off + threadIdx.x;
  for (unsigned i = 0; i < F::elements_per_thread; i++) {
    if ((i * st + threadIdx.x) < size_of<F>::value) {
      data[idx] = td[i];
      idx += st;
    }
  }
}

int main() {
  const long BATCH = 1L << 18;   // number of length-8 FFTs
  const long N = BATCH * NP;

  cpx* d;
  cpx* d2;
  CK(cudaMallocManaged(&d, N * sizeof(cpx)));
  CK(cudaMallocManaged(&d2, N * sizeof(cpx)));
  for (long i = 0; i < N; i++) {
    d[i].x = std::sin(0.1 * i);
    d[i].y = std::cos(0.2 * i);
    d2[i] = d[i];
  }

  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  cudaError_t ec = cudaSuccess;
  auto ws = make_workspace<FFT>(ec, s);
  CK(ec);
  CK(cudaFuncSetAttribute(dxk<FFT>, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));
  long nb = BATCH / FFT::ffts_per_block;

  // cuFFT reference plan (batched length-8)
  cufftHandle plan;
  int n[1] = {NP};
  cufftPlanMany(&plan, 1, n, n, 1, NP, n, 1, NP, CUFFT_Z2Z, BATCH);
  cufftSetStream(plan, s);

  // ---- correctness: single forward, compare cuFFTDx (d) vs cuFFT (d2) on the 1st few FFTs ----
  dxk<FFT><<<nb, FFT::block_dim, FFT::shared_memory_size, s>>>(d, ws);
  cufftExecZ2Z(plan, (cufftDoubleComplex*)d2, (cufftDoubleComplex*)d2, CUFFT_FORWARD);
  CK(cudaStreamSynchronize(s));
  double maxd = 0.0;
  double nrm = 0.0;
  for (long i = 0; i < NP * 1024; i++) {
    double dr = d[i].x - d2[i].x;
    double di = d[i].y - d2[i].y;
    maxd += dr * dr + di * di;
    nrm += d2[i].x * d2[i].x + d2[i].y * d2[i].y;
  }
  printf("correctness: cuFFTDx vs cuFFT rel diff = %.3e\n", std::sqrt(maxd / nrm));

  // ---- timing ----
  int reps = 100;
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);

  for (int r = 0; r < 5; r++) {
    dxk<FFT><<<nb, FFT::block_dim, FFT::shared_memory_size, s>>>(d, ws);
  }
  CK(cudaStreamSynchronize(s));
  cudaEventRecord(a, s);
  for (int r = 0; r < reps; r++) {
    dxk<FFT><<<nb, FFT::block_dim, FFT::shared_memory_size, s>>>(d, ws);
  }
  cudaEventRecord(b, s);
  CK(cudaStreamSynchronize(s));
  float ms_dx = 0;
  cudaEventElapsedTime(&ms_dx, a, b);

  for (int r = 0; r < 5; r++) {
    cufftExecZ2Z(plan, (cufftDoubleComplex*)d2, (cufftDoubleComplex*)d2, CUFFT_FORWARD);
  }
  CK(cudaStreamSynchronize(s));
  cudaEventRecord(a, s);
  for (int r = 0; r < reps; r++) {
    cufftExecZ2Z(plan, (cufftDoubleComplex*)d2, (cufftDoubleComplex*)d2, CUFFT_FORWARD);
  }
  cudaEventRecord(b, s);
  CK(cudaStreamSynchronize(s));
  float ms_cf = 0;
  cudaEventElapsedTime(&ms_cf, a, b);

  printf("length-%d batched C2C double, BATCH=%ld, reps=%d\n", NP, BATCH, reps);
  printf("cuFFTDx : %.4f ms/rep\n", ms_dx / reps);
  printf("cuFFT   : %.4f ms/rep\n", ms_cf / reps);
  printf("ratio (cuFFTDx / cuFFT) = %.3f  (<1 => cuFFTDx faster)\n", ms_dx / ms_cf);

  cufftDestroy(plan);
  CK(cudaFree(d));
  CK(cudaFree(d2));
  return 0;
}
