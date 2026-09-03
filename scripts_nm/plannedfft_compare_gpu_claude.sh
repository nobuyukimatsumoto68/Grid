#!/bin/bash
# Compare our FFT_claude (cached plan + cached pgbuf) vs upstream's native PlannedFFT (cached plan, but
# re-allocs pgbuf every call -- FFT_dim_execute). Both in the fp32 DEFAULT regime. Builds two binaries and
# runs each 8^4 --config --ops m0 back-to-back. CORRECTNESS (load-independent): both must PASS gate 1 +
# converge in 79 iters (confirms PlannedFFT is a valid drop-in). TIMING (needs a QUIET gpu to trust): compare
# the [F timers] fft_fwd + fft_bwd -- if PlannedFFT ties FFT_claude, we can RETIRE our copy for the native one;
# if the per-call pgbuf alloc hurts, FFT_claude stays. grid-fft-profile memory.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_dwf_freeprec_claude.cc
LOG=${ROOT}/plannedfft_compare_gpu_claude.log
GRID=8.8.8.8
CFG=/mnt/baracuda_14/dwms/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

build_run () {
  local tag="$1"
  local extra="$2"
  local obj="${BUILD}/Test_dwf_freeprec_${tag}_claude.o"
  local bin="${BUILD}/Test_dwf_freeprec_${tag}_claude"
  echo "======== compile ${tag} (${extra}) $(date) ========"
  ${CXX} ${CXXFLAGS} ${extra} -I${SRC} -c ${TEST} -o ${obj}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "${tag} COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${obj} -o ${bin} ${LDFLAGS} ${LIBS}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "${tag} LINK FAILED (rc=${rc})"; exit ${rc}; fi
  echo "======== run ${tag} 8^4 --config --ops m0 ========"
  ${bin} --grid ${GRID} --mpi 1.1.1.1 --config ${CFG} --ops m0
}

{
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"

  build_run fftclaude ""                              # our FFT_claude (default fp32)
  build_run plannedfft "-DFREEMOBIUS5D_PLANNED_FFT"   # upstream native PlannedFFT (fp32)
  echo "done"
} 2>&1 | tee ${LOG}
