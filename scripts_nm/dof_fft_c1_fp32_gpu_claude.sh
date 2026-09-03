#!/bin/bash
# C1b PERFORMANCE, FP32 (the F deployment precision). Times DofFFT one-dim vs PlannedFFT one-dim vs the
# GATHER8/GATHER16 bracket + COPY floor, LatticeFermionF, at L=8 and L=16. On V100 fp32 is ~2x throughput
# + half the bytes, so it eases both the arithmetic (binding constraint for the dense O(L^2) DFT) and
# memory -- this is the honest F-path number. Compiles standalone vs the install tree (-I source); NO
# libGrid rebuild. *** TIMING NEEDS A QUIET GPU (nothing else on CUDA_VISIBLE_DEVICES=0). ***
# Read dof_fft_c1_fp32_gpu_claude.log: DofFFT/GATHER and PlannedFFT/DofFFT at each L.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
PERF_SRC=${SRC}/tests/solver/Test_dof_fft_c1_perf_f_claude.cc
PERF_BIN=${BUILD}/Test_dof_fft_c1_perf_f_claude
LOG=${ROOT}/dof_fft_c1_fp32_gpu_claude.log

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

{
  echo "======== compile $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"

  ${CXX} ${CXXFLAGS} -I${SRC} -c ${PERF_SRC} -o ${PERF_BIN}.o
  rc=$?; if [ ${rc} -ne 0 ]; then echo "PERF COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${PERF_BIN}.o -o ${PERF_BIN} ${LDFLAGS} ${LIBS}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "PERF LINK FAILED (rc=${rc})"; exit ${rc}; fi

  echo "======== C1b FP32 PERFORMANCE : L = 8  (needs a QUIET gpu) ========"
  ${PERF_BIN} --grid 8.8.8.8 --mpi 1.1.1.1
  echo "======== C1b FP32 PERFORMANCE : L = 16 (needs a QUIET gpu) ========"
  ${PERF_BIN} --grid 16.16.16.16 --mpi 1.1.1.1
  echo "done"
} 2>&1 | tee ${LOG}
