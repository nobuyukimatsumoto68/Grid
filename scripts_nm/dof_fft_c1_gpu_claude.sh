#!/bin/bash
# C1 gate for the custom DOF-payload FFT (DofFFT_claude.h). TWO parts, correctness BEFORE perf:
#   C1a CORRECTNESS (Test_dof_fft_c0_claude): roundtrip + PlannedFFT-oracle at L=8 AND L=16. Bit-exact,
#       spill-independent. Runs first; if either L FAILS the script stops before timing.
#   C1b PERFORMANCE (Test_dof_fft_c1_perf_claude): times DofFFT one-dim vs PlannedFFT one-dim vs the
#       GATHER8/GATHER16 bracket + COPY floor, back-to-back on the same field, at L=8 and L=16.
#
#   *** C1b TIMING NEEDS A QUIET GPU (no other jobs on CUDA_VISIBLE_DEVICES=0). Correctness (C1a) does
#       not. If the box is busy, the C1a numbers are still valid; re-run for clean C1b timing. ***
#
# GO/NO-GO (for proceeding to C2 = full multi-dim + t): DofFFT/GATHER <~1.5x AND PlannedFFT/DofFFT >1.
# Compiles both tests standalone vs the install tree (-I source for DofFFT_claude.h); NO libGrid rebuild.
# Read dof_fft_c1_gpu_claude.log. Plan: dwf4_qcd_claude/grid_custom_dof_fft_impl_plan_claude.md (sec 3c/4).

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
CORR_SRC=${SRC}/tests/solver/Test_dof_fft_c0_claude.cc
PERF_SRC=${SRC}/tests/solver/Test_dof_fft_c1_perf_claude.cc
CORR_BIN=${BUILD}/Test_dof_fft_c0_claude
PERF_BIN=${BUILD}/Test_dof_fft_c1_perf_claude
LOG=${ROOT}/dof_fft_c1_gpu_claude.log

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

{
  echo "======== compile $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"

  ${CXX} ${CXXFLAGS} -I${SRC} -c ${CORR_SRC} -o ${CORR_BIN}.o
  rc=$?; if [ ${rc} -ne 0 ]; then echo "CORR COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${CORR_BIN}.o -o ${CORR_BIN} ${LDFLAGS} ${LIBS}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "CORR LINK FAILED (rc=${rc})"; exit ${rc}; fi

  ${CXX} ${CXXFLAGS} -I${SRC} -c ${PERF_SRC} -o ${PERF_BIN}.o
  rc=$?; if [ ${rc} -ne 0 ]; then echo "PERF COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${PERF_BIN}.o -o ${PERF_BIN} ${LDFLAGS} ${LIBS}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "PERF LINK FAILED (rc=${rc})"; exit ${rc}; fi

  echo "======== C1a CORRECTNESS : L = 8 ========"
  ${CORR_BIN} --grid 8.8.8.8 --mpi 1.1.1.1
  rc8=$?
  echo "======== C1a CORRECTNESS : L = 16 ========"
  ${CORR_BIN} --grid 16.16.16.16 --mpi 1.1.1.1
  rc16=$?
  if [ ${rc8} -ne 0 ] || [ ${rc16} -ne 0 ]; then
    echo "C1a CORRECTNESS FAILED (rc8=${rc8}, rc16=${rc16}) -- STOP before timing."
    exit 1
  fi
  echo "C1a PASS at both L -- proceeding to C1b timing."

  echo "======== C1b PERFORMANCE : L = 8  (needs a QUIET gpu) ========"
  ${PERF_BIN} --grid 8.8.8.8 --mpi 1.1.1.1
  echo "======== C1b PERFORMANCE : L = 16 (needs a QUIET gpu) ========"
  ${PERF_BIN} --grid 16.16.16.16 --mpi 1.1.1.1
  echo "done"
} 2>&1 | tee ${LOG}
