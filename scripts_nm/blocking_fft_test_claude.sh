#!/bin/bash
# A/B harness for the SIMD-blocking FFT (grid_blocking_fft_impl_plan_claude.md). Compiles
# Test_blocking_fft_claude.cc vs build_mpi and runs it at 4^4 and 8^4 single-rank. Chunk A candidate is a
# pass-through wrapper, so diffs must be ~0; the run also PRINTS simd_layout (confirm [1,1,1,1,2], t sd=2).
# Later chunks (strided cuFFT + t butterfly) reuse this to stay at machine eps. No rm, no kill.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build_mpi
GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_blocking_fft_claude.cc
OBJ=${BUILD}/Test_blocking_fft_claude.o
BIN=${BUILD}/Test_blocking_fft_claude
LOG=${ROOT}/blocking_fft_test_claude.log
MPIRUN=/mnt/hdd_barracuda/opt/openmpi_cuda/bin/mpirun

export OMP_NUM_THREADS=4

{
  echo "======== compile + link (vs build_mpi) $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${TEST} -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "COMPILE FAILED (rc=${rc}); stopping."
    exit ${rc}
  fi
  ${CXX} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "LINK FAILED (rc=${rc}); stopping."
    exit ${rc}
  fi

  echo "======== run 4^4 ========"
  ${MPIRUN} -np 1 ${BIN} --grid 4.4.4.4 --mpi 1.1.1.1
  echo "exit = $?"
  echo "======== run 8^4 ========"
  ${MPIRUN} -np 1 ${BIN} --grid 8.8.8.8 --mpi 1.1.1.1
  echo "exit = $?"
} 2>&1 | tee ${LOG}
