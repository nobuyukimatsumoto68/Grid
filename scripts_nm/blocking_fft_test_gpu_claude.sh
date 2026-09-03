#!/bin/bash
# GPU (cuFFT path) validation of the SIMD-blocking FFT: builds Test_blocking_fft_claude.cc vs the CUDA
# build/, runs on gpu0. The cuFFT coarse path + the on-device butterfly must give the same roundtrip
# identity (~machine eps) as the CPU/FFTW path. Forward vs Grid still differs (t digit-reversal, expected).

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_blocking_fft_claude.cc
OBJ=${BUILD}/Test_blocking_fft_claude.o
BIN=${BUILD}/Test_blocking_fft_claude
LOG=${ROOT}/blocking_fft_test_gpu_claude.log

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

{
  echo "======== compile + link (nvcc; CXXLD swaps -x cu -> -link) $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${TEST} -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "COMPILE FAILED (rc=${rc}); stopping."
    exit ${rc}
  fi
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "LINK FAILED (rc=${rc}); stopping."
    exit ${rc}
  fi

  echo "======== run 4^4 (gpu0) ========"
  ${BIN} --grid 4.4.4.4 --mpi 1.1.1.1
  echo "exit = $?"
  echo "======== run 8^4 (gpu0) ========"
  ${BIN} --grid 8.8.8.8 --mpi 1.1.1.1
  echo "exit = $?"
} 2>&1 | tee ${LOG}
