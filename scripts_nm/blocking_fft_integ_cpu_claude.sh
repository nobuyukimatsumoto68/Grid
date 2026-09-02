#!/bin/bash
# Chunk D validation (CPU/FFTW path): build Test_dwf_freeprec_claude.cc with -DFREEMOBIUS5D_BLOCKING_FFT
# (blocking FFT + re-keyed Minv_dev) vs build_mpi, run 4^4 no-config. DECISIVE: gate 1 cold gate
# ||F D v - v|| ~ machine eps (proves blocking FFT + re-key give a correct F). gate 0a/0b/2 also checked.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build_mpi
GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_dwf_freeprec_claude.cc
OBJ=${BUILD}/Test_dwf_freeprec_blk_claude.o
BIN=${BUILD}/Test_dwf_freeprec_blk_claude
LOG=${ROOT}/blocking_fft_integ_cpu_claude.log
MPIRUN=/mnt/hdd_barracuda/opt/openmpi_cuda/bin/mpirun

export OMP_NUM_THREADS=4

{
  echo "======== compile + link (build_mpi, -DFREEMOBIUS5D_BLOCKING_FFT) $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  ${CXX} ${CXXFLAGS} -DFREEMOBIUS5D_BLOCKING_FFT -I${SRC} -c ${TEST} -o ${OBJ}
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

  echo "======== run 4^4 no-config (watch gate 1) ========"
  ${MPIRUN} -np 1 ${BIN} --grid 4.4.4.4 --mpi 1.1.1.1 --ops m0
  echo "exit = $?"
} 2>&1 | tee ${LOG}
