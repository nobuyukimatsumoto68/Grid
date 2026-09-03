#!/bin/bash
# DOF-FFT proxy benchmark: brackets the radix-8 DOF-FFT cost (streaming COPY floor .. strided GATHER8
# ceiling) vs PlannedFFT, at L=8,16,32 -- a GO/NO-GO before building the custom FFT. Compiles the probe
# standalone against the merged Grid install tree (PlannedFFT). Run on a CLEAN gpu (timing).
# Read the VERDICT line at each L. grid_custom_dof_fft_impl_plan_claude.md.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_dof_fft_probe_claude.cc
OBJ=${BUILD}/Test_dof_fft_probe_claude.o
BIN=${BUILD}/Test_dof_fft_probe_claude
LOG=${ROOT}/dof_fft_probe_gpu_claude.log

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

{
  echo "======== compile $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${TEST} -o ${OBJ}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "LINK FAILED (rc=${rc})"; exit ${rc}; fi

  for G in 8.8.8.8 16.16.16.16 32.32.32.32; do
    echo "======== run L = ${G} ========"
    ${BIN} --grid ${G} --mpi 1.1.1.1
  done
  echo "done"
} 2>&1 | tee ${LOG}
