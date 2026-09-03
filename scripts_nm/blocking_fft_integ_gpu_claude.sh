#!/bin/bash
# Chunk D validation + measurement on GPU (cuFFT path): build Test_dwf_freeprec_claude.cc with
# -DFREEMOBIUS5D_BLOCKING_FFT vs the CUDA build/, run 8^4 on the SU(3) config with --ops m0, on gpu0.
# Checks: gate 1 cold gate ~ machine eps (blocking FFT + re-key correct on GPU); then the [F timers]
# fft_fwd/fft_bwd -- does the blocking FFT (no pack) drop the FFT us vs Grid's packed cuFFT (was ~2200/2000)?

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_dwf_freeprec_claude.cc
OBJ=${BUILD}/Test_dwf_freeprec_blk_claude.o
BIN=${BUILD}/Test_dwf_freeprec_blk_claude
LOG=${ROOT}/blocking_fft_integ_gpu_claude.log
GRID=8.8.8.8
CFG=/mnt/baracuda_14/dwms/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

{
  echo "======== compile + link (build/, nvcc, -DFREEMOBIUS5D_BLOCKING_FFT) $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  ${CXX} ${CXXFLAGS} -DFREEMOBIUS5D_BLOCKING_FFT -I${SRC} -c ${TEST} -o ${OBJ}
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

  echo "======== run 8^4 --ops m0 (gpu0): gate 1 + M0 breakdown ========"
  ${BIN} --grid ${GRID} --mpi 1.1.1.1 --config ${CFG} --ops m0
  echo "exit = $?"
} 2>&1 | tee ${LOG}
