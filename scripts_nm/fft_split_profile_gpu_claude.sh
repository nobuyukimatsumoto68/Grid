#!/bin/bash
# FFT-internal split profile (single-GPU): decides lever 1 (fp32, if cuFFT-KERNEL-dominated) vs lever 2
# (fuse passes, if PACK/UNPACK-dominated) for the local M0 ~26 D_W. Builds Test_dwf_freeprec with
# -DFFT_CLAUDE_PROFILE (device-synced pack/fftk/unpack timers inside FFT_claude::FFT_dim), runs gpu0 8^4
# --config --ops m0. Read the "[FFT_dim split]" block (pack% / fftk% / unpack%) at the end of the log.
# NB the barriers perturb absolute timing -- trust the RATIOS, not the absolute us (compare vs the quiet
# no-profile numbers in grid_freeprec_cost_benchmark_v2_claude.md Section 3).

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_dwf_freeprec_claude.cc
OBJ=${BUILD}/Test_dwf_freeprec_fftsplit_claude.o
BIN=${BUILD}/Test_dwf_freeprec_fftsplit_claude
LOG=${ROOT}/fft_split_profile_gpu_claude.log
GRID=8.8.8.8
CFG=/mnt/baracuda_14/dwms/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

{
  echo "======== compile (-DFFT_CLAUDE_PROFILE) $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  ${CXX} ${CXXFLAGS} -DFFT_CLAUDE_PROFILE -I${SRC} -c ${TEST} -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "LINK FAILED (rc=${rc})"; exit ${rc}; fi

  echo "======== run 8^4 --config --ops m0 (FFT split) ========"
  ${BIN} --grid ${GRID} --mpi 1.1.1.1 --config ${CFG} --ops m0
  echo "done"
} 2>&1 | tee ${LOG}
