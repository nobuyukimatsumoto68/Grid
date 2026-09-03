#!/bin/bash
# Chunk A validation: pack-once fused F (one pack + fused contiguous cuFFT + solve-in-buffer + one unpack)
# vs the current default (barrel-shift per-dim). Builds TWO binaries -- DEFAULT and -DFREEMOBIUS5D_PACKONCE
# -- and runs each 8^4 --config --ops m0 back-to-back (same GPU load). Check:
#   (1) cold gate 0a/0b PASS under PACKONCE (correctness);
#   (2) FGMRES M0 iteration count IDENTICAL default vs packonce (same math -> only faster);
#   (3) [F timers] FFT-part (fft_fwd+fft_bwd) drops (target ~4-8x); total M0 drops.
# grid_packonce_fft_impl_plan_claude.md Chunk A.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_dwf_freeprec_claude.cc
LOG=${ROOT}/packonce_validate_gpu_claude.log
GRID=8.8.8.8
CFG=/mnt/baracuda_14/dwms/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

{
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"

  echo "======== compile DEFAULT $(date) ========"
  OBJ_D=${BUILD}/Test_dwf_freeprec_default_claude.o
  BIN_D=${BUILD}/Test_dwf_freeprec_default_claude
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${TEST} -o ${OBJ_D}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "DEFAULT COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${OBJ_D} -o ${BIN_D} ${LDFLAGS} ${LIBS}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "DEFAULT LINK FAILED (rc=${rc})"; exit ${rc}; fi

  echo "======== compile PACKONCE (-DFREEMOBIUS5D_PACKONCE) ========"
  OBJ_P=${BUILD}/Test_dwf_freeprec_packonce_claude.o
  BIN_P=${BUILD}/Test_dwf_freeprec_packonce_claude
  ${CXX} ${CXXFLAGS} -DFREEMOBIUS5D_PACKONCE -I${SRC} -c ${TEST} -o ${OBJ_P}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "PACKONCE COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${OBJ_P} -o ${BIN_P} ${LDFLAGS} ${LIBS}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "PACKONCE LINK FAILED (rc=${rc})"; exit ${rc}; fi

  echo "======== run DEFAULT 8^4 --config --ops m0 ========"
  ${BIN_D} --grid ${GRID} --mpi 1.1.1.1 --config ${CFG} --ops m0

  echo "======== run PACKONCE 8^4 --config --ops m0 ========"
  ${BIN_P} --grid ${GRID} --mpi 1.1.1.1 --config ${CFG} --ops m0
  echo "done"
} 2>&1 | tee ${LOG}
