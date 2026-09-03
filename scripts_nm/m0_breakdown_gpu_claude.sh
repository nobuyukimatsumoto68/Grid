#!/bin/bash
# GPU M0/Dhop breakdown: same run as the CPU m0_breakdown, but on the CUDA build/ (cuFFT), pinned to
# gpu0. Recompiles Test_dwf_freeprec_claude.cc (M0.report_timers() + Dhop loop in run_m0) against
# build/grid-config, then runs single-rank 8^4 on the SU(3) config with --ops m0. Watch: "[M0 timers]",
# "[F timers]", "[Dhop] us/apply", "[M0/Dhop] ... Dhop-equivalents". Log -> m0_breakdown_gpu_claude.log.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_dwf_freeprec_claude.cc
OBJ=${BUILD}/Test_dwf_freeprec_claude.o
BIN=${BUILD}/Test_dwf_freeprec_claude
LOG=${ROOT}/m0_breakdown_gpu_claude.log
GRID=8.8.8.8
CFG=/mnt/baracuda_14/dwms/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

{
  echo "======== [0/2] CUDA install-tree guard $(date) ========"
  if [ ! -f ${BUILD}/lib/libGrid.a ] || [ ! -f ${BUILD}/include/Grid/Grid.h ]; then
    echo "  CUDA build tree missing -> run scripts_nm/grid_freeprec_build_claude.sh first; stopping."
    exit 1
  fi
  echo "  present"

  echo "======== [1/2] COMPILE + LINK (nvcc; CXXLD swaps -x cu -> -link) ========"
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

  echo "======== [2/2] RUN on gpu0 (CUDA_VISIBLE_DEVICES=0), single-rank 8^4 --ops m0 ========"
  ${BIN} --grid ${GRID} --mpi 1.1.1.1 --config ${CFG} --ops m0
  echo "exit = $?"
} 2>&1 | tee ${LOG}
