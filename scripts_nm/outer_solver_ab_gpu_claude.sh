#!/bin/bash
# Outer-solver A/B (targets the ~0.53 s no-restart FGMRES orthogonalization, ~half the solve wall).
# Builds the default (cached FFT + on-device solve + phase), runs on gpu0 8^4 --config:
#   (1) --ops m0,bcg --restart 256 : FGMRES(no-restart) WALL + BiCGSTAB(M0) WALL, back-to-back (same frame)
#   (2..4) --ops m0 --restart {50,30,20} : FGMRES restart sweep WALL
# Compare the "WALL=" lines. BiCGSTAB: ~2 M0/iter, no orthogonalization -> wins iff fewer iters.
# NB each run re-determines Omega (~8 s, one-time); only the SOLVE WALL is compared.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_dwf_freeprec_claude.cc
OBJ=${BUILD}/Test_dwf_freeprec_solver_claude.o
BIN=${BUILD}/Test_dwf_freeprec_solver_claude
LOG=${ROOT}/outer_solver_ab_gpu_claude.log
GRID=8.8.8.8
CFG=/mnt/baracuda_14/dwms/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

{
  echo "======== compile (default: cached FFT + on-device solve + phase) $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${TEST} -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "LINK FAILED (rc=${rc})"; exit ${rc}; fi

  echo "======== (1) FGMRES(no-restart) + BiCGSTAB, same frame ========"
  ${BIN} --grid ${GRID} --mpi 1.1.1.1 --config ${CFG} --ops m0,bcg --restart 256
  for R in 50 30 20; do
    echo "======== FGMRES restart=${R} ========"
    ${BIN} --grid ${GRID} --mpi 1.1.1.1 --config ${CFG} --ops m0 --restart ${R}
  done
  echo "done"
} 2>&1 | tee ${LOG}
