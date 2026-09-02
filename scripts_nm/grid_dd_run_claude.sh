#!/bin/bash
# G0a of the domain-decomposed (additive-Schwarz) free-limit Mobius DWF preconditioner, Grid port.
# Design: qed2/dwf4_qcd_claude/grid_dd_freeprec_impl_plan_claude.md. Builds + runs
# tests/solver/Test_dwf_freeprec_dd_claude.cc against the CPU/MPI Grid build (build_mpi/, from
# grid_mpi_build_claude.sh). SINGLE RANK (--mpi 1.1.1.1): G0 block-local gather uses the full per-rank
# lex array. Reuses FreeMobius5DInverse on an ext^4 block grid (BlockFreeMobius5D_claude.h, additive --
# does NOT modify FreeMobius5D_claude.h).
#
# What to look for in the log:
#   [validation] core=L,halo=0 iters == exact-F iters : PASS   (the block machinery is correct)
#   then the block (core=4/2, halo=0/1/2) FGMRES counts vs the exact-F baseline -- should reproduce the
#   dwf4 8^4 pattern (block-local stalls without a near-full halo; RAS halo recovers it). 16^4 is the
#   real gate but needs a 16^4 config (not yet generated).
# User runs this; Claude reads the log. No rm, no kill here.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build_mpi
GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_dwf_freeprec_dd_claude.cc
OBJ=${BUILD}/Test_dwf_freeprec_dd_claude.o
BIN=${BUILD}/Test_dwf_freeprec_dd_claude
LOG=${ROOT}/grid_dd_run_claude.log
CFG=${ROOT}/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc

export OMP_NUM_THREADS=8

{
  echo "======== [0/3] build_mpi install-tree guard $(date) ========"
  if [ ! -f ${BUILD}/lib/libGrid.a ] || [ ! -f ${BUILD}/include/Grid/Grid.h ]; then
    echo "  build_mpi tree missing -> run scripts_nm/grid_mpi_build_claude.sh first; stopping."
    exit 1
  fi
  if [ ! -f ${CFG} ]; then
    echo "  config ${CFG} missing; stopping."
    exit 1
  fi
  echo "  present"

  echo "======== [1/3] COMPILE + LINK G0 DD test (vs build_mpi grid-config) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  echo "CXX = ${CXX}"
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
  echo "  built ${BIN}"

  echo "======== [2/3] G0 DD scan 8^4  m=0.1  (single rank; validation + block pattern) ========"
  ${BIN} --grid 8.8.8.8 --mpi 1.1.1.1 --config ${CFG} --mass 0.1
  rc=$?
  echo "  (m=0.1 exit ${rc})"

  echo "======== [3/3] G0 DD scan 8^4  m=0.02  (light-mass point) ========"
  ${BIN} --grid 8.8.8.8 --mpi 1.1.1.1 --config ${CFG} --mass 0.02
  rc=$?
  echo "  (m=0.02 exit ${rc})"

  echo "======== G0a DD run complete ========"
} 2>&1 | tee ${LOG}
