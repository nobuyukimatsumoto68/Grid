#!/bin/bash
# G0b of the domain-decomposed (additive-Schwarz) free-limit Mobius DWF preconditioner, Grid port:
# the 16^4 quality GATE (the decisive DD-viability test; 8^4 has no locality headroom). Builds + runs
# tests/solver/Test_dwf_freeprec_dd_claude.cc SINGLE RANK on the shared 16^4 Iwasaki beta2.6 config
# (../iwasaki16_b2.6_Qspread_claude/). Design: qed2/dwf4_qcd_claude/grid_dd_freeprec_impl_plan_claude.md.
#
# --min-core 4: at 16^4 the genuinely-LOCAL blocks are core=8 (2/dim) and core=4 (4/dim); core=2 blocks
# (8/dim, 4096 blocks, below the correlation length) are slow AND guaranteed bad, so skip them. Read:
#   [validation] core=L,halo=0 iters == exact-F iters : PASS   (port correct at 16^4)
#   then block(core=8, halo 0/1/2) vs exact -- DOES a genuinely-local block (ext 8-12 < 16) hold the win?
# NB config is IWASAKI beta2.6 (Q!=0), not the 8^4 Wilson beta6 point -- win MAGNITUDE differs, but the
# DD relative degradation (block vs exact) is the measurement; also probes R2 (Q-dependence).
# MEMORY: at 16^4 Ls8 each FGMRES Krylov vector is ~96 MB, so a no-restart (256) basis is ~49 GB/solve
# -> OOM on a 62 GB box when two coexist. Mitigated: the exact-F solve is now scoped (freed before the
# scan) and --restart 128 bounds each basis to ~25 GB. The block-vs-exact comparison stays fair (same
# restart). This memory pressure is itself a single-rank artifact -- G1 (MPI node=block) distributes it.
# User runs this (heavier than 8^4); Claude reads the log. No rm, no kill here.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build_mpi
GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_dwf_freeprec_dd_claude.cc
OBJ=${BUILD}/Test_dwf_freeprec_dd_claude.o
BIN=${BUILD}/Test_dwf_freeprec_dd_claude
LOG=${ROOT}/grid_dd_16_run_claude.log
CFG=${ROOT}/iwasaki16_b2.6_Qspread_claude/ckpoint_lat.160   # Q = -6 (5Li)

export OMP_NUM_THREADS=8

{
  echo "======== [0/2] guards $(date) ========"
  if [ ! -f ${BUILD}/lib/libGrid.a ] || [ ! -f ${BUILD}/include/Grid/Grid.h ]; then
    echo "  build_mpi tree missing -> run scripts_nm/grid_mpi_build_claude.sh first; stopping."
    exit 1
  fi
  if [ ! -f ${CFG} ]; then
    echo "  config ${CFG} missing; stopping."
    exit 1
  fi
  echo "  present"

  echo "======== [1/2] COMPILE + LINK (vs build_mpi grid-config) ========"
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

  echo "======== [2/2] G0b 16^4 DD gate  ckpoint_lat.160 (Q=-6)  m=0.1  min-core 4 (single rank) ========"
  ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1 --config ${CFG} --mass 0.1 --min-core 8 --restart 128
  rc=$?
  echo "  (16^4 m=0.1 exit ${rc})"

  echo "======== G0b 16^4 run complete ========"
} 2>&1 | tee ${LOG}
