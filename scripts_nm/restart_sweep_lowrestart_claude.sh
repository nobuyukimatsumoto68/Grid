#!/bin/bash
# LOW-RESTART SWEEP: push the FGMRES(M0) restart length below 20 to find the wall optimum vs RB-CGNE.
# The shootout showed restart-20 (0.409s) STILL descending; the O(iters^2) orthogonalization keeps
# shrinking with restart, but too-small restart eventually blows up the iteration count -> there is an
# optimum. Sweep 20,15,10,8 back-to-back (same frame + GPU state -> clean A/B), m=0.1, 8^4.
#   --ops cgne,m0   : RB-CGNE reference + FGMRES(M0); BiCGSTAB dropped (it lost the shootout).
# Watch: WALL bottoms then rises as iters blow up; the min-WALL restart = the outer-solver default.
#
# *** PURE WALL-CLOCK TEST -- RUN ONLY ON A QUIET GPU (counts are load-independent, wall is not). ***
# Single GPU. No rm / no kill.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/restart_sweep_lowrestart_claude.log
CFG=${ROOT}/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

{
  echo "======== compile Test_dwf_freeprec_claude.cc $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"

  OBJ=${BUILD}/Test_dwf_freeprec_lowrestart_claude.o
  BIN=${BUILD}/Test_dwf_freeprec_lowrestart_claude
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${SRC}/tests/solver/Test_dwf_freeprec_claude.cc -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]
  then
    echo "COMPILE FAILED (rc=${rc})"
    exit ${rc}
  fi
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]
  then
    echo "LINK FAILED (rc=${rc})"
    exit ${rc}
  fi

  echo "======== run: 8^4 m=0.1 --ops cgne,m0 --restart-sweep 20,15,10,8 ========"
  ${BIN} --grid 8.8.8.8 --mpi 1.1.1.1 --config ${CFG} --ops cgne,m0 --restart-sweep 20,15,10,8

  echo "======== SUMMARY ========"
  grep -nE "RB-CGNE:|FGMRES\(M0\) restart|WALL speedup \(RB-CGNE|M0/Dhop" ${LOG}
  echo "======== done ========"
  echo "READ: WALL bottoms at some restart then rises (iters blow up). min-WALL restart = outer default;"
  echo "      WALL speedup (RB-CGNE/FGMRES) >1 = free-prec WINS wall vs RB-CGNE at 8^4 m=0.1."
} 2>&1 | tee ${LOG}
