#!/bin/bash
# OUTER-SOLVER SHOOTOUT: no-restart FGMRES vs restarted GMRES vs BiCGSTAB, same M0, m=0.1, 8^4.
# All three already exist in Test_dwf_freeprec_claude.cc -- NO code change:
#   --ops cgne,m0,bcg   : RB-CGNE baseline + FGMRES(M0) + BiCGSTAB(M0)
#   --restart-sweep 256,50,30,20 : FGMRES solved BACK-TO-BACK at each restart length (same frame + GPU
#       state -> removes cross-run drift). 256 = no-restart (79 iters < 256), 50/30/20 cap the O(iters^2)
#       Arnoldi orthogonalization (the 0.51s Linalg that dominated the no-restart wall).
# WHAT WE MEASURE (per solver): iters, M0 applies, WALL, and (FGMRES) WALL speedup vs RB-CGNE.
#   - Restarted GMRES: fewer/cheaper orthogonalization per iter, but more iters -> find the wall optimum.
#   - BiCGSTAB: NO orthogonalization (short recurrence, O(1) linalg) but ~2 M0 applies/iter -> wins only
#     if it converges in few enough iters. Non-monotonic; err_on_no_conv=false so it caps, never aborts.
#
# *** THIS IS A PURE WALL-CLOCK TEST -- RUN ONLY ON A QUIET GPU. *** A congested GPU makes the wall
# numbers meaningless (the iters/D_W counts would still be valid, but the whole point here is wall).
# Single GPU. No rm / no kill.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/outer_solver_shootout_claude.log
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

  OBJ=${BUILD}/Test_dwf_freeprec_shootout_claude.o
  BIN=${BUILD}/Test_dwf_freeprec_shootout_claude
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

  echo "======== run: 8^4 m=0.1 --ops cgne,m0,bcg --restart-sweep 256,50,30,20 ========"
  ${BIN} --grid 8.8.8.8 --mpi 1.1.1.1 --config ${CFG} --ops cgne,m0,bcg --restart-sweep 256,50,30,20

  echo "======== SUMMARY ========"
  grep -nE "RB-CGNE:|FGMRES\(M0\) restart|WALL speedup \(RB-CGNE|BiCGSTAB\(M0\):|M0/Dhop" ${LOG}
  echo "======== done ========"
  echo "READ: lowest WALL among {FGMRES no-restart(256), FGMRES restart 50/30/20, BiCGSTAB} = best outer"
  echo "      solver; compare each to RB-CGNE WALL. Watch BiCGSTAB true res (non-monotonic can stall)."
} 2>&1 | tee ${LOG}
