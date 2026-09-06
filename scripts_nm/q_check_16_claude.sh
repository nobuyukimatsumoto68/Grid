#!/bin/bash
# TOPOLOGY CHECK of the new 16^4 config -- CHEAP (flow + Landau + Q only, NO solves). run_headline in
# Test_dwf_freeprec_claude.cc measures the topological charge in its HEADER before any solve, so --ops none
# (no cgne/m0/m1) runs just: Wilson flow (eps 0.02 x 100 -> tau=2) + Landau fix + Q(orig) clover +
# Q_5Li(frame-flowed). The per-step [WilsonFlow] Top. charge trajectory shows whether Q settles to a
# nonzero INTEGER (nontrivial topology) or ~0 (trivial, like the 8^4 which was Q_5Li~0.001).
# Q is a PHYSICS quantity (load-independent) -> a busy GPU is fine. Single GPU. No rm / no kill.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/q_check_16_claude.log
CFG16=${ROOT}/dwf4_qcd_claude/cfg_su3_16161616_b6.0_claude.nersc

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

{
  if [ ! -f "${CFG16}" ]
  then
    echo "CONFIG NOT FOUND: ${CFG16}"
    exit 2
  fi

  echo "======== compile Test_dwf_freeprec_claude.cc $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"

  OBJ=${BUILD}/Test_dwf_freeprec_qcheck_claude.o
  BIN=${BUILD}/Test_dwf_freeprec_qcheck_claude
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

  echo "======== run: 16^4 --config --ops none (flow + Landau + Q only, NO solves) --log Message ========"
  ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1 --config ${CFG16} --ops none

  echo "======== SUMMARY ========"
  echo "-- Q trajectory (Wilson flow, per step) [want it to settle to an integer]:"
  grep -nE "\[WilsonFlow\] Top. charge" ${LOG} | grep -vE "warning" | tail -20
  echo "-- header Q line (Q(orig) clover + Q_5Li frame-flowed tau=2):"
  grep -nE "Q\(orig\)=|Q_5Li|flowed-fixed Landau" ${LOG} | grep -vE "warning" | tail
  echo "======== done ========"
  echo "READ: if Q_5Li ~ integer != 0 -> NONTRIVIAL topology (the interesting case). ~0 -> trivial."
} 2>&1 | tee ${LOG}
