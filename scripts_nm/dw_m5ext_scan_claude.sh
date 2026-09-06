#!/bin/bash
# EXTENDED M5-shift scan for the DW free-prec. The 2D scan found lowering the preconditioner's Wilson-kernel
# M5 (below the operator's 1.8) is the DOMINANT lever (FGMRES 79 matched -> 49 at M5_prec=1.3, m_prec=0.4),
# and it was STILL DROPPING at the edge M5_prec=1.3. Push lower to find the bottom. Operator FIXED at
# (M5=1.8, m=0.1); scan M5_prec (absolute --m5prec) x m_prec (absolute --mprec, near the 0.3-0.5 optimum).
# --ops m0, no-restart (8^4 fits, clean N_it); RB=181 fixed. Watch: does FGMRES keep dropping below 49, or
# bottom out / blow up as M5_prec -> small (free DW block near-singular at low M5)?
# Single GPU; wall not the point here (iteration count). No rm / no kill.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/dw_m5ext_scan_claude.log
CFG8=${ROOT}/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc

M5PRECS="1.5 1.3 1.1 0.9 0.7 0.5 0.3"
MPRECS="0.3 0.4 0.5"

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

{
  echo "======== compile Test_dwf_freeprec_claude.cc $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  OBJ=${BUILD}/Test_dwf_freeprec_m5ext_claude.o
  BIN=${BUILD}/Test_dwf_freeprec_m5ext_claude
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${SRC}/tests/solver/Test_dwf_freeprec_claude.cc -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "LINK FAILED (rc=${rc})"; exit ${rc}; fi

  for M5P in ${M5PRECS}
  do
    for MP in ${MPRECS}
    do
      echo "POINT M5prec=${M5P} mprec=${MP}"
      ${BIN} --grid 8.8.8.8 --mpi 1.1.1.1 --config ${CFG8} --ops m0 --m5prec ${M5P} --mprec ${MP} --restart 256
    done
  done

  echo "======== SUMMARY (POINT + FGMRES iters) ========"
  grep -nE "^POINT |FGMRES\(M0\) restart" ${LOG} | grep -vE "warning|::write|strlen|\.h\(:"
  echo "======== done ========"
  echo "READ: min FGMRES over (M5prec, mprec); the 2D scan bottomed at 49 (M5prec=1.3,mprec=0.4)."
  echo "      does it drop further below M5prec=1.3, or bottom/blow up as M5prec->small?"
} 2>&1 | tee ${LOG}
