#!/bin/bash
# RB-CGNE-DEFAULT benchmark + MASS SCAN. RB-CGNE (SchurRedBlackDiagMooee) is now the HONEST default
# baseline (full CGNE kept but labeled soft). The FGMRES(M0) block prints, per mass:
#   D_W-apply speedup (RB-CGNE / FGMRES-M0)  [count, M0-free -- misleading]
#   WALL speedup     (RB-CGNE / FGMRES-M0)  [HONEST -- >1 = free-prec wins wall]
# Runs 8^4 --ops cgne,m0 at m=0.1 (reference) and m=0.01 (small-m target -- tests the win-condition
# crossover: does RB-CGNE's iter count grow ~kappa~1/m while FGMRES(M0) stays flatter?). Baselines are
# now non-erroring + solve_maxit=20000 so a slow small-m CGNE caps instead of aborting. Single GPU.
# No rm / no kill. GPU-congestion-safe for correctness, but the WALL numbers need a QUIET GPU to trust --
# run when the GPU is free (the D_W COUNTS are load-independent; only wall speedups need isolation).

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/rbcgne_baseline_claude.log
CFG=${ROOT}/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc
MASSES="0.1 0.01"

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

{
  echo "======== compile Test_dwf_freeprec_claude.cc (RB-CGNE default + --mass) $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"

  OBJ=${BUILD}/Test_dwf_freeprec_rbcgne_claude.o
  BIN=${BUILD}/Test_dwf_freeprec_rbcgne_claude
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

  for M in ${MASSES}
  do
    echo "======== run: 8^4 m=${M} --ops cgne,m0 (RB-CGNE default + full CGNE + FGMRES(M0)) ========"
    ${BIN} --grid 8.8.8.8 --mpi 1.1.1.1 --config ${CFG} --ops cgne,m0 --mass ${M}
  done

  echo "======== SUMMARY (per mass) ========"
  grep -nE "headline mass m =|CGNE\(full\):|RB-CGNE:|FGMRES\(M0\) restart|speedup \(RB-CGNE|WALL speedup" ${LOG}
  echo "======== done ========"
  echo "READ: at each mass compare RB-CGNE iters (should GROW ~1/m) vs FGMRES(M0) iters (flatter?),"
  echo "      and the WALL speedup (RB-CGNE/FGMRES-M0): >1 = free-prec wins wall. m=0.01 is the target."
} 2>&1 | tee ${LOG}
