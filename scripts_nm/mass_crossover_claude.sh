#!/bin/bash
# LIGHT-MASS CROSSOVER: m = 0.1, 0.01, 0.001 at 8^4, restart-20 (wall-optimal), --ops cgne,m0.
# Tests the win-condition directly: does RB-CGNE's iter count keep GROWING ~1/m (kappa penalty of CG on
# the normal eqns) while FGMRES(M0)'s N_it stays FLAT (frame-limited)? m=0.001 is the deep-chiral probe.
# The COUNT metric (RB iters, FGMRES iters) is LOAD-INDEPENDENT -> the crossover signal is clean even on a
# busy GPU; the WALL numbers need a quiet GPU (only the wall-speedup line is load-sensitive).
# Baselines are non-erroring + solve_maxit=20000, so a slow deep-chiral CGNE caps, never aborts.
# Single GPU. No rm / no kill.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/mass_crossover_claude.log
CFG=${ROOT}/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc
MASSES="0.1 0.01 0.001"

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

{
  echo "======== compile Test_dwf_freeprec_claude.cc (default: PlannedFFT, fp32) $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"

  OBJ=${BUILD}/Test_dwf_freeprec_masscross_claude.o
  BIN=${BUILD}/Test_dwf_freeprec_masscross_claude
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
    echo "======== run: 8^4 m=${M} --ops cgne,m0 --restart 20 ========"
    ${BIN} --grid 8.8.8.8 --mpi 1.1.1.1 --config ${CFG} --ops cgne,m0 --restart 20 --mass ${M}
  done

  echo "======== SUMMARY (per mass) ========"
  grep -nE "headline mass m =|RB-CGNE:|FGMRES\(M0\) restart|WALL speedup \(RB-CGNE" ${LOG} | grep -vE "warning|::write|strlen|\.h\("
  echo "======== done ========"
  echo "READ (crossover): RB-CGNE iters should GROW toward small m (~1/m, kappa penalty); FGMRES(M0) iters"
  echo "      should stay FLAT (frame-limited). count-win = RB_iters*2 / FGMRES_iters widens toward small m."
  echo "      m=0.001 = deep-chiral probe; watch whether 8^4 volume-floor still masks the growth."
} 2>&1 | tee ${LOG}
