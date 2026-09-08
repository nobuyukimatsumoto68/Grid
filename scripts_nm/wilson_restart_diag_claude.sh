#!/bin/bash
# Wilson small-m DIAGNOSTIC: is FGMRES(M0_W)'s explosion at m=-0.7 (3578 iters @restart-80) RESTART
# GRINDING or a GENUINE bad preconditioned spectrum (low modes)? Wilson FITS a big restart (400*2*12MiB
# ~10 GiB @16^4), which is the whole point of the pivot -- so sweep restart 80,200,400 at m=-0.7 (one
# invocation, --mass-list "-0.7", Omega built once). If N_it DROPS to hundreds at restart 400 -> was
# grinding (use a big restart). If it STAYS in the thousands -> genuine low-mode failure: the free-prec
# alone can't precondition small-m Wilson, needs deflation (Dir-2). Single GPU. Wall needs quiet GPU.
# NB: the test takes ONE --restart, so this runs 3 invocations (re-flows Omega each -- acceptable for a
# diagnostic). No rm / no kill.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/wilson_restart_diag_claude.log
CFG16=${ROOT}/dwf4_qcd_claude/cfg_su3_16161616_b6.0_claude.nersc
RESTARTS="80 200 400"

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

{
  echo "======== compile Test_wilson_freeprec_claude.cc $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  OBJ=${BUILD}/Test_wilson_freeprec_rd_claude.o
  BIN=${BUILD}/Test_wilson_freeprec_rd_claude
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${SRC}/tests/solver/Test_wilson_freeprec_claude.cc -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "LINK FAILED (rc=${rc})"; exit ${rc}; fi

  for R in ${RESTARTS}
  do
    echo "======== run: 16^4 Wilson m=-0.7 restart=${R} ========"
    ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1 --config ${CFG16} --mass-list "-0.7" --restart ${R}
  done

  echo "======== SUMMARY ========"
  grep -nE "residual proxy|RB-CGNE\(Wilson\)|FGMRES\(M0_W\) restart|WALL speedup" ${LOG} | grep -vE "warning|::write|strlen|\.h\(:"
  echo "======== done ========"
  echo "READ: N_it(restart 80 -> 200 -> 400). DROPS to hundreds = grinding (use big restart). STAYS in"
  echo "      thousands = genuine low-mode failure -> free-prec alone insufficient for small-m Wilson."
} 2>&1 | tee ${LOG}
