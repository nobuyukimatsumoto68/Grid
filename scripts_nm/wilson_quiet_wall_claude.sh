#!/usr/bin/env bash
# QUIET-GPU wall comparison, RB-CGNE vs free-prec FGMRES(M0_W), with the RESTART param swept to find the
# optimal wall (user 2026-09-04, run on a cleared gpu0). The congested runs gave ~0.30-0.33x (free-prec 3x
# slower) at restart=256 (no-restart); restart is the wall lever (DW 8^4: no-restart 0.52x -> restart-20
# 0.91x). This finds the best restart on a QUIET GPU with min-of-N timing (drops GPU warmup).
#
# One config: raw ckpoint_lat.240 CONVENTIONALLY flowed to 4 t0 (--op-reflow-nstep 232) = genuine SMOOTH
# Q=-3 (the reflow study showed Q is really -3). m_bare=0.1, m_prec=0.3 (Tikhonov optimum from the Wilson
# memo). Sweep restart {256,50,30,20}; --repeat 3 = min wall. Frame (flow+Landau) is rebuilt per invocation
# (same each time) -- a few min overhead, fine for a one-off. Test = Test_wilson_freeprec_defect_scan_claude.cc
# (branched copy, now with --repeat). Single QUIET gpu0. No rm / no kill.
set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/wilson_quiet_wall_claude.log
CFGDIR=${ROOT}/iwasaki16_b2.6_Qspread_claude
RAW=${CFGDIR}/ckpoint_lat.240

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0   # the cleared/quiet GPU

MASS=0.1
MPREC=0.3
REPEAT=3
RESTARTS="256 50 30 20"

{
  echo "======== compile Test_wilson_freeprec_defect_scan_claude.cc $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  OBJ=${BUILD}/Test_wilson_freeprec_defect_scan_claude.o
  BIN=${BUILD}/Test_wilson_freeprec_defect_scan_claude
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${SRC}/tests/solver/Test_wilson_freeprec_defect_scan_claude.cc -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "LINK FAILED (rc=${rc})"; exit ${rc}; fi

  for R in ${RESTARTS}
  do
    echo ""
    echo "############## smooth Q=-3 (op-reflow 4t0)  m=${MASS} mprec=${MPREC} restart=${R} repeat=${REPEAT} ##############"
    ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1 --mass ${MASS} --mprec ${MPREC} --restart ${R} --repeat ${REPEAT} \
        --config ${RAW} --op-reflow-eps 0.05 --op-reflow-nstep 232 --flow-nstep 0
  done

  echo ""
  echo "======== SUMMARY (min wall per restart; RB is restart-independent) ========"
  grep -nE "restart=|RB-CGNE\(Wilson\)|FGMRES\(M0_W\)|WALL speedup \(RB-CGNE" ${LOG} | grep -vE "warning|::write|\.h\(:"
  echo "======== done ========"
} 2>&1 | tee ${LOG}
