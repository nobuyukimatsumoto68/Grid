#!/usr/bin/env bash
# SOLVER COUNTS incl. DEFLATION (user 2026-09-05): actually invert D_W x = b and count outer FGMRES
# iterations for RB-CGNE (baseline), FGMRES(M0), FGMRES(M1), FGMRES(M0+deflation), FGMRES(M1+deflation),
# where deflation = exact solve on span{the 3 chiral zero modes the IRA found} (DeflatedPrec, additive
# two-level). Does deflating the topological blocker sector cut the outer count -- and does M1 help on top?
# Honest note printed inline: M1 adds 1 D_W[U^L]/apply, deflation adds 1 D_W/apply. Modes via SHIFT-INVERT
# IRA. Smooth Q=-3 (ckpoint_lat.240 op-reflow 4t0), m=0.1 mprec=0.3, Nstop=16, FGMRES restart=20.
# Driver Test_wilson_frameopt_claude.cc. Single QUIET gpu0. No rm / no kill.
set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/wilson_solve_deflate_claude.log
CFGDIR=${ROOT}/iwasaki16_b2.6_Qspread_claude
RAW=${CFGDIR}/ckpoint_lat.240

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

MASS=0.10
MPREC=0.3
NSTOP=16
NK=32
NM=64
SITOL=1.0e-8
SRST=20
MXACUT=0.7

{
  echo "======== compile Test_wilson_frameopt_claude.cc $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  OBJ=${BUILD}/Test_wilson_frameopt_claude.o
  BIN=${BUILD}/Test_wilson_frameopt_claude
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${SRC}/tests/solver/Test_wilson_frameopt_claude.cc -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "LINK FAILED (rc=${rc})"; exit ${rc}; fi

  echo ""
  echo "-------- shift-invert IRA + solve counts (M0/M1 + deflation), smooth Q=-3, m=${MASS} mprec=${MPREC} --------"
  ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1 --config ${RAW} --op-reflow-eps 0.05 --op-reflow-nstep 232 \
      --flow-nstep 0 --spectrum --shiftinvert --si-tol ${SITOL} --mass ${MASS} --mprec ${MPREC} \
      --spec-nstop ${NSTOP} --spec-nk ${NK} --spec-nm ${NM} \
      --solve-deflate --solve-restart ${SRST} --mx-acut ${MXACUT}

  echo ""
  echo "======== SUMMARY ========"
  grep -nE "SOLVE-DEFLATE|RB-CGNE|FGMRES\(M|outer iteration|shift-invert|Nconv" ${LOG} | grep -vE "warning|::write|\.h\(:"
  echo "======== done ========"
} 2>&1 | tee ${LOG}
