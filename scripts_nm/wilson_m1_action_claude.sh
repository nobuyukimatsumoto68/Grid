#!/usr/bin/env bash
# M1-action test (user 2026-09-05): does the next-order (hopping-expansion) correction M1 rescue the chiral
# sector that M0 fails on? Find the low D_W modes with SHIFT-INVERT IRA (CGNE D_W^{-1}, sigma=0, the robust
# near-zero eigensolver = diagnosis fix 3), SAVE them (LIME + meta), then apply BOTH M0 and M1 to each and
# report |M0 DW phi|/|phi| and |M1 DW phi|/|phi| (=1 for a perfect prec). Smooth Q=-3 (ckpoint_lat.240
# op-reflow 4t0), m=0.1 mprec=0.3, Nstop=16. Driver Test_wilson_frameopt_claude.cc. Single QUIET gpu0.
# No rm / no kill. (Modes -> dw_modes_claude/; if a stale .lime blocks a rerun, remove it yourself.)
set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/wilson_m1_action_claude.log
CFGDIR=${ROOT}/iwasaki16_b2.6_Qspread_claude
RAW=${CFGDIR}/ckpoint_lat.240
MODEDIR=${ROOT}/dw_modes_claude

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

MASS=0.10
MPREC=0.3
NSTOP=16
NK=32
NM=64
SITOL=1.0e-8

mkdir -p ${MODEDIR}

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
  echo "-------- shift-invert IRA + M0/M1 action, smooth Q=-3, m=${MASS} mprec=${MPREC} --------"
  ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1 --config ${RAW} --op-reflow-eps 0.05 --op-reflow-nstep 232 \
      --flow-nstep 0 --spectrum --shiftinvert --si-tol ${SITOL} --mass ${MASS} --mprec ${MPREC} \
      --spec-nstop ${NSTOP} --spec-nk ${NK} --spec-nm ${NM} \
      --save-modes ${MODEDIR}/dw_modes_q3_m${MASS}

  echo ""
  echo "======== SUMMARY ========"
  grep -nE "SHIFT-INVERT|shift-invert|D_W RR mode\[|M0act\[|M1act\[|SAVED|Nconv" ${LOG} | grep -vE "warning|::write|\.h\(:"
  echo "======== done ========"
} 2>&1 | tee ${LOG}
