#!/usr/bin/env bash
# OPERATOR SPECTRUM SCAN (user 2026-09-05): extract the LOWEST (smallest modulus) and HIGHEST (largest
# modulus) eigenpair of D_W, H_W = gamma5 D_W, M0 D_W, and M1 D_W, and on each extreme eigenvector psi
# evaluate the preconditioner quality C = ||(1 - M D_W) psi||/||psi|| (= |1 - mu| when psi is an eigenmode of
# M D_W) for BOTH M0 and M1. C>=1 (equivalently Re mu<0 for the M D_W eigenvalues) = RED FLAG (the prec
# amplifies/flips that mode). Also finds the D_W low modes via SHIFT-INVERT IRA. Smooth Q=-3
# (ckpoint_lat.240 op-reflow 4t0), m=0.1 mprec=0.3. Driver Test_wilson_frameopt_claude.cc. Single QUIET gpu0.
# No rm / no kill.
set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/wilson_opscan_claude.log
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
  echo "-------- operator spectrum scan, smooth Q=-3, m=${MASS} mprec=${MPREC} --------"
  ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1 --config ${RAW} --op-reflow-eps 0.05 --op-reflow-nstep 232 \
      --flow-nstep 0 --spectrum --shiftinvert --si-tol ${SITOL} --mass ${MASS} --mprec ${MPREC} \
      --spec-nstop ${NSTOP} --spec-nk ${NK} --spec-nm ${NM} --opscan

  echo ""
  echo "======== SUMMARY ========"
  grep -nE "OPSCAN|RED" ${LOG} | grep -vE "warning|::write|\.h\(:"
  echo "======== done ========"
} 2>&1 | tee ${LOG}
