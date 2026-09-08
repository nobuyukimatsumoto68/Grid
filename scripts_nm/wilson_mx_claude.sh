#!/usr/bin/env bash
# SCALE-MIXED Mx (user 2026-09-05): the M1 hopping correction band-limited to the LOW free-eval band
# A(p) = m_prec + sum(1-cos p_mu) < Acut (IR), M0 in the UV. Two variants: INNER (P before D(tildeA), exact,
# +1 D_free apply) and OUTER (P after, cheap, leaks in UV). Compare the quality C=||(1-M D_W)phi||/||phi|| for
# M0 / M1 / Mx_in / Mx_out on the D_W eigenmodes, and the extreme-mode spectra (does Mx remove M1's UV
# red-flag mu=0.865+1.238i, C=1.25?). Modes via SHIFT-INVERT IRA. Smooth Q=-3 (ckpoint_lat.240 op-reflow
# 4t0), m=0.1 mprec=0.3, Acut=1.0 sharp. Driver Test_wilson_frameopt_claude.cc. Single QUIET gpu0. No rm/kill.
set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/wilson_mx_claude.log
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
# cut variable = |sin p| = sqrt(sum_j sin^2 p_j) in [0,2] (user: kinetic free eigenvalue, with sin p_j)
ACUTLIST=0.3,0.5,0.7,1.0,1.4

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
  echo "-------- Mx inner vs outer, Acut sweep ${ACUTLIST}, smooth Q=-3, m=${MASS} mprec=${MPREC} --------"
  ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1 --config ${RAW} --op-reflow-eps 0.05 --op-reflow-nstep 232 \
      --flow-nstep 0 --spectrum --shiftinvert --si-tol ${SITOL} --mass ${MASS} --mprec ${MPREC} \
      --spec-nstop ${NSTOP} --spec-nk ${NK} --spec-nm ${NM} \
      --mx --mx-acut-list ${ACUTLIST}

  echo ""
  echo "======== SUMMARY ========"
  grep -nE "== Mx|Acut=|MXSCAN|RED" ${LOG} | grep -vE "warning|::write|\.h\(:"
  echo "======== done ========"
} 2>&1 | tee ${LOG}
