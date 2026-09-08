#!/usr/bin/env bash
# DEFLATED-SUBSPACE FRAME OPT experiment (user 2026-09-05): determine Omega by descending the M0 L2 loss
# L = sum||M0(Omega) D_W phi - phi||^2 ONLY on the 3 chiral zero modes the IRA found (the deflated subspace),
# then build BOTH M0 and M1 from that Omega and compare their action on every mode vs the Landau frame.
# Question: does a frame tuned to the chiral blocker sector improve M0/M1 there -- and at what cost to the
# bulk? Modes via SHIFT-INVERT IRA (CGNE D_W^{-1}). Smooth Q=-3 (ckpoint_lat.240 op-reflow 4t0), m=0.1
# mprec=0.3, Nstop=16. Driver Test_wilson_frameopt_claude.cc. Single QUIET gpu0. No rm / no kill.
set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/wilson_deflate_frameopt_claude.log
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
DFITER=300
DFETA=0.1

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
  echo "-------- shift-invert IRA + M0/M1 + deflate-frameopt, smooth Q=-3, m=${MASS} mprec=${MPREC} --------"
  ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1 --config ${RAW} --op-reflow-eps 0.05 --op-reflow-nstep 232 \
      --flow-nstep 0 --spectrum --shiftinvert --si-tol ${SITOL} --mass ${MASS} --mprec ${MPREC} \
      --spec-nstop ${NSTOP} --spec-nk ${NK} --spec-nm ${NM} \
      --deflate-frameopt --df-iter ${DFITER} --df-eta ${DFETA}

  echo ""
  echo "======== SUMMARY ========"
  grep -nE "shift-invert|D_W RR mode\[|DEFLATE-FRAMEOPT|fo_descend|CMP\[|Nconv" ${LOG} | grep -vE "warning|::write|\.h\(:"
  echo "======== done ========"
} 2>&1 | tee ${LOG}
