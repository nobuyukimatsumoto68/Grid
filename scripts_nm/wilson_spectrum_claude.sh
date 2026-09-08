#!/usr/bin/env bash
# SPECTRUM of D_W and M0 D_W via IRA (non-Hermitian Arnoldi, ImplicitlyRestartedArnoldi_claude.h), smallest
# |lambda| (near 0). "Get the spectrum of D_W first" (user 2026-09-04): are there ~|Q|=3 near-zero (IR/
# topological) modes limiting convergence, and does the free-prec M0 cluster them near 1? On the smooth
# genuine Q=-3 config (raw ckpoint_lat.240 op-reflow to 4 t0). Test = Test_wilson_frameopt_claude.cc.
# Single QUIET gpu0. No rm / no kill.
#
# CAVEAT: smallest-|lambda| Arnoldi on the bare D_W (spectral radius ~8) can converge slowly for the deep-
# interior near-0 modes; the M0 D_W outliers (isolated near 0) converge more reliably. If D_W struggles,
# the robust fallback is Hermitian Lanczos (IRL) on gamma5*D_W -- ask.
set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/wilson_spectrum_claude.log
CFGDIR=${ROOT}/iwasaki16_b2.6_Qspread_claude
RAW=${CFGDIR}/ckpoint_lat.240

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

MASS=0.1
MPREC=0.3
NSTOP=16
NK=32
NM=64

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
  echo "-------- spectrum: smooth Q=-3 (op-reflow 4t0), m=${MASS} mprec=${MPREC} --------"
  ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1 --config ${RAW} --op-reflow-eps 0.05 --op-reflow-nstep 232 \
      --flow-nstep 0 --spectrum --mass ${MASS} --mprec ${MPREC} \
      --spec-nstop ${NSTOP} --spec-nk ${NK} --spec-nm ${NM}

  echo ""
  echo "======== SUMMARY (low eigenvalues) ========"
  grep -nE "SPECTRUM|D_W low|D_W RR mode\[|M0act\[|Nconv" ${LOG} | grep -vE "warning|::write|\.h\(:"
  echo "======== done ========"
} 2>&1 | tee ${LOG}
