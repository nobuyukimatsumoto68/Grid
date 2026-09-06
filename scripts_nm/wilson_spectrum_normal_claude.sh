#!/usr/bin/env bash
# LOW SPECTRUM of the HERMITIAN squared normal Wilson operator D_W^dag D_W via Grid's TESTED Lanczos (IRL) --
# robust (no IRA restart-accuracy caveat), low modes converge cleanly. Same diagnostics as the D_W run:
# per mode eval=|lambda|^2, |lambda|, Rayleigh quotient lambda_RQ=<psi|D_W|psi>, CHIRALITY chi, resid; then
# how M0 (Landau frame) acts on each eigenmode (|M0 DW psi|/|psi| + residual). Smooth Q=-3 config
# (raw ckpoint_lat.240 op-reflow to 4 t0). Separate driver Test_wilson_spectrum_normal_claude.cc.
# Chebyshev acceleration for the small end: --cheby-lo (cutoff, find eval<lo) / --cheby-hi (~||D||^2) /
# --cheby-ord. TUNE lo/ord if too few modes converge. Single QUIET gpu0. No rm / no kill.
set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/wilson_spectrum_normal_claude.log
CFGDIR=${ROOT}/iwasaki16_b2.6_Qspread_claude
RAW=${CFGDIR}/ckpoint_lat.240

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

MASS=0.1
MPREC=0.3
NSTOP=80
NK=100
NM=160
CLO=0.5
CHI=70
CORD=81

{
  echo "======== compile Test_wilson_spectrum_normal_claude.cc $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  OBJ=${BUILD}/Test_wilson_spectrum_normal_claude.o
  BIN=${BUILD}/Test_wilson_spectrum_normal_claude
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${SRC}/tests/solver/Test_wilson_spectrum_normal_claude.cc -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "LINK FAILED (rc=${rc})"; exit ${rc}; fi

  echo ""
  echo "-------- DdagD spectrum: smooth Q=-3 (op-reflow 4t0), m=${MASS} mprec=${MPREC} --------"
  ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1 --config ${RAW} --op-reflow-eps 0.05 --op-reflow-nstep 232 \
      --flow-nstep 0 --spectrum --mass ${MASS} --mprec ${MPREC} \
      --spec-nstop ${NSTOP} --spec-nk ${NK} --spec-nm ${NM} \
      --cheby-lo ${CLO} --cheby-ord ${CORD}

  echo ""
  echo "======== SUMMARY ========"
  grep -nE "SPECTRUM|PowerMethod|D_W\^dag|DdagD: Nconv|DdD mode\[|M0act\[|DWproj\[" ${LOG} | grep -vE "warning|::write|\.h\(:"
  echo "======== done ========"
} 2>&1 | tee ${LOG}
