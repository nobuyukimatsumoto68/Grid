#!/usr/bin/env bash
# DIRECT FRAME OPTIMIZER in Grid (imported from the dwf4 concept, gate-validated there). Gradient-descend the
# frame Omega on L = sum_v ||M0(Omega) D_W v - v||^2 vs the Landau frame; compare FGMRES N_it. Runs the FD
# GATE first (validates the Grid port of the gradient -- rel err < 1e-4 = PASS), then descends, then FGMRES
# N_it(Landau) vs N_it(optimized). Smooth genuine Q=-3 config (raw ckpoint_lat.240 op-reflow to 4 t0).
# Test = Test_wilson_frameopt_claude.cc (branched copy of the defect-scan test + the frameopt functions).
# Single QUIET gpu0. No rm / no kill.
set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/wilson_frameopt_claude.log
CFGDIR=${ROOT}/iwasaki16_b2.6_Qspread_claude
RAW=${CFGDIR}/ckpoint_lat.240

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

MASS=0.1
MPREC=0.3
RESTART=20
FO_ITER=60
FO_ETA=0.1
FO_PROBES=4

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
  echo "-------- frameopt: smooth Q=-3 (op-reflow 4t0), m=${MASS} mprec=${MPREC} restart=${RESTART} --------"
  ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1 --config ${RAW} --op-reflow-eps 0.05 --op-reflow-nstep 232 \
      --flow-nstep 0 --frameopt --mass ${MASS} --mprec ${MPREC} --restart ${RESTART} \
      --fo-iter ${FO_ITER} --fo-eta ${FO_ETA} --fo-probes ${FO_PROBES}

  echo ""
  echo "======== SUMMARY ========"
  grep -nE "FD gate|L\(Landau|fo_descend|FGMRES\(M0, LANDAU|FGMRES\(M0, OPTIMIZED" ${LOG} | grep -vE "warning|::write|\.h\(:"
  echo "======== done ========"
} 2>&1 | tee ${LOG}
