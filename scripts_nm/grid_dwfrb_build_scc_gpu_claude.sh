#!/bin/bash
# SCC GPU (CUDA) build of the DWF Fourier/RB interacting solve (chunk D1b) into
# build_merged/Test_dwf_freeprec_rb_solve_claude. Same single-file nvcc recipe as
# grid_twolevel_build_scc_gpu_claude.sh: link the merged GPU tree via build_merged grid-config,
# -I<src> FIRST + -I<build>/include/Grid for Config.h, CXXLD swaps -x cu -> -link.
# Plan: scripts_nm/../tests/solver/dwf_fourier_rb_impl_plan_claude.md.
#
# This reads a NERSC config (BinaryIO / AggregateExchange), so it MUST build against build_merged
# (the current GPU install), NOT build_mpi (whose libGrid.a lacks AllToAllV / aggregateTargetBytes).
#
# Run:  bash grid_dwfrb_build_scc_gpu_claude.sh   (user runs; Claude reads the log)

set -u

module load cuda/12.8
module load gcc/13.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
SRC=${ROOT}/Grid
BUILD=${BUILD:-${ROOT}/build_merged}
GC=${BUILD}/bin/grid-config
[ -x "${GC}" ] || GC=${BUILD}/grid-config
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"
LOG=${LOGDIR}/grid_dwfrb_build_scc_gpu_claude.log

TARGETS=${TARGETS:-"Test_dwf_freeprec_rb_solve_claude"}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

# PREC=fp32 (default) uses the single-precision F apply (fine for a preconditioner; RGMRES is flexible).
# PREC=fp64 builds F in double (-DFREEMOBIUS5D_FP64) if a tighter free apply is ever wanted.
PREC=${PREC:-fp32}
PRECDEF=""
if [ "${PREC}" = "fp64" ]; then PRECDEF="-DFREEMOBIUS5D_FP64"; fi

{
  echo "======== [0] merged GPU tree check $(date) ========"
  if [ ! -f "${BUILD}/lib/libGrid.a" ] || [ ! -f "${BUILD}/include/Grid/Grid.h" ]; then
    echo "  merged GPU tree missing at ${BUILD} -> run grid_build_scc_gpu_merged_claude.sh first; stopping."
    exit 1
  fi
  echo "  present -> compiling  (PREC=${PREC} ${PRECDEF})  TARGETS=${TARGETS}"

  echo "======== [1] flags from grid-config (${GC}) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  echo "CXX   = ${CXX}"
  echo "CXXLD = ${CXXLD}"

  for T in ${TARGETS}; do
    TEST=${SRC}/tests/solver/${T}.cc
    OBJ=${BUILD}/${T}.o
    BIN=${BUILD}/${T}
    if [ ! -f "${TEST}" ]; then echo "MISSING SOURCE ${TEST}; stopping."; exit 1; fi
    echo "======== COMPILE ${T} (-I${SRC} FIRST + build Config.h) ========"
    ${CXX} -I${SRC} -I${BUILD}/include/Grid ${PRECDEF} ${CXXFLAGS} -c ${TEST} -o ${OBJ}
    rc=$?
    if [ ${rc} -ne 0 ]; then echo "COMPILE FAILED ${T} (rc=${rc}); stopping."; exit ${rc}; fi
    echo "======== LINK ${T} ========"
    ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
    rc=$?
    if [ ${rc} -ne 0 ]; then echo "LINK FAILED ${T} (rc=${rc}); stopping."; exit ${rc}; fi
    echo "  built ${BIN}"
  done
  echo "======== built; run via grid_dwfrb_gpu_qsub_claude.sh ========"
} 2>&1 | tee ${LOG}
