#!/bin/bash
# SCC GPU (CUDA) build of the shrink-flow frame driver into build_merged/Test_wilson_frameopt_claude.
# Two-stage frame flow (Wilson to t0, then under-improved c1>0 shrink flow) + baseline/shrink frame
# A/B M0 solve. See tests/solver/shrinkflow_impl_plan_claude.md. Same recipe as
# grid_spectrum_build_scc_gpu_claude.sh: links the merged GPU tree via build_merged grid-config,
# -I<src> FIRST + -I<build>/include/Grid for Config.h, CXXLD swaps -x cu -> -link.

set -u

module load cuda/12.8
module load gcc/13.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
SRC=${ROOT}/Grid
BUILD=${BUILD:-${ROOT}/build_merged}
GC=${BUILD}/bin/grid-config
[ -x "${GC}" ] || GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_wilson_frameopt_claude.cc
OBJ=${BUILD}/Test_wilson_frameopt_claude.o
BIN=${BUILD}/Test_wilson_frameopt_claude
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"
LOG=${LOGDIR}/grid_wilson_frameopt_build_scc_gpu_claude.log

PREC=${PREC:-fp32}
PRECDEF=""
if [ "${PREC}" = "fp64" ]; then PRECDEF="-DFREEMOBIUS5D_FP64"; fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

{
  echo "======== [0/3] merged GPU tree check $(date) ========"
  if [ ! -f "${BUILD}/lib/libGrid.a" ] || [ ! -f "${BUILD}/include/Grid/Grid.h" ]; then
    echo "  merged GPU tree missing at ${BUILD} -> run grid_build_scc_gpu_merged_claude.sh first; stopping."
    exit 1
  fi
  echo "  present -> compiling the shrinkflow test only  (PREC=${PREC} ${PRECDEF})"

  echo "======== [1/3] flags from grid-config (${GC}) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  echo "CXX   = ${CXX}"
  echo "CXXLD = ${CXXLD}"

  echo "======== [2/3] COMPILE (-I${SRC} FIRST + build Config.h) ========"
  ${CXX} -I${SRC} -I${BUILD}/include/Grid ${PRECDEF} ${CXXFLAGS} -c ${TEST} -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "COMPILE FAILED (rc=${rc}); stopping."; exit ${rc}; fi

  echo "======== [3/3] LINK ========"
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "LINK FAILED (rc=${rc}); stopping."; exit ${rc}; fi
  echo "  built ${BIN}"
  echo "  submit: qsub -v CONFIG=${ROOT}/configs_iwasaki_16_b2.6/trivialQ_lat.747 ${SRC}/scripts_nm/grid_wilson_frameopt_gpu_qsub_claude.sh"
} 2>&1 | tee ${LOG}
