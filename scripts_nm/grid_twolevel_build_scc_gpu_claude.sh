#!/bin/bash
# SCC GPU (CUDA) build of the two-level deflation CHECK driver into build_merged/Test_dwf_twolevel_claude.
# Chunk 1 (now): computes the lowest N |D_DW|^2 modes (Chebyshev-IRL). Same single-file nvcc recipe as
# grid_freeprec_build_scc_gpu_merged_claude.sh: link the merged GPU tree via build_merged grid-config,
# -I<src> FIRST + -I<build>/include/Grid for Config.h, CXXLD swaps -x cu -> -link.
# Plan: scripts_nm/twolevel_deflation_check_impl_plan_claude.md.

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
LOG=${LOGDIR}/grid_twolevel_build_scc_gpu_claude.log

# The four split single-purpose binaries (share tests/solver/twolevel_common_claude.h). Override with
# TARGETS="Test_dwf_deflprec_claude" to rebuild just one. The old monolithic Test_dwf_twolevel_claude is
# no longer built by default (kept as reference).
TARGETS=${TARGETS:-"Test_dwf_svddump_rbcgne_claude Test_dwf_svddump_deflprec_claude Test_dwf_rbcgne_claude Test_dwf_deflprec_claude Test_dwf_m0d5_overlap_claude"}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

# PREC=fp64 builds the free Mobius inverse F in double -> clean loss for the FD gate (fp32 noise ~1e-7
# corrupts the finite difference). PREC=fp32 (default) for production speed.
PREC=${PREC:-fp32}
PRECDEF=""
if [ "${PREC}" = "fp64" ]; then PRECDEF="-DFREEMOBIUS5D_FP64"; fi

{
  echo "======== [0/3] merged GPU tree check $(date) ========"
  if [ ! -f "${BUILD}/lib/libGrid.a" ] || [ ! -f "${BUILD}/include/Grid/Grid.h" ]; then
    echo "  merged GPU tree missing at ${BUILD} -> run grid_build_scc_gpu_merged_claude.sh first; stopping."
    exit 1
  fi
  echo "  present -> compiling the split two-level binaries  (PREC=${PREC} ${PRECDEF})"
  echo "  TARGETS = ${TARGETS}"

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
  echo "======== all targets built ========"
  echo "  run order: svddump_rbcgne + svddump_deflprec  (dump)  ->  rbcgne + deflprec  (read+solve)"
} 2>&1 | tee ${LOG}
