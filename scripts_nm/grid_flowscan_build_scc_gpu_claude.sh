#!/bin/bash
# SCC GPU (CUDA) build of the standalone flow-scan driver into build_merged/Test_dwf_flowscan_claude.
# Links against the MERGED GPU tree via build_merged/bin/grid-config (nvcc, multi-arch). Mirrors
# grid_freeprec_build_scc_claude.sh: -I<src> FIRST (repo header wins) + -I<build>/include/Grid for
# generated Config.h; CXXLD = compile driver with `-x cu` -> `-link`. nvcc compiles on a login node.
# Requires grid_build_scc_gpu_merged_claude.sh to have populated build_merged first.

set -u

module load cuda/12.8
module load gcc/13.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
SRC=${ROOT}/Grid
BUILD=${BUILD:-${ROOT}/build_merged}
# grid-config is at the build ROOT (configure-generated); bin/grid-config only exists after a full
# (root) make install -- we did `make -C Grid install` (library only), so prefer the root script.
GC=${BUILD}/bin/grid-config
[ -x "${GC}" ] || GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_dwf_flowscan_claude.cc
OBJ=${BUILD}/Test_dwf_flowscan_claude.o
BIN=${BUILD}/Test_dwf_flowscan_claude
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"
LOG=${LOGDIR}/grid_flowscan_build_scc_gpu_claude.log

# PREC=fp32 (default post-merge) or fp64 (-DFREEMOBIUS5D_FP64, double-F reference).
PREC=${PREC:-fp32}
PRECDEF=""
if [ "${PREC}" = "fp64" ]; then PRECDEF="-DFREEMOBIUS5D_FP64"; fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

{
  echo "======== [0/3] install tree check $(date) ========"
  if [ ! -f "${BUILD}/lib/libGrid.a" ] || [ ! -f "${BUILD}/include/Grid/Grid.h" ]; then
    echo "  merged GPU tree missing at ${BUILD} -> run grid_build_scc_gpu_merged_claude.sh first; stopping."
    exit 1
  fi
  echo "  merged GPU tree present -> compiling the flowscan test only  (PREC=${PREC} ${PRECDEF})"

  echo "======== [1/3] flags from grid-config ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  echo "CXX      = ${CXX}"
  echo "CXXLD    = ${CXXLD}"

  echo "======== [2/3] COMPILE (-I${SRC} FIRST + build Config.h) ========"
  ${CXX} -I${SRC} -I${BUILD}/include/Grid ${PRECDEF} ${CXXFLAGS} -c ${TEST} -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "COMPILE FAILED (rc=${rc}); stopping."; exit ${rc}; fi

  echo "======== [3/3] LINK ========"
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "LINK FAILED (rc=${rc}); stopping."; exit ${rc}; fi
  echo "  built ${BIN}"
} 2>&1 | tee ${LOG}
