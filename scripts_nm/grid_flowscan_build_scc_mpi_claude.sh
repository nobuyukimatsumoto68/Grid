#!/bin/bash
# SCC CPU/MPI build of the standalone frame-flow-TIME scan driver into
# build_mpi/Test_dwf_flowscan_claude. Same recipe as grid_freeprec_build_scc_mpi_claude.sh (grid-config,
# -I<src> FIRST so the repo header wins + -I<build>/include/Grid for generated Config.h), just a different
# .cc. Compile-only; runs via qsub (grid_freeprec_flowscan_qsub/wrapper). User runs this; Claude reads log.
#
# POST-MERGE NOTE (upstream/develop merged into dwf_prec): the install tree (build_mpi/{include,lib}) must
# be REFRESHED (bootstrap.sh -> reconfigure -> make -> make install) BEFORE this, else you link a stale
# pre-merge libGrid.a. Coordinate with the local agent's grid_freeprec_build_v2_claude.sh full-rebuild.

set -u

module load gcc/12.2.0
module load openmpi/4.1.5_gnu-12.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
SRC=${ROOT}/Grid
BUILD=${BUILD:-${ROOT}/build_mpi_merged}   # merged tree (build grid_build_scc_mpi_merged_claude.sh first)
# grid-config is at the build ROOT; bin/grid-config only exists after a full (root) make install.
GC=${BUILD}/bin/grid-config
[ -x "${GC}" ] || GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_dwf_flowscan_claude.cc
OBJ=${BUILD}/Test_dwf_flowscan_claude.o
BIN=${BUILD}/Test_dwf_flowscan_claude
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"
LOG=${LOGDIR}/grid_flowscan_build_scc_mpi_claude.log

# Precision of the free-limit F apply (FreeMobius5D_claude.h). PREC=fp32 (default post-merge) = faster,
# FGMRES iters identical-or-+-1-2. PREC=fp64 = the DOUBLE-F reference (-DFREEMOBIUS5D_FP64), iters exactly
# match the pre-merge running scan -- use it if you want the flow-scan counts bit-comparable to the 40
# in-flight (pre-merge, double-F) jobs.
PREC=${PREC:-fp32}
PRECDEF=""
if [ "${PREC}" = "fp64" ]; then PRECDEF="-DFREEMOBIUS5D_FP64"; fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

{
  echo "======== [0/3] install tree check $(date) ========"
  if [ ! -f "${BUILD}/lib/libGrid.a" ] || [ ! -f "${BUILD}/include/Grid/Grid.h" ]; then
    echo "  install tree missing at ${BUILD} -> run the full Grid build first; stopping."
    exit 1
  fi
  echo "  install tree present -> compiling the flowscan test only"

  echo "======== [1/3] flags from grid-config ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  echo "CXX      = ${CXX}"
  echo "CXXFLAGS = ${CXXFLAGS}"

  echo "======== [2/3] COMPILE (-I${SRC} FIRST + build Config.h)  PREC=${PREC} ${PRECDEF} ========"
  ${CXX} -I${SRC} -I${BUILD}/include/Grid ${PRECDEF} ${CXXFLAGS} -c ${TEST} -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "COMPILE FAILED (rc=${rc}); stopping."
    exit ${rc}
  fi

  echo "======== [3/3] LINK ========"
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "LINK FAILED (rc=${rc}); stopping."
    exit ${rc}
  fi
  echo "  built ${BIN}"
} 2>&1 | tee ${LOG}
