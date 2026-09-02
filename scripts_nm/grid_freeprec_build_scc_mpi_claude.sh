#!/bin/bash
# SCC CPU/MPI build of the free-limit preconditioner test into build_mpi/Test_dwf_freeprec_claude.
# Companion to grid_freeprec_build_scc_claude.sh (that one targets the GPU build/ tree); this one links
# against the AVX+MPI install tree build_mpi/ via its grid-config (mpicxx, no CUDA). Compile-only -- the
# binary RUNS via qsub (grid_freeprec_run_1node_qsub_claude.sh). User runs this; Claude reads the log.
#
# We only add -I<source-root> so the header-only FreeMobius5D_claude.h (not in the installed tree)
# resolves. Rebuild after editing Test_dwf_freeprec_claude.cc OR FreeMobius5D_claude.h (neither is part
# of libGrid.a). Re-run grid_build_scc_mpi_claude.sh yourself only after changing Grid library source.
#
# Run:  bash grid_freeprec_build_scc_mpi_claude.sh   (single-thread compile is fine; ~1-2 min)

set -u

module load gcc/12.2.0
module load openmpi/4.1.5_gnu-12.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
SRC=${ROOT}/Grid
BUILD=${ROOT}/build_mpi
GC=${BUILD}/bin/grid-config
TEST=${SRC}/tests/solver/Test_dwf_freeprec_claude.cc
OBJ=${BUILD}/Test_dwf_freeprec_claude.o
BIN=${BUILD}/Test_dwf_freeprec_claude
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"
LOG=${LOGDIR}/grid_freeprec_build_scc_mpi_claude.log

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

{
  echo "======== [0/3] install tree check $(date) ========"
  if [ ! -f "${BUILD}/lib/libGrid.a" ] || [ ! -f "${BUILD}/include/Grid/Grid.h" ]; then
    echo "  install tree missing at ${BUILD} -> run grid_build_scc_mpi_claude.sh first; stopping."
    exit 1
  fi
  echo "  install tree present -> compiling the test only"

  echo "======== [1/3] flags from grid-config ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  # CPU/mpicxx build carries no `-x cu`, so this swap is a harmless no-op (kept to mirror the GPU script).
  CXXLD="${CXX/-x cu/-link}"
  echo "CXX      = ${CXX}"
  echo "CXXFLAGS = ${CXXFLAGS}"
  echo "LDFLAGS  = ${LDFLAGS}"
  echo "LIBS     = ${LIBS}"

  echo "======== [2/3] COMPILE (-I${SRC} FIRST for FreeMobius5D_claude.h) ========"
  # Include order matters. grid-config puts -I<build_mpi>/include in CXXFLAGS, where an OLD INSTALLED
  # copy of FreeMobius5D_claude.h lives (make install does NOT refresh this hand-added header). First -I
  # match wins per header, so we put the SOURCE tree first -> the repo header (with M1 =
  # FreeLimitPreconditioner1) shadows the stale installed one, and every Grid header comes from source.
  # But the generated Config.h is bare-quoted (#include "Config.h") and exists ONLY in the build tree, so
  # we add -I${BUILD}/include/Grid (which holds Config.h) right after the source dir to resolve it.
  ${CXX} -I${SRC} -I${BUILD}/include/Grid ${CXXFLAGS} -c ${TEST} -o ${OBJ}
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
  echo "  next: qsub -v CONFIG=<cfg>,GRID=16.16.16.16,OPS=m1 grid_freeprec_run_1node_qsub_claude.sh"
} 2>&1 | tee ${LOG}
