#!/bin/bash
# SCC CPU/MPI build of the instanton-size probe into build_mpi_merged/Test_instsize_claude.
# Pure gauge-field analysis (clover q(x) + greedy lump extraction) for the shrink-flow checkpoints --
# no solver, cheap, runs single-rank on a login/interactive node. Mimics
# grid_freeprec_build_scc_mpi_claude.sh but targets the MERGED CPU tree build_mpi_merged (the
# pre-merge build_mpi libGrid.a lacks the merged BinaryIO symbols).
#
# Run:  bash grid_instsize_build_scc_mpi_claude.sh   (~1-2 min)

set -u

module load gcc/12.2.0
module load openmpi/4.1.5_gnu-12.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
SRC=${ROOT}/Grid
BUILD=${BUILD:-${ROOT}/build_mpi_merged}
GC=${BUILD}/bin/grid-config
[ -x "${GC}" ] || GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_instsize_claude.cc
OBJ=${BUILD}/Test_instsize_claude.o
BIN=${BUILD}/Test_instsize_claude
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"
LOG=${LOGDIR}/grid_instsize_build_scc_mpi_claude.log

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

{
  echo "======== [0/3] install tree check $(date) ========"
  if [ ! -f "${BUILD}/lib/libGrid.a" ] || [ ! -f "${BUILD}/include/Grid/Grid.h" ]; then
    echo "  merged CPU tree missing at ${BUILD} -> run grid_build_scc_mpi_merged_claude.sh first; stopping."
    exit 1
  fi
  echo "  install tree present -> compiling the probe only"

  echo "======== [1/3] flags from grid-config (${GC}) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  echo "CXX = ${CXX}"

  echo "======== [2/3] COMPILE ========"
  ${CXX} -I${SRC} -I${BUILD}/include/Grid ${CXXFLAGS} -c ${TEST} -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "COMPILE FAILED (rc=${rc}); stopping."; exit ${rc}; fi

  echo "======== [3/3] LINK ========"
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "LINK FAILED (rc=${rc}); stopping."; exit ${rc}; fi
  echo "  built ${BIN}"
  echo "  run: bash ${SRC}/scripts_nm/run_instsize_claude.sh   (loops over shrinkflow_ckp/)"
} 2>&1 | tee ${LOG}
