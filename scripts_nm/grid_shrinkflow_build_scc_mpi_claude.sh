#!/bin/bash
# SCC CPU/MPI build of the shrinkflow driver into build_mpi_merged/Test_dwf_shrinkflow_claude.
# Purpose: run the QSqueezeGaugeAction --fdcheck (force validation, small lattice, single rank) on a
# CPU node without waiting for a GPU; production runs use the GPU build
# (grid_shrinkflow_build_scc_gpu_claude.sh). Mimics grid_instsize_build_scc_mpi_claude.sh.
#
# Run:  bash grid_shrinkflow_build_scc_mpi_claude.sh   (~2-3 min)

set -u

module load gcc/12.2.0
module load openmpi/4.1.5_gnu-12.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
SRC=${ROOT}/Grid
BUILD=${BUILD:-${ROOT}/build_mpi_merged}
GC=${BUILD}/bin/grid-config
[ -x "${GC}" ] || GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_dwf_shrinkflow_claude.cc
OBJ=${BUILD}/Test_dwf_shrinkflow_claude.o
BIN=${BUILD}/Test_dwf_shrinkflow_claude
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"
LOG=${LOGDIR}/grid_shrinkflow_build_scc_mpi_claude.log

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

{
  echo "======== [0/3] install tree check $(date) ========"
  if [ ! -f "${BUILD}/lib/libGrid.a" ] || [ ! -f "${BUILD}/include/Grid/Grid.h" ]; then
    echo "  merged CPU tree missing at ${BUILD} -> run grid_build_scc_mpi_merged_claude.sh first; stopping."
    exit 1
  fi
  echo "  install tree present -> compiling the driver only"

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
  echo "  fdcheck: mpirun -np 1 ${BIN} --grid 8.8.8.8 --mpi 1.1.1.1 --fdcheck --qn 2,4,6"
} 2>&1 | tee ${LOG}
