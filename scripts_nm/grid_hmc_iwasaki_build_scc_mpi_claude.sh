#!/bin/bash
# SCC CPU/MPI build of the quenched-Iwasaki HMC driver into build_mpi/Test_hmc_IwasakiGauge_claude.
# Standalone grid-config compile (the _claude test is NOT in tests/hmc/Makefile.am). Rebuild after
# editing Test_hmc_IwasakiGauge_claude.cc -- e.g. the new --beta CLI for the lattice-spacing scan.
# Compile-only; the binary RUNS via qsub (grid_gen_quenched_16_1node_scc_mpi_qsub_claude.sh, BETA=...).
# User runs this; Claude reads the log. ~1-2 min single-thread.

set -u

module load gcc/12.2.0
module load openmpi/4.1.5_gnu-12.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
SRC=${ROOT}/Grid
BUILD=${ROOT}/build_mpi
GC=${BUILD}/bin/grid-config
TEST=${SRC}/tests/hmc/Test_hmc_IwasakiGauge_claude.cc
OBJ=${BUILD}/Test_hmc_IwasakiGauge_claude.o
BIN=${BUILD}/Test_hmc_IwasakiGauge_claude
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"
LOG=${LOGDIR}/grid_hmc_iwasaki_build_scc_mpi_claude.log

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

{
  echo "======== [0/3] install tree check $(date) ========"
  if [ ! -f "${BUILD}/lib/libGrid.a" ] || [ ! -f "${BUILD}/include/Grid/Grid.h" ]; then
    echo "  install tree missing at ${BUILD} -> run grid_build_scc_mpi_claude.sh first; stopping."
    exit 1
  fi
  echo "  install tree present -> compiling the HMC test only"

  echo "======== [1/3] flags from grid-config ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  echo "CXX      = ${CXX}"
  echo "CXXFLAGS = ${CXXFLAGS}"
  echo "LDFLAGS  = ${LDFLAGS}"
  echo "LIBS     = ${LIBS}"

  echo "======== [2/3] COMPILE ========"
  # -I${SRC} FIRST (see grid_freeprec_build_scc_mpi_claude.sh): source tree wins over any stale installed
  # header, and -I${BUILD}/include/Grid resolves the bare-quoted generated Config.h.
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
  echo "  next: qsub -N gen24_b213 -v GRID=24.24.24.24,BETA=2.13,MDSTEPS=24 grid_gen_quenched_16_1node_scc_mpi_qsub_claude.sh"
} 2>&1 | tee ${LOG}
