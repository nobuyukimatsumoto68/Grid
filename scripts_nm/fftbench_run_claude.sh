#!/bin/bash
# Chunk 4: build + run the standalone FFT scaling bench (Test_freeprec_fftbench_claude.cc) against the
# CPU/MPI Grid build (build_mpi/, from grid_mpi_build_claude.sh) across MPI decompositions. Measures the
# FFT comm-vs-compute cost and FFT-per-D_W-apply as ranks grow, on the 20 CPU cores here.
#
# Each run prints a "[fftbench] ... FFT/Dhop = ..." summary line; with --log Performance, FFT.h emits a
# per-dim breakdown -- grep "of which shift" (Cshift/comm) and "FFT kernels" (FFTW) to get the comm
# fraction. WEAK scaling keeps a fixed 8^4 local volume per rank; STRONG keeps a fixed 16^4 global
# volume. OMP_NUM_THREADS=1 so per-rank timing is clean. User runs this; Claude reads the log. No rm,
# no kill here.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build_mpi
GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_freeprec_fftbench_claude.cc
OBJ=${BUILD}/Test_freeprec_fftbench_claude.o
BIN=${BUILD}/Test_freeprec_fftbench_claude
LOG=${ROOT}/fftbench_run_claude.log
MPIRUN=/mnt/hdd_barracuda/opt/openmpi_cuda/bin/mpirun

export OMP_NUM_THREADS=1

# --- scaling sweeps: "np grid mpi" (20 cores here -> up to 16 ranks) ---
# WEAK: fixed 8^4 local volume per rank.
WEAK=(
  "1  8.8.8.8      1.1.1.1"
  "4  16.16.8.8    2.2.1.1"
  "8  16.16.16.8   2.2.2.1"
  "16 16.16.16.16  2.2.2.2"
)
# STRONG: fixed 16^4 global volume.
STRONG=(
  "1  16.16.16.16  1.1.1.1"
  "4  16.16.16.16  2.2.1.1"
  "8  16.16.16.16  2.2.2.1"
  "16 16.16.16.16  2.2.2.2"
)

{
  echo "======== [0/2] build_mpi install-tree guard $(date) ========"
  if [ ! -f ${BUILD}/lib/libGrid.a ] || [ ! -f ${BUILD}/include/Grid/Grid.h ]; then
    echo "  build_mpi tree missing -> run scripts_nm/grid_mpi_build_claude.sh first; stopping."
    exit 1
  fi
  echo "  present"

  echo "======== [1/2] COMPILE + LINK bench (vs build_mpi grid-config) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  echo "CXX = ${CXX}"
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${TEST} -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "COMPILE FAILED (rc=${rc}); stopping."
    exit ${rc}
  fi
  ${CXX} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "LINK FAILED (rc=${rc}); stopping."
    exit ${rc}
  fi

  echo "======== [2/2] SCALING RUNS (--log Message,Performance for the comm/kernel split) ========"
  echo "-------- WEAK scaling (fixed 8^4 local/rank) --------"
  for row in "${WEAK[@]}"; do
    set -- ${row}
    NP=$1
    GRID=$2
    MPI=$3
    echo "===== WEAK np=${NP} grid=${GRID} mpi=${MPI} ====="
    ${MPIRUN} -np ${NP} ${BIN} --grid ${GRID} --mpi ${MPI} --log Message,Performance
    echo "exit = $?"
  done

  echo "-------- STRONG scaling (fixed 16^4 global) --------"
  for row in "${STRONG[@]}"; do
    set -- ${row}
    NP=$1
    GRID=$2
    MPI=$3
    echo "===== STRONG np=${NP} grid=${GRID} mpi=${MPI} ====="
    ${MPIRUN} -np ${NP} ${BIN} --grid ${GRID} --mpi ${MPI} --log Message,Performance
    echo "exit = $?"
  done
} 2>&1 | tee ${LOG}
