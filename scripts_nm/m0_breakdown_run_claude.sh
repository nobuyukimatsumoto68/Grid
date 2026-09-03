#!/bin/bash
# Complete-picture benchmark: the full M0 apply breakdown (the "leftovers" fftbench did not measure --
# the (4Ls)x(4Ls) block solve + the Omega colour-matmuls), plus M0-vs-D_W(Dhop) in the same run. Builds
# Test_dwf_freeprec_claude.cc (now with M0.report_timers() + a Dhop loop in the run_m0 block) against the
# CPU/MPI build_mpi, runs single-rank 8^4 on the SU(3) config with --ops m0 (skips the slow CGNE/M1).
# Watch: "[M0 timers]" (omega/F), "[F timers]" (phase/fft_fwd/solve/fft_bwd), "[Dhop] us/apply",
# "[M0/Dhop] ... Dhop-equivalents". User runs this; Claude reads the log. No rm, no kill.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build_mpi
GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_dwf_freeprec_claude.cc
OBJ=${BUILD}/Test_dwf_freeprec_claude.o
BIN=${BUILD}/Test_dwf_freeprec_claude
LOG=${ROOT}/m0_breakdown_run_claude.log
MPIRUN=/mnt/hdd_barracuda/opt/openmpi_cuda/bin/mpirun
GRID=8.8.8.8
CFG=/mnt/baracuda_14/dwms/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc

export OMP_NUM_THREADS=8

{
  echo "======== [0/2] build_mpi guard $(date) ========"
  if [ ! -f ${BUILD}/lib/libGrid.a ] || [ ! -f ${BUILD}/include/Grid/Grid.h ]; then
    echo "  build_mpi missing -> run scripts_nm/grid_mpi_build_claude.sh first; stopping."
    exit 1
  fi
  echo "  present"

  echo "======== [1/2] COMPILE + LINK test (vs build_mpi) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
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

  echo "======== [2/2] RUN single-rank 8^4 --ops m0 (M0 breakdown + M0/Dhop) ========"
  ${MPIRUN} -np 1 ${BIN} --grid ${GRID} --mpi 1.1.1.1 --config ${CFG} --ops m0
  echo "exit = $?"
} 2>&1 | tee ${LOG}
