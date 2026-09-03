#!/bin/bash
# C0 gate for the custom DOF-payload FFT (DofFFT_claude.h): direct radix-L DFT along ONE unsplit dim.
# Compiles Test_dof_fft_c0_claude.cc standalone against the merged Grid install tree (uses PlannedFFT as
# the oracle) -- NO libGrid rebuild, NO scripts/filelist/reconfigure. The new header DofFFT_claude.h is
# picked up from SOURCE via -I${SRC} (it is not yet in build/include); every other Grid header comes from
# the install tree. Read the three numbers + the "C0 PASS/FAIL" line back.
#   (1) roundtrip  iFFT(FFT(v)) - v            -> ~1e-24 (squared rel norm)
#   (2a/2b) DofFFT vs PlannedFFT fwd/bwd on x  -> ~1e-24 (pins sign, ordering=freq-coord, 1/L scale)
# Plan: dwf4_qcd_claude/grid_custom_dof_fft_impl_plan_claude.md (sec 4b). Runs on GPU (the CUDA build).

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_dof_fft_c0_claude.cc
OBJ=${BUILD}/Test_dof_fft_c0_claude.o
BIN=${BUILD}/Test_dof_fft_c0_claude
LOG=${ROOT}/dof_fft_c0_gpu_claude.log

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

{
  echo "======== compile $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${TEST} -o ${OBJ}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "LINK FAILED (rc=${rc})"; exit ${rc}; fi

  # Correctness at BOTH L=8 (radix-8) and L=16 (radix-16) -- the kernel is L-GENERIC (same direct-DFT
  # code, bigger matrix/more vObjs; no separate radix-16 path). Any L=16 register spill only hurts SPEED,
  # not the bit-exact oracle. PlannedFFT is slow at 16^4 (~20ms/fft) -- fine, correctness ignores speed.
  echo "======== run C0 gate : L = 8 (8^4, Ls=8) ========"
  ${BIN} --grid 8.8.8.8 --mpi 1.1.1.1
  echo "======== run C0 gate : L = 16 (16^4, Ls=8) ========"
  ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1
  echo "done"
} 2>&1 | tee ${LOG}
