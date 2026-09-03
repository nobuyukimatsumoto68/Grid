#!/bin/bash
# Post-merge compile check: does our _claude code compile against the 264-commit-newer Grid API?
# Compiles Test_dwf_freeprec_claude.cc (pulls in FreeMobius5D_claude.h + FFT_claude.h + the whole Grid
# header tree) to an object file ONLY -- no link, no run -- so it surfaces header/API breakage fast.
# If it needs a fresh configure/bootstrap first (configure.ac changed in the merge), that error shows here.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_dwf_freeprec_claude.cc
OBJ=${BUILD}/Test_dwf_freeprec_mergecheck_claude.o
LOG=${ROOT}/merge_compile_check_claude.log

export OMP_NUM_THREADS=4

{
  echo "======== post-merge compile-only check $(date) ========"
  echo "HEAD = $(cd ${SRC} && git log --oneline -1)"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  echo "--- compiling (default = fp32) Test_dwf_freeprec_claude.cc -> .o (no link) ---"
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${TEST} -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "COMPILE FAILED (rc=${rc}) -- see errors above; our _claude code needs adapting to the new API"
    exit ${rc}
  fi
  echo "COMPILE OK -- our _claude code is API-compatible with the merged Grid"
} 2>&1 | tee ${LOG}
