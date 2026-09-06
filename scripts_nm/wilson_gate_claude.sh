#!/bin/bash
# Chunk 0 gate for the free WILSON preconditioner (FreeWilson_claude.h). Build + run the cold gate at 8^4
# (unit gauge, periodic BC): ||F_W D_W v - v||/||v|| and ||D_W F_W v - v||/||v|| must be ~eps vs Grid's
# WilsonFermionD -> validates F_W = D_W^free^{-1} AND the mass/gamma convention (free_dw_p M5=-mass).
# Correctness (not timing) -> a busy GPU is fine. Single GPU. No rm / no kill.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/wilson_gate_claude.log

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

{
  echo "======== compile Test_wilson_freeprec_claude.cc $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"

  OBJ=${BUILD}/Test_wilson_freeprec_claude.o
  BIN=${BUILD}/Test_wilson_freeprec_claude
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${SRC}/tests/solver/Test_wilson_freeprec_claude.cc -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]
  then
    echo "COMPILE FAILED (rc=${rc}) -- likely a convention/type fix in FreeWilson_claude.h"
    exit ${rc}
  fi
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]
  then
    echo "LINK FAILED (rc=${rc})"
    exit ${rc}
  fi

  echo "======== run: cold gate 8^4 (unit gauge, periodic, m=0.1) ========"
  ${BIN} --grid 8.8.8.8 --mpi 1.1.1.1

  echo "======== also m=0.5 (convention robustness) ========"
  ${BIN} --grid 8.8.8.8 --mpi 1.1.1.1 --mass 0.5

  echo "======== SUMMARY ========"
  grep -nE "cold gate|F_W D_W v - v|GATE (PASS|FAIL)|convention mismatch|COMPILE FAILED" ${LOG} | grep -vE "warning|::write|strlen|\.h\(:"
  echo "======== done ========"
  echo "READ: GATE PASS (e1,e2 ~ 1e-11 or below) = F_W correct + convention matched. FAIL -> the printed"
  echo "      hint says to adjust the mass offset/sign or gamma sign (calibrate vs Grid WilsonFermionD)."
} 2>&1 | tee ${LOG}
