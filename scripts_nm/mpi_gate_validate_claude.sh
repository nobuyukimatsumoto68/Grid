#!/bin/bash
# MULTI-RANK VALIDATION of the free-prec block solve (the only open item after the Chunk-1 MPI fix).
# The fix is IN CODE (FreeMobius5D_claude.h: Minv keyed/sized by LOCAL V4loc :487,489; momentum VALUE
# from GLOBAL coord kglob=pcoor*Ll+kc, global extent Lg :500-501; MomentumSpaceSolve iterates idx4<V4loc
# :1280). This script EXECUTES the multi-rank gate that had not been run: compile the free-prec test
# against the GPU+MPI build/ (accelerator=cuda + accelerator-aware-mpi, sm_70), then run 8^4 at
#   [A] --mpi 1.1.1.1  (1 GPU, the reference)
#   [B] --mpi 2.1.1.1  (2 GPUs, X split -> local 4.8.8.8; exercises BOTH the block-solve local indexing
#                        AND the parallel Cshift-ring FFT over the split X dim)
# PASS CRITERIA (printed at the end):
#   1. gate 1 (cold gate ||F D v - v|| and ||D F v - v||, unit gauge) PASS at BOTH [A] and [B]
#      -- this is the pure MPI-indexing correctness test (config-independent).
#   2. FGMRES headline 'Converged on iteration N' IDENTICAL between [A] and [B] on the real config
#      -- proves Omega + block solve + FFT decompose correctly end-to-end (expect N=79).
# No rm / no kill anywhere. Correctness is load-independent, so a busy GPU is fine (numbers, not wall).

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/mpi_gate_validate_claude.log
CFG=${ROOT}/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc
MPIRUN=/mnt/hdd_barracuda/opt/openmpi_cuda/bin/mpirun

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1

{
  echo "======== [0] compile Test_dwf_freeprec_claude.cc against GPU+MPI build/ $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"

  OBJ=${BUILD}/Test_dwf_freeprec_mpigate_claude.o
  BIN=${BUILD}/Test_dwf_freeprec_mpigate_claude
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${SRC}/tests/solver/Test_dwf_freeprec_claude.cc -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]
  then
    echo "COMPILE FAILED (rc=${rc}) -- if a stale build/include, run grid_freeprec_build_v2_claude.sh first"
    exit ${rc}
  fi
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]
  then
    echo "LINK FAILED (rc=${rc})"
    exit ${rc}
  fi

  echo "======== [A] REFERENCE: 8^4 --mpi 1.1.1.1 (1 GPU) ========"
  ${MPIRUN} -np 1 ${BIN} --grid 8.8.8.8 --mpi 1.1.1.1 --config ${CFG} --ops m0

  echo "======== [B] 2-GPU: 8^4 --mpi 2.1.1.1 (X split -> local 4.8.8.8) ========"
  ${MPIRUN} -np 2 ${BIN} --grid 8.8.8.8 --mpi 2.1.1.1 --config ${CFG} --ops m0

  echo "======== SUMMARY (extracted from this log) ========"
  echo "-- gate 1 (cold gate) lines (want PASS in BOTH [A] and [B]):"
  grep -nE "gate 1:|F D v - v|D F v - v|gate 1 .* (PASS|FAIL)" ${LOG} | grep -iE "PASS|FAIL"
  echo "-- FGMRES convergence lines (want the SAME iteration N in [A] and [B]):"
  grep -niE "Converged on iteration|FGMRES.*iteration" ${LOG}
  echo "======== MPI GATE VALIDATION: done ========"
  echo "PASS if: gate 1 PASS in BOTH runs AND the two 'Converged on iteration N' match (expect 79)."
} 2>&1 | tee ${LOG}
