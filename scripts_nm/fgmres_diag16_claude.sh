#!/bin/bash
# FGMRES CONVERGENCE DIAGNOSTIC at 16^4 m=0.1. Facts so far: config thermalized (plaq 0.594 = b6.0 ref),
# M0 correct (16^4 cold gate PASS 3.8e-7), RB-CGNE fine (248 iters, GREW from 181@8^4). FGMRES gave NO
# output -> restart-80 likely OOM'd (flexible GMRES stores ~2R fp64 vectors; 16^4 fp64 fermion ~100 MiB
# -> 2*80*100 = 16 GiB > 12 GiB GPU). "No-restart" OOMs worse. So use the memory-safe restart-40 (~8 GiB)
# and TURN ON the iterative residual log to SEE the process:
#   --ops m0 (FGMRES only; RB already done) --restart 40 --repeat 1 --log Message,Iterative
# WATCH: FlexibleGeneralisedMinimalResidual per-iteration residual -- does it DESCEND toward 1e-8
# (converging, maybe slowly) or FLATTEN (stalling, restart too small)? If it stalls at restart-40, the fix
# is a mixed-precision (fp32 Krylov) outer so a bigger restart fits; if it descends, we have N_it + wall.
# Single GPU. No rm / no kill.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/fgmres_diag16_claude.log
CFG16=${ROOT}/dwf4_qcd_claude/cfg_su3_16161616_b6.0_claude.nersc

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

{
  if [ ! -f "${CFG16}" ]
  then
    echo "CONFIG NOT FOUND: ${CFG16}"
    exit 2
  fi

  echo "======== compile Test_dwf_freeprec_claude.cc $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"

  OBJ=${BUILD}/Test_dwf_freeprec_diag16_claude.o
  BIN=${BUILD}/Test_dwf_freeprec_diag16_claude
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${SRC}/tests/solver/Test_dwf_freeprec_claude.cc -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]
  then
    echo "COMPILE FAILED (rc=${rc})"
    exit ${rc}
  fi
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]
  then
    echo "LINK FAILED (rc=${rc})"
    exit ${rc}
  fi

  echo "======== run: 16^4 m=0.1 --ops m0 --restart 40 --repeat 1 --log Message,Iterative ========"
  ${BIN} --grid 16.16.16.16 --mpi 1.1.1.1 --config ${CFG16} --ops m0 --restart 40 --repeat 1 --log Message,Iterative

  echo "======== SUMMARY (FGMRES residual trajectory) ========"
  grep -nE "FlexibleGeneralisedMinimalResidual: Iteration|FGMRES\(M0\)|Converged on iteration|cudaMalloc|out of memory" ${LOG} | grep -vE "::write|strlen|\.h\(:" | tail -40
  echo "======== done ========"
  echo "READ: residual DESCENDING to 1e-8 = converging (note N_it); FLAT = stalling (restart 40 too small"
  echo "      -> need mixed-precision outer for a bigger restart). If it OOMs even at 40, log shows cudaMalloc."
} 2>&1 | tee ${LOG}
