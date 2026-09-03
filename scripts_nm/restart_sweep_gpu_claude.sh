#!/bin/bash
# Back-to-back FGMRES restart sweep in ONE process (same frame + GPU load -> removes cross-run drift,
# pins the restart optimum). Builds default (cached FFT + on-device solve + phase), runs gpu0 8^4 --config
# --ops m0 --restart-sweep 256,64,32,24,20,16,12,8. Compare the WALL= lines. Then set the default restart.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_dwf_freeprec_claude.cc
OBJ=${BUILD}/Test_dwf_freeprec_sweep_claude.o
BIN=${BUILD}/Test_dwf_freeprec_sweep_claude
LOG=${ROOT}/restart_sweep_gpu_claude.log
GRID=8.8.8.8
CFG=/mnt/baracuda_14/dwms/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

{
  echo "======== compile (default) $(date) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${TEST} -o ${OBJ}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "COMPILE FAILED"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "LINK FAILED"; exit ${rc}; fi

  echo "======== restart sweep (one process, same frame) ========"
  ${BIN} --grid ${GRID} --mpi 1.1.1.1 --config ${CFG} --ops m0 --restart-sweep 256,64,32,24,20,16,12,8
  echo "done"
} 2>&1 | tee ${LOG}
