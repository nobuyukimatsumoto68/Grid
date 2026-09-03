#!/bin/bash
# Fair (matched-load) A/B of the FFT working-space cache: build DEFAULT (FFT_claude cached plan/pgbuf) and
# -DFREEMOBIUS5D_GRID_FFT (Grid's uncached FFT), run BOTH back-to-back at 8^4 --ops m0 on gpu0 so they see
# the SAME GPU load. Compare [F timers] fft_fwd/fft_bwd + [M0/Dhop]. Confounded cross-run timing (busier
# GPU) is removed by adjacency. Both must gate1 PASS (~5e-16).

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_dwf_freeprec_claude.cc
LOG=${ROOT}/fft_cache_ab_gpu_claude.log
GRID=8.8.8.8
CFG=/mnt/baracuda_14/dwms/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

build_one() {
  local tag="$1"; shift
  local extra="$1"; shift
  local obj="${BUILD}/Test_dwf_freeprec_${tag}_claude.o"
  local bin="${BUILD}/Test_dwf_freeprec_${tag}_claude"
  local CXX="$(${GC} --cxx)"
  local CXXFLAGS="$(${GC} --cxxflags)"
  local LDFLAGS="$(${GC} --ldflags)"
  local LIBS="$(${GC} --libs)"
  local CXXLD="${CXX/-x cu/-link}"
  echo "---- build ${tag} (${extra:-default}) ----"
  ${CXX} ${CXXFLAGS} ${extra} -I${SRC} -c ${TEST} -o ${obj} || { echo "COMPILE ${tag} FAILED"; return 1; }
  ${CXXLD} ${CXXFLAGS} ${obj} -o ${bin} ${LDFLAGS} ${LIBS} || { echo "LINK ${tag} FAILED"; return 1; }
  echo "${bin}"
}

{
  echo "======== build both $(date) ========"
  BIN_CACHED=$(build_one cached "" | tail -1)
  BIN_UNCACHED=$(build_one uncached "-DFREEMOBIUS5D_GRID_FFT" | tail -1)
  echo "cached=${BIN_CACHED}"
  echo "uncached=${BIN_UNCACHED}"

  echo "======== run A: CACHED (default) 8^4 --ops m0 ========"
  ${BIN_CACHED}   --grid ${GRID} --mpi 1.1.1.1 --config ${CFG} --ops m0
  echo "======== run B: UNCACHED (Grid FFT) 8^4 --ops m0 ========"
  ${BIN_UNCACHED} --grid ${GRID} --mpi 1.1.1.1 --config ${CFG} --ops m0
  echo "======== run A2: CACHED again (bracket the load) ========"
  ${BIN_CACHED}   --grid ${GRID} --mpi 1.1.1.1 --config ${CFG} --ops m0
  echo "done"
} 2>&1 | tee ${LOG}
