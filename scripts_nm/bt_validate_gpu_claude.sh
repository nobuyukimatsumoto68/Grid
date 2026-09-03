#!/bin/bash
# BT-1 validation: on-device block-Thomas solve vs the dense (4Ls)^2 solve. Three builds, each 8^4
# --config --ops m0 (same GPU load, back-to-back):
#   (1) DEFAULT           = fp32 + block-Thomas (the new default solve)
#   (2) fp32 + DENSE      = -DFREEMOBIUS5D_DENSE_SOLVE (fp32 dense reference; solve-timer A/B + iter match)
#   (3) fp64 + block-Thomas = -DFREEMOBIUS5D_FP64 (double BT; STRONG correctness: cold gate must be ~1e-16)
# Check: [BT-0 gate] PASS ~1e-16 (host); cold gate 1 PASS (fp64 BT ~1e-16, fp32 BT ~1e-6); FGMRES 79 iters
# in all three; [F timers] solve DROPS block-Thomas vs dense (target ~2-3x). grid_block_thomas_impl_plan.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_dwf_freeprec_claude.cc
LOG=${ROOT}/bt_validate_gpu_claude.log
GRID=8.8.8.8
CFG=/mnt/baracuda_14/dwms/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

build_one () {
  local tag="$1"
  local extra="$2"
  local obj="${BUILD}/Test_dwf_freeprec_${tag}_claude.o"
  local bin="${BUILD}/Test_dwf_freeprec_${tag}_claude"
  echo "======== compile ${tag} (${extra}) $(date) ========"
  ${CXX} ${CXXFLAGS} ${extra} -I${SRC} -c ${TEST} -o ${obj}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "${tag} COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${obj} -o ${bin} ${LDFLAGS} ${LIBS}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "${tag} LINK FAILED (rc=${rc})"; exit ${rc}; fi
  echo "======== run ${tag} 8^4 --config --ops m0 ========"
  ${bin} --grid ${GRID} --mpi 1.1.1.1 --config ${CFG} --ops m0
}

{
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"

  build_one bt_default ""
  build_one bt_fp32dense "-DFREEMOBIUS5D_DENSE_SOLVE"
  build_one bt_fp64 "-DFREEMOBIUS5D_FP64"
  echo "done"
} 2>&1 | tee ${LOG}
