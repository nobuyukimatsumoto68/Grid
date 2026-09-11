#!/bin/bash
# SCC GPU (CUDA) build of the 24^4 fp32 comparison driver into
# build_merged/Test_dwf_freeprec_fp32_claude. Single-file nvcc recipe (same as the two-level build):
# link the merged GPU tree via build_merged grid-config, -I<src> FIRST + -I<build>/include/Grid for
# Config.h, CXXLD swaps -x cu -> -link. Always defines FREEMOBIUS5D_FP32 so the free-inverse momentum
# core scalar is ComplexF (single Minv -> less memory). The SOLVE fields (D, source, RB-CGNE, GMRES-DR
# basis) are LatticeFermionF; the free inverse F stays WilsonImplD (double field, single core) and is
# bridged to the single outer solve by FreeInvF32Adapter in the driver.
# Source: tests/solver/Test_dwf_freeprec_fp32_claude.cc; plan dwf_freeprec_fp32_impl_plan_claude.md.

set -u

module load cuda/12.8
module load gcc/13.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
SRC=${ROOT}/Grid
BUILD=${BUILD:-${ROOT}/build_merged}
GC=${BUILD}/bin/grid-config
[ -x "${GC}" ] || GC=${BUILD}/grid-config
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"
LOG=${LOGDIR}/grid_freeprec_fp32_build_scc_gpu_claude.log

T=Test_dwf_freeprec_fp32_claude
PRECDEF="-DFREEMOBIUS5D_FP32"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

{
  echo "======== [0] merged GPU tree check $(date) ========"
  if [ ! -f "${BUILD}/lib/libGrid.a" ] || [ ! -f "${BUILD}/include/Grid/Grid.h" ]; then
    echo "  merged GPU tree missing at ${BUILD} -> run grid_build_scc_gpu_merged_claude.sh first; stopping."
    exit 1
  fi
  echo "  present -> compiling ${T}  (${PRECDEF})"

  echo "======== [1] flags from grid-config (${GC}) ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  echo "CXX   = ${CXX}"
  echo "CXXLD = ${CXXLD}"

  TEST=${SRC}/tests/solver/${T}.cc
  OBJ=${BUILD}/${T}.o
  BIN=${BUILD}/${T}
  if [ ! -f "${TEST}" ]; then echo "MISSING SOURCE ${TEST}; stopping."; exit 1; fi
  echo "======== COMPILE ${T} (-I${SRC} FIRST + build Config.h) ========"
  ${CXX} -I${SRC} -I${BUILD}/include/Grid ${PRECDEF} ${CXXFLAGS} -c ${TEST} -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "COMPILE FAILED ${T} (rc=${rc}); stopping."; exit ${rc}; fi
  echo "======== LINK ${T} ========"
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "LINK FAILED ${T} (rc=${rc}); stopping."; exit ${rc}; fi
  echo "  built ${BIN}"
} 2>&1 | tee ${LOG}
