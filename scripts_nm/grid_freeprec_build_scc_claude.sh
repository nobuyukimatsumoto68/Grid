#!/bin/bash
# SCC variant of grid_freeprec_build_claude.sh: build + run the free-limit Mobius preconditioner gate.
# The peer's original (Ubuntu paths, inline GPU run) is left untouched for A/B.
#
# Links against Grid NATIVELY via build/bin/grid-config (Grid's own flag script), which now carries the
# SCC CUDA flags (nvcc, -ccbin g++, -gencode compute_70, -ldl -lrt) from grid_build_scc_claude.sh.
# We only add -I<source-root> so the one NEW header (FreeMobius5D_claude.h, not in the installed tree)
# resolves. Env mimics ../qed3/env.sh. Output tee'd to grid_freeprec_build_scc_claude.log.
#
# SPLIT for SCC: the login/interactive node has NO GPU, so nvcc COMPILES here fine but the binary can
# only RUN on a GPU node. Easiest path -- grab a GPU node and run this one script there (it builds AND
# runs in a single shot; nvidia-smi is then present so step [4] executes):
#     qrsh -P qfe -l gpus=1 -l gpu_c=7.0 -l h_rt=2:00:00 -pe omp 4
#     GRID=8.8.8.8 CFG=<path/to/config.nersc> bash grid_freeprec_build_scc_claude.sh
# (gpu_c 7.0 = V100/sm_70; use 8.0 for A100/sm_80, 9.0 for H200/sm_90 -- match the SM= used at build.)

set -u

module load cuda/12.8
module load gcc/13.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/bin/grid-config
TEST=${SRC}/tests/solver/Test_dwf_freeprec_claude.cc
OBJ=${BUILD}/Test_dwf_freeprec_claude.o
BIN=${BUILD}/Test_dwf_freeprec_claude
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"
LOG=${LOGDIR}/grid_freeprec_build_scc_claude.log

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

# RUN parameters (overridable). CFG is a NERSC-format SU(3) gauge config from the cloned qed2 repo.
# 8^4 HEADLINE default; for the 4^4 validation run (dense D_W cross-check also fires) set instead:
#   GRID=4.4.4.4 CFG=${ROOT}/qed2/dwf4_qcd_claude/cfg_su3_4444_b6.0_claude.nersc
GRID=${GRID:-8.8.8.8}
CFG=${CFG:-${ROOT}/qed2/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc}

{
  echo "======== [0/4] Grid install tree (only if missing) $(date) ========"
  # grid-config points at build/{include,lib}. Iteration edits only the test .cc + the header-only
  # FreeMobius5D_claude.h -- neither is part of libGrid.a -- so once installed we compile just the test.
  # Re-run grid_build_scc_claude.sh yourself only after changing Grid library source.
  if [ ! -f ${BUILD}/lib/libGrid.a ] || [ ! -f ${BUILD}/include/Grid/Grid.h ]; then
    echo "  install tree missing -> run grid_build_scc_claude.sh first; stopping."
    exit 1
  else
    echo "  install tree present -> compiling the test only"
  fi

  echo "======== [1/4] flags from grid-config ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  # link driver = compile driver with `-x cu` swapped for `-link` (matches Grid Makefile CXXLD).
  # WITHOUT this the linker inherits `-x cu` and tries to COMPILE the .o as CUDA source (garbage output).
  CXXLD="${CXX/-x cu/-link}"
  echo "CXX      = ${CXX}"
  echo "CXXLD    = ${CXXLD}"
  echo "CXXFLAGS = ${CXXFLAGS}"
  echo "LDFLAGS  = ${LDFLAGS}"
  echo "LIBS     = ${LIBS}"

  echo "======== [2/4] COMPILE ========"
  echo "+ compile Test_dwf_freeprec_claude.cc  (-I${SRC} FIRST for the new header)"
  # Include order matters. grid-config puts -I<build>/include in CXXFLAGS where an OLD INSTALLED copy of
  # FreeMobius5D_claude.h lives (make install does NOT refresh this hand-added header). First -I match
  # wins per header, so the SOURCE tree goes first -> the repo header (with M1 = FreeLimitPreconditioner1)
  # shadows the stale installed one. The generated Config.h is bare-quoted and lives ONLY in the build
  # tree, so add -I${BUILD}/include/Grid (holds Config.h) right after the source dir to resolve it.
  ${CXX} -I${SRC} -I${BUILD}/include/Grid ${CXXFLAGS} -c ${TEST} -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "COMPILE FAILED (rc=${rc}); stopping."
    exit ${rc}
  fi

  echo "======== [3/4] LINK (native: grid-config --ldflags --libs, -lGrid) ========"
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "LINK FAILED (rc=${rc}); stopping."
    exit ${rc}
  fi
  echo "  built ${BIN}"

  echo "======== [4/4] RUN (needs a GPU node) ========"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "  no GPU on this node -> BUILT ONLY (binary is ready at ${BIN})."
    echo "  To run: grab a GPU node and re-run this script there, e.g."
    echo "    qrsh -P qfe -l gpus=1 -l gpu_c=7.0 -l h_rt=2:00:00 -pe omp ${OMP_NUM_THREADS}"
    echo "    GRID=${GRID} CFG=<path/to/config.nersc> bash $0"
    exit 0
  fi
  if [ -z "${CFG}" ] || [ ! -f "${CFG}" ]; then
    echo "  CFG not set or missing: '${CFG}'"
    echo "  set CFG=<NERSC-format SU(3) config> (and GRID to match) and re-run on this GPU node."
    exit 0
  fi
  echo "+ ${BIN} --grid ${GRID} --mpi 1.1.1.1 --config ${CFG}"
  ${BIN} --grid ${GRID} --mpi 1.1.1.1 --config ${CFG}
  rc=$?
  echo "run exit code = ${rc}"
} 2>&1 | tee ${LOG}
