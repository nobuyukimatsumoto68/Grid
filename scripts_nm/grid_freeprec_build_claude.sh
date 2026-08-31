#!/bin/bash
# Chunk 0 handoff: build + run the free-limit Mobius preconditioner convention gate (Grid port).
#
# Links against Grid NATIVELY via build/grid-config (Grid's own flag script). That requires the install
# tree (build/include, build/lib) to be populated -- the earlier `make install` FAILED on the old
# __rdtsc error (see log_install), so step [0] re-runs `make install` now that the build is fixed.
# After that, grid-config gives the exact cxx/cxxflags/ldflags/libs Grid was built with, and we only
# add -I<source-root> so the one NEW header (FreeMobius5D_claude.h, not yet in the installed tree)
# resolves. Output tee'd to grid_freeprec_build_claude.log. User runs this; Claude reads the log.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_dwf_freeprec_claude.cc
OBJ=${BUILD}/Test_dwf_freeprec_claude.o
BIN=${BUILD}/Test_dwf_freeprec_claude
LOG=${ROOT}/grid_freeprec_build_claude.log

export OMP_NUM_THREADS=8

{
  echo "======== [0/4] Grid install tree (only if missing) $(date) ========"
  # grid-config points at build/{include,lib}. Populate them ONCE via `make install`. Our iteration
  # edits only the test .cc + the header-only FreeMobius5D_claude.h -- neither is part of libGrid.a --
  # so once installed we SKIP make install and compile just the test (seconds). Re-run make install
  # yourself only after changing Grid library source.
  if [ ! -f ${BUILD}/lib/libGrid.a ] || [ ! -f ${BUILD}/include/Grid/Grid.h ]; then
    echo "  install tree missing -> make install"
    cd ${BUILD}
    make install
    rc=$?
    if [ ${rc} -ne 0 ]; then
      echo "MAKE INSTALL FAILED (rc=${rc}); stopping."
      exit ${rc}
    fi
  else
    echo "  install tree present -> skipping make install (compiling the test only)"
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
  echo "+ compile Test_dwf_freeprec_claude.cc  (+ -I${SRC} for the new header)"
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${TEST} -o ${OBJ}
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

  echo "======== [4/4] RUN (gate 0 + gate 1 cold gate + chunk-3b headline on the config) ========"
  # 8^4 HEADLINE. For a 4^4 validation run instead, set:
  #   GRID=4.4.4.4 ; CFG=.../cfg_su3_4444_b6.0_claude.nersc  (then the 3b dense cross-check also runs)
  GRID=8.8.8.8
  CFG=/mnt/baracuda_14/dwms/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc
  ${BIN} --grid ${GRID} --mpi 1.1.1.1 --config ${CFG}
  rc=$?
  echo "run exit code = ${rc}"
} 2>&1 | tee ${LOG}
