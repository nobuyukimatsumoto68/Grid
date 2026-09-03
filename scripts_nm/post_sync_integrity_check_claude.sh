#!/bin/bash
# POST-SYNC INTEGRITY CHECK -- confirm the dwf_prec reconciliation merge (ab6d70df: local merged-upstream
# dwf_prec  <->  origin/SCC dwf_prec) left the merged Grid + our _claude code building and passing gates.
# The sync touched only _claude HEADERS/scripts (resolved via -I source, NOT part of libGrid) + the upstream
# 264-commit merge was already built (grid_freeprec_build_v2). So a full bootstrap/reconfigure is NOT needed;
# an incremental library make+install (safety, ~instant if unchanged) + two gates suffices. Correctness is
# load-independent, so a congested GPU is fine.
#   [1] F fp32 gate  : Test_dwf_freeprec_claude.cc default (fp32 + PlannedFFT + dense solve), 8^4 --ops m0
#       -> gate 0a/0b/1 PASS + [BT-0 gate] PASS + FGMRES 'Converged on iteration 79'.
#   [2] DofFFT gate  : Test_dof_fft_c0_claude.cc at L=8 and 16 -> roundtrip + PlannedFFT oracle ~eps.
# If [0] make fails on a NEW library file (unlikely -- sync was _claude only), run grid_freeprec_build_v2
# instead (full bootstrap+configure+make+install), then re-run this.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
LOG=${ROOT}/post_sync_integrity_check_claude.log
CFG=/mnt/baracuda_14/dwms/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc
JOBS=${JOBS:-8}

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

{
  echo "======== [0] incremental libGrid rebuild + install (safety; fast if unchanged) $(date) ========"
  cd ${BUILD}
  make -C Grid -j${JOBS}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "MAKE (Grid lib) FAILED (rc=${rc}) -- if a NEW library file, use grid_freeprec_build_v2_claude.sh"; exit ${rc}; fi
  make -C Grid install
  rc=$?; if [ ${rc} -ne 0 ]; then echo "MAKE INSTALL FAILED (rc=${rc})"; exit ${rc}; fi

  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"

  echo "======== [1] F fp32 gate: Test_dwf_freeprec_claude.cc (default), 8^4 --config --ops m0 ========"
  OBJ=${BUILD}/Test_dwf_freeprec_postsync_claude.o
  BIN=${BUILD}/Test_dwf_freeprec_postsync_claude
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${SRC}/tests/solver/Test_dwf_freeprec_claude.cc -o ${OBJ}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "F COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "F LINK FAILED (rc=${rc})"; exit ${rc}; fi
  ${BIN} --grid 8.8.8.8 --mpi 1.1.1.1 --config ${CFG} --ops m0

  echo "======== [2] DofFFT C0/C1 gate: Test_dof_fft_c0_claude.cc, L=8 and 16 ========"
  OBJD=${BUILD}/Test_dof_fft_c0_postsync_claude.o
  BIND=${BUILD}/Test_dof_fft_c0_postsync_claude
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${SRC}/tests/solver/Test_dof_fft_c0_claude.cc -o ${OBJD}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "DofFFT COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${OBJD} -o ${BIND} ${LDFLAGS} ${LIBS}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "DofFFT LINK FAILED (rc=${rc})"; exit ${rc}; fi
  ${BIND} --grid 8.8.8.8 --mpi 1.1.1.1
  ${BIND} --grid 16.16.16.16 --mpi 1.1.1.1

  echo "======== POST-SYNC INTEGRITY: done ========"
  echo "PASS if: [1] gate 0a/0b/1 PASS + [BT-0 gate] PASS + 'Converged on iteration 79';"
  echo "         [2] roundtrip + oracle ~eps PASS at BOTH L=8 and L=16."
  echo "done"
} 2>&1 | tee ${LOG}
