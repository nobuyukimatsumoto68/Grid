#!/bin/bash
# grid_freeprec_build_v2 -- FULL reproducible Grid rebuild + reinstall, then build+run the free-prec test.
#
# WHY v2 (vs grid_freeprec_build_claude.sh): the v1 script SKIPS `make install` when build/include exists,
# so after the upstream merge (paboyle/Grid, 264 commits: new FFT.h/PlannedFFT, configure.ac + new files)
# the compile still saw the STALE pre-merge headers in build/include. v2 does the whole chain so the merged
# source actually reaches the compiler:
#   [0] bootstrap.sh   (configure.ac changed + new .cc/.h -> regenerates configure, Make.inc via scripts/filelist)
#   [1] configure      (EXACT flags from the current build/config.status -- reproducible)
#   [2] make           (rebuild libGrid from the merged source)
#   [3] make install   (REFRESH build/{include,lib} -- the step v1 skipped; grid-config points here)
#   [4] compile+link+run the free-prec test against the now-merged install tree
#
# User runs this; Claude reads grid_freeprec_build_v2_claude.log. Big build (merge touched core headers) --
# expect a long compile. nvcc is memory-heavy: lower GRID_BUILD_JOBS if the box OOMs.
# NOTE: if an incremental `make` misbehaves after such a large merge, run `make clean` in build/ YOURSELF
# (this script does not, per the no-destructive-commands rule) and re-run.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
GC=${BUILD}/grid-config
TEST=${SRC}/tests/solver/Test_dwf_freeprec_claude.cc
OBJ=${BUILD}/Test_dwf_freeprec_v2_claude.o
BIN=${BUILD}/Test_dwf_freeprec_v2_claude
LOG=${ROOT}/grid_freeprec_build_v2_claude.log
JOBS=${GRID_BUILD_JOBS:-16}

# ---- run config (8^4 headline). For a 4^4 validation: GRID=4.4.4.4 + the 4444 nersc config. ----
GRID=8.8.8.8
CFG=/mnt/baracuda_14/dwms/dwf4_qcd_claude/cfg_su3_8888_b6.0_claude.nersc

{
  echo "======== [0/5] bootstrap (regenerate configure + Make.inc after the merge) $(date) ========"
  cd ${SRC}
  ./bootstrap.sh
  rc=$?; if [ ${rc} -ne 0 ]; then echo "BOOTSTRAP FAILED (rc=${rc})"; exit ${rc}; fi

  echo "======== [1/5] configure (EXACT flags from build/config.status) ========"
  cd ${BUILD}
  ${SRC}/configure \
    --prefix=${BUILD}/ \
    --with-lime= \
    MPICC=/mnt/hdd_barracuda/opt/openmpi_cuda/bin/mpicc \
    MPICXX=/mnt/hdd_barracuda/opt/openmpi_cuda/bin/mpic++ \
    CXXFLAGS="-std=c++17 -I/usr/local/include -I/mnt/hdd_barracuda/opt/openmpi_cuda/include -I/mnt/hdd_barracuda/opt/myhdfstuff/hdf5-2.1.0/include/ -ccbin /mnt/hdd_barracuda/opt/openmpi_cuda/bin/mpic++ -gencode arch=compute_70,code=sm_70 -Xcompiler -fPIC -Xcompiler -fopenmp" \
    LDFLAGS="-L/usr/local/lib -L/opt/openssl/lib -L/mnt/hdd_barracuda/opt/openmpi_cuda/lib/ -L/mnt/hdd_barracuda/opt/myhdfstuff/hdf5-2.1.0/lib/ -lhdf5 -lhdf5_cpp" \
    --enable-comms=mpi \
    --enable-unified=no \
    --enable-shm=no \
    --enable-tracing=timer \
    --enable-accelerator=cuda \
    --enable-gen-simd-width=32 \
    --enable-simd=GPU \
    --enable-accelerator-aware-mpi \
    --enable-openmp \
    --disable-gparity \
    --disable-fermion-reps \
    --enable-accelerator-cshift \
    CXX=/usr/local/cuda-12.6/bin//nvcc \
    CC=/mnt/hdd_barracuda/opt/openmpi_cuda/bin/mpicc
  rc=$?; if [ ${rc} -ne 0 ]; then echo "CONFIGURE FAILED (rc=${rc})"; exit ${rc}; fi

  echo "======== [2/5] make -j${JOBS} (rebuild libGrid from merged source) ========"
  cd ${BUILD}
  make -j${JOBS}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "MAKE FAILED (rc=${rc})"; exit ${rc}; fi

  echo "======== [3/5] make install (REFRESH build/include + build/lib with the merged headers) ========"
  cd ${BUILD}
  make install
  rc=$?; if [ ${rc} -ne 0 ]; then echo "MAKE INSTALL FAILED (rc=${rc})"; exit ${rc}; fi

  echo "======== [4/5] compile + link the free-prec test against the merged install tree ========"
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  ${CXX} ${CXXFLAGS} -I${SRC} -c ${TEST} -o ${OBJ}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "COMPILE FAILED (rc=${rc})"; exit ${rc}; fi
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?; if [ ${rc} -ne 0 ]; then echo "LINK FAILED (rc=${rc})"; exit ${rc}; fi

  echo "======== [5/5] run (gate 0 + cold gate + headline) on the merged Grid ========"
  ${BIN} --grid ${GRID} --mpi 1.1.1.1 --config ${CFG} --ops m0
  rc=$?; if [ ${rc} -ne 0 ]; then echo "RUN FAILED (rc=${rc})"; exit ${rc}; fi
  echo "done"
} 2>&1 | tee ${LOG}
