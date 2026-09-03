#!/bin/bash
# FULL rebuild of the CPU AVX+MPI Grid into a SEPARATE tree build_mpi_merged/ after the upstream/develop
# merge (264 commits). Race-free: the live build_mpi/ (and its already-built binaries for the 40 in-flight
# flowscan jobs + any queued campaign jobs) is left UNTOUCHED; only NEW work points at build_mpi_merged.
# (Local agent marlborough's guidance: install into a NEW tree, don't overwrite the live one.)
#
# Exact configure flags are those build_mpi was built with (from build_mpi/config.log):
#   --with-gmp=/usr --with-mpfr=/usr --with-fftw=/usr --enable-comms=mpi --enable-simd=AVX --enable-openmp
#   --disable-gparity --disable-fermion-reps  CXX=mpicxx MPICXX=mpicxx  CXXFLAGS="-std=c++17 -O3 -fopenmp"
#   LDFLAGS="-fopenmp -ldl -lrt"
#
# CRITICAL (merge gotcha): bootstrap.sh MUST re-run (merge changed configure.ac + added instantiation
# files -> regenerates configure + runs scripts/filelist), and `make install` MUST run (refreshes
# build_mpi_merged/include so the merged headers actually compile). Skipping either => stale pre-merge
# headers. User runs this (heavy compile); Claude reads the log. Consider a compute node / more cores:
#   qsub -P qfe -pe omp 16 -l h_rt=8:00:00 -b y "bash grid_build_scc_mpi_merged_claude.sh"
#   or interactive: JOBS=16 bash grid_build_scc_mpi_merged_claude.sh

set -u

module load gcc/12.2.0
module load openmpi/4.1.5_gnu-12.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
SRC=${ROOT}/Grid
BUILD=${ROOT}/build_mpi_merged
JOBS=${JOBS:-8}
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"
LOG=${LOGDIR}/grid_build_scc_mpi_merged_claude.log

{
  echo "======== [0/3] bootstrap (FORCED -- merge changed configure.ac + added files) $(date) ========"
  cd "${SRC}"
  ./bootstrap.sh
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "BOOTSTRAP FAILED (rc=${rc}); stopping."; exit ${rc}; fi

  echo "======== [1/3] configure into ${BUILD} (out-of-tree; live build_mpi untouched) ========"
  mkdir -p "${BUILD}"
  cd "${BUILD}"
  "${SRC}/configure" \
    --prefix="${BUILD}" \
    --with-gmp=/usr \
    --with-mpfr=/usr \
    --with-fftw=/usr \
    --enable-comms=mpi \
    --enable-simd=AVX \
    --enable-openmp \
    --disable-gparity \
    --disable-fermion-reps \
    CXX=mpicxx \
    MPICXX=mpicxx \
    CXXFLAGS="-std=c++17 -O3 -fopenmp" \
    LDFLAGS="-fopenmp -ldl -lrt"
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "CONFIGURE FAILED (rc=${rc}); stopping."; exit ${rc}; fi

  # LIBRARY ONLY (-C Grid): skip benchmarks/tests/examples/HMC (some upstream benchmarks from the merge
  # don't compile in every comms config; we don't need them -- the flowscan test builds standalone against
  # libGrid). Build + install just the library subdir.
  echo "======== [2/3] make -j${JOBS} -C Grid (library only) ========"
  make -j"${JOBS}" -C Grid
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "MAKE FAILED (rc=${rc}); stopping."; exit ${rc}; fi

  echo "======== [3/3] make -C Grid install (refreshes ${BUILD}/include + lib) ========"
  make -C Grid install
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "MAKE INSTALL FAILED (rc=${rc}); stopping."; exit ${rc}; fi
  echo "  merged CPU tree ready at ${BUILD}   $(date)"
  echo "  next: build the flowscan binary against it (grid_flowscan_build_scc_mpi_claude.sh)"
} 2>&1 | tee ${LOG}
