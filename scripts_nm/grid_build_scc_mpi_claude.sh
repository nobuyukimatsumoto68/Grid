#!/bin/bash
# SCC CPU + MPI build of Grid -- runs on the CPU interactive/login nodes (no GPU) for interactive
# quenched generation + flowed-Q tuning. Separate from the CUDA build: installs into build_mpi/ so the
# GPU build/ (grid-config etc.) is untouched and both coexist.
#
# SIMD = AVX: verified best for the SCC interactive node CPU (Intel Xeon E5-2650 v2, Ivy Bridge --
# has avx + sse4_2 but NO avx2/fma/avx512, so AVX is the highest Grid --enable-simd it supports;
# AVXFMA/AVX2 need Haswell+, AVX512 needs Skylake-X, either would fault here). Override with SIMD=... on
# a newer node (e.g. SIMD=AVX2 on Haswell+, SIMD=SKL on Skylake-X) -- but AVX runs on every SCC x86 node.
#
# Toolchain: matched gcc/12.2.0 + openmpi/4.1.5_gnu-12.2.0 (mpicxx -> g++ 12.2.0). Deps from /usr.
# The heavy 24^4 FGMRES SOLVE still wants the GPU build; this CPU build is for gen + flow-Q + tuning.
# Output tee'd to grid_build_scc_mpi_claude.log.

set -u

ROOT=/projectnb/qfe/nmatsum/dwf
SRC=${ROOT}/Grid
BUILD=${ROOT}/build_mpi
LOG=${ROOT}/grid_build_scc_mpi_claude.log

SIMD=${SIMD:-AVX}
JOBS=${JOBS:-8}              # this interactive node has 8 slots -> parallel CPU compile

module load gcc/12.2.0
module load openmpi/4.1.5_gnu-12.2.0

{
  echo "======== SCC Grid CPU+MPI build  $(date) ========"
  echo "ROOT=${ROOT}  SRC=${SRC}  BUILD=${BUILD}  SIMD=${SIMD}  JOBS=${JOBS}"
  echo "mpicxx: $(command -v mpicxx)"
  mpicxx -show 2>/dev/null | head -1

  echo "======== [0/3] bootstrap (only if missing -- shared with the GPU source tree) ========"
  if [ ! -f "${SRC}/configure" ] || [ ! -e "${SRC}/Grid/Eigen" ]; then
    cd "${SRC}"
    ./bootstrap.sh
    rc=$?
    if [ ${rc} -ne 0 ]; then
      echo "BOOTSTRAP FAILED (rc=${rc}); stopping."
      exit ${rc}
    fi
  else
    echo "  configure + Eigen present -> skipping bootstrap"
  fi

  echo "======== [1/3] configure (always, in build_mpi) ========"
  mkdir -p "${BUILD}"
  cd "${BUILD}"
  "${SRC}/configure" \
    --prefix="${BUILD}" \
    --with-gmp=/usr \
    --with-mpfr=/usr \
    --with-fftw=/usr \
    --enable-comms=mpi \
    --enable-simd="${SIMD}" \
    --enable-openmp \
    --disable-gparity \
    --disable-fermion-reps \
    CXX=mpicxx \
    MPICXX=mpicxx \
    CXXFLAGS="-std=c++17 -O3 -fopenmp" \
    LDFLAGS="-fopenmp -ldl -lrt"
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "CONFIGURE FAILED (rc=${rc}); stopping. See config.log."
    exit ${rc}
  fi

  echo "======== [2/3] make -j${JOBS} ========"
  make -j"${JOBS}"
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "MAKE FAILED (rc=${rc}); stopping."
    exit ${rc}
  fi

  echo "======== [3/3] make install ========"
  make install
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "MAKE INSTALL FAILED (rc=${rc}); stopping."
    exit ${rc}
  fi

  echo "======== DONE  $(date) ========"
  echo "grid-config -> ${BUILD}/bin/grid-config"
  ls -la "${BUILD}/bin/grid-config" 2>/dev/null || echo "  (grid-config not found -- check log)"
} 2>&1 | tee "${LOG}"
