#!/bin/bash
# One-time CPU/MPI(FFTW) build of THIS repo's Grid (with the FreeMobius5D_claude.h edits + the FFT
# bench) into a SEPARATE tree build_mpi/, leaving the CUDA build/ untouched. Mirrors the proven
# ../grid_mpi/build_grid_ubuntu.sh configure line, changed only for this repo: --enable-Nc=3 (SU(3),
# not grid_mpi's Nc=4) and --enable-tracing dropped (our usecond() timers + FFT.h --log Performance
# need no tracing build). Purpose: measure the distributed-FFT comm on CPU cores (mpirun over the 20
# cores here); the Cshift "barrel-shift" ring (FFT.h:367-417) is backend-independent, so CPU multi-rank
# exercises the SAME transpose comm as GPU, with real MPI transport. Heavy (~full libGrid.a rebuild).
# User runs this ONCE; Claude reads the log. No rm here.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build_mpi
LOG=${ROOT}/grid_mpi_build_claude.log
MPICC=/mnt/hdd_barracuda/opt/openmpi_cuda/bin/mpicc
MPICXX=/mnt/hdd_barracuda/opt/openmpi_cuda/bin/mpic++
MPIROOT=/mnt/hdd_barracuda/opt/openmpi_cuda
HDF5=/mnt/hdd_barracuda/opt/myhdfstuff/hdf5-2.1.0

mkdir -p ${BUILD}

{
  echo "======== [1/3] CONFIGURE (CPU, mpi-auto, AVX/256, FFTW, Nc=3) $(date) ========"
  cd ${BUILD}
  ${SRC}/configure \
    --prefix=${BUILD}/ --with-lime= \
    MPICC="${MPICC}" \
    MPICXX="${MPICXX}" \
    CXXFLAGS="-std=c++17 -I/usr/local/include -I${MPIROOT}/include/ -I${HDF5}/include/ -fopenmp" \
    LDFLAGS="-L/usr/local/lib -L/opt/openssl/lib -L${MPIROOT}/lib/ -L${HDF5}/lib/ -lhdf5 -lhdf5_cpp" \
    --enable-unified=no \
    --enable-comms=mpi-auto \
    --enable-simd=AVX \
    --enable-gen-simd-width=256 \
    --enable-openmp \
    --enable-Nc=3 \
    --disable-gparity \
    --disable-fermion-reps
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "CONFIGURE FAILED (rc=${rc}); stopping."
    exit ${rc}
  fi

  echo "======== [2/3] MAKE (-j12) ========"
  make -j12
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "MAKE FAILED (rc=${rc}); stopping."
    exit ${rc}
  fi

  echo "======== [3/3] MAKE INSTALL (populates build_mpi/{include,lib,bin/grid-config}) ========"
  make install
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "MAKE INSTALL FAILED (rc=${rc}); stopping."
    exit ${rc}
  fi
  echo "MPI/CPU build OK. grid-config summary:"
  ${BUILD}/grid-config --summary 2>/dev/null | grep -iE "simd|comm|acce|fftw|Nc"
} 2>&1 | tee ${LOG}
