#!/bin/bash
# One-time CPU/FFTW build of Grid into a SEPARATE tree (build_cpu/), leaving the CUDA build/ untouched.
# Purpose: measure the distributed-FFT comm on CPU cores via mpirun (the Cshift "barrel-shift" ring in
# FFT.h:367-417 is backend-independent, so CPU multi-rank exercises the SAME transpose comm as GPU, with
# real MPI transport -- no multi-GPU needed). Mirrors the original CUDA configure line MINUS the CUDA
# accelerator options, PLUS --enable-simd=AVX512 and CXX=mpic++ (g++-backed, not nvcc).
#
# Heavy: full libGrid.a rebuild (~20-40 min). User runs this ONCE; Claude reads the log. No rm here.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid
BUILD=${ROOT}/build_cpu
LOG=${ROOT}/grid_cpu_build_claude.log
MPICXX=/mnt/hdd_barracuda/opt/openmpi_cuda/bin/mpic++
MPICC=/mnt/hdd_barracuda/opt/openmpi_cuda/bin/mpicc
HDF5=/mnt/hdd_barracuda/opt/myhdfstuff/hdf5-2.1.0
MPIROOT=/mnt/hdd_barracuda/opt/openmpi_cuda

mkdir -p ${BUILD}

{
  echo "======== [1/3] CONFIGURE (CPU/AVX512, FFTW, mpi3) $(date) ========"
  cd ${BUILD}
  ${SRC}/configure \
    --prefix=${BUILD}/ \
    --with-lime= \
    CXX=${MPICXX} \
    MPICC=${MPICC} \
    MPICXX=${MPICXX} \
    "CXXFLAGS=-std=c++17 -O3 -fopenmp -I/usr/local/include -I${MPIROOT}/include -I${HDF5}/include/" \
    "LDFLAGS=-L/usr/local/lib -L${MPIROOT}/lib/ -L${HDF5}/lib/ -lhdf5 -lhdf5_cpp" \
    --enable-comms=mpi \
    --enable-shm=no \
    --enable-tracing=timer \
    --enable-simd=AVX512 \
    --enable-openmp \
    --disable-gparity \
    --disable-fermion-reps
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "CONFIGURE FAILED (rc=${rc}); stopping."
    exit ${rc}
  fi

  echo "======== [2/3] MAKE (-j20) ========"
  make -j20
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "MAKE FAILED (rc=${rc}); stopping."
    exit ${rc}
  fi

  echo "======== [3/3] MAKE INSTALL (populates build_cpu/{include,lib,bin/grid-config}) ========"
  make install
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "MAKE INSTALL FAILED (rc=${rc}); stopping."
    exit ${rc}
  fi
  echo "CPU build OK. grid-config:"
  ${BUILD}/grid-config --summary 2>/dev/null | grep -iE "simd|comm|acce|fftw"
} 2>&1 | tee ${LOG}
