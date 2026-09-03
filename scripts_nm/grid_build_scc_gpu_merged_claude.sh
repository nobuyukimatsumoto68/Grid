#!/bin/bash
# FULL rebuild of the GPU (CUDA multi-arch) Grid into a SEPARATE tree build_merged/ after the
# upstream/develop merge. Race-free: the live build/ (pre-merge GPU tree + binaries) is left UNTOUCHED.
# nvcc COMPILES on a login/compute node without a GPU (GPU only needed to RUN). Heavy compile.
#
# Exact flags from build/config.log (SCC GPU tree): --enable-simd=GPU --enable-gen-simd-width=32
#   --enable-accelerator=cuda --enable-comms=none --enable-unified=no, multi-arch sm_70/80/90 + PTX 90,
#   CXX=nvcc -ccbin g++, LDFLAGS "-cudart shared -ldl -lrt". Modules cuda/12.8 + gcc/13.2.0 (../qed3/env.sh).
#
# CRITICAL: bootstrap.sh MUST re-run (merge changed configure.ac + added instantiation files) and
# `make install` MUST run (refreshes build_merged/include). User runs this; Claude reads the log.
#   qsub -P qfe -pe omp 16 -l h_rt=8:00:00 -b y "bash grid_build_scc_gpu_merged_claude.sh"   (no GPU needed to compile)

set -u

module load cuda/12.8
module load gcc/13.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
SRC=${ROOT}/Grid
BUILD=${ROOT}/build_merged
JOBS=${JOBS:-8}
GENCODE=${GENCODE:-"-gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90"}
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"
LOG=${LOGDIR}/grid_build_scc_gpu_merged_claude.log

{
  echo "======== [0/3] bootstrap (FORCED -- merge changed configure.ac) $(date) ========"
  cd "${SRC}"
  ./bootstrap.sh
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "BOOTSTRAP FAILED (rc=${rc}); stopping."; exit ${rc}; fi

  echo "======== [1/3] configure into ${BUILD} (out-of-tree; live build/ untouched) ========"
  mkdir -p "${BUILD}"
  cd "${BUILD}"
  "${SRC}/configure" \
    --prefix="${BUILD}" \
    --with-gmp=/usr \
    --with-mpfr=/usr \
    --with-fftw=/usr \
    --enable-comms=none \
    --enable-simd=GPU \
    --enable-gen-simd-width=32 \
    --enable-accelerator=cuda \
    --enable-unified=no \
    --enable-openmp \
    --disable-gparity \
    --disable-fermion-reps \
    CXX=nvcc \
    LDFLAGS="-cudart shared -ldl -lrt" \
    CXXFLAGS="-ccbin g++ ${GENCODE} -std=c++17 -cudart shared -Xcompiler -fPIC -Xcompiler -fopenmp"
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "CONFIGURE FAILED (rc=${rc}); stopping."; exit ${rc}; fi

  # LIBRARY ONLY (-C Grid): the merge's benchmarks/Benchmark_allreduce.cc uses MPI_Datatype directly and
  # does NOT compile under --enable-comms=none (GPU). We never build benchmarks/tests/examples/HMC here --
  # the flowscan test compiles standalone against libGrid -- so build+install just the library subdir.
  echo "======== [2/3] make -j${JOBS} -C Grid (library only) ========"
  make -j"${JOBS}" -C Grid
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "MAKE FAILED (rc=${rc}); stopping."; exit ${rc}; fi

  echo "======== [3/3] make -C Grid install (refreshes ${BUILD}/include + lib) ========"
  make -C Grid install
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "MAKE INSTALL FAILED (rc=${rc}); stopping."; exit ${rc}; fi
  echo "  merged GPU tree ready at ${BUILD}   $(date)"
  echo "  next: grid_flowscan_build_scc_gpu_claude.sh (build the flowscan binary against it)"
} 2>&1 | tee ${LOG}
