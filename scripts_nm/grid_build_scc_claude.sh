#!/bin/bash
# SCC (BU Shared Computing Cluster) CUDA build of Grid for the free-limit Mobius preconditioner.
# SCC variant of build_grid_ubuntu.sh -- that script is left untouched for reference/A-B.
#
# Environment mimics ../qed3/env.sh (single source of truth for Nobu's SCC CUDA toolchain):
#     module load cuda/12.8      # nvcc; compiles device code on the login node (NO GPU needed to build)
#     module load gcc/13.2.0     # host compiler for nvcc -ccbin
# Dependencies (GMP, MPFR, FFTW single+double) are all present in the system prefix /usr on SCC, so
# no gmplib/mpfr/fftw modules are required -- we point configure at --with-...=/usr.
#
# Layout (per Nobu's request): the build/install tree lives at the working-tree ROOT, OUTSIDE the
# Grid git repo:
#     ROOT  = /projectnb/qfe/nmatsum/dwf          <- outside repo
#     SRC   = ROOT/Grid                            <- the git checkout (branch dwf_prec)
#     BUILD = ROOT/build                           <- configure --prefix; grid-config lands here
#
# Compile parallelism: this SCC login/interactive node reports nproc=1, so JOBS defaults to 1
# ("single thread to compile"). Grid is large; a single-threaded nvcc build is SLOW. To build faster,
# submit as a batch job with more slots, e.g.  qsub -P qfe -pe omp 8 ...  and run with JOBS=8.
#
# The user runs this one script; Claude reads the tee'd log grid_build_scc_claude.log.
# GPU RUN of the compiled binary goes to a batch node, e.g. (V100 = sm_70 = gpu_c 7.0):
#     qsub -P qfe -l gpus=1 -l gpu_c=7.0 -l h_rt=12:00:00 -pe omp 1 <run-script>

set -u

# ---- overridable configuration -------------------------------------------------------------------
ROOT=/projectnb/qfe/nmatsum/dwf
SRC=${ROOT}/Grid
BUILD=${ROOT}/build
LOG=${ROOT}/grid_build_scc_claude.log

# Multi-arch by default so the binary runs on ANY SCC GPU (SGE's gpu_c=N is a *floor*, so a job can
# land on a newer card than requested). Native SASS: 70=V100, 80=A100 (also covers 8.6=A40, 8.9=L40S
# since a cubin runs on higher minor revs of the same major), 90=H200; plus compute_90 PTX for anything
# newer. sm_70-only would die at the first device kernel on Ampere/Ada with "no kernel image available".
# Override e.g. GENCODE="-gencode arch=compute_80,code=sm_80" to trim compile time if you only use 8.x.
GENCODE=${GENCODE:-"-gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90"}
COMMS=${COMMS:-none}         # single-GPU R2 default. Set COMMS=mpi + load a CUDA-aware openmpi module for multi-GPU.
JOBS=${JOBS:-1}              # single thread to compile (this node has nproc=1)

# host compiler for nvcc: g++ (from gcc/13.2.0) for comms=none; mpic++ for comms=mpi
if [ "${COMMS}" = "none" ]; then
  HOSTCXX=g++
else
  HOSTCXX=mpic++
fi
# --------------------------------------------------------------------------------------------------

module load cuda/12.8
module load gcc/13.2.0

{
  echo "======== SCC Grid CUDA build  $(date) ========"
  echo "ROOT=${ROOT}  SRC=${SRC}  BUILD=${BUILD}"
  echo "GENCODE=${GENCODE}"
  echo "COMMS=${COMMS}  JOBS=${JOBS}  HOSTCXX=${HOSTCXX}"
  echo "nvcc: $(command -v nvcc)  ($(nvcc --version | tail -1))"
  echo "host g++: $(g++ --version | head -1)"

  echo "======== [0/3] bootstrap (Eigen + generate configure), only if missing ========"
  if [ ! -f "${SRC}/configure" ] || [ ! -e "${SRC}/Grid/Eigen" ]; then
    command -v wget      >/dev/null || { echo "ERROR: wget not found (bootstrap fetches Eigen)"; exit 1; }
    command -v autoreconf >/dev/null || { echo "ERROR: autoreconf not found (need autotools)"; exit 1; }
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

  echo "======== [1/3] configure (only if not already configured) ========"
  mkdir -p "${BUILD}"
  cd "${BUILD}"
  # Always (re)configure: autotools configure is idempotent, and re-running it in place is how we pick
  # up a flag change (e.g. adding -ldl) so the generated grid-config propagates it. Objects do not
  # depend on the Makefiles here, so an unchanged tree just relinks rather than fully rebuilds.
  # NOTE: nvcc-as-linker does NOT auto-add -ldl, but Grid's shm allocator + CUDA driver use
  # dlopen/dlclose -> without -ldl the benchmark/test links die with "DSO missing from command line".
  # It must be in configure LDFLAGS so grid-config --ldflags carries it to the freeprec test too.
  "${SRC}/configure" \
    --prefix="${BUILD}" \
    --with-gmp=/usr \
    --with-mpfr=/usr \
    --with-fftw=/usr \
    --enable-comms="${COMMS}" \
    --enable-simd=GPU \
    --enable-gen-simd-width=32 \
    --enable-accelerator=cuda \
    --enable-unified=no \
    --enable-openmp \
    --disable-gparity \
    --disable-fermion-reps \
    CXX=nvcc \
    LDFLAGS="-cudart shared -ldl -lrt" \
    CXXFLAGS="-ccbin ${HOSTCXX} ${GENCODE} -std=c++17 -cudart shared -Xcompiler -fPIC -Xcompiler -fopenmp"
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

  echo "======== [3/3] make install (populates ${BUILD}/{include,lib,bin/grid-config}) ========"
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
