#!/bin/bash
# Build + run the cuFFTDx-vs-cuFFT length-8 throughput gate (cufftdx_bench_claude.cu) on gpu0.
# Decision: ratio (cuFFTDx/cuFFT) < 1 => cuFFTDx faster for our size -> fused-apply worth building.

set -u

ROOT=/mnt/baracuda_14/dwms
SRC=${ROOT}/Grid/scripts_nm/cufftdx_bench_claude.cu
BIN=${ROOT}/build/cufftdx_bench_claude
LOG=${ROOT}/cufftdx_bench_claude.log
NVCC=/usr/local/cuda-12.6/bin/nvcc
MD=/mnt/hdd_barracuda/opt/nvidia-mathdx-24.08.0/nvidia/mathdx/24.08

export CUDA_VISIBLE_DEVICES=0

{
  echo "======== compile (nvcc, sm_70, cuFFTDx + cutlass) $(date) ========"
  ${NVCC} -std=c++17 -O3 -arch=sm_70 \
    -I"${MD}/include" -I"${MD}/external/cutlass/include" \
    "${SRC}" -o "${BIN}" -lcufft
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "COMPILE FAILED (rc=${rc}); stopping."
    exit ${rc}
  fi
  echo "======== run (gpu0) ========"
  "${BIN}"
  echo "exit = $?"
} 2>&1 | tee ${LOG}
