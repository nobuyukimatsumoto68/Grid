#!/bin/bash
# Interactive-GPU-node run helper for the deflation experiment (|D_DW|^2 low modes as a spectrum
# transform; GMRES vs CGNE, plain vs deflated). Run it on a qrsh GPU node (or any node with a visible
# GPU). Not a batch script -- for SGE use grid_deflate_gpu_qsub_claude.sh. Override any knob via env:
#   CONFIG=... K=12 TOL=1e-8 bash run_deflate_claude.sh
set -u

module load cuda/12.8
module load gcc/13.2.0
export LD_LIBRARY_PATH=/share/pkg.8/cuda/12.8/install/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}

ROOT=/projectnb/qfe/nmatsum/dwf
BIN=${ROOT}/build_merged/Test_dwf_deflate_claude
CONFIG=${CONFIG:-${ROOT}/configs_iwasaki_16_b2.6/ckpoint_lat.640}
K=${K:-3}
TOL=${TOL:-1e-8}
CG_MAXIT=${CG_MAXIT:-20000}
GMRES_MAXIT=${GMRES_MAXIT:-20000}
GMRES_RESTART=${GMRES_RESTART:-50}
CHEB_LO=${CHEB_LO:-0.5}
CHEB_HI=${CHEB_HI:--1.0}      # <=0 -> auto (power-method lambda_max)
CHEB_ORD=${CHEB_ORD:-21}      # forced odd by the binary
SV_RESID=${SV_RESID:-1e-5}
SV_MAXIT=${SV_MAXIT:-200}
NSTOP=${NSTOP:-12}
NK=${NK:-24}
NM=${NM:-48}
LOG=${ROOT}/log/deflate_$(basename "${CONFIG}")_claude.log
mkdir -p "${ROOT}/log"

if [ ! -x "${BIN}" ]; then echo "ERROR: deflate binary missing ${BIN} (build first: grid_deflate_build_scc_gpu_claude.sh)"; exit 1; fi

echo "=== deflate $(date)  CONFIG=${CONFIG}  K=${K}  TOL=${TOL}  gmres_restart=${GMRES_RESTART} ==="
"${BIN}" --grid 16.16.16.16 --mpi 1.1.1.1 --accelerator-threads 8 \
  --config "${CONFIG}" --k "${K}" --tol "${TOL}" \
  --cg_maxit "${CG_MAXIT}" --gmres_maxit "${GMRES_MAXIT}" --gmres_restart "${GMRES_RESTART}" \
  --cheb_lo "${CHEB_LO}" --cheb_hi "${CHEB_HI}" --cheb_ord "${CHEB_ORD}" \
  --sv_resid "${SV_RESID}" --sv_maxit "${SV_MAXIT}" \
  --nstop "${NSTOP}" --nk "${NK}" --nm "${NM}" 2>&1 | tee "${LOG}"
echo "log -> ${LOG}"
