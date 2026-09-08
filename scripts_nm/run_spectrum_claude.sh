#!/bin/bash
# Interactive-GPU-node run helper for the spectrum-transform diagnostic (Direction 2). Sets the modules +
# the cuBLAS lib path (interactive nodes don't have it by default), runs the binary, tees to log/. Not a
# batch script -- run it on a qrsh GPU node. Override any knob via env, e.g.
#   NM=96 MAXIT=400 WHICH=d bash run_spectrum_claude.sh
set -u

module load cuda/12.8
module load gcc/13.2.0
export LD_LIBRARY_PATH=/share/pkg.8/cuda/12.8/install/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}

ROOT=/projectnb/qfe/nmatsum/dwf
BIN=${ROOT}/build_merged/Test_dwf_spectrum_transform_claude
CONFIG=${CONFIG:-${ROOT}/configs_iwasaki_16_b2.6/ckpoint_lat.640}
WHICH=${WHICH:-both}
NSTOP=${NSTOP:-12}
NK=${NK:-24}
NM=${NM:-48}
ERESID=${ERESID:-1e-2}
MAXIT=${MAXIT:-200}
GFMAXIT=${GFMAXIT:-3000}
FLOWNSTEP=${FLOWNSTEP:-146}   # frame s/t0=1.0 (tau=t0=2.91 at eps=0.02; flowed plaq ~0.9990) -- was 58 (s/t0=0.4)
# shift-invert (default): D_DW near sigma_d=-1.8 (physical Wilson branch), M0 D_DW near sigma_m0d=0
# (near-zero stragglers + negative-real check). PLAIN=1 -> old smallest-modulus Arnoldi.
SIGMAD=${SIGMAD:--1.8}
SIGMAM0D=${SIGMAM0D:-0.0}
INNER_TOL=${INNER_TOL:-1e-5}
INNER_MAXIT=${INNER_MAXIT:-4000}   # CGNE at sigma near the spectrum is ill-conditioned -> needs many iters
INNER_RESTART=${INNER_RESTART:-50}  # GMRES restart length (M0 D_DW inner solve)
# WHICH=sv : Lanczos on |D_DW|^2 -> singular values + chirality (index) in 5D.
#   CHEB_LO = cut: modes with lambda<CHEB_LO extracted -> keep ABOVE the near-zero cluster (~m^2=0.01).
#   CHEB_HI <= 0 -> auto (power-method lambda_max). CHEB_ORD forced ODD by the binary. Reuses NSTOP/NK/NM.
CHEB_LO=${CHEB_LO:-0.5}
CHEB_HI=${CHEB_HI:--1.0}
CHEB_ORD=${CHEB_ORD:-21}
SV_RESID=${SV_RESID:-1e-5}
SV_MAXIT=${SV_MAXIT:-200}
PLAIN=${PLAIN:-0}
PLAINFLAG=""
if [ "${PLAIN}" = "1" ]; then PLAINFLAG="--plain"; fi
LOG=${ROOT}/log/spectrum_$(basename "${CONFIG}")_claude.log
mkdir -p "${ROOT}/log"

if [ ! -x "${BIN}" ]; then echo "ERROR: spectrum binary missing ${BIN} (build first)"; exit 1; fi

echo "=== spectrum $(date)  CONFIG=${CONFIG}  WHICH=${WHICH}  Nstop=${NSTOP} Nk=${NK} Nm=${NM} eresid=${ERESID} maxit=${MAXIT} gf_maxit=${GFMAXIT} flow_nstep=${FLOWNSTEP} ==="
"${BIN}" --grid 16.16.16.16 --mpi 1.1.1.1 --accelerator-threads 8 \
  --config "${CONFIG}" --which "${WHICH}" --nstop "${NSTOP}" --nk "${NK}" --nm "${NM}" \
  --eresid "${ERESID}" --maxit "${MAXIT}" --gf_maxit "${GFMAXIT}" --flow_nstep "${FLOWNSTEP}" \
  --sigma_d "${SIGMAD}" --sigma_m0d "${SIGMAM0D}" --inner_tol "${INNER_TOL}" --inner_maxit "${INNER_MAXIT}" \
  --inner_restart "${INNER_RESTART}" \
  --cheb_lo "${CHEB_LO}" --cheb_hi "${CHEB_HI}" --cheb_ord "${CHEB_ORD}" \
  --sv_resid "${SV_RESID}" --sv_maxit "${SV_MAXIT}" ${PLAINFLAG} 2>&1 | tee "${LOG}"
echo "log -> ${LOG}"
