#!/bin/bash -l
# BU SCC SGE batch: spectrum-transform diagnostic (Direction 2) for ONE config on a GPU. Batch form of
# run_spectrum_claude.sh -- lets the SLOW M0^{-1} D_DW shift-invert (GMRES inner solve at sigma~0) run to
# completion unattended (12h) instead of on a qrsh node. Runs the merged GPU binary directly (comms=none
# -> NO mpirun). Every knob is overridable via -v (SGE splits -v values on commas, but none of these
# values contain commas -- grids use dots, WHICH is a single token -- so no hyphen-encoding needed).
#
# Submit (Nobu submits; Claude never qsub/qdel):
#   qsub -v CONFIG=/projectnb/qfe/nmatsum/dwf/configs_iwasaki_16_b2.6/ckpoint_lat.640,WHICH=m0d grid_spectrum_gpu_qsub_claude.sh
#   # sv (fast, singular values + chirality):  qsub -v CONFIG=...,WHICH=sv grid_spectrum_gpu_qsub_claude.sh
#   # D_DW shift-invert only:                   qsub -v CONFIG=...,WHICH=d  grid_spectrum_gpu_qsub_claude.sh
# Watch:  qstat -u $USER ; tail -f log/spectrum_<cfg>_claude.log

#$ -P qfe
##$ -M mtsmtnbyk@gmail.com
#$ -N spectrumG
#$ -j y
#$ -o /projectnb/qfe/nmatsum/dwf/log/
##$ -m n
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -pe omp 8

set -u

module load cuda/12.8
module load gcc/13.2.0
# interactive/compute nodes: cuda module adds .../install/lib64 but libcublas.so.12 lives in targets/.../lib
export LD_LIBRARY_PATH=/share/pkg.8/cuda/12.8/install/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}

ROOT=/projectnb/qfe/nmatsum/dwf
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"
BIN=${BIN:-${ROOT}/build_merged/Test_dwf_spectrum_transform_claude}
GRID=${GRID:-16.16.16.16}
CONFIG=${CONFIG:?set CONFIG=<NERSC config>}
# WHICH: m0d (slow M0^{-1} D_DW shift-invert -- the reason to batch) | sv | d | both. Default m0d.
WHICH=${WHICH:-m0d}
NSTOP=${NSTOP:-12}
NK=${NK:-24}
NM=${NM:-48}
ERESID=${ERESID:-1e-2}
MAXIT=${MAXIT:-200}
GFMAXIT=${GFMAXIT:-3000}
FLOWNSTEP=${FLOWNSTEP:-146}   # frame s/t0=1.0 (tau=t0=2.91 at eps=0.02; flowed plaq ~0.9990); was 100 (s/t0=0.69). frame only for m0d/both
SIGMAD=${SIGMAD:--1.8}
SIGMAM0D=${SIGMAM0D:-0.0}
INNER_TOL=${INNER_TOL:-1e-5}
INNER_MAXIT=${INNER_MAXIT:-4000}
INNER_RESTART=${INNER_RESTART:-50}
CHEB_LO=${CHEB_LO:-0.5}
CHEB_HI=${CHEB_HI:--1.0}       # <=0 -> auto (power-method lambda_max)
CHEB_ORD=${CHEB_ORD:-21}       # forced odd by the binary
SV_RESID=${SV_RESID:-1e-5}
SV_MAXIT=${SV_MAXIT:-200}
PLAIN=${PLAIN:-0}
PLAINFLAG=""
if [ "${PLAIN}" = "1" ]; then PLAINFLAG="--plain"; fi
TAG=${TAG:-$(basename "${CONFIG}")}
# Per-submission UNIQUE log (never overwrite a prior run's log -- Nobu 2026-09-07). JOB_ID is set by SGE.
OUTLOG=${OUTLOG:-${LOGDIR}/spectrum_${TAG}_j${JOB_ID:-manual}_claude.log}

export OMP_NUM_THREADS=${NSLOTS:-8}

echo "=== spectrumG $(date)  host ${HOSTNAME:-?}  GRID=${GRID}  CONFIG=${CONFIG}  WHICH=${WHICH} ==="
echo "    Nstop=${NSTOP} Nk=${NK} Nm=${NM} flow_nstep=${FLOWNSTEP} sigma_m0d=${SIGMAM0D} cheb_lo=${CHEB_LO} cheb_ord=${CHEB_ORD}"
if [ ! -x "${BIN}" ]; then echo "ERROR: spectrum GPU binary missing ${BIN}"; exit 1; fi
if [ ! -f "${CONFIG}" ]; then echo "ERROR: config missing ${CONFIG}"; exit 1; fi

# comms=none GPU build -> run the binary directly (no mpirun).
"${BIN}" --grid "${GRID}" --mpi 1.1.1.1 --accelerator-threads 8 \
         --config "${CONFIG}" --which "${WHICH}" --nstop "${NSTOP}" --nk "${NK}" --nm "${NM}" \
         --eresid "${ERESID}" --maxit "${MAXIT}" --gf_maxit "${GFMAXIT}" --flow_nstep "${FLOWNSTEP}" \
         --sigma_d "${SIGMAD}" --sigma_m0d "${SIGMAM0D}" --inner_tol "${INNER_TOL}" --inner_maxit "${INNER_MAXIT}" \
         --inner_restart "${INNER_RESTART}" \
         --cheb_lo "${CHEB_LO}" --cheb_hi "${CHEB_HI}" --cheb_ord "${CHEB_ORD}" \
         --sv_resid "${SV_RESID}" --sv_maxit "${SV_MAXIT}" ${PLAINFLAG} 2>&1 | tee "${OUTLOG}"
echo "spectrumG exit = ${PIPESTATUS[0]}   $(date)"
echo "log -> ${OUTLOG}"
