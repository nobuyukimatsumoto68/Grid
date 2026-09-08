#!/bin/bash -l
# BU SCC SGE batch: deflation experiment (|D_DW|^2 low modes as a spectrum transform) for ONE config on a
# GPU. GMRES vs CGNE, plain vs deflated. Runs the merged GPU binary directly (comms=none -> NO mpirun). 12h.
#
# Submit (Nobu submits; Claude never qsub/qdel):
#   qsub -v CONFIG=/projectnb/qfe/nmatsum/dwf/configs_iwasaki_16_b2.6/ckpoint_lat.640,K=3 grid_deflate_gpu_qsub_claude.sh
# Watch:  qstat -u $USER ; tail -f log/deflate_<cfg>_claude.log

#$ -P qfe
##$ -M mtsmtnbyk@gmail.com
#$ -N deflateG
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
export LD_LIBRARY_PATH=/share/pkg.8/cuda/12.8/install/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}

ROOT=/projectnb/qfe/nmatsum/dwf
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"
BIN=${BIN:-${ROOT}/build_merged/Test_dwf_deflate_claude}
GRID=${GRID:-16.16.16.16}
CONFIG=${CONFIG:?set CONFIG=<NERSC config>}
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
TAG=${TAG:-$(basename "${CONFIG}")}
OUTLOG=${OUTLOG:-${LOGDIR}/deflate_${TAG}_claude.log}

export OMP_NUM_THREADS=${NSLOTS:-8}

echo "=== deflateG $(date)  host ${HOSTNAME:-?}  GRID=${GRID}  CONFIG=${CONFIG}  K=${K}  TOL=${TOL} ==="
echo "    gmres_restart=${GMRES_RESTART}  cheb_lo=${CHEB_LO} cheb_ord=${CHEB_ORD}"
if [ ! -x "${BIN}" ]; then echo "ERROR: deflate GPU binary missing ${BIN}"; exit 1; fi
if [ ! -f "${CONFIG}" ]; then echo "ERROR: config missing ${CONFIG}"; exit 1; fi

"${BIN}" --grid "${GRID}" --mpi 1.1.1.1 --accelerator-threads 8 \
         --config "${CONFIG}" --k "${K}" --tol "${TOL}" \
         --cg_maxit "${CG_MAXIT}" --gmres_maxit "${GMRES_MAXIT}" --gmres_restart "${GMRES_RESTART}" \
         --cheb_lo "${CHEB_LO}" --cheb_hi "${CHEB_HI}" --cheb_ord "${CHEB_ORD}" \
         --sv_resid "${SV_RESID}" --sv_maxit "${SV_MAXIT}" \
         --nstop "${NSTOP}" --nk "${NK}" --nm "${NM}" 2>&1 | tee "${OUTLOG}"
echo "deflateG exit = ${PIPESTATUS[0]}   $(date)"
echo "log -> ${OUTLOG}"
