#!/bin/bash -l
# BU SCC SGE batch: 2D (restart m, deflation k) scan of GMRES-DR on ONE config, ONE frame, all masses.
# PHASE A of the deflation production plan: find a realistic FIXED (m,k) that gives a good D_W-count win
# across masses (m=0.1,0.01,0.001) on the HARD config 640, before looping it over the ensemble (Phase B,
# grid_freeprec_wrapper_claude.sh with OPS=gmresdr RESTART=m DEFLATEK=k).
# Each (m,k) pair runs BACK-TO-BACK on the same frame (--restart-sweep x --deflate-sweep). All at true 1e-8.
#
# Submit:  qsub -v CONFIG=/.../ckpoint_lat.640 grid_freeprec_2dscan_gpu_qsub_claude.sh
#   override grid: qsub -v CONFIG=...,MSWEEP=16:24:32,KSWEEP=16:24:32,MASSLIST=0.1:0.01:0.001 ...
# Watch:   qstat -u $USER ; tail -f log/freeprec_2dscan_<cfg>_j<jid>_claude.log

#$ -P qfe
##$ -M mtsmtnbyk@gmail.com
#$ -N freeprec2D
#$ -j y
#$ -o /projectnb/qfe/nmatsum/dwf/log/
##$ -m n
#$ -l h_rt=8:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -pe omp 4

set -u

module load cuda/12.8
module load gcc/13.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"
BIN=${ROOT}/build_merged/Test_dwf_freeprec_claude
GRID=${GRID:-16.16.16.16}
CONFIG=${CONFIG:-${ROOT}/configs_iwasaki_16_b2.6/ckpoint_lat.640}
MSWEEP=${MSWEEP:-16:24:32}               # restart windows m, COLON-sep
KSWEEP=${KSWEEP:-16:24:32}               # saved low modes k, COLON-sep
MASSLIST=${MASSLIST:-0.1:0.01:0.001}     # masses (the mass-scaling is the point), COLON-sep
MCOMMA=$(echo "${MSWEEP}"   | tr ':' ',')
KCOMMA=$(echo "${KSWEEP}"   | tr ':' ',')
MLCOMMA=$(echo "${MASSLIST}" | tr ':' ',')
OUTLOG=${OUTLOG:-${LOGDIR}/freeprec_2dscan_$(basename "${CONFIG}")_j${JOB_ID:-manual}_claude.log}

export OMP_NUM_THREADS=${NSLOTS:-4}

echo "=== freeprec 2D (m,k) scan $(date)  host ${HOSTNAME:-?}  CUDA=${CUDA_VISIBLE_DEVICES:-?} ==="
nvidia-smi -L 2>/dev/null | head -1
echo "GRID=${GRID}  CONFIG=${CONFIG}"
echo "MSWEEP=${MCOMMA}  KSWEEP=${KCOMMA}  MASSLIST=${MLCOMMA}  (frame s/t0=6, gf_maxit 4000 defaults; GMRES-DR)"
if [ ! -x "${BIN}" ]; then echo "ERROR: GPU binary missing ${BIN}"; exit 1; fi
if [ ! -f "${CONFIG}" ]; then echo "ERROR: config missing ${CONFIG}"; exit 1; fi

"${BIN}" --grid "${GRID}" --mpi 1.1.1.1 --config "${CONFIG}" \
         --mass-list "${MLCOMMA}" --ops gmresdr --restart-sweep "${MCOMMA}" --deflate-sweep "${KCOMMA}" \
         --accelerator-threads 8 2>&1 | tee "${OUTLOG}"
echo "freeprec 2D scan exit = ${PIPESTATUS[0]}   $(date)"
echo "log -> ${OUTLOG}"
