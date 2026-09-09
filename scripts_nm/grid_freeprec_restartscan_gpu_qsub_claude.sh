#!/bin/bash -l
# BU SCC SGE batch: FGMRES(M0) RESTART-LENGTH scan on ONE config, ONE converged frame, on a GPU.
# Purpose: confirm the 20000-iter cap seen in the config scan is restarted-GMRES STAGNATION (the isolated
# near-null cluster of M0 D_DW), NOT a bug -- and pin the restart length at which M0 recovers convergence.
# Uses the binary's built-in --restart-sweep: each restart length runs BACK-TO-BACK on the SAME frame +
# GPU load, so cross-run drift is removed and the restart-vs-iters curve is clean.
#
# Memory note: FGMRES Krylov dim = restart length; at 16^4 Ls8 restart 256 ~ 26 GB -> request gpu_c=8.0
# (Ampere+, >=40 GB) so restart 256 fits (avoids a 16 GB V100).
#
# Submit:  qsub -v CONFIG=/.../ckpoint_lat.640 grid_freeprec_restartscan_gpu_qsub_claude.sh
#   override sweep/masses: qsub -v CONFIG=...,RESTARTS=20:40:80:128:256,MASSLIST=0.1:0.01 ...
# Watch:   qstat -u $USER ; tail -f log/freeprec_restartscan_<cfg>_j<jid>_claude.log

#$ -P qfe
##$ -M mtsmtnbyk@gmail.com
#$ -N freeprecRS
#$ -j y
#$ -o /projectnb/qfe/nmatsum/dwf/log/
##$ -m n
#$ -l h_rt=6:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -pe omp 4

set -u

module load cuda/12.8
module load gcc/13.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"
BIN=${ROOT}/build_merged/Test_dwf_freeprec_claude   # merged GPU tree (has --restart-sweep + --mass-list)
GRID=${GRID:-16.16.16.16}
CONFIG=${CONFIG:-${ROOT}/configs_iwasaki_16_b2.6/ckpoint_lat.640}   # hard |Q|=3 default reference
RESTARTS=${RESTARTS:-20:40:80:128:256}     # restart lengths, COLON-sep (SGE -v splits commas)
MASSLIST=${MASSLIST:-0.1:0.01}             # DWF masses looped over the SINGLE frame, COLON-sep
RSCOMMA=$(echo "${RESTARTS}" | tr ':' ',') # --restart-sweep wants commas
MLCOMMA=$(echo "${MASSLIST}" | tr ':' ',') # --mass-list wants commas
OPS=${OPS:-cgne,m0}                         # RB-CGNE baseline + M0 sweep; skip M1 to keep it fast
OUTLOG=${OUTLOG:-${LOGDIR}/freeprec_restartscan_$(basename "${CONFIG}")_j${JOB_ID:-manual}_claude.log}

export OMP_NUM_THREADS=${NSLOTS:-4}

echo "=== freeprec restart-scan $(date)  host ${HOSTNAME:-?}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?} ==="
nvidia-smi -L 2>/dev/null | head -1
echo "GRID=${GRID}  CONFIG=${CONFIG}"
echo "RESTARTS=${RSCOMMA}  MASSLIST=${MLCOMMA}  OPS=${OPS}  (frame s/t0=6, gf_maxit 4000 = binary defaults)"
if [ ! -x "${BIN}" ]; then echo "ERROR: GPU freeprec binary missing ${BIN}"; exit 1; fi
if [ ! -f "${CONFIG}" ]; then echo "ERROR: config missing ${CONFIG}"; exit 1; fi

# comms=none CUDA binary: run directly (no mpirun), one GPU.
"${BIN}" --grid "${GRID}" --mpi 1.1.1.1 --config "${CONFIG}" \
         --mass-list "${MLCOMMA}" --restart-sweep "${RSCOMMA}" --ops "${OPS}" \
         --accelerator-threads 8 2>&1 | tee "${OUTLOG}"
echo "freeprec restart-scan exit = ${PIPESTATUS[0]}   $(date)"
echo "log -> ${OUTLOG}"
