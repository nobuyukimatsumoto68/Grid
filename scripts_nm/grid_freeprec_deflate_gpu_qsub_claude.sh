#!/bin/bash -l
# BU SCC SGE batch: DEFLATED GMRES (GMRES-DR) deflation-dim (k) sweep on ONE config, ONE frame, GPU.
# Validates chunk 2 of gmres_deflation_impl_plan_claude.md: at restart m=20, GMRES-DR(20,k) should
# converge in ~the no-restart iteration count (640 m=0.1 floor = 175 iters @ restart 256) once k exceeds
# the effective near-null count -- the restart-N scan put the plain-GMRES knee at window ~80, so expect
# the GMRES-DR knee around k ~ 40-80. --deflate-sweep runs each k BACK-TO-BACK on the same frame.
#
# Submit:  qsub -v CONFIG=/.../ckpoint_lat.640 grid_freeprec_deflate_gpu_qsub_claude.sh
#   override: qsub -v CONFIG=...,KSWEEP=8:16:24:40:64:96,MASSLIST=0.1:0.01,VARIANT=gmresdr ...
# Watch:   qstat -u $USER ; tail -f log/freeprec_deflate_<cfg>_j<jid>_claude.log

#$ -P qfe
##$ -M mtsmtnbyk@gmail.com
#$ -N freeprecDF
#$ -j y
#$ -o /projectnb/qfe/nmatsum/dwf/log/
##$ -m n
#$ -l h_rt=4:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -pe omp 4

set -u

module load cuda/12.8
module load gcc/13.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"
BIN=${ROOT}/build_merged/Test_dwf_freeprec_claude   # merged GPU tree (has GMRES-DR/GCRO-DR ops)
GRID=${GRID:-16.16.16.16}
CONFIG=${CONFIG:-${ROOT}/configs_iwasaki_16_b2.6/ckpoint_lat.640}
KSWEEP=${KSWEEP:-8:16:24:40:64:96}       # deflation dims, COLON-sep (SGE -v splits commas)
MASSLIST=${MASSLIST:-0.1:0.01}           # masses looped over the SINGLE frame, COLON-sep
VARIANT=${VARIANT:-gmresdr}              # gmresdr (deflated restart) or gcrodr (recycling)
RESTART=${RESTART:-20}                   # restart window m paired with deflation
KCOMMA=$(echo "${KSWEEP}" | tr ':' ',')
MLCOMMA=$(echo "${MASSLIST}" | tr ':' ',')
OUTLOG=${OUTLOG:-${LOGDIR}/freeprec_deflate_$(basename "${CONFIG}")_j${JOB_ID:-manual}_claude.log}

export OMP_NUM_THREADS=${NSLOTS:-4}

echo "=== freeprec deflate-scan $(date)  host ${HOSTNAME:-?}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?} ==="
nvidia-smi -L 2>/dev/null | head -1
echo "GRID=${GRID}  CONFIG=${CONFIG}"
echo "VARIANT=${VARIANT}  RESTART(m)=${RESTART}  KSWEEP=${KCOMMA}  MASSLIST=${MLCOMMA}  (frame s/t0=6, gf_maxit 4000 defaults)"
if [ ! -x "${BIN}" ]; then echo "ERROR: GPU freeprec binary missing ${BIN} -> run grid_freeprec_build_scc_gpu_merged_claude.sh"; exit 1; fi
if [ ! -f "${CONFIG}" ]; then echo "ERROR: config missing ${CONFIG}"; exit 1; fi

"${BIN}" --grid "${GRID}" --mpi 1.1.1.1 --config "${CONFIG}" \
         --mass-list "${MLCOMMA}" --ops "${VARIANT}" --restart "${RESTART}" --deflate-sweep "${KCOMMA}" \
         --accelerator-threads 8 2>&1 | tee "${OUTLOG}"
echo "freeprec deflate-scan exit = ${PIPESTATUS[0]}   $(date)"
echo "log -> ${OUTLOG}"
