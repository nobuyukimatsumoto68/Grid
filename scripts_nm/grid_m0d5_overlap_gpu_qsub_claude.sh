#!/bin/bash -l
# BU SCC SGE batch: low modes of M0 D5 by shift-invert IRA + overlap with the |D5|^2 (deflfull) low modes.
# Test_dwf_m0d5_overlap_claude.cc. Reads deflfull from --ckpt_dir (must already be dumped, e.g. 48 modes).
# Submit: qsub -v CONFIG=.../ckpoint_lat.640,NSTOP=24 grid_m0d5_overlap_gpu_qsub_claude.sh

#$ -P qfe
#$ -N m0d5ovlp
#$ -j y
#$ -o /projectnb/qfe/nmatsum/dwf/log/
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
BIN=${ROOT}/build_merged/Test_dwf_m0d5_overlap_claude
GRID=${GRID:-16.16.16.16}
CONFIG=${CONFIG:-${ROOT}/configs_iwasaki_16_b2.6/ckpoint_lat.640}
CKPT_DIR=${CKPT_DIR:-${LOGDIR}/deflvec_$(basename "${CONFIG}")}
NSTOP=${NSTOP:-24}
NK=${NK:-36}
NM=${NM:-64}
FLOW_NSTEP=${FLOW_NSTEP:-873}
INNER_RESTART=${INNER_RESTART:-200}
INNER_TOL=${INNER_TOL:-1e-6}
OUTLOG=${OUTLOG:-${LOGDIR}/m0d5_overlap_$(basename "${CONFIG}")_j${JOB_ID:-manual}_claude.log}

export OMP_NUM_THREADS=${NSLOTS:-4}

echo "=== m0d5_overlap $(date)  host ${HOSTNAME:-?}  CUDA=${CUDA_VISIBLE_DEVICES:-?} ==="
nvidia-smi -L 2>/dev/null | head -1
echo "CONFIG=${CONFIG}  CKPT_DIR=${CKPT_DIR}  NSTOP=${NSTOP}"
if [ ! -x "${BIN}" ]; then echo "ERROR: binary missing ${BIN} -> run grid_twolevel_build_scc_gpu_claude.sh"; exit 1; fi
if [ ! -f "${CONFIG}" ]; then echo "ERROR: config missing ${CONFIG}"; exit 1; fi

"${BIN}" --grid "${GRID}" --mpi 1.1.1.1 --config "${CONFIG}" --ckpt_dir "${CKPT_DIR}" \
         --nstop "${NSTOP}" --nk "${NK}" --nm "${NM}" --flow_nstep "${FLOW_NSTEP}" \
         --inner_restart "${INNER_RESTART}" --inner_tol "${INNER_TOL}" \
         --accelerator-threads 8 2>&1 | tee "${OUTLOG}"
echo "m0d5_overlap exit = ${PIPESTATUS[0]}   $(date)"
echo "log -> ${OUTLOG}"
