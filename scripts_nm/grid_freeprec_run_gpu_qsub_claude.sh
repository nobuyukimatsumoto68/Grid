#!/bin/bash -l
# BU SCC SGE batch: free-limit preconditioner test on ONE config, on a GPU (single-GPU, 16^4 fits).
# Runs the multi-arch CUDA freeprec binary (build/). FGMRES R=256 at 16^4 Ls8 ~ 26 GB -> request
# gpu_c=8.0 (Ampere+, >=40 GB) so it avoids a 16 GB V100. 24^4 (~132 GB) does NOT fit one GPU -> multi-GPU.
#
# Submit:  qsub -v CONFIG=/.../ckpoint_lat.200,GRID=16.16.16.16 grid_freeprec_run_gpu_qsub_claude.sh
# Watch:   qstat -u $USER ; tail -f log/freeprec_gpu_<cfg>_claude.log
# All text output (the tee'd program log AND the SGE job .o file) goes to ${ROOT}/log/.

#$ -P qfe
#$ -M mtsmtnbyk@gmail.com
#$ -N freeprecG
#$ -j y
#$ -o /projectnb/qfe/nmatsum/dwf/log/
#$ -m beas
#$ -l h_rt=2:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -pe omp 4

set -u

module load cuda/12.8
module load gcc/13.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"
BIN=${ROOT}/build/Test_dwf_freeprec_claude
GRID=${GRID:-16.16.16.16}
CONFIG=${CONFIG:?set CONFIG=<NERSC config>}
OUTLOG=${OUTLOG:-${LOGDIR}/freeprec_gpu_$(basename "${CONFIG}")_claude.log}

export OMP_NUM_THREADS=${NSLOTS:-4}

echo "=== freeprec GPU $(date)  host ${HOSTNAME:-?}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?} ==="
nvidia-smi -L 2>/dev/null | head -1
echo "GRID=${GRID}  CONFIG=${CONFIG}"
if [ ! -x "${BIN}" ]; then echo "ERROR: GPU freeprec binary missing ${BIN}"; exit 1; fi
if [ ! -f "${CONFIG}" ]; then echo "ERROR: config missing ${CONFIG}"; exit 1; fi

# comms=none CUDA binary: run directly (no mpirun), one GPU.
"${BIN}" --grid "${GRID}" --mpi 1.1.1.1 --config "${CONFIG}" --accelerator-threads 8 2>&1 | tee "${OUTLOG}"
echo "freeprec exit = ${PIPESTATUS[0]}   $(date)"
echo "log -> ${OUTLOG}"
