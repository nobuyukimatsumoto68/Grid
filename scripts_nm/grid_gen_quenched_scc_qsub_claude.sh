#!/bin/bash -l
# BU SCC SGE batch script: generate quenched Iwasaki SU(3) configs (R2 chunk C). ONE GPU, single stream.
# Delegates to grid_gen_quenched_scc_claude.sh (build + HMC run). Multi-arch lib -> gpu_c=7.0 is a safe
# floor (runs on any SCC GPU >= cc 7.0). Quenched gen is cheap and needs only ONE GPU.
#
# Submit:  qsub grid_gen_quenched_scc_qsub_claude.sh
#          qsub -v GRID=24.24.24.24,TRAJ=200,THERM=50,START=HotStart grid_gen_quenched_scc_qsub_claude.sh
#          qsub -v START=ColdStart,OUTDIR=/.../configs_iwasaki_24_b2.6_cold grid_gen_quenched_scc_qsub_claude.sh
# Watch:   qstat -u $USER ; tail -f grid_gen_quenched_scc_claude.log

#$ -P qfe
##$ -M nmatsum@bu.edu
##$ -m n
#$ -j y
#$ -N gen_iwa
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=12:00:00
#$ -pe omp 4

set -u

echo "=========================================================="
echo "Start date : $(date)   Job ID : ${JOB_ID:-?}   Host : ${HOSTNAME:-?}"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-unset}   NSLOTS : ${NSLOTS:-?}"
nvidia-smi -L 2>/dev/null || echo "WARNING: nvidia-smi -L failed"
echo "=========================================================="

GENRUN=/projectnb/qfe/nmatsum/dwf/Grid/scripts_nm/grid_gen_quenched_scc_claude.sh

export OMP_NUM_THREADS=${NSLOTS:-4}
export OPENBLAS_NUM_THREADS=${NSLOTS:-4}

# pass-through knobs (defaults live in the gen script)
export GRID=${GRID:-24.24.24.24}
[ -n "${OUTDIR:-}" ] && export OUTDIR
[ -n "${TRAJ:-}" ]   && export TRAJ
[ -n "${THERM:-}" ]  && export THERM
[ -n "${START:-}" ]  && export START

echo "+ bash ${GENRUN}   (GRID=${GRID} START=${START:-HotStart} TRAJ=${TRAJ:-200} THERM=${THERM:-50})"
bash "${GENRUN}"
echo "end date : $(date)   gen script exit = $?"
echo "finished"
