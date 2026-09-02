#!/bin/bash -l
# BU SCC SGE *BATCH* script for the free-limit Mobius preconditioner gate (Grid port).
# Modeled on qed3/src/production/run_L4_scc_claude.sh: ONE job == ONE GPU, single stream, NO MPS.
#
# Submit (resource requests live in the #$ directives below, so a bare qsub works):
#     qsub grid_freeprec_scc_qsub_claude.sh                         # 8^4 headline (default)
#     qsub -v GRID=4.4.4.4,CFG=<...cfg_su3_4444...nersc> grid_freeprec_scc_qsub_claude.sh   # 4^4 validation
# Watch:  qstat -u $USER   ;  tail -f grid_freeprec_build_scc_claude.log
#
# The heavy lifting (module load, nvcc compile+link of the test, the run) is delegated to
# grid_freeprec_build_scc_claude.sh -- on a GPU node nvidia-smi is present so that script executes
# the binary rather than stopping at "built only".
#
# NOTE ON ARCH: grid_build_scc_claude.sh now builds MULTI-ARCH (sm_70/80/90 + compute_90 PTX), so the
# binary runs on any SCC GPU >= cc 7.0 and gpu_c=7.0 (a floor) is safe. If instead you run an OLD
# sm_70-only library, SGE may place the job on a newer card (A40 cc8.6, L40S cc8.9, ...) and the run
# dies at the first device kernel with "no kernel image is available" -- then pin the card by adding
#   qsub -l gpu_type=V100 ...   (V100 = cc 7.0), or rebuild the library multi-arch.

#$ -P qfe
#$ -M nmatsum@bu.edu
#$ -m ea
#$ -j y
#$ -N freeprec
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=2:00:00
#$ -pe omp 4

set -u

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name   : ${JOB_NAME:-?}   Job ID : ${JOB_ID:-?}"
echo "Host name  : ${HOSTNAME:-?}"
echo "TMPDIR     : ${TMPDIR:-?}   NSLOTS : ${NSLOTS:-?}"
echo "CUDA_VISIBLE_DEVICES (SGE-assigned) : ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>/dev/null || echo "WARNING: nvidia-smi -L failed"
echo "=========================================================="

BUILD_RUN=/projectnb/qfe/nmatsum/dwf/Grid/scripts_nm/grid_freeprec_build_scc_claude.sh

# single stream on the assigned GPU: give all CPU slots to host OpenMP
export OMP_NUM_THREADS=${NSLOTS:-4}
export OPENBLAS_NUM_THREADS=${NSLOTS:-4}

# GRID / CFG pass straight through to the build+run script (which defaults to the 8^4 headline config).
# Exported here so they survive into the child shell whether set via qsub -v or left to defaults.
export GRID=${GRID:-8.8.8.8}
[ -n "${CFG:-}" ] && export CFG

echo "+ bash ${BUILD_RUN}   (GRID=${GRID} CFG=${CFG:-<default 8^4>})"
bash "${BUILD_RUN}"
rc=$?

echo "end date : $(date)"
echo "build+run script exit code = ${rc}"
echo "finished"
