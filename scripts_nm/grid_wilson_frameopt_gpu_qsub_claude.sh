#!/bin/bash -l
# BU SCC SGE batch: Wilson frame-optimization with the JOINT (Omega, theta) holonomy optimisation on a
# GPU. --frameopt --opt_theta: flow+Landau -> Omega0; then (a) Omega-only descent, (b) joint (Omega,theta)
# descent with the holonomy-twisted free kernel (FreeWilsonTwisted_claude.h). Reports L(Landau) vs
# L(joint) and FGMRES iters Landau / Omega-only / joint. See holonomy_frameopt_impl_plan_claude.md.
# Runs the merged GPU binary directly (comms=none -> NO mpirun).
#
# NEEDS the GPU binary built: bash grid_wilson_frameopt_build_scc_gpu_claude.sh
# Submit (Nobu; Claude never qsub/qdel):
#   qsub grid_wilson_frameopt_gpu_qsub_claude.sh                              # default config 747 (Q=0)
#   qsub -v CONFIG=/projectnb/qfe/nmatsum/dwf/configs_iwasaki_16_b2.6/ckpoint_lat.640 grid_wilson_frameopt_gpu_qsub_claude.sh
# Watch:  qstat -u $USER ; tail -f log/frameopt_holonomy_<cfg>_j<jobid>_claude.log

#$ -P qfe
##$ -M mtsmtnbyk@gmail.com
#$ -N frameoptG
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
BIN=${ROOT}/build_merged/Test_wilson_frameopt_claude
GRID=${GRID:-16.16.16.16}
CONFIG=${CONFIG:-${ROOT}/configs_iwasaki_16_b2.6/trivialQ_lat.747}

# ---- parameters ----
MASS=${MASS:-0.1}
MPREC=${MPREC:-0.1}
FLOWEPS=${FLOWEPS:-0.02}
FLOWNSTEP=${FLOWNSTEP:-146}   # Landau frame at s/t0=1 (tau=t0 on b2.6)
FOPROBES=${FOPROBES:-4}
FOITER=${FOITER:-60}
FOETA=${FOETA:-0.1}
THETAETA=${THETAETA:-0.05}
RESTART=${RESTART:-20}
LOGCH=${LOGCH:-Message,Error,Warning}
TAG=$(basename "${CONFIG}")
# Per-submission UNIQUE log (never overwrite -- feedback-unique-logs). JOB_ID set by SGE.
OUTLOG=${LOGDIR}/frameopt_holonomy_${TAG}_j${JOB_ID:-manual}_claude.log

export OMP_NUM_THREADS=${NSLOTS:-8}

echo "=== frameoptG $(date)  host ${HOSTNAME:-?}  CONFIG=${CONFIG}  mass=${MASS} flow_nstep=${FLOWNSTEP} ==="
if [ ! -x "${BIN}" ]; then echo "ERROR: frameopt GPU binary missing ${BIN} (build: grid_wilson_frameopt_build_scc_gpu_claude.sh)"; exit 1; fi
if [ ! -f "${CONFIG}" ]; then echo "ERROR: config missing ${CONFIG}"; exit 1; fi

# comms=none GPU build -> run the binary directly (no mpirun).
"${BIN}" --grid "${GRID}" --mpi 1.1.1.1 --accelerator-threads 8 --log "${LOGCH}" \
         --config "${CONFIG}" --mass "${MASS}" --mprec "${MPREC}" \
         --flow-eps "${FLOWEPS}" --flow-nstep "${FLOWNSTEP}" \
         --frameopt --opt_theta --fo-probes "${FOPROBES}" --fo-iter "${FOITER}" \
         --fo-eta "${FOETA}" --theta-eta "${THETAETA}" --restart "${RESTART}" \
         2>&1 | tee "${OUTLOG}"
echo "frameoptG exit = ${PIPESTATUS[0]}   $(date)"
echo "log -> ${OUTLOG}"
