#!/bin/bash -l
# BU SCC SGE batch: Q-FRAME experiment (Nobu 2026-09-07) on a GPU. q^n-flow the ORIGINAL config (NO
# Wilson flow) to localise the topological deficit to ~1 site (U^q); frame-optimise Omega on D_W(U^q)
# (lump measure-zero in the loss -> clean bulk frame); then precondition the ORIGINAL D_W(U) and compare
# FGMRES iters: Landau(U) frame vs the U^q-optimised frame. Same binary as the frameopt runs
# (Test_wilson_frameopt_claude, --qframe). See holonomy_frameopt_impl_plan_claude.md / qsqueeze notes.
#
# NEEDS the GPU binary built: bash grid_wilson_frameopt_build_scc_gpu_claude.sh
# Submit (Nobu; Claude never qsub/qdel):
#   qsub -v CONFIG=/projectnb/qfe/nmatsum/dwf/configs_iwasaki_16_b2.6/ckpoint_lat.640 grid_wilson_qframe_gpu_qsub_claude.sh
#   qsub grid_wilson_qframe_gpu_qsub_claude.sh                              # default config 640
# Watch:  qstat -u $USER ; tail -f log/qframe_<cfg>_j<jobid>_claude.log

#$ -P qfe
##$ -M mtsmtnbyk@gmail.com
#$ -N qframeG
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
CONFIG=${CONFIG:-${ROOT}/configs_iwasaki_16_b2.6/ckpoint_lat.640}

MASS=${MASS:-0.1}
MPREC=${MPREC:-0.1}
QN=${QN:-6}                 # ||q||_n order for the localizing flow
QFLOWNSTEP=${QFLOWNSTEP:-2000}   # q-flow steps (eps 0.4 -> tau2 up to 800; qstop not used here, full flow)
QFLOWEPS=${QFLOWEPS:-0.4}
FLOWEPS=${FLOWEPS:-0.02}
FLOWNSTEP=${FLOWNSTEP:-146}   # Wilson flow ONLY for the Landau(U) baseline frame (xform), s/t0=1
FOPROBES=${FOPROBES:-4}
FOITER=${FOITER:-60}
FOETA=${FOETA:-0.1}
RESTART=${RESTART:-20}
LOGCH=${LOGCH:-Message,Error,Warning}
TAG=$(basename "${CONFIG}")
OUTLOG=${LOGDIR}/qframe_${TAG}_j${JOB_ID:-manual}_claude.log

export OMP_NUM_THREADS=${NSLOTS:-8}

echo "=== qframeG $(date)  host ${HOSTNAME:-?}  CONFIG=${CONFIG}  qn=${QN} qflow_nstep=${QFLOWNSTEP} qeps=${QFLOWEPS} ==="
if [ ! -x "${BIN}" ]; then echo "ERROR: frameopt GPU binary missing ${BIN} (build: grid_wilson_frameopt_build_scc_gpu_claude.sh)"; exit 1; fi
if [ ! -f "${CONFIG}" ]; then echo "ERROR: config missing ${CONFIG}"; exit 1; fi

"${BIN}" --grid "${GRID}" --mpi 1.1.1.1 --accelerator-threads 8 --log "${LOGCH}" \
         --config "${CONFIG}" --mass "${MASS}" --mprec "${MPREC}" \
         --flow-eps "${FLOWEPS}" --flow-nstep "${FLOWNSTEP}" \
         --qframe --qn "${QN}" --qflow-nstep "${QFLOWNSTEP}" --qflow-eps "${QFLOWEPS}" \
         --fo-probes "${FOPROBES}" --fo-iter "${FOITER}" --fo-eta "${FOETA}" --restart "${RESTART}" \
         2>&1 | tee "${OUTLOG}"
echo "qframeG exit = ${PIPESTATUS[0]}   $(date)"
echo "log -> ${OUTLOG}"
