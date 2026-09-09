#!/bin/bash -l
# BU SCC SGE batch: LONG q-squeeze n-SCAN at FIXED nstep1=t0 for ONE config on a GPU. Companion to
# grid_shrinkflow_qsqlong_gpu_qsub_claude.sh (that one scans nstep1 at fixed n=6; this one scans
# n = 6,8,10,12 at fixed nstep1 = 146 = t0). Both use the big stage-2 step (EPS2, lambda absorbed into
# flow time) + long cap so the 1/rho^5 runaway reaches fall-through. Higher n = more core-concentrated
# force (6c): expect faster shrink until the force width ~ rho/sqrt(n) drops below the profile
# (~n>10 at rho~3) and it starts manufacturing a single-site spike instead of shrinking the instanton
# -- the turnover this scan is meant to find. See qsqueeze_impl_plan_claude.md.
# Dedicated wrapper -> NO fragile multi-var `qsub -v`.
#
# NEEDS the GPU binary rebuilt first (flow_eps2 + nstep1-scan, 2026-09-06):
#   bash grid_shrinkflow_build_scc_gpu_claude.sh
# Submit (Nobu; Claude never qsub/qdel):
#   qsub grid_shrinkflow_qsqlong_nscan_gpu_qsub_claude.sh                 # default config 640
#   qsub -v CONFIG=/path/to/ckpoint_lat.NNN grid_shrinkflow_qsqlong_nscan_gpu_qsub_claude.sh
# Watch:  qstat -u $USER ; tail -f log/shrinkflow_<cfg>_qsqlong_nscan_claude.log

#$ -P qfe
##$ -M mtsmtnbyk@gmail.com
#$ -N shrinkQLn
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
BIN=${ROOT}/build_merged/Test_dwf_shrinkflow_claude
GRID=16.16.16.16
CONFIG=${CONFIG:-${ROOT}/configs_iwasaki_16_b2.6/ckpoint_lat.640}

# ---- parameters (baked in) ----
T0=2.91
EPS=0.02          # stage 1 Wilson (t0 scale)
EPS2=0.4          # stage 2 ||q||_n step: 20x eps; lambda absorbed into flow time
NSTEP1=146        # FIXED Wilson pre-flow = t0 (the frame we already use)
QN=6,8,10,12      # the n-scan (all EVEN, required by QSqueezeGaugeAction)
NSTEP2=2000       # stage-2 cap: tau2 up to EPS2*2000 = 800 (~275 t0); qstop stops at fall-through
QCHUNK=10
QSTOP=0.5
NSTEP3=20
OPS=cgne,m0
TOL=1e-6
RESTART=20
GFMAXIT=3000
CKPDIR=${ROOT}/shrinkflow_ckp_qsqlong_nscan
TAG=$(basename "${CONFIG}")_qsqlong_nscan
LOGCH=Message,Error,Warning
# Per-submission UNIQUE log (never overwrite a prior run's log -- Nobu 2026-09-07). JOB_ID is set by SGE.
OUTLOG=${LOGDIR}/shrinkflow_${TAG}_j${JOB_ID:-manual}_claude.log

mkdir -p "${CKPDIR}"
export OMP_NUM_THREADS=${NSLOTS:-8}

echo "=== shrinkQLn $(date)  host ${HOSTNAME:-?}  CONFIG=${CONFIG}  QN=${QN}  NSTEP1=${NSTEP1}  EPS2=${EPS2} nstep2<=${NSTEP2} ==="
if [ ! -x "${BIN}" ]; then echo "ERROR: shrinkflow GPU binary missing ${BIN} (build: grid_shrinkflow_build_scc_gpu_claude.sh)"; exit 1; fi
if [ ! -f "${CONFIG}" ]; then echo "ERROR: config missing ${CONFIG}"; exit 1; fi

# comms=none GPU build -> run the binary directly (no mpirun).
"${BIN}" --grid "${GRID}" --mpi 1.1.1.1 --accelerator-threads 8 --log "${LOGCH}" \
         --config "${CONFIG}" --t0 "${T0}" --flow_eps "${EPS}" --flow_eps2 "${EPS2}" \
         --nstep1 "${NSTEP1}" --qsqueeze --qn "${QN}" --nstep2 "${NSTEP2}" --qchunk "${QCHUNK}" \
         --qstop "${QSTOP}" --nstep3 "${NSTEP3}" --ckp_dir "${CKPDIR}" \
         --ops "${OPS}" --solve_tol "${TOL}" --fgmres_restart "${RESTART}" --gf_maxit "${GFMAXIT}" \
         2>&1 | tee "${OUTLOG}"
echo "shrinkQLn exit = ${PIPESTATUS[0]}   $(date)"
echo "log -> ${OUTLOG}"
