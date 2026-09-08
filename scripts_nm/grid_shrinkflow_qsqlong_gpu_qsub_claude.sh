#!/bin/bash -l
# BU SCC SGE batch: LONG q-squeeze + nstep1 scan for ONE config on a GPU. Follow-up to the short
# n-scan (grid_shrinkflow_qsqueeze_gpu_qsub_claude.sh) which showed the ||q||_n force shrinks the lump
# in the right direction but ~100x too slowly at eps 0.02 / 10 t0. Here: BIG stage-2 step (EPS2, lambda
# absorbed into flow time) + long cap so the 1/rho^5 runaway reaches fall-through, crossed with a scan
# of the Wilson pre-flow time NSTEP1 (bulk-smoothing banked before the squeeze). Fixed n=6 (fastest
# shrink, no turnover through n=6). Baseline (Wilson-only frame) printed per NSTEP1 for the A/B.
# See qsqueeze_impl_plan_claude.md. Dedicated wrapper -> NO fragile multi-var `qsub -v`.
#
# NEEDS the GPU binary rebuilt first (flow_eps2 + nstep1-scan were added 2026-09-06):
#   bash grid_shrinkflow_build_scc_gpu_claude.sh
# Submit (Nobu; Claude never qsub/qdel):
#   qsub grid_shrinkflow_qsqlong_gpu_qsub_claude.sh                 # default config 640
#   qsub -v CONFIG=/path/to/ckpoint_lat.NNN grid_shrinkflow_qsqlong_gpu_qsub_claude.sh
# Watch:  qstat -u $USER ; tail -f log/shrinkflow_<cfg>_qsqlong_claude.log

#$ -P qfe
##$ -M mtsmtnbyk@gmail.com
#$ -N shrinkQLn1
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
EPS=0.02          # stage 1 Wilson (t0 scale) -- keep small
EPS2=0.4          # stage 2 ||q||_n step: 20x eps (aggressive: 1.0 = 50x); lambda absorbed in flow time
NSTEP1=146,730,1460   # Wilson pre-flow scan: 1, 5, 10 t0 (bulk-smoothing vs starting lump size)
QN=6              # n=6 was the fastest shrink, no turnover
NSTEP2=2000       # stage-2 cap: tau2 up to EPS2*2000 = 800 (~275 t0); qstop stops at fall-through
QCHUNK=10
QSTOP=0.5         # stop a track when |Q_5Li| < QSTOP (lump fell through)
NSTEP3=20
OPS=cgne,m0
TOL=1e-6
RESTART=256
GFMAXIT=3000
CKPDIR=${ROOT}/shrinkflow_ckp_qsqlong_n1scan
TAG=$(basename "${CONFIG}")_qsqlong_n1scan
LOGCH=Message,Error,Warning
# Per-submission UNIQUE log (never overwrite a prior run's log -- Nobu 2026-09-07). JOB_ID is set by SGE.
OUTLOG=${LOGDIR}/shrinkflow_${TAG}_j${JOB_ID:-manual}_claude.log

mkdir -p "${CKPDIR}"
export OMP_NUM_THREADS=${NSLOTS:-8}

echo "=== shrinkQL $(date)  host ${HOSTNAME:-?}  CONFIG=${CONFIG}  QN=${QN}  NSTEP1=${NSTEP1}  EPS2=${EPS2} nstep2<=${NSTEP2} ==="
if [ ! -x "${BIN}" ]; then echo "ERROR: shrinkflow GPU binary missing ${BIN} (build: grid_shrinkflow_build_scc_gpu_claude.sh)"; exit 1; fi
if [ ! -f "${CONFIG}" ]; then echo "ERROR: config missing ${CONFIG}"; exit 1; fi

# comms=none GPU build -> run the binary directly (no mpirun).
"${BIN}" --grid "${GRID}" --mpi 1.1.1.1 --accelerator-threads 8 --log "${LOGCH}" \
         --config "${CONFIG}" --t0 "${T0}" --flow_eps "${EPS}" --flow_eps2 "${EPS2}" \
         --nstep1 "${NSTEP1}" --qsqueeze --qn "${QN}" --nstep2 "${NSTEP2}" --qchunk "${QCHUNK}" \
         --qstop "${QSTOP}" --nstep3 "${NSTEP3}" --ckp_dir "${CKPDIR}" \
         --ops "${OPS}" --solve_tol "${TOL}" --fgmres_restart "${RESTART}" --gf_maxit "${GFMAXIT}" \
         2>&1 | tee "${OUTLOG}"
echo "shrinkQL exit = ${PIPESTATUS[0]}   $(date)"
echo "log -> ${OUTLOG}"
