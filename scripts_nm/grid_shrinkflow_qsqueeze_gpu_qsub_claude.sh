#!/bin/bash -l
# BU SCC SGE batch: Q-SQUEEZE flow n-scan for ONE config on a GPU. DEDICATED wrapper for the
# ||q||_n localization flow (qsqueeze_impl_plan_claude.md) so the run does NOT depend on a long,
# fragile `qsub -v A=..,B=..` list (that dropped every var but CONFIG on job 7474005 and ran plain
# defaults). Everything qsqueeze-specific is baked in HERE; only CONFIG is overridable via -v.
#
# Submit (Nobu submits; Claude never qsub/qdel):
#   qsub grid_shrinkflow_qsqueeze_gpu_qsub_claude.sh                       # default config 640
#   qsub -v CONFIG=/path/to/ckpoint_lat.NNN grid_shrinkflow_qsqueeze_gpu_qsub_claude.sh
# Watch:  qstat -u $USER ; tail -f log/shrinkflow_<cfg>_qsq_claude.log

#$ -P qfe
##$ -M mtsmtnbyk@gmail.com
#$ -N shrinkQsq
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

# ---- qsqueeze parameters (baked in) ----
T0=2.91          # b2.6 16^4
EPS=0.02
NSTEP1=146       # stage 1 = Wilson to tau1 = t0 (smooth the config before the squeeze acts)
QN=2,4,6         # the n-scan (pure ||q||_n stage-2 flow)
NSTEP2=1455      # stage-2 cap = 10 t0 (tau2 ~ 29 at eps 0.02)
QCHUNK=10
QSTOP=0.5        # stop a track early if |Q_5Li| < QSTOP (lump fell through)
NSTEP3=20        # short Wilson re-smooth before Landau
OPS=cgne,m0
TOL=1e-6
RESTART=20
GFMAXIT=3000
CKPDIR=${ROOT}/shrinkflow_ckp_qsq
TAG=$(basename "${CONFIG}")_qsq
LOGCH=Message,Error,Warning
# Per-submission UNIQUE log (never overwrite a prior run's log -- Nobu 2026-09-07). JOB_ID is set by SGE.
OUTLOG=${LOGDIR}/shrinkflow_${TAG}_j${JOB_ID:-manual}_claude.log

mkdir -p "${CKPDIR}"
export OMP_NUM_THREADS=${NSLOTS:-8}

echo "=== shrinkQsq $(date)  host ${HOSTNAME:-?}  CONFIG=${CONFIG}  QN=${QN}  nstep1=${NSTEP1} nstep2<=${NSTEP2} ==="
if [ ! -x "${BIN}" ]; then echo "ERROR: shrinkflow GPU binary missing ${BIN} (build: grid_shrinkflow_build_scc_gpu_claude.sh)"; exit 1; fi
if [ ! -f "${CONFIG}" ]; then echo "ERROR: config missing ${CONFIG}"; exit 1; fi

# comms=none GPU build -> run the binary directly (no mpirun).
"${BIN}" --grid "${GRID}" --mpi 1.1.1.1 --accelerator-threads 8 --log "${LOGCH}" \
         --config "${CONFIG}" --t0 "${T0}" --flow_eps "${EPS}" \
         --nstep1 "${NSTEP1}" --qsqueeze --qn "${QN}" --nstep2 "${NSTEP2}" --qchunk "${QCHUNK}" \
         --qstop "${QSTOP}" --nstep3 "${NSTEP3}" --ckp_dir "${CKPDIR}" \
         --ops "${OPS}" --solve_tol "${TOL}" --fgmres_restart "${RESTART}" --gf_maxit "${GFMAXIT}" \
         2>&1 | tee "${OUTLOG}"
echo "shrinkQsq exit = ${PIPESTATUS[0]}   $(date)"
echo "log -> ${OUTLOG}"
