#!/bin/bash -l
# BU SCC SGE batch: shrink-flow frame test for ONE config on a GPU. Two-stage frame flow --
# Stage 1 Wilson to tau1 ~ t0, baseline Landau frame + FGMRES(M0) (the A side), Stage 2
# under-improved flow (c1=+1/12, epsilon=2) chunked with a Q_5Li monitor until |Q| < QSTOP (lumps
# fall through -> trivial-sector frame copy), Stage 3 short Wilson re-smooth, shrink Landau frame +
# FGMRES(M0) (the B side). CGNE baseline once. Runs the merged GPU binary directly (comms=none ->
# NO mpirun). See tests/solver/shrinkflow_impl_plan_claude.md.
#
# Submit (Nobu submits; Claude never qsub/qdel). C1 is a SPACE-separated list in the env (SGE -v
# splits values on commas; the script converts spaces -> commas for the binary's --c1 list):
#   qsub -v CONFIG=/projectnb/qfe/nmatsum/dwf/configs_iwasaki_16_b2.6/ckpoint_lat.640 grid_shrinkflow_gpu_qsub_claude.sh
#   # c1 trend scan (epsilon = 1, 1.5, 2, 2.4):
#   qsub -v CONFIG=...,C1='0 0.0417 0.0833 0.1167' grid_shrinkflow_gpu_qsub_claude.sh
#   # Experiment 2 -- c1 flow from the BEGINNING (no Wilson stage 1), distinct log + ckp dir:
#   qsub -v CONFIG=...,NSTEP1=0,NOBASELINE=1,C1='0 0.0417 0.0833 0.1167',TAG=640_c1fromstart,CKPDIR=/projectnb/qfe/nmatsum/dwf/shrinkflow_ckp_fromstart grid_shrinkflow_gpu_qsub_claude.sh
# Watch:  qstat -u $USER ; tail -f log/shrinkflow_<cfg>_claude.log

#$ -P qfe
##$ -M mtsmtnbyk@gmail.com
#$ -N shrinkG
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
# interactive/compute nodes: cuda module adds .../install/lib64 but libcublas.so.12 lives in targets/.../lib
export LD_LIBRARY_PATH=/share/pkg.8/cuda/12.8/install/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}

ROOT=/projectnb/qfe/nmatsum/dwf
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"
BIN=${BIN:-${ROOT}/build_merged/Test_dwf_shrinkflow_claude}
GRID=${GRID:-16.16.16.16}
CONFIG=${CONFIG:?set CONFIG=<NERSC config>}
T0=${T0:-2.91}          # b2.6 16^4; other betas: 0.759 (b2.13) / 1.140 (b2.25) / 1.697 (b2.37)
EPS=${EPS:-0.02}
NSTEP1=${NSTEP1:-146}   # tau1 = EPS*NSTEP1 = t0 at b2.6 (the current default frame)
C1=${C1:-0.0833333}     # list, separated by space OR colon; epsilon = 1+12*c1 each; stability c1 < 1/8
# SANITIZE (the Exp-2 crash, 2026-09-05): a multi-line paste fed literal quotes + a newline into C1 via
# qsub -v, and the binary's stod threw. Strip quote/backslash chars, turn ALL whitespace + colons into
# commas, squeeze repeats, trim leading/trailing commas. Colon-separated C1 (no quotes needed) is the
# paste-safe form: C1=0:0.0417:0.0833:0.1167
C1LIST=$(echo "${C1}" | tr -d "'\"\\\\" | tr -s " \t\n:" ',' | sed 's/^,//;s/,$//')
NSTEP2=${NSTEP2:-2000}  # shrink-stage cap (tau2 up to 40 ~ 14 t0; pure gauge = inert, long is safe)
QCHUNK=${QCHUNK:-10}
QSTOP=${QSTOP:-0.5}     # stop stage 2 when |Q_5Li| < QSTOP; 0 disables (flow the full NSTEP2)
NSTEP3=${NSTEP3:-20}
CKPDIR=${CKPDIR:-${ROOT}/shrinkflow_ckp}   # flowed-config checkpoints every t0 (k*t0 crossings) + _final; set empty to disable
# NOBASELINE=1 -> skip the post-stage-1 baseline frame. Use for the "c1 from the beginning" run
# (NSTEP1=0): Landau-fixing the bare (unflowed) config is meaningless.
NOBASELINE=${NOBASELINE:-0}
NOBASEFLAG=""
if [ "${NOBASELINE}" = "1" ]; then NOBASEFLAG="--no_baseline"; fi
# QSQUEEZE=1 -> stage 2 = PURE ||q||_n flow (QSqueezeGaugeAction, FD-validated 2026-09-06) scanning
# QN (colon/space-separated list, e.g. QN=2:4:6); C1 is then ignored. See qsqueeze_impl_plan_claude.md.
QSQUEEZE=${QSQUEEZE:-0}
QN=${QN:-2}
QNLIST=$(echo "${QN}" | tr -d "'\"\\\\" | tr -s " \t\n:" ',' | sed 's/^,//;s/,$//')
QSQFLAG=""
if [ "${QSQUEEZE}" = "1" ]; then QSQFLAG="--qsqueeze --qn ${QNLIST}"; fi
OPS=${OPS:-cgne,m0}
TOL=${TOL:-1e-6}
RESTART=${RESTART:-256}
GFMAXIT=${GFMAXIT:-3000}
# Grid log channels. Iterative -> per-iteration FGMRES(M0)/CG progress. Drop it for quiet.
LOGCH=${LOGCH:-Message,Error,Warning}
TAG=${TAG:-$(basename "${CONFIG}")}
# Per-submission UNIQUE log (never overwrite a prior run's log -- Nobu 2026-09-07). JOB_ID is set by SGE.
OUTLOG=${OUTLOG:-${LOGDIR}/shrinkflow_${TAG}_j${JOB_ID:-manual}_claude.log}

export OMP_NUM_THREADS=${NSLOTS:-8}

if [ -n "${CKPDIR}" ]; then mkdir -p "${CKPDIR}"; fi

echo "=== shrinkG $(date)  host ${HOSTNAME:-?}  CONFIG=${CONFIG}  c1=${C1}  nstep1=${NSTEP1} nstep2<=${NSTEP2} qstop=${QSTOP} ==="
if [ ! -x "${BIN}" ]; then echo "ERROR: shrinkflow GPU binary missing ${BIN} (build: grid_shrinkflow_build_scc_gpu_claude.sh)"; exit 1; fi
if [ ! -f "${CONFIG}" ]; then echo "ERROR: config missing ${CONFIG}"; exit 1; fi

# comms=none GPU build -> run the binary directly (no mpirun).
"${BIN}" --grid "${GRID}" --mpi 1.1.1.1 --accelerator-threads 8 --log "${LOGCH}" \
         --config "${CONFIG}" --t0 "${T0}" --flow_eps "${EPS}" \
         --nstep1 "${NSTEP1}" --c1 "${C1LIST}" --nstep2 "${NSTEP2}" --qchunk "${QCHUNK}" \
         --qstop "${QSTOP}" --nstep3 "${NSTEP3}" --ckp_dir "${CKPDIR}" ${NOBASEFLAG} ${QSQFLAG} \
         --ops "${OPS}" --solve_tol "${TOL}" --fgmres_restart "${RESTART}" --gf_maxit "${GFMAXIT}" \
         2>&1 | tee "${OUTLOG}"
echo "shrinkG exit = ${PIPESTATUS[0]}   $(date)"
echo "log -> ${OUTLOG}"
