#!/bin/bash -l
# BU SCC SGE batch: 24^4 fp32 comparison RB-CGNE vs deflated FGMRES (GMRES-DR) on ONE config.
# fp32 halves the GMRES-DR Krylov basis so the 24^4 x Ls8 solve fits one GPU (~32 GB). Frame (flow +
# FA-Landau) is DOUBLE and mass-independent -> built once, reused over the mass list; flowed config is
# CACHED (NERSC) keyed by (config, eps, nstep) so a rerun skips the flow.
# Binary: build_merged/Test_dwf_freeprec_fp32_claude (grid_freeprec_fp32_build_scc_gpu_claude.sh).
#
# Submit: qsub -v CONFIG=/.../configs_iwasaki_24_b2.6/ckpoint_lat.1100 grid_freeprec_fp32_gpu_qsub_claude.sh
# Watch:  qstat -u $USER ; tail -f log/freeprec_fp32_<cfg>_j<JID>_claude.log

#$ -P qfe
#$ -N freeprecF32
#$ -j y
#$ -o /projectnb/qfe/nmatsum/dwf/log/
#$ -l h_rt=8:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -pe omp 4

set -u

module load cuda/12.8
module load gcc/13.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"
BIN=${ROOT}/build_merged/Test_dwf_freeprec_fp32_claude
GRID=${GRID:-24.24.24.24}
CONFIG=${CONFIG:-${ROOT}/configs_iwasaki_24_b2.6/ckpoint_lat.1100}
MASSLIST=${MASSLIST:-0.1:0.01:0.001}   # masses LOOPED over the SINGLE frame; COLON-sep (SGE -v splits commas)
MLCOMMA=$(echo "${MASSLIST}" | tr ':' ',')
RESTART=${RESTART:-}       # GMRES-DR restart window m (empty = binary default 16)
DEFLATEK=${DEFLATEK:-}     # GMRES-DR deflation dim k (empty = binary default 32)
TOL=${TOL:-}               # solver tol (empty = binary default 1e-6; fp32 floor ~3e-7)
RSOPT=""; [ -n "${RESTART}" ]  && RSOPT="--restart ${RESTART}"
DKOPT=""; [ -n "${DEFLATEK}" ] && DKOPT="--deflate-k ${DEFLATEK}"
TOLOPT=""; [ -n "${TOL}" ]     && TOLOPT="--tol ${TOL}"
FLOWCACHE=${FLOWCACHE:-${ROOT}/flowcache}   # dir must exist; flowed config cached here
OUTLOG=${OUTLOG:-${LOGDIR}/freeprec_fp32_$(basename "${CONFIG}")_j${JOB_ID:-manual}_claude.log}

export OMP_NUM_THREADS=${NSLOTS:-4}

echo "=== freeprec fp32 $(date)  host ${HOSTNAME:-?}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?} ==="
nvidia-smi -L 2>/dev/null | head -1
echo "GRID=${GRID}  CONFIG=${CONFIG}  MASSLIST=${MASSLIST}  RESTART=${RESTART:-16}  DEFLATEK=${DEFLATEK:-32}  TOL=${TOL:-1e-6}"
if [ ! -x "${BIN}" ]; then echo "ERROR: binary missing ${BIN} -> run grid_freeprec_fp32_build_scc_gpu_claude.sh"; exit 1; fi
if [ ! -f "${CONFIG}" ]; then echo "ERROR: config missing ${CONFIG}"; exit 1; fi
if [ ! -d "${FLOWCACHE}" ]; then echo "ERROR: flowcache dir missing ${FLOWCACHE}"; exit 1; fi

# comms=none CUDA binary: run directly (no mpirun), one GPU.
"${BIN}" --grid "${GRID}" --mpi 1.1.1.1 --config "${CONFIG}" --mass-list "${MLCOMMA}" \
         --flowcache "${FLOWCACHE}" ${RSOPT} ${DKOPT} ${TOLOPT} \
         --accelerator-threads 8 2>&1 | tee "${OUTLOG}"
echo "freeprec_fp32 exit = ${PIPESTATUS[0]}   $(date)"
echo "log -> ${OUTLOG}"
