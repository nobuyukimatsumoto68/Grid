#!/bin/bash -l
# BU SCC SGE batch: DWF Fourier/RB interacting solve, GPU.
# Plan: tests/solver/dwf_fourier_rb_impl_plan_claude.md.
# Dedicated framed Fourier-preconditioned RB-FGMRES: M0^rb = P_o (Omega^dag F Omega) P_o, GMRES-DR on the
# odd-checkerboard Schur S_o. D_W = Ls*iters; compare to plain RB-CGNE from Test_dwf_rbcgne_claude.cc.
#
# Submit:
#   qsub -v CONFIG=/projectnb/qfe/nmatsum/dwf/configs_iwasaki_16_b2.6/ckpoint_lat.640 grid_dwfrb_gpu_qsub_claude.sh
# Optional: -v ...,RESTART=30,MAXIT=40000

#$ -P qfe
#$ -N dwfrbG
#$ -j y
#$ -o /projectnb/qfe/nmatsum/dwf/log/
#$ -l h_rt=6:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -pe omp 4

set -u

module load cuda/12.8
module load gcc/13.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"

GRID=${GRID:-16.16.16.16}
CONFIG=${CONFIG:-${ROOT}/configs_iwasaki_16_b2.6/ckpoint_lat.640}
RESTART=${RESTART:-16}
MAXIT=${MAXIT:-20000}
TOL=${TOL:-1e-8}
FLOW_EPS=${FLOW_EPS:-0.02}
FLOW_NSTEP=${FLOW_NSTEP:-873}    # s/t0=6 Landau warm-start flow length (0 = frame the unflowed config)
GMRES_K=${GMRES_K:-32}          # GMRES-DR deflation dim (latest twolevel value; deflates topological stragglers)

BIN=${ROOT}/build_merged/Test_dwf_freeprec_rb_solve_claude
OUTLOG=${OUTLOG:-${LOGDIR}/dwf_freeprec_rb_solve_$(basename "${CONFIG}")_j${JOB_ID:-manual}_claude.log}

export OMP_NUM_THREADS=${NSLOTS:-4}

echo "=== dwfrb solve $(date)  host ${HOSTNAME:-?}  CUDA=${CUDA_VISIBLE_DEVICES:-?} ==="
nvidia-smi -L 2>/dev/null | head -1
echo "GRID=${GRID}  CONFIG=${CONFIG}  RESTART=${RESTART}  MAXIT=${MAXIT}  TOL=${TOL}  GMRES_K=${GMRES_K}  FLOW=${FLOW_EPS}x${FLOW_NSTEP}"
if [ ! -x "${BIN}" ]; then echo "ERROR: binary missing ${BIN} -> run grid_dwfrb_build_scc_gpu_claude.sh"; exit 1; fi
if [ ! -f "${CONFIG}" ]; then echo "ERROR: config missing ${CONFIG}"; exit 1; fi

echo "BIN=${BIN}"
"${BIN}" --grid "${GRID}" --mpi 1.1.1.1 --config "${CONFIG}" \
         --restart "${RESTART}" --maxit "${MAXIT}" --tol "${TOL}" \
         --gmres_k "${GMRES_K}" --flow_eps "${FLOW_EPS}" --flow_nstep "${FLOW_NSTEP}" \
         --accelerator-threads 8 2>&1 | tee "${OUTLOG}"
echo "dwfrb solve exit = ${PIPESTATUS[0]}   $(date)"
echo "log -> ${OUTLOG}"
