#!/bin/bash -l
# BU SCC SGE batch: SPLIT two-level deflation binaries (one job = one binary), GPU.
# Design: scripts_nm/svd_deflation_topological_twolevel_claude.md.
#
# Four binaries (build with grid_twolevel_build_scc_gpu_claude.sh), selected by WHICH=:
#   svddump_rbcgne   -- dump RB-Schur^2 low modes -> deflrb   (Lanczos; run FIRST)
#   svddump_deflprec -- dump D5^H D5 low modes    -> deflfull (Lanczos; run FIRST)
#   rbcgne           -- read deflrb  -> RB-CGNE plain vs deflated
#   deflprec         -- read deflfull -> SVD-deflation + M0-FGMRES
# The dumps and the solves share ONE CKPT_DIR (deflvec_<cfg>), so run a dump before its solve.
#
# Submit (2 dumps, then 2 solves):
#   qsub -v CONFIG=/.../ckpoint_lat.640,WHICH=svddump_deflprec grid_twolevel_gpu_qsub_claude.sh
#   qsub -v CONFIG=/.../ckpoint_lat.640,WHICH=svddump_rbcgne   grid_twolevel_gpu_qsub_claude.sh
#   qsub -v CONFIG=/.../ckpoint_lat.640,WHICH=deflprec         grid_twolevel_gpu_qsub_claude.sh
#   qsub -v CONFIG=/.../ckpoint_lat.640,WHICH=rbcgne           grid_twolevel_gpu_qsub_claude.sh

#$ -P qfe
##$ -M mtsmtnbyk@gmail.com
#$ -N twolevelG
#$ -j y
#$ -o /projectnb/qfe/nmatsum/dwf/log/
##$ -m n
#$ -l h_rt=6:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -pe omp 4
## -l mem_per_core=8G   # NOT NEEDED: peak physical RSS is only ~3 GB (fits the default node), and it does
##                      # not gate the failure -- the RB-Schur^2 dump reached 880 GB *virtual* (address
##                      # space) and still succeeded (qacct failed=0), so mem_per_core wasn't capping vmem.
##                      # The earlier rbcgne crashes were RLIMIT_AS (address-space) ceilings on small nodes,
##                      # hit by the RB path's ~880 GB VA reservation -- orthogonal to this request.

set -u

module load cuda/12.8
module load gcc/13.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"

WHICH=${WHICH:-deflprec}            # svddump_rbcgne | svddump_deflprec | rbcgne | deflprec
GRID=${GRID:-16.16.16.16}
CONFIG=${CONFIG:-${ROOT}/configs_iwasaki_16_b2.6/ckpoint_lat.640}
NMODES=${NMODES:-24}               # svddump_deflprec: # of D5^H D5 modes to dump
DEFL_K=${DEFL_K:-10}               # svddump_rbcgne: # of RB-Schur^2 modes to dump (set 48 for the w_o scan)
WO_LIST=${WO_LIST:-16,24,32,40,48} # rbcgne: outer-deflation sizes w_o to scan (prefixes of the dumped modes)
CHEB_ORD=${CHEB_ORD:-21}
SVD_KLIST=${SVD_KLIST:-3,10}       # deflprec: deflation sizes w_o to loop (outer-deflation scan, e.g. 16,24,32,40,48)
FLOW_NSTEP=${FLOW_NSTEP:-873}      # deflprec: Landau warm-start Wilson flow (s/t0=6 at eps 0.02)
GMRES_K=${GMRES_K:-32}             # deflprec: GMRES-DR adaptive recycle on the complement. Set 0 to ISOLATE
                                   # the outer (SVD) deflation -- with k=32 adaptive masks the w_o effect.

# dumps and solves share this dir (a dump writes it; the matching solve reads it)
CKPT_DIR=${CKPT_DIR:-${LOGDIR}/deflvec_$(basename "${CONFIG}")}
mkdir -p "${CKPT_DIR}"

BIN=${ROOT}/build_merged/Test_dwf_${WHICH}_claude
case "${WHICH}" in
  svddump_rbcgne)   ARGS="--config ${CONFIG} --ckpt_dir ${CKPT_DIR} --defl_k ${DEFL_K} --cheb_ord ${CHEB_ORD}" ;;
  svddump_deflprec) ARGS="--config ${CONFIG} --ckpt_dir ${CKPT_DIR} --nmodes ${NMODES} --cheb_ord ${CHEB_ORD}" ;;
  rbcgne)           ARGS="--config ${CONFIG} --ckpt_dir ${CKPT_DIR} --wo_list ${WO_LIST}" ;;
  deflprec)         ARGS="--config ${CONFIG} --ckpt_dir ${CKPT_DIR} --svd_klist ${SVD_KLIST} --flow_nstep ${FLOW_NSTEP} --gmres_k ${GMRES_K}" ;;
  *) echo "ERROR: unknown WHICH=${WHICH}"; exit 1 ;;
esac

OUTLOG=${OUTLOG:-${LOGDIR}/twolevel_$(basename "${CONFIG}")_${WHICH}_j${JOB_ID:-manual}_claude.log}

export OMP_NUM_THREADS=${NSLOTS:-4}

echo "=== ${WHICH} $(date)  host ${HOSTNAME:-?}  CUDA=${CUDA_VISIBLE_DEVICES:-?} ==="
nvidia-smi -L 2>/dev/null | head -1
echo "GRID=${GRID}  CONFIG=${CONFIG}  CKPT_DIR=${CKPT_DIR}"
if [ ! -x "${BIN}" ]; then echo "ERROR: binary missing ${BIN} -> run grid_twolevel_build_scc_gpu_claude.sh"; exit 1; fi
if [ ! -f "${CONFIG}" ]; then echo "ERROR: config missing ${CONFIG}"; exit 1; fi

echo "BIN=${BIN}"
echo "ARGS=${ARGS}"
"${BIN}" --grid "${GRID}" --mpi 1.1.1.1 ${ARGS} --accelerator-threads 8 2>&1 | tee "${OUTLOG}"
echo "${WHICH} exit = ${PIPESTATUS[0]}   $(date)"
echo "log -> ${OUTLOG}"
