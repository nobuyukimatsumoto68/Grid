#!/bin/bash -l
# BU SCC SGE batch: run the free-limit preconditioner test (Landau frame M0 + leading D_W correction M1)
# on ONE config, single node. Runs the already-built CPU freeprec binary. Prints the validation gates +
# the 3b headline on --config: flowed-fixed Landau functional, Q(orig), and CGNE vs FGMRES(M0) vs
# FGMRES(M1) D_W-apply counts + speedups. M1 total D_W = Ls*outer + Ls*(internal D_DW[U^L] applies).
#
# Submit:  qsub -v CONFIG=/.../ckpoint_lat.200,GRID=16.16.16.16 grid_freeprec_run_1node_qsub_claude.sh
# ADDITIVE op selection: pass OPS=<comma-list of cgne,m0,m1> to run only those solves (default = all).
#   e.g. -v CONFIG=...,OPS=m1  reruns ONLY FGMRES(M1) (skips the already-logged CGNE + M0 recompute).
# Watch:   qstat -u $USER ; tail -f log/freeprec_<cfg>_claude.log
# All text output (the tee'd program log AND the SGE job .o file) goes to ${ROOT}/log/.

#$ -P qfe
##$ -M mtsmtnbyk@gmail.com
#$ -N freeprec
#$ -j y
#$ -o /projectnb/qfe/nmatsum/dwf/log/
##$ -m n
#$ -l h_rt=1:45:00
#$ -pe omp 16

set -u

module load gcc/12.2.0
module load openmpi/4.1.5_gnu-12.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"
BIN=${ROOT}/build_mpi/Test_dwf_freeprec_claude
GRID=${GRID:-16.16.16.16}
CONFIG=${CONFIG:?set CONFIG=<NERSC config>}
OPS=${OPS:-}  # additive op list cgne,m0,m1 (empty = all)
# OPS-tagged log so an --ops rerun (e.g. OPS=m1) does NOT overwrite the existing full freeprec log.
if [ -n "${OPS}" ]; then
  OPSTAG=$(echo "${OPS}" | tr ',' '-')
  OUTLOG=${OUTLOG:-${LOGDIR}/freeprec_$(basename "${CONFIG}")_${OPSTAG}_claude.log}
else
  OUTLOG=${OUTLOG:-${LOGDIR}/freeprec_$(basename "${CONFIG}")_claude.log}
fi

export OMP_NUM_THREADS=${NSLOTS:-16}
export OPENBLAS_NUM_THREADS=${NSLOTS:-16}

echo "=== freeprec $(date)  host ${HOSTNAME:-?}  NSLOTS=${NSLOTS:-?}  GRID=${GRID}  CONFIG=${CONFIG} ==="
if [ ! -x "${BIN}" ]; then echo "ERROR: freeprec binary missing ${BIN}"; exit 1; fi
if [ ! -f "${CONFIG}" ]; then echo "ERROR: config missing ${CONFIG}"; exit 1; fi

mpirun -np 1 "${BIN}" --grid "${GRID}" --mpi 1.1.1.1 --threads "${NSLOTS:-16}" --config "${CONFIG}" ${OPS:+--ops ${OPS}} 2>&1 | tee "${OUTLOG}"
echo "freeprec exit = ${PIPESTATUS[0]}   $(date)"
echo "log -> ${OUTLOG}"
