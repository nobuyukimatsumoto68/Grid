#!/bin/bash -l
# BU SCC SGE batch: flow-TIME scan for ONE config (Direction 1b Stage 1). Runs the freeprec binary in
# --flow_nsteps mode: CGNE once, then Wilson-flow -> Landau -> M0/M1 FGMRES at each tau, printing the
# D_W-apply ratio vs s/t0. One config per job (the WRAPPER loops configs -- NO array jobs). 12h wall.
#
# Submit (usually via grid_freeprec_flowscan_wrapper_claude.sh):
#   qsub -v CONFIG=/.../ckpoint_lat.200,GRID=16.16.16.16,NSTEPS=58,73,87,...,T0=2.91,TOL=1e-6,OPS=m0,m1 \
#        grid_freeprec_flowscan_qsub_claude.sh
# Watch:   qstat -u $USER ; tail -f log/flowscan_<cfg>_claude.log

#$ -P qfe
##$ -M mtsmtnbyk@gmail.com
#$ -N flowscan
#$ -j y
#$ -o /projectnb/qfe/nmatsum/dwf/log/
##$ -m n
#$ -l h_rt=12:00:00
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
# NSTEPS and OPS are transported HYPHEN-joined (SGE -v splits values on commas) -> convert back to commas.
NSTEPS=${NSTEPS:?set NSTEPS=<hyphen list of flow nsteps, e.g. 58-73-87>}
NSTEPS_C=$(echo "${NSTEPS}" | tr '-' ',')
T0=${T0:-1.0}
TOL=${TOL:-1e-6}
OPS=${OPS:-cgne-m0-m1}
OPS_C=$(echo "${OPS}" | tr '-' ',')
TAG=${TAG:-$(basename "${CONFIG}")}
OUTLOG=${OUTLOG:-${LOGDIR}/flowscan_${TAG}_claude.log}

export OMP_NUM_THREADS=${NSLOTS:-16}
export OPENBLAS_NUM_THREADS=${NSLOTS:-16}

echo "=== flowscan $(date)  host ${HOSTNAME:-?}  GRID=${GRID}  CONFIG=${CONFIG} ==="
echo "    NSTEPS=${NSTEPS_C}  T0=${T0}  TOL=${TOL}  OPS=${OPS_C}"
if [ ! -x "${BIN}" ]; then echo "ERROR: freeprec binary missing ${BIN}"; exit 1; fi
if [ ! -f "${CONFIG}" ]; then echo "ERROR: config missing ${CONFIG}"; exit 1; fi

mpirun -np 1 "${BIN}" --grid "${GRID}" --mpi 1.1.1.1 --threads "${NSLOTS:-16}" \
       --config "${CONFIG}" --ops "${OPS_C}" \
       --flow_nsteps "${NSTEPS_C}" --t0 "${T0}" --solve_tol "${TOL}" 2>&1 | tee "${OUTLOG}"
echo "flowscan exit = ${PIPESTATUS[0]}   $(date)"
echo "log -> ${OUTLOG}"
