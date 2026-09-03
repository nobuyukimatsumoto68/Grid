#!/bin/bash -l
# BU SCC SGE batch: frame-flow-TIME scan for ONE config on a GPU (Direction 1b Stage 1). Runs the merged
# GPU flowscan binary directly (comms=none -> NO mpirun). One config/job (the WRAPPER loops configs). 12h.
# 16^4 Ls8 FGMRES R=256 ~26 GB -> gpu_c=8.0 (Ampere+, >=40 GB) so it avoids a 16 GB V100.
#
# Submit (usually via the wrapper with JOBSCRIPT= this file):
#   qsub -v CONFIG=...,GRID=16.16.16.16,NSTEPS=58-73-...,T0=2.91,TOL=1e-6,OPS=cgne-m0-m1 grid_flowscan_gpu_qsub_claude.sh
# Watch:  qstat -u $USER ; tail -f log/flowscan_<cfg>_claude.log

#$ -P qfe
##$ -M mtsmtnbyk@gmail.com
#$ -N flowscanG
#$ -j y
#$ -o /projectnb/qfe/nmatsum/dwf/log/
##$ -m n
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -pe omp 4

set -u

module load cuda/12.8
module load gcc/13.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"
BIN=${BIN:-${ROOT}/build_merged/Test_dwf_flowscan_claude}
GRID=${GRID:-16.16.16.16}
CONFIG=${CONFIG:?set CONFIG=<NERSC config>}
# NSTEPS and OPS are HYPHEN-joined in -v (SGE splits values on commas) -> convert back to commas.
NSTEPS=${NSTEPS:?set NSTEPS=<hyphen list, e.g. 58-73-87>}
NSTEPS_C=$(echo "${NSTEPS}" | tr '-' ',')
T0=${T0:-1.0}
TOL=${TOL:-1e-6}
OPS=${OPS:-cgne-m0-m1}
OPS_C=$(echo "${OPS}" | tr '-' ',')
FLOWS=${FLOWS:-wilson}
FLOWS_C=$(echo "${FLOWS}" | tr '-' ',')
TAG=${TAG:-$(basename "${CONFIG}")}
OUTLOG=${OUTLOG:-${LOGDIR}/flowscan_${TAG}_claude.log}

export OMP_NUM_THREADS=${NSLOTS:-4}

echo "=== flowscanG $(date)  host ${HOSTNAME:-?}  GRID=${GRID}  CONFIG=${CONFIG} ==="
echo "    NSTEPS=${NSTEPS_C}  T0=${T0}  TOL=${TOL}  OPS=${OPS_C}"
if [ ! -x "${BIN}" ]; then echo "ERROR: flowscan GPU binary missing ${BIN}"; exit 1; fi
if [ ! -f "${CONFIG}" ]; then echo "ERROR: config missing ${CONFIG}"; exit 1; fi

# comms=none GPU build -> run the binary directly (no mpirun).
"${BIN}" --grid "${GRID}" --mpi 1.1.1.1 --accelerator-threads 8 \
         --config "${CONFIG}" --ops "${OPS_C}" --frame_flows "${FLOWS_C}" \
         --flow_nsteps "${NSTEPS_C}" --t0 "${T0}" --solve_tol "${TOL}" 2>&1 | tee "${OUTLOG}"
echo "flowscanG exit = ${PIPESTATUS[0]}   $(date)"
echo "log -> ${OUTLOG}"
