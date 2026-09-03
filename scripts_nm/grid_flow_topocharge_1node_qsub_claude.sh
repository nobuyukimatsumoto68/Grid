#!/bin/bash -l
# BU SCC SGE batch: flowed topological charge Q(tau) for ONE (or more) NERSC config(s), single node.
# Runs the ALREADY-BUILT topology driver (build_mpi/Test_flowed_topocharge_claude) -- no recompile.
# Iwasaki-action gradient flow (default), fixed-step Luscher RK3. Writes one flowQ_<cfg>_claude.dat
# (cols: tau plaq Q_clover Q_5Li) per config into DATDIR; plot with plot_flowQ_claude.gp.
#
# Submit:  qsub -v CONFIG=/.../ckpoint_lat.40,GRID=16.16.16.16 grid_flow_topocharge_1node_qsub_claude.sh
#   several configs: qsub -v CONFIGDIR=/.../configs_iwasaki_16_b2.6,GRID=16.16.16.16 ...
# Watch:   qstat -u $USER ; ls <DATDIR>/flowQ_*.dat

#$ -P qfe
##$ -M mtsmtnbyk@gmail.com
#$ -N flowQ
#$ -j y
#$ -o /projectnb/qfe/nmatsum/dwf/log/
##$ -m n
#$ -l h_rt=2:00:00
#$ -pe omp 16

set -u

module load gcc/12.2.0
module load openmpi/4.1.5_gnu-12.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
BIN=${ROOT}/build_mpi/Test_flowed_topocharge_claude

GRID=${GRID:-16.16.16.16}
FLOW_ACTION=${FLOW_ACTION:-iwasaki}
FLOW_EPS=${FLOW_EPS:-0.01}
FLOW_NSTEP=${FLOW_NSTEP:-400}        # tau up to eps*nstep = 4 (enough to reach the 16^4 plateau)
MEAS=${MEAS:-20}
CONFIG=${CONFIG:-}
CONFIGDIR=${CONFIGDIR:-}
DATDIR=${DATDIR:-${ROOT}/flowQ_dat}
# Read a decorrelated subset from the EXISTING stream (no new stream): CONFIGDIR mode only.
MINTRAJ=${MINTRAJ:-0}                 # skip configs with trajectory < MINTRAJ (thermalization burn-in)
SKIP=${SKIP:-1}                       # stride: take every SKIPth surviving config (SKIP=1 = all)

export OMP_NUM_THREADS=${NSLOTS:-16}
export OPENBLAS_NUM_THREADS=${NSLOTS:-16}

echo "=== flowQ $(date)  host ${HOSTNAME:-?}  NSLOTS=${NSLOTS:-?}  GRID=${GRID} flow=${FLOW_ACTION} eps=${FLOW_EPS} nstep=${FLOW_NSTEP} ==="
if [ ! -x "${BIN}" ]; then echo "ERROR: driver missing ${BIN}"; exit 1; fi

cfglist=""
if [ -n "${CONFIGDIR}" ]; then
  # numeric-sort by trajectory (dir name has a dot, so key off the number AFTER the last dot),
  # drop configs below MINTRAJ (thermalization), then keep every SKIPth (stride).
  cfglist=$(ls ${CONFIGDIR}/ckpoint_lat.* 2>/dev/null \
            | while read f; do n=${f##*.}; [ "${n}" -ge "${MINTRAJ}" ] 2>/dev/null && echo "${n} ${f}"; done \
            | sort -n \
            | awk -v s="${SKIP}" '(NR-1) % s == 0 {print $2}')
  echo "  CONFIGDIR selection: MINTRAJ=${MINTRAJ} SKIP=${SKIP} -> $(echo ${cfglist} | wc -w) configs"
elif [ -n "${CONFIG}" ]; then
  cfglist="${CONFIG}"
fi
if [ -z "${cfglist}" ]; then echo "no configs: set CONFIG=<file> or CONFIGDIR=<dir>"; exit 0; fi

mkdir -p "${DATDIR}"
for C in ${cfglist}; do
  [ -f "${C}" ] || { echo "skip missing ${C}"; continue; }
  base=$(basename "${C}")
  dat="${DATDIR}/flowQ_${base}_claude.dat"
  if [ -s "${dat}" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "  already flowed ${base} (FORCE=1 to redo); skipping"
    continue
  fi
  echo "+ flow ${C} -> ${dat}"
  mpirun -np 1 "${BIN}" --grid "${GRID}" --mpi 1.1.1.1 --threads "${NSLOTS:-16}" \
         --config "${C}" --flow_action "${FLOW_ACTION}" \
         --flow_eps "${FLOW_EPS}" --flow_nstep "${FLOW_NSTEP}" --meas_interval "${MEAS}" \
         2> "${dat}" 1> "${DATDIR}/flowQ_${base}_stdout.log"
  echo "  wrote ${dat} (rc=$?)"
done
echo "finished $(date)"
