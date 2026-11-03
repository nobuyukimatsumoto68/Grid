#!/bin/bash
# submit_wflow_q_tuolumne_claude.sh
# FLUX batch launcher for Grid's native Wilson-flow program HMC/ComputeWilsonFlow (gluonic topological charge
# Q, clover + 5Li, and the flow energy density) on one config range of one 24c SU(4) ensemble.
# Plan: Grid/scripts/wflow_topo_q_impl_plan_claude.md. Wilson flow / RK3: M. Luescher, JHEP 08 (2010) 071,
# arXiv:1006.4518.
#
# ComputeWilsonFlow reads input.xml from the CWD, so this script writes input.xml into RUNDIR and runs there.
# flow_type is left EMPTY on purpose: "Wilson" would select WilsonGaugeAction(3.0) (SU(3) normalization,
# flow time x 3/4 for Nc = 4); empty keeps the library default WilsonGaugeAction(Nc) (correct).
# A missing config file aborts the binary's loop -> START..END step SKIP must only hit existing files.
#
# Env (passed by run_wflow_q_tuo_claude.sh or on the command line):
#   CONF_PATH    dir of the NERSC configs                 CONF_PREFIX  e.g. conf_nc4nf1_2448_b11p045_m0p4000_lat
#   START END SKIP  config range                          RUNDIR       output dir (input.xml + log)
#   STEPS (600) STEP_SIZE (0.01) MEAS (20)                 BIN          ComputeWilsonFlow binary
# USER submits. Example (debug, one config):
#   CONF_PATH=/p/lustre5/matsumoto5/conf_nc4nf1_2448/conf_nc4nf1_2448_b11p045_m0p4000 \
#   CONF_PREFIX=conf_nc4nf1_2448_b11p045_m0p4000_lat START=1000 END=1000 SKIP=20 \
#   RUNDIR=/p/lustre5/matsumoto5/wflow_nc4nf1_2448/debug_b11p045_m0p4000 \
#   flux batch -q pdebug -t 30m submit_wflow_q_tuolumne_claude.sh
#FLUX: -t 240m
#FLUX: --output=wflow_q_{{id}}
#FLUX: -q pbatch
#FLUX: -N 1
#FLUX: -n 4
#FLUX: -g 1
#FLUX: --exclusive

set -u

Lattice="24.24.24.48"
MpiGrid="1.1.2.2"

STEPS=${STEPS:-600}
STEP_SIZE=${STEP_SIZE:-0.01}
MEAS=${MEAS:-20}
BIN=${BIN:-/usr/workspace/lsd/matsumoto5/su4_32c/build/HMC/ComputeWilsonFlow}
MAXTAU=$(awk -v n="${STEPS}" -v e="${STEP_SIZE}" 'BEGIN{printf "%.4f", n*e}')

date
hostname
export FASTLOAD_VERBOSE=1
export SPINDLE_FLUXOPT=off
source /usr/workspace/lsd/matsumoto5/su4_32c/env.sh

if [ ! -x "${BIN}" ]; then
    echo "ERROR: binary not found: ${BIN} (make -C /usr/workspace/lsd/matsumoto5/su4_32c/build/HMC ComputeWilsonFlow)" >&2
    exit 1
fi
if [ ! -f "${CONF_PATH}/${CONF_PREFIX}.${START}" ]; then
    echo "ERROR: first config missing: ${CONF_PATH}/${CONF_PREFIX}.${START}" >&2
    exit 1
fi

mkdir -p "${RUNDIR}"
cd "${RUNDIR}" || exit 1

cat > input.xml <<EOF
<?xml version="1.0"?>
<root>
  <WilsonFlow>
    <steps>${STEPS}</steps>
    <step_size>${STEP_SIZE}</step_size>
    <meas_interval>${MEAS}</meas_interval>
    <maxTau>${MAXTAU}</maxTau>
    <meas_interval_density>${MEAS}</meas_interval_density>
    <flow_type></flow_type>
    <path>./</path>
  </WilsonFlow>
  <Configurations>
    <conf_path>${CONF_PATH}</conf_path>
    <conf_prefix>${CONF_PREFIX}</conf_prefix>
    <conf_smr_prefix>${CONF_PREFIX}_smr</conf_smr_prefix>
    <rng_prefix>${CONF_PREFIX%_lat}_rng</rng_prefix>
    <StartConfiguration>${START}</StartConfiguration>
    <EndConfiguration>${END}</EndConfiguration>
    <Skip>${SKIP}</Skip>
  </Configurations>
</root>
EOF
echo "==== input.xml ===="
cat input.xml

OPTIONS="--decomposition --comms-concurrent --comms-overlap --debug-mem --shm 2048 --shm-mpi 1"
PARAMS=" --grid ${Lattice} --mpi ${MpiGrid} --threads 8 --accelerator-threads 8 ${OPTIONS}"

LOG=wflow_q_${CONF_PREFIX}_${START}_${END}_claude.log
echo "--start " $(date) $(date +%s)
flux run -N 1 --tasks-per-node=4 --verbose --exclusive --setopt=mpibind=verbose:1 \
    ${BIN} ${PARAMS} 2>&1 | tee ${LOG}
echo "--end " $(date) $(date +%s)
