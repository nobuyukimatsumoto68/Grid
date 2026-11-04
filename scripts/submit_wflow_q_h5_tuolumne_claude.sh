#!/bin/bash
# submit_wflow_q_h5_tuolumne_claude.sh
# FLUX batch launcher (round 2, plan Grid/scripts/wflow_topo_q_impl_plan_claude.md): 4 INDEPENDENT single-rank
# Wilson-flow processes on one node (1 GPU each, no MPI comms), wflow_q_h5_claude, sharing the config range
# START..END step SKIP by --part p --nparts 4. One h5 per config in OUTDIR (self-skip -> rerun = resume),
# graceful stop before the wall (WFLOW_DEADLINE_EPOCH from flux job timeleft). Side-by-side with the round-1
# submit_wflow_q_tuolumne_claude.sh (kept).
# Env: CONF_PATH CONF_PREFIX START END SKIP OUTDIR [BIN] [WFLOW_STEPS WFLOW_STEP_SIZE WFLOW_MEAS WFLOW_TPT_SECONDS]
# USER submits (run_wflow_q_h5_tuo_claude.sh, or by hand, e.g. the debug:
#   CONF_PATH=/p/lustre5/matsumoto5/conf_nc4nf1_2448/conf_nc4nf1_2448_b11p045_m0p4000 \
#   CONF_PREFIX=conf_nc4nf1_2448_b11p045_m0p4000_lat START=1000 END=1060 SKIP=20 \
#   OUTDIR=/p/lustre5/matsumoto5/wflow_nc4nf1_2448_h5/debug \
#   flux batch -q pdebug -t 30m submit_wflow_q_h5_tuolumne_claude.sh )
#FLUX: -t 240m
#FLUX: --output=wflow_h5_{{id}}
#FLUX: -q pbatch
#FLUX: -N 1
#FLUX: --exclusive

set -u

Lattice="24.24.24.48"
NPARTS=4
BIN=${BIN:-/usr/workspace/lsd/matsumoto5/su4_32c/Grid_sdm_build/src/gauge_gen_Nc4/wflow_q_h5_claude}
export WFLOW_STEPS=${WFLOW_STEPS:-600}
export WFLOW_STEP_SIZE=${WFLOW_STEP_SIZE:-0.01}
export WFLOW_MEAS=${WFLOW_MEAS:-20}
export WFLOW_TPT_SECONDS=${WFLOW_TPT_SECONDS:-900}

date
hostname
export FASTLOAD_VERBOSE=1
export SPINDLE_FLUXOPT=off
export HDF5_USE_FILE_LOCKING=FALSE
source /usr/workspace/lsd/matsumoto5/su4_32c/env.sh

if [ ! -x "${BIN}" ]; then
    echo "ERROR: binary not found: ${BIN} (make -C .../gauge_gen_Nc4 -f Makefile_claude wflow_q_h5_claude)" >&2
    exit 1
fi
mkdir -p "${OUTDIR}/logs"

BLOCKER_OVERHEAD=${BLOCKER_OVERHEAD:-120}
TIMELEFT=$(flux job timeleft 2>/dev/null)
case "${TIMELEFT}" in ''|*[!0-9.]*) TIMELEFT=0 ;; esac
TIMELEFT=${TIMELEFT%.*}
if [ "${TIMELEFT}" -gt 0 ]; then
    export WFLOW_DEADLINE_EPOCH=$(( $(date +%s) + TIMELEFT - BLOCKER_OVERHEAD ))
    echo "blocker: timeleft=${TIMELEFT}s -> WFLOW_DEADLINE_EPOCH=${WFLOW_DEADLINE_EPOCH}"
fi

OPTIONS="--decomposition --debug-mem --shm 2048"
PARAMS=" --grid ${Lattice} --mpi 1.1.1.1 --threads 8 --accelerator-threads 8 ${OPTIONS}"
APPPARAMS="--dir ${CONF_PATH} --prefix ${CONF_PREFIX} --start ${START} --end ${END} --step ${SKIP} --outdir ${OUTDIR}"

echo "--start " $(date) $(date +%s)
for((p=0;p<NPARTS;p++)); do
    LOG=${OUTDIR}/logs/wflow_h5_${CONF_PREFIX}_${START}_${END}_p${p}_claude.log
    flux run -N 1 -n 1 -g 1 --verbose ${BIN} ${APPPARAMS} --part ${p} --nparts ${NPARTS} ${PARAMS} > ${LOG} 2>&1 &
done
wait
echo "--end " $(date) $(date +%s)
