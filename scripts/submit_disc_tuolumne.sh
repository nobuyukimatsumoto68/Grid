#!/bin/bash
#FLUX: -t 480m
#FLUX: --output=disc_{{id}}
#FLUX: -q pbatch
#FLUX: -N 8
#FLUX: -n 32
#FLUX: -g 1
#FLUX: --exclusive

Lattice="24.24.24.48"
MpiGrid="2.2.2.4"

### four gpus per node
date
here=`pwd`

# To verify fastload2 is loading or not, set env variable FASTLOAD_VERBOSE=1 and rank 0 (or serial tasks) will print out something of the form if fastload2 is being used:
# email notice on 10/30/24
export FASTLOAD_VERBOSE=1
# but causing the issue?
#export FLUX_FASTLOAD=off

# email notice on 12/11/24
export SPINDLE_FLUXOPT=off


### Launch parallel executable
source /usr/workspace/lsd/matsumoto5/su4_32c/env.sh
APP="${builddir}/examples/disc_multipleGamma_binary_claude"

echo "--start " `date` `date +%s`

OPTIONS="--decomposition --comms-concurrent --comms-overlap --debug-mem  --shm 2048 --shm-mpi 1"
APPPARAMS="--mass ${mass} --beta ${beta} --dir ${latdir} --obsdir ${obsdir}"
PARAMS=" --grid ${Lattice} --mpi ${MpiGrid} --threads 8 --accelerator-threads 8 ${OPTIONS}"

# ----------------- graceful wall-time blocker -----------------
# The disc binary loops ALL configs inside this single flux run, so it cannot be
# stopped from bash between configs. Instead we pass it the wall deadline (epoch
# s) and it stops cleanly before a config it cannot finish+save -- it measures
# per-config wall time itself, and DISC_TPT_SECONDS bootstraps the first-config
# guard (measured ~950s; generous default below). The binary self-skips completed
# configs, so resubmit continues. If timeleft is unavailable, no deadline is set
# and the binary runs all configs (old behavior).
BLOCKER_OVERHEAD=${BLOCKER_OVERHEAD:-300}     # final write + teardown margin (s)
export DISC_TPT_SECONDS=${DISC_TPT_SECONDS:-1200}  # generous bootstrap estimate (s)
TIMELEFT=$(flux job timeleft 2>/dev/null)
case "${TIMELEFT}" in
  ''|*[!0-9.]*) TIMELEFT=0 ;;
esac
TIMELEFT=${TIMELEFT%.*}
if [ "${TIMELEFT}" -gt 0 ]; then
    export DISC_DEADLINE_EPOCH=$(( $(date +%s) + TIMELEFT - BLOCKER_OVERHEAD ))
    echo "blocker: timeleft=${TIMELEFT}s -> DISC_DEADLINE_EPOCH=${DISC_DEADLINE_EPOCH} (bootstrap TPT=${DISC_TPT_SECONDS}s)"
fi
# --------------------------------------------------------------

flux run -N 8 --tasks-per-node=4 --verbose --exclusive --setopt=mpibind=verbose:1 $APP $APPPARAMS $PARAMS

echo "--end " `date` `date +%s`
