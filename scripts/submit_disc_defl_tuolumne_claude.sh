#!/bin/bash
# submit_disc_defl_tuolumne_claude.sh
# FLUX batch launcher for the SPEED (deflation+mixed+batched) disc binary
# disc_multipleGamma_binary_defl_claude. SAME stochastic estimator as the original
# disc binary (identical source/dilution/contraction), so it gap-fills the SAME
# obsdir consistently; it just solves faster (eigensolve once/config + 16-RHS
# batched mixed-prec deflated solve). Driver run_disc_defl_tuo_claude.sh passes the
# env vars (builddir, mass, beta, latdir, obsdir). Eigensolve knobs use the m=0.01
# tuning (conservative for heavier mass: lambda_max ~ mass-independent, low-mode
# region milder at heavier mass). USER submits.
#FLUX: -t 480m
#FLUX: --output=disc_defl_{{id}}
#FLUX: -q pbatch
#FLUX: -N 8
#FLUX: -n 32
#FLUX: -g 1
#FLUX: --exclusive

Lattice="24.24.24.48"
MpiGrid="2.2.2.4"

date
hostname
export FASTLOAD_VERBOSE=1
export SPINDLE_FLUXOPT=off

source /usr/workspace/lsd/matsumoto5/su4_32c/env.sh
APP="${builddir}/examples/disc_multipleGamma_binary_defl_claude"

# ---- eigensolve + solver knobs (see disc_tuning_routine_claude.md) ----
export NEV=150
export NSTOP=150
export NK=150
export NM=240
export INV_TOL=1e-5
export INNER_TOL=1e-4
export MAXPATCH=1000
export ERESID=1e-5
export MAXITER=300

echo "--start " $(date) $(date +%s)

OPTIONS="--decomposition --comms-concurrent --comms-overlap --debug-mem --shm 2048 --shm-mpi 1"
APPPARAMS="--mass ${mass} --beta ${beta} --dir ${latdir} --obsdir ${obsdir}"
PARAMS=" --grid ${Lattice} --mpi ${MpiGrid} --threads 8 --accelerator-threads 8 ${OPTIONS}"

# ----------------- graceful wall-time blocker -----------------
# Identical scheme to submit_disc_tuolumne.sh; the binary self-skips completed
# configs (BEFORE the eigensolve) and stops cleanly before a config it cannot
# finish. Per-config time now INCLUDES the once-per-config eigensolve, so the
# bootstrap estimate is larger.
BLOCKER_OVERHEAD=${BLOCKER_OVERHEAD:-300}
export DISC_TPT_SECONDS=${DISC_TPT_SECONDS:-2000}   # bootstrap: solve + eigensolve
TIMELEFT=$(flux job timeleft 2>/dev/null)
case "${TIMELEFT}" in ''|*[!0-9.]*) TIMELEFT=0 ;; esac
TIMELEFT=${TIMELEFT%.*}
if [ "${TIMELEFT}" -gt 0 ]; then
    export DISC_DEADLINE_EPOCH=$(( $(date +%s) + TIMELEFT - BLOCKER_OVERHEAD ))
    echo "blocker: timeleft=${TIMELEFT}s -> DISC_DEADLINE_EPOCH=${DISC_DEADLINE_EPOCH}"
fi
# --------------------------------------------------------------

flux run -N 8 --tasks-per-node=4 --verbose --exclusive --setopt=mpibind=verbose:1 \
    $APP $APPPARAMS $PARAMS

echo "--end " $(date) $(date +%s)
