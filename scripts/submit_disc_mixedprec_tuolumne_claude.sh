#!/bin/bash
# submit_disc_mixedprec_tuolumne_claude.sh
# FLUX batch launcher for the MIXED-PRECISION disc binary
# (disc_multipleGamma_binary_mixedprec_claude): ~1.4x over plain double, NO
# eigensolve overhead -- the right tool for HEAVIER masses (m>=0.1) where there are
# no low modes worth deflating. SAME stochastic estimator as the original disc
# binary (identical source/dilution/contraction) -> gap-fills the SAME obsdir
# consistently. Driver run_disc_mixedprec_tuo_claude.sh passes mass/beta/latdir/
# obsdir via env. USER submits.
#FLUX: -t 480m
#FLUX: --output=disc_mp_{{id}}
#FLUX: -q pbatch
#FLUX: -N 8
#FLUX: -n 32
#FLUX: -g 1
#FLUX: --exclusive

Lattice="24.24.24.48"
MpiGrid="2.2.2.4"

date; hostname
export FASTLOAD_VERBOSE=1
export SPINDLE_FLUXOPT=off

source /usr/workspace/lsd/matsumoto5/su4_32c/env.sh
APP="${builddir}/examples/disc_multipleGamma_binary_mixedprec_claude"

export INNER_TOL=1e-4    # mixed-prec inner single-CG tol (outer stays 1e-8)

OPTIONS="--decomposition --comms-concurrent --comms-overlap --debug-mem --shm 2048 --shm-mpi 1"
APPPARAMS="--mass ${mass} --beta ${beta} --dir ${latdir} --obsdir ${obsdir}"
PARAMS=" --grid ${Lattice} --mpi ${MpiGrid} --threads 8 --accelerator-threads 8 ${OPTIONS}"

echo "--start " $(date) $(date +%s)

# graceful wall blocker (same scheme as submit_disc_tuolumne.sh)
BLOCKER_OVERHEAD=${BLOCKER_OVERHEAD:-300}
export DISC_TPT_SECONDS=${DISC_TPT_SECONDS:-1200}
TIMELEFT=$(flux job timeleft 2>/dev/null)
case "${TIMELEFT}" in ''|*[!0-9.]*) TIMELEFT=0 ;; esac
TIMELEFT=${TIMELEFT%.*}
if [ "${TIMELEFT}" -gt 0 ]; then
    export DISC_DEADLINE_EPOCH=$(( $(date +%s) + TIMELEFT - BLOCKER_OVERHEAD ))
    echo "blocker: timeleft=${TIMELEFT}s -> DISC_DEADLINE_EPOCH=${DISC_DEADLINE_EPOCH}"
fi

flux run -N 8 --tasks-per-node=4 --verbose --exclusive --setopt=mpibind=verbose:1 \
    $APP $APPPARAMS $PARAMS

echo "--end " $(date) $(date +%s)
