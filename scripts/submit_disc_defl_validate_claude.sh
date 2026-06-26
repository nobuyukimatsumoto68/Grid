#!/bin/bash
# submit_disc_defl_validate_claude.sh
# pdebug single-config VALIDATION + TIMING run for the SPEED disc binary
# (disc_multipleGamma_binary_defl_claude). The driver run_disc_defl_validate_claude.sh
# sets up a SCRATCH latdir holding ONE config + a SCRATCH obsdir (which already holds
# the old reference traces as traces.<gam>.<conf>.ref), then passes mass/beta/latdir/
# obsdir here. The defl binary auto-detects the single config, computes traces.<gam>.
# <conf> into the scratch obsdir, and prints the per-config time (eigensolve + solve).
# Compare traces.<gam>.<conf> vs traces.<gam>.<conf>.ref afterwards (must agree ~1e-7).
# Production / scratch dirs are NEVER overwritten by this script.
#FLUX: -t 60m
#FLUX: --output=disc_defl_val_{{id}}
#FLUX: -q pdebug
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
APP="${builddir}/examples/disc_multipleGamma_binary_defl_claude"

# eigensolve + solver knobs (same as production, see disc_tuning_routine_claude.md)
export NEV=150
export NSTOP=150
export NK=150
export NM=240
export INV_TOL=1e-5
export INNER_TOL=1e-4
export MAXPATCH=1000
export ERESID=1e-5
export MAXITER=300

OPTIONS="--decomposition --comms-concurrent --comms-overlap --debug-mem --shm 2048 --shm-mpi 1"
APPPARAMS="--mass ${mass} --beta ${beta} --dir ${latdir} --obsdir ${obsdir}"
PARAMS=" --grid ${Lattice} --mpi ${MpiGrid} --threads 8 --accelerator-threads 8 ${OPTIONS}"

echo "--start " $(date) $(date +%s)
echo "validate: mass=${mass} beta=${beta} latdir=${latdir} obsdir=${obsdir}"
flux run -N 8 --tasks-per-node=4 --verbose --exclusive --setopt=mpibind=verbose:1 \
    $APP $APPPARAMS $PARAMS
echo "--end " $(date) $(date +%s)
