#!/bin/bash
# submit_disc_mixedprec_validate_claude.sh
# pdebug single-config timing run for the MIXED-PRECISION disc binary. Driver passes
# mass/beta/latdir/obsdir (scratch single-config dir). Prints "conf <n> took <s>s".
#FLUX: -t 60m
#FLUX: --output=disc_mp_val_{{id}}
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
APP="${builddir}/examples/disc_multipleGamma_binary_mixedprec_claude"
export INNER_TOL=1e-4

OPTIONS="--decomposition --comms-concurrent --comms-overlap --debug-mem --shm 2048 --shm-mpi 1"
APPPARAMS="--mass ${mass} --beta ${beta} --dir ${latdir} --obsdir ${obsdir}"
PARAMS=" --grid ${Lattice} --mpi ${MpiGrid} --threads 8 --accelerator-threads 8 ${OPTIONS}"

echo "--start " $(date) $(date +%s)
echo "mixedprec validate: mass=${mass} beta=${beta} latdir=${latdir} obsdir=${obsdir}"
flux run -N 8 --tasks-per-node=4 --verbose --exclusive --setopt=mpibind=verbose:1 \
    $APP $APPPARAMS $PARAMS
echo "--end " $(date) $(date +%s)
