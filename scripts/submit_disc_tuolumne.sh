#!/bin/bash
#FLUX: -t 360m
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

flux run -N 8 --tasks-per-node=4 --verbose --exclusive --setopt=mpibind=verbose:1 $APP $APPPARAMS $PARAMS

echo "--end " `date` `date +%s`
