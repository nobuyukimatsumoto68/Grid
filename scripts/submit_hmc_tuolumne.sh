#!/bin/bash
#FLUX: -t 720m
#FLUX: --output=32c_{{id}}
#FLUX: -q pbatch
#FLUX: -N 8
#FLUX: -n 32
#FLUX: -g 1
#FLUX: --exclusive


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


# # create xml for CheckpointStart from lasttraj
# mkdir -p xml

XML=ip_hmc_mobius.xml



### Launch parallel executable
source /usr/workspace/lsd/matsumoto5/su4_32c/env.sh
APP=/usr/workspace/lsd/matsumoto5/su4_32c/Grid_sdm_build/src/gauge_gen_Nc4/bin/dweofa_mobius_HSDM_v4

echo "--start " `date` `date +%s`

# OPTIONS="--decomposition --comms-concurrent --comms-overlap --debug-mem  --shm 2048 --shm-mpi 1"
OPTIONS="--decomposition --comms-concurrent --comms-overlap --debug-mem  --shm 2048 --shm-mpi 1"

# PARAMS=" --grid 24.24.24.8 --mpi 4.2.2.1 --threads 8 --accelerator-threads 8 ${OPTIONS} --ParameterFile ${XML}"
PARAMS=" --grid 32.32.32.64 --mpi 2.2.2.4 --threads 8 --accelerator-threads 8 ${OPTIONS} --ParameterFile ${XML}"

flux run -N 8 --tasks-per-node=4 --verbose --exclusive --setopt=mpibind=verbose:1 $APP $PARAMS

echo "--end " `date` `date +%s`
