#!/bin/bash

date; hostname

builddir=/mnt/baracuda_14/grid_claude/build
make -C ${builddir}/examples disc_multipleGamma_binary_claude

rundir=/mnt/baracuda_14/grid_claude/run
mkdir -p ${rundir}
cd ${rundir}

Lattice="16.16.16.8"
# MpiGrid="2.2.1.1"
# MpiGrid="2.1.1.1"
MpiGrid="1.1.1.1"

echo "--start " `date` `date +%s`
TotalTic=`date +%s`

mass=0.4
beta=11.08
latdir="/mnt/baracuda_14/grid_claude/16c"
obsdir="/mnt/baracuda_14/grid_claude/16c_obs"

APP="${builddir}/examples/disc_multipleGamma_binary_claude"
# OPTIONS="--decomposition --comms-concurrent --comms-overlap --shm 2048 --shm-mpi 4"
# PARAMS="--grid ${Lattice} --mpi ${MpiGrid} --threads 20 --accelerator-threads 4 ${OPTIONS}"
OPTIONS="--decomposition"
PARAMS="--grid ${Lattice} --mpi ${MpiGrid} --threads 1 ${OPTIONS}"
APPPARAMS="--mass ${mass} --beta ${beta} --dir ${latdir} --obsdir ${obsdir}"

# mpirun --prefix /mnt/hdd_barracuda/opt/openmpi_cuda/ -np 4 $APP $PARAMS 2>&1 | tee disc_multipleGamma_binary.log
CUDA_VISIBLE_DEVICES=0 mpirun -np 1 $APP $APPPARAMS $PARAMS 2>&1 | tee disc_multipleGamma_binary.log

TotalToc=`date +%s`
echo "--end " `date` `date +%s`

TotalTime=$(( $TotalToc - $TotalTic ))
TotalHours=`echo "$TotalTime / 3600" | bc -l`
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "Total time  $TotalTime [sec] = $TotalHours [h]"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

echo 'Done'
