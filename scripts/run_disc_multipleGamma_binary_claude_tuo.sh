#!/bin/bash

date; hostname

builddir=/usr/workspace/lsd/matsumoto5/su4_32c/build
make -C ${builddir}/examples disc_multipleGamma_binary_claude

mass=0.4
massstr=0p4000

beta=11.045
betastr=11p045

latdir="/p/lustre5/matsumoto5/conf_nc4nf1_2448/conf_nc4nf1_2448_b${betastr}_m${massstr}"
obsdir="/p/lustre5/matsumoto5/obs_nc4nf1_2448/obs_nc4nf1_2448_b${betastr}_m${massstr}"

########
script=submit_disc_tuolumne.sh

rundir=$obsdir
mkdir -p ${rundir}
cp $script $rundir
cd ${rundir}


if [ "$#" -eq 1 ]; then
    flux batch --dependency=afterany:$1 --env=builddir=$builddir --env=mass=$mass --env=beta=$beta --env=latdir=$latdir --env=obsdir=$obsdir $script
else
    flux batch --env=builddir=$builddir --env=mass=$mass --env=beta=$beta --env=latdir=$latdir --env=obsdir=$obsdir $script
fi
