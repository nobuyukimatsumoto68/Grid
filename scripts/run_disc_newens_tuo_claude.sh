#!/bin/bash
#
# Driver (run from workdir) for the DISCONNECTED loop correlator
# (disc_multipleGamma_binary_claude) on the NEW light-mass ensembles
# (mass 0.01 @ beta 10.8, mass 0.05 @ beta 10.84) in conf_nc4nf1_2448.
# Same logic as run_disc_multipleGamma_binary_claude_tuo.sh, but loops the two
# new ensembles via index-aligned arrays; reuses submit_disc_tuolumne.sh.
#
# The disc binary auto-detects the config range from latdir and loops internally
# at the native config interval (= 4 for these ensembles). NOTE: the connected
# side (run_meson_momproj_newens_tuo_claude.sh) MUST use the SAME config set so
# they combine as 2D - C; if you thin one, thin the other identically.

date; hostname

basedir=$(pwd)

source env.sh
builddir=/usr/workspace/lsd/matsumoto5/su4_32c/build
make -C ${builddir}/examples disc_multipleGamma_binary_claude

# New ensembles to process (index-aligned arrays). beta is the float passed to
# the binary; betastr/massstr build the directory names.
masses=(0.01    0.05)
massstrs=(0p0100 0p0500)
betas=(10.8     10.84)
betastrs=(10p800 10p840)

script=submit_disc_tuolumne.sh

jmax=${#masses[@]}
for((j=0;j<$jmax;j++))
do
    cd ${basedir}

    mass=${masses[$j]}
    massstr=${massstrs[$j]}
    beta=${betas[$j]}
    betastr=${betastrs[$j]}

    cfgfilename="conf_nc4nf1_2448_b${betastr}_m${massstr}"
    latdir="/p/lustre5/matsumoto5/conf_nc4nf1_2448/${cfgfilename}"
    obsdir="/p/lustre5/matsumoto5/obs_nc4nf1_2448/obs_nc4nf1_2448_b${betastr}_m${massstr}"

    rundir=$obsdir
    mkdir -p ${rundir}
    cp $script $rundir
    cd ${rundir}

    # Per-ensemble job name so resubmissions can find their predecessor.
    jobname="disc_${betastr}_${massstr}"

    # Auto-chain: depend (afterany) on any ACTIVE (pending/running) job of the
    # same name, so a re-run queues BEHIND a still-running job for this ensemble
    # instead of racing it. Plus an optional explicit dependency: pass a jobid as
    # $1 to the driver. If flux is unreachable the query yields nothing and the
    # job is submitted with no dependency.
    deps=""
    for id in $(flux jobs --filter=pending,running --no-header --format="{id.dec} {name}" 2>/dev/null | awk -v n="$jobname" '$2==n{print $1}'); do
        deps="${deps} --dependency=afterany:${id}"
    done
    if [ "$#" -eq 1 ]; then
        deps="${deps} --dependency=afterany:$1"
    fi

    echo "submitting ${cfgfilename} as ${jobname}${deps:+ (deps:${deps# })}"
    flux batch --job-name=${jobname} ${deps} --env=builddir=$builddir --env=mass=$mass --env=beta=$beta --env=latdir=$latdir --env=obsdir=$obsdir $script
done
