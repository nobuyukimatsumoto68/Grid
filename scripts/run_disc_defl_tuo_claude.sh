#!/bin/bash
# run_disc_defl_tuo_claude.sh
# Driver (run from a flux node) for the SPEED disc binary
# (disc_multipleGamma_binary_defl_claude) over the HEAVIER ensembles m=0.1/0.3/0.4
# -- it GAP-FILLS each ensemble's existing obsdir (same estimator as the old disc
# binary, self-skips done configs). The two LIGHT ensembles (m=0.01, 0.05) are NOT
# here -- they go to the LMA program in a new directory (different estimator).
#
# Build first: `bash build_disc_speedup_claude.sh` (builds the defl binary).
# Then run this on a flux node; it submits one job/ensemble with afterany
# auto-chain (re-run later to resume after a wall cutoff).

date; hostname
basedir=$(pwd)
source env.sh
builddir=/usr/workspace/lsd/matsumoto5/su4_32c/build

script=submit_disc_defl_tuolumne_claude.sh

# heavier ensembles to GAP-FILL (index-aligned). m=0.2 (b10p990) is already DONE.
masses=(0.1     0.3     0.4)
massstrs=(0p1000 0p3000 0p4000)
betas=(10.865   11.035  11.045)
betastrs=(10p865 11p035 11p045)

jmax=${#masses[@]}
for((j=0;j<$jmax;j++)); do
    cd ${basedir}
    mass=${masses[$j]}; massstr=${massstrs[$j]}
    beta=${betas[$j]}; betastr=${betastrs[$j]}

    cfgfilename="conf_nc4nf1_2448_b${betastr}_m${massstr}"
    latdir="/p/lustre5/matsumoto5/conf_nc4nf1_2448/${cfgfilename}"
    obsdir="/p/lustre5/matsumoto5/obs_nc4nf1_2448/obs_nc4nf1_2448_b${betastr}_m${massstr}"

    rundir=$obsdir
    mkdir -p ${rundir}
    cp $script $rundir
    cd ${rundir}

    # per-ensemble job name; afterany-chain onto any ACTIVE same-name job so a
    # re-run queues behind a still-running one instead of racing it.
    jobname="discdefl_${betastr}_${massstr}"
    deps=""
    for id in $(flux jobs --filter=pending,running --no-header --format="{id.dec} {name}" 2>/dev/null | awk -v n="$jobname" '$2==n{print $1}'); do
        deps="${deps} --dependency=afterany:${id}"
    done
    if [ "$#" -eq 1 ]; then deps="${deps} --dependency=afterany:$1"; fi

    echo "submitting ${cfgfilename} as ${jobname}${deps:+ (deps:${deps# })}"
    flux batch --job-name=${jobname} ${deps} \
        --env=builddir=$builddir --env=mass=$mass --env=beta=$beta \
        --env=latdir=$latdir --env=obsdir=$obsdir $script
done
