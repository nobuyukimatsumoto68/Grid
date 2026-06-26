#!/bin/bash
# run_disc_mixedprec_tuo_claude.sh
# Driver for the MIXED-PRECISION disc binary over the HEAVIER ensembles m=0.1/0.3/0.4
# -- gap-fills each existing obsdir (~1.4x over double, no eigensolve overhead, same
# estimator, self-skips done configs). This is the right tool for heavy mass where
# deflation does NOT pay (see the validation timings: defl was eigensolve-overhead
# bound at these masses). The LIGHT ensembles (m=0.01, 0.05) go to the defl/LMA
# program. Build first: bash build_disc_speedup_claude.sh. Run from a flux node.

date; hostname
basedir=$(pwd)
source env.sh
builddir=/usr/workspace/lsd/matsumoto5/su4_32c/build
script=submit_disc_mixedprec_tuolumne_claude.sh

masses=(0.1     0.3     0.4)
massstrs=(0p1000 0p3000 0p4000)
betas=(10.865   11.035  11.045)
betastrs=(10p865 11p035 11p045)

jmax=${#masses[@]}
for((j=0;j<$jmax;j++)); do
    cd ${basedir}
    mass=${masses[$j]}; massstr=${massstrs[$j]}
    beta=${betas[$j]};  betastr=${betastrs[$j]}
    cfgfilename="conf_nc4nf1_2448_b${betastr}_m${massstr}"
    latdir="/p/lustre5/matsumoto5/conf_nc4nf1_2448/${cfgfilename}"
    obsdir="/p/lustre5/matsumoto5/obs_nc4nf1_2448/obs_nc4nf1_2448_b${betastr}_m${massstr}"

    rundir=$obsdir; mkdir -p ${rundir}; cp $script $rundir; cd ${rundir}
    jobname="discmp_${betastr}_${massstr}"
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
