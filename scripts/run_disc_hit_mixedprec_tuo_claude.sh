#!/bin/bash
# run_disc_hit_mixedprec_tuo_claude.sh
# Driver for an ADDITIONAL noise hit (DISC_HIT, default 1) of the disc loops on the HEAVY
# ensembles m=0.1/0.2/0.3/0.4 with the mixed-precision disc binary
# (Grid/examples/disc_second_hit_impl_plan_claude.md). Side-by-side copy of
# run_disc_mixedprec_tuo_claude.sh (hit 0, left intact). The binary seeds the noise with
# cfg_path + "_hit<N>" and writes into <obsdir>_hit<N>/ itself; obsdir passed here is the
# hit-0 dir (never written). Self-skips done configs -> re-run to extend / resume.
# Usage (tuolumne login, from su4_32c/Grid/scripts, after build_disc_hit_claude.sh):
#   bash run_disc_hit_mixedprec_tuo_claude.sh            # all four ensembles, hit 1
#   ONLY=3 bash run_disc_hit_mixedprec_tuo_claude.sh     # m0.4 only (index into masses)
#   DISC_HIT=2 bash run_disc_hit_mixedprec_tuo_claude.sh # a third hit
# Optional explicit predecessor jobid as $1. Claude does NOT submit -- run this yourself.

date; hostname
basedir=$(pwd)
source /usr/workspace/lsd/matsumoto5/su4_32c/env.sh
builddir=/usr/workspace/lsd/matsumoto5/su4_32c/build
script=submit_disc_mixedprec_tuolumne_claude.sh

DISC_HIT=${DISC_HIT:-1}
ONLY=${ONLY:-}               # empty = all; 0 = m0.1, 1 = m0.2, 2 = m0.3, 3 = m0.4

masses=(0.1     0.2     0.3     0.4)
massstrs=(0p1000 0p2000 0p3000 0p4000)
betas=(10.865   10.99   11.035  11.045)
betastrs=(10p865 10p990 11p035 11p045)

if [ "${DISC_HIT}" -lt 1 ]; then
    echo "ERROR: DISC_HIT must be >= 1 (hit 0 = run_disc_mixedprec_tuo_claude.sh)" >&2
    exit 1
fi

jmax=${#masses[@]}
for((j=0;j<$jmax;j++)); do
    if [ -n "${ONLY}" ] && [ "${ONLY}" != "${j}" ]; then
        continue
    fi
    cd ${basedir}
    mass=${masses[$j]}
    massstr=${massstrs[$j]}
    beta=${betas[$j]}
    betastr=${betastrs[$j]}
    cfgfilename="conf_nc4nf1_2448_b${betastr}_m${massstr}"
    latdir="/p/lustre5/matsumoto5/conf_nc4nf1_2448/${cfgfilename}"
    obsdir="/p/lustre5/matsumoto5/obs_nc4nf1_2448/obs_nc4nf1_2448_b${betastr}_m${massstr}"

    # job logs go into the hit-N dir (the binary writes its traces there too)
    rundir=${obsdir}_hit${DISC_HIT}
    mkdir -p ${rundir}
    cp ${script} ${rundir}
    cd ${rundir}
    jobname="discmp_hit${DISC_HIT}_${betastr}_${massstr}"
    deps=""
    for id in $(flux jobs --filter=pending,running --no-header --format="{id.dec} {name}" 2>/dev/null | awk -v n="$jobname" '$2==n{print $1}'); do
        deps="${deps} --dependency=afterany:${id}"
    done
    if [ "$#" -eq 1 ]; then
        deps="${deps} --dependency=afterany:$1"
    fi

    echo "submitting ${cfgfilename} hit ${DISC_HIT} as ${jobname}${deps:+ (deps:${deps# })}"
    flux batch --job-name=${jobname} ${deps} \
        --env=builddir=$builddir --env=mass=$mass --env=beta=$beta \
        --env=latdir=$latdir --env=obsdir=$obsdir --env=DISC_HIT=${DISC_HIT} $script
done
