#!/bin/bash
#
# Driver: RESUME the 32^3x64 SU(4) SDM HEAVY HMC streams m0.1, m0.2, m0.3 up to 100 saved configs (user 2026-11-06),
# AUTO-CHAINED like run_prod32c_claude.sh. The streams were started by run_tuo.sh / submit_hmc_tuolumne.sh
# (dweofa_mobius_HSDM_v4, Force Gradient, saveInterval 20, ckpoint_{lat,rng}.<traj>) in
#   /p/lustre5/matsumoto5/32_64_<m>/beta<beta>m<m>/
# 100 configs at saveInterval 20 (first saved traj 20) = trajectory 2000 (m0.4 is already there; not touched).
#
# Per stream: the stream's OWN last-used XML (ip_hmc_mobius.xml in the stream dir: beta, mass, MDsteps, saveInterval as
# run) is copied to ip_hmc_mobius_claude.xml (the original file is left untouched) and the batch script
# submit_hmc_prod_tuolumne_claude.sh is copied next to it. The batch script resumes at RUNTIME from the latest
# ckpoint_lat.<traj> (CheckpointStart, NoMetropolisUntil 0), caps the trajectories to what fits the wall, and is a
# no-op once the stream reaches NTRAJ. One job per stream per run, queued --dependency=afterany behind every
# still-active job of the same name (job name hmc_<betastr>_<massstr>): re-run this driver to queue the next
# allocation. Optional explicit predecessor: pass a jobid as $1. Claude does NOT submit.
#
# Usage (on tuolumne, from su4_32c/Grid/scripts):
#   bash run_prod32c_heavy_claude.sh            # queue one allocation per stream
#   ONLY=0.3 bash run_prod32c_heavy_claude.sh   # one stream only

masses=(0.1      0.2      0.3)
betas=(10.865    10.99    11.035)
massstrs=(0p1000 0p2000   0p3000)   # used only for the flux --job-name
betastrs=(10p865 10p990   11p035)

NTRAJ=${NTRAJ:-2000}               # overall trajectory target = 100 configs at saveInterval 20
WALL=${WALL:-360m}                 # as the original submit_hmc_tuolumne.sh
WALL_SECONDS=${WALL_SECONDS:-21600}
ONLY=${ONLY:-}

basedir=$(pwd)
xml=ip_hmc_mobius_claude.xml
script=submit_hmc_prod_tuolumne_claude.sh

if [ ! -f "${basedir}/${script}" ]; then
    echo "ERROR: run from the directory holding ${script}" >&2
    exit 1
fi

jmax=${#masses[@]}
for((j=0;j<$jmax;j++))
do
    m=${masses[$j]}
    beta=${betas[$j]}
    massstr=${massstrs[$j]}
    betastr=${betastrs[$j]}
    if [ -n "${ONLY}" ] && [ "${ONLY}" != "${m}" ]; then
        continue
    fi

    dir=/p/lustre5/matsumoto5/32_64_${m}/beta${beta}m${m}
    if [ ! -f "${dir}/ip_hmc_mobius.xml" ]; then
        echo "ERROR: ${dir}/ip_hmc_mobius.xml missing (not an existing stream); skipping" >&2
        continue
    fi
    latest=$(ls ${dir}/ckpoint_lat.* 2>/dev/null | sed 's/.*\.//' | sort -n | tail -n1)
    latest=${latest:-0}
    if [ "${latest}" -ge "${NTRAJ}" ]; then
        echo "32_64_${m}/beta${beta}m${m}: at ${latest} >= ${NTRAJ}, nothing to queue"
        continue
    fi

    cd ${dir}
    # the stream's own last-used XML -> the chained-run XML (original kept)
    cp -f ip_hmc_mobius.xml ${xml}
    cp -f ${basedir}/${script} .
    sed -i "/<StartingType>/{s/ColdStart/CheckpointStart/}" ${xml}
    sed -i "/<StartingType>/{s/HotStart/CheckpointStart/}" ${xml}
    echo "32_64_${m}/beta${beta}m${m}: latest ${latest} -> target ${NTRAJ};" \
         "beta $(grep -oP '(?<=<gauge_beta>)[^<]+' ${xml}) mass $(grep -oP '(?<=<mass>)[^<]+' ${xml})" \
         "MDsteps $(grep -oP '(?<=<MDsteps>)[^<]+' ${xml}) saveInterval $(grep -oP '(?<=<saveInterval>)[^<]+' ${xml})"

    jobname=hmc_${betastr}_${massstr}
    deps=""
    for id in $(flux jobs --filter=pending,running --no-header --format="{id.dec} {name}" 2>/dev/null | awk -v n="$jobname" '$2==n{print $1}'); do
        deps="${deps} --dependency=afterany:${id}"
    done
    if [ "$#" -eq 1 ]; then
        deps="${deps} --dependency=afterany:$1"
    fi
    echo "  submitting as ${jobname}${deps:+ (deps:${deps# })}"
    flux batch --job-name=${jobname} ${deps} -t ${WALL} --env=NTRAJ_TARGET=${NTRAJ} --env=WALL_SECONDS=${WALL_SECONDS} ${script}

    cd ${basedir}
done
