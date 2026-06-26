#!/bin/bash
#
# Driver: PRODUCTION HMC streams for the 32^3x64 SU(4) SDM ensemble, AUTO-CHAINED.
# Two INDEX-ALIGNED streams (beta paired with mass), in the existing run_tuo.sh layout:
#   /p/lustre5/matsumoto5/32_64_<m>/beta<beta>m<m>/ckpoint_{lat,rng}.<traj>
# (default ckpoint prefix -- the batch script's runtime resume reads config_prefix from
# the XML, so it picks up ckpoint_lat.* automatically.)
#
# Auto-chain (mirrors the conn/disc drivers' convention): each stream gets a unique
# --job-name=hmc_<betastr>_<massstr>, and we submit ONE job per stream per run, queued
# --dependency=afterany BEHIND every still-active same-name job. Each job RESUMES at
# runtime from the latest checkpoint and the wall blocker stops it cleanly; once a stream
# reaches NTRAJ the job is a no-op. So you advance a stream ONE allocation at a time by
# re-running this driver (or via /loop) -- new jobs always queue behind the active ones,
# never race. Resume + blocker + target logic live in submit_hmc_prod_tuolumne_claude.sh.
# Optional explicit predecessor: pass a jobid as $1.

masses=(0.01     0.05)
betas=(10.8      10.84)
massstrs=(0p0100 0p0500)   # used only for the per-stream flux --job-name
betastrs=(10p800 10p840)

NTRAJ=100                # overall trajectory target per stream (batch script no-ops past it)
SAVEINT=4                # saveInterval (save a config every 4 trajectories)
WALL=240m                # flux batch wall per job (4 h)
WALL_SECONDS=14400       # same wall in seconds (blocker fallback when flux timeleft missing)

basedir=$(pwd)
Nt=64
xml=ip_hmc_mobius_claude.xml
script=submit_hmc_prod_tuolumne_claude.sh

jmax=${#masses[@]}
for((j=0;j<$jmax;j++))
do
    m=${masses[$j]}
    beta=${betas[$j]}
    massstr=${massstrs[$j]}
    betastr=${betastrs[$j]}

    rundir=/p/lustre5/matsumoto5/32_64_${m}
    dir=${rundir}/beta${beta}m${m}
    mkdir -p $dir

    # copy the pristine template + batch script into the stream dir
    cp -f ${basedir}/$xml $dir
    cp -f ${basedir}/$script $dir
    cd $dir

    # set up the XML; CheckpointStart/StartTrajectory are handled at RUNTIME by the batch
    # script (essential for afterany chaining), so the driver leaves them pristine.
    # config_prefix/rng_prefix left at the default ./ckpoint_lat / ./ckpoint_rng.
    sed -i "/<gauge_beta>/{s/@BETA@/${beta}/}" $xml
    sed -i "/<mass>/{s/0.01/${m}/}" $xml
    sed -i "/<Trajectories>/{s/40/${NTRAJ}/}" $xml
    sed -i "/<saveInterval>/{s/5/${SAVEINT}/}" $xml

    jobname=hmc_${betastr}_${massstr}
    # one job per run, queued BEHIND every still-active same-name job (so re-running this
    # driver advances the stream one allocation at a time without racing). Optional explicit
    # predecessor: pass a jobid as $1. If flux is unreachable the query yields no deps.
    deps=""
    for id in $(flux jobs --filter=pending,running --no-header --format="{id.dec} {name}" 2>/dev/null | awk -v n="$jobname" '$2==n{print $1}'); do
        deps="${deps} --dependency=afterany:${id}"
    done
    if [ "$#" -eq 1 ]; then
        deps="${deps} --dependency=afterany:$1"
    fi

    echo "submitting 32_64_${m}/beta${beta}m${m} as ${jobname}${deps:+ (deps:${deps# })}"
    flux batch --job-name=${jobname} ${deps} -t ${WALL} --env=NTRAJ_TARGET=${NTRAJ} --env=WALL_SECONDS=${WALL_SECONDS} $script

    cd ${basedir}
done
