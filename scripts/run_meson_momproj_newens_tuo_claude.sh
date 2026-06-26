#!/bin/bash
#
# Driver (run from workdir) for the momentum-projected CONNECTED meson correlator
# on the NEW light-mass ensembles (mass 0.01 @ beta 10.8, mass 0.05 @ beta 10.84)
# in conf_nc4nf1_2448. Same logic as run_meson_momproj_tuo_claude.sh, just a
# different (index-aligned) ensemble list; reuses the same flux batch launcher
# submit_meson_momproj_tuolumne_claude.sh.
#
# Output h5 land next to the disc traces in obs_nc4nf1_2448_b<betastr>_m<massstr>/.
# The binary self-skips COMPLETE outputs (all channel keys) and recomputes
# partial ones, so re-running this driver resumes.
#
# These ensembles are saved at config interval 4 (much denser than the older
# set). The launcher processes the native interval by default; export STRIDE
# (a multiple of 4) to thin. NOTE: conn and disc MUST use the SAME stride so
# they combine as 2D - C.

date; hostname

basedir=$(pwd)

source env.sh
# Build the meson binary with the incremental in-place Makefile_claude (only
# recompiles when the .cc changed; original Makefile rebuilt every time).
make -C /usr/workspace/lsd/matsumoto5/su4_32c/Grid_sdm_build/src/gauge_gen_Nc4 -f Makefile_claude baryons_0000_dirac_claude
BIN=/usr/workspace/lsd/matsumoto5/su4_32c/Grid_sdm_build/src/gauge_gen_Nc4/baryons_0000_dirac_claude

# Domain-wall height (fixed for this ensemble set).
M5=1.5

# New ensembles to process (index-aligned arrays).
masses=(0.01    0.05)
massstrs=(0p0100 0p0500)
betastrs=(10p800 10p840)

# script=submit_meson_momproj_tuolumne_claude.sh   # old per-config (one flux run/config)
script=submit_meson_momproj_loop_claude.sh         # binary internal --dir loop (no re-skip tax)

jmax=${#masses[@]}
for((j=0;j<$jmax;j++))
do
    cd ${basedir}

    mass=${masses[$j]}
    massstr=${massstrs[$j]}
    betastr=${betastrs[$j]}

    cfgfilename="conf_nc4nf1_2448_b${betastr}_m${massstr}"
    latdir="/p/lustre5/matsumoto5/conf_nc4nf1_2448/${cfgfilename}"
    obsdir="/p/lustre5/matsumoto5/obs_nc4nf1_2448/obs_nc4nf1_2448_b${betastr}_m${massstr}"

    rundir=$obsdir
    mkdir -p ${rundir}
    cp $script $rundir
    cd ${rundir}

    # Per-ensemble job name so resubmissions can find their predecessor.
    jobname="conn_${betastr}_${massstr}"

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
    flux batch --job-name=${jobname} ${deps} --env=BIN=$BIN --env=M5=$M5 --env=mass=$mass --env=cfgfilename=$cfgfilename --env=latdir=$latdir --env=obsdir=$obsdir $script
done
