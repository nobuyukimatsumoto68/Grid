#!/bin/bash
#
# Driver (run from workdir) for the residual-mass (mres) measurement on the two LIGHT
# 24^3x48 ensembles (b10p800_m0p0100, b10p840_m0p0500 -- same beta/mass as the 32c
# production point), via Mobius_mesons_xt. Modeled on run_meson_momproj_tuo_claude.sh.
# Builds the binary on the login node, then flux-batches submit_mres_tuolumne_claude.sh
# per ensemble; each job processes the LAST 20 saved configs and writes mres.<conf>.h5
# into the obs dir. Ls=16 (hardcoded in the binary), M5=1.5, valence mass = sea mass.
#
# Analysis: mres(t) = sum_t PJ5q_t / sum_t G5_G5_t, plateau at large t.

date; hostname

basedir=$(pwd)

source env.sh
# Build the mres binary from its own Makefile (pattern rule -> ./bin/).
make -C /usr/workspace/lsd/matsumoto5/su4_32c/Grid_sdm_build/src/gauge_gen_Nc4 Mobius_mesons_xt_claude
BIN=/usr/workspace/lsd/matsumoto5/su4_32c/Grid_sdm_build/src/gauge_gen_Nc4/bin/Mobius_mesons_xt_claude

# Domain-wall height (fixed for this ensemble set).
M5=1.5

# Ensembles to process (index-aligned arrays) -- the two light ensembles.
masses=(0.01     0.05)
massstrs=(0p0100 0p0500)
betastrs=(10p800 10p840)

script=submit_mres_tuolumne_claude.sh

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

    # auto-chain (mirrors conn/disc): queue BEHIND any still-active same-name mres job so a
    # re-run never races. Optional explicit predecessor: pass a jobid as $1.
    jobname=mres_${betastr}_${massstr}
    deps=""
    for id in $(flux jobs --filter=pending,running --no-header --format="{id.dec} {name}" 2>/dev/null | awk -v n="$jobname" '$2==n{print $1}'); do
        deps="${deps} --dependency=afterany:${id}"
    done
    if [ "$#" -eq 1 ]; then
        deps="${deps} --dependency=afterany:$1"
    fi

    echo "submitting mres ${cfgfilename} as ${jobname}${deps:+ (deps:${deps# })}"
    flux batch --job-name=${jobname} ${deps} --env=BIN=$BIN --env=M5=$M5 --env=mass=$mass --env=cfgfilename=$cfgfilename --env=latdir=$latdir --env=obsdir=$obsdir $script
done
