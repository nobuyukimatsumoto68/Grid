#!/bin/bash
#
# Driver: CONTINUATION HMC for the 24^3x48 light ensembles (m0.01 @ b10.8, m0.05 @ b10.84),
# AUTO-CHAINED like run_prod32c_claude.sh (Grid/scripts/hmc_prod24c_impl_plan_claude.md).
#
#   Sungwoo's original streams (READ ONLY, never written):
#     /p/lustre5/matsumoto5/conf_nc4nf1_2448/<cfgname>/<cfgname>_{lat,rng}.<traj>
#   New continuation streams (this driver):
#     ${TOPDIR}/<cfgname>/<cfgname>_{lat,rng}.<traj>   (same prefix -> trajectory numbers continue)
#   Seeding: if the new stream dir has no checkpoint yet, the LATEST lat/rng pair is COPIED
#   (cp -n, never overwrites) from Sungwoo's dir.
#
# One job per stream per run, --job-name=hmc24_<betastr>_<massstr>, queued
# --dependency=afterany BEHIND every still-active same-name job. Each job resumes at runtime
# (submit_hmc_prod24c_tuolumne_claude.sh). Advance a stream one allocation at a time by
# re-running this driver. Optional explicit predecessor: pass a jobid as $1.
#
# Usage (from the su4_32c workdir holding this script, the XML template and the submit script):
#   bash run_prod24c_claude.sh
# Each run = ONE 4h job per stream (graceful wall stop); re-run to extend. Per-stream overall
# targets ntrajs[] = 2x the latest trajectory at continuation start (user 2026-10-06).
# (Claude does NOT submit -- run this yourself.)

set -u

# NTRAJ=${NTRAJ:?set NTRAJ=<overall target trajectory number>}

# masses=(0.01     0.05)
# betas=(10.8      10.84)
# massstrs=(0p0100 0p0500)
# betastrs=(10p800 10p840)
# saveints=(2      4)          # as in Sungwoo's streams (m0.01 every 2 since traj 74; m0.05 every 4)
# ntrajs=(1516     4272)       # 2 x latest at start (758, 2136)
masses=(0.01     0.05     0.1)
betas=(10.8      10.84    10.865)
massstrs=(0p0100 0p0500   0p1000)
betastrs=(10p800 10p840   10p865)
saveints=(2      4        20)  # as in Sungwoo's streams (m0.01 every 2 since traj 74; m0.05 every 4; m0.1 every 20)
ntrajs=(1516     4272     8300) # m0.01/m0.05: 2 x latest at start (758, 2136); m0.1: 4640 + 183 cfgs x 20 (2 x configs)

GAUGE_MULT=10                # gauge-level MD multiplier (user decision 2026-10-06)
# optional env overrides (defaults = production); e.g. a debug-queue test:
#   QUEUE=pdebug WALL=30m WALL_SECONDS=1800 TPT_OVERRIDE=600 TOPDIR=/p/lustre5/matsumoto5/conf_nc4nf1_2448_cont_debug ONLY=0 bash run_prod24c_claude.sh
# WALL=240m
# WALL_SECONDS=14400
WALL=${WALL:-240m}
WALL_SECONDS=${WALL_SECONDS:-14400}
QUEUE=${QUEUE:-pbatch}
ONLY=${ONLY:-}               # empty = all streams; 0 = m0.01 only; 1 = m0.05 only; 2 = m0.1 only
# per-trajectory wall estimate (s) forwarded to the blocker; empty = submit script's own
# (measured from prior logs, else 750 s bootstrap). Needed for short debug walls.
TPT_OVERRIDE=${TPT_OVERRIDE:-}
tptenv=""
if [ -n "${TPT_OVERRIDE}" ]; then
    tptenv="--env=TPT_OVERRIDE=${TPT_OVERRIDE}"
fi
APP_BIN=/usr/workspace/lsd/matsumoto5/su4_32c/Grid_sdm_build/src/gauge_gen_Nc4/bin/dweofa_mobius_HSDM_v5_gmult_claude

SRCTOP=/p/lustre5/matsumoto5/conf_nc4nf1_2448
# TOPDIR=/p/lustre5/matsumoto5/conf_nc4nf1_2448_cont
TOPDIR=${TOPDIR:-/p/lustre5/matsumoto5/conf_nc4nf1_2448_cont}

basedir=$(pwd)
xml=ip_hmc_mobius_24c_claude.xml
script=submit_hmc_prod24c_tuolumne_claude.sh

if [ ! -x "${APP_BIN}" ]; then
    echo "ERROR: binary not found: ${APP_BIN} (run build_dweofa_v5_gmult_claude.sh first)" >&2
    exit 1
fi

jmax=${#masses[@]}
for((j=0;j<$jmax;j++))
do
    if [ -n "${ONLY}" ] && [ "${ONLY}" != "${j}" ]; then
        continue
    fi
    m=${masses[$j]}
    beta=${betas[$j]}
    massstr=${massstrs[$j]}
    betastr=${betastrs[$j]}
    saveint=${saveints[$j]}
    NTRAJ=${ntrajs[$j]}

    cfgname=conf_nc4nf1_2448_b${betastr}_m${massstr}
    srcdir=${SRCTOP}/${cfgname}
    dir=${TOPDIR}/${cfgname}
    mkdir -p ${dir}

    # seed the continuation dir from Sungwoo's latest checkpoint (copy, never overwrite)
    have=$(ls ${dir}/${cfgname}_lat.* 2>/dev/null | wc -l)
    if [ "${have}" -eq 0 ]; then
        seed=$(ls ${srcdir}/${cfgname}_lat.* 2>/dev/null | sed 's/.*\.//' | sort -n | tail -n1)
        if [ -z "${seed}" ] || [ ! -f ${srcdir}/${cfgname}_rng.${seed} ]; then
            echo "ERROR: no lat/rng pair to seed from in ${srcdir}" >&2
            continue
        fi
        cp -n ${srcdir}/${cfgname}_lat.${seed} ${dir}/
        cp -n ${srcdir}/${cfgname}_rng.${seed} ${dir}/
        echo "seeded ${dir} from ${srcdir} traj ${seed}"
    fi

    # pristine template + batch script into the stream dir, fill placeholders
    cp -f ${basedir}/${xml} ${dir}
    cp -f ${basedir}/${script} ${dir}
    cd ${dir}
    sed -i "s/@BETA@/${beta}/" ${xml}
    sed -i "s/@MASS@/${m}/" ${xml}
    sed -i "s/@SAVEINT@/${saveint}/" ${xml}
    sed -i "s/@CFGNAME@/${cfgname}/g" ${xml}

    jobname=hmc24_${betastr}_${massstr}
    deps=""
    for id in $(flux jobs --filter=pending,running --no-header --format="{id.dec} {name}" 2>/dev/null | awk -v n="$jobname" '$2==n{print $1}'); do
        deps="${deps} --dependency=afterany:${id}"
    done
    if [ "$#" -eq 1 ]; then
        deps="${deps} --dependency=afterany:$1"
    fi

    echo "submitting ${cfgname} (target ${NTRAJ}, saveInterval ${saveint}, GAUGE_MULT ${GAUGE_MULT}) as ${jobname}${deps:+ (deps:${deps# })}"
    # flux batch --job-name=${jobname} ${deps} -t ${WALL} --env=NTRAJ_TARGET=${NTRAJ} --env=WALL_SECONDS=${WALL_SECONDS} \
    flux batch --job-name=${jobname} ${deps} -q ${QUEUE} -t ${WALL} --env=NTRAJ_TARGET=${NTRAJ} --env=WALL_SECONDS=${WALL_SECONDS} \
         --env=APP_BIN=${APP_BIN} --env=GAUGE_MULT=${GAUGE_MULT} ${tptenv} ${script}

    cd ${basedir}
done
