#!/bin/bash
# run_wflow_q_tuo_claude.sh
# Driver: gluonic topological charge (Wilson flow, Grid HMC/ComputeWilsonFlow) on every config used in the
# 24c meson analysis, original + continuation streams (plan Grid/scripts/wflow_topo_q_impl_plan_claude.md).
# Wilson flow / RK3: M. Luescher, JHEP 08 (2010) 071, arXiv:1006.4518.
#
# For each (stream dir, start, step): list the EXISTING configs start, start+step, ... (the binary aborts on a
# missing file), split them into contiguous runs, cut the runs into chunks of at most NPER configs and submit
# one 1-node job per chunk (submit_wflow_q_tuolumne_claude.sh). A chunk whose log already holds
# "TopologicalCharge5Li <last conf>" is skipped (re-run = resume). Debug 2026-11-03: ~225 s/config (600 steps).
# Usage (tuolumne login, from su4_32c/Grid/scripts):
#   DRYRUN=1 bash run_wflow_q_tuo_claude.sh      # list the chunks, submit nothing
#   bash run_wflow_q_tuo_claude.sh
#   ONLY=b11p045_m0p4000 bash run_wflow_q_tuo_claude.sh
# Claude does NOT submit -- run this yourself.

set -u

NPER=${NPER:-55}                 # configs per 4h job (~225 s/config + margin)
WALL=${WALL:-240m}
QUEUE=${QUEUE:-pbatch}
ONLY=${ONLY:-}
DRYRUN=${DRYRUN:-0}
OUTTOP=/p/lustre5/matsumoto5/wflow_nc4nf1_2448
ORIG=/p/lustre5/matsumoto5/conf_nc4nf1_2448
CONT=/p/lustre5/matsumoto5/conf_nc4nf1_2448_cont
script=submit_wflow_q_tuolumne_claude.sh

# stream table: tag  topdir  start  step  [end]   (end empty = last existing config)
streams=(
  "b10p800_m0p0100 ${ORIG} 4 4"
  "b10p800_m0p0100 ${CONT} 760 4"
  "b10p840_m0p0500 ${ORIG} 4 4 96"
  "b10p840_m0p0500 ${ORIG} 500 4"
  "b10p840_m0p0500 ${CONT} 2140 4"
  "b10p865_m0p1000 ${ORIG} 1000 20"
  "b10p865_m0p1000 ${CONT} 4660 20"
  "b10p990_m0p2000 ${ORIG} 1000 20"
  "b11p035_m0p3000 ${ORIG} 1000 20"
  "b11p045_m0p4000 ${ORIG} 1000 20"
)

basedir=$(pwd)
for line in "${streams[@]}"; do
    read -r tag top start step end <<< "${line}"
    end=${end:-}
    if [ -n "${ONLY}" ] && [ "${ONLY}" != "${tag}" ]; then
        continue
    fi
    cfgname=conf_nc4nf1_2448_${tag}
    cdir=${top}/${cfgname}
    prefix=${cfgname}_lat
    kind=$(basename ${top})
    # existing configs on the start/step grid (optionally capped at end), sorted
    confs=$(ls ${cdir}/${prefix}.* 2>/dev/null | sed 's/.*\.//' | grep -E '^[0-9]+$' | sort -n | \
            awk -v s="${start}" -v k="${step}" -v e="${end}" '$1 >= s && ($1 - s) % k == 0 && (e == "" || $1 <= e + 0)')
    if [ -z "${confs}" ]; then
        echo "no configs: ${cdir} from ${start} step ${step}"
        continue
    fi
    # contiguous runs cut into chunks of <= NPER: print "first last" per chunk
    chunks=$(echo "${confs}" | awk -v k="${step}" -v n="${NPER}" '
        NR == 1 { a = $1; p = $1; c = 1; next }
        { if ($1 != p + k || c >= n) { print a, p; a = $1; c = 0 } p = $1; c++ }
        END { print a, p }')
    while read -r a b; do
        rundir=${OUTTOP}/${kind}/${tag}/${a}_${b}
        log=${rundir}/wflow_q_${prefix}_${a}_${b}_claude.log
        if [ -f "${log}" ] && grep -q "TopologicalCharge5Li ${b} " "${log}"; then
            echo "done : ${kind} ${tag} ${a}..${b}"
            continue
        fi
        echo "chunk: ${kind} ${tag} ${a}..${b} step ${step} -> ${rundir}"
        if [ "${DRYRUN}" = "1" ]; then
            continue
        fi
        mkdir -p ${rundir}
        cd ${rundir}
        flux batch --job-name=wflow_${tag} -q ${QUEUE} -t ${WALL} \
            --env=CONF_PATH=${cdir} --env=CONF_PREFIX=${prefix} --env=START=${a} --env=END=${b} \
            --env=SKIP=${step} --env=RUNDIR=${rundir} ${basedir}/${script}
        cd ${basedir}
    done <<< "${chunks}"
done
