#!/bin/bash
# run_wflow_q_h5_tuo_claude.sh
# Driver, round 2 (h5 per config, 4 single-rank flows per node; plan Grid/scripts/wflow_topo_q_impl_plan_claude.md).
# For each stream (tag, conf dir, start, step [, end]) the existing configs on the start/step grid are cut into
# ranges of NPER configs (one 1-node job per range, 4 processes share it). A range whose configs all have
# OUTDIR/wflow.<conf>.h5 is skipped; otherwise it is (re)submitted -- the binary skips finished configs, so a
# rerun only does what is missing. OUTDIR = /p/lustre5/matsumoto5/wflow_nc4nf1_2448_h5/<tag>/ (orig + cont share
# it: config numbers do not overlap). Side-by-side with the round-1 run_wflow_q_tuo_claude.sh (kept).
# Usage (tuolumne login, from su4_32c/Grid/scripts; after building wflow_q_h5_claude):
#   DRYRUN=1 bash run_wflow_q_h5_tuo_claude.sh
#   bash run_wflow_q_h5_tuo_claude.sh                    # all
#   ONLY=b11p045_m0p4000 bash run_wflow_q_h5_tuo_claude.sh
# Claude does NOT submit -- run this yourself.

set -u

# NPER=${NPER:-60}
NPER=${NPER:-52}              # configs per job: debug 2026-11-04 ~955 s/config per GPU, 4 GPUs -> 13 per GPU in 4h
WALL=${WALL:-240m}
QUEUE=${QUEUE:-pbatch}
ONLY=${ONLY:-}
DRYRUN=${DRYRUN:-0}
OUTTOP=/p/lustre5/matsumoto5/wflow_nc4nf1_2448_h5
ORIG=/p/lustre5/matsumoto5/conf_nc4nf1_2448
CONT=/p/lustre5/matsumoto5/conf_nc4nf1_2448_cont
script=submit_wflow_q_h5_tuolumne_claude.sh

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
    outdir=${OUTTOP}/${tag}
    confs=$(ls ${cdir}/${prefix}.* 2>/dev/null | sed 's/.*\.//' | grep -E '^[0-9]+$' | sort -n | \
            awk -v s="${start}" -v k="${step}" -v e="${end}" '$1 >= s && ($1 - s) % k == 0 && (e == "" || $1 <= e + 0)')
    if [ -z "${confs}" ]; then
        echo "no configs: ${cdir} from ${start} step ${step}"
        continue
    fi
    # ranges of NPER existing configs: "first last" (gaps inside a range are fine, the binary skips them)
    ranges=$(echo "${confs}" | awk -v n="${NPER}" 'NR % n == 1 { a = $1 } { p = $1 } NR % n == 0 { print a, p } END { if (NR % n != 0) print a, p }')
    while read -r a b; do
        todo=$(echo "${confs}" | awk -v a="${a}" -v b="${b}" '$1 >= a && $1 <= b' | while read -r c; do
                   [ -f ${outdir}/wflow.${c}.h5 ] || echo ${c}
               done | wc -l)
        if [ "${todo}" -eq 0 ]; then
            echo "done : ${tag} $(basename ${top}) ${a}..${b}"
            continue
        fi
        echo "range: ${tag} $(basename ${top}) ${a}..${b} step ${step} (${todo} to do) -> ${outdir}"
        if [ "${DRYRUN}" = "1" ]; then
            continue
        fi
        mkdir -p ${outdir}
        flux batch --job-name=wflowh5_${tag} -q ${QUEUE} -t ${WALL} \
            --env=CONF_PATH=${cdir} --env=CONF_PREFIX=${prefix} --env=START=${a} --env=END=${b} \
            --env=SKIP=${step} --env=OUTDIR=${outdir} ${basedir}/${script}
    done <<< "${ranges}"
done
