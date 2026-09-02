#!/bin/bash
# 2024-style submission wrapper (mimics ../../2024_latticeCPN_dev epoch_driver/run_wrapper): loop over a
# set of configs and qsub the freeprec run for each. You run this to SUBMIT; it launches no compute.
# Freeprec runs are independent per config, so these fan out in parallel (no -hold_jid chaining).
#
# Usage (DRYRUN first to preview, then submit):
#   DRYRUN=1 bash grid_freeprec_wrapper_claude.sh            # print the qsub lines, submit nothing
#   bash grid_freeprec_wrapper_claude.sh                     # submit
# Config selection (one of): explicit list, or a dir with MINTRAJ/SKIP:
#   CONFIGS="/.../ckpoint_lat.200 /.../ckpoint_lat.240" bash grid_freeprec_wrapper_claude.sh
#   CONFIGDIR=/.../configs_iwasaki_16_b2.6 MINTRAJ=40 SKIP=2 bash grid_freeprec_wrapper_claude.sh
# Point at a different job (gen/flow) via JOBSCRIPT=... ; pass GRID=... .

set -u

SNM=/projectnb/qfe/nmatsum/dwf/Grid/scripts_nm
JOBSCRIPT=${JOBSCRIPT:-${SNM}/grid_freeprec_run_1node_qsub_claude.sh}
GRID=${GRID:-16.16.16.16}
OPS=${OPS:-}                          # additive op list cgne,m0,m1 forwarded to the run script (empty=all)
DRYRUN=${DRYRUN:-0}

# config selection: CONFIGS (explicit) wins; else CONFIGDIR + MINTRAJ/SKIP (numeric-sort by trajectory,
# drop < MINTRAJ, keep every SKIPth) -- same selection idiom as grid_flow_topocharge_1node_qsub.
CONFIGS=${CONFIGS:-}
CONFIGDIR=${CONFIGDIR:-}
MINTRAJ=${MINTRAJ:-0}
SKIP=${SKIP:-1}

[ -f "${JOBSCRIPT}" ] || { echo "ERROR: jobscript not found: ${JOBSCRIPT}"; exit 1; }

if [ -n "${CONFIGS}" ]; then
  cfglist="${CONFIGS}"
elif [ -n "${CONFIGDIR}" ]; then
  cfglist=$(ls ${CONFIGDIR}/ckpoint_lat.* 2>/dev/null \
            | while read f; do n=${f##*.}; [ "${n}" -ge "${MINTRAJ}" ] 2>/dev/null && echo "${n} ${f}"; done \
            | sort -n \
            | awk -v s="${SKIP}" '(NR-1) % s == 0 {print $2}')
else
  echo "no configs: set CONFIGS='f1 f2 ...' or CONFIGDIR=<dir> (+ MINTRAJ/SKIP)"
  exit 1
fi

n=$(echo ${cfglist} | wc -w)
echo "wrapper: $(basename "${JOBSCRIPT}")  GRID=${GRID}  DRYRUN=${DRYRUN}  -> ${n} configs"
for C in ${cfglist}; do
  [ -f "${C}" ] || { echo "  skip missing ${C}"; continue; }
  base=$(basename "${C}")
  opts=(-terse -v "CONFIG=${C},GRID=${GRID}${OPS:+,OPS=${OPS}}")
  if [ "${DRYRUN}" = "1" ]; then
    echo "  [dryrun] qsub ${opts[*]} $(basename "${JOBSCRIPT}")   # ${base}"
  else
    jid=$(qsub "${opts[@]}" "${JOBSCRIPT}" | tr -d '[:space:]')
    echo "  submitted ${base} -> job ${jid}"
  fi
done
