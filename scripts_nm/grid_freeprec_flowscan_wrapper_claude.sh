#!/bin/bash
# 2024-style submission wrapper for the flow-TIME scan (Direction 1b Stage 1). LOOPS over ensembles and
# configs and fires a plain per-config qsub (NO array jobs, Nobu). For each ensemble it looks up the
# measured t0, builds the flow-nstep list from the s/t0 grid, picks NCFG configs, and submits one
# grid_freeprec_flowscan_qsub_claude.sh per config (each internally scans all s/t0). You run this to
# SUBMIT; it launches no compute.
#
# s/t0 grid (Nobu): 0.4 0.5 ... 1.2 ; tau_lattice = (s/t0)*t0 ; nstep = tau/eps (eps=0.02).
# t0/a^2 (memory project-r2-config-gen): 16^4 -> 0.759/1.140/1.697/2.91 for b2.13/2.25/2.37/2.6.
#
# Usage (DRYRUN first):
#   DRYRUN=1 bash grid_freeprec_flowscan_wrapper_claude.sh
#   bash grid_freeprec_flowscan_wrapper_claude.sh
# Knobs: VOLS="16"  BETAS="2.13 2.25 2.37 2.6"  NCFG=10  MINTRAJ=100  SKIP=3  TOL=1e-6  OPS=m0,m1
#        SOT="0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2"   (the s/t0 grid)

set -u

SNM=/projectnb/qfe/nmatsum/dwf/Grid/scripts_nm
ROOT=/projectnb/qfe/nmatsum/dwf
JOBSCRIPT=${JOBSCRIPT:-${SNM}/grid_freeprec_flowscan_qsub_claude.sh}
VOLS=${VOLS:-"16"}
BETAS=${BETAS:-"2.13 2.25 2.37 2.6"}
SOT=${SOT:-"0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2"}
EPS=${EPS:-0.02}
NCFG=${NCFG:-10}
MINTRAJ=${MINTRAJ:-100}
SKIP=${SKIP:-3}
TOL=${TOL:-1e-6}
OPS=${OPS:-cgne,m0,m1}          # MUST include cgne: the ratio needs the (frame-independent) baseline
FLOWS=${FLOWS:-wilson}         # frame-flow KERNELS (comma list). Kernel scan: FLOWS=wilson,iwasaki,antiiwasaki
DRYRUN=${DRYRUN:-0}
OPS_T=$(echo "${OPS}" | tr ',' '-')     # hyphen transport (SGE -v splits values on commas)
FLOWS_T=$(echo "${FLOWS}" | tr ',' '-')

[ -f "${JOBSCRIPT}" ] || { echo "ERROR: jobscript not found: ${JOBSCRIPT}"; exit 1; }

# measured t0/a^2 lookup keyed by "<L>_<beta>"
t0_of() {
  case "$1" in
    16_2.13) echo 0.759 ;;
    16_2.25) echo 1.140 ;;
    16_2.37) echo 1.697 ;;
    16_2.6)  echo 2.91  ;;
    24_2.13) echo 0.756 ;;
    24_2.25) echo 1.127 ;;
    24_2.37) echo 1.526 ;;
    *)       echo "" ;;
  esac
}

echo "flowscan wrapper: VOLS='${VOLS}' BETAS='${BETAS}' NCFG=${NCFG} TOL=${TOL} OPS=${OPS} DRYRUN=${DRYRUN}"
echo "  s/t0 grid = ${SOT}"
for L in ${VOLS}; do
  GRID="${L}.${L}.${L}.${L}"
  for B in ${BETAS}; do
    t0=$(t0_of "${L}_${B}")
    if [ -z "${t0}" ]; then
      echo "  [skip] no measured t0 for ${L}^4 b${B} -- flow that ensemble first"
      continue
    fi
    cdir="${ROOT}/configs_iwasaki_${L}_b${B}"
    if [ ! -d "${cdir}" ]; then
      echo "  [skip] no config dir ${cdir}"
      continue
    fi
    # build the nstep list from the s/t0 grid and this ensemble's t0 (HYPHEN-joined for SGE -v transport)
    nsteps=""
    for r in ${SOT}; do
      ns=$(awk -v r="${r}" -v t="${t0}" -v e="${EPS}" 'BEGIN{printf "%d", r*t/e + 0.5}')
      nsteps="${nsteps:+${nsteps}-}${ns}"
    done
    # pick NCFG configs: trajectory >= MINTRAJ, every SKIPth, take the first NCFG
    cfgs=$(ls ${cdir}/ckpoint_lat.* 2>/dev/null \
           | while read f; do n=${f##*.}; [ "${n}" -ge "${MINTRAJ}" ] 2>/dev/null && echo "${n} ${f}"; done \
           | sort -n \
           | awk -v s="${SKIP}" '(NR-1) % s == 0 {print $2}' \
           | head -n "${NCFG}")
    nc=$(echo ${cfgs} | wc -w)
    echo "  == ${L}^4 b${B}  t0=${t0}  nsteps=${nsteps}  -> ${nc} configs =="
    for C in ${cfgs}; do
      base=$(basename "${C}")
      tag="${base}_${L}_b${B}"
      vars="CONFIG=${C},GRID=${GRID},NSTEPS=${nsteps},T0=${t0},TOL=${TOL},OPS=${OPS_T},FLOWS=${FLOWS_T},TAG=${tag}"
      if [ "${DRYRUN}" = "1" ]; then
        echo "    [dryrun] qsub -N flowscan_${L}b${B//./}_${base##*.} -v ${vars} $(basename "${JOBSCRIPT}")"
      else
        jid=$(qsub -N "flowscan_${L}b${B//./}_${base##*.}" -terse -v "${vars}" "${JOBSCRIPT}" | tr -d '[:space:]')
        echo "    submitted ${tag} -> job ${jid}"
      fi
    done
  done
done
