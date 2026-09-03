#!/bin/bash
# 2024-style submission wrapper: qsub the (Iwasaki) gradient flow for each beta-scan ensemble so we can
# reconstruct t0 per ensemble. You run this to SUBMIT; it launches no compute. One flow job per (L,beta);
# each job loops internally over a decorrelated config subset and writes flowQ_<cfg>_claude.dat into a
# per-ensemble DATDIR. Then: grid_t0_reconstruct_claude.sh <DATDIR>/flowQ_*.dat  -> t0/a^2 per ensemble.
#
# Flow to tau = FLOW_EPS*FLOW_NSTEP = 4.0 by default (ample: coarser betas < 2.6 reach the 0.3 crossing
# before tau=3). If a finer ensemble undershoots 0.3, bump FLOW_NSTEP (Nobu: "longer flow if necessary").
#
# Usage (DRYRUN first):
#   DRYRUN=1 bash grid_flow_betascan_wrapper_claude.sh
#   bash grid_flow_betascan_wrapper_claude.sh
# Knobs: VOLS="16 24"  BETAS="2.13 2.25 2.37"  MINTRAJ=100  SKIP=2  FLOW_NSTEP=400

set -u

SNM=/projectnb/qfe/nmatsum/dwf/Grid/scripts_nm
ROOT=/projectnb/qfe/nmatsum/dwf
JOBSCRIPT=${JOBSCRIPT:-${SNM}/grid_flow_topocharge_1node_qsub_claude.sh}
VOLS=${VOLS:-"16 24"}
BETAS=${BETAS:-"2.13 2.25 2.37"}
MINTRAJ=${MINTRAJ:-100}              # thermalization burn-in cut (24^4 has few configs -> see per-vol below)
SKIP=${SKIP:-2}                      # decorrelation stride for the t0 subset
FLOW_NSTEP=${FLOW_NSTEP:-400}        # tau up to eps(0.01)*nstep = 4.0
DRYRUN=${DRYRUN:-0}

[ -f "${JOBSCRIPT}" ] || { echo "ERROR: jobscript not found: ${JOBSCRIPT}"; exit 1; }

echo "flow betascan wrapper: VOLS='${VOLS}'  BETAS='${BETAS}'  MINTRAJ=${MINTRAJ} SKIP=${SKIP} nstep=${FLOW_NSTEP}  DRYRUN=${DRYRUN}"
for L in ${VOLS}; do
  GRID="${L}.${L}.${L}.${L}"
  # 24^4 has only a handful of configs so far -> flow all of them (small MINTRAJ, no stride).
  mtj=${MINTRAJ}
  skp=${SKIP}
  if [ "${L}" -ge 24 ] 2>/dev/null; then
    mtj=40
    skp=1
  fi
  for B in ${BETAS}; do
    btag=$(echo "${B}" | tr -d '.')
    cdir="${ROOT}/configs_iwasaki_${L}_b${B}"
    ddir="${ROOT}/flowQ_dat_${L}_b${B}"
    if [ ! -d "${cdir}" ]; then
      echo "  [skip] no config dir ${cdir}"
      continue
    fi
    nc=$(ls "${cdir}"/ckpoint_lat.* 2>/dev/null | wc -l)
    name="flow${L}_b${btag}"
    vars="GRID=${GRID},CONFIGDIR=${cdir},DATDIR=${ddir},MINTRAJ=${mtj},SKIP=${skp},FLOW_NSTEP=${FLOW_NSTEP}"
    if [ "${DRYRUN}" = "1" ]; then
      echo "  [dryrun] qsub -N ${name} -v ${vars} $(basename "${JOBSCRIPT}")   # ${nc} configs available"
    else
      jid=$(qsub -N "${name}" -terse -v "${vars}" "${JOBSCRIPT}" | tr -d '[:space:]')
      echo "  submitted ${name} (${cdir}, ${nc} configs) -> job ${jid}"
    fi
  done
done
