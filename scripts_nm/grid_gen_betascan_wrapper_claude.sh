#!/bin/bash
# 2024-style submission wrapper: qsub the single-node quenched-Iwasaki HMC gen for each (volume, beta) in
# the lattice-spacing scan. You run this to SUBMIT; it launches no compute. Jobs are independent -> fan
# out in parallel (no -hold_jid). OUTDIR is auto-derived per (L,beta) inside the gen script as
# configs_iwasaki_<L>_b<beta>, so nothing is overwritten across betas.
#
# RBC/UKQCD Iwasaki gauge anchors: 2.13 (24I), 2.25 (32I), 2.37 (32Ifine). NB those spacings are for the
# DYNAMICAL 2+1f theory; quenched at the same beta is finer. Your existing 16^4 b2.6 is a finer 4th point.
#
# Usage (DRYRUN first to preview, then submit):
#   DRYRUN=1 bash grid_gen_betascan_wrapper_claude.sh                 # print the qsub lines, submit nothing
#   bash grid_gen_betascan_wrapper_claude.sh                          # submit the full 2x3 scan
# Knobs (override any):
#   VOLS="16 24"  BETAS="2.13 2.25 2.37"  FINALTRAJ=400  bash grid_gen_betascan_wrapper_claude.sh
#   FINALTRAJ sets #configs (=/SAVE, SAVE=20 => 400->20 configs). MDSTEPS defaults per volume (16->20,
#   24->24); override with MDSTEPS=... to force one value for all.

set -u

SNM=/projectnb/qfe/nmatsum/dwf/Grid/scripts_nm
JOBSCRIPT=${JOBSCRIPT:-${SNM}/grid_gen_quenched_16_1node_scc_mpi_qsub_claude.sh}
VOLS=${VOLS:-"16 24"}
BETAS=${BETAS:-"2.13 2.25 2.37"}
FINALTRAJ=${FINALTRAJ:-800}          # #configs = FINALTRAJ / SAVE(=20); 800 => 40, 400 => 20, 200 => 10
DRYRUN=${DRYRUN:-0}

[ -f "${JOBSCRIPT}" ] || { echo "ERROR: jobscript not found: ${JOBSCRIPT}"; exit 1; }

echo "betascan wrapper: $(basename "${JOBSCRIPT}")  VOLS='${VOLS}'  BETAS='${BETAS}'  FINALTRAJ=${FINALTRAJ}  DRYRUN=${DRYRUN}"
for L in ${VOLS}; do
  # MD steps per volume (integrator error grows with volume); MDSTEPS env overrides for all.
  if [ -n "${MDSTEPS:-}" ]; then
    md=${MDSTEPS}
  elif [ "${L}" -ge 24 ] 2>/dev/null; then
    md=24
  else
    md=20
  fi
  GRID="${L}.${L}.${L}.${L}"
  for B in ${BETAS}; do
    btag=$(echo "${B}" | tr -d '.')
    name="gen${L}_b${btag}"
    vars="GRID=${GRID},BETA=${B},MDSTEPS=${md},FINALTRAJ=${FINALTRAJ}"
    if [ "${DRYRUN}" = "1" ]; then
      echo "  [dryrun] qsub -N ${name} -terse -v ${vars} $(basename "${JOBSCRIPT}")"
    else
      jid=$(qsub -N "${name}" -terse -v "${vars}" "${JOBSCRIPT}" | tr -d '[:space:]')
      echo "  submitted ${name} (${GRID} beta=${B} mdsteps=${md}) -> job ${jid}"
    fi
  done
done
