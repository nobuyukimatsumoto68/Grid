#!/bin/bash
# Chain N HMC gen links via SGE -hold_jid (mimics 2024 epoch_driver_claude.sh). Each link is the SAME
# AUTORESUME=1 gen job script; it detects the on-disk frontier and CheckpointStarts from it, so a chain
# of 6 h links walks the SAME stream forward to FINALTRAJ (each starts when the previous finishes/walls).
# No compute here -- pure qsub submission (run on the login node).
#
# Usage:  bash grid_gen_chain_claude.sh <jobscript> <n_links> [first_hold_jid]
#   e.g. 4 links of the 16^4 gen, first held on the currently-running job 7406497:
#     bash grid_gen_chain_claude.sh grid_gen_quenched_16_1node_scc_mpi_qsub_claude.sh 4 7406497
#   (omit first_hold_jid to start the chain immediately)

set -u

SNM=/projectnb/qfe/nmatsum/dwf/Grid/scripts_nm
JOBSCRIPT=${1:?usage: grid_gen_chain_claude.sh <jobscript> <n_links> [first_hold_jid]}
NLINKS=${2:-4}
prev=${3:-}

# resolve jobscript path (accept bare name)
[ -f "${JOBSCRIPT}" ] || JOBSCRIPT="${SNM}/$(basename "${JOBSCRIPT}")"
[ -f "${JOBSCRIPT}" ] || { echo "ERROR: jobscript not found: ${JOBSCRIPT}"; exit 1; }

echo "chaining ${NLINKS} links of $(basename "${JOBSCRIPT}")  (first hold_jid = ${prev:-none})"
for k in $(seq 1 "${NLINKS}"); do
  if [ -n "${prev}" ]; then
    jid=$(qsub -terse -hold_jid "${prev}" "${JOBSCRIPT}")
  else
    jid=$(qsub -terse "${JOBSCRIPT}")
  fi
  jid=$(echo "${jid}" | tr -d '[:space:]')
  echo "  link ${k}: job ${jid}   held on ${prev:-none}"
  prev="${jid}"
done
echo "chain tail job = ${prev}"
echo "watch: qstat -u \$USER"
