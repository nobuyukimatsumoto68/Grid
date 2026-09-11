#!/bin/bash
# Chained outer-deflation w_o scan (2026-09-10). One command does the whole dependency graph:
#   deflprec(48) re-dump  --hold_jid-->  deflprec w_o scan
#   RB-CGNE w_o scan  (no dep: deflrb is already 48)
#
# Usage:
#   bash submit_wo_scan_claude.sh [DUMP_JOB_ID]
#     DUMP_JOB_ID : an already-running svddump_deflprec(48) job to depend on (reuse it, no new dump).
#     (omitted)   : if deflfull<48, submit a fresh re-dump and depend on it; if deflfull>=48, no dep.
#
# Fixes both traps we hit: (1) rebuilt-binary guard; (2) SGE `-v` comma-split -> pass list vars BY NAME
# (`-v ...,WO_LIST`) so the exported comma value is imported whole (SGE splits the -v STRING on commas).
set -u

ROOT=/projectnb/qfe/nmatsum/dwf
QSUB=${ROOT}/Grid/scripts_nm/grid_twolevel_gpu_qsub_claude.sh
CKPT=${ROOT}/log/deflvec_ckpoint_lat.640
BIN=${ROOT}/build_merged/Test_dwf_rbcgne_claude

export WO_LIST="16,24,32,40,48"
export SVD_KLIST="16,24,32,40,48"

# guard: binaries rebuilt (w_o loop compiled in)
if ! grep -aq "w_o=" "${BIN}" ; then
  echo "ERROR: ${BIN} lacks the w_o loop -> rebuild: bash grid_twolevel_build_scc_gpu_claude.sh"
  exit 1
fi

# resolve the deflprec-dump dependency
DJ="${1:-}"
if [ -z "${DJ}" ]; then
  nfull=$(ls "${CKPT}"/deflfull_v*.bin 2>/dev/null | wc -l)
  if [ "${nfull}" -lt 48 ]; then
    echo "deflfull has ${nfull} (<48) -> submitting a fresh deflprec(48) re-dump ..."
    DJ=$(qsub -terse -v WHICH=svddump_deflprec,NMODES=48 "${QSUB}")
    echo "  re-dump job = ${DJ}"
  else
    echo "deflfull already 48 -> no re-dump needed."
  fi
else
  echo "depending on existing deflprec re-dump job ${DJ}"
fi

# RB-CGNE w_o scan: deflrb already 48 -> no dependency
echo "submitting RB-CGNE w_o scan ..."
qsub -v WHICH=rbcgne,WO_LIST "${QSUB}"

# FGMRES SVD-defl w_o scan (gmres_k=0 isolates outer deflation): hold on the deflprec dump if we have one
echo "submitting deflprec w_o scan${DJ:+ (held on ${DJ})} ..."
if [ -n "${DJ}" ]; then
  qsub -hold_jid "${DJ}" -v WHICH=deflprec,SVD_KLIST,GMRES_K=0 "${QSUB}"
else
  qsub -v WHICH=deflprec,SVD_KLIST,GMRES_K=0 "${QSUB}"
fi
echo "done. watch: qstat -u \$USER"
