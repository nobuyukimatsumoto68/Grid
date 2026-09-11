#!/bin/bash -l
# mprec scan (Wilson additive-renormalization shift) for the topological-correction diagnostics.
# For each (config, mprec): run BOTH
#   (1) Test_dwf_wtop_eigs_claude  -> C_M0 / C_M1 on the D_W (set a) and H_W (set b) low modes
#   (2) Test_dwf_precmodes_claude  -> low modes of the PRECONDITIONED operators M0 D_W and M1 D_W
# mass_c stays 0 (massless interacting operator); only the preconditioner mass mprec is shifted.
# Runs are STRICTLY SEQUENTIAL (1 GPU). Each measurement gets its OWN dedicated log file.
# Plan: tests/solver/wtop_precond_impl_plan_claude.md.

set -u
module load cuda/12.8
module load gcc/13.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
BIN_WTOP=${ROOT}/build_merged/Test_dwf_wtop_eigs_claude
BIN_PM=${ROOT}/build_merged/Test_dwf_precmodes_claude
CFGDIR=${ROOT}/configs_iwasaki_16_b2.6
FLOWCACHE=${ROOT}/flowcache
OUTDIR=${ROOT}/log/mprec_scan
mkdir -p "${OUTDIR}"

GRID=16.16.16.16
CONFIGS="640 240"
MPRECS="0.5 0.6 0.7 0.8 0.9"

export OMP_NUM_THREADS=4

echo "=== mprec scan START $(date)  host ${HOSTNAME:-?} ==="
nvidia-smi -L 2>/dev/null | head -1
if [ ! -x "${BIN_WTOP}" ]; then echo "ERROR: missing ${BIN_WTOP}"; exit 1; fi
if [ ! -x "${BIN_PM}" ]; then echo "ERROR: missing ${BIN_PM}"; exit 1; fi

for CFG in ${CONFIGS}
do
  CONFIG=${CFGDIR}/ckpoint_lat.${CFG}
  if [ ! -f "${CONFIG}" ]; then echo "ERROR: config missing ${CONFIG}"; exit 1; fi
  for MP in ${MPRECS}
  do
    echo "======== config ${CFG}  mprec=${MP}  $(date) ========"

    OUT_WTOP=${OUTDIR}/wtop_${CFG}_mprec${MP}_claude.log
    echo "  [wtop -> ${OUT_WTOP}]"
    "${BIN_WTOP}" --grid "${GRID}" --mpi 1.1.1.1 --config "${CONFIG}" --flowcache "${FLOWCACHE}" \
        --ckpt_dir "${ROOT}/log/wtop_${CFG}" --nstop 24 --nk 36 --nm 64 --cheb_lo 0.5 --cheb_ord 21 \
        --nreport 24 --mass_c 0 --mprec "${MP}" --accelerator-threads 8 > "${OUT_WTOP}" 2>&1
    echo "    wtop exit=$?"

    OUT_PM=${OUTDIR}/precmodes_${CFG}_mprec${MP}_claude.log
    echo "  [precmodes -> ${OUT_PM}]"
    "${BIN_PM}" --grid "${GRID}" --mpi 1.1.1.1 --config "${CONFIG}" --flowcache "${FLOWCACHE}" \
        --nstop 4 --nk 8 --nm 16 --ira_maxit 30 --nreport 8 --mass_c 0 --mprec "${MP}" --accelerator-threads 8 \
        > "${OUT_PM}" 2>&1
    echo "    precmodes exit=$?"
  done
done

echo "=== mprec scan DONE $(date) ==="
echo "logs in ${OUTDIR}/"
