#!/bin/bash
# Instanton-size probe over the shrink-flow checkpoints. Loops the CPU probe binary over every file
# in CKPDIR (or an explicit FILES list), one single-rank run each, and concatenates the per-config
# .dat outputs into one aggregate. Cheap (pure gauge analysis) -- fine on a login/interactive node.
#   FILES="shrinkflow_ckp/shrink_ckpoint_lat.640_c10.0833_k*" bash run_instsize_claude.sh
set -u

module load gcc/12.2.0
module load openmpi/4.1.5_gnu-12.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
BIN=${BIN:-${ROOT}/build_mpi_merged/Test_instsize_claude}
GRID=${GRID:-16.16.16.16}
CKPDIR=${CKPDIR:-${ROOT}/shrinkflow_ckp}
MINQPEAK=${MINQPEAK:-1e-4}
MASKFAC=${MASKFAC:-2.0}
MAXLUMPS=${MAXLUMPS:-30}
# SciDAC q(x) dumps for the VTK visualisation tools; needs a --with-lime Grid build (else the binary
# warns and skips). Empty = off.
SCIDACDIR=${SCIDACDIR:-}
LOGDIR=${ROOT}/log
mkdir -p "${LOGDIR}"
RUNLOG=${LOGDIR}/instsize_run_claude.log
AGG=${LOGDIR}/instsize_all_claude.dat

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

if [ ! -x "${BIN}" ]; then echo "ERROR: probe binary missing ${BIN} (build: grid_instsize_build_scc_mpi_claude.sh)"; exit 1; fi

if [ -n "${FILES:-}" ]; then
  LIST=$(ls ${FILES} 2>/dev/null)
else
  LIST=$(ls ${CKPDIR}/shrink_* 2>/dev/null)
fi
if [ -z "${LIST}" ]; then echo "no checkpoint files found (CKPDIR=${CKPDIR})"; exit 1; fi

: > "${AGG}"
{
  for f in ${LIST}; do
    base=$(basename "${f}")
    dat=${LOGDIR}/instsize_${base}_claude.dat
    echo "======== probe ${base}  $(date) ========"
    SCIDACOPT=""
    if [ -n "${SCIDACDIR}" ]; then
      mkdir -p "${SCIDACDIR}"
      SCIDACOPT="--scidac_dir ${SCIDACDIR}"
    fi
    mpirun -np 1 "${BIN}" --grid "${GRID}" --mpi 1.1.1.1 \
      --config "${f}" --min_qpeak "${MINQPEAK}" --mask_fac "${MASKFAC}" --max_lumps "${MAXLUMPS}" \
      --dat "${dat}" ${SCIDACOPT}
    cat "${dat}" >> "${AGG}"
  done
  echo "aggregate -> ${AGG}"
} 2>&1 | tee "${RUNLOG}"
