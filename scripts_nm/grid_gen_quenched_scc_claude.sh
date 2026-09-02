#!/bin/bash
# R2 chunk C: generate a few QUENCHED SU(3) configs on SCC via Grid's stock Iwasaki gauge HMC.
# Iwasaki beta=2.6 (already hardcoded in tests/hmc/Test_hmc_IwasakiGauge.cc -- matches Nobu's choice).
# Pure gauge (no fermion force) -> cheap, single GPU, uses the current comms=none build. The multi-GPU
# comms=mpi build is needed only for the 24^4 FGMRES SOLVE later, NOT for generation or flow.
#
# NERSC checkpoints (ckpoint_lat.<traj>) land in OUTDIR every 20 trajectories (saveInterval in the test).
# We then measure flowed topological charge (chunk D) and pick a Q!=0 config + a Q=0 control.
#
# Build compiles on the login node (nvcc, no GPU needed); the HMC RUN needs a GPU node (guarded by
# nvidia-smi). Easiest: qsub grid_gen_quenched_scc_qsub_claude.sh (single GPU). Output tee'd to LOG.

set -u

# MODE=gpu -> CUDA build/ (GPU node). MODE=cpu -> AVX+MPI build_mpi/ (runs HERE via mpirun).
MODE=${MODE:-gpu}
ROOT=/projectnb/qfe/nmatsum/dwf
SRC=${ROOT}/Grid
if [ "${MODE}" = "cpu" ]; then
  module load gcc/12.2.0
  module load openmpi/4.1.5_gnu-12.2.0
  BUILD=${BUILD:-${ROOT}/build_mpi}
  NP=${NP:-2}                          # 2 ranks x 4 OpenMP threads (= 8 cores); MPIDECOMP product must equal NP
  MPIDECOMP=${MPIDECOMP:-2.1.1.1}
  LAUNCH="mpirun -np ${NP}"
else
  module load cuda/12.8
  module load gcc/13.2.0
  BUILD=${BUILD:-${ROOT}/build}
  MPIDECOMP=${MPIDECOMP:-1.1.1.1}
  LAUNCH=""
fi
BIN=${BUILD}/tests/hmc/Test_hmc_IwasakiGauge

GRID=${GRID:-24.24.24.24}
OUTDIR=${OUTDIR:-${ROOT}/configs_iwasaki_24_b2.6}
TRAJ=${TRAJ:-200}
THERM=${THERM:-50}
START=${START:-HotStart}           # HotStart -> chain wanders in Q; add a ColdStart chain for a guaranteed Q=0 control
LOG=${ROOT}/grid_gen_quenched_scc_claude.log

# Threads per MPI rank. Enforce ranks x threads = physical cores so we do NOT oversubscribe.
# CPU: NP ranks (product of MPIDECOMP) x (CORES/NP) threads = CORES. Default NP=8 -> 1 thread/rank.
# For a hybrid split set e.g. NP=2 MPIDECOMP=2.1.1.1 -> 4 threads/rank; NP=4 MPIDECOMP=2.2.1.1 -> 2.
CORES=${CORES:-8}
if [ "${MODE}" = "cpu" ]; then
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( CORES / NP ))}
else
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
fi

{
  echo "======== [0/2] build Test_hmc_IwasakiGauge  $(date) ========"
  if [ ! -f "${BUILD}/lib/libGrid.a" ]; then
    echo "  libGrid.a missing -> run grid_build_scc_claude.sh first; stopping."
    exit 1
  fi
  make -C "${BUILD}/tests/hmc" Test_hmc_IwasakiGauge
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "BUILD FAILED (rc=${rc}); stopping."
    exit ${rc}
  fi
  echo "  built ${BIN}"

  echo "======== [1/2] generate quenched Iwasaki configs (MODE=${MODE} GRID=${GRID} beta=2.6) ========"
  if [ "${MODE}" = "gpu" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "  MODE=gpu but no GPU here -> BUILT ONLY. Run on a GPU node (qsub grid_gen_quenched_scc_qsub_claude.sh)"
    echo "  or use MODE=cpu to generate here via mpirun."
    exit 0
  fi
  mkdir -p "${OUTDIR}"
  # Guard: do not silently overwrite a previous generation. If HotStart would clobber existing
  # checkpoints, stop and let the user choose a fresh OUTDIR or resume with START=CheckpointStart.
  if ls "${OUTDIR}"/ckpoint_lat.* >/dev/null 2>&1 && [ "${START}" != "CheckpointStart" ]; then
    echo "  OUTDIR already has ckpoint_lat.* and START=${START}."
    echo "  Pick a fresh OUTDIR, or set START=CheckpointStart to resume. Not overwriting. Stopping."
    exit 0
  fi
  cd "${OUTDIR}"
  echo "  ranks=${MPIDECOMP} (NP inferred) x OMP_NUM_THREADS=${OMP_NUM_THREADS} threads/rank"
  echo "+ ${LAUNCH} ${BIN} --grid ${GRID} --mpi ${MPIDECOMP} --threads ${OMP_NUM_THREADS} --StartingType ${START} --Thermalizations ${THERM} --Trajectories ${TRAJ}"
  ${LAUNCH} "${BIN}" --grid "${GRID}" --mpi "${MPIDECOMP}" --threads "${OMP_NUM_THREADS}" \
           --StartingType "${START}" --Thermalizations "${THERM}" --Trajectories "${TRAJ}"
  rc=$?
  echo "HMC exit code = ${rc}"
  echo "  configs written to ${OUTDIR}:"
  ls -la "${OUTDIR}"/ckpoint_lat.* 2>/dev/null || echo "  (no ckpoint_lat.* found -- check log)"
} 2>&1 | tee "${LOG}"
