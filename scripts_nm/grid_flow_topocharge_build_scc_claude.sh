#!/bin/bash
# R2 chunk D (SCC): build + run the flowed-topological-charge measurement on a NERSC config.
# Compiles tests/smearing/Test_flowed_topocharge_claude.cc natively via build/bin/grid-config (the .cc
# is not in automake), then flows a COPY long and prints Q(tau) [clover + 5Li] for binning.
# Build compiles on the login node (nvcc); the RUN needs a GPU node (guarded by nvidia-smi).
# Output tee'd to grid_flow_topocharge_build_scc_claude.log.

set -u

# MODE=gpu -> CUDA build/ (run on a GPU node). MODE=cpu -> AVX+MPI build_mpi/ (run HERE via mpirun).
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
GC=${BUILD}/bin/grid-config
TEST=${SRC}/tests/smearing/Test_flowed_topocharge_claude.cc
OBJ=${BUILD}/Test_flowed_topocharge_claude.o
BIN=${BUILD}/Test_flowed_topocharge_claude
LOG=${ROOT}/grid_flow_topocharge_build_scc_claude.log

# Threads per MPI rank = CORES/NP so ranks x threads = physical cores (no oversubscription).
CORES=${CORES:-8}
if [ "${MODE}" = "cpu" ]; then
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( CORES / NP ))}
else
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
fi

# RUN params. We do NOT iterate nstep: one LONG flow per config writes the WHOLE Q(tau) trajectory to a
# per-config .dat, then we plot the few configs together and read off the plateau flow time.
# This is the Q-MEASUREMENT flow (long), distinct from the tau=2 FRAME flow used to build Omega.
GRID=${GRID:-24.24.24.24}
FLOW_ACTION=${FLOW_ACTION:-iwasaki}   # Iwasaki-action gradient flow: preserves the topological lump
FLOW_EPS=${FLOW_EPS:-0.01}
FLOW_NSTEP=${FLOW_NSTEP:-800}         # long: covers tau up to eps*nstep (=8) to reach the plateau
MEAS=${MEAS:-20}                      # sample Q every MEAS steps -> dense trajectory
# config selection (one of): CONFIGDIR (globs ckpoint_lat.*), CONFIGS (space list), or single CONFIG
CONFIGDIR=${CONFIGDIR:-}
CONFIGS=${CONFIGS:-}
CONFIG=${CONFIG:-}
DATDIR=${DATDIR:-${ROOT}/flowQ_dat}   # per-config Q(tau) trajectories land here (columns: tau plaq Q_clover Q_5Li)

{
  echo "======== [0/3] flags from grid-config  $(date) ========"
  if [ ! -f "${GC}" ]; then
    echo "  grid-config missing -> run grid_build_scc_claude.sh first; stopping."
    exit 1
  fi
  CXX="$(${GC} --cxx)"
  CXXFLAGS="$(${GC} --cxxflags)"
  LDFLAGS="$(${GC} --ldflags)"
  LIBS="$(${GC} --libs)"
  CXXLD="${CXX/-x cu/-link}"
  echo "CXX   = ${CXX}"
  echo "CXXLD = ${CXXLD}"

  echo "======== [1/3] COMPILE ========"
  ${CXX} ${CXXFLAGS} -c ${TEST} -o ${OBJ}
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "COMPILE FAILED (rc=${rc}); stopping."
    exit ${rc}
  fi

  echo "======== [2/3] LINK ========"
  ${CXXLD} ${CXXFLAGS} ${OBJ} -o ${BIN} ${LDFLAGS} ${LIBS}
  rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "LINK FAILED (rc=${rc}); stopping."
    exit ${rc}
  fi
  echo "  built ${BIN}"

  echo "======== [3/3] RUN (MODE=${MODE}) -- one long flow per config, write full Q(tau) ========"
  if [ "${MODE}" = "gpu" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "  MODE=gpu but no GPU here -> BUILT ONLY (${BIN}). Run on a GPU node, or use MODE=cpu."
    exit 0
  fi
  # assemble the config list (CONFIGS > CONFIGDIR > CONFIG)
  cfglist=""
  if [ -n "${CONFIGS}" ]; then
    cfglist="${CONFIGS}"
  elif [ -n "${CONFIGDIR}" ]; then
    cfglist=$(ls ${CONFIGDIR}/ckpoint_lat.* 2>/dev/null)
  elif [ -n "${CONFIG}" ]; then
    cfglist="${CONFIG}"
  fi
  if [ -z "${cfglist}" ]; then
    echo "  no configs -> set CONFIGDIR=<dir>, CONFIGS='f1 f2 ...', or CONFIG=<file>. Stopping."
    exit 0
  fi
  mkdir -p "${DATDIR}"
  echo "  ranks=${MPIDECOMP} x OMP_NUM_THREADS=${OMP_NUM_THREADS} threads/rank ; DATDIR=${DATDIR}"
  for C in ${cfglist}; do
    if [ ! -f "${C}" ]; then
      echo "  skip (missing): ${C}"
      continue
    fi
    base=$(basename "${C}")
    dat="${DATDIR}/flowQ_${base}_claude.dat"
    echo "+ flow ${C} (action=${FLOW_ACTION} eps=${FLOW_EPS} nstep=${FLOW_NSTEP}) -> ${dat}"
    # binary's stderr (the numeric Q(tau) table) -> per-config .dat; its stdout (Grid logs) -> the tee'd log
    ${LAUNCH} ${BIN} --grid "${GRID}" --mpi "${MPIDECOMP}" --threads "${OMP_NUM_THREADS}" \
           --config "${C}" --flow_action "${FLOW_ACTION}" \
           --flow_eps "${FLOW_EPS}" --flow_nstep "${FLOW_NSTEP}" --meas_interval "${MEAS}" 2> "${dat}"
    echo "  wrote ${dat} (rc=$?)"
  done
  echo "  done. Plot all trajectories:  gnuplot -e \"datdir='${DATDIR}'\" ${SRC}/scripts_nm/plot_flowQ_claude.gp"
} 2>&1 | tee "${LOG}"
