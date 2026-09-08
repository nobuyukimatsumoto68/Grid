#!/bin/bash
# Interactive-GPU-node run helper for the frame-flow-TIME scan (M0/M1 win vs s/t0). Run it on a qrsh GPU
# node (or any node with a visible GPU). Not a batch script -- for SGE use grid_flowscan_gpu_qsub_claude.sh.
# comms=none GPU build -> run the binary directly (no mpirun). Override any knob via env, e.g.
#   CONFIG=.../ckpoint_lat.140 T0=0.759 NSTEPS=15,19,23,27,30,34,38,42,46 bash run_flowscan_claude.sh
set -u

module load cuda/12.8
module load gcc/13.2.0
export LD_LIBRARY_PATH=/share/pkg.8/cuda/12.8/install/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}

ROOT=/projectnb/qfe/nmatsum/dwf
BIN=${ROOT}/build_merged/Test_dwf_flowscan_claude
GRID=${GRID:-16.16.16.16}
# default target = the trivial-Q config (b2.13 traj 140, Q_5Li~0.0002); t0(b2.13)=0.759
CONFIG=${CONFIG:-${ROOT}/configs_iwasaki_16_b2.13/ckpoint_lat.140}
T0=${T0:-0.759}
# s/t0 = 0.4..1.2 grid at eps=0.02, t0=0.759 -> tau=0.30..0.91 -> nstep=15..46 (comma list, run directly)
NSTEPS=${NSTEPS:-15,19,23,27,30,34,38,42,46}
EPS=${EPS:-0.02}
TOL=${TOL:-1e-6}
OPS=${OPS:-cgne,m0,m1}
FLOWS=${FLOWS:-wilson}
TAG=${TAG:-16_b2.13}
RESTART=${RESTART:-256}   # FGMRES Krylov RestartLength; 256=~no-restart (pricey iters), 40=cheap restarted iters
# Grid log channels. Iterative -> per-iteration FGMRES(M0)/CG progress (GridLogIterative). Drop it for quiet.
LOGCH=${LOGCH:-Message,Error,Warning,Iterative}
LOG=${ROOT}/log/flowscan_$(basename "${CONFIG}")_${TAG}_claude.log
mkdir -p "${ROOT}/log"

if [ ! -x "${BIN}" ]; then echo "ERROR: flowscan binary missing ${BIN} (build: grid_flowscan_build_scc_gpu_claude.sh)"; exit 1; fi
if [ ! -f "${CONFIG}" ]; then echo "ERROR: config missing ${CONFIG}"; exit 1; fi

echo "=== flowscan $(date)  CONFIG=${CONFIG}  T0=${T0}  NSTEPS=${NSTEPS}  EPS=${EPS}  OPS=${OPS} ==="
"${BIN}" --grid "${GRID}" --mpi 1.1.1.1 --accelerator-threads 8 --log "${LOGCH}" \
  --config "${CONFIG}" --ops "${OPS}" --frame_flows "${FLOWS}" \
  --flow_nsteps "${NSTEPS}" --flow_eps "${EPS}" --t0 "${T0}" --solve_tol "${TOL}" \
  --fgmres_restart "${RESTART}" 2>&1 | tee "${LOG}"
echo "log -> ${LOG}"
