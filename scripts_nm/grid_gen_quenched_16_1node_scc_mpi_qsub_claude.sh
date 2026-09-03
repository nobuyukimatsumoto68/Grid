#!/bin/bash -l
# BU SCC SGE *BATCH* job: quenched Iwasaki SU(3) generation at 16^4 on a SINGLE NODE (throughput monitor).
# Single-node SMP: "-pe omp 16" (16 cores on ONE node -- schedules almost immediately, no multi-node
# wait). 1 MPI rank x OpenMP=16 (mpirun -np 1). Purpose: measure seconds/trajectory to size the batches.
# Same recipe otherwise: MinimumNorm2, trajL=1.6, saves ckpoint_lat.<t> + ckpoint_rng.<t> (resumable),
# #configs = TRAJ/SAVE. Shares OUTDIR with the multi-node 16^4 run (configs_iwasaki_16_b2.6).
#
# Submit:  qsub grid_gen_quenched_16_1node_scc_mpi_qsub_claude.sh
#   throughput probe (few traj, quick):
#          qsub -l h_rt=1:00:00 -v THERM=0,TRAJ=20,SAVE=100 grid_gen_quenched_16_1node_scc_mpi_qsub_claude.sh
# Watch:   qstat -u $USER ; tail -f <OUTDIR>/gen_mpi_claude.log   (Grid prints per-trajectory wall time)

#$ -P qfe
##$ -M mtsmtnbyk@gmail.com
#$ -N gen_iwa16_1n
#$ -j y
#$ -o /projectnb/qfe/nmatsum/dwf/log/
##$ -m n
#$ -l h_rt=6:00:00
#$ -pe omp 16

set -u

module load gcc/12.2.0
module load openmpi/4.1.5_gnu-12.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
BIN=${ROOT}/build_mpi/Test_hmc_IwasakiGauge_claude

# ---- runtime knobs (qsub -v) ----
GRID=${GRID:-16.16.16.16}
THREADS=${THREADS:-${NSLOTS:-16}}    # OpenMP threads = all cores on the single node
MPIDECOMP=${MPIDECOMP:-1.1.1.1}      # single rank
TRAJ=${TRAJ:-800}                    # #configs = TRAJ/SAVE
THERM=${THERM:-10}                   # = Grid NoMetropolisUntil (ignored on CheckpointStart)
FINALTRAJ=${FINALTRAJ:-800}          # AUTORESUME target = last-config trajectory (must be a multiple of SAVE); 800 => 40 configs
AUTORESUME=${AUTORESUME:-1}          # 1 = detect on-disk frontier and CheckpointStart from it (for -hold_jid chaining)
SAVE=${SAVE:-20}                     # 800/20 = 40 configs
MDSTEPS=${MDSTEPS:-20}
TRAJL=${TRAJL:-1.6}
BETA=${BETA:-2.6}                    # Iwasaki gauge coupling (--beta). RBC anchors: 2.13, 2.25, 2.37
START=${START:-HotStart}             # HotStart | ColdStart | CheckpointStart (+ STARTTRAJ)
STARTTRAJ=${STARTTRAJ:-0}
LSITE=${GRID%%.*}                    # spatial extent from GRID (16.16.16.16 -> 16) for the default OUTDIR
OUTDIR=${OUTDIR:-${ROOT}/configs_iwasaki_${LSITE}_b${BETA}}

export OMP_NUM_THREADS=${THREADS}
export OPENBLAS_NUM_THREADS=${THREADS}

echo "=========================================================="
echo "Start date : $(date)   Job ID : ${JOB_ID:-?}   Host : ${HOSTNAME:-?}"
echo "NSLOTS=${NSLOTS:-?}  1 rank x ${THREADS} OpenMP threads (single node)  MPIDECOMP=${MPIDECOMP}"
echo "GRID=${GRID}  START=${START}  THERM=${THERM}  TRAJ=${TRAJ}  SAVE=${SAVE} (=> $((TRAJ/SAVE)) configs)"
echo "Iwasaki beta=${BETA}  integrator=MinimumNorm2  trajL=${TRAJL}  MDSTEPS=${MDSTEPS}  dt=$(awk "BEGIN{print ${TRAJL}/${MDSTEPS}}")"
echo "OUTDIR=${OUTDIR}   BIN=${BIN}"
echo "=========================================================="

if [ ! -x "${BIN}" ]; then
  echo "ERROR: binary missing (${BIN}); build the CPU variant first. Stopping."
  exit 1
fi

mkdir -p "${OUTDIR}"
cd "${OUTDIR}"

# AUTORESUME: continue the SAME stream from the highest on-disk config up to FINALTRAJ (for -hold_jid
# chaining). Each chained 6 h link resumes RNG-exactly from where the previous was walltime-killed.
if [ "${AUTORESUME}" = "1" ]; then
  frontier=$(ls ckpoint_lat.* 2>/dev/null | while read f; do echo "${f##*.}"; done | sort -n | tail -1)
  frontier=${frontier:-0}
  if [ "${frontier}" -ge "${FINALTRAJ}" ]; then
    echo "AUTORESUME: frontier=${frontier} >= FINALTRAJ=${FINALTRAJ} -> generation complete, nothing to do."
    exit 0
  fi
  if [ "${frontier}" -gt 0 ]; then
    START=CheckpointStart
    STARTTRAJ=${frontier}
    THERM=0
    TRAJ=$(( FINALTRAJ - frontier ))
    echo "AUTORESUME: frontier=${frontier} -> CheckpointStart, Trajectories=${TRAJ} (to FINALTRAJ=${FINALTRAJ})"
  else
    START=HotStart
    STARTTRAJ=0
    TRAJ=$(( FINALTRAJ - THERM ))
    echo "AUTORESUME: no configs -> HotStart, THERM=${THERM}, Trajectories=${TRAJ}"
  fi
elif [ "${START}" = "HotStart" ] && ls ckpoint_lat.* >/dev/null 2>&1; then
  echo "OUTDIR has ckpoint_lat.* and START=HotStart (AUTORESUME=0) -> refusing to overwrite. Stopping."
  exit 0
fi

ARGS="--grid ${GRID} --mpi ${MPIDECOMP} --threads ${THREADS}"
ARGS="${ARGS} --StartingType ${START} --Thermalizations ${THERM} --Trajectories ${TRAJ}"
ARGS="${ARGS} --save_interval ${SAVE} --trajL ${TRAJL} --mdsteps ${MDSTEPS} --beta ${BETA}"
if [ "${START}" = "CheckpointStart" ]; then
  ARGS="${ARGS} --StartTrajectory ${STARTTRAJ}"
fi

echo "+ mpirun -np 1 ${BIN} ${ARGS}"
mpirun -np 1 "${BIN}" ${ARGS} 2>&1 | tee gen_mpi_claude.log
echo "mpirun exit = ${PIPESTATUS[0]}   $(date)"
echo "configs:"
ls -la "${OUTDIR}"/ckpoint_lat.* 2>/dev/null | tail
echo "finished"
