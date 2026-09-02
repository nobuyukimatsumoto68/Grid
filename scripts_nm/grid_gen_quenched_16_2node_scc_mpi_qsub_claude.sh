#!/bin/bash -l
# BU SCC SGE *BATCH* job: quenched Iwasaki SU(3) generation at 16^4 on a 2-NODE MPI job (R2 warmup).
# Same recipe as grid_gen_quenched_scc_mpi_qsub_claude.sh (the 24^4 / 4-node version), scaled down:
#   16^4 volume, 2 x 16-core MPI nodes -> "-pe mpi_16_tasks_per_node 32", 1 rank/node x OpenMP=16
#   (mpirun -np 2 --map-by ppr:1:node --bind-to none), --mpi 2.1.1.1. MinimumNorm2, trajL=1.6.
# Saves ckpoint_lat.<t> + ckpoint_rng.<t> every SAVE traj -> RESUMABLE via START=CheckpointStart.
# #configs = TRAJ/SAVE ; 40 configs = TRAJ 800, SAVE 20. Separate OUTDIR from the 24^4 run.
# 16^4 is ~5x smaller than 24^4, so 40 configs on 32 cores should fit inside one 6 h job.
#
# Submit:  qsub grid_gen_quenched_16_2node_scc_mpi_qsub_claude.sh
#   brief acceptance test first:
#          qsub -l h_rt=1:00:00 -v THERM=0,TRAJ=10,SAVE=100 grid_gen_quenched_16_2node_scc_mpi_qsub_claude.sh
#   resume after a walltime cut:
#          qsub -v START=CheckpointStart,STARTTRAJ=<N> grid_gen_quenched_16_2node_scc_mpi_qsub_claude.sh
# Watch:   qstat -u $USER ; tail -f <OUTDIR>/gen_mpi_claude.log

#$ -P qfe
#$ -M mtsmtnbyk@gmail.com
#$ -N gen_iwa16
#$ -j y
#$ -m beas
#$ -l h_rt=6:00:00
#$ -pe mpi_16_tasks_per_node 32

set -u

module load gcc/12.2.0
module load openmpi/4.1.5_gnu-12.2.0

ROOT=/projectnb/qfe/nmatsum/dwf
BIN=${ROOT}/build_mpi/Test_hmc_IwasakiGauge_claude     # MinimumNorm2 variant (cmdline trajL/mdsteps/save)

# ---- runtime knobs (qsub -v) ----
GRID=${GRID:-16.16.16.16}
THREADS=${THREADS:-16}               # OpenMP threads per rank = cores per 16-core node
MPIDECOMP=${MPIDECOMP:-2.1.1.1}      # product MUST equal #ranks (= NSLOTS/THREADS = 2)
TRAJ=${TRAJ:-800}                    # production trajectories; #configs = TRAJ/SAVE
THERM=${THERM:-10}                   # = Grid NoMetropolisUntil: always-accept burn-in (ignored on CheckpointStart)
SAVE=${SAVE:-20}                     # save every SAVE -> 800/20 = 40 configs
MDSTEPS=${MDSTEPS:-20}               # MinimumNorm2 steps (dt = trajL/MDSTEPS); tune by acceptance
TRAJL=${TRAJL:-1.6}
START=${START:-HotStart}             # HotStart | ColdStart | CheckpointStart (+ STARTTRAJ to resume)
STARTTRAJ=${STARTTRAJ:-0}
OUTDIR=${OUTDIR:-${ROOT}/configs_iwasaki_16_b2.6}

export OMP_NUM_THREADS=${THREADS}
export OPENBLAS_NUM_THREADS=${THREADS}
NP=$(( ${NSLOTS:-32} / THREADS ))    # 32/16 = 2 ranks (1 per node)

echo "=========================================================="
echo "Start date : $(date)   Job ID : ${JOB_ID:-?}   Host : ${HOSTNAME:-?}"
echo "NSLOTS=${NSLOTS:-?}  NP(ranks)=${NP}  THREADS/rank=${THREADS}  MPIDECOMP=${MPIDECOMP}"
echo "GRID=${GRID}  START=${START}  THERM=${THERM}  TRAJ=${TRAJ}  SAVE=${SAVE} (=> $((TRAJ/SAVE)) configs)"
echo "integrator=MinimumNorm2  trajL=${TRAJL}  MDSTEPS=${MDSTEPS}  dt=$(awk "BEGIN{print ${TRAJL}/${MDSTEPS}}")"
echo "OUTDIR=${OUTDIR}   BIN=${BIN}"
echo "=========================================================="

if [ ! -x "${BIN}" ]; then
  echo "ERROR: binary missing (${BIN}); build the CPU variant first. Stopping."
  exit 1
fi

mkdir -p "${OUTDIR}"
cd "${OUTDIR}"

# Refuse to clobber a fresh HotStart over existing configs (no rm here -- use CheckpointStart to resume).
if [ "${START}" = "HotStart" ] && ls ckpoint_lat.* >/dev/null 2>&1; then
  echo "OUTDIR already has ckpoint_lat.* and START=HotStart -> refusing to overwrite."
  echo "Resume with START=CheckpointStart,STARTTRAJ=<last N>, or use a fresh OUTDIR. Stopping."
  exit 0
fi

ARGS="--grid ${GRID} --mpi ${MPIDECOMP} --threads ${THREADS}"
ARGS="${ARGS} --StartingType ${START} --Thermalizations ${THERM} --Trajectories ${TRAJ}"
ARGS="${ARGS} --save_interval ${SAVE} --trajL ${TRAJL} --mdsteps ${MDSTEPS}"
if [ "${START}" = "CheckpointStart" ]; then
  ARGS="${ARGS} --StartTrajectory ${STARTTRAJ}"
fi

# 1 rank per node (--map-by ppr:1:node) so each rank gets a whole 16-core node for its 16 OpenMP threads.
echo "+ mpirun -np ${NP} --map-by ppr:1:node --bind-to none ${BIN} ${ARGS}"
mpirun -np ${NP} --map-by ppr:1:node --bind-to none "${BIN}" ${ARGS} 2>&1 | tee gen_mpi_claude.log
echo "mpirun exit = ${PIPESTATUS[0]}   $(date)"
echo "configs:"
ls -la "${OUTDIR}"/ckpoint_lat.* 2>/dev/null | tail
echo "finished"
