#!/bin/bash
#FLUX: -t 240m
#FLUX: --output=hmc_prod24c_{{id}}
#FLUX: -q pbatch
#FLUX: -N 2
#FLUX: -n 8
#FLUX: -g 1
#FLUX: --exclusive
#
# FLUX batch launcher for ONE allocation of a 24^3x48 light-ensemble CONTINUATION HMC stream
# (Grid/scripts/hmc_prod24c_impl_plan_claude.md). Copy of the 32c
# submit_hmc_prod_tuolumne_claude.sh (cluster version with APP_BIN / TPT_OVERRIDE); changes:
#   - 2 nodes / 8 GPUs, --grid 24.24.24.48 --mpi 1.2.2.2
#   - XML ip_hmc_mobius_24c_claude.xml
#   - binary default dweofa_mobius_HSDM_v5_gmult_claude; GAUGE_MULT (gauge-level MD
#     multiplier) exported from the driver's --env
#   - Hasenbusch M1_HASEN block dropped (this binary has no Hasenbusch)
# Built for --dependency=afterany AUTO-CHAINING by run_prod24c_claude.sh; each job resumes at
# RUNTIME from the latest checkpoint in cwd:
#   1. read config_prefix from the XML, detect the latest saved trajectory in cwd;
#   2. switch the XML to CheckpointStart / StartTrajectory=<latest> / NoMetropolisUntil=0;
#   3. if <latest> >= NTRAJ_TARGET the stream is done -> exit without launching;
#   4. graceful wall blocker: cap <Trajectories> to the largest whole-saveInterval block that
#      fits the time left;
#   5. run the binary.

date
here=`pwd`
export FASTLOAD_VERBOSE=1
export SPINDLE_FLUXOPT=off

XML=ip_hmc_mobius_24c_claude.xml

source /usr/workspace/lsd/matsumoto5/su4_32c/env.sh
APP=${APP_BIN:-/usr/workspace/lsd/matsumoto5/su4_32c/Grid_sdm_build/src/gauge_gen_Nc4/bin/dweofa_mobius_HSDM_v5_gmult_claude}
export GAUGE_MULT=${GAUGE_MULT:-4}
echo "binary: ${APP}"
echo "GAUGE_MULT=${GAUGE_MULT}"

echo "--start " `date` `date +%s`

OPTIONS="--decomposition --comms-concurrent --comms-overlap --debug-mem  --shm 2048 --shm-mpi 1"
PARAMS=" --grid 24.24.24.48 --mpi 1.2.2.2 --threads 8 --accelerator-threads 8 ${OPTIONS} --ParameterFile ${XML}"

# ---- runtime resume: continue from the latest saved config (prefix read from the XML) ----
PREFIX=$(grep -oP '(?<=<config_prefix>)[^<]+' ${XML} | head -1)
latest=$(ls ${PREFIX}.* 2>/dev/null | sed 's/.*\.//' | sort -n | tail -n1)
latest=${latest:-0}
CUR_START=$(grep -oP '(?<=<StartTrajectory>)[0-9]+' ${XML} | head -1)
CUR_START=${CUR_START:-0}
if [ "${latest}" -gt 0 ]; then
    sed -i "/<StartTrajectory>/{s/>${CUR_START}</>${latest}</}" ${XML}
    sed -i "/<StartingType>/{s/HotStart/CheckpointStart/}" ${XML}
    sed -i "/<NoMetropolisUntil>/{s/>[0-9]*</>0</}" ${XML}
    echo "resume: latest checkpoint = ${latest} -> CheckpointStart"
else
    echo "ERROR: no checkpoint ${PREFIX}.* in $(pwd); a continuation stream must be seeded first (run_prod24c_claude.sh)." >&2
    echo "--end " `date` `date +%s`
    exit 1
fi

# ---- target check (overall goal passed by the driver) ----
TARGET=${NTRAJ_TARGET:?NTRAJ_TARGET not set}
if [ "${latest}" -ge "${TARGET}" ]; then
    echo "stream already at target (${latest} >= ${TARGET}); nothing to do (chain tail)."
    echo "--end " `date` `date +%s`
    exit 0
fi

# ---- graceful wall-time blocker: cap Trajectories to whole saveInterval blocks that fit ----
# Per-traj time MEASURED from prior hmc_prod24c_* logs of this stream ("Total time for
# trajectory (s): X", max x MARGIN); TPT_OVERRIDE takes precedence; TPT_SECONDS bootstraps the
# first job. The 750 s bootstrap is the 32c value (GUESS for 24c on 2 nodes; per-GPU volume
# 24.12.12.24 = 82944 is close to the 32c 16.16.16.16 = 65536).
OVERHEAD_SECONDS=${OVERHEAD_SECONDS:-400}
MARGIN_NUM=${MARGIN_NUM:-12}
if [ -n "${TPT_OVERRIDE:-}" ]; then
    TPT_SECONDS=${TPT_OVERRIDE}
    echo "blocker: TPT_OVERRIDE set -> per-traj=${TPT_SECONDS}s"
else
    measured=$(grep -hoE "Total time for trajectory \(s\): [0-9]+" hmc_prod24c_* 2>/dev/null | grep -oE "[0-9]+$" | sort -n | tail -n1)
    if [ -n "${measured}" ]; then
        TPT_SECONDS=$(( measured * MARGIN_NUM / 10 ))
        echo "blocker: measured per-traj (max, prior jobs this stream) = ${measured}s -> TPT=${TPT_SECONDS}s (x${MARGIN_NUM}/10)"
    else
        TPT_SECONDS=${TPT_SECONDS:-750}
        echo "blocker: no prior trajectory timing in this stream -> bootstrap TPT=${TPT_SECONDS}s"
    fi
fi
TIMELEFT=$(flux job timeleft 2>/dev/null)
case "${TIMELEFT}" in ''|*[!0-9.]*) TIMELEFT=${WALL_SECONDS:-7200} ;; esac
TIMELEFT=${TIMELEFT%.*}
SAVEINT=$(grep -oP '(?<=<saveInterval>)[0-9]+' ${XML} | head -1)
SAVEINT=${SAVEINT:-4}
CUR_TRAJ=$(grep -oP '(?<=<Trajectories>)[0-9]+' ${XML} | head -1)
NFIT=$(( (TIMELEFT - OVERHEAD_SECONDS) / TPT_SECONDS ))
[ ${NFIT} -lt 0 ] && NFIT=0
NFIT=$(( (NFIT / SAVEINT) * SAVEINT ))
CAP=$(( latest + NFIT ))
[ ${CAP} -gt ${TARGET} ] && CAP=${TARGET}
echo "blocker: timeleft=${TIMELEFT}s latest=${latest} target=${TARGET} saveInt=${SAVEINT} TPT=${TPT_SECONDS}s -> Trajectories cap=${CAP}"

if [ ${CAP} -le ${latest} ]; then
    echo "blocker: not enough time for another ${SAVEINT}-traj block; stopping gracefully (the next chained job retries with a fresh wall)."
    echo "--end " `date` `date +%s`
    exit 0
fi

# <Trajectories> is a COUNT of sweeps: run NRUN = CAP - latest.
NRUN=$(( CAP - latest ))
sed -i "/<Trajectories>/{s/>${CUR_TRAJ}</>${NRUN}</}" ${XML}
echo "this job: run ${NRUN} trajectories (${latest} -> ${CAP}, overall target ${TARGET})"

# ---- run ----
flux run -N 2 --tasks-per-node=4 --verbose --exclusive --setopt=mpibind=verbose:1 $APP $PARAMS

echo "--end " `date` `date +%s`
