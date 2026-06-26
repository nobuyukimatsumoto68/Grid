#!/bin/bash
#FLUX: -t 240m
#FLUX: --output=hmc_prod_{{id}}
#FLUX: -q pbatch
#FLUX: -N 8
#FLUX: -n 32
#FLUX: -g 1
#FLUX: --exclusive
#
# FLUX batch launcher for ONE allocation of a 32^3x64 PRODUCTION HMC stream, built for
# --dependency=afterany AUTO-CHAINING: run_prod32c_claude.sh submits a chain of these (each
# depending on the previous), and each job picks up wherever the previous one stopped via
# RUNTIME resume (essential for chaining -- StartTrajectory is NOT known at submit time):
#   1. read config_prefix from the XML, detect the latest saved trajectory in cwd;
#   2. if a checkpoint exists, switch the XML to CheckpointStart / StartTrajectory=<latest> /
#      NoMetropolisUntil=0 (idempotent: parse-current-then-replace);
#   3. if <latest> >= NTRAJ_TARGET the stream is done -> exit without launching;
#   4. graceful wall blocker: cap <Trajectories> to the largest whole-saveInterval block that
#      fits the time left, so the binary stops cleanly on a checkpoint instead of being killed;
#   5. run the binary.
# Driver passes NTRAJ_TARGET (overall trajectory goal) and WALL_SECONDS (blocker fallback when
# flux job timeleft is unavailable) via --env. TPT_SECONDS / OVERHEAD_SECONDS tune the blocker.

date
here=`pwd`
export FASTLOAD_VERBOSE=1
export SPINDLE_FLUXOPT=off

XML=ip_hmc_mobius_claude.xml

source /usr/workspace/lsd/matsumoto5/su4_32c/env.sh
APP=/usr/workspace/lsd/matsumoto5/su4_32c/Grid_sdm_build/src/gauge_gen_Nc4/bin/dweofa_mobius_HSDM_v4

echo "--start " `date` `date +%s`

OPTIONS="--decomposition --comms-concurrent --comms-overlap --debug-mem  --shm 2048 --shm-mpi 1"
PARAMS=" --grid 32.32.32.64 --mpi 2.2.2.4 --threads 8 --accelerator-threads 8 ${OPTIONS} --ParameterFile ${XML}"

# ---- runtime resume: continue from the latest saved config (prefix read from the XML) ----
PREFIX=$(grep -oP '(?<=<config_prefix>)[^<]+' ${XML} | head -1)   # e.g. ./conf_nc4nf1_3264_..._lat
latest=$(ls ${PREFIX}.* 2>/dev/null | sed 's/.*\.//' | sort -n | tail -n1)
latest=${latest:-0}
CUR_START=$(grep -oP '(?<=<StartTrajectory>)[0-9]+' ${XML} | head -1); CUR_START=${CUR_START:-0}
if [ "${latest}" -gt 0 ]; then
    sed -i "/<StartTrajectory>/{s/>${CUR_START}</>${latest}</}" ${XML}
    sed -i "/<StartingType>/{s/HotStart/CheckpointStart/}" ${XML}
    sed -i "/<NoMetropolisUntil>/{s/>[0-9]*</>0</}" ${XML}
    echo "resume: latest checkpoint = ${latest} -> CheckpointStart"
else
    echo "resume: no checkpoint found -> fresh HotStart"
fi

# ---- target check (overall goal passed by the driver) ----
TARGET=${NTRAJ_TARGET:-$(grep -oP '(?<=<Trajectories>)[0-9]+' ${XML} | head -1)}
if [ "${latest}" -ge "${TARGET}" ]; then
    echo "stream already at target (${latest} >= ${TARGET}); nothing to do (chain tail)."
    echo "--end " `date` `date +%s`
    exit 0
fi

# ---- graceful wall-time blocker: cap Trajectories to whole saveInterval blocks that fit ----
# Per-trajectory time is MEASURED LIVE: the binary (Grid's own timer) prints
# "Total time for trajectory (s): X" into the per-job hmc_prod_* logs. We take the LARGEST
# such value from PRIOR jobs of this stream (integer s) x a safety margin as the per-traj
# estimate; the fixed TPT_SECONDS only bootstraps the FIRST job (no measurement yet).
OVERHEAD_SECONDS=${OVERHEAD_SECONDS:-400}  # startup + final ckpt write + teardown margin
MARGIN_NUM=${MARGIN_NUM:-12}               # 12/10 = 1.2x safety on the measured per-traj time
measured=$(grep -hoE "Total time for trajectory \(s\): [0-9]+" hmc_prod_* 2>/dev/null | grep -oE "[0-9]+$" | sort -n | tail -n1)
if [ -n "${measured}" ]; then
    TPT_SECONDS=$(( measured * MARGIN_NUM / 10 ))
    echo "blocker: measured per-traj (max, prior jobs this stream) = ${measured}s -> TPT=${TPT_SECONDS}s (x${MARGIN_NUM}/10)"
else
    TPT_SECONDS=${TPT_SECONDS:-750}
    echo "blocker: no prior trajectory timing in this stream -> bootstrap TPT=${TPT_SECONDS}s"
fi
TIMELEFT=$(flux job timeleft 2>/dev/null)
case "${TIMELEFT}" in ''|*[!0-9.]*) TIMELEFT=${WALL_SECONDS:-7200} ;; esac
TIMELEFT=${TIMELEFT%.*}
SAVEINT=$(grep -oP '(?<=<saveInterval>)[0-9]+' ${XML} | head -1); SAVEINT=${SAVEINT:-4}
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

sed -i "/<Trajectories>/{s/>${CUR_TRAJ}</>${CAP}</}" ${XML}
echo "this job: trajectories ${latest} -> ${CAP} (overall target ${TARGET})"

# ---- run ----
flux run -N 8 --tasks-per-node=4 --verbose --exclusive --setopt=mpibind=verbose:1 $APP $PARAMS

echo "--end " `date` `date +%s`
