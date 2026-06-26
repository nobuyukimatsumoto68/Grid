#!/bin/bash
#FLUX: -t 120m
#FLUX: --output=hmc_tune_{{id}}
#FLUX: -q pbatch
#FLUX: -N 8
#FLUX: -n 32
#FLUX: -g 1
#FLUX: --exclusive
#
# FLUX batch launcher for one HMC stream of the M5/Ls tuning ensemble.
#   32^3 x 64, m = 0.01, Ls = 16, M5 = 1.5, b/c = 1.5/0.5 (Mobius EOFA, SU(4) SDM).
# One gauge_beta per job; run_tuo_tune_claude.sh seds @BETA@ into the XML and
# submits one of these per stream (beta = 10.7, 10.8, 10.9).
# Modeled on ../../../../submit_hmc_tuolumne.sh.


### four gpus per node
date
here=`pwd`

# To verify fastload2 is loading or not, set env variable FASTLOAD_VERBOSE=1 and rank 0
# (or serial tasks) will print out something if fastload2 is being used (email 10/30/24).
export FASTLOAD_VERBOSE=1

# email notice on 12/11/24
export SPINDLE_FLUXOPT=off


XML=ip_hmc_mobius_claude.xml


### Launch parallel executable
source /usr/workspace/lsd/matsumoto5/su4_32c/env.sh
APP=/usr/workspace/lsd/matsumoto5/su4_32c/Grid_sdm_build/src/gauge_gen_Nc4/bin/dweofa_mobius_HSDM_v4

echo "--start " `date` `date +%s`

OPTIONS="--decomposition --comms-concurrent --comms-overlap --debug-mem  --shm 2048 --shm-mpi 1"
PARAMS=" --grid 32.32.32.64 --mpi 2.2.2.4 --threads 8 --accelerator-threads 8 ${OPTIONS} --ParameterFile ${XML}"

# ----------------- graceful wall-time blocker -----------------
# The binary runs its whole trajectory loop inside one flux run, and the NERSC
# checkpointer saves only every <saveInterval> trajectories. If the wall time hits
# mid-loop the job is HARD-KILLED -- wasting the rest of the allocation and risking a
# half-written checkpoint. Instead we cap <Trajectories> for THIS job to the largest
# whole-saveInterval block that fits in the time left, so the binary stops cleanly on a
# checkpoint boundary. If not even one block fits, we stop WITHOUT launching.
# Tune these if trajectories run faster/slower than assumed:
TPT_SECONDS=${TPT_SECONDS:-750}            # conservative wall-time per trajectory (measured ~646)
OVERHEAD_SECONDS=${OVERHEAD_SECONDS:-400}  # startup (config load + action build) + final ckpt write + teardown

# time left in this allocation (s); if flux job timeleft is unavailable fall back to
# WALL_SECONDS (passed by the driver to match its -t; default 7200 = the 120m directive)
TIMELEFT=$(flux job timeleft 2>/dev/null)
case "${TIMELEFT}" in
  ''|*[!0-9.]*) TIMELEFT=${WALL_SECONDS:-7200} ;;
esac
TIMELEFT=${TIMELEFT%.*}

# what the driver wrote into the XML
TARGET=$(grep -oP '(?<=<Trajectories>)[0-9]+' ${XML} | head -1)
SAVEINT=$(grep -oP '(?<=<saveInterval>)[0-9]+' ${XML} | head -1)
START=$(grep -oP '(?<=<StartTrajectory>)[0-9]+' ${XML} | head -1)
: "${TARGET:=40}" "${SAVEINT:=5}" "${START:=0}"

# whole saveInterval blocks that fit -> cap Trajectories at START + that (never above TARGET)
NFIT=$(( (TIMELEFT - OVERHEAD_SECONDS) / TPT_SECONDS ))
[ ${NFIT} -lt 0 ] && NFIT=0
NFIT=$(( (NFIT / SAVEINT) * SAVEINT ))
CAP=$(( START + NFIT ))
[ ${CAP} -gt ${TARGET} ] && CAP=${TARGET}

echo "blocker: timeleft=${TIMELEFT}s start=${START} target=${TARGET} saveInterval=${SAVEINT} TPT=${TPT_SECONDS}s -> Trajectories cap=${CAP}"

if [ ${CAP} -le ${START} ]; then
    echo "blocker: not enough time for another ${SAVEINT}-trajectory block; stopping gracefully to save the allocation (no new checkpoint would be reached)."
    echo "--end " `date` `date +%s`
    exit 0
fi

# only ever LOWER Trajectories (this job stops at CAP; resubmit to continue toward TARGET)
sed -i "/<Trajectories>/{s/>${TARGET}</>${CAP}</}" ${XML}
echo "blocker: this job will run trajectories ${START} -> ${CAP} (overall target ${TARGET})"
# --------------------------------------------------------------

flux run -N 8 --tasks-per-node=4 --verbose --exclusive --setopt=mpibind=verbose:1 $APP $PARAMS

echo "--end " `date` `date +%s`
