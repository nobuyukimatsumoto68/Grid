#!/bin/bash
#FLUX: -t 60m
#FLUX: --output=meson_momproj_{{id}}
#FLUX: -q pbatch
#FLUX: -N 8
#FLUX: -n 32
#FLUX: -g 1
#FLUX: --exclusive
#
# FLUX batch launcher for the momentum-projected CONNECTED meson correlator
# (baryons_0000_dirac_claude). Mirrors submit_disc_tuolumne.sh: one job per
# ensemble, configs processed serially. Per config it runs one point-source
# inversion + meson contraction and writes mesons_conn.<conf>.h5 into obsdir.
# The driver run_meson_momproj_tuo_claude.sh passes the env vars below
# (BIN, M5, mass, cfgfilename, latdir, obsdir; optional STRIDE).
#
# Config indices are auto-detected from latdir (conf_min, interval) exactly as
# the disc binary does, so the stride equals the disc native stride (= 20 here).
# Set STRIDE to a multiple of the interval to thin the set. Configs whose output
# h5 already exists are SKIPPED, so re-submitting this job resumes where an
# earlier (wall-time-limited) run stopped.

Lattice="24.24.24.48"
MpiGrid="2.2.2.4"

### four gpus per node
date
here=`pwd`

# To verify fastload2 is loading or not, set env variable FASTLOAD_VERBOSE=1 and rank 0
# (or serial tasks) will print out something if fastload2 is being used (email 10/30/24).
export FASTLOAD_VERBOSE=1

# email notice on 12/11/24
export SPINDLE_FLUXOPT=off

# meson driver writes HDF5; disable file locking on the parallel FS.
export HDF5_USE_FILE_LOCKING=FALSE


### environment + binary (binary built by the driver on the login node)
source /usr/workspace/lsd/matsumoto5/su4_32c/env.sh
APP=${BIN}

echo "--start " `date` `date +%s`

OPTIONS="--decomposition --comms-concurrent --comms-overlap --debug-mem  --shm 2048 --shm-mpi 1"
PARAMS=" --grid ${Lattice} --mpi ${MpiGrid} --threads 8 --accelerator-threads 8 ${OPTIONS}"

mkdir -p ${obsdir}

# ----------------- graceful wall-time blocker -----------------
# Each loop iteration runs one config (one inversion) and writes one h5; if the
# wall time hits mid-config the job is HARD-KILLED, wasting that config's compute
# and possibly leaving a half-written h5. So before launching each config we
# check the time left and stop CLEANLY if there is not enough for one more
# compute+save. Rather than a fixed per-config time we MEASURE each config's wall
# time as we go (date +%s around the flux run) and keep the largest seen, with a
# generous safety margin; TPT_SECONDS only bootstraps the FIRST config's guard
# (measured ~20s on MI300A). Resubmit (the binary self-skips complete h5) to
# continue. Skipped/complete configs are fast so they do not inflate the estimate.
TPT_SECONDS=${TPT_SECONDS:-120}            # generous bootstrap wall-time per config (measured ~20)
OVERHEAD_SECONDS=${OVERHEAD_SECONDS:-180}  # flux run launch + h5 write + teardown margin
MARGIN_NUM=${MARGIN_NUM:-12}               # safety factor numerator (12/10 = 1.2x)
max_dur=${TPT_SECONDS}                     # largest per-config wall time seen so far (s)

# auto-detect config indices from latdir (same logic as the disc binary):
# conf_min = first index, interval = gap between the first two, loop to the last.
indices=$(ls ${latdir}/${cfgfilename}_lat.* 2>/dev/null | sed "s|.*_lat\.||" | sort -n)
first=$(echo "${indices}" | head -n1)
second=$(echo "${indices}" | sed -n 2p)
last=$(echo "${indices}" | tail -n1)
interval=$((second - first))
step=${STRIDE:-$interval}
echo "configs: first=${first} last=${last} interval=${interval} step=${step}"

for((conf=first; conf<=last; conf+=step))
do
    cfg=${latdir}/${cfgfilename}_lat.${conf}
    outfile=${obsdir}/mesons_conn.${conf}.h5

    if [ ! -f "${cfg}" ]; then
        echo "skip conf ${conf}: config not found ${cfg}"
        continue
    fi

    # graceful wall blocker: stop before starting a config we cannot finish+save.
    # NEED = margin*max_dur + overhead, using the largest config time seen so far.
    # If timeleft is unavailable (empty/non-numeric) we do NOT block, so the run
    # proceeds rather than stopping prematurely.
    NEED_SECONDS=$(( max_dur * MARGIN_NUM / 10 + OVERHEAD_SECONDS ))
    TIMELEFT=$(flux job timeleft 2>/dev/null)
    case "${TIMELEFT}" in
      ''|*[!0-9.]*) TIMELEFT=0 ;;
    esac
    TIMELEFT=${TIMELEFT%.*}
    if [ "${TIMELEFT}" -gt 0 ] && [ "${TIMELEFT}" -lt "${NEED_SECONDS}" ]; then
        echo "blocker: ${TIMELEFT}s left < ${NEED_SECONDS}s needed (max_dur=${max_dur}s); stopping gracefully at conf ${conf} (resubmit to continue)."
        break
    fi

    # Completeness is now decided inside the binary (output_complete): it skips
    # only files that already contain ALL channel keys and recomputes partial
    # ones. So launch every existing checkpoint to be safe rather than skipping
    # on mere file existence here. (Old existence-only skip kept for reference.)
    # if [ -f "${outfile}" ]; then
    #     echo "skip conf ${conf}: output exists ${outfile}"
    #     continue
    # fi

    echo "==== conf ${conf} -> ${outfile} ===="
    # binary positional args: <config> <M5> <mass> <outfile>
    t0=$(date +%s)
    flux run -N 8 --tasks-per-node=4 --verbose --exclusive --setopt=mpibind=verbose:1 $APP ${cfg} ${M5} ${mass} ${outfile} $PARAMS
    dur=$(( $(date +%s) - t0 ))
    # track the largest config wall time seen (skips/complete configs are fast and
    # do not inflate it); used as the estimate for the next blocker decision.
    [ "${dur}" -gt "${max_dur}" ] && max_dur=${dur}
    echo "conf ${conf} took ${dur}s (max so far ${max_dur}s)"
done

echo "--end " `date` `date +%s`
