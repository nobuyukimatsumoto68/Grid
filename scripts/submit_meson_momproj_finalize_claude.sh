#!/bin/bash
#FLUX: -t 240m
#FLUX: --output=finalize_conn_{{id}}
#FLUX: -q pbatch
#FLUX: -N 8
#FLUX: -n 32
#FLUX: -g 1
#FLUX: --exclusive
#
# 4h FINALIZER batch job for the CONNECTED meson correlator. Processes ONLY the
# configs listed in CONFLIST (space-separated conf indices), so it does NOT
# waste wall time relaunching the binary on already-complete configs. The binary
# DOES skip a complete config (the skip decision takes ~0.65s of Grid time), but
# each config is a separate flux run, so the surrounding MPI init + Grid setup +
# teardown costs ~6s per skip (measured). Re-skipping hundreds of done configs
# can therefore eat the whole wall before reaching the missing ones. The finalizer
# driver finalize_old_ensembles_claude.sh computes CONFLIST and passes the env
# below (BIN, M5, mass, cfgfilename, latdir, obsdir, CONFLIST). Same dynamic
# graceful wall blocker as submit_meson_momproj_tuolumne_claude.sh.

Lattice="24.24.24.48"
MpiGrid="2.2.2.4"

date
here=`pwd`

export FASTLOAD_VERBOSE=1
export SPINDLE_FLUXOPT=off
export HDF5_USE_FILE_LOCKING=FALSE

source /usr/workspace/lsd/matsumoto5/su4_32c/env.sh
APP=${BIN}

echo "--start " `date` `date +%s`

OPTIONS="--decomposition --comms-concurrent --comms-overlap --debug-mem  --shm 2048 --shm-mpi 1"
PARAMS=" --grid ${Lattice} --mpi ${MpiGrid} --threads 8 --accelerator-threads 8 ${OPTIONS}"

mkdir -p ${obsdir}

# graceful wall blocker (dynamic, measured) -- identical logic to the main submit:
# stop before starting a config we cannot finish+save. max_dur is the largest
# config time seen so far; TPT_SECONDS bootstraps the first one.
OVERHEAD_SECONDS=${OVERHEAD_SECONDS:-180}
MARGIN_NUM=${MARGIN_NUM:-12}
max_dur=${TPT_SECONDS:-120}

echo "CONFLIST (${0##*/}): ${CONFLIST}"

for conf in ${CONFLIST}
do
    cfg=${latdir}/${cfgfilename}_lat.${conf}
    outfile=${obsdir}/mesons_conn.${conf}.h5

    if [ ! -f "${cfg}" ]; then
        echo "skip conf ${conf}: config not found ${cfg}"
        continue
    fi

    NEED_SECONDS=$(( max_dur * MARGIN_NUM / 10 + OVERHEAD_SECONDS ))
    TIMELEFT=$(flux job timeleft 2>/dev/null)
    case "${TIMELEFT}" in
      ''|*[!0-9.]*) TIMELEFT=0 ;;
    esac
    TIMELEFT=${TIMELEFT%.*}
    if [ "${TIMELEFT}" -gt 0 ] && [ "${TIMELEFT}" -lt "${NEED_SECONDS}" ]; then
        echo "blocker: ${TIMELEFT}s left < ${NEED_SECONDS}s needed (max_dur=${max_dur}s); stopping gracefully at conf ${conf} (resubmit/re-finalize to continue)."
        break
    fi

    echo "==== conf ${conf} -> ${outfile} ===="
    # binary positional args: <config> <M5> <mass> <outfile>
    t0=$(date +%s)
    flux run -N 8 --tasks-per-node=4 --verbose --exclusive --setopt=mpibind=verbose:1 $APP ${cfg} ${M5} ${mass} ${outfile} $PARAMS
    dur=$(( $(date +%s) - t0 ))
    [ "${dur}" -gt "${max_dur}" ] && max_dur=${dur}
    echo "conf ${conf} took ${dur}s (max so far ${max_dur}s)"
done

echo "--end " `date` `date +%s`
