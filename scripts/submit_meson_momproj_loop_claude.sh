#!/bin/bash
#FLUX: -t 240m
#FLUX: --output=meson_loop_{{id}}
#FLUX: -q pbatch
#FLUX: -N 8
#FLUX: -n 32
#FLUX: -g 1
#FLUX: --exclusive
#
# Single-flux-run CONN job using the binary's INTERNAL config loop (--dir mode,
# like submit_disc_tuolumne.sh). One resident process loops every config in
# latdir, cheaply self-skips complete mesons_conn.<conf>.h5 (output_complete,
# no per-config MPI relaunch -- unlike the old per-config submit), and stops
# gracefully before the wall via the IN-BINARY blocker (MESON_DEADLINE_EPOCH).
# Driver env: BIN, M5, mass, latdir, obsdir (optional STRIDE = multiplier of the
# native config interval; 1 = every config, matches disc).
# Side-by-side variant of submit_meson_momproj_tuolumne_claude.sh (per-config),
# which is left intact and is still used by the finalizer.

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

# in-binary graceful wall blocker: pass the deadline (epoch s) computed from
# `flux job timeleft`; the binary measures per-config time itself and
# MESON_TPT_SECONDS bootstraps the first config. Inert if timeleft unavailable.
BLOCKER_OVERHEAD=${BLOCKER_OVERHEAD:-180}
export MESON_TPT_SECONDS=${MESON_TPT_SECONDS:-120}
TIMELEFT=$(flux job timeleft 2>/dev/null)
case "${TIMELEFT}" in
  ''|*[!0-9.]*) TIMELEFT=0 ;;
esac
TIMELEFT=${TIMELEFT%.*}
if [ "${TIMELEFT}" -gt 0 ]; then
    export MESON_DEADLINE_EPOCH=$(( $(date +%s) + TIMELEFT - BLOCKER_OVERHEAD ))
    echo "blocker: timeleft=${TIMELEFT}s -> MESON_DEADLINE_EPOCH=${MESON_DEADLINE_EPOCH} (bootstrap TPT=${MESON_TPT_SECONDS}s)"
fi

# optional thinning: STRIDE multiplies the native config interval (1 = every config).
STRIDE_ARG=""
[ -n "${STRIDE:-}" ] && STRIDE_ARG="--stride ${STRIDE}"

# binary loop mode: --dir/--obsdir/--mass/--M5 (auto-detects conf range from latdir)
flux run -N 8 --tasks-per-node=4 --verbose --exclusive --setopt=mpibind=verbose:1 $APP --dir ${latdir} --obsdir ${obsdir} --mass ${mass} --M5 ${M5} ${STRIDE_ARG} $PARAMS

echo "--end " `date` `date +%s`
