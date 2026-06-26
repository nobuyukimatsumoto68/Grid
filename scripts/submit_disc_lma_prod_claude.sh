#!/bin/bash
# submit_disc_lma_prod_claude.sh
# FULL PRODUCTION of the LMA disconnected loop (disc_multipleGamma_binary_lma_claude) over
# the m=0.01 @ b10.8 LIGHT ensemble. Per config: Cheby+RR eigensolve (reads the per-ensemble
# eigref) + source-PROJECTED stochastic high solve + exact L^low -> traces.<gam>.<conf>
# (Scidac, drop-in) in a NEW obs dir (LMA = a different estimator -> do NOT mix with the old
# plain traces). evec.<conf>.scidac/eval.<conf>.h5 checkpoints saved alongside (reload to skip
# the eigensolve on rerun). Self-skips done configs; graceful wall blocker stops cleanly before
# a config that would overrun -> RESUBMIT to continue (or add an afterany chain later).
# Submit yourself:  flux batch submit_disc_lma_prod_claude.sh  (Claude does not submit.)
#
# RUN THE ONE-CONFIG VALIDATION FIRST (submit_disc_lma_prod_validate_claude.sh) to confirm
# end-to-end + measure the per-config wall, then lower DISC_TPT_SECONDS to that value here.
#FLUX: -t 480m
#FLUX: --output=disc_lma_prod_{{id}}
#FLUX: -q pbatch
#FLUX: -N 8
#FLUX: -n 32
#FLUX: -g 1
#FLUX: --exclusive

set -u

Lattice="24.24.24.48"
MpiGrid="2.2.2.4"

date; hostname
export FASTLOAD_VERBOSE=1
export SPINDLE_FLUXOPT=off

ROOT=/usr/workspace/lsd/matsumoto5/su4_32c
source ${ROOT}/env.sh
builddir=${ROOT}/build
APP="${builddir}/examples/disc_multipleGamma_binary_lma_claude"

# ----- ensemble (m=0.01 @ b10.8): real config dir IN, NEW LMA obs dir OUT -----
mass=0.01
beta=10.8
betastr=10p800
massstr=0p0100
LATDIR=/p/lustre5/matsumoto5/conf_nc4nf1_2448/conf_nc4nf1_2448_b${betastr}_m${massstr}
OBSDIR=${OBSDIR:-/p/lustre5/matsumoto5/lma_nc4nf1_2448_b${betastr}_m${massstr}}
EIGREF=${EIGREF:-${ROOT}/eigref_2448_b${betastr}_m${massstr}.h5}
mkdir -p ${OBSDIR}

# ----- eigensolve knobs: Cheby+RR (band-top auto order), reading the eigref -----
export EIG_METHOD=1
export EIG_PREC=1
export RR_REFINE=1
export NSTOP=100
export NK=140
export NM=240
export NEV=100
export ERESID=1e-4
export MAXITER=500
export CHEB_LO_AUTO=1
export CHEB_LO_FAC=1.5
export CHEB_HI_FAC=1.1
export CHEB_ATOP=8

if [ ! -x "${APP}" ]; then
    echo "ERROR: LMA production binary not found/executable: ${APP}" >&2
    echo "       build it first via build_disc_lma_v2_claude.sh" >&2
    exit 1
fi
if [ ! -f "${EIGREF}" ]; then
    echo "ERROR: eigref not found: '${EIGREF}' -- run submit_disc_lma_eigref_claude.sh first" >&2
    exit 1
fi

OPTIONS="--decomposition --comms-concurrent --comms-overlap --debug-mem --shm 2048 --shm-mpi 1"
APPPARAMS="--mass ${mass} --beta ${beta} --dir ${LATDIR} --obsdir ${OBSDIR} --eigref ${EIGREF}"
PARAMS=" --grid ${Lattice} --mpi ${MpiGrid} --threads 8 --accelerator-threads 8 ${OPTIONS}"

echo "--start " $(date) $(date +%s)
echo "latdir = ${LATDIR}"
echo "obsdir = ${OBSDIR}  (NEW LMA estimator dir)"
echo "eigref = ${EIGREF}"

# graceful wall blocker (same scheme as submit_disc_mixedprec_tuolumne_claude.sh). Bootstrap
# DISC_TPT_SECONDS guards the FIRST config; lower it to the validation-measured per-config wall.
BLOCKER_OVERHEAD=${BLOCKER_OVERHEAD:-300}
export DISC_TPT_SECONDS=${DISC_TPT_SECONDS:-7200}
TIMELEFT=$(flux job timeleft 2>/dev/null)
case "${TIMELEFT}" in ''|*[!0-9.]*) TIMELEFT=0 ;; esac
TIMELEFT=${TIMELEFT%.*}
if [ "${TIMELEFT}" -gt 0 ]; then
    export DISC_DEADLINE_EPOCH=$(( $(date +%s) + TIMELEFT - BLOCKER_OVERHEAD ))
    echo "blocker: timeleft=${TIMELEFT}s -> DISC_DEADLINE_EPOCH=${DISC_DEADLINE_EPOCH}"
fi

flux run -N 8 --tasks-per-node=4 --verbose --exclusive --setopt=mpibind=verbose:1 \
    $APP $APPPARAMS $PARAMS

echo "--end " $(date) $(date +%s)
