#!/bin/bash
# submit_disc_lma_eigref_claude.sh
# FLUX launcher for the per-ENSEMBLE LMA eigref pre-calc (disc_lma_eigref_v2_claude):
# robust SHIFT-INVERT IRL on ONE real config -> writes the spectral landscape
# (lambda/sigma/lambda_max) to an HDF5 the per-config Cheby+RR production binary reads.
# Run ONCE per ensemble. m=0.01 @ b10.8 (the light/near-zero case where LMA pays off).
# Submit yourself:  flux batch submit_disc_lma_eigref_claude.sh  (Claude does not submit.)
#
# Shift-invert eigensolve (~1000-1500s for 100 modes at NM=240; chunk A was 763s at
# NM=140) -> fits pdebug's 1h cap. Only the eigenVALUES are written, so single prec is
# fine (vectors may be rough; values are accurate -- the per-config Cheby+RR recomputes
# the actual evecs in production). If it ever walls, raise -t / move to pbatch.
#FLUX: -t 60m
#FLUX: --output=disc_lma_eigref_{{id}}
#FLUX: -q pdebug
#FLUX: -N 8
#FLUX: -n 32
#FLUX: -g 1
#FLUX: --exclusive

set -u

Lattice="24.24.24.48"
MpiGrid="2.2.2.4"

date
hostname

export FASTLOAD_VERBOSE=1
export SPINDLE_FLUXOPT=off

ROOT=/usr/workspace/lsd/matsumoto5/su4_32c
source ${ROOT}/env.sh
builddir=${ROOT}/build
APP=${builddir}/examples/disc_lma_eigref_v2_claude

# ----- ensemble / config (m=0.01 @ b10.8); pick the LATEST lat unless overridden -----
mass=0.01
confdir=/p/lustre5/matsumoto5/conf_nc4nf1_2448/conf_nc4nf1_2448_b10p800_m0p0100
CONFIG=${CONFIG:-$(ls ${confdir}/*_lat.* 2>/dev/null | sort -V | tail -n1)}
# per-ensemble eigref landscape (the production binary reads this via --eigref)
EIGREF=${EIGREF:-${ROOT}/eigref_2448_b10p800_m0p0100.h5}

# ----- eigensolve knobs (shift-invert; values converge faster than vectors) -----
export EIG_PREC=1
export NSTOP=100
export NK=140
export NM=240
export ERESID=1e-5
export INV_TOL=1e-5
export INV_MAXIT=50000
export MAXITER=2000

if [ ! -x "${APP}" ]; then
    echo "ERROR: eigref binary not found/executable: ${APP}" >&2
    echo "       build it first via build_disc_lma_v2_claude.sh" >&2
    exit 1
fi
if [ -z "${CONFIG}" ] || [ ! -f "${CONFIG}" ]; then
    echo "ERROR: config not found: '${CONFIG}'" >&2
    exit 1
fi

echo "--start " $(date) $(date +%s)
echo "config = ${CONFIG}"
echo "eigref = ${EIGREF}"
echo "knobs: EIG_PREC=${EIG_PREC} NSTOP=${NSTOP} NK=${NK} NM=${NM} ERESID=${ERESID} INV_TOL=${INV_TOL}"

OPTIONS="--decomposition --comms-concurrent --comms-overlap --debug-mem --shm 2048 --shm-mpi 1"
PARAMS="--grid ${Lattice} --mpi ${MpiGrid} --threads 8 --accelerator-threads 8 ${OPTIONS}"

flux run -N 8 --tasks-per-node=4 --verbose --exclusive --setopt=mpibind=verbose:1 \
    ${APP} --config ${CONFIG} --mass ${mass} --eigref ${EIGREF} ${PARAMS}

echo "--end " $(date) $(date +%s)
