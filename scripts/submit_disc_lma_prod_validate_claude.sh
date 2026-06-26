#!/bin/bash
# submit_disc_lma_prod_validate_claude.sh
# ONE-CONFIG validation of the LMA PRODUCTION binary (disc_multipleGamma_binary_lma_claude)
# on a real m=0.01 @ b10.8 config (lat.758), in SCRATCH (touches no production). Confirms:
#   - runs end-to-end -> writes traces.<gam>.758 (Scidac, drop-in format);
#   - the per-config WALL (eigensolve Cheby+RR + 1536 projected solves) -- expect ~1.5-2h,
#     i.e. ~3-4x faster than the original disc binary's ~5.4h (the SPEED deliverable);
#   - the evec CHECKPOINT (evec.758.scidac + eval.758.h5 written in the scratch obsdir).
# Submit yourself:  flux batch submit_disc_lma_prod_validate_claude.sh  (Claude does not submit.)
#
# Per-config ~1.5-2h -> pbatch (exceeds pdebug's 1h cap).
#FLUX: -t 240m
#FLUX: --output=disc_lma_prodval_{{id}}
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

# ----- ensemble / config (m=0.01 @ b10.8); pick the LATEST lat unless overridden -----
mass=0.01
beta=10.8
betastr=10p800
massstr=0p0100
confdir=/p/lustre5/matsumoto5/conf_nc4nf1_2448/conf_nc4nf1_2448_b${betastr}_m${massstr}
CONFIG=${CONFIG:-$(ls ${confdir}/*_lat.* 2>/dev/null | sort -V | tail -n1)}
EIGREF=${EIGREF:-${ROOT}/eigref_2448_b${betastr}_m${massstr}.h5}

# ----- scratch single-config latdir (symlink) + scratch obsdir (NO production touched) -----
SCRATCH=/p/lustre5/matsumoto5/validate_lma/b${betastr}_m${massstr}
slat=${SCRATCH}/lat
sobs=${SCRATCH}/obs
mkdir -p ${slat} ${sobs}
cfgname=$(basename ${CONFIG})
ln -sf ${CONFIG} ${slat}/${cfgname}

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
if [ -z "${CONFIG}" ] || [ ! -f "${CONFIG}" ]; then
    echo "ERROR: config not found: '${CONFIG}'" >&2
    exit 1
fi
if [ ! -f "${EIGREF}" ]; then
    echo "ERROR: eigref not found: '${EIGREF}' -- run submit_disc_lma_eigref_claude.sh first" >&2
    exit 1
fi

OPTIONS="--decomposition --comms-concurrent --comms-overlap --debug-mem --shm 2048 --shm-mpi 1"
APPPARAMS="--mass ${mass} --beta ${beta} --dir ${slat} --obsdir ${sobs} --eigref ${EIGREF}"
PARAMS=" --grid ${Lattice} --mpi ${MpiGrid} --threads 8 --accelerator-threads 8 ${OPTIONS}"

echo "--start " $(date) $(date +%s)
echo "config = ${CONFIG}"
echo "eigref = ${EIGREF}"
echo "scratch obs = ${sobs}"

flux run -N 8 --tasks-per-node=4 --verbose --exclusive --setopt=mpibind=verbose:1 \
    $APP $APPPARAMS $PARAMS

echo "--end " $(date) $(date +%s)
echo "validation outputs (expect 10 traces.<gam>.* + evec.*.scidac + eval.*.h5):"
ls -l ${sobs}
