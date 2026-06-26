#!/bin/bash
# submit_disc_lma_bench_v2_claude.sh
# STEP 3: validate the per-config CHEBYSHEV+RR eigensolve on the REAL m=0.01 dense
# spectrum (the gating risk -- the agent only validated Cheby on a gapped 4^4 HOT
# testbed). Runs disc_lma_bench_v2_claude on the SAME config as the eigref (lat.758),
# EIG_METHOD=1 (Cheby, auto window/order from the eigref) + RR refine, and checks:
#   - Cheby converges (Nconv>=Nstop) on the dense spectrum (was the odd-degree bug);
#   - eval reldiff vs the shift-invert reference ~0 (same config);
#   - GATE1 small after RR; GATE2 r1 ~machine (the lift); GATE3 settles b=gamma5 a;
#   - wall vs the shift-invert eigref (~763s chunk A / the eigref run).
# Submit yourself:  flux batch submit_disc_lma_bench_v2_claude.sh  (Claude does not submit.)
#
# Cheby auto-order on this spectrum lands ~395 (lambda_min~5e-4 far below lo) -> each
# matvec ~395 H-applies ~ a shift-invert inner solve, so expect ~parity wall, not a big
# speed win here; the benefit is no inner-CG floor (cleaner modes). If it walls, bump -t.
#FLUX: -t 60m
#FLUX: --output=disc_lma_bench_v2_{{id}}
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
APP=${builddir}/examples/disc_lma_bench_v2_claude

# ----- SAME config + eigref as the eigref pre-calc (so eval-vs-ref is a clean check) -----
mass=0.01
confdir=/p/lustre5/matsumoto5/conf_nc4nf1_2448/conf_nc4nf1_2448_b10p800_m0p0100
CONFIG=${CONFIG:-$(ls ${confdir}/*_lat.* 2>/dev/null | sort -V | tail -n1)}
EIGREF=${EIGREF:-${ROOT}/eigref_2448_b10p800_m0p0100.h5}

# ----- eigensolve: Chebyshev (auto window from eigref) + double Rayleigh-Ritz refine -----
export EIG_METHOD=1     # 1 = Chebyshev, 2 = shift-invert
export EIG_PREC=1       # single Cheby supplies the subspace; RR polishes in double
export RR_REFINE=1
export NSTOP=100
export NK=140
export NM=240
export NEV=100
export NCHECK=5
export ERESID=1e-4
export MAXITER=500
# Chebyshev window: lo = CHEB_LO_FAC * lambda_ref[Nstop-1], hi = CHEB_HI_FAC * lambda_max
# (from the eigref). ORDER: AUTO via the BAND-TOP anchor CHEB_ATOP (target amplification of
# the cut-boundary wanted mode; config-ROBUST, vs the lambda_min CHEB_GAIN fallback which
# under-orders near-zero configs). MEASURED m=0.01: A_top~5 -> order 151 (95s, reldiff 9e-7,
# the winner); A_top=8 -> order ~187 (safe margin). Even degree enforced internally.
export CHEB_LO_AUTO=1
export CHEB_LO_FAC=1.5
export CHEB_HI_FAC=1.1
export CHEB_ATOP=8

if [ ! -x "${APP}" ]; then
    echo "ERROR: bench v2 binary not found/executable: ${APP}" >&2
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

echo "--start " $(date) $(date +%s)
echo "config = ${CONFIG}"
echo "eigref = ${EIGREF}"
echo "knobs: EIG_METHOD=${EIG_METHOD} EIG_PREC=${EIG_PREC} RR_REFINE=${RR_REFINE} NSTOP=${NSTOP} NK=${NK} NM=${NM} NEV=${NEV}"
echo "       CHEB_LO_FAC=${CHEB_LO_FAC} CHEB_HI_FAC=${CHEB_HI_FAC} CHEB_GAIN=${CHEB_GAIN} ERESID=${ERESID} MAXITER=${MAXITER}"

OPTIONS="--decomposition --comms-concurrent --comms-overlap --debug-mem --shm 2048 --shm-mpi 1"
PARAMS="--grid ${Lattice} --mpi ${MpiGrid} --threads 8 --accelerator-threads 8 ${OPTIONS}"

flux run -N 8 --tasks-per-node=4 --verbose --exclusive --setopt=mpibind=verbose:1 \
    ${APP} --config ${CONFIG} --mass ${mass} --eigref ${EIGREF} ${PARAMS}

echo "--end " $(date) $(date +%s)
