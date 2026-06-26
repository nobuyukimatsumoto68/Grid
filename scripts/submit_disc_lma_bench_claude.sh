#!/bin/bash
# submit_disc_lma_bench_claude.sh
# FLUX batch launcher for the disc LMA CHUNK-A correctness bench
# (disc_lma_bench_claude): on ONE m=0.01 config it eigensolves the low modes,
# reconstructs the physical 4D A2A pair (a_i,b_i), runs the three gates (eigen-
# residual / operator-lift / gamma5-herm), and prints the exact low-mode loop
# L^low_Gamma(t,p=0). Writes NO data files -- everything goes to disc_lma_bench_<id>.
# Submit yourself:  flux batch submit_disc_lma_bench_claude.sh  (Claude does not submit.)
#
# Production layout: 8 nodes / 32 ranks (mpi 2.2.2.4); 1 node host-OOMs.
# CONFIGURED FOR THE CHEBYSHEV TEST (single prec, EIG_METHOD=1). RETUNED window
# (lo=0.02 ABOVE the cut, hi~100, ord=150 -> cut-mode ~11x / bottom ~33x amplified)
# after lo=0.012/ord=100 gave 0/100 (band sat on the filter edge). Tests: (a) does it
# converge now and how fast vs shift-invert's 763s, (b) GATE1 < 0.13? -- Cheby's matvec
# has NO inner CG so NO single-prec residual-drift floor -> likely more accurate modes.
# Cheap matvecs -> fits pdebug. If still not converging, bump CHEB_ORD / NM.
#FLUX: -t 60m
#FLUX: --output=disc_lma_bench_{{id}}
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

source /usr/workspace/lsd/matsumoto5/su4_32c/env.sh
builddir=/usr/workspace/lsd/matsumoto5/su4_32c/build
APP=${builddir}/examples/disc_lma_bench_claude

# ----- ensemble / config (m=0.01 @ b10.8); pick the LATEST lat unless overridden -----
mass=0.01
confdir=/p/lustre5/matsumoto5/conf_nc4nf1_2448/conf_nc4nf1_2448_b10p800_m0p0100
CONFIG=${CONFIG:-$(ls ${confdir}/*_lat.* 2>/dev/null | sort -V | tail -n1)}

# ----- eigensolve knobs -----
# 100 low modes; larger Krylov (NK=140/NM=240) gives the dense low band room to
# separate. Push NSTOP/NEV later for the variance study (chunk C).
export NSTOP=100
export NK=140
export NM=240
export ERESID=1e-5
export MAXITER=300
# eigensolve method: 1 = Chebyshev (THIS test), 2 = shift-invert (robust fallback).
export EIG_METHOD=1
# eigensolve PRECISION: 1 = single (Cheby has no inner-CG floor, so single is fine and
# tests accuracy), 2 = DOUBLE (slow, pbatch). Compare GATE1 vs the shift-invert 0.13.
export EIG_PREC=1
export INV_TOL=1e-5
export INV_MAXIT=50000
# Chebyshev window for m=0.01 (retune: lo ABOVE the cut so the wanted band sits
# clearly below the filter edge -> strong amplification). lo=0.02 (> eval[99]~0.0097),
# ord=150. hi = 1.1 * power-method lambda_max (self-tuning safety factor; ~1.1*81 ~ 90,
# the bench measures lambda_max each run). Smaller hi -> slightly MORE amplification.
export CHEB_LO=0.02
export CHEB_ORD=150
export CHEB_HI_FAC=1.1
# how many low modes to use in the loop + gates, and how many to print per-mode detail for
export NEV=100
export NCHECK=5

if [ ! -x "${APP}" ]; then
    echo "ERROR: lma bench binary not found/executable: ${APP}" >&2
    echo "       build it first via build_disc_speedup_claude.sh" >&2
    exit 1
fi
if [ -z "${CONFIG}" ] || [ ! -f "${CONFIG}" ]; then
    echo "ERROR: config not found: '${CONFIG}'" >&2
    exit 1
fi

echo "--start " $(date) $(date +%s)
echo "config = ${CONFIG}"
echo "knobs: EIG_METHOD=${EIG_METHOD} EIG_PREC=${EIG_PREC} NSTOP=${NSTOP} NK=${NK} NM=${NM} NEV=${NEV}"
echo "       CHEB_LO=${CHEB_LO} CHEB_ORD=${CHEB_ORD} CHEB_HI_FAC=${CHEB_HI_FAC} INV_TOL=${INV_TOL}"

OPTIONS="--decomposition --comms-concurrent --comms-overlap --debug-mem --shm 2048 --shm-mpi 1"
PARAMS="--grid ${Lattice} --mpi ${MpiGrid} --threads 8 --accelerator-threads 8 ${OPTIONS}"

flux run -N 8 --tasks-per-node=4 --verbose --exclusive --setopt=mpibind=verbose:1 \
    ${APP} --config ${CONFIG} --mass ${mass} ${PARAMS}

echo "--end " $(date) $(date +%s)
