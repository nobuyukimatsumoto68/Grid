#!/bin/bash
# submit_disc_bench_claude.sh
# FLUX batch launcher for the disc speedup #2+#3 BENCHMARK
# (disc_mrhs_defl_bench_claude): multi-RHS + exact low-mode deflation break-even
# on ONE m=0.01 config. Modeled on submit_disc_tuolumne.sh but on the pdebug
# queue, a single node (4 GCD). Submit yourself:  flux batch submit_disc_bench_claude.sh
# (Claude does not submit.) Writes no data files; the break-even table goes to the
# flux output file disc_bench_<id>.
#
# -t may exceed the pdebug wall cap on this system -- lower it (or the NSTOP/NM
# knobs) if the queue rejects it.
#FLUX: -t 60m
#FLUX: --output=disc_bench_{{id}}
#FLUX: -q pdebug
#FLUX: -N 8
#FLUX: -n 32
#FLUX: -g 1
#FLUX: --exclusive

set -u

Lattice="24.24.24.48"
MpiGrid="2.2.2.4"     # 32 ranks on 8 nodes (4 GCD/node) = production layout;
                     # per-rank fields ~20 MB (1 node/4 ranks was 162 MB -> host OOM)

date
hostname

# email-notice runtime flags (as in submit_disc_tuolumne.sh)
export FASTLOAD_VERBOSE=1
export SPINDLE_FLUXOPT=off

source /usr/workspace/lsd/matsumoto5/su4_32c/env.sh
builddir=/usr/workspace/lsd/matsumoto5/su4_32c/build
APP=${builddir}/examples/disc_mrhs_defl_bench_claude

# ----- ensemble / config (m=0.01 @ b10.8); pick the LATEST lat unless overridden -----
mass=0.01
confdir=/p/lustre5/matsumoto5/conf_nc4nf1_2448/conf_nc4nf1_2448_b10p800_m0p0100
CONFIG=${CONFIG:-$(ls ${confdir}/*_lat.* 2>/dev/null | sort -V | tail -n1)}

# ----- benchmark knobs -----
# --- SERIOUS run: shift-invert IRL for 100 low modes of the 5D Mobius operator ---
# EIG_METHOD=2 = shift-invert (Lanczos on H^{-1} via inner CG; no window tuning,
# robust). EIG_METHOD=1 = Chebyshev IRL (needs CHEB_LO/ORD). lambda_min already
# measured (~5e-4), so DO_LMIN=0. The eval[] this prints IS the real spectrum ->
# sets beta for any future cheap Chebyshev runs and is the deflation basis.
export DO_LMIN=0
export DO_SOLVERCMP=0     # already measured (mixed 1.41x; batching no gain at Nev=0)
export DO_BATCHCMP=0      # already measured (batching DOES help w/ deflation: 1.65x)
export NEV_CMP=100
export DO_DEFL=1
export EIG_METHOD=2
export NRHS=16
export NSTOP=250
export NK=250
export NM=400
export NEV_LIST=0,200,250
# shift-invert inner solve: loose tol (inexact, fine for deflation vectors) keeps
# each Lanczos step cheap. INV_MAXIT generous since the solve is slow at small lambda.
export INV_TOL=1e-5      # shift-invert inner tol; 1e-4 broke down (NaN) at 200 modes
export INV_MAXIT=50000
export ERESID=1e-5
export MAXITER=300
# mixed-prec inner single-CG rel tol. Was 1e-8 (== outer) -> 3 full outer iters;
# 1e-4 lets each inner quit early and the double outer refine (test the speedup).
export INNER_TOL=1e-4
# batched mixed-CG caps. MAXPATCH=50 (old default) was marginal -> Nev=50 needed 51
# in the final double patch-up and CG aborted; give it headroom.
export MAXINNER=10000
export MAXOUTER=50
export MAXPATCH=1000
# (Chebyshev path only, EIG_METHOD=1)
export CHEB_LO=0.01
export CHEB_ORD=150
export CHEB_HI_FAC=1.1   # UV edge = factor * power-method lambda_max (~82 -> ~90)

if [ ! -x "${APP}" ]; then
    echo "ERROR: bench binary not found/executable: ${APP}" >&2
    echo "       build it first via build_disc_speedup_claude.sh" >&2
    exit 1
fi
if [ -z "${CONFIG}" ] || [ ! -f "${CONFIG}" ]; then
    echo "ERROR: config not found: '${CONFIG}'" >&2
    exit 1
fi

echo "--start " $(date) $(date +%s)
echo "config = ${CONFIG}"
echo "knobs: NRHS=${NRHS} NSTOP=${NSTOP} NK=${NK} NM=${NM} NEV_LIST=${NEV_LIST} CHEB_LO=${CHEB_LO} CHEB_ORD=${CHEB_ORD}"

# EXACTLY the production OPTIONS (submit_disc_tuolumne.sh) -- no extra flags, so
# the only differences from the known-good production run are the binary + its
# args. At the production 8-node/32-rank layout the default device cache is fine.
OPTIONS="--decomposition --comms-concurrent --comms-overlap --debug-mem --shm 2048 --shm-mpi 1"
PARAMS="--grid ${Lattice} --mpi ${MpiGrid} --threads 8 --accelerator-threads 8 ${OPTIONS}"

flux run -N 8 --tasks-per-node=4 --verbose --exclusive --setopt=mpibind=verbose:1 \
    ${APP} --config ${CONFIG} --mass ${mass} ${PARAMS}

echo "--end " $(date) $(date +%s)
